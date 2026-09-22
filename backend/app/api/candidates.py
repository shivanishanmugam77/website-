"""Reviewing the product candidates a catalogue's finds have been grouped into (Phase 6).

Read-only except for the one action that (re)builds them. Turning a candidate into a real,
publishable ``Product`` is a deliberate separate step for a human to take (Phase 8) - nothing
here ever creates or changes a ``Product`` row.
"""

from __future__ import annotations

import html
import secrets
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import HTMLResponse
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from app.api.catalogues import load_catalogue
from app.api.deps import AdminUser, ClientInfoDep, DbSession, SettingsDep, require_admin
from app.api.product_finds import _urls
from app.models import DuplicateFlag, ProductCandidate
from app.models.enums import AuditAction
from app.schemas.candidates import (
    CandidateAssembleOut,
    DuplicateFlagOut,
    ProductCandidateOut,
    ProductCandidateSummaryOut,
)
from app.schemas.catalogue import Page
from app.services import candidate_assembly as assembly
from app.services.audit import record_audit

router = APIRouter(
    prefix="/admin/catalogues", tags=["candidates"], dependencies=[Depends(require_admin)]
)

REVIEW_TEMPLATE = Path(__file__).resolve().parent.parent / "web" / "candidate_review.html"


@router.get("/{catalogue_id}/candidates/review", response_class=HTMLResponse)
def candidate_review_page(catalogue_id: uuid.UUID, db: DbSession) -> HTMLResponse:
    """A read-only page showing what the AI found for this catalogue: pictures, names,
    model codes, confidence, and anything worth checking. For looking things over with a
    client today - approving an item into the real catalogue is a later step (Phase 8),
    not something this page can do."""
    catalogue = load_catalogue(db, catalogue_id)
    nonce = secrets.token_urlsafe(16)
    page_html = (
        REVIEW_TEMPLATE.read_text(encoding="utf-8")
        .replace("__NONCE__", nonce)
        .replace("__CATALOGUE_ID__", str(catalogue_id))
        .replace("__CATALOGUE_NAME__", html.escape(catalogue.name))
    )
    return HTMLResponse(
        page_html,
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": (
                "default-src 'none'; img-src 'self'; connect-src 'self'; "
                f"script-src 'nonce-{nonce}'; style-src 'nonce-{nonce}'; "
                "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
            ),
        },
    )


class ProductCandidateList(Page):
    items: list[ProductCandidateSummaryOut]


def _candidate_query():  # noqa: ANN201
    return select(ProductCandidate).options(
        selectinload(ProductCandidate.pages),
        selectinload(ProductCandidate.detected_objects),
    )


def _field_value(candidate: ProductCandidate, key: str) -> str | None:
    field = candidate.fields.get(key)
    return field["value"] if field else None


def _thumbnail_url(catalogue_id: uuid.UUID, candidate: ProductCandidate) -> str | None:
    for detected in candidate.detected_objects:
        outline = detected.segmentation
        if outline is not None and outline.thumbnail_key:
            return _urls(catalogue_id, detected.id, outline).get("thumbnail")
    return None


def _summary(catalogue_id: uuid.UUID, candidate: ProductCandidate) -> ProductCandidateSummaryOut:
    return ProductCandidateSummaryOut(
        id=candidate.id,
        status=candidate.status,
        origin=candidate.origin,
        overall_confidence=candidate.overall_confidence,
        flags=candidate.flags,
        name=_field_value(candidate, "name"),
        model_code=_field_value(candidate, "model_code"),
        page_numbers=sorted(page.page_number for page in candidate.pages),
        thumbnail_url=_thumbnail_url(catalogue_id, candidate),
        created_at=candidate.created_at,
    )


@router.post("/{catalogue_id}/candidates/assemble", response_model=CandidateAssembleOut)
def assemble(
    catalogue_id: uuid.UUID,
    db: DbSession,
    settings: SettingsDep,
    admin: AdminUser,
    client: ClientInfoDep,
) -> CandidateAssembleOut:
    """Group this catalogue's finds into product candidates.

    Rebuilds every candidate the AI proposed that no one has reviewed yet; a candidate an
    admin has approved, rejected, merged, or edited is left untouched.
    """
    catalogue = load_catalogue(db, catalogue_id)
    try:
        result = assembly.assemble_candidates(db, settings, catalogue)
    except assembly.CatalogueNotReadyError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from None
    record_audit(
        db,
        AuditAction.CANDIDATES_ASSEMBLED,
        actor=admin,
        entity_type="catalogue",
        entity_id=str(catalogue_id),
        details={
            "candidates_created": result.candidates_created,
            "ready": result.ready,
            "needs_review": result.needs_review,
            "duplicate_flags_created": result.duplicate_flags_created,
        },
        ip_address=client.ip,
    )
    db.commit()
    return CandidateAssembleOut(**result.__dict__)


@router.get("/{catalogue_id}/candidates", response_model=ProductCandidateList)
def list_candidates(
    catalogue_id: uuid.UUID,
    db: DbSession,
    limit: int = Query(default=25, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> ProductCandidateList:
    """Every candidate for this catalogue, newest first."""
    load_catalogue(db, catalogue_id)
    base = _candidate_query().where(ProductCandidate.catalogue_id == catalogue_id)
    total = db.scalar(select(func.count()).select_from(base.subquery()))
    rows = list(
        db.scalars(base.order_by(ProductCandidate.created_at.desc()).limit(limit).offset(offset))
    )
    return ProductCandidateList(
        total=total or 0,
        limit=limit,
        offset=offset,
        items=[_summary(catalogue_id, row) for row in rows],
    )


@router.get("/{catalogue_id}/candidates/{candidate_id}", response_model=ProductCandidateOut)
def get_candidate(
    catalogue_id: uuid.UUID, candidate_id: uuid.UUID, db: DbSession
) -> ProductCandidateOut:
    """One candidate in full, including every find behind it and any duplicate flags."""
    load_catalogue(db, catalogue_id)
    candidate = db.scalar(
        _candidate_query().where(
            ProductCandidate.id == candidate_id, ProductCandidate.catalogue_id == catalogue_id
        )
    )
    if candidate is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Candidate not found")

    images: dict[uuid.UUID, dict[str, str]] = {}
    for detected in candidate.detected_objects:
        if detected.segmentation is not None:
            images[detected.id] = _urls(catalogue_id, detected.id, detected.segmentation)
        else:
            images[detected.id] = {}

    duplicate_flags = list(
        db.scalars(select(DuplicateFlag).where(DuplicateFlag.candidate_id == candidate.id))
    )
    return ProductCandidateOut(
        id=candidate.id,
        catalogue_id=candidate.catalogue_id,
        status=candidate.status,
        origin=candidate.origin,
        fields=candidate.fields,
        flags=candidate.flags,
        reasoning=candidate.reasoning,
        detection_confidence=candidate.detection_confidence,
        segmentation_confidence=candidate.segmentation_confidence,
        ocr_confidence=candidate.ocr_confidence,
        association_confidence=candidate.association_confidence,
        metadata_confidence=candidate.metadata_confidence,
        overall_confidence=candidate.overall_confidence,
        page_numbers=sorted(page.page_number for page in candidate.pages),
        detected_object_ids=[d.id for d in candidate.detected_objects],
        images=images,
        duplicate_flags=[
            DuplicateFlagOut(
                id=flag.id,
                other_candidate_id=flag.other_candidate_id,
                existing_product_id=flag.existing_product_id,
                signal=flag.signal.value,
                score=flag.score,
                status=flag.status.value,
                details=flag.details,
            )
            for flag in duplicate_flags
        ],
        created_at=candidate.created_at,
        updated_at=candidate.updated_at,
    )

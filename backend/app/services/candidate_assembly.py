"""Turning the products found in a catalogue's panels into candidates for an admin to review.

Run once a catalogue has processed (or partly processed) - typically after the admin has
finished any panel corrections, since re-running this throws away and rebuilds every
candidate that no one has looked at yet. A candidate an admin has approved, rejected, merged,
or otherwise touched (``reviewed_at`` is set, or its origin shows human involvement) is never
touched by this: only the AI's own untouched proposals are replaced.

See ``app.services.candidates`` for the actual grouping and scoring rules; this module is
just the database plumbing around it.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.models import (
    Catalogue,
    CataloguePage,
    DetectedObject,
    DuplicateFlag,
    OCRResult,
    ProductCandidate,
    SegmentationResult,
    SourceImage,
)
from app.models.candidate import candidate_pages
from app.models.enums import (
    CandidateStatus,
    CatalogueStatus,
    DataOrigin,
    DuplicateSignal,
    DuplicateStatus,
)
from app.services.candidates import CandidateDraft, Find, build_page_candidate
from app.services.settings_store import get_review_thresholds

logger = logging.getLogger(__name__)

# Only an AI candidate no one has acted on yet is safe to rebuild; anything reviewed, or
# entered/edited by a person, is left exactly as it is.
_REBUILDABLE = (ProductCandidate.origin == DataOrigin.AI) & (ProductCandidate.reviewed_at.is_(None))


class CatalogueNotReadyError(Exception):
    """The catalogue has not processed far enough for its candidates to be assembled."""


@dataclass(frozen=True)
class AssemblyResult:
    pages_with_finds: int
    candidates_created: int
    ready: int
    needs_review: int
    duplicate_flags_created: int


def assemble_candidates(
    session: Session, settings: Settings, catalogue: Catalogue
) -> AssemblyResult:
    if catalogue.status not in (CatalogueStatus.COMPLETED, CatalogueStatus.PARTIALLY_COMPLETED):
        raise CatalogueNotReadyError(
            f"Catalogue status is {catalogue.status.value}; it must have finished processing "
            "(at least partly) before its products can be grouped into candidates."
        )
    auto_review_threshold, low_confidence_threshold = get_review_thresholds(session, settings)

    session.execute(
        delete(ProductCandidate).where(ProductCandidate.catalogue_id == catalogue.id, _REBUILDABLE)
    )
    session.flush()

    # Whatever candidate rows are left for this catalogue now are ones we must not touch
    # (reviewed, or entered/edited by a person) - the pages they cover are not rebuilt, so a
    # page never ends up described by two overlapping candidates.
    kept_page_ids = set(
        session.scalars(
            select(candidate_pages.c.page_id)
            .join(ProductCandidate, ProductCandidate.id == candidate_pages.c.candidate_id)
            .where(ProductCandidate.catalogue_id == catalogue.id)
        )
    )

    pages = list(
        session.scalars(
            select(CataloguePage)
            .where(CataloguePage.catalogue_id == catalogue.id)
            .order_by(CataloguePage.page_number)
        )
    )
    created: list[ProductCandidate] = []
    for page in pages:
        if page.id in kept_page_ids:
            continue
        candidate = _assemble_page(
            session,
            page,
            auto_review_threshold=auto_review_threshold,
            low_confidence_threshold=low_confidence_threshold,
        )
        if candidate is not None:
            created.append(candidate)
    session.flush()  # every candidate has an id before duplicate-flagging links to it

    duplicate_flags_created = _flag_duplicates(session, created)
    ready = sum(1 for c in created if c.status is CandidateStatus.READY)
    return AssemblyResult(
        pages_with_finds=len(created),
        candidates_created=len(created),
        ready=ready,
        needs_review=len(created) - ready,
        duplicate_flags_created=duplicate_flags_created,
    )


def _assemble_page(
    session: Session,
    page: CataloguePage,
    *,
    auto_review_threshold: float,
    low_confidence_threshold: float,
) -> ProductCandidate | None:
    rows = session.execute(
        select(DetectedObject, SegmentationResult)
        .select_from(DetectedObject)
        .outerjoin(SegmentationResult, SegmentationResult.detected_object_id == DetectedObject.id)
        .join(SourceImage, SourceImage.id == DetectedObject.source_image_id)
        .where(SourceImage.page_id == page.id)
        .order_by(
            SourceImage.bbox_y0, SourceImage.bbox_x0, DetectedObject.bbox_y0, DetectedObject.bbox_x0
        )
    ).all()
    if not rows:
        return None

    finds = [
        Find(
            id=detected.id,
            panel_id=detected.source_image_id,
            label=detected.label,
            confidence=detected.confidence,
            outline_status=outline.status if outline else None,
            outline_score=outline.confidence if outline else None,
        )
        for detected, outline in rows
    ]
    ocr_rows = list(
        session.scalars(
            select(OCRResult).where(OCRResult.page_id == page.id).order_by(OCRResult.line_index)
        )
    )
    draft = build_page_candidate(
        finds,
        [r.text for r in ocr_rows],
        [r.confidence for r in ocr_rows],
        auto_review_threshold=auto_review_threshold,
        low_confidence_threshold=low_confidence_threshold,
    )
    if draft is None:  # pragma: no cover - rows is non-empty, so build_page_candidate cannot
        return None    # return None here; guarded for safety if that ever changes

    detected_by_id = {detected.id: detected for detected, _outline in rows}
    panel_ids = {detected.source_image_id for detected, _outline in rows}
    panels = list(
        session.scalars(select(SourceImage).where(SourceImage.id.in_(panel_ids)))
    )

    candidate = _to_candidate(page.catalogue_id, draft)
    candidate.pages = [page]
    candidate.source_images = panels
    candidate.detected_objects = [
        detected_by_id[object_id] for object_id in draft.detected_object_ids
    ]
    session.add(candidate)
    return candidate


def _to_candidate(catalogue_id: uuid.UUID, draft: CandidateDraft) -> ProductCandidate:
    return ProductCandidate(
        catalogue_id=catalogue_id,
        status=draft.status,
        origin=DataOrigin.AI,
        fields=draft.fields,
        flags=draft.flags,
        reasoning=draft.reasoning,
        detection_confidence=draft.detection_confidence,
        segmentation_confidence=draft.segmentation_confidence,
        ocr_confidence=draft.ocr_confidence,
        association_confidence=draft.association_confidence,
        metadata_confidence=draft.metadata_confidence,
        overall_confidence=draft.overall_confidence,
    )


def _model_code_of(candidate: ProductCandidate) -> str | None:
    field = candidate.fields.get("model_code")
    return field["value"] if field else None


def _flag_duplicates(session: Session, candidates: list[ProductCandidate]) -> int:
    """Candidates in this same run that printed the same model code: almost certainly
    the same product (e.g. a page processed twice, or a series shown on two pages)."""
    by_code: dict[str, list[ProductCandidate]] = {}
    for candidate in candidates:
        code = _model_code_of(candidate)
        if code:
            by_code.setdefault(code, []).append(candidate)

    created = 0
    for code, group in by_code.items():
        if len(group) < 2:
            continue
        first, rest = group[0], group[1:]
        for other in rest:
            session.add(
                DuplicateFlag(
                    candidate_id=other.id,
                    other_candidate_id=first.id,
                    signal=DuplicateSignal.SKU,
                    score=1.0,
                    status=DuplicateStatus.OPEN,
                    details={"model_code": code},
                )
            )
            created += 1
    return created

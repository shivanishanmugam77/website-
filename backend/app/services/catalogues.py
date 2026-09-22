"""Catalogue ingestion: accept an upload, store it, record it, queue processing."""

from __future__ import annotations

import itertools
import logging
import uuid
from collections.abc import Iterable

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, selectinload

from app.core.config import Settings
from app.models import (
    Catalogue,
    CataloguePage,
    ProcessingJob,
    SourceImage,
    Supplier,
    User,
)
from app.models.enums import (
    AuditAction,
    CatalogueStatus,
    JobStatus,
    JobType,
    SourceImageKind,
)
from app.services.audit import record_audit
from app.services.panel_history import count_corrections
from app.services.pipeline import catalogue_prefix, original_pdf_key
from app.services.sessions import utcnow
from app.services.storage import ObjectTooLargeError, Storage
from app.utils.files import default_catalogue_name, has_pdf_signature, sanitize_filename

logger = logging.getLogger(__name__)

ACTIVE_JOB_STATUSES = (JobStatus.QUEUED, JobStatus.RUNNING)
ENQUEUE_FAILED_MESSAGE = (
    "Could not queue processing (is the worker running?). Use reprocess to retry."
)


class SupplierNotFoundError(Exception):
    pass


class NotAPdfError(Exception):
    pass


class UploadTooLargeError(Exception):
    pass


class DuplicateCatalogueError(Exception):
    def __init__(self, existing_id: uuid.UUID) -> None:
        super().__init__(str(existing_id))
        self.existing_id = existing_id


class CorrectionsWouldBeLostError(Exception):
    """Reprocessing would throw away manual panel corrections the admin has not agreed to lose."""

    def __init__(self, count: int) -> None:
        super().__init__(f"{count} manual correction(s) would be discarded")
        self.count = count


class CatalogueBusyError(Exception):
    """A processing job is already queued or running for this catalogue."""


def enqueue_processing(catalogue_id: uuid.UUID, job_id: uuid.UUID) -> str:
    """Hand the job to a worker. Returns the Celery task id.

    Imported lazily so this module can be imported without the Celery app (and so tests
    can replace this one function).
    """
    from app.core.constants import QUEUE_PIPELINE
    from app.workers.tasks.pipeline import process_catalogue

    result = process_catalogue.apply_async(
        args=[str(catalogue_id), str(job_id)], queue=QUEUE_PIPELINE
    )
    return str(result.id)


def _enqueue_or_fail(session: Session, catalogue_id: uuid.UUID, job_id: uuid.UUID) -> None:
    """Queue the job *after* it is committed; if the broker is down, record that visibly
    instead of losing the upload or leaving a job that will never run."""
    try:
        task_id = enqueue_processing(catalogue_id, job_id)
        job = session.get(ProcessingJob, job_id)
        job.celery_task_id = task_id
        session.commit()
    except Exception:
        logger.exception("enqueue_failed", extra={"catalogue_id": str(catalogue_id)})
        session.rollback()
        job = session.get(ProcessingJob, job_id)
        catalogue = session.get(Catalogue, catalogue_id)
        job.status = JobStatus.FAILED
        job.error = ENQUEUE_FAILED_MESSAGE
        job.finished_at = utcnow()
        catalogue.status = CatalogueStatus.FAILED
        catalogue.processing_error = ENQUEUE_FAILED_MESSAGE
        session.commit()


def _new_job(catalogue: Catalogue, actor: User) -> ProcessingJob:
    return ProcessingJob(
        catalogue_id=catalogue.id,
        job_type=JobType.CATALOGUE_PROCESS,
        status=JobStatus.QUEUED,
        created_by_id=actor.id,
    )


def has_active_job(session: Session, catalogue_id: uuid.UUID) -> bool:
    return bool(
        session.scalar(
            select(func.count())
            .select_from(ProcessingJob)
            .where(
                ProcessingJob.catalogue_id == catalogue_id,
                ProcessingJob.status.in_(ACTIVE_JOB_STATUSES),
            )
        )
    )


def ingest_catalogue(
    session: Session,
    storage: Storage,
    settings: Settings,
    *,
    chunks: Iterable[bytes],
    filename: str | None,
    supplier_id: uuid.UUID | None,
    name: str | None,
    allow_duplicate: bool,
    actor: User,
    ip: str | None,
) -> tuple[Catalogue, ProcessingJob]:
    """Store an uploaded PDF and queue its processing.

    The API process only sniffs the header and streams bytes to storage; the PDF is never
    *parsed* here (parsing untrusted PDFs is done in the worker, where a parser crash cannot
    take the API down).
    """
    supplier: Supplier | None = None
    if supplier_id is not None:
        supplier = session.get(Supplier, supplier_id)
        if supplier is None:
            raise SupplierNotFoundError

    iterator = iter(chunks)
    first = next(iterator, b"")
    if not has_pdf_signature(first):
        raise NotAPdfError

    catalogue_id = uuid.uuid4()
    key = original_pdf_key(catalogue_id)
    try:
        stored = storage.put_stream(
            key, itertools.chain([first], iterator), max_bytes=settings.max_upload_bytes
        )
    except ObjectTooLargeError:
        raise UploadTooLargeError from None

    if not allow_duplicate:
        existing_id = session.scalar(
            select(Catalogue.id).where(Catalogue.sha256 == stored.sha256).limit(1)
        )
        if existing_id is not None:
            storage.delete_prefix(catalogue_prefix(catalogue_id))
            raise DuplicateCatalogueError(existing_id)

    display_name = sanitize_filename(filename)
    catalogue = Catalogue(
        id=catalogue_id,
        supplier_id=supplier.id if supplier else None,
        name=(name or "").strip() or default_catalogue_name(display_name),
        original_filename=display_name,
        storage_key=key,
        file_size_bytes=stored.size,
        sha256=stored.sha256,
        status=CatalogueStatus.UPLOADED,
        uploaded_by_id=actor.id,
    )
    session.add(catalogue)
    session.flush()
    job = _new_job(catalogue, actor)
    session.add(job)
    record_audit(
        session,
        AuditAction.UPLOAD_CATALOGUE,
        actor=actor,
        entity_type="catalogue",
        entity_id=str(catalogue.id),
        details={
            "filename": display_name,
            "size_bytes": stored.size,
            "sha256": stored.sha256,
            "supplier_id": str(supplier.id) if supplier else None,
        },
        ip_address=ip,
    )
    try:
        session.commit()
    except Exception:
        storage.delete_prefix(catalogue_prefix(catalogue_id))  # do not orphan the stored file
        raise
    _enqueue_or_fail(session, catalogue.id, job.id)
    return catalogue, job


def reprocess_catalogue(
    session: Session,
    catalogue: Catalogue,
    *,
    actor: User,
    ip: str | None,
    force: bool = False,
    discard_corrections: bool = False,
) -> ProcessingJob:
    """Queue a fresh run. Refused while one is queued/running unless ``force`` (e.g. the
    worker died and left a job stuck), and refused while manual panel corrections exist
    unless ``discard_corrections`` (a fresh run rebuilds every page and loses them)."""
    if not force and has_active_job(session, catalogue.id):
        raise CatalogueBusyError
    corrections = count_corrections(session, catalogue.id)
    if corrections and not discard_corrections:
        raise CorrectionsWouldBeLostError(corrections)
    job = _new_job(catalogue, actor)
    session.add(job)
    catalogue.status = CatalogueStatus.UPLOADED
    catalogue.processing_progress = 0
    catalogue.processing_error = None
    record_audit(
        session,
        AuditAction.REPROCESS_CATALOGUE,
        actor=actor,
        entity_type="catalogue",
        entity_id=str(catalogue.id),
        details={"forced": force},
        ip_address=ip,
    )
    if corrections:
        record_audit(
            session,
            AuditAction.CORRECTIONS_DISCARDED,
            actor=actor,
            entity_type="catalogue",
            entity_id=str(catalogue.id),
            details={"corrections": corrections},
            ip_address=ip,
        )
    session.commit()
    _enqueue_or_fail(session, catalogue.id, job.id)
    return job


def delete_catalogue(
    session: Session,
    storage: Storage,
    catalogue: Catalogue,
    *,
    actor: User,
    ip: str | None,
    force: bool = False,
) -> None:
    """Delete a catalogue, its derived data and its files.

    Published products survive (their source links become NULL; see docs/database.md).
    """
    if not force and has_active_job(session, catalogue.id):
        raise CatalogueBusyError
    catalogue_id = catalogue.id
    record_audit(
        session,
        AuditAction.DELETE_CATALOGUE,
        actor=actor,
        entity_type="catalogue",
        entity_id=str(catalogue_id),
        details={
            "name": catalogue.name,
            "filename": catalogue.original_filename,
            "sha256": catalogue.sha256,
            "page_count": catalogue.page_count,
        },
        ip_address=ip,
    )
    session.execute(delete(Catalogue).where(Catalogue.id == catalogue_id))
    session.commit()
    # Files last: if this fails we have orphaned files (harmless), never a catalogue row
    # pointing at files that are gone.
    storage.delete_prefix(catalogue_prefix(catalogue_id))


# ---------------------------------------------------------------------------------- queries
def list_catalogues(
    session: Session,
    *,
    status: CatalogueStatus | None,
    supplier_id: uuid.UUID | None,
    search: str | None,
    limit: int,
    offset: int,
) -> tuple[list[Catalogue], int]:
    filters = []
    if status is not None:
        filters.append(Catalogue.status == status)
    if supplier_id is not None:
        filters.append(Catalogue.supplier_id == supplier_id)
    if search:
        escaped = search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        filters.append(Catalogue.name.ilike(f"%{escaped}%", escape="\\"))
    total = session.scalar(select(func.count()).select_from(Catalogue).where(*filters)) or 0
    rows = session.scalars(
        select(Catalogue)
        .where(*filters)
        .options(selectinload(Catalogue.supplier))
        .order_by(Catalogue.created_at.desc(), Catalogue.id)
        .limit(limit)
        .offset(offset)
    ).all()
    return list(rows), total


def latest_job(session: Session, catalogue_id: uuid.UUID) -> ProcessingJob | None:
    return session.scalar(
        select(ProcessingJob)
        .where(ProcessingJob.catalogue_id == catalogue_id)
        .order_by(ProcessingJob.created_at.desc(), ProcessingJob.id)
        .limit(1)
    )


def page_status_counts(session: Session, catalogue_id: uuid.UUID) -> dict[str, int]:
    rows = session.execute(
        select(CataloguePage.status, func.count())
        .where(CataloguePage.catalogue_id == catalogue_id)
        .group_by(CataloguePage.status)
    ).all()
    return {status.value: count for status, count in rows}


def embedded_image_counts(session: Session, page_ids: list[uuid.UUID]) -> dict[uuid.UUID, int]:
    if not page_ids:
        return {}
    rows = session.execute(
        select(SourceImage.page_id, func.count())
        .where(SourceImage.page_id.in_(page_ids), SourceImage.kind == SourceImageKind.EMBEDDED)
        .group_by(SourceImage.page_id)
    ).all()
    return dict(rows)


def region_counts(session: Session, page_ids: list[uuid.UUID]) -> dict[uuid.UUID, int]:
    """Photo panels found per page."""
    if not page_ids:
        return {}
    rows = session.execute(
        select(SourceImage.page_id, func.count())
        .where(SourceImage.page_id.in_(page_ids), SourceImage.kind == SourceImageKind.REGION)
        .group_by(SourceImage.page_id)
    ).all()
    return dict(rows)


def count_source_images(session: Session, catalogue_id: uuid.UUID) -> int:
    return (
        session.scalar(
            select(func.count())
            .select_from(SourceImage)
            .join(CataloguePage, SourceImage.page_id == CataloguePage.id)
            .where(
                CataloguePage.catalogue_id == catalogue_id,
                SourceImage.kind == SourceImageKind.EMBEDDED,
            )
        )
        or 0
    )

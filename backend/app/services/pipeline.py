"""The catalogue processing pipeline (worker side).

Stage 1 (Phase 3) renders every page to an image and extracts the raster images embedded
in the PDF. Stage 2 (Phase 4) reads the text on each page and decides what kind of page it
is. Stage 3 (Phase 5a) cuts each page into its photo panels. Stage 4 (Phase 5b) finds the
products inside every panel, outlines each one and saves its pictures. Later phases add stages
after these (candidate assembly) without changing how jobs, progress or failures work.

Design rules:
* Idempotent: running it again on the same catalogue wipes its previous results first.
* One bad page never loses the catalogue: failures are recorded per page.
* A page's rows are written in a savepoint and its files are tracked, so a failure leaves
  neither half-written rows nor orphaned files.
* A page whose text cannot be read (or whose panels cannot be cut, or whose products cannot be
  found) keeps what it has: its
  status stays at the last stage it completed, the error is recorded on the page, and the
  catalogue ends up PARTIALLY_COMPLETED. Later stages skip pages that missed an earlier one.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from io import BytesIO
from typing import Any

from celery.exceptions import SoftTimeLimitExceeded
from PIL import Image
from sqlalchemy import Connection, delete, func, select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.ml.detection import DetectionProvider, get_detection_provider
from app.ml.errors import ModelUnavailableError
from app.ml.ocr import OcrEngineError, OcrProvider, get_ocr_provider
from app.ml.segmentation import SegmentationProvider, get_segmentation_provider
from app.models import (
    Catalogue,
    CataloguePage,
    DetectedObject,
    OCRResult,
    ProcessingJob,
    SegmentationResult,
    SourceImage,
)
from app.models.enums import CatalogueStatus, JobStatus, PageStatus, PageType, SourceImageKind
from app.services.detections import parse_prompts
from app.services.ocr import read_page_text
from app.services.page_classifier import classify_page
from app.services.panels import detect_panels
from app.services.pdf import (
    PdfFormatError,
    dhash,
    encode_jpeg,
    iter_embedded_images,
    make_thumbnail,
    open_pdf,
    render_page,
    render_scale,
)
from app.services.product_finding import Found, find_products, render_find
from app.services.sessions import utcnow
from app.services.storage import ObjectNotFoundError, Storage
from app.services.text_signals import extract_signals

logger = logging.getLogger(__name__)

STAGE_RENDER = "render_pages"
STAGE_TEXT = "read_text"
STAGE_PANELS = "find_panels"
STAGE_PRODUCTS = "find_products"
# Pages of these types show no products, so cutting them into panels would only make noise.
PAGE_TYPES_WITHOUT_PANELS = frozenset({PageType.COVER})
# A page that reached any of these has a usable picture for later stages.
USABLE_STATUSES = (PageStatus.RENDERED, PageStatus.TEXT_READ, PageStatus.PANELS_FOUND)
THUMBNAIL_QUALITY = 85
# The two models that find products: the finder names them, the outliner draws their edges.
ProductModels = tuple[DetectionProvider, SegmentationProvider]


class PipelineFailure(Exception):
    """A controlled failure of the whole catalogue; its message is shown to admins."""


@dataclass(frozen=True)
class PipelineLock:
    connection: Connection
    key: int


# ------------------------------------------------------------------------------ storage keys
def catalogue_prefix(catalogue_id: uuid.UUID) -> str:
    return f"catalogues/{catalogue_id}"


def original_pdf_key(catalogue_id: uuid.UUID) -> str:
    return f"{catalogue_prefix(catalogue_id)}/original.pdf"


def page_image_key(catalogue_id: uuid.UUID, page_number: int) -> str:
    return f"{catalogue_prefix(catalogue_id)}/pages/{page_number:04d}.jpg"


def page_thumbnail_key(catalogue_id: uuid.UUID, page_number: int) -> str:
    return f"{catalogue_prefix(catalogue_id)}/pages/{page_number:04d}_thumb.jpg"


def source_image_key(catalogue_id: uuid.UUID, image_id: uuid.UUID) -> str:
    return f"{catalogue_prefix(catalogue_id)}/source-images/{image_id}.jpg"


def product_file_key(
    catalogue_id: uuid.UUID, object_id: uuid.UUID, kind: str, extension: str
) -> str:
    """One picture of a product found in a panel (kind: crop, mask, cutout, white, thumb)."""
    return f"{catalogue_prefix(catalogue_id)}/products/{object_id}/{kind}.{extension}"


# ------------------------------------------------------------------------------ entry point
def run_catalogue_pipeline(
    session: Session,
    storage: Storage,
    settings: Settings,
    *,
    catalogue_id: uuid.UUID,
    job_id: uuid.UUID,
    ocr_provider: OcrProvider | None = None,
    product_models: ProductModels | None = None,
) -> None:
    """Run every stage for one catalogue. ``ocr_provider`` and ``product_models`` override the
    configured engines (tests use them); each is ignored when its stage is switched off."""
    catalogue = session.get(Catalogue, catalogue_id)
    job = session.get(ProcessingJob, job_id)
    if catalogue is None or job is None:
        logger.warning("pipeline_target_missing", extra={"catalogue_id": str(catalogue_id)})
        return  # deleted while queued: nothing to do

    # Two runs on one catalogue would delete each other's pages, so only one may run at a time.
    lock = _acquire_lock(session, catalogue_lock_key(catalogue_id))
    if lock is None:
        logger.warning("pipeline_already_running", extra={"catalogue_id": str(catalogue_id)})
        job.status = JobStatus.FAILED
        job.error = (
            "This catalogue is already being processed by another job; "
            "start again once it has finished"
        )
        job.finished_at = utcnow()
        session.commit()
        return  # the catalogue itself is left exactly as the running job has it
    try:
        _run(session, storage, settings, catalogue, job, ocr_provider, product_models)
    except PipelineFailure as exc:
        _mark_failed(session, catalogue_id, job_id, str(exc))
    except Exception:
        logger.exception("pipeline_crashed", extra={"catalogue_id": str(catalogue_id)})
        _mark_failed(
            session, catalogue_id, job_id, "Unexpected error while processing (see server logs)"
        )
        raise  # let Celery record the failure too
    finally:
        _release_lock(lock)


def catalogue_lock_key(catalogue_id: uuid.UUID) -> int:
    """The PostgreSQL advisory-lock number for a catalogue (a signed 64-bit integer)."""
    return int.from_bytes(catalogue_id.bytes[:8], "big", signed=True)


def _acquire_lock(session: Session, lock_key: int) -> PipelineLock | None:
    """Take the catalogue's lock, or return None when another run holds it.

    The lock lives on a connection of its own, not on the session's: the session hands its
    connection back to the pool at every commit, and a PostgreSQL advisory lock belongs to
    the connection that took it. It is released explicitly when the run ends and, should the
    worker die, automatically when its connection closes.
    """
    connection = session.get_bind().engine.connect().execution_options(isolation_level="AUTOCOMMIT")
    try:
        if connection.execute(select(func.pg_try_advisory_lock(lock_key))).scalar():
            return PipelineLock(connection, lock_key)
    except Exception:
        connection.close()
        raise
    connection.close()
    return None


def _release_lock(lock: PipelineLock) -> None:
    try:
        lock.connection.execute(select(func.pg_advisory_unlock(lock.key)))
    except Exception:
        # A dead connection drops its locks anyway; never hide the real outcome behind this.
        logger.warning("pipeline_lock_release_failed", extra={"lock_key": lock.key})
    finally:
        lock.connection.close()


def _mark_failed(
    session: Session, catalogue_id: uuid.UUID, job_id: uuid.UUID, message: str
) -> None:
    session.rollback()  # discard whatever the failed run left uncommitted
    catalogue = session.get(Catalogue, catalogue_id)
    job = session.get(ProcessingJob, job_id)
    now = utcnow()
    if catalogue is not None:
        catalogue.status = CatalogueStatus.FAILED
        catalogue.processing_error = message[:2000]
        catalogue.current_stage = None
    if job is not None:
        job.status = JobStatus.FAILED
        job.error = message[:2000]
        job.finished_at = now
        job.current_stage = None
    session.commit()


def _load_provider(settings: Settings, override: OcrProvider | None) -> OcrProvider | None:
    if not settings.ocr_enabled:
        return None
    if override is not None:
        return override
    try:
        return get_ocr_provider(settings)
    except OcrEngineError as exc:
        logger.error("ocr_engine_unavailable", extra={"reason": str(exc)})
        raise PipelineFailure(
            "Text recognition is switched on but its engine could not be started "
            "(see the worker log). Fix the engine or set OCR_ENABLED=false."
        ) from None


def _products_enabled(settings: Settings) -> bool:
    return settings.panel_detection_enabled and settings.product_detection_enabled


def _load_product_models(
    settings: Settings, override: ProductModels | None
) -> ProductModels | None:
    if not _products_enabled(settings):
        return None
    if override is not None:
        return override
    try:
        return get_detection_provider(settings), get_segmentation_provider(settings)
    except ModelUnavailableError as exc:
        logger.error("product_models_unavailable", extra={"reason": str(exc)})
        raise PipelineFailure(
            f"Product finding is switched on but its models could not be loaded: {exc} "
            "(or set PRODUCT_DETECTION_ENABLED=false)"
        ) from None


def _run(
    session: Session,
    storage: Storage,
    settings: Settings,
    catalogue: Catalogue,
    job: ProcessingJob,
    ocr_override: OcrProvider | None,
    product_override: ProductModels | None = None,
) -> None:
    catalogue_id = catalogue.id
    job.status = JobStatus.RUNNING
    job.started_at = utcnow()
    job.finished_at = None
    job.error = None
    job.progress = 0
    job.pages_processed = 0
    job.current_stage = STAGE_RENDER
    catalogue.status = CatalogueStatus.PROCESSING
    catalogue.processing_progress = 0
    catalogue.processing_error = None
    catalogue.current_stage = STAGE_RENDER
    session.commit()

    # Fail fast if the text engine is broken, before minutes are spent rendering pages.
    provider = _load_provider(settings, ocr_override)
    models = _load_product_models(settings, product_override)

    # Fresh start: drop the artefacts of any earlier run (database rows cascade).
    session.execute(delete(CataloguePage).where(CataloguePage.catalogue_id == catalogue_id))
    prefix = catalogue_prefix(catalogue_id)
    storage.delete_prefix(f"{prefix}/pages")
    storage.delete_prefix(f"{prefix}/source-images")
    storage.delete_prefix(f"{prefix}/products")
    session.commit()

    try:
        with storage.local_copy(catalogue.storage_key) as pdf_path:
            document = open_pdf(pdf_path)
            try:
                total = len(document)
                if total == 0:
                    raise PipelineFailure("The PDF contains no pages")
                if total > settings.max_pdf_pages:
                    raise PipelineFailure(
                        f"The PDF has {total} pages; the limit is {settings.max_pdf_pages}"
                    )
                catalogue.page_count = total
                job.total_pages = total
                session.commit()

                # Progress counts one unit per page per stage; 100 is reserved for "finished".
                units = total * _stage_count(settings, provider)
                for index in range(total):
                    _process_page(session, storage, settings, catalogue, document, index)
                    done = index + 1
                    progress = min(99, done * 100 // units)
                    job.pages_processed = done
                    job.progress = progress
                    catalogue.processing_progress = progress
                    session.commit()
            finally:
                document.close()
    except ObjectNotFoundError:
        raise PipelineFailure("The stored PDF file is missing") from None
    except PdfFormatError as exc:
        raise PipelineFailure(str(exc)) from None

    usable = _count_pages(session, catalogue_id, *USABLE_STATUSES)
    if not usable:
        raise PipelineFailure("None of the pages could be processed")

    units = total * _stage_count(settings, provider)
    if provider is not None:
        _read_text_stage(session, storage, settings, catalogue, job, provider, total, units)
    if settings.panel_detection_enabled:
        offset = total * (2 if provider is not None else 1)
        _find_panels_stage(
            session, storage, settings, catalogue, job, provider is not None, offset, units
        )
    if models is not None:
        offset = total * ((2 if provider is not None else 1) + 1)
        _find_products_stage(session, storage, settings, catalogue, job, models, offset, units)
    finished_status = _final_status(settings, provider)
    finished = _count_pages(session, catalogue_id, finished_status)

    now = utcnow()
    catalogue.status = (
        CatalogueStatus.COMPLETED if finished == total else CatalogueStatus.PARTIALLY_COMPLETED
    )
    catalogue.processing_progress = 100
    catalogue.current_stage = None
    job.status = JobStatus.SUCCEEDED
    job.progress = 100
    job.current_stage = None
    job.finished_at = now
    session.commit()
    logger.info(
        "pipeline_finished",
        extra={
            "catalogue_id": str(catalogue_id),
            "pages": total,
            "rendered": usable,
            "finished": finished,
        },
    )


def _stage_count(settings: Settings, provider: OcrProvider | None) -> int:
    return (
        1
        + (provider is not None)
        + settings.panel_detection_enabled
        + _products_enabled(settings)
    )


def _final_status(settings: Settings, provider: OcrProvider | None) -> PageStatus:
    """The status a page has once every stage that is switched on has succeeded."""
    if _products_enabled(settings):
        return PageStatus.COMPLETED
    if settings.panel_detection_enabled:
        return PageStatus.PANELS_FOUND
    return PageStatus.TEXT_READ if provider is not None else PageStatus.RENDERED


def _count_pages(session: Session, catalogue_id: uuid.UUID, *statuses: PageStatus) -> int:
    return (
        session.scalar(
            select(func.count())
            .select_from(CataloguePage)
            .where(CataloguePage.catalogue_id == catalogue_id, CataloguePage.status.in_(statuses))
        )
        or 0
    )


# ------------------------------------------------------------------------------ one page
def _process_page(
    session: Session,
    storage: Storage,
    settings: Settings,
    catalogue: Catalogue,
    document: Any,
    index: int,
) -> bool:
    number = index + 1
    written: list[str] = []  # keys created for this page, removed again if the page fails

    def store(key: str, data: bytes) -> Any:
        stored = storage.put_bytes(key, data)
        written.append(key)
        return stored

    try:
        with session.begin_nested():
            page_row = CataloguePage(
                catalogue_id=catalogue.id, page_number=number, status=PageStatus.PROCESSING
            )
            session.add(page_row)
            session.flush()

            page = document[index]
            try:
                width_pt, height_pt = page.get_size()
                scale = render_scale(
                    width_pt, height_pt, settings.render_dpi, settings.max_page_pixels
                )
                rendered = render_page(page, scale)

                image_key = page_image_key(catalogue.id, number)
                stored_page = store(image_key, encode_jpeg(rendered.image, settings.jpeg_quality))
                thumb_key = page_thumbnail_key(catalogue.id, number)
                thumbnail = make_thumbnail(rendered.image, settings.thumbnail_max_px)
                store(thumb_key, encode_jpeg(thumbnail, THUMBNAIL_QUALITY))
                page_row.image_key = image_key
                page_row.thumbnail_key = thumb_key
                page_row.width_px, page_row.height_px = rendered.image.size

                # The whole page is itself a source image: later stages detect products on it.
                session.add(
                    SourceImage(
                        page_id=page_row.id,
                        kind=SourceImageKind.PAGE,
                        storage_key=image_key,
                        width_px=rendered.image.width,
                        height_px=rendered.image.height,
                        sha256=stored_page.sha256,
                        phash=dhash(rendered.image),
                    )
                )

                for embedded in iter_embedded_images(
                    page, rendered, min_px=settings.min_embedded_image_px
                ):
                    image_id = uuid.uuid4()
                    key = source_image_key(catalogue.id, image_id)
                    stored = store(key, encode_jpeg(embedded.image, settings.jpeg_quality))
                    bbox = embedded.bbox or (None, None, None, None)
                    session.add(
                        SourceImage(
                            id=image_id,
                            page_id=page_row.id,
                            kind=SourceImageKind.EMBEDDED,
                            storage_key=key,
                            width_px=embedded.image.width,
                            height_px=embedded.image.height,
                            bbox_x0=bbox[0],
                            bbox_y0=bbox[1],
                            bbox_x1=bbox[2],
                            bbox_y1=bbox[3],
                            sha256=stored.sha256,
                            phash=dhash(embedded.image),
                        )
                    )
            finally:
                page.close()
            page_row.status = PageStatus.RENDERED
        return True
    except SoftTimeLimitExceeded:
        raise  # the whole job is out of time: do not swallow it as a "bad page"
    except Exception as exc:
        logger.exception(
            "page_failed", extra={"catalogue_id": str(catalogue.id), "page_number": number}
        )
        for key in written:
            storage.delete(key)
        session.add(
            CataloguePage(
                catalogue_id=catalogue.id,
                page_number=number,
                status=PageStatus.FAILED,
                error_stage=STAGE_RENDER,
                processing_error=f"{type(exc).__name__}: {str(exc)[:300]}",
            )
        )
        session.flush()
        return False


# ------------------------------------------------------------------------------ stage 2: text
def _read_text_stage(
    session: Session,
    storage: Storage,
    settings: Settings,
    catalogue: Catalogue,
    job: ProcessingJob,
    provider: OcrProvider,
    total: int,
    units: int,
) -> None:
    catalogue.current_stage = STAGE_TEXT
    job.current_stage = STAGE_TEXT
    session.commit()

    pages = list(
        session.scalars(
            select(CataloguePage)
            .where(
                CataloguePage.catalogue_id == catalogue.id,
                CataloguePage.status == PageStatus.RENDERED,
            )
            .order_by(CataloguePage.page_number)
        )
    )
    for done, page_row in enumerate(pages, start=1):
        _read_page_text(session, storage, settings, provider, page_row)
        progress = min(99, (total + done) * 100 // units)
        job.progress = progress
        catalogue.processing_progress = progress
        session.commit()


def _read_page_text(
    session: Session,
    storage: Storage,
    settings: Settings,
    provider: OcrProvider,
    page_row: CataloguePage,
) -> bool:
    number = page_row.page_number
    try:
        if page_row.image_key is None:
            raise ValueError("the page image is missing")
        with Image.open(BytesIO(storage.read_bytes(page_row.image_key))) as opened:
            image = opened.convert("RGB")
        lines = read_page_text(
            image,
            provider,
            min_confidence=settings.ocr_min_confidence,
            tile_px=settings.ocr_tile_px,
            overlap_px=settings.ocr_tile_overlap_px,
            max_lines=settings.ocr_max_lines_per_page,
        )
        signals = extract_signals([line.text for line in lines])
        outcome = classify_page(page_number=number, line_count=len(lines), signals=signals)

        # Only the new rows live in the savepoint: if saving them fails, nothing of this page
        # is written and the page row itself was never touched.
        with session.begin_nested():
            session.add_all(
                OCRResult(
                    page_id=page_row.id,
                    line_index=index,
                    text=line.text,
                    confidence=line.confidence,
                    language=line.language,
                    polygon=[[x, y] for x, y in line.polygon],
                    bbox_x0=line.bbox[0],
                    bbox_y0=line.bbox[1],
                    bbox_x1=line.bbox[2],
                    bbox_y1=line.bbox[3],
                    engine=provider.name,
                    engine_version=provider.version,
                )
                for index, line in enumerate(lines)
            )
        page_row.page_type = outcome.page_type
        page_row.page_type_confidence = outcome.confidence
        page_row.status = PageStatus.TEXT_READ
        page_row.error_stage = None
        page_row.processing_error = None
        return True
    except SoftTimeLimitExceeded:
        raise
    except Exception as exc:
        logger.exception(
            "page_text_failed", extra={"page_id": str(page_row.id), "page_number": number}
        )
        # The rendered page stays usable; the reason is recorded on it for the admin.
        page_row.error_stage = STAGE_TEXT
        page_row.processing_error = f"{type(exc).__name__}: {str(exc)[:300]}"
        session.flush()
        return False


# ------------------------------------------------------------------------------ stage 3: panels
def _find_panels_stage(
    session: Session,
    storage: Storage,
    settings: Settings,
    catalogue: Catalogue,
    job: ProcessingJob,
    text_was_read: bool,
    offset: int,
    units: int,
) -> None:
    catalogue.current_stage = STAGE_PANELS
    job.current_stage = STAGE_PANELS
    session.commit()

    # Only pages that completed the stage before this one: a page whose text failed to read
    # keeps its error visible instead of being carried forward.
    previous = PageStatus.TEXT_READ if text_was_read else PageStatus.RENDERED
    pages = list(
        session.scalars(
            select(CataloguePage)
            .where(CataloguePage.catalogue_id == catalogue.id, CataloguePage.status == previous)
            .order_by(CataloguePage.page_number)
        )
    )
    for done, page_row in enumerate(pages, start=1):
        _find_page_panels(session, storage, settings, catalogue, page_row)
        progress = min(99, (offset + done) * 100 // units)
        job.progress = progress
        catalogue.processing_progress = progress
        session.commit()


def _find_page_panels(
    session: Session,
    storage: Storage,
    settings: Settings,
    catalogue: Catalogue,
    page_row: CataloguePage,
) -> bool:
    number = page_row.page_number
    written: list[str] = []
    try:
        if page_row.image_key is None:
            raise ValueError("the page image is missing")
        panels = []
        image = None
        if page_row.page_type not in PAGE_TYPES_WITHOUT_PANELS:
            with Image.open(BytesIO(storage.read_bytes(page_row.image_key))) as opened:
                image = opened.convert("RGB")
            text_boxes = [
                (row.bbox_x0, row.bbox_y0, row.bbox_x1, row.bbox_y1)
                for row in session.scalars(
                    select(OCRResult).where(OCRResult.page_id == page_row.id)
                )
            ]
            panels = detect_panels(
                image, text_boxes, min_area_share=settings.panel_min_area_share
            )

        with session.begin_nested():
            for panel in panels:
                crop = image.crop(panel.bbox)
                image_id = uuid.uuid4()
                key = source_image_key(catalogue.id, image_id)
                stored = storage.put_bytes(key, encode_jpeg(crop, settings.jpeg_quality))
                written.append(key)
                session.add(
                    SourceImage(
                        id=image_id,
                        page_id=page_row.id,
                        kind=SourceImageKind.REGION,
                        storage_key=key,
                        width_px=crop.width,
                        height_px=crop.height,
                        bbox_x0=panel.bbox[0],
                        bbox_y0=panel.bbox[1],
                        bbox_x1=panel.bbox[2],
                        bbox_y1=panel.bbox[3],
                        sha256=stored.sha256,
                        phash=dhash(crop),
                    )
                )
        page_row.status = PageStatus.PANELS_FOUND
        page_row.error_stage = None
        page_row.processing_error = None
        return True
    except SoftTimeLimitExceeded:
        raise
    except Exception as exc:
        logger.exception(
            "page_panels_failed", extra={"page_id": str(page_row.id), "page_number": number}
        )
        for key in written:
            storage.delete(key)
        page_row.error_stage = STAGE_PANELS
        page_row.processing_error = f"{type(exc).__name__}: {str(exc)[:300]}"
        session.flush()
        return False


# ------------------------------------------------------------------------------ stage 4: products
def _find_products_stage(
    session: Session,
    storage: Storage,
    settings: Settings,
    catalogue: Catalogue,
    job: ProcessingJob,
    models: ProductModels,
    offset: int,
    units: int,
) -> None:
    catalogue.current_stage = STAGE_PRODUCTS
    job.current_stage = STAGE_PRODUCTS
    session.commit()

    # Only pages whose panels were cut: a page that failed an earlier stage keeps its error.
    pages = list(
        session.scalars(
            select(CataloguePage)
            .where(
                CataloguePage.catalogue_id == catalogue.id,
                CataloguePage.status == PageStatus.PANELS_FOUND,
            )
            .order_by(CataloguePage.page_number)
        )
    )
    for done, page_row in enumerate(pages, start=1):
        _find_page_products(session, storage, settings, catalogue, page_row, models)
        progress = min(99, (offset + done) * 100 // units)
        job.progress = progress
        catalogue.processing_progress = progress
        session.commit()


def _find_page_products(
    session: Session,
    storage: Storage,
    settings: Settings,
    catalogue: Catalogue,
    page_row: CataloguePage,
    models: ProductModels,
) -> bool:
    finder, outliner = models
    prompts = parse_prompts(settings.detection_prompts)
    written: list[str] = []
    try:
        panels = list(
            session.scalars(
                select(SourceImage)
                .where(
                    SourceImage.page_id == page_row.id, SourceImage.kind == SourceImageKind.REGION
                )
                .order_by(SourceImage.bbox_y0, SourceImage.bbox_x0)
            )
        )
        with session.begin_nested():
            for panel in panels:
                with Image.open(BytesIO(storage.read_bytes(panel.storage_key))) as opened:
                    picture = opened.convert("RGB")
                finds = find_products(
                    picture,
                    finder,
                    outliner,
                    prompts,
                    min_score=settings.product_min_outline_score,
                    min_coverage=settings.product_min_outline_coverage,
                )
                for found in finds:
                    _save_find(
                        session,
                        storage,
                        settings,
                        catalogue,
                        panel,
                        picture,
                        found,
                        models,
                        written,
                    )
        page_row.status = PageStatus.COMPLETED
        page_row.error_stage = None
        page_row.processing_error = None
        return True
    except SoftTimeLimitExceeded:
        raise
    except Exception as exc:
        logger.exception(
            "page_products_failed",
            extra={"page_id": str(page_row.id), "page_number": page_row.page_number},
        )
        for key in written:
            storage.delete(key)
        page_row.error_stage = STAGE_PRODUCTS
        page_row.processing_error = f"{type(exc).__name__}: {str(exc)[:300]}"
        session.flush()
        return False


def _version(value: str | None) -> str | None:
    return value[:64] if value else None


def _save_find(
    session: Session,
    storage: Storage,
    settings: Settings,
    catalogue: Catalogue,
    panel: SourceImage,
    picture: Image.Image,
    found: Found,
    models: ProductModels,
    written: list[str],
) -> uuid.UUID:
    """Save one product found in a panel (its rows and its pictures); returns its id."""
    finder, outliner = models
    object_id = uuid.uuid4()
    renders = render_find(
        picture,
        found,
        jpeg_quality=settings.jpeg_quality,
        thumbnail_px=settings.thumbnail_max_px,
    )

    def put(kind: str, extension: str, data: bytes | None) -> str | None:
        if data is None:
            return None
        key = product_file_key(catalogue.id, object_id, kind, extension)
        storage.put_bytes(key, data)
        written.append(key)
        return key

    box = found.detection.box
    session.add(
        DetectedObject(
            id=object_id,
            source_image_id=panel.id,
            label=found.detection.label,
            prompt=found.detection.label,
            confidence=found.detection.confidence,
            model_name=finder.name,
            model_version=_version(finder.version),
            bbox_x0=box[0],
            bbox_y0=box[1],
            bbox_x1=box[2],
            bbox_y1=box[3],
        )
    )
    session.flush()  # the find exists before the outline that refers to it
    session.add(
        SegmentationResult(
            detected_object_id=object_id,
            status=found.status,
            confidence=found.score,
            mask_key=put("mask", "png", renders.mask_png),
            crop_key=put("crop", "jpg", renders.crop_jpeg),
            cutout_key=put("cutout", "png", renders.cutout_png),
            white_bg_key=put("white", "jpg", renders.white_jpeg),
            thumbnail_key=put("thumb", "jpg", renders.thumbnail_jpeg),
            error=found.note,
            model_name=outliner.name,
            model_version=_version(outliner.version),
        )
    )
    return object_id

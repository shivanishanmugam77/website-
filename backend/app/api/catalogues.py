from __future__ import annotations

import uuid
from collections.abc import Iterator
from io import BytesIO
from typing import BinaryIO, Literal

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import StreamingResponse
from PIL import Image
from sqlalchemy import func, select

from app.api.deps import (
    AdminUser,
    ClientInfoDep,
    DbSession,
    SettingsDep,
    StorageDep,
    require_admin,
)
from app.models import Catalogue, CataloguePage, OCRResult, SourceImage
from app.models.enums import CatalogueStatus, SourceImageKind
from app.schemas.catalogue import (
    CatalogueDetail,
    CatalogueList,
    CatalogueOut,
    CataloguePageList,
    CataloguePageOut,
    JobOut,
    OcrLineOut,
    PageTextOut,
    PanelOut,
    SourceImageOut,
    TextSignalsOut,
)
from app.services import catalogues as service
from app.services.panels import (
    SheetItem,
    assign_lines,
    draw_contact_sheet,
    draw_overlay,
    sort_reading_order,
)
from app.services.pdf import encode_jpeg
from app.services.storage import ObjectNotFoundError, Storage
from app.services.text_signals import extract_signals

router = APIRouter(
    prefix="/admin/catalogues", tags=["catalogues"], dependencies=[Depends(require_admin)]
)

READ_CHUNK = 64 * 1024
# Multipart framing and form fields add a little to the file size.
UPLOAD_OVERHEAD_BYTES = 1024 * 1024
IMAGE_HEADERS = {"Cache-Control": "private, max-age=3600", "X-Content-Type-Options": "nosniff"}


# ------------------------------------------------------------------------------ helpers
def load_catalogue(db: DbSession, catalogue_id: uuid.UUID) -> Catalogue:
    catalogue = db.get(Catalogue, catalogue_id)
    if catalogue is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Catalogue not found")
    return catalogue


def _iter_chunks(handle: BinaryIO, size: int = 1024 * 1024) -> Iterator[bytes]:
    while chunk := handle.read(size):
        yield chunk


def stream_image(
    storage: Storage, key: str | None, media_type: str = "image/jpeg"
) -> StreamingResponse:
    if key is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Image not available")
    try:
        handle = storage.open(key)
        size = storage.size(key)
    except ObjectNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Image not available") from None

    def body() -> Iterator[bytes]:
        try:
            while chunk := handle.read(READ_CHUNK):
                yield chunk
        finally:
            handle.close()

    return StreamingResponse(
        body(), media_type=media_type, headers={**IMAGE_HEADERS, "Content-Length": str(size)}
    )


def limit_upload_size(request: Request, settings: SettingsDep) -> None:
    """Reject obviously oversized uploads before the body is read.

    Honest clients declare Content-Length; the streaming cap in the storage layer covers
    the rest. In production a reverse proxy should also cap request bodies.
    """
    declared = request.headers.get("content-length")
    if declared and declared.isdigit():
        if int(declared) > settings.max_upload_bytes + UPLOAD_OVERHEAD_BYTES:
            raise HTTPException(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large (limit {settings.max_upload_mb} MB)",
            )


def _detail(db: DbSession, catalogue: Catalogue) -> CatalogueDetail:
    job = service.latest_job(db, catalogue.id)
    return CatalogueDetail(
        **CatalogueOut.model_validate(catalogue).model_dump(),
        job=JobOut.model_validate(job) if job else None,
        page_status_counts=service.page_status_counts(db, catalogue.id),
        embedded_image_count=service.count_source_images(db, catalogue.id),
    )


# ------------------------------------------------------------------------------ upload
@router.post(
    "",
    response_model=CatalogueDetail,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(limit_upload_size)],
)
def upload_catalogue(
    db: DbSession,
    storage: StorageDep,
    settings: SettingsDep,
    admin: AdminUser,
    client: ClientInfoDep,
    file: UploadFile = File(description="The supplier's catalogue PDF"),
    supplier_id: uuid.UUID | None = Form(default=None),
    name: str | None = Form(default=None, max_length=255),
    allow_duplicate: bool = Form(default=False),
) -> CatalogueDetail:
    """Upload a catalogue PDF. Processing runs in the background: poll the returned catalogue
    (``status`` / ``processing_progress`` / ``job``) to follow it."""
    try:
        catalogue, _job = service.ingest_catalogue(
            db,
            storage,
            settings,
            chunks=_iter_chunks(file.file),
            filename=file.filename,
            supplier_id=supplier_id,
            name=name,
            allow_duplicate=allow_duplicate,
            actor=admin,
            ip=client.ip,
        )
    except service.SupplierNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Supplier not found") from None
    except service.NotAPdfError:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="The file is not a PDF"
        ) from None
    except service.UploadTooLargeError:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large (limit {settings.max_upload_mb} MB)",
        ) from None
    except service.DuplicateCatalogueError as exc:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail={
                "message": "This exact file has already been uploaded",
                "existing_catalogue_id": str(exc.existing_id),
            },
        ) from None
    return _detail(db, catalogue)


# ------------------------------------------------------------------------------ browse
@router.get("", response_model=CatalogueList)
def list_catalogues(
    db: DbSession,
    status_filter: CatalogueStatus | None = Query(default=None, alias="status"),
    supplier_id: uuid.UUID | None = None,
    q: str | None = Query(default=None, max_length=100, description="Search in the name"),
    limit: int = Query(default=25, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> CatalogueList:
    rows, total = service.list_catalogues(
        db, status=status_filter, supplier_id=supplier_id, search=q, limit=limit, offset=offset
    )
    return CatalogueList(
        items=[CatalogueOut.model_validate(row) for row in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{catalogue_id}", response_model=CatalogueDetail)
def get_catalogue(catalogue_id: uuid.UUID, db: DbSession) -> CatalogueDetail:
    return _detail(db, load_catalogue(db, catalogue_id))


@router.get("/{catalogue_id}/pages", response_model=CataloguePageList)
def list_pages(
    catalogue_id: uuid.UUID,
    db: DbSession,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> CataloguePageList:
    load_catalogue(db, catalogue_id)
    base = f"/api/admin/catalogues/{catalogue_id}/pages"
    pages = list(
        db.scalars(
            select(CataloguePage)
            .where(CataloguePage.catalogue_id == catalogue_id)
            .order_by(CataloguePage.page_number)
            .limit(limit)
            .offset(offset)
        )
    )
    total = (
        db.scalar(
            select(func.count())
            .select_from(CataloguePage)
            .where(CataloguePage.catalogue_id == catalogue_id)
        )
        or 0
    )
    counts = service.embedded_image_counts(db, [p.id for p in pages])
    panel_counts = service.region_counts(db, [p.id for p in pages])
    items = [
        CataloguePageOut(
            id=page.id,
            page_number=page.page_number,
            status=page.status,
            page_type=page.page_type,
            page_type_confidence=page.page_type_confidence,
            width_px=page.width_px,
            height_px=page.height_px,
            error_stage=page.error_stage,
            processing_error=page.processing_error,
            embedded_image_count=counts.get(page.id, 0),
            panel_count=panel_counts.get(page.id, 0),
            image_url=f"{base}/{page.page_number}/image" if page.image_key else None,
            thumbnail_url=(
                f"{base}/{page.page_number}/image?variant=thumbnail" if page.thumbnail_key else None
            ),
        )
        for page in pages
    ]
    return CataloguePageList(items=items, total=total, limit=limit, offset=offset)


def load_page(db: DbSession, catalogue_id: uuid.UUID, page_number: int) -> CataloguePage:
    page = db.scalar(
        select(CataloguePage).where(
            CataloguePage.catalogue_id == catalogue_id, CataloguePage.page_number == page_number
        )
    )
    if page is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Page not found")
    return page


@router.get("/{catalogue_id}/pages/{page_number}/image", response_class=StreamingResponse)
def page_image(
    catalogue_id: uuid.UUID,
    page_number: int,
    db: DbSession,
    storage: StorageDep,
    variant: Literal["full", "thumbnail"] = "full",
) -> StreamingResponse:
    page = load_page(db, catalogue_id, page_number)
    return stream_image(storage, page.image_key if variant == "full" else page.thumbnail_key)


@router.get("/{catalogue_id}/pages/{page_number}/images", response_model=list[SourceImageOut])
def page_source_images(
    catalogue_id: uuid.UUID, page_number: int, db: DbSession
) -> list[SourceImageOut]:
    """The images found on one page: the page render itself plus native embedded images."""
    page = load_page(db, catalogue_id, page_number)
    images = db.scalars(
        select(SourceImage)
        .where(SourceImage.page_id == page.id)
        .order_by(SourceImage.kind, SourceImage.created_at, SourceImage.id)
    )
    base = f"/api/admin/catalogues/{catalogue_id}/source-images"
    out = []
    for image in images:
        box = (image.bbox_x0, image.bbox_y0, image.bbox_x1, image.bbox_y1)
        out.append(
            SourceImageOut(
                id=image.id,
                kind=image.kind,
                width_px=image.width_px,
                height_px=image.height_px,
                bbox=None if None in box else box,
                phash=image.phash,
                url=f"{base}/{image.id}/image",
            )
        )
    return out


@router.get("/{catalogue_id}/pages/{page_number}/text", response_model=PageTextOut)
def page_text(catalogue_id: uuid.UUID, page_number: int, db: DbSession) -> PageTextOut:
    """The text read on one page (in reading order, with positions) and the product facts
    spotted in it: model codes, sizes and series labels."""
    page = load_page(db, catalogue_id, page_number)
    rows = list(
        db.scalars(
            select(OCRResult).where(OCRResult.page_id == page.id).order_by(OCRResult.line_index)
        )
    )
    lines = [
        OcrLineOut(
            line_index=row.line_index,
            text=row.text,
            confidence=row.confidence,
            language=row.language,
            bbox=(row.bbox_x0, row.bbox_y0, row.bbox_x1, row.bbox_y1),
            polygon=row.polygon,
        )
        for row in rows
    ]
    return PageTextOut(
        page_number=page.page_number,
        status=page.status,
        page_type=page.page_type,
        page_type_confidence=page.page_type_confidence,
        engine=rows[0].engine if rows else None,
        engine_version=rows[0].engine_version if rows else None,
        line_count=len(lines),
        lines=lines,
        signals=TextSignalsOut.model_validate(extract_signals([row.text for row in rows])),
    )


def region_box(region: SourceImage) -> tuple[float, float, float, float]:
    return (
        region.bbox_x0 or 0.0,
        region.bbox_y0 or 0.0,
        region.bbox_x1 or 0.0,
        region.bbox_y1 or 0.0,
    )


def panels_in_reading_order(db: DbSession, page: CataloguePage) -> list[SourceImage]:
    regions = list(
        db.scalars(
            select(SourceImage).where(
                SourceImage.page_id == page.id, SourceImage.kind == SourceImageKind.REGION
            )
        )
    )
    return sort_reading_order(regions, page.height_px or 1, key=region_box)


def build_panel_list(db: DbSession, catalogue_id: uuid.UUID, page: CataloguePage) -> list[PanelOut]:
    """The page's photo panels in reading order, each with the text that belongs to it."""
    regions = panels_in_reading_order(db, page)
    lines = list(
        db.scalars(
            select(OCRResult).where(OCRResult.page_id == page.id).order_by(OCRResult.line_index)
        )
    )
    boxes = [region_box(r) for r in regions]
    owned = assign_lines(
        boxes,
        [(row.line_index, (row.bbox_x0, row.bbox_y0, row.bbox_x1, row.bbox_y1)) for row in lines],
        page.height_px or 1,
    )
    by_index = {row.line_index: row.text for row in lines}
    page_area = (page.width_px or 1) * (page.height_px or 1)
    base = f"/api/admin/catalogues/{catalogue_id}/source-images"
    return [
        PanelOut(
            id=region.id,
            index=number,
            origin=region.origin,
            bbox=box,
            width_px=region.width_px,
            height_px=region.height_px,
            area_share=region.width_px * region.height_px / page_area,
            # ?v= changes whenever the cut-out does, so an edited panel is never shown stale
            url=f"{base}/{region.id}/image" + (f"?v={region.sha256[:8]}" if region.sha256 else ""),
            text_line_indexes=indexes,
            text=[by_index[i] for i in indexes],
            signals=TextSignalsOut.model_validate(
                extract_signals([by_index[i] for i in indexes], indexes)
            ),
        )
        for number, (region, box, indexes) in enumerate(zip(regions, boxes, owned, strict=True), 1)
    ]


@router.get("/{catalogue_id}/pages/{page_number}/panels", response_model=list[PanelOut])
def page_panels(catalogue_id: uuid.UUID, page_number: int, db: DbSession) -> list[PanelOut]:
    """The photo panels found on one page, in reading order, each with its own text."""
    return build_panel_list(db, catalogue_id, load_page(db, catalogue_id, page_number))


@router.get("/{catalogue_id}/pages/{page_number}/panels/preview", response_class=Response)
def page_panels_preview(
    catalogue_id: uuid.UUID, page_number: int, db: DbSession, storage: StorageDep
) -> Response:
    """The page picture with every panel outlined and numbered, to check the result by eye."""
    page = load_page(db, catalogue_id, page_number)
    if page.image_key is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Image not available")
    try:
        data = storage.read_bytes(page.image_key)
    except ObjectNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Image not available") from None
    boxes = [
        (r.bbox_x0, r.bbox_y0, r.bbox_x1, r.bbox_y1) for r in panels_in_reading_order(db, page)
    ]
    with Image.open(BytesIO(data)) as opened:
        overlay = draw_overlay(opened, boxes)
    # Never cached: the URL stays the same when a catalogue is reprocessed, and a browser
    # showing yesterday's panels would be worse than no cache at all.
    return Response(
        content=encode_jpeg(overlay, 80),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
    )


@router.get("/{catalogue_id}/panels/overview", response_class=Response)
def panels_overview(
    catalogue_id: uuid.UUID,
    db: DbSession,
    storage: StorageDep,
    start: int = Query(default=1, ge=1, description="first page number to show"),
    limit: int = Query(default=16, ge=1, le=40, description="how many pages to show"),
    columns: int = Query(default=4, ge=1, le=8),
) -> Response:
    """Several pages in one picture, each with its panels outlined and numbered: a quick way
    to check a whole catalogue by eye. Page thumbnails are used, so it is small and fast."""
    load_catalogue(db, catalogue_id)
    pages = list(
        db.scalars(
            select(CataloguePage)
            .where(
                CataloguePage.catalogue_id == catalogue_id,
                CataloguePage.page_number >= start,
                CataloguePage.thumbnail_key.is_not(None),
            )
            .order_by(CataloguePage.page_number)
            .limit(limit)
        )
    )
    regions = list(
        db.scalars(
            select(SourceImage).where(
                SourceImage.page_id.in_([p.id for p in pages]),
                SourceImage.kind == SourceImageKind.REGION,
            )
        )
    )
    items = []
    for page in pages:
        try:
            data = storage.read_bytes(page.thumbnail_key or "")
        except ObjectNotFoundError:
            continue
        own = [r for r in regions if r.page_id == page.id]
        ordered = sort_reading_order(own, page.height_px or 1, key=region_box)
        boxes = [region_box(r) for r in ordered]
        with Image.open(BytesIO(data)) as opened:
            thumbnail = opened.convert("RGB")
        kind = page.page_type.value if page.page_type else "-"
        items.append(
            SheetItem(
                label=f"Page {page.page_number}  {kind}  {len(boxes)} panels",
                thumbnail=thumbnail,
                page_size=(page.width_px or thumbnail.width, page.height_px or thumbnail.height),
                boxes=boxes,
            )
        )
    if not items:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="No page pictures to show")
    return Response(
        content=encode_jpeg(draw_contact_sheet(items, columns), 80),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
    )


@router.get("/{catalogue_id}/source-images/{image_id}/image", response_class=StreamingResponse)
def source_image_file(
    catalogue_id: uuid.UUID, image_id: uuid.UUID, db: DbSession, storage: StorageDep
) -> StreamingResponse:
    image = db.scalar(
        select(SourceImage)
        .join(CataloguePage, SourceImage.page_id == CataloguePage.id)
        .where(SourceImage.id == image_id, CataloguePage.catalogue_id == catalogue_id)
    )
    if image is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Image not found")
    return stream_image(storage, image.storage_key)


# ------------------------------------------------------------------------------ actions
@router.post(
    "/{catalogue_id}/reprocess", response_model=JobOut, status_code=status.HTTP_202_ACCEPTED
)
def reprocess(
    catalogue_id: uuid.UUID,
    db: DbSession,
    admin: AdminUser,
    client: ClientInfoDep,
    force: bool = False,
    discard_corrections: bool = False,
) -> JobOut:
    """Run processing again from scratch (e.g. after a failure or once new stages exist).

    Refused when admins have corrected panels by hand (reprocessing would throw that work
    away) unless ``discard_corrections`` is true.
    """
    catalogue = load_catalogue(db, catalogue_id)
    try:
        job = service.reprocess_catalogue(
            db,
            catalogue,
            actor=admin,
            ip=client.ip,
            force=force,
            discard_corrections=discard_corrections,
        )
    except service.CorrectionsWouldBeLostError as exc:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail=(
                f"This catalogue has {exc.count} manual panel correction(s) that reprocessing "
                "would discard. Use discard_corrections=true to reprocess anyway."
            ),
        ) from None
    except service.CatalogueBusyError:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail="Processing is already queued or running (use force=true if it is stuck)",
        ) from None
    return JobOut.model_validate(job)


@router.delete(
    "/{catalogue_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,  # a 204 has no body ...
    response_model=None,  # ... so FastAPI must not infer one from the annotation
)
def delete_catalogue(
    catalogue_id: uuid.UUID,
    db: DbSession,
    storage: StorageDep,
    admin: AdminUser,
    client: ClientInfoDep,
    force: bool = False,
) -> None:
    catalogue = load_catalogue(db, catalogue_id)
    try:
        service.delete_catalogue(db, storage, catalogue, actor=admin, ip=client.ip, force=force)
    except service.CatalogueBusyError:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail="Processing is still queued or running (use force=true if it is stuck)",
        ) from None


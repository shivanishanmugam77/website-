"""Looking at the products the system found inside the photo panels (read only).

The correction editor for these arrives with the next step; until then the pictures below
are how an admin checks a catalogue by eye.
"""

from __future__ import annotations

import uuid
from io import BytesIO
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.responses import StreamingResponse
from PIL import Image
from sqlalchemy import select

from app.api.catalogues import load_catalogue, load_page, panels_in_reading_order, stream_image
from app.api.deps import DbSession, StorageDep, require_admin
from app.ml.detection.base import Detection
from app.models import CataloguePage, DetectedObject, SegmentationResult, SourceImage
from app.models.enums import SegmentationStatus
from app.schemas.product_finds import ProductFindOut
from app.services.detection_preview import draw_detections, draw_sheet
from app.services.pdf import encode_jpeg
from app.services.storage import ObjectNotFoundError, Storage

router = APIRouter(
    prefix="/admin/catalogues", tags=["products found"], dependencies=[Depends(require_admin)]
)

# What an address asks for -> (the column that holds the file, its content type)
PICTURES = {
    "crop": ("crop_key", "image/jpeg"),
    "mask": ("mask_key", "image/png"),
    "cutout": ("cutout_key", "image/png"),
    "white": ("white_bg_key", "image/jpeg"),
    "thumbnail": ("thumbnail_key", "image/jpeg"),
}
# Never cached: the address of a page stays the same when a catalogue is processed again.
NO_STORE = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}


def _finds(
    db: DbSession, panel_ids: list[uuid.UUID]
) -> list[tuple[DetectedObject, SegmentationResult | None]]:
    """Every find inside the given panels with its outline, top to bottom, left to right."""
    if not panel_ids:
        return []
    rows = db.execute(
        select(DetectedObject, SegmentationResult)
        .select_from(DetectedObject)
        .outerjoin(SegmentationResult, SegmentationResult.detected_object_id == DetectedObject.id)
        .where(DetectedObject.source_image_id.in_(panel_ids))
        .order_by(DetectedObject.bbox_y0, DetectedObject.bbox_x0)
    ).all()
    return [(found, outline) for found, outline in rows]


def _urls(
    catalogue_id: uuid.UUID, object_id: uuid.UUID, outline: SegmentationResult | None
) -> dict[str, str]:
    if outline is None:
        return {}
    base = f"/api/admin/catalogues/{catalogue_id}/products/{object_id}/image"
    return {
        kind: f"{base}/{kind}"
        for kind, (column, _media_type) in PICTURES.items()
        if getattr(outline, column)
    }


def _read_mask(storage: Storage, outline: SegmentationResult | None) -> Image.Image | None:
    if outline is None or outline.mask_key is None:
        return None
    try:
        data = storage.read_bytes(outline.mask_key)
    except ObjectNotFoundError:
        return None
    with Image.open(BytesIO(data)) as opened:
        return opened.convert("L")


@router.get("/{catalogue_id}/pages/{page_number}/products", response_model=list[ProductFindOut])
def page_products(
    catalogue_id: uuid.UUID, page_number: int, db: DbSession
) -> list[ProductFindOut]:
    """The products found on one page, panel by panel, each with its outline and pictures."""
    page = load_page(db, catalogue_id, page_number)
    panels = panels_in_reading_order(db, page)
    number_of = {panel.id: number for number, panel in enumerate(panels, start=1)}
    finds = sorted(_finds(db, list(number_of)), key=lambda pair: number_of[pair[0].source_image_id])
    return [
        ProductFindOut(
            id=found.id,
            panel_id=found.source_image_id,
            panel_index=number_of[found.source_image_id],
            label=found.label,
            confidence=found.confidence,
            bbox=(found.bbox_x0, found.bbox_y0, found.bbox_x1, found.bbox_y1),
            outline_status=outline.status if outline else None,
            outline_score=outline.confidence if outline else None,
            note=outline.error if outline else None,
            images=_urls(catalogue_id, found.id, outline),
        )
        for found, outline in finds
    ]


@router.get("/{catalogue_id}/pages/{page_number}/products/preview", response_class=Response)
def page_products_preview(
    catalogue_id: uuid.UUID, page_number: int, db: DbSession, storage: StorageDep
) -> Response:
    """Every panel of the page with its finds boxed, named and tinted, to check them by eye."""
    page = load_page(db, catalogue_id, page_number)
    panels = panels_in_reading_order(db, page)
    by_panel: dict[uuid.UUID, list[tuple[DetectedObject, SegmentationResult | None]]] = {}
    for found, outline in _finds(db, [panel.id for panel in panels]):
        by_panel.setdefault(found.source_image_id, []).append((found, outline))
    items = []
    for number, panel in enumerate(panels, start=1):
        try:
            data = storage.read_bytes(panel.storage_key)
        except ObjectNotFoundError:
            continue
        with Image.open(BytesIO(data)) as opened:
            picture = opened.convert("RGB")
        finds = by_panel.get(panel.id, [])
        detections = [
            Detection(f.label, f.confidence, (f.bbox_x0, f.bbox_y0, f.bbox_x1, f.bbox_y1))
            for f, _ in finds
        ]
        masks = [_read_mask(storage, outline) for _, outline in finds]
        annotated = draw_detections(picture, detections, masks)
        items.append((f"Panel {number}: {len(finds)} found", annotated))
    if not items:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="This page has no panel pictures")
    return Response(
        content=encode_jpeg(draw_sheet(items), 80), media_type="image/jpeg", headers=NO_STORE
    )


@router.get("/{catalogue_id}/products/overview", response_class=Response)
def products_overview(
    catalogue_id: uuid.UUID,
    db: DbSession,
    storage: StorageDep,
    start: int = Query(default=1, ge=1, description="number of the first product to show"),
    limit: int = Query(default=24, ge=1, le=60, description="how many products to show"),
    columns: int = Query(default=4, ge=1, le=8),
) -> Response:
    """Many products in one picture, each cut out on white with its page and name underneath.
    Outlines the system is unsure about are marked 'check'."""
    load_catalogue(db, catalogue_id)
    rows = db.execute(
        select(DetectedObject, SegmentationResult, CataloguePage.page_number)
        .select_from(DetectedObject)
        .join(SegmentationResult, SegmentationResult.detected_object_id == DetectedObject.id)
        .join(SourceImage, SourceImage.id == DetectedObject.source_image_id)
        .join(CataloguePage, CataloguePage.id == SourceImage.page_id)
        .where(
            CataloguePage.catalogue_id == catalogue_id,
            SegmentationResult.thumbnail_key.is_not(None),
        )
        .order_by(
            CataloguePage.page_number,
            SourceImage.bbox_y0,
            SourceImage.bbox_x0,
            DetectedObject.bbox_y0,
            DetectedObject.bbox_x0,
        )
        .offset(start - 1)
        .limit(limit)
    ).all()
    items = []
    for found, outline, page_number in rows:
        try:
            data = storage.read_bytes(outline.thumbnail_key or "")
        except ObjectNotFoundError:
            continue
        caption = f"p{page_number} {found.label} {found.confidence:.0%}"
        if outline.status is SegmentationStatus.LOW_CONFIDENCE:
            caption += " - check"
        with Image.open(BytesIO(data)) as opened:
            items.append((caption, opened.convert("RGB")))
    if not items:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="No product pictures to show")
    return Response(
        content=encode_jpeg(draw_sheet(items, columns, cell=300), 80),
        media_type="image/jpeg",
        headers=NO_STORE,
    )


@router.get(
    "/{catalogue_id}/products/{object_id}/image/{kind}", response_class=StreamingResponse
)
def product_picture(
    catalogue_id: uuid.UUID,
    object_id: uuid.UUID,
    kind: Literal["crop", "mask", "cutout", "white", "thumbnail"],
    db: DbSession,
    storage: StorageDep,
) -> StreamingResponse:
    """One picture of one product: the crop, its mask, the transparent cut-out, the cut-out on
    white, or a thumbnail of that."""
    outline = db.scalar(
        select(SegmentationResult)
        .join(DetectedObject, DetectedObject.id == SegmentationResult.detected_object_id)
        .join(SourceImage, SourceImage.id == DetectedObject.source_image_id)
        .join(CataloguePage, CataloguePage.id == SourceImage.page_id)
        .where(DetectedObject.id == object_id, CataloguePage.catalogue_id == catalogue_id)
    )
    if outline is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Product not found")
    column, media_type = PICTURES[kind]
    return stream_image(storage, getattr(outline, column), media_type)

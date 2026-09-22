"""A page with finds and OCR text, ready for candidate assembly."""

from __future__ import annotations

from PIL import ImageDraw

from app.models import OCRResult
from app.models.enums import CatalogueStatus, PageStatus
from tests.factories import make_catalogue, make_page
from app.services.pipeline import page_image_key
from tests.page_factory import blank_page
from tests.product_factory import add_find
from tests.test_panel_editing_api import BIG, RED, SMALL, add_region, jpeg

PAGE_8_LINES = [""] * 11 + ["型号：YY-11", "四人位职员桌2400Wx1200D×750H"]


def add_ocr_lines(session, page, lines=PAGE_8_LINES) -> None:  # noqa: ANN001
    for index, text in enumerate(lines):
        session.add(
            OCRResult(
                page=page,
                line_index=index,
                text=text,
                confidence=0.95,
                engine="rapidocr",
                bbox_x0=0.0,
                bbox_y0=float(index),
                bbox_x1=10.0,
                bbox_y1=float(index) + 1.0,
            )
        )
    session.flush()


def catalogue_with_candidates(  # noqa: ANN201
    session, storage, *, catalogue_status=CatalogueStatus.COMPLETED, lines=PAGE_8_LINES
):
    """One catalogue, one page: a red panel (desk) and a blue panel (chair), with the
    BOGAO-style OCR text that names the red one's product."""
    catalogue = make_catalogue(session, status=catalogue_status)
    page = make_page(session, catalogue, 1)
    image = blank_page(1200, 800)
    ImageDraw.Draw(image).rectangle(BIG, fill=RED)
    ImageDraw.Draw(image).rectangle(SMALL, fill=(0, 0, 255))
    page.image_key = page_image_key(catalogue.id, 1)
    storage.put_bytes(page.image_key, jpeg(image))
    page.width_px, page.height_px, page.status = 1200, 800, PageStatus.COMPLETED
    big = add_region(session, storage, catalogue, page, image, BIG)
    small = add_region(session, storage, catalogue, page, image, SMALL)
    add_ocr_lines(session, page, lines)
    desk = add_find(session, storage, catalogue, big, label="desk", box=(50.0, 50.0, 300.0, 400.0))
    chair = add_find(
        session, storage, catalogue, small, label="office chair", box=(20.0, 20.0, 200.0, 150.0)
    )
    session.flush()
    return catalogue, page, big, small, desk, chair

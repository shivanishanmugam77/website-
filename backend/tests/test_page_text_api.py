"""Reading a page's recognised text through the admin API."""

from __future__ import annotations

import uuid

import pytest

from app.models import CataloguePage, OCRResult
from app.models.enums import PageStatus, PageType, UserRole
from app.services.pipeline import run_catalogue_pipeline
from tests.factories import TEST_PASSWORD, make_catalogue, make_page, make_user, settings_with
from tests.helpers import sign_in, upload
from tests.ocr_fakes import FunctionOcr, line
from tests.pdf_factory import ImageSpec, PageSpec, build_pdf


@pytest.fixture
def admin(client, db_session):  # noqa: ANN001, ANN201
    user = make_user(db_session, password=TEST_PASSWORD, role=UserRole.ADMIN)
    sign_in(client, user)
    return user


def add_text(session, page, index: int, text: str, **overrides) -> OCRResult:  # noqa: ANN001, ANN003
    values = {
        "page_id": page.id,
        "line_index": index,
        "text": text,
        "confidence": 0.9,
        "language": "en",
        "polygon": [[10, 20 + index], [110, 20 + index], [110, 40 + index], [10, 40 + index]],
        "bbox_x0": 10,
        "bbox_y0": 20 + index,
        "bbox_x1": 110,
        "bbox_y1": 40 + index,
        "engine": "rapidocr",
        "engine_version": "1.4.4",
    }
    row = OCRResult(**{**values, **overrides})
    session.add(row)
    session.flush()
    return row


def test_a_pages_text_signals_and_type_are_returned(client, admin, db_session) -> None:  # noqa: ANN001
    catalogue = make_catalogue(db_session)
    page = make_page(db_session, catalogue, 8)
    page.status = PageStatus.TEXT_READ
    page.page_type = PageType.PRODUCT
    page.page_type_confidence = 0.9
    add_text(db_session, page, 1, "3200Wx1400Dx750H")  # inserted out of order on purpose
    add_text(db_session, page, 0, "\u578b\u53f7: YY-21", language="mixed")
    add_text(db_session, page, 2, "09series", polygon=None)
    db_session.flush()

    response = client.get(f"/api/admin/catalogues/{catalogue.id}/pages/8/text")
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["page_number"] == 8 and body["status"] == "TEXT_READ"
    assert body["page_type"] == "PRODUCT" and body["page_type_confidence"] == 0.9
    assert (body["engine"], body["engine_version"], body["line_count"]) == ("rapidocr", "1.4.4", 3)
    assert [item["line_index"] for item in body["lines"]] == [0, 1, 2]  # reading order
    first = body["lines"][0]
    assert first["text"] == "\u578b\u53f7: YY-21" and first["language"] == "mixed"
    assert first["bbox"] == [10, 20, 110, 40] and first["confidence"] == 0.9
    assert first["polygon"][0] == [10, 20] and body["lines"][2]["polygon"] is None

    signals = body["signals"]
    assert signals["model_codes"] == [{"code": "YY-21", "line_index": 0, "labelled": True}]
    (size,) = signals["dimensions"]
    assert (size["width"], size["depth"], size["height"], size["unit"]) == (3200, 1400, 750, "mm")
    assert size["line_index"] == 1 and size["labelled"] is True
    assert signals["series"] == [{"label": "09 series", "line_index": 2}]


def test_a_page_without_text_returns_empty_results(client, admin, db_session) -> None:  # noqa: ANN001
    catalogue = make_catalogue(db_session)
    make_page(db_session, catalogue, 1)
    body = client.get(f"/api/admin/catalogues/{catalogue.id}/pages/1/text").json()
    assert body["lines"] == [] and body["line_count"] == 0
    assert body["engine"] is None and body["page_type"] is None
    assert body["signals"] == {"model_codes": [], "dimensions": [], "series": []}


def test_text_of_one_page_never_includes_another_pages_lines(client, admin, db_session) -> None:  # noqa: ANN001
    catalogue = make_catalogue(db_session)
    first, second = make_page(db_session, catalogue, 1), make_page(db_session, catalogue, 2)
    add_text(db_session, first, 0, "ON PAGE ONE")
    add_text(db_session, second, 0, "ON PAGE TWO")
    body = client.get(f"/api/admin/catalogues/{catalogue.id}/pages/2/text").json()
    assert [item["text"] for item in body["lines"]] == ["ON PAGE TWO"]


@pytest.mark.parametrize("page_number", [99, 0])
def test_unknown_pages_and_catalogues_are_404(client, admin, db_session, page_number: int) -> None:  # noqa: ANN001
    catalogue = make_catalogue(db_session)
    assert client.get(f"/api/admin/catalogues/{catalogue.id}/pages/{page_number}/text").status_code == 404
    assert client.get(f"/api/admin/catalogues/{uuid.uuid4()}/pages/1/text").status_code == 404


def test_the_page_list_shows_each_pages_type_and_confidence(client, admin, db_session) -> None:  # noqa: ANN001
    catalogue = make_catalogue(db_session)
    page = make_page(db_session, catalogue, 1)
    page.page_type, page.page_type_confidence = PageType.COVER, 0.7
    make_page(db_session, catalogue, 2)
    db_session.flush()
    items = client.get(f"/api/admin/catalogues/{catalogue.id}/pages").json()["items"]
    assert [(i["page_type"], i["page_type_confidence"]) for i in items] == [("COVER", 0.7), (None, None)]


def test_from_upload_to_readable_text(client, admin, db_session, storage) -> None:  # noqa: ANN001
    """The whole path: upload a PDF, let the worker process it, read the text back."""
    pdf = build_pdf(
        [PageSpec(images=[ImageSpec(50, 500, 300, 200, px=160, color=(200, 40, 40))]) for _ in range(2)]
    )
    body = upload(client, pdf, "BOGAO sample.pdf").json()
    pages_read = [
        [line("SERVICE PROVIDER FOR COMMERCIAL SPACE", 100, 100, 900, 160)],
        [line("MODEL: YY-15", 100, 1000, 400, 1040), line("1200Wx1200Dx750H", 100, 1050, 500, 1090)],
    ]
    run_catalogue_pipeline(
        db_session,
        storage,
        settings_with(min_embedded_image_px=100, ocr_enabled=True),
        catalogue_id=uuid.UUID(body["id"]),
        job_id=uuid.UUID(body["job"]["id"]),
        ocr_provider=FunctionOcr(lambda index, _image: list(pages_read[index])),
    )

    detail = client.get(f"/api/admin/catalogues/{body['id']}").json()
    assert detail["status"] == "COMPLETED" and detail["page_status_counts"] == {"TEXT_READ": 2}

    pages = client.get(f"/api/admin/catalogues/{body['id']}/pages").json()["items"]
    assert [p["page_type"] for p in pages] == ["COVER", "PRODUCT"]

    text = client.get(f"/api/admin/catalogues/{body['id']}/pages/2/text").json()
    assert [item["text"] for item in text["lines"]] == ["MODEL: YY-15", "1200Wx1200Dx750H"]
    assert text["signals"]["model_codes"][0]["code"] == "YY-15"
    assert text["signals"]["dimensions"][0]["width"] == 1200
    stored = db_session.query(CataloguePage).filter_by(catalogue_id=uuid.UUID(body["id"])).count()
    assert stored == 2

"""Reading a page's photo panels, their text, and the preview picture through the admin API."""

from __future__ import annotations

import io
import uuid

import pytest
from PIL import Image

from app.models import OCRResult, SourceImage
from app.models.enums import PageStatus, SourceImageKind, UserRole
from app.services.pipeline import run_catalogue_pipeline
from tests.factories import TEST_PASSWORD, make_catalogue, make_page, make_user, settings_with
from tests.helpers import sign_in, upload
from tests.pdf_factory import ImageSpec, PageSpec, build_pdf


@pytest.fixture
def admin(client, db_session):  # noqa: ANN001, ANN201
    user = make_user(db_session, password=TEST_PASSWORD, role=UserRole.ADMIN)
    sign_in(client, user)
    return user


def add_panel(session, page, box: tuple[int, int, int, int]) -> SourceImage:  # noqa: ANN001
    width, height = box[2] - box[0], box[3] - box[1]
    image = SourceImage(
        page_id=page.id,
        kind=SourceImageKind.REGION,
        storage_key=f"catalogues/x/source-images/{uuid.uuid4()}.jpg",
        width_px=width,
        height_px=height,
        bbox_x0=box[0],
        bbox_y0=box[1],
        bbox_x1=box[2],
        bbox_y1=box[3],
    )
    session.add(image)
    session.flush()
    return image


def add_line(session, page, index: int, text: str, box: tuple[int, int, int, int]) -> None:  # noqa: ANN001
    session.add(
        OCRResult(
            page_id=page.id,
            line_index=index,
            text=text,
            confidence=0.9,
            language="en",
            polygon=None,
            bbox_x0=box[0],
            bbox_y0=box[1],
            bbox_x1=box[2],
            bbox_y1=box[3],
            engine="rapidocr",
            engine_version="1.4.4",
        )
    )
    session.flush()


def page_with_layout(session):  # noqa: ANN001, ANN201
    """A 2000 x 1000 page: a big panel on the left, two stacked on the right, and text."""
    catalogue = make_catalogue(session)
    page = make_page(session, catalogue, 3)
    page.width_px, page.height_px, page.status = 2000, 1000, PageStatus.PANELS_FOUND
    # created in a scrambled order on purpose: the API must return reading order
    lower_right = add_panel(session, page, (1100, 520, 1900, 900))
    big = add_panel(session, page, (100, 100, 1000, 900))
    upper_right = add_panel(session, page, (1100, 100, 1900, 480))
    add_line(session, page, 0, "MODEL: YY-11", (150, 820, 500, 860))  # inside the big panel
    add_line(session, page, 1, "4-person desk 2400Wx1200Dx750H", (1100, 905, 1700, 945))  # a caption
    add_line(session, page, 2, "OUR PRODUCTS", (100, 20, 600, 70))  # a heading: no panel's
    session.flush()
    return catalogue, page, big, upper_right, lower_right


def test_panels_come_back_in_reading_order_with_their_own_text(client, admin, db_session) -> None:  # noqa: ANN001
    catalogue, _page, big, upper_right, lower_right = page_with_layout(db_session)
    response = client.get(f"/api/admin/catalogues/{catalogue.id}/pages/3/panels")
    assert response.status_code == 200, response.text
    panels = response.json()

    assert [p["id"] for p in panels] == [str(big.id), str(upper_right.id), str(lower_right.id)]
    assert [p["index"] for p in panels] == [1, 2, 3]
    first, second, third = panels
    assert first["bbox"] == [100, 100, 1000, 900] and (first["width_px"], first["height_px"]) == (900, 800)
    assert first["area_share"] == pytest.approx(0.36)
    assert first["url"] == f"/api/admin/catalogues/{catalogue.id}/source-images/{big.id}/image"

    assert first["text"] == ["MODEL: YY-11"] and first["text_line_indexes"] == [0]
    assert second["text"] == [] and second["signals"]["model_codes"] == []
    assert third["text"] == ["4-person desk 2400Wx1200Dx750H"]  # the caption under it
    assert first["signals"]["model_codes"] == [{"code": "YY-11", "line_index": 0, "labelled": True}]
    (size,) = third["signals"]["dimensions"]
    assert (size["width"], size["depth"], size["height"]) == (2400, 1200, 750)
    assert size["line_index"] == 1  # the line's number on the page, not within the panel


def test_a_page_without_panels_returns_an_empty_list(client, admin, db_session) -> None:  # noqa: ANN001
    catalogue = make_catalogue(db_session)
    make_page(db_session, catalogue, 1)
    assert client.get(f"/api/admin/catalogues/{catalogue.id}/pages/1/panels").json() == []


def test_unknown_pages_give_404_for_panels_and_preview(client, admin, db_session) -> None:  # noqa: ANN001
    catalogue = make_catalogue(db_session)
    for path in ("panels", "panels/preview"):
        assert client.get(f"/api/admin/catalogues/{catalogue.id}/pages/9/{path}").status_code == 404
        assert client.get(f"/api/admin/catalogues/{uuid.uuid4()}/pages/1/{path}").status_code == 404


def test_the_preview_is_the_page_with_panels_outlined(client, admin, db_session, storage) -> None:  # noqa: ANN001
    catalogue, page, *_ = page_with_layout(db_session)
    page.image_key = f"catalogues/{catalogue.id}/pages/0003.jpg"
    paper = Image.new("RGB", (2000, 1000), (243, 241, 238))
    buffer = io.BytesIO()
    paper.save(buffer, "JPEG", quality=95)
    storage.put_bytes(page.image_key, buffer.getvalue())
    db_session.flush()

    response = client.get(f"/api/admin/catalogues/{catalogue.id}/pages/3/panels/preview")
    assert response.status_code == 200 and response.headers["content-type"] == "image/jpeg"
    assert response.headers["cache-control"] == "no-store"  # a reprocess must show at once
    image = Image.open(io.BytesIO(response.content)).convert("RGB")
    assert image.size == (2000, 1000)
    outline = image.getpixel((100, 500))  # on the big panel's left edge
    assert max(outline) - min(outline) > 60  # a saturated colour, not paper
    assert all(abs(a - b) < 12 for a, b in zip(image.getpixel((40, 500)), (243, 241, 238), strict=True))


def test_a_page_without_an_image_has_no_preview(client, admin, db_session) -> None:  # noqa: ANN001
    catalogue = make_catalogue(db_session)
    make_page(db_session, catalogue, 1)
    assert client.get(f"/api/admin/catalogues/{catalogue.id}/pages/1/panels/preview").status_code == 404


def test_the_page_list_counts_panels(client, admin, db_session) -> None:  # noqa: ANN001
    catalogue, *_ = page_with_layout(db_session)
    make_page(db_session, catalogue, 4)
    items = client.get(f"/api/admin/catalogues/{catalogue.id}/pages").json()["items"]
    assert {i["page_number"]: i["panel_count"] for i in items} == {3: 3, 4: 0}


def test_from_upload_to_panels(client, admin, db_session, storage) -> None:  # noqa: ANN001
    """The whole path: upload, let the worker process it, then read the panels back."""
    page = PageSpec(
        images=[
            ImageSpec(50, 300, 350, 400, px=160, color=(200, 60, 40)),
            ImageSpec(420, 300, 120, 180, px=160, color=(40, 90, 200)),
        ]
    )
    body = upload(client, build_pdf([page, page]), "BOGAO sample.pdf").json()
    run_catalogue_pipeline(
        db_session,
        storage,
        settings_with(min_embedded_image_px=100, ocr_enabled=False, panel_detection_enabled=True),
        catalogue_id=uuid.UUID(body["id"]),
        job_id=uuid.UUID(body["job"]["id"]),
    )

    detail = client.get(f"/api/admin/catalogues/{body['id']}").json()
    assert detail["status"] == "COMPLETED" and detail["page_status_counts"] == {"PANELS_FOUND": 2}

    panels = client.get(f"/api/admin/catalogues/{body['id']}/pages/2/panels").json()
    assert [p["index"] for p in panels] == [1, 2]
    assert panels[0]["width_px"] > panels[1]["width_px"]  # the big picture comes first

    crop = client.get(panels[0]["url"])
    assert crop.status_code == 200 and crop.headers["content-type"] == "image/jpeg"
    assert Image.open(io.BytesIO(crop.content)).size == (panels[0]["width_px"], panels[0]["height_px"])
    assert client.get(f"/api/admin/catalogues/{body['id']}/pages/2/panels/preview").status_code == 200


# ------------------------------------------------------------------------------ overview
def add_thumbnail(storage, page, size: tuple[int, int]) -> None:  # noqa: ANN001
    buffer = io.BytesIO()
    Image.new("RGB", size, (235, 232, 226)).save(buffer, "JPEG", quality=90)
    page.thumbnail_key = f"catalogues/x/pages/{page.page_number:04d}_thumb.jpg"
    storage.put_bytes(page.thumbnail_key, buffer.getvalue())


def test_the_overview_shows_several_pages_in_one_picture(client, admin, db_session, storage) -> None:  # noqa: ANN001
    catalogue, page, *_ = page_with_layout(db_session)  # page 3: 2000 x 1000 with three panels
    add_thumbnail(storage, page, (400, 200))
    second = make_page(db_session, catalogue, 4)
    second.width_px, second.height_px = 1000, 1500
    add_thumbnail(storage, second, (267, 400))
    db_session.flush()

    response = client.get(f"/api/admin/catalogues/{catalogue.id}/panels/overview")
    assert response.status_code == 200 and response.headers["content-type"] == "image/jpeg"
    assert response.headers["cache-control"] == "no-store"
    assert Image.open(io.BytesIO(response.content)).size == (4 * 420, 420 + 34)  # two pages, one row


def test_the_overview_can_show_a_range_of_pages_in_fewer_columns(client, admin, db_session, storage) -> None:  # noqa: ANN001
    catalogue, page, *_ = page_with_layout(db_session)
    add_thumbnail(storage, page, (400, 200))
    for number in (4, 5):
        extra = make_page(db_session, catalogue, number)
        extra.width_px, extra.height_px = 1000, 1000
        add_thumbnail(storage, extra, (400, 400))
    db_session.flush()

    url = f"/api/admin/catalogues/{catalogue.id}/panels/overview"
    picture = Image.open(io.BytesIO(client.get(f"{url}?start=4&limit=2&columns=1").content))
    assert picture.size == (420, 2 * (420 + 34))  # pages 4 and 5, stacked


def test_the_overview_needs_page_pictures(client, admin, db_session) -> None:  # noqa: ANN001
    catalogue = make_catalogue(db_session)
    make_page(db_session, catalogue, 1)
    assert client.get(f"/api/admin/catalogues/{catalogue.id}/panels/overview").status_code == 404
    assert client.get(f"/api/admin/catalogues/{uuid.uuid4()}/panels/overview").status_code == 404
    assert client.get(f"/api/admin/catalogues/{catalogue.id}/panels/overview?limit=0").status_code == 422

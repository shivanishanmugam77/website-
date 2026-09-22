"""Looking at the products found inside the panels, and forgetting them when a panel changes,
through the real API, database and file store."""

from __future__ import annotations

import io
import uuid

import pytest
from PIL import Image
from sqlalchemy import func, select

from app.models import DetectedObject
from app.models.enums import SegmentationStatus, UserRole
from tests.factories import TEST_PASSWORD, make_catalogue, make_page, make_user
from tests.helpers import sign_in
from tests.product_factory import add_find
from tests.test_panel_editing_api import editable_page

LOW, FAILED = SegmentationStatus.LOW_CONFIDENCE, SegmentationStatus.FAILED
KINDS = ["crop", "cutout", "mask", "thumbnail", "white"]


@pytest.fixture
def admin(client, db_session):  # noqa: ANN001, ANN201
    user = make_user(db_session, password=TEST_PASSWORD, role=UserRole.ADMIN)
    sign_in(client, user)
    return user


def base(catalogue) -> str:  # noqa: ANN001
    return f"/api/admin/catalogues/{catalogue.id}"


def picture(response) -> Image.Image:  # noqa: ANN001
    assert response.status_code == 200, response.text
    return Image.open(io.BytesIO(response.content))


def keys_of(found) -> list[str]:  # noqa: ANN001
    outline = found.segmentation
    columns = (outline.mask_key, outline.crop_key, outline.cutout_key, outline.white_bg_key, outline.thumbnail_key)
    return [key for key in columns if key]


def finds_in(session, panel) -> int:  # noqa: ANN001
    return session.scalar(
        select(func.count()).select_from(DetectedObject).where(DetectedObject.source_image_id == panel.id)
    )


def page_with_finds(session, storage):  # noqa: ANN001, ANN201
    """The 1200 x 800 test page: a desk and a cabinet (no outline) in the big panel, a chair in
    the small one whose outline is doubtful."""
    catalogue, page, big, small = editable_page(session, storage)
    desk = add_find(session, storage, catalogue, big)
    cabinet = add_find(
        session, storage, catalogue, big, label="cabinet", box=(300.0, 450.0, 450.0, 600.0),
        status=FAILED, note="the outline is empty",
    )  # fmt: skip
    chair = add_find(
        session, storage, catalogue, small, label="office chair", box=(20.0, 20.0, 200.0, 150.0),
        status=LOW, note="the outline fills only 20% of the find's box",
    )  # fmt: skip
    return catalogue, page, big, small, desk, cabinet, chair


# ------------------------------------------------------------------------------ access
@pytest.mark.parametrize(
    "path",
    [
        f"/api/admin/catalogues/{uuid.uuid4()}/pages/1/products",
        f"/api/admin/catalogues/{uuid.uuid4()}/pages/1/products/preview",
        f"/api/admin/catalogues/{uuid.uuid4()}/products/overview",
        f"/api/admin/catalogues/{uuid.uuid4()}/products/{uuid.uuid4()}/image/cutout",
    ],
)
def test_only_admins_may_look_at_the_products(client, db_session, path: str) -> None:  # noqa: ANN001
    assert client.get(path).status_code == 401  # not signed in
    customer = make_user(db_session, password=TEST_PASSWORD, role=UserRole.CUSTOMER)
    sign_in(client, customer)
    assert client.get(path).status_code == 403  # signed in, but not an admin


# ------------------------------------------------------------------------------ the list
def test_a_page_lists_its_finds_panel_by_panel(client, admin, db_session, storage) -> None:  # noqa: ANN001
    catalogue, _page, big, small, desk, cabinet, chair = page_with_finds(db_session, storage)
    response = client.get(f"{base(catalogue)}/pages/1/products")
    assert response.status_code == 200, response.text
    body = response.json()
    assert [f["id"] for f in body] == [str(desk.id), str(cabinet.id), str(chair.id)]

    first = body[0]
    assert (first["panel_id"], first["panel_index"], first["label"]) == (str(big.id), 1, "desk")
    assert first["confidence"] == 0.8 and first["bbox"] == [50, 50, 300, 400]
    assert (first["outline_status"], first["outline_score"], first["note"]) == ("SUCCEEDED", 0.9, None)
    assert sorted(first["images"]) == KINDS
    assert first["images"]["cutout"] == f"{base(catalogue)}/products/{desk.id}/image/cutout"

    assert (body[1]["outline_status"], body[1]["note"]) == ("FAILED", "the outline is empty")
    assert sorted(body[1]["images"]) == ["crop"]  # a failed outline has only the crop
    assert (body[2]["panel_id"], body[2]["panel_index"]) == (str(small.id), 2)
    assert body[2]["outline_status"] == "LOW_CONFIDENCE" and "20%" in body[2]["note"]


def test_a_page_without_finds_lists_nothing(client, admin, db_session, storage) -> None:  # noqa: ANN001
    catalogue, *_ = editable_page(db_session, storage)
    assert client.get(f"{base(catalogue)}/pages/1/products").json() == []
    assert client.get(f"{base(catalogue)}/pages/9/products").status_code == 404


# ------------------------------------------------------------------------------ the pictures
def test_every_kind_of_picture_can_be_fetched(client, admin, db_session, storage) -> None:  # noqa: ANN001
    catalogue, _page, _big, _small, desk, *_ = page_with_finds(db_session, storage)
    where = f"{base(catalogue)}/products/{desk.id}/image"

    crop = client.get(f"{where}/crop")
    assert crop.headers["content-type"] == "image/jpeg" and picture(crop).size == (250, 350)
    mask = client.get(f"{where}/mask")
    assert mask.headers["content-type"] == "image/png"
    assert (picture(mask).mode, picture(mask).size) == ("L", (500, 600))  # the size of the panel
    cutout = client.get(f"{where}/cutout")
    assert cutout.headers["content-type"] == "image/png"
    assert (picture(cutout).mode, picture(cutout).size) == ("RGBA", (250, 350))
    white = client.get(f"{where}/white")
    assert white.headers["content-type"] == "image/jpeg" and picture(white).size == (250, 350)
    thumbnail = client.get(f"{where}/thumbnail")
    assert thumbnail.headers["content-type"] == "image/jpeg" and max(picture(thumbnail).size) == 64


def test_a_picture_that_was_never_made_is_not_found(client, admin, db_session, storage) -> None:  # noqa: ANN001
    catalogue, _page, _big, _small, _desk, cabinet, _chair = page_with_finds(db_session, storage)
    where = f"{base(catalogue)}/products/{cabinet.id}/image"
    assert client.get(f"{where}/crop").status_code == 200
    for kind in ("mask", "cutout", "white", "thumbnail"):
        assert client.get(f"{where}/{kind}").status_code == 404, kind


def test_pictures_are_only_served_for_their_own_catalogue(client, admin, db_session, storage) -> None:  # noqa: ANN001
    _catalogue, _page, _big, _small, desk, *_ = page_with_finds(db_session, storage)
    other = make_catalogue(db_session)
    assert client.get(f"{base(other)}/products/{desk.id}/image/cutout").status_code == 404
    unknown = client.get(f"{base(other)}/products/{uuid.uuid4()}/image/cutout")
    assert unknown.status_code == 404


def test_an_unknown_kind_of_picture_is_refused(client, admin, db_session, storage) -> None:  # noqa: ANN001
    catalogue, _page, _big, _small, desk, *_ = page_with_finds(db_session, storage)
    assert client.get(f"{base(catalogue)}/products/{desk.id}/image/poster").status_code == 422


# ------------------------------------------------------------------------------ the preview
def test_the_preview_shows_every_panel_with_its_finds(client, admin, db_session, storage) -> None:  # noqa: ANN001
    catalogue, *_ = page_with_finds(db_session, storage)
    response = client.get(f"{base(catalogue)}/pages/1/products/preview")
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["cache-control"] == "no-store"
    assert picture(response).format == "JPEG"


def test_the_preview_still_works_when_an_outline_file_is_missing(client, admin, db_session, storage) -> None:  # noqa: ANN001
    catalogue, _page, _big, _small, desk, *_ = page_with_finds(db_session, storage)
    storage.delete(desk.segmentation.mask_key)
    assert client.get(f"{base(catalogue)}/pages/1/products/preview").status_code == 200


def test_the_preview_of_a_page_without_panels_is_not_found(client, admin, db_session) -> None:  # noqa: ANN001
    catalogue = make_catalogue(db_session)
    make_page(db_session, catalogue, 1)
    assert client.get(f"{base(catalogue)}/pages/1/products/preview").status_code == 404


# ------------------------------------------------------------------------------ the overview
def test_the_overview_shows_the_products_that_have_a_picture(client, admin, db_session, storage) -> None:  # noqa: ANN001
    catalogue, *_ = page_with_finds(db_session, storage)
    two = client.get(f"{base(catalogue)}/products/overview?columns=2")
    assert two.headers["cache-control"] == "no-store"
    assert picture(two).size == (600, 334)  # the desk and the chair: the failed cabinet has no thumbnail
    assert picture(client.get(f"{base(catalogue)}/products/overview?columns=1")).size == (300, 668)


def test_the_overview_can_start_further_on(client, admin, db_session, storage) -> None:  # noqa: ANN001
    catalogue, *_ = page_with_finds(db_session, storage)
    later = client.get(f"{base(catalogue)}/products/overview?columns=1&start=2")
    assert picture(later).size == (300, 334)
    assert client.get(f"{base(catalogue)}/products/overview?start=99").status_code == 404


@pytest.mark.parametrize("query", ["limit=0", "limit=61", "columns=0", "columns=9", "start=0"])
def test_the_overview_refuses_silly_numbers(client, admin, db_session, storage, query: str) -> None:  # noqa: ANN001
    catalogue, *_ = page_with_finds(db_session, storage)
    assert client.get(f"{base(catalogue)}/products/overview?{query}").status_code == 422


def test_the_overview_of_an_unknown_or_empty_catalogue_is_not_found(client, admin, db_session) -> None:  # noqa: ANN001
    assert client.get(f"/api/admin/catalogues/{uuid.uuid4()}/products/overview").status_code == 404
    empty = make_catalogue(db_session)
    assert client.get(f"{base(empty)}/products/overview").status_code == 404


# ------------------------------------------------------------------------------ panel changes
def test_moving_a_panel_forgets_the_products_found_in_it(client, admin, db_session, storage) -> None:  # noqa: ANN001
    catalogue, _page, big, small, desk, cabinet, chair = page_with_finds(db_session, storage)
    gone = keys_of(desk) + keys_of(cabinet)
    kept = keys_of(chair)
    assert all(storage.exists(key) for key in gone + kept)

    moved = client.patch(f"{base(catalogue)}/pages/1/panels/{big.id}", json={"bbox": [110, 110, 500, 600]})
    assert moved.status_code == 200, moved.text

    assert finds_in(db_session, big) == 0 and finds_in(db_session, small) == 1
    assert not any(storage.exists(key) for key in gone)  # their pictures went with them
    assert all(storage.exists(key) for key in kept)  # the other panel is untouched


def test_a_move_that_changes_nothing_keeps_the_finds(client, admin, db_session, storage) -> None:  # noqa: ANN001
    catalogue, _page, big, *_ = page_with_finds(db_session, storage)
    same = client.patch(f"{base(catalogue)}/pages/1/panels/{big.id}", json={"bbox": [100, 100, 600, 700]})
    assert same.status_code == 200
    assert finds_in(db_session, big) == 2


def test_deleting_a_panel_forgets_its_products_and_their_pictures(client, admin, db_session, storage) -> None:  # noqa: ANN001
    catalogue, _page, big, small, desk, cabinet, _chair = page_with_finds(db_session, storage)
    gone = keys_of(desk) + keys_of(cabinet)

    assert client.delete(f"{base(catalogue)}/pages/1/panels/{big.id}").status_code == 200
    assert not any(storage.exists(key) for key in gone)
    assert finds_in(db_session, small) == 1

    # Undoing the deletion brings the panel back, but not the products found on the old one.
    assert client.post(f"{base(catalogue)}/pages/1/panels/undo").status_code == 200
    assert finds_in(db_session, big) == 0


def test_undoing_a_move_forgets_products_found_in_the_meantime(client, admin, db_session, storage) -> None:  # noqa: ANN001
    catalogue, _page, big, _small, desk, cabinet, _chair = page_with_finds(db_session, storage)
    client.patch(f"{base(catalogue)}/pages/1/panels/{big.id}", json={"bbox": [110, 110, 500, 600]})
    again = add_find(db_session, storage, catalogue, big, box=(20.0, 20.0, 200.0, 300.0))
    assert finds_in(db_session, big) == 1 and all(storage.exists(k) for k in keys_of(again))
    keys = keys_of(again)

    assert client.post(f"{base(catalogue)}/pages/1/panels/undo").status_code == 200
    assert finds_in(db_session, big) == 0  # the panel changed back, so those finds no longer fit
    assert not any(storage.exists(key) for key in keys)

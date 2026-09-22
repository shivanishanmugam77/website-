"""Reviewing product candidates through the real API, database and file store."""

from __future__ import annotations

import uuid

import pytest

from app.models.enums import CatalogueStatus, UserRole
from tests.candidate_factory import add_ocr_lines, catalogue_with_candidates
from tests.factories import TEST_PASSWORD, make_catalogue, make_page, make_user
from tests.helpers import sign_in
from tests.page_factory import blank_page
from tests.product_factory import add_find
from tests.test_panel_editing_api import BIG, add_region, jpeg


@pytest.fixture
def admin(client, db_session):  # noqa: ANN001, ANN201
    user = make_user(db_session, password=TEST_PASSWORD, role=UserRole.ADMIN)
    sign_in(client, user)
    return user


def base(catalogue) -> str:  # noqa: ANN001
    return f"/api/admin/catalogues/{catalogue.id}"


# ------------------------------------------------------------------------------ access
@pytest.mark.parametrize(
    "path",
    [
        f"/api/admin/catalogues/{uuid.uuid4()}/candidates/assemble",
        f"/api/admin/catalogues/{uuid.uuid4()}/candidates",
        f"/api/admin/catalogues/{uuid.uuid4()}/candidates/{uuid.uuid4()}",
        f"/api/admin/catalogues/{uuid.uuid4()}/candidates/review",
    ],
)
def test_only_admins_may_reach_candidates(client, db_session, path: str) -> None:  # noqa: ANN001
    method = client.post if path.endswith("/assemble") else client.get
    assert method(path).status_code == 401
    customer = make_user(db_session, password=TEST_PASSWORD, role=UserRole.CUSTOMER)
    sign_in(client, customer)
    assert method(path).status_code == 403


# ------------------------------------------------------------------------------ review page
def test_the_review_page_loads_and_names_the_catalogue(client, admin, db_session, storage) -> None:  # noqa: ANN001
    catalogue, *_ = catalogue_with_candidates(db_session, storage)
    response = client.get(f"{base(catalogue)}/candidates/review")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["cache-control"] == "no-store"
    assert "content-security-policy" in response.headers
    assert catalogue.name in response.text
    assert str(catalogue.id) in response.text


def test_the_review_page_escapes_the_catalogue_name(client, admin, db_session) -> None:  # noqa: ANN001
    catalogue = make_catalogue(db_session, name="<script>evil()</script>")
    response = client.get(f"{base(catalogue)}/candidates/review")
    assert response.status_code == 200
    assert "<script>evil()" not in response.text
    assert "&lt;script&gt;" in response.text


def test_the_review_page_of_an_unknown_catalogue_is_not_found(client, admin) -> None:  # noqa: ANN001
    assert client.get(f"/api/admin/catalogues/{uuid.uuid4()}/candidates/review").status_code == 404


# ------------------------------------------------------------------------------ assemble
def test_assembling_a_finished_catalogue_groups_its_finds(client, admin, db_session, storage) -> None:  # noqa: ANN001
    catalogue, *_ = catalogue_with_candidates(db_session, storage)
    response = client.post(f"{base(catalogue)}/candidates/assemble")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["candidates_created"] == 1
    assert body["ready"] + body["needs_review"] == 1
    assert body["duplicate_flags_created"] == 0


def test_assembling_before_processing_has_finished_is_refused(client, admin, db_session, storage) -> None:  # noqa: ANN001
    catalogue, *_ = catalogue_with_candidates(db_session, storage, catalogue_status=CatalogueStatus.PROCESSING)
    response = client.post(f"{base(catalogue)}/candidates/assemble")
    assert response.status_code == 409
    assert "finished processing" in response.json()["detail"]


def test_assembling_an_unknown_catalogue_is_not_found(client, admin) -> None:  # noqa: ANN001
    response = client.post(f"/api/admin/catalogues/{uuid.uuid4()}/candidates/assemble")
    assert response.status_code == 404


# ------------------------------------------------------------------------------ list
def test_the_list_shows_a_summary_of_each_candidate(client, admin, db_session, storage) -> None:  # noqa: ANN001
    catalogue, *_ = catalogue_with_candidates(db_session, storage)
    client.post(f"{base(catalogue)}/candidates/assemble")

    response = client.get(f"{base(catalogue)}/candidates")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    (item,) = body["items"]
    assert item["name"] == "四人位职员桌"
    assert item["model_code"] == "YY-11"
    assert item["page_numbers"] == [1]
    assert item["status"] in ("READY", "NEEDS_REVIEW")
    assert item["thumbnail_url"] is not None and item["thumbnail_url"].startswith(base(catalogue))


def test_an_empty_catalogue_lists_no_candidates(client, admin, db_session) -> None:  # noqa: ANN001
    catalogue = make_catalogue(db_session, status=CatalogueStatus.COMPLETED)
    response = client.get(f"{base(catalogue)}/candidates")
    assert response.status_code == 200
    assert response.json() == {"total": 0, "limit": 25, "offset": 0, "items": []}


def test_the_list_can_be_paged(client, admin, db_session, storage) -> None:  # noqa: ANN001
    catalogue = make_catalogue(db_session, status=CatalogueStatus.COMPLETED)
    for number in (1, 2, 3):
        page = make_page(db_session, catalogue, number)
        image = blank_page(1200, 800)
        page.image_key = f"catalogues/{catalogue.id}/pages/{number}.jpg"
        storage.put_bytes(page.image_key, jpeg(image))
        page.width_px, page.height_px = 1200, 800
        panel = add_region(db_session, storage, catalogue, page, image, BIG)
        add_ocr_lines(db_session, page, lines=[])
        add_find(db_session, storage, catalogue, panel, label="desk")
    client.post(f"{base(catalogue)}/candidates/assemble")

    page1 = client.get(f"{base(catalogue)}/candidates?limit=2&offset=0").json()
    page2 = client.get(f"{base(catalogue)}/candidates?limit=2&offset=2").json()
    assert page1["total"] == 3 and len(page1["items"]) == 2
    assert len(page2["items"]) == 1
    assert {i["id"] for i in page1["items"]} & {i["id"] for i in page2["items"]} == set()


# ------------------------------------------------------------------------------ detail
def test_the_detail_view_has_every_field_flag_and_find(client, admin, db_session, storage) -> None:  # noqa: ANN001
    catalogue, page, big, small, desk, chair = catalogue_with_candidates(db_session, storage)
    client.post(f"{base(catalogue)}/candidates/assemble")
    (item,) = client.get(f"{base(catalogue)}/candidates").json()["items"]

    response = client.get(f"{base(catalogue)}/candidates/{item['id']}")
    assert response.status_code == 200
    body = response.json()
    assert body["catalogue_id"] == str(catalogue.id)
    assert body["origin"] == "AI"
    assert sorted(body["detected_object_ids"]) == sorted([str(desk.id), str(chair.id)])
    assert set(body["images"].keys()) == {str(desk.id), str(chair.id)}
    assert "cutout" in body["images"][str(desk.id)]
    assert body["fields"]["model_code"]["value"] == "YY-11"
    assert body["reasoning"] and body["duplicate_flags"] == []


def test_the_detail_view_lists_open_duplicate_flags(client, admin, db_session, storage) -> None:  # noqa: ANN001
    catalogue = make_catalogue(db_session, status=CatalogueStatus.COMPLETED)
    for number in (1, 2):
        page = make_page(db_session, catalogue, number)
        image = blank_page(1200, 800)
        page.image_key = f"catalogues/{catalogue.id}/pages/{number}.jpg"
        storage.put_bytes(page.image_key, jpeg(image))
        page.width_px, page.height_px = 1200, 800
        panel = add_region(db_session, storage, catalogue, page, image, BIG)
        add_ocr_lines(db_session, page)
        add_find(db_session, storage, catalogue, panel, label="desk")
    client.post(f"{base(catalogue)}/candidates/assemble")

    items = client.get(f"{base(catalogue)}/candidates").json()["items"]
    detail = client.get(f"{base(catalogue)}/candidates/{items[1]['id']}").json()
    assert len(detail["duplicate_flags"]) == 1
    flag = detail["duplicate_flags"][0]
    assert flag["other_candidate_id"] == items[0]["id"]
    assert flag["signal"] == "SKU" and flag["status"] == "OPEN"


def test_an_unknown_candidate_is_not_found(client, admin, db_session, storage) -> None:  # noqa: ANN001
    catalogue, *_ = catalogue_with_candidates(db_session, storage)
    response = client.get(f"{base(catalogue)}/candidates/{uuid.uuid4()}")
    assert response.status_code == 404


def test_a_candidate_is_only_served_under_its_own_catalogue(client, admin, db_session, storage) -> None:  # noqa: ANN001
    catalogue, *_ = catalogue_with_candidates(db_session, storage)
    client.post(f"{base(catalogue)}/candidates/assemble")
    (item,) = client.get(f"{base(catalogue)}/candidates").json()["items"]

    other = make_catalogue(db_session)
    response = client.get(f"{base(other)}/candidates/{item['id']}")
    assert response.status_code == 404

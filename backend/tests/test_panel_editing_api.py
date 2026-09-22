"""Correcting a page's photo panels by hand, through the real API, database and file store."""

from __future__ import annotations

import io
import uuid

import pytest
from PIL import Image, ImageDraw
from sqlalchemy import select

from app.models import AuditLog, PanelEdit, ProcessingJob, SourceImage
from app.models.enums import (
    AuditAction,
    DataOrigin,
    JobStatus,
    JobType,
    PageStatus,
    PanelEditAction,
    SourceImageKind,
    UserRole,
)
from app.services import panel_edits
from app.services.pipeline import page_image_key, source_image_key
from tests.factories import TEST_PASSWORD, make_catalogue, make_page, make_user
from tests.helpers import sign_in
from tests.page_factory import blank_page

RED, BLUE = (200, 30, 30), (30, 30, 200)
BIG, SMALL = (100, 100, 600, 700), (700, 100, 1100, 400)  # the two panels a page starts with


@pytest.fixture
def admin(client, db_session):  # noqa: ANN001, ANN201
    user = make_user(db_session, password=TEST_PASSWORD, role=UserRole.ADMIN)
    sign_in(client, user)
    return user


def jpeg(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=95)
    return buffer.getvalue()


def add_region(session, storage, catalogue, page, image, box, origin=DataOrigin.AI):  # noqa: ANN001, ANN201
    """A panel exactly as the pipeline would have left it."""
    crop = image.crop(box)
    panel_id = uuid.uuid4()
    key = source_image_key(catalogue.id, panel_id)
    stored = storage.put_bytes(key, jpeg(crop))
    region = SourceImage(
        id=panel_id,
        page_id=page.id,
        kind=SourceImageKind.REGION,
        storage_key=key,
        width_px=crop.width,
        height_px=crop.height,
        bbox_x0=box[0],
        bbox_y0=box[1],
        bbox_x1=box[2],
        bbox_y1=box[3],
        sha256=stored.sha256,
        phash="0" * 16,
        origin=origin,
    )
    session.add(region)
    session.flush()
    return region


def editable_page(session, storage, number: int = 1):  # noqa: ANN001, ANN201
    """A 1200 x 800 page with a red photo and a blue one, and a panel around each."""
    catalogue = make_catalogue(session)
    page = make_page(session, catalogue, number)
    image = blank_page(1200, 800)
    draw = ImageDraw.Draw(image)
    draw.rectangle(BIG, fill=RED)
    draw.rectangle(SMALL, fill=BLUE)
    page.image_key = page_image_key(catalogue.id, number)
    storage.put_bytes(page.image_key, jpeg(image))
    page.width_px, page.height_px, page.status = 1200, 800, PageStatus.PANELS_FOUND
    big = add_region(session, storage, catalogue, page, image, BIG)
    small = add_region(session, storage, catalogue, page, image, SMALL)
    session.flush()
    return catalogue, page, big, small


def base(catalogue, page_number: int = 1) -> str:  # noqa: ANN001
    return f"/api/admin/catalogues/{catalogue.id}/pages/{page_number}/panels"


def by_id(panels: list[dict], panel_id) -> dict:  # noqa: ANN001
    return next(p for p in panels if p["id"] == str(panel_id))


def cut_out(storage, catalogue, panel_id) -> Image.Image:  # noqa: ANN001
    data = storage.read_bytes(source_image_key(catalogue.id, uuid.UUID(str(panel_id))))
    return Image.open(io.BytesIO(data)).convert("RGB")


def edits_of(session, page) -> list[PanelEdit]:  # noqa: ANN001
    return list(
        session.scalars(select(PanelEdit).where(PanelEdit.page_id == page.id).order_by(PanelEdit.seq))
    )


def audit_actions(session) -> set[AuditAction]:  # noqa: ANN001
    """Which actions are in the audit trail (rows of one test share a timestamp: no order)."""
    return {row.action for row in session.scalars(select(AuditLog))}


def drawn(panels: list[dict]) -> dict:
    """The panel an admin drew (the only one with origin HUMAN)."""
    return next(p for p in panels if p["origin"] == "HUMAN")


def is_red(pixel: tuple) -> bool:
    return pixel[0] > 150 and pixel[1] < 90 and pixel[2] < 90


# ------------------------------------------------------------------------------ drawing
def test_drawing_a_panel_adds_it_and_cuts_it_out(client, admin, db_session, storage) -> None:  # noqa: ANN001
    catalogue, page, *_ = editable_page(db_session, storage)
    response = client.post(base(catalogue), json={"bbox": [110, 110, 300, 300]})
    assert response.status_code == 201, response.text
    panels = response.json()
    assert len(panels) == 3
    new = next(p for p in panels if p["origin"] == "HUMAN")
    assert new["bbox"] == [110, 110, 300, 300] and (new["width_px"], new["height_px"]) == (190, 190)

    crop = cut_out(storage, catalogue, new["id"])
    assert crop.size == (190, 190) and is_red(crop.getpixel((95, 95)))

    (edit,) = edits_of(db_session, page)
    assert edit.action is PanelEditAction.CREATE and edit.seq == 1 and edit.before is None
    assert edit.after == {"bbox": [110, 110, 300, 300], "origin": "HUMAN"}
    assert edit.actor_id == admin.id and edit.undone_at is None
    audit = db_session.scalar(select(AuditLog).where(AuditLog.action == AuditAction.PANEL_CREATED))
    assert audit.actor_id == admin.id and audit.entity_id == new["id"]
    assert audit.details["page_number"] == 1 and audit.details["after"]["origin"] == "HUMAN"


def test_a_box_is_clamped_to_the_page(client, admin, db_session, storage) -> None:  # noqa: ANN001
    catalogue, *_ = editable_page(db_session, storage)
    new = drawn(client.post(base(catalogue), json={"bbox": [-50, -20, 300.4, 300.6]}).json())
    assert new["bbox"] == [0, 0, 300, 301]


@pytest.mark.parametrize(
    "box",
    [[10, 10, 15, 300], [300, 10, 10, 300], [10, 10, 10, 10], [5000, 5000, 6000, 6000], [1, 2, 3]],
)
def test_bad_boxes_are_refused_and_change_nothing(client, admin, db_session, storage, box) -> None:  # noqa: ANN001
    catalogue, page, *_ = editable_page(db_session, storage)
    assert client.post(base(catalogue), json={"bbox": box}).status_code == 422
    assert edits_of(db_session, page) == []
    assert len(client.get(base(catalogue)).json()) == 2


def test_a_page_holds_a_limited_number_of_panels(client, admin, db_session, storage, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(panel_edits, "MAX_PANELS_PER_PAGE", 3)
    catalogue, *_ = editable_page(db_session, storage)
    assert client.post(base(catalogue), json={"bbox": [110, 110, 300, 300]}).status_code == 201
    refused = client.post(base(catalogue), json={"bbox": [310, 110, 500, 300]})
    assert refused.status_code == 409 and "at most 3" in refused.json()["detail"]


# ------------------------------------------------------------------------------ moving
def test_moving_a_panel_recuts_it_and_marks_it_adjusted(client, admin, db_session, storage) -> None:  # noqa: ANN001
    catalogue, page, big, _small = editable_page(db_session, storage)
    old_url = by_id(client.get(base(catalogue)).json(), big.id)["url"]

    response = client.patch(f"{base(catalogue)}/{big.id}", json={"bbox": [110, 110, 500, 600]})
    assert response.status_code == 200, response.text
    moved = by_id(response.json(), big.id)
    assert moved["bbox"] == [110, 110, 500, 600] and moved["origin"] == "AI_HUMAN_REVIEW"
    assert (moved["width_px"], moved["height_px"]) == (390, 490)
    assert moved["url"] != old_url  # the picture changed, so browsers must fetch it again

    crop = cut_out(storage, catalogue, big.id)
    assert crop.size == (390, 490) and is_red(crop.getpixel((195, 245)))
    (edit,) = edits_of(db_session, page)
    assert edit.action is PanelEditAction.UPDATE and edit.panel_id == big.id
    assert edit.before == {"bbox": [100, 100, 600, 700], "origin": "AI"}
    assert edit.after == {"bbox": [110, 110, 500, 600], "origin": "AI_HUMAN_REVIEW"}
    assert AuditAction.PANEL_UPDATED in audit_actions(db_session)


def test_a_move_that_changes_nothing_is_not_recorded(client, admin, db_session, storage) -> None:  # noqa: ANN001
    catalogue, page, big, _small = editable_page(db_session, storage)
    assert client.patch(f"{base(catalogue)}/{big.id}", json={"bbox": list(BIG)}).status_code == 200
    assert edits_of(db_session, page) == []


def test_a_panel_the_admin_drew_stays_theirs_when_moved(client, admin, db_session, storage) -> None:  # noqa: ANN001
    catalogue, *_ = editable_page(db_session, storage)
    new = drawn(client.post(base(catalogue), json={"bbox": [110, 110, 300, 300]}).json())
    moved = client.patch(f"{base(catalogue)}/{new['id']}", json={"bbox": [120, 120, 320, 320]})
    assert by_id(moved.json(), new["id"])["origin"] == "HUMAN"


# ------------------------------------------------------------------------------ deleting
def test_deleting_removes_the_panel_and_its_file(client, admin, db_session, storage) -> None:  # noqa: ANN001
    catalogue, page, big, small = editable_page(db_session, storage)
    response = client.delete(f"{base(catalogue)}/{big.id}")
    assert response.status_code == 200
    assert [p["id"] for p in response.json()] == [str(small.id)]
    assert [p["index"] for p in response.json()] == [1]  # renumbered
    assert not storage.exists(source_image_key(catalogue.id, big.id))
    assert db_session.get(SourceImage, big.id) is None
    (edit,) = edits_of(db_session, page)
    assert edit.action is PanelEditAction.DELETE and edit.after is None
    assert edit.before == {"bbox": [100, 100, 600, 700], "origin": "AI"}
    assert AuditAction.PANEL_DELETED in audit_actions(db_session)


# ------------------------------------------------------------------------------ undoing
def test_undoing_a_deletion_brings_the_panel_back_as_it_was(client, admin, db_session, storage) -> None:  # noqa: ANN001
    catalogue, page, big, _small = editable_page(db_session, storage)
    client.delete(f"{base(catalogue)}/{big.id}")
    response = client.post(f"{base(catalogue)}/undo")
    assert response.status_code == 200, response.text
    back = by_id(response.json(), big.id)  # the same id
    assert back["bbox"] == [100, 100, 600, 700] and back["origin"] == "AI"
    assert is_red(cut_out(storage, catalogue, big.id).getpixel((250, 300)))
    (edit,) = edits_of(db_session, page)
    assert edit.undone_at is not None
    assert AuditAction.PANEL_EDIT_UNDONE in audit_actions(db_session)


def test_undoing_a_move_restores_the_old_box_and_origin(client, admin, db_session, storage) -> None:  # noqa: ANN001
    catalogue, _page, big, _small = editable_page(db_session, storage)
    client.patch(f"{base(catalogue)}/{big.id}", json={"bbox": [110, 110, 500, 600]})
    restored = by_id(client.post(f"{base(catalogue)}/undo").json(), big.id)
    assert restored["bbox"] == [100, 100, 600, 700] and restored["origin"] == "AI"
    assert cut_out(storage, catalogue, big.id).size == (500, 600)


def test_undoing_a_drawn_panel_removes_it_and_its_file(client, admin, db_session, storage) -> None:  # noqa: ANN001
    catalogue, *_ = editable_page(db_session, storage)
    new = drawn(client.post(base(catalogue), json={"bbox": [110, 110, 300, 300]}).json())
    assert len(client.post(f"{base(catalogue)}/undo").json()) == 2
    assert not storage.exists(source_image_key(catalogue.id, uuid.UUID(new["id"])))


def test_undo_walks_back_through_the_changes_one_by_one_and_then_stops(client, admin, db_session, storage) -> None:  # noqa: ANN001
    catalogue, page, big, small = editable_page(db_session, storage)
    client.post(base(catalogue), json={"bbox": [110, 110, 300, 300]})  # 1: draw
    client.patch(f"{base(catalogue)}/{big.id}", json={"bbox": [110, 110, 500, 600]})  # 2: move
    client.delete(f"{base(catalogue)}/{small.id}")  # 3: delete

    after_undo = [client.post(f"{base(catalogue)}/undo") for _ in range(3)]
    assert [len(r.json()) for r in after_undo] == [3, 3, 2]  # back, still moved-back, drawn gone
    assert by_id(after_undo[0].json(), small.id)["bbox"] == list(SMALL)
    assert by_id(after_undo[1].json(), big.id)["bbox"] == list(BIG)

    nothing = client.post(f"{base(catalogue)}/undo")
    assert nothing.status_code == 409 and "nothing to undo" in nothing.json()["detail"].lower()
    assert all(e.undone_at is not None for e in edits_of(db_session, page))


def test_a_new_change_after_an_undo_is_undone_first(client, admin, db_session, storage) -> None:  # noqa: ANN001
    catalogue, page, big, _small = editable_page(db_session, storage)
    client.patch(f"{base(catalogue)}/{big.id}", json={"bbox": [110, 110, 500, 600]})
    client.post(f"{base(catalogue)}/undo")
    client.patch(f"{base(catalogue)}/{big.id}", json={"bbox": [120, 120, 400, 500]})
    undone = client.post(f"{base(catalogue)}/undo").json()
    assert by_id(undone, big.id)["bbox"] == [100, 100, 600, 700]
    assert [e.seq for e in edits_of(db_session, page)] == [1, 2]


# ------------------------------------------------------------------------------ refusals
def test_unknown_panels_and_pages_are_404(client, admin, db_session, storage) -> None:  # noqa: ANN001
    catalogue, _page, big, _small = editable_page(db_session, storage)
    _other, _p, other_big, _ = editable_page(db_session, storage)
    ghost = uuid.uuid4()
    box = {"bbox": [110, 110, 300, 300]}
    assert client.patch(f"{base(catalogue)}/{ghost}", json=box).status_code == 404
    assert client.delete(f"{base(catalogue)}/{ghost}").status_code == 404
    assert client.patch(f"{base(catalogue)}/{other_big.id}", json=box).status_code == 404  # not this page
    assert client.patch(f"{base(catalogue, 9)}/{big.id}", json=box).status_code == 404
    assert client.post(base(catalogue, 9), json=box).status_code == 404
    assert client.post(f"{base(catalogue, 9)}/undo").status_code == 404


def test_a_page_without_a_picture_cannot_be_edited(client, admin, db_session) -> None:  # noqa: ANN001
    catalogue = make_catalogue(db_session)
    make_page(db_session, catalogue, 1)
    refused = client.post(base(catalogue), json={"bbox": [10, 10, 300, 300]})
    assert refused.status_code == 409 and "no picture" in refused.json()["detail"]


def test_editing_waits_until_the_catalogue_has_finished_processing(client, admin, db_session, storage) -> None:  # noqa: ANN001
    catalogue, page, big, _small = editable_page(db_session, storage)
    db_session.add(
        ProcessingJob(
            catalogue_id=catalogue.id, job_type=JobType.CATALOGUE_PROCESS, status=JobStatus.RUNNING
        )
    )
    db_session.flush()
    box = {"bbox": [110, 110, 300, 300]}
    for response in (
        client.post(base(catalogue), json=box),
        client.patch(f"{base(catalogue)}/{big.id}", json=box),
        client.delete(f"{base(catalogue)}/{big.id}"),
        client.post(f"{base(catalogue)}/undo"),
    ):
        assert response.status_code == 409 and "being processed" in response.json()["detail"]
    assert edits_of(db_session, page) == []


# ------------------------------------------------------------------------------ who may edit
def _requests(catalogue, panel_id):  # noqa: ANN001, ANN202
    box = {"bbox": [110, 110, 300, 300]}
    return [
        ("POST", base(catalogue), box),
        ("PATCH", f"{base(catalogue)}/{panel_id}", box),
        ("DELETE", f"{base(catalogue)}/{panel_id}", None),
        ("POST", f"{base(catalogue)}/undo", None),
        ("GET", f"{base(catalogue)}/editor", None),
    ]


def test_only_admins_may_edit_panels(client, db_session, storage) -> None:  # noqa: ANN001
    catalogue, _page, big, _small = editable_page(db_session, storage)
    for method, path, body in _requests(catalogue, big.id):
        assert client.request(method, path, json=body).status_code == 401  # not signed in
    sign_in(client, make_user(db_session, password=TEST_PASSWORD, role=UserRole.CUSTOMER))
    for method, path, body in _requests(catalogue, big.id):
        assert client.request(method, path, json=body).status_code == 403  # signed in, not admin


def test_edits_from_an_untrusted_origin_are_refused(client, admin, db_session, storage) -> None:  # noqa: ANN001
    catalogue, page, big, _small = editable_page(db_session, storage)
    evil = {"Origin": "https://evil.example"}
    box = {"bbox": [110, 110, 300, 300]}
    assert client.post(base(catalogue), json=box, headers=evil).status_code == 403
    assert client.patch(f"{base(catalogue)}/{big.id}", json=box, headers=evil).status_code == 403
    assert client.delete(f"{base(catalogue)}/{big.id}", headers=evil).status_code == 403
    assert client.post(f"{base(catalogue)}/undo", headers=evil).status_code == 403
    assert edits_of(db_session, page) == []


# ------------------------------------------------------------------------------ the editor page
def test_the_editor_page_is_served_with_a_matching_nonce(client, admin, db_session, storage) -> None:  # noqa: ANN001
    catalogue, *_ = editable_page(db_session, storage, number=4)
    response = client.get(f"{base(catalogue, 4)}/editor")
    assert response.status_code == 200 and response.headers["content-type"].startswith("text/html")
    assert response.headers["cache-control"] == "no-store"
    policy = response.headers["content-security-policy"]
    nonce = policy.split("script-src 'nonce-")[1].split("'")[0]
    assert len(nonce) >= 16 and f'<script nonce="{nonce}">' in response.text
    assert f'<style nonce="{nonce}">' in response.text
    assert f'const CATALOGUE_ID = "{catalogue.id}";' in response.text
    assert "const PAGE = 4;" in response.text
    assert "__" not in response.text.replace("__proto__", "")  # every placeholder was filled
    assert "default-src 'none'" in policy and "'unsafe-inline'" not in policy


def test_each_editor_page_gets_a_fresh_nonce(client, admin, db_session, storage) -> None:  # noqa: ANN001
    catalogue, *_ = editable_page(db_session, storage)
    first = client.get(f"{base(catalogue)}/editor").headers["content-security-policy"]
    second = client.get(f"{base(catalogue)}/editor").headers["content-security-policy"]
    assert first != second


def test_an_unknown_page_has_no_editor(client, admin, db_session, storage) -> None:  # noqa: ANN001
    catalogue, *_ = editable_page(db_session, storage)
    assert client.get(f"{base(catalogue, 9)}/editor").status_code == 404


# ------------------------------------------------------------------------------ reprocessing
def reprocess(client, catalogue, **query):  # noqa: ANN001, ANN201
    return client.post(f"/api/admin/catalogues/{catalogue.id}/reprocess", params=query)


def test_reprocessing_is_refused_while_manual_corrections_exist(client, admin, db_session, storage, enqueued) -> None:  # noqa: ANN001
    catalogue, _page, big, _small = editable_page(db_session, storage)
    client.patch(f"{base(catalogue)}/{big.id}", json={"bbox": [110, 110, 500, 600]})

    refused = reprocess(client, catalogue)
    assert refused.status_code == 409
    assert "1 manual panel correction" in refused.json()["detail"]
    assert "discard_corrections=true" in refused.json()["detail"]
    assert enqueued == []


def test_reprocessing_can_discard_the_corrections_on_purpose(client, admin, db_session, storage, enqueued) -> None:  # noqa: ANN001
    catalogue, _page, big, small = editable_page(db_session, storage)
    client.patch(f"{base(catalogue)}/{big.id}", json={"bbox": [110, 110, 500, 600]})
    client.delete(f"{base(catalogue)}/{small.id}")

    assert reprocess(client, catalogue, discard_corrections="true").status_code == 202
    assert len(enqueued) == 1
    discarded = db_session.scalar(
        select(AuditLog).where(AuditLog.action == AuditAction.CORRECTIONS_DISCARDED)
    )
    assert discarded.details["corrections"] == 2 and discarded.entity_id == str(catalogue.id)


def test_undone_corrections_no_longer_block_reprocessing(client, admin, db_session, storage, enqueued) -> None:  # noqa: ANN001
    catalogue, *_ = editable_page(db_session, storage)
    client.post(base(catalogue), json={"bbox": [110, 110, 300, 300]})
    client.post(f"{base(catalogue)}/undo")
    assert reprocess(client, catalogue).status_code == 202
    assert db_session.scalar(
        select(AuditLog).where(AuditLog.action == AuditAction.CORRECTIONS_DISCARDED)
    ) is None


def test_reprocessing_an_untouched_catalogue_is_unchanged(client, admin, db_session, storage, enqueued) -> None:  # noqa: ANN001
    catalogue, *_ = editable_page(db_session, storage)
    assert reprocess(client, catalogue).status_code == 202


def test_corrections_on_one_catalogue_do_not_block_another(client, admin, db_session, storage, enqueued) -> None:  # noqa: ANN001
    edited, _page, big, _small = editable_page(db_session, storage)
    untouched, *_ = editable_page(db_session, storage)
    client.patch(f"{base(edited)}/{big.id}", json={"bbox": [110, 110, 500, 600]})
    assert reprocess(client, untouched).status_code == 202


# ------------------------------------------------------------------------------ listing
def test_the_panel_list_shows_where_each_panel_came_from(client, admin, db_session, storage) -> None:  # noqa: ANN001
    catalogue, _page, big, _small = editable_page(db_session, storage)
    client.patch(f"{base(catalogue)}/{big.id}", json={"bbox": [110, 110, 500, 600]})
    client.post(base(catalogue), json={"bbox": [700, 500, 900, 700]})
    origins = sorted(p["origin"] for p in client.get(base(catalogue)).json())
    assert origins == ["AI", "AI_HUMAN_REVIEW", "HUMAN"]

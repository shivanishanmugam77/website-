"""Upload, browse, view, reprocess and delete catalogues through the real API."""

from __future__ import annotations

import hashlib
import uuid

import pytest
from sqlalchemy import select

from app.core.config import get_settings
from app.models import AuditLog, Catalogue, ProcessingJob, Product
from app.models.enums import AuditAction, CatalogueStatus, JobStatus, UserRole
from app.services.pipeline import original_pdf_key, run_catalogue_pipeline
from tests.factories import (
    TEST_PASSWORD,
    make_product,
    make_stored_catalogue,
    make_supplier,
    make_user,
    settings_with,
)
from tests.helpers import sign_in, upload
from tests.pdf_factory import ImageSpec, PageSpec, blank_pages, build_pdf

PDF = build_pdf(
    [
        PageSpec(images=[ImageSpec(50, 600, 200, 150, px=150, color=(255, 0, 0))]),
        PageSpec(),
    ]
)
PIPELINE_SETTINGS = settings_with(min_embedded_image_px=100)


@pytest.fixture
def admin(client, db_session):  # noqa: ANN001, ANN201
    user = make_user(db_session, password=TEST_PASSWORD, role=UserRole.ADMIN)
    sign_in(client, user)
    return user


def stored_files(storage) -> list[str]:  # noqa: ANN001
    return sorted(p.name for p in storage.root.rglob("*") if p.is_file())


def process(db_session, storage, body: dict) -> None:  # noqa: ANN001
    """Run the worker's job in-process (the test double for a live Celery worker)."""
    run_catalogue_pipeline(
        db_session,
        storage,
        PIPELINE_SETTINGS,
        catalogue_id=uuid.UUID(body["id"]),
        job_id=uuid.UUID(body["job"]["id"]),
    )


# ------------------------------------------------------------------------------ access control
@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/api/admin/catalogues"),
        ("POST", "/api/admin/catalogues"),
        ("GET", f"/api/admin/catalogues/{uuid.uuid4()}"),
        ("GET", f"/api/admin/catalogues/{uuid.uuid4()}/pages"),
        ("GET", f"/api/admin/catalogues/{uuid.uuid4()}/pages/1/image"),
        ("GET", f"/api/admin/catalogues/{uuid.uuid4()}/pages/1/text"),
        ("GET", f"/api/admin/catalogues/{uuid.uuid4()}/pages/1/panels"),
        ("GET", f"/api/admin/catalogues/{uuid.uuid4()}/pages/1/panels/preview"),
        ("GET", f"/api/admin/catalogues/{uuid.uuid4()}/panels/overview"),
        ("POST", f"/api/admin/catalogues/{uuid.uuid4()}/reprocess"),
        ("DELETE", f"/api/admin/catalogues/{uuid.uuid4()}"),
        ("GET", "/api/admin/suppliers"),
    ],
)
def test_only_admins_may_touch_catalogues(client, db_session, method: str, path: str) -> None:  # noqa: ANN001
    assert client.request(method, path).status_code == 401  # not signed in
    customer = make_user(db_session, password=TEST_PASSWORD, role=UserRole.CUSTOMER)
    sign_in(client, customer)
    assert client.request(method, path).status_code == 403  # signed in, but not an admin


# ------------------------------------------------------------------------------ upload
def test_upload_stores_the_file_records_it_and_queues_processing(  # noqa: ANN001
    client, admin, db_session, storage, enqueued
) -> None:
    supplier = make_supplier(db_session)
    response = upload(client, PDF, "BOGAO Spring 2026.pdf", supplier_id=str(supplier.id))
    assert response.status_code == 201, response.text
    body = response.json()

    assert body["name"] == "BOGAO Spring 2026"  # derived from the file name
    assert body["original_filename"] == "BOGAO Spring 2026.pdf"
    assert body["supplier"]["id"] == str(supplier.id)
    assert body["status"] == "UPLOADED"
    assert body["file_size_bytes"] == len(PDF)
    assert body["sha256"] == hashlib.sha256(PDF).hexdigest()
    assert body["job"]["status"] == "QUEUED" and body["job"]["job_type"] == "CATALOGUE_PROCESS"

    catalogue_id = uuid.UUID(body["id"])
    assert storage.read_bytes(original_pdf_key(catalogue_id)) == PDF
    assert enqueued == [(catalogue_id, uuid.UUID(body["job"]["id"]))]
    job = db_session.get(ProcessingJob, uuid.UUID(body["job"]["id"]))
    assert job.celery_task_id == "test-task-1" and job.created_by_id == admin.id

    audit = db_session.scalar(select(AuditLog).where(AuditLog.action == AuditAction.UPLOAD_CATALOGUE))
    assert audit.actor_id == admin.id and audit.entity_id == body["id"]
    assert audit.details["sha256"] == body["sha256"]


def test_a_custom_name_overrides_the_file_name(client, admin) -> None:  # noqa: ANN001
    response = upload(client, PDF, "x.pdf", name="  Autumn range  ")
    assert response.json()["name"] == "Autumn range"


def test_hostile_file_names_never_reach_storage_paths(client, admin, storage) -> None:  # noqa: ANN001
    response = upload(client, PDF, "../../../etc/passwd.pdf")
    assert response.status_code == 201
    assert response.json()["original_filename"] == "passwd.pdf"
    (stored,) = [p for p in storage.root.rglob("*") if p.is_file()]
    assert stored.name == "original.pdf"  # keys are generated, never derived from user input
    assert storage.root in stored.resolve().parents


def test_unknown_suppliers_are_rejected_and_nothing_is_stored(client, admin, storage) -> None:  # noqa: ANN001
    response = upload(client, PDF, supplier_id=str(uuid.uuid4()))
    assert response.status_code == 404
    assert stored_files(storage) == []


@pytest.mark.parametrize(
    "data", [b"", b"just some text", b"PK\x03\x04 pretending to be a pdf"], ids=["empty", "text", "zip"]
)
def test_files_that_are_not_pdfs_are_rejected_and_nothing_is_stored(  # noqa: ANN001
    client, admin, storage, data: bytes
) -> None:
    assert upload(client, data).status_code in (415, 422)  # (an empty part may be "missing")
    assert stored_files(storage) == []


def test_oversized_uploads_are_rejected_before_and_while_streaming(  # noqa: ANN001
    client, app, admin, storage
) -> None:
    tiny = get_settings().model_copy(update={"max_upload_mb": 1})
    app.dependency_overrides[get_settings] = lambda: tiny

    too_big_declared = b"%PDF-1.4\n" + b"0" * (3 * 1024 * 1024)  # Content-Length alone gives it away
    assert upload(client, too_big_declared).status_code == 413

    over_the_stream_cap = b"%PDF-1.4\n" + b"0" * int(1.5 * 1024 * 1024)  # passes the header check
    assert upload(client, over_the_stream_cap).status_code == 413
    assert stored_files(storage) == []  # neither attempt left a file behind


def test_uploading_the_same_file_twice_is_flagged_not_duplicated(client, admin, storage, enqueued) -> None:  # noqa: ANN001
    first = upload(client, PDF, "one.pdf").json()
    again = upload(client, PDF, "one-again.pdf")
    assert again.status_code == 409
    assert again.json()["detail"]["existing_catalogue_id"] == first["id"]
    assert len(stored_files(storage)) == 1 and len(enqueued) == 1  # nothing extra stored or queued


def test_a_duplicate_can_be_uploaded_on_purpose(client, admin, storage) -> None:  # noqa: ANN001
    upload(client, PDF, "one.pdf")
    forced = upload(client, PDF, "two.pdf", allow_duplicate="true")
    assert forced.status_code == 201
    assert len(stored_files(storage)) == 2


def test_uploads_from_untrusted_origins_are_refused(client, admin, storage) -> None:  # noqa: ANN001
    response = client.post(
        "/api/admin/catalogues",
        files={"file": ("x.pdf", PDF, "application/pdf")},
        headers={"Origin": "https://evil.example"},
    )
    assert response.status_code == 403
    assert stored_files(storage) == []


def test_if_the_queue_is_unavailable_the_upload_is_kept_and_visibly_failed(  # noqa: ANN001
    client, admin, monkeypatch, storage
) -> None:
    def broken(catalogue_id, job_id):  # noqa: ANN001, ANN202
        raise ConnectionError("redis is down")

    monkeypatch.setattr("app.services.catalogues.enqueue_processing", broken)
    response = upload(client, PDF)
    assert response.status_code == 201  # the file is safe; the admin is told what to do
    body = response.json()
    assert body["status"] == "FAILED" and "worker" in body["processing_error"]
    assert body["job"]["status"] == "FAILED"
    assert len(stored_files(storage)) == 1

    # ... and once the queue is back, reprocessing recovers it.
    monkeypatch.setattr("app.services.catalogues.enqueue_processing", lambda c, j: "task-ok")
    assert client.post(f"/api/admin/catalogues/{body['id']}/reprocess").status_code == 202


# ------------------------------------------------------------------------------ browse
def test_detail_and_pages_after_processing(client, admin, db_session, storage) -> None:  # noqa: ANN001
    body = upload(client, PDF).json()
    process(db_session, storage, body)

    detail = client.get(f"/api/admin/catalogues/{body['id']}").json()
    assert detail["status"] == "COMPLETED" and detail["page_count"] == 2
    assert detail["processing_progress"] == 100
    assert detail["page_status_counts"] == {"RENDERED": 2}
    assert detail["embedded_image_count"] == 1
    assert detail["job"]["status"] == "SUCCEEDED" and detail["job"]["progress"] == 100

    pages = client.get(f"/api/admin/catalogues/{body['id']}/pages").json()
    assert pages["total"] == 2
    first, second = pages["items"]
    assert (first["page_number"], first["status"], first["embedded_image_count"]) == (1, "RENDERED", 1)
    assert second["embedded_image_count"] == 0
    assert first["thumbnail_url"].endswith("/pages/1/image?variant=thumbnail")

    paged = client.get(f"/api/admin/catalogues/{body['id']}/pages", params={"limit": 1, "offset": 1})
    assert [p["page_number"] for p in paged.json()["items"]] == [2]


def test_listing_filters_and_paginates(client, admin, db_session, storage) -> None:  # noqa: ANN001
    acme, other = make_supplier(db_session), make_supplier(db_session)
    make_stored_catalogue(db_session, storage, PDF, name="Acme Spring", supplier=acme)
    make_stored_catalogue(
        db_session, storage, PDF, name="Acme Autumn", supplier=acme, status=CatalogueStatus.COMPLETED
    )
    make_stored_catalogue(db_session, storage, PDF, name="100% Wood_Works", supplier=other)

    everything = client.get("/api/admin/catalogues").json()
    assert everything["total"] == 3 and len(everything["items"]) == 3

    by_supplier = client.get("/api/admin/catalogues", params={"supplier_id": str(acme.id)}).json()
    assert by_supplier["total"] == 2
    assert client.get("/api/admin/catalogues", params={"status": "COMPLETED"}).json()["total"] == 1
    assert client.get("/api/admin/catalogues", params={"q": "autumn"}).json()["total"] == 1
    # LIKE wildcards in the search text are matched literally, not interpreted.
    assert client.get("/api/admin/catalogues", params={"q": "100%"}).json()["total"] == 1
    assert client.get("/api/admin/catalogues", params={"q": "%"}).json()["total"] == 1
    assert client.get("/api/admin/catalogues", params={"q": "_"}).json()["total"] == 1

    page = client.get("/api/admin/catalogues", params={"limit": 2, "offset": 2}).json()
    assert len(page["items"]) == 1 and page["total"] == 3


def test_invalid_query_parameters_and_ids_are_rejected(client, admin) -> None:  # noqa: ANN001
    assert client.get("/api/admin/catalogues", params={"limit": 0}).status_code == 422
    assert client.get("/api/admin/catalogues", params={"limit": 101}).status_code == 422
    assert client.get("/api/admin/catalogues", params={"status": "NOPE"}).status_code == 422
    assert client.get("/api/admin/catalogues/not-a-uuid").status_code == 422
    assert client.get(f"/api/admin/catalogues/{uuid.uuid4()}").status_code == 404


# ------------------------------------------------------------------------------ images
def test_page_images_are_served_privately_as_jpeg(client, admin, db_session, storage) -> None:  # noqa: ANN001
    body = upload(client, PDF).json()
    process(db_session, storage, body)
    base = f"/api/admin/catalogues/{body['id']}/pages/1/image"

    full = client.get(base)
    assert full.status_code == 200 and full.headers["content-type"] == "image/jpeg"
    assert full.content[:3] == b"\xff\xd8\xff"
    assert full.headers["cache-control"].startswith("private")  # never cached by shared proxies
    assert full.headers["x-content-type-options"] == "nosniff"
    assert int(full.headers["content-length"]) == len(full.content)

    thumb = client.get(base, params={"variant": "thumbnail"})
    assert thumb.status_code == 200 and len(thumb.content) < len(full.content)

    assert client.get(base, params={"variant": "huge"}).status_code == 422
    assert client.get(f"/api/admin/catalogues/{body['id']}/pages/99/image").status_code == 404
    assert client.get(f"/api/admin/catalogues/{uuid.uuid4()}/pages/1/image").status_code == 404


def test_source_images_are_listed_and_downloadable(client, admin, db_session, storage) -> None:  # noqa: ANN001
    body = upload(client, PDF).json()
    process(db_session, storage, body)

    images = client.get(f"/api/admin/catalogues/{body['id']}/pages/1/images").json()
    assert sorted(i["kind"] for i in images) == ["EMBEDDED", "PAGE"]
    embedded = next(i for i in images if i["kind"] == "EMBEDDED")
    assert (embedded["width_px"], embedded["height_px"]) == (150, 150)
    assert embedded["bbox"] is not None and len(embedded["bbox"]) == 4
    assert next(i for i in images if i["kind"] == "PAGE")["bbox"] is None

    download = client.get(embedded["url"])
    assert download.status_code == 200 and download.content[:3] == b"\xff\xd8\xff"


def test_an_image_cannot_be_fetched_through_the_wrong_catalogue(client, admin, db_session, storage) -> None:  # noqa: ANN001
    one = upload(client, PDF, "one.pdf").json()
    two = upload(client, blank_pages(1), "two.pdf").json()
    process(db_session, storage, one)
    process(db_session, storage, two)

    images = client.get(f"/api/admin/catalogues/{one['id']}/pages/1/images").json()
    image_id = images[0]["id"]
    assert client.get(f"/api/admin/catalogues/{one['id']}/source-images/{image_id}/image").status_code == 200
    assert client.get(f"/api/admin/catalogues/{two['id']}/source-images/{image_id}/image").status_code == 404


# ------------------------------------------------------------------------------ reprocess
def test_reprocess_queues_a_new_job_but_not_while_one_is_active(client, admin, db_session, storage, enqueued) -> None:  # noqa: ANN001
    body = upload(client, PDF).json()
    busy = client.post(f"/api/admin/catalogues/{body['id']}/reprocess")
    assert busy.status_code == 409  # the upload's own job is still queued

    process(db_session, storage, body)  # the worker finishes it
    again = client.post(f"/api/admin/catalogues/{body['id']}/reprocess")
    assert again.status_code == 202
    assert again.json()["status"] == "QUEUED" and len(enqueued) == 2
    assert client.get(f"/api/admin/catalogues/{body['id']}").json()["status"] == "UPLOADED"
    assert db_session.scalar(select(AuditLog).where(AuditLog.action == AuditAction.REPROCESS_CATALOGUE))


def test_a_stuck_job_can_be_overridden_with_force(client, admin, enqueued) -> None:  # noqa: ANN001
    body = upload(client, PDF).json()
    forced = client.post(f"/api/admin/catalogues/{body['id']}/reprocess", params={"force": "true"})
    assert forced.status_code == 202 and len(enqueued) == 2


# ------------------------------------------------------------------------------ delete
def test_deleting_removes_data_and_files_but_keeps_published_products(  # noqa: ANN001
    client, admin, db_session, storage
) -> None:
    body = upload(client, PDF).json()
    process(db_session, storage, body)
    product = make_product(db_session, catalogue=db_session.get(Catalogue, uuid.UUID(body["id"])))

    response = client.delete(f"/api/admin/catalogues/{body['id']}")
    assert response.status_code == 204 and response.content == b""

    assert db_session.get(Catalogue, uuid.UUID(body["id"])) is None
    assert stored_files(storage) == []  # original, page renders and extracted images are gone
    db_session.refresh(product)
    assert product.catalogue_id is None  # the product itself survives
    assert db_session.get(Product, product.id) is not None

    audit = db_session.scalar(select(AuditLog).where(AuditLog.action == AuditAction.DELETE_CATALOGUE))
    assert audit.entity_id == body["id"] and audit.details["filename"] == "catalogue.pdf"
    assert client.get(f"/api/admin/catalogues/{body['id']}").status_code == 404


def test_delete_is_refused_while_processing_is_active_unless_forced(client, admin, db_session, storage) -> None:  # noqa: ANN001
    body = upload(client, PDF).json()  # its job is QUEUED
    assert client.delete(f"/api/admin/catalogues/{body['id']}").status_code == 409
    assert db_session.get(Catalogue, uuid.UUID(body["id"])) is not None
    assert client.delete(f"/api/admin/catalogues/{body['id']}", params={"force": "true"}).status_code == 204
    assert stored_files(storage) == []


def test_deleting_a_catalogue_never_touches_another_ones_files(client, admin, db_session, storage) -> None:  # noqa: ANN001
    one = upload(client, PDF, "one.pdf").json()
    two = upload(client, blank_pages(1), "two.pdf").json()
    for body in (one, two):
        process(db_session, storage, body)
    client.delete(f"/api/admin/catalogues/{one['id']}")
    assert storage.exists(original_pdf_key(uuid.UUID(two["id"])))
    assert client.get(f"/api/admin/catalogues/{two['id']}/pages/1/image").status_code == 200


def test_job_statuses_are_visible_on_the_detail_endpoint(client, admin, db_session) -> None:  # noqa: ANN001
    body = upload(client, PDF).json()
    job = db_session.get(ProcessingJob, uuid.UUID(body["job"]["id"]))
    job.status = JobStatus.RUNNING
    job.progress = 40
    db_session.flush()
    detail = client.get(f"/api/admin/catalogues/{body['id']}").json()
    assert detail["job"]["status"] == "RUNNING" and detail["job"]["progress"] == 40

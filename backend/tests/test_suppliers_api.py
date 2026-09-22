from __future__ import annotations

import pytest
from sqlalchemy import select

from app.models import AuditLog
from app.models.enums import AuditAction, UserRole
from tests.factories import TEST_PASSWORD, make_user
from tests.helpers import sign_in


@pytest.fixture
def admin(client, db_session):  # noqa: ANN001, ANN201
    user = make_user(db_session, password=TEST_PASSWORD, role=UserRole.ADMIN)
    sign_in(client, user)
    return user


def test_create_and_list_suppliers(client, admin, db_session) -> None:  # noqa: ANN001
    created = client.post(
        "/api/admin/suppliers",
        json={"name": "  BOGAO Furniture  ", "website": "https://bogao.example"},
    )
    assert created.status_code == 201
    body = created.json()
    assert body["name"] == "BOGAO Furniture" and body["slug"] == "bogao-furniture"
    assert body["is_active"] is True

    client.post("/api/admin/suppliers", json={"name": "Alpha Lighting"})
    names = [s["name"] for s in client.get("/api/admin/suppliers").json()]
    assert names == ["Alpha Lighting", "BOGAO Furniture"]  # alphabetical

    audit = db_session.scalar(select(AuditLog).where(AuditLog.action == AuditAction.MANAGE_SUPPLIER))
    assert audit is not None and audit.actor_id == admin.id


def test_names_that_slugify_the_same_are_a_conflict(client, admin) -> None:  # noqa: ANN001
    assert client.post("/api/admin/suppliers", json={"name": "Acme & Sons"}).status_code == 201
    assert client.post("/api/admin/suppliers", json={"name": "Acme & Sons"}).status_code == 409
    assert client.post("/api/admin/suppliers", json={"name": "Acme  Sons"}).status_code == 409


def test_non_latin_names_get_a_fallback_slug(client, admin) -> None:  # noqa: ANN001
    body = client.post("/api/admin/suppliers", json={"name": "博高家具"}).json()
    assert body["slug"].startswith("supplier-")
    assert body["name"] == "博高家具"


@pytest.mark.parametrize("payload", [{}, {"name": ""}, {"name": "x" * 256}])
def test_invalid_suppliers_are_rejected(client, admin, payload: dict) -> None:  # noqa: ANN001
    assert client.post("/api/admin/suppliers", json=payload).status_code == 422

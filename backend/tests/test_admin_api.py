from __future__ import annotations

import pytest

from app.models.enums import AuditAction, UserRole
from app.services.audit import record_audit
from tests.factories import TEST_PASSWORD, make_user


def sign_in(client, user) -> None:  # noqa: ANN001
    response = client.post(
        "/api/auth/login", json={"email": user.email, "password": TEST_PASSWORD}
    )
    assert response.status_code == 200


@pytest.fixture
def admin(db_session):  # noqa: ANN001, ANN201
    return make_user(db_session, password=TEST_PASSWORD, role=UserRole.ADMIN)


def test_anonymous_users_are_unauthenticated(client) -> None:  # noqa: ANN001
    assert client.get("/api/admin/audit-logs").status_code == 401


def test_customers_are_forbidden(client, db_session) -> None:  # noqa: ANN001
    customer = make_user(db_session, password=TEST_PASSWORD, role=UserRole.CUSTOMER)
    sign_in(client, customer)
    assert client.get("/api/admin/audit-logs").status_code == 403


def test_admins_can_read_the_audit_log(client, admin) -> None:  # noqa: ANN001
    sign_in(client, admin)
    response = client.get("/api/admin/audit-logs")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] >= 1
    assert body["items"][0]["action"] == "LOGIN_SUCCESS"  # newest first
    assert body["items"][0]["actor_email"] == admin.email
    assert {"limit", "offset"} <= set(body)


def test_audit_log_can_be_filtered_and_paginated(client, admin, db_session) -> None:  # noqa: ANN001
    for index in range(5):
        record_audit(
            db_session,
            AuditAction.APPROVED_PRODUCT,
            actor=admin,
            entity_type="product",
            entity_id=f"product-{index}",
        )
    db_session.flush()
    sign_in(client, admin)

    approved = client.get("/api/admin/audit-logs", params={"action": "APPROVED_PRODUCT"}).json()
    assert approved["total"] == 5

    page = client.get(
        "/api/admin/audit-logs", params={"action": "APPROVED_PRODUCT", "limit": 2, "offset": 4}
    ).json()
    assert len(page["items"]) == 1 and page["total"] == 5

    one = client.get(
        "/api/admin/audit-logs", params={"entity_type": "product", "entity_id": "product-3"}
    ).json()
    assert [item["entity_id"] for item in one["items"]] == ["product-3"]


@pytest.mark.parametrize(
    "params",
    [{"limit": 0}, {"limit": 201}, {"offset": -1}, {"action": "NOT_AN_ACTION"}],
)
def test_invalid_query_parameters_are_rejected(client, admin, params) -> None:  # noqa: ANN001
    sign_in(client, admin)
    assert client.get("/api/admin/audit-logs", params=params).status_code == 422


def test_secrets_in_audit_details_are_masked(db_session) -> None:  # noqa: ANN001
    entry = record_audit(
        db_session,
        AuditAction.EDIT_PRODUCT,
        details={"field": "name", "api_key": "sk-live-123", "nested": {"password": "hunter2"}},
    )
    assert entry.details == {
        "field": "name",
        "api_key": "[REDACTED]",
        "nested": {"password": "[REDACTED]"},
    }

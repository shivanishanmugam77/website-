from __future__ import annotations

import logging

from app.core import health
from app.core.health import CheckResult


def test_live_returns_ok(client) -> None:  # noqa: ANN001
    response = client.get("/api/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_ok_against_real_dependencies(client) -> None:  # noqa: ANN001
    """Needs the compose Postgres (migrated) and Redis; see README 'Running the tests'."""
    response = client.get("/api/health/ready")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "ok"
    assert body["checks"]["database"]["ok"] is True
    assert body["checks"]["redis"]["ok"] is True


def test_ready_returns_503_when_a_dependency_fails(client, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(health, "check_redis", lambda: CheckResult("redis", False, "unreachable"))
    response = client.get("/api/health/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["checks"]["redis"] == {"ok": False, "detail": "unreachable"}
    assert body["checks"]["database"]["ok"] is True


def test_database_check_detects_pending_migrations(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(health, "head_revision", lambda: "some-future-revision")
    result = health.check_database()
    assert result.ok is False
    assert result.detail == "migrations pending"


def test_response_carries_request_id(client) -> None:  # noqa: ANN001
    response = client.get("/api/health/live")
    assert len(response.headers["X-Request-ID"]) >= 8


def test_valid_incoming_request_id_is_echoed(client) -> None:  # noqa: ANN001
    response = client.get("/api/health/live", headers={"X-Request-ID": "trace-abc-12345"})
    assert response.headers["X-Request-ID"] == "trace-abc-12345"


def test_malicious_request_id_is_replaced(client) -> None:  # noqa: ANN001
    response = client.get("/api/health/live", headers={"X-Request-ID": "bad id\twith spaces"})
    assert response.headers["X-Request-ID"] != "bad id\twith spaces"
    assert " " not in response.headers["X-Request-ID"]


def test_unhandled_errors_do_not_leak_details(app, client, caplog) -> None:  # noqa: ANN001
    @app.get("/api/_boom")
    def boom() -> None:
        raise RuntimeError("database password is hunter2")

    with caplog.at_level(logging.ERROR):
        response = client.get("/api/_boom")
    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error"
    assert "hunter2" not in response.text
    assert response.json()["request_id"]


def test_openapi_docs_available_outside_production(client) -> None:  # noqa: ANN001
    assert client.get("/api/openapi.json").status_code == 200


def test_cors_allows_configured_origin_only(client) -> None:  # noqa: ANN001
    allowed = client.get("/api/health/live", headers={"Origin": "http://localhost:3000"})
    assert allowed.headers.get("access-control-allow-origin") == "http://localhost:3000"
    denied = client.get("/api/health/live", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in denied.headers

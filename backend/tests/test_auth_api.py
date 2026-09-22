"""End-to-end authentication behaviour through the real FastAPI app and PostgreSQL."""

from __future__ import annotations

import json
import re
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.tokens import create_access_token, hash_refresh_token
from app.models import AuditLog, RefreshToken, User
from app.models.enums import AuditAction, UserRole
from tests.factories import TEST_PASSWORD, make_user

WRONG_PASSWORD = "definitely-not-the-password"
NEW_PASSWORD = "a-brand-new-passphrase-2026"
GENERIC_LOGIN_ERROR = {"detail": "Invalid email or password"}


def login(client, email: str, password: str = TEST_PASSWORD):  # noqa: ANN001, ANN201
    return client.post("/api/auth/login", json={"email": email, "password": password})


def cookie_header(**cookies: str) -> dict[str, str]:
    """Send exactly these cookies, bypassing the client's cookie jar (to replay old ones)."""
    return {"Cookie": "; ".join(f"{k}={v}" for k, v in cookies.items())}


def _family_of(session: Session, user: User) -> uuid.UUID:
    return session.scalar(select(RefreshToken.family_id).where(RefreshToken.user_id == user.id))


def audit_actions(session: Session, action: AuditAction) -> list[AuditLog]:
    return list(session.scalars(select(AuditLog).where(AuditLog.action == action)))


# ------------------------------------------------------------------------- registration
def test_register_creates_a_customer_and_signs_in(client) -> None:  # noqa: ANN001
    response = client.post(
        "/api/auth/register",
        json={
            "email": "New.User@Example.com",
            "password": TEST_PASSWORD,
            "full_name": "New User",
            "role": "ADMIN",  # must be ignored: nobody can self-assign a role
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert body["email"] == "new.user@example.com"
    assert body["role"] == "CUSTOMER"
    # Exactly the public fields: no password hash, no lockout state.
    assert set(body) == {
        "id",
        "email",
        "full_name",
        "role",
        "is_active",
        "created_at",
        "last_login_at",
    }
    assert client.get("/api/auth/me").json()["email"] == "new.user@example.com"


def test_register_rejects_duplicate_email_case_insensitively(client) -> None:  # noqa: ANN001
    payload = {"email": "dup@example.com", "password": TEST_PASSWORD}
    assert client.post("/api/auth/register", json=payload).status_code == 201
    again = client.post("/api/auth/register", json={**payload, "email": "DUP@Example.com"})
    assert again.status_code == 409


def test_register_rejects_weak_passwords_and_bad_emails(client) -> None:  # noqa: ANN001
    weak = client.post(
        "/api/auth/register", json={"email": "weak@example.com", "password": "short"}
    )
    assert weak.status_code == 422
    assert any("password" in item["loc"] for item in weak.json()["detail"])

    bad_email = client.post(
        "/api/auth/register", json={"email": "not-an-email", "password": TEST_PASSWORD}
    )
    assert bad_email.status_code == 422


def test_register_never_stores_the_password_in_plain_text(client, db_session) -> None:  # noqa: ANN001
    client.post("/api/auth/register", json={"email": "hash@example.com", "password": TEST_PASSWORD})
    user = db_session.scalar(select(User).where(User.email == "hash@example.com"))
    assert TEST_PASSWORD not in user.password_hash
    assert user.password_hash.startswith("$argon2id$")


# ------------------------------------------------------------------------- login
def test_login_sets_httponly_scoped_cookies(client, db_session) -> None:  # noqa: ANN001
    user = make_user(db_session, password=TEST_PASSWORD)
    response = login(client, user.email)
    assert response.status_code == 200
    assert response.json()["email"] == user.email
    assert response.headers["cache-control"] == "no-store"

    cookies = {c.split("=", 1)[0]: c.lower() for c in response.headers.get_list("set-cookie")}
    assert set(cookies) == {"access_token", "refresh_token"}
    for header in cookies.values():
        assert "httponly" in header
        assert "samesite=lax" in header
        assert "; secure" not in header  # local development over http
    assert re.search(r"path=/api(;|$)", cookies["access_token"])
    assert "path=/api/auth" in cookies["refresh_token"]  # refresh token only travels to /auth


def test_login_is_case_insensitive_on_email(client, db_session) -> None:  # noqa: ANN001
    user = make_user(db_session, password=TEST_PASSWORD)
    assert login(client, user.email.upper()).status_code == 200


def test_wrong_password_and_unknown_email_are_indistinguishable(client, db_session) -> None:  # noqa: ANN001
    user = make_user(db_session, password=TEST_PASSWORD)
    wrong = login(client, user.email, WRONG_PASSWORD)
    unknown = login(client, "nobody@example.com", WRONG_PASSWORD)
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json() == GENERIC_LOGIN_ERROR
    assert "set-cookie" not in wrong.headers and "set-cookie" not in unknown.headers


def test_login_rejects_oversized_passwords_before_hashing(client, db_session) -> None:  # noqa: ANN001
    user = make_user(db_session, password=TEST_PASSWORD)
    assert login(client, user.email, "x" * 5000).status_code == 422


def test_account_locks_after_repeated_failures_even_for_the_right_password(  # noqa: ANN001
    client, db_session
) -> None:
    user = make_user(db_session, password=TEST_PASSWORD)
    for _ in range(get_settings().login_max_failed_attempts):
        assert login(client, user.email, WRONG_PASSWORD).status_code == 401

    locked = login(client, user.email)  # correct password, but locked
    assert locked.status_code == 401
    assert locked.json() == GENERIC_LOGIN_ERROR  # a lock must not reveal the account exists

    db_session.refresh(user)
    assert user.locked_until is not None and user.locked_until > datetime.now(UTC)

    # Once the lockout has been served, login works and the counters reset.
    user.locked_until = datetime.now(UTC) - timedelta(seconds=1)
    db_session.flush()
    assert login(client, user.email).status_code == 200
    db_session.refresh(user)
    assert user.failed_login_attempts == 0 and user.locked_until is None


def test_a_successful_login_resets_the_failure_counter(client, db_session) -> None:  # noqa: ANN001
    user = make_user(db_session, password=TEST_PASSWORD)
    for _ in range(3):
        login(client, user.email, WRONG_PASSWORD)
    assert login(client, user.email).status_code == 200
    db_session.refresh(user)
    assert user.failed_login_attempts == 0


def test_disabled_accounts_cannot_log_in_and_lose_existing_sessions(  # noqa: ANN001
    client, db_session
) -> None:
    user = make_user(db_session, password=TEST_PASSWORD)
    assert login(client, user.email).status_code == 200
    assert client.get("/api/auth/me").status_code == 200

    user.is_active = False
    db_session.flush()
    assert client.get("/api/auth/me").status_code == 401  # immediate, no waiting for expiry
    client.cookies.clear()
    assert login(client, user.email).status_code == 401


def test_me_requires_authentication(client) -> None:  # noqa: ANN001
    response = client.get("/api/auth/me")
    assert response.status_code == 401
    assert response.json() == {"detail": "Not authenticated"}


def test_a_demoted_admin_loses_admin_access_immediately(client, db_session) -> None:  # noqa: ANN001
    admin = make_user(db_session, password=TEST_PASSWORD, role=UserRole.ADMIN)
    login(client, admin.email)
    assert client.get("/api/admin/audit-logs").status_code == 200
    admin.role = UserRole.CUSTOMER  # role is read from the database, not from the token
    db_session.flush()
    assert client.get("/api/admin/audit-logs").status_code == 403


# ------------------------------------------------------------------------- forged tokens
def _forged_cookie(**overrides) -> dict[str, str]:  # noqa: ANN003
    values = {
        "user_id": uuid.uuid4(),
        "session_id": uuid.uuid4(),
        "secret": get_settings().secret_key.get_secret_value(),
        "ttl": timedelta(minutes=15),
    }
    values.update(overrides)
    user_id, session_id = values.pop("user_id"), values.pop("session_id")
    return cookie_header(access_token=create_access_token(user_id, session_id, **values))


def test_expired_access_tokens_are_rejected(client, db_session) -> None:  # noqa: ANN001
    user = make_user(db_session, password=TEST_PASSWORD)
    login(client, user.email)
    family = _family_of(db_session, user)
    headers = _forged_cookie(
        user_id=user.id, session_id=family, now=datetime.now(UTC) - timedelta(hours=1)
    )
    assert client.get("/api/auth/me", headers=headers).status_code == 401


def test_tokens_signed_with_another_secret_are_rejected(client, db_session) -> None:  # noqa: ANN001
    user = make_user(db_session, password=TEST_PASSWORD)
    login(client, user.email)
    family = _family_of(db_session, user)
    headers = _forged_cookie(user_id=user.id, session_id=family, secret="attacker-key-" + "z" * 40)
    assert client.get("/api/auth/me", headers=headers).status_code == 401


def test_a_valid_signature_is_not_enough_without_a_live_session(client, db_session) -> None:  # noqa: ANN001
    user = make_user(db_session, password=TEST_PASSWORD)
    headers = _forged_cookie(user_id=user.id)  # correctly signed, but no such session exists
    assert client.get("/api/auth/me", headers=headers).status_code == 401


def test_tokens_for_deleted_users_are_rejected(client) -> None:  # noqa: ANN001
    assert client.get("/api/auth/me", headers=_forged_cookie()).status_code == 401


# ------------------------------------------------------------------------- refresh
def test_refresh_rotates_the_token_and_keeps_the_session_alive(client, db_session) -> None:  # noqa: ANN001
    user = make_user(db_session, password=TEST_PASSWORD)
    login(client, user.email)
    old_refresh = client.cookies.get("refresh_token")

    assert client.post("/api/auth/refresh").status_code == 200
    assert client.cookies.get("refresh_token") != old_refresh
    assert client.get("/api/auth/me").status_code == 200

    # Replaying the old token immediately is a benign race (two tabs): rejected, but the
    # live session is not destroyed.
    replay = client.post("/api/auth/refresh", headers=cookie_header(refresh_token=old_refresh))
    assert replay.status_code == 401
    assert client.get("/api/auth/me").status_code == 200
    assert client.post("/api/auth/refresh").status_code == 200


def test_replaying_an_old_refresh_token_later_revokes_the_whole_session(  # noqa: ANN001
    client, db_session
) -> None:
    user = make_user(db_session, password=TEST_PASSWORD)
    login(client, user.email)
    old_refresh = client.cookies.get("refresh_token")
    client.post("/api/auth/refresh")
    assert client.get("/api/auth/me").status_code == 200

    # Pretend the rotation happened five minutes ago (outside the grace window).
    db_session.execute(
        update(RefreshToken)
        .where(RefreshToken.token_hash == hash_refresh_token(old_refresh))
        .values(revoked_at=datetime.now(UTC) - timedelta(minutes=5))
    )
    db_session.flush()

    replay = client.post("/api/auth/refresh", headers=cookie_header(refresh_token=old_refresh))
    assert replay.status_code == 401
    # Theft is assumed: the legitimate tokens are dead too.
    assert client.get("/api/auth/me").status_code == 401
    assert client.post("/api/auth/refresh").status_code == 401
    assert len(audit_actions(db_session, AuditAction.TOKEN_REUSE_DETECTED)) == 1


def test_refresh_without_a_cookie_or_with_garbage_is_rejected(client) -> None:  # noqa: ANN001
    assert client.post("/api/auth/refresh").status_code == 401
    garbage = client.post("/api/auth/refresh", headers=cookie_header(refresh_token="nonsense"))
    assert garbage.status_code == 401


def test_expired_refresh_tokens_are_rejected(client, db_session) -> None:  # noqa: ANN001
    user = make_user(db_session, password=TEST_PASSWORD)
    login(client, user.email)
    db_session.execute(
        update(RefreshToken)
        .where(RefreshToken.user_id == user.id)
        .values(expires_at=datetime.now(UTC) - timedelta(seconds=1))
    )
    db_session.flush()
    assert client.post("/api/auth/refresh").status_code == 401


# ------------------------------------------------------------------------- logout
def test_logout_revokes_the_session_server_side(client, db_session) -> None:  # noqa: ANN001
    user = make_user(db_session, password=TEST_PASSWORD)
    login(client, user.email)
    access, refresh = client.cookies.get("access_token"), client.cookies.get("refresh_token")

    assert client.post("/api/auth/logout").status_code == 204
    assert client.get("/api/auth/me").status_code == 401

    # A stolen copy of the old cookies is useless too (not merely deleted client-side).
    stolen = cookie_header(access_token=access, refresh_token=refresh)
    assert client.get("/api/auth/me", headers=stolen).status_code == 401
    assert client.post("/api/auth/refresh", headers=stolen).status_code == 401
    assert len(audit_actions(db_session, AuditAction.LOGOUT)) == 1


def test_logout_when_not_signed_in_is_not_an_error(client) -> None:  # noqa: ANN001
    assert client.post("/api/auth/logout").status_code == 204


# ------------------------------------------------------------------------- change password
def test_change_password_ends_other_sessions_but_keeps_this_one(client, db_session) -> None:  # noqa: ANN001
    user = make_user(db_session, password=TEST_PASSWORD)
    login(client, user.email)
    other_device_refresh = client.cookies.get("refresh_token")

    response = client.post(
        "/api/auth/change-password",
        json={"current_password": TEST_PASSWORD, "new_password": NEW_PASSWORD},
    )
    assert response.status_code == 200
    assert client.get("/api/auth/me").status_code == 200  # this device stays signed in
    stale = client.post(
        "/api/auth/refresh", headers=cookie_header(refresh_token=other_device_refresh)
    )
    assert stale.status_code == 401  # every previous session is gone

    client.cookies.clear()
    assert login(client, user.email, TEST_PASSWORD).status_code == 401
    assert login(client, user.email, NEW_PASSWORD).status_code == 200
    assert len(audit_actions(db_session, AuditAction.PASSWORD_CHANGED)) == 1


def test_change_password_requires_the_current_password(client, db_session) -> None:  # noqa: ANN001
    user = make_user(db_session, password=TEST_PASSWORD)
    login(client, user.email)
    response = client.post(
        "/api/auth/change-password",
        json={"current_password": WRONG_PASSWORD, "new_password": NEW_PASSWORD},
    )
    assert response.status_code == 401
    assert len(audit_actions(db_session, AuditAction.PASSWORD_CHANGE_FAILED)) == 1
    client.cookies.clear()
    assert login(client, user.email, TEST_PASSWORD).status_code == 200  # unchanged


def test_change_password_enforces_the_strength_policy(client, db_session) -> None:  # noqa: ANN001
    user = make_user(db_session, password=TEST_PASSWORD)
    login(client, user.email)
    response = client.post(
        "/api/auth/change-password",
        json={"current_password": TEST_PASSWORD, "new_password": "short"},
    )
    assert response.status_code == 422
    client.cookies.clear()
    assert login(client, user.email, TEST_PASSWORD).status_code == 200  # unchanged


def test_change_password_requires_authentication(client) -> None:  # noqa: ANN001
    response = client.post(
        "/api/auth/change-password",
        json={"current_password": TEST_PASSWORD, "new_password": NEW_PASSWORD},
    )
    assert response.status_code == 401


# ------------------------------------------------------------------------- CSRF / origin
def test_state_changing_requests_from_untrusted_origins_are_refused(client, db_session) -> None:  # noqa: ANN001
    user = make_user(db_session, password=TEST_PASSWORD)
    payload = {"email": user.email, "password": TEST_PASSWORD}
    evil = client.post("/api/auth/login", json=payload, headers={"Origin": "https://evil.example"})
    assert evil.status_code == 403
    assert evil.json() == {"detail": "Untrusted origin"}


def test_state_changing_requests_without_an_origin_are_refused(client, db_session) -> None:  # noqa: ANN001
    user = make_user(db_session, password=TEST_PASSWORD)
    response = client.post(
        "/api/auth/login",
        json={"email": user.email, "password": TEST_PASSWORD},
        headers={"Origin": ""},
    )
    assert response.status_code == 403


def test_referer_is_accepted_when_origin_is_absent(client, db_session) -> None:  # noqa: ANN001
    user = make_user(db_session, password=TEST_PASSWORD)
    response = client.post(
        "/api/auth/login",
        json={"email": user.email, "password": TEST_PASSWORD},
        headers={"Origin": "", "Referer": "http://localhost:3000/login"},
    )
    assert response.status_code == 200


def test_the_api_docs_origin_is_trusted_in_development(client, db_session) -> None:  # noqa: ANN001
    user = make_user(db_session, password=TEST_PASSWORD)
    response = client.post(
        "/api/auth/login",
        json={"email": user.email, "password": TEST_PASSWORD},
        headers={"Origin": "http://localhost:8000"},
    )
    assert response.status_code == 200


def test_safe_methods_do_not_need_a_trusted_origin(client) -> None:  # noqa: ANN001
    response = client.get("/api/health/live", headers={"Origin": "https://evil.example"})
    assert response.status_code == 200


def test_logout_is_also_protected_against_cross_site_requests(client, db_session) -> None:  # noqa: ANN001
    user = make_user(db_session, password=TEST_PASSWORD)
    login(client, user.email)
    response = client.post("/api/auth/logout", headers={"Origin": "https://evil.example"})
    assert response.status_code == 403
    assert client.get("/api/auth/me").status_code == 200  # the session was not ended


# ------------------------------------------------------------------------- audit trail
def test_authentication_events_are_audited_without_leaking_secrets(client, db_session) -> None:  # noqa: ANN001
    user = make_user(db_session, password=TEST_PASSWORD)
    login(client, user.email, WRONG_PASSWORD)
    login(client, "Ghost@Example.com", WRONG_PASSWORD)
    login(client, user.email)

    failures = audit_actions(db_session, AuditAction.LOGIN_FAILED)
    reasons = {entry.details["reason"] for entry in failures}
    assert reasons == {"bad_password", "unknown_user"}
    ghost = next(e for e in failures if e.details["reason"] == "unknown_user")
    assert ghost.actor_id is None and ghost.actor_email == "ghost@example.com"

    successes = audit_actions(db_session, AuditAction.LOGIN_SUCCESS)
    assert len(successes) == 1 and successes[0].actor_id == user.id
    assert successes[0].ip_address  # recorded

    everything = json.dumps(
        [
            [e.actor_email, e.entity_id, e.details, e.ip_address]
            for e in db_session.scalars(select(AuditLog))
        ],
        default=str,
    )
    for secret in (WRONG_PASSWORD, TEST_PASSWORD, "access_token", "refresh_token"):
        assert secret not in everything

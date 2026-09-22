"""Credential checking, login and password change."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import NoReturn

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.passwords import hash_password, needs_rehash, verify_dummy, verify_password
from app.models import User
from app.models.enums import AuditAction
from app.services.audit import record_audit
from app.services.sessions import (
    IssuedTokens,
    revoke_user_sessions,
    start_session,
    utcnow,
)
from app.services.users import normalize_email, set_password


class InvalidCredentialsError(Exception):
    """Deliberately vague: callers must not learn *why* authentication failed."""


def _register_failure(session: Session, settings: Settings, user: User, now: datetime) -> None:
    """Atomically count a failed attempt and lock the account when the limit is reached."""
    attempts = session.execute(
        update(User)
        .where(User.id == user.id)
        .values(failed_login_attempts=User.failed_login_attempts + 1)
        .returning(User.failed_login_attempts)
        .execution_options(synchronize_session=False)
    ).scalar_one()
    session.expire(user, ["failed_login_attempts"])
    if attempts >= settings.login_max_failed_attempts:
        user.locked_until = now + timedelta(minutes=settings.login_lockout_minutes)


def _fail(
    session: Session,
    *,
    reason: str,
    actor: User | None,
    attempted_email: str,
    ip: str | None,
    action: AuditAction = AuditAction.LOGIN_FAILED,
) -> NoReturn:
    record_audit(
        session,
        action,
        actor=actor,
        actor_email=attempted_email,
        entity_type="user",
        entity_id=str(actor.id) if actor is not None else None,
        details={"reason": reason},
        ip_address=ip,
    )
    session.commit()  # persist the audit row and failure counter before reporting failure
    raise InvalidCredentialsError


def authenticate(
    session: Session, settings: Settings, email: str, password: str, *, ip: str | None
) -> User:
    """Return the user if the credentials are valid, else raise InvalidCredentialsError.

    Every failure path (unknown email, wrong password, locked, disabled) costs one hash
    verification and raises the same exception, so neither response nor timing reveals
    whether an account exists. The real reason goes to the audit log only.
    """
    normalized = normalize_email(email)
    user = session.scalar(select(User).where(User.email == normalized))
    now = utcnow()

    if user is None:
        verify_dummy(password)
        _fail(session, reason="unknown_user", actor=None, attempted_email=normalized, ip=ip)

    if user.locked_until is not None and user.locked_until <= now:
        user.locked_until = None  # lockout served
        user.failed_login_attempts = 0

    password_ok = verify_password(password, user.password_hash)

    if user.locked_until is not None:
        _fail(session, reason="locked", actor=user, attempted_email=normalized, ip=ip)
    if not user.is_active:
        _fail(session, reason="inactive", actor=user, attempted_email=normalized, ip=ip)
    if not password_ok:
        _register_failure(session, settings, user, now)
        _fail(session, reason="bad_password", actor=user, attempted_email=normalized, ip=ip)

    if needs_rehash(user.password_hash):
        user.password_hash = hash_password(password)
    user.failed_login_attempts = 0
    user.locked_until = None
    user.last_login_at = now
    return user


def login(
    session: Session,
    settings: Settings,
    email: str,
    password: str,
    *,
    ip: str | None,
    user_agent: str | None,
) -> tuple[User, IssuedTokens]:
    user = authenticate(session, settings, email, password, ip=ip)
    record_audit(
        session,
        AuditAction.LOGIN_SUCCESS,
        actor=user,
        entity_type="user",
        entity_id=str(user.id),
        ip_address=ip,
    )
    tokens = start_session(session, settings, user, ip=ip, user_agent=user_agent)
    session.commit()
    return user, tokens


def change_password(
    session: Session,
    settings: Settings,
    user: User,
    *,
    current_password: str,
    new_password: str,
    ip: str | None,
    user_agent: str | None,
) -> IssuedTokens:
    """Change the password, end every session and start a fresh one for this device."""
    if not verify_password(current_password, user.password_hash):
        _register_failure(session, settings, user, utcnow())
        _fail(
            session,
            reason="bad_current_password",
            actor=user,
            attempted_email=user.email,
            ip=ip,
            action=AuditAction.PASSWORD_CHANGE_FAILED,
        )

    set_password(session, user, new_password)  # may raise WeakPasswordError (nothing saved)
    revoke_user_sessions(session, user.id)
    record_audit(
        session,
        AuditAction.PASSWORD_CHANGED,
        actor=user,
        entity_type="user",
        entity_id=str(user.id),
        ip_address=ip,
    )
    tokens = start_session(session, settings, user, ip=ip, user_agent=user_agent)
    session.commit()
    return tokens

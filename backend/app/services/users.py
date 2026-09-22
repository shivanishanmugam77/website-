"""User creation and password management."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.password_policy import ensure_strong_password
from app.core.passwords import hash_password
from app.models import User
from app.models.enums import AuditAction, UserRole
from app.services.audit import record_audit
from app.services.sessions import revoke_user_sessions


class EmailAlreadyRegisteredError(Exception):
    pass


class UserExistsError(Exception):
    pass


def normalize_email(email: str) -> str:
    return email.strip().lower()


def create_user(
    session: Session,
    *,
    email: str,
    password: str,
    full_name: str | None = None,
    role: UserRole = UserRole.CUSTOMER,
) -> User:
    """Create and flush a user. Raises WeakPasswordError / EmailAlreadyRegisteredError.

    The caller commits. Uniqueness is enforced by the database (race-free), not by a
    check-then-insert.
    """
    normalized = normalize_email(email)
    ensure_strong_password(password, normalized)
    user = User(
        email=normalized,
        password_hash=hash_password(password),
        full_name=full_name.strip() if full_name else None,
        role=role,
    )
    try:
        with session.begin_nested():
            session.add(user)
    except IntegrityError as exc:
        raise EmailAlreadyRegisteredError from exc
    return user


def set_password(session: Session, user: User, new_password: str) -> None:
    """Replace a user's password and clear any lockout. The caller commits."""
    ensure_strong_password(new_password, user.email)
    user.password_hash = hash_password(new_password)
    user.failed_login_attempts = 0
    user.locked_until = None


def create_admin(
    session: Session,
    *,
    email: str,
    password: str,
    full_name: str | None = None,
    reset_existing: bool = False,
) -> tuple[User, str]:
    """Create an admin, or reset an existing admin's password. Returns ``(user, outcome)``.

    Admins can only be created here (via the CLI): there is deliberately no HTTP path
    that grants the ADMIN role. The caller commits.
    """
    normalized = normalize_email(email)
    existing = session.scalar(select(User).where(User.email == normalized))

    if existing is None:
        user = create_user(
            session, email=normalized, password=password, full_name=full_name, role=UserRole.ADMIN
        )
        record_audit(
            session,
            AuditAction.ADMIN_CREATED,
            entity_type="user",
            entity_id=str(user.id),
            details={"via": "cli", "target_email": normalized},
        )
        return user, "created"

    if not reset_existing or existing.role is not UserRole.ADMIN:
        raise UserExistsError
    set_password(session, existing, password)
    revoke_user_sessions(session, existing.id)
    record_audit(
        session,
        AuditAction.PASSWORD_CHANGED,
        entity_type="user",
        entity_id=str(existing.id),
        details={"via": "cli", "target_email": normalized},
    )
    return existing, "password reset"

"""Session lifecycle: issue, rotate, revoke.

A *session* is a refresh-token family. The short-lived access JWT carries the family id
(``sid``) and is only honoured while the family still has an unrevoked, unexpired token,
so logout, password change and account deactivation take effect immediately.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import exists, select, update
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.tokens import create_access_token, hash_refresh_token, new_refresh_token
from app.models import RefreshToken, User
from app.models.enums import AuditAction
from app.services.audit import record_audit

# Two browser tabs can refresh at the same instant; the loser presents a token that was
# revoked a moment ago. Inside this window that is a benign race, not theft.
REFRESH_REUSE_GRACE_SECONDS = 10


class InvalidSessionError(Exception):
    """The refresh token is unknown, expired, revoked or its user is disabled."""


@dataclass(frozen=True)
class IssuedTokens:
    access_token: str
    refresh_token: str
    access_max_age: int  # seconds
    refresh_max_age: int  # seconds


def utcnow() -> datetime:
    return datetime.now(UTC)


def _issue(
    session: Session,
    settings: Settings,
    user_id: uuid.UUID,
    family_id: uuid.UUID,
    *,
    ip: str | None,
    user_agent: str | None,
    now: datetime,
) -> IssuedTokens:
    refresh_token, digest = new_refresh_token()
    session.add(
        RefreshToken(
            user_id=user_id,
            family_id=family_id,
            token_hash=digest,
            expires_at=now + timedelta(days=settings.refresh_token_expire_days),
            ip_address=ip,
            user_agent=user_agent,
        )
    )
    access_token = create_access_token(
        user_id,
        family_id,
        secret=settings.secret_key.get_secret_value(),
        ttl=timedelta(minutes=settings.access_token_expire_minutes),
        now=now,
    )
    return IssuedTokens(
        access_token=access_token,
        refresh_token=refresh_token,
        access_max_age=settings.access_token_expire_minutes * 60,
        refresh_max_age=settings.refresh_token_expire_days * 86400,
    )


def start_session(
    session: Session,
    settings: Settings,
    user: User,
    *,
    ip: str | None,
    user_agent: str | None,
) -> IssuedTokens:
    """Begin a new session (new family). The caller commits."""
    return _issue(
        session, settings, user.id, uuid.uuid4(), ip=ip, user_agent=user_agent, now=utcnow()
    )


def revoke_family(session: Session, family_id: uuid.UUID, now: datetime | None = None) -> None:
    session.execute(
        update(RefreshToken)
        .where(RefreshToken.family_id == family_id, RefreshToken.revoked_at.is_(None))
        .values(revoked_at=now or utcnow())
        .execution_options(synchronize_session="fetch")
    )


def revoke_user_sessions(
    session: Session, user_id: uuid.UUID, now: datetime | None = None
) -> None:
    session.execute(
        update(RefreshToken)
        .where(RefreshToken.user_id == user_id, RefreshToken.revoked_at.is_(None))
        .values(revoked_at=now or utcnow())
        .execution_options(synchronize_session="fetch")
    )


def is_session_active(session: Session, family_id: uuid.UUID, now: datetime | None = None) -> bool:
    return bool(
        session.scalar(
            select(
                exists().where(
                    RefreshToken.family_id == family_id,
                    RefreshToken.revoked_at.is_(None),
                    RefreshToken.expires_at > (now or utcnow()),
                )
            )
        )
    )


def rotate_session(
    session: Session,
    settings: Settings,
    refresh_token: str,
    *,
    ip: str | None,
    user_agent: str | None,
) -> tuple[User, IssuedTokens]:
    """Exchange a refresh token for a new pair. Commits on success and on theft detection."""
    now = utcnow()
    token = session.scalar(
        select(RefreshToken)
        .where(RefreshToken.token_hash == hash_refresh_token(refresh_token))
        .with_for_update()  # serialise concurrent refreshes of the same token
    )
    if token is None:
        raise InvalidSessionError

    if token.revoked_at is not None:
        if now - token.revoked_at <= timedelta(seconds=REFRESH_REUSE_GRACE_SECONDS):
            raise InvalidSessionError  # benign race; leave the live session alone
        # A token that was already rotated away is being replayed: assume it was stolen.
        revoke_family(session, token.family_id, now)
        record_audit(
            session,
            AuditAction.TOKEN_REUSE_DETECTED,
            actor=session.get(User, token.user_id),
            entity_type="user",
            entity_id=str(token.user_id),
            details={"family_id": str(token.family_id)},
            ip_address=ip,
        )
        session.commit()
        raise InvalidSessionError

    if token.expires_at <= now:
        raise InvalidSessionError

    user = session.get(User, token.user_id)
    if user is None or not user.is_active:
        revoke_family(session, token.family_id, now)
        session.commit()
        raise InvalidSessionError

    token.revoked_at = now
    tokens = _issue(
        session, settings, user.id, token.family_id, ip=ip, user_agent=user_agent, now=now
    )
    session.commit()
    return user, tokens

"""Append-only audit trail. ``record_audit`` never commits: it joins the caller's
transaction so the trail and the action it describes succeed or fail together."""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.logging_config import redact
from app.models import AuditLog, User
from app.models.enums import AuditAction

MAX_EMAIL_LENGTH = 320


def record_audit(
    session: Session,
    action: AuditAction,
    *,
    actor: User | None = None,
    actor_email: str | None = None,
    entity_type: str | None = None,
    entity_id: str | None = None,
    details: dict[str, Any] | None = None,
    ip_address: str | None = None,
) -> AuditLog:
    """Add an audit row to the session. ``actor_email`` is used when there is no
    authenticated actor (e.g. the email typed into a failed login)."""
    email = actor.email if actor is not None else actor_email
    entry = AuditLog(
        actor_id=actor.id if actor is not None else None,
        actor_email=email[:MAX_EMAIL_LENGTH] if email else None,
        action=action,
        entity_type=entity_type,
        entity_id=entity_id,
        # Defence in depth: anything that looks like a secret is masked before storage.
        details=redact(details or {}),
        ip_address=ip_address,
    )
    session.add(entry)
    return entry


def list_audit_logs(
    session: Session,
    *,
    action: AuditAction | None = None,
    entity_type: str | None = None,
    entity_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[AuditLog], int]:
    filters = []
    if action is not None:
        filters.append(AuditLog.action == action)
    if entity_type is not None:
        filters.append(AuditLog.entity_type == entity_type)
    if entity_id is not None:
        filters.append(AuditLog.entity_id == entity_id)

    total = session.scalar(select(func.count()).select_from(AuditLog).where(*filters)) or 0
    rows = session.scalars(
        select(AuditLog)
        .where(*filters)
        .order_by(AuditLog.created_at.desc(), AuditLog.id)
        .limit(limit)
        .offset(offset)
    ).all()
    return list(rows), total

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.api.deps import DbSession, require_admin
from app.models.enums import AuditAction
from app.schemas.audit import AuditLogOut, AuditLogPage
from app.services.audit import list_audit_logs

# Every route in this router requires the ADMIN role.
router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin)])


@router.get("/audit-logs", response_model=AuditLogPage)
def get_audit_logs(
    db: DbSession,
    action: AuditAction | None = None,
    entity_type: str | None = Query(default=None, max_length=64),
    entity_id: str | None = Query(default=None, max_length=64),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> AuditLogPage:
    rows, total = list_audit_logs(
        db,
        action=action,
        entity_type=entity_type,
        entity_id=entity_id,
        limit=limit,
        offset=offset,
    )
    return AuditLogPage(
        items=[AuditLogOut.model_validate(row) for row in rows],
        total=total,
        limit=limit,
        offset=offset,
    )

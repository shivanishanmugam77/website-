from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from app.models.enums import AuditAction


class AuditLogOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    created_at: datetime
    actor_id: uuid.UUID | None
    actor_email: str | None
    action: AuditAction
    entity_type: str | None
    entity_id: str | None
    details: dict[str, Any]
    ip_address: str | None


class AuditLogPage(BaseModel):
    items: list[AuditLogOut]
    total: int
    limit: int
    offset: int

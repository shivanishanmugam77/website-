from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import ForeignKey, Index, String, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, CreatedAtMixin, UUIDPrimaryKeyMixin, string_enum
from app.models.enums import AuditAction, CorrectionType


class AuditLog(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    """Append-only record of important admin actions."""

    __tablename__ = "audit_logs"
    __table_args__ = (
        Index("ix_audit_logs_entity", "entity_type", "entity_id"),
        Index("ix_audit_logs_created_at", "created_at"),
    )

    actor_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), index=True
    )
    # Denormalised so the trail stays readable if the user account is deleted.
    actor_email: Mapped[str | None] = mapped_column(String(320))
    action: Mapped[AuditAction] = mapped_column(string_enum(AuditAction), nullable=False)
    entity_type: Mapped[str | None] = mapped_column(String(64))
    entity_id: Mapped[str | None] = mapped_column(String(64))
    details: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )
    ip_address: Mapped[str | None] = mapped_column(String(45))


class CorrectionRecord(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    """AI value vs. human value. This table is the future fine-tuning dataset."""

    __tablename__ = "correction_records"

    candidate_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("product_candidates.id", ondelete="SET NULL"), index=True
    )
    product_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("products.id", ondelete="SET NULL"), index=True
    )
    correction_type: Mapped[CorrectionType] = mapped_column(
        string_enum(CorrectionType), nullable=False
    )
    field_name: Mapped[str | None] = mapped_column(String(128))
    ai_value: Mapped[Any | None] = mapped_column(JSONB(none_as_null=True))
    human_value: Mapped[Any | None] = mapped_column(JSONB(none_as_null=True))
    # Snapshot (catalogue id, page number, boxes...) so the record outlives its candidate.
    context: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )
    corrected_by_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL")
    )

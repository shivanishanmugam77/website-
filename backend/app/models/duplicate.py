from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import CheckConstraint, DateTime, Float, ForeignKey, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin, string_enum
from app.models.enums import DuplicateSignal, DuplicateStatus


class DuplicateFlag(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A *suspected* duplicate for an admin to confirm or dismiss. Never auto-merged."""

    __tablename__ = "duplicate_flags"
    __table_args__ = (
        CheckConstraint(
            "num_nonnulls(existing_product_id, other_candidate_id) = 1", name="exactly_one_target"
        ),
    )

    candidate_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("product_candidates.id", ondelete="CASCADE"), nullable=False, index=True
    )
    existing_product_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("products.id", ondelete="CASCADE"), index=True
    )
    other_candidate_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("product_candidates.id", ondelete="CASCADE")
    )
    signal: Mapped[DuplicateSignal] = mapped_column(string_enum(DuplicateSignal), nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[DuplicateStatus] = mapped_column(
        string_enum(DuplicateStatus),
        nullable=False,
        default=DuplicateStatus.OPEN,
        server_default=DuplicateStatus.OPEN.value,
        index=True,
    )
    details: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )
    resolved_by_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL")
    )
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

"""The history of manual corrections to a page's photo panels (for undo and for the record)."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Integer, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, CreatedAtMixin, UUIDPrimaryKeyMixin, string_enum
from app.models.enums import PanelEditAction


class PanelEdit(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    """One change an admin made to a page's panels.

    ``before`` / ``after`` are snapshots ``{"bbox": [x0, y0, x1, y1], "origin": "AI"}`` (None
    where there was nothing: no ``before`` for a new panel, no ``after`` for a deleted one).
    ``seq`` orders the changes on a page; undo reverts the highest ``seq`` that is not yet
    undone. ``panel_id`` is deliberately not a foreign key: a deleted panel's row is gone
    but its history must stay (and undoing the deletion brings the same id back).
    """

    __tablename__ = "panel_edits"
    __table_args__ = (UniqueConstraint("page_id", "seq", name="uq_panel_edits_page_seq"),)

    page_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("catalogue_pages.id", ondelete="CASCADE"), nullable=False, index=True
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    action: Mapped[PanelEditAction] = mapped_column(string_enum(PanelEditAction), nullable=False)
    panel_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    before: Mapped[dict[str, Any] | None] = mapped_column(JSONB(none_as_null=True))
    after: Mapped[dict[str, Any] | None] = mapped_column(JSONB(none_as_null=True))
    actor_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    undone_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

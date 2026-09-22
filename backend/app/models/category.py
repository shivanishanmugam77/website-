from __future__ import annotations

import uuid

from sqlalchemy import Boolean, ForeignKey, Integer, String, Text, text, true
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class Category(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Self-referencing tree stored in the database (never hard-coded in the UI)."""

    __tablename__ = "categories"

    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("categories.id", ondelete="RESTRICT"), index=True
    )
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    slug: Mapped[str] = mapped_column(String(160), unique=True, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    sort_order: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0")
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default=true()
    )

    parent: Mapped[Category | None] = relationship(
        remote_side="Category.id", back_populates="children"
    )
    # RESTRICT is enforced by the database; the ORM must not silently NULL children.
    children: Mapped[list[Category]] = relationship(
        back_populates="parent", passive_deletes="all", order_by="Category.sort_order"
    )

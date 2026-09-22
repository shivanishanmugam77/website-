"""Customer-facing interaction records."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin, string_enum
from app.models.enums import EnquiryStatus


class Enquiry(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Guests may enquire, so ``user_id`` is optional."""

    __tablename__ = "enquiries"

    user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), index=True
    )
    product_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("products.id", ondelete="SET NULL"), index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str] = mapped_column(String(320), nullable=False)
    phone: Mapped[str | None] = mapped_column(String(64))
    message: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[EnquiryStatus] = mapped_column(
        string_enum(EnquiryStatus),
        nullable=False,
        default=EnquiryStatus.NEW,
        server_default=EnquiryStatus.NEW.value,
        index=True,
    )


class WishlistItem(Base):
    """One (user, product) favourite. Composite primary key prevents duplicates."""

    __tablename__ = "wishlist_items"

    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    product_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("products.id", ondelete="CASCADE"), primary_key=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

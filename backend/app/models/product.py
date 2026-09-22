"""Published-catalogue entities.

Customer-facing values live in typed columns (the *verified* values). Where each value
came from (OCR / AI / human) is recorded per field in ``field_provenance``.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    false,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, CreatedAtMixin, TimestampMixin, UUIDPrimaryKeyMixin, string_enum
from app.models.enums import DataOrigin, ProductImageType, ProductStatus

if TYPE_CHECKING:
    from app.models.candidate import ProductCandidate
    from app.models.catalogue import Catalogue, CataloguePage
    from app.models.category import Category
    from app.models.media import DetectedObject, SegmentationResult, SourceImage
    from app.models.supplier import Supplier


class Product(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "products"
    __table_args__ = (
        CheckConstraint("price IS NULL OR price >= 0", name="price_non_negative"),
        CheckConstraint("price IS NULL OR currency IS NOT NULL", name="price_requires_currency"),
    )

    slug: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    sku: Mapped[str | None] = mapped_column(String(128), index=True)
    model: Mapped[str | None] = mapped_column(String(128))
    description: Mapped[str | None] = mapped_column(Text)
    material: Mapped[str | None] = mapped_column(String(255))
    colour: Mapped[str | None] = mapped_column(String(128))
    product_type: Mapped[str | None] = mapped_column(String(64))
    # {"length": 1200, "width": 600, "height": 750, "unit": "mm", "raw_text": "..."}
    dimensions: Mapped[dict[str, Any] | None] = mapped_column(JSONB(none_as_null=True))
    specifications: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )
    # Only set when a real price exists in the source or was entered by an admin.
    price: Mapped[Decimal | None] = mapped_column(Numeric(12, 2))
    currency: Mapped[str | None] = mapped_column(String(3))

    status: Mapped[ProductStatus] = mapped_column(
        string_enum(ProductStatus),
        nullable=False,
        default=ProductStatus.DRAFT,
        server_default=ProductStatus.DRAFT.value,
        index=True,
    )
    is_featured: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=false()
    )
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    origin: Mapped[DataOrigin] = mapped_column(
        string_enum(DataOrigin),
        nullable=False,
        default=DataOrigin.HUMAN,
        server_default=DataOrigin.HUMAN.value,
    )
    # {"name": {"source": "OCR", "confidence": 0.97, "verified": true}, ...}
    field_provenance: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )

    category_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("categories.id", ondelete="RESTRICT"), index=True
    )
    supplier_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("suppliers.id", ondelete="RESTRICT"), index=True
    )
    # Traceability back to where the product came from.
    candidate_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("product_candidates.id", ondelete="SET NULL"), index=True
    )
    catalogue_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("catalogues.id", ondelete="SET NULL"), index=True
    )
    source_page_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("catalogue_pages.id", ondelete="SET NULL")
    )
    created_by_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL")
    )

    category: Mapped[Category | None] = relationship()
    supplier: Mapped[Supplier | None] = relationship()
    candidate: Mapped[ProductCandidate | None] = relationship()
    catalogue: Mapped[Catalogue | None] = relationship()
    source_page: Mapped[CataloguePage | None] = relationship()
    images: Mapped[list[ProductImage]] = relationship(
        back_populates="product",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="ProductImage.sort_order",
    )


class ProductImage(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "product_images"

    product_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("products.id", ondelete="CASCADE"), nullable=False, index=True
    )
    image_type: Mapped[ProductImageType] = mapped_column(
        string_enum(ProductImageType), nullable=False
    )
    storage_key: Mapped[str] = mapped_column(String(512), nullable=False)
    width_px: Mapped[int | None] = mapped_column(Integer)
    height_px: Mapped[int | None] = mapped_column(Integer)
    alt_text: Mapped[str | None] = mapped_column(String(255))
    sort_order: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0")
    )
    # Traceability: which AI artefacts produced this image (NULL for manual uploads).
    source_image_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("source_images.id", ondelete="SET NULL")
    )
    detected_object_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("detected_objects.id", ondelete="SET NULL")
    )
    segmentation_result_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("segmentation_results.id", ondelete="SET NULL")
    )

    product: Mapped[Product] = relationship(back_populates="images")
    source_image: Mapped[SourceImage | None] = relationship()
    detected_object: Mapped[DetectedObject | None] = relationship()
    segmentation_result: Mapped[SegmentationResult | None] = relationship()

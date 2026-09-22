from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Float,
    ForeignKey,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin, string_enum
from app.models.enums import CatalogueStatus, PageStatus, PageType

if TYPE_CHECKING:
    from app.models.media import OCRResult, SourceImage
    from app.models.supplier import Supplier


class Catalogue(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "catalogues"
    __table_args__ = (
        CheckConstraint("processing_progress BETWEEN 0 AND 100", name="progress_range"),
    )

    supplier_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("suppliers.id", ondelete="RESTRICT"), index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    storage_key: Mapped[str] = mapped_column(String(512), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    page_count: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[CatalogueStatus] = mapped_column(
        string_enum(CatalogueStatus),
        nullable=False,
        default=CatalogueStatus.UPLOADED,
        server_default=CatalogueStatus.UPLOADED.value,
        index=True,
    )
    processing_progress: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, default=0, server_default=text("0")
    )
    current_stage: Mapped[str | None] = mapped_column(String(64))
    processing_error: Mapped[str | None] = mapped_column(Text)
    uploaded_by_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL")
    )

    supplier: Mapped[Supplier | None] = relationship()
    pages: Mapped[list[CataloguePage]] = relationship(
        back_populates="catalogue",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="CataloguePage.page_number",
    )


class CataloguePage(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "catalogue_pages"
    __table_args__ = (
        UniqueConstraint("catalogue_id", "page_number", name="uq_catalogue_pages_page_number"),
        CheckConstraint("page_number >= 1", name="page_number_positive"),
    )

    catalogue_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("catalogues.id", ondelete="CASCADE"), nullable=False
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[PageStatus] = mapped_column(
        string_enum(PageStatus),
        nullable=False,
        default=PageStatus.PENDING,
        server_default=PageStatus.PENDING.value,
    )
    page_type: Mapped[PageType | None] = mapped_column(string_enum(PageType))
    page_type_confidence: Mapped[float | None] = mapped_column(Float)
    image_key: Mapped[str | None] = mapped_column(String(512))
    thumbnail_key: Mapped[str | None] = mapped_column(String(512))
    width_px: Mapped[int | None] = mapped_column(Integer)
    height_px: Mapped[int | None] = mapped_column(Integer)
    error_stage: Mapped[str | None] = mapped_column(String(64))
    processing_error: Mapped[str | None] = mapped_column(Text)

    catalogue: Mapped[Catalogue] = relationship(back_populates="pages")
    source_images: Mapped[list[SourceImage]] = relationship(
        back_populates="page", cascade="all, delete-orphan", passive_deletes=True
    )
    ocr_results: Mapped[list[OCRResult]] = relationship(
        back_populates="page",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="OCRResult.line_index",
    )

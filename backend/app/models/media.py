"""Everything derived from a catalogue page: images, OCR, detections, masks, embeddings."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.constants import EMBEDDING_DIM
from app.models.base import (
    Base,
    BoundingBoxMixin,
    CreatedAtMixin,
    UUIDPrimaryKeyMixin,
    string_enum,
)
from app.models.enums import DataOrigin, SegmentationStatus, SourceImageKind

if TYPE_CHECKING:
    from app.models.catalogue import CataloguePage


class SourceImage(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    """An image belonging to a page: the page render itself, an embedded image or a region."""

    __tablename__ = "source_images"

    page_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("catalogue_pages.id", ondelete="CASCADE"), nullable=False, index=True
    )
    kind: Mapped[SourceImageKind] = mapped_column(string_enum(SourceImageKind), nullable=False)
    storage_key: Mapped[str] = mapped_column(String(512), nullable=False)
    width_px: Mapped[int] = mapped_column(Integer, nullable=False)
    height_px: Mapped[int] = mapped_column(Integer, nullable=False)
    # Location on the rendered page (NULL for the full-page image).
    bbox_x0: Mapped[float | None] = mapped_column(Float)
    bbox_y0: Mapped[float | None] = mapped_column(Float)
    bbox_x1: Mapped[float | None] = mapped_column(Float)
    bbox_y1: Mapped[float | None] = mapped_column(Float)
    region_confidence: Mapped[float | None] = mapped_column(Float)
    # AI = as the system cut it; AI_HUMAN_REVIEW = the system's box, moved or resized by an
    # admin; HUMAN = drawn by an admin. Reprocessing refuses to discard non-AI work unasked.
    origin: Mapped[DataOrigin] = mapped_column(
        string_enum(DataOrigin),
        nullable=False,
        default=DataOrigin.AI,
        server_default=DataOrigin.AI.value,
    )
    sha256: Mapped[str | None] = mapped_column(String(64))
    phash: Mapped[str | None] = mapped_column(String(16), index=True)

    page: Mapped[CataloguePage] = relationship(back_populates="source_images")
    detected_objects: Mapped[list[DetectedObject]] = relationship(
        back_populates="source_image", cascade="all, delete-orphan", passive_deletes=True
    )


class OCRResult(UUIDPrimaryKeyMixin, CreatedAtMixin, BoundingBoxMixin, Base):
    """One recognised text line on a page."""

    __tablename__ = "ocr_results"

    page_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("catalogue_pages.id", ondelete="CASCADE"), nullable=False, index=True
    )
    line_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float)
    language: Mapped[str | None] = mapped_column(String(16))
    polygon: Mapped[list[Any] | None] = mapped_column(JSONB(none_as_null=True))
    engine: Mapped[str] = mapped_column(String(64), nullable=False)
    engine_version: Mapped[str | None] = mapped_column(String(64))

    page: Mapped[CataloguePage] = relationship(back_populates="ocr_results")


class DetectedObject(UUIDPrimaryKeyMixin, CreatedAtMixin, BoundingBoxMixin, Base):
    __tablename__ = "detected_objects"

    source_image_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("source_images.id", ondelete="CASCADE"), nullable=False, index=True
    )
    label: Mapped[str] = mapped_column(String(128), nullable=False)
    prompt: Mapped[str | None] = mapped_column(String(255))
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    model_version: Mapped[str | None] = mapped_column(String(64))

    source_image: Mapped[SourceImage] = relationship(back_populates="detected_objects")
    segmentation: Mapped[SegmentationResult | None] = relationship(
        back_populates="detected_object",
        uselist=False,
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class SegmentationResult(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "segmentation_results"

    detected_object_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("detected_objects.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    status: Mapped[SegmentationStatus] = mapped_column(
        string_enum(SegmentationStatus), nullable=False
    )
    confidence: Mapped[float | None] = mapped_column(Float)
    mask_key: Mapped[str | None] = mapped_column(String(512))
    crop_key: Mapped[str | None] = mapped_column(String(512))
    cutout_key: Mapped[str | None] = mapped_column(String(512))  # transparent PNG
    white_bg_key: Mapped[str | None] = mapped_column(String(512))
    thumbnail_key: Mapped[str | None] = mapped_column(String(512))
    error: Mapped[str | None] = mapped_column(Text)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    model_version: Mapped[str | None] = mapped_column(String(64))

    detected_object: Mapped[DetectedObject] = relationship(back_populates="segmentation")


class ImageEmbedding(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    """Embedding used for duplicate detection / similarity.

    No ANN index yet: it is added together with the first query that needs it
    (Phase 6+), once real data exists to size it against.
    """

    __tablename__ = "image_embeddings"

    source_image_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("source_images.id", ondelete="CASCADE"), nullable=False, index=True
    )
    detected_object_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("detected_objects.id", ondelete="CASCADE"), index=True
    )
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)

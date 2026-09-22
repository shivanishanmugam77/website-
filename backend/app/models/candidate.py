"""Product candidates: AI (or human) proposals that an admin reviews before publishing."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Table,
    Uuid,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin, string_enum
from app.models.enums import CandidateStatus, DataOrigin

if TYPE_CHECKING:
    from app.models.catalogue import Catalogue, CataloguePage
    from app.models.media import DetectedObject, SourceImage

CONFIDENCE_COLUMNS = (
    "detection_confidence",
    "segmentation_confidence",
    "ocr_confidence",
    "association_confidence",
    "metadata_confidence",
    "overall_confidence",
)

# Association tables: one candidate can span several pages/images/objects, which is how
# a multi-piece "system" or a multi-angle product becomes ONE candidate.
candidate_pages = Table(
    "candidate_pages",
    Base.metadata,
    Column(
        "candidate_id",
        Uuid,
        ForeignKey("product_candidates.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "page_id",
        Uuid,
        ForeignKey("catalogue_pages.id", ondelete="CASCADE"),
        primary_key=True,
        index=True,
    ),
)

candidate_source_images = Table(
    "candidate_images",
    Base.metadata,
    Column(
        "candidate_id",
        Uuid,
        ForeignKey("product_candidates.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "source_image_id",
        Uuid,
        ForeignKey("source_images.id", ondelete="CASCADE"),
        primary_key=True,
        index=True,
    ),
)

candidate_detected_objects = Table(
    "candidate_objects",
    Base.metadata,
    Column(
        "candidate_id",
        Uuid,
        ForeignKey("product_candidates.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "detected_object_id",
        Uuid,
        ForeignKey("detected_objects.id", ondelete="CASCADE"),
        primary_key=True,
        index=True,
    ),
)


class ProductCandidate(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "product_candidates"
    __table_args__ = (
        Index("ix_product_candidates_catalogue_status", "catalogue_id", "status"),
        *(
            CheckConstraint(f"{column} IS NULL OR {column} BETWEEN 0 AND 1", name=f"{column}_range")
            for column in CONFIDENCE_COLUMNS
        ),
    )

    catalogue_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("catalogues.id", ondelete="CASCADE"), nullable=False
    )
    status: Mapped[CandidateStatus] = mapped_column(
        string_enum(CandidateStatus),
        nullable=False,
        default=CandidateStatus.NEEDS_REVIEW,
        server_default=CandidateStatus.NEEDS_REVIEW.value,
    )
    origin: Mapped[DataOrigin] = mapped_column(
        string_enum(DataOrigin),
        nullable=False,
        default=DataOrigin.AI,
        server_default=DataOrigin.AI.value,
    )
    # {"name": {"value": "YY-11", "source": "OCR", "confidence": 0.97}, "dimensions": null, ...}
    fields: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )
    # e.g. ["SEGMENTATION_FAILED", "POSSIBLE_DUPLICATE", "UNKNOWN_PRODUCT"]
    flags: Mapped[list[Any]] = mapped_column(
        JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb")
    )
    # Human-readable evidence, e.g. ["same SKU on pages 7 and 8"].
    reasoning: Mapped[list[Any]] = mapped_column(
        JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb")
    )

    detection_confidence: Mapped[float | None] = mapped_column(Float)
    segmentation_confidence: Mapped[float | None] = mapped_column(Float)
    ocr_confidence: Mapped[float | None] = mapped_column(Float)
    association_confidence: Mapped[float | None] = mapped_column(Float)
    metadata_confidence: Mapped[float | None] = mapped_column(Float)
    overall_confidence: Mapped[float | None] = mapped_column(Float)

    merged_into_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("product_candidates.id", ondelete="SET NULL")
    )
    reviewed_by_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL")
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    catalogue: Mapped[Catalogue] = relationship()
    pages: Mapped[list[CataloguePage]] = relationship(secondary=candidate_pages)
    source_images: Mapped[list[SourceImage]] = relationship(secondary=candidate_source_images)
    detected_objects: Mapped[list[DetectedObject]] = relationship(
        secondary=candidate_detected_objects
    )

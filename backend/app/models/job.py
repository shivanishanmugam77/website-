from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    SmallInteger,
    String,
    Text,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin, string_enum
from app.models.enums import JobStatus, JobType


class ProcessingJob(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "processing_jobs"
    __table_args__ = (CheckConstraint("progress BETWEEN 0 AND 100", name="progress_range"),)

    catalogue_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("catalogues.id", ondelete="CASCADE"), nullable=False, index=True
    )
    page_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("catalogue_pages.id", ondelete="CASCADE")
    )
    candidate_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("product_candidates.id", ondelete="SET NULL")
    )
    job_type: Mapped[JobType] = mapped_column(string_enum(JobType), nullable=False)
    status: Mapped[JobStatus] = mapped_column(
        string_enum(JobStatus),
        nullable=False,
        default=JobStatus.QUEUED,
        server_default=JobStatus.QUEUED.value,
        index=True,
    )
    celery_task_id: Mapped[str | None] = mapped_column(String(64))
    current_stage: Mapped[str | None] = mapped_column(String(64))
    progress: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, default=0, server_default=text("0")
    )
    pages_processed: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0")
    )
    total_pages: Mapped[int | None] = mapped_column(Integer)
    products_found: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0")
    )
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_by_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL")
    )

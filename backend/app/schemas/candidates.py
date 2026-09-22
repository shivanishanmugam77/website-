from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from app.models.enums import CandidateStatus, DataOrigin


class CandidateAssembleOut(BaseModel):
    """The result of (re)building a catalogue's product candidates."""

    pages_with_finds: int
    candidates_created: int
    ready: int
    needs_review: int
    duplicate_flags_created: int


class DuplicateFlagOut(BaseModel):
    id: uuid.UUID
    other_candidate_id: uuid.UUID | None
    existing_product_id: uuid.UUID | None
    signal: str
    score: float
    status: str
    details: dict[str, Any]


class ProductCandidateSummaryOut(BaseModel):
    """One row in a catalogue's candidate list: enough to triage without opening it."""

    id: uuid.UUID
    status: CandidateStatus
    origin: DataOrigin
    overall_confidence: float | None
    flags: list[Any]
    name: str | None
    model_code: str | None
    page_numbers: list[int]
    thumbnail_url: str | None
    created_at: datetime


class ProductCandidateOut(BaseModel):
    """One candidate in full: every field, every flag, every find behind it."""

    id: uuid.UUID
    catalogue_id: uuid.UUID
    status: CandidateStatus
    origin: DataOrigin
    fields: dict[str, Any]
    flags: list[Any]
    reasoning: list[Any]
    detection_confidence: float | None
    segmentation_confidence: float | None
    ocr_confidence: float | None
    association_confidence: float | None
    metadata_confidence: float | None
    overall_confidence: float | None
    page_numbers: list[int]
    detected_object_ids: list[uuid.UUID]
    images: dict[uuid.UUID, dict[str, str]]
    duplicate_flags: list[DuplicateFlagOut]
    created_at: datetime
    updated_at: datetime

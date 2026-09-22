from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.enums import (
    CatalogueStatus,
    DataOrigin,
    JobStatus,
    JobType,
    PageStatus,
    PageType,
    SourceImageKind,
)


class SupplierCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    website: str | None = Field(default=None, max_length=512)
    contact_email: str | None = Field(default=None, max_length=320)
    description: str | None = Field(default=None, max_length=5000)


class SupplierOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    slug: str
    website: str | None
    contact_email: str | None
    description: str | None
    is_active: bool
    created_at: datetime


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    job_type: JobType
    status: JobStatus
    current_stage: str | None
    progress: int
    pages_processed: int
    total_pages: int | None
    products_found: int
    error: str | None
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime


class CatalogueOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    original_filename: str
    supplier_id: uuid.UUID | None
    supplier: SupplierOut | None
    status: CatalogueStatus
    page_count: int | None
    processing_progress: int
    current_stage: str | None
    processing_error: str | None
    file_size_bytes: int
    sha256: str
    uploaded_by_id: uuid.UUID | None
    created_at: datetime
    updated_at: datetime


class CatalogueDetail(CatalogueOut):
    job: JobOut | None
    page_status_counts: dict[str, int]
    embedded_image_count: int


class CataloguePageOut(BaseModel):
    id: uuid.UUID
    page_number: int
    status: PageStatus
    page_type: PageType | None
    page_type_confidence: float | None
    width_px: int | None
    height_px: int | None
    error_stage: str | None
    processing_error: str | None
    embedded_image_count: int
    panel_count: int
    image_url: str | None
    thumbnail_url: str | None


class SourceImageOut(BaseModel):
    id: uuid.UUID
    kind: SourceImageKind
    width_px: int
    height_px: int
    # Position on the rendered page image (pixels), or null when unknown.
    bbox: tuple[float, float, float, float] | None
    phash: str | None
    url: str


class OcrLineOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    line_index: int
    text: str
    confidence: float | None
    language: str | None
    # Position on the rendered page image (pixels).
    bbox: tuple[float, float, float, float]
    polygon: list[list[float]] | None


class ModelCodeOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    code: str
    line_index: int
    labelled: bool


class DimensionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    width: float | None
    depth: float | None
    height: float | None
    length: float | None
    unit: str
    labelled: bool
    raw_text: str
    line_index: int


class SeriesLabelOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    label: str
    line_index: int


class TextSignalsOut(BaseModel):
    """Product facts spotted in the text by pattern rules. Hints, not verified data."""

    model_config = ConfigDict(from_attributes=True)

    model_codes: list[ModelCodeOut]
    dimensions: list[DimensionOut]
    series: list[SeriesLabelOut]


class PageTextOut(BaseModel):
    page_number: int
    status: PageStatus
    page_type: PageType | None
    page_type_confidence: float | None
    engine: str | None
    engine_version: str | None
    line_count: int
    lines: list[OcrLineOut]
    signals: TextSignalsOut


class PanelOut(BaseModel):
    """One photo panel on a page, with the text that belongs to it."""

    id: uuid.UUID
    index: int  # 1, 2, 3 ... in reading order: the numbers drawn by the preview picture
    origin: DataOrigin  # AI = as cut by the system; AI_HUMAN_REVIEW = adjusted; HUMAN = drawn
    bbox: tuple[float, float, float, float]  # on the rendered page image (pixels)
    width_px: int
    height_px: int
    area_share: float  # of the page area, 0..1
    url: str  # the panel cut out of the page, full resolution
    text_line_indexes: list[int]  # which of the page's text lines belong to this panel
    text: list[str]
    signals: TextSignalsOut  # model codes / sizes / series in that text (line_index = page line)


class PanelBoxIn(BaseModel):
    """A panel's position on the rendered page image: left, top, right, bottom in pixels."""

    bbox: list[float] = Field(min_length=4, max_length=4)


class Page(BaseModel):
    """Generic pagination envelope."""

    total: int
    limit: int
    offset: int


class CatalogueList(Page):
    items: list[CatalogueOut]


class CataloguePageList(Page):
    items: list[CataloguePageOut]

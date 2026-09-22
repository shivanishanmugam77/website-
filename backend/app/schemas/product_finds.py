from __future__ import annotations

import uuid

from pydantic import BaseModel

from app.models.enums import SegmentationStatus


class ProductFindOut(BaseModel):
    """One product found inside a photo panel, with its outline and pictures."""

    id: uuid.UUID
    panel_id: uuid.UUID
    panel_index: int  # the panel's number on its page, in reading order (as in the previews)
    label: str  # what the finder called it: "desk", "office chair" ...
    confidence: float  # the finder's confidence, 0..1
    bbox: tuple[float, float, float, float]  # in the pixels of the panel picture
    outline_status: SegmentationStatus | None  # None only if no outline was made at all
    outline_score: float | None  # the outlining model's own estimate, 0..1
    note: str | None  # why the outline is doubtful or missing
    images: dict[str, str]  # picture kind -> address, for the pictures that exist

"""Forgetting the products found in a panel when the panel itself changes.

A find is a box and an outline drawn on one particular picture. When an admin moves, resizes
or deletes the panel, that picture is no longer the one the find was made on, so the finds
(and their pictures) are removed rather than left pointing at the wrong place.
"""

from __future__ import annotations

import uuid

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.models import DetectedObject, SegmentationResult


def forget_finds(session: Session, panel_id: uuid.UUID) -> list[str]:
    """Delete the products found inside one panel.

    Returns the keys of their stored pictures. The caller removes those files after the change
    has been committed, so a failed change never loses pictures the database still refers to.
    """
    rows = session.execute(
        select(
            SegmentationResult.mask_key,
            SegmentationResult.crop_key,
            SegmentationResult.cutout_key,
            SegmentationResult.white_bg_key,
            SegmentationResult.thumbnail_key,
        )
        .join(DetectedObject, DetectedObject.id == SegmentationResult.detected_object_id)
        .where(DetectedObject.source_image_id == panel_id)
    ).all()
    session.execute(delete(DetectedObject).where(DetectedObject.source_image_id == panel_id))
    return [key for row in rows for key in row if key]

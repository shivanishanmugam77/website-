"""Questions about the manual panel corrections made to a catalogue."""

from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import CataloguePage, PanelEdit


def count_corrections(session: Session, catalogue_id: uuid.UUID) -> int:
    """Manual changes still in force on the catalogue (undone ones no longer count)."""
    return (
        session.scalar(
            select(func.count())
            .select_from(PanelEdit)
            .join(CataloguePage, PanelEdit.page_id == CataloguePage.id)
            .where(CataloguePage.catalogue_id == catalogue_id, PanelEdit.undone_at.is_(None))
        )
        or 0
    )

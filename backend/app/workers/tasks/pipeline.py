from __future__ import annotations

import uuid

from app.core.config import get_settings
from app.core.database import session_scope
from app.services.pipeline import run_catalogue_pipeline
from app.services.storage import get_storage
from app.workers.celery_app import celery_app

_settings = get_settings()
_TIME_LIMIT = _settings.pipeline_time_limit_minutes * 60


@celery_app.task(
    name="pipeline.process_catalogue",
    soft_time_limit=_TIME_LIMIT,  # raises inside the task so it can record a clean failure
    time_limit=_TIME_LIMIT + 300,  # hard kill if the task ignores the soft limit
)
def process_catalogue(catalogue_id: str, job_id: str) -> None:
    """Render and extract a catalogue. Arguments are strings so they are JSON-serialisable."""
    with session_scope() as session:
        run_catalogue_pipeline(
            session,
            get_storage(),
            get_settings(),
            catalogue_id=uuid.UUID(catalogue_id),
            job_id=uuid.UUID(job_id),
        )

"""Celery application. Start a worker with:

    celery -A app.workers.celery_app:celery_app worker -Q default,pipeline,interactive
"""

from __future__ import annotations

from typing import Any

from celery import Celery
from celery.signals import setup_logging

from app.core.config import get_settings
from app.core.constants import QUEUE_DEFAULT
from app.core.logging_config import configure_logging
from app.workers.queues import QUEUES

_settings = get_settings()

celery_app = Celery(
    "catalogue",
    broker=_settings.redis_url,
    backend=_settings.redis_url,
    include=["app.workers.tasks.system", "app.workers.tasks.pipeline"],
)

celery_app.conf.update(
    task_default_queue=QUEUE_DEFAULT,
    task_queues=QUEUES,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    result_expires=3600,
    task_track_started=True,
    # A crashed worker must not lose a long job: acknowledge after completion and
    # take one task at a time (jobs are minutes long, not milliseconds).
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
    # Redis re-delivers an unacknowledged task after `visibility_timeout` (default 1 hour).
    # Pipeline jobs can legitimately run longer, which would start a duplicate run of a job
    # that is still in progress. Keep this comfortably above the hard task time limit.
    broker_transport_options={
        "visibility_timeout": (_settings.pipeline_time_limit_minutes + 30) * 60
    },
)


@setup_logging.connect
def _configure_worker_logging(**_: Any) -> None:
    """Replace Celery's logging setup with the same structured logging the API uses."""
    configure_logging(_settings.log_level, _settings.log_json)

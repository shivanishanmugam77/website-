from __future__ import annotations

from app.workers.celery_app import celery_app


@celery_app.task(name="system.ping")
def ping() -> str:
    """Round-trip probe used to verify that workers are consuming from the broker."""
    return "pong"

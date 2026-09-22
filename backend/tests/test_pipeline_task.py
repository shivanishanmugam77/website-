from __future__ import annotations

from contextlib import contextmanager

from app.core.config import get_settings
from app.models.enums import CatalogueStatus
from app.workers.celery_app import celery_app
from app.workers.tasks import pipeline as task_module
from app.workers.tasks.pipeline import process_catalogue
from tests.factories import make_stored_catalogue
from tests.pdf_factory import blank_pages


def test_task_is_registered_with_time_limits() -> None:
    assert "pipeline.process_catalogue" in celery_app.tasks
    limit = get_settings().pipeline_time_limit_minutes * 60
    assert process_catalogue.soft_time_limit == limit
    assert process_catalogue.time_limit > limit  # hard kill comes after the graceful one


def test_redis_will_not_redeliver_a_job_that_is_still_running() -> None:
    """Redis re-delivers unacknowledged tasks after `visibility_timeout`; if that were shorter
    than a long job, the same catalogue would be processed twice at once."""
    visibility = celery_app.conf.broker_transport_options["visibility_timeout"]
    assert visibility > process_catalogue.time_limit


def test_task_runs_the_pipeline_end_to_end(db_session, storage, monkeypatch) -> None:  # noqa: ANN001
    catalogue, job = make_stored_catalogue(db_session, storage, blank_pages(2))

    @contextmanager
    def fake_scope():  # noqa: ANN202
        yield db_session

    monkeypatch.setattr(task_module, "session_scope", fake_scope)
    monkeypatch.setattr(task_module, "get_storage", lambda: storage)

    result = process_catalogue.apply(args=[str(catalogue.id), str(job.id)])
    assert result.successful()
    db_session.refresh(catalogue)
    assert catalogue.status is CatalogueStatus.COMPLETED and catalogue.page_count == 2

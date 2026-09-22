"""Only one run at a time per catalogue (two would delete each other's pages)."""

from __future__ import annotations

from contextlib import closing

import pytest
from sqlalchemy import func, select

from app.models import CataloguePage, ProcessingJob
from app.models.enums import CatalogueStatus, JobStatus, JobType
from app.services import pipeline
from app.services.pipeline import catalogue_lock_key, run_catalogue_pipeline
from tests.factories import make_stored_catalogue, settings_with
from tests.pdf_factory import blank_pages

SETTINGS = settings_with(ocr_enabled=False, panel_detection_enabled=False)


def run(session, storage, catalogue, job) -> None:  # noqa: ANN001
    run_catalogue_pipeline(session, storage, SETTINGS, catalogue_id=catalogue.id, job_id=job.id)


def new_job(session, catalogue) -> ProcessingJob:  # noqa: ANN001
    job = ProcessingJob(catalogue_id=catalogue.id, job_type=JobType.CATALOGUE_PROCESS)
    session.add(job)
    session.flush()
    return job


def is_free(engine, key: int) -> bool:  # noqa: ANN001
    """Can another connection take the lock right now? (Releases it again straight away.)"""
    with closing(engine.connect()) as other:
        got = other.execute(select(func.pg_try_advisory_lock(key))).scalar()
        if got:
            other.execute(select(func.pg_advisory_unlock(key)))
        return bool(got)


def test_lock_keys_are_stable_and_fit_a_signed_64_bit_integer(db_session, storage) -> None:  # noqa: ANN001
    catalogue, _job = make_stored_catalogue(db_session, storage, blank_pages(1))
    key = catalogue_lock_key(catalogue.id)
    assert key == catalogue_lock_key(catalogue.id)
    assert -(2**63) <= key < 2**63


def test_a_run_is_refused_while_another_holds_the_catalogue(db_session, storage, engine) -> None:  # noqa: ANN001
    catalogue, job = make_stored_catalogue(db_session, storage, blank_pages(2))
    key = catalogue_lock_key(catalogue.id)
    with closing(engine.connect()) as other:
        other.execute(select(func.pg_advisory_lock(key)))  # "the other run"
        try:
            run(db_session, storage, catalogue, job)
        finally:
            other.execute(select(func.pg_advisory_unlock(key)))

    db_session.refresh(job)
    db_session.refresh(catalogue)
    assert job.status is JobStatus.FAILED and "already being processed" in job.error
    assert job.finished_at is not None
    assert catalogue.status is CatalogueStatus.UPLOADED  # left exactly as the other run has it
    assert db_session.scalars(select(CataloguePage)).all() == []  # nothing was wiped or made

    run(db_session, storage, catalogue, new_job(db_session, catalogue))  # free again: works
    db_session.refresh(catalogue)
    assert catalogue.status is CatalogueStatus.COMPLETED


def test_the_lock_is_released_when_a_run_finishes(db_session, storage, engine) -> None:  # noqa: ANN001
    catalogue, job = make_stored_catalogue(db_session, storage, blank_pages(1))
    run(db_session, storage, catalogue, job)
    assert is_free(engine, catalogue_lock_key(catalogue.id))


def test_the_lock_is_released_when_a_run_crashes(db_session, storage, engine, monkeypatch) -> None:  # noqa: ANN001
    def crash(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        raise RuntimeError("boom")

    monkeypatch.setattr(pipeline, "_run", crash)
    catalogue, job = make_stored_catalogue(db_session, storage, blank_pages(1))
    with pytest.raises(RuntimeError):
        run(db_session, storage, catalogue, job)
    assert is_free(engine, catalogue_lock_key(catalogue.id))


def test_other_catalogues_are_not_blocked(db_session, storage, engine) -> None:  # noqa: ANN001
    busy, _ = make_stored_catalogue(db_session, storage, blank_pages(1))
    other, job = make_stored_catalogue(db_session, storage, blank_pages(1))
    with closing(engine.connect()) as holder:
        holder.execute(select(func.pg_advisory_lock(catalogue_lock_key(busy.id))))
        try:
            run(db_session, storage, other, job)
        finally:
            holder.execute(select(func.pg_advisory_unlock(catalogue_lock_key(busy.id))))
    db_session.refresh(other)
    assert other.status is CatalogueStatus.COMPLETED

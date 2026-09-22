"""Stage 2 of the pipeline (reading the text) against a real database and file store.

A stand-in engine returns known lines, so these tests check what the pipeline does with
text - storing it, typing the pages, surviving failures - not how well an engine reads.
"""

from __future__ import annotations

import pytest
from sqlalchemy import func, select

from app.ml.ocr import OcrEngineError
from app.models import CataloguePage, OCRResult, ProcessingJob
from app.models.enums import CatalogueStatus, JobStatus, PageStatus, PageType
from app.services import pipeline
from app.services.pdf import render_page as real_render_page
from app.services.pipeline import run_catalogue_pipeline
from tests.factories import make_stored_catalogue, settings_with
from tests.ocr_fakes import FunctionOcr, line
from tests.pdf_factory import ImageSpec, PageSpec, build_pdf

SETTINGS = settings_with(min_embedded_image_px=100, thumbnail_max_px=200, ocr_enabled=True)

COVER_TEXT = [
    line("SERVICE PROVIDER FOR COMMERCIAL SPACE", 100, 100, 900, 160),
    line("\u5546\u7528\u7a7a\u95f4\u670d\u52a1\u5546", 100, 200, 500, 250, 0.8),
]
PRODUCT_TEXT = [
    line("\u578b\u53f7: YY-21", 100, 1000, 400, 1040),
    line("3200Wx1400Dx750H", 100, 1050, 500, 1090),
]


def three_page_pdf() -> bytes:
    """Every page has a picture, so no page is blank (blank tiles are never sent to the engine)."""
    return build_pdf(
        [PageSpec(images=[ImageSpec(50, 500, 300, 200, px=160, color=(200, 40, 40))]) for _ in range(3)]
    )


def engine_saying(*pages: list) -> FunctionOcr:  # noqa: ANN401
    """One reading per page, in page order (each A4 page is exactly one tile)."""
    return FunctionOcr(lambda index, _image: list(pages[index]) if index < len(pages) else [])


def run(session, storage, catalogue, job, engine, settings=SETTINGS) -> None:  # noqa: ANN001
    run_catalogue_pipeline(
        session, storage, settings, catalogue_id=catalogue.id, job_id=job.id, ocr_provider=engine
    )


def pages_of(session, catalogue) -> list[CataloguePage]:  # noqa: ANN001
    return list(
        session.scalars(
            select(CataloguePage)
            .where(CataloguePage.catalogue_id == catalogue.id)
            .order_by(CataloguePage.page_number)
        )
    )


def rows_of(session, page) -> list[OCRResult]:  # noqa: ANN001
    return list(
        session.scalars(
            select(OCRResult).where(OCRResult.page_id == page.id).order_by(OCRResult.line_index)
        )
    )


def total_rows(session) -> int:  # noqa: ANN001
    return session.scalar(select(func.count()).select_from(OCRResult)) or 0


# ------------------------------------------------------------------------------ success
def test_text_is_stored_and_pages_are_typed(db_session, storage) -> None:  # noqa: ANN001
    catalogue, job = make_stored_catalogue(db_session, storage, three_page_pdf())
    run(db_session, storage, catalogue, job, engine_saying(COVER_TEXT, PRODUCT_TEXT, []))

    db_session.refresh(catalogue)
    db_session.refresh(job)
    assert catalogue.status is CatalogueStatus.COMPLETED
    assert catalogue.processing_progress == 100 and catalogue.current_stage is None
    assert job.status is JobStatus.SUCCEEDED and job.progress == 100 and job.error is None

    cover, product, empty = pages_of(db_session, catalogue)
    assert [p.status for p in (cover, product, empty)] == [PageStatus.TEXT_READ] * 3
    assert [p.page_type for p in (cover, product, empty)] == [
        PageType.COVER,
        PageType.PRODUCT,
        PageType.UNKNOWN,
    ]
    assert [p.page_type_confidence for p in (cover, product, empty)] == [0.7, 0.9, 0.3]

    first, second = rows_of(db_session, cover)
    assert (first.line_index, first.text, first.language) == (0, COVER_TEXT[0].text, "en")
    assert (second.line_index, second.language) == (1, "zh")
    assert first.confidence == pytest.approx(0.95) and second.confidence == pytest.approx(0.8)
    assert (first.bbox_x0, first.bbox_y0, first.bbox_x1, first.bbox_y1) == (100, 100, 900, 160)
    assert first.polygon == [[100, 100], [900, 100], [900, 160], [100, 160]]
    assert (first.engine, first.engine_version) == ("function", "test")
    assert [r.text for r in rows_of(db_session, product)] == ["\u578b\u53f7: YY-21", "3200Wx1400Dx750H"]
    assert rows_of(db_session, empty) == []


def test_running_again_replaces_the_text_instead_of_piling_it_up(db_session, storage) -> None:  # noqa: ANN001
    catalogue, first_job = make_stored_catalogue(db_session, storage, three_page_pdf())
    run(db_session, storage, catalogue, first_job, engine_saying(COVER_TEXT, PRODUCT_TEXT, []))
    assert total_rows(db_session) == 4

    second_job = ProcessingJob(catalogue_id=catalogue.id, job_type=first_job.job_type)
    db_session.add(second_job)
    db_session.flush()
    run(db_session, storage, catalogue, second_job, engine_saying(COVER_TEXT, [], PRODUCT_TEXT))

    assert total_rows(db_session) == 4  # replaced, not doubled
    _, middle, last = pages_of(db_session, catalogue)
    assert rows_of(db_session, middle) == [] and len(rows_of(db_session, last)) == 2


def test_progress_and_stage_cover_both_stages(db_session, storage) -> None:  # noqa: ANN001
    catalogue, job = make_stored_catalogue(db_session, storage, three_page_pdf())
    seen: list[tuple[str | None, int]] = []

    def spy(_index, _image):  # noqa: ANN001, ANN202
        current = db_session.get(type(catalogue), catalogue.id)
        db_session.refresh(current)
        seen.append((current.current_stage, current.processing_progress))
        return []

    run(db_session, storage, catalogue, job, FunctionOcr(spy))
    assert [stage for stage, _ in seen] == ["read_text"] * 3
    progress = [value for _, value in seen]
    assert progress == sorted(progress) and progress[0] >= 50  # rendering was the first half
    assert progress[-1] < 100  # 100 is reserved for "finished"


# ------------------------------------------------------------------------------ failures
def test_a_page_whose_text_cannot_be_read_stays_usable(db_session, storage) -> None:  # noqa: ANN001
    catalogue, job = make_stored_catalogue(db_session, storage, three_page_pdf())

    def respond(index, _image):  # noqa: ANN001, ANN202
        if index == 1:
            raise RuntimeError("engine exploded")
        return list(COVER_TEXT)

    run(db_session, storage, catalogue, job, FunctionOcr(respond))

    db_session.refresh(catalogue)
    assert catalogue.status is CatalogueStatus.PARTIALLY_COMPLETED
    first, broken, third = pages_of(db_session, catalogue)
    assert first.status is PageStatus.TEXT_READ and third.status is PageStatus.TEXT_READ
    assert broken.status is PageStatus.RENDERED  # its picture is still there for later stages
    assert broken.image_key is not None and broken.page_type is None
    assert broken.error_stage == "read_text" and "engine exploded" in broken.processing_error
    assert rows_of(db_session, broken) == []
    assert len(rows_of(db_session, first)) == 2 and len(rows_of(db_session, third)) == 2


def test_a_failure_while_saving_a_pages_text_leaves_no_half_written_rows(  # noqa: ANN001
    db_session, storage
) -> None:
    # A lone surrogate cannot be encoded for the database, so the INSERT itself fails.
    unsavable = [line("GOOD LINE", 100, 100, 300, 130), line("BAD\ud800LINE", 100, 200, 300, 230)]
    catalogue, job = make_stored_catalogue(db_session, storage, three_page_pdf())
    run(db_session, storage, catalogue, job, engine_saying(COVER_TEXT, unsavable, COVER_TEXT))

    db_session.refresh(catalogue)
    assert catalogue.status is CatalogueStatus.PARTIALLY_COMPLETED
    first, broken, third = pages_of(db_session, catalogue)
    assert broken.status is PageStatus.RENDERED and broken.error_stage == "read_text"
    assert rows_of(db_session, broken) == []  # not even the good line
    assert first.status is PageStatus.TEXT_READ and third.status is PageStatus.TEXT_READ


def test_pages_that_failed_to_render_are_skipped_by_the_text_stage(  # noqa: ANN001
    db_session, storage, monkeypatch
) -> None:
    calls = {"n": 0}

    def flaky(page, scale):  # noqa: ANN001, ANN202
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("renderer exploded")
        return real_render_page(page, scale)

    monkeypatch.setattr(pipeline, "render_page", flaky)
    catalogue, job = make_stored_catalogue(db_session, storage, three_page_pdf())
    engine = engine_saying(COVER_TEXT, COVER_TEXT)
    run(db_session, storage, catalogue, job, engine)

    db_session.refresh(catalogue)
    assert catalogue.status is CatalogueStatus.PARTIALLY_COMPLETED
    first, failed, third = pages_of(db_session, catalogue)
    assert failed.status is PageStatus.FAILED and failed.error_stage == "render_pages"
    assert first.status is PageStatus.TEXT_READ and third.status is PageStatus.TEXT_READ
    assert engine.calls == 2  # the failed page was never offered to the engine


def test_an_engine_that_cannot_start_fails_the_job_before_any_work_is_done(  # noqa: ANN001
    db_session, storage, monkeypatch
) -> None:
    def broken(_settings):  # noqa: ANN001, ANN202
        raise OcrEngineError("secret path /opt/models/missing.onnx")

    monkeypatch.setattr(pipeline, "get_ocr_provider", broken)
    catalogue, job = make_stored_catalogue(db_session, storage, three_page_pdf())
    run_catalogue_pipeline(db_session, storage, SETTINGS, catalogue_id=catalogue.id, job_id=job.id)

    db_session.refresh(catalogue)
    db_session.refresh(job)
    assert catalogue.status is CatalogueStatus.FAILED and job.status is JobStatus.FAILED
    assert "Text recognition" in catalogue.processing_error and "OCR_ENABLED" in job.error
    assert "/opt/models" not in catalogue.processing_error  # engine details stay in the log
    assert pages_of(db_session, catalogue) == []  # failed fast: nothing was rendered


# ------------------------------------------------------------------------------ switched off
def test_text_recognition_can_be_switched_off(db_session, storage) -> None:  # noqa: ANN001
    def must_not_be_called(_index, _image):  # noqa: ANN001, ANN202
        raise AssertionError("the engine must not be used when OCR is disabled")

    catalogue, job = make_stored_catalogue(db_session, storage, three_page_pdf())
    off = settings_with(min_embedded_image_px=100, thumbnail_max_px=200, ocr_enabled=False)
    run(db_session, storage, catalogue, job, FunctionOcr(must_not_be_called), settings=off)

    db_session.refresh(catalogue)
    assert catalogue.status is CatalogueStatus.COMPLETED
    assert all(p.status is PageStatus.RENDERED for p in pages_of(db_session, catalogue))
    assert total_rows(db_session) == 0

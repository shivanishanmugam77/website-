"""The catalogue pipeline, run against real PDFs, a real database and a real file store."""

from __future__ import annotations

import hashlib
import io
import re
import uuid

import pytest
from PIL import Image
from sqlalchemy import select

from app.models import CataloguePage, ProcessingJob, SourceImage
from app.models.enums import CatalogueStatus, JobStatus, PageStatus, SourceImageKind
from app.services import pipeline
from app.services.pdf import render_page as real_render_page
from app.services.pipeline import original_pdf_key, run_catalogue_pipeline
from tests.factories import make_stored_catalogue, settings_with
from tests.pdf_factory import ImageSpec, PageSpec, blank_pages, build_nested_form_pdf, build_pdf

SETTINGS = settings_with(min_embedded_image_px=100, thumbnail_max_px=200)


def catalogue_pdf() -> bytes:
    return build_pdf(
        [
            PageSpec(
                images=[
                    ImageSpec(50, 600, 200, 150, px=150, color=(255, 0, 0)),
                    ImageSpec(300, 100, 250, 250, px=160, color=(0, 0, 255)),
                ]
            ),
            PageSpec(
                images=[ImageSpec(100, 100, 300, 300, px=200, color=(0, 160, 0), codec="jpeg")]
            ),
            PageSpec(),
        ]
    )


def stored_files(storage) -> list[str]:  # noqa: ANN001
    return sorted(
        str(path.relative_to(storage.root)).replace("\\", "/")
        for path in storage.root.rglob("*")
        if path.is_file()
    )


def run(session, storage, catalogue, job, settings=SETTINGS) -> None:  # noqa: ANN001
    run_catalogue_pipeline(
        session, storage, settings, catalogue_id=catalogue.id, job_id=job.id
    )


def pages_of(session, catalogue) -> list[CataloguePage]:  # noqa: ANN001
    return list(
        session.scalars(
            select(CataloguePage)
            .where(CataloguePage.catalogue_id == catalogue.id)
            .order_by(CataloguePage.page_number)
        )
    )


def images_of(session, page) -> list[SourceImage]:  # noqa: ANN001
    return list(session.scalars(select(SourceImage).where(SourceImage.page_id == page.id)))


# ------------------------------------------------------------------------------ success
def test_pages_are_rendered_and_embedded_images_extracted(db_session, storage) -> None:  # noqa: ANN001
    catalogue, job = make_stored_catalogue(db_session, storage, catalogue_pdf())
    run(db_session, storage, catalogue, job)

    db_session.refresh(catalogue)
    db_session.refresh(job)
    assert catalogue.status is CatalogueStatus.COMPLETED
    assert catalogue.page_count == 3
    assert catalogue.processing_progress == 100
    assert catalogue.current_stage is None and catalogue.processing_error is None
    assert job.status is JobStatus.SUCCEEDED
    assert (job.progress, job.pages_processed, job.total_pages) == (100, 3, 3)
    assert job.started_at is not None and job.finished_at is not None and job.error is None

    pages = pages_of(db_session, catalogue)
    assert [p.page_number for p in pages] == [1, 2, 3]
    assert all(p.status is PageStatus.RENDERED for p in pages)

    first = pages[0]
    assert (first.width_px, first.height_px) == (1240, 1755)  # A4 at 150 dpi
    page_jpeg = Image.open(io.BytesIO(storage.read_bytes(first.image_key)))
    assert page_jpeg.size == (first.width_px, first.height_px)
    thumbnail = Image.open(io.BytesIO(storage.read_bytes(first.thumbnail_key)))
    assert max(thumbnail.size) == 200

    def counts(page):  # noqa: ANN001, ANN202
        kinds = [i.kind for i in images_of(db_session, page)]
        return kinds.count(SourceImageKind.PAGE), kinds.count(SourceImageKind.EMBEDDED)

    assert [counts(p) for p in pages] == [(1, 2), (1, 1), (1, 0)]


def test_extracted_images_are_traceable_and_hashed(db_session, storage) -> None:  # noqa: ANN001
    catalogue, job = make_stored_catalogue(db_session, storage, catalogue_pdf())
    run(db_session, storage, catalogue, job)
    first = pages_of(db_session, catalogue)[0]

    page_image = next(i for i in images_of(db_session, first) if i.kind is SourceImageKind.PAGE)
    assert page_image.storage_key == first.image_key  # the page render *is* its source image
    assert page_image.bbox_x0 is None

    embedded = sorted(
        (i for i in images_of(db_session, first) if i.kind is SourceImageKind.EMBEDDED),
        key=lambda i: i.width_px,
    )
    assert [(i.width_px, i.height_px) for i in embedded] == [(150, 150), (160, 160)]  # native size
    for image in embedded:
        assert None not in (image.bbox_x0, image.bbox_y0, image.bbox_x1, image.bbox_y1)
        assert image.bbox_x1 > image.bbox_x0 and image.bbox_y1 > image.bbox_y0
        assert image.sha256 == hashlib.sha256(storage.read_bytes(image.storage_key)).hexdigest()
        assert re.fullmatch(r"[0-9a-f]{16}", image.phash)


def test_only_expected_files_are_stored(db_session, storage) -> None:  # noqa: ANN001
    catalogue, job = make_stored_catalogue(db_session, storage, catalogue_pdf())
    run(db_session, storage, catalogue, job)
    # original + (page + thumbnail) x 3 + 3 embedded images
    assert len(stored_files(storage)) == 1 + 6 + 3
    assert original_pdf_key(catalogue.id) in stored_files(storage)


def test_images_nested_in_forms_are_kept_without_a_position(db_session, storage) -> None:  # noqa: ANN001
    catalogue, job = make_stored_catalogue(db_session, storage, build_nested_form_pdf())
    run(db_session, storage, catalogue, job, settings_with(min_embedded_image_px=8))
    (page,) = pages_of(db_session, catalogue)
    (embedded,) = [i for i in images_of(db_session, page) if i.kind is SourceImageKind.EMBEDDED]
    assert (embedded.width_px, embedded.bbox_x0, embedded.bbox_y1) == (16, None, None)


def test_enormous_pages_are_rendered_within_the_pixel_cap(db_session, storage) -> None:  # noqa: ANN001
    catalogue, job = make_stored_catalogue(
        db_session, storage, blank_pages(1, width=7200, height=7200)  # 100 x 100 inches
    )
    run(db_session, storage, catalogue, job, settings_with(max_page_pixels=1_000_000))
    (page,) = pages_of(db_session, catalogue)
    assert page.status is PageStatus.RENDERED
    assert page.width_px * page.height_px <= 1_000_000 * 1.02


def test_progress_is_reported_as_pages_complete(db_session, storage, monkeypatch) -> None:  # noqa: ANN001
    catalogue, job = make_stored_catalogue(db_session, storage, catalogue_pdf())
    seen: list[int] = []

    def spy(page, scale):  # noqa: ANN001, ANN202
        seen.append(db_session.get(ProcessingJob, job.id).progress)
        return real_render_page(page, scale)

    monkeypatch.setattr(pipeline, "render_page", spy)
    run(db_session, storage, catalogue, job)
    assert seen == [0, 33, 66]  # 100 is reserved for "finished"


# ------------------------------------------------------------------------------ re-runs
def test_running_twice_replaces_results_instead_of_duplicating_them(db_session, storage) -> None:  # noqa: ANN001
    catalogue, first_job = make_stored_catalogue(db_session, storage, catalogue_pdf())
    run(db_session, storage, catalogue, first_job)

    second_job = ProcessingJob(catalogue_id=catalogue.id, job_type=first_job.job_type)
    db_session.add(second_job)
    db_session.flush()
    run(db_session, storage, catalogue, second_job)

    assert len(pages_of(db_session, catalogue)) == 3
    total_images = db_session.scalars(
        select(SourceImage).join(CataloguePage).where(CataloguePage.catalogue_id == catalogue.id)
    ).all()
    assert len(total_images) == 3 + 3  # 3 page images + 3 embedded, not doubled
    assert len(stored_files(storage)) == 1 + 6 + 3  # and no orphaned files from the first run


def test_a_missing_catalogue_is_ignored(db_session, storage) -> None:  # noqa: ANN001
    run_catalogue_pipeline(
        db_session, storage, SETTINGS, catalogue_id=uuid.uuid4(), job_id=uuid.uuid4()
    )  # e.g. deleted while the job waited in the queue: must not raise


# ------------------------------------------------------------------------------ failures
def test_a_corrupt_pdf_fails_cleanly_with_a_readable_reason(db_session, storage) -> None:  # noqa: ANN001
    broken = b"%PDF-1.4 this is not really a pdf"
    catalogue, job = make_stored_catalogue(db_session, storage, broken)
    run(db_session, storage, catalogue, job)
    db_session.refresh(catalogue)
    db_session.refresh(job)
    assert catalogue.status is CatalogueStatus.FAILED
    assert "not a readable PDF" in catalogue.processing_error
    assert job.status is JobStatus.FAILED and job.error == catalogue.processing_error
    assert job.finished_at is not None
    assert pages_of(db_session, catalogue) == []


def test_pdfs_over_the_page_limit_are_rejected(db_session, storage) -> None:  # noqa: ANN001
    catalogue, job = make_stored_catalogue(db_session, storage, catalogue_pdf())
    run(db_session, storage, catalogue, job, settings_with(max_pdf_pages=2))
    db_session.refresh(catalogue)
    assert catalogue.status is CatalogueStatus.FAILED
    assert "3 pages" in catalogue.processing_error and "limit is 2" in catalogue.processing_error


def test_a_missing_stored_file_is_reported(db_session, storage) -> None:  # noqa: ANN001
    catalogue, job = make_stored_catalogue(db_session, storage, catalogue_pdf())
    storage.delete(catalogue.storage_key)
    run(db_session, storage, catalogue, job)
    db_session.refresh(catalogue)
    assert catalogue.status is CatalogueStatus.FAILED
    assert "missing" in catalogue.processing_error


@pytest.mark.parametrize("failure_point", ["render", "after-files-written"])
def test_one_bad_page_does_not_lose_the_catalogue_or_leave_debris(  # noqa: ANN001
    db_session, storage, monkeypatch, failure_point: str
) -> None:
    catalogue, job = make_stored_catalogue(db_session, storage, catalogue_pdf())
    calls = {"n": 0}

    if failure_point == "render":

        def flaky(page, scale):  # noqa: ANN001, ANN202
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("renderer exploded")
            return real_render_page(page, scale)

        monkeypatch.setattr(pipeline, "render_page", flaky)
    else:
        real_iter = pipeline.iter_embedded_images

        def flaky_iter(page, rendered, *, min_px):  # noqa: ANN001, ANN202
            calls["n"] += 1
            if calls["n"] == 2:  # page 2: its page image and thumbnail are already stored
                raise RuntimeError("extractor exploded")
            yield from real_iter(page, rendered, min_px=min_px)

        monkeypatch.setattr(pipeline, "iter_embedded_images", flaky_iter)

    run(db_session, storage, catalogue, job)
    db_session.refresh(catalogue)
    db_session.refresh(job)

    assert catalogue.status is CatalogueStatus.PARTIALLY_COMPLETED
    assert job.status is JobStatus.SUCCEEDED  # the run finished; one page needs attention
    pages = pages_of(db_session, catalogue)
    assert [p.status for p in pages] == [
        PageStatus.RENDERED,
        PageStatus.FAILED,
        PageStatus.RENDERED,
    ]

    bad = pages[1]
    assert bad.error_stage == "render_pages" and "RuntimeError" in bad.processing_error
    assert bad.image_key is None and images_of(db_session, bad) == []
    # Nothing was left behind for the failed page, and the other pages are intact.
    assert not [f for f in stored_files(storage) if "/pages/0002" in f]
    assert len(images_of(db_session, pages[0])) == 3 and len(images_of(db_session, pages[2])) == 1


def test_when_every_page_fails_the_catalogue_fails(db_session, storage, monkeypatch) -> None:  # noqa: ANN001
    catalogue, job = make_stored_catalogue(db_session, storage, catalogue_pdf())

    def always(page, scale):  # noqa: ANN001, ANN202
        raise RuntimeError("nope")

    monkeypatch.setattr(pipeline, "render_page", always)
    run(db_session, storage, catalogue, job)
    db_session.refresh(catalogue)
    assert catalogue.status is CatalogueStatus.FAILED
    assert "None of the pages" in catalogue.processing_error
    assert [p.status for p in pages_of(db_session, catalogue)] == [PageStatus.FAILED] * 3


def test_an_unexpected_crash_is_recorded_without_leaking_details_then_reraised(  # noqa: ANN001
    db_session, storage, monkeypatch
) -> None:
    catalogue, job = make_stored_catalogue(db_session, storage, catalogue_pdf())

    def boom(path):  # noqa: ANN001, ANN202
        raise RuntimeError("secret internal detail: /var/lib/whatever")

    monkeypatch.setattr(pipeline, "open_pdf", boom)
    with pytest.raises(RuntimeError):
        run(db_session, storage, catalogue, job)

    db_session.refresh(catalogue)
    db_session.refresh(job)
    assert catalogue.status is CatalogueStatus.FAILED and job.status is JobStatus.FAILED
    assert "secret internal detail" not in (catalogue.processing_error or "")
    assert "server logs" in catalogue.processing_error

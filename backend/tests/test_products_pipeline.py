"""Stage 4 of the pipeline (finding the products inside the panels) with a real database and
files, and stand-in models.

Each PDF page holds two flat-colour pictures on white paper, so every page has a big panel
(the stand-in finder finds a desk in it) and a small one (where it finds nothing).
"""

from __future__ import annotations

import io

from PIL import Image
from sqlalchemy import select

from app.ml.detection.base import Detection
from app.ml.errors import ModelUnavailableError
from app.models import (
    CataloguePage,
    DetectedObject,
    ProcessingJob,
    SegmentationResult,
    SourceImage,
)
from app.models.enums import CatalogueStatus, JobStatus, PageStatus, SegmentationStatus
from app.services import pipeline
from app.services.detections import parse_prompts
from app.services.pipeline import run_catalogue_pipeline
from tests.factories import make_stored_catalogue, settings_with
from tests.ocr_fakes import FunctionOcr, line
from tests.product_factory import FakeFinder, FakeOutliner
from tests.test_panels_pipeline import pages_of, three_page_pdf

PRODUCTS = settings_with(
    min_embedded_image_px=100,
    thumbnail_max_px=200,
    ocr_enabled=False,
    panel_detection_enabled=True,
    product_detection_enabled=True,
)
DESK_BOX = (50.0, 60.0, 650.0, 760.0)  # inside the big panel (about 729 x 833 pixels)


def desk_in_the_big_panel(_index: int, image: Image.Image) -> list[Detection]:
    return [Detection("desk", 0.8, DESK_BOX)] if image.width > 500 else []


def run(session, storage, catalogue, job, settings=PRODUCTS, models=None, ocr=None) -> None:  # noqa: ANN001
    if models is None:
        models = (FakeFinder(desk_in_the_big_panel), FakeOutliner())
    run_catalogue_pipeline(
        session,
        storage,
        settings,
        catalogue_id=catalogue.id,
        job_id=job.id,
        ocr_provider=ocr,
        product_models=models,
    )


def finds_of(session, catalogue) -> list[tuple[DetectedObject, SegmentationResult]]:  # noqa: ANN001
    rows = session.execute(
        select(DetectedObject, SegmentationResult)
        .select_from(DetectedObject)
        .join(SegmentationResult, SegmentationResult.detected_object_id == DetectedObject.id)
        .join(SourceImage, SourceImage.id == DetectedObject.source_image_id)
        .join(CataloguePage, CataloguePage.id == SourceImage.page_id)
        .where(CataloguePage.catalogue_id == catalogue.id)
        .order_by(CataloguePage.page_number)
    ).all()
    return [(found, outline) for found, outline in rows]


def product_files(storage) -> list[str]:  # noqa: ANN001
    return sorted(
        str(p.relative_to(storage.root))
        for p in (storage.root / "catalogues").rglob("*")
        if p.is_file() and "products" in p.parts
    )


# ------------------------------------------------------------------------------ success
def test_every_find_is_saved_with_its_outline_and_pictures(db_session, storage) -> None:  # noqa: ANN001
    catalogue, job = make_stored_catalogue(db_session, storage, three_page_pdf())
    finder = FakeFinder(desk_in_the_big_panel)
    run(db_session, storage, catalogue, job, models=(finder, FakeOutliner(score=0.9)))

    db_session.refresh(catalogue)
    db_session.refresh(job)
    assert catalogue.status is CatalogueStatus.COMPLETED and catalogue.current_stage is None
    assert job.status is JobStatus.SUCCEEDED and job.progress == 100
    assert all(page.status is PageStatus.COMPLETED for page in pages_of(db_session, catalogue))

    finds = finds_of(db_session, catalogue)
    assert len(finds) == 3  # one desk on each page, none in the small panels
    found, outline = finds[0]
    assert (found.label, found.prompt, found.confidence) == ("desk", "desk", 0.8)
    assert (found.model_name, found.model_version) == ("fake-finder", "fake 1")
    assert (found.bbox_x0, found.bbox_y0, found.bbox_x1, found.bbox_y1) == DESK_BOX
    assert outline.status is SegmentationStatus.SUCCEEDED and outline.confidence == 0.9
    assert outline.error is None
    assert (outline.model_name, outline.model_version) == ("fake-outliner", "fake 1")

    kinds = {
        "mask": (outline.mask_key, "PNG"),
        "crop": (outline.crop_key, "JPEG"),
        "cutout": (outline.cutout_key, "PNG"),
        "white": (outline.white_bg_key, "JPEG"),
        "thumb": (outline.thumbnail_key, "JPEG"),
    }
    for name, (key, image_format) in kinds.items():
        assert key is not None and f"/products/{found.id}/{name}." in key
        assert Image.open(io.BytesIO(storage.read_bytes(key))).format == image_format
    cutout = Image.open(io.BytesIO(storage.read_bytes(outline.cutout_key)))
    assert cutout.mode == "RGBA" and cutout.size == (600, 700)  # the outline, trimmed
    assert len(product_files(storage)) == 15  # 5 pictures for each of the 3 finds


def test_the_finder_is_asked_for_the_configured_things_in_every_panel(db_session, storage) -> None:  # noqa: ANN001
    catalogue, job = make_stored_catalogue(db_session, storage, three_page_pdf())
    finder = FakeFinder(desk_in_the_big_panel)
    run(db_session, storage, catalogue, job, models=(finder, FakeOutliner()))
    assert len(finder.calls) == 6  # 3 pages x 2 panels
    assert all(prompts == parse_prompts(PRODUCTS.detection_prompts) for _size, prompts in finder.calls)


def test_the_stage_and_progress_are_reported_while_it_runs(db_session, storage) -> None:  # noqa: ANN001
    catalogue, job = make_stored_catalogue(db_session, storage, three_page_pdf())
    seen: list[tuple[str | None, int]] = []

    def respond(index: int, image: Image.Image) -> list[Detection]:
        db_session.refresh(catalogue)
        seen.append((catalogue.current_stage, catalogue.processing_progress))
        return desk_in_the_big_panel(index, image)

    run(db_session, storage, catalogue, job, models=(FakeFinder(respond), FakeOutliner()))
    assert [stage for stage, _ in seen] == [pipeline.STAGE_PRODUCTS] * 6
    progress = [value for _stage, value in seen]
    assert progress[0] >= 66 and progress == sorted(progress)  # two of the three stages were done
    db_session.refresh(catalogue)
    assert catalogue.processing_progress == 100 and catalogue.current_stage is None


def test_covers_are_finished_without_looking_for_products(db_session, storage) -> None:  # noqa: ANN001
    catalogue, job = make_stored_catalogue(db_session, storage, three_page_pdf())
    cover_text = [line("SERVICE PROVIDER FOR COMMERCIAL SPACE", 100, 100, 900, 160)]
    product_text = [line("MODEL: YY-11", 100, 1200, 400, 1240), line("2400Wx1200Dx750H", 100, 1250, 500, 1290)]
    engine = FunctionOcr(lambda i, _img: list([cover_text, product_text, product_text][i]))
    finder = FakeFinder(desk_in_the_big_panel)
    settings = settings_with(
        min_embedded_image_px=100, thumbnail_max_px=200, ocr_enabled=True,
        panel_detection_enabled=True, product_detection_enabled=True,
    )  # fmt: skip
    run(db_session, storage, catalogue, job, settings, (finder, FakeOutliner()), engine)

    assert all(page.status is PageStatus.COMPLETED for page in pages_of(db_session, catalogue))
    assert len(finder.calls) == 4  # the cover has no panels, so only two pages were searched
    assert len(finds_of(db_session, catalogue)) == 2


def test_running_again_replaces_the_finds_instead_of_piling_them_up(db_session, storage) -> None:  # noqa: ANN001
    catalogue, first = make_stored_catalogue(db_session, storage, three_page_pdf())
    run(db_session, storage, catalogue, first)
    first_ids = {found.id for found, _ in finds_of(db_session, catalogue)}

    second = ProcessingJob(catalogue_id=catalogue.id, job_type=first.job_type)
    db_session.add(second)
    db_session.flush()
    run(db_session, storage, catalogue, second)

    second_ids = {found.id for found, _ in finds_of(db_session, catalogue)}
    assert len(second_ids) == 3 and not (first_ids & second_ids)
    assert len(product_files(storage)) == 15  # the first run's pictures are gone


# ------------------------------------------------------------------------------ outlines
def test_an_outline_the_model_doubts_is_kept_and_marked(db_session, storage) -> None:  # noqa: ANN001
    catalogue, job = make_stored_catalogue(db_session, storage, three_page_pdf())
    run(db_session, storage, catalogue, job, models=(FakeFinder(desk_in_the_big_panel), FakeOutliner(score=0.5)))
    db_session.refresh(catalogue)
    assert catalogue.status is CatalogueStatus.COMPLETED  # doubt is for a person to look at
    _found, outline = finds_of(db_session, catalogue)[0]
    assert outline.status is SegmentationStatus.LOW_CONFIDENCE and outline.confidence == 0.5
    assert "not sure" in outline.error
    assert outline.cutout_key is not None  # still usable, just to be checked


def test_the_limits_come_from_the_settings(db_session, storage) -> None:  # noqa: ANN001
    catalogue, job = make_stored_catalogue(db_session, storage, three_page_pdf())
    strict = PRODUCTS.model_copy(update={"product_min_outline_score": 0.95})
    run(db_session, storage, catalogue, job, strict)  # the stand-in outliner rates 0.9
    assert {o.status for _f, o in finds_of(db_session, catalogue)} == {SegmentationStatus.LOW_CONFIDENCE}


def test_an_empty_outline_leaves_only_the_crop(db_session, storage) -> None:  # noqa: ANN001
    catalogue, job = make_stored_catalogue(db_session, storage, three_page_pdf())
    run(db_session, storage, catalogue, job, models=(FakeFinder(desk_in_the_big_panel), FakeOutliner(fill=False)))
    _found, outline = finds_of(db_session, catalogue)[0]
    assert outline.status is SegmentationStatus.FAILED and outline.error == "the outline is empty"
    assert outline.crop_key is not None
    assert (outline.mask_key, outline.cutout_key, outline.white_bg_key, outline.thumbnail_key) == (None,) * 4
    assert len(product_files(storage)) == 3  # just the three crops


# ------------------------------------------------------------------------------ failures
def test_a_page_that_fails_keeps_its_panels_and_leaves_nothing_behind(db_session, storage) -> None:  # noqa: ANN001
    catalogue, job = make_stored_catalogue(db_session, storage, three_page_pdf())

    def respond(index: int, image: Image.Image) -> list[Detection]:
        if index == 3:  # page 2, second panel: after its first find has already been saved
            raise RuntimeError("the model crashed")
        return desk_in_the_big_panel(index, image)

    run(db_session, storage, catalogue, job, models=(FakeFinder(respond), FakeOutliner()))

    db_session.refresh(catalogue)
    first, second, third = pages_of(db_session, catalogue)
    assert catalogue.status is CatalogueStatus.PARTIALLY_COMPLETED
    assert (first.status, third.status) == (PageStatus.COMPLETED, PageStatus.COMPLETED)
    assert second.status is PageStatus.PANELS_FOUND  # it keeps what it had
    assert second.error_stage == pipeline.STAGE_PRODUCTS
    assert second.processing_error == "RuntimeError: the model crashed"
    assert len(finds_of(db_session, catalogue)) == 2  # pages 1 and 3
    assert len(product_files(storage)) == 10  # nothing orphaned by the page that failed


def test_missing_models_stop_the_job_before_any_work_is_done(db_session, storage, monkeypatch) -> None:  # noqa: ANN001
    def broken(_settings):  # noqa: ANN001, ANN202
        raise ModelUnavailableError("the model is not downloaded yet")

    monkeypatch.setattr(pipeline, "get_detection_provider", broken)
    catalogue, job = make_stored_catalogue(db_session, storage, three_page_pdf())
    run_catalogue_pipeline(db_session, storage, PRODUCTS, catalogue_id=catalogue.id, job_id=job.id)

    db_session.refresh(catalogue)
    db_session.refresh(job)
    assert catalogue.status is CatalogueStatus.FAILED and job.status is JobStatus.FAILED
    assert "could not be loaded" in catalogue.processing_error
    assert "the model is not downloaded yet" in catalogue.processing_error
    assert pages_of(db_session, catalogue) == []  # nothing was rendered first


# ------------------------------------------------------------------------------ switched off
def test_nothing_is_searched_when_product_finding_is_off(db_session, storage) -> None:  # noqa: ANN001
    catalogue, job = make_stored_catalogue(db_session, storage, three_page_pdf())
    finder = FakeFinder(desk_in_the_big_panel)
    off = PRODUCTS.model_copy(update={"product_detection_enabled": False})
    run(db_session, storage, catalogue, job, off, (finder, FakeOutliner()))

    assert finder.calls == [] and finds_of(db_session, catalogue) == []
    assert all(page.status is PageStatus.PANELS_FOUND for page in pages_of(db_session, catalogue))


def test_products_need_panels_to_be_cut_first(db_session, storage) -> None:  # noqa: ANN001
    catalogue, job = make_stored_catalogue(db_session, storage, three_page_pdf())
    finder = FakeFinder(desk_in_the_big_panel)
    no_panels = PRODUCTS.model_copy(update={"panel_detection_enabled": False})
    run(db_session, storage, catalogue, job, no_panels, (finder, FakeOutliner()))
    assert finder.calls == []
    assert all(page.status is PageStatus.RENDERED for page in pages_of(db_session, catalogue))

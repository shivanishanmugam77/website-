"""Stage 3 of the pipeline (cutting pages into photo panels) with a real database and files.

The PDFs hold two flat-colour pictures on white paper, so each rendered page has a known
layout: a big picture and a smaller one beside it, separated by paper.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image
from sqlalchemy import select

from app.models import Catalogue, CataloguePage, SourceImage
from app.models.enums import CatalogueStatus, PageStatus, PageType, SourceImageKind
from app.services import pipeline
from app.services.pipeline import run_catalogue_pipeline
from tests.factories import make_stored_catalogue, settings_with
from tests.ocr_fakes import FunctionOcr, line
from tests.pdf_factory import ImageSpec, PageSpec, build_pdf

SETTINGS = settings_with(
    min_embedded_image_px=100, thumbnail_max_px=200, ocr_enabled=False, panel_detection_enabled=True
)
WITH_TEXT = settings_with(
    min_embedded_image_px=100, thumbnail_max_px=200, ocr_enabled=True, panel_detection_enabled=True
)

# In pixels at 150 dpi (A4 = 1240 x 1755): a big picture and a smaller one beside it.
BIG = (104, 296, 833, 1129)
SMALL = (875, 754, 1125, 1129)


def layout_page() -> PageSpec:
    return PageSpec(
        images=[
            ImageSpec(50, 300, 350, 400, px=160, color=(200, 60, 40)),  # PDF points, origin bottom-left
            ImageSpec(420, 300, 120, 180, px=160, color=(40, 90, 200)),
        ]
    )


def three_page_pdf() -> bytes:
    return build_pdf([layout_page(), layout_page(), layout_page()])


def run(session, storage, catalogue, job, settings=SETTINGS, ocr=None) -> None:  # noqa: ANN001
    run_catalogue_pipeline(
        session, storage, settings, catalogue_id=catalogue.id, job_id=job.id, ocr_provider=ocr
    )


def pages_of(session, catalogue) -> list[CataloguePage]:  # noqa: ANN001
    return list(
        session.scalars(
            select(CataloguePage)
            .where(CataloguePage.catalogue_id == catalogue.id)
            .order_by(CataloguePage.page_number)
        )
    )


def regions_of(session, page) -> list[SourceImage]:  # noqa: ANN001
    return list(
        session.scalars(
            select(SourceImage)
            .where(SourceImage.page_id == page.id, SourceImage.kind == SourceImageKind.REGION)
            .order_by(SourceImage.bbox_x0)
        )
    )


def region_files(storage) -> list[str]:  # noqa: ANN001
    return sorted(p.name for p in (storage.root / "catalogues").rglob("*.jpg") if "pages" not in p.parts)


def close(got: tuple, want: tuple, tolerance: int = 6) -> bool:
    return all(abs(a - b) <= tolerance for a, b in zip(got, want, strict=True))


# ------------------------------------------------------------------------------ success
def test_every_page_is_cut_into_its_panels(db_session, storage) -> None:  # noqa: ANN001
    catalogue, job = make_stored_catalogue(db_session, storage, three_page_pdf())
    run(db_session, storage, catalogue, job)

    db_session.refresh(catalogue)
    assert catalogue.status is CatalogueStatus.COMPLETED and catalogue.current_stage is None
    pages = pages_of(db_session, catalogue)
    assert all(p.status is PageStatus.PANELS_FOUND for p in pages)

    big, small = regions_of(db_session, pages[0])
    assert close((big.bbox_x0, big.bbox_y0, big.bbox_x1, big.bbox_y1), BIG)
    assert close((small.bbox_x0, small.bbox_y0, small.bbox_x1, small.bbox_y1), SMALL)
    assert (big.width_px, big.height_px) == (
        round(big.bbox_x1 - big.bbox_x0),
        round(big.bbox_y1 - big.bbox_y0),
    )
    assert big.sha256 and big.phash and big.region_confidence is None  # geometric, not scored

    stored = Image.open(io.BytesIO(storage.read_bytes(big.storage_key)))
    assert stored.size == (big.width_px, big.height_px)
    red, _green, _blue = stored.convert("RGB").getpixel((stored.width // 2, stored.height // 2))
    assert red > 150  # the crop really is the red picture
    assert len(region_files(storage)) == 12  # 6 panels + the 6 embedded images


def test_the_page_and_embedded_images_are_kept_alongside_the_panels(db_session, storage) -> None:  # noqa: ANN001
    catalogue, job = make_stored_catalogue(db_session, storage, three_page_pdf())
    run(db_session, storage, catalogue, job)
    kinds = [
        img.kind
        for img in db_session.scalars(
            select(SourceImage).join(CataloguePage).where(CataloguePage.catalogue_id == catalogue.id)
        )
    ]
    assert kinds.count(SourceImageKind.PAGE) == 3
    assert kinds.count(SourceImageKind.EMBEDDED) == 6
    assert kinds.count(SourceImageKind.REGION) == 6


def test_cover_pages_are_not_cut_into_panels(db_session, storage) -> None:  # noqa: ANN001
    catalogue, job = make_stored_catalogue(db_session, storage, three_page_pdf())
    cover_text = [line("SERVICE PROVIDER FOR COMMERCIAL SPACE", 100, 100, 900, 160)]
    product_text = [line("MODEL: YY-11", 100, 1200, 400, 1240), line("2400Wx1200Dx750H", 100, 1250, 500, 1290)]
    engine = FunctionOcr(lambda i, _img: list([cover_text, product_text, product_text][i]))
    run(db_session, storage, catalogue, job, WITH_TEXT, engine)

    cover, product, _third = pages_of(db_session, catalogue)
    assert cover.page_type is PageType.COVER and product.page_type is PageType.PRODUCT
    assert all(p.status is PageStatus.PANELS_FOUND for p in pages_of(db_session, catalogue))
    assert regions_of(db_session, cover) == []  # a cover shows no products
    assert len(regions_of(db_session, product)) == 2


def test_running_again_replaces_the_panels_instead_of_piling_them_up(db_session, storage) -> None:  # noqa: ANN001
    from app.models import ProcessingJob

    catalogue, first = make_stored_catalogue(db_session, storage, three_page_pdf())
    run(db_session, storage, catalogue, first)
    files_after_first = region_files(storage)
    second = ProcessingJob(catalogue_id=catalogue.id, job_type=first.job_type)
    db_session.add(second)
    db_session.flush()
    run(db_session, storage, catalogue, second)

    counts = [len(regions_of(db_session, p)) for p in pages_of(db_session, catalogue)]
    assert counts == [2, 2, 2]
    assert len(region_files(storage)) == len(files_after_first)  # no orphaned files


def test_progress_covers_all_three_stages(db_session, storage, monkeypatch) -> None:  # noqa: ANN001
    catalogue, job = make_stored_catalogue(db_session, storage, three_page_pdf())
    seen: list[tuple[str | None, int]] = []
    real = pipeline.detect_panels

    def spy(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        current = db_session.get(Catalogue, catalogue.id)
        db_session.refresh(current)
        seen.append((current.current_stage, current.processing_progress))
        return real(*args, **kwargs)

    monkeypatch.setattr(pipeline, "detect_panels", spy)
    run(db_session, storage, catalogue, job, WITH_TEXT, FunctionOcr(lambda _i, _img: []))
    assert [stage for stage, _ in seen] == ["find_panels"] * 2  # page 1 is a cover: skipped
    assert seen[0][1] >= 66  # rendering and text reading were the first two thirds
    db_session.refresh(catalogue)
    assert catalogue.processing_progress == 100


# ------------------------------------------------------------------------------ failures
def test_a_page_whose_panels_fail_keeps_its_earlier_results(db_session, storage, monkeypatch) -> None:  # noqa: ANN001
    calls = {"n": 0}
    real = pipeline.detect_panels

    def flaky(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("detector exploded")
        return real(*args, **kwargs)

    monkeypatch.setattr(pipeline, "detect_panels", flaky)
    catalogue, job = make_stored_catalogue(db_session, storage, three_page_pdf())
    run(db_session, storage, catalogue, job)

    db_session.refresh(catalogue)
    assert catalogue.status is CatalogueStatus.PARTIALLY_COMPLETED
    first, broken, third = pages_of(db_session, catalogue)
    assert first.status is PageStatus.PANELS_FOUND and third.status is PageStatus.PANELS_FOUND
    assert broken.status is PageStatus.RENDERED  # no text stage here, so it stays where it was
    assert broken.error_stage == "find_panels" and "detector exploded" in broken.processing_error
    assert regions_of(db_session, broken) == []


def test_a_failure_while_saving_panels_leaves_no_rows_and_no_files(db_session, storage, monkeypatch) -> None:  # noqa: ANN001
    catalogue, job = make_stored_catalogue(db_session, storage, build_pdf([layout_page()]))
    baseline = {"n": 0}
    real = pipeline.dhash

    def failing_on_second_panel(image):  # noqa: ANN001, ANN202
        baseline["n"] += 1
        if baseline["n"] == 5:  # the page image and 2 embedded images come first, then panel 1, 2
            raise RuntimeError("hash exploded")
        return real(image)

    monkeypatch.setattr(pipeline, "dhash", failing_on_second_panel)
    run(db_session, storage, catalogue, job)

    (page,) = pages_of(db_session, catalogue)
    assert page.status is PageStatus.RENDERED and page.error_stage == "find_panels"
    assert regions_of(db_session, page) == []  # not even the first panel
    assert len(region_files(storage)) == 2  # only the two embedded images remain


def test_pages_that_missed_the_text_stage_are_not_carried_forward(db_session, storage) -> None:  # noqa: ANN001
    def respond(index, _image):  # noqa: ANN001, ANN202
        if index == 1:
            raise RuntimeError("engine exploded")
        return []

    catalogue, job = make_stored_catalogue(db_session, storage, three_page_pdf())
    run(db_session, storage, catalogue, job, WITH_TEXT, FunctionOcr(respond))

    db_session.refresh(catalogue)
    assert catalogue.status is CatalogueStatus.PARTIALLY_COMPLETED
    first, skipped, third = pages_of(db_session, catalogue)
    assert first.status is PageStatus.PANELS_FOUND and third.status is PageStatus.PANELS_FOUND
    assert skipped.status is PageStatus.RENDERED and skipped.error_stage == "read_text"
    assert regions_of(db_session, skipped) == []


def test_panel_detection_can_be_switched_off(db_session, storage) -> None:  # noqa: ANN001
    catalogue, job = make_stored_catalogue(db_session, storage, three_page_pdf())
    off = settings_with(
        min_embedded_image_px=100, thumbnail_max_px=200, ocr_enabled=False, panel_detection_enabled=False
    )
    run(db_session, storage, catalogue, job, off)
    db_session.refresh(catalogue)
    assert catalogue.status is CatalogueStatus.COMPLETED
    assert all(p.status is PageStatus.RENDERED for p in pages_of(db_session, catalogue))
    assert all(regions_of(db_session, p) == [] for p in pages_of(db_session, catalogue))


@pytest.mark.parametrize("enabled", [True, False])
def test_the_final_page_status_follows_the_stages_that_are_on(enabled: bool) -> None:
    settings = settings_with(ocr_enabled=False, panel_detection_enabled=enabled)
    expected = PageStatus.PANELS_FOUND if enabled else PageStatus.RENDERED
    assert pipeline._final_status(settings, None) is expected

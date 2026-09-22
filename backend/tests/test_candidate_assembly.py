"""Grouping a catalogue's finds into product candidates, through the real database."""

from __future__ import annotations

import pytest
from sqlalchemy import select

from app.models import DuplicateFlag, ProductCandidate
from app.models.enums import (
    CandidateStatus,
    CatalogueStatus,
    DataOrigin,
    DuplicateSignal,
    DuplicateStatus,
)
from app.services.candidate_assembly import CatalogueNotReadyError, assemble_candidates
from tests.candidate_factory import add_ocr_lines, catalogue_with_candidates
from tests.factories import settings_with
from tests.product_factory import add_find

SETTINGS = settings_with()


def candidates_of(session, catalogue) -> list[ProductCandidate]:  # noqa: ANN001
    return list(session.scalars(select(ProductCandidate).where(ProductCandidate.catalogue_id == catalogue.id)))


# ------------------------------------------------------------------------------ readiness
@pytest.mark.parametrize("status", [CatalogueStatus.UPLOADED, CatalogueStatus.PROCESSING, CatalogueStatus.FAILED])
def test_a_catalogue_that_has_not_finished_processing_is_refused(db_session, storage, status) -> None:  # noqa: ANN001
    catalogue, *_ = catalogue_with_candidates(db_session, storage, catalogue_status=status)
    with pytest.raises(CatalogueNotReadyError):
        assemble_candidates(db_session, SETTINGS, catalogue)
    assert candidates_of(db_session, catalogue) == []


@pytest.mark.parametrize("status", [CatalogueStatus.COMPLETED, CatalogueStatus.PARTIALLY_COMPLETED])
def test_a_finished_or_partly_finished_catalogue_is_accepted(db_session, storage, status) -> None:  # noqa: ANN001
    catalogue, *_ = catalogue_with_candidates(db_session, storage, catalogue_status=status)
    result = assemble_candidates(db_session, SETTINGS, catalogue)
    assert result.candidates_created == 1


# ------------------------------------------------------------------------------ grouping
def test_every_find_on_the_page_joins_one_candidate(db_session, storage) -> None:  # noqa: ANN001
    catalogue, page, big, small, desk, chair = catalogue_with_candidates(db_session, storage)
    result = assemble_candidates(db_session, SETTINGS, catalogue)
    assert (result.candidates_created, result.pages_with_finds) == (1, 1)

    (candidate,) = candidates_of(db_session, catalogue)
    assert {o.id for o in candidate.detected_objects} == {desk.id, chair.id}
    assert [p.id for p in candidate.pages] == [page.id]
    assert sorted(i.id for i in candidate.source_images) == sorted([big.id, small.id])
    assert candidate.origin is DataOrigin.AI
    assert candidate.status in (CandidateStatus.READY, CandidateStatus.NEEDS_REVIEW)


def test_a_page_with_no_finds_gets_no_candidate(db_session, storage) -> None:  # noqa: ANN001
    from tests.factories import make_catalogue, make_page

    catalogue = make_catalogue(db_session, status=CatalogueStatus.COMPLETED)
    make_page(db_session, catalogue, 1)
    result = assemble_candidates(db_session, SETTINGS, catalogue)
    assert result.candidates_created == 0
    assert candidates_of(db_session, catalogue) == []


def test_the_fields_and_scores_match_what_the_pure_grouping_logic_would_say(db_session, storage) -> None:  # noqa: ANN001
    catalogue, *_ = catalogue_with_candidates(db_session, storage)
    assemble_candidates(db_session, SETTINGS, catalogue)
    (candidate,) = candidates_of(db_session, catalogue)
    assert candidate.fields["name"]["value"] == "四人位职员桌"
    assert candidate.fields["model_code"]["value"] == "YY-11"
    assert candidate.fields["dimensions"]["value"] == {
        "width": 2400.0, "depth": 1200.0, "height": 750.0, "unit": "mm",
    }  # fmt: skip
    assert candidate.association_confidence == 1.0  # one model code and one size on the page
    assert candidate.overall_confidence is not None


# ------------------------------------------------------------------------------ rebuilding
def test_running_it_again_replaces_untouched_ai_candidates(db_session, storage) -> None:  # noqa: ANN001
    catalogue, *_ = catalogue_with_candidates(db_session, storage)
    assemble_candidates(db_session, SETTINGS, catalogue)
    (first,) = candidates_of(db_session, catalogue)

    assemble_candidates(db_session, SETTINGS, catalogue)
    (second,) = candidates_of(db_session, catalogue)
    assert second.id != first.id  # rebuilt, not reused


def test_a_reviewed_candidate_is_left_alone(db_session, storage) -> None:  # noqa: ANN001
    catalogue, *_ = catalogue_with_candidates(db_session, storage)
    assemble_candidates(db_session, SETTINGS, catalogue)
    (candidate,) = candidates_of(db_session, catalogue)
    candidate.status = CandidateStatus.APPROVED
    from datetime import datetime, timezone

    candidate.reviewed_at = datetime.now(timezone.utc)
    db_session.flush()

    assemble_candidates(db_session, SETTINGS, catalogue)
    remaining = candidates_of(db_session, catalogue)
    assert [c.id for c in remaining] == [candidate.id]
    assert remaining[0].status is CandidateStatus.APPROVED


def test_a_pages_reviewed_candidate_stops_that_page_being_rebuilt(db_session, storage) -> None:  # noqa: ANN001
    """A page with a kept (reviewed) candidate must not also get a second, fresh one - the
    two would describe the same finds twice."""
    from datetime import datetime, timezone

    catalogue, *_ = catalogue_with_candidates(db_session, storage)
    result = assemble_candidates(db_session, SETTINGS, catalogue)
    assert result.candidates_created == 1
    (candidate,) = candidates_of(db_session, catalogue)
    candidate.status, candidate.reviewed_at = CandidateStatus.REJECTED, datetime.now(timezone.utc)
    db_session.flush()

    result = assemble_candidates(db_session, SETTINGS, catalogue)
    assert result.candidates_created == 0  # the page's only candidate is already decided
    remaining = candidates_of(db_session, catalogue)
    assert [c.id for c in remaining] == [candidate.id]


def test_a_human_entered_candidate_is_left_alone(db_session, storage) -> None:  # noqa: ANN001
    catalogue, *_ = catalogue_with_candidates(db_session, storage)
    manual = ProductCandidate(catalogue_id=catalogue.id, origin=DataOrigin.HUMAN, fields={}, flags=[], reasoning=[])
    db_session.add(manual)
    db_session.flush()

    assemble_candidates(db_session, SETTINGS, catalogue)
    remaining = {c.id for c in candidates_of(db_session, catalogue)}
    assert manual.id in remaining and len(remaining) == 2  # the human one, plus the fresh AI one


# ------------------------------------------------------------------------------ thresholds
def test_thresholds_come_from_the_database_when_an_admin_has_set_them(db_session, storage) -> None:  # noqa: ANN001
    from app.core.constants import SETTING_AI_AUTO_REVIEW_THRESHOLD
    from app.models import AppSetting

    catalogue, *_ = catalogue_with_candidates(db_session, storage)
    db_session.add(AppSetting(key=SETTING_AI_AUTO_REVIEW_THRESHOLD, value=0.0))
    db_session.flush()

    assemble_candidates(db_session, SETTINGS, catalogue)
    (candidate,) = candidates_of(db_session, catalogue)
    assert candidate.status is CandidateStatus.READY  # every score clears a threshold of 0


# ------------------------------------------------------------------------------ duplicate flags
def test_two_pages_with_the_same_model_code_are_flagged_as_duplicates(db_session, storage) -> None:  # noqa: ANN001
    from tests.factories import make_catalogue, make_page
    from tests.page_factory import blank_page
    from tests.test_panel_editing_api import BIG, add_region, jpeg

    catalogue = make_catalogue(db_session, status=CatalogueStatus.COMPLETED)
    for number in (1, 2):
        page = make_page(db_session, catalogue, number)
        image = blank_page(1200, 800)
        page.image_key = f"catalogues/{catalogue.id}/pages/{number}.jpg"
        storage.put_bytes(page.image_key, jpeg(image))
        page.width_px, page.height_px = 1200, 800
        panel = add_region(db_session, storage, catalogue, page, image, BIG)
        add_ocr_lines(db_session, page)
        add_find(db_session, storage, catalogue, panel, label="desk")
    db_session.flush()

    result = assemble_candidates(db_session, SETTINGS, catalogue)
    assert result.candidates_created == 2
    assert result.duplicate_flags_created == 1

    flags = list(db_session.scalars(select(DuplicateFlag)))
    assert len(flags) == 1
    assert flags[0].signal is DuplicateSignal.SKU and flags[0].status is DuplicateStatus.OPEN
    assert flags[0].details == {"model_code": "YY-11"}


def test_different_model_codes_are_not_flagged(db_session, storage) -> None:  # noqa: ANN001
    catalogue, *_ = catalogue_with_candidates(db_session, storage)  # only one page: nothing to compare
    result = assemble_candidates(db_session, SETTINGS, catalogue)
    assert result.duplicate_flags_created == 0
    assert list(db_session.scalars(select(DuplicateFlag))) == []


def test_pages_with_no_model_code_are_never_flagged_against_each_other(db_session, storage) -> None:  # noqa: ANN001
    catalogue, page, big, small, desk, chair = catalogue_with_candidates(db_session, storage, lines=[])
    result = assemble_candidates(db_session, SETTINGS, catalogue)
    assert result.duplicate_flags_created == 0

"""Grouping the finds on a page into one candidate, and scoring how sure we are of it.

The page-8 fixture below mirrors the real BOGAO page: a blank run of lines, then a labelled
model code ("型号：YY-11") and a size line that also carries the product's name
("四人位职员桌2400Wx1200D×750H").
"""

from __future__ import annotations

import uuid

import pytest

from app.models.enums import CandidateStatus, SegmentationStatus
from app.services.candidates import (
    WEIGHT_ASSOCIATION,
    WEIGHT_DETECTION,
    WEIGHT_METADATA,
    WEIGHT_OCR,
    WEIGHT_SEGMENTATION,
    Find,
    build_page_candidate,
    pick_name,
    suggest_category,
)
from app.services.text_signals import Dimension, SeriesLabel, extract_signals

OK, LOW, FAILED = SegmentationStatus.SUCCEEDED, SegmentationStatus.LOW_CONFIDENCE, SegmentationStatus.FAILED
PAGE_8_LINES = [""] * 11 + ["型号：YY-11", "四人位职员桌2400Wx1200D×750H"]
THRESHOLDS = {"auto_review_threshold": 0.85, "low_confidence_threshold": 0.60}


def find(label: str, confidence: float = 0.8, status=OK, score: float | None = 0.9, panel=None) -> Find:
    return Find(uuid.uuid4(), panel or uuid.uuid4(), label, confidence, status, score)


# ------------------------------------------------------------------------------ weights
def test_the_scoring_weights_add_up_to_one() -> None:
    total = WEIGHT_DETECTION + WEIGHT_SEGMENTATION + WEIGHT_OCR + WEIGHT_ASSOCIATION + WEIGHT_METADATA
    assert total == pytest.approx(1.0)


# ------------------------------------------------------------------------------ pick_name
def test_the_name_is_the_size_line_with_the_size_removed() -> None:
    signals = extract_signals(PAGE_8_LINES)
    name = pick_name(PAGE_8_LINES, signals.dimensions[0], None)
    assert name == "四人位职员桌"


def test_a_size_line_that_is_only_the_size_gives_no_name() -> None:
    dimension = extract_signals(["2400Wx1200Dx750H"]).dimensions[0]
    assert pick_name(["2400Wx1200Dx750H"], dimension, None) is None


def test_a_series_label_is_used_when_there_is_no_size() -> None:
    series = SeriesLabel(label="09 series", line_index=0)
    assert pick_name(["09 series"], None, series) == "09 series"


def test_the_series_labels_own_clean_value_is_used_not_the_raw_line_it_came_from() -> None:
    # a real bug: the matched line was a whole marketing sentence, not a short label
    sentence = "09系列采用仿生设计手法，设计元素来源大树根深叶茂的形态，"
    series = SeriesLabel(label="09 series", line_index=0)
    assert pick_name([sentence], None, series) == "09 series"


def test_a_name_far_longer_than_a_real_name_is_dropped() -> None:
    long_line = "x" * (41)
    dimension = extract_signals([f"{long_line}2400x1200x750"]).dimensions[0]
    assert pick_name([f"{long_line}2400x1200x750"], dimension, None) is None


def test_a_name_right_at_the_length_limit_is_kept() -> None:
    exactly_forty = "x" * 40
    dimension = extract_signals([f"{exactly_forty}2400x1200x750"]).dimensions[0]
    assert pick_name([f"{exactly_forty}2400x1200x750"], dimension, None) == exactly_forty


def test_a_name_from_the_size_line_wins_over_a_series_label() -> None:
    dimension = extract_signals(["Chair2400x1200x750"]).dimensions[0]
    series = SeriesLabel(label="09 series", line_index=5)
    assert pick_name(["Chair2400x1200x750"], dimension, series) == "Chair"


def test_no_size_and_no_series_gives_no_name() -> None:
    assert pick_name(["some other text"], None, None) is None


def test_an_out_of_range_line_index_is_ignored_safely() -> None:
    dimension = Dimension(2400, 1200, 750, None, "mm", True, "2400x1200x750", line_index=99)
    assert pick_name(["short list"], dimension, None) is None


# ------------------------------------------------------------------------------ suggest_category
def test_the_majority_label_wins_the_category() -> None:
    assert suggest_category(["desk", "desk", "office chair"]) == "workstations"


def test_a_tie_goes_to_the_first_in_reading_order() -> None:
    assert suggest_category(["office chair", "desk"]) == "office-chairs"


def test_matching_is_not_case_sensitive() -> None:
    assert suggest_category(["Desk", "DESK"]) == "workstations"


def test_unrecognised_labels_suggest_nothing() -> None:
    assert suggest_category(["a mysterious object"]) is None


def test_no_labels_suggest_nothing() -> None:
    assert suggest_category([]) is None


# ------------------------------------------------------------------------------ build_page_candidate: shape
def test_no_finds_means_no_candidate() -> None:
    assert build_page_candidate([], PAGE_8_LINES, **THRESHOLDS) is None


def test_a_clean_page_makes_one_candidate_covering_every_find() -> None:
    panel = uuid.uuid4()
    finds = [find("cabinet", panel=panel), find("office chair", panel=panel), find("desk", panel=panel)]
    draft = build_page_candidate(finds, PAGE_8_LINES, **THRESHOLDS)
    assert draft is not None
    assert set(draft.detected_object_ids) == {f.id for f in finds}
    assert draft.panel_ids == [panel]


def test_finds_from_several_panels_are_all_kept_and_panels_listed_once_each() -> None:
    p1, p2 = uuid.uuid4(), uuid.uuid4()
    finds = [find("desk", panel=p1), find("desk", panel=p1), find("office chair", panel=p2)]
    draft = build_page_candidate(finds, PAGE_8_LINES, **THRESHOLDS)
    assert sorted(draft.panel_ids, key=str) == sorted([p1, p2], key=str)


# ------------------------------------------------------------------------------ fields
def test_fields_carry_the_name_code_size_and_labels() -> None:
    draft = build_page_candidate([find("desk")], PAGE_8_LINES, **THRESHOLDS)
    assert draft.fields["name"] == {"value": "四人位职员桌", "source": "OCR", "confidence": None}
    assert draft.fields["model_code"] == {"value": "YY-11", "source": "OCR", "confidence": None}
    assert draft.fields["dimensions"]["value"] == {"width": 2400.0, "depth": 1200.0, "height": 750.0, "unit": "mm"}
    assert draft.fields["labels"]["value"] == ["desk"]
    assert draft.fields["category_suggestion"]["value"] == "workstations"
    assert draft.fields["series"] is None


def test_missing_facts_leave_their_fields_as_none_not_a_placeholder() -> None:
    draft = build_page_candidate([find("a mysterious object")], [], **THRESHOLDS)
    assert draft.fields["name"] is None
    assert draft.fields["model_code"] is None
    assert draft.fields["dimensions"] is None
    assert draft.fields["category_suggestion"] is None
    assert draft.fields["labels"]["value"] == ["a mysterious object"]


def test_labels_in_fields_are_sorted_and_deduplicated() -> None:
    draft = build_page_candidate([find("desk"), find("chair"), find("desk")], [], **THRESHOLDS)
    assert draft.fields["labels"]["value"] == ["chair", "desk"]


def test_labels_are_in_true_alphabetical_order_not_incidental_order() -> None:
    finds = [find("zebra chair"), find("mango table"), find("apple desk")]
    draft = build_page_candidate(finds, [], **THRESHOLDS)
    assert draft.fields["labels"]["value"] == ["apple desk", "mango table", "zebra chair"]


def test_metadata_confidence_is_the_share_of_three_facts_present() -> None:
    draft = build_page_candidate([find("desk")], PAGE_8_LINES, **THRESHOLDS)  # name + code + size
    assert draft.metadata_confidence == pytest.approx(1.0)
    partial = build_page_candidate([find("desk")], ["Model: YY-11"], **THRESHOLDS)  # code only
    assert partial.metadata_confidence == pytest.approx(1 / 3)
    none = build_page_candidate([find("desk")], [], **THRESHOLDS)
    assert none.metadata_confidence == pytest.approx(0.0)


# ------------------------------------------------------------------------------ flags
def test_a_clean_single_product_page_raises_no_flags() -> None:
    draft = build_page_candidate([find("desk")], PAGE_8_LINES, **THRESHOLDS)
    assert draft.flags == []


def test_two_model_codes_on_one_page_is_flagged() -> None:
    lines = ["Model: AA-1", "Model: BB-2"]
    draft = build_page_candidate([find("desk")], lines, **THRESHOLDS)
    assert "MULTIPLE_MODEL_CODES_ON_PAGE" in draft.flags


def test_two_sizes_on_one_page_is_flagged() -> None:
    lines = ["1200x600x750", "1400x700x800"]
    draft = build_page_candidate([find("desk")], lines, **THRESHOLDS)
    assert "MULTIPLE_DIMENSIONS_ON_PAGE" in draft.flags


def test_no_model_code_or_dimensions_is_flagged() -> None:
    draft = build_page_candidate([find("desk")], [], **THRESHOLDS)
    assert "NO_MODEL_CODE" in draft.flags and "NO_DIMENSIONS" in draft.flags


def test_a_failed_outline_is_flagged() -> None:
    draft = build_page_candidate([find("desk", status=FAILED, score=None)], PAGE_8_LINES, **THRESHOLDS)
    assert "SOME_OUTLINES_FAILED" in draft.flags


def test_a_low_confidence_outline_is_flagged() -> None:
    draft = build_page_candidate([find("desk", status=LOW, score=0.4)], PAGE_8_LINES, **THRESHOLDS)
    assert "SOME_OUTLINES_LOW_CONFIDENCE" in draft.flags
    assert "SOME_OUTLINES_FAILED" not in draft.flags


def test_a_good_outline_among_others_does_not_trip_the_outline_flags() -> None:
    draft = build_page_candidate([find("desk"), find("chair")], PAGE_8_LINES, **THRESHOLDS)
    assert "SOME_OUTLINES_FAILED" not in draft.flags
    assert "SOME_OUTLINES_LOW_CONFIDENCE" not in draft.flags


# ------------------------------------------------------------------------------ scoring and status
def test_everything_good_scores_high_and_is_ready() -> None:
    finds = [find("desk", confidence=0.9, score=0.95), find("office chair", confidence=0.9, score=0.95)]
    draft = build_page_candidate(finds, PAGE_8_LINES, **THRESHOLDS)
    assert draft.overall_confidence >= THRESHOLDS["auto_review_threshold"]
    assert draft.status is CandidateStatus.READY
    assert "LOW_CONFIDENCE" not in draft.flags


def test_nothing_to_go_on_scores_low_and_needs_review() -> None:
    draft = build_page_candidate([find("a mysterious object", confidence=0.3, score=0.3)], [], **THRESHOLDS)
    assert draft.overall_confidence < THRESHOLDS["low_confidence_threshold"]
    assert draft.status is CandidateStatus.NEEDS_REVIEW
    assert "LOW_CONFIDENCE" in draft.flags


def test_a_failed_outline_counts_as_zero_even_if_a_score_is_on_file() -> None:
    # the model was confident about an outline that turned out unusable - it must not help the score
    without = build_page_candidate([find("desk", status=OK, score=0.9)], PAGE_8_LINES, **THRESHOLDS)
    failed = build_page_candidate([find("desk", status=FAILED, score=0.9)], PAGE_8_LINES, **THRESHOLDS)
    assert failed.segmentation_confidence == 0.0
    assert failed.overall_confidence < without.overall_confidence


def test_a_score_exactly_at_the_ready_threshold_is_ready() -> None:
    loose = build_page_candidate([find("desk")], PAGE_8_LINES, auto_review_threshold=0.0, low_confidence_threshold=0.0)
    exact = build_page_candidate(
        [find("desk")], PAGE_8_LINES,
        auto_review_threshold=loose.overall_confidence, low_confidence_threshold=0.0,
    )
    assert exact.status is CandidateStatus.READY


def test_a_score_exactly_at_the_low_confidence_threshold_is_not_flagged() -> None:
    loose = build_page_candidate([find("desk")], PAGE_8_LINES, auto_review_threshold=1.0, low_confidence_threshold=0.0)
    exact = build_page_candidate(
        [find("desk")], PAGE_8_LINES,
        auto_review_threshold=1.0, low_confidence_threshold=loose.overall_confidence,
    )
    assert "LOW_CONFIDENCE" not in exact.flags


# ------------------------------------------------------------------------------ association confidence
def test_a_model_code_and_a_size_together_give_full_association_confidence() -> None:
    draft = build_page_candidate([find("desk")], PAGE_8_LINES, **THRESHOLDS)
    assert draft.association_confidence == 1.0


def test_only_one_of_code_or_size_gives_partial_association_confidence() -> None:
    draft = build_page_candidate([find("desk")], ["Model: YY-11"], **THRESHOLDS)
    assert draft.association_confidence == 0.6


def test_neither_code_nor_size_gives_the_weakest_association_confidence() -> None:
    draft = build_page_candidate([find("desk")], [], **THRESHOLDS)
    assert draft.association_confidence == 0.3


def test_multiple_codes_on_the_page_halve_association_confidence() -> None:
    lines = ["Model: AA-1", "Model: BB-2", "2400x1200x750"]
    draft = build_page_candidate([find("desk")], lines, **THRESHOLDS)
    assert draft.association_confidence == pytest.approx(0.5)  # 1.0 (code + size) halved


# ------------------------------------------------------------------------------ ocr confidence
def test_ocr_confidence_blends_in_the_real_line_confidences() -> None:
    confidences = [None] * 11 + [0.5, 1.0]  # low confidence on the model-code line only
    draft = build_page_candidate([find("desk")], PAGE_8_LINES, confidences, **THRESHOLDS)
    assert draft.ocr_confidence == pytest.approx(0.4 * 0.5 + 0.4 * 1.0 + 0.2 * 1.0)


def test_missing_line_confidences_default_to_full_confidence() -> None:
    draft = build_page_candidate([find("desk")], PAGE_8_LINES, **THRESHOLDS)
    assert draft.ocr_confidence == pytest.approx(1.0)


def test_a_recorded_but_unknown_line_confidence_also_defaults_to_full_confidence() -> None:
    confidences = [None] * len(PAGE_8_LINES)  # OCR ran but did not report a confidence
    draft = build_page_candidate([find("desk")], PAGE_8_LINES, confidences, **THRESHOLDS)
    assert draft.ocr_confidence == pytest.approx(1.0)


def test_ocr_confidence_only_counts_facts_that_were_actually_found() -> None:
    draft = build_page_candidate([find("desk")], [], **THRESHOLDS)
    assert draft.ocr_confidence == 0.0


# ------------------------------------------------------------------------------ reasoning
def test_reasoning_mentions_the_count_and_the_model_code() -> None:
    draft = build_page_candidate([find("desk"), find("office chair")], PAGE_8_LINES, **THRESHOLDS)
    assert any("2 item(s)" in line for line in draft.reasoning)
    assert any("YY-11" in line for line in draft.reasoning)

"""Tidying a finder's answer: wording, boxes, duplicates (no model needed)."""

from __future__ import annotations

import pytest

from app.ml.detection.base import Detection
from app.services.detections import (
    MAX_PROMPT_CHARS,
    MAX_PROMPTS,
    box_area,
    clamp_box,
    drop_tiny,
    format_prompt,
    iou,
    match_prompt,
    parse_prompts,
    suppress_overlaps,
    tidy,
    to_detections,
)

PROMPTS = ["desk", "office chair", "chair", "meeting table", "table"]


# ------------------------------------------------------------------------------ the request
def test_prompts_are_split_lowercased_and_deduplicated() -> None:
    assert parse_prompts("Desk, Office  chair. cabinet\nDESK; sofa") == [
        "desk",
        "office chair",
        "cabinet",
        "sofa",
    ]


@pytest.mark.parametrize("text", ["", " , . ; ", "x" * (MAX_PROMPT_CHARS + 1)])
def test_bad_prompt_lists_are_refused(text: str) -> None:
    with pytest.raises(ValueError):
        parse_prompts(text)


def test_too_many_prompts_are_refused() -> None:
    with pytest.raises(ValueError):
        parse_prompts(", ".join(f"thing{n}" for n in range(MAX_PROMPTS + 1)))
    assert len(parse_prompts(", ".join(f"thing{n}" for n in range(MAX_PROMPTS)))) == MAX_PROMPTS


def test_the_prompt_has_the_form_the_model_expects() -> None:
    assert format_prompt(["desk", "office chair"]) == "desk. office chair."


# ------------------------------------------------------------------------------ wording
@pytest.mark.parametrize(
    ("wording", "expected"),
    [
        ("office chair", "office chair"),  # exact
        ("chair", "chair"),  # exact, even though "office chair" contains it
        ("table.", "table"),  # trailing full stop
        ("Desk", "desk"),
        ("office", "office chair"),  # part of a phrase
        ("meeting", "meeting table"),
        ("desk chair", "desk"),  # two phrases run together: the first full match with most words
        ("cat", None),  # nothing asked for
        ("", None),
        ("   ", None),
    ],
)
def test_the_models_wording_is_mapped_back_to_a_requested_phrase(
    wording: str, expected: str | None
) -> None:
    assert match_prompt(wording, PROMPTS) == expected


# ------------------------------------------------------------------------------ boxes
def test_box_arithmetic() -> None:
    assert box_area((0, 0, 10, 20)) == 200 and box_area((10, 10, 5, 5)) == 0
    assert clamp_box((-5, -5, 120, 90), 100, 80) == (0, 0, 100, 80)
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_model_answers_become_detections_in_the_original_pictures_pixels() -> None:
    raw = [
        ([10, 20, 110, 220], 0.9, "office chair"),
        ([0, 0, 50, 50], 0.8, "cat"),  # not asked for
        ([30, 30, 30, 90], 0.7, "desk"),  # no width
        ([-40, 5, 60, 500], 0.6, "desk"),  # sticks out of the picture
    ]
    found = to_detections(raw, PROMPTS, scale=2.0, image_size=(400, 600))
    assert [d.label for d in found] == ["office chair", "desk"]
    assert found[0].box == (20, 40, 220, 440)  # doubled: the model saw a half-size picture
    assert found[1].box == (0, 10, 120, 600)  # doubled, then cut to the picture
    assert found[0].confidence == 0.9


def test_confidence_is_kept_between_zero_and_one() -> None:
    (found,) = to_detections([([0, 0, 10, 10], 1.7, "desk")], PROMPTS, scale=1.0, image_size=(50, 50))
    assert found.confidence == 1.0


# ------------------------------------------------------------------------------ tidying
def det(label: str, confidence: float, box: tuple) -> Detection:
    return Detection(label, confidence, box)


def test_specks_are_dropped() -> None:
    big = det("desk", 0.9, (0, 0, 300, 200))
    thin = det("desk", 0.9, (0, 0, 20, 600))  # big enough in area, but only 20 px wide
    small = det("chair", 0.9, (0, 0, 30, 30))  # under 1 % of a 1000 x 1000 picture
    assert drop_tiny([big, thin, small], (1000, 1000)) == [big]


def test_the_same_object_reported_twice_is_kept_once_with_the_best_score() -> None:
    a = det("desk", 0.6, (100, 100, 400, 300))
    b = det("desk", 0.9, (110, 105, 410, 305))
    assert suppress_overlaps([a, b]) == [b]


def test_two_separate_objects_of_one_kind_are_both_kept() -> None:
    left = det("office chair", 0.8, (0, 0, 100, 100))
    right = det("office chair", 0.7, (200, 0, 300, 100))
    assert suppress_overlaps([left, right]) == [left, right]


def test_different_names_for_one_object_are_merged_only_when_the_boxes_nearly_coincide() -> None:
    desk = det("desk", 0.9, (100, 100, 500, 300))
    table = det("table", 0.7, (102, 101, 498, 299))  # same furniture, other word
    assert suppress_overlaps([desk, table]) == [desk]

    chair = det("chair", 0.8, (300, 150, 500, 320))  # a chair pushed under the desk
    assert suppress_overlaps([desk, chair]) == [desk, chair]

    shelf = det("bookshelf", 0.8, (200, 100, 500, 300))  # overlaps the desk by 75 %: still two things
    assert suppress_overlaps([desk, shelf]) == [desk, shelf]


def test_tidy_cuts_filters_merges_and_orders_the_finds() -> None:
    finds = [
        det("cabinet", 0.95, (600, 500, 900, 900)),  # the most confident, but last to read
        det("desk", 0.6, (100, 100, 400, 300)),
        det("desk", 0.9, (105, 100, 405, 300)),  # duplicate of the one above
        det("chair", 0.8, (-50, 350, 200, 5000)),  # sticks out of the 1000 x 1000 picture
        det("mug", 0.9, (10, 10, 30, 30)),  # a speck
    ]
    result = tidy(finds, (1000, 1000))
    assert [(d.label, d.box) for d in result] == [
        ("desk", (105, 100, 405, 300)),
        ("chair", (0, 350, 200, 1000)),
        ("cabinet", (600, 500, 900, 900)),
    ]

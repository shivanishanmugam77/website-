"""The rules for a hand-drawn panel box."""

from __future__ import annotations

import pytest

from app.services.panel_boxes import (
    MIN_SIDE_PX,
    InvalidBoxError,
    PanelEditError,
    validate_box,
)

W, H = 1200, 800


def test_a_good_box_becomes_whole_pixels() -> None:
    assert validate_box([100.4, 50.6, 600.5, 400.2], W, H) == (100, 51, 600, 400)


def test_a_box_is_clamped_to_the_page() -> None:
    assert validate_box([-50, -20, 300, 300], W, H) == (0, 0, 300, 300)
    assert validate_box([900, 500, 5000, 5000], W, H) == (900, 500, W, H)


@pytest.mark.parametrize(
    "box",
    [
        [10, 10, 10, 200],  # no width
        [10, 10, 200, 10],  # no height
        [200, 10, 10, 200],  # right of left
        [10, 200, 200, 10],  # bottom above top
        [10, 10, 10 + MIN_SIDE_PX - 1, 200],  # too narrow
        [10, 10, 200, 10 + MIN_SIDE_PX - 1],  # too short
        [-500, -500, 5, 5],  # ends up too small once clamped
        [W + 10, H + 10, W + 500, H + 500],  # entirely off the page
    ],
)
def test_impossible_or_tiny_boxes_are_refused(box: list[float]) -> None:
    with pytest.raises(InvalidBoxError):
        validate_box(box, W, H)


@pytest.mark.parametrize(
    "box",
    [[], [1, 2, 3], [1, 2, 3, 4, 5], ["a", 2, 300, 400], [None, 2, 300, 400], [float("nan"), 0, 300, 400], [0, 0, float("inf"), 400], [True, 0, 300, 400]],
)
def test_things_that_are_not_four_numbers_are_refused(box: list) -> None:
    with pytest.raises(InvalidBoxError):
        validate_box(box, W, H)


def test_the_smallest_allowed_box_is_accepted() -> None:
    assert validate_box([0, 0, MIN_SIDE_PX, MIN_SIDE_PX], W, H) == (0, 0, MIN_SIDE_PX, MIN_SIDE_PX)


def test_every_refusal_is_a_panel_edit_error() -> None:
    assert issubclass(InvalidBoxError, PanelEditError)

"""Choosing among the outlines the model offers, and measuring how much of a box one fills."""

from __future__ import annotations

import numpy as np
import pytest

from app.services.masks import SCORE_TOLERANCE, choose_mask, clip_to_box, coverage


def filled(height: int, width: int, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


BOX = (10, 10, 90, 50)  # 80 x 40 = 3200 pixels, inside a 100 x 60 picture


# ------------------------------------------------------------------------------ clip_to_box
def test_everything_outside_the_box_is_removed() -> None:
    clipped = clip_to_box(np.ones((60, 100), dtype=bool), BOX)
    assert int(clipped.sum()) == 3200
    assert clipped[10:50, 10:90].all() and not clipped[:10].any() and not clipped[:, 90:].any()


def test_a_box_hanging_over_the_picture_edge_is_clamped() -> None:
    clipped = clip_to_box(np.ones((60, 100), dtype=bool), (-20, -5, 30, 20))
    assert int(clipped.sum()) == 30 * 20 and clipped[:20, :30].all()


def test_fractional_box_edges_keep_the_whole_pixels_they_touch() -> None:
    clipped = clip_to_box(np.ones((60, 100), dtype=bool), (10.6, 10.6, 20.2, 20.2))
    assert int(clipped.sum()) == 11 * 11  # x and y from 10 to 21 (exclusive)


@pytest.mark.parametrize("box", [(50, 20, 50, 40), (60, 20, 40, 40), (200, 200, 300, 300), (-50, -50, -10, -10), (-50, 10, -10, 40), (10, -50, 40, -10)])
def test_an_empty_or_outside_box_leaves_nothing(box) -> None:  # noqa: ANN001
    assert not clip_to_box(np.ones((60, 100), dtype=bool), box).any()


def test_the_input_outline_is_not_changed() -> None:
    mask = np.ones((60, 100), dtype=bool)
    clip_to_box(mask, BOX)
    assert mask.all()


# ------------------------------------------------------------------------------ choose_mask
def test_the_more_complete_outline_wins_when_the_model_rates_it_nearly_as_well() -> None:
    near_half = filled(60, 100, 10, 10, 50, 50)
    whole = filled(60, 100, 10, 10, 90, 50)
    index, mask = choose_mask([near_half, whole], [0.95, 0.95 - SCORE_TOLERANCE], BOX)
    assert index == 1 and int(mask.sum()) == 3200


def test_a_much_lower_rated_bigger_outline_is_not_used() -> None:
    small = filled(60, 100, 10, 10, 50, 50)
    huge = filled(60, 100, 10, 10, 90, 50)
    index, _ = choose_mask([small, huge], [0.95, 0.95 - SCORE_TOLERANCE - 0.01], BOX)
    assert index == 0


def test_only_the_part_inside_the_box_counts_when_comparing_sizes() -> None:
    inside = filled(60, 100, 10, 10, 60, 50)  # 50 x 40 inside the box
    leaks = filled(60, 100, 0, 0, 100, 60)  # everything, but only 80 x 40 of it is inside
    index, mask = choose_mask([inside, leaks], [0.9, 0.9], BOX)
    assert index == 1 and int(mask.sum()) == 3200 and not mask[:10].any()  # clipped to the box


def test_equal_outlines_go_to_the_higher_score() -> None:
    same = filled(60, 100, 10, 10, 90, 50)
    index, _ = choose_mask([same, same.copy(), same.copy()], [0.86, 0.93, 0.90], BOX)
    assert index == 1


def test_a_single_outline_is_used_as_it_is() -> None:
    only = filled(60, 100, 20, 20, 30, 30)
    index, mask = choose_mask([only], [0.4], BOX)
    assert index == 0 and int(mask.sum()) == 100


def test_the_best_score_defines_who_is_eligible() -> None:
    parts = [filled(60, 100, 10, 10, 30, 50), filled(60, 100, 10, 10, 60, 50), filled(60, 100, 10, 10, 90, 50)]
    index, _ = choose_mask(parts, [0.90, 0.97, 0.80], BOX)
    assert index == 1  # 0.80 is more than the tolerance below 0.97, so the biggest is out


@pytest.mark.parametrize("candidates,scores", [([], []), ([np.ones((6, 10), dtype=bool)], [0.9, 0.8])])
def test_mismatched_or_missing_candidates_are_an_error(candidates, scores) -> None:  # noqa: ANN001
    with pytest.raises(ValueError, match="one score for each"):
        choose_mask(candidates, scores, (0, 0, 5, 5))


# ------------------------------------------------------------------------------ coverage
def test_a_full_outline_fills_the_whole_box() -> None:
    assert coverage(filled(60, 100, 10, 10, 90, 50), BOX) == 1.0


def test_half_an_outline_fills_half_the_box() -> None:
    assert coverage(filled(60, 100, 10, 10, 50, 50), BOX) == pytest.approx(0.5)


def test_an_outline_elsewhere_fills_nothing_of_the_box() -> None:
    assert coverage(filled(60, 100, 92, 52, 99, 59), BOX) == 0.0


def test_only_the_part_inside_the_box_is_measured() -> None:
    assert coverage(np.ones((60, 100), dtype=bool), BOX) == 1.0


@pytest.mark.parametrize("box", [(5, 5, 5, 20), (30, 30, 10, 10)])
def test_a_box_without_area_has_no_coverage(box) -> None:  # noqa: ANN001
    assert coverage(np.ones((60, 100), dtype=bool), box) == 0.0


def test_a_box_between_pixels_never_reports_more_than_full() -> None:
    assert coverage(np.ones((60, 100), dtype=bool), (10.4, 10.4, 20.2, 20.2)) == 1.0

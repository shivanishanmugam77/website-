"""Finding products in a panel and preparing their pictures, with stand-in models."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from app.ml.detection.base import Detection
from app.ml.segmentation.base import Segment
from app.models.enums import SegmentationStatus
from app.services.product_finding import Found, find_products, render_find

SIZE = (200, 100)
BOX = (40.0, 20.0, 160.0, 80.0)  # 120 x 60 = 7200 pixels
PROMPTS = ["desk", "office chair"]
OK, LOW, FAILED = (
    SegmentationStatus.SUCCEEDED,
    SegmentationStatus.LOW_CONFIDENCE,
    SegmentationStatus.FAILED,
)


class FakeFinder:
    name = "fake-finder"
    version = "1"

    def __init__(self, detections) -> None:  # noqa: ANN001
        self.detections = detections
        self.calls: list[tuple[tuple[int, int], list[str]]] = []

    def detect(self, image, prompts):  # noqa: ANN001, ANN201
        self.calls.append((image.size, list(prompts)))
        return list(self.detections)


class FakeOutliner:
    name = "fake-outliner"
    version = "1"

    def __init__(self, outlines) -> None:  # noqa: ANN001
        self.outlines = outlines
        self.boxes: list[list[tuple]] = []

    def segment(self, image, boxes):  # noqa: ANN001, ANN201
        self.boxes.append(list(boxes))
        return list(self.outlines)


def rect_mask(box, size=SIZE) -> Image.Image:  # noqa: ANN001
    mask = np.zeros((size[1], size[0]), dtype="uint8")
    mask[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])] = 255
    return Image.fromarray(mask)


def picture(size=SIZE) -> Image.Image:  # noqa: ANN001
    return Image.new("RGB", size, (30, 60, 200))


def find(outline: Segment, *, min_score: float = 0.7, min_coverage: float = 0.3) -> Found:
    finder = FakeFinder([Detection("desk", 0.8, BOX)])
    (found,) = find_products(
        picture(), finder, FakeOutliner([outline]), PROMPTS, min_score=min_score, min_coverage=min_coverage
    )
    return found


# ------------------------------------------------------------------------------ finding
def test_nothing_found_means_the_outliner_is_never_asked() -> None:
    outliner = FakeOutliner([])
    assert find_products(picture(), FakeFinder([]), outliner, PROMPTS, min_score=0.7, min_coverage=0.3) == []
    assert outliner.boxes == []


def test_the_finder_is_asked_for_the_wanted_things_in_the_panel() -> None:
    finder = FakeFinder([])
    find_products(picture(), finder, FakeOutliner([]), PROMPTS, min_score=0.7, min_coverage=0.3)
    assert finder.calls == [(SIZE, PROMPTS)]


def test_a_good_outline_is_accepted_and_measured() -> None:
    found = find(Segment(rect_mask(BOX), 0.9))
    assert found.status is OK and found.note is None
    assert found.detection.label == "desk" and found.score == 0.9
    assert found.coverage == pytest.approx(1.0)
    assert found.mask is not None and found.mask.size == SIZE and found.mask.mode == "L"


def test_outlines_are_asked_for_in_reading_order() -> None:
    low, high = Detection("desk", 0.8, (50, 250, 450, 500)), Detection("office chair", 0.7, (60, 60, 220, 300))
    outliner = FakeOutliner([Segment(Image.new("L", (500, 600), 255), 0.9)] * 2)
    found = find_products(
        Image.new("RGB", (500, 600)), FakeFinder([low, high]), outliner, PROMPTS, min_score=0.7, min_coverage=0.3
    )
    assert outliner.boxes == [[(60, 60, 220, 300), (50, 250, 450, 500)]]
    assert [f.detection.label for f in found] == ["office chair", "desk"]


def test_an_outline_the_model_is_unsure_of_is_kept_but_marked() -> None:
    found = find(Segment(rect_mask(BOX), 0.5))
    assert found.status is LOW and found.mask is not None
    assert "not sure" in found.note and "50%" in found.note


def test_an_outline_that_fills_little_of_its_box_is_marked() -> None:
    found = find(Segment(rect_mask((40, 20, 64, 80)), 0.95))  # 24 x 60 = 1440 of 7200 = 20%
    assert found.status is LOW and found.coverage == pytest.approx(0.2)
    assert "fills only 20% of the find's box" in found.note


def test_when_both_doubts_apply_the_models_own_doubt_is_reported() -> None:
    found = find(Segment(rect_mask((40, 20, 64, 80)), 0.4))
    assert found.status is LOW and "not sure" in found.note


def test_a_score_exactly_at_the_limit_is_accepted() -> None:
    assert find(Segment(rect_mask(BOX), 0.7), min_score=0.7).status is OK


def test_a_share_exactly_at_the_limit_is_accepted() -> None:
    thirty = (40, 20, 160, 38)  # 120 x 18 = 2160 of 7200 = exactly 30%
    assert find(Segment(rect_mask(thirty), 0.9), min_coverage=0.3).status is OK
    assert find(Segment(rect_mask((40, 20, 160, 37)), 0.9), min_coverage=0.3).status is LOW


def test_an_empty_outline_is_a_failure_without_a_mask() -> None:
    found = find(Segment(Image.new("L", SIZE, 0), 0.9))
    assert found.status is FAILED and found.mask is None and found.coverage == 0.0
    assert found.note == "the outline is empty"


def test_an_outline_only_outside_the_box_is_empty_too() -> None:
    found = find(Segment(rect_mask((0, 0, 30, 100)), 0.9))
    assert found.status is FAILED and found.mask is None


def test_an_outline_of_the_wrong_size_is_a_failure() -> None:
    found = find(Segment(Image.new("L", (50, 50), 255), 0.9))
    assert found.status is FAILED and found.score is None and "did not fit" in found.note


def test_whatever_is_outside_the_box_is_dropped_from_the_outline() -> None:
    found = find(Segment(Image.new("L", SIZE, 255), 0.9))  # the model outlined the whole panel
    assert found.status is OK and found.coverage == pytest.approx(1.0)
    covered = np.asarray(found.mask) > 127
    assert int(covered.sum()) == 7200 and covered[20:80, 40:160].all() and not covered[:20].any()


def test_a_different_number_of_outlines_than_boxes_is_an_error() -> None:
    with pytest.raises(ValueError, match="different number"):
        find_products(
            picture(),
            FakeFinder([Detection("desk", 0.8, BOX)]),
            FakeOutliner([]),
            PROMPTS,
            min_score=0.7,
            min_coverage=0.3,
        )


# ------------------------------------------------------------------------------ pictures
def scene() -> tuple[Image.Image, Found]:
    """Blue panel with a red 80 x 40 object (a corner missing) inside a bigger find box."""
    image = picture()
    for x in range(60, 140):
        for y in range(30, 70):
            image.putpixel((x, y), (220, 40, 40))
    mask = np.zeros((SIZE[1], SIZE[0]), dtype="uint8")
    mask[30:70, 60:140] = 255
    mask[30:40, 60:70] = 0  # the top left corner of the object is not part of it
    found = Found(Detection("desk", 0.8, BOX), OK, 0.9, 0.5, Image.fromarray(mask), None)
    return image, found


def opened(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data))


def test_the_crop_is_the_finds_box_cut_from_the_panel() -> None:
    image, found = scene()
    renders = render_find(image, found, jpeg_quality=90, thumbnail_px=64)
    crop = opened(renders.crop_jpeg)
    assert crop.format == "JPEG" and crop.size == (120, 60)


def test_the_mask_picture_is_the_size_of_the_panel() -> None:
    image, found = scene()
    mask = opened(render_find(image, found, jpeg_quality=90, thumbnail_px=64).mask_png)
    assert mask.format == "PNG" and mask.mode == "L" and mask.size == SIZE
    assert set(np.unique(np.asarray(mask))) == {0, 255}


def test_the_cutout_is_trimmed_to_the_object_with_a_transparent_background() -> None:
    image, found = scene()
    cutout = opened(render_find(image, found, jpeg_quality=90, thumbnail_px=64).cutout_png)
    assert cutout.format == "PNG" and cutout.mode == "RGBA" and cutout.size == (80, 40)
    assert cutout.getpixel((0, 0))[3] == 0  # the missing corner is see-through
    assert cutout.getpixel((40, 20)) == (220, 40, 40, 255)  # the object itself, exactly
    assert cutout.getpixel((79, 39))[3] == 255


def test_the_white_picture_shows_the_object_on_white() -> None:
    image, found = scene()
    white = opened(render_find(image, found, jpeg_quality=95, thumbnail_px=64).white_jpeg)
    assert white.format == "JPEG" and white.size == (80, 40)
    assert min(white.getpixel((0, 0))) > 240  # the see-through corner became white
    red, green, blue = white.getpixel((40, 20))
    assert red > 190 and green < 90 and blue < 90


def test_the_thumbnail_fits_the_requested_size() -> None:
    image, found = scene()
    thumbnail = opened(render_find(image, found, jpeg_quality=90, thumbnail_px=40).thumbnail_jpeg)
    assert thumbnail.format == "JPEG" and thumbnail.size == (40, 20)


def test_a_find_without_an_outline_only_has_its_crop() -> None:
    image, _ = scene()
    failed = Found(Detection("desk", 0.8, BOX), FAILED, None, None, None, "the outline is empty")
    renders = render_find(image, failed, jpeg_quality=90, thumbnail_px=64)
    assert opened(renders.crop_jpeg).size == (120, 60)
    assert (renders.mask_png, renders.cutout_png, renders.white_jpeg, renders.thumbnail_jpeg) == (None,) * 4


def test_a_box_hanging_over_the_panel_is_cut_at_the_edge() -> None:
    image = picture((100, 50))
    found = Found(Detection("desk", 0.8, (-5.3, 2.2, 150.0, 90.0)), FAILED, None, None, None, "x")
    assert opened(render_find(image, found, jpeg_quality=90, thumbnail_px=64).crop_jpeg).size == (100, 48)


def test_a_box_without_area_still_gives_a_one_pixel_crop() -> None:
    image = picture((100, 50))
    found = Found(Detection("desk", 0.8, (10.0, 10.0, 10.0, 10.0)), FAILED, None, None, None, "x")
    assert opened(render_find(image, found, jpeg_quality=90, thumbnail_px=64).crop_jpeg).size == (1, 1)


def test_a_box_at_the_far_corner_is_kept_inside_the_panel() -> None:
    image = picture((100, 50))
    found = Found(Detection("desk", 0.8, (100.0, 50.0, 120.0, 70.0)), FAILED, None, None, None, "x")
    assert opened(render_find(image, found, jpeg_quality=90, thumbnail_px=64).crop_jpeg).size == (1, 1)

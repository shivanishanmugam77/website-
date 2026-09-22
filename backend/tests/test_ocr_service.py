"""Page-level text reading: tiling, merging, filtering and ordering (no real engine needed)."""

from __future__ import annotations

import pytest
from PIL import Image, ImageDraw

from app.services.ocr import detect_language, read_page_text, tile_boxes
from tests.ocr_fakes import FunctionOcr, SceneLine, SceneOcr, coordinate_page, line

TILE, OVERLAP = 1000, 200


def read(page: Image.Image, engine, **overrides):  # noqa: ANN001, ANN201
    options = {"min_confidence": 0.5, "tile_px": TILE, "overlap_px": OVERLAP, "max_lines": 1000}
    return read_page_text(page, engine, **{**options, **overrides})


# ------------------------------------------------------------------------------ tiling
@pytest.mark.parametrize(
    ("width", "height"),
    [(2500, 1800), (4970, 2487), (1300, 6000), (3000, 1000), (1251, 1251), (1250, 4000)],
)
def test_tiles_cover_the_page_and_overlap_by_at_least_the_requested_amount(
    width: int, height: int
) -> None:
    boxes = tile_boxes(width, height, TILE, OVERLAP)
    xs = sorted({box[0] for box in boxes})
    ys = sorted({box[1] for box in boxes})
    assert xs[0] == 0 and ys[0] == 0
    assert max(box[2] for box in boxes) == width and max(box[3] for box in boxes) == height
    assert len(boxes) == len(xs) * len(ys)  # a regular grid
    for starts, size in ((xs, TILE), (ys, TILE)):
        for previous, following in zip(starts, starts[1:], strict=False):
            assert following <= previous + size - OVERLAP  # neighbours overlap enough
    for x0, y0, x1, y1 in boxes:
        assert x1 - x0 <= 1250 and y1 - y0 <= 1250  # never bigger than the engine copes with


def test_a_page_only_slightly_larger_than_a_tile_is_read_in_one_piece() -> None:
    # An A4 page at 150 dpi is 1240 x 1755: splitting it would double the work for nothing.
    assert tile_boxes(1240, 1755, 1600, 320) == [(0, 0, 1240, 1755)]
    assert tile_boxes(800, 600, 1600, 320) == [(0, 0, 800, 600)]


def test_a_wide_page_is_split_into_a_grid() -> None:
    boxes = tile_boxes(4970, 2487, 1600, 320)
    assert len(boxes) == 8  # 4 columns x 2 rows
    assert boxes[0] == (0, 0, 1600, 1600) and boxes[-1] == (3370, 887, 4970, 2487)


def test_the_tile_must_be_larger_than_the_overlap() -> None:
    with pytest.raises(ValueError):
        tile_boxes(3000, 3000, 500, 500)


# ------------------------------------------------------------------------------ merging
def test_lines_are_reported_in_page_coordinates() -> None:
    page = coordinate_page(2500, 1800)
    engine = SceneOcr([SceneLine("MODEL YY-21", 1600, 1300, 1780, 1330)])
    (found,) = read(page, engine)
    assert found.text == "MODEL YY-21"
    assert found.bbox == pytest.approx((1600, 1300, 1780, 1330))
    assert found.polygon[0] == pytest.approx((1600, 1300))
    assert found.polygon[2] == pytest.approx((1780, 1330))


def test_a_line_seen_by_two_overlapping_tiles_is_reported_once() -> None:
    page = coordinate_page(2500, 1800)
    engine = SceneOcr([SceneLine("3200W x 1400D x 750H", 780, 100, 960, 130)])  # in the overlap
    found = read(page, engine)
    assert [item.text for item in found] == ["3200W x 1400D x 750H"]
    assert len(engine.crops) == len(tile_boxes(2500, 1800, TILE, OVERLAP))  # no extra reads


def test_a_line_cut_by_a_tile_edge_is_read_again_and_comes_out_whole() -> None:
    page = coordinate_page(2500, 1800)
    text = "MODEL: YY-21 MEETING TABLE 3200W x 1400D x 750H"
    engine = SceneOcr([SceneLine(text, 700, 400, 1500, 430)])  # wider than the overlap
    found = read(page, engine)
    assert [item.text for item in found] == [text]
    assert found[0].bbox == pytest.approx((700, 400, 1500, 430))
    strips = [crop for crop in engine.crops if crop[3] - crop[1] < 300]
    assert len(strips) == 1  # one cheap re-read (a thin strip), not another full tile


def test_a_line_cut_at_both_ends_by_different_tiles_is_still_read_whole() -> None:
    page = coordinate_page(2500, 1800)
    text = "A LONG CAPTION THAT CROSSES SEVERAL TILE EDGES AT ONCE"
    engine = SceneOcr([SceneLine(text, 400, 500, 1900, 530)])
    assert [item.text for item in read(page, engine)] == [text]


def test_identical_text_in_different_places_is_kept_twice() -> None:
    page = coordinate_page(2500, 1800)
    engine = SceneOcr(
        [SceneLine("YY-21", 100, 100, 200, 130), SceneLine("YY-21", 100, 400, 200, 430)]
    )
    assert [item.text for item in read(page, engine)] == ["YY-21", "YY-21"]


def test_stacked_lines_are_not_mistaken_for_duplicates() -> None:
    page = coordinate_page(2500, 1800)
    engine = SceneOcr(
        [SceneLine("FIRST LINE", 100, 100, 400, 130), SceneLine("SECOND LINE", 100, 135, 400, 165)]
    )
    assert [item.text for item in read(page, engine)] == ["FIRST LINE", "SECOND LINE"]


# ------------------------------------------------------------------------------ filtering
def test_low_confidence_lines_are_dropped() -> None:
    page = coordinate_page(1200, 800)
    engine = SceneOcr(
        [
            SceneLine("SURE", 100, 100, 300, 130, confidence=0.9),
            SceneLine("UNSURE", 100, 300, 300, 330, confidence=0.3),
        ]
    )
    assert [item.text for item in read(page, engine, min_confidence=0.5)] == ["SURE"]


def test_marks_without_letters_or_digits_are_dropped() -> None:
    page = coordinate_page(1200, 800)
    engine = SceneOcr(
        [
            SceneLine("|||", 100, 100, 300, 130),
            SceneLine("— . —", 100, 200, 300, 230),
            SceneLine("A1", 100, 300, 300, 330),
            SceneLine("\u578b\u53f7", 100, 400, 300, 430),  # Chinese characters are content
        ]
    )
    assert [item.text for item in read(page, engine)] == ["A1", "\u578b\u53f7"]


def test_whitespace_is_collapsed_and_nul_characters_removed() -> None:
    engine = FunctionOcr(lambda _i, _img: [line("  YY-21 \n\t MEETING\x00TABLE ", 10, 10, 200, 40)])
    page = Image.new("RGB", (400, 200))
    ImageDraw.Draw(page).rectangle((0, 0, 200, 100), fill="white")  # not blank
    (found,) = read(page, engine)
    assert found.text == "YY-21 MEETING TABLE"


def test_boxes_are_clamped_to_the_page_and_rotated_polygons_are_boxed() -> None:
    page = Image.new("RGB", (400, 200), "white")
    ImageDraw.Draw(page).rectangle((0, 0, 100, 100), fill="black")
    rotated = ((-6.0, 10.0), (100.0, 4.0), (104.0, 30.0), (-2.0, 36.0))
    engine = FunctionOcr(lambda _i, _img: [type(line("x", 0, 0, 1, 1))("ROTATED", 0.9, rotated)])
    (found,) = read(page, engine)
    assert found.bbox == (0.0, 4.0, 104.0, 36.0)  # left edge clamped to the page
    assert found.polygon == rotated  # the engine's polygon is kept as it was


def test_the_line_cap_keeps_the_most_confident_lines_in_reading_order() -> None:
    page = coordinate_page(1200, 800)
    engine = SceneOcr(
        [
            SceneLine("TOP", 100, 100, 200, 130, confidence=0.6),
            SceneLine("MIDDLE", 100, 300, 200, 330, confidence=0.99),
            SceneLine("BOTTOM", 100, 500, 200, 530, confidence=0.9),
        ]
    )
    assert [item.text for item in read(page, engine, max_lines=2)] == ["MIDDLE", "BOTTOM"]


def test_blank_tiles_are_never_sent_to_the_engine() -> None:
    blank = Image.new("RGB", (2500, 1800), "white")
    engine = FunctionOcr(lambda _i, _img: [line("GHOST", 0, 0, 50, 20)])
    assert read(blank, engine) == [] and engine.calls == 0

    one_corner = blank.copy()
    ImageDraw.Draw(one_corner).rectangle((0, 0, 500, 500), fill="black")
    engine = FunctionOcr(lambda _i, _img: [])
    read(one_corner, engine)
    assert engine.calls == 1  # only the tile that has something in it


def test_any_image_mode_is_accepted_and_the_engine_gets_rgb() -> None:
    seen = []
    engine = FunctionOcr(lambda _i, image: seen.append(image.mode) or [])
    grey = Image.new("L", (300, 200), 0)
    ImageDraw.Draw(grey).rectangle((0, 0, 100, 100), fill=255)
    read(grey, engine)
    assert seen == ["RGB"]


# ------------------------------------------------------------------------------ ordering
def test_reading_order_is_top_to_bottom_then_left_to_right() -> None:
    page = coordinate_page(1200, 800)
    engine = SceneOcr(
        [
            SceneLine("FOURTH", 500, 500, 700, 530),
            SceneLine("SECOND", 500, 100, 700, 130),  # same row as FIRST, slightly higher
            SceneLine("THIRD", 100, 300, 300, 330),
            SceneLine("FIRST", 100, 104, 300, 134),
        ]
    )
    assert [item.text for item in read(page, engine)] == ["FIRST", "SECOND", "THIRD", "FOURTH"]


# ------------------------------------------------------------------------------ language
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("MODEL YY-21", "en"),
        ("\u4f1a\u8bae\u684c", "zh"),
        ("\u578b\u53f7: YY-21 \u4f1a\u8bae\u684c", "mixed"),
        ("3200 - 1400", None),
        ("", None),
    ],
)
def test_language_is_told_from_the_characters_used(text: str, expected: str | None) -> None:
    assert detect_language(text) == expected

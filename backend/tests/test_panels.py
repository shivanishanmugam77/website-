"""Finding the photo panels on a page (synthetic pages with known layouts)."""

from __future__ import annotations

import pytest
from PIL import Image, ImageDraw

from app.services.panels import (
    SHEET_CELL_PX,
    SHEET_LABEL_PX,
    SheetItem,
    assign_lines,
    detect_panels,
    draw_contact_sheet,
    draw_overlay,
    sort_reading_order,
)
from tests.page_factory import (
    PAPER,
    blank_page,
    draw_chair,
    draw_text,
    framed_print_photo,
    iou,
    place_photo,
    light_wall_photo,
    photo,
)


def boxes(panels) -> list[tuple]:  # noqa: ANN001
    return [panel.bbox for panel in panels]


def assert_layout(panels, truth: list[tuple], minimum_iou: float = 0.97) -> None:  # noqa: ANN001
    found = boxes(panels)
    assert len(found) == len(truth), f"found {found}"
    for got, want in zip(found, truth, strict=True):
        assert iou(got, want) >= minimum_iou, f"{got} is not {want}"


# ------------------------------------------------------------------------------ layouts
def test_a_big_photo_beside_two_stacked_photos_gives_three_panels_in_reading_order() -> None:
    page = blank_page(3200, 1600)
    truth = [
        place_photo(page, (120, 80, 2000, 1500), 1),
        place_photo(page, (2060, 80, 3080, 780), 2, tone=(170, 120, 90)),
        place_photo(page, (2060, 830, 3080, 1500), 3, tone=(120, 110, 100)),
    ]
    panels = detect_panels(page)
    assert_layout(panels, truth)
    assert all(panel.fill > 0.95 for panel in panels)
    assert panels[0].area_share == pytest.approx(0.5286, abs=0.01)


def test_products_on_paper_are_found_next_to_a_photo() -> None:
    page = blank_page(3200, 1600)
    truth = [
        place_photo(page, (100, 100, 1900, 1500), 4),
        draw_chair(page, 2200, 300),
        draw_chair(page, 2500, 300),
        draw_chair(page, 2800, 300),
    ]
    assert_layout(detect_panels(page), truth, minimum_iou=0.9)


def test_a_photo_with_a_light_wall_in_it_is_not_cut_in_two() -> None:
    page = blank_page(3200, 1600)
    box = (200, 100, 3000, 1500)
    light_wall_photo(page, box, seed=5)
    assert_layout(detect_panels(page), [box])


def test_a_framed_print_on_the_wall_of_a_photo_is_not_a_panel_of_its_own() -> None:
    page = blank_page(3200, 1600)
    box = (200, 100, 3000, 1500)
    framed_print_photo(page, box, seed=17)
    assert_layout(detect_panels(page), [box])


def test_a_print_reaching_past_the_top_edge_of_its_photo_is_still_part_of_it() -> None:
    page = blank_page(3200, 1600)
    box = (200, 100, 3000, 1500)
    framed_print_photo(page, box, seed=18, poke=80)  # the print pokes 80 px (18 % of it) out
    assert_layout(detect_panels(page), [(200, 180, 3000, 1500)], minimum_iou=0.97)


def test_slanted_photos_separated_by_a_diagonal_gap_are_two_panels() -> None:
    page = blank_page(3200, 1600)
    for source, polygon in (
        (photo(3200, 1600, 6), [(200, 100), (1500, 100), (1200, 1450), (200, 1450)]),
        (photo(3200, 1600, 7, tone=(90, 130, 160)), [(1700, 100), (3000, 100), (3000, 1450), (1400, 1450)]),
    ):
        mask = Image.new("L", page.size, 0)
        ImageDraw.Draw(mask).polygon(polygon, fill=255)
        page.paste(source, (0, 0), mask)
    panels = detect_panels(page)
    assert len(panels) == 2
    assert boxes(panels)[0][0] < 260 and boxes(panels)[1][2] > 2940


def test_photos_joined_by_a_thin_line_are_still_two_panels() -> None:
    page = blank_page(3200, 1600)
    truth = [place_photo(page, (100, 100, 1500, 1500), 20), place_photo(page, (1700, 100, 3100, 1500), 21)]
    ImageDraw.Draw(page).rectangle((1500, 780, 1700, 784), fill=(30, 30, 30))  # a connector line
    assert_layout(detect_panels(page), truth)


def test_a_dark_page_works_the_same_way() -> None:
    page = blank_page(2400, 1200, paper=(25, 25, 30))
    truth = [
        place_photo(page, (100, 100, 1200, 1100), 8, tone=(220, 200, 170)),
        place_photo(page, (1300, 100, 2300, 1100), 9, tone=(200, 220, 180)),
    ]
    assert_layout(detect_panels(page), truth)


def test_a_page_larger_than_the_working_size_gives_page_coordinates() -> None:
    page = blank_page(4970, 2487)
    truth = [place_photo(page, (300, 200, 3000, 2300), 10), place_photo(page, (3200, 200, 4700, 1200), 11)]
    assert_layout(detect_panels(page), truth, minimum_iou=0.98)


# ------------------------------------------------------------------------------ non-pictures
def test_a_full_bleed_picture_is_one_panel_covering_the_page() -> None:
    page = blank_page(2000, 1000)
    place_photo(page, (0, 0, 2000, 1000), 12)
    (panel,) = detect_panels(page)
    assert panel.bbox == (0, 0, 2000, 1000) and panel.area_share == 1.0


def test_a_smooth_full_bleed_picture_is_not_mistaken_for_paper() -> None:
    page = Image.new("RGB", (2000, 1000))
    draw = ImageDraw.Draw(page)
    for x in range(2000):  # a gentle left-to-right shade: no sharp variation, but not paper
        draw.line((x, 0, x, 999), fill=(120 + x // 66, 90 + x // 66, 70 + x // 66))
    (panel,) = detect_panels(page)
    assert panel.bbox == (0, 0, 2000, 1000)


def test_an_empty_page_has_no_panels() -> None:
    assert detect_panels(blank_page(2000, 1000)) == []


def test_headings_and_captions_are_not_panels() -> None:
    page = blank_page(3200, 1600)
    photo_box = place_photo(page, (200, 300, 3000, 1300), 13)
    heading = draw_text(page, (200, 60), "SERVICE PROVIDER FOR COMMERCIAL SPACE", size=90)
    caption = draw_text(page, (200, 1320), "MODEL: YY-11  2400Wx1200Dx750H", size=36)
    assert_layout(detect_panels(page, [heading, caption]), [photo_box], minimum_iou=0.95)


def test_small_print_is_not_a_panel_even_without_the_text_engine() -> None:
    page = blank_page(3200, 1600)
    photo_box = place_photo(page, (200, 300, 3000, 1300), 13)
    draw_text(page, (200, 1320), "MODEL: YY-11  2400Wx1200Dx750H", size=36)
    assert_layout(detect_panels(page), [photo_box], minimum_iou=0.95)


def test_a_caption_just_under_a_photo_does_not_stretch_the_panel() -> None:
    page = blank_page(3200, 1600)
    photo_box = place_photo(page, (200, 200, 3000, 1300), 14)
    caption = draw_text(page, (210, 1306), "TWO PERSON DESK 1200Wx1200Dx750H", size=36)  # 6 px gap
    assert_layout(detect_panels(page, [caption]), [photo_box], minimum_iou=0.99)


def test_an_outline_is_not_a_picture() -> None:
    page = blank_page(3200, 1600)
    ImageDraw.Draw(page).rectangle((300, 300, 1500, 1200), outline=(30, 30, 30), width=8)
    assert detect_panels(page) == []


def test_text_printed_on_a_photo_does_not_eat_into_it() -> None:
    page = blank_page(3200, 1600)
    photo_box = place_photo(page, (200, 200, 3000, 1300), 15)
    strip = (200.0, 1240.0, 3000.0, 1300.0)  # a caption band along the photo's bottom edge
    assert_layout(detect_panels(page, [strip]), [photo_box], minimum_iou=0.99)


def test_small_pictures_are_ignored_unless_the_limit_is_lowered() -> None:
    page = blank_page(3200, 1600)
    small = place_photo(page, (500, 500, 620, 620), 16)  # 0.28 % of the page
    assert detect_panels(page) == []
    assert_layout(detect_panels(page, min_area_share=0.001), [small], minimum_iou=0.9)


def test_thin_rules_and_marks_are_not_panels() -> None:
    page = blank_page(3200, 1600)
    draw = ImageDraw.Draw(page)
    draw.rectangle((100, 100, 3100, 108), fill=(30, 30, 30))  # a long thin rule
    draw.ellipse((1000, 700, 1030, 730), fill=(200, 50, 50))  # a bullet
    assert detect_panels(page) == []


# ------------------------------------------------------------------------------ ordering
def test_reading_order_groups_panels_into_rows() -> None:
    items = [(1800, 900, 2000, 1000), (100, 105, 300, 400), (1000, 100, 1300, 400), (100, 900, 300, 1000)]
    ordered = sort_reading_order(items, 1600, key=lambda box: box)
    assert ordered == [(100, 105, 300, 400), (1000, 100, 1300, 400), (100, 900, 300, 1000), (1800, 900, 2000, 1000)]


# ------------------------------------------------------------------------------ captions
PANELS = [(100, 100, 1000, 800), (1100, 100, 2000, 450), (1100, 500, 2000, 800)]


def test_a_line_belongs_to_the_panel_that_contains_it() -> None:
    lines = [(0, (150, 700, 500, 740)), (1, (1200, 150, 1500, 190)), (2, (1200, 520, 1500, 560))]
    assert assign_lines(PANELS, lines, page_height=1000) == [[0], [1], [2]]


def test_a_line_inside_two_overlapping_panels_goes_to_the_smaller_one() -> None:
    panels = [(0, 0, 1000, 800), (100, 100, 400, 400)]
    assert assign_lines(panels, [(0, (150, 150, 300, 200))], page_height=1000) == [[], [0]]


def test_a_caption_below_a_panel_goes_to_that_panel() -> None:
    lines = [(0, (120, 820, 600, 860)), (1, (1150, 815, 1800, 850))]
    assert assign_lines(PANELS, lines, page_height=1000) == [[0], [], [1]]


def test_text_that_belongs_to_no_panel_is_left_out() -> None:
    lines = [
        (0, (150, 20, 900, 60)),  # a heading above everything
        (1, (120, 900, 600, 940)),  # far below the panel: outside the caption reach
        (2, (2100, 200, 2400, 240)),  # to the side, no panel underneath
    ]
    assert assign_lines(PANELS, lines, page_height=1000) == [[], [], []]


def test_a_caption_must_line_up_with_its_panel() -> None:
    assert assign_lines(PANELS, [(0, (2050, 820, 2500, 860))], page_height=1000) == [[], [], []]


# ------------------------------------------------------------------------------ preview
def test_the_overlay_outlines_every_panel_and_leaves_the_rest_alone() -> None:
    page = blank_page(800, 400)
    overlay = draw_overlay(page, [(100, 100, 300, 300), (400, 100, 700, 300)])
    assert overlay.size == page.size
    assert overlay.getpixel((100, 200)) != PAPER  # the outline is drawn
    assert overlay.getpixel((5, 5)) == PAPER  # elsewhere untouched
    assert page.getpixel((100, 200)) == PAPER  # and the original is not modified


# ------------------------------------------------------------------------------ contact sheet
def sheet_item(label: str = "Page 1", boxes=((400, 300, 2000, 1500),)) -> SheetItem:  # noqa: ANN001
    return SheetItem(label, Image.new("RGB", (400, 300), (200, 200, 200)), (4000, 3000), boxes)


def test_the_contact_sheet_lays_pages_out_in_a_grid() -> None:
    sheet = draw_contact_sheet([sheet_item(f"Page {n}") for n in range(1, 6)], columns=4)
    assert sheet.size == (4 * SHEET_CELL_PX, 2 * (SHEET_CELL_PX + SHEET_LABEL_PX))
    assert draw_contact_sheet([sheet_item()], columns=4).size == (4 * SHEET_CELL_PX, SHEET_CELL_PX + SHEET_LABEL_PX)


def test_panel_boxes_are_scaled_from_page_pixels_to_the_thumbnail() -> None:
    sheet = draw_contact_sheet([sheet_item()], columns=1)
    coloured = [
        (x, y)
        for x in range(sheet.width)
        for y in range(SHEET_CELL_PX)
        if max(sheet.getpixel((x, y))) - min(sheet.getpixel((x, y))) > 60
    ]
    xs, ys = [c[0] for c in coloured], [c[1] for c in coloured]
    left, top = (SHEET_CELL_PX - 400) // 2, (SHEET_CELL_PX - 300) // 2  # the thumbnail's corner
    # the box (400,300)-(2000,1500) on a 4000 x 3000 page is (40,30)-(200,150) on a 400 x 300 picture
    assert abs(min(xs) - (left + 40)) <= 6 and abs(max(xs) - (left + 200)) <= 6
    assert abs(min(ys) - (top + 30)) <= 6 and abs(max(ys) - (top + 150)) <= 6


def test_a_page_without_panels_is_shown_untouched_with_its_label() -> None:
    sheet = draw_contact_sheet([sheet_item("Page 7  COVER  0 panels", boxes=())], columns=1)
    picture = sheet.crop((10, 60, 410, 360))
    assert all(max(px) - min(px) < 10 for px in picture.getdata())  # no outline drawn
    label_strip = sheet.crop((0, SHEET_CELL_PX, SHEET_CELL_PX, SHEET_CELL_PX + SHEET_LABEL_PX))
    assert any(px[0] > 200 for px in label_strip.getdata())  # the caption text is there

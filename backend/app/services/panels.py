"""Cut a catalogue page into its photo panels.

Most supplier pages are a sheet of paper with several pictures laid out on it: a big room
scene, close-ups beside it, a row of products on white. Each of those pictures is a
*panel*. Finding them needs no machine learning, only the observation that panels are
separated from each other by the paper:

1. estimate the paper colour from the page margins (if there are no margins the page is
   one full-bleed picture and is returned as a single panel);
2. mark every pixel that differs from the paper;
3. remove the marks made by printed text (the text engine already knows where the text is),
   join tiny gaps, and take each connected blob as a candidate;
4. split a blob wherever a clean strip of paper runs right across it (a layout such as one
   big photo beside two stacked ones can be joined by a thin line or shadow);
5. keep candidates that are big enough and filled enough to be pictures rather than
   headings, rules or stray marks, and drop any that lie inside a bigger one (a framed print
   on the wall of a room photo is part of that photo).

Known limits, on purpose: pictures that touch with no paper between them are returned as
one panel; a photo that contains a strip of paper-coloured pixels running the whole way
across it (a white wall) wider than ``SPLIT_GUTTER_PX`` may be cut in two; and when the text
engine found nothing (it is off, or failed) large printed words may show up as panels. Real
catalogues will show how often these matter; ``/pages/{n}/panels/preview`` draws the result
on the page for checking.

numpy and OpenCV are imported inside the functions so that importing this module (the API
process does) stays cheap; only worker processes and tests run the detection.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    import numpy as np

Box = tuple[float, float, float, float]

WORK_MAX_PX = 1600  # the detection runs on a copy no larger than this along its long side
BACKGROUND_TOLERANCE = 24  # max per-channel difference from the paper that still counts as paper
MARGIN_BAND = 0.02  # width of the page-edge band used to find the paper colour
# Paper is very uniform (a few levels of noise); a photograph, even a smooth one, is not. The
# page edge only counts as paper when this much of it lies within PAPER_UNIFORMITY of one colour.
PAPER_UNIFORMITY = 10
CLEAN_MARGIN_SHARE = 0.55  # share of the band that must be paper for the page to have margins
CLOSING_PX = 5  # gaps of paper narrower than this inside a picture are filled
SPLIT_GUTTER_PX = 8  # a strip of paper at least this wide separates two panels
MIN_SIDE_PX = 20  # a panel is at least this wide and tall (in working pixels)
MIN_FILL = 0.12  # share of a panel's box that must differ from the paper (rejects rules, text)
TEXT_ERASE_MAX_FOREGROUND = 0.45  # text over paper is erased from the mask; text on a photo is not
EMPTY_PROFILE = 0.005  # a row/column with at most this share of marks counts as empty ...
EMPTY_MARKS = 6  # ... or at most this many marks (a thin connector line crossing the gutter)
ROW_BAND = 0.08  # panels whose tops are within this share of the page height share a row
CAPTION_REACH = 0.06  # captions may sit this share of the page height below their panel
# A panel with this much of its area inside a bigger one is dropped as part of it. Well below
# 1.0 on purpose: a print on a wall often reaches a little past the edge the rest of the photo
# is measured by, so it only overlaps its photo by 80-95 %.
NESTED_SHARE = 0.75


@dataclass(frozen=True)
class Panel:
    """One picture on a page, in the page image's pixel coordinates."""

    bbox: tuple[int, int, int, int]
    area_share: float  # share of the page area (0..1)
    fill: float  # share of the box that differs from the paper (0..1)


# ------------------------------------------------------------------------------ detection
def detect_panels(
    image: Image.Image,
    text_boxes: Sequence[Box] = (),
    *,
    min_area_share: float = 0.004,
) -> list[Panel]:
    """The pictures on the page, in reading order.

    ``text_boxes`` are the boxes of recognised text (page pixels); text printed on paper is
    left out of the picture mask so a caption just under a photo does not become part of it.
    """
    import cv2
    import numpy as np

    page = image if image.mode == "RGB" else image.convert("RGB")
    width, height = page.size
    scale = max(1.0, max(width, height) / WORK_MAX_PX)
    work = page if scale == 1.0 else page.resize((round(width / scale), round(height / scale)))
    rgb = np.asarray(work, dtype=np.uint8)
    work_h, work_w = rgb.shape[:2]

    paper = _paper_colour(rgb)
    if paper is None:
        return [Panel((0, 0, width, height), 1.0, 1.0)]  # no margins: one full-bleed picture

    difference = np.abs(rgb.astype(np.int16) - paper.astype(np.int16)).max(axis=2)
    marks = cv2.blur(difference.astype(np.uint8), (3, 3)) > BACKGROUND_TOLERANCE
    picture = marks.copy()
    for box in text_boxes:
        x0, y0, x1, y1 = _clamp_box(
            (box[0] / scale - 2, box[1] / scale - 2, box[2] / scale + 2, box[3] / scale + 2),
            work_w,
            work_h,
        )
        if x1 > x0 and y1 > y0 and picture[y0:y1, x0:x1].mean() < TEXT_ERASE_MAX_FOREGROUND:
            picture[y0:y1, x0:x1] = False  # printed text on paper: not part of any picture

    kernel = np.ones((CLOSING_PX, CLOSING_PX), np.uint8)
    closed = cv2.morphologyEx(picture.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)

    page_area = work_w * work_h
    found: list[Panel] = []
    for index in range(1, count):
        x, y, box_w, box_h, _area = (int(v) for v in stats[index])
        blob = labels[y : y + box_h, x : x + box_w] == index
        for x0, y0, x1, y1 in _split(blob, x, y):
            area = (x1 - x0) * (y1 - y0)
            if min(x1 - x0, y1 - y0) < MIN_SIDE_PX or area / page_area < min_area_share:
                continue
            fill = float(marks[y0:y1, x0:x1].mean())
            if fill < MIN_FILL:
                continue
            found.append(
                Panel(
                    bbox=_to_page_box((x0, y0, x1, y1), scale, width, height),
                    area_share=area / page_area,
                    fill=fill,
                )
            )
    found = [panel for panel in found if not _is_inside_another(panel, found)]
    return sort_reading_order(found, height, key=lambda panel: panel.bbox)


def _is_inside_another(panel: Panel, panels: Sequence[Panel]) -> bool:
    """A blob lying inside a bigger panel is part of that picture, not a picture of its own
    (a framed print on a wall in a room photo: its white mat cuts it off from the photo)."""
    own = _box_area(panel.bbox)
    for other in panels:
        if other is panel or _box_area(other.bbox) <= own:
            continue
        width = min(panel.bbox[2], other.bbox[2]) - max(panel.bbox[0], other.bbox[0])
        height = min(panel.bbox[3], other.bbox[3]) - max(panel.bbox[1], other.bbox[1])
        if width > 0 and height > 0 and width * height >= NESTED_SHARE * own:
            return True
    return False


def _box_area(box: tuple[int, int, int, int]) -> int:
    return (box[2] - box[0]) * (box[3] - box[1])


def _paper_colour(rgb: np.ndarray) -> np.ndarray | None:
    """The colour of the paper, or None when the page edge is not mostly one colour."""
    import numpy as np

    height, width = rgb.shape[:2]
    band = max(2, round(min(height, width) * MARGIN_BAND))
    edge = np.concatenate(
        [
            rgb[:band].reshape(-1, 3),
            rgb[-band:].reshape(-1, 3),
            rgb[band:-band, :band].reshape(-1, 3),
            rgb[band:-band, -band:].reshape(-1, 3),
        ]
    )
    paper = np.median(edge, axis=0)
    close = np.abs(edge.astype(np.int16) - paper).max(axis=1) <= PAPER_UNIFORMITY
    return paper.astype(np.uint8) if close.mean() >= CLEAN_MARGIN_SHARE else None


def _split(mask: np.ndarray, ox: int, oy: int) -> list[tuple[int, int, int, int]]:
    """Boxes of the pictures in a blob, cutting wherever paper runs right across it."""
    import numpy as np

    rows, cols = np.nonzero(mask.any(axis=1))[0], np.nonzero(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return []
    mask = mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]  # trim to the marks
    ox, oy = ox + int(cols[0]), oy + int(rows[0])
    cut = _widest_gutter(mask)
    if cut is None:
        return [(ox, oy, ox + mask.shape[1], oy + mask.shape[0])]
    axis, start, end = cut
    if axis == 0:  # a horizontal strip of paper: split into top and bottom
        return _split(mask[:start], ox, oy) + _split(mask[end:], ox, oy + end)
    return _split(mask[:, :start], ox, oy) + _split(mask[:, end:], ox + end, oy)


def _widest_gutter(mask: np.ndarray) -> tuple[int, int, int] | None:
    """(axis, start, end) of the widest empty strip that separates two solid parts."""
    best: tuple[int, int, int] | None = None
    for axis in (0, 1):
        profile = mask.sum(axis=1 - axis)  # marks per row (axis 0) or per column (axis 1)
        length = mask.shape[1 - axis]
        empty = profile <= max(EMPTY_MARKS, EMPTY_PROFILE * length)
        position = 0
        while position < len(empty):
            if not empty[position]:
                position += 1
                continue
            end = position
            while end < len(empty) and empty[end]:
                end += 1
            inside = position >= MIN_SIDE_PX and len(empty) - end >= MIN_SIDE_PX
            if inside and end - position >= SPLIT_GUTTER_PX:
                if best is None or end - position > best[2] - best[1]:
                    best = (axis, position, end)
            position = end
    return best


def _clamp_box(box: Box, width: int, height: int) -> tuple[int, int, int, int]:
    return (
        max(0, math.floor(box[0])),
        max(0, math.floor(box[1])),
        min(width, math.ceil(box[2])),
        min(height, math.ceil(box[3])),
    )


def _to_page_box(
    box: tuple[int, int, int, int], scale: float, width: int, height: int
) -> tuple[int, int, int, int]:
    return _clamp_box(
        (box[0] * scale, box[1] * scale, box[2] * scale, box[3] * scale), width, height
    )


# ------------------------------------------------------------------------------ ordering
def sort_reading_order(items: Sequence, page_height: float, *, key) -> list:  # noqa: ANN001
    """Top to bottom in rows (tops within ``ROW_BAND`` of the page height share a row), then
    left to right. ``key(item)`` returns the item's box."""
    band = max(1.0, page_height * ROW_BAND)
    return sorted(items, key=lambda item: (math.floor(key(item)[1] / band), key(item)[0]))


# ------------------------------------------------------------------------------ captions
def assign_lines(
    panels: Sequence[Box], lines: Sequence[tuple[int, Box]], page_height: float
) -> list[list[int]]:
    """Which recognised text lines belong to which panel.

    A line belongs to the smallest panel that contains its centre. A line outside every
    panel is a caption if it sits just below a panel (within ``CAPTION_REACH`` of the page
    height) and lines up with it horizontally; it goes to the nearest such panel. Anything
    else (page headings, margin notes) belongs to no panel. ``lines`` are
    ``(line_index, box)``; the result has one sorted list of line indexes per panel.
    """
    owned: list[list[int]] = [[] for _ in panels]
    reach = page_height * CAPTION_REACH
    for index, box in lines:
        centre_x, centre_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        inside = [
            p for p, panel in enumerate(panels)
            if panel[0] <= centre_x <= panel[2] and panel[1] <= centre_y <= panel[3]
        ]
        if inside:
            owned[min(inside, key=lambda p: _area(panels[p]))].append(index)
            continue
        below = [
            (box[1] - panel[3], p)
            for p, panel in enumerate(panels)
            if 0 <= box[1] - panel[3] <= reach
            and min(box[2], panel[2]) - max(box[0], panel[0]) >= 0.5 * (box[2] - box[0])
        ]
        if below:
            owned[min(below)[1]].append(index)
    return [sorted(indexes) for indexes in owned]


def _area(box: Box) -> float:
    return (box[2] - box[0]) * (box[3] - box[1])


# ------------------------------------------------------------------------------ preview
_OUTLINES = ((230, 40, 40), (30, 110, 235), (20, 165, 70), (240, 150, 0), (150, 60, 200))


def draw_overlay(image: Image.Image, boxes: Sequence[Box]) -> Image.Image:
    """The page with each panel outlined and numbered 1, 2, 3 ... (for checking the result)."""
    canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    line = max(4, canvas.width // 400)
    font = ImageFont.load_default(size=max(24, canvas.width // 45))
    for number, box in enumerate(boxes, start=1):
        colour = _OUTLINES[(number - 1) % len(_OUTLINES)]
        draw.rectangle(box, outline=colour, width=line)
        label = str(number)
        left, top, right, bottom = draw.textbbox((box[0] + line, box[1] + line), label, font=font)
        draw.rectangle((left - 6, top - 4, right + 6, bottom + 4), fill=colour)
        draw.text((box[0] + line, box[1] + line), label, fill="white", font=font)
    return canvas


# ------------------------------------------------------------------------------ contact sheet
SHEET_CELL_PX = 420  # each page gets a square cell this big
SHEET_LABEL_PX = 34  # and a caption strip under it
SHEET_BACKGROUND = (43, 42, 40)


@dataclass(frozen=True)
class SheetItem:
    """One page on the contact sheet: its thumbnail, its full size and its panels' boxes."""

    label: str
    thumbnail: Image.Image
    page_size: tuple[int, int]  # the rendered page, in pixels: the boxes' coordinate space
    boxes: Sequence[Box]


def draw_contact_sheet(items: Sequence[SheetItem], columns: int = 4) -> Image.Image:
    """Every page in one picture, each with its panels outlined and numbered, so a whole
    catalogue can be checked at a glance."""
    rows = math.ceil(len(items) / columns)
    cell_h = SHEET_CELL_PX + SHEET_LABEL_PX
    sheet = Image.new("RGB", (columns * SHEET_CELL_PX, rows * cell_h), SHEET_BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=20)
    for index, item in enumerate(items):
        left = (index % columns) * SHEET_CELL_PX
        top = (index // columns) * cell_h
        thumb = item.thumbnail.convert("RGB")
        factor_x, factor_y = thumb.width / item.page_size[0], thumb.height / item.page_size[1]
        scaled = [
            (box[0] * factor_x, box[1] * factor_y, box[2] * factor_x, box[3] * factor_y)
            for box in item.boxes
        ]
        picture = draw_overlay(thumb, scaled)
        picture.thumbnail((SHEET_CELL_PX - 8, SHEET_CELL_PX - 8))
        x = left + (SHEET_CELL_PX - picture.width) // 2
        y = top + (SHEET_CELL_PX - picture.height) // 2
        sheet.paste(picture, (x, y))
        draw.text((left + 12, top + SHEET_CELL_PX + 6), item.label, fill=(235, 232, 226), font=font)
    return sheet

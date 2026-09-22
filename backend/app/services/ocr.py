"""Read all the text on one rendered page.

Catalogue pages are large (a wide page at 150 dpi can be 5000 px across) while text engines
shrink their input to a couple of thousand pixels, which makes small print - model codes,
sizes - unreadable. So the page is cut into overlapping square tiles, each tile is read at
full resolution, and the results are merged back into page coordinates:

* every line's box is moved to the page's pixel coordinates;
* the same line seen in two overlapping tiles (or cut in half by one tile's edge and whole
  in the other) is kept once, preferring the longer reading;
* a line that is still cut by a tile edge (it is wider than the overlap) is read again in a
  thin, wide strip centred on it, so a model code is never silently clipped at a seam;
* the survivors are put into reading order (top to bottom, then left to right).
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass

from PIL import Image, ImageStat

from app.ml.ocr import OcrLine, OcrProvider

logger = logging.getLogger(__name__)

# Engines shrink inputs larger than about this; up to this size a tile is read unshrunk.
ENGINE_COMFORT_PX = 2000
# An axis up to this multiple of the tile size is read as a single tile.
SINGLE_TILE_SLACK = 1.25
# A tile whose pixels barely vary cannot hold text; skipping it saves seconds per page.
BLANK_TILE_STDDEV = 2.0
# Two boxes are the same text when this share of the narrower one lies inside the other
# horizontally and this share of the shorter one's height is shared vertically.
DUPLICATE_OVERLAP = 0.6
SAME_ROW_OVERLAP = 0.5
# A line whose box comes this close to the edge of a tile (that is not the page edge) may
# have been cut off by it.
SEAM_MARGIN_PX = 6
# When re-reading a cut line, the strip extends this many line-heights above and below it.
REPAIR_STRIP_LINES = 1.5
# Lines closer than this fraction of a typical line height (vertically) share a row.
ROW_TOLERANCE = 0.6


@dataclass(frozen=True)
class TextLine:
    """A line of text in page pixel coordinates."""

    text: str
    confidence: float
    polygon: tuple[tuple[float, float], ...]
    bbox: tuple[float, float, float, float]
    language: str | None


# ------------------------------------------------------------------------------ tiling
def _axis(length: int, tile: int, overlap: int) -> tuple[list[int], int]:
    """Tile origins and tile size along one axis.

    An axis only slightly longer than a tile (an A4 page at 150 dpi is 1755 px) is read
    in one piece: splitting it would double the work for no gain. Longer axes get evenly
    spread tiles, each overlapping the next by at least ``overlap`` pixels, the last one
    ending exactly at the page edge.
    """
    if length <= max(tile, min(int(tile * SINGLE_TILE_SLACK), ENGINE_COMFORT_PX)):
        return [0], length
    count = max(2, math.ceil((length - overlap) / (tile - overlap)))
    return [round(i * (length - tile) / (count - 1)) for i in range(count)], tile


def tile_boxes(
    width: int, height: int, tile_px: int, overlap_px: int
) -> list[tuple[int, int, int, int]]:
    """Boxes ``(x0, y0, x1, y1)`` that together cover the page, row by row."""
    if tile_px <= overlap_px:
        raise ValueError("tile size must be larger than the overlap")
    xs, tile_w = _axis(width, tile_px, overlap_px)
    ys, tile_h = _axis(height, tile_px, overlap_px)
    return [(x, y, x + tile_w, y + tile_h) for y in ys for x in xs]


def _is_blank(tile: Image.Image) -> bool:
    return ImageStat.Stat(tile.convert("L")).stddev[0] < BLANK_TILE_STDDEV


# ------------------------------------------------------------------------------ text helpers
def detect_language(text: str) -> str | None:
    """``zh`` / ``en`` / ``mixed`` from the characters used; ``None`` when there are no letters."""
    has_cjk = any("\u4e00" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf" for ch in text)
    has_latin = any(ch.isascii() and ch.isalpha() for ch in text)
    if has_cjk and has_latin:
        return "mixed"
    if has_cjk:
        return "zh"
    if has_latin:
        return "en"
    return None


def _clean(text: str) -> str:
    # NUL is not allowed in PostgreSQL text and is never real content.
    return " ".join(text.replace("\x00", " ").split())


def _has_content(text: str) -> bool:
    """False for stray marks the engine reads from rules, borders and textures."""
    return any(ch.isalnum() for ch in text)


# ------------------------------------------------------------------------------ merging
def _is_same_text(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    """Do two boxes cover the same piece of text?

    True when they share a row (most of the shorter box's height overlaps) and one lies
    mostly inside the other horizontally. Heights are compared separately from widths
    because a line cut through a letter often gets a taller box than the whole line.
    """
    overlap_w = min(a[2], b[2]) - max(a[0], b[0])
    overlap_h = min(a[3], b[3]) - max(a[1], b[1])
    if overlap_w <= 0 or overlap_h <= 0:
        return False
    return (
        overlap_w >= DUPLICATE_OVERLAP * min(a[2] - a[0], b[2] - b[0])
        and overlap_h >= SAME_ROW_OVERLAP * min(a[3] - a[1], b[3] - b[1])
    )


@dataclass(frozen=True)
class _Found:
    line: TextLine
    at_seam: bool  # the line touches the edge of the tile it was read in: it may be cut off


def _drop_duplicates(found: list[_Found]) -> list[_Found]:
    kept: list[_Found] = []
    # Longest reading first, then the most confident: it wins over its duplicates.
    for item in sorted(found, key=lambda f: (-len(f.line.text), -f.line.confidence)):
        if not any(
            _is_same_text(item.line.bbox, other.line.bbox) for other in kept
        ):
            kept.append(item)
    return kept


def _reading_order(lines: list[TextLine]) -> list[TextLine]:
    if not lines:
        return []
    heights = [line.bbox[3] - line.bbox[1] for line in lines]
    band = max(4.0, ROW_TOLERANCE * statistics.median(heights))
    ordered = sorted(lines, key=lambda line: (line.bbox[1] + line.bbox[3]) / 2)
    rows: list[list[TextLine]] = []
    row_centre = 0.0
    for line in ordered:
        centre = (line.bbox[1] + line.bbox[3]) / 2
        if rows and centre - row_centre <= band:
            rows[-1].append(line)
            centres = [(item.bbox[1] + item.bbox[3]) / 2 for item in rows[-1]]
            row_centre = sum(centres) / len(centres)
        else:
            rows.append([line])
            row_centre = centre
    return [line for row in rows for line in sorted(row, key=lambda item: item.bbox[0])]


# ------------------------------------------------------------------------------ entry point
def read_page_text(
    image: Image.Image,
    provider: OcrProvider,
    *,
    min_confidence: float,
    tile_px: int,
    overlap_px: int,
    max_lines: int,
) -> list[TextLine]:
    """All text on the page in reading order, in the page image's pixel coordinates."""
    page = image if image.mode == "RGB" else image.convert("RGB")
    width, height = page.size
    found: list[_Found] = []
    for box in tile_boxes(width, height, tile_px, overlap_px):
        tile = page.crop(box)
        if _is_blank(tile):
            continue
        for raw in provider.recognize(tile):
            line = _to_page_line(raw, box[0], box[1], width, height, min_confidence)
            if line is not None:
                found.append(_Found(line, _touches_seam(raw, box, width, height)))

    kept = _drop_duplicates(found)
    repaired = [
        _repair(item, page, provider, min_confidence, tile_px) if item.at_seam else item
        for item in kept
    ]
    lines = [item.line for item in _drop_duplicates(repaired)]
    if len(lines) > max_lines:
        logger.warning("ocr_line_cap_hit", extra={"found": len(lines), "kept": max_lines})
        lines = sorted(lines, key=lambda item: -item.confidence)[:max_lines]
    return _reading_order(lines)


def _touches_seam(
    raw: OcrLine, box: tuple[int, int, int, int], width: int, height: int
) -> bool:
    """Does the line reach an edge of its tile that is not also the edge of the page?"""
    x0, y0, x1, y1 = raw.bbox
    tile_w, tile_h = box[2] - box[0], box[3] - box[1]
    return (
        (x0 <= SEAM_MARGIN_PX and box[0] > 0)
        or (x1 >= tile_w - SEAM_MARGIN_PX and box[2] < width)
        or (y0 <= SEAM_MARGIN_PX and box[1] > 0)
        or (y1 >= tile_h - SEAM_MARGIN_PX and box[3] < height)
    )


def _repair(
    item: _Found, page: Image.Image, provider: OcrProvider, min_confidence: float, tile_px: int
) -> _Found:
    """Read a possibly cut line again in a strip centred on it (still a modest image, so
    it costs a fraction of a tile). Keeps the original if the strip reads nothing longer."""
    width, height = page.size
    x0, y0, x1, y1 = item.line.bbox
    # Wide enough for the fragment plus half a tile of room on each side (the cut-off part
    # of the line must fit), but never wider than an engine reads without shrinking.
    strip_w = min(width, max(tile_px, ENGINE_COMFORT_PX), round(x1 - x0 + tile_px))
    strip_h = min(height, max(64, round((y1 - y0) * (1 + 2 * REPAIR_STRIP_LINES))))
    left = min(max(0, round((x0 + x1) / 2 - strip_w / 2)), width - strip_w)
    top = min(max(0, round((y0 + y1) / 2 - strip_h / 2)), height - strip_h)
    strip = page.crop((left, top, left + strip_w, top + strip_h))
    best = item.line
    if not _is_blank(strip):
        for raw in provider.recognize(strip):
            candidate = _to_page_line(raw, left, top, width, height, min_confidence)
            if (
                candidate is not None
                and _is_same_text(candidate.bbox, item.line.bbox)
                and len(candidate.text) > len(best.text)
            ):
                best = candidate
    return _Found(best, at_seam=False)


def _to_page_line(
    raw: OcrLine, x0: int, y0: int, width: int, height: int, min_confidence: float
) -> TextLine | None:
    text = _clean(raw.text)
    if raw.confidence < min_confidence or not _has_content(text):
        return None
    polygon = tuple((x + x0, y + y0) for x, y in raw.polygon)
    xs = [x for x, _ in polygon]
    ys = [y for _, y in polygon]
    bbox = (
        max(0.0, min(xs)),
        max(0.0, min(ys)),
        min(float(width), max(xs)),
        min(float(height), max(ys)),
    )
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    return TextLine(
        text=text,
        confidence=raw.confidence,
        polygon=polygon,
        bbox=bbox,
        language=detect_language(text),
    )

"""Finding the products inside one photo panel and preparing their pictures (no database).

For each panel the finder names the pieces of furniture it can see (a desk, an office chair,
a cabinet ...) and the outliner draws the exact edge of each. Every find then gets its own
set of pictures, so later steps (and people) never have to cut anything out again:

* ``crop``      the find's box cut from the panel (with its background)
* ``mask``      black and white, white where the object is (the size of the panel)
* ``cutout``    the object alone on a transparent background, trimmed to its edges (PNG)
* ``white``     the cut-out on plain white (JPEG), the usual shop picture
* ``thumbnail`` a small version of ``white``

An outline the model is unsure about (or that fills little of its box) is kept but marked
LOW_CONFIDENCE, so a person looks at it; an empty one is FAILED and has only the crop.
"""

from __future__ import annotations

import io
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image

from app.ml.detection.base import Detection, DetectionProvider
from app.ml.segmentation.base import Segment, SegmentationProvider
from app.models.enums import SegmentationStatus
from app.services.detections import tidy
from app.services.masks import clip_to_box, coverage
from app.services.pdf import encode_jpeg, make_thumbnail

MASK_THRESHOLD = 127  # a mask picture pixel above this is part of the object


@dataclass(frozen=True)
class Found:
    """One product found in a panel, with the verdict on its outline."""

    detection: Detection
    status: SegmentationStatus
    score: float | None  # the outlining model's own estimate of the outline, 0..1
    coverage: float | None  # the share of the find's box that the outline fills, 0..1
    mask: Image.Image | None  # mode "L", the size of the panel; None when there is no outline
    note: str | None  # why the outline is doubtful or missing (None when it is fine)


@dataclass(frozen=True)
class Renders:
    """The pictures of one find, as file contents. Anything that could not be made is None."""

    crop_jpeg: bytes
    mask_png: bytes | None
    cutout_png: bytes | None
    white_jpeg: bytes | None
    thumbnail_jpeg: bytes | None


def find_products(
    picture: Image.Image,
    finder: DetectionProvider,
    outliner: SegmentationProvider,
    prompts: Sequence[str],
    *,
    min_score: float,
    min_coverage: float,
) -> list[Found]:
    """Everything the models can find in one panel, in reading order."""
    detections = tidy(finder.detect(picture, prompts), picture.size)
    if not detections:
        return []
    outlines = outliner.segment(picture, [d.box for d in detections])
    if len(outlines) != len(detections):
        raise ValueError("The outlining model returned a different number of outlines than boxes")
    return [
        _judge(picture.size, detection, outline, min_score, min_coverage)
        for detection, outline in zip(detections, outlines, strict=True)
    ]


def _judge(
    size: tuple[int, int],
    detection: Detection,
    outline: Segment,
    min_score: float,
    min_coverage: float,
) -> Found:
    width, height = size
    drawn = np.asarray(outline.mask.convert("L")) > MASK_THRESHOLD
    if drawn.shape != (height, width):
        note = "the outline did not fit the picture"
        return Found(detection, SegmentationStatus.FAILED, None, None, None, note)
    inside = clip_to_box(drawn, detection.box)  # whatever lies outside the find's box is dropped
    if not inside.any():
        note = "the outline is empty"
        return Found(detection, SegmentationStatus.FAILED, outline.score, 0.0, None, note)
    share = coverage(inside, detection.box)
    mask = Image.fromarray(inside.astype("uint8") * 255)
    if outline.score < min_score:
        status = SegmentationStatus.LOW_CONFIDENCE
        note = f"the outlining model is not sure of this outline ({outline.score:.0%})"
    elif share < min_coverage:
        status = SegmentationStatus.LOW_CONFIDENCE
        note = f"the outline fills only {share:.0%} of the find's box"
    else:
        status, note = SegmentationStatus.SUCCEEDED, None
    return Found(detection, status, outline.score, share, mask, note)


def _pixel_box(
    box: tuple[float, float, float, float], size: tuple[int, int]
) -> tuple[int, int, int, int]:
    """The whole pixels a box touches, inside the picture and at least one pixel across."""
    width, height = size
    left = min(max(0, math.floor(box[0])), width - 1)
    top = min(max(0, math.floor(box[1])), height - 1)
    right = min(width, max(left + 1, math.ceil(box[2])))
    bottom = min(height, max(top + 1, math.ceil(box[3])))
    return left, top, right, bottom


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, "PNG", optimize=True)
    return buffer.getvalue()


def render_find(
    picture: Image.Image, found: Found, *, jpeg_quality: int, thumbnail_px: int
) -> Renders:
    """Make the pictures of one find from the panel it was found in."""
    rgb = picture.convert("RGB")
    left, top, right, bottom = _pixel_box(found.detection.box, rgb.size)
    crop = rgb.crop((left, top, right, bottom))
    crop_jpeg = encode_jpeg(crop, jpeg_quality)
    if found.mask is None:
        return Renders(crop_jpeg, None, None, None, None)

    alpha = found.mask.convert("L").crop((left, top, right, bottom))
    covered = np.argwhere(np.asarray(alpha) > MASK_THRESHOLD)
    if covered.size == 0:  # cannot happen for a judged find, but never divide by nothing
        return Renders(crop_jpeg, _png(found.mask), None, None, None)
    (row0, col0), (row1, col1) = covered.min(axis=0), covered.max(axis=0) + 1
    trimmed = (int(col0), int(row0), int(col1), int(row1))
    cutout = crop.convert("RGBA").crop(trimmed)
    cutout.putalpha(alpha.crop(trimmed))
    white = Image.new("RGB", cutout.size, (255, 255, 255))
    white.paste(cutout, mask=cutout.getchannel("A"))
    return Renders(
        crop_jpeg=crop_jpeg,
        mask_png=_png(found.mask),
        cutout_png=_png(cutout),
        white_jpeg=encode_jpeg(white, jpeg_quality),
        thumbnail_jpeg=encode_jpeg(make_thumbnail(white, thumbnail_px), jpeg_quality),
    )

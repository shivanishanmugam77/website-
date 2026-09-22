"""Pictures for checking what the finder found: boxes, labels and (optionally) outlines."""

from __future__ import annotations

import math
from collections.abc import Sequence

from PIL import Image, ImageDraw, ImageFont

from app.ml.detection.base import Detection

_COLOURS = (
    (230, 40, 40),
    (30, 110, 235),
    (20, 165, 70),
    (240, 150, 0),
    (150, 60, 200),
    (0, 160, 180),
)
SHEET_BACKGROUND = (43, 42, 40)


def draw_detections(
    image: Image.Image,
    detections: Sequence[Detection],
    masks: Sequence[Image.Image | None] = (),
) -> Image.Image:
    """The picture with each find outlined and named ("office chair 82%").

    ``masks`` (one per detection, same order, black and white) tint the exact outline of each
    object in its colour so the cut can be judged before it is used.
    """
    canvas = image.convert("RGB")
    line = max(3, canvas.width // 250)
    font = ImageFont.load_default(size=max(16, canvas.width // 35))
    for index, detection in enumerate(detections):
        colour = _COLOURS[index % len(_COLOURS)]
        mask = masks[index] if index < len(masks) else None
        if mask is not None:
            tint = mask.convert("L").resize(canvas.size).point(lambda v: 110 if v > 127 else 0)
            canvas.paste(Image.new("RGB", canvas.size, colour), mask=tint)
    draw = ImageDraw.Draw(canvas)
    for index, detection in enumerate(detections):
        colour = _COLOURS[index % len(_COLOURS)]
        x0, y0, x1, y1 = detection.box
        draw.rectangle(detection.box, outline=colour, width=line)
        text = f"{detection.label} {detection.confidence:.0%}"
        left, top, right, bottom = draw.textbbox((x0 + line, y0 + line), text, font=font)
        draw.rectangle((left - 4, top - 3, right + 4, bottom + 3), fill=colour)
        draw.text((x0 + line, y0 + line), text, fill="white", font=font)
    return canvas


def draw_sheet(
    items: Sequence[tuple[str, Image.Image]], columns: int = 3, cell: int = 520
) -> Image.Image:
    """Several annotated pictures in one, each with a caption underneath."""
    if not items:
        raise ValueError("Nothing to draw")
    rows = math.ceil(len(items) / columns)
    caption = 34
    sheet = Image.new("RGB", (columns * cell, rows * (cell + caption)), SHEET_BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=20)
    for index, (label, picture) in enumerate(items):
        left, top = (index % columns) * cell, (index // columns) * (cell + caption)
        fitted = picture.convert("RGB")
        fitted.thumbnail((cell - 8, cell - 8))
        sheet.paste(fitted, (left + (cell - fitted.width) // 2, top + (cell - fitted.height) // 2))
        draw.text((left + 12, top + cell + 6), label, fill=(235, 232, 226), font=font)
    return sheet

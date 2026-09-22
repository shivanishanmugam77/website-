"""The pictures used to check what the finder found."""

from __future__ import annotations

import pytest
from PIL import Image

from app.ml.detection.base import Detection
from app.services.detection_preview import draw_detections, draw_sheet

PAPER = (243, 241, 238)


def test_each_find_is_outlined_and_the_original_is_left_alone() -> None:
    picture = Image.new("RGB", (800, 600), PAPER)
    result = draw_detections(picture, [Detection("desk", 0.82, (100, 150, 500, 450))])
    assert result.size == picture.size
    assert result.getpixel((100, 300)) != PAPER  # the left edge of the box is drawn
    assert result.getpixel((300, 300)) == PAPER  # the inside is untouched
    assert picture.getpixel((100, 300)) == PAPER  # and so is the original picture


def test_an_outline_tints_only_the_object() -> None:
    picture = Image.new("RGB", (400, 300), PAPER)
    mask = Image.new("L", (400, 300), 0)
    mask.paste(255, (150, 100, 250, 200))  # the object is a square in the middle
    result = draw_detections(picture, [Detection("desk", 0.9, (100, 50, 300, 250))], [mask])
    inside, outside = result.getpixel((200, 150)), result.getpixel((120, 220))
    assert inside != PAPER and outside == PAPER


def test_a_missing_outline_is_simply_not_drawn() -> None:
    picture = Image.new("RGB", (400, 300), PAPER)
    result = draw_detections(
        picture, [Detection("desk", 0.9, (100, 50, 300, 250))], [None]
    )
    assert result.getpixel((200, 150)) == PAPER


def test_the_sheet_holds_every_picture_in_a_grid_with_captions() -> None:
    items = [(f"Panel {n}", Image.new("RGB", (300, 200), PAPER)) for n in range(1, 6)]
    sheet = draw_sheet(items, columns=3, cell=200)
    assert sheet.size == (600, 2 * (200 + 34))


def test_an_empty_sheet_is_refused() -> None:
    with pytest.raises(ValueError):
        draw_sheet([])

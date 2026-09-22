"""Stand-in text engines for tests (test doubles: nothing here is used by the application).

``SceneOcr`` behaves like a real engine on a page whose text we know exactly. It works out
where in the page each crop it is handed comes from (the test page encodes its own
coordinates in its pixel colours) and returns the lines visible in that crop - including the
half-line a real engine returns when a crop edge cuts through text. That makes tiling,
de-duplication and seam repair testable without a real engine and without flakiness.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PIL import Image

from app.ml.ocr import OcrLine


def coordinate_page(width: int, height: int) -> Image.Image:
    """A page where pixel (x, y) is coloured (x % 256, y % 256, x // 256 + 16 * (y // 256)).

    Every crop's top-left pixel therefore tells us where the crop was taken. Not blank,
    so the tiler never skips it. Limited to 4096 x 4096.
    """
    assert width <= 4096 and height <= 4096
    ramp = bytes(range(256))
    red_row = (ramp * (width // 256 + 1))[:width]
    red = Image.frombytes("L", (width, height), red_row * height)
    green = Image.frombytes(
        "L", (width, height), b"".join(bytes([y % 256]) * width for y in range(height))
    )
    blue_rows = []
    for band in range(height // 256 + 1):
        row = bytes((x // 256) + 16 * band for x in range(width))
        blue_rows.append(row * min(256, height - band * 256) if band * 256 < height else b"")
    blue = Image.frombytes("L", (width, height), b"".join(blue_rows))
    return Image.merge("RGB", (red, green, blue))


def crop_origin(image: Image.Image) -> tuple[int, int]:
    red, green, blue = image.getpixel((0, 0))
    return (blue % 16) * 256 + red, (blue // 16) * 256 + green


@dataclass(frozen=True)
class SceneLine:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    confidence: float = 0.95


class SceneOcr:
    """Reads the known ``lines`` of a :func:`coordinate_page`, honouring crop edges."""

    name = "scene"
    version = "test"

    def __init__(self, lines: list[SceneLine]) -> None:
        self.lines = lines
        self.crops: list[tuple[int, int, int, int]] = []  # every crop it was asked to read

    def recognize(self, image: Image.Image) -> list[OcrLine]:
        ox, oy = crop_origin(image)
        width, height = image.size
        self.crops.append((ox, oy, ox + width, oy + height))
        found = []
        for line in self.lines:
            x0, x1 = max(line.x0, ox), min(line.x1, ox + width)
            y0, y1 = max(line.y0, oy), min(line.y1, oy + height)
            if x1 <= x0 or y1 <= y0:
                continue  # not in this crop
            if (y1 - y0) < 0.5 * (line.y1 - line.y0):
                continue  # a sliver of a line is unreadable
            span = line.x1 - line.x0
            first = round(len(line.text) * (x0 - line.x0) / span)
            last = round(len(line.text) * (x1 - line.x0) / span)
            text = line.text[first:last]
            if not text:
                continue
            local = ((x0 - ox, y0 - oy), (x1 - ox, y0 - oy), (x1 - ox, y1 - oy), (x0 - ox, y1 - oy))
            found.append(OcrLine(text, line.confidence, local))
        return found


class FunctionOcr:
    """Returns whatever ``respond(call_index, image)`` says; counts its calls."""

    name = "function"
    version = "test"

    def __init__(self, respond: Callable[[int, Image.Image], list[OcrLine]]) -> None:
        self.respond = respond
        self.calls = 0

    def recognize(self, image: Image.Image) -> list[OcrLine]:
        index = self.calls
        self.calls += 1
        return self.respond(index, image)


def line(text: str, x0: float, y0: float, x1: float, y1: float, confidence: float = 0.95) -> OcrLine:
    """An engine result with an axis-aligned box."""
    return OcrLine(text, confidence, ((x0, y0), (x1, y0), (x1, y1), (x0, y1)))

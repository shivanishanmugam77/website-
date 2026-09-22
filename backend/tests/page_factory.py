"""Synthetic catalogue pages for testing panel detection.

Photos are noisy colour blocks (like real photographs they have no flat regions), the paper
is off-white, and captions are real rendered text. Nothing is random between runs.
"""

from __future__ import annotations

import random

from PIL import Image, ImageDraw, ImageFont

PAPER = (243, 241, 238)


def blank_page(width: int, height: int, paper: tuple[int, int, int] = PAPER) -> Image.Image:
    return Image.new("RGB", (width, height), paper)


def photo(
    width: int, height: int, seed: int = 0, tone: tuple[int, int, int] = (150, 105, 75)
) -> Image.Image:
    """Something that looks like a photograph to a pixel-level detector: broad patches of
    different tones (objects, walls, shadows) plus a little grain, and nothing paper-coloured."""
    rng = random.Random(seed)
    image = Image.new("RGB", (width, height), tone)
    draw = ImageDraw.Draw(image)
    for _ in range(60):
        w = rng.randint(width // 30, width // 3)
        h = rng.randint(height // 30, height // 3)
        x, y = rng.randint(-w // 2, width), rng.randint(-h // 2, height)
        colour = tuple(max(0, min(200, c + rng.randint(-70, 70))) for c in tone)
        if rng.random() < 0.5:
            draw.rectangle((x, y, x + w, y + h), fill=colour)
        else:
            draw.ellipse((x, y, x + w, y + h), fill=colour)
    grain = Image.frombytes("L", (width, height), rng.randbytes(width * height))
    return Image.blend(image, grain.convert("RGB"), 0.08)


def place_photo(
    page: Image.Image, box: tuple[int, int, int, int], seed: int = 0, **kwargs  # noqa: ANN003
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    page.paste(photo(x1 - x0, y1 - y0, seed, **kwargs), (x0, y0))
    return box


def light_wall_photo(page: Image.Image, box: tuple[int, int, int, int], seed: int = 0) -> None:
    """A photo with a paper-coloured wall across most (not all) of its width, like a room
    shot where furniture and window frames break up the wall."""
    x0, y0, x1, y1 = box
    place_photo(page, box, seed)
    middle = (y0 + y1) // 2
    ImageDraw.Draw(page).rectangle(
        (x0 + 200, middle - 150, x1 - 300, middle + 150), fill=(236, 234, 231)
    )


def framed_print_photo(
    page: Image.Image, box: tuple[int, int, int, int], seed: int = 0, poke: int = 0
) -> None:
    """A photo with a framed print on its wall: a paper-coloured mat with a dark picture in it,
    so the print is cut off from the rest of the photo by paper-coloured pixels.

    With ``poke`` the top ``poke`` pixels of the photo (except the print) are paper-coloured
    too, so the print reaches that far above the rest of the photo, as one does in a real
    room photo whose top edge is pale.
    """
    x0, y0, x1, y1 = box
    place_photo(page, box, seed)
    draw = ImageDraw.Draw(page)
    if poke:
        draw.rectangle((x0, y0, x1, y0 + poke), fill=PAPER)
    draw.rectangle((x0 + 800, y0, x0 + 1100, y0 + 500), fill=(238, 236, 233))  # the mat
    draw.rectangle((x0 + 850, y0 + (0 if poke else 200), x0 + 1050, y0 + 450), fill=(30, 70, 70))  # the art


def draw_text(
    page: Image.Image, xy: tuple[int, int], text: str, size: int = 28, fill=(40, 40, 40)  # noqa: ANN001
) -> tuple[int, int, int, int]:
    draw = ImageDraw.Draw(page)
    font = ImageFont.load_default(size=size)
    draw.text(xy, text, fill=fill, font=font)
    return draw.textbbox(xy, text, font=font)


def draw_chair(page: Image.Image, x: int, y: int, width: int = 140, height: int = 220) -> tuple:
    """A simple chair on the paper: an orange back, a grey seat and a dark base."""
    draw = ImageDraw.Draw(page)
    draw.rounded_rectangle((x + 20, y, x + width - 20, y + height * 0.5), 20, fill=(205, 110, 50))
    draw.ellipse((x, y + height * 0.5, x + width, y + height * 0.68), fill=(120, 120, 125))
    draw.rectangle((x + width * 0.46, y + height * 0.68, x + width * 0.54, y + height * 0.9), fill=(60, 60, 60))
    draw.polygon(
        [(x + 10, y + height), (x + width - 10, y + height), (x + width * 0.5, y + height * 0.88)],
        fill=(60, 60, 60),
    )
    return (x, y, x + width, y + height)


def iou(a: tuple, b: tuple) -> float:
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    inter = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union

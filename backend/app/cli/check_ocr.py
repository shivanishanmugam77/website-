"""Check that the text-recognition engine works, without involving any catalogue:

    docker compose run --rm backend python -m app.cli.check_ocr

It draws a small test picture (a model code, a size and a series label), reads it back with
the configured engine and prints what it found. Run it after changing the engine or the
Docker image; it takes a few seconds and needs no database.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Sequence

from PIL import Image, ImageDraw, ImageFont

from app.core.config import get_settings
from app.core.logging_config import configure_logging
from app.ml.ocr import OcrEngineError, OcrProvider, get_ocr_provider
from app.services.ocr import read_page_text
from app.services.text_signals import extract_signals

TEST_LINES = ("MODEL: YY-21 MEETING TABLE", "3200W x 1400D x 750H", "09 series")


def draw_test_picture() -> Image.Image:
    image = Image.new("RGB", (1400, 420), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=64)
    for row, text in enumerate(TEST_LINES):
        draw.text((40, 30 + row * 120), text, fill="black", font=font)
    return image


def main(argv: Sequence[str] | None = None, provider: OcrProvider | None = None) -> int:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    print(f"Text recognition engine: {settings.ocr_provider}")
    try:
        started = time.perf_counter()
        engine = provider or get_ocr_provider(settings)
        print(f"Engine started in {time.perf_counter() - started:.1f}s (version {engine.version})")
        started = time.perf_counter()
        lines = read_page_text(
            draw_test_picture(),
            engine,
            min_confidence=settings.ocr_min_confidence,
            tile_px=settings.ocr_tile_px,
            overlap_px=settings.ocr_tile_overlap_px,
            max_lines=settings.ocr_max_lines_per_page,
        )
        print(f"Read the test picture in {time.perf_counter() - started:.1f}s:")
    except OcrEngineError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    for line in lines:
        print(f"  {line.confidence:.2f}  {line.text}")
    signals = extract_signals([line.text for line in lines])
    codes = [item.code for item in signals.model_codes]
    sizes = [item.raw_text for item in signals.dimensions]
    print(f"Model codes found: {codes or 'none'}")
    print(f"Sizes found:       {sizes or 'none'}")
    print(f"Series found:      {[item.label for item in signals.series] or 'none'}")

    if "YY-21" in codes and sizes:
        print("OK: the engine reads text correctly.")
        return 0
    print("PROBLEM: the engine started but did not read the test picture correctly.")
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

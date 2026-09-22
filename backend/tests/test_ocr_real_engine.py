"""The real text engine, end to end on a picture with known text.

Skipped where the engine is not installed (a bare development machine). In the Docker image
and in CI it is installed, so this is the test that proves the engine really works there.
"""

from __future__ import annotations

import pytest

from app.cli.check_ocr import draw_test_picture
from app.services.ocr import read_page_text
from app.services.text_signals import extract_signals

pytest.importorskip("rapidocr_onnxruntime")


def test_the_engine_reads_a_model_code_a_size_and_a_series_label() -> None:
    from app.ml.ocr.rapidocr_provider import RapidOcrProvider

    provider = RapidOcrProvider()
    assert provider.name == "rapidocr" and provider.version

    lines = read_page_text(
        draw_test_picture(),
        provider,
        min_confidence=0.5,
        tile_px=1600,
        overlap_px=320,
        max_lines=100,
    )
    texts = [line.text for line in lines]
    signals = extract_signals(texts)

    assert "YY-21" in [c.code for c in signals.model_codes], f"engine read: {texts}"
    assert any(
        (d.width, d.depth, d.height) == (3200, 1400, 750) for d in signals.dimensions
    ), f"engine read: {texts}"
    assert [s.label for s in signals.series] == ["09 series"], f"engine read: {texts}"
    assert all(0.0 <= line.confidence <= 1.0 for line in lines)
    assert all(line.bbox[2] > line.bbox[0] and line.bbox[3] > line.bbox[1] for line in lines)

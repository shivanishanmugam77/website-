"""The engine interface, the provider factory and the RapidOCR result parser.

The parser and factory are tested without the engine installed; the engine itself is
exercised in test_ocr_real_engine.py.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.ml.ocr import OcrEngineError, OcrLine, get_ocr_provider
from app.ml.ocr.rapidocr_provider import _to_line

BOX = [[10.0, 20.0], [110.0, 20.0], [110.0, 50.0], [10.0, 50.0]]


def test_a_line_knows_its_bounding_box() -> None:
    rotated = OcrLine("x", 0.9, ((10.0, 12.0), (100.0, 8.0), (104.0, 40.0), (14.0, 44.0)))
    assert rotated.bbox == (10.0, 8.0, 104.0, 44.0)


@pytest.mark.parametrize("score", ["0.9876", 0.9876, "1", 1])
def test_rapidocr_scores_may_be_strings_or_numbers(score: object) -> None:
    line = _to_line([BOX, "YY-21", score])
    assert line is not None and line.text == "YY-21"
    assert line.confidence == pytest.approx(float(score))  # type: ignore[arg-type]
    assert line.polygon[0] == (10.0, 20.0) and line.bbox == (10.0, 20.0, 110.0, 50.0)


def test_scores_are_kept_within_zero_and_one() -> None:
    assert _to_line([BOX, "x", "1.5"]).confidence == 1.0  # type: ignore[union-attr]
    assert _to_line([BOX, "x", "-0.2"]).confidence == 0.0  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "broken",
    [None, [], [BOX], [BOX, "text"], [BOX, "text", "not-a-number"], [[[1, 2]], "text", 0.9], [BOX, 5, 0.9]],
)
def test_malformed_engine_results_are_skipped_not_fatal(broken: object) -> None:
    assert _to_line(broken) is None


def test_an_unknown_provider_is_reported_clearly() -> None:
    with pytest.raises(OcrEngineError, match="Unknown OCR provider"):
        get_ocr_provider(SimpleNamespace(ocr_provider="nonexistent"))  # type: ignore[arg-type]

"""The ``check_ocr`` self-test command, with stand-in engines."""

from __future__ import annotations

from app.cli import check_ocr
from tests.ocr_fakes import FunctionOcr, line

GOOD = [
    line("MODEL: YY-21 MEETING TABLE", 40, 30, 900, 100),
    line("3200W x 1400D x 750H", 40, 150, 700, 220),
    line("09 series", 40, 270, 400, 340),
]


def test_a_working_engine_passes(capsys) -> None:  # noqa: ANN001
    engine = FunctionOcr(lambda _i, _img: list(GOOD))
    assert check_ocr.main([], provider=engine) == 0
    output = capsys.readouterr().out
    assert "YY-21" in output and "OK: the engine reads text correctly." in output


def test_an_engine_that_reads_nothing_is_reported_as_a_problem(capsys) -> None:  # noqa: ANN001
    assert check_ocr.main([], provider=FunctionOcr(lambda _i, _img: [])) == 1
    assert "PROBLEM" in capsys.readouterr().out


def test_an_engine_that_cannot_start_is_reported_clearly(monkeypatch, capsys) -> None:  # noqa: ANN001
    from app.ml.ocr import OcrEngineError

    def broken(_settings):  # noqa: ANN001, ANN202
        raise OcrEngineError("package missing")

    monkeypatch.setattr(check_ocr, "get_ocr_provider", broken)
    assert check_ocr.main([]) == 1
    assert "FAILED: package missing" in capsys.readouterr().err

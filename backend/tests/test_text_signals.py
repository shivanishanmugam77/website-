"""Model codes, sizes and series labels picked out of recognised text."""

from __future__ import annotations

import pytest

from app.services.text_signals import extract_signals


def codes(*lines: str) -> list[str]:
    return [item.code for item in extract_signals(list(lines)).model_codes]


def sizes(*lines: str):  # noqa: ANN201
    return extract_signals(list(lines)).dimensions


# ------------------------------------------------------------------------------ model codes
@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("\u578b\u53f7: YY-15", ["YY-15"]),  # 型号: in Chinese
        ("\u578b\u53f7\uff1aYY-15", ["YY-15"]),  # full-width colon
        ("\u578b\u53f7 : YY\u201321 \u4f1a\u8bae\u684c", ["YY-21"]),  # en dash, text after
        ("\u578b\u53f7:YY-21\u4f1a\u8bae\u684c", ["YY-21"]),  # no spaces at all
        ("MODEL: AB-123C", ["AB-123C"]),
        ("Item No. XZ-9", ["XZ-9"]),
        ("YY-21", ["YY-21"]),  # a bare code
        ("yy-21", []),  # bare codes must be upper case: too many false hits otherwise
        ("Model of the year", []),  # a label followed by a plain word
        ("Tel: 138-1234-5678", []),
        ("Windows 10", []),
        ("", []),
    ],
)
def test_model_codes(line: str, expected: list[str]) -> None:
    assert codes(line) == expected


def test_a_code_is_reported_once_and_remembers_its_line() -> None:
    signals = extract_signals(["intro", "MODEL: YY-15 (YY-15)", "YY-21"])
    assert [(c.code, c.line_index, c.labelled) for c in signals.model_codes] == [
        ("YY-15", 1, True),
        ("YY-21", 2, False),
    ]


# ------------------------------------------------------------------------------ sizes
@pytest.mark.parametrize(
    "line",
    [
        "3200Wx1400Dx750H",
        "3200W x 1400D x 750H",
        "3200W\u00d71400D\u00d7750H",  # multiplication sign
        "3200W*1400D*750H",
        "\u4e24\u4eba\u4f4d\u804c\u5458\u684c 3200Wx1400Dx750H",  # Chinese words before it
        "W3200 x D1400 x H750",  # letters in front
    ],
)
def test_labelled_sizes(line: str) -> None:
    (size,) = sizes(line)
    assert (size.width, size.depth, size.height) == (3200, 1400, 750)
    assert size.length is None and size.labelled and size.unit == "mm"


def test_unlabelled_sizes_are_assumed_to_be_width_depth_height() -> None:
    (size,) = sizes("1200x600x750")
    assert (size.width, size.depth, size.height, size.labelled) == (1200, 600, 750, False)


def test_labels_decide_which_number_is_which() -> None:
    (size,) = sizes("H750 x W1200 x D600")
    assert (size.width, size.depth, size.height) == (1200, 600, 750)
    (size,) = sizes("L1200*W600*H750 mm")
    assert (size.length, size.width, size.height) == (1200, 600, 750)


def test_contradictory_labels_are_ignored_in_favour_of_the_order() -> None:
    (size,) = sizes("1200W x 600W x 750H")  # two widths: the letters cannot be trusted
    assert (size.width, size.depth, size.height, size.labelled) == (1200, 600, 750, False)


def test_units_and_source_text_are_kept() -> None:
    (size,) = sizes("Size 120 x 60 x 75 cm")
    assert size.unit == "cm" and size.raw_text == "120 x 60 x 75 cm"


@pytest.mark.parametrize(
    "line",
    ["W x D x H", "5W x 6D x 7H", "1 x 2 x 3", "Tel 0755 8888 9999", "99999x99999x99999", "1200 x 600"],
)
def test_things_that_are_not_sizes_are_ignored(line: str) -> None:
    assert sizes(line) == []


def test_two_sizes_on_one_line_are_both_found() -> None:
    found = sizes("1200x600x750 / 1400x700x750")
    assert [(s.width, s.depth) for s in found] == [(1200, 600), (1400, 700)]


# ------------------------------------------------------------------------------ series
@pytest.mark.parametrize("line", ["09series", "09 series", "09 SERIES", "09-Series", "09\u7cfb\u5217"])
def test_series_labels(line: str) -> None:
    (label,) = extract_signals([line]).series
    assert label.label == "09 series" and label.line_index == 0


def test_text_without_a_series_number_is_not_a_series() -> None:
    assert extract_signals(["Our new series of chairs", "series"]).series == []


def test_results_can_carry_line_numbers_from_a_larger_list() -> None:
    signals = extract_signals(["MODEL: YY-15", "1200x600x750"], indexes=[7, 12])
    assert signals.model_codes[0].line_index == 7
    assert signals.dimensions[0].line_index == 12

"""Pick product facts out of recognised text lines.

These are plain pattern rules, not machine learning: they recognise the *shapes* that
supplier catalogues print - a model code such as ``YY-21``, a size such as
``3200W x 1400D x 750H`` and a series label such as ``09 series``. They are deliberately
forgiving about OCR noise (``x``/``X``/``×``/``*`` between numbers, full-width colons,
dashes of every length).

Everything here is a *hint*: the results say which line each fact came from so a later
stage (or an admin) can check it. Nothing is written to the product record from here.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

# Sizes outside this range (in the printed unit) are not furniture: phone numbers, years...
MIN_DIMENSION = 20.0
MAX_DIMENSION = 20000.0

_DASHES = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-"})


@dataclass(frozen=True)
class ModelCode:
    code: str
    line_index: int
    labelled: bool  # True when preceded by a word such as 型号 / Model


@dataclass(frozen=True)
class Dimension:
    """A size. Values are in ``unit``; ``labelled`` is False when W/D/H letters were absent
    and the conventional width x depth x height order was assumed."""

    width: float | None
    depth: float | None
    height: float | None
    length: float | None
    unit: str
    labelled: bool
    raw_text: str
    line_index: int


@dataclass(frozen=True)
class SeriesLabel:
    label: str
    line_index: int


@dataclass
class TextSignals:
    model_codes: list[ModelCode] = field(default_factory=list)
    dimensions: list[Dimension] = field(default_factory=list)
    series: list[SeriesLabel] = field(default_factory=list)


# ------------------------------------------------------------------------------ patterns
_LABEL = r"[WLDH\u5bbd\u957f\u6df1\u9ad8]"  # W L D H and the Chinese 宽 长 深 高
_NUMBER = r"\d{2,5}(?:\.\d)?"
_SEPARATOR = r"\s*[xX\u00d7\u2715*\uff0a]\s*"


def _part(n: int) -> str:
    return rf"(?:(?P<p{n}>{_LABEL})\s*)?(?P<n{n}>{_NUMBER})\s*(?P<s{n}>{_LABEL})?"


_DIMENSION = re.compile(
    rf"(?<![\d.]){_part(1)}{_SEPARATOR}{_part(2)}{_SEPARATOR}{_part(3)}"
    r"(?:\s*(?P<unit>mm|cm))?(?![\d])",
    re.IGNORECASE,
)
_LABEL_FIELDS = {
    "W": "width",
    "\u5bbd": "width",
    "D": "depth",
    "\u6df1": "depth",
    "H": "height",
    "\u9ad8": "height",
    "L": "length",
    "\u957f": "length",
}
_DEFAULT_ORDER = ("width", "depth", "height")

_LABELLED_CODE = re.compile(
    r"(?:\u578b\u53f7|\u6b3e\u53f7|\u8d27\u53f7|model|item\s*no\.?|code)\s*[:\uff1a#]?\s*"
    r"(?P<code>[A-Za-z0-9][A-Za-z0-9\-_/.]{1,23})",
    re.IGNORECASE,
)
_BARE_CODE = re.compile(r"(?<![A-Za-z0-9])([A-Z]{1,4}-\d{1,4}[A-Z]{0,2})(?![A-Za-z0-9])")

_SERIES = re.compile(
    r"(?<![\d])(?P<number>\d{1,3})\s*[- ]?\s*(?:series|\u7cfb\u5217)", re.IGNORECASE
)


# ------------------------------------------------------------------------------ parsers
def _clean_code(code: str) -> str:
    return code.translate(_DASHES).strip("-_/. ").upper()


def _find_codes(text: str, line_index: int) -> list[ModelCode]:
    text = text.translate(_DASHES)
    found: list[ModelCode] = []
    seen: set[str] = set()
    for match in _LABELLED_CODE.finditer(text):
        code = _clean_code(match.group("code"))
        # A label followed by something without a digit is a word ("model of ..."), not a code.
        if any(ch.isdigit() for ch in code) and code not in seen:
            seen.add(code)
            found.append(ModelCode(code, line_index, labelled=True))
    for match in _BARE_CODE.finditer(text):
        code = _clean_code(match.group(1))
        if code not in seen:
            seen.add(code)
            found.append(ModelCode(code, line_index, labelled=False))
    return found


def _find_dimensions(text: str, line_index: int) -> list[Dimension]:
    found: list[Dimension] = []
    for match in _DIMENSION.finditer(text):
        values = [float(match.group(f"n{n}").replace(",", ".")) for n in (1, 2, 3)]
        if not all(MIN_DIMENSION <= value <= MAX_DIMENSION for value in values):
            continue
        labels = [(match.group(f"p{n}") or match.group(f"s{n}") or "").upper() for n in (1, 2, 3)]
        fields = [_LABEL_FIELDS.get(label) for label in labels]
        named = [name for name in fields if name]
        usable = len(named) == len(set(named))  # "W, W, H" is noise: fall back to the order
        assigned: dict[str, float] = {}
        if usable:
            for name, value in zip(fields, values, strict=True):
                if name:
                    assigned[name] = value
        free = [name for name in _DEFAULT_ORDER if name not in assigned]
        for name, value in zip(fields, values, strict=True):
            if (not usable or not name) and free:
                assigned[free.pop(0)] = value
        found.append(
            Dimension(
                width=assigned.get("width"),
                depth=assigned.get("depth"),
                height=assigned.get("height"),
                length=assigned.get("length"),
                unit=(match.group("unit") or "mm").lower(),
                labelled=usable and len(named) == 3,
                raw_text=match.group(0).strip(),
                line_index=line_index,
            )
        )
    return found


def _find_series(text: str, line_index: int) -> list[SeriesLabel]:
    return [
        SeriesLabel(f"{match.group('number')} series", line_index)
        for match in _SERIES.finditer(text)
    ]


def extract_signals(lines: Sequence[str], indexes: Sequence[int] | None = None) -> TextSignals:
    """Facts found in ``lines``; each result carries the index of the line it came from.

    ``indexes`` gives each line's number in a larger list (e.g. the whole page) when
    ``lines`` is only part of it; results then carry those numbers instead of 0, 1, 2 ...
    """
    signals = TextSignals()
    for position, text in enumerate(lines):
        index = indexes[position] if indexes is not None else position
        signals.model_codes.extend(_find_codes(text, index))
        signals.dimensions.extend(_find_dimensions(text, index))
        signals.series.extend(_find_series(text, index))
    return signals

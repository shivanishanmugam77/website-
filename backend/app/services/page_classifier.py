"""Decide what kind of page this is, from the text found on it.

This is a rule-based first pass, **not** a vision model: it can tell product pages (they
print model codes and sizes) from covers, and nothing finer. Anything it cannot justify
from the text is ``UNKNOWN`` with a low confidence, so that later stages (image analysis
in Phase 5) know they still have to look. The confidence values are the rules' own
certainty, not a statistical probability.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.models.enums import PageType
from app.services.text_signals import TextSignals

CONFIDENCE_PRODUCT_CODE_AND_SIZE = 0.9
CONFIDENCE_PRODUCT_ONE_SIGNAL = 0.75
CONFIDENCE_COVER = 0.7
CONFIDENCE_COVER_NO_TEXT = 0.5
CONFIDENCE_UNKNOWN = 0.3


@dataclass(frozen=True)
class PageClassification:
    page_type: PageType
    confidence: float
    reason: str


def classify_page(*, page_number: int, line_count: int, signals: TextSignals) -> PageClassification:
    has_codes = bool(signals.model_codes)
    has_sizes = bool(signals.dimensions)
    if has_codes and has_sizes:
        return PageClassification(
            PageType.PRODUCT, CONFIDENCE_PRODUCT_CODE_AND_SIZE, "model codes and sizes found"
        )
    if has_codes or has_sizes:
        found = "model codes" if has_codes else "sizes"
        return PageClassification(
            PageType.PRODUCT, CONFIDENCE_PRODUCT_ONE_SIGNAL, f"{found} found, no other product text"
        )
    if page_number == 1:
        if line_count:
            return PageClassification(
                PageType.COVER, CONFIDENCE_COVER, "first page without product text"
            )
        return PageClassification(
            PageType.COVER, CONFIDENCE_COVER_NO_TEXT, "first page without any readable text"
        )
    return PageClassification(PageType.UNKNOWN, CONFIDENCE_UNKNOWN, "no product text found")

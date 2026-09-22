"""What kind of page is it, judged from its text."""

from __future__ import annotations

from app.models.enums import PageType
from app.services.page_classifier import classify_page
from app.services.text_signals import extract_signals


def classify(number: int, *lines: str):  # noqa: ANN201
    return classify_page(
        page_number=number, line_count=len(lines), signals=extract_signals(list(lines))
    )


def test_codes_and_sizes_together_make_a_product_page() -> None:
    result = classify(8, "\u578b\u53f7: YY-21", "3200Wx1400Dx750H")
    assert result.page_type is PageType.PRODUCT and result.confidence == 0.9


def test_one_kind_of_product_text_is_enough_but_less_certain() -> None:
    only_code = classify(8, "MODEL: YY-21")
    only_size = classify(8, "1200Wx1200Dx750H")
    assert only_code.page_type is PageType.PRODUCT and only_size.page_type is PageType.PRODUCT
    assert only_code.confidence == only_size.confidence == 0.75


def test_a_first_page_without_product_text_is_a_cover() -> None:
    result = classify(1, "SERVICE PROVIDER FOR COMMERCIAL SPACE")
    assert result.page_type is PageType.COVER and result.confidence == 0.7


def test_a_first_page_with_no_text_at_all_is_still_probably_a_cover() -> None:
    result = classify(1)
    assert result.page_type is PageType.COVER and result.confidence < 0.7


def test_a_first_page_that_shows_products_is_a_product_page() -> None:
    assert classify(1, "MODEL: YY-21", "3200Wx1400Dx750H").page_type is PageType.PRODUCT


def test_other_pages_without_product_text_are_unknown_not_guessed() -> None:
    result = classify(5, "Thank you for your support")
    assert result.page_type is PageType.UNKNOWN and result.confidence == 0.3
    assert classify(5).page_type is PageType.UNKNOWN

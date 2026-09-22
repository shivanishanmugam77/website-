from __future__ import annotations

import pytest

from app.utils.slugs import slugify


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Office Chairs", "office-chairs"),
        ("  Pendant   Lights!! ", "pendant-lights"),
        ("Café Table", "cafe-table"),
        ("A/B & C", "a-b-c"),
        ("办公椅", ""),  # Chinese-only: caller must provide a fallback
        ("YY-11 办公椅", "yy-11"),
    ],
)
def test_slugify(value: str, expected: str) -> None:
    assert slugify(value) == expected


def test_slugify_respects_max_length_without_trailing_dash() -> None:
    slug = slugify("word " * 100, max_length=12)
    assert len(slug) <= 12
    assert not slug.endswith("-")

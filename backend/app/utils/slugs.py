"""Slug generation."""

from __future__ import annotations

import re
import unicodedata

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def slugify(value: str, max_length: int = 160) -> str:
    """Lower-case ASCII slug. Returns ``""`` if nothing usable remains.

    Names that are entirely non-Latin (e.g. Chinese-only) slugify to ``""``; callers
    must supply their own fallback (SKU, short id, ...) instead of relying on this.
    """
    ascii_text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    slug = _NON_ALNUM.sub("-", ascii_text.lower()).strip("-")
    return slug[:max_length].rstrip("-")

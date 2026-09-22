"""The rules for a panel's box, and the refusals the editing service raises.

Kept apart from the database code so the rules can be read (and tested) on their own.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

MIN_SIDE_PX = 20  # a panel is at least this wide and tall


class PanelEditError(Exception):
    """Base of every refusal; the message is safe to show to an admin."""


class InvalidBoxError(PanelEditError):
    pass


class PanelNotFoundError(PanelEditError):
    pass


class PageNotEditableError(PanelEditError):
    pass


class NothingToUndoError(PanelEditError):
    pass


class TooManyPanelsError(PanelEditError):
    pass


def validate_box(
    box: Sequence[float], page_width: int, page_height: int
) -> tuple[int, int, int, int]:
    """A drawn box as whole page pixels, clamped to the page.

    Raises :class:`InvalidBoxError` for anything that is not four finite numbers with the
    first corner above and left of the second, or that ends up smaller than ``MIN_SIDE_PX``
    once clamped to the page.
    """
    numbers = [v for v in box if isinstance(v, int | float) and not isinstance(v, bool)]
    if len(box) != 4 or len(numbers) != 4 or not all(math.isfinite(v) for v in numbers):
        raise InvalidBoxError("A panel is four numbers: left, top, right, bottom")
    x0, y0, x1, y1 = numbers
    if x1 <= x0 or y1 <= y0:
        raise InvalidBoxError("The box must run from its top-left corner to its bottom-right one")
    clamped = (
        max(0, round(x0)),
        max(0, round(y0)),
        min(page_width, round(x1)),
        min(page_height, round(y1)),
    )
    if clamped[2] - clamped[0] < MIN_SIDE_PX or clamped[3] - clamped[1] < MIN_SIDE_PX:
        raise InvalidBoxError(f"A panel must be at least {MIN_SIDE_PX} pixels wide and tall")
    return clamped

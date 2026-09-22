"""The contract between the pipeline and any "outline this object" model."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from PIL import Image

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Segment:
    """The outline of one object."""

    mask: Image.Image  # mode "L", the size of the picture: 255 inside the object, 0 outside
    score: float  # 0..1, the model's own estimate of how good the outline is


class SegmentationProvider(Protocol):
    name: str
    version: str | None

    def segment(self, image: Image.Image, boxes: Sequence[Box]) -> list[Segment]:
        """Outline the object inside each box. Returns one :class:`Segment` per box, in order."""
        ...

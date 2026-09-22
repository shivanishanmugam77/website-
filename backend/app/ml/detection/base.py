"""The contract between the pipeline and any "find these things in a picture" model."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from PIL import Image

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    """One thing found in a picture."""

    label: str  # which of the requested phrases it matched, e.g. "office chair"
    confidence: float  # 0..1, as reported by the model
    box: Box  # left, top, right, bottom in the pixels of the picture that was searched


class DetectionProvider(Protocol):
    #: short stable identifier stored with every result, e.g. "grounding-dino"
    name: str
    #: model id / version, stored with every result
    version: str | None

    def detect(self, image: Image.Image, prompts: Sequence[str]) -> list[Detection]:
        """Find things matching the phrases in ``prompts`` (e.g. ``["desk", "office chair"]``).
        Boxes are in the pixels of ``image``. Returns ``[]`` when nothing is found."""
        ...

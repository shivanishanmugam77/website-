"""The contract between the pipeline and any text-recognition engine.

The pipeline only ever talks to :class:`OcrProvider`. To try another engine, write a class
with the same two members and register it in ``app.ml.ocr.get_ocr_provider``; nothing else
changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from PIL import Image


class OcrEngineError(RuntimeError):
    """The engine is missing, misconfigured or failed to start (an operator problem, not a
    problem with the uploaded catalogue)."""


@dataclass(frozen=True)
class OcrLine:
    """One line of text the engine found, in the pixel coordinates of the image it was given."""

    text: str
    confidence: float  # 0..1, as reported by the engine
    polygon: tuple[tuple[float, float], ...]  # the 4 corners of the (possibly rotated) box

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        xs = [point[0] for point in self.polygon]
        ys = [point[1] for point in self.polygon]
        return (min(xs), min(ys), max(xs), max(ys))


class OcrProvider(Protocol):
    #: short stable identifier stored with every result, e.g. "rapidocr"
    name: str
    #: engine/package version when known (stored with every result)
    version: str | None

    def recognize(self, image: Image.Image) -> list[OcrLine]:
        """Read every line of text in ``image`` (any mode). Returns ``[]`` for no text."""
        ...

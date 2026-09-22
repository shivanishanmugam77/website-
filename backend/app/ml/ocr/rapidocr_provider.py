"""RapidOCR: the PaddleOCR text models running on ONNX Runtime.

Why this engine: one model reads Chinese and English together (supplier catalogues mix
both), it needs no GPU, and the models ship *inside* the pip package, so nothing is
downloaded when the worker starts and an unprivileged container can use it as is.

The engine is heavy to start (a second or two), so one instance is kept per worker process
(see ``app.ml.ocr.get_ocr_provider``). The package is imported lazily so the API process,
which never reads text, does not pay for it.
"""

from __future__ import annotations

import logging
from importlib import metadata

from PIL import Image

from app.ml.ocr.base import OcrEngineError, OcrLine

logger = logging.getLogger(__name__)

PACKAGE = "rapidocr-onnxruntime"


class RapidOcrProvider:
    name = "rapidocr"

    def __init__(self) -> None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:  # pragma: no cover - depends on the image
            raise OcrEngineError(
                f"The '{PACKAGE}' package is not installed in this environment"
            ) from exc
        try:
            self._engine = RapidOCR()
        except Exception as exc:  # pragma: no cover - depends on the image
            raise OcrEngineError(f"RapidOCR could not start: {exc}") from exc
        try:
            self.version: str | None = metadata.version(PACKAGE)
        except metadata.PackageNotFoundError:  # pragma: no cover
            self.version = None

    def recognize(self, image: Image.Image) -> list[OcrLine]:
        import numpy as np

        # RapidOCR follows OpenCV's convention: colour channels in BGR order.
        pixels = np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])
        result, _timings = self._engine(pixels)
        if not result:
            return []
        lines: list[OcrLine] = []
        for item in result:
            line = _to_line(item)
            if line is not None:
                lines.append(line)
        return lines


def _to_line(item: object) -> OcrLine | None:
    """Turn one ``[box, text, score]`` entry into an :class:`OcrLine`.

    Scores arrive as strings in some versions of the package, floats in others, so both
    are accepted. A malformed entry is skipped rather than failing the whole page.
    """
    try:
        box, text, score = item[0], item[1], item[2]  # type: ignore[index]
        polygon = tuple((float(point[0]), float(point[1])) for point in box)
        confidence = min(1.0, max(0.0, float(score)))
        if len(polygon) < 3 or not isinstance(text, str):
            return None
        return OcrLine(text=text, confidence=confidence, polygon=polygon)
    except (TypeError, ValueError, IndexError, KeyError):
        logger.warning("ocr_line_skipped", extra={"raw": repr(item)[:200]})
        return None

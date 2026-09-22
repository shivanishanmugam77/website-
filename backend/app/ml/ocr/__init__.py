"""Text-recognition (OCR) providers.

Only worker processes ever load an engine; the API process imports this package for the
interface types alone, which is why the heavy engine module is imported lazily.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from app.ml.ocr.base import OcrEngineError, OcrLine, OcrProvider

if TYPE_CHECKING:
    from app.core.config import Settings

__all__ = ["OcrEngineError", "OcrLine", "OcrProvider", "get_ocr_provider"]


@lru_cache(maxsize=4)
def _load(name: str) -> OcrProvider:
    # lru_cache does not cache exceptions, so a failed start is retried on the next call.
    if name == "rapidocr":
        from app.ml.ocr.rapidocr_provider import RapidOcrProvider

        return RapidOcrProvider()
    raise OcrEngineError(f"Unknown OCR provider {name!r}")


def get_ocr_provider(settings: Settings) -> OcrProvider:
    """The engine named in the settings; started once per process and then reused."""
    return _load(settings.ocr_provider)

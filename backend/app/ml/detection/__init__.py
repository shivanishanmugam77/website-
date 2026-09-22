"""Object-finding providers ("find the desks and chairs in this picture").

Only worker processes and command-line tools ever load a model; the API imports this package
for the interface types alone, which is why the model module is imported lazily.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from app.ml.detection.base import Detection, DetectionProvider
from app.ml.errors import ModelUnavailableError

if TYPE_CHECKING:
    from app.core.config import Settings

__all__ = ["Detection", "DetectionProvider", "ModelUnavailableError", "get_detection_provider"]


@lru_cache(maxsize=2)
def _load(
    name: str, model: str, box: float, text: float, max_px: int, threads: int, device: str
) -> DetectionProvider:
    # lru_cache does not cache exceptions, so a failed start is retried on the next call.
    if name == "grounding_dino":
        from app.ml.detection.grounding_dino import GroundingDinoProvider

        return GroundingDinoProvider(
            model,
            box_threshold=box,
            text_threshold=text,
            max_px=max_px,
            threads=threads,
            device=device,
        )
    raise ModelUnavailableError(f"Unknown detection provider {name!r}")


def get_detection_provider(settings: Settings) -> DetectionProvider:
    """The finder named in the settings; loaded once per process and then reused."""
    return _load(
        settings.detection_provider,
        settings.detection_model,
        settings.detection_box_threshold,
        settings.detection_text_threshold,
        settings.detection_max_px,
        settings.ml_num_threads,
        settings.ml_device,
    )

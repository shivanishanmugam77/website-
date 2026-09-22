"""Object-outlining providers ("draw the exact edge of this desk").

Only worker processes and command-line tools ever load a model; the API imports this package
for the interface types alone, which is why the model module is imported lazily.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from app.ml.errors import ModelUnavailableError
from app.ml.segmentation.base import Segment, SegmentationProvider

if TYPE_CHECKING:
    from app.core.config import Settings

__all__ = ["ModelUnavailableError", "Segment", "SegmentationProvider", "get_segmentation_provider"]


@lru_cache(maxsize=2)
def _load(name: str, model: str, threads: int, device: str) -> SegmentationProvider:
    if name == "sam":
        from app.ml.segmentation.sam import SamProvider

        return SamProvider(model, threads=threads, device=device)
    raise ModelUnavailableError(f"Unknown segmentation provider {name!r}")


def get_segmentation_provider(settings: Settings) -> SegmentationProvider:
    """The outliner named in the settings; loaded once per process and then reused."""
    return _load(
        settings.segmentation_provider,
        settings.segmentation_model,
        settings.ml_num_threads,
        settings.ml_device,
    )

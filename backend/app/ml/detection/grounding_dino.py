"""Grounding DINO: finds things in a picture from a description in words.

"Tiny" size (about 700 MB), Apache-2.0 licensed. It is asked for phrases like "desk" or
"office chair" and answers with a box, a score and the words it matched for each thing it
finds. It runs through the Hugging Face ``transformers`` library on the CPU (or a GPU when
one is available and ``ML_DEVICE`` allows it).

torch and transformers are imported inside ``__init__`` so importing this module is cheap
and the API process (which never looks at pictures) does not load them.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from PIL import Image

from app.ml.detection.base import Detection
from app.ml.errors import ModelUnavailableError
from app.services.detections import format_prompt, to_detections

logger = logging.getLogger(__name__)

DOWNLOAD_HINT = "run: docker compose run --rm backend python -m app.cli.download_models"


def pick_device(torch: Any, wanted: str) -> str:
    """``auto`` uses a CUDA graphics card when there is one, otherwise the CPU."""
    if wanted == "cuda" or (wanted == "auto" and torch.cuda.is_available()):
        return "cuda"
    return "cpu"


def shrink(image: Image.Image, max_px: int) -> tuple[Image.Image, float]:
    """The picture scaled so its longest side is at most ``max_px``, and the factor that
    converts the smaller picture's pixels back to the original's (1.0 when unchanged)."""
    longest = max(image.size)
    if longest <= max_px:
        return image, 1.0
    factor = longest / max_px
    size = (max(1, round(image.width / factor)), max(1, round(image.height / factor)))
    return image.resize(size, Image.Resampling.LANCZOS), factor


class GroundingDinoProvider:
    name = "grounding-dino"

    def __init__(
        self,
        model_id: str,
        *,
        box_threshold: float,
        text_threshold: float,
        max_px: int,
        threads: int,
        device: str,
    ) -> None:
        try:
            import torch
            import transformers
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:  # pragma: no cover - depends on the image
            raise ModelUnavailableError(
                f"The machine-learning packages are not installed ({exc}); rebuild the image "
                "with: docker compose build"
            ) from exc
        torch.set_num_threads(threads)
        self._torch = torch
        self._device = pick_device(torch, device)
        self._box_threshold = box_threshold
        self._text_threshold = text_threshold
        self._max_px = max_px
        try:
            self._processor = AutoProcessor.from_pretrained(model_id)
            self._model = (
                AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
                .to(self._device)
                .eval()
            )
        except Exception as exc:  # pragma: no cover - depends on the network and cache
            logger.error(
                "detection_model_load_failed", extra={"model": model_id, "error": str(exc)}
            )
            raise ModelUnavailableError(
                f"The finder model {model_id!r} could not be loaded ({type(exc).__name__}). "
                f"If it has not been downloaded yet, {DOWNLOAD_HINT}"
            ) from exc
        self.version = f"{model_id} (transformers {transformers.__version__}, {self._device})"

    def detect(self, image: Image.Image, prompts: Sequence[str]) -> list[Detection]:
        picture, factor = shrink(image.convert("RGB"), self._max_px)
        inputs = self._processor(
            images=picture, text=format_prompt(prompts), return_tensors="pt"
        ).to(self._device)
        with self._torch.inference_mode():
            outputs = self._model(**inputs)
        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self._box_threshold,
            text_threshold=self._text_threshold,
            target_sizes=[picture.size[::-1]],
        )[0]
        raw = zip(
            results["boxes"].tolist(),
            results["scores"].tolist(),
            [str(label) for label in results["labels"]],
            strict=True,
        )
        return to_detections(raw, prompts, scale=factor, image_size=image.size)

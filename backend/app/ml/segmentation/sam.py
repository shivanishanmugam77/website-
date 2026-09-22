"""SAM (Segment Anything): draws the exact outline of an object inside a box.

"ViT-B" size (about 375 MB), Apache-2.0 licensed. Given a picture and boxes it returns, for
each box, a mask: white where the object is, black elsewhere. Several boxes on one picture
are outlined in a single pass (the picture is analysed once). The model offers three outlines
per box (part, bigger part, whole); ``app.services.masks`` picks the most complete good one.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from PIL import Image

from app.ml.detection.grounding_dino import DOWNLOAD_HINT, pick_device
from app.ml.errors import ModelUnavailableError
from app.ml.segmentation.base import Box, Segment
from app.services.masks import choose_mask

logger = logging.getLogger(__name__)


class SamProvider:
    name = "sam"

    def __init__(self, model_id: str, *, threads: int, device: str) -> None:
        try:
            import torch
            import transformers
            from transformers import SamModel, SamProcessor
        except ImportError as exc:  # pragma: no cover - depends on the image
            raise ModelUnavailableError(
                f"The machine-learning packages are not installed ({exc}); rebuild the image "
                "with: docker compose build"
            ) from exc
        torch.set_num_threads(threads)
        self._torch = torch
        self._device = pick_device(torch, device)
        try:
            self._processor = SamProcessor.from_pretrained(model_id)
            self._model = SamModel.from_pretrained(model_id).to(self._device).eval()
        except Exception as exc:  # pragma: no cover - depends on the network and cache
            logger.error(
                "segmentation_model_load_failed", extra={"model": model_id, "error": str(exc)}
            )
            raise ModelUnavailableError(
                f"The outlining model {model_id!r} could not be loaded ({type(exc).__name__}). "
                f"If it has not been downloaded yet, {DOWNLOAD_HINT}"
            ) from exc
        self.version = f"{model_id} (transformers {transformers.__version__}, {self._device})"

    def segment(self, image: Image.Image, boxes: Sequence[Box]) -> list[Segment]:
        if not boxes:
            return []
        picture = image.convert("RGB")
        inputs = self._processor(
            picture, input_boxes=[[list(box) for box in boxes]], return_tensors="pt"
        ).to(self._device)
        with self._torch.inference_mode():
            outputs = self._model(**inputs, multimask_output=True)
        masks = self._processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0]  # (boxes, 3 outlines, height, width) booleans, at the original picture's size
        scores = outputs.iou_scores.cpu().reshape(len(boxes), -1).tolist()
        segments = []
        for index, box in enumerate(boxes):
            candidates = [outline.numpy() for outline in masks[index]]
            chosen, mask = choose_mask(candidates, scores[index], box)
            score = min(1.0, max(0.0, float(scores[index][chosen])))
            segments.append(Segment(mask=_mask_image(mask), score=score))
        return segments


def _mask_image(mask: Any) -> Image.Image:
    """A boolean (height, width) array as a black-and-white picture."""
    return Image.fromarray(mask.astype("uint8") * 255)

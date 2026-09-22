"""Choosing and measuring the outline of one product (no model needed).

For every box the outlining model offers three outlines: a small part of the object, a bigger
part, and the whole object. The model's own favourite is often the small part (for a long desk
it can be just the near half), so the outline used is the *largest* one among those the model
rates nearly as highly as its favourite. Whatever the model draws outside the find's box is
cut away, because the box is what the finder saw.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from app.ml.segmentation.base import Box

# A bigger outline wins while its score is within this much of the best score.
SCORE_TOLERANCE = 0.10


def clip_to_box(mask: np.ndarray, box: Box) -> np.ndarray:
    """The outline with everything outside the box removed (True = inside the object)."""
    height, width = mask.shape
    left = int(np.clip(np.floor(box[0]), 0, width))
    top = int(np.clip(np.floor(box[1]), 0, height))
    right = int(np.clip(np.ceil(box[2]), 0, width))
    bottom = int(np.clip(np.ceil(box[3]), 0, height))
    clipped = np.zeros((height, width), dtype=bool)
    clipped[top:bottom, left:right] = mask[top:bottom, left:right].astype(bool)  # empty if no area
    return clipped


def choose_mask(
    candidates: Sequence[np.ndarray], scores: Sequence[float], box: Box
) -> tuple[int, np.ndarray]:
    """The position and the (box-clipped) picture of the outline to use."""
    if not candidates or len(candidates) != len(scores):
        raise ValueError("Need one score for each candidate outline, and at least one outline.")
    clipped = [clip_to_box(candidate, box) for candidate in candidates]
    best = max(scores)
    eligible = [i for i, score in enumerate(scores) if score >= best - SCORE_TOLERANCE]
    chosen = max(eligible, key=lambda i: (int(clipped[i].sum()), scores[i]))
    return chosen, clipped[chosen]


def coverage(mask: np.ndarray, box: Box) -> float:
    """The share (0 to 1) of the box that the outline fills."""
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    if area <= 0:
        return 0.0
    return min(1.0, float(clip_to_box(mask, box).sum()) / area)

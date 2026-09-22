"""Turn a model's raw answer into a clean list of things found in a picture.

Everything here is plain arithmetic on boxes and words, kept apart from the models so it can
be tested exactly. The models make suggestions; these rules tidy them: map the model's wording
back to the phrases that were asked for, cut boxes to the picture, drop specks, and keep one
box per object where the model reported the same object several times.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from app.ml.detection.base import Box, Detection

MAX_PROMPTS = 30
MAX_PROMPT_CHARS = 40


# ------------------------------------------------------------------------------ the request
def parse_prompts(text: str) -> list[str]:
    """``"Desk, Office chair. cabinet"`` -> ``["desk", "office chair", "cabinet"]``."""
    prompts: list[str] = []
    for part in re.split(r"[,.;\n]", text):
        phrase = " ".join(part.lower().split())
        if not phrase:
            continue
        if len(phrase) > MAX_PROMPT_CHARS:
            raise ValueError(f"'{phrase[:20]}...' is longer than {MAX_PROMPT_CHARS} letters")
        if phrase not in prompts:
            prompts.append(phrase)
    if not prompts:
        raise ValueError("At least one phrase to look for is needed")
    if len(prompts) > MAX_PROMPTS:
        raise ValueError(f"At most {MAX_PROMPTS} phrases can be looked for at once")
    return prompts


def format_prompt(prompts: Sequence[str]) -> str:
    """The form Grounding DINO expects: ``"desk. office chair. cabinet."``."""
    return " ".join(f"{phrase}." for phrase in prompts)


def match_prompt(label_text: str, prompts: Sequence[str]) -> str | None:
    """Which requested phrase does the model's wording stand for?

    The model answers with the words it matched, which can be a whole phrase ("office chair"),
    part of one ("office") or two run together ("desk chair"). An exact match wins; otherwise
    the phrase whose words are all present, then with the most words in common, is chosen.
    ``None`` when the wording matches nothing asked for.
    """
    words = label_text.lower().replace(".", " ").split()
    if not words:
        return None
    text = " ".join(words)
    if text in prompts:
        return text
    best: str | None = None
    best_score = (0, 0, 0)
    for phrase in prompts:
        phrase_words = phrase.split()
        common = len(set(phrase_words) & set(words))
        if common == 0:
            continue
        score = (int(common == len(phrase_words)), common, len(phrase_words))
        if score > best_score:
            best, best_score = phrase, score
    return best


# ------------------------------------------------------------------------------ boxes
def clamp_box(box: Box, width: int, height: int) -> Box:
    return (
        min(max(box[0], 0.0), width),
        min(max(box[1], 0.0), height),
        min(max(box[2], 0.0), width),
        min(max(box[3], 0.0), height),
    )


def box_area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def iou(a: Box, b: Box) -> float:
    """Overlap divided by combined area: 1.0 for identical boxes, 0.0 for disjoint ones."""
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    inter = width * height
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def to_detections(
    raw: Iterable[tuple[Sequence[float], float, str]],
    prompts: Sequence[str],
    *,
    scale: float,
    image_size: tuple[int, int],
) -> list[Detection]:
    """Model output ``(box, score, wording)`` -> :class:`Detection`.

    ``scale`` converts the model's pixels back to the original picture's (a picture that was
    shrunk before the model saw it has ``scale > 1``). Answers that match no requested phrase
    or that have no area are dropped.
    """
    found: list[Detection] = []
    for box, score, wording in raw:
        label = match_prompt(wording, prompts)
        if label is None:
            continue
        scaled = clamp_box(
            (box[0] * scale, box[1] * scale, box[2] * scale, box[3] * scale), *image_size
        )
        if box_area(scaled) <= 0:
            continue
        confidence = min(1.0, max(0.0, float(score)))
        found.append(Detection(label=label, confidence=confidence, box=scaled))
    return found


# ------------------------------------------------------------------------------ tidying
def drop_tiny(
    detections: Sequence[Detection],
    image_size: tuple[int, int],
    *,
    min_area_share: float = 0.01,
    min_side_px: float = 24.0,
) -> list[Detection]:
    """Remove boxes too small to be a product (handles, cups, specks)."""
    page_area = image_size[0] * image_size[1]
    return [
        d
        for d in detections
        if box_area(d.box) >= min_area_share * page_area
        and min(d.box[2] - d.box[0], d.box[3] - d.box[1]) >= min_side_px
    ]


def suppress_overlaps(
    detections: Sequence[Detection],
    *,
    same_label_iou: float = 0.5,
    cross_label_iou: float = 0.85,
) -> list[Detection]:
    """Keep one box per object, the most confident.

    Boxes with the same label that overlap by ``same_label_iou`` or more are the same object
    seen twice. Boxes with different labels are only merged when they overlap almost exactly
    (``cross_label_iou``): "desk" and "table" on one piece of furniture, whereas a chair pushed
    under a desk is two objects with overlapping boxes.
    """
    kept: list[Detection] = []
    for candidate in sorted(detections, key=lambda d: -d.confidence):
        duplicate = any(
            iou(candidate.box, other.box)
            >= (same_label_iou if candidate.label == other.label else cross_label_iou)
            for other in kept
        )
        if not duplicate:
            kept.append(candidate)
    return kept


def tidy(
    detections: Sequence[Detection],
    image_size: tuple[int, int],
    *,
    min_area_share: float = 0.01,
) -> list[Detection]:
    """The whole clean-up: cut to the picture, drop specks, keep one box per object, and list
    them top to bottom, left to right."""
    inside = [
        Detection(d.label, d.confidence, clamp_box(d.box, *image_size)) for d in detections
    ]
    kept = suppress_overlaps(drop_tiny(inside, image_size, min_area_share=min_area_share))
    return sorted(kept, key=lambda d: (round(d.box[1] / max(1, image_size[1]) * 12), d.box[0]))

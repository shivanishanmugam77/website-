"""Grouping the products found on a page into one product candidate for an admin to review.

A supplier's PRODUCT page almost always shows one product: a room photo plus one or more
close-ups, with one model code and one size printed once on the page. So the rule here is
simple - **everything found on one PRODUCT page becomes one candidate** - and what this
module mostly does is turn that grouping into evidence a person can judge quickly: a
suggested name (read off the page), the model code and size, a suggested category, and a
confidence score built from how sure each step along the way was, plus plain-English flags
for anything that looks off (two model codes on one page, a find whose outline failed, ...).

Nothing here writes to the database or decides anything final: a candidate is always
``NEEDS_REVIEW`` or, when everything lines up, ``READY`` - never approved. A human still
turns a candidate into a real ``Product`` (Phase 8).
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from app.models.enums import CandidateStatus, SegmentationStatus
from app.services.text_signals import (
    Dimension,
    ModelCode,
    SeriesLabel,
    TextSignals,
    extract_signals,
)

# How much each ingredient counts towards the overall score. Kept as named constants (rather
# than folded into the arithmetic) so they can be retuned from real catalogues without
# touching the logic, and so a test can check they still add up to one.
WEIGHT_DETECTION = 0.25
WEIGHT_SEGMENTATION = 0.25
WEIGHT_OCR = 0.15
WEIGHT_ASSOCIATION = 0.20
WEIGHT_METADATA = 0.15

# A find whose outline is unusable still has a "score" on file sometimes (e.g. the model was
# confident about an outline that turned out empty) - that number would overstate how good
# the segmentation step actually was, so a failed outline always counts as zero here.
_FAILED_OUTLINE_COMPONENT = 0.0

# label (as the detector prints it, case-insensitive) -> suggested category slug. Slugs match
# app.services.seed.DEFAULT_CATEGORY_TREE. A label with no good match is left unset rather
# than guessed.
CATEGORY_HINTS: dict[str, str] = {
    "desk": "workstations",
    "workstation": "workstations",
    "reception desk": "workstations",
    "table": "office-tables",
    "meeting table": "conference-tables",
    "conference table": "conference-tables",
    "office chair": "office-chairs",
    "chair": "office-chairs",
    "executive chair": "executive-chairs",
    "visitor chair": "visitor-chairs",
    "cabinet": "cabinets",
    "filing cabinet": "cabinets",
    "locker": "cabinets",
    "bookshelf": "shelves",
    "shelf": "shelves",
    "sofa": "sofas",
    "chandelier": "chandeliers",
    "pendant light": "pendant-lights",
    "ceiling light": "ceiling-lights",
    "wall lamp": "decorative-lighting",
    "floor lamp": "decorative-lighting",
    "table lamp": "decorative-lighting",
}

_TRIM = " \t\u3000:：-—–,、。.·"
# A genuine product name or series label, printed once beside a size or on its own, is short.
# Anything longer than this is marketing copy or a description that got matched by mistake,
# not a name - it is dropped rather than shown, so a person is never handed a wall of text
# where a short name belongs.
MAX_NAME_LENGTH = 40


@dataclass(frozen=True)
class Find:
    """One product found in a panel - just enough to group and score it (see ``ProductFindOut``
    for the full picture; this is the DB-independent subset)."""

    id: uuid.UUID
    panel_id: uuid.UUID
    label: str
    confidence: float
    outline_status: SegmentationStatus | None
    outline_score: float | None


@dataclass(frozen=True)
class CandidateDraft:
    """Everything needed to create one ``ProductCandidate`` row."""

    detected_object_ids: list[uuid.UUID]
    panel_ids: list[uuid.UUID]
    status: CandidateStatus
    fields: dict[str, Any]
    flags: list[str]
    reasoning: list[str]
    detection_confidence: float
    segmentation_confidence: float
    ocr_confidence: float
    association_confidence: float
    metadata_confidence: float
    overall_confidence: float


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _outline_component(find: Find) -> float:
    if find.outline_status is not SegmentationStatus.FAILED and find.outline_score is not None:
        return find.outline_score
    return _FAILED_OUTLINE_COMPONENT


def _strip_dimension(text: str, dimension: Dimension) -> str:
    without = text.replace(dimension.raw_text, "", 1)
    return without.strip(_TRIM)


def pick_name(
    line_texts: Sequence[str], dimension: Dimension | None, series: SeriesLabel | None
) -> str | None:
    """The best guess at a product name, read straight off the page.

    Suppliers usually print the name and the size on the same line (``"四人位职员桌
    2400Wx1200Dx750H"``), so the name is whatever is left after the recognised size is
    removed. A series label (already a short, cleaned-up value such as ``"09 series"``, never
    the raw line it was matched from) is used only when there is no size to anchor on.
    Anything that comes out longer than a real name ever is (``MAX_NAME_LENGTH``) is dropped
    rather than shown, since it is more likely a caption or marketing line than a name.
    """
    name = None
    if dimension is not None and 0 <= dimension.line_index < len(line_texts):
        stripped = _strip_dimension(line_texts[dimension.line_index], dimension)
        name = stripped or None
    if name is None and series is not None:
        name = series.label or None
    if name is not None and len(name) > MAX_NAME_LENGTH:
        return None
    return name


def suggest_category(labels: Sequence[str]) -> str | None:
    """The category slug most of the found labels agree on, or None if they disagree or
    match nothing we recognise. Ties fall back to the first label in reading order."""
    matched = [CATEGORY_HINTS[label.lower()] for label in labels if label.lower() in CATEGORY_HINTS]
    if not matched:
        return None
    counts: dict[str, int] = {}
    for slug in matched:
        counts[slug] = counts.get(slug, 0) + 1
    best = max(counts.values())
    for slug in matched:  # first in reading order among the tied leaders
        if counts[slug] == best:
            return slug
    return None  # pragma: no cover - unreachable, matched is non-empty


def _field(value: Any, source: str, confidence: float | None = None) -> dict[str, Any]:
    return {"value": value, "source": source, "confidence": confidence}


def _association_confidence(signals: TextSignals, flags: list[str]) -> float:
    has_code, has_dim = bool(signals.model_codes), bool(signals.dimensions)
    if has_code and has_dim:
        base = 1.0
    elif has_code or has_dim:
        base = 0.6
    else:
        base = 0.3  # grouped only because they share a PRODUCT page - the weakest anchor
    if "MULTIPLE_MODEL_CODES_ON_PAGE" in flags or "MULTIPLE_DIMENSIONS_ON_PAGE" in flags:
        base *= 0.5  # the page may actually describe more than one product
    return base


def _metadata_confidence(
    name: str | None, model_code: ModelCode | None, dimension: Dimension | None
) -> float:
    present = sum(1 for value in (name, model_code, dimension) if value is not None)
    return present / 3


def _ocr_confidence(
    line_confidences: Sequence[float | None],
    *,
    name: str | None,
    name_line: int | None,
    model_code: ModelCode | None,
    dimension: Dimension | None,
) -> float:
    def confidence_at(index: int | None) -> float:
        if index is None or not (0 <= index < len(line_confidences)):
            return 1.0
        value = line_confidences[index]
        return 1.0 if value is None else value

    weighted = 0.0
    if model_code is not None:
        weighted += 0.4 * confidence_at(model_code.line_index)
    if dimension is not None:
        weighted += 0.4 * confidence_at(dimension.line_index)
    if name is not None:
        weighted += 0.2 * confidence_at(name_line)
    return weighted


def build_page_candidate(
    finds: Sequence[Find],
    line_texts: Sequence[str],
    line_confidences: Sequence[float | None] = (),
    *,
    auto_review_threshold: float,
    low_confidence_threshold: float,
) -> CandidateDraft | None:
    """One candidate for everything found on a PRODUCT page, or ``None`` if nothing was found.

    ``line_texts`` must be every OCR line on the page, in reading order (line ``i`` is
    ``extract_signals``'s line ``i``) - the same lines the page's ``/text`` endpoint used.
    ``line_confidences`` is the matching per-line OCR confidence, or omit it to skip that part
    of the scoring.
    """
    if not finds:
        return None

    signals = extract_signals(list(line_texts))
    model_code = signals.model_codes[0] if signals.model_codes else None
    dimension = signals.dimensions[0] if signals.dimensions else None
    series = signals.series[0] if signals.series else None
    name = pick_name(line_texts, dimension, series)
    name_line = None
    if dimension is not None:
        name_line = dimension.line_index
    elif series is not None:
        name_line = series.line_index

    flags: list[str] = []
    if len(signals.model_codes) > 1:
        flags.append("MULTIPLE_MODEL_CODES_ON_PAGE")
    if len(signals.dimensions) > 1:
        flags.append("MULTIPLE_DIMENSIONS_ON_PAGE")
    if model_code is None:
        flags.append("NO_MODEL_CODE")
    if dimension is None:
        flags.append("NO_DIMENSIONS")
    if any(f.outline_status is SegmentationStatus.FAILED for f in finds):
        flags.append("SOME_OUTLINES_FAILED")
    if any(f.outline_status is SegmentationStatus.LOW_CONFIDENCE for f in finds):
        flags.append("SOME_OUTLINES_LOW_CONFIDENCE")

    detection_confidence = _mean([f.confidence for f in finds])
    segmentation_confidence = _mean([_outline_component(f) for f in finds])
    ocr_confidence = _ocr_confidence(
        line_confidences, name=name, name_line=name_line, model_code=model_code, dimension=dimension
    )
    association_confidence = _association_confidence(signals, flags)
    metadata_confidence = _metadata_confidence(name, model_code, dimension)
    overall_confidence = (
        WEIGHT_DETECTION * detection_confidence
        + WEIGHT_SEGMENTATION * segmentation_confidence
        + WEIGHT_OCR * ocr_confidence
        + WEIGHT_ASSOCIATION * association_confidence
        + WEIGHT_METADATA * metadata_confidence
    )
    if overall_confidence < low_confidence_threshold:
        flags.append("LOW_CONFIDENCE")
    status = (
        CandidateStatus.READY
        if overall_confidence >= auto_review_threshold
        else CandidateStatus.NEEDS_REVIEW
    )

    panel_count = len({f.panel_id for f in finds})
    reasoning = [f"{len(finds)} item(s) found across {panel_count} panel(s) on this page."]
    if model_code is not None:
        reasoning.append(f"Model code {model_code.code!r} found on the page.")
    if dimension is not None:
        reasoning.append(f"Size {dimension.raw_text!r} found on the page.")
    if "MULTIPLE_MODEL_CODES_ON_PAGE" in flags:
        reasoning.append(
            "More than one model code appears on this page - it may describe more than one product."
        )

    fields: dict[str, Any] = {
        "name": _field(name, "OCR") if name is not None else None,
        "model_code": _field(model_code.code, "OCR") if model_code is not None else None,
        "dimensions": (
            _field(
                {
                    "width": dimension.width,
                    "depth": dimension.depth,
                    "height": dimension.height,
                    "unit": dimension.unit,
                },
                "OCR",
            )
            if dimension is not None
            else None
        ),
        "series": _field(series.label, "OCR") if series is not None else None,
        "labels": _field(sorted({f.label for f in finds}), "AI"),
        "category_suggestion": (
            _field(category, "AI")
            if (category := suggest_category([f.label for f in finds]))
            else None
        ),
    }

    return CandidateDraft(
        detected_object_ids=[f.id for f in finds],
        panel_ids=sorted({f.panel_id for f in finds}, key=str),
        status=status,
        fields=fields,
        flags=flags,
        reasoning=reasoning,
        detection_confidence=detection_confidence,
        segmentation_confidence=segmentation_confidence,
        ocr_confidence=ocr_confidence,
        association_confidence=association_confidence,
        metadata_confidence=metadata_confidence,
        overall_confidence=overall_confidence,
    )

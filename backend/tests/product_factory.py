"""Stand-in models for finding products, and a way to put finds into the database directly."""

from __future__ import annotations

import io
from collections.abc import Callable

import numpy as np
from PIL import Image

from app.ml.detection.base import Detection
from app.ml.segmentation.base import Segment
from app.models import DetectedObject
from app.models.enums import SegmentationStatus
from app.services import pipeline
from app.services.product_finding import Found
from tests.factories import settings_with


class FakeFinder:
    """Finds whatever ``respond(call_index, image)`` says (nothing by default); counts its calls."""

    name = "fake-finder"
    version = "fake 1"

    def __init__(self, respond: Callable[[int, Image.Image], list[Detection]] | None = None) -> None:
        self.respond = respond or (lambda _index, _image: [])
        self.calls: list[tuple[tuple[int, int], list[str]]] = []

    def detect(self, image: Image.Image, prompts):  # noqa: ANN001, ANN201
        index = len(self.calls)
        self.calls.append((image.size, list(prompts)))
        return self.respond(index, image)


class FakeOutliner:
    """Outlines every box as a filled rectangle (or as nothing, with ``fill=False``)."""

    name = "fake-outliner"
    version = "fake 1"

    def __init__(self, score: float = 0.9, fill: bool = True) -> None:
        self.score, self.fill, self.calls = score, fill, 0

    def segment(self, image: Image.Image, boxes):  # noqa: ANN001, ANN201
        self.calls += 1
        outlines = []
        for box in boxes:
            mask = np.zeros((image.height, image.width), dtype="uint8")
            if self.fill:
                mask[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])] = 255
            outlines.append(Segment(Image.fromarray(mask), self.score))
        return outlines


def add_find(  # noqa: ANN201
    session,  # noqa: ANN001
    storage,  # noqa: ANN001
    catalogue,  # noqa: ANN001
    panel,  # noqa: ANN001
    *,
    label: str = "desk",
    box: tuple[float, float, float, float] = (50.0, 50.0, 300.0, 400.0),
    status: SegmentationStatus = SegmentationStatus.SUCCEEDED,
    note: str | None = None,
) -> DetectedObject:
    """A product found in a panel, saved exactly as the pipeline saves it (rows and files)."""
    with Image.open(io.BytesIO(storage.read_bytes(panel.storage_key))) as opened:
        picture = opened.convert("RGB")
    mask = None
    if status is not SegmentationStatus.FAILED:
        drawn = np.zeros((picture.height, picture.width), dtype="uint8")
        drawn[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])] = 255
        mask = Image.fromarray(drawn)
    found = Found(
        Detection(label, 0.8, box), status, 0.9, 1.0 if mask is not None else 0.0, mask, note
    )
    object_id = pipeline._save_find(
        session,
        storage,
        settings_with(thumbnail_max_px=64),
        catalogue,
        panel,
        picture,
        found,
        (FakeFinder(), FakeOutliner()),
        [],
    )
    session.flush()
    return session.get(DetectedObject, object_id)

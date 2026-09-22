"""Try the product finder on the pictures of one catalogue page, without changing anything:

    docker compose run --rm backend python -m app.cli.check_products --catalogue <id> --page 8

For every photo panel on the page it asks the finder for desks, chairs, cabinets and the like,
prints what it found (name, confidence, position, time taken) and writes a picture with the
finds outlined to the ``debug`` folder of the project. Add ``--masks`` to also draw the exact
outline of each find (slower) and to print how much of its box each outline fills. Nothing is
saved to the database.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
import uuid
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import session_scope
from app.core.logging_config import configure_logging
from app.ml.detection import DetectionProvider, get_detection_provider
from app.ml.errors import ModelUnavailableError
from app.ml.segmentation import SegmentationProvider, get_segmentation_provider
from app.models import CataloguePage, SourceImage
from app.models.enums import SourceImageKind
from app.services.detection_preview import draw_detections, draw_sheet
from app.services.detections import box_area, parse_prompts, tidy
from app.services.masks import coverage
from app.services.panels import sort_reading_order
from app.services.storage import Storage, get_storage

DEBUG_DIR = Path("/app/debug")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Try the product finder on one page.")
    parser.add_argument("--catalogue", required=True, type=uuid.UUID, help="catalogue id")
    parser.add_argument("--page", required=True, type=int, help="page number")
    parser.add_argument("--panel", type=int, help="only this panel (1, 2, 3 ... as numbered)")
    parser.add_argument("--masks", action="store_true", help="also outline each find exactly")
    parser.add_argument("--out", type=Path, help="where to write the picture")
    return parser.parse_args(argv)


def _panels(db: Session, catalogue_id: uuid.UUID, page_number: int) -> list[SourceImage] | None:
    page = db.scalar(
        select(CataloguePage).where(
            CataloguePage.catalogue_id == catalogue_id, CataloguePage.page_number == page_number
        )
    )
    if page is None:
        return None
    regions = list(
        db.scalars(
            select(SourceImage).where(
                SourceImage.page_id == page.id, SourceImage.kind == SourceImageKind.REGION
            )
        )
    )
    return sort_reading_order(
        regions,
        page.height_px or 1,
        key=lambda r: (r.bbox_x0 or 0.0, r.bbox_y0 or 0.0, r.bbox_x1 or 0.0, r.bbox_y1 or 0.0),
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    session: Session | None = None,
    storage: Storage | None = None,
    detector: DetectionProvider | None = None,
    segmenter: SegmentationProvider | None = None,
) -> int:
    args = _parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    prompts = parse_prompts(settings.detection_prompts)
    scope = contextlib.nullcontext(session) if session is not None else session_scope()

    with scope as db:
        panels = _panels(db, args.catalogue, args.page)
        if panels is None:
            print(f"No page {args.page} in catalogue {args.catalogue}.", file=sys.stderr)
            return 1
        if not panels:
            print("That page has no photo panels to look at.")
            return 1
        files = storage or get_storage()
        try:
            started = time.perf_counter()
            finder = detector or get_detection_provider(settings)
            outliner = (segmenter or get_segmentation_provider(settings)) if args.masks else None
            print(f"Models ready in {time.perf_counter() - started:.1f}s ({finder.version})")
        except ModelUnavailableError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1

        print(f"Looking for: {', '.join(prompts)}")
        sheet_items: list[tuple[str, Image.Image]] = []
        total = 0
        for number, panel in enumerate(panels, start=1):
            if args.panel is not None and number != args.panel:
                continue
            with Image.open(BytesIO(files.read_bytes(panel.storage_key))) as opened:
                picture = opened.convert("RGB")
            started = time.perf_counter()
            found = tidy(finder.detect(picture, prompts), picture.size)
            outlines = []
            if outliner is not None and found:
                outlines = outliner.segment(picture, [d.box for d in found])
            seconds = time.perf_counter() - started
            print(f"\nPanel {number} ({picture.width} x {picture.height}): "
                  f"{len(found)} found in {seconds:.1f}s")
            for position, detection in enumerate(found):
                share = box_area(detection.box) / (picture.width * picture.height)
                line = (f"  {detection.label:<16} {detection.confidence:.0%}  "
                        f"box {tuple(round(v) for v in detection.box)}  ({share:.0%} of the panel)")
                if position < len(outlines):
                    outline = np.asarray(outlines[position].mask.convert("L")) > 127
                    filled = coverage(outline, detection.box)
                    line += f"  outline fills {filled:.0%} of its box"
                print(line)
            total += len(found)
            annotated = draw_detections(picture, found, [o.mask for o in outlines])
            sheet_items.append((f"Panel {number}: {len(found)} found", annotated))

    if not sheet_items:
        print("No such panel on that page.", file=sys.stderr)
        return 1
    out = args.out or DEBUG_DIR / f"products-{str(args.catalogue)[:8]}-page{args.page}.jpg"
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        draw_sheet(sheet_items).save(out, "JPEG", quality=85)
        print(f"\nPicture written to {out} (in the 'debug' folder of your project).")
    except OSError as exc:
        print(f"\nCould not write the picture: {exc}", file=sys.stderr)
    print(f"{total} things found in all.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

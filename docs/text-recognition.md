# Text recognition (Phase 4)

The worker reads the text on every page after the pages are rendered. This is what turns a flat
page picture into words the system can use: product names, model codes, sizes.

## What happens to each page

1. The rendered page picture is cut into overlapping square tiles (default 1600 px, overlapping by 320 px).
   Big pages need this: a text engine shrinks its input to about 2000 px, which would make small print
   (captions, model codes) unreadable. Pages only slightly larger than a tile (an A4 page at 150 dpi) are
   read in one piece; tiles with nothing on them (plain white) are skipped.
2. Each tile is read by the engine and every line's position is moved back to page coordinates.
3. The same line seen in two overlapping tiles is kept once. A line still cut by a tile edge is re-read in a
   thin strip centred on it, so a model code is not silently clipped.
4. Lines the engine is unsure of (below `OCR_MIN_CONFIDENCE`) and stray marks (rules, borders) are dropped;
   the rest is put in reading order (top to bottom, then left to right) and stored in `ocr_results`.
5. Pattern rules pick product facts out of the lines (see below) and decide the page type.

## The engine

**RapidOCR** — the PaddleOCR text models running on ONNX Runtime. One model reads Chinese and English
together (supplier catalogues mix both), it needs no GPU, and the models are inside the pip package, so
nothing is downloaded when the worker starts. It sits behind the `OcrProvider` interface
(`app/ml/ocr/base.py`); another engine can be added by implementing that interface and naming it in
`get_ocr_provider`. Every stored line records the engine name and version that produced it.

Known limits: rotated (vertical) text such as a "09 series" label printed along a page edge may be read
badly; text over busy photographs is harder than text on plain paper; lines wider than about 2000 px
may stay in two pieces. Real catalogues will show which of these matter; the engine, tile size and
confidence threshold are all settings.

## Product facts picked out of the text

`app/services/text_signals.py` — plain pattern rules, not machine learning:

| Fact | Examples recognised |
| --- | --- |
| model code | `型号: YY-21`, `MODEL: AB-123C`, a bare upper-case `YY-21` |
| size | `3200W x 1400D x 750H`, `3600W×1200D×750H`, `L1200*W600*H750 mm`, unlabelled `1200x600x750` (assumed width x depth x height) |
| series | `09 series`, `09series`, `09系列` |

These are **hints**: each result says which line it came from. Nothing is written to a product record
from here; a later phase combines them with the images, and an admin approves the result.

## Page types

Decided from the text alone, so deliberately modest (`app/services/page_classifier.py`):

| Type | Rule | Confidence |
| --- | --- | --- |
| `PRODUCT` | model codes **and** sizes found | 0.9 |
| `PRODUCT` | only one of them found | 0.75 |
| `COVER` | first page, no product text | 0.7 (0.5 if no text at all) |
| `UNKNOWN` | anything else — later stages must look at the picture | 0.3 |

The confidence is the rule's own certainty, not a statistical probability.

## Settings

| Setting | Default | Purpose |
| --- | --- | --- |
| `OCR_ENABLED` | true | `false` skips text reading (pages stay `RENDERED`) |
| `OCR_MIN_CONFIDENCE` | 0.5 | drop lines the engine is less sure of |
| `OCR_TILE_PX` | 1600 | tile size (640–2000) |
| `OCR_TILE_OVERLAP_PX` | 320 | overlap between tiles (at most half the tile) |

If the engine cannot start the job fails immediately with a clear message, before any page is rendered.

## Checking the engine

```powershell
docker compose run --rm backend python -m app.cli.check_ocr
```

Draws a small picture (a model code, a size, a series label), reads it back and prints the result.
Then read a real page with `GET /api/admin/catalogues/{id}/pages/{n}/text`.

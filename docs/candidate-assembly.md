# Grouping finds into product candidates (Phase 6)

Once a catalogue's panels and products have been found (Phase 5), this step groups the raw
finds on each page into one **product candidate** — an admin's starting point for turning
"a desk, a chair and a cabinet were found on page 8" into "this is the YY-11 four-person
staff desk". Nothing here is published: a candidate is always `NEEDS_REVIEW` or `READY`
(meaning "little for a person to fix", not "live"). Turning a candidate into a real, visible
`Product` is a separate, deliberate action for Phase 8.

## The rule

A supplier's PRODUCT page almost always shows one product: a room photo plus close-ups, with
one model code and one size printed once. So **every find on one page becomes one candidate**.
This is a simple, stated v1 rule — a page that genuinely shows more than one product (rare in
the BOGAO catalogue) is still grouped as one candidate today, flagged `MULTIPLE_MODEL_CODES_ON_PAGE`
or `MULTIPLE_DIMENSIONS_ON_PAGE` so a person notices and can split it by hand later.

## What a candidate gets

- **A suggested name**, read straight off the page: suppliers usually print the name and the
  size on the same OCR line ("四人位职员桌2400Wx1200D×750H"), so the name is whatever is left
  after the recognised size is removed.
- **The model code and size**, from the existing text-signal extraction (Phase 4).
- **A suggested category**, guessed from the AI labels of its finds against the category tree
  seeded in Phase 1 (e.g. `desk` → Workstations, `chandelier` → Chandeliers). Left unset when
  the labels don't agree or match nothing.
- **A confidence score**, `overall_confidence` (0 to 1), built from five weighted ingredients:
  how confident the finder was (25%), how good the outlines were (25%), how much of the name/
  code/size was actually read (15% OCR presence, 15% metadata completeness), and how sure we
  are the finds truly belong together (20% — full marks only when the page has exactly one
  model code and one size). The weights live as named constants in `app/services/candidates.py`
  so they can be retuned from real catalogues.
- **Plain-English flags** for anything to double-check: `NO_MODEL_CODE`, `NO_DIMENSIONS`,
  `MULTIPLE_MODEL_CODES_ON_PAGE`, `MULTIPLE_DIMENSIONS_ON_PAGE`, `SOME_OUTLINES_FAILED`,
  `SOME_OUTLINES_LOW_CONFIDENCE`, `LOW_CONFIDENCE` (overall score below the low-confidence
  threshold).

A candidate at or above `ai.auto_review_threshold` (default 0.85) is marked `READY`; otherwise
`NEEDS_REVIEW`. Both thresholds are stored in `app_settings` and can be changed from the
database without a redeploy (`app/services/settings_store.py`), falling back to the
`AI_AUTO_REVIEW_THRESHOLD` / `AI_LOW_CONFIDENCE_THRESHOLD` environment settings.

## Duplicate flags

If two candidates from the same run share a model code (a page processed twice, or a series
shown on two pages), a `DuplicateFlag` is created — `OPEN`, never auto-resolved — so an admin
can confirm or dismiss it. Comparing against already-published `Product` rows is not built yet
(candidates are only compared against each other within one assembly run); matching a new
catalogue's finds against the existing published catalogue, and merging the same product shown
in several sizes into one product with size options, are later steps.

## Running it

Assembly is a **separate, on-demand action** — it does not run automatically as part of the
processing pipeline, and can be re-run at any time:

```
POST /api/admin/catalogues/{catalogue_id}/candidates/assemble
```

Requires the catalogue to be `COMPLETED` or `PARTIALLY_COMPLETED` (409 otherwise). Rebuilds
every AI-origin candidate that no one has reviewed yet (`reviewed_at` is null); a candidate an
admin has approved, rejected, merged, or entered by hand is always left untouched, and the
page(s) it covers are skipped entirely rather than also getting a second, overlapping candidate.

| Address (under `/api/admin/catalogues/<id>`) | What it gives |
| --- | --- |
| `POST /candidates/assemble` | (re)builds the candidates; returns counts |
| `GET /candidates` | paginated summaries: name, model code, pages, thumbnail, status |
| `GET /candidates/{candidate_id}` | full detail: every field, flag, find, and duplicate flag |
| `GET /candidates/review` | a plain visual page (pictures, names, flags) for looking things over with someone who won't use the API directly - read only, no approve/reject yet |

## Not yet

* Turning a candidate into a real `Product` (approve / reject / edit-then-approve): Phase 8.
* Comparing a new candidate against already-published products, not just other candidates in
  the same run.
* Merging the same product shown in several sizes (4-, 6-, 8-person desk) into one product with
  size options — the decided design, still to build.
* Splitting a candidate that a flag says covers more than one product.

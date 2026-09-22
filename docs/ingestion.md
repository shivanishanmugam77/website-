# Catalogue ingestion

How a supplier's PDF becomes data the platform can work with. Phase 3 implements the first
stage; later phases add stages to the same pipeline without changing how uploads, jobs or
failures work.

## The flow

```
admin uploads PDF ──► API: check it looks like a PDF, stream it to storage, hash it
                      record Catalogue (UPLOADED) + Job (QUEUED) + audit entry
                      queue a background job            ──►  returns immediately (HTTP 201)

worker picks up the job (queue "pipeline")
   1. open the PDF                     (parse untrusted PDFs ONLY here, never in the API)
   2. for every page:
        render to a JPEG (150 dpi)  + a thumbnail
        extract the raster images embedded in the page, at their ORIGINAL resolution,
        with their position on the page                          [Phase 3 - done]
   3. OCR: read the text on each page, pick out model codes / sizes,
      decide the page type                                       [Phase 4 - done]
   4. cut every page into its photo panels (the pictures laid out on it), keep the
      text that belongs to each                                   [Phase 5a - done]
   5. detect + cut out each product inside a panel                [Phase 5b]
   6. group everything into product candidates, score confidence, flag duplicates  [Phase 6]
   ──► catalogue status COMPLETED / PARTIALLY_COMPLETED / FAILED
```

## Why native image extraction matters

Many catalogues embed each product photo as a real image inside the PDF. Extracting those
gives the *original pixels* — sharp, uncropped, with no text baked in — and needs no machine
learning at all. AI detection (Phase 5) is only needed for pages where products are part of
a flattened page image.

## Statuses

| Catalogue status | Meaning |
| --- | --- |
| `UPLOADED` | stored, waiting for a worker |
| `PROCESSING` | a worker is running the pipeline (see `processing_progress`, `current_stage`) |
| `COMPLETED` | every page completed every stage that is switched on |
| `PARTIALLY_COMPLETED` | finished, but some pages failed (see each page's `processing_error`) |
| `FAILED` | nothing usable: unreadable PDF, too many pages, no page could be processed, or the job could not be queued |

A page's status is the last stage it completed: `RENDERED` (picture and images done), `TEXT_READ` (text read
and page typed), `PANELS_FOUND` (cut into photo panels) or `FAILED`. If a later stage fails for one page, the
page keeps its status and records `error_stage` (`read_text` or `find_panels`) with the reason, later stages
skip it, and the catalogue ends `PARTIALLY_COMPLETED`. One bad page never loses the catalogue.
Failures always carry a human-readable reason; internal error details go to the server log only.

## Uploading repeatedly

Suppliers send new catalogues over time; each upload is independent.

* The **exact same file** (SHA-256) uploaded twice is refused with `409` and the id of the
  existing catalogue, unless `allow_duplicate=true`.
* A **new version** of a catalogue is a different file, so it is accepted as a new catalogue.
  Products that appear in both are caught later by duplicate detection (Phase 6), which
  *flags* them for an admin and never merges automatically.
* **Reprocess** re-runs a catalogue from scratch (after a failure, or when a new pipeline
  stage exists) and replaces its previous results. **Delete** removes the catalogue, its
  derived data and its files; published products survive it.

## Limits and safety

| Setting | Default | Purpose |
| --- | --- | --- |
| `MAX_UPLOAD_MB` | 200 | upload size cap (checked up front and while streaming) |
| `MAX_PDF_PAGES` | 500 | refuse absurdly long PDFs |
| `MAX_PAGE_PIXELS` | 40 million | a huge page is rendered smaller instead of exhausting memory |
| `PIPELINE_TIME_LIMIT_MINUTES` | 120 | graceful stop, then hard kill 5 minutes later |
| `WORKER_CONCURRENCY` | 2 | parallel jobs per worker (memory) |

* The API never parses a PDF; PDFium (native code) runs only in worker processes.
* Storage keys are generated from UUIDs, never from file names; every key is validated
  and cannot escape the storage root. Uploaded file names are sanitised and only displayed.
* Page and image files are served only to admins, marked `private` and `nosniff`.
* Redis' visibility timeout is set above the task time limit so a long job is never
  delivered to a second worker while the first is still running.

## Storage layout

```
catalogues/<catalogue-id>/original.pdf
catalogues/<catalogue-id>/pages/0001.jpg          full page render
catalogues/<catalogue-id>/pages/0001_thumb.jpg    thumbnail
catalogues/<catalogue-id>/source-images/<id>.jpg  an embedded image, native resolution
```

Locally this is the Docker volume `storage_data`. An S3-compatible backend is planned
(Phase 13) behind the same interface (`app/services/storage.py`).

## What is recorded for every extracted image

`source_images` rows carry: the page it belongs to, its kind (`PAGE` = the whole rendered
page, `EMBEDDED` = an image from the PDF), pixel size, position on the page image (or none when
it cannot be determined reliably, e.g. images nested inside form XObjects), SHA-256 and a
64-bit perceptual hash (`phash`) for near-duplicate detection.

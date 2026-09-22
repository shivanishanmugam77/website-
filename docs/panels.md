# Photo panels (Phase 5a)

A catalogue page is usually a sheet of paper with several pictures laid out on it: a big room
scene, close-ups beside it, a row of products on white. Each picture is a **panel**. After the
text is read, the worker cuts every page into its panels and stores each one as its own image.
No machine-learning model is involved, so this step needs no downloads and runs in a fraction of
a second per page.

Why panels matter: the pictures a customer swipes through on a product page (the whole set, the
close-ups) are exactly these panels, and a panel is where the later AI step (Phase 5b) looks for
the individual desk, chair or cabinet to cut out for the main page.

## How a page is cut up

`app/services/panels.py`:

1. The paper colour is estimated from the page margins. If the margins are not one plain colour
   the page is a single full-bleed picture and is returned as one panel.
2. Every pixel that differs from the paper is marked.
3. Printed text on paper is removed from the marks (the text engine already knows where the text
   is), so a caption right under a photo does not become part of it. Text printed *on* a photo is
   left alone.
4. Tiny gaps are closed and each connected blob becomes a candidate; a blob is split wherever a
   clean strip of paper runs right across it (this also separates photos joined by a thin line).
5. Candidates that are too small (`PANEL_MIN_AREA_SHARE`, default 0.4 % of the page), too thin or
   too empty (rules, outlines) are dropped, and so is any candidate that lies inside a bigger one (a
   framed print on the wall of a room photo is part of that photo, not a picture of its own).
6. Panels are ordered top to bottom in rows, then left to right.

Each panel is stored as a `source_images` row of kind `REGION` (box on the page, cut-out JPEG at
the full page resolution). Cover pages are skipped.

## Which text belongs to which panel

Computed when you ask (`/panels`), not stored: a text line belongs to the smallest panel that
contains its centre; a line just below a panel (within 6 % of the page height) and lined up with it
is that panel's caption; anything else (page headings, margin notes) belongs to no panel. For each
panel the answer includes the lines, and the model codes / sizes / series found in them.

## Checking the result

A whole catalogue in one picture (each page's thumbnail with its panels outlined and numbered;
add `?start=9&limit=8` to page through a long one):

```
http://localhost:8000/api/admin/catalogues/<catalogue id>/panels/overview
```

One page in detail, in the browser (you must be signed in):

```
http://localhost:8000/api/admin/catalogues/<catalogue id>/pages/<page number>/panels/preview
```

It shows the page with every panel outlined and numbered, in the same order as `/panels`.

## Known limits

* Pictures that touch with no paper between them come out as one panel.
* A photo with a strip of paper-coloured pixels running all the way across it (a white wall) may be
  cut in two.
* Slanted or irregular panels get a rectangular box around them.
* If the text engine found nothing on a page (it is off, or failed), large printed words can show up
  as panels.
* A product photographed against a white studio background *inside* a panel is not separated from it
  here; that is the job of Phase 5b.

Real catalogues will show which of these matter; the preview picture is the way to see it.

## Settings

| Setting | Default | Purpose |
| --- | --- | --- |
| `PANEL_DETECTION_ENABLED` | true | `false` skips this step (pages stay `TEXT_READ`) |
| `PANEL_MIN_AREA_SHARE` | 0.004 | ignore pictures smaller than this share of the page |

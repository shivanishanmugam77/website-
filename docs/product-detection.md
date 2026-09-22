# Finding and cutting out products (Phase 5b, steps 2 and 3)

A photo panel shows a room, a desk, or a row of chairs. This step finds each product inside a
panel (a desk, an office chair, a cabinet ...) so it can be cut out on its own for the main page,
and so the panel's other pictures can be linked to it. Step 2 delivered the models and a way to
try them; step 3 (below) runs them as the fourth stage of the processing pipeline and saves the
results.

## The two models

| Job | Model | Size | Licence |
| --- | --- | --- | --- |
| Find things described in words ("desk", "office chair") | Grounding DINO, tiny | ~700 MB | Apache-2.0 |
| Draw the exact outline of each find | SAM (Segment Anything), ViT-B | ~375 MB | Apache-2.0 |

Both run on the CPU through PyTorch (CPU build) and Hugging Face `transformers`; a graphics card
is used automatically if one is available (`ML_DEVICE`). Expect roughly 5 to 40 seconds per panel
on a laptop without one. Both licences allow commercial use.

They sit behind two small interfaces, `DetectionProvider` and `SegmentationProvider`
(`app/ml/detection/base.py`, `app/ml/segmentation/base.py`), so either model can be swapped for a
different one without touching the rest of the system. Every result records the model that made it.

## What the finder is asked

`DETECTION_PROMPTS` (a comma separated list, default: desk, office chair, cabinet, sofa, meeting
table, bookshelf, filing cabinet, locker, reception desk, workstation, table, chair). The model
answers with a box, a score and the words it matched; `app/services/detections.py` then

1. maps the words back to one of the phrases asked for (`office chair`, not `office`),
2. cuts boxes to the picture and drops specks (under 1 % of the picture or under 24 px wide),
3. keeps one box per object: two boxes of the same kind overlapping by half are one object seen
   twice; boxes of different kinds are merged only when they almost coincide (a chair under a
   desk is two things).

`DETECTION_BOX_THRESHOLD` (default 0.35) and `DETECTION_TEXT_THRESHOLD` (0.25) say how sure the
model must be. Lower finds more, and more mistakes. Panels larger than `DETECTION_MAX_PX` (1024)
on their longest side are shrunk first, which keeps it fast; boxes are converted back.

## Trying it

One-time download of the models (about 1.1 GB, kept in a Docker volume so a rebuild keeps them):

```
docker compose run --rm backend python -m app.cli.download_models
```

Then try the finder on a page of a catalogue already processed (nothing is saved):

```
docker compose run --rm backend python -m app.cli.check_products --catalogue <catalogue id> --page 8
```

It prints, for every photo panel, what it found with confidence, position and the time taken, and
writes a picture with the finds outlined to the **`debug`** folder of the project. Options:
`--panel 2` (only that panel), `--masks` (also draw each find's exact outline; slower).

### Choosing the outline

For every find the outlining model offers three outlines: a small part of the object, a bigger
part, and the whole object. Its own favourite is often the small part (for a long desk, just the
near half). `app/services/masks.py` therefore uses the **largest** outline among those the model
rates within 0.10 of its best one, and cuts away anything outside the find's box. With `--masks`
the command prints, for each find, how much of its box the outline fills ("outline fills 62% of
its box"). A very low number (for example under 30%) means the outline probably covers only part
of the product and will need a correction. The 0.10 tolerance is a tuning knob
(`SCORE_TOLERANCE`) to be set from real catalogue pages.

## As a pipeline stage (step 3)

Switched on by `PRODUCT_DETECTION_ENABLED` (on in `docker-compose.yml`, off in the code so that
tests and small set-ups never load the models). It needs `PANEL_DETECTION_ENABLED` and the
downloaded models; if the models cannot be loaded the job fails at once, before any page is
rendered, with a message that says so.

After the panels are cut, every panel of every page goes through the finder and the outliner.
For each find the system saves a row in `detected_objects` and one in `segmentation_results`,
and five pictures under `catalogues/<id>/products/<find id>/`:

| Picture | What it is |
| --- | --- |
| `crop.jpg` | the find's box cut from the panel, with its background |
| `mask.png` | black and white, white where the object is (the size of the panel) |
| `cutout.png` | the object alone on a transparent background, trimmed to its edges |
| `white.jpg` | the cut-out on plain white: the usual shop picture |
| `thumb.jpg` | a small version of `white.jpg` |

Each outline gets a status: `SUCCEEDED`; `LOW_CONFIDENCE` when the outlining model rates it below
`PRODUCT_MIN_OUTLINE_SCORE` (0.70) or it fills less than `PRODUCT_MIN_OUTLINE_COVERAGE` (0.30) of
its box (kept, but a person should look, and the reason is in `error`); or `FAILED` when the
outline is empty (only the crop is saved). A page whose products cannot be found keeps its panels,
records the error on the page, and the catalogue ends up `PARTIALLY_COMPLETED`.

Once every stage has succeeded a page's status is `COMPLETED` (it was `PANELS_FOUND` before this
stage existed, and still is when `PRODUCT_DETECTION_ENABLED=false`). Expect roughly 15 to 40
minutes more for a 16-page catalogue on a laptop processor (about 12 to 16 seconds per panel was
measured). While the catalogue is processing its panels cannot be edited.

### Looking at the results (admin only, read only)

| Address (under `/api/admin/catalogues/<id>`) | What it gives |
| --- | --- |
| `/pages/<n>/products` | the page's finds as data: label, confidence, box, outline status and score, note, picture addresses |
| `/pages/<n>/products/preview` | the picture you have seen in `debug`: every panel with its finds boxed and tinted |
| `/products/overview` | many products cut out on white, each captioned with its page and name (`- check` marks doubtful outlines); options `start`, `limit`, `columns` |
| `/products/<find id>/image/<kind>` | one picture: `crop`, `mask`, `cutout`, `white` or `thumbnail` |

### When a panel is changed by hand

The finds belong to the picture they were found on. Moving, resizing or deleting a panel in the
panel editor (or undoing such a change) therefore forgets the products found in it, together with
their pictures. Panels an admin draws or changes are not searched again automatically: a way to
search just those panels arrives with the correction editor for products.

## Not yet

* Correcting a wrong find or outline by hand, and searching again in a panel you changed: next.
* Turning finds into product candidates (grouping the room photo, close-ups, model code and
  sizes): Phase 6.

# Correcting panels by hand (Phase 5b, step 1)

The automatic panel cutting (see `panels.md`) is right most of the time, not always: pictures that
touch come out as one panel, a caption can pull a box too wide, a missed picture has no box. The
panel editor lets an admin fix that with the mouse, before the AI steps build on the panels.

## Opening it

While signed in as an admin, open (change the page number at the end):

```
http://localhost:8000/api/admin/catalogues/<catalogue id>/pages/<page number>/panels/editor
```

Find the pages that need attention with the overview picture
(`.../panels/overview`, see `panels.md`).

## Using it

| To | Do this |
| --- | --- |
| Move a panel | drag the box |
| Resize a panel | click it, then drag one of its round handles |
| Add a panel the system missed | press **Add a panel**, then drag over the picture |
| Delete a panel | click it, then **Delete selected** (or the Delete key) |
| Undo the last change | **Undo** (or Ctrl+Z); press it again to go further back |
| Change page | **Previous page** / **Next page** |

Every change is saved the moment you let go of the mouse, the cut-out picture is re-made from
the new box, and the text lines that belong to each panel update by themselves. The list on the
right shows each panel's cut-out, size and text; panels you drew or adjusted are labelled.

## What is recorded

* **`panel_edits`** — one row per change with the box before and after, who made it and when.
  This is what Undo replays. Undone changes stay in the table, marked as undone.
* **The audit trail** — `PANEL_CREATED`, `PANEL_UPDATED`, `PANEL_DELETED` and `PANEL_EDIT_UNDONE`
  with the admin, the time, the address the request came from and the before/after boxes.
* **`source_images.origin`** — `AI` for a panel as the system cut it, `AI_HUMAN_REVIEW` for one an
  admin moved or resized, `HUMAN` for one an admin drew.

## Rules

* A panel is at least 20 pixels wide and tall, and inside the page (boxes that stick out are
  trimmed to the page). A page holds at most 60 panels.
* Panels can only be edited while the catalogue is not being processed.
* **Reprocessing a catalogue rebuilds every page, so it would lose your corrections.** It is
  therefore refused (HTTP 409) while corrections are in force, unless you deliberately add
  `discard_corrections=true`; that is recorded in the audit trail (`CORRECTIONS_DISCARDED`).
  Changes you have undone no longer count.
* Two people editing one page cannot interleave: edits to a page are taken one at a time.

## The API behind it

All under `/api/admin/catalogues/{id}/pages/{n}/panels`, admin only; each call returns the
page's panels afterwards (the same shape as `GET .../panels`).

| Method and path | Purpose | Body |
| --- | --- | --- |
| `POST .../panels` | add a panel | `{"bbox": [left, top, right, bottom]}` in page pixels |
| `PATCH .../panels/{panel id}` | move or resize | the new `bbox` |
| `DELETE .../panels/{panel id}` | delete | none |
| `POST .../panels/undo` | undo the last change | none |
| `GET .../panels/editor` | the editor page | none |

The editor page is an interim tool: the real admin interface (Phase 7) will use the same calls.
It is served with a strict content-security policy (a fresh nonce for its one script), and every
change is sent with the same-origin and admin checks as any other admin request.

## Not included

* Redo (undo a change, then bring it back). Undo a different way instead: make the change again.
* Keeping corrections through a reprocess (they are protected from it, not merged into it).

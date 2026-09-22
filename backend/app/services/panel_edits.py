"""Manual correction of a page's photo panels: draw a missed one, move or resize one, delete
one that is not needed, and undo the last change.

Rules:
* Every change is recorded twice: in ``panel_edits`` (before/after snapshots, which is what
  undo replays) and in the audit trail (who, when, from where).
* A panel the system cut and an admin then adjusted is marked ``AI_HUMAN_REVIEW``; one an
  admin drew is ``HUMAN``. Reprocessing refuses to discard such work unless told to.
* Edits to one page are serialised by locking the page row, so two admins (or two browser
  tabs) cannot interleave and confuse the undo history.
* The cut-out picture is always re-made from the stored page render, and its file is written
  only after the database accepted the change; a file that has to disappear is removed only
  after the commit.
* Products found inside a panel belong to the picture they were found on, so moving, resizing
  or deleting the panel forgets them (their pictures are removed after the commit too).
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

from PIL import Image
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.models import Catalogue, CataloguePage, PanelEdit, SourceImage, User
from app.models.enums import AuditAction, DataOrigin, PanelEditAction, SourceImageKind
from app.services.audit import record_audit
from app.services.catalogues import has_active_job
from app.services.panel_boxes import (
    NothingToUndoError,
    PageNotEditableError,
    PanelNotFoundError,
    TooManyPanelsError,
    validate_box,
)
from app.services.pdf import dhash, encode_jpeg
from app.services.pipeline import source_image_key
from app.services.product_cleanup import forget_finds
from app.services.sessions import utcnow
from app.services.storage import ObjectNotFoundError, Storage

MAX_PANELS_PER_PAGE = 60


@dataclass(frozen=True)
class EditContext:
    """Everything an edit needs besides the panel itself."""

    session: Session
    storage: Storage
    settings: Settings
    catalogue: Catalogue
    page: CataloguePage
    actor: User
    ip: str | None
    # Files that stop belonging to anything during this edit (the pictures of products found
    # in a panel that has changed); removed once the change is committed.
    stale_files: list[str] = field(default_factory=list)


# ------------------------------------------------------------------------------ boxes
def _checked(ctx: EditContext, box: Sequence[float]) -> tuple[int, int, int, int]:
    return validate_box(box, ctx.page.width_px or 0, ctx.page.height_px or 0)


def _snapshot(panel: SourceImage) -> dict[str, Any]:
    return {
        "bbox": [panel.bbox_x0, panel.bbox_y0, panel.bbox_x1, panel.bbox_y1],
        "origin": panel.origin.value,
    }


def _box_of(panel: SourceImage) -> tuple[float, float, float, float]:
    return (panel.bbox_x0 or 0.0, panel.bbox_y0 or 0.0, panel.bbox_x1 or 0.0, panel.bbox_y1 or 0.0)


# ------------------------------------------------------------------------------ plumbing
def _begin(ctx: EditContext) -> None:
    """Refuse unless the page can be edited right now, and take the page's edit lock."""
    if ctx.page.image_key is None or ctx.page.width_px is None or ctx.page.height_px is None:
        raise PageNotEditableError("This page has no picture to edit")
    if has_active_job(ctx.session, ctx.catalogue.id):
        raise PageNotEditableError(
            "This catalogue is being processed; panels can be edited once it has finished "
            "(if a job is stuck, reprocess with force=true first)"
        )
    ctx.session.execute(
        select(CataloguePage.id).where(CataloguePage.id == ctx.page.id).with_for_update()
    )


def _find(ctx: EditContext, panel_id: uuid.UUID) -> SourceImage:
    panel = ctx.session.scalar(
        select(SourceImage).where(
            SourceImage.id == panel_id,
            SourceImage.page_id == ctx.page.id,
            SourceImage.kind == SourceImageKind.REGION,
        )
    )
    if panel is None:
        raise PanelNotFoundError("That panel does not exist on this page")
    return panel


def _cut_out(ctx: EditContext, box: tuple[int, int, int, int]) -> tuple[bytes, Image.Image]:
    try:
        data = ctx.storage.read_bytes(ctx.page.image_key or "")
    except ObjectNotFoundError:
        raise PageNotEditableError("The stored page picture is missing") from None
    with Image.open(BytesIO(data)) as opened:
        crop = opened.convert("RGB").crop(box)
    return encode_jpeg(crop, ctx.settings.jpeg_quality), crop


def _apply(panel: SourceImage, box: tuple[int, int, int, int], crop: Image.Image) -> None:
    panel.bbox_x0, panel.bbox_y0, panel.bbox_x1, panel.bbox_y1 = box
    panel.width_px, panel.height_px = crop.size
    panel.phash = dhash(crop)


def _log(
    ctx: EditContext,
    action: PanelEditAction,
    audit_action: AuditAction,
    panel_id: uuid.UUID,
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
) -> None:
    last = ctx.session.scalar(
        select(func.max(PanelEdit.seq)).where(PanelEdit.page_id == ctx.page.id)
    )
    ctx.session.add(
        PanelEdit(
            page_id=ctx.page.id,
            seq=(last or 0) + 1,
            action=action,
            panel_id=panel_id,
            before=before,
            after=after,
            actor_id=ctx.actor.id,
        )
    )
    _audit(ctx, audit_action, panel_id, {"before": before, "after": after})


def _audit(
    ctx: EditContext, action: AuditAction, panel_id: uuid.UUID, details: dict[str, Any]
) -> None:
    record_audit(
        ctx.session,
        action,
        actor=ctx.actor,
        entity_type="source_image",
        entity_id=str(panel_id),
        details={
            "catalogue_id": str(ctx.catalogue.id),
            "page_number": ctx.page.page_number,
            **details,
        },
        ip_address=ctx.ip,
    )


def _finish(ctx: EditContext, stale_keys: Sequence[str] = ()) -> None:
    """Commit, then remove files that no longer belong to anything."""
    ctx.session.commit()
    for key in (*stale_keys, *ctx.stale_files):
        ctx.storage.delete(key)


# ------------------------------------------------------------------------------ the edits
def create_panel(ctx: EditContext, box: Sequence[float]) -> SourceImage:
    """Add a panel the system missed."""
    _begin(ctx)
    checked = _checked(ctx, box)
    existing = ctx.session.scalar(
        select(func.count())
        .select_from(SourceImage)
        .where(SourceImage.page_id == ctx.page.id, SourceImage.kind == SourceImageKind.REGION)
    )
    if (existing or 0) >= MAX_PANELS_PER_PAGE:
        raise TooManyPanelsError(f"A page can have at most {MAX_PANELS_PER_PAGE} panels")
    return _add(ctx, uuid.uuid4(), checked, DataOrigin.HUMAN, log=True)


def _add(
    ctx: EditContext,
    panel_id: uuid.UUID,
    box: tuple[int, int, int, int],
    origin: DataOrigin,
    *,
    log: bool,
) -> SourceImage:
    jpeg, crop = _cut_out(ctx, box)
    key = source_image_key(ctx.catalogue.id, panel_id)
    panel = SourceImage(
        id=panel_id,
        page_id=ctx.page.id,
        kind=SourceImageKind.REGION,
        storage_key=key,
        width_px=crop.width,
        height_px=crop.height,
        origin=origin,
        sha256=hashlib.sha256(jpeg).hexdigest(),
    )
    _apply(panel, box, crop)
    ctx.session.add(panel)
    ctx.session.flush()  # the database accepts the row before any file is written
    ctx.storage.put_bytes(key, jpeg)
    if log:
        _log(
            ctx,
            PanelEditAction.CREATE,
            AuditAction.PANEL_CREATED,
            panel_id,
            None,
            _snapshot(panel),
        )
        _finish(ctx)
    return panel


def update_panel(ctx: EditContext, panel_id: uuid.UUID, box: Sequence[float]) -> SourceImage:
    """Move or resize a panel."""
    _begin(ctx)
    panel = _find(ctx, panel_id)
    checked = _checked(ctx, box)
    if tuple(float(v) for v in checked) == _box_of(panel):
        return panel  # nothing changed: nothing to record
    before = _snapshot(panel)
    origin = DataOrigin.HUMAN if panel.origin is DataOrigin.HUMAN else DataOrigin.AI_HUMAN_REVIEW
    _recut(ctx, panel, checked, origin)
    _log(ctx, PanelEditAction.UPDATE, AuditAction.PANEL_UPDATED, panel_id, before, _snapshot(panel))
    _finish(ctx)
    return panel


def _recut(
    ctx: EditContext, panel: SourceImage, box: tuple[int, int, int, int], origin: DataOrigin
) -> None:
    jpeg, crop = _cut_out(ctx, box)
    ctx.stale_files.extend(forget_finds(ctx.session, panel.id))  # found on the old picture
    _apply(panel, box, crop)
    panel.origin = origin
    panel.sha256 = hashlib.sha256(jpeg).hexdigest()
    ctx.session.flush()  # the database accepts the change before the file is replaced
    ctx.storage.put_bytes(panel.storage_key, jpeg)


def delete_panel(ctx: EditContext, panel_id: uuid.UUID) -> None:
    """Remove a panel that is not needed (its history stays, so it can be undone)."""
    _begin(ctx)
    panel = _find(ctx, panel_id)
    before, key = _snapshot(panel), panel.storage_key
    ctx.stale_files.extend(forget_finds(ctx.session, panel.id))
    ctx.session.delete(panel)
    ctx.session.flush()
    _log(ctx, PanelEditAction.DELETE, AuditAction.PANEL_DELETED, panel_id, before, None)
    _finish(ctx, [key])


def undo_last(ctx: EditContext) -> PanelEditAction:
    """Revert the most recent change on the page that has not been undone yet."""
    _begin(ctx)
    edit = ctx.session.scalar(
        select(PanelEdit)
        .where(PanelEdit.page_id == ctx.page.id, PanelEdit.undone_at.is_(None))
        .order_by(PanelEdit.seq.desc())
        .limit(1)
    )
    if edit is None:
        raise NothingToUndoError("There is nothing to undo on this page")

    stale: list[str] = []
    panel = ctx.session.scalar(
        select(SourceImage).where(
            SourceImage.id == edit.panel_id, SourceImage.page_id == ctx.page.id
        )
    )
    if edit.action is PanelEditAction.CREATE:
        if panel is not None:
            stale.append(panel.storage_key)
            ctx.stale_files.extend(forget_finds(ctx.session, panel.id))
            ctx.session.delete(panel)
    elif edit.action is PanelEditAction.UPDATE:
        if panel is not None and edit.before:
            box = _checked(ctx, edit.before["bbox"])
            _recut(ctx, panel, box, DataOrigin(edit.before["origin"]))
    elif edit.action is PanelEditAction.DELETE and panel is None and edit.before:
        # a deletion: bring the panel back under its old id
        box = _checked(ctx, edit.before["bbox"])
        _add(ctx, edit.panel_id, box, DataOrigin(edit.before["origin"]), log=False)

    edit.undone_at = utcnow()
    _audit(
        ctx,
        AuditAction.PANEL_EDIT_UNDONE,
        edit.panel_id,
        {"undone": edit.action.value, "seq": edit.seq, "before": edit.before, "after": edit.after},
    )
    _finish(ctx, stale)
    return edit.action

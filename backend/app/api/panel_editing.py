"""Editing a page's photo panels by hand: draw, move/resize, delete, undo, and the page that
lets an admin do it with the mouse. (The real admin interface arrives in Phase 7; this
editor talks to the same API that interface will use.)"""

from __future__ import annotations

import secrets
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import HTMLResponse

from app.api.catalogues import build_panel_list, load_catalogue, load_page
from app.api.deps import (
    AdminUser,
    ClientInfoDep,
    DbSession,
    SettingsDep,
    StorageDep,
    require_admin,
)
from app.schemas.catalogue import PanelBoxIn, PanelOut
from app.services import panel_boxes as boxes
from app.services import panel_edits as edits

router = APIRouter(
    prefix="/admin/catalogues", tags=["panel editing"], dependencies=[Depends(require_admin)]
)

EDITOR_TEMPLATE = Path(__file__).resolve().parent.parent / "web" / "panel_editor.html"


def _context(
    db: DbSession,
    storage: StorageDep,
    settings: SettingsDep,
    admin: AdminUser,
    client: ClientInfoDep,
    catalogue_id: uuid.UUID,
    page_number: int,
) -> edits.EditContext:
    catalogue = load_catalogue(db, catalogue_id)
    page = load_page(db, catalogue_id, page_number)
    return edits.EditContext(
        session=db,
        storage=storage,
        settings=settings,
        catalogue=catalogue,
        page=page,
        actor=admin,
        ip=client.ip,
    )


def _refuse(exc: boxes.PanelEditError) -> HTTPException:
    if isinstance(exc, boxes.InvalidBoxError):
        code = status.HTTP_422_UNPROCESSABLE_ENTITY
    elif isinstance(exc, boxes.PanelNotFoundError):
        code = status.HTTP_404_NOT_FOUND
    else:  # page not editable, nothing to undo, too many panels
        code = status.HTTP_409_CONFLICT
    return HTTPException(code, detail=str(exc))


@router.post(
    "/{catalogue_id}/pages/{page_number}/panels",
    response_model=list[PanelOut],
    status_code=status.HTTP_201_CREATED,
)
def add_panel(
    catalogue_id: uuid.UUID,
    page_number: int,
    body: PanelBoxIn,
    db: DbSession,
    storage: StorageDep,
    settings: SettingsDep,
    admin: AdminUser,
    client: ClientInfoDep,
) -> list[PanelOut]:
    """Add a panel the system missed. Returns the page's panels afterwards."""
    ctx = _context(db, storage, settings, admin, client, catalogue_id, page_number)
    try:
        edits.create_panel(ctx, body.bbox)
    except boxes.PanelEditError as exc:
        raise _refuse(exc) from None
    return build_panel_list(db, catalogue_id, ctx.page)


@router.post("/{catalogue_id}/pages/{page_number}/panels/undo", response_model=list[PanelOut])
def undo_panel_edit(
    catalogue_id: uuid.UUID,
    page_number: int,
    db: DbSession,
    storage: StorageDep,
    settings: SettingsDep,
    admin: AdminUser,
    client: ClientInfoDep,
) -> list[PanelOut]:
    """Revert the last change made to this page's panels. Returns the panels afterwards."""
    ctx = _context(db, storage, settings, admin, client, catalogue_id, page_number)
    try:
        edits.undo_last(ctx)
    except boxes.PanelEditError as exc:
        raise _refuse(exc) from None
    return build_panel_list(db, catalogue_id, ctx.page)


@router.patch(
    "/{catalogue_id}/pages/{page_number}/panels/{panel_id}", response_model=list[PanelOut]
)
def move_panel(
    catalogue_id: uuid.UUID,
    page_number: int,
    panel_id: uuid.UUID,
    body: PanelBoxIn,
    db: DbSession,
    storage: StorageDep,
    settings: SettingsDep,
    admin: AdminUser,
    client: ClientInfoDep,
) -> list[PanelOut]:
    """Move or resize a panel. Returns the page's panels afterwards."""
    ctx = _context(db, storage, settings, admin, client, catalogue_id, page_number)
    try:
        edits.update_panel(ctx, panel_id, body.bbox)
    except boxes.PanelEditError as exc:
        raise _refuse(exc) from None
    return build_panel_list(db, catalogue_id, ctx.page)


@router.delete(
    "/{catalogue_id}/pages/{page_number}/panels/{panel_id}", response_model=list[PanelOut]
)
def remove_panel(
    catalogue_id: uuid.UUID,
    page_number: int,
    panel_id: uuid.UUID,
    db: DbSession,
    storage: StorageDep,
    settings: SettingsDep,
    admin: AdminUser,
    client: ClientInfoDep,
) -> list[PanelOut]:
    """Delete a panel that is not needed (it can be brought back with undo). Returns the
    page's panels afterwards."""
    ctx = _context(db, storage, settings, admin, client, catalogue_id, page_number)
    try:
        edits.delete_panel(ctx, panel_id)
    except boxes.PanelEditError as exc:
        raise _refuse(exc) from None
    return build_panel_list(db, catalogue_id, ctx.page)


def _editor_template() -> str:
    # Read per request (a small file): editing it never needs a server restart.
    return EDITOR_TEMPLATE.read_text(encoding="utf-8")


@router.get(
    "/{catalogue_id}/pages/{page_number}/panels/editor", response_class=HTMLResponse
)
def panel_editor(catalogue_id: uuid.UUID, page_number: int, db: DbSession) -> HTMLResponse:
    """The page to correct panels with the mouse: drag to move, drag the corners to resize,
    draw to add, Delete to remove, Ctrl+Z to undo."""
    load_page(db, catalogue_id, page_number)  # 404 for a page that does not exist
    nonce = secrets.token_urlsafe(16)
    html = (
        _editor_template()
        .replace("__NONCE__", nonce)
        .replace("__CATALOGUE_ID__", str(catalogue_id))
        .replace("__PAGE_NUMBER__", str(int(page_number)))
    )
    return HTMLResponse(
        html,
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": (
                "default-src 'none'; img-src 'self'; connect-src 'self'; "
                f"script-src 'nonce-{nonce}'; style-src 'nonce-{nonce}'; "
                "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
            ),
        },
    )

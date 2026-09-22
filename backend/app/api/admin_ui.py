"""The first real screens of the admin website (Phase 7): sign in, a list of catalogues with
an upload form, and a per-catalogue page with the actions an admin actually needs (reprocess,
delete, group into candidates) plus links out to the panel editor, the panels/products
pictures, and the candidate review page.

Same pattern as the panel editor and candidate review page: one self-contained HTML file per
screen, served same-origin (so the existing session cookies just work, no CORS needed), with
a nonce-based Content-Security-Policy. The sign-in page is the one route here that must stay
reachable without already being signed in - everything else requires the admin role.
"""

from __future__ import annotations

import html
import secrets
import uuid
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from app.api.catalogues import load_catalogue
from app.api.deps import DbSession

router = APIRouter(prefix="/admin", tags=["admin ui"])

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
NO_STORE_HEADERS = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}


def _csp(nonce: str) -> str:
    return (
        "default-src 'none'; img-src 'self'; connect-src 'self'; "
        f"script-src 'nonce-{nonce}'; style-src 'nonce-{nonce}'; "
        "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    )


def _page(template_name: str, replacements: dict[str, str]) -> HTMLResponse:
    nonce = secrets.token_urlsafe(16)
    text = (WEB_DIR / template_name).read_text(encoding="utf-8")
    text = text.replace("__NONCE__", nonce)
    for placeholder, value in replacements.items():
        text = text.replace(placeholder, value)
    return HTMLResponse(text, headers={**NO_STORE_HEADERS, "Content-Security-Policy": _csp(nonce)})


@router.get("/login", response_class=HTMLResponse)
def admin_login_page() -> HTMLResponse:
    """Sign in and land on the dashboard - the one admin page that works while signed out."""
    return _page("admin_login.html", {})


@router.get("/dashboard", response_class=HTMLResponse)
def admin_dashboard_page() -> HTMLResponse:
    """Every catalogue, with an upload form.

    This page itself carries no admin check: it is just a shell, and its own JavaScript calls
    admin-only JSON endpoints straight away, which sends a signed-out (or non-admin) visitor
    to ``/admin/login`` immediately. Checking here too would only replace that friendly
    redirect with a raw JSON error on a first visit or a stale bookmark.
    """
    return _page("admin_dashboard.html", {})


@router.get("/catalogues/{catalogue_id}/dashboard", response_class=HTMLResponse)
def admin_catalogue_dashboard_page(catalogue_id: uuid.UUID, db: DbSession) -> HTMLResponse:
    """One catalogue: status, progress, reprocess/delete/assemble, and links to the rest of
    what exists for it (panel editor, panels/products pictures, candidate review). Like the
    dashboard above, the admin check happens in the page's own API calls, not the page itself."""
    load_catalogue(db, catalogue_id)  # 404 for a catalogue that does not exist
    return _page("admin_catalogue.html", {"__CATALOGUE_ID__": html.escape(str(catalogue_id))})

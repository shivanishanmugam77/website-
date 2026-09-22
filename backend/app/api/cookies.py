"""Session cookies.

Both cookies are HttpOnly (invisible to JavaScript, so XSS cannot steal them) and
SameSite. The access cookie is sent to every API route; the refresh cookie is scoped to
/api/auth so it travels only where it is needed.
"""

from __future__ import annotations

from fastapi import Response

from app.core.config import Settings
from app.services.sessions import IssuedTokens

ACCESS_COOKIE = "access_token"
REFRESH_COOKIE = "refresh_token"
ACCESS_COOKIE_PATH = "/api"
REFRESH_COOKIE_PATH = "/api/auth"


def set_session_cookies(response: Response, tokens: IssuedTokens, settings: Settings) -> None:
    common = {
        "httponly": True,
        "secure": settings.use_secure_cookies,
        "samesite": settings.cookie_samesite,
        "domain": settings.cookie_domain,
    }
    response.set_cookie(
        ACCESS_COOKIE,
        tokens.access_token,
        max_age=tokens.access_max_age,
        path=ACCESS_COOKIE_PATH,
        **common,
    )
    response.set_cookie(
        REFRESH_COOKIE,
        tokens.refresh_token,
        max_age=tokens.refresh_max_age,
        path=REFRESH_COOKIE_PATH,
        **common,
    )
    # Responses that set credentials must never be cached by browsers or proxies.
    response.headers["Cache-Control"] = "no-store"


def clear_session_cookies(response: Response, settings: Settings) -> None:
    for name, path in ((ACCESS_COOKIE, ACCESS_COOKIE_PATH), (REFRESH_COOKIE, REFRESH_COOKIE_PATH)):
        response.delete_cookie(
            name,
            path=path,
            domain=settings.cookie_domain,
            secure=settings.use_secure_cookies,
            httponly=True,
            samesite=settings.cookie_samesite,
        )
    response.headers["Cache-Control"] = "no-store"

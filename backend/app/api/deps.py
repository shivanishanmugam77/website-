"""Shared FastAPI dependencies: database, settings, client info, authentication, RBAC, CSRF."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated
from urllib.parse import urlsplit

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.api.cookies import ACCESS_COOKIE
from app.core.config import Settings, get_settings
from app.core.database import get_db
from app.core.request_info import resolve_client_ip
from app.core.tokens import InvalidTokenError, decode_access_token
from app.models import User
from app.models.enums import UserRole
from app.services.sessions import is_session_active
from app.services.storage import Storage, get_storage

DbSession = Annotated[Session, Depends(get_db)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
StorageDep = Annotated[Storage, Depends(get_storage)]

UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
MAX_USER_AGENT_LENGTH = 255


def require_trusted_origin(request: Request, settings: SettingsDep) -> None:
    """CSRF defence for cookie authentication.

    Browsers attach the Origin header to every cross-origin and same-origin state-changing
    request and scripts cannot forge it. Requiring it to match our own origins stops a
    malicious site from making a logged-in user's browser change data. Combined with
    SameSite cookies this is robust without per-request CSRF tokens. Non-browser clients
    must send an ``Origin`` header that matches one of the trusted origins.
    """
    if request.method not in UNSAFE_METHODS:
        return
    origin = request.headers.get("origin")
    if not origin:
        referer = request.headers.get("referer")
        if referer:
            parts = urlsplit(referer)
            origin = f"{parts.scheme}://{parts.netloc}" if parts.scheme and parts.netloc else None
    if not origin or origin not in settings.trusted_origin_list:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Untrusted origin")


@dataclass(frozen=True)
class ClientInfo:
    ip: str | None
    user_agent: str | None


def get_client_info(request: Request, settings: SettingsDep) -> ClientInfo:
    user_agent = request.headers.get("user-agent")
    return ClientInfo(
        ip=resolve_client_ip(
            request.client.host if request.client else None,
            request.headers.get("x-forwarded-for"),
            settings.trusted_proxy_count,
        ),
        user_agent=user_agent[:MAX_USER_AGENT_LENGTH] if user_agent else None,
    )


ClientInfoDep = Annotated[ClientInfo, Depends(get_client_info)]


def _unauthenticated() -> HTTPException:
    return HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")


def get_current_user(request: Request, db: DbSession, settings: SettingsDep) -> User:
    """Resolve the logged-in user from the access cookie.

    The user's role and active flag are read from the database on every request (not from
    the token), and the session must still be live, so demotion, deactivation and logout
    apply immediately.
    """
    token = request.cookies.get(ACCESS_COOKIE)
    if not token:
        raise _unauthenticated()
    try:
        claims = decode_access_token(token, secret=settings.secret_key.get_secret_value())
    except InvalidTokenError:
        raise _unauthenticated() from None
    user = db.get(User, claims.user_id)
    if user is None or not user.is_active or not is_session_active(db, claims.session_id):
        raise _unauthenticated()
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


def require_admin(user: CurrentUser) -> User:
    if user.role is not UserRole.ADMIN:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Forbidden")
    return user


AdminUser = Annotated[User, Depends(require_admin)]

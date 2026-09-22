"""Pure-ASGI middleware: request-id propagation and one structured access-log line."""

from __future__ import annotations

import logging
import re
import time
import uuid

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.constants import REQUEST_ID_HEADER
from app.core.logging_config import request_id_ctx

logger = logging.getLogger("app.request")

# Accept a caller-supplied id only if it is boring; otherwise generate one. This
# prevents log injection through a crafted header.
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{8,64}$")
_HEADER_NAME = REQUEST_ID_HEADER.lower().encode("latin-1")


def _resolve_request_id(scope: Scope) -> str:
    for name, value in scope.get("headers", []):
        if name == _HEADER_NAME:
            candidate = value.decode("latin-1")
            if _VALID_REQUEST_ID.match(candidate):
                return candidate
            break
    return uuid.uuid4().hex


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _resolve_request_id(scope)
        scope.setdefault("state", {})["request_id"] = request_id
        token = request_id_ctx.set(request_id)
        started = time.perf_counter()
        status_code = 500

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                message.setdefault("headers", [])
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            # Path only: query strings can carry tokens and are never logged.
            logger.info(
                "request",
                extra={
                    "method": scope["method"],
                    "path": scope["path"],
                    "status_code": status_code,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                },
            )
            request_id_ctx.reset(token)

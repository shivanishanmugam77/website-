"""Structured (JSON) logging with secret redaction and request-id correlation.

This module deliberately depends only on the standard library so it can be used
by the API, Celery workers, Alembic and CLI scripts alike.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

request_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)

REDACTED = "[REDACTED]"
# Any key containing one of these fragments (case-insensitive) has its value redacted.
SENSITIVE_KEY_FRAGMENTS = (
    "password",
    "passwd",
    "secret",
    "token",
    "authorization",
    "api_key",
    "apikey",
    "cookie",
    "credential",
)

# Attributes every LogRecord has; everything else was passed through ``extra=``.
_STANDARD_RECORD_ATTRS = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {"message", "asctime", "taskName"}


def _is_sensitive(key: object) -> bool:
    lowered = str(key).lower()
    return any(fragment in lowered for fragment in SENSITIVE_KEY_FRAGMENTS)


def redact(value: Any) -> Any:
    """Return a copy of ``value`` with values of sensitive-looking keys masked."""
    if isinstance(value, dict):
        return {k: REDACTED if _is_sensitive(k) else redact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    return value


class JsonFormatter(logging.Formatter):
    """One JSON object per line; extra fields are merged in and redacted."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = request_id_ctx.get()
        if request_id:
            payload["request_id"] = request_id

        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _STANDARD_RECORD_ATTRS and not key.startswith("_")
        }
        for key, value in redact(extras).items():
            payload.setdefault(key, value)  # core keys always win

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Human-friendly format for local development (LOG_JSON=false)."""

    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)-8s %(name)s: %(message)s")


def configure_logging(level: str = "INFO", json_output: bool = True) -> None:
    """(Re)configure the root logger. Safe to call more than once."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if json_output else TextFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Route uvicorn's own loggers through the root handler for a uniform format.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True

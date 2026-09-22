"""Standard-library-only tests (no database needed)."""

from __future__ import annotations

import io
import json
import logging

from app.core.logging_config import (
    REDACTED,
    JsonFormatter,
    redact,
    request_id_ctx,
)


def _capture(message_fn) -> dict:  # noqa: ANN001
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger(f"test.{id(stream)}")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    message_fn(logger)
    return json.loads(stream.getvalue().strip().splitlines()[-1])


def test_basic_fields_are_present() -> None:
    record = _capture(lambda log: log.info("hello %s", "world"))
    assert record["message"] == "hello world"
    assert record["level"] == "INFO"
    assert record["timestamp"].endswith("+00:00")


def test_extra_fields_are_included() -> None:
    record = _capture(lambda log: log.info("evt", extra={"page": 7, "stage": "OCR"}))
    assert record["page"] == 7
    assert record["stage"] == "OCR"


def test_sensitive_extras_are_redacted() -> None:
    record = _capture(
        lambda log: log.info(
            "login",
            extra={"password": "hunter2", "jwt_token": "abc", "Authorization": "Bearer x"},
        )
    )
    assert record["password"] == REDACTED
    assert record["jwt_token"] == REDACTED
    assert record["Authorization"] == REDACTED
    assert "hunter2" not in json.dumps(record)


def test_redact_is_recursive_and_non_destructive() -> None:
    original = {"user": {"name": "a", "secret_key": "s"}, "items": [{"api_key": "k"}]}
    cleaned = redact(original)
    assert cleaned == {
        "user": {"name": "a", "secret_key": REDACTED},
        "items": [{"api_key": REDACTED}],
    }
    assert original["user"]["secret_key"] == "s"


def test_request_id_is_attached_from_context() -> None:
    token = request_id_ctx.set("req-1234567890")
    try:
        record = _capture(lambda log: log.info("x"))
    finally:
        request_id_ctx.reset(token)
    assert record["request_id"] == "req-1234567890"


def test_exception_is_serialised() -> None:
    def emit(log: logging.Logger) -> None:
        try:
            raise ValueError("boom")
        except ValueError:
            log.exception("failed")

    record = _capture(emit)
    assert "ValueError: boom" in record["exception"]


def test_extras_cannot_overwrite_core_fields() -> None:
    record = _capture(lambda log: log.info("real", extra={"level": "FAKE", "logger": "x"}))
    assert record["level"] == "INFO"
    assert record["logger"] != "x"

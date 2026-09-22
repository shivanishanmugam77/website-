"""Readiness checks. Details are logged server-side, never returned to callers."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import redis
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import get_settings
from app.core.database import get_engine
from app.core.migrations import head_revision

logger = logging.getLogger(__name__)

REDIS_TIMEOUT_SECONDS = 2


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str | None = None


def check_database() -> CheckResult:
    try:
        with get_engine().connect() as connection:
            connection.execute(text("SELECT 1"))
            current = connection.execute(
                text("SELECT version_num FROM alembic_version")
            ).scalar_one_or_none()
    except SQLAlchemyError:
        logger.exception("readiness_database_check_failed")
        return CheckResult("database", False, "unreachable or not migrated")
    if current != head_revision():
        logger.error(
            "readiness_migrations_pending",
            extra={"current_revision": current, "head_revision": head_revision()},
        )
        return CheckResult("database", False, "migrations pending")
    return CheckResult("database", True)


def check_redis() -> CheckResult:
    client = redis.Redis.from_url(
        get_settings().redis_url,
        socket_connect_timeout=REDIS_TIMEOUT_SECONDS,
        socket_timeout=REDIS_TIMEOUT_SECONDS,
    )
    try:
        client.ping()
    except redis.RedisError:
        logger.exception("readiness_redis_check_failed")
        return CheckResult("redis", False, "unreachable")
    finally:
        client.close()
    return CheckResult("redis", True)


def run_readiness_checks() -> list[CheckResult]:
    return [check_database(), check_redis()]

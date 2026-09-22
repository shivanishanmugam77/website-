"""Helpers for locating Alembic's configuration and the current migration head."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory

BACKEND_ROOT = Path(__file__).resolve().parents[2]


def alembic_config() -> Config:
    config = Config(str(BACKEND_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_ROOT / "alembic"))
    return config


@lru_cache
def head_revision() -> str | None:
    return ScriptDirectory.from_config(alembic_config()).get_current_head()

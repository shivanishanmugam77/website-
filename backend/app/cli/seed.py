"""Seed reference data:  python -m app.cli.seed"""

from __future__ import annotations

import logging

from app.core.config import get_settings
from app.core.database import session_scope
from app.core.logging_config import configure_logging
from app.services.seed import seed_categories, seed_settings

logger = logging.getLogger(__name__)


def main() -> int:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    with session_scope() as session:
        settings_created = seed_settings(session, settings)
        categories_created = seed_categories(session)
    logger.info(
        "seed_complete",
        extra={"settings_created": settings_created, "categories_created": categories_created},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

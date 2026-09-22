"""Idempotent seeding of reference data. Never overwrites values an admin has changed."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.constants import (
    SETTING_AI_AUTO_REVIEW_THRESHOLD,
    SETTING_AI_LOW_CONFIDENCE_THRESHOLD,
)
from app.models import AppSetting, Category
from app.utils.slugs import slugify

logger = logging.getLogger(__name__)

# Nested mapping: name -> children. Purely initial data; admins edit it in the database.
DEFAULT_CATEGORY_TREE: Mapping[str, Mapping[str, Any]] = {
    "Furniture": {
        "Chairs": {"Office Chairs": {}, "Executive Chairs": {}, "Visitor Chairs": {}},
        "Tables": {"Office Tables": {}, "Conference Tables": {}, "Workstations": {}},
        "Storage": {"Cabinets": {}, "Shelves": {}, "Drawers": {}},
        "Sofas": {},
    },
    "Lighting": {
        "Chandeliers": {},
        "Pendant Lights": {},
        "Ceiling Lights": {},
        "Decorative Lighting": {},
    },
}


def seed_settings(session: Session, settings: Settings) -> int:
    """Insert missing AI-threshold rows from the environment. Returns rows created."""
    defaults = {
        SETTING_AI_AUTO_REVIEW_THRESHOLD: (
            settings.ai_auto_review_threshold,
            "At or above this overall confidence a candidate is prepared for approval.",
        ),
        SETTING_AI_LOW_CONFIDENCE_THRESHOLD: (
            settings.ai_low_confidence_threshold,
            "Below this overall confidence a candidate is flagged for manual intervention.",
        ),
    }
    created = 0
    for key, (value, description) in defaults.items():
        if session.get(AppSetting, key) is None:
            session.add(AppSetting(key=key, value=value, description=description))
            created += 1
    session.flush()
    return created


def seed_categories(
    session: Session,
    tree: Mapping[str, Mapping[str, Any]] = DEFAULT_CATEGORY_TREE,
) -> int:
    """Create any missing categories (matched by slug). Returns rows created."""
    created = 0

    def walk(nodes: Mapping[str, Mapping[str, Any]], parent: Category | None) -> None:
        nonlocal created
        for order, (name, children) in enumerate(nodes.items()):
            slug = slugify(name)
            category = session.scalar(select(Category).where(Category.slug == slug))
            if category is None:
                category = Category(name=name, slug=slug, parent=parent, sort_order=order)
                session.add(category)
                session.flush()
                created += 1
            walk(children, category)

    walk(tree, None)
    return created

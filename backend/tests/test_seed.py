from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.constants import (
    SETTING_AI_AUTO_REVIEW_THRESHOLD,
    SETTING_AI_LOW_CONFIDENCE_THRESHOLD,
)
from app.models import AppSetting, Category
from app.services.seed import DEFAULT_CATEGORY_TREE, seed_categories, seed_settings


def _count_tree(tree) -> int:  # noqa: ANN001
    return sum(1 + _count_tree(children) for children in tree.values())


def _category_path(category: Category) -> str:
    names = []
    node: Category | None = category
    while node is not None:
        names.append(node.name)
        node = node.parent
    return " > ".join(reversed(names))


def test_seed_categories_builds_the_default_tree(db_session: Session) -> None:
    created = seed_categories(db_session)
    assert created == _count_tree(DEFAULT_CATEGORY_TREE)

    paths = {_category_path(c) for c in db_session.scalars(select(Category))}
    assert "Furniture > Chairs > Executive Chairs" in paths
    assert "Furniture > Tables > Workstations" in paths
    assert "Furniture > Sofas" in paths
    assert "Lighting > Chandeliers" in paths


def test_seed_categories_is_idempotent(db_session: Session) -> None:
    seed_categories(db_session)
    total = db_session.scalar(select(func.count()).select_from(Category))
    assert seed_categories(db_session) == 0
    assert db_session.scalar(select(func.count()).select_from(Category)) == total


def test_seed_never_overwrites_admin_edits(db_session: Session) -> None:
    seed_categories(db_session)
    sofas = db_session.scalar(select(Category).where(Category.slug == "sofas"))
    sofas.name = "Sofas & Couches"
    db_session.flush()
    seed_categories(db_session)
    db_session.expire_all()
    assert db_session.scalar(select(Category).where(Category.slug == "sofas")).name == (
        "Sofas & Couches"
    )


def test_seed_settings_uses_configured_thresholds_and_is_idempotent(db_session: Session) -> None:
    settings = get_settings()
    assert seed_settings(db_session, settings) == 2
    assert db_session.get(AppSetting, SETTING_AI_AUTO_REVIEW_THRESHOLD).value == (
        settings.ai_auto_review_threshold
    )
    assert db_session.get(AppSetting, SETTING_AI_LOW_CONFIDENCE_THRESHOLD).value == (
        settings.ai_low_confidence_threshold
    )
    assert seed_settings(db_session, settings) == 0


def test_seed_settings_does_not_overwrite_tuned_values(db_session: Session) -> None:
    settings = get_settings()
    seed_settings(db_session, settings)
    row = db_session.get(AppSetting, SETTING_AI_AUTO_REVIEW_THRESHOLD)
    row.value = 0.92
    db_session.flush()
    seed_settings(db_session, settings)
    db_session.expire_all()
    assert db_session.get(AppSetting, SETTING_AI_AUTO_REVIEW_THRESHOLD).value == 0.92

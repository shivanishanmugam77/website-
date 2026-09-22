"""Runtime-tunable settings (``app_settings``): an admin's saved value wins over the
environment default, so a threshold can be retuned without redeploying."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.constants import SETTING_AI_AUTO_REVIEW_THRESHOLD, SETTING_AI_LOW_CONFIDENCE_THRESHOLD
from app.models import AppSetting


def get_float_setting(session: Session, key: str, default: float) -> float:
    value = session.scalar(select(AppSetting.value).where(AppSetting.key == key))
    return default if value is None else float(value)


def get_review_thresholds(session: Session, settings: Settings) -> tuple[float, float]:
    """(auto_review_threshold, low_confidence_threshold), each from the database if an admin
    has set it there, else the environment default."""
    auto_review = get_float_setting(
        session,
        SETTING_AI_AUTO_REVIEW_THRESHOLD,
        settings.ai_auto_review_threshold,
    )
    low_confidence = get_float_setting(
        session,
        SETTING_AI_LOW_CONFIDENCE_THRESHOLD,
        settings.ai_low_confidence_threshold,
    )
    return auto_review, low_confidence

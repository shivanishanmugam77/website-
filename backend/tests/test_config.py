from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.config import Settings

@pytest.fixture(autouse=True)
def _isolate_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make these tests independent of whatever the developer put in .env / compose."""
    for field in Settings.model_fields:
        monkeypatch.delenv(field.upper(), raising=False)


BASE = {
    "secret_key": "x" * 40,
    "database_url": "postgresql+psycopg://u:p@localhost/db",
}


def make(**overrides) -> Settings:  # noqa: ANN003
    # _env_file=None: ignore any real .env; init kwargs beat environment variables.
    return Settings(_env_file=None, **{**BASE, **overrides})


def test_defaults_are_sane() -> None:
    settings = make()
    assert settings.storage_type == "local"
    assert settings.ml_device == "auto"
    assert settings.ai_auto_review_threshold == 0.85
    assert settings.ai_low_confidence_threshold == 0.60
    assert settings.cors_origin_list == ["http://localhost:3000"]


def test_cors_origins_are_split_and_trimmed() -> None:
    settings = make(cors_origins=" https://a.example , https://b.example ,,")
    assert settings.cors_origin_list == ["https://a.example", "https://b.example"]


def test_thresholds_must_be_ordered() -> None:
    with pytest.raises(ValidationError, match="LOW_CONFIDENCE"):
        make(ai_low_confidence_threshold=0.9, ai_auto_review_threshold=0.8)


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_thresholds_must_be_probabilities(bad: float) -> None:
    with pytest.raises(ValidationError):
        make(ai_auto_review_threshold=bad)


def test_production_rejects_short_secret() -> None:
    with pytest.raises(ValidationError, match="SECRET_KEY"):
        make(app_env="production", secret_key="short")


def test_production_accepts_long_secret() -> None:
    assert make(app_env="production").is_production is True


def test_s3_requires_credentials_and_bucket() -> None:
    with pytest.raises(ValidationError, match="S3_BUCKET"):
        make(storage_type="s3")
    settings = make(
        storage_type="s3", s3_bucket="b", s3_access_key="a", s3_secret_key="s"
    )
    assert settings.s3_bucket == "b"


def test_invalid_ml_device_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make(ml_device="tpu")


def test_secrets_are_not_leaked_by_repr() -> None:
    settings = make(secret_key="super-secret-value-" + "x" * 30)
    assert "super-secret-value" not in repr(settings)
    assert "super-secret-value" not in str(settings)


def test_connection_urls_are_not_leaked_by_repr() -> None:
    settings = make(database_url="postgresql+psycopg://user:dbpassword123@db/app")
    assert "dbpassword123" not in repr(settings)
    assert "dbpassword123" not in str(settings)


def test_required_fields_are_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SECRET_KEY", raising=False)
    with pytest.raises(ValidationError, match="secret_key"):
        Settings(_env_file=None, database_url="postgresql+psycopg://u:p@h/d")


def test_production_requires_secure_cookies() -> None:
    with pytest.raises(ValidationError, match="COOKIE_SECURE"):
        make(app_env="production", cookie_secure=False)


def test_cookie_security_defaults_follow_the_environment() -> None:
    assert make(app_env="production").use_secure_cookies is True
    assert make(app_env="development").use_secure_cookies is False
    assert make(app_env="development", cookie_secure=True).use_secure_cookies is True


def test_production_enforces_argon2_minimums() -> None:
    with pytest.raises(ValidationError, match="ARGON2"):
        make(app_env="production", argon2_memory_cost_kib=1024)
    with pytest.raises(ValidationError, match="ARGON2"):
        make(app_env="production", argon2_time_cost=1)


def test_argon2_memory_must_scale_with_parallelism() -> None:
    with pytest.raises(ValidationError, match="ARGON2_MEMORY_COST_KIB"):
        make(argon2_memory_cost_kib=16, argon2_parallelism=4)


def test_trusted_origins_include_the_dev_api_origin_only_outside_production() -> None:
    development = make(app_env="development").trusted_origin_list
    assert "http://localhost:3000" in development
    assert "http://localhost:8000" in development
    production = make(app_env="production", cors_origins="https://shop.example").trusted_origin_list
    assert production == ["https://shop.example"]


@pytest.mark.parametrize("field", ["login_max_failed_attempts", "trusted_proxy_count"])
def test_auth_limits_are_bounded(field: str) -> None:
    with pytest.raises(ValidationError):
        make(**{field: 10_000})


def test_upload_limit_is_exposed_in_bytes() -> None:
    assert make(max_upload_mb=5).max_upload_bytes == 5 * 1024 * 1024


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_upload_mb", 0),
        ("max_pdf_pages", 0),
        ("render_dpi", 10),
        ("max_page_pixels", 10),
        ("jpeg_quality", 10),
        ("min_embedded_image_px", 1),
        ("pipeline_time_limit_minutes", 0),
    ],
)
def test_ingestion_limits_are_bounded(field: str, value: int) -> None:
    with pytest.raises(ValidationError):
        make(**{field: value})


def test_text_recognition_defaults() -> None:
    settings = make()
    assert settings.ocr_enabled is True and settings.ocr_provider == "rapidocr"
    assert settings.ocr_min_confidence == 0.5
    assert (settings.ocr_tile_px, settings.ocr_tile_overlap_px) == (1600, 320)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ocr_min_confidence", -0.1),
        ("ocr_min_confidence", 1.1),
        ("ocr_tile_px", 100),
        ("ocr_tile_px", 5000),
        ("ocr_tile_overlap_px", -1),
        ("ocr_max_lines_per_page", 1),
        ("ocr_provider", "tesseract"),
    ],
)
def test_text_recognition_settings_are_bounded(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        make(**{field: value})


def test_the_tile_overlap_must_leave_room_for_new_pixels() -> None:
    with pytest.raises(ValidationError, match="OCR_TILE_OVERLAP_PX"):
        make(ocr_tile_px=800, ocr_tile_overlap_px=500)
    make(ocr_tile_px=800, ocr_tile_overlap_px=400)  # exactly half is still allowed


def test_panel_detection_defaults_and_bounds() -> None:
    settings = make()
    assert settings.panel_detection_enabled is True and settings.panel_min_area_share == 0.004
    for bad in (0.0, 0.0001, 0.9):
        with pytest.raises(ValidationError):
            make(panel_min_area_share=bad)


def test_product_finding_defaults() -> None:
    settings = make()
    assert settings.detection_model == "IDEA-Research/grounding-dino-tiny"
    assert settings.segmentation_model == "facebook/sam-vit-base"
    assert "desk" in settings.detection_prompts and settings.ml_num_threads == 4
    assert (settings.detection_box_threshold, settings.detection_text_threshold) == (0.35, 0.25)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("detection_model", "no-slash"),
        ("detection_model", "../../etc/passwd"),
        ("segmentation_model", "org/name with spaces"),
        ("detection_box_threshold", 0.0),
        ("detection_text_threshold", 1.0),
        ("detection_max_px", 100),
        ("ml_num_threads", 0),
        ("detection_provider", "yolo"),
    ],
)
def test_product_finding_settings_are_bounded(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        make(**{field: value})


def test_product_detection_is_off_in_the_code_and_needs_panels() -> None:
    assert Settings.model_fields["product_detection_enabled"].default is False  # compose turns it on
    settings = make(panel_detection_enabled=True, product_detection_enabled=True)
    assert settings.product_detection_enabled is True
    assert (settings.product_min_outline_score, settings.product_min_outline_coverage) == (0.70, 0.30)
    with pytest.raises(ValidationError, match="PRODUCT_DETECTION_ENABLED needs PANEL_DETECTION_ENABLED"):
        make(panel_detection_enabled=False, product_detection_enabled=True)
    make(panel_detection_enabled=False, product_detection_enabled=False)  # both off is fine


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("product_min_outline_score", -0.1),
        ("product_min_outline_score", 1.1),
        ("product_min_outline_coverage", -0.1),
        ("product_min_outline_coverage", 1.5),
    ],
)
def test_product_outline_limits_are_bounded(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        make(**{field: value})
    make(**{field: 0.0})
    make(**{field: 1.0})

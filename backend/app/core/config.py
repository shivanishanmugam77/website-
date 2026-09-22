"""Application configuration, loaded from environment variables (and an optional .env)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# repo-root/.env when running from a checkout; harmlessly absent inside containers.
_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"

MIN_PRODUCTION_SECRET_LENGTH = 32
# OWASP minimums for argon2id (19 MiB, 2 iterations), enforced in production only.
MIN_PRODUCTION_ARGON2_MEMORY_KIB = 19456
MIN_PRODUCTION_ARGON2_TIME_COST = 2
# Swagger UI (development only) calls the API from its own origin.
DEV_API_ORIGINS = ("http://localhost:8000", "http://127.0.0.1:8000")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",  # the shared .env also holds POSTGRES_*, frontend vars, ...
        env_ignore_empty=True,  # `FOO=` in .env means "unset", not empty string
    )

    # --- general ---
    app_name: str = "Catalogue Platform"
    app_env: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_json: bool = True

    # --- secrets / infrastructure ---
    secret_key: SecretStr
    # repr=False: these URLs may embed credentials and must never appear in logs.
    database_url: str = Field(repr=False)
    redis_url: str = Field(default="redis://localhost:6379/0", repr=False)
    cors_origins: str = "http://localhost:3000"

    # --- auth ---
    access_token_expire_minutes: int = Field(default=15, ge=1, le=1440)
    refresh_token_expire_days: int = Field(default=14, ge=1, le=90)
    # None = decide by environment (secure cookies everywhere except local development).
    cookie_secure: bool | None = None
    cookie_samesite: Literal["lax", "strict"] = "lax"
    cookie_domain: str | None = None
    login_max_failed_attempts: int = Field(default=5, ge=3, le=20)
    login_lockout_minutes: int = Field(default=15, ge=1, le=1440)
    # Number of reverse proxies whose X-Forwarded-For entry may be trusted (0 = ignore header).
    trusted_proxy_count: int = Field(default=0, ge=0, le=5)
    argon2_time_cost: int = Field(default=3, ge=1, le=10)
    argon2_memory_cost_kib: int = Field(default=65536, ge=8, le=1048576)
    argon2_parallelism: int = Field(default=4, ge=1, le=16)

    # --- storage ---
    storage_type: Literal["local", "s3"] = "local"
    storage_local_root: Path = Path("./storage")
    s3_endpoint: str | None = None
    s3_region: str | None = None
    s3_access_key: SecretStr | None = None
    s3_secret_key: SecretStr | None = None
    s3_bucket: str | None = None

    # --- catalogue ingestion ---
    max_upload_mb: int = Field(default=200, ge=1, le=2048)
    max_pdf_pages: int = Field(default=500, ge=1, le=5000)
    render_dpi: int = Field(default=150, ge=72, le=400)
    # Hard cap on a rendered page's pixel count (protects worker memory from huge pages).
    max_page_pixels: int = Field(default=40_000_000, ge=1_000_000)
    thumbnail_max_px: int = Field(default=400, ge=64, le=1024)
    # Embedded images smaller than this (either side) are ignored: icons, bullets, rules.
    min_embedded_image_px: int = Field(default=120, ge=16)
    jpeg_quality: int = Field(default=90, ge=50, le=100)
    pipeline_time_limit_minutes: int = Field(default=120, ge=5, le=1440)

    # --- text recognition (OCR) ---
    ocr_enabled: bool = True
    # The engine behind the OcrProvider interface. Only "rapidocr" exists today (PaddleOCR
    # models on ONNX Runtime: Chinese + English, models ship inside the pip package).
    ocr_provider: Literal["rapidocr"] = "rapidocr"
    # Lines the engine is less sure of than this are dropped.
    ocr_min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    # Pages are read in overlapping square tiles so small print is not lost when a large
    # page is shrunk to fit the engine's input size.
    ocr_tile_px: int = Field(default=1600, ge=640, le=2000)
    ocr_tile_overlap_px: int = Field(default=320, ge=0, le=600)
    # Safety valve: a page of pure texture can "hallucinate" thousands of lines.
    ocr_max_lines_per_page: int = Field(default=3000, ge=10, le=20000)

    # --- photo panels (Phase 5a) ---
    panel_detection_enabled: bool = True
    # A picture smaller than this share of the page area is ignored (icons, logos, bullets).
    panel_min_area_share: float = Field(default=0.004, ge=0.0005, le=0.5)

    # --- ML ---
    ml_device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    ai_auto_review_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    ai_low_confidence_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    ml_num_threads: int = Field(default=4, ge=1, le=64)  # CPU threads one model process may use

    # --- product finding (Phase 5b) ---
    detection_provider: Literal["grounding_dino"] = "grounding_dino"
    segmentation_provider: Literal["sam"] = "sam"
    # Hugging Face model ids. Both are Apache-2.0 licensed (commercial use allowed).
    detection_model: str = Field(
        default="IDEA-Research/grounding-dino-tiny", pattern=r"^[\w.\-]+/[\w.\-]+$"
    )
    segmentation_model: str = Field(
        default="facebook/sam-vit-base", pattern=r"^[\w.\-]+/[\w.\-]+$"
    )
    # What to look for, in words: comma separated, lower case is fine.
    detection_prompts: str = (
        "desk, office chair, cabinet, sofa, meeting table, bookshelf, filing cabinet, "
        "locker, reception desk, workstation, table, chair"
    )
    detection_box_threshold: float = Field(default=0.35, ge=0.05, le=0.95)
    detection_text_threshold: float = Field(default=0.25, ge=0.05, le=0.95)
    # A panel larger than this (longest side, pixels) is shrunk before the finder looks at it.
    detection_max_px: int = Field(default=1024, ge=320, le=2048)

    # --- products inside the panels: a pipeline stage after the panels (Phase 5b step 3) ---
    # Off in the code so tests and small set-ups never load the models; docker-compose turns
    # it on. It needs the models to be downloaded (see docs/product-detection.md).
    product_detection_enabled: bool = False
    # An outline is marked LOW_CONFIDENCE (kept, but to be checked) when the outlining model
    # rates it below this, or when it fills less than this share of the find's box.
    product_min_outline_score: float = Field(default=0.70, ge=0.0, le=1.0)
    product_min_outline_coverage: float = Field(default=0.30, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_consistency(self) -> Settings:
        if self.ai_low_confidence_threshold >= self.ai_auto_review_threshold:
            raise ValueError(
                "AI_LOW_CONFIDENCE_THRESHOLD must be lower than AI_AUTO_REVIEW_THRESHOLD"
            )
        if self.app_env == "production":
            secret = self.secret_key.get_secret_value()
            if len(secret) < MIN_PRODUCTION_SECRET_LENGTH:
                raise ValueError(
                    f"SECRET_KEY must be at least {MIN_PRODUCTION_SECRET_LENGTH} "
                    "characters in production"
                )
        if self.product_detection_enabled and not self.panel_detection_enabled:
            raise ValueError(
                "PRODUCT_DETECTION_ENABLED needs PANEL_DETECTION_ENABLED "
                "(products are found inside the photo panels)"
            )
        if self.ocr_tile_overlap_px * 2 > self.ocr_tile_px:
            raise ValueError("OCR_TILE_OVERLAP_PX must be at most half of OCR_TILE_PX")
        if self.argon2_memory_cost_kib < 8 * self.argon2_parallelism:
            raise ValueError("ARGON2_MEMORY_COST_KIB must be at least 8 x ARGON2_PARALLELISM")
        if self.app_env == "production":
            if self.cookie_secure is False:
                raise ValueError("COOKIE_SECURE must not be false in production")
            if (
                self.argon2_memory_cost_kib < MIN_PRODUCTION_ARGON2_MEMORY_KIB
                or self.argon2_time_cost < MIN_PRODUCTION_ARGON2_TIME_COST
            ):
                raise ValueError("ARGON2_* parameters are below the production minimum")
        if self.storage_type == "s3":
            missing = [
                name
                for name, value in (
                    ("S3_BUCKET", self.s3_bucket),
                    ("S3_ACCESS_KEY", self.s3_access_key),
                    ("S3_SECRET_KEY", self.s3_secret_key),
                )
                if value is None
            ]
            if missing:
                raise ValueError(f"STORAGE_TYPE=s3 requires: {', '.join(missing)}")
        return self

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def use_secure_cookies(self) -> bool:
        return self.is_production if self.cookie_secure is None else self.cookie_secure

    @property
    def trusted_origin_list(self) -> list[str]:
        """Origins allowed to make state-changing (cookie-authenticated) requests."""
        origins = list(self.cors_origin_list)
        if not self.is_production:
            origins.extend(o for o in DEV_API_ORIGINS if o not in origins)
        return origins

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]  # required fields come from the environment

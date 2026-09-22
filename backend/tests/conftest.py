"""Shared fixtures.

The suite runs against a REAL PostgreSQL server (with pgvector) because the schema uses
JSONB, vector columns, partial features and check constraints that SQLite cannot emulate.
A dedicated ``*_test`` database is dropped and recreated at the start of every run and is
never the development database.
"""

from __future__ import annotations

import os
import re
import tempfile
import uuid
from collections.abc import Iterator

import pytest
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.orm import Session

# --- environment must be settled BEFORE any application module reads settings ----------
os.environ["APP_ENV"] = "test"
os.environ["LOG_JSON"] = "false"
os.environ.setdefault("SECRET_KEY", "test-only-secret-key-do-not-use-elsewhere-0123456789")
# Cheap argon2 so hundreds of password hashes in the suite stay fast (never used outside tests).
os.environ["ARGON2_TIME_COST"] = "1"
os.environ["ARGON2_MEMORY_COST_KIB"] = "8"
os.environ["ARGON2_PARALLELISM"] = "1"
# Make the suite independent of whatever the developer has in .env / the shell.
os.environ["CORS_ORIGINS"] = "http://localhost:3000"
for _name in (
    "COOKIE_SECURE",
    "COOKIE_DOMAIN",
    "COOKIE_SAMESITE",
    "TRUSTED_PROXY_COUNT",
    "ACCESS_TOKEN_EXPIRE_MINUTES",
    "REFRESH_TOKEN_EXPIRE_DAYS",
    "LOGIN_MAX_FAILED_ATTEMPTS",
    "LOGIN_LOCKOUT_MINUTES",
    "ADMIN_PASSWORD",
    "MAX_UPLOAD_MB",
    "MAX_PDF_PAGES",
    "RENDER_DPI",
    "MAX_PAGE_PIXELS",
    "THUMBNAIL_MAX_PX",
    "MIN_EMBEDDED_IMAGE_PX",
    "JPEG_QUALITY",
    "PIPELINE_TIME_LIMIT_MINUTES",
    "OCR_MIN_CONFIDENCE",
    "OCR_TILE_PX",
    "OCR_TILE_OVERLAP_PX",
    "OCR_MAX_LINES_PER_PAGE",
    "PANEL_MIN_AREA_SHARE",
    "PRODUCT_MIN_OUTLINE_SCORE",
    "PRODUCT_MIN_OUTLINE_COVERAGE",
):
    os.environ.pop(_name, None)
# Text recognition is off by default in tests: they must never load a real engine. The tests
# that cover it switch it on with settings_with(ocr_enabled=True) and pass a stand-in engine.
os.environ["OCR_ENABLED"] = "false"
# Likewise panel detection: the tests about it switch it on with settings_with(...).
os.environ["PANEL_DETECTION_ENABLED"] = "false"
# And product finding: it loads two large models, so only the tests about it switch it on
# (with stand-in models).
os.environ["PRODUCT_DETECTION_ENABLED"] = "false"
# Never let a test touch the real storage volume.
os.environ["STORAGE_LOCAL_ROOT"] = tempfile.mkdtemp(prefix="catalogue-test-storage-")

_SAFE_DB_NAME = re.compile(r"^[a-z0-9_]+$")


def _test_database_url() -> URL:
    explicit = os.environ.get("TEST_DATABASE_URL")
    base = explicit or os.environ.get("DATABASE_URL")
    if not base:
        raise RuntimeError("Set DATABASE_URL (or TEST_DATABASE_URL) to run the test-suite.")
    url = make_url(base)
    if not explicit and not (url.database or "").endswith("_test"):
        url = url.set(database=f"{url.database}_test")
    name = url.database or ""
    if not name.endswith("_test") or not _SAFE_DB_NAME.match(name):
        raise RuntimeError(
            f"Refusing to run tests against database {name!r}: the name must match "
            "[a-z0-9_]+ and end with '_test' (the database is dropped on every run)."
        )
    return url


TEST_DB_URL = _test_database_url()
os.environ["DATABASE_URL"] = TEST_DB_URL.render_as_string(hide_password=False)


def recreate_database(url: URL) -> None:
    """Drop and recreate ``url.database`` using the server's maintenance database."""
    assert url.database and _SAFE_DB_NAME.match(url.database)
    admin = create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as connection:
            connection.execute(text(f'DROP DATABASE IF EXISTS "{url.database}" WITH (FORCE)'))
            connection.execute(text(f'CREATE DATABASE "{url.database}"'))
    finally:
        admin.dispose()


@pytest.fixture(scope="session", autouse=True)
def _migrated_database() -> Iterator[None]:
    from alembic import command

    from app.core.database import get_engine
    from app.core.migrations import alembic_config

    recreate_database(TEST_DB_URL)
    command.upgrade(alembic_config(), "head")
    yield
    get_engine().dispose()


@pytest.fixture(scope="session")
def engine(_migrated_database: None) -> Engine:
    from app.core.database import get_engine

    return get_engine()


@pytest.fixture
def db_session(engine: Engine) -> Iterator[Session]:
    """A session wrapped in an outer transaction that is rolled back after each test.

    ``create_savepoint`` lets tests trigger and recover from IntegrityErrors.
    """
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture
def storage(tmp_path):  # noqa: ANN001, ANN201
    """An isolated, throw-away file store."""
    from app.services.storage import LocalStorage

    return LocalStorage(tmp_path / "storage")


@pytest.fixture(autouse=True)
def enqueued(monkeypatch) -> list[tuple[uuid.UUID, uuid.UUID]]:  # noqa: ANN001
    """Record processing jobs instead of sending them to the real Celery broker.

    Autouse on purpose: a test must never queue work that a live worker (pointed at the
    development database) could pick up.
    """
    calls: list[tuple[uuid.UUID, uuid.UUID]] = []

    def fake_enqueue(catalogue_id: uuid.UUID, job_id: uuid.UUID) -> str:
        calls.append((catalogue_id, job_id))
        return f"test-task-{len(calls)}"

    monkeypatch.setattr("app.services.catalogues.enqueue_processing", fake_enqueue)
    return calls


@pytest.fixture
def app(db_session: Session, storage):  # noqa: ANN001, ANN201
    from app.core.database import get_db
    from app.main import create_app
    from app.services.storage import get_storage

    application = create_app()
    application.dependency_overrides[get_storage] = lambda: storage

    def _override_get_db() -> Iterator[Session]:
        yield db_session

    application.dependency_overrides[get_db] = _override_get_db
    return application


@pytest.fixture
def client(app):  # noqa: ANN001, ANN201
    from fastapi.testclient import TestClient

    # raise_server_exceptions=False: assert on the 500 response our handler produces.
    with TestClient(app, raise_server_exceptions=False) as test_client:
        # State-changing requests must come from a trusted origin (CSRF defence); browsers
        # always send this header, so the test client does too.
        test_client.headers["Origin"] = "http://localhost:3000"
        yield test_client

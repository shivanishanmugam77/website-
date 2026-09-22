"""Migration integrity, run against its own throw-away database."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from pgvector.sqlalchemy import Vector
from sqlalchemy import Engine, Float, create_engine, inspect

from app.core.migrations import alembic_config
from app.models import Base
from tests.conftest import TEST_DB_URL, recreate_database

MIGRATION_DB_URL = TEST_DB_URL.set(
    database=f"{TEST_DB_URL.database.removesuffix('_test')}_migrations_test"
)


def _config():  # noqa: ANN202
    config = alembic_config()
    config.attributes["sqlalchemy_url"] = MIGRATION_DB_URL.render_as_string(hide_password=False)
    return config


@pytest.fixture
def migration_engine() -> Iterator[Engine]:
    recreate_database(MIGRATION_DB_URL)
    engine = create_engine(MIGRATION_DB_URL)
    try:
        yield engine
    finally:
        engine.dispose()


def _ignore_vector_type(context, inspected_column, metadata_column, inspected_type, metadata_type):  # noqa: ANN001, ANN202
    # None defers to Alembic's default comparison. Two known reflection quirks are exempted:
    # * pgvector's type does not round-trip through reflection (dimension is covered by
    #   tests/test_models.py);
    # * PostgreSQL reports FLOAT as DOUBLE PRECISION.
    if isinstance(metadata_type, Vector):
        return False
    if isinstance(metadata_type, Float) and isinstance(inspected_type, Float):
        return False
    return None


def test_there_is_exactly_one_migration_head() -> None:
    heads = ScriptDirectory.from_config(alembic_config()).get_heads()
    assert len(heads) == 1


def test_models_and_migrations_are_in_sync(migration_engine: Engine) -> None:
    """Fails if someone changes a model without writing a migration (or vice-versa)."""
    command.upgrade(_config(), "head")
    with migration_engine.connect() as connection:
        context = MigrationContext.configure(
            connection, opts={"compare_type": _ignore_vector_type}
        )
        differences = compare_metadata(context, Base.metadata)
    assert differences == [], f"models and migrations have drifted: {differences}"


def test_downgrade_to_base_removes_every_table_and_upgrade_is_repeatable(
    migration_engine: Engine,
) -> None:
    command.upgrade(_config(), "head")
    command.downgrade(_config(), "base")
    assert set(inspect(migration_engine).get_table_names()) <= {"alembic_version"}
    command.upgrade(_config(), "head")
    assert "products" in inspect(migration_engine).get_table_names()

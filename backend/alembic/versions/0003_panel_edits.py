"""Manual correction of photo panels: who drew each panel, and the history for undo.

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-21
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Existing rows are the system's own output, hence the default.
    op.add_column(
        "source_images",
        sa.Column("origin", sa.String(32), server_default="AI", nullable=False),
    )

    op.create_table(
        "panel_edits",
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("page_id", sa.Uuid(), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("action", sa.String(32), nullable=False),
        sa.Column("panel_id", sa.Uuid(), nullable=False),
        sa.Column("before", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("after", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("actor_id", sa.Uuid(), nullable=True),
        sa.Column("undone_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_panel_edits")),
        sa.ForeignKeyConstraint(
            ["page_id"],
            ["catalogue_pages.id"],
            name=op.f("fk_panel_edits_page_id_catalogue_pages"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["actor_id"],
            ["users.id"],
            name=op.f("fk_panel_edits_actor_id_users"),
            ondelete="SET NULL",
        ),
        sa.UniqueConstraint("page_id", "seq", name=op.f("uq_panel_edits_page_seq")),
    )
    op.create_index(op.f("ix_panel_edits_page_id"), "panel_edits", ["page_id"])


def downgrade() -> None:
    op.drop_table("panel_edits")
    op.drop_column("source_images", "origin")

"""job matrix: ppe_items, jobs, job_requirements

Revision ID: 0001
Revises:
Create Date: 2026-10-15 20:14:42

Generated with `alembic revision --autogenerate` against an empty database,
then tidied. Data comes from `cli.py db seed`, not from this migration.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "jobs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("key", sa.String(length=64), nullable=False),
        sa.Column("display_name", sa.String(length=100), nullable=False),
        sa.Column("position", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("key ~ '^[a-z][a-z0-9_]*$'", name=op.f("ck_jobs_key_format")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_jobs")),
        sa.UniqueConstraint("key", name=op.f("uq_jobs_key")),
    )
    op.create_table(
        "ppe_items",
        sa.Column("key", sa.String(length=64), nullable=False),
        sa.Column("display_name", sa.String(length=100), nullable=False),
        sa.Column("detectable", sa.Boolean(), nullable=False),
        sa.CheckConstraint("key ~ '^[a-z][a-z0-9_]*$'", name=op.f("ck_ppe_items_key_format")),
        sa.PrimaryKeyConstraint("key", name=op.f("pk_ppe_items")),
    )
    op.create_table(
        "job_requirements",
        sa.Column("job_id", sa.Integer(), nullable=False),
        sa.Column("item_key", sa.String(length=64), nullable=False),
        sa.Column("mode", sa.String(length=16), nullable=False),
        sa.Column("position", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.CheckConstraint(
            "mode IN ('required', 'pending')", name=op.f("ck_job_requirements_mode_valid")
        ),
        sa.ForeignKeyConstraint(
            ["item_key"], ["ppe_items.key"],
            name=op.f("fk_job_requirements_item_key_ppe_items"), ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["job_id"], ["jobs.id"],
            name=op.f("fk_job_requirements_job_id_jobs"), ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("job_id", "item_key", name=op.f("pk_job_requirements")),
    )


def downgrade() -> None:
    op.drop_table("job_requirements")
    op.drop_table("ppe_items")
    op.drop_table("jobs")

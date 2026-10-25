"""admin system: sites, job versions, app_settings, audit_log

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-26

Written by hand, because it moves data: it creates a "main" site and puts
every existing worker and supervisor in it before site_id becomes required.
tests/test_db_matrix.py checks it still matches the models. The contract
these tables serve is docs/admin-api.md.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

MAIN_SITE = "(SELECT id FROM sites WHERE key = 'main')"


def upgrade() -> None:
    op.create_table(
        "sites",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("key", sa.String(length=64), nullable=False),
        sa.Column("display_name", sa.String(length=100), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("key ~ '^[a-z][a-z0-9_]*$'", name=op.f("ck_sites_key_format")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_sites")),
        sa.UniqueConstraint("key", name=op.f("uq_sites_key")),
    )
    op.execute("INSERT INTO sites (key, display_name) VALUES ('main', 'Main site')")

    # Every worker belongs to a site: backfill, then make it required.
    op.add_column("workers", sa.Column("site_id", sa.Integer(), nullable=True))
    op.execute(f"UPDATE workers SET site_id = {MAIN_SITE}")
    op.alter_column("workers", "site_id", nullable=False)
    op.create_foreign_key(
        op.f("fk_workers_site_id_sites"), "workers", "sites", ["site_id"], ["id"],
        ondelete="RESTRICT",
    )
    op.create_index(op.f("ix_workers_site_id"), "workers", ["site_id"], unique=False)

    # Supervisors belong to a site; admins have none.
    op.add_column("users", sa.Column("site_id", sa.Integer(), nullable=True))
    op.execute(f"UPDATE users SET site_id = {MAIN_SITE} WHERE role = 'supervisor'")
    op.create_foreign_key(
        op.f("fk_users_site_id_sites"), "users", "sites", ["site_id"], ["id"],
        ondelete="RESTRICT",
    )
    op.create_index(op.f("ix_users_site_id"), "users", ["site_id"], unique=False)
    op.create_check_constraint(
        op.f("ck_users_site_matches_role"), "users",
        "(role = 'supervisor') = (site_id IS NOT NULL)",
    )

    op.add_column(
        "jobs", sa.Column("version", sa.Integer(), server_default=sa.text("1"), nullable=False)
    )

    op.create_table(
        "app_settings",
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("value", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("updated_by", sa.String(length=254), nullable=True),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("name", name=op.f("pk_app_settings")),
    )

    op.create_table(
        "audit_log",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column(
            "at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.Column("actor", sa.String(length=254), nullable=False),
        sa.Column("action", sa.String(length=64), nullable=False),
        sa.Column("target", sa.String(length=160), nullable=False),
        sa.Column("site", sa.String(length=64), nullable=True),
        sa.Column("before", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("after", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("ip", sa.String(length=64), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_audit_log")),
    )
    for column in ("at", "action", "target", "site"):
        op.create_index(op.f(f"ix_audit_log_{column}"), "audit_log", [column], unique=False)


def downgrade() -> None:
    for column in ("site", "target", "action", "at"):
        op.drop_index(op.f(f"ix_audit_log_{column}"), table_name="audit_log")
    op.drop_table("audit_log")
    op.drop_table("app_settings")
    op.drop_column("jobs", "version")
    op.drop_constraint(op.f("ck_users_site_matches_role"), "users", type_="check")
    op.drop_index(op.f("ix_users_site_id"), table_name="users")
    op.drop_constraint(op.f("fk_users_site_id_sites"), "users", type_="foreignkey")
    op.drop_column("users", "site_id")
    op.drop_index(op.f("ix_workers_site_id"), table_name="workers")
    op.drop_constraint(op.f("fk_workers_site_id_sites"), "workers", type_="foreignkey")
    op.drop_column("workers", "site_id")
    op.drop_table("sites")

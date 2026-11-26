"""checks, and the worker's answer to the photo notice

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-25

The whole of Phase 6's schema, in one step: the ``checks`` table, with its
review columns ready for the dashboard endpoints, and the photo notice's
answer on ``workers``. No existing data moves. tests/test_db_matrix.py checks
it still matches the models. The contract these serve is docs/checks-api.md.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _jsonb() -> postgresql.JSONB:
    return postgresql.JSONB(astext_type=sa.Text())


def upgrade() -> None:
    op.create_table(
        "checks",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("at", sa.DateTime(timezone=True), server_default=sa.text("now()"),
                  nullable=False),
        sa.Column("worker_id", sa.Integer(), nullable=True),
        sa.Column("worker_code", sa.String(length=64), nullable=False),
        sa.Column("signed_in", sa.Boolean(), nullable=False),
        sa.Column("site_id", sa.Integer(), nullable=True),
        sa.Column("job_key", sa.String(length=64), nullable=False),
        sa.Column("job_name", sa.String(length=100), nullable=False),
        sa.Column("job_version", sa.Integer(), nullable=True),
        sa.Column("required", _jsonb(), nullable=False),
        sa.Column("pending", _jsonb(), nullable=False),
        sa.Column("verdict", sa.String(length=16), nullable=False),
        sa.Column("verdict_source", sa.String(length=32), nullable=False),
        sa.Column("confidence_avg", sa.Float(), nullable=False),
        sa.Column("missing_equipment", _jsonb(), nullable=False),
        sa.Column("violations_detected", _jsonb(), nullable=False),
        sa.Column("unverifiable_equipment", _jsonb(), nullable=False),
        sa.Column("detections", _jsonb(), nullable=False),
        sa.Column("verification", _jsonb(), nullable=True),
        sa.Column("explanation", sa.Text(), nullable=True),
        sa.Column("model_used", sa.String(length=64), nullable=False),
        sa.Column("inference_time_ms", sa.Float(), nullable=False),
        sa.Column("gemini_invoked", sa.Boolean(), nullable=False),
        sa.Column("confidence_threshold", sa.Float(), nullable=False),
        sa.Column("gemini_verify", sa.Boolean(), nullable=False),
        sa.Column("gemini_min_confidence", sa.Float(), nullable=False),
        sa.Column("photo_state", sa.String(length=16), nullable=False),
        sa.Column("photo_key", sa.String(length=255), nullable=True),
        sa.Column("photo_type", sa.String(length=32), nullable=True),
        sa.Column("photo_width", sa.Integer(), nullable=True),
        sa.Column("photo_height", sa.Integer(), nullable=True),
        sa.Column("photo_bytes", sa.Integer(), nullable=True),
        sa.Column("photo_deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("training_consent", sa.Boolean(), server_default=sa.text("false"),
                  nullable=False),
        sa.Column("review_outcome", sa.String(length=16), nullable=True),
        sa.Column("review_note", sa.String(length=500), nullable=True),
        sa.Column("reviewed_by", sa.Integer(), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("(photo_state = 'kept') = (photo_key IS NOT NULL)",
                           name=op.f("ck_checks_photo_key_when_kept")),
        sa.CheckConstraint("photo_state IN ('kept', 'deleted', 'not_stored')",
                           name=op.f("ck_checks_photo_state_valid")),
        sa.CheckConstraint(
            "review_outcome IS NULL OR review_outcome IN ('compliant', 'not_compliant')",
            name=op.f("ck_checks_review_outcome_valid"),
        ),
        sa.CheckConstraint("verdict IN ('PASS', 'FAIL', 'NEEDS_REVIEW')",
                           name=op.f("ck_checks_verdict_valid")),
        sa.CheckConstraint("(review_outcome IS NULL) = (reviewed_at IS NULL)",
                           name=op.f("ck_checks_review_complete")),
        sa.ForeignKeyConstraint(["reviewed_by"], ["users.id"],
                                name=op.f("fk_checks_reviewed_by_users"), ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["site_id"], ["sites.id"],
                                name=op.f("fk_checks_site_id_sites"), ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["worker_id"], ["workers.id"],
                                name=op.f("fk_checks_worker_id_workers"), ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_checks")),
    )
    op.create_index("ix_checks_at", "checks", ["at"], unique=False)
    op.create_index("ix_checks_photo_state_at", "checks", ["photo_state", "at"], unique=False)
    op.create_index("ix_checks_site_id_at", "checks", ["site_id", "at"], unique=False)
    op.create_index("ix_checks_worker_id_at", "checks", ["worker_id", "at"], unique=False)

    op.add_column("workers", sa.Column("consent_seen_version", sa.Integer(), nullable=True))
    op.add_column("workers", sa.Column("training_consent", sa.Boolean(),
                                       server_default=sa.text("false"), nullable=False))
    op.add_column("workers", sa.Column("consent_updated_at", sa.DateTime(timezone=True),
                                       nullable=True))


def downgrade() -> None:
    op.drop_column("workers", "consent_updated_at")
    op.drop_column("workers", "training_consent")
    op.drop_column("workers", "consent_seen_version")
    op.drop_index("ix_checks_worker_id_at", table_name="checks")
    op.drop_index("ix_checks_site_id_at", table_name="checks")
    op.drop_index("ix_checks_photo_state_at", table_name="checks")
    op.drop_index("ix_checks_at", table_name="checks")
    op.drop_table("checks")

"""consent_notices: each version of the photo notice

Revision ID: 0005
Revises: 0004
Create Date: 2026-11-26

The server writes the photo notice the app shows workers, with the retention
periods filled in, and every change to its words is a new version
(docs/checks-api.md §5). This keeps each version, so the version a worker
answered still says what they read. The table starts empty: the first request
for the notice writes version 1.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "consent_notices",
        sa.Column("version", sa.Integer(), autoincrement=False, nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("training_question", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"),
                  nullable=False),
        sa.PrimaryKeyConstraint("version", name=op.f("pk_consent_notices")),
    )


def downgrade() -> None:
    op.drop_table("consent_notices")

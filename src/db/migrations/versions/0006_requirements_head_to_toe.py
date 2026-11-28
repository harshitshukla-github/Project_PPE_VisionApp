"""job_requirements: each job's equipment head to toe

Revision ID: 0006
Revises: 0005
Create Date: 2026-11-28

A job's equipment used to keep the order it was entered in: a new job's
followed the order its boxes were ticked, and an item switched on in the
matrix went last. The server now keeps every list in the items' order, head
to toe (docs/admin-api.md §5, decision 9 in §12). This puts the lists already
stored into that order. Only ``position`` changes: no job's version, and no
audit entry, since what each job requires is the same.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# The items' order when this migration was written (src/db/matrix.py ITEMS),
# copied so a later change there can't change what this migration does.
HEAD_TO_TOE = ("hardhat", "goggles", "face_shield", "ear_protection", "mask", "safety_vest",
               "harness", "gloves", "boots")


def upgrade() -> None:
    rank = " ".join(f"WHEN '{key}' THEN {index}" for index, key in enumerate(HEAD_TO_TOE))
    # An item not in the list (none today) keeps its place after the known ones.
    op.execute(f"""
        UPDATE job_requirements AS r
        SET position = ordered.new_position
        FROM (
            SELECT job_id, item_key,
                   ROW_NUMBER() OVER (
                       PARTITION BY job_id
                       ORDER BY CASE item_key {rank} ELSE {len(HEAD_TO_TOE)} END, position
                   ) - 1 AS new_position
            FROM job_requirements
        ) AS ordered
        WHERE r.job_id = ordered.job_id AND r.item_key = ordered.item_key
    """)


def downgrade() -> None:
    # The order things were entered in isn't kept anywhere, so there is nothing
    # to go back to; head to toe is a valid order for the old code too.
    pass

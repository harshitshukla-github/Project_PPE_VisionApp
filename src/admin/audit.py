"""The audit log: who changed what, and when (docs/admin-api.md §9).

:func:`record` adds an entry to the caller's database session without
committing, so the entry is saved in the same transaction as the change it
describes: if the change is saved, so is its record, and neither without
the other.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.admin.common import decode_cursor, encode_cursor
from src.api.errors import ApiError
from src.db.models import AuditEntry

CLI_ACTOR = "cli"


def record(
    db: Session,
    *,
    actor: str,
    action: str,
    target: str,
    site: str | None = None,
    before: dict[str, Any] | None = None,
    after: dict[str, Any] | None = None,
    ip: str | None = None,
) -> None:
    """Add an audit entry to the current transaction.

    Args:
        db: The session the change is being made in. Not committed here.
        actor: The acting user's email, or ``cli``.
        action: e.g. ``job.requirements`` (the table in §9 lists them).
        target: e.g. ``job:welder``.
        site: The site key the change concerns, if any.
        before: The documented keys before the change. Never a secret.
        after: The documented keys after the change. Never a secret.
        ip: The client's address, for changes made through the API.
    """
    db.add(AuditEntry(actor=actor, action=action, target=target, site=site,
                      before=before, after=after, ip=(ip or "")[:64] or None))


def changes(old: dict[str, Any], new: dict[str, Any]) -> tuple[dict, dict]:
    """The keys whose values differ, as ``(before, after)``."""
    keys = [k for k in new if old.get(k) != new[k]]
    return {k: old.get(k) for k in keys}, {k: new[k] for k in keys}


def list_entries(db: Session, *, action: str | None, target: str | None, actor: str | None,
                 site: str | None, start: datetime | None, end: datetime | None, limit: int,
                 cursor: str | None) -> tuple[list[AuditEntry], str | None, int]:
    """One page of the audit log, newest first, with the total that match."""
    filters = []
    for column, value in ((AuditEntry.action, action), (AuditEntry.target, target),
                          (AuditEntry.actor, actor), (AuditEntry.site, site)):
        if value:
            filters.append(column == value)
    if start is not None:
        filters.append(AuditEntry.at >= start)
    if end is not None:
        filters.append(AuditEntry.at < end)

    total = db.scalar(select(func.count()).select_from(AuditEntry).where(*filters)) or 0
    query = select(AuditEntry).where(*filters).order_by(AuditEntry.id.desc())
    if cursor:
        before_id = decode_cursor(cursor)
        if not isinstance(before_id, int):
            raise ApiError(422, "invalid", "That page cursor isn't valid.", field="cursor")
        query = query.where(AuditEntry.id < before_id)
    rows = list(db.scalars(query.limit(limit + 1)))
    next_cursor = encode_cursor(rows[limit - 1].id) if len(rows) > limit else None
    return rows[:limit], next_cursor, total

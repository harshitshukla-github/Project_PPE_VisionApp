"""Stored checks for the dashboard (docs/checks-api.md §4).

Admins see every check; a supervisor only their own site's. A check's site
is the worker's site when it was made (§1). A check with no site (sent
without a login, under an ID no worker has) is visible to admins only.

A supervisor asking for another site's check gets ``404``, the same as for a
check that doesn't exist, so the API reveals nothing about other sites.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import false, func, select, tuple_
from sqlalchemy.orm import Session, selectinload

from src.admin import audit
from src.admin.common import decode_cursor, encode_cursor, is_admin
from src.admin.views import check_view
from src.api.errors import ApiError
from src.auth.service import utcnow
from src.checks.photos import forget_photos, photo_store
from src.db.models import Check, Site, User, Worker

VERDICTS = ("PASS", "FAIL", "NEEDS_REVIEW")
OUTCOMES = ("compliant", "not_compliant")
NOTE_MAX = 500

_LOADS = (selectinload(Check.worker), selectinload(Check.site), selectinload(Check.reviewer))


def _scoped(query, user: User):
    # A supervisor always has a site; a check with none never matches it.
    return query if is_admin(user) else query.where(Check.site_id == user.site_id)


# ---------------------------------------------------------------------------
# Reading (§4.1, §4.2)
# ---------------------------------------------------------------------------


def list_checks(db: Session, user: User, *, site: str | None, worker: str | None,
                job: str | None, verdict: str | None, reviewed: bool | None,
                signed_in: bool | None, start: datetime | None, end: datetime | None,
                limit: int, cursor: str | None) -> tuple[list[Check], str | None, int]:
    """One page of checks, newest first, with the total that match."""
    filters = []
    if not is_admin(user):
        filters.append(Check.site_id == user.site_id)  # `site` is ignored for supervisors
    elif site:
        site_id = db.scalar(select(Site.id).where(Site.key == site))
        filters.append(Check.site_id == site_id if site_id is not None else false())
    code = (worker or "").strip().upper()
    if code:
        # The ID as the check gave it, so this also finds checks sent without a
        # login under an ID that names no worker.
        filters.append(func.upper(func.btrim(Check.worker_code)) == code)
    if job:
        filters.append(Check.job_key == job)
    if verdict:
        if verdict not in VERDICTS:
            raise ApiError(422, "invalid", "verdict is PASS, FAIL or NEEDS_REVIEW.",
                           field="verdict")
        filters.append(Check.verdict == verdict)
    if reviewed is not None:
        filters.append(Check.review_outcome.is_not(None) if reviewed
                       else Check.review_outcome.is_(None))
    if signed_in is not None:
        filters.append(Check.signed_in.is_(signed_in))
    if start is not None:
        filters.append(Check.at >= start)
    if end is not None:
        filters.append(Check.at < end)

    total = db.scalar(select(func.count()).select_from(Check).where(*filters)) or 0
    query = (select(Check).where(*filters).options(*_LOADS)
             .order_by(Check.at.desc(), Check.id.desc()))
    if cursor:
        query = query.where(tuple_(Check.at, Check.id) < tuple_(*_after(cursor)))
    rows = list(db.scalars(query.limit(limit + 1)))
    next_cursor = None
    if len(rows) > limit:
        last = rows[limit - 1]
        next_cursor = encode_cursor([last.at.isoformat(), last.id])
    return rows[:limit], next_cursor, total


def _after(cursor: str) -> tuple[datetime, int]:
    value = decode_cursor(cursor)
    try:
        at, check_id = value
        if not isinstance(at, str) or not isinstance(check_id, int) or isinstance(check_id, bool):
            raise ValueError
        return datetime.fromisoformat(at), check_id
    except (TypeError, ValueError):
        raise ApiError(422, "invalid", "That page cursor isn't valid.", field="cursor") from None


def get_check(db: Session, user: User, check_id: int, *, lock: bool = False) -> Check:
    """A check this user may see, or ``404 not_found``."""
    query = _scoped(select(Check).where(Check.id == check_id), user).options(*_LOADS)
    if lock:
        query = query.with_for_update(of=Check)
    check = db.scalar(query)
    if check is None:
        raise ApiError(404, "not_found", f"No check {check_id}.")
    return check


def read_photo(db: Session, user: User, check_id: int) -> tuple[bytes, str]:
    """The check's photo and its content type (§4.2).

    ``404 not_found`` if it was never stored, ``410 photo_deleted`` if it
    was and has since been deleted.
    """
    check = get_check(db, user, check_id)
    if check.photo_state == "not_stored":
        raise ApiError(404, "not_found", "This check's photo was never stored.")
    if check.photo_state == "deleted":
        raise ApiError(410, "photo_deleted", "This check's photo has been deleted.")
    return photo_store().read(check.photo_key), check.photo_type or "application/octet-stream"


# ---------------------------------------------------------------------------
# Reviewing (§4.3)
# ---------------------------------------------------------------------------


def review_check(db: Session, user: User, check_id: int, *, outcome: str | None,
                 note: str | None, replace: bool, actor: str, ip: str | None) -> Check:
    """Record a person's decision beside the check's verdict, which stays as it was.

    Any verdict can be reviewed, once. A second review gets
    ``409 already_reviewed`` with the check as it is now, unless an admin
    sends ``replace``. Every review is its own ``check.review`` audit entry.
    """
    check = get_check(db, user, check_id, lock=True)
    if outcome not in OUTCOMES:
        raise ApiError(422, "invalid", "A review's outcome is compliant or not_compliant.",
                       field="outcome")
    text = (note or "").strip() or None
    if text is not None and len(text) > NOTE_MAX:
        raise ApiError(422, "invalid", f"A note is at most {NOTE_MAX} characters.", field="note")
    if replace and not is_admin(user):
        raise ApiError(422, "invalid", "Only an admin can replace a review.", field="replace")
    if check.review_outcome is not None and not replace:
        raise ApiError(409, "already_reviewed", "Someone has reviewed this check already.",
                       extra={"current": check_view(check)})

    before = ({"outcome": check.review_outcome, "note": check.review_note}
              if check.review_outcome is not None else None)
    check.review_outcome, check.review_note = outcome, text
    check.reviewed_by, check.reviewed_at = user.id, utcnow()
    audit.record(db, actor=actor, action="check.review", target=f"check:{check.id}",
                 site=check.site.key if check.site else None, before=before,
                 after={"outcome": outcome, "note": text}, ip=ip)
    db.commit()
    return get_check(db, user, check_id)


# ---------------------------------------------------------------------------
# Deleting a worker's photos (§4.4)
# ---------------------------------------------------------------------------


def kept_photo_count(db: Session, worker: Worker) -> int:
    return db.scalar(select(func.count()).select_from(Check)
                     .where(Check.worker_id == worker.id, Check.photo_state == "kept")) or 0


def delete_worker_photos(db: Session, worker: Worker, actor: str, ip: str | None) -> int:
    """Delete every kept photo of this worker, files included; return how many.

    Their checks stay, with ``photo.state: "deleted"``. If a file can't be
    deleted, the ones already gone are saved with their audit entry, and the
    error is raised: asking again deletes the rest.
    """
    checks = db.scalars(select(Check)
                        .where(Check.worker_id == worker.id, Check.photo_state == "kept")
                        .order_by(Check.id).with_for_update()).all()
    deleted, error = forget_photos(checks, photo_store())
    audit.record(db, actor=actor, action="worker.photos_delete",
                 target=f"worker:{worker.worker_code}", site=worker.site.key,
                 after={"deleted": deleted}, ip=ip)
    db.commit()
    if error is not None:
        raise error
    return deleted

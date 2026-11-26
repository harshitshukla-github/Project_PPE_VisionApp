"""How long photos are kept, and deleting them when that time is up (docs/checks-api.md §6).

Two periods, both admin settings: ``photo_review_days`` for photos without
training consent and ``photo_training_days`` for photos with it. A photo's
consent is the one stamped on its check, which a withdrawal clears (§5), so
a withdrawn worker's photos fall back to the review period.

The API server runs :func:`purge` once a day by itself (:func:`run_daily`),
and ``cli.py photos purge`` runs it on demand. Check records stay: only the
photos go, and each run writes one ``photos.purge`` audit entry.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from loguru import logger
from sqlalchemy import and_, func, or_, select, text
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from src.admin import audit, runtime
from src.checks.photos import PhotoStore, forget_photos, photo_store
from src.db.models import AuditEntry, Check

PURGE_ACTION = "photos.purge"
# The daily job looks every hour whether a day has passed since the last run,
# whichever server or command made it, so restarts don't reset the clock.
PURGE_EVERY = timedelta(hours=24)
LOOK_EVERY_S = 3600.0
FIRST_LOOK_AFTER_S = 60.0
# A Postgres advisory lock, so two servers never purge at once.
_LOCK_KEY = 0x50504236  # "PPB6"


def periods() -> tuple[int, int]:
    """The review and training periods in force, in days. Training is never the shorter."""
    review = int(runtime.get("photo_review_days"))
    return review, max(int(runtime.get("photo_training_days")), review)


def expired(review_days: int, training_days: int, now: datetime):
    """The SQL condition for a kept photo that is past its period."""
    return and_(
        Check.photo_state == "kept",
        or_(
            and_(Check.training_consent.is_(False), Check.at < now - timedelta(days=review_days)),
            and_(Check.training_consent.is_(True), Check.at < now - timedelta(days=training_days)),
        ),
    )


def count_expired(db: Session, review_days: int, training_days: int) -> int:
    """How many photos a purge with these periods would delete now."""
    now = datetime.now(timezone.utc)
    return db.scalar(select(func.count()).select_from(Check)
                     .where(expired(review_days, training_days, now))) or 0


def purge(db: Session, store: PhotoStore | None = None, *, actor: str = audit.CLI_ACTOR) -> int:
    """Delete every photo past its period; return how many.

    Writes one ``photos.purge`` audit entry, even when nothing was due. If a
    file can't be deleted, the ones already gone are saved with the entry
    and the error is raised; the next run deletes the rest.
    """
    review, training = periods()
    now = datetime.now(timezone.utc)
    due = db.scalars(select(Check).where(expired(review, training, now)).order_by(Check.id)
                     .with_for_update(skip_locked=True)).all()
    deleted, error = forget_photos(due, store or photo_store())
    audit.record(db, actor=actor, action=PURGE_ACTION, target="photos",
                 after={"deleted": deleted})
    db.commit()
    if error is not None:
        raise error
    logger.info(f"Photo retention: deleted {deleted} photo{'' if deleted == 1 else 's'} "
                f"past {review} days (or {training} with training consent).")
    return deleted


def purge_if_due(*, force: bool = False) -> int | None:
    """Run :func:`purge` if a day has passed since the last run anywhere.

    Returns how many photos were deleted, or ``None`` if it wasn't due or
    another server is purging right now.
    """
    from src.db.engine import get_engine  # noqa: PLC0415 — only with a database

    with Session(get_engine()) as db:
        if not db.scalar(select(func.pg_try_advisory_xact_lock(text(str(_LOCK_KEY))))):
            return None
        last = db.scalar(select(func.max(AuditEntry.at)).where(AuditEntry.action == PURGE_ACTION))
        if not force and last is not None and datetime.now(timezone.utc) - last < PURGE_EVERY:
            return None
        return purge(db)


async def run_daily() -> None:
    """The server's daily purge: look every hour, purge when a day has passed.

    A run that fails is tried again at the next look rather than a day
    later. Runs until cancelled at shutdown.
    """
    await asyncio.sleep(FIRST_LOOK_AFTER_S)
    force = False
    while True:
        try:
            await run_in_threadpool(purge_if_due, force=force)
            force = False
        except Exception as exc:  # noqa: BLE001 — logged; the next look tries again
            reason = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
            logger.error(f"Photo retention run failed, trying again within the hour: {reason}")
            force = True
        await asyncio.sleep(LOOK_EVERY_S)

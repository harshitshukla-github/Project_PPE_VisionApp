"""The photo notice, and each worker's answer to it (docs/checks-api.md §5).

The server writes the notice, with the retention periods in force filled in,
so an admin changing a period, or a change of wording, needs no app release.
Any change to the words is a new version: the app shows the notice again,
and an answer to an older version gets ``409 notice_changed``, so a worker
always answers the words they were shown.

Two uses of a photo, two different asks: workers are *told* their photos are
kept for review, with nothing to agree to; keeping them longer for training
is *their choice*, off until they say yes. Withdrawing that choice clears it
on every earlier check too, so those photos fall back to the review period
and are never exported for training again.

The wording is the contract's first draft. It goes past whoever handles
privacy for the company before it ships in an app release.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.admin import audit
from src.admin.views import iso
from src.api.errors import ApiError
from src.auth.service import utcnow
from src.checks.retention import periods
from src.db.models import Check, ConsentNotice, Worker

NOTICE_TEXT = (
    "Photos from your PPE checks are kept on the company's server for up to "
    "{review_days} days, so supervisors at your site can review checks. When the app can't "
    "tell from the photo alone, it is also sent to Google's Gemini service to check it. Ask "
    "your supervisor if you want your photos deleted."
)
TRAINING_QUESTION = (
    "Can we also keep your photos for up to {training_days} days to improve the PPE detector? "
    "You can change this in Settings at any time."
)


def render() -> tuple[str, str]:
    """The notice and the training question, with the periods in force."""
    review_days, training_days = periods()
    return (NOTICE_TEXT.format(review_days=review_days),
            TRAINING_QUESTION.format(training_days=training_days))


def current_notice(db: Session) -> ConsentNotice:
    """The notice as it reads now, adding a version if its words changed.

    Not committed here: the caller commits, with whatever else it changes.
    """
    text, question = render()
    latest = _latest(db)
    if latest is not None and (latest.text, latest.training_question) == (text, question):
        return latest
    notice = ConsentNotice(version=latest.version + 1 if latest else 1, text=text,
                           training_question=question)
    try:
        with db.begin_nested():
            db.add(notice)
    except IntegrityError:
        # Another request wrote this version a moment ago, from the same settings.
        return _latest(db)
    return notice


def _latest(db: Session) -> ConsentNotice | None:
    return db.scalar(select(ConsentNotice).order_by(ConsentNotice.version.desc()).limit(1))


def notice_view(notice: ConsentNotice) -> dict[str, Any]:
    return {"version": notice.version, "text": notice.text,
            "training_question": notice.training_question}


def consent_view(notice: ConsentNotice, worker: Worker) -> dict[str, Any]:
    """What ``GET /auth/worker/consent`` answers."""
    return {
        "notice": notice_view(notice),
        "seen_version": worker.consent_seen_version,
        "training": worker.training_consent,
        "updated_at": iso(worker.consent_updated_at),
    }


def get_consent(db: Session, worker: Worker) -> dict[str, Any]:
    notice = current_notice(db)
    db.commit()
    return consent_view(notice, worker)


def answer(db: Session, worker_id: int, *, version: int, training: bool,
           ip: str | None) -> dict[str, Any]:
    """Record the worker's answer to the notice they were shown.

    ``409 notice_changed``, with the current notice, if that isn't the
    current version. A withdrawal clears training consent on all their
    earlier checks, in the same transaction as the ``worker.consent`` entry.
    """
    notice = current_notice(db)
    if version != notice.version:
        db.commit()  # keep a version written just now
        raise ApiError(409, "notice_changed",
                       "The notice has changed since the app showed it. Show this one.",
                       extra={"notice": notice_view(notice)})

    worker = db.scalar(select(Worker).where(Worker.id == worker_id).with_for_update()
                       .execution_options(populate_existing=True))
    old = {"version": worker.consent_seen_version, "training": worker.training_consent}
    new = {"version": version, "training": training}
    if old != new:
        if old["training"] and not training:
            db.execute(update(Check)
                       .where(Check.worker_id == worker.id, Check.training_consent.is_(True))
                       .values(training_consent=False))
        worker.consent_seen_version, worker.training_consent = version, training
        worker.consent_updated_at = utcnow()
        actor = f"worker:{worker.worker_code}"
        audit.record(db, actor=actor, action="worker.consent", target=actor,
                     site=worker.site.key, before=old, after=new, ip=ip)
    db.commit()
    return consent_view(notice, worker)

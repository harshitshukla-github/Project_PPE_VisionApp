"""Stores a check once its verdict is decided (docs/checks-api.md §2).

Storing never costs the worker a verdict: :func:`record_check` never raises.
If the database or the photo store fails, the worker still gets their answer,
the server logs what wasn't stored, and the response's ``check_id`` is null.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.admin import runtime
from src.checks.location import content_type, remove_location
from src.checks.photos import photo_store
from src.db.matrix import display_name_for
from src.db.models import Check, Job, Worker
from src.settings import settings

if TYPE_CHECKING:
    from src.auth.deps import SignedInWorker
    from src.inference.compliance import JobRequirements
    from src.inference.schemas import ComplianceResponse

# A worker ID typed into an app without login can be anything; keep it bounded.
_MAX_TYPED_ID = 64


def photo_may_be_kept(signed_in: SignedInWorker | None, worker: Worker | None) -> bool:
    """Whether this check's photo may be kept (§5, and decisions Q1 and Q8 in #34).

    Only for a signed-in worker who has answered the photo notice, and only
    while ``store_photos`` is on. An app without login never shows the
    notice, so its checks never keep a photo; nor does a signed-in worker
    whose app hasn't shown it yet.
    """
    return (
        signed_in is not None
        and worker is not None
        and worker.consent_seen_version is not None
        and bool(runtime.get("store_photos"))
    )


def record_check(
    response: ComplianceResponse,
    *,
    signed_in: SignedInWorker | None,
    requirements: JobRequirements,
    upload: bytes,
    width: int,
    height: int,
    confidence_threshold: float,
) -> int | None:
    """Store the check, and its photo when that's allowed; return its ID.

    Args:
        response: What the app is about to get.
        signed_in: The worker behind the request, or ``None`` without a login.
        requirements: The job's requirements the verdict was judged against.
        upload: The photo as uploaded.
        width: The photo's width after orientation: the space of the boxes.
        height: Its height after orientation.
        confidence_threshold: The detector threshold the verdict used.

    Returns:
        The new check's ID, or ``None`` if it wasn't stored (no database
        configured, or something failed; the reason is logged).
    """
    if not settings.database_url:
        return None

    store = photo_store()
    kept: str | None = None
    try:
        from src.db.engine import get_engine  # noqa: PLC0415 — only with a database

        with Session(get_engine()) as db:
            worker = _worker_for(db, signed_in, response.worker_id)
            job = db.scalar(select(Job).where(Job.key == response.job_category))
            photo = {
                "photo_state": "not_stored",
                "photo_key": None,
                "photo_type": content_type(upload),
                "photo_bytes": len(upload),
            }
            if photo_may_be_kept(signed_in, worker):
                try:
                    data, kind = remove_location(upload)
                    kept = store.save(data, kind)
                    photo.update(photo_state="kept", photo_key=kept, photo_type=kind,
                                 photo_bytes=len(data))
                except Exception as exc:  # noqa: BLE001 — the check is stored without it
                    logger.warning(f"Check photo not kept: {_reason(exc)}")

            row = Check(
                at=response.timestamp,
                worker_id=worker.id if worker else None,
                worker_code=signed_in.worker_id if signed_in else response.worker_id[:_MAX_TYPED_ID],
                signed_in=signed_in is not None,
                site_id=worker.site_id if worker else None,
                job_key=response.job_category,
                job_name=job.display_name if job else display_name_for(response.job_category),
                job_version=job.version if job else None,
                required=list(requirements.required),
                pending=list(requirements.pending),
                verdict=response.verdict.value,
                verdict_source=response.verdict_source,
                confidence_avg=response.confidence_avg,
                missing_equipment=list(response.missing_equipment),
                violations_detected=list(response.violations_detected),
                unverifiable_equipment=list(response.unverifiable_equipment),
                detections=[d.model_dump(mode="json", by_alias=True) for d in response.detections],
                verification=(response.verification.model_dump(mode="json")
                              if response.verification else None),
                explanation=response.explanation,
                model_used=response.model_used,
                inference_time_ms=response.inference_time_ms,
                gemini_invoked=response.gemini_invoked,
                confidence_threshold=confidence_threshold,
                gemini_verify=bool(runtime.get("gemini_verify")),
                gemini_min_confidence=float(runtime.get("gemini_min_confidence")),
                photo_width=width,
                photo_height=height,
                training_consent=bool(signed_in and worker and worker.training_consent),
                **photo,
            )
            db.add(row)
            db.commit()
            return row.id
    except Exception as exc:  # noqa: BLE001 — storing must never cost the worker a verdict
        logger.error(f"Check not stored ({response.job_category}, {response.verdict.value}): "
                     f"{_reason(exc)}")
        if kept is not None:
            _forget(store, kept)
        return None


def _worker_for(db: Session, signed_in: SignedInWorker | None, typed_id: str) -> Worker | None:
    """The worker behind the check: from the login, or the typed ID if it names one."""
    code = signed_in.worker_id if signed_in else typed_id.strip().upper()
    return db.scalar(select(Worker).where(func.upper(Worker.worker_code) == code))


def _forget(store, key: str) -> None:
    """Remove a photo whose check row never made it, so no file is left unaccounted for."""
    try:
        store.delete(key)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"A kept photo whose check wasn't stored couldn't be removed ({key}): "
                     f"{_reason(exc)}")


def _reason(exc: Exception) -> str:
    text = str(exc).strip()
    return text.splitlines()[0] if text else type(exc).__name__

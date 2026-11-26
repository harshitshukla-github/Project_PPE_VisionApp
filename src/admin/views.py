"""How admin API objects are written out (docs/admin-api.md).

Plain dictionaries rather than response models, so every timestamp goes
through :func:`iso` and comes out in UTC with a ``Z``, as the contracts say.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.db.models import AuditEntry, Check, Job, Site, User, Worker


def iso(moment: datetime | None) -> str | None:
    """``2026-10-16T10:12:00Z``: UTC, with a Z, or ``None``."""
    if moment is None:
        return None
    return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def site_view(site: Site, worker_count: int) -> dict[str, Any]:
    return {
        "key": site.key,
        "display_name": site.display_name,
        "is_active": site.is_active,
        "worker_count": worker_count,
        "created_at": iso(site.created_at),
    }


def requirements_view(job: Job) -> list[dict[str, str]]:
    return [{"item": r.item_key, "mode": r.mode} for r in job.requirements]


def job_view(job: Job) -> dict[str, Any]:
    return {
        "key": job.key,
        "display_name": job.display_name,
        "position": job.position,
        "is_active": job.is_active,
        "version": job.version,
        "requirements": requirements_view(job),
        "updated_at": iso(job.updated_at),
    }


def user_view(user: User) -> dict[str, Any]:
    return {
        "id": user.id,
        "email": user.email,
        "display_name": user.display_name,
        "role": user.role,
        "site": user.site.key if user.site else None,
        "is_active": user.is_active,
        "must_change_password": user.must_change_password,
        "locked_until": iso(user.locked_until),
        "last_login_at": iso(user.last_login_at),
        "created_at": iso(user.created_at),
    }


def worker_view(worker: Worker) -> dict[str, Any]:
    return {
        "worker_id": worker.worker_code,
        "display_name": worker.display_name,
        "site": worker.site.key,
        "is_active": worker.is_active,
        "must_change_pin": worker.must_change_pin,
        "locked_until": iso(worker.locked_until),
        "last_login_at": iso(worker.last_login_at),
        "created_at": iso(worker.created_at),
        # null for a worker made from the CLI (docs/admin-api.md §8)
        "created_by": worker.creator.email if worker.creator else None,
    }


def review_view(check: Check) -> dict[str, Any] | None:
    if check.review_outcome is None:
        return None
    return {
        "outcome": check.review_outcome,
        "note": check.review_note,
        "by": check.reviewer.email if check.reviewer else None,
        "at": iso(check.reviewed_at),
    }


def check_view(check: Check, *, full: bool = True) -> dict[str, Any]:
    """A check as docs/checks-api.md §2 writes it.

    Args:
        check: With its worker, site and reviewer loaded.
        full: ``False`` for the list, which leaves out ``detections`` and
            ``verification`` to stay small (§4.1).
    """
    view: dict[str, Any] = {
        "id": check.id,
        "at": iso(check.at),
        "worker_id": check.worker_code,
        # null when the ID a check without login gave names no worker (§2)
        "worker_name": check.worker.display_name if check.worker else None,
        # the worker's site when the check was made (§1)
        "site": check.site.key if check.site else None,
        "signed_in": check.signed_in,
        "job": check.job_key,
        "job_name": check.job_name,
        "job_version": check.job_version,
        "required": check.required,
        "pending": check.pending,
        "verdict": check.verdict,
        "verdict_source": check.verdict_source,
        "missing_equipment": check.missing_equipment,
        "violations_detected": check.violations_detected,
        "unverifiable_equipment": check.unverifiable_equipment,
    }
    if full:
        view["detections"] = check.detections
        view["verification"] = check.verification
    view.update({
        "explanation": check.explanation,
        "model_used": check.model_used,
        "confidence_threshold": check.confidence_threshold,
        "gemini_verify": check.gemini_verify,
        "gemini_min_confidence": check.gemini_min_confidence,
        "photo": {
            "state": check.photo_state,
            "width": check.photo_width,
            "height": check.photo_height,
            "bytes": check.photo_bytes,
            "type": check.photo_type,
        },
        "training_consent": check.training_consent,
        "review": review_view(check),
    })
    return view


def audit_view(entry: AuditEntry) -> dict[str, Any]:
    return {
        "id": entry.id,
        "at": iso(entry.at),
        "actor": entry.actor,
        "action": entry.action,
        "target": entry.target,
        "site": entry.site,
        "before": entry.before,
        "after": entry.after,
        "ip": entry.ip,
    }

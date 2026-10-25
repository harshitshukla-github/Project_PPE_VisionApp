"""Jobs and the matrix (docs/admin-api.md §5).

Every write invalidates the cached job matrix, so the server that made the
change uses it on its next check; other processes follow within
``PPB_MATRIX_CACHE_S``.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

from src.admin import audit, runtime
from src.admin.common import check_key, clean_name
from src.admin.views import job_view, requirements_view
from src.api.errors import ApiError
from src.db.models import Job, JobRequirement, PpeItem, RequirementMode
from src.inference.compliance import invalidate_job_matrix_cache

MODES = {m.value for m in RequirementMode}


def list_jobs(db: Session) -> list[Job]:
    """Every job, active or not, in display order."""
    return list(db.scalars(
        select(Job).order_by(Job.position, Job.id).options(selectinload(Job.requirements))
    ))


def _items(db: Session) -> dict[str, PpeItem]:
    return {item.key: item for item in db.scalars(select(PpeItem))}


def _validated(db: Session, requirements: list[dict[str, str]]) -> dict[str, PpeItem]:
    """Check a requirements list; name the exact entry in ``field`` when it's wrong."""
    items = _items(db)
    seen: set[str] = set()
    for index, req in enumerate(requirements):
        item, mode = req.get("item"), req.get("mode")
        if mode not in MODES:
            raise ApiError(422, "invalid", "mode must be 'required' or 'pending'.",
                           field=f"requirements[{index}].mode")
        if item not in items:
            raise ApiError(422, "unknown_item", f'"{item}" isn\'t a known item.',
                           field=f"requirements[{index}].item")
        if item in seen:
            raise ApiError(422, "invalid", f'"{item}" is listed twice.',
                           field=f"requirements[{index}].item")
        seen.add(item)
    if not any(req["mode"] == RequirementMode.REQUIRED.value for req in requirements):
        raise ApiError(422, "no_required_item", "A job needs at least one required item.",
                       field="requirements")
    return items


def _warnings(items: dict[str, PpeItem], requirements: list[dict[str, str]]) -> list[dict]:
    warnings: list[dict] = []
    required = [r["item"] for r in requirements if r["mode"] == RequirementMode.REQUIRED.value]
    for key in required:
        if not items[key].detectable:
            warnings.append({
                "code": "gemini_only_item", "item": key,
                "message": f"{items[key].display_name} can only be judged by Gemini, whose "
                           "accuracy on it hasn't been measured.",
            })
    if required and not runtime.get("gemini_verify"):
        warnings.append({
            "code": "gemini_verify_off",
            "message": "Gemini verification is switched off, so anything the detector misses "
                       "stays \"Needs review\".",
        })
    return warnings


def _set_requirements(db: Session, job: Job, requirements: list[dict[str, str]]) -> None:
    # Delete then insert: rows keyed (job_id, item_key) may come back with the
    # same key, which a single flush can't replace in place.
    db.execute(delete(JobRequirement).where(JobRequirement.job_id == job.id))
    db.flush()
    db.expire(job, ["requirements"])
    for position, req in enumerate(requirements):
        db.add(JobRequirement(job_id=job.id, item_key=req["item"], mode=req["mode"],
                              position=position))
    db.flush()
    db.refresh(job, ["requirements"])


def _locked(db: Session, key: str) -> Job:
    job = db.scalar(select(Job).where(Job.key == key).with_for_update())
    if job is None:
        raise ApiError(404, "not_found", f"No job '{key}'.", field="key")
    return job


def _check_version(job: Job, version: Any) -> None:
    if version != job.version:
        raise ApiError(409, "version_conflict",
                       "Someone changed this job while you were editing it.",
                       extra={"current": job_view(job)})


def _done(db: Session) -> None:
    db.commit()
    invalidate_job_matrix_cache()


def create_job(db: Session, key: str, display_name: str, requirements: list[dict[str, str]],
               actor: str, ip: str | None) -> tuple[Job, list[dict]]:
    key = check_key(key)
    name = clean_name(display_name)
    items = _validated(db, requirements)
    position = (db.scalar(select(func.max(Job.position))) or -1) + 1
    job = Job(key=key, display_name=name, position=position)
    db.add(job)
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        raise ApiError(409, "already_exists", f"Job '{key}' already exists.",
                       field="key") from None
    _set_requirements(db, job, requirements)
    audit.record(db, actor=actor, action="job.create", target=f"job:{key}",
                 after={"key": key, "display_name": name,
                        "requirements": requirements_view(job)}, ip=ip)
    _done(db)
    return job, _warnings(items, requirements)


def update_job(db: Session, key: str, version: Any, changes: dict[str, Any], actor: str,
               ip: str | None) -> Job:
    """Rename or (de)activate a job. A change bumps its version."""
    job = _locked(db, key)
    _check_version(job, version)
    old = {"display_name": job.display_name, "is_active": job.is_active}
    new = dict(old)
    if "display_name" in changes:
        new["display_name"] = clean_name(changes["display_name"])
    if "is_active" in changes:
        if not isinstance(changes["is_active"], bool):
            raise ApiError(422, "invalid", "is_active must be true or false.", field="is_active")
        new["is_active"] = changes["is_active"]

    if old["is_active"] and not new["is_active"]:
        others = db.scalar(select(func.count()).select_from(Job).where(
            Job.is_active.is_(True), Job.id != job.id))
        if not others:
            raise ApiError(409, "last_active_job",
                           "This is the last active job; the app would have none to offer.")

    before, after = audit.changes(old, new)
    if after:
        job.display_name, job.is_active = new["display_name"], new["is_active"]
        job.version += 1
        audit.record(db, actor=actor, action="job.update", target=f"job:{key}",
                     before=before, after=after, ip=ip)
        _done(db)
    return job


def replace_requirements(db: Session, key: str, version: Any,
                         requirements: list[dict[str, str]], actor: str,
                         ip: str | None) -> tuple[Job, list[dict]]:
    """Replace a job's whole equipment list, as the matrix grid saves a row."""
    job = _locked(db, key)
    _check_version(job, version)
    items = _validated(db, requirements)
    before = requirements_view(job)
    if before != requirements:
        _set_requirements(db, job, requirements)
        job.version += 1
        audit.record(db, actor=actor, action="job.requirements", target=f"job:{key}",
                     before={"requirements": before},
                     after={"requirements": requirements_view(job)}, ip=ip)
        _done(db)
    return job, _warnings(items, requirements)


def reorder(db: Session, keys: list[str], actor: str, ip: str | None) -> list[Job]:
    """Set the display order. Versions don't change (docs/admin-api.md §5)."""
    jobs = {job.key: job for job in db.scalars(select(Job).with_for_update())}
    seen: set[str] = set()
    for index, key in enumerate(keys):
        if key not in jobs:
            raise ApiError(422, "invalid", f"No job '{key}'.", field=f"keys[{index}]")
        if key in seen:
            raise ApiError(422, "invalid", f"'{key}' is listed twice.", field=f"keys[{index}]")
        seen.add(key)
    missing = sorted(set(jobs) - seen)
    if missing:
        raise ApiError(422, "invalid",
                       f"Every job must appear exactly once; missing: {', '.join(missing)}.",
                       field="keys")

    before = [job.key for job in sorted(jobs.values(), key=lambda j: (j.position, j.id))]
    if before != keys:
        for position, key in enumerate(keys):
            jobs[key].position = position
        audit.record(db, actor=actor, action="job.order", target="jobs",
                     before={"keys": before}, after={"keys": keys}, ip=ip)
        _done(db)
    return list_jobs(db)


def view(job: Job) -> dict[str, Any]:
    return job_view(job)

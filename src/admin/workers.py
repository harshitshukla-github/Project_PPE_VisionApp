"""Workers (docs/admin-api.md §8). Admins see every site; a supervisor only theirs.

A supervisor asking for another site's worker gets ``404``, the same as for
a worker that doesn't exist, so the API reveals nothing about other sites.
The one accepted exception is ``409 already_exists`` when creating an ID
that is taken elsewhere.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session, selectinload

from src.admin import audit
from src.admin.common import (
    active_site,
    clean_name,
    decode_cursor,
    encode_cursor,
    end_sessions,
    find_site,
    is_admin,
)
from src.api.errors import ApiError
from src.auth import service
from src.auth.crypto import is_valid_worker_code, normalize_worker_code
from src.db.models import User, Worker

_LOADS = (selectinload(Worker.site), selectinload(Worker.creator))


def _scoped(query, user: User):
    return query if is_admin(user) else query.where(Worker.site_id == user.site_id)


def list_workers(db: Session, user: User, *, search: str | None, site: str | None,
                 active: bool | None, locked: bool | None, limit: int,
                 cursor: str | None) -> tuple[list[Worker], str | None, int]:
    """One page of workers, ordered by worker ID, with the total that match."""
    filters = []
    if is_admin(user):
        if site:
            filters.append(Worker.site_id == find_site(db, site).id)
    else:
        filters.append(Worker.site_id == user.site_id)  # `site` is ignored for supervisors
    if search:
        pattern = "%" + search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        filters.append(or_(Worker.worker_code.ilike(pattern, escape="\\"),
                           Worker.display_name.ilike(pattern, escape="\\")))
    if active is not None:
        filters.append(Worker.is_active.is_(active))
    if locked is not None:
        now = service.utcnow()
        filters.append(Worker.locked_until > now if locked else or_(
            Worker.locked_until.is_(None), Worker.locked_until <= now))

    total = db.scalar(select(func.count()).select_from(Worker).where(*filters)) or 0
    query = select(Worker).where(*filters).options(*_LOADS).order_by(Worker.worker_code)
    if cursor:
        after = decode_cursor(cursor)
        if not isinstance(after, str):
            raise ApiError(422, "invalid", "That page cursor isn't valid.", field="cursor")
        query = query.where(Worker.worker_code > after)
    rows = list(db.scalars(query.limit(limit + 1)))
    next_cursor = encode_cursor(rows[limit - 1].worker_code) if len(rows) > limit else None
    return rows[:limit], next_cursor, total


def get_worker(db: Session, user: User, worker_id: str, *, lock: bool = False) -> Worker:
    query = _scoped(select(Worker).where(Worker.worker_code == normalize_worker_code(worker_id)),
                    user).options(*_LOADS)
    if lock:
        query = query.with_for_update(of=Worker)
    worker = db.scalar(query)
    if worker is None:
        raise ApiError(404, "not_found", f"No worker {normalize_worker_code(worker_id)}.",
                       field="worker_id")
    return worker


def create_worker(db: Session, user: User, body: dict[str, Any], actor: str,
                  ip: str | None) -> tuple[Worker, str]:
    code = normalize_worker_code(body.get("worker_id") or "")
    if not is_valid_worker_code(code):
        raise ApiError(422, "invalid", "A worker ID is 2-32 letters, digits and hyphens, not "
                       "starting with a hyphen.", field="worker_id")
    name = clean_name(body.get("display_name"))
    site_key = body.get("site")
    if is_admin(user):
        if not site_key:
            raise ApiError(422, "invalid", "Choose the worker's site.", field="site")
        site = active_site(db, site_key)
    else:
        if site_key is not None and site_key != user.site.key:
            raise ApiError(403, "forbidden", "You can only add workers to your own site.",
                           field="site")
        site = active_site(db, user.site.key)

    if db.scalar(select(Worker.id).where(Worker.worker_code == code)) is not None:
        raise ApiError(409, "already_exists", f"Worker ID {code} is already taken.",
                       field="worker_id")
    try:
        worker, pin = service.create_worker(db, code, name, site_id=site.id,
                                            created_by=user.id, actor=actor, ip=ip)
    except ValueError as exc:  # a race on the ID
        raise ApiError(409, "already_exists", str(exc), field="worker_id") from None
    return get_worker(db, user, code), pin


def update_worker(db: Session, user: User, worker_id: str, changes: dict[str, Any],
                  actor: str, ip: str | None) -> Worker:
    """Rename, move (admins only) or (de)activate a worker."""
    worker = get_worker(db, user, worker_id, lock=True)
    old = {"display_name": worker.display_name, "site": worker.site.key,
           "is_active": worker.is_active}
    new = dict(old)
    site_id = worker.site_id

    if "site" in changes:
        if not is_admin(user):
            raise ApiError(403, "forbidden", "Only an admin can move a worker to another site.",
                           field="site")
        if not changes["site"]:
            raise ApiError(422, "invalid", "A worker needs a site.", field="site")
        if changes["site"] != old["site"]:
            site = active_site(db, changes["site"])
            new["site"], site_id = site.key, site.id
    if "display_name" in changes:
        new["display_name"] = clean_name(changes["display_name"])
    if "is_active" in changes:
        if not isinstance(changes["is_active"], bool):
            raise ApiError(422, "invalid", "is_active must be true or false.", field="is_active")
        new["is_active"] = changes["is_active"]

    before, after = audit.changes(old, new)
    if after:
        worker.display_name, worker.site_id = new["display_name"], site_id
        worker.is_active = new["is_active"]
        if old["is_active"] and not new["is_active"]:
            end_sessions(db, worker_id=worker.id)
        audit.record(db, actor=actor, action="worker.update", target=f"worker:{worker.worker_code}",
                     site=new["site"], before=before, after=after, ip=ip)
        db.commit()
        db.refresh(worker)
    return worker


def reset_pin(db: Session, user: User, worker_id: str, actor: str, ip: str | None) -> str:
    """New one-time PIN; unlocks the worker and ends their sessions."""
    worker = get_worker(db, user, worker_id)
    _, pin = service.reset_worker_pin(db, worker.worker_code, actor=actor, ip=ip)
    return pin

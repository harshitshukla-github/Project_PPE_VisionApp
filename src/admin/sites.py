"""Sites (docs/admin-api.md §3)."""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.admin import audit
from src.admin.common import check_key, clean_name, find_site, is_admin
from src.admin.views import site_view
from src.api.errors import ApiError
from src.db.models import Role, Site, User, Worker


def _active_worker_counts(db: Session) -> dict[int, int]:
    rows = db.execute(
        select(Worker.site_id, func.count())
        .where(Worker.is_active.is_(True))
        .group_by(Worker.site_id)
    ).all()
    return dict(rows)


def list_sites(db: Session, user: User) -> list[dict[str, Any]]:
    """Every site for an admin; only their own for a supervisor."""
    query = select(Site).order_by(Site.key)
    if not is_admin(user):
        query = query.where(Site.id == user.site_id)
    counts = _active_worker_counts(db)
    return [site_view(site, counts.get(site.id, 0)) for site in db.scalars(query)]


def view(db: Session, site: Site) -> dict[str, Any]:
    return site_view(site, _active_worker_counts(db).get(site.id, 0))


def create_site(db: Session, key: str, display_name: str, actor: str,
                ip: str | None) -> Site:
    key = check_key(key)
    name = clean_name(display_name)
    site = Site(key=key, display_name=name)
    db.add(site)
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        raise ApiError(409, "already_exists", f"Site '{key}' already exists.",
                       field="key") from None
    audit.record(db, actor=actor, action="site.create", target=f"site:{key}", site=key,
                 after={"key": key, "display_name": name}, ip=ip)
    db.commit()
    return site


def update_site(db: Session, key: str, changes: dict[str, Any], actor: str,
                ip: str | None) -> Site:
    """Rename or (de)activate a site.

    A site with active workers or supervisors can't be deactivated
    (``409 site_in_use``): they would be left somewhere that doesn't exist.
    """
    site = find_site(db, key, field="key")
    old = {"display_name": site.display_name, "is_active": site.is_active}
    new = dict(old)
    if "display_name" in changes:
        new["display_name"] = clean_name(changes["display_name"])
    if "is_active" in changes:
        if not isinstance(changes["is_active"], bool):
            raise ApiError(422, "invalid", "is_active must be true or false.", field="is_active")
        new["is_active"] = changes["is_active"]

    if old["is_active"] and not new["is_active"]:
        workers = db.scalar(select(func.count()).select_from(Worker).where(
            Worker.site_id == site.id, Worker.is_active.is_(True)))
        supervisors = db.scalar(select(func.count()).select_from(User).where(
            User.site_id == site.id, User.role == Role.SUPERVISOR.value,
            User.is_active.is_(True)))
        if workers or supervisors:
            raise ApiError(
                409, "site_in_use",
                f"Site '{key}' still has {workers} active workers and {supervisors} active "
                "supervisors. Move or deactivate them first.",
            )

    before, after = audit.changes(old, new)
    if after:
        site.display_name = new["display_name"]
        site.is_active = new["is_active"]
        audit.record(db, actor=actor, action="site.update", target=f"site:{key}", site=key,
                     before=before, after=after, ip=ip)
        db.commit()
    return site

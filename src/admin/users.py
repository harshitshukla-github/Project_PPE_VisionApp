"""Dashboard users (docs/admin-api.md §7). Admins only."""

from __future__ import annotations

import secrets
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from src.admin import audit
from src.admin.common import active_site, clean_name, end_sessions
from src.api.errors import ApiError
from src.auth import service
from src.auth.crypto import hash_secret, is_valid_email, normalize_email
from src.db.models import Role, User

ROLES = {r.value for r in Role}


def temporary_password() -> str:
    """A password for the admin to hand over; replaced at first login."""
    return secrets.token_urlsafe(12)


def list_users(db: Session) -> list[User]:
    return list(db.scalars(select(User).order_by(User.email).options(selectinload(User.site))))


def get_user(db: Session, user_id: int, *, lock: bool = False) -> User:
    query = select(User).where(User.id == user_id)
    if lock:
        query = query.with_for_update()
    user = db.scalar(query)
    if user is None:
        raise ApiError(404, "not_found", f"No user {user_id}.", field="id")
    return user


def create_user(db: Session, body: dict[str, Any], actor: str,
                ip: str | None) -> tuple[User, str]:
    email = normalize_email(body.get("email") or "")
    if not is_valid_email(email):
        raise ApiError(422, "invalid", "That isn't an email address.", field="email")
    name = clean_name(body.get("display_name"))
    role = body.get("role")
    if role not in ROLES:
        raise ApiError(422, "invalid", "role must be 'admin' or 'supervisor'.", field="role")
    site_key = body.get("site")
    site_id = None
    if role == Role.SUPERVISOR.value:
        if not site_key:
            raise ApiError(422, "invalid", "A supervisor needs a site.", field="site")
        site_id = active_site(db, site_key).id
    elif site_key is not None:
        raise ApiError(422, "invalid", "An admin works across all sites and has no site.",
                       field="site")

    if db.scalar(select(User.id).where(User.email == email)) is not None:
        raise ApiError(409, "already_exists", f"An account for {email} already exists.",
                       field="email")
    password = temporary_password()
    try:
        user = service.create_user(db, email, name, Role(role), password,
                                   must_change_password=True, site_id=site_id,
                                   actor=actor, ip=ip)
    except ValueError as exc:  # a race on the email, or a rule the checks above missed
        raise ApiError(409 if "already exists" in str(exc) else 422,
                       "already_exists" if "already exists" in str(exc) else "invalid",
                       str(exc)) from None
    return user, password


def update_user(db: Session, user_id: int, changes: dict[str, Any], me: User, actor: str,
                ip: str | None) -> User:
    """Change a user's name, role, site or active flag, with the §7 guards."""
    user = get_user(db, user_id, lock=True)
    old = {"display_name": user.display_name, "role": user.role,
           "site": user.site.key if user.site else None, "is_active": user.is_active}
    new = dict(old)

    if "display_name" in changes:
        new["display_name"] = clean_name(changes["display_name"])
    if "role" in changes:
        if changes["role"] not in ROLES:
            raise ApiError(422, "invalid", "role must be 'admin' or 'supervisor'.", field="role")
        new["role"] = changes["role"]
    if "is_active" in changes:
        if not isinstance(changes["is_active"], bool):
            raise ApiError(422, "invalid", "is_active must be true or false.", field="is_active")
        new["is_active"] = changes["is_active"]

    site_id = user.site_id
    if new["role"] == Role.ADMIN.value:
        if changes.get("site") is not None:
            raise ApiError(422, "invalid", "An admin works across all sites and has no site.",
                           field="site")
        new["site"], site_id = None, None
    elif "site" in changes:
        if not changes["site"]:
            raise ApiError(422, "invalid", "A supervisor needs a site.", field="site")
        site = active_site(db, changes["site"])
        new["site"], site_id = site.key, site.id
    elif site_id is None:
        raise ApiError(422, "invalid", "A supervisor needs a site; send one with the role.",
                       field="site")

    if user.id == me.id and any(new[k] != old[k] for k in ("role", "site", "is_active")):
        raise ApiError(409, "cannot_change_self",
                       "You can't change your own role, site or active flag.")
    loses_admin = old["role"] == Role.ADMIN.value and old["is_active"] and (
        new["role"] != Role.ADMIN.value or not new["is_active"])
    if loses_admin:
        admins = db.scalar(select(func.count()).select_from(User).where(
            User.role == Role.ADMIN.value, User.is_active.is_(True)))
        if admins <= 1:
            raise ApiError(409, "last_admin", "That would leave no active admin.")

    before, after = audit.changes(old, new)
    if after:
        user.display_name, user.role, user.site_id = new["display_name"], new["role"], site_id
        user.is_active = new["is_active"]
        if old["is_active"] and not new["is_active"]:
            end_sessions(db, user_id=user.id)
        audit.record(db, actor=actor, action="user.update", target=f"user:{user.email}",
                     site=new["site"] or old["site"], before=before, after=after, ip=ip)
        db.commit()
        db.refresh(user)
    return user


def reset_password(db: Session, user_id: int, me: User, actor: str, ip: str | None) -> str:
    """Issue a temporary password, unlock the user, and end their sessions."""
    user = get_user(db, user_id, lock=True)
    if user.id == me.id:
        raise ApiError(409, "cannot_change_self",
                       "Change your own password with POST /auth/password.")
    password = temporary_password()
    user.password_hash = hash_secret(password)
    user.must_change_password = True
    user.failed_attempts = 0
    user.locked_until = None
    end_sessions(db, user_id=user.id)
    audit.record(db, actor=actor, action="user.reset_password", target=f"user:{user.email}",
                 site=user.site.key if user.site else None, ip=ip)
    db.commit()
    return password

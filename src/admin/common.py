"""Small pieces the admin modules share: paging, validation, site lookups."""

from __future__ import annotations

import base64
import json
import re
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from src.api.errors import ApiError
from src.db.models import AuthSession, Role, Site, User
from src.settings import settings

# Site and job keys (docs/admin-api.md §2): a lowercase identifier, 2–64 characters.
KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
NAME_MAX = 100


def applies_within_s() -> int:
    """How long other server processes take to see a change (docs/admin-api.md §2)."""
    return int(settings.ppb_matrix_cache_s)


def is_admin(user: User) -> bool:
    return user.role == Role.ADMIN.value


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def clean_name(value: Any, field: str = "display_name") -> str:
    """A display name with its whitespace tidied, or ``422 invalid``."""
    name = " ".join(str(value or "").split()) if isinstance(value, str) else ""
    if not name or len(name) > NAME_MAX:
        raise ApiError(422, "invalid", f"A name is needed, of at most {NAME_MAX} characters.",
                       field=field)
    return name


def check_key(value: str, field: str = "key") -> str:
    """A site or job key, or ``422 invalid_key``."""
    if not KEY_PATTERN.fullmatch(value or ""):
        raise ApiError(
            422, "invalid_key",
            "A key is 2-64 characters: a lowercase letter, then lowercase letters, "
            "digits or underscores.",
            field=field,
        )
    return value


# ---------------------------------------------------------------------------
# Sites
# ---------------------------------------------------------------------------


def find_site(db: Session, key: str, field: str = "site") -> Site:
    """A site by key, active or not, or ``404 not_found``."""
    site = db.scalar(select(Site).where(Site.key == key))
    if site is None:
        raise ApiError(404, "not_found", f"No site '{key}'.", field=field)
    return site


def active_site(db: Session, key: str, field: str = "site") -> Site:
    """A site to put someone in: ``404`` if unknown, ``422 invalid`` if deactivated."""
    site = find_site(db, key, field)
    if not site.is_active:
        raise ApiError(422, "invalid", f"Site '{key}' has been deactivated.", field=field)
    return site


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


def end_sessions(db: Session, *, user_id: int | None = None, worker_id: int | None = None) -> None:
    """Revoke every live session of a user or a worker (no commit)."""
    owner = AuthSession.user_id == user_id if user_id is not None else (
        AuthSession.worker_id == worker_id
    )
    db.execute(
        update(AuthSession)
        .where(owner, AuthSession.revoked_at.is_(None))
        .values(revoked_at=func.now())
    )


# ---------------------------------------------------------------------------
# Paging (docs/admin-api.md §2)
# ---------------------------------------------------------------------------


def encode_cursor(value: Any) -> str:
    raw = json.dumps(value, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_cursor(cursor: str) -> Any:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        return json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))
    except (ValueError, UnicodeDecodeError):
        raise ApiError(422, "invalid", "That page cursor isn't valid.", field="cursor") from None


def page(items: list[Any], next_cursor: str | None, total: int) -> dict[str, Any]:
    return {"items": items, "next_cursor": next_cursor, "total": total}

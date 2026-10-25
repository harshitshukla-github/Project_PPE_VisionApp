"""Runtime settings: the few values an admin can change without a restart.

Each setting has a default from ``.env`` (via :mod:`src.settings`). A value
saved from the dashboard lives in the ``app_settings`` table and overrides
the default until it is cleared (docs/admin-api.md §6).

The inference path reads settings through :func:`get`, so a change reaches
every check within ``PPB_MATRIX_CACHE_S`` seconds, the same cache period as
the job matrix. Without ``DATABASE_URL`` there are no overrides and the
``.env`` values apply.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Literal

from loguru import logger

from src.settings import settings


@dataclass(frozen=True)
class SettingDef:
    """What a runtime setting is, and how the dashboard should edit it."""

    name: str
    type: Literal["number", "boolean", "choice"]
    env_attr: str
    description: str
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: tuple[str, ...] | None = None
    # The dashboard shows the old and new value and asks before saving.
    confirm: bool = False


DEFINITIONS: tuple[SettingDef, ...] = (
    SettingDef(
        "confidence_threshold", "number", "ppb_confidence_threshold",
        "Minimum detector confidence for a detection to count.",
        min=0.25, max=0.80, step=0.05, confirm=True,
    ),
    SettingDef(
        "gemini_verify", "boolean", "ppb_gemini_verify",
        "Whether Gemini looks at the photo when the detector can't find a required item.",
    ),
    SettingDef(
        "gemini_min_confidence", "number", "ppb_gemini_verify_min_confidence",
        "How sure Gemini must be for its answer to change a verdict.",
        min=0.50, max=1.00, step=0.05,
    ),
    SettingDef(
        "worker_auth", "choice", "ppb_worker_auth",
        "Whether a check needs a worker login (auth-api.md §5).",
        choices=("optional", "required"), confirm=True,
    ),
)
BY_NAME: dict[str, SettingDef] = {d.name: d for d in DEFINITIONS}


def default(name: str) -> Any:
    """The ``.env`` value of a runtime setting."""
    return getattr(settings, BY_NAME[name].env_attr)


def get(name: str) -> Any:
    """The value in force: the saved override if there is one, else the default."""
    found = overrides()
    return found[name] if name in found else default(name)


# Last overrides read from the database, and when (time.monotonic()).
_cache: dict[str, Any] | None = None
_cache_at = 0.0
_lock = threading.Lock()


def invalidate() -> None:
    """Forget the cached overrides, so the next read goes to the database."""
    global _cache, _cache_at
    with _lock:
        _cache, _cache_at = None, 0.0


def overrides() -> dict[str, Any]:
    """Saved overrides by name, cached like the job matrix.

    If the database can't be read, the last overrides read stay in force;
    if none were ever read, the ``.env`` defaults apply. A setting must
    never make a check fail.
    """
    global _cache, _cache_at
    if not settings.database_url:
        return {}
    ttl = settings.ppb_matrix_cache_s
    if _cache is not None and time.monotonic() - _cache_at < ttl:
        return _cache

    with _lock:
        now = time.monotonic()
        if _cache is not None and now - _cache_at < ttl:
            return _cache

        from sqlalchemy import select  # noqa: PLC0415 — only when a DB is used
        from sqlalchemy.orm import Session  # noqa: PLC0415

        from src.db.engine import get_engine  # noqa: PLC0415
        from src.db.models import AppSetting  # noqa: PLC0415

        try:
            with Session(get_engine()) as db:
                rows = db.scalars(select(AppSetting)).all()
                found = {row.name: row.value for row in rows if row.name in BY_NAME}
        except Exception as exc:  # noqa: BLE001 — a setting must never make a check fail
            reason = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
            logger.warning(f"Runtime settings refresh failed ({reason}); keeping what's in force.")
            _cache_at = now
            if _cache is None:
                _cache = {}
            return _cache

        _cache, _cache_at = found, now
        return found

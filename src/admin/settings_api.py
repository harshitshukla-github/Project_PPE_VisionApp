"""Runtime settings over the admin API (docs/admin-api.md §6). Admins only."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.admin import audit, runtime
from src.admin.runtime import BY_NAME, DEFINITIONS, SettingDef
from src.admin.views import iso
from src.api.errors import ApiError
from src.checks.retention import count_expired
from src.db.models import AppSetting

# The two photo retention periods (docs/checks-api.md §6, §7).
RETENTION = ("photo_review_days", "photo_training_days")


def _rows(db: Session) -> dict[str, AppSetting]:
    return {row.name: row for row in db.scalars(select(AppSetting)) if row.name in BY_NAME}


def list_settings(db: Session) -> list[dict[str, Any]]:
    """Each setting, describing itself so the dashboard can build its form."""
    rows = _rows(db)
    out = []
    for d in DEFINITIONS:
        row = rows.get(d.name)
        out.append({
            "name": d.name, "type": d.type, "min": d.min, "max": d.max, "step": d.step,
            "choices": list(d.choices) if d.choices else None, "confirm": d.confirm,
            "description": d.description,
            "value": row.value if row else runtime.default(d.name),
            "default": runtime.default(d.name),
            "overridden": row is not None,
            "updated_by": row.updated_by if row else None,
            "updated_at": iso(row.updated_at) if row else None,
        })
    return out


def _validate(d: SettingDef, value: Any) -> Any:
    if d.type == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ApiError(422, "invalid", f"{d.name} must be a number.", field=d.name)
        if not d.min <= value <= d.max:
            raise ApiError(422, "out_of_range",
                           f"{d.name} must be between {d.min:g} and {d.max:g}.", field=d.name)
        if d.integer:
            if value != int(value):
                raise ApiError(422, "invalid", f"{d.name} must be a whole number.", field=d.name)
            return int(value)
        return float(value)
    if d.type == "boolean":
        if not isinstance(value, bool):
            raise ApiError(422, "invalid", f"{d.name} must be true or false.", field=d.name)
        return value
    if value not in (d.choices or ()):
        raise ApiError(422, "invalid", f"{d.name} must be one of: {', '.join(d.choices)}.",
                       field=d.name)
    return value


def patch_settings(db: Session, patch: dict[str, Any], actor: str,
                   ip: str | None) -> list[dict[str, Any]]:
    """Apply an RFC 7396 merge patch, all or nothing.

    A name left out is unchanged, a value sets it, ``null`` clears the
    override. Every value is validated before anything is saved, so one bad
    value saves nothing. Returns the warnings for the dashboard.
    """
    for name in patch:
        if name not in BY_NAME:
            raise ApiError(422, "invalid", f"There is no setting '{name}'.", field=name)
    validated = {name: None if value is None else _validate(BY_NAME[name], value)
                 for name, value in patch.items()}

    rows = _rows(db)

    def after(name: str) -> Any:
        """The value in force once the patch is saved."""
        if name in validated:
            return validated[name] if validated[name] is not None else runtime.default(name)
        return rows[name].value if name in rows else runtime.default(name)

    # checks-api.md §7: saying yes to training can't mean a photo is kept for less time.
    retention_sent = [n for n in RETENTION if n in validated]
    if retention_sent and after("photo_training_days") < after("photo_review_days"):
        field = "photo_training_days" if "photo_training_days" in validated else "photo_review_days"
        raise ApiError(422, "out_of_range",
                       "Photos kept for training can't be kept for less time than photos kept "
                       f"for review ({after('photo_review_days')} days).", field=field)

    warnings: list[dict] = []
    changed = False
    retention_changed: list[str] = []
    for name, value in validated.items():
        row = rows.get(name)
        old_override = row.value if row else None
        if value == old_override:
            continue
        old_effective = old_override if row else runtime.default(name)
        new_effective = value if value is not None else runtime.default(name)
        if value is None:
            db.delete(row)
        elif row:
            row.value, row.updated_by = value, actor
        else:
            db.add(AppSetting(name=name, value=value, updated_by=actor))
        audit.record(db, actor=actor, action="settings.update", target=f"setting:{name}",
                     before={"value": old_override}, after={"value": value}, ip=ip)
        changed = True
        warnings.extend(_warnings(name, old_effective, new_effective))
        if name in RETENTION and new_effective != old_effective:
            retention_changed.append(name)
    if retention_changed:
        # checks-api.md §7: say what the next daily run will delete under the new periods.
        review = after("photo_review_days")
        due = count_expired(db, review, max(after("photo_training_days"), review))
        photos = "no photos" if due == 0 else f"{due} photo{'' if due == 1 else 's'}"
        warnings.extend({"code": "retention_changed", "setting": name, "count": due,
                         "message": f"The next daily run will delete {photos} past the new "
                                    "period."} for name in retention_changed)
    if changed:
        db.commit()
        runtime.invalidate()
    return warnings


def _warnings(name: str, old: Any, new: Any) -> list[dict]:
    if name == "worker_auth" and new == "required" and old != "required":
        return [{"code": "worker_auth_required", "setting": name,
                 "message": "Checks from an app without login will now be refused."}]
    if name == "gemini_verify" and old and not new:
        return [{"code": "gemini_verify_switched_off", "setting": name,
                 "message": "Gemini verification is off: anything the detector misses stays "
                            "\"Needs review\"."}]
    if name == "confidence_threshold" and new != old:
        return [{"code": "threshold_changed", "setting": name,
                 "message": f"Every verdict from now on uses a detector threshold of {new:g}."}]
    return []

"""Exporting photos for training: only those the worker agreed to (docs/checks-api.md §5).

From the CLI on the server only (§1). A photo is exported when its check
carries training consent, which a withdrawal clears on every earlier check,
and when it isn't past its retention period. The export says nothing about
who is in a photo: no worker ID, name or review note, only what training
needs.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import not_, select
from sqlalchemy.orm import Session

from src.admin import audit
from src.admin.views import iso
from src.checks.photos import PhotoStore, photo_store
from src.checks.retention import expired, periods
from src.db.models import Check

EXPORT_ACTION = "photos.export"


def consented(db: Session) -> list[Check]:
    """The checks whose photo may be used for training now."""
    review, training = periods()
    now = datetime.now(timezone.utc)
    return list(db.scalars(
        select(Check)
        .where(Check.photo_state == "kept", Check.training_consent.is_(True),
               not_(expired(review, training, now)))
        .order_by(Check.id)
    ))


def export_consented(db: Session, folder: Path, store: PhotoStore | None = None) -> int:
    """Copy every consented photo into ``folder``, with a manifest; return how many.

    Writes ``images/<check id>.<ext>`` and ``manifest.jsonl``, one line per
    photo, and a ``photos.export`` audit entry. ``folder`` must be new or
    empty, so an export never mixes with an older one.
    """
    folder = folder.resolve()
    if folder.exists() and (not folder.is_dir() or any(folder.iterdir())):
        raise ValueError(f"{folder} isn't an empty folder. Choose a new one.")
    store = store or photo_store()
    checks = consented(db)
    images = folder / "images"
    images.mkdir(parents=True, exist_ok=True)
    copied = 0
    try:
        with (folder / "manifest.jsonl").open("w", encoding="utf-8") as manifest:
            for check in checks:
                _copy(check, store, images, manifest)
                copied += 1
    finally:
        # Photos that left the store are on record, even if the export stopped part way.
        audit.record(db, actor=audit.CLI_ACTOR, action=EXPORT_ACTION, target="photos",
                     after={"exported": copied, "folder": str(folder)})
        db.commit()
    return copied


def _copy(check: Check, store: PhotoStore, images: Path, manifest) -> None:
    name = f"{check.id}{Path(check.photo_key).suffix}"
    (images / name).write_bytes(store.read(check.photo_key))
    manifest.write(json.dumps({
        "check_id": check.id,
        "image": f"images/{name}",
        "at": iso(check.at),
        "job": check.job_key,
        "required": check.required,
        "pending": check.pending,
        "verdict": check.verdict,
        "verdict_source": check.verdict_source,
        "review": check.review_outcome,
        "model_used": check.model_used,
        "width": check.photo_width,
        "height": check.photo_height,
        # In the photo as shown upright: its EXIF orientation applied.
        "detections": check.detections,
    }) + "\n")

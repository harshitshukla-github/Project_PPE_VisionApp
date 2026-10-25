"""Per-class accuracy saved next to a model's weights (docs/admin-api.md §4).

Accuracy belongs to a model, not to an item, so it travels with the weights:
``models/yolo11l_ppe_best.pt`` → ``models/yolo11l_ppe_best.metrics.json``.
Training writes the file; ``GET /admin/items`` reads it for the loaded model.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from loguru import logger


def metrics_path(weights_path: Path) -> Path:
    return weights_path.with_suffix(".metrics.json")


def write_metrics(weights_path: Path, model_name: str, ap50: dict[str, float],
                  source: str) -> Path:
    """Save AP50 per class next to the weights.

    Args:
        weights_path: The ``.pt`` file the figures describe.
        model_name: e.g. ``yolo11l``.
        ap50: Class name → AP50 (0–1). Classes the model can't detect are left out.
        source: Where the figures come from, e.g. ``test split, 2026-10-05``.

    Returns:
        The path written.
    """
    path = metrics_path(weights_path)
    path.write_text(json.dumps({
        "model": model_name,
        "source": source,
        "written_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "ap50": {name: round(float(value), 4) for name, value in sorted(ap50.items())},
    }, indent=2) + "\n", encoding="utf-8")
    return path


def load_ap50(weights_path: Path) -> dict[str, float]:
    """AP50 per class for a model, or ``{}`` if it has no (readable) metrics file."""
    path = metrics_path(weights_path)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}
    return _read(str(path), mtime)


@lru_cache(maxsize=8)
def _read(path: str, mtime: float) -> dict[str, float]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return {str(k): float(v) for k, v in data.get("ap50", {}).items()}
    except (OSError, ValueError, AttributeError, TypeError) as exc:
        logger.warning(f"Unreadable model metrics file {path}: {exc}")
        return {}

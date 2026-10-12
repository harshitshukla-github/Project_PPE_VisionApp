"""Class label remapper for the PPB Vision Module.

Reads each dataset's downloaded annotations (YOLO TXT format) and rewrites
them using our unified 12-class schema defined in ``configs/class_maps.yaml``.

- Source class IDs are resolved via each dataset's ``data.yaml``.
- Classes marked ``null`` in the map are discarded (annotation line removed).
- Images whose every annotation is discarded become background images
  (kept with an empty .txt file).

Usage::

    uv run python -m src.data.remapper
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path

import yaml
from loguru import logger

from src.settings import settings


# ---------------------------------------------------------------------------
# Windows MAX_PATH workaround
# ---------------------------------------------------------------------------

def _safe(path: Path) -> str:
    """Return an extended-length path string on Windows to bypass MAX_PATH (260 chars).

    Roboflow dataset filenames can exceed 260 characters. Prefixing the
    resolved absolute path with ``\\\\?\\`` tells the Windows API to skip
    the length limit.  On other platforms the plain string is returned.

    Args:
        path: Absolute or relative path to convert.

    Returns:
        Path string safe for use in ``open()``, ``shutil``, and ``os`` calls.
    """
    if sys.platform == "win32":
        return "\\\\?\\" + str(path.resolve())
    return str(path.resolve())


# ---------------------------------------------------------------------------
# Target schema (our 12 classes)
# ---------------------------------------------------------------------------

TARGET_CLASSES: dict[int, str] = {
    0: "hardhat",
    1: "no_hardhat",
    2: "safety_vest",
    3: "no_vest",
    4: "goggles",
    5: "mask",
    6: "gloves",
    7: "boots",
    8: "ear_protection",
    9: "face_shield",
    10: "harness",
    11: "person",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_class_maps() -> dict[str, dict[str, int | None]]:
    """Load the class remapping config from ``configs/class_maps.yaml``.

    Returns:
        Mapping of dataset_key → {source_class_name: target_class_id | None}.
    """
    map_path = settings.configs_dir / "class_maps.yaml"
    with map_path.open() as f:
        raw = yaml.safe_load(f)
    return raw["datasets"]


def _find_dataset_root(dataset_dir: Path) -> Path:
    """Resolve the actual directory containing the dataset splits.

    The Roboflow SDK sometimes creates a project-named subfolder inside
    the download location (e.g. ``data/raw/ppe_combined/<project-slug>/``).
    This function walks one level deep to find where ``train/`` or
    ``valid/`` actually lives.

    Args:
        dataset_dir: The directory passed as ``location`` to the SDK.

    Returns:
        The resolved dataset root (may be ``dataset_dir`` itself, or a
        single subdirectory inside it).
    """
    # Direct layout — splits are right here
    if (dataset_dir / "train").exists() or (dataset_dir / "valid").exists():
        return dataset_dir

    # Roboflow created a project-named subfolder — find it
    for subdir in sorted(dataset_dir.iterdir()):
        if subdir.is_dir() and (
            (subdir / "train").exists() or (subdir / "valid").exists()
        ):
            logger.debug(f"Dataset root resolved to subfolder: {subdir.name}")
            return subdir

    # Nothing found — return as-is and let the caller surface a clear error
    return dataset_dir


def _load_source_classes(dataset_dir: Path) -> dict[int, str]:
    """Read class names from a downloaded dataset's ``data.yaml``.

    Handles the case where the Roboflow SDK places ``data.yaml`` inside a
    project-named subdirectory rather than directly in ``dataset_dir``.

    Args:
        dataset_dir: Root directory of the downloaded dataset (may contain
            a Roboflow-created subfolder).

    Returns:
        Mapping of source class ID → source class name.

    Raises:
        FileNotFoundError: If ``data.yaml`` is not found in the dataset dir
            or any immediate subdirectory.
        KeyError: If ``names`` key is missing from the YAML.
    """
    root = _find_dataset_root(dataset_dir)
    yaml_path = root / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"data.yaml not found in {dataset_dir} or any immediate subdirectory.\n"
            "  Ensure the dataset downloaded successfully before running remapper."
        )

    with yaml_path.open() as f:
        data = yaml.safe_load(f)

    names = data.get("names", {})
    # Roboflow data.yaml can be a list or a dict
    if isinstance(names, list):
        return {i: name for i, name in enumerate(names)}
    return {int(k): v for k, v in names.items()}


def _normalize_class_name(name: str) -> str:
    """Normalise a class name for tolerant matching.

    Lowercases and collapses any run of spaces, hyphens, or underscores into a
    single space. This makes ``Safety-Helmet``, ``safety helmet``, and
    ``Safety_Helmet`` all equivalent.

    Dataset authors spell the same class a dozen ways, and an exact-match miss
    is silent-but-destructive here: the class is discarded, and every image
    that contained it becomes a background image teaching the model that the
    object is *not* there. That is strictly worse than not having the dataset.

    Args:
        name: Raw class name from a dataset's ``data.yaml``.

    Returns:
        Normalised lookup key.
    """
    return re.sub(r"[\s_-]+", " ", str(name).strip().lower())


def _build_id_map(
    source_classes: dict[int, str],
    class_map: dict[str, int | None],
) -> dict[int, int | None]:
    """Build a source_class_id → target_class_id mapping.

    Matching is exact first, then falls back to a normalised comparison
    (case-, hyphen-, and underscore-insensitive). The exact pass runs first so
    existing behaviour is preserved verbatim; normalisation only rescues names
    that would otherwise be discarded.

    Args:
        source_classes: Source dataset class IDs and names.
        class_map: Name-level mapping from class_maps.yaml.

    Returns:
        Mapping of source ID → target ID (None means discard).
    """
    normalized: dict[str, int | None] = {}
    for key, target in class_map.items():
        norm_key = _normalize_class_name(key)
        if norm_key in normalized and normalized[norm_key] != target:
            logger.warning(
                f"  Class map ambiguity: '{key}' normalises to '{norm_key}', "
                f"which already maps to {normalized[norm_key]} (not {target}). "
                "Keeping the first mapping."
            )
            continue
        normalized[norm_key] = target

    id_map: dict[int, int | None] = {}
    for src_id, src_name in source_classes.items():
        if src_name in class_map:
            id_map[src_id] = class_map[src_name]
            continue

        norm_name = _normalize_class_name(src_name)
        if norm_name in normalized:
            id_map[src_id] = normalized[norm_name]
            logger.info(
                f"  Source class '{src_name}' (id={src_id}) matched as "
                f"'{norm_name}' → {normalized[norm_name]}"
            )
            continue

        logger.warning(
            f"  Source class '{src_name}' (id={src_id}) not in class map → discarding. "
            f"Add '{norm_name}' to the class map if this class is wanted."
        )
        id_map[src_id] = None
    return id_map


def polygon_to_bbox(coords: list[float]) -> list[float]:
    """Convert a YOLO segmentation polygon to a normalised bounding box.

    Args:
        coords: Flat ``[x1, y1, x2, y2, ...]`` normalised polygon points.

    Returns:
        ``[cx, cy, w, h]`` clamped to ``[0, 1]``.
    """
    xs = coords[0::2]
    ys = coords[1::2]
    x_min, x_max = max(0.0, min(xs)), min(1.0, max(xs))
    y_min, y_max = max(0.0, min(ys)), min(1.0, max(ys))
    return [
        (x_min + x_max) / 2,
        (y_min + y_max) / 2,
        x_max - x_min,
        y_max - y_min,
    ]


def normalise_label_line(parts: list[str]) -> list[str] | None:
    """Normalise one annotation row to detection format ``class cx cy w h``.

    Some sources ship segmentation polygons (``class x1 y1 x2 y2 ...``) mixed
    into otherwise box-shaped label files. Ultralytics decides a file is
    segmentation if *any* row has more than 6 fields, and then reinterprets
    the 5-field box rows in that same file as 2-point polygons — silently
    producing garbage boxes. Converting polygons to boxes here keeps every
    file uniformly 5-field, so that ambiguity never arises.

    Args:
        parts: Whitespace-split fields of one annotation line.

    Returns:
        A 5-element ``[class, cx, cy, w, h]`` list, or None if the row is
        malformed beyond repair.
    """
    if len(parts) == 5:
        return parts
    # A polygon needs a class plus at least 3 (x, y) points.
    if len(parts) >= 7 and len(parts) % 2 == 1:
        try:
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            return None
        box = polygon_to_bbox(coords)
        return [parts[0], *(f"{v:.6f}" for v in box)]
    return None


def _remap_annotation_file(
    src_txt: Path,
    dst_txt: Path,
    id_map: dict[int, int | None],
) -> tuple[int, int]:
    """Remap a single YOLO TXT annotation file.

    Each line in YOLO format: ``<class_id> <cx> <cy> <w> <h>``

    Args:
        src_txt: Source annotation file.
        dst_txt: Destination annotation file.
        id_map: Source class ID → target class ID (None = discard).

    Returns:
        Tuple of (kept_annotations, discarded_annotations).
    """
    os.makedirs(_safe(dst_txt.parent), exist_ok=True)
    kept = 0
    discarded = 0
    output_lines: list[str] = []

    with open(_safe(src_txt)) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            src_class_id = int(parts[0])
            target_id = id_map.get(src_class_id)

            if target_id is None:
                discarded += 1
                continue

            # Collapse segmentation polygons to boxes so every row is 5-field.
            normalised = normalise_label_line(parts)
            if normalised is None:
                logger.warning(
                    f"  Unparseable annotation in {src_txt.name} "
                    f"({len(parts)} fields) → discarding"
                )
                discarded += 1
                continue

            normalised[0] = str(target_id)
            output_lines.append(" ".join(normalised))
            kept += 1

    with open(_safe(dst_txt), "w") as f:
        f.write("\n".join(output_lines))
        if output_lines:
            f.write("\n")

    return kept, discarded


def remap_dataset(
    dataset_key: str,
    class_maps: dict[str, dict[str, int | None]],
) -> dict[str, int]:
    """Remap all annotations in a single downloaded dataset.

    Reads from ``data/raw/<dataset_key>/``, writes to
    ``data/processed/<dataset_key>/``.

    Args:
        dataset_key: Key matching a dataset in ``configs/class_maps.yaml``.
        class_maps: Full class maps dict loaded from YAML.

    Returns:
        Statistics dict with keys: images, kept, discarded, background.
    """
    src_root = settings.data_raw_dir / dataset_key
    dst_root = settings.data_processed_dir / dataset_key

    if not src_root.exists():
        logger.error(f"[{dataset_key}] Raw dataset not found at {src_root}. Run downloader first.")
        return {}

    class_map = class_maps.get(dataset_key)
    if not class_map:
        logger.error(f"[{dataset_key}] No class map entry found in class_maps.yaml.")
        return {}

    # Roboflow SDK may place files one folder deeper — resolve the real root
    actual_root = _find_dataset_root(src_root)
    if actual_root != src_root:
        logger.debug(f"[{dataset_key}] Using dataset subfolder: {actual_root.name}")

    # Load source class names from the dataset's own data.yaml
    source_classes = _load_source_classes(src_root)
    id_map = _build_id_map(source_classes, class_map)

    logger.info(f"[{dataset_key}] Source classes: {source_classes}")
    logger.info(f"[{dataset_key}] ID remapping: {id_map}")

    stats = {"images": 0, "kept": 0, "discarded": 0, "background": 0}

    # Walk all splits (train / valid / test) inside the dataset
    # Use actual_root in case Roboflow created a project-named subfolder
    for split in ("train", "valid", "test"):
        img_dir = actual_root / split / "images"
        lbl_dir = actual_root / split / "labels"

        if not img_dir.exists():
            continue

        dst_img_dir = dst_root / split / "images"
        dst_lbl_dir = dst_root / split / "labels"
        os.makedirs(_safe(dst_img_dir), exist_ok=True)
        os.makedirs(_safe(dst_lbl_dir), exist_ok=True)

        image_files = list(img_dir.glob("*.[jJpPbB][pPnNmM][gGpPpP]*"))
        logger.info(f"[{dataset_key}/{split}] Processing {len(image_files)} images...")

        for img_path in image_files:
            stats["images"] += 1
            txt_path = lbl_dir / (img_path.stem + ".txt")
            dst_img = dst_img_dir / img_path.name
            dst_txt = dst_lbl_dir / (img_path.stem + ".txt")

            # Copy image using safe extended-length paths (Windows MAX_PATH bypass)
            shutil.copy2(_safe(img_path), _safe(dst_img))

            if not txt_path.exists():
                # No annotation → background image, create empty label
                open(_safe(dst_txt), "a").close()
                stats["background"] += 1
                continue

            kept, discarded = _remap_annotation_file(txt_path, dst_txt, id_map)
            stats["kept"] += kept
            stats["discarded"] += discarded

            if kept == 0:
                stats["background"] += 1

    logger.success(
        f"[{dataset_key}] Done — "
        f"images={stats['images']}, "
        f"annotations kept={stats['kept']}, "
        f"discarded={stats['discarded']}, "
        f"background images={stats['background']}"
    )
    return stats


def remap_all(dataset_keys: list[str] | None = None) -> None:
    """Remap all (or a subset of) datasets.

    Args:
        dataset_keys: Optional list of dataset keys to process.
            If None, all keys in class_maps.yaml are processed.
    """
    class_maps = _load_class_maps()
    keys = dataset_keys or list(class_maps.keys())

    # A dataset is ready only if its data.yaml has been extracted. A directory
    # holding just a partially-downloaded roboflow.zip is non-empty but not yet
    # remappable, so "non-empty" is too weak a check to distinguish "still
    # downloading" from "ready".
    available = []
    for key in keys:
        raw_dir = settings.data_raw_dir / key
        if not raw_dir.exists() or not any(raw_dir.iterdir()):
            logger.warning(f"[{key}] Raw directory empty or missing — skipping remap.")
            continue
        if not (_find_dataset_root(raw_dir) / "data.yaml").exists():
            logger.warning(
                f"[{key}] No data.yaml yet — the download may still be running or "
                "the archive was never extracted. Skipping remap."
            )
            continue
        available.append(key)

    if not available:
        logger.error("No downloaded datasets found. Run `data download` first.")
        return

    logger.info(f"Remapping {len(available)} dataset(s): {available}")
    total_stats: dict[str, int] = {"images": 0, "kept": 0, "discarded": 0, "background": 0}
    failed: list[str] = []

    for key in available:
        # One broken dataset must not discard the work done on the others —
        # these runs take tens of minutes over tens of thousands of images.
        try:
            stats = remap_dataset(key, class_maps)
        except Exception as exc:  # noqa: BLE001 — isolate per-dataset failures
            logger.error(f"[{key}] Remap failed: {exc}")
            failed.append(key)
            continue
        for k in total_stats:
            total_stats[k] += stats.get(k, 0)

    logger.info("─" * 50)
    logger.info("Remapping summary (all datasets combined):")
    logger.info(f"  Datasets remapped      : {len(available) - len(failed)}/{len(available)}")
    logger.info(f"  Total images processed : {total_stats['images']:,}")
    logger.info(f"  Annotations kept       : {total_stats['kept']:,}")
    logger.info(f"  Annotations discarded  : {total_stats['discarded']:,}")
    logger.info(f"  Background images      : {total_stats['background']:,}")
    logger.info(f"  Output dir             : {settings.data_processed_dir.resolve()}")
    for key in failed:
        logger.error(f"  ✗ {key} — see the error above; re-run after fixing it.")
    logger.info("─" * 50)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
        colorize=True,
    )

    parser = argparse.ArgumentParser(description="Remap PPE dataset class labels")
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=None,
        help="One or more dataset keys to remap (default: all)",
    )
    args = parser.parse_args()
    remap_all(dataset_keys=args.dataset)

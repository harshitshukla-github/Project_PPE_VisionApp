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

import shutil
import sys
from pathlib import Path

import yaml
from loguru import logger

from src.settings import settings


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


def _load_source_classes(dataset_dir: Path) -> dict[int, str]:
    """Read class names from a downloaded dataset's ``data.yaml``.

    Args:
        dataset_dir: Root directory of the downloaded dataset.

    Returns:
        Mapping of source class ID → source class name.

    Raises:
        FileNotFoundError: If ``data.yaml`` is not found in the dataset dir.
        KeyError: If ``names`` key is missing from the YAML.
    """
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found in {dataset_dir}")

    with yaml_path.open() as f:
        data = yaml.safe_load(f)

    names = data.get("names", {})
    # Roboflow data.yaml can be a list or a dict
    if isinstance(names, list):
        return {i: name for i, name in enumerate(names)}
    return {int(k): v for k, v in names.items()}


def _build_id_map(
    source_classes: dict[int, str],
    class_map: dict[str, int | None],
) -> dict[int, int | None]:
    """Build a source_class_id → target_class_id mapping.

    Args:
        source_classes: Source dataset class IDs and names.
        class_map: Name-level mapping from class_maps.yaml.

    Returns:
        Mapping of source ID → target ID (None means discard).
    """
    id_map: dict[int, int | None] = {}
    for src_id, src_name in source_classes.items():
        if src_name in class_map:
            id_map[src_id] = class_map[src_name]
        else:
            logger.warning(
                f"  Source class '{src_name}' (id={src_id}) not in class map → discarding"
            )
            id_map[src_id] = None
    return id_map


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
    dst_txt.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    discarded = 0
    output_lines: list[str] = []

    with src_txt.open() as f:
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

            parts[0] = str(target_id)
            output_lines.append(" ".join(parts))
            kept += 1

    with dst_txt.open("w") as f:
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

    # Load source class names from the dataset's own data.yaml
    source_classes = _load_source_classes(src_root)
    id_map = _build_id_map(source_classes, class_map)

    logger.info(f"[{dataset_key}] Source classes: {source_classes}")
    logger.info(f"[{dataset_key}] ID remapping: {id_map}")

    stats = {"images": 0, "kept": 0, "discarded": 0, "background": 0}

    # Walk all splits (train / valid / test) inside the dataset
    for split in ("train", "valid", "test"):
        img_dir = src_root / split / "images"
        lbl_dir = src_root / split / "labels"

        if not img_dir.exists():
            continue

        dst_img_dir = dst_root / split / "images"
        dst_lbl_dir = dst_root / split / "labels"
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        image_files = list(img_dir.glob("*.[jJpPbB][pPnNmM][gGpPpP]*"))
        logger.info(f"[{dataset_key}/{split}] Processing {len(image_files)} images...")

        for img_path in image_files:
            stats["images"] += 1
            txt_path = lbl_dir / (img_path.stem + ".txt")
            dst_img = dst_img_dir / img_path.name
            dst_txt = dst_lbl_dir / (img_path.stem + ".txt")

            # Copy image
            shutil.copy2(img_path, dst_img)

            if not txt_path.exists():
                # No annotation → background image, create empty label
                dst_txt.touch()
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

    logger.info(f"Remapping {len(keys)} dataset(s): {keys}")
    total_stats: dict[str, int] = {"images": 0, "kept": 0, "discarded": 0, "background": 0}

    for key in keys:
        stats = remap_dataset(key, class_maps)
        for k in total_stats:
            total_stats[k] += stats.get(k, 0)

    logger.info("─" * 50)
    logger.info("Remapping summary (all datasets combined):")
    logger.info(f"  Total images processed : {total_stats['images']:,}")
    logger.info(f"  Annotations kept       : {total_stats['kept']:,}")
    logger.info(f"  Annotations discarded  : {total_stats['discarded']:,}")
    logger.info(f"  Background images      : {total_stats['background']:,}")
    logger.info(f"  Output dir             : {settings.data_processed_dir.resolve()}")
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

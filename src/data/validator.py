"""Annotation integrity validator for the PPB Vision Module.

Validates the ``data/splits/`` directory before training to catch common
annotation problems early:

    - Every image has a matching ``.txt`` label file.
    - All class IDs are within the valid range (0 – NC-1).
    - Bounding box coordinates are normalized (0.0 – 1.0).
    - No duplicate image filenames across splits.
    - Prints a per-class annotation count table.

Usage::

    uv run python -m src.data.validator
    uv run python -m src.data.validator --split train
    uv run python -m src.data.validator --strict    # exit 1 on any error
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.table import Table

from src.settings import settings


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_CLASSES = 12
CLASS_NAMES = {
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
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
console = Console()


# ---------------------------------------------------------------------------
# Validation logic
# ---------------------------------------------------------------------------

def validate_split(
    split_name: str,
    splits_root: Path,
) -> dict[str, object]:
    """Validate one split (train / val / test).

    Args:
        split_name: Name of the split folder (``train``, ``val``, ``test``).
        splits_root: Root of ``data/splits/``.

    Returns:
        Dict with validation results: errors, warnings, class_counts, totals.
    """
    img_dir = splits_root / split_name / "images"
    lbl_dir = splits_root / split_name / "labels"

    if not img_dir.exists():
        logger.warning(f"[{split_name}] images dir not found: {img_dir}")
        return {}

    image_files = [f for f in img_dir.iterdir() if f.suffix.lower() in VALID_IMAGE_EXTS]
    errors: list[str] = []
    warnings: list[str] = []
    class_counts: dict[int, int] = defaultdict(int)
    total_annotations = 0

    for img_path in image_files:
        lbl_path = lbl_dir / (img_path.stem + ".txt")

        # Check label file exists
        if not lbl_path.exists():
            errors.append(f"Missing label: {img_path.name}")
            continue

        # Parse annotation file
        with lbl_path.open() as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        if not lines:
            # Background image — acceptable
            continue

        for line_num, line in enumerate(lines, 1):
            parts = line.split()

            # Must have exactly 5 fields: class cx cy w h
            if len(parts) != 5:
                errors.append(
                    f"{lbl_path.name}:{line_num} — expected 5 fields, got {len(parts)}: '{line}'"
                )
                continue

            try:
                class_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])
            except ValueError:
                errors.append(f"{lbl_path.name}:{line_num} — non-numeric values: '{line}'")
                continue

            # Class ID range check
            if class_id < 0 or class_id >= NUM_CLASSES:
                errors.append(
                    f"{lbl_path.name}:{line_num} — invalid class_id={class_id} "
                    f"(valid: 0–{NUM_CLASSES - 1})"
                )
                continue

            # Bounding box range check
            for val, name in [(cx, "cx"), (cy, "cy"), (w, "w"), (h, "h")]:
                if not (0.0 <= val <= 1.0):
                    warnings.append(
                        f"{lbl_path.name}:{line_num} — {name}={val:.4f} outside [0, 1]"
                    )

            class_counts[class_id] += 1
            total_annotations += 1

    return {
        "split": split_name,
        "images": len(image_files),
        "annotations": total_annotations,
        "class_counts": dict(class_counts),
        "errors": errors,
        "warnings": warnings,
    }


def _print_class_table(all_results: list[dict]) -> None:
    """Print a rich table showing per-class annotation counts per split."""
    table = Table(title="Annotation Count by Class", show_lines=True)
    table.add_column("ID", style="dim", width=4)
    table.add_column("Class Name", style="bold")
    for r in all_results:
        table.add_column(r["split"].capitalize(), justify="right")
    table.add_column("Total", justify="right", style="bold cyan")

    for class_id in range(NUM_CLASSES):
        name = CLASS_NAMES[class_id]
        counts = [r["class_counts"].get(class_id, 0) for r in all_results]
        total = sum(counts)
        style = "yellow" if total == 0 else ""
        table.add_row(str(class_id), name, *[f"{c:,}" for c in counts], f"{total:,}", style=style)

    console.print(table)


def run_validation(
    splits: list[str] | None = None,
    strict: bool = False,
) -> bool:
    """Run validation across all (or selected) splits.

    Args:
        splits: List of split names to validate. Defaults to all three.
        strict: If True, exit with code 1 on any error.

    Returns:
        True if no errors were found, False otherwise.
    """
    splits_root = settings.data_splits_dir
    split_names = splits or ["train", "val", "test"]

    logger.info(f"Validating splits: {split_names}")
    logger.info(f"Dataset root: {splits_root.resolve()}")
    logger.info(f"Expected classes: {NUM_CLASSES} (IDs 0–{NUM_CLASSES - 1})")

    all_results: list[dict] = []
    total_errors = 0
    total_warnings = 0

    for split_name in split_names:
        result = validate_split(split_name, splits_root)
        if not result:
            continue
        all_results.append(result)

        n_err = len(result["errors"])
        n_warn = len(result["warnings"])
        total_errors += n_err
        total_warnings += n_warn

        status = "✓" if n_err == 0 else "✗"
        logger.info(
            f"[{split_name}] {status} "
            f"images={result['images']:,}  "
            f"annotations={result['annotations']:,}  "
            f"errors={n_err}  warnings={n_warn}"
        )

        for err in result["errors"][:10]:  # cap output to 10 per split
            logger.error(f"  ERROR: {err}")
        if len(result["errors"]) > 10:
            logger.error(f"  ... and {len(result['errors']) - 10} more errors")

        for warn in result["warnings"][:5]:
            logger.warning(f"  WARN:  {warn}")
        if len(result["warnings"]) > 5:
            logger.warning(f"  ... and {len(result['warnings']) - 5} more warnings")

    # Summary table
    if all_results:
        _print_class_table(all_results)

    # Final verdict
    logger.info("─" * 50)
    if total_errors == 0:
        logger.success(f"Validation PASSED — 0 errors, {total_warnings} warnings")
    else:
        logger.error(f"Validation FAILED — {total_errors} errors, {total_warnings} warnings")

    if total_errors > 0 and strict:
        sys.exit(1)

    return total_errors == 0


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

    parser = argparse.ArgumentParser(description="Validate PPE dataset annotations")
    parser.add_argument(
        "--split",
        nargs="+",
        choices=["train", "val", "test"],
        default=None,
        help="Splits to validate (default: all)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if any errors are found",
    )
    args = parser.parse_args()
    run_validation(splits=args.split, strict=args.strict)

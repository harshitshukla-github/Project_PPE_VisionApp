"""Train / val / test dataset splitter for the PPB Vision Module.

Collects all remapped images from ``data/processed/`` across all datasets,
shuffles them, and splits into:
    - train : 80%
    - val   : 10%
    - test  : 10%

Output is written to ``data/splits/`` in standard YOLO layout::

    data/splits/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/

Usage::

    uv run python -m src.data.splitter
    uv run python -m src.data.splitter --train 0.8 --val 0.1 --test 0.1 --seed 42
"""

from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path

from loguru import logger

from src.settings import settings


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def collect_pairs(processed_root: Path) -> list[tuple[Path, Path]]:
    """Collect all (image, label) pairs from the processed dataset directory.

    Args:
        processed_root: Root of ``data/processed/``.

    Returns:
        List of (image_path, label_path) tuples where both files exist.
    """
    pairs: list[tuple[Path, Path]] = []
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    for img_path in sorted(processed_root.rglob("images/*")):
        if img_path.suffix.lower() not in image_exts:
            continue
        lbl_path = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
        if lbl_path.exists():
            pairs.append((img_path, lbl_path))
        else:
            logger.warning(f"No label found for {img_path.name} — skipping")

    return pairs


def split_and_copy(
    pairs: list[tuple[Path, Path]],
    output_root: Path,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> dict[str, int]:
    """Shuffle, split, and copy image+label pairs to the output directory.

    Args:
        pairs: List of (image_path, label_path) tuples.
        output_root: Destination root (``data/splits/``).
        train_ratio: Fraction for training set (default 0.80).
        val_ratio: Fraction for validation set (default 0.10).
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys ``train``, ``val``, ``test`` and counts.
    """
    if not pairs:
        logger.error("No image/label pairs found. Run remapper first.")
        return {}

    random.seed(seed)
    shuffled = pairs.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }

    counts: dict[str, int] = {}

    for split_name, split_pairs in splits.items():
        img_out = output_root / split_name / "images"
        lbl_out = output_root / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        logger.info(f"[{split_name}] Copying {len(split_pairs)} pairs...")

        for img_src, lbl_src in split_pairs:
            shutil.copy2(img_src, img_out / img_src.name)
            shutil.copy2(lbl_src, lbl_out / lbl_src.name)

        counts[split_name] = len(split_pairs)
        logger.success(f"[{split_name}] {len(split_pairs):,} images → {img_out}")

    return counts


def run_split(
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> None:
    """Full pipeline: collect → shuffle → split → copy.

    Args:
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
        seed: Random seed.
    """
    test_ratio = round(1.0 - train_ratio - val_ratio, 6)
    if test_ratio < 0:
        raise ValueError("train_ratio + val_ratio must be <= 1.0")

    logger.info(f"Split ratios — train:{train_ratio} val:{val_ratio} test:{test_ratio}")
    logger.info(f"Source: {settings.data_processed_dir.resolve()}")
    logger.info(f"Output: {settings.data_splits_dir.resolve()}")

    pairs = collect_pairs(settings.data_processed_dir)
    if not pairs:
        return

    logger.info(f"Total image/label pairs found: {len(pairs):,}")

    counts = split_and_copy(
        pairs,
        output_root=settings.data_splits_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    logger.info("─" * 50)
    logger.info("Split complete:")
    for split_name, count in counts.items():
        pct = count / len(pairs) * 100
        logger.info(f"  {split_name:<6} : {count:>6,} images ({pct:.1f}%)")
    logger.info(f"  Total  : {len(pairs):>6,} images")
    logger.info("─" * 50)
    logger.info("Next step: uv run python -m src.data.validator")


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

    parser = argparse.ArgumentParser(description="Split PPE dataset into train/val/test")
    parser.add_argument("--train", type=float, default=0.80, help="Train fraction (default: 0.80)")
    parser.add_argument("--val",   type=float, default=0.10, help="Val fraction (default: 0.10)")
    parser.add_argument("--seed",  type=int,   default=42,   help="Random seed (default: 42)")
    args = parser.parse_args()

    run_split(train_ratio=args.train, val_ratio=args.val, seed=args.seed)

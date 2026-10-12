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

import json
import random
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from src.settings import settings

#: Filename of the provenance manifest written alongside the splits.
MANIFEST_NAME = "manifest.json"


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ImagePair:
    """One image and its label, tagged with the dataset it came from.

    Attributes:
        image: Path to the image under ``data/processed/``.
        label: Path to the matching YOLO ``.txt`` label.
        source: Dataset key, e.g. ``"kaggle_ppe"`` — the first path component
            beneath ``data/processed/``.
    """

    image: Path
    label: Path
    source: str


def collect_pairs(processed_root: Path) -> list[ImagePair]:
    """Collect all (image, label) pairs from the processed dataset directory.

    Each pair is tagged with its source dataset so provenance survives the
    shuffle. Source attribution is what makes per-source loss masking possible
    later: the merged datasets annotate different class vocabularies, so an
    image from a helmet-only source should not be treated as negative evidence
    for vests.

    Args:
        processed_root: Root of ``data/processed/``.

    Returns:
        List of :class:`ImagePair` where both files exist.
    """
    pairs: list[ImagePair] = []
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    for img_path in sorted(processed_root.rglob("images/*")):
        if img_path.suffix.lower() not in image_exts:
            continue
        lbl_path = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
        if not lbl_path.exists():
            logger.warning(f"No label found for {img_path.name} — skipping")
            continue

        try:
            source = img_path.relative_to(processed_root).parts[0]
        except (ValueError, IndexError):
            source = "unknown"

        pairs.append(ImagePair(image=img_path, label=lbl_path, source=source))

    return pairs


def _count_existing_images(output_root: Path) -> int:
    """Count images already present in the split directories.

    Args:
        output_root: Split root (``data/splits/``).

    Returns:
        Total image count across train/val/test.
    """
    total = 0
    for split_name in ("train", "val", "test"):
        img_dir = output_root / split_name / "images"
        if img_dir.is_dir():
            total += sum(1 for _ in img_dir.iterdir())
    return total


def _resolve_destination_names(
    split_pairs: list[ImagePair],
) -> tuple[list[tuple[ImagePair, str]], int]:
    """Assign a unique destination filename to every pair within one split.

    The splits are flat directories keyed by filename, so two datasets that
    happen to use the same filename would silently overwrite one another and
    the image would simply disappear from the run. Colliding names are
    prefixed with their source key instead; the manifest records both the
    original path and the final name, so nothing becomes untraceable.

    Args:
        split_pairs: Pairs assigned to a single split.

    Returns:
        Tuple of (list of (pair, destination filename), number renamed).
    """
    seen: set[str] = set()
    resolved: list[tuple[ImagePair, str]] = []
    renamed = 0

    for pair in split_pairs:
        name = pair.image.name
        if name in seen:
            name = f"{pair.source}__{pair.image.name}"
            renamed += 1
            # Vanishingly unlikely, but keep going until genuinely unique.
            suffix = 1
            while name in seen:
                name = f"{pair.source}__{suffix}__{pair.image.name}"
                suffix += 1
        seen.add(name)
        resolved.append((pair, name))

    return resolved, renamed


def split_and_copy(
    pairs: list[ImagePair],
    output_root: Path,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42,
    write_manifest: bool = True,
    clean: bool = False,
) -> dict[str, int]:
    """Shuffle, split, and copy image+label pairs to the output directory.

    Also writes ``manifest.json`` recording, for every copied image, which
    source dataset it came from and where it originally lived.

    Args:
        pairs: Collected :class:`ImagePair` list.
        output_root: Destination root (``data/splits/``).
        train_ratio: Fraction for training set (default 0.80).
        val_ratio: Fraction for validation set (default 0.10).
        seed: Random seed for reproducibility.
        write_manifest: Whether to emit the provenance manifest.
        clean: Delete any existing splits first. Without this, splitting into
            a non-empty directory is refused, because the result would be two
            runs mixed together with a manifest describing only one of them.

    Returns:
        Dict with keys ``train``, ``val``, ``test`` and counts, or an empty
        dict if the split was refused.
    """
    if not pairs:
        logger.error("No image/label pairs found. Run remapper first.")
        return {}

    existing = _count_existing_images(output_root)
    if existing:
        if not clean:
            logger.error(
                f"{output_root} already contains {existing:,} images. Splitting on "
                "top of them would mix two runs together, and the manifest would "
                "describe only the new files — silently misdescribing the rest.\n"
                "  Re-split from scratch:  uv run python cli.py data split --clean\n"
                "  (or move the existing data/splits/ aside first)"
            )
            return {}
        logger.warning(f"--clean: removing {existing:,} existing images from {output_root}")
        for split_name in ("train", "val", "test"):
            shutil.rmtree(output_root / split_name, ignore_errors=True)
        (output_root / MANIFEST_NAME).unlink(missing_ok=True)

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
    entries: dict[str, dict[str, str]] = {}
    per_source: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total_renamed = 0
    processed_root = settings.data_processed_dir

    for split_name, split_pairs in splits.items():
        img_out = output_root / split_name / "images"
        lbl_out = output_root / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        resolved, renamed = _resolve_destination_names(split_pairs)
        total_renamed += renamed
        logger.info(f"[{split_name}] Copying {len(resolved)} pairs...")

        for pair, dest_name in resolved:
            dest_stem = Path(dest_name).stem
            shutil.copy2(pair.image, img_out / dest_name)
            shutil.copy2(pair.label, lbl_out / f"{dest_stem}.txt")

            try:
                origin = str(pair.image.relative_to(processed_root)).replace("\\", "/")
            except ValueError:
                origin = str(pair.image).replace("\\", "/")

            entries[f"{split_name}/{dest_name}"] = {
                "source": pair.source,
                "source_path": origin,
            }
            per_source[pair.source][split_name] += 1

        counts[split_name] = len(resolved)
        logger.success(f"[{split_name}] {len(resolved):,} images → {img_out}")

    if total_renamed:
        logger.warning(
            f"{total_renamed:,} filename collision(s) between source datasets were "
            "renamed with a source prefix. Without this they would have silently "
            "overwritten each other — see manifest.json."
        )

    if write_manifest:
        _write_manifest(
            output_root=output_root,
            entries=entries,
            per_source=per_source,
            counts=counts,
            seed=seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            renamed=total_renamed,
        )

    return counts


def _write_manifest(
    output_root: Path,
    entries: dict[str, dict[str, str]],
    per_source: dict[str, dict[str, int]],
    counts: dict[str, int],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    renamed: int,
) -> Path:
    """Write the provenance manifest to ``data/splits/manifest.json``.

    Args:
        output_root: Split root the manifest describes.
        entries: Mapping of ``"<split>/<filename>"`` → source metadata.
        per_source: Per-source, per-split image counts.
        counts: Total images per split.
        seed: Shuffle seed used.
        train_ratio: Train fraction used.
        val_ratio: Val fraction used.
        renamed: Number of collision renames applied.

    Returns:
        Path to the written manifest.
    """
    sources = {
        src: {**dict(splits), "total": sum(splits.values())}
        for src, splits in sorted(per_source.items())
    }
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": round(1.0 - train_ratio - val_ratio, 6),
        },
        "counts": counts,
        "sources": sources,
        "renamed_on_collision": renamed,
        "images": entries,
    }

    path = output_root / MANIFEST_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    logger.success(f"Provenance manifest → {path}  ({len(entries):,} images)")
    return path


def load_manifest(splits_root: Path | None = None) -> dict:
    """Load the provenance manifest produced by the splitter.

    Training code needs this to know which source each image came from, and
    therefore which classes that source actually annotates.

    Args:
        splits_root: Split root. Defaults to ``settings.data_splits_dir``.

    Returns:
        Parsed manifest dict.

    Raises:
        FileNotFoundError: If no manifest exists — re-run the splitter.
    """
    root = splits_root or settings.data_splits_dir
    path = root / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"No provenance manifest at {path}.\n"
            "  Re-run: uv run python cli.py data split"
        )
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def run_split(
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42,
    clean: bool = False,
) -> None:
    """Full pipeline: collect → shuffle → split → copy → manifest.

    Args:
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
        seed: Random seed.
        clean: Wipe existing splits first. Required when ``data/splits/``
            already holds a previous run.
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

    by_source: dict[str, int] = defaultdict(int)
    for pair in pairs:
        by_source[pair.source] += 1
    logger.info("Source datasets going into this split:")
    for src, n in sorted(by_source.items(), key=lambda kv: -kv[1]):
        logger.info(f"  {src:<24} {n:>7,} images ({n / len(pairs) * 100:5.1f}%)")

    counts = split_and_copy(
        pairs,
        output_root=settings.data_splits_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        clean=clean,
    )
    if not counts:
        return

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

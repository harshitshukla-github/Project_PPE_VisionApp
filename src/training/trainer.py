"""Unified YOLO trainer for the PPB Vision Module.

Supports YOLO26 and YOLO11 variants through a single ``train()`` entry
point. Model selection is entirely runtime — pass the model name and the
trainer loads the corresponding hyperparameter config, registers callbacks,
and launches ultralytics training.

After training, the best checkpoint is copied to ``models/`` for easy
retrieval by the inference pipeline.

Usage — Python API::

    from src.training.trainer import train

    best_pt = train("yolo26l")                        # use default config
    best_pt = train("yolo11n", epochs=50, batch=64)   # override specific params
    best_pt = train("yolo26l", use_mlflow=True)        # enable MLflow logging

Usage — CLI::

    uv run python -m src.training.trainer --model yolo26l
    uv run python -m src.training.trainer --model yolo11n --epochs 50 --device cpu
    uv run python -m src.training.trainer --model yolo26l --mlflow
"""

from __future__ import annotations

import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

# ── Windows virtual-address-space fragmentation mitigation ───────────────────
# Must happen before torch / CUDA is first imported anywhere in the process.
# CUDA's default allocator reserves one large contiguous VA region; on Windows
# that fragments the process VA space so badly that even a 1.2 MB cv2 buffer
# allocation fails with OutOfMemoryError despite plenty of physical RAM.
#
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#       → allocator grows in small expandable segments instead of one giant slab
#   OMP_NUM_THREADS=1
#       → caps OpenMP thread-pool VA consumption (each thread reserves a stack)
#
# setdefault() is used so an explicit env var from the shell always wins.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OMP_NUM_THREADS", "1")
# Defer CUDA kernel/module loading until first use — reduces upfront VA mapping
# that fragments the process address space on Windows.
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
# MLflow ≥2.x dropped the filesystem backend by default.  Ultralytics registers
# its own MLflow callback which fires even when use_mlflow=False.  Point to a
# local SQLite DB so the callback has a valid store without needing a server.
# An explicit MLFLOW_TRACKING_URI in .env / the shell always takes precedence.
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///runs/mlflow.db")

from loguru import logger

from src.settings import settings
from src.training.callbacks import build_loguru_callbacks, build_mlflow_callbacks
from src.training.hyperparams import TrainingConfig, config_path_for, load_training_config

# Models directory (gitignored) — stores the best weights per run
MODELS_DIR = Path("models")


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

def _check_data_yaml() -> Path:
    """Verify the dataset YAML exists before training starts.

    Returns:
        Resolved path to ``configs/data.yaml``.

    Raises:
        FileNotFoundError: If ``data.yaml`` is missing — the dataset
            pipeline (download → remap → split) hasn't been run yet.
    """
    data_yaml = settings.configs_dir / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Dataset config not found: {data_yaml}\n"
            "  Run the data pipeline first:\n"
            "    uv run python -m src.data.downloader\n"
            "    uv run python -m src.data.remapper\n"
            "    uv run python -m src.data.splitter\n"
            "    uv run python -m src.data.validator"
        )

    splits_dir = settings.data_splits_dir
    for split in ("train", "val"):
        split_images = splits_dir / split / "images"
        if not split_images.exists() or not any(split_images.iterdir()):
            raise FileNotFoundError(
                f"Split directory empty or missing: {split_images}\n"
                "  Run: uv run python -m src.data.splitter"
            )

    logger.debug(f"Data YAML verified: {data_yaml}")
    return data_yaml.resolve()


# ---------------------------------------------------------------------------
# Post-training: save best weights to models/
# ---------------------------------------------------------------------------

def _archive_existing(dest: Path) -> Path | None:
    """Move an existing checkpoint into ``models/archive/`` before it is replaced.

    The archive name carries the old file's modification date, so the
    previous model stays identifiable and can be restored or benchmarked
    against its successor.

    Args:
        dest: Checkpoint that is about to be overwritten.

    Returns:
        Where the old file was moved, or None if there was nothing to archive.
    """
    if not dest.exists():
        return None

    stamp = datetime.fromtimestamp(dest.stat().st_mtime).strftime("%Y-%m-%d_%H%M")
    archive_dir = MODELS_DIR / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    target = archive_dir / f"{dest.stem}.{stamp}{dest.suffix}"
    suffix = 1
    while target.exists():
        target = archive_dir / f"{dest.stem}.{stamp}_{suffix}{dest.suffix}"
        suffix += 1
    shutil.move(str(dest), target)
    logger.info(f"Previous checkpoint archived → {target}")
    return target


def _save_best_weights(
    run_dir: Path,
    model_name: str,
    run_name: str | None = None,
) -> Path | None:
    """Copy the best training checkpoint to ``models/``.

    ``models/<model>_ppe_best.pt`` is what the API and CLI load by default, so
    only a default-named run is promoted into that slot. A run started with
    ``--run-name`` (a smoke test, an experiment) is saved as
    ``models/<run_name>_best.pt`` instead and never displaces the served model.

    Whatever the slot, an existing file there is archived rather than
    overwritten — a one-epoch test run once replaced a 50-epoch model this way.

    Args:
        run_dir: The ultralytics run output directory.
        model_name: Model variant name, e.g. ``"yolo26l"``.
        run_name: Custom run name, if one was given.

    Returns:
        Path to the saved checkpoint, or None if ``best.pt`` was not found
        (training may have failed).
    """
    best_src = run_dir / "weights" / "best.pt"
    if not best_src.exists():
        logger.warning(f"best.pt not found at {best_src} — skipping copy to models/")
        return None

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if run_name:
        dest = MODELS_DIR / f"{run_name}_best.pt"
    else:
        dest = MODELS_DIR / f"{model_name}_ppe_best.pt"

    _archive_existing(dest)
    shutil.copy2(best_src, dest)
    logger.success(f"Best weights saved → {dest}")
    if run_name:
        logger.info(
            f"Custom run name, so the served model models/{model_name}_ppe_best.pt "
            "was left untouched. Promote this checkpoint by copying it there "
            "once you've benchmarked it."
        )
    return dest


# ---------------------------------------------------------------------------
# Core train function
# ---------------------------------------------------------------------------

def train(
    model_name: str,
    config_path: Path | None = None,
    epochs: int | None = None,
    batch: int | None = None,
    device: str | None = None,
    use_mlflow: bool = False,
    run_name: str | None = None,
) -> Path | None:
    """Train a YOLO model on the PPE dataset.

    Loads the hyperparameter config for ``model_name``, registers loguru
    (and optionally MLflow) callbacks, then calls ``YOLO.train()``.

    Args:
        model_name: Model variant to train, e.g. ``"yolo26l"``, ``"yolo11n"``.
            A pretrained ``{model_name}.pt`` checkpoint will be downloaded
            automatically by ultralytics on first use.
        config_path: Override the default ``configs/{model_name}_train.yaml``.
            If None, the default config for ``model_name`` is used.
        epochs: Override the number of training epochs from the config.
        batch: Override the batch size from the config.
        device: GPU/CPU device string, e.g. ``"0"``, ``"0,1"``, ``"cpu"``.
            Defaults to ``settings.ppb_device``.
        use_mlflow: Whether to register MLflow callbacks for experiment
            tracking. Requires ``mlflow`` installed and ``MLFLOW_TRACKING_URI``
            set in ``.env``.
        run_name: Custom name for the ultralytics run directory under ``runs/``.
            Defaults to ``{model_name}_ppe``.

    Returns:
        Path to the best checkpoint saved under ``models/``, or None if
        training completed but no best.pt was produced.

    Raises:
        FileNotFoundError: If the data pipeline hasn't been run yet, or if
            the training config YAML is missing.
        ImportError: If ``ultralytics`` is not installed.
    """
    # --- Resolve config ---
    cfg_path = config_path or config_path_for(model_name)
    cfg: TrainingConfig = load_training_config(cfg_path)

    # Apply CLI / API overrides
    if epochs is not None:
        cfg = cfg.model_copy(update={"epochs": epochs})
    if batch is not None:
        cfg = cfg.model_copy(update={"batch": batch})

    # --- Pre-flight ---
    data_yaml = _check_data_yaml()
    active_device = device or settings.ppb_device
    active_run_name = run_name or f"{model_name}_ppe"

    logger.info(f"Training {model_name} | config={cfg_path.name} | device={active_device}")

    # --- Load model ---
    # Disable cv2's internal thread pool before CUDA/YOLO initialises.
    # Each cv2 worker thread reserves its own stack in the VA space; on Windows
    # this compounds CUDA's VA reservation and causes OutOfMemoryError on even
    # small image allocations (e.g. 640×640×3 = 1.2 MB).
    try:
        import cv2  # noqa: PLC0415
        cv2.setNumThreads(0)
        logger.debug("cv2 thread pool disabled (Windows VA fragmentation mitigation)")
    except ImportError:
        pass

    try:
        from ultralytics import YOLO  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "ultralytics not installed. Run: uv add ultralytics"
        ) from exc

    # ultralytics downloads the pretrained checkpoint automatically on first use
    model = YOLO(f"{model_name}.pt")

    # --- Register callbacks ---
    for event, fn in build_loguru_callbacks().items():
        model.add_callback(event, fn)

    if use_mlflow:
        for event, fn in build_mlflow_callbacks(model_name=model_name, cfg=cfg).items():
            model.add_callback(event, fn)

    # --- Train ---
    results = model.train(
        data=str(data_yaml),
        project="runs",
        name=active_run_name,
        device=active_device,
        **cfg.to_ultralytics_kwargs(),
    )

    # --- Save best weights ---
    run_dir = Path("runs") / active_run_name
    # ultralytics may suffix run_name with a number if exist_ok=False
    # find the actual output dir from the results object
    if results is not None and hasattr(results, "save_dir"):
        run_dir = Path(results.save_dir)

    best_pt = _save_best_weights(run_dir, model_name, run_name=run_name)
    return best_pt


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _configure_logger() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
        colorize=True,
    )


def main() -> None:
    """CLI entry point for the training pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="PPB Vision — YOLO Trainer (YOLO26 + YOLO11)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model variant to train (e.g. yolo26l, yolo11n, yolo26n, yolo11l)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epochs from the config YAML.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Override batch size from the config YAML.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to train on: '0', '0,1', 'cpu'. Defaults to PPB_DEVICE env var.",
    )
    parser.add_argument(
        "--config",
        default=None,
        type=Path,
        help="Override config YAML path (default: configs/<model>_train.yaml).",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        default=False,
        help="Enable MLflow experiment tracking.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Custom run name for the ultralytics output directory under runs/.",
    )

    args = parser.parse_args()
    _configure_logger()

    logger.info(f"PPB Vision — Trainer starting for model: {args.model}")

    best_pt = train(
        model_name=args.model,
        config_path=args.config,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        use_mlflow=args.mlflow,
        run_name=args.run_name,
    )

    if best_pt:
        logger.success(f"Training complete. Best weights: {best_pt}")
    else:
        logger.warning("Training complete but best.pt was not found.")


if __name__ == "__main__":
    main()

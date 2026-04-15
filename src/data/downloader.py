"""Dataset downloader for the PPB Vision Module.

Downloads PPE datasets from Roboflow Universe and Kaggle into ``data/raw/``.
Each dataset is saved in YOLOv8 format (images + YOLO TXT annotations).

Roboflow datasets:
    - PPE Combined Model v8      (~44,000 images)
    - PPE Annotation v7          (~10,129 images)
    - Construction Site Safety   (~2,801 images)

Kaggle datasets:
    - Safety Helmet & Reflective Jacket  (~5,000 images)
      Set KAGGLE_PPE_DATASET=owner/dataset-name in .env before running.

Usage::

    # Download all datasets (Roboflow + Kaggle)
    uv run python -m src.data.downloader

    # Download a single dataset by key
    uv run python -m src.data.downloader --dataset ppe_combined
    uv run python -m src.data.downloader --dataset kaggle_ppe
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from roboflow import Roboflow

from src.settings import settings


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """Describes a single Roboflow dataset to download.

    Attributes:
        key: Internal short name used as the output folder name.
        workspace: Roboflow workspace slug.
        project: Roboflow project slug.
        version: Dataset version number to download.
        description: Human-readable name for log messages.
    """

    key: str
    workspace: str
    project: str
    version: int
    description: str


@dataclass
class KaggleDatasetConfig:
    """Describes a single Kaggle dataset to download.

    Attributes:
        key: Internal short name used as the output folder name.
        identifier: Kaggle dataset in ``owner/dataset-name`` format.
        description: Human-readable name for log messages.
    """

    key: str
    identifier: str  # owner/dataset-name from kaggle.com/datasets/...
    description: str


ROBOFLOW_DATASETS: list[DatasetConfig] = [
    DatasetConfig(
        key="ppe_combined",
        workspace="roboflow-universe-projects",
        project="personal-protective-equipment-combined-model",
        version=8,
        description="PPE Combined Model v8 (~44,000 images)",
    ),
    DatasetConfig(
        key="ppe_annotation",
        workspace="ppe-detection-4ewkm",
        project="ppe-annotation",
        version=7,
        description="PPE Annotation v7 (~10,129 images)",
    ),
    DatasetConfig(
        key="construction_safety",
        workspace="roboflow-universe-projects",
        project="construction-site-safety",
        version=1,
        description="Construction Site Safety (~2,801 images)",
    ),
]


def _build_kaggle_registry() -> list[KaggleDatasetConfig]:
    """Build the Kaggle dataset registry from settings.

    The Kaggle dataset identifier is read from ``KAGGLE_PPE_DATASET`` in .env.
    Returns an empty list if the env var is not set, so Roboflow-only runs
    still work without Kaggle credentials.

    Returns:
        List of configured Kaggle datasets.
    """
    if not settings.kaggle_ppe_dataset:
        return []
    return [
        KaggleDatasetConfig(
            key="kaggle_ppe",
            identifier=settings.kaggle_ppe_dataset,
            description="Safety Helmet & Reflective Jacket (~5,000 images)",
        ),
    ]


# ---------------------------------------------------------------------------
# Roboflow download
# ---------------------------------------------------------------------------

def download_roboflow_dataset(cfg: DatasetConfig, output_root: Path) -> Path:
    """Download a single Roboflow dataset in YOLOv8 format.

    Args:
        cfg: Dataset configuration (workspace, project, version).
        output_root: Root directory where the dataset folder will be created.

    Returns:
        Path to the downloaded dataset directory.

    Raises:
        RuntimeError: If the Roboflow download fails.
    """
    dest = output_root / cfg.key
    if dest.exists() and any(dest.iterdir()):
        logger.info(f"[{cfg.key}] Already downloaded at {dest} — skipping.")
        return dest

    dest.mkdir(parents=True, exist_ok=True)
    logger.info(f"[{cfg.key}] Downloading: {cfg.description}")
    logger.info(f"[{cfg.key}] → {cfg.workspace}/{cfg.project} version {cfg.version}")

    try:
        rf = Roboflow(api_key=settings.roboflow_api_key)
        project = rf.workspace(cfg.workspace).project(cfg.project)
        dataset = project.version(cfg.version).download(
            model_format="yolov8",
            location=str(dest),
            overwrite=False,
        )
        logger.success(f"[{cfg.key}] Saved to: {dataset.location}")
        return Path(dataset.location)

    except Exception as exc:
        logger.error(f"[{cfg.key}] Download failed: {exc}")
        raise RuntimeError(f"Failed to download {cfg.key}") from exc


# ---------------------------------------------------------------------------
# Kaggle download
# ---------------------------------------------------------------------------

def _configure_kaggle_credentials() -> None:
    """Inject Kaggle credentials from settings into the environment.

    The ``kaggle`` package reads ``KAGGLE_USERNAME`` and ``KAGGLE_KEY`` from
    the environment (or ``~/.kaggle/kaggle.json``). Setting them here ensures
    credentials loaded from ``.env`` are picked up before the first API call.
    """
    if settings.kaggle_username:
        os.environ["KAGGLE_USERNAME"] = settings.kaggle_username
    if settings.kaggle_key:
        os.environ["KAGGLE_KEY"] = settings.kaggle_key

    if not settings.kaggle_username or not settings.kaggle_key:
        logger.warning(
            "KAGGLE_USERNAME or KAGGLE_KEY is not set. "
            "Kaggle download will only succeed if ~/.kaggle/kaggle.json exists."
        )


def download_kaggle_dataset(cfg: KaggleDatasetConfig, output_root: Path) -> Path:
    """Download a Kaggle dataset and extract it to ``data/raw/<key>/``.

    The dataset must be in YOLO TXT format with a ``train/valid/test`` folder
    structure inside the zip. After download verify the class names in the
    extracted ``data.yaml`` match ``configs/class_maps.yaml``.

    Args:
        cfg: Kaggle dataset configuration.
        output_root: Root directory where the dataset folder will be created.

    Returns:
        Path to the extracted dataset directory.

    Raises:
        RuntimeError: If Kaggle credentials are missing or download fails.
        ImportError: If the ``kaggle`` package is not installed.
    """
    dest = output_root / cfg.key
    if dest.exists() and any(dest.iterdir()):
        logger.info(f"[{cfg.key}] Already downloaded at {dest} — skipping.")
        return dest

    dest.mkdir(parents=True, exist_ok=True)
    logger.info(f"[{cfg.key}] Downloading Kaggle dataset: {cfg.description}")
    logger.info(f"[{cfg.key}] → kaggle.com/datasets/{cfg.identifier}")

    try:
        import kaggle  # noqa: PLC0415 — intentional lazy import

        _configure_kaggle_credentials()
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            cfg.identifier,
            path=str(dest),
            unzip=True,
            quiet=False,
        )
        logger.success(f"[{cfg.key}] Extracted to: {dest}")
        logger.info(
            f"[{cfg.key}] Verify class names in {dest}/data.yaml match "
            "configs/class_maps.yaml → kaggle_ppe section."
        )
        return dest

    except ImportError as exc:
        logger.error("kaggle package not installed. Run: uv add kaggle")
        raise RuntimeError("kaggle package missing") from exc
    except Exception as exc:
        logger.error(f"[{cfg.key}] Kaggle download failed: {exc}")
        raise RuntimeError(f"Failed to download Kaggle dataset {cfg.key}") from exc


# ---------------------------------------------------------------------------
# Unified download orchestration
# ---------------------------------------------------------------------------

def download_all(dataset_keys: list[str] | None = None) -> dict[str, Path]:
    """Download all (or a subset of) registered datasets.

    Combines Roboflow and Kaggle datasets into a single registry. Pass
    ``dataset_keys`` to restrict which datasets are downloaded.

    Args:
        dataset_keys: Optional list of dataset keys to download.
            If None, all datasets (Roboflow + Kaggle) are downloaded.

    Returns:
        Mapping of dataset key → downloaded directory path.
    """
    output_root = settings.data_raw_dir
    output_root.mkdir(parents=True, exist_ok=True)

    kaggle_datasets = _build_kaggle_registry()

    # Build a flat key→config lookup for both source types
    all_roboflow: dict[str, DatasetConfig] = {d.key: d for d in ROBOFLOW_DATASETS}
    all_kaggle: dict[str, KaggleDatasetConfig] = {d.key: d for d in kaggle_datasets}
    all_keys = list(all_roboflow) + list(all_kaggle)

    active_keys = dataset_keys if dataset_keys else all_keys
    unknown = [k for k in active_keys if k not in all_roboflow and k not in all_kaggle]
    if unknown:
        logger.warning(f"Unknown dataset key(s) — skipping: {unknown}")

    results: dict[str, Path] = {}
    failed: list[str] = []

    for key in active_keys:
        if key in unknown:
            continue
        try:
            if key in all_roboflow:
                results[key] = download_roboflow_dataset(all_roboflow[key], output_root)
            else:
                results[key] = download_kaggle_dataset(all_kaggle[key], output_root)
        except RuntimeError:
            failed.append(key)

    # Summary
    logger.info("─" * 50)
    logger.info(f"Download complete: {len(results)}/{len(active_keys) - len(unknown)} succeeded")
    for key, path in results.items():
        logger.info(f"  ✓ {key:25s} → {path}")
    for key in failed:
        logger.error(f"  ✗ {key:25s} → FAILED")
    logger.info("─" * 50)

    if failed:
        logger.warning(
            f"{len(failed)} dataset(s) failed. "
            "Check your API keys and network connection."
        )

    return results


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


def _all_keys() -> list[str]:
    """Return all registered dataset keys (Roboflow + Kaggle)."""
    return [d.key for d in ROBOFLOW_DATASETS] + [d.key for d in _build_kaggle_registry()]


def main(dataset: str | None = None) -> None:
    """CLI entry point.

    Args:
        dataset: Single dataset key to download, or None for all.
    """
    _configure_logger()
    logger.info("PPB Vision — Dataset Downloader")
    logger.info(f"Output directory: {settings.data_raw_dir.resolve()}")

    keys = [dataset] if dataset else None
    download_all(dataset_keys=keys)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download PPE datasets from Roboflow and Kaggle"
    )
    parser.add_argument(
        "--dataset",
        choices=_all_keys(),
        default=None,
        help="Download a single dataset by key (default: all)",
    )
    args = parser.parse_args()
    main(dataset=args.dataset)

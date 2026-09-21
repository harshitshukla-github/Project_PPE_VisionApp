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


def _is_valid_slug(slug: str) -> bool:
    """Check that a Kaggle dataset identifier is a usable ``owner/name`` pair.

    A malformed slug fails as an opaque 404 from the Kaggle API, which reads
    like a network or permissions problem rather than a config typo. This has
    already cost the project one silently-skipped dataset, so the shape is
    checked up front instead.

    Rejects: missing ``/``, empty halves, more than one ``/``, and the
    ``owner`` placeholder carried over from .env.example.

    Args:
        slug: Candidate identifier, e.g. ``"niravnaik/safety-helmet"``.

    Returns:
        True if the slug is well-formed.
    """
    parts = slug.strip().split("/")
    if len(parts) != 2:
        return False
    owner, name = (p.strip() for p in parts)
    return bool(owner) and bool(name) and owner.lower() != "owner"


def _build_kaggle_registry() -> list[KaggleDatasetConfig]:
    """Build the Kaggle dataset registry from settings.

    The identifier comes from ``KAGGLE_PPE_DATASET``, defaulting to the
    Safety Helmet & Reflective Jacket dataset. Returns an empty list when the
    slug is unset or malformed, so Roboflow-only runs still work.

    Returns:
        List of configured Kaggle datasets.
    """
    slug = settings.kaggle_ppe_dataset.strip()
    if not slug:
        return []

    if not _is_valid_slug(slug):
        logger.error(
            f"KAGGLE_PPE_DATASET is malformed: '{slug}'. "
            "Expected '<owner>/<name>' exactly as it appears in the URL "
            "kaggle.com/datasets/<owner>/<name> — skipping the Kaggle dataset."
        )
        return []

    return [
        KaggleDatasetConfig(
            key="kaggle_ppe",
            identifier=slug,
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

    # Ensure the parent exists but do NOT pre-create dest itself.
    # Pre-creating dest (even as empty) causes the Roboflow SDK to treat it as
    # an existing download and skip the transfer when overwrite=False.
    output_root.mkdir(parents=True, exist_ok=True)
    logger.info(f"[{cfg.key}] Downloading: {cfg.description}")
    logger.info(f"[{cfg.key}] → {cfg.workspace}/{cfg.project} version {cfg.version}")

    try:
        rf = Roboflow(api_key=settings.roboflow_api_key)
        project = rf.workspace(cfg.workspace).project(cfg.project)
        dataset = project.version(cfg.version).download(
            model_format="yolov8",
            location=str(dest),
            overwrite=True,  # safe — our own check above already confirmed dest is empty
        )
        location = Path(dataset.location)
        logger.success(f"[{cfg.key}] Saved to: {location}")
        # The SDK reports success even when its extraction stopped early at
        # MAX_PATH, so never trust the exit status alone — count the files.
        verify_and_repair(cfg.key, location)
        return location

    except Exception as exc:
        logger.error(f"[{cfg.key}] Download failed: {exc}")
        raise RuntimeError(f"Failed to download {cfg.key}") from exc


# ---------------------------------------------------------------------------
# Archive extraction (Windows MAX_PATH-safe)
# ---------------------------------------------------------------------------

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def _long_path(path: Path) -> str:
    r"""Return an extended-length path string on Windows to bypass MAX_PATH.

    Roboflow filenames routinely exceed the 260-character limit (the
    ppe_combined archive reaches 307). Prefixing the resolved absolute path
    with ``\\?\`` tells the Windows API to skip the check. No-op elsewhere.

    ``src/data/remapper.py`` carries the same shim; it is duplicated rather
    than shared so the two modules stay independent.

    Args:
        path: Path to convert.

    Returns:
        Path string safe for ``open()`` and ``os`` calls.
    """
    if sys.platform == "win32":
        return "\\\\?\\" + str(path.resolve())
    return str(path.resolve())


def count_images_in_zip(zip_path: Path) -> int:
    """Count image entries inside an archive.

    Args:
        zip_path: Archive to inspect.

    Returns:
        Number of image members.
    """
    import zipfile  # noqa: PLC0415

    with zipfile.ZipFile(zip_path) as zf:
        return sum(1 for m in zf.namelist() if m.lower().endswith(IMAGE_SUFFIXES))


def count_images_on_disk(root: Path) -> int:
    """Count extracted images beneath a directory.

    Args:
        root: Directory to walk.

    Returns:
        Number of image files found.
    """
    total = 0
    for dirpath, _, filenames in os.walk(root):
        if os.path.basename(dirpath) == "images":
            total += sum(1 for f in filenames if f.lower().endswith(IMAGE_SUFFIXES))
    return total


def extract_archive(zip_path: Path, dest: Path) -> int:
    """Extract an archive using Windows-safe long paths.

    The Roboflow SDK's own extraction stops silently when it hits MAX_PATH,
    leaving a partial dataset and still reporting success — which previously
    cost this project 25,000 images without any error surfacing.

    Args:
        zip_path: Archive to extract.
        dest: Directory to extract into.

    Returns:
        Number of files written.

    Raises:
        FileNotFoundError: If the archive is missing.
    """
    import zipfile  # noqa: PLC0415

    if not zip_path.exists():
        raise FileNotFoundError(f"Archive not found: {zip_path}")

    written = 0
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if member.endswith("/"):
                continue
            target = dest / member
            os.makedirs(_long_path(target.parent), exist_ok=True)
            with zf.open(member) as src, open(_long_path(target), "wb") as out:
                out.write(src.read())
            written += 1
    return written


def verify_and_repair(dataset_key: str, dataset_dir: Path) -> bool:
    """Check an extracted dataset against its archive, re-extracting if short.

    Args:
        dataset_key: Dataset key, for log messages.
        dataset_dir: Directory holding the extracted dataset (and its zip).

    Returns:
        True if the dataset is complete (or was successfully repaired).
    """
    zip_path = dataset_dir / "roboflow.zip"
    if not zip_path.exists():
        # No archive to compare against — nothing we can verify.
        return True

    expected = count_images_in_zip(zip_path)
    actual = count_images_on_disk(dataset_dir)
    if actual >= expected:
        logger.success(f"[{dataset_key}] Extraction verified: {actual:,} images.")
        return True

    logger.warning(
        f"[{dataset_key}] Incomplete extraction: {actual:,} of {expected:,} images "
        "on disk. This is the Windows MAX_PATH limit — re-extracting with "
        "long-path support."
    )
    written = extract_archive(zip_path, dataset_dir)
    repaired = count_images_on_disk(dataset_dir)
    if repaired >= expected:
        logger.success(
            f"[{dataset_key}] Repaired: {written:,} files written, {repaired:,} images."
        )
        return True

    logger.error(
        f"[{dataset_key}] Still short after re-extraction: {repaired:,}/{expected:,}."
    )
    return False


def repair_all(dataset_keys: list[str] | None = None) -> dict[str, bool]:
    """Verify every downloaded dataset and repair truncated extractions.

    Args:
        dataset_keys: Datasets to check. Defaults to all Roboflow datasets.

    Returns:
        Mapping of dataset key → whether it is now complete.
    """
    keys = dataset_keys or [d.key for d in ROBOFLOW_DATASETS]
    results: dict[str, bool] = {}
    for key in keys:
        dataset_dir = settings.data_raw_dir / key
        if not dataset_dir.exists():
            logger.warning(f"[{key}] Not downloaded — skipping.")
            continue
        results[key] = verify_and_repair(key, dataset_dir)
    return results


# ---------------------------------------------------------------------------
# Kaggle download
# ---------------------------------------------------------------------------

def _configure_kaggle_credentials() -> None:
    """Inject Kaggle credentials from settings into the environment.

    pydantic-settings loads ``.env`` into the ``settings`` object but never
    writes to ``os.environ``, so values must be exported here before the
    ``kaggle`` client reads them.

    The client tries auth in this order (kagglesdk ``get_access_token_from_env``):

    1. ``KAGGLE_API_TOKEN`` — the current ``KGAT_...`` bearer token. The
       username is introspected from the token, so no username is needed.
    2. ``~/.kaggle/access_token`` (or ``access_token.txt``) — same token,
       read from disk.
    3. ``KAGGLE_USERNAME`` + ``KAGGLE_KEY`` — the legacy ``kaggle.json`` pair.

    Any one of these is sufficient; the warning fires only when none is set.
    """
    if settings.kaggle_api_token:
        os.environ["KAGGLE_API_TOKEN"] = settings.kaggle_api_token
    if settings.kaggle_username:
        os.environ["KAGGLE_USERNAME"] = settings.kaggle_username
    if settings.kaggle_key:
        os.environ["KAGGLE_KEY"] = settings.kaggle_key

    has_token = bool(settings.kaggle_api_token)
    has_legacy = bool(settings.kaggle_username and settings.kaggle_key)
    token_file = Path.home() / ".kaggle" / "access_token"
    legacy_file = Path.home() / ".kaggle" / "kaggle.json"

    if has_token:
        logger.debug("Kaggle auth: using KAGGLE_API_TOKEN from settings.")
    elif has_legacy:
        logger.debug("Kaggle auth: using legacy KAGGLE_USERNAME + KAGGLE_KEY.")
    elif token_file.exists() or legacy_file.exists():
        logger.debug(f"Kaggle auth: relying on {token_file.parent} on disk.")
    else:
        logger.warning(
            "No Kaggle credentials found. Set KAGGLE_API_TOKEN in .env "
            "(get a KGAT_... token from kaggle.com -> Settings -> API), or "
            "save it to ~/.kaggle/access_token. The legacy "
            "KAGGLE_USERNAME + KAGGLE_KEY pair also works."
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

    output_root.mkdir(parents=True, exist_ok=True)
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

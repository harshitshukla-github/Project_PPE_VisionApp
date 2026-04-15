"""PPB Vision Module — Command-Line Interface.

Built with Typer. Provides commands for every stage of the pipeline:

    data        Dataset management (download / remap / validate / split)
    train       Train a YOLO model (YOLO26 or YOLO11 variant)
    benchmark   Evaluate and compare trained models  [coming in Phase 1]
    export      Export a trained model to ONNX       [coming in Phase 1]
    serve       Launch the FastAPI inference server   [coming in Phase 2]

Quick-start::

    # 1. Pull all datasets
    uv run python cli.py data pipeline

    # 2. Train production + edge models
    uv run python cli.py train --model yolo26l
    uv run python cli.py train --model yolo11l

    # 3. Compare results
    uv run python cli.py benchmark

    # 4. Serve the best model
    uv run python cli.py serve --model yolo26l
"""

from __future__ import annotations

import sys
from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.settings import settings

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="ppb",
    help="PPB Vision — PPE Compliance AI Pipeline",
    rich_markup_mode="markdown",
    no_args_is_help=True,
)

data_app = typer.Typer(
    help="Dataset management: download, remap, validate, split.",
    rich_markup_mode="markdown",
    no_args_is_help=True,
)
app.add_typer(data_app, name="data")

console = Console()

# ---------------------------------------------------------------------------
# Enums for constrained options
# ---------------------------------------------------------------------------

class ModelVariant(str, Enum):
    yolo26l = "yolo26l"
    yolo26n = "yolo26n"
    yolo26m = "yolo26m"
    yolo26s = "yolo26s"
    yolo26x = "yolo26x"
    yolo11l = "yolo11l"
    yolo11n = "yolo11n"
    yolo11m = "yolo11m"
    yolo11s = "yolo11s"
    yolo11x = "yolo11x"


class DatasetKey(str, Enum):
    ppe_combined = "ppe_combined"
    ppe_annotation = "ppe_annotation"
    construction_safety = "construction_safety"
    kaggle_ppe = "kaggle_ppe"


class ExportFormat(str, Enum):
    onnx = "onnx"
    tflite = "tflite"
    torchscript = "torchscript"
    engine = "engine"  # TensorRT


# ---------------------------------------------------------------------------
# Shared logger setup
# ---------------------------------------------------------------------------

def _setup_logger(level: str = settings.log_level) -> None:
    """Configure loguru for CLI output."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
        colorize=True,
    )


# ---------------------------------------------------------------------------
# data commands
# ---------------------------------------------------------------------------

@data_app.command("download")
def data_download(
    dataset: Optional[list[DatasetKey]] = typer.Option(
        None,
        "--dataset", "-d",
        help=(
            "Dataset(s) to download. Repeat the flag for multiple: "
            "`-d ppe_combined -d kaggle_ppe`. "
            "Omit to download **all** datasets."
        ),
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging verbosity."),
) -> None:
    """Download PPE datasets from Roboflow and Kaggle into `data/raw/`.

    **Datasets:**
    - `ppe_combined`       — Roboflow PPE Combined Model v8 (~44k images)
    - `ppe_annotation`     — Roboflow PPE Annotation v7 (~10k images)
    - `construction_safety`— Roboflow Construction Site Safety (~2.8k images)
    - `kaggle_ppe`         — Kaggle Safety Helmet & Reflective Jacket (~5k images)

    **Requires** `ROBOFLOW_API_KEY` (and optionally `KAGGLE_USERNAME` + `KAGGLE_KEY`) in `.env`.
    """
    _setup_logger(log_level)
    from src.data.downloader import download_all

    keys = [d.value for d in dataset] if dataset else None
    console.print(Panel(
        f"Downloading: [bold cyan]{keys or 'all datasets'}[/]",
        title="PPB — Data Download",
        border_style="blue",
    ))
    download_all(dataset_keys=keys)


@data_app.command("remap")
def data_remap(
    dataset: Optional[list[DatasetKey]] = typer.Option(
        None,
        "--dataset", "-d",
        help="Dataset(s) to remap. Omit to remap all.",
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging verbosity."),
) -> None:
    """Remap downloaded dataset class labels to the unified 12-class schema.

    Reads from `data/raw/<dataset>/` and writes remapped annotations
    to `data/processed/<dataset>/`. Discards classes not in the schema.

    Run **after** `data download`.
    """
    _setup_logger(log_level)
    from src.data.remapper import remap_all

    keys = [d.value for d in dataset] if dataset else None
    console.print(Panel(
        f"Remapping: [bold cyan]{keys or 'all datasets'}[/]",
        title="PPB — Class Remapper",
        border_style="blue",
    ))
    remap_all(dataset_keys=keys)


@data_app.command("validate")
def data_validate(
    split: Optional[list[str]] = typer.Option(
        None,
        "--split", "-s",
        help="Split(s) to validate: train, val, test. Omit for all.",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Exit with code 1 if any annotation errors are found.",
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging verbosity."),
) -> None:
    """Validate annotation integrity in `data/splits/`.

    Checks:
    - Every image has a matching `.txt` label
    - All class IDs are in range [0, 11]
    - Bounding boxes are normalized (0.0 – 1.0)
    - No duplicate filenames across splits

    Run **after** `data split`.
    """
    _setup_logger(log_level)
    from src.data.validator import run_validation

    ok = run_validation(splits=split or None, strict=strict)
    if not ok and strict:
        raise typer.Exit(code=1)


@data_app.command("split")
def data_split(
    train: float = typer.Option(0.80, "--train", help="Train fraction."),
    val: float = typer.Option(0.10, "--val", help="Validation fraction."),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility."),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging verbosity."),
) -> None:
    """Split processed images into train / val / test sets.

    Shuffles all `data/processed/` images and copies them into
    `data/splits/` in standard YOLO layout.

    Default split: **80% train / 10% val / 10% test**.

    Run **after** `data remap`.
    """
    _setup_logger(log_level)
    from src.data.splitter import run_split

    test = round(1.0 - train - val, 6)
    if test < 0:
        console.print("[red]Error:[/] --train + --val must be ≤ 1.0")
        raise typer.Exit(code=1)

    console.print(Panel(
        f"Split ratios — train:[bold]{train:.0%}[/]  val:[bold]{val:.0%}[/]  test:[bold]{test:.0%}[/]  seed={seed}",
        title="PPB — Dataset Splitter",
        border_style="blue",
    ))
    run_split(train_ratio=train, val_ratio=val, seed=seed)


@data_app.command("pipeline")
def data_pipeline(
    log_level: str = typer.Option("INFO", "--log-level", help="Logging verbosity."),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Fail the pipeline if validation finds annotation errors.",
    ),
) -> None:
    """Run the full data pipeline in sequence.

    Executes all four stages automatically:

    1. **download** — pull all 4 datasets from Roboflow + Kaggle
    2. **remap**    — unify class labels to 12-class schema
    3. **split**    — 80/10/10 train/val/test split
    4. **validate** — check annotation integrity

    Equivalent to running each sub-command individually.
    """
    _setup_logger(log_level)
    from src.data.downloader import download_all
    from src.data.remapper import remap_all
    from src.data.splitter import run_split
    from src.data.validator import run_validation

    stages = [
        ("Download", lambda: download_all()),
        ("Remap", lambda: remap_all()),
        ("Split", lambda: run_split()),
        ("Validate", lambda: run_validation(strict=strict)),
    ]

    for i, (name, fn) in enumerate(stages, 1):
        console.rule(f"[bold blue]Stage {i}/4 — {name}")
        fn()

    console.print(Panel(
        "[green]Data pipeline complete.[/]\n"
        "Next: [bold cyan]uv run python cli.py train --model yolo26l[/]",
        title="PPB — Pipeline Done",
        border_style="green",
    ))


# ---------------------------------------------------------------------------
# train command
# ---------------------------------------------------------------------------

@app.command("train")
def train_cmd(
    model: ModelVariant = typer.Option(
        ...,
        "--model", "-m",
        help="Model variant to train.",
        show_choices=True,
    ),
    epochs: Optional[int] = typer.Option(
        None,
        "--epochs", "-e",
        help="Override epoch count from the config YAML.",
    ),
    batch: Optional[int] = typer.Option(
        None,
        "--batch", "-b",
        help="Override batch size from the config YAML.",
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device",
        help="Training device: '0', '0,1', 'cpu'. Default: PPB_DEVICE env var.",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="Override the hyperparameter YAML (default: configs/<model>_train.yaml).",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    mlflow: bool = typer.Option(
        False,
        "--mlflow",
        help="Enable MLflow experiment tracking (requires MLFLOW_TRACKING_URI in .env).",
    ),
    run_name: Optional[str] = typer.Option(
        None,
        "--run-name",
        help="Custom name for the output directory under runs/ (default: <model>_ppe).",
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging verbosity."),
) -> None:
    """Train a YOLO model on the PPE dataset.

    Loads hyperparameters from `configs/<model>_train.yaml`, registers
    loguru (and optionally MLflow) callbacks, then runs ultralytics training.

    The best checkpoint is automatically saved to `models/<model>_ppe_best.pt`.

    **Examples:**

    ```bash
    # Train YOLO26-L (production backend candidate)
    uv run python cli.py train --model yolo26l

    # Train YOLO11-N (edge candidate), override batch and epochs
    uv run python cli.py train --model yolo11n --epochs 50 --batch 64

    # Train with MLflow tracking on GPU 1
    uv run python cli.py train --model yolo26l --mlflow --device 1
    ```
    """
    _setup_logger(log_level)
    from src.training.trainer import train

    model_name = model.value
    console.print(Panel(
        f"Model: [bold cyan]{model_name}[/]  |  "
        f"Device: [bold]{device or settings.ppb_device}[/]  |  "
        f"MLflow: [bold]{'on' if mlflow else 'off'}[/]",
        title="PPB — Training",
        border_style="blue",
    ))

    best_pt = train(
        model_name=model_name,
        config_path=config,
        epochs=epochs,
        batch=batch,
        device=device,
        use_mlflow=mlflow,
        run_name=run_name,
    )

    if best_pt:
        console.print(Panel(
            f"[green]Best weights saved:[/] {best_pt}",
            title="Training Complete",
            border_style="green",
        ))
    else:
        console.print("[yellow]Training finished but best.pt was not found.[/]")


# ---------------------------------------------------------------------------
# benchmark command  (Phase 1 — evaluation module not yet built)
# ---------------------------------------------------------------------------

@app.command("benchmark")
def benchmark_cmd(
    models: Optional[list[ModelVariant]] = typer.Option(
        None,
        "--model", "-m",
        help="Model(s) to benchmark. Omit to benchmark all trained models.",
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging verbosity."),
) -> None:
    """Compare trained YOLO26 vs YOLO11 models on the test split.

    Generates a benchmark report at `runs/benchmark_report.md` with:
    mAP50, mAP50-95, per-class AP, inference latency, model size.

    **Status: coming in Phase 1 — evaluation module not yet built.**
    """
    _setup_logger(log_level)
    _not_yet_implemented(
        command="benchmark",
        phase="Phase 1",
        module="src/evaluation/benchmark.py",
        workaround=(
            "Run ultralytics val manually:\n"
            "  uv run python -c \"\n"
            "  from ultralytics import YOLO\n"
            "  YOLO('models/yolo26l_ppe_best.pt').val(data='configs/data.yaml')\n"
            "  \""
        ),
    )
    raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# export command  (Phase 1 — export module not yet built)
# ---------------------------------------------------------------------------

@app.command("export")
def export_cmd(
    model: ModelVariant = typer.Option(
        ..., "--model", "-m", help="Trained model variant to export."
    ),
    fmt: ExportFormat = typer.Option(
        ExportFormat.onnx,
        "--format", "-f",
        help="Export format.",
        show_choices=True,
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging verbosity."),
) -> None:
    """Export a trained model to ONNX / TFLite / TorchScript / TensorRT.

    Reads from `models/<model>_ppe_best.pt` and writes the export to
    `models/exports/`.

    **Status: coming in Phase 1 — export module not yet built.**
    """
    _setup_logger(log_level)
    _not_yet_implemented(
        command="export",
        phase="Phase 1",
        module="src/export/onnx_export.py",
        workaround=(
            f"Export manually with ultralytics:\n"
            f"  uv run python -c \"\n"
            f"  from ultralytics import YOLO\n"
            f"  YOLO('models/{model.value}_ppe_best.pt').export(format='{fmt.value}')\n"
            f"  \""
        ),
    )
    raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# serve command  (Phase 2 — FastAPI not yet built)
# ---------------------------------------------------------------------------

@app.command("serve")
def serve_cmd(
    model: ModelVariant = typer.Option(
        ..., "--model", "-m", help="Model variant to load for inference."
    ),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind the server to."),
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind the server to."),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging verbosity."),
) -> None:
    """Launch the FastAPI PPE compliance inference server.

    Serves the `/detect`, `/compliance/check`, `/health`, and `/model/info`
    endpoints. Gemini 2.5 Flash is invoked automatically on FAIL verdicts.

    **Status: coming in Phase 2 — FastAPI server not yet built.**
    """
    _setup_logger(log_level)
    _not_yet_implemented(
        command="serve",
        phase="Phase 2",
        module="src/api/main.py",
        workaround=None,
    )
    raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# info command — quick project status
# ---------------------------------------------------------------------------

@app.command("info")
def info_cmd() -> None:
    """Show project configuration and pipeline status."""
    table = Table(title="PPB Vision — Project Info", show_header=True, header_style="bold blue")
    table.add_column("Setting", style="cyan")
    table.add_column("Value")

    table.add_row("Active model", settings.ppb_model_name)
    table.add_row("Device", settings.ppb_device)
    table.add_row("Image size", str(settings.ppb_img_size))
    table.add_row("Confidence threshold", str(settings.ppb_confidence_threshold))
    table.add_row("IoU threshold", str(settings.ppb_iou_threshold))
    table.add_row("MLflow URI", settings.mlflow_tracking_uri)
    table.add_row("Log level", settings.log_level)
    table.add_row("Data splits dir", str(settings.data_splits_dir))
    table.add_row("Models dir", "models/")

    console.print(table)

    # Pipeline status
    status_table = Table(title="Pipeline Status", show_header=True, header_style="bold blue")
    status_table.add_column("Asset", style="cyan")
    status_table.add_column("Status")

    checks = [
        ("configs/data.yaml", settings.configs_dir / "data.yaml"),
        ("configs/yolo26l_train.yaml", settings.configs_dir / "yolo26l_train.yaml"),
        ("configs/yolo11l_train.yaml", settings.configs_dir / "yolo11l_train.yaml"),
        ("data/raw/ppe_combined", settings.data_raw_dir / "ppe_combined"),
        ("data/splits/train", settings.data_splits_dir / "train" / "images"),
        ("models/yolo26l_ppe_best.pt", Path("models") / "yolo26l_ppe_best.pt"),
        ("models/yolo11l_ppe_best.pt", Path("models") / "yolo11l_ppe_best.pt"),
    ]

    for label, path in checks:
        exists = path.exists() and (not path.is_dir() or any(path.iterdir()))
        status_table.add_row(label, "[green]✓ ready[/]" if exists else "[yellow]✗ missing[/]")

    console.print(status_table)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _not_yet_implemented(
    command: str,
    phase: str,
    module: str,
    workaround: str | None,
) -> None:
    """Print a consistent 'not yet implemented' message."""
    msg = (
        f"[bold yellow]`{command}`[/] is not yet implemented.\n\n"
        f"  Planned for: [bold]{phase}[/]\n"
        f"  Module:      [cyan]{module}[/]"
    )
    if workaround:
        msg += f"\n\n[bold]Workaround:[/]\n  {workaround}"
    console.print(Panel(msg, title="Coming Soon", border_style="yellow"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()

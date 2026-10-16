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

db_app = typer.Typer(
    help="PostgreSQL for the admin system: setup, status.",
    rich_markup_mode="markdown",
    no_args_is_help=True,
)
app.add_typer(db_app, name="db")

auth_app = typer.Typer(
    help="Accounts: create the first admin, create workers, reset PINs.",
    rich_markup_mode="markdown",
    no_args_is_help=True,
)
app.add_typer(auth_app, name="auth")

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


@data_app.command("extract")
def data_extract(
    dataset: Optional[list[DatasetKey]] = typer.Option(
        None,
        "--dataset", "-d",
        help="Dataset(s) to verify. Omit to check all Roboflow datasets.",
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging verbosity."),
) -> None:
    """Verify downloaded archives extracted fully, and repair them if not.

    On Windows the Roboflow SDK's extraction stops silently when a path
    exceeds the 260-character `MAX_PATH` limit — leaving a partial dataset
    while still reporting success. Roboflow filenames routinely exceed it.

    This compares the image count inside each `roboflow.zip` against what is
    actually on disk, and re-extracts with long-path support when short.

    **Example:**

    ```bash
    uv run python cli.py data extract
    ```
    """
    _setup_logger(log_level)
    from src.data.downloader import repair_all

    keys = [d.value for d in dataset] if dataset else None
    console.print(Panel(
        f"Verifying extraction: [bold cyan]{keys or 'all Roboflow datasets'}[/]",
        title="PPB — Archive Check",
        border_style="blue",
    ))

    results = repair_all(dataset_keys=keys)
    if not results:
        console.print("[yellow]Nothing to check.[/]")
        return

    incomplete = [k for k, ok in results.items() if not ok]
    if incomplete:
        console.print(f"[red]Still incomplete:[/] {incomplete}")
        raise typer.Exit(code=1)
    console.print(Panel(
        f"[green]All {len(results)} dataset(s) verified complete.[/]",
        border_style="green",
    ))


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
    clean: bool = typer.Option(
        False,
        "--clean",
        help="Delete any existing splits first. Required when `data/splits/` "
             "already holds a previous run — otherwise the two get mixed.",
    ),
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
    run_split(train_ratio=train, val_ratio=val, seed=seed, clean=clean)


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

    downloaded: dict = {}

    def _download() -> None:
        nonlocal downloaded
        downloaded = download_all()

    def _remap() -> None:
        keys = list(downloaded.keys())
        if not keys:
            console.print("[yellow]No datasets downloaded — skipping remap.[/]")
            return
        remap_all(dataset_keys=keys)

    stages = [
        ("Download", _download),
        ("Remap", _remap),
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
    device: Optional[str] = typer.Option(
        None,
        "--device",
        help="Evaluation device: '0' or 'cpu'. Default: PPB_DEVICE env var.",
    ),
    split: str = typer.Option(
        "test",
        "--split",
        help="Dataset split to evaluate: 'test' (unbiased) or 'val'.",
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging verbosity."),
) -> None:
    """Compare trained YOLO26 vs YOLO11 models on the test split.

    Evaluates every trained checkpoint in `models/`, then generates:

    - `runs/benchmark/benchmark_results.json` — raw metrics
    - `runs/benchmark/*.png` — comparison plots
    - `runs/benchmark_report.md` — the decision report

    **Example:**

    ```bash
    uv run python cli.py benchmark                      # all trained models
    uv run python cli.py benchmark -m yolo26l -m yolo11l --device 0
    ```
    """
    _setup_logger(log_level)
    from src.evaluation.benchmark import run_benchmark
    from src.evaluation.report import generate_report
    from src.evaluation.visualize import generate_plots

    keys = [m.value for m in models] if models else None
    console.print(Panel(
        f"Benchmarking: [bold cyan]{keys or 'all trained models'}[/]  |  "
        f"Split: [bold]{split}[/]  |  Device: [bold]{device or settings.ppb_device}[/]",
        title="PPB — Benchmark",
        border_style="blue",
    ))

    results = run_benchmark(keys, device=device, split=split)
    generate_plots(results)
    report_path = generate_report(results)

    # Summary table in the terminal
    table = Table(title=f"Benchmark — {split} split", header_style="bold blue")
    table.add_column("Model", style="cyan")
    table.add_column("mAP50")
    table.add_column("mAP50-95")
    table.add_column("Latency")
    table.add_column("FPS")
    table.add_column("Size")
    best = max(r.map50_95 for r in results)
    for r in results:
        mark = " 🥇" if r.map50_95 == best else ""
        table.add_row(
            r.model_name,
            f"{r.map50:.4f}",
            f"{r.map50_95:.4f}{mark}",
            f"{r.total_latency_ms:.1f} ms",
            f"{r.fps:.1f}",
            f"{r.size_mb:.1f} MB",
        )
    console.print(table)
    console.print(Panel(
        f"[green]Report:[/] {report_path}",
        title="Benchmark Complete",
        border_style="green",
    ))


# ---------------------------------------------------------------------------
# probe command — single-image diagnostic
# ---------------------------------------------------------------------------

@app.command("probe")
def probe_cmd(
    image: Path = typer.Argument(
        ...,
        help="Photo to analyse (JPEG/PNG).",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    job: Optional[str] = typer.Option(
        None,
        "--job", "-j",
        help="Job category from the PPE matrix, e.g. `general_labor`. "
             "Adds the per-class diagnosis and the verdict the API would return.",
    ),
    model: Optional[ModelVariant] = typer.Option(
        None,
        "--model", "-m",
        help="Model variant to probe. Default: PPB_MODEL_NAME env var.",
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device",
        help="Inference device: '0' or 'cpu'. Default: PPB_DEVICE env var.",
    ),
    conf: float = typer.Option(
        0.01,
        "--conf",
        help="Probe confidence floor. Lower surfaces more of what the model considered.",
    ),
    save: bool = typer.Option(
        False,
        "--save",
        help="Write an annotated copy to `runs/probe/` "
             "(green = kept, red = below production threshold).",
    ),
    log_level: str = typer.Option("WARNING", "--log-level", help="Logging verbosity."),
) -> None:
    """Diagnose why one photo produced its verdict.

    Runs the image at a near-zero confidence floor so you can see **every**
    detection the model considered — including the ones the production
    threshold discards — then re-runs it at larger input sizes and with TTA.

    For each required PPE class it reports one of:

    - `DETECTED`   — found above the production threshold
    - `CONFIG FIX` — missed at baseline, recovered by bigger imgsz / TTA
    - `THRESHOLD`  — the model saw it, the threshold discarded it
    - `DATA GAP`   — never seen at any config; needs training data

    Use this when a real photo produces a surprising FAIL: it separates a
    one-line settings fix from a genuine model problem.

    **Examples:**

    ```bash
    uv run python cli.py probe worker.jpg --job general_labor --save
    uv run python cli.py probe worker.jpg -j welder -m yolo26l --device cpu
    ```
    """
    _setup_logger(log_level)
    from src.evaluation.probe import probe_image

    try:
        probe_image(
            image_path=image,
            job_category=job,
            model_name=model.value if model else None,
            device=device,
            conf_floor=conf,
            save_annotated=save,
        )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        console.print(f"[red]Error:[/] {exc}")
        raise typer.Exit(code=1) from exc


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
    half: bool = typer.Option(
        False, "--half", help="Export FP16 weights (~50% smaller)."
    ),
    int8: bool = typer.Option(
        False, "--int8",
        help="Additionally produce an INT8-quantized ONNX (~75% smaller, CPU-fast). "
             "Only valid with --format onnx.",
    ),
    imgsz: Optional[int] = typer.Option(
        None, "--imgsz", help="Export image size. Default: PPB_IMG_SIZE env var."
    ),
    device: str = typer.Option(
        "cpu", "--device",
        help="Device for the export trace. TensorRT ('engine') needs a GPU, e.g. '0'.",
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging verbosity."),
) -> None:
    """Export a trained model to ONNX / TFLite / TorchScript / TensorRT.

    Reads from `models/<model>_ppe_best.pt` and writes the artifact to
    `models/exports/`.

    **Examples:**

    ```bash
    uv run python cli.py export --model yolo11l                    # FP32 ONNX
    uv run python cli.py export --model yolo26n --half --int8      # edge bundle
    ```
    """
    _setup_logger(log_level)
    from src.export.onnx_export import export_model

    if int8 and fmt is not ExportFormat.onnx:
        console.print("[red]Error:[/] --int8 requires --format onnx")
        raise typer.Exit(code=1)

    artifact = export_model(
        model.value, fmt=fmt.value, half=half, imgsz=imgsz, device=device
    )
    lines = [f"[green]Exported:[/] {artifact}"]

    if int8:
        from src.export.quantize import quantize_onnx_int8

        int8_path = quantize_onnx_int8(artifact)
        lines.append(f"[green]INT8:[/]     {int8_path}")

    console.print(Panel("\n".join(lines), title="Export Complete", border_style="green"))


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
    device: Optional[str] = typer.Option(
        None,
        "--device",
        help="Inference device: '0' or 'cpu'. Default: PPB_DEVICE env var.",
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging verbosity."),
) -> None:
    """Launch the FastAPI PPE compliance inference server.

    Serves the `/detect`, `/compliance/check`, `/health`, and `/model/info`
    endpoints under `/api/v1`. Gemini 2.5 Flash is invoked automatically on
    FAIL verdicts (never on PASS — cost optimization).

    **Examples:**

    ```bash
    uv run python cli.py serve --model yolo11l              # production pick
    uv run python cli.py serve --model yolo26n --port 8080  # edge model
    ```
    """
    import os

    import uvicorn

    _setup_logger(log_level)
    from src.api.main import DEVICE_ENV_VAR, MODEL_ENV_VAR

    # The lifespan hook reads these to select the model at startup.
    os.environ[MODEL_ENV_VAR] = model.value
    if device:
        os.environ[DEVICE_ENV_VAR] = device

    console.print(Panel(
        f"Serving [bold cyan]{model.value}[/] on [bold]http://{host}:{port}[/]\n"
        f"Docs: [bold]http://{host if host != '0.0.0.0' else 'localhost'}:{port}/docs[/]",
        title="PPB — Inference Server",
        border_style="blue",
    ))
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        log_level=log_level.lower(),
    )


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
# db commands
# ---------------------------------------------------------------------------

def _print_connection_report(label: str, url: str) -> bool:
    """Check one connection URL and print what was found. Returns success."""
    import psycopg

    from src.db.bootstrap import check_connection, mask_url

    try:
        report = check_connection(url)
    except (psycopg.Error, ValueError) as exc:
        console.print(f"[red]✗ {label}[/red] {mask_url(url)}\n  {str(exc).strip()}")
        return False
    tables = "[green]yes[/green]" if report.can_create_tables else "[red]no[/red]"
    console.print(
        f"[green]✓ {label}[/green] {mask_url(url)}\n"
        f"  PostgreSQL {report.server_version} · user {report.current_user} · "
        f"database {report.database} · can create tables: {tables}"
    )
    return report.can_create_tables


@db_app.command("setup")
def db_setup_cmd(
    host: str = typer.Option("localhost", "--host", help="PostgreSQL server host."),
    port: int = typer.Option(5432, "--port", help="PostgreSQL server port."),
    superuser: str = typer.Option(
        "postgres", "--superuser", help="Existing superuser to provision with."
    ),
    env_file: Path = typer.Option(Path(".env"), "--env-file", help=".env file to update."),
) -> None:
    """
    Create the **ppb_app** role and the **ppb** and **ppb_test** databases,
    then write `DATABASE_URL` and `TEST_DATABASE_URL` to `.env`.

    Asks once for the superuser's password. It is used for this one
    connection and never stored. The app role's password is generated and
    written only to `.env`; re-running reuses it. Safe to run again.

    ```bash
    uv run python cli.py db setup
    ```
    """
    import psycopg

    from src.db import bootstrap as b

    _setup_logger()
    password = None
    existing = b.read_env_var(env_file, b.DATABASE_URL_KEY)
    if existing:
        try:
            parts = b.parse_url(existing)
            if parts.user == b.APP_ROLE and parts.password:
                password = parts.password
                console.print(f"Reusing the {b.APP_ROLE} password already in {env_file}.")
        except ValueError:
            pass
    password = password or b.generate_password()

    su_password = typer.prompt(
        f"Password for PostgreSQL superuser '{superuser}' (not stored)", hide_input=True
    )
    try:
        with psycopg.connect(
            host=host,
            port=port,
            user=superuser,
            password=su_password,
            dbname="postgres",
            autocommit=True,
            connect_timeout=5,
        ) as conn:
            actions = b.provision(conn, b.APP_ROLE, password, [b.APP_DB, b.TEST_DB])
    except psycopg.Error as exc:
        console.print(
            f"[red]Provisioning as '{superuser}' on {host}:{port} failed.[/red]\n"
            f"  {str(exc).strip()}"
        )
        raise typer.Exit(code=1) from None

    for action in actions:
        console.print(f"  • {action}")

    b.upsert_env_var(
        env_file,
        b.DATABASE_URL_KEY,
        b.build_url(b.APP_ROLE, password, host, port, b.APP_DB),
        comment="PostgreSQL — admin system. Written by `cli.py db setup`; holds a password.",
    )
    b.upsert_env_var(
        env_file,
        b.TEST_DATABASE_URL_KEY,
        b.build_url(b.APP_ROLE, password, host, port, b.TEST_DB),
        comment="Throwaway database the test suite may wipe. Never point this at real data.",
    )
    console.print(f"Wrote {b.DATABASE_URL_KEY} and {b.TEST_DATABASE_URL_KEY} to {env_file}.\n")

    ok = all(
        _print_connection_report(label, b.read_env_var(env_file, key) or "")
        for label, key in (
            ("app", b.DATABASE_URL_KEY),
            ("test", b.TEST_DATABASE_URL_KEY),
        )
    )
    if not ok:
        raise typer.Exit(code=1)


@db_app.command("status")
def db_status_cmd() -> None:
    """Check that `DATABASE_URL` and `TEST_DATABASE_URL` connect and can create tables."""
    from src.db.bootstrap import DATABASE_URL_KEY, TEST_DATABASE_URL_KEY

    urls = [
        ("app", DATABASE_URL_KEY, settings.database_url),
        ("test", TEST_DATABASE_URL_KEY, settings.test_database_url),
    ]
    ok = True
    for label, key, url in urls:
        if not url:
            console.print(f"[yellow]✗ {label}[/yellow] {key} is not set — run `cli.py db setup`.")
            ok = False
            continue
        ok = _print_connection_report(label, url) and ok
    if not ok:
        raise typer.Exit(code=1)
    _print_schema_report(settings.database_url)


def _print_schema_report(url: str) -> None:
    """Print the app database's migration revision and how many jobs it holds."""
    from sqlalchemy.exc import SQLAlchemyError

    from src.db.engine import get_engine
    from src.db.matrix import count_jobs
    from src.db.migrate import current_revision, head_revision

    head = head_revision()
    try:
        current = current_revision(url)
        jobs = count_jobs(get_engine(url)) if current else 0
    except SQLAlchemyError as exc:
        console.print(f"[red]✗ schema[/red] could not be read: {str(exc).splitlines()[0]}")
        raise typer.Exit(code=1) from None

    if current != head:
        console.print(
            f"[yellow]✗ schema[/yellow] at revision {current or 'none'}, latest is {head} "
            "— run `cli.py db migrate`."
        )
    elif not jobs:
        console.print(f"[yellow]✗ schema[/yellow] at {head}, but no jobs — run `cli.py db seed`.")
    else:
        console.print(f"[green]✓ schema[/green] at {head} · {jobs} jobs")


@db_app.command("migrate")
def db_migrate_cmd() -> None:
    """
    Bring the database in `DATABASE_URL` up to the latest schema.

    Safe to run any time: migrations already applied are skipped.
    """
    from sqlalchemy.exc import SQLAlchemyError

    from src.db.migrate import current_revision, head_revision, upgrade_to_head

    _setup_logger()
    if not settings.database_url:
        console.print("[red]DATABASE_URL is not set — run `cli.py db setup` first.[/red]")
        raise typer.Exit(code=1)
    try:
        before = current_revision()
        upgrade_to_head()
        after = current_revision()
    except SQLAlchemyError as exc:
        console.print(f"[red]Migration failed:[/red] {str(exc).splitlines()[0]}")
        raise typer.Exit(code=1) from None
    if before == after:
        console.print(f"Already up to date at revision {after}.")
    else:
        console.print(f"Migrated from {before or 'an empty database'} to {after} (latest {head_revision()}).")


@db_app.command("seed")
def db_seed_cmd(
    matrix_file: Path = typer.Option(
        Path("configs/job_matrix.yaml"), "--matrix", help="Job matrix YAML to load."
    ),
) -> None:
    """
    Load the PPE items and the job matrix from YAML into an empty database.

    Jobs are only added when the database has none: after that, the
    database is the source of truth and the YAML is not applied again.
    """
    from sqlalchemy.exc import SQLAlchemyError

    from src.db.engine import get_engine
    from src.db.matrix import seed_from_yaml

    _setup_logger()
    if not settings.database_url:
        console.print("[red]DATABASE_URL is not set — run `cli.py db setup` first.[/red]")
        raise typer.Exit(code=1)
    try:
        report = seed_from_yaml(get_engine(), matrix_file)
    except (SQLAlchemyError, ValueError, FileNotFoundError) as exc:
        console.print(f"[red]Seeding failed:[/red] {str(exc).splitlines()[0]}")
        raise typer.Exit(code=1) from None
    console.print(f"Items added: {report.items_added}")
    if report.skipped_reason:
        console.print(f"Jobs not loaded: {report.skipped_reason}.")
    else:
        console.print(f"Jobs added: {report.jobs_added} (from {matrix_file})")


# ---------------------------------------------------------------------------
# auth commands (docs/auth-api.md §7)
# ---------------------------------------------------------------------------

def _auth_db():
    """A database session for an auth command, or exit with the fix."""
    from sqlalchemy.orm import Session

    from src.db.engine import get_engine

    if not settings.database_url:
        console.print("[red]DATABASE_URL is not set — run `cli.py db setup` first.[/red]")
        raise typer.Exit(code=1)
    # Keep loaded values after commit: the command prints them once the
    # session has closed.
    return Session(get_engine(), expire_on_commit=False)


def _run_account_command(action):
    """Run ``action(db)``, turning the expected failures into one red line."""
    from sqlalchemy.exc import SQLAlchemyError

    with _auth_db() as db:
        try:
            return action(db)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from None
        except SQLAlchemyError as exc:
            console.print(
                f"[red]Database error:[/red] {str(exc).splitlines()[0]}\n"
                "Run `cli.py db migrate` if the login tables don't exist yet."
            )
            raise typer.Exit(code=1) from None


def _print_one_time_pin(worker_code: str, name: str, pin: str) -> None:
    console.print(
        Panel.fit(
            f"Worker [bold]{worker_code}[/bold] ({name})\n"
            f"One-time PIN: [bold]{pin}[/bold]\n\n"
            "Hand this to the worker. They must choose their own PIN at first\n"
            "login. It is not stored anywhere and can't be shown again.",
            title="New PIN",
        )
    )


@auth_app.command("create-admin")
def auth_create_admin_cmd(
    email: str = typer.Option(..., "--email", help="The admin's email; used to sign in."),
    name: str = typer.Option(..., "--name", help="Name shown in the dashboard."),
) -> None:
    """
    Create a dashboard **admin**. Asks for the password twice (hidden).

    The only way to create the first account: there is no sign-up.
    """
    from src.auth.service import create_user
    from src.db.models import Role

    password = typer.prompt("Password (at least 12 characters)", hide_input=True,
                            confirmation_prompt=True)
    user = _run_account_command(
        lambda db: create_user(db, email, name, Role.ADMIN, password)
    )
    console.print(f"[green]Created admin {user.email}.[/green]")


@auth_app.command("create-worker")
def auth_create_worker_cmd(
    worker_id: str = typer.Option(..., "--worker-id", help="e.g. W-0042"),
    name: str = typer.Option(..., "--name", help="The worker's name."),
) -> None:
    """Create a worker and print their **one-time PIN**."""
    from src.auth.service import create_worker

    worker, pin = _run_account_command(lambda db: create_worker(db, worker_id, name))
    _print_one_time_pin(worker.worker_code, worker.display_name, pin)


@auth_app.command("reset-pin")
def auth_reset_pin_cmd(
    worker_id: str = typer.Option(..., "--worker-id", help="e.g. W-0042"),
) -> None:
    """
    Issue a new **one-time PIN**, unlock the worker, and sign them out everywhere.
    """
    from src.auth.service import reset_worker_pin

    worker, pin = _run_account_command(lambda db: reset_worker_pin(db, worker_id))
    _print_one_time_pin(worker.worker_code, worker.display_name, pin)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()

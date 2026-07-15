"""Benchmark trained YOLO models on the held-out test split.

Runs ultralytics validation for each trained checkpoint in ``models/`` and
collects accuracy, per-class AP, latency, and model size into a single
comparable structure. Results are saved as JSON under ``runs/benchmark/``
so report generation can run without re-evaluating.

Usage — Python API::

    from src.evaluation.benchmark import run_benchmark

    results = run_benchmark(["yolo26l", "yolo11l"], device="0")

Usage — CLI::

    uv run python -m src.evaluation.benchmark --model yolo26l --model yolo11l
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Windows CUDA VA-fragmentation mitigation — must precede any torch import.
# Same rationale as src/training/trainer.py; harmless on Linux.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

from loguru import logger
from pydantic import BaseModel, Field

from src.settings import settings

# Where trained weights live (written by src/training/trainer.py)
MODELS_DIR = Path("models")
# Where benchmark artifacts (JSON, plots, report) are written
BENCHMARK_DIR = Path("runs") / "benchmark"

# Default variants benchmarked when none are specified — the Phase 1 matrix.
DEFAULT_VARIANTS: tuple[str, ...] = ("yolo26l", "yolo11l", "yolo26n", "yolo11n")


class ModelBenchmark(BaseModel):
    """All benchmark measurements for a single trained model."""

    model_name: str = Field(description="Variant name, e.g. 'yolo26l'.")
    weights_path: str = Field(description="Path to the evaluated .pt checkpoint.")
    split: str = Field(description="Dataset split evaluated: 'test' or 'val'.")
    device: str = Field(description="Device used for evaluation.")

    # --- Accuracy ---
    map50: float = Field(description="mAP@0.5 across classes with test instances.")
    map50_95: float = Field(description="mAP@0.5:0.95 across classes with test instances.")
    precision: float = Field(description="Mean precision across classes.")
    recall: float = Field(description="Mean recall across classes.")
    per_class_ap50: dict[str, float | None] = Field(
        description="AP@0.5 per class name. None = class has no test instances."
    )

    # --- Efficiency ---
    size_mb: float = Field(description="Checkpoint file size in MB.")
    preprocess_ms: float = Field(description="Mean preprocess time per image (ms).")
    inference_ms: float = Field(description="Mean inference time per image (ms).")
    postprocess_ms: float = Field(description="Mean postprocess (NMS) time per image (ms).")

    @property
    def total_latency_ms(self) -> float:
        """End-to-end per-image latency in milliseconds."""
        return self.preprocess_ms + self.inference_ms + self.postprocess_ms

    @property
    def fps(self) -> float:
        """Throughput in frames per second implied by total latency."""
        total = self.total_latency_ms
        return 1000.0 / total if total > 0 else 0.0


def weights_path_for(model_name: str) -> Path:
    """Return the expected checkpoint path for a trained variant.

    Args:
        model_name: Variant name, e.g. ``"yolo26l"``.

    Returns:
        ``models/<model_name>_ppe_best.pt``.

    Raises:
        FileNotFoundError: If the checkpoint does not exist — the variant
            hasn't been trained (or weights weren't copied to ``models/``).
    """
    path = MODELS_DIR / f"{model_name}_ppe_best.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"No trained weights for '{model_name}' at {path}.\n"
            f"  Train it first: uv run python cli.py train --model {model_name}"
        )
    return path


def benchmark_model(
    model_name: str,
    device: str | None = None,
    split: str = "test",
) -> ModelBenchmark:
    """Evaluate one trained checkpoint on a dataset split.

    Args:
        model_name: Variant name, e.g. ``"yolo26l"``.
        device: Device string ('0', 'cpu'). Defaults to ``settings.ppb_device``.
        split: Dataset split to evaluate: ``"test"`` (default) or ``"val"``.
            Use ``"test"`` for the unbiased final comparison — the val split
            was used for checkpoint selection during training.

    Returns:
        Populated :class:`ModelBenchmark`.
    """
    from ultralytics import YOLO  # noqa: PLC0415 — heavy import deferred

    weights = weights_path_for(model_name)
    active_device = device or settings.ppb_device
    data_yaml = settings.configs_dir / "data.yaml"

    logger.info(f"[{model_name}] Evaluating {weights} on split='{split}' device={active_device}")

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        batch=8,  # small val batch — fits 4 GB laptop GPUs even for L models
        device=active_device,
        project=str(BENCHMARK_DIR),
        name=f"{model_name}_{split}",
        exist_ok=True,
        verbose=False,
    )

    # Per-class AP@0.5: metrics.box.ap50 only covers classes that had test
    # instances; ap_class_index maps those rows back to global class ids.
    names: dict[int, str] = metrics.names
    per_class: dict[str, float | None] = {name: None for name in names.values()}
    for row, class_id in enumerate(metrics.box.ap_class_index):
        per_class[names[int(class_id)]] = round(float(metrics.box.ap50[row]), 4)

    speed: dict[str, float] = metrics.speed  # ms per image

    result = ModelBenchmark(
        model_name=model_name,
        weights_path=str(weights),
        split=split,
        device=str(active_device),
        map50=round(float(metrics.box.map50), 4),
        map50_95=round(float(metrics.box.map), 4),
        precision=round(float(metrics.box.mp), 4),
        recall=round(float(metrics.box.mr), 4),
        per_class_ap50=per_class,
        size_mb=round(weights.stat().st_size / 1e6, 2),
        preprocess_ms=round(float(speed.get("preprocess", 0.0)), 2),
        inference_ms=round(float(speed.get("inference", 0.0)), 2),
        postprocess_ms=round(float(speed.get("postprocess", 0.0)), 2),
    )
    logger.success(
        f"[{model_name}] mAP50={result.map50:.4f}  mAP50-95={result.map50_95:.4f}  "
        f"latency={result.total_latency_ms:.1f}ms/img ({result.fps:.1f} FPS)  "
        f"size={result.size_mb}MB"
    )
    return result


def run_benchmark(
    model_names: list[str] | None = None,
    device: str | None = None,
    split: str = "test",
) -> list[ModelBenchmark]:
    """Benchmark multiple trained models and persist results to JSON.

    Args:
        model_names: Variants to evaluate. Defaults to every variant in
            :data:`DEFAULT_VARIANTS` that has trained weights in ``models/``.
        device: Device string forwarded to each evaluation.
        split: Dataset split to evaluate on.

    Returns:
        One :class:`ModelBenchmark` per successfully evaluated model.

    Raises:
        FileNotFoundError: If no trained weights are found at all.
    """
    if model_names is None:
        model_names = [m for m in DEFAULT_VARIANTS if (MODELS_DIR / f"{m}_ppe_best.pt").exists()]
        if not model_names:
            raise FileNotFoundError(
                f"No trained weights found in {MODELS_DIR}/. "
                "Train at least one model first: uv run python cli.py train --model yolo26l"
            )
        logger.info(f"Auto-discovered trained models: {model_names}")

    results = [benchmark_model(name, device=device, split=split) for name in model_names]

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    out_json = BENCHMARK_DIR / "benchmark_results.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "results": [r.model_dump() for r in results],
    }
    out_json.write_text(json.dumps(payload, indent=2))
    logger.success(f"Benchmark results saved → {out_json}")
    return results


def load_results(path: Path | None = None) -> list[ModelBenchmark]:
    """Load previously saved benchmark results from JSON.

    Args:
        path: JSON file written by :func:`run_benchmark`. Defaults to
            ``runs/benchmark/benchmark_results.json``.

    Returns:
        The deserialized benchmark list.

    Raises:
        FileNotFoundError: If the results file doesn't exist yet.
    """
    path = path or BENCHMARK_DIR / "benchmark_results.json"
    if not path.exists():
        raise FileNotFoundError(f"No benchmark results at {path}. Run the benchmark first.")
    payload = json.loads(path.read_text())
    return [ModelBenchmark(**r) for r in payload["results"]]


def main() -> None:
    """CLI entry point: benchmark models and regenerate plots + report."""
    import argparse

    from src.evaluation.report import generate_report
    from src.evaluation.visualize import generate_plots

    parser = argparse.ArgumentParser(description="PPB Vision — Model Benchmark")
    parser.add_argument(
        "--model", action="append", default=None,
        help="Variant to benchmark (repeatable). Default: all trained variants.",
    )
    parser.add_argument("--device", default=None, help="Device: '0' or 'cpu'.")
    parser.add_argument("--split", default="test", choices=("test", "val"))
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level=settings.log_level)

    results = run_benchmark(args.model, device=args.device, split=args.split)
    generate_plots(results)
    report_path = generate_report(results)
    logger.success(f"Benchmark complete → {report_path}")


if __name__ == "__main__":
    main()

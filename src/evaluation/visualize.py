"""Comparison plots for benchmark results.

Generates PNG charts under ``runs/benchmark/`` from a list of
:class:`~src.evaluation.benchmark.ModelBenchmark` results:

* ``accuracy_comparison.png`` — grouped bars: mAP50 and mAP50-95 per model.
* ``latency_vs_accuracy.png`` — scatter: per-image latency vs mAP50-95,
  point size ∝ checkpoint size. The production trade-off in one picture.
* ``per_class_ap.png`` — heatmap of AP@0.5 per class per model; classes
  with no test instances are greyed out.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")  # headless — never require a display
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from src.evaluation.benchmark import ModelBenchmark

BENCHMARK_DIR = Path("runs") / "benchmark"


def _plot_accuracy(results: list["ModelBenchmark"], out: Path) -> None:
    """Grouped bar chart of mAP50 / mAP50-95 per model."""
    names = [r.model_name for r in results]
    map50 = [r.map50 for r in results]
    map5095 = [r.map50_95 for r in results]

    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, map50, width, label="mAP@0.5")
    bars2 = ax.bar(x + width / 2, map5095, width, label="mAP@0.5:0.95")
    ax.bar_label(bars1, fmt="%.3f", padding=2, fontsize=8)
    ax.bar_label(bars2, fmt="%.3f", padding=2, fontsize=8)

    ax.set_ylabel("mAP")
    ax.set_title(f"Accuracy on {results[0].split} split")
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_latency_vs_accuracy(results: list["ModelBenchmark"], out: Path) -> None:
    """Scatter of latency vs accuracy — point size encodes model size."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in results:
        ax.scatter(
            r.total_latency_ms, r.map50_95,
            s=max(r.size_mb * 6, 40), alpha=0.7, edgecolors="black", linewidths=0.8,
        )
        ax.annotate(
            f"{r.model_name}\n{r.size_mb:.0f} MB",
            (r.total_latency_ms, r.map50_95),
            textcoords="offset points", xytext=(10, 6), fontsize=9,
        )
    ax.set_xlabel(f"Latency per image (ms) — device: {results[0].device}")
    ax.set_ylabel("mAP@0.5:0.95")
    ax.set_title("Speed vs accuracy trade-off (bubble size = checkpoint size)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_per_class_ap(results: list["ModelBenchmark"], out: Path) -> None:
    """Heatmap of per-class AP@0.5 across models."""
    class_names = list(results[0].per_class_ap50.keys())
    matrix = np.full((len(results), len(class_names)), np.nan)
    for i, r in enumerate(results):
        for j, cls in enumerate(class_names):
            ap = r.per_class_ap50.get(cls)
            if ap is not None:
                matrix[i, j] = ap

    fig, ax = plt.subplots(figsize=(max(9, len(class_names) * 0.9), 1.2 + len(results) * 0.7))
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad(color="#d0d0d0")  # grey = class has no test instances
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(class_names)), class_names, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(results)), [r.model_name for r in results], fontsize=10)
    for i in range(len(results)):
        for j in range(len(class_names)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="#606060")

    fig.colorbar(im, ax=ax, label="AP@0.5", shrink=0.8)
    ax.set_title("Per-class AP@0.5 (grey = no test instances)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def generate_plots(results: list["ModelBenchmark"], out_dir: Path | None = None) -> list[Path]:
    """Render all comparison plots for a benchmark run.

    Args:
        results: Benchmark results (at least one model).
        out_dir: Output directory. Defaults to ``runs/benchmark/``.

    Returns:
        Paths of the PNG files written.
    """
    out_dir = out_dir or BENCHMARK_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = [
        (out_dir / "accuracy_comparison.png", _plot_accuracy),
        (out_dir / "latency_vs_accuracy.png", _plot_latency_vs_accuracy),
        (out_dir / "per_class_ap.png", _plot_per_class_ap),
    ]
    written: list[Path] = []
    for path, fn in outputs:
        fn(results, path)
        written.append(path)
        logger.info(f"Plot saved → {path}")
    return written

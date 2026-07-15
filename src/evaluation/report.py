"""Markdown benchmark report generation.

Turns a list of :class:`~src.evaluation.benchmark.ModelBenchmark` results
into ``runs/benchmark_report.md`` — the deliverable that drives the
production model decision (agents.md §8).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from src.evaluation.benchmark import ModelBenchmark

REPORT_PATH = Path("runs") / "benchmark_report.md"

# Classes where a miss means an undetected safety violation — weighted
# heavily in the recommendation notes.
VIOLATION_CLASSES: tuple[str, ...] = ("no_hardhat", "no_vest")


def _fmt(v: float | None, spec: str = ".4f") -> str:
    return "—" if v is None else format(v, spec)


def _summary_table(results: list["ModelBenchmark"]) -> str:
    best_acc = max(r.map50_95 for r in results)
    fastest = min(r.total_latency_ms for r in results)
    lines = [
        "| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Latency (ms/img) | FPS | Size (MB) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        acc_mark = " 🥇" if r.map50_95 == best_acc else ""
        fast_mark = " ⚡" if r.total_latency_ms == fastest else ""
        lines.append(
            f"| {r.model_name} | {_fmt(r.map50)} | {_fmt(r.map50_95)}{acc_mark} "
            f"| {_fmt(r.precision)} | {_fmt(r.recall)} "
            f"| {r.total_latency_ms:.1f}{fast_mark} | {r.fps:.1f} | {r.size_mb:.1f} |"
        )
    return "\n".join(lines)


def _per_class_table(results: list["ModelBenchmark"]) -> str:
    class_names = list(results[0].per_class_ap50.keys())
    header = "| Class | " + " | ".join(r.model_name for r in results) + " |"
    sep = "|---|" + "---|" * len(results)
    lines = [header, sep]
    for cls in class_names:
        flag = " ⚠️" if cls in VIOLATION_CLASSES else ""
        row = [f"| {cls}{flag} "]
        for r in results:
            row.append(f"| {_fmt(r.per_class_ap50.get(cls), '.3f')} ")
        lines.append("".join(row) + "|")
    return "\n".join(lines)


def _latency_table(results: list["ModelBenchmark"]) -> str:
    lines = [
        "| Model | Preprocess | Inference | Postprocess (NMS) | Total |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.model_name} | {r.preprocess_ms:.2f} ms | {r.inference_ms:.2f} ms "
            f"| {r.postprocess_ms:.2f} ms | {r.total_latency_ms:.2f} ms |"
        )
    return "\n".join(lines)


def _verdicts(results: list["ModelBenchmark"]) -> str:
    by_acc = max(results, key=lambda r: r.map50_95)
    by_speed = min(results, key=lambda r: r.total_latency_ms)
    by_size = min(results, key=lambda r: r.size_mb)

    # Recall on violation classes drives the compliance recommendation:
    # a missed violation is the costly failure mode for this product.
    def violation_ap(r: "ModelBenchmark") -> float:
        vals = [r.per_class_ap50.get(c) for c in VIOLATION_CLASSES]
        known = [v for v in vals if v is not None]
        return sum(known) / len(known) if known else 0.0

    by_violation = max(results, key=violation_ap)

    untrained = [c for c, v in results[0].per_class_ap50.items() if v is None]
    lines = [
        f"- **Best accuracy**: `{by_acc.model_name}` (mAP50-95 = {by_acc.map50_95:.4f})",
        f"- **Fastest**: `{by_speed.model_name}` ({by_speed.total_latency_ms:.1f} ms/img, "
        f"{by_speed.fps:.0f} FPS on device {by_speed.device})",
        f"- **Smallest**: `{by_size.model_name}` ({by_size.size_mb:.1f} MB) — edge candidate",
        f"- **Best on violation classes** (`{'`, `'.join(VIOLATION_CLASSES)}`): "
        f"`{by_violation.model_name}` (mean AP@0.5 = {violation_ap(by_violation):.3f})",
    ]
    if untrained:
        lines.append(
            f"- ⚠️ **Classes with no test instances / never trained**: "
            f"`{'`, `'.join(untrained)}` — the model cannot detect these. "
            "They must be excluded from the Job PPE Matrix until data exists."
        )
    return "\n".join(lines)


def generate_report(
    results: list["ModelBenchmark"], out_path: Path | None = None
) -> Path:
    """Write the benchmark comparison report.

    Args:
        results: Benchmark results (at least one model).
        out_path: Output markdown path. Defaults to ``runs/benchmark_report.md``.

    Returns:
        The path written.
    """
    out_path = out_path or REPORT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    md = f"""# PPB Vision — Model Benchmark Report

Generated: {generated}
Split: **{results[0].split}** | Device: **{results[0].device}**

> Latency numbers are specific to the benchmark device and batch size 1;
> re-measure on the production target before finalizing SLAs.

## Summary

{_summary_table(results)}

## Verdicts

{_verdicts(results)}

## Latency breakdown (per image)

{_latency_table(results)}

Note: YOLO26 variants are NMS-free — their postprocess column reflects the
architectural advantage most visible on CPU targets.

## Per-class AP@0.5

⚠️ = violation class (miss ⇒ undetected safety breach). “—” = no test instances.

{_per_class_table(results)}

## Plots

![Accuracy comparison](benchmark/accuracy_comparison.png)
![Latency vs accuracy](benchmark/latency_vs_accuracy.png)
![Per-class AP](benchmark/per_class_ap.png)
"""
    out_path.write_text(md, encoding="utf-8")
    logger.success(f"Benchmark report written → {out_path}")
    return out_path

"""Model evaluation and benchmarking for the PPB Vision Module.

Compares trained YOLO26 vs YOLO11 checkpoints on the held-out test split:
accuracy (mAP), per-class AP, inference latency, and model size — then
generates a markdown report to drive the production model decision.
"""

from src.evaluation.benchmark import ModelBenchmark, run_benchmark
from src.evaluation.report import generate_report
from src.evaluation.visualize import generate_plots

__all__ = ["ModelBenchmark", "run_benchmark", "generate_report", "generate_plots"]

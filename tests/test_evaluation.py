"""Unit tests for src/evaluation — report/plots/persistence with synthetic data."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation.benchmark import ModelBenchmark, load_results, weights_path_for
from src.evaluation.report import generate_report
from src.evaluation.visualize import generate_plots


def make_benchmark(name: str, map50_95: float, latency: float, size: float) -> ModelBenchmark:
    return ModelBenchmark(
        model_name=name,
        weights_path=f"models/{name}_ppe_best.pt",
        split="test",
        device="cpu",
        map50=map50_95 + 0.25,
        map50_95=map50_95,
        precision=0.61,
        recall=0.79,
        per_class_ap50={
            "hardhat": 0.9, "no_hardhat": 0.82, "safety_vest": 0.66,
            "no_vest": 0.24, "person": 0.83, "boots": None, "harness": None,
        },
        size_mb=size,
        preprocess_ms=1.5,
        inference_ms=latency,
        postprocess_ms=0.5,
    )


@pytest.fixture
def results() -> list[ModelBenchmark]:
    return [
        make_benchmark("yolo11l", 0.4714, 50.0, 51.2),
        make_benchmark("yolo26n", 0.4223, 5.4, 5.4),
    ]


class TestModelBenchmark:
    def test_latency_and_fps_properties(self, results) -> None:
        r = results[0]
        assert r.total_latency_ms == pytest.approx(52.0)
        assert r.fps == pytest.approx(1000 / 52.0)

    def test_weights_path_for_missing_model_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="yolo99x"):
            weights_path_for("yolo99x")


class TestPersistence:
    def test_load_results_roundtrip(self, results, tmp_path: Path) -> None:
        out = tmp_path / "benchmark_results.json"
        out.write_text(json.dumps({
            "generated_at": "2026-07-15T00:00:00Z",
            "split": "test",
            "results": [r.model_dump() for r in results],
        }))
        loaded = load_results(out)
        assert [r.model_name for r in loaded] == ["yolo11l", "yolo26n"]
        assert loaded[0].map50_95 == pytest.approx(0.4714)

    def test_load_results_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_results(tmp_path / "nope.json")


class TestReport:
    def test_report_contains_verdicts_and_warnings(self, results, tmp_path: Path) -> None:
        report = generate_report(results, out_path=tmp_path / "report.md")
        text = report.read_text(encoding="utf-8")

        assert "yolo11l" in text and "yolo26n" in text
        assert "Best accuracy" in text and "`yolo11l`" in text
        assert "Fastest" in text and "`yolo26n`" in text
        # Untrained classes must be flagged, never silently dropped
        assert "boots" in text and "harness" in text
        # Violation classes marked
        assert "no_vest ⚠️" in text

    def test_report_marks_best_model(self, results, tmp_path: Path) -> None:
        text = generate_report(results, out_path=tmp_path / "r.md").read_text(encoding="utf-8")
        best_line = next(line for line in text.splitlines() if "| yolo11l |" in line)
        assert "🥇" in best_line


class TestPlots:
    def test_generates_all_three_plots(self, results, tmp_path: Path) -> None:
        written = generate_plots(results, out_dir=tmp_path)
        assert [p.name for p in written] == [
            "accuracy_comparison.png",
            "latency_vs_accuracy.png",
            "per_class_ap.png",
        ]
        for path in written:
            assert path.exists() and path.stat().st_size > 1000

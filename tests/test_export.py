"""Unit tests for src/export — path/validation logic without real exports."""

from __future__ import annotations

from pathlib import Path

import pytest

import src.export.onnx_export as onnx_export
from src.export.onnx_export import SUPPORTED_FORMATS, export_model
from src.export.quantize import quantize_onnx_int8


class TestExportValidation:
    def test_unsupported_format_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported format"):
            export_model("yolo11l", fmt="coreml")

    def test_missing_weights_raises_with_train_hint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(onnx_export, "MODELS_DIR", tmp_path)
        with pytest.raises(FileNotFoundError, match="train --model yolo11l"):
            export_model("yolo11l")

    def test_supported_formats_cover_agents_md_targets(self) -> None:
        assert {"onnx", "torchscript", "tflite", "engine"} == set(SUPPORTED_FORMATS)


class TestExportFlow:
    def test_artifact_moved_to_exports_with_precision_suffix(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """export_model must relocate ultralytics output into models/exports/."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "yolo26n_ppe_best.pt").write_bytes(b"fake-weights")
        produced = tmp_path / "raw_export.onnx"

        class _FakeYOLO:
            def __init__(self, weights: str) -> None:
                self.weights = weights

            def export(self, **kwargs) -> str:
                produced.write_bytes(b"fake-onnx" * 100)
                self.export_kwargs = kwargs
                return str(produced)

        import ultralytics

        monkeypatch.setattr(ultralytics, "YOLO", _FakeYOLO)
        monkeypatch.setattr(onnx_export, "MODELS_DIR", models_dir)
        monkeypatch.setattr(onnx_export, "EXPORTS_DIR", models_dir / "exports")

        dest = export_model("yolo26n", fmt="onnx", half=True, imgsz=416)

        assert dest == models_dir / "exports" / "yolo26n_ppe_fp16.onnx"
        assert dest.exists()
        assert not produced.exists()  # moved, not copied


class TestQuantize:
    def test_missing_onnx_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            quantize_onnx_int8(tmp_path / "ghost.onnx")

    def test_output_path_gets_int8_suffix(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        src = tmp_path / "model.onnx"
        src.write_bytes(b"x" * 1000)

        def fake_quantize_dynamic(model_input: str, model_output: str, weight_type) -> None:
            Path(model_output).write_bytes(b"y" * 250)

        import onnxruntime.quantization

        monkeypatch.setattr(
            onnxruntime.quantization, "quantize_dynamic", fake_quantize_dynamic
        )
        out = quantize_onnx_int8(src)
        assert out.name == "model_int8.onnx"
        assert out.exists()

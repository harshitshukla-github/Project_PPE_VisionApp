"""Unit tests for src/inference/detector.py — model I/O without a real model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import src.inference.detector as detector_module
from src.inference.detector import PPEDetector


class _FakeBox:
    def __init__(self, cls_id: int, conf: float, xyxy: list[float]) -> None:
        self.cls = [cls_id]
        self.conf = [conf]
        self.xyxy = [np.array(xyxy)]


class _FakeResult:
    def __init__(self, boxes: list[_FakeBox]) -> None:
        self.boxes = boxes


class _FakeYOLO:
    """Mimics the slice of ultralytics.YOLO the detector uses."""

    def __init__(self, weights: str) -> None:
        self.weights = weights
        self.names = {0: "hardhat", 1: "no_hardhat", 2: "safety_vest"}
        self.last_predict_kwargs: dict = {}

    def predict(self, source, **kwargs) -> list[_FakeResult]:
        self.last_predict_kwargs = kwargs
        return [
            _FakeResult([
                _FakeBox(2, 0.55, [1.0, 2.0, 3.0, 4.0]),
                _FakeBox(0, 0.91, [10.0, 20.0, 30.0, 40.0]),
            ])
        ]


@pytest.fixture
def patched_detector(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> PPEDetector:
    """PPEDetector wired to _FakeYOLO with a dummy weights file."""
    import ultralytics

    weights = tmp_path / "yolo11l_ppe_best.pt"
    weights.write_bytes(b"fake")
    monkeypatch.setattr(ultralytics, "YOLO", _FakeYOLO)
    return PPEDetector(model_name="yolo11l", weights_path=weights, device="cpu")


class TestInit:
    def test_missing_weights_raises_with_train_hint(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(detector_module, "MODELS_DIR", tmp_path)
        with pytest.raises(FileNotFoundError, match="train --model yolo26x"):
            PPEDetector(model_name="yolo26x")

    def test_settings_defaults_applied(self, patched_detector: PPEDetector) -> None:
        from src.settings import settings

        assert patched_detector.confidence == settings.ppb_confidence_threshold
        assert patched_detector.iou == settings.ppb_iou_threshold
        assert patched_detector.class_names == {0: "hardhat", 1: "no_hardhat", 2: "safety_vest"}


class TestDetect:
    def test_detections_parsed_and_sorted_by_confidence(
        self, patched_detector: PPEDetector, sample_image: np.ndarray
    ) -> None:
        detections, elapsed_ms = patched_detector.detect(sample_image)

        assert [d.class_name for d in detections] == ["hardhat", "safety_vest"]
        assert detections[0].confidence == pytest.approx(0.91)
        assert detections[1].bbox == [1.0, 2.0, 3.0, 4.0]
        assert elapsed_ms >= 0  # fake model can round to 0.0 ms

    def test_thresholds_forwarded_to_predict(
        self, patched_detector: PPEDetector, sample_image: np.ndarray
    ) -> None:
        patched_detector.detect(sample_image)
        kwargs = patched_detector._model.last_predict_kwargs
        assert kwargs["conf"] == patched_detector.confidence
        assert kwargs["iou"] == patched_detector.iou
        assert kwargs["device"] == "cpu"

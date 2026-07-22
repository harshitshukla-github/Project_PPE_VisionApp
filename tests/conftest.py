"""Shared pytest fixtures for the PPB Vision test suite.

Design rule: no test loads a real YOLO model, touches the network, or
needs a GPU. The detector and Gemini are replaced with fakes so the suite
runs in seconds on any machine (including CI).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

from src.inference.schemas import Detection


# ---------------------------------------------------------------------------
# Detections
# ---------------------------------------------------------------------------

def make_detection(class_name: str, confidence: float = 0.9) -> Detection:
    """Build a Detection with a dummy bbox."""
    return Detection(
        class_name=class_name, confidence=confidence, bbox=[10.0, 10.0, 100.0, 200.0]
    )


@pytest.fixture
def compliant_detections() -> list[Detection]:
    """Worker wearing hardhat + vest — passes general_labor."""
    return [make_detection("hardhat", 0.92), make_detection("safety_vest", 0.88)]


@pytest.fixture
def missing_vest_detections() -> list[Detection]:
    """Worker with hardhat only — fails any job requiring a vest."""
    return [make_detection("hardhat", 0.85)]


@pytest.fixture
def violation_detections() -> list[Detection]:
    """Hardhat AND no_hardhat detected (multi-person frame) — must FAIL."""
    return [
        make_detection("hardhat", 0.9),
        make_detection("safety_vest", 0.87),
        make_detection("no_hardhat", 0.8),
    ]


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_image() -> np.ndarray:
    """Small BGR image with structure (not all-zero)."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)


@pytest.fixture
def sample_image_bytes(sample_image: np.ndarray) -> bytes:
    """The sample image encoded as PNG — for upload endpoints."""
    import cv2

    ok, buf = cv2.imencode(".png", sample_image)
    assert ok
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Fake detector (no model load)
# ---------------------------------------------------------------------------

class FakeDetector:
    """Stands in for PPEDetector in API tests.

    Attribute-compatible with the parts of PPEDetector the API touches.
    ``detections`` can be swapped per-test to steer the verdict.
    """

    def __init__(self, detections: list[Detection] | None = None) -> None:
        self.model_name = "yolo11l"
        self.device = "cpu"
        self.confidence = 0.45
        self.iou = 0.50
        self.weights_path = Path("models/yolo11l_ppe_best.pt")
        self.class_names = {0: "hardhat", 1: "no_hardhat", 2: "safety_vest", 11: "person"}
        self.detections: list[Detection] = detections or []

    def detect(self, image: object) -> tuple[list[Detection], float]:
        return list(self.detections), 12.3


@pytest.fixture
def fake_detector() -> FakeDetector:
    return FakeDetector()


@pytest.fixture
def api_client(fake_detector: FakeDetector, monkeypatch: pytest.MonkeyPatch) -> Iterator:
    """FastAPI TestClient wired to the fake detector.

    Patches both the lifespan hook (so startup doesn't load a real model)
    and the deps singleton (so get_detector returns the fake).
    """
    from fastapi.testclient import TestClient

    from src.api import deps, main

    monkeypatch.setattr(main, "init_detector", lambda model_name=None, device=None: fake_detector)
    monkeypatch.setattr(deps, "_detector", fake_detector)

    with TestClient(main.app) as client:
        yield client

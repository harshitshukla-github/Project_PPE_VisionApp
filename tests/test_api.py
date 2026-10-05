"""Integration tests for the FastAPI endpoints (fake detector, no model)."""

from __future__ import annotations

import pytest

from src.settings import settings
from tests.conftest import make_detection


@pytest.fixture(autouse=True)
def offline_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the deterministic fallback explainer — tests never call Gemini."""
    import src.inference.gemini_explainer as explainer

    monkeypatch.setattr(settings, "google_api_key", "")
    explainer._cached_explain.cache_clear()


class TestHealthAndInfo:
    def test_health(self, api_client) -> None:
        response = api_client.get("/api/v1/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["model_name"] == "yolo11l"

    def test_model_info(self, api_client) -> None:
        response = api_client.get("/api/v1/model/info")
        assert response.status_code == 200
        body = response.json()
        assert body["confidence_threshold"] == 0.45
        assert body["class_names"]["0"] == "hardhat"

    def test_jobs_lists_matrix(self, api_client) -> None:
        response = api_client.get("/api/v1/jobs")
        assert response.status_code == 200
        body = response.json()
        assert body["welder"]["required"] == ["hardhat", "safety_vest", "goggles", "gloves"]
        assert "face_shield" in body["welder"]["pending"]


class TestDetect:
    def test_detect_returns_boxes(self, api_client, fake_detector, sample_image_bytes) -> None:
        fake_detector.detections = [make_detection("hardhat", 0.9)]
        response = api_client.post(
            "/api/v1/detect",
            files={"image": ("worker.png", sample_image_bytes, "image/png")},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["detections"][0]["class"] == "hardhat"
        assert body["image_width"] == 64
        assert body["model_used"] == "yolo11l"

    def test_undecodable_image_is_400(self, api_client) -> None:
        response = api_client.post(
            "/api/v1/detect",
            files={"image": ("junk.png", b"this is not an image", "image/png")},
        )
        assert response.status_code == 400
        assert "decode" in response.json()["detail"].lower()

    def test_empty_upload_is_400(self, api_client) -> None:
        response = api_client.post(
            "/api/v1/detect", files={"image": ("empty.png", b"", "image/png")}
        )
        assert response.status_code == 400


class TestComplianceCheck:
    def _check(self, api_client, image_bytes: bytes, job: str = "general_labor") -> object:
        return api_client.post(
            "/api/v1/compliance/check",
            files={"image": ("worker.png", image_bytes, "image/png")},
            data={"worker_id": "W-0042", "job_category": job},
        )

    def test_pass_verdict_skips_gemini(
        self, api_client, fake_detector, sample_image_bytes
    ) -> None:
        fake_detector.detections = [
            make_detection("hardhat", 0.92),
            make_detection("safety_vest", 0.88),
        ]
        response = self._check(api_client, sample_image_bytes)
        assert response.status_code == 200
        body = response.json()
        assert body["verdict"] == "PASS"
        assert body["gemini_invoked"] is False
        assert body["explanation"] is None
        assert body["missing_equipment"] == []
        assert body["unverifiable_equipment"] == ["boots"]

    def test_fail_verdict_includes_explanation(
        self, api_client, fake_detector, sample_image_bytes
    ) -> None:
        """FAIL needs a violation box, so Gemini explains a real finding."""
        fake_detector.detections = [
            make_detection("hardhat", 0.85),
            make_detection("no_vest", 0.79),
        ]
        response = self._check(api_client, sample_image_bytes)
        body = response.json()
        assert body["verdict"] == "FAIL"
        assert body["violations_detected"] == ["no_vest"]
        assert body["gemini_invoked"] is True
        assert body["explanation"]  # fallback text present

    def test_undetected_item_is_review_and_skips_gemini(
        self, api_client, fake_detector, sample_image_bytes
    ) -> None:
        """A merely-undetected vest must not accuse, and must not spend money.

        This is the reported defect: the worker was wearing a vest the model
        missed. It is now NEEDS_REVIEW, and Gemini is not called — there is no
        finding to explain, and these are far more common than FAIL.
        """
        fake_detector.detections = [make_detection("hardhat", 0.85)]
        response = self._check(api_client, sample_image_bytes)
        body = response.json()
        assert body["verdict"] == "NEEDS_REVIEW"
        assert body["missing_equipment"] == ["safety_vest"]
        assert body["violations_detected"] == []
        assert body["gemini_invoked"] is False
        assert body["explanation"] is None

    def test_violation_detection_fails_worker(
        self, api_client, fake_detector, sample_image_bytes
    ) -> None:
        fake_detector.detections = [
            make_detection("hardhat", 0.9),
            make_detection("safety_vest", 0.9),
            make_detection("no_hardhat", 0.75),
        ]
        response = self._check(api_client, sample_image_bytes)
        body = response.json()
        assert body["verdict"] == "FAIL"
        assert body["violations_detected"] == ["no_hardhat"]

    def test_unknown_job_is_422_with_choices(self, api_client, sample_image_bytes) -> None:
        response = self._check(api_client, sample_image_bytes, job="astronaut")
        assert response.status_code == 422
        assert "general_labor" in response.json()["detail"]

    def test_response_echoes_worker_and_job(
        self, api_client, fake_detector, sample_image_bytes
    ) -> None:
        fake_detector.detections = []
        response = self._check(api_client, sample_image_bytes, job="welder")
        body = response.json()
        assert body["worker_id"] == "W-0042"
        assert body["job_category"] == "welder"
        assert body["confidence_avg"] == 0.0

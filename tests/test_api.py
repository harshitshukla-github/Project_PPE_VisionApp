"""Integration tests for the FastAPI endpoints (fake detector, no model)."""

from __future__ import annotations

import pytest

import src.api.routes as routes
from src.inference.schemas import GeminiVerification, ItemStatus, ItemVerification
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

    def test_undetected_item_stays_review_when_gemini_unavailable(
        self, api_client, fake_detector, sample_image_bytes
    ) -> None:
        """No Gemini (no key, outage) must never turn doubt into a verdict.

        The worker was wearing a vest the detector missed. Without a second
        opinion that stays NEEDS_REVIEW: not an accusation, not a free pass.
        """
        fake_detector.detections = [make_detection("hardhat", 0.85)]
        response = self._check(api_client, sample_image_bytes)
        body = response.json()
        assert body["verdict"] == "NEEDS_REVIEW"
        assert body["missing_equipment"] == ["safety_vest"]
        assert body["violations_detected"] == []
        assert body["gemini_invoked"] is False
        assert body["explanation"] is None
        assert body["verification"] is None
        assert body["verdict_source"] == "detector"


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


class TestGeminiSecondOpinion:
    """End to end: the detector is unsure, Gemini looks at the photo."""

    @staticmethod
    def _check(client, image_bytes):
        return client.post(
            "/api/v1/compliance/check",
            files={"image": ("worker.png", image_bytes, "image/png")},
            data={"worker_id": "W-0042", "job_category": "general_labor"},
        )

    @staticmethod
    def _fake_verifier(monkeypatch, status: str, confidence: float = 0.92) -> list[tuple]:
        """Stand in for Gemini; records what the route asked it to check."""
        calls: list[tuple] = []

        def fake(image, job_category, items):
            calls.append((job_category, list(items)))
            return GeminiVerification(
                model="gemini-test",
                elapsed_ms=5.0,
                items=[
                    ItemVerification(item=i, status=ItemStatus(status),
                                     confidence=confidence, reason="test")
                    for i in items
                ],
            )

        monkeypatch.setattr(routes, "verify_items", fake)
        return calls

    def test_gemini_sees_the_vest_and_worker_passes(
        self, api_client, fake_detector, sample_image_bytes, monkeypatch
    ) -> None:
        """The reported defect, end to end: missed by YOLO, cleared by Gemini."""
        calls = self._fake_verifier(monkeypatch, "worn")
        fake_detector.detections = [make_detection("hardhat", 0.85)]
        body = self._check(api_client, sample_image_bytes).json()
        assert body["verdict"] == "PASS"
        assert body["missing_equipment"] == []
        assert body["verdict_source"] == "detector+gemini"
        assert body["gemini_invoked"] is True
        assert body["explanation"] is None
        assert body["verification"]["items"][0]["status"] == "worn"
        assert calls == [("general_labor", ["safety_vest"])]

    def test_gemini_sees_no_vest_and_worker_fails_with_explanation(
        self, api_client, fake_detector, sample_image_bytes, monkeypatch
    ) -> None:
        self._fake_verifier(monkeypatch, "not_worn")
        fake_detector.detections = [make_detection("hardhat", 0.85)]
        body = self._check(api_client, sample_image_bytes).json()
        assert body["verdict"] == "FAIL"
        assert body["missing_equipment"] == ["safety_vest"]
        assert body["verdict_source"] == "detector+gemini"
        assert body["explanation"]

    def test_gemini_unsure_stays_review(
        self, api_client, fake_detector, sample_image_bytes, monkeypatch
    ) -> None:
        self._fake_verifier(monkeypatch, "uncertain")
        fake_detector.detections = [make_detection("hardhat", 0.85)]
        body = self._check(api_client, sample_image_bytes).json()
        assert body["verdict"] == "NEEDS_REVIEW"
        assert body["missing_equipment"] == ["safety_vest"]
        assert body["gemini_invoked"] is True
        assert body["verification"] is not None

    def test_hesitant_gemini_cannot_clear_a_worker(
        self, api_client, fake_detector, sample_image_bytes, monkeypatch
    ) -> None:
        self._fake_verifier(monkeypatch, "worn", confidence=0.5)
        fake_detector.detections = [make_detection("hardhat", 0.85)]
        assert self._check(api_client, sample_image_bytes).json()["verdict"] == "NEEDS_REVIEW"

    def test_gemini_not_consulted_when_detector_is_sure(
        self, api_client, fake_detector, sample_image_bytes, monkeypatch
    ) -> None:
        """PASS and detector FAIL never trigger the vision call."""
        calls = self._fake_verifier(monkeypatch, "worn")
        fake_detector.detections = [
            make_detection("hardhat", 0.9), make_detection("safety_vest", 0.9),
        ]
        assert self._check(api_client, sample_image_bytes).json()["verdict"] == "PASS"
        fake_detector.detections = [
            make_detection("hardhat", 0.9), make_detection("no_vest", 0.8),
        ]
        assert self._check(api_client, sample_image_bytes).json()["verdict"] == "FAIL"
        assert calls == []

    def test_only_undetected_items_are_sent_for_checking(
        self, api_client, fake_detector, sample_image_bytes, monkeypatch
    ) -> None:
        calls = self._fake_verifier(monkeypatch, "worn")
        fake_detector.detections = [make_detection("safety_vest", 0.9)]
        self._check(api_client, sample_image_bytes)
        assert calls == [("general_labor", ["hardhat"])]

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

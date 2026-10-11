"""Unit tests for src/inference/gemini_verifier.py — no real API calls.

The verifier can clear a worker or accuse one, so the rule under test is that
anything short of a clean, confident answer changes nothing: failures return
None, and a malformed finding is read as "uncertain".
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import cv2
import google.genai
import numpy as np
import pytest

import src.inference.gemini_verifier as verifier
from src.inference.schemas import ItemStatus
from src.settings import settings


@pytest.fixture
def photo() -> np.ndarray:
    """A frame large enough to be verified (the shared sample_image is 64px)."""
    rng = np.random.default_rng(7)
    return rng.integers(0, 255, size=(640, 480, 3), dtype=np.uint8)


def reply(*entries: dict) -> str:
    return json.dumps({"items": list(entries)})


def install_fake_gemini(monkeypatch: pytest.MonkeyPatch, text: str | Exception) -> list[dict]:
    """Replace google.genai.Client; returns a list that records each call."""
    calls: list[dict] = []

    class _Models:
        def generate_content(self, **kwargs: object) -> SimpleNamespace:
            calls.append(kwargs)
            if isinstance(text, Exception):
                raise text
            return SimpleNamespace(text=text)

    class _Client:
        def __init__(self, **kwargs: object) -> None:
            self.models = _Models()

    monkeypatch.setattr(google.genai, "Client", _Client)
    monkeypatch.setattr(settings, "google_api_key", "fake-key")
    monkeypatch.setattr(settings, "ppb_gemini_verify", True)
    return calls


class TestParseReply:
    def test_reads_status_confidence_and_reason(self) -> None:
        text = reply({"item": "safety_vest", "status": "worn", "confidence": 0.93,
                      "reason": "Orange hi-vis vest on torso."})
        (finding,) = verifier.parse_reply(text, ["safety_vest"])
        assert finding.status is ItemStatus.WORN
        assert finding.confidence == 0.93
        assert "vest" in finding.reason

    def test_item_missing_from_reply_is_uncertain(self) -> None:
        text = reply({"item": "hardhat", "status": "worn", "confidence": 0.9, "reason": ""})
        findings = verifier.parse_reply(text, ["hardhat", "gloves"])
        assert findings[1].item == "gloves"
        assert findings[1].status is ItemStatus.UNCERTAIN
        assert findings[1].confidence == 0.0

    def test_unknown_status_is_uncertain(self) -> None:
        text = reply({"item": "gloves", "status": "probably", "confidence": 0.9, "reason": ""})
        assert verifier.parse_reply(text, ["gloves"])[0].status is ItemStatus.UNCERTAIN

    def test_non_numeric_confidence_is_uncertain(self) -> None:
        text = reply({"item": "gloves", "status": "worn", "confidence": "high", "reason": ""})
        (finding,) = verifier.parse_reply(text, ["gloves"])
        assert finding.status is ItemStatus.UNCERTAIN
        assert finding.confidence == 0.0

    def test_confidence_is_clamped_to_unit_range(self) -> None:
        text = reply({"item": "gloves", "status": "worn", "confidence": 7, "reason": ""})
        assert verifier.parse_reply(text, ["gloves"])[0].confidence == 1.0

    def test_item_names_match_despite_spelling(self) -> None:
        text = reply({"item": "Safety Vest", "status": "not_worn", "confidence": 0.8, "reason": ""})
        (finding,) = verifier.parse_reply(text, ["safety_vest"])
        assert finding.item == "safety_vest"
        assert finding.status is ItemStatus.NOT_WORN

    def test_items_gemini_was_not_asked_about_are_dropped(self) -> None:
        text = reply(
            {"item": "gloves", "status": "worn", "confidence": 0.9, "reason": ""},
            {"item": "harness", "status": "not_worn", "confidence": 0.9, "reason": ""},
        )
        assert [f.item for f in verifier.parse_reply(text, ["gloves"])] == ["gloves"]

    @pytest.mark.parametrize("text", ["not json", '"just a string"', '{"items": "nope"}'])
    def test_unusable_reply_raises(self, text: str) -> None:
        with pytest.raises(ValueError):
            verifier.parse_reply(text, ["gloves"])


class TestPromptAndImage:
    def test_prompt_lists_each_item_with_its_description(self) -> None:
        prompt = verifier.build_prompt("welder", ["gloves", "safety_vest"])
        assert "- gloves:" in prompt and "- safety_vest:" in prompt
        assert "reflective jacket" in prompt
        assert "welder" in prompt

    def test_prompt_never_reveals_what_the_detector_found(self) -> None:
        """The second opinion must be independent of the first."""
        prompt = verifier.build_prompt("welder", ["gloves"]).lower()
        assert "hardhat" not in prompt  # an item that was detected is not mentioned

    def test_large_image_is_downscaled(self, sample_image: np.ndarray) -> None:
        big = cv2.resize(sample_image, (4000, 3000))
        decoded = cv2.imdecode(
            np.frombuffer(verifier.encode_for_gemini(big), np.uint8), cv2.IMREAD_COLOR
        )
        assert max(decoded.shape[:2]) == verifier.MAX_EDGE_PX

    def test_small_image_is_not_upscaled(self, sample_image: np.ndarray) -> None:
        decoded = cv2.imdecode(
            np.frombuffer(verifier.encode_for_gemini(sample_image), np.uint8), cv2.IMREAD_COLOR
        )
        assert decoded.shape[:2] == sample_image.shape[:2]


class TestVerifyItems:
    def test_returns_findings_and_sends_the_image(
        self, monkeypatch: pytest.MonkeyPatch, photo: np.ndarray
    ) -> None:
        calls = install_fake_gemini(monkeypatch, reply(
            {"item": "safety_vest", "status": "worn", "confidence": 0.95, "reason": "seen"}
        ))
        result = verifier.verify_items(photo, "general_labor", ["safety_vest"])
        assert result is not None
        assert result.model == verifier.GEMINI_MODEL
        assert result.items[0].status is ItemStatus.WORN
        assert len(calls) == 1
        assert len(calls[0]["contents"]) == 2  # image part + prompt

    def test_no_api_key_returns_none(
        self, monkeypatch: pytest.MonkeyPatch, photo: np.ndarray
    ) -> None:
        calls = install_fake_gemini(monkeypatch, reply())
        monkeypatch.setattr(settings, "google_api_key", "")
        assert verifier.verify_items(photo, "welder", ["gloves"]) is None
        assert calls == []

    def test_switched_off_returns_none_without_calling(
        self, monkeypatch: pytest.MonkeyPatch, photo: np.ndarray
    ) -> None:
        calls = install_fake_gemini(monkeypatch, reply())
        monkeypatch.setattr(settings, "ppb_gemini_verify", False)
        assert verifier.verify_items(photo, "welder", ["gloves"]) is None
        assert calls == []

    def test_nothing_to_check_returns_none_without_calling(
        self, monkeypatch: pytest.MonkeyPatch, photo: np.ndarray
    ) -> None:
        calls = install_fake_gemini(monkeypatch, reply())
        assert verifier.verify_items(photo, "welder", []) is None
        assert calls == []

    def test_network_failure_returns_none(
        self, monkeypatch: pytest.MonkeyPatch, photo: np.ndarray
    ) -> None:
        install_fake_gemini(monkeypatch, TimeoutError("deadline exceeded"))
        assert verifier.verify_items(photo, "welder", ["gloves"]) is None

    def test_garbage_reply_returns_none(
        self, monkeypatch: pytest.MonkeyPatch, photo: np.ndarray
    ) -> None:
        install_fake_gemini(monkeypatch, "I think the worker looks fine!")
        assert verifier.verify_items(photo, "welder", ["gloves"]) is None


class TestResolutionFloor:
    """Gemini once answered not_worn @ 1.00 on a 183x275 thumbnail.

    A confidence threshold cannot catch a model that is confidently wrong, so
    photos too small to judge are never sent.
    """

    def test_thumbnail_is_not_verified(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = install_fake_gemini(monkeypatch, reply(
            {"item": "goggles", "status": "not_worn", "confidence": 1.0, "reason": ""}
        ))
        thumb = np.zeros((275, 183, 3), dtype=np.uint8)
        assert verifier.verify_items(thumb, "welder", ["goggles"]) is None
        assert calls == []

    def test_photo_at_the_floor_is_verified(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = install_fake_gemini(monkeypatch, reply(
            {"item": "goggles", "status": "worn", "confidence": 0.9, "reason": ""}
        ))
        ok = np.zeros((verifier.MIN_EDGE_PX, 300, 3), dtype=np.uint8)
        assert verifier.verify_items(ok, "welder", ["goggles"]) is not None
        assert len(calls) == 1

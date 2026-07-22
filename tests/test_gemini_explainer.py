"""Unit tests for src/inference/gemini_explainer.py — no real API calls."""

from __future__ import annotations

import pytest

import src.inference.gemini_explainer as explainer
from src.settings import settings


@pytest.fixture(autouse=True)
def clear_cache() -> None:
    """Each test starts with an empty explanation cache."""
    explainer._cached_explain.cache_clear()


class TestFallback:
    def test_no_api_key_uses_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings, "google_api_key", "")
        text = explainer.explain_failure("welder", ["gloves"], ["no_hardhat"])
        assert "gloves" in text
        assert "not wearing hardhat" in text
        assert "welder" in text

    def test_gemini_error_degrades_to_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings, "google_api_key", "not-a-real-key")

        class _BoomClient:
            def __init__(self, api_key: str) -> None:
                raise ConnectionError("network down")

        import google.genai

        monkeypatch.setattr(google.genai, "Client", _BoomClient)
        text = explainer.explain_failure("painter", ["mask"], [])
        assert "mask" in text  # fallback produced, no exception escaped


class TestCaching:
    def test_identical_failures_hit_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings, "google_api_key", "")
        explainer.explain_failure("welder", ["gloves"], [])
        explainer.explain_failure("welder", ["gloves"], [])
        info = explainer._cached_explain.cache_info()
        assert info.hits == 1
        assert info.misses == 1

    def test_missing_order_does_not_break_cache(self, monkeypatch) -> None:
        """['a','b'] and ['b','a'] must map to the same cache entry."""
        monkeypatch.setattr(settings, "google_api_key", "")
        explainer.explain_failure("welder", ["gloves", "goggles"], [])
        explainer.explain_failure("welder", ["goggles", "gloves"], [])
        assert explainer._cached_explain.cache_info().hits == 1

    def test_different_jobs_are_separate_entries(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "google_api_key", "")
        explainer.explain_failure("welder", ["gloves"], [])
        explainer.explain_failure("painter", ["gloves"], [])
        assert explainer._cached_explain.cache_info().misses == 2

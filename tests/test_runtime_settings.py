"""Runtime settings without a database: defaults, caching and outages.

tests/test_admin_api.py covers saving them through the API.
"""

from __future__ import annotations

import pytest

import src.admin.runtime as runtime
from src.inference.detector import PPEDetector
from src.settings import settings


class _Rows:
    """Stands in for the database read inside runtime.overrides()."""

    def __init__(self) -> None:
        self.calls = 0
        self.result: dict | Exception = {}


@pytest.fixture
def database(monkeypatch: pytest.MonkeyPatch) -> _Rows:
    fake = _Rows()

    class _Session:
        def __init__(self, engine: object) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def scalars(self, query: object):
            fake.calls += 1
            if isinstance(fake.result, Exception):
                raise fake.result
            return type("R", (), {"all": lambda _self: [
                type("Row", (), {"name": k, "value": v})() for k, v in fake.result.items()]})()

    monkeypatch.setattr(settings, "database_url", "postgresql://u:p@localhost:5432/ppb")
    monkeypatch.setattr("src.db.engine.get_engine", lambda url=None: object())
    monkeypatch.setattr("sqlalchemy.orm.Session", _Session)
    return fake


class TestDefaults:
    def test_env_values_apply_without_a_database(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "ppb_worker_auth", "required")
        assert runtime.get("worker_auth") == "required"
        assert runtime.overrides() == {}

    def test_every_definition_has_an_env_default(self) -> None:
        for d in runtime.DEFINITIONS:
            assert runtime.default(d.name) is not None

    def test_threshold_range_is_the_agreed_one(self) -> None:
        d = runtime.BY_NAME["confidence_threshold"]
        assert (d.min, d.max, d.confirm) == (0.25, 0.80, True)


class TestOverrides:
    def test_a_saved_value_wins_over_env(self, database: _Rows) -> None:
        database.result = {"gemini_min_confidence": 0.9}
        assert runtime.get("gemini_min_confidence") == 0.9
        assert runtime.get("gemini_verify") == settings.ppb_gemini_verify

    def test_unknown_names_in_the_table_are_ignored(self, database: _Rows) -> None:
        database.result = {"model": "yolo26l"}
        assert runtime.overrides() == {}

    def test_cached_until_invalidated(self, database: _Rows) -> None:
        runtime.get("worker_auth")
        runtime.get("worker_auth")
        assert database.calls == 1
        runtime.invalidate()
        runtime.get("worker_auth")
        assert database.calls == 2

    def test_outage_never_breaks_a_check(self, database: _Rows) -> None:
        database.result = RuntimeError("connection refused")
        assert runtime.get("worker_auth") == settings.ppb_worker_auth

    def test_outage_keeps_the_last_good_values(self, database: _Rows,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
        database.result = {"gemini_min_confidence": 0.9}
        runtime.get("gemini_min_confidence")
        monkeypatch.setattr(runtime, "_cache_at", 0.0)  # cache period over
        database.result = RuntimeError("connection refused")
        assert runtime.get("gemini_min_confidence") == 0.9


def test_detector_threshold_follows_the_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    """PPEDetector reads the live threshold unless one was passed in."""
    detector = PPEDetector.__new__(PPEDetector)
    detector._fixed_confidence = None
    monkeypatch.setattr(runtime, "get", lambda name: 0.6)
    assert detector.confidence == 0.6
    detector._fixed_confidence = 0.3
    assert detector.confidence == 0.3

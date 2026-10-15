"""Where the job matrix comes from, and what happens when the database fails.

The database is replaced by a stand-in for ``read_matrix``, so these tests
need no PostgreSQL. tests/test_db_matrix.py covers the real database.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

import src.db.matrix as db_matrix
import src.inference.compliance as compliance
from src.api import main
from src.inference.compliance import (
    MatrixUnavailableError,
    invalidate_job_matrix_cache,
    load_job_matrix,
    matrix_source,
)
from src.settings import settings
from tests.conftest import make_detection

YAML_PATH = Path("configs/job_matrix.yaml")
DB_MATRIX = {
    "only_job": {"required": ["hardhat"], "pending": []},
}


class _FakeDatabase:
    """Stands in for read_matrix; can be told to fail or return nothing."""

    def __init__(self) -> None:
        self.calls = 0
        self.result: dict | Exception = DB_MATRIX

    def __call__(self, engine: object) -> dict:
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.fixture
def database(monkeypatch: pytest.MonkeyPatch) -> _FakeDatabase:
    """Point the matrix at a (fake) database."""
    fake = _FakeDatabase()
    monkeypatch.setattr(settings, "database_url", "postgresql://u:p@localhost:5432/ppb")
    monkeypatch.setattr(db_matrix, "read_matrix", fake)
    monkeypatch.setattr("src.db.engine.get_engine", lambda url=None: object())
    return fake


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Control time.monotonic() as seen by the matrix cache."""
    now = [1000.0]
    monkeypatch.setattr(compliance.time, "monotonic", lambda: now[0])
    return now


def _outage() -> OperationalError:
    return OperationalError("SELECT 1", {}, Exception("connection refused"))


class TestSource:
    def test_yaml_when_no_database_is_configured(self) -> None:
        assert matrix_source() == "yaml"
        assert "general_labor" in load_job_matrix()

    def test_explicit_path_always_reads_yaml(self, database: _FakeDatabase) -> None:
        assert "general_labor" in load_job_matrix(path=YAML_PATH)
        assert database.calls == 0

    def test_database_when_configured(self, database: _FakeDatabase) -> None:
        assert matrix_source() == "database"
        matrix = load_job_matrix()
        assert list(matrix) == ["only_job"]
        assert matrix["only_job"].required == ["hardhat"]

    def test_verdicts_follow_the_database_matrix(self, database: _FakeDatabase) -> None:
        """A job that exists only in the database can be checked."""
        result = compliance.check_compliance([make_detection("hardhat")], "only_job")
        assert result.verdict.value == "PASS"


class TestCache:
    def test_reads_once_within_the_cache_period(
        self, database: _FakeDatabase, clock: list[float]
    ) -> None:
        load_job_matrix()
        clock[0] += settings.ppb_matrix_cache_s - 1
        load_job_matrix()
        assert database.calls == 1

    def test_rereads_after_the_cache_period(
        self, database: _FakeDatabase, clock: list[float]
    ) -> None:
        load_job_matrix()
        clock[0] += settings.ppb_matrix_cache_s + 1
        database.result = {"other_job": {"required": ["boots"], "pending": []}}
        assert list(load_job_matrix()) == ["other_job"]
        assert database.calls == 2

    def test_invalidate_forces_a_fresh_read(self, database: _FakeDatabase) -> None:
        """What the admin API will call after an edit."""
        load_job_matrix()
        invalidate_job_matrix_cache()
        load_job_matrix()
        assert database.calls == 2


class TestOutage:
    def test_no_matrix_ever_read_raises(self, database: _FakeDatabase) -> None:
        database.result = _outage()
        with pytest.raises(MatrixUnavailableError, match="unavailable"):
            load_job_matrix()

    def test_empty_database_raises_with_the_fix(self, database: _FakeDatabase) -> None:
        database.result = {}
        with pytest.raises(MatrixUnavailableError, match="db seed"):
            load_job_matrix()

    def test_outage_after_a_good_read_keeps_the_last_matrix(
        self, database: _FakeDatabase, clock: list[float]
    ) -> None:
        good = load_job_matrix()
        clock[0] += settings.ppb_matrix_cache_s + 1
        database.result = _outage()
        assert load_job_matrix() == good

    def test_outage_waits_a_cache_period_before_retrying(
        self, database: _FakeDatabase, clock: list[float]
    ) -> None:
        """A down database must not add a connection timeout to every check."""
        load_job_matrix()
        clock[0] += settings.ppb_matrix_cache_s + 1
        database.result = _outage()
        load_job_matrix()
        load_job_matrix()
        assert database.calls == 2


class TestApi:
    """``api_client`` is listed before ``database``, so the app starts on the
    YAML matrix and the database is switched in (or broken) afterwards."""

    def test_startup_fails_with_the_fix_when_the_database_has_no_matrix(
        self, database: _FakeDatabase, fake_detector, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(main, "init_detector", lambda model_name=None, device=None: fake_detector)
        database.result = {}
        with pytest.raises(RuntimeError, match="cli.py db migrate"), TestClient(main.app):
            pass

    def test_jobs_returns_503_when_the_matrix_is_unavailable(
        self, api_client, database: _FakeDatabase
    ) -> None:
        database.result = _outage()
        response = api_client.get("/api/v1/jobs")
        assert response.status_code == 503
        assert "unavailable" in response.json()["detail"]

    def test_check_returns_503_rather_than_guessing(
        self, api_client, database: _FakeDatabase, sample_image_bytes: bytes
    ) -> None:
        database.result = _outage()
        response = api_client.post(
            "/api/v1/compliance/check",
            files={"image": ("worker.png", sample_image_bytes, "image/png")},
            data={"worker_id": "W-0042", "job_category": "general_labor"},
        )
        assert response.status_code == 503

    def test_jobs_lists_the_database_matrix(self, api_client, database: _FakeDatabase) -> None:
        assert api_client.get("/api/v1/jobs").json() == DB_MATRIX

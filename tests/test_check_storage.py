"""Every check is stored, and its photo only when that's allowed (docs/checks-api.md §2, §5).

Against a real PostgreSQL (TEST_DATABASE_URL); skipped without it, as the
other database tests are. The detector is the fake one from conftest.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

import pytest
from argon2 import PasswordHasher
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import select, update
from sqlalchemy.orm import Session

import src.auth.crypto as crypto
from src.admin import runtime
from src.api import deps, main
from src.auth import service
from src.auth.ratelimit import ip_limiter
from src.checks import record
from src.db.models import AppSetting, Check, Site, Worker
from src.settings import settings
from tests.conftest import TEST_DB_URL, make_detection

WORKER = "W-0042"
PIN = "730154"
COMPLIANT = [make_detection("hardhat", 0.92), make_detection("safety_vest", 0.88),
             make_detection("boots", 0.8)]


def photo_with_location() -> bytes:
    """40x20 as stored, shown upright as 20x40, with a GPS position."""
    exif = Image.Exif()
    exif[0x0112] = 6
    exif[0x8825] = {1: "N", 2: (18.0, 31.0, 12.5), 3: "E", 4: (73.0, 51.0, 2.25)}
    out = io.BytesIO()
    Image.new("RGB", (40, 20), (10, 120, 200)).save(out, "JPEG", exif=exif.tobytes())
    return out.getvalue()


UPLOAD = photo_with_location()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fast_hashing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(crypto, "_hasher", PasswordHasher(time_cost=1, memory_cost=8,
                                                          parallelism=1))


@pytest.fixture
def photo_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    folder = tmp_path / "photos"
    monkeypatch.setattr(settings, "ppb_photo_dir", folder)
    return folder


@pytest.fixture
def db_engine(seeded_db, photo_dir: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "database_url", TEST_DB_URL)
    ip_limiter.reset()
    yield seeded_db
    ip_limiter.reset()


@pytest.fixture
def db(db_engine) -> Iterator[Session]:
    with Session(db_engine) as session:
        yield session


@pytest.fixture
def client(db_engine, fake_detector, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    fake_detector.detections = list(COMPLIANT)
    monkeypatch.setattr(main, "init_detector", lambda model_name=None, device=None: fake_detector)
    monkeypatch.setattr(deps, "_detector", fake_detector)
    with TestClient(main.app, base_url="https://testserver") as test_client:
        yield test_client


@pytest.fixture
def token(client: TestClient, db: Session) -> str:
    """W-0042, signed in with their own PIN."""
    _, issued = service.create_worker(db, WORKER, "Asha Verma")
    first = client.post("/api/v1/auth/worker/login", json={"worker_id": WORKER, "pin": issued})
    signed_in = first.json()["token"]
    changed = client.post("/api/v1/auth/worker/pin", headers={"Authorization": f"Bearer {signed_in}"},
                          json={"current_pin": issued, "new_pin": PIN})
    assert changed.status_code == 204
    return signed_in


@pytest.fixture
def seen_notice(db: Session, token: str) -> None:
    """The worker has answered the photo notice."""
    db.execute(update(Worker).where(Worker.worker_code == WORKER).values(consent_seen_version=1))
    db.commit()


def check(client: TestClient, token: str | None = None, worker_id: str | None = None,
          job: str = "general_labor"):
    data = {"job_category": job}
    if worker_id is not None:
        data["worker_id"] = worker_id
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    response = client.post("/api/v1/compliance/check", headers=headers, data=data,
                           files={"image": ("photo.jpg", UPLOAD, "image/jpeg")})
    assert response.status_code == 200
    return response.json()


def stored(db: Session, check_id: int) -> Check:
    db.expire_all()
    return db.get(Check, check_id)


def kept_files(folder: Path) -> list[Path]:
    return [p for p in folder.rglob("*") if p.is_file()] if folder.exists() else []


# ---------------------------------------------------------------------------
# What is stored
# ---------------------------------------------------------------------------


class TestEveryCheckIsStored:
    def test_with_what_the_verdict_was_judged_against(self, client, db, token) -> None:
        answer = check(client, token)
        row = stored(db, answer["check_id"])

        assert answer["verdict"] == "PASS"
        assert (row.worker_code, row.signed_in, row.worker.worker_code) == (WORKER, True, WORKER)
        assert row.site_id == db.scalar(select(Site.id).where(Site.key == "main"))
        assert (row.job_key, row.job_name, row.job_version) == ("general_labor", "General labor", 1)
        assert (row.required, row.pending) == (["hardhat", "safety_vest", "boots"], [])
        assert (row.verdict, row.verdict_source) == ("PASS", answer["verdict_source"])
        assert row.detections == answer["detections"]
        assert row.missing_equipment == answer["missing_equipment"]
        assert (row.model_used, row.inference_time_ms) == ("yolo11l", answer["inference_time_ms"])
        assert row.at.isoformat().replace("+00:00", "Z").startswith(answer["timestamp"][:19])
        assert row.review_outcome is None

    def test_with_the_three_settings_that_decide_a_verdict(self, client, db, token) -> None:
        row = stored(db, check(client, token)["check_id"])
        assert row.confidence_threshold == 0.45  # the fake detector's
        assert row.gemini_verify is bool(runtime.get("gemini_verify"))
        assert row.gemini_min_confidence == float(runtime.get("gemini_min_confidence"))

    def test_the_photo_size_is_after_orientation(self, client, db, token) -> None:
        row = stored(db, check(client, token)["check_id"])
        assert (row.photo_width, row.photo_height) == (20, 40)
        assert (row.photo_type, row.photo_bytes) == ("image/jpeg", len(UPLOAD))

    def test_without_a_database_there_is_no_check_id(self, api_client, fake_detector,
                                                     monkeypatch) -> None:
        monkeypatch.setattr(settings, "database_url", "")
        fake_detector.detections = list(COMPLIANT)
        response = api_client.post("/api/v1/compliance/check",
                                   data={"job_category": "general_labor", "worker_id": WORKER},
                                   files={"image": ("photo.jpg", UPLOAD, "image/jpeg")})
        assert (response.status_code, response.json()["check_id"]) == (200, None)


# ---------------------------------------------------------------------------
# When the photo is kept
# ---------------------------------------------------------------------------


class TestThePhoto:
    def test_not_kept_before_the_worker_has_seen_the_notice(self, client, db, token,
                                                             photo_dir) -> None:
        row = stored(db, check(client, token)["check_id"])
        assert (row.photo_state, row.photo_key) == ("not_stored", None)
        assert kept_files(photo_dir) == []

    def test_kept_once_they_have_without_its_location(self, client, db, token, seen_notice,
                                                      photo_dir) -> None:
        row = stored(db, check(client, token)["check_id"])
        assert row.photo_state == "kept"
        [path] = kept_files(photo_dir)
        assert path.relative_to(photo_dir).as_posix() == row.photo_key
        assert WORKER not in row.photo_key  # the file name doesn't say whose photo it is

        kept = path.read_bytes()
        exif = Image.open(io.BytesIO(kept)).getexif()
        assert dict(exif.get_ifd(0x8825)) == {}  # no location
        assert exif.get(0x0112) == 6  # the orientation stays
        assert (len(kept), row.photo_bytes) == (len(UPLOAD), len(UPLOAD))  # not re-encoded

    def test_not_kept_with_store_photos_off(self, client, db, token, seen_notice,
                                            photo_dir) -> None:
        db.add(AppSetting(name="store_photos", value=False, updated_by="admin@example.com"))
        db.commit()
        runtime.invalidate()
        row = stored(db, check(client, token)["check_id"])
        assert row.photo_state == "not_stored"
        assert kept_files(photo_dir) == []

    def test_the_training_choice_at_the_time_is_stamped(self, client, db, token,
                                                        seen_notice) -> None:
        assert stored(db, check(client, token)["check_id"]).training_consent is False
        db.execute(update(Worker).where(Worker.worker_code == WORKER).values(training_consent=True))
        db.commit()
        assert stored(db, check(client, token)["check_id"]).training_consent is True


class TestChecksWithoutALogin:
    """While PPB_WORKER_AUTH=optional (decision Q8 in #34)."""

    def test_stored_and_linked_to_the_worker_the_id_names(self, client, db, token,
                                                          seen_notice, photo_dir) -> None:
        # The worker said yes to training, but this check can't be shown to be theirs.
        db.execute(update(Worker).where(Worker.worker_code == WORKER).values(training_consent=True))
        db.commit()
        row = stored(db, check(client, worker_id="w-0042")["check_id"])
        assert (row.signed_in, row.worker.worker_code, row.worker_code) == (False, WORKER, "w-0042")
        assert row.site_id is not None
        # Never a photo: an app without login never showed the notice.
        assert (row.photo_state, row.training_consent) == ("not_stored", False)
        assert kept_files(photo_dir) == []

    def test_an_id_that_names_nobody_has_no_worker_and_no_site(self, client, db) -> None:
        row = stored(db, check(client, worker_id="NOBODY-1")["check_id"])
        assert (row.worker_id, row.site_id, row.worker_code) == (None, None, "NOBODY-1")


# ---------------------------------------------------------------------------
# Storing never costs the worker a verdict
# ---------------------------------------------------------------------------


class TestWhenStoringFails:
    def test_the_photo_store_failing_still_stores_the_check(self, client, db, token, seen_notice,
                                                            monkeypatch) -> None:
        class BrokenStore:
            def save(self, data: bytes, content_type: str) -> str:
                raise OSError("disk full")

        monkeypatch.setattr(record, "photo_store", BrokenStore)
        answer = check(client, token)
        assert answer["verdict"] == "PASS"
        assert stored(db, answer["check_id"]).photo_state == "not_stored"

    def test_the_database_failing_still_answers_and_leaves_no_photo(self, client, db, token,
                                                                     seen_notice, photo_dir,
                                                                     monkeypatch) -> None:
        def broken_row(**_fields):
            raise RuntimeError("database gone")

        monkeypatch.setattr(record, "Check", broken_row)
        answer = check(client, token)
        assert (answer["verdict"], answer["check_id"]) == ("PASS", None)
        assert kept_files(photo_dir) == []  # the photo kept just before was removed again
        assert db.scalar(select(Check.id)) is None

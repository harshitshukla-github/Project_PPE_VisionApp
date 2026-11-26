"""The photo notice, how long photos are kept, and exporting them for training.

docs/checks-api.md §5 (consent), §6 and §7 (retention), §8 (audit). Against
a real PostgreSQL (TEST_DATABASE_URL); skipped without it, as the other
database tests are.
"""

from __future__ import annotations

import asyncio
import io
import json
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from argon2 import PasswordHasher
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError
from sqlalchemy import select, text
from sqlalchemy.orm import Session
from typer.testing import CliRunner

import cli
import src.auth.crypto as crypto
from src.admin import runtime
from src.api import deps, main
from src.auth import service
from src.auth.ratelimit import ip_limiter
from src.checks import retention
from src.checks.photos import FolderPhotoStore
from src.db.models import AppSetting, AuditEntry, Check, ConsentNotice, Role, Site, Worker
from src.settings import Settings, settings
from tests.conftest import TEST_DB_URL, make_detection

WORKER, PIN = "W-0042", "730154"
ADMIN_EMAIL, PASSWORD = "admin@example.com", "correct horse battery"
NOW = datetime.now(timezone.utc)


def jpeg() -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (40, 20), (10, 120, 200)).save(out, "JPEG")
    return out.getvalue()


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
    runtime.invalidate()
    ip_limiter.reset()
    yield seeded_db
    ip_limiter.reset()
    runtime.invalidate()


@pytest.fixture
def db(db_engine) -> Iterator[Session]:
    with Session(db_engine, expire_on_commit=False) as session:
        yield session


@pytest.fixture
def store(photo_dir: Path) -> FolderPhotoStore:
    return FolderPhotoStore(photo_dir)


@pytest.fixture
def client(db_engine, fake_detector, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    fake_detector.detections = [make_detection("hardhat", 0.92),
                                make_detection("safety_vest", 0.88), make_detection("boots", 0.8)]
    monkeypatch.setattr(main, "init_detector", lambda model_name=None, device=None: fake_detector)
    monkeypatch.setattr(deps, "_detector", fake_detector)
    with TestClient(main.app, base_url="https://testserver") as test_client:
        yield test_client


@pytest.fixture
def bearer(client: TestClient, db: Session) -> dict[str, str]:
    """W-0042, signed in with their own PIN."""
    _, issued = service.create_worker(db, WORKER, "Asha Verma")
    token = client.post("/api/v1/auth/worker/login",
                        json={"worker_id": WORKER, "pin": issued}).json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    assert client.post("/api/v1/auth/worker/pin", headers=headers,
                       json={"current_pin": issued, "new_pin": PIN}).status_code == 204
    return headers


def consent(client: TestClient, bearer: dict) -> dict:
    response = client.get("/api/v1/auth/worker/consent", headers=bearer)
    assert response.status_code == 200, response.text
    return response.json()


def answer(client: TestClient, bearer: dict, version: int, training: bool):
    return client.put("/api/v1/auth/worker/consent", headers=bearer,
                      json={"version": version, "training": training})


def set_setting(db: Session, name: str, value) -> None:
    db.merge(AppSetting(name=name, value=value, updated_by=ADMIN_EMAIL))
    db.commit()
    runtime.invalidate()


def audit(db: Session, action: str) -> list[AuditEntry]:
    db.expire_all()
    return list(db.scalars(select(AuditEntry).where(AuditEntry.action == action)
                           .order_by(AuditEntry.id)))


def add_check(db: Session, *, worker: Worker | None, age_days: float, consent: bool,
              store: FolderPhotoStore | None, site: Site | None = None) -> Check:
    """A stored check made ``age_days`` ago, with a kept photo when ``store`` is given."""
    key = store.save(jpeg(), "image/jpeg") if store else None
    row = Check(
        at=NOW - timedelta(days=age_days), worker_id=worker.id if worker else None,
        worker_code=worker.worker_code if worker else "NOBODY-1", signed_in=worker is not None,
        site_id=(worker.site_id if worker else site.id if site else None),
        job_key="general_labor", job_name="General labor", job_version=1,
        required=["hardhat", "safety_vest", "boots"], pending=[], verdict="PASS",
        verdict_source="detector", confidence_avg=0.9, missing_equipment=[],
        violations_detected=[], unverifiable_equipment=[],
        detections=[{"class": "hardhat", "confidence": 0.92, "bbox": [10.0, 10.0, 30.0, 18.0]}],
        verification=None, explanation=None, model_used="yolo11l", inference_time_ms=20.0,
        gemini_invoked=False, confidence_threshold=0.45, gemini_verify=False,
        gemini_min_confidence=0.75, photo_state="kept" if key else "not_stored", photo_key=key,
        photo_type="image/jpeg", photo_width=40, photo_height=20, photo_bytes=1000,
        training_consent=consent,
    )
    db.add(row)
    db.commit()
    return row


def fresh(db: Session, check: Check) -> Check:
    db.expire_all()
    return db.get(Check, check.id)


# ---------------------------------------------------------------------------
# The notice (§5)
# ---------------------------------------------------------------------------


class TestTheNotice:
    def test_the_server_writes_it_with_the_periods_in_force(self, client, bearer, db) -> None:
        body = consent(client, bearer)
        notice = body["notice"]
        assert notice["version"] == 1
        assert "for up to 90 days, so supervisors at your site can review checks" in notice["text"]
        assert "Gemini" in notice["text"] and "Ask your supervisor" in notice["text"]
        assert notice["training_question"].startswith("Can we also keep your photos for up to "
                                                       "730 days")
        # Not answered yet: the app shows it, with training off.
        assert (body["seen_version"], body["training"], body["updated_at"]) == (None, False, None)

    def test_reading_it_again_makes_no_new_version(self, client, bearer, db) -> None:
        consent(client, bearer)
        assert consent(client, bearer)["notice"]["version"] == 1
        assert db.scalar(select(text("count(*)")).select_from(ConsentNotice)) == 1

    def test_a_changed_period_is_a_new_version_and_the_old_words_stay(self, client, bearer,
                                                                       db) -> None:
        consent(client, bearer)
        set_setting(db, "photo_review_days", 60)
        notice = consent(client, bearer)["notice"]
        assert notice["version"] == 2 and "up to 60 days" in notice["text"]
        assert "up to 90 days" in db.get(ConsentNotice, 1).text

    def test_needs_a_worker_who_chose_their_pin(self, client, db) -> None:
        assert client.get("/api/v1/auth/worker/consent").status_code == 401
        _, issued = service.create_worker(db, "W-0099", "New Starter")
        token = client.post("/api/v1/auth/worker/login",
                            json={"worker_id": "W-0099", "pin": issued}).json()["token"]
        response = client.get("/api/v1/auth/worker/consent",
                              headers={"Authorization": f"Bearer {token}"})
        assert (response.status_code, response.json()["code"]) == (403, "pin_change_required")


class TestTheAnswer:
    def test_is_recorded_and_audited(self, client, bearer, db) -> None:
        response = answer(client, bearer, 1, True)
        assert response.status_code == 200, response.text
        body = response.json()
        assert (body["seen_version"], body["training"]) == (1, True)
        assert body["updated_at"].endswith("Z") and body["notice"]["version"] == 1

        [entry] = audit(db, "worker.consent")
        assert (entry.actor, entry.target, entry.site) == ("worker:W-0042", "worker:W-0042", "main")
        assert (entry.before, entry.after) == ({"version": None, "training": False},
                                               {"version": 1, "training": True})

    def test_the_same_answer_again_changes_nothing(self, client, bearer, db) -> None:
        answer(client, bearer, 1, False)
        assert answer(client, bearer, 1, False).status_code == 200
        assert len(audit(db, "worker.consent")) == 1

    def test_to_an_older_notice_is_refused_with_the_current_one(self, client, bearer, db) -> None:
        consent(client, bearer)
        set_setting(db, "photo_training_days", 1000)
        response = answer(client, bearer, 1, True)
        assert (response.status_code, response.json()["code"]) == (409, "notice_changed")
        notice = response.json()["notice"]
        assert notice["version"] == 2 and "up to 1000 days" in notice["training_question"]
        worker = db.scalar(select(Worker).where(Worker.worker_code == WORKER))
        db.refresh(worker)
        assert (worker.consent_seen_version, worker.training_consent) == (None, False)
        assert audit(db, "worker.consent") == []

    @pytest.mark.parametrize("body", [{"version": 1, "training": "yes"},
                                      {"version": "1", "training": True}, {"version": 1}])
    def test_a_malformed_answer(self, client, bearer, body) -> None:
        response = client.put("/api/v1/auth/worker/consent", headers=bearer, json=body)
        assert response.status_code == 422

    def test_photos_are_kept_once_answered_and_stamped_with_the_choice(self, client, bearer,
                                                                       db, photo_dir) -> None:
        """Part 1's rule, through the real endpoint: no photo until the notice is answered."""
        files = {"image": ("photo.jpg", jpeg(), "image/jpeg")}

        def check() -> Check:
            answer_ = client.post("/api/v1/compliance/check", headers=bearer,
                                  data={"job_category": "general_labor"}, files=files).json()
            db.expire_all()
            return db.get(Check, answer_["check_id"])

        assert check().photo_state == "not_stored"
        answer(client, bearer, 1, True)
        kept = check()
        assert (kept.photo_state, kept.training_consent) == ("kept", True)

    def test_withdrawing_clears_consent_on_earlier_checks(self, client, bearer, db,
                                                          store) -> None:
        answer(client, bearer, 1, True)
        worker = db.scalar(select(Worker).where(Worker.worker_code == WORKER))
        mine = [add_check(db, worker=worker, age_days=d, consent=True, store=store)
                for d in (1, 200)]
        other_worker, _ = service.create_worker(db, "W-0043", "Ravi Kumar")
        theirs = add_check(db, worker=other_worker, age_days=1, consent=True, store=store)

        assert answer(client, bearer, 1, False).status_code == 200
        assert [fresh(db, c).training_consent for c in mine] == [False, False]
        assert fresh(db, theirs).training_consent is True
        last = audit(db, "worker.consent")[-1]
        assert (last.before, last.after) == ({"version": 1, "training": True},
                                             {"version": 1, "training": False})


# ---------------------------------------------------------------------------
# The two periods as settings (§7)
# ---------------------------------------------------------------------------


@pytest.fixture
def admin(db_engine, fake_detector, monkeypatch) -> Iterator[tuple[TestClient, dict]]:
    with Session(db_engine) as session:
        service.create_user(session, ADMIN_EMAIL, "Admin", Role.ADMIN, PASSWORD)
    monkeypatch.setattr(main, "init_detector", lambda model_name=None, device=None: fake_detector)
    monkeypatch.setattr(deps, "_detector", fake_detector)
    with TestClient(main.app, base_url="https://testserver") as test_client:
        login = test_client.post("/api/v1/auth/login",
                                 json={"email": ADMIN_EMAIL, "password": PASSWORD})
        yield test_client, {"X-CSRF-Token": login.json()["csrf_token"]}


def patch(admin, body: dict):
    client, csrf = admin
    return client.patch("/api/v1/admin/settings", json=body, headers=csrf)


class TestTheSettings:
    @pytest.mark.parametrize(("body", "code", "field"), [
        ({"photo_review_days": 6}, "out_of_range", "photo_review_days"),
        ({"photo_training_days": 3651}, "out_of_range", "photo_training_days"),
        ({"photo_review_days": 90.5}, "invalid", "photo_review_days"),
        # training is never shorter than review, whichever one the patch sends
        ({"photo_training_days": 60}, "out_of_range", "photo_training_days"),
        ({"photo_review_days": 800}, "out_of_range", "photo_review_days"),
        ({"photo_review_days": 800, "photo_training_days": 700}, "out_of_range",
         "photo_training_days"),
    ])
    def test_out_of_range_saves_nothing(self, admin, db, body, code, field) -> None:
        response = patch(admin, body)
        assert response.status_code == 422, response.text
        assert (response.json()["code"], response.json()["field"]) == (code, field)
        assert db.scalars(select(AppSetting)).all() == []

    def test_both_at_once_and_whole_days(self, admin, db) -> None:
        response = patch(admin, {"photo_review_days": 800, "photo_training_days": 900.0})
        assert response.status_code == 200, response.text
        saved = {s.name: s.value for s in db.scalars(select(AppSetting))}
        assert saved == {"photo_review_days": 800, "photo_training_days": 900}
        assert retention.periods() == (800, 900)

    def test_clearing_training_cant_leave_it_shorter(self, admin, db) -> None:
        patch(admin, {"photo_review_days": 800, "photo_training_days": 900})
        response = patch(admin, {"photo_training_days": None})  # back to 730
        assert (response.status_code, response.json()["field"]) == (422, "photo_training_days")

    def test_a_changed_period_says_what_the_next_run_deletes(self, admin, db, store) -> None:
        site = db.scalar(select(Site).where(Site.key == "main"))
        for age, agreed in ((100, False), (50, False), (800, True), (400, True)):
            add_check(db, worker=None, site=site, age_days=age, consent=agreed, store=store)
        response = patch(admin, {"photo_review_days": 30})
        assert response.status_code == 200
        [warning] = response.json()["warnings"]
        # 100 and 50 days without consent, 800 with: all past 30 / 730
        assert (warning["code"], warning["setting"], warning["count"]) == (
            "retention_changed", "photo_review_days", 3)
        assert warning["message"] == "The next daily run will delete 3 photos past the new period."

    def test_env_values_where_training_is_shorter_are_refused(self, monkeypatch) -> None:
        monkeypatch.setenv("PPB_PHOTO_REVIEW_DAYS", "90")
        monkeypatch.setenv("PPB_PHOTO_TRAINING_DAYS", "60")
        with pytest.raises(ValidationError, match="can't be shorter"):
            Settings()


# ---------------------------------------------------------------------------
# Deleting photos past their period (§6)
# ---------------------------------------------------------------------------


@pytest.fixture
def aged(db: Session, db_engine, store) -> dict[str, Check]:
    site = db.scalar(select(Site).where(Site.key == "main"))
    made = {name: add_check(db, worker=None, site=site, age_days=age, consent=agreed, store=store)
            for name, age, agreed in (("old", 100, False), ("recent", 50, False),
                                      ("old_agreed", 800, True), ("agreed", 400, True))}
    made["no_photo"] = add_check(db, worker=None, site=site, age_days=1000, consent=False,
                                 store=None)
    return made


def on_disk(folder: Path) -> set[str]:
    return {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}


class TestPurge:
    def test_deletes_photos_past_their_period_and_keeps_the_checks(self, db, aged,
                                                                   photo_dir) -> None:
        keys = {name: c.photo_key for name, c in aged.items()}
        assert retention.purge(db) == 2
        assert on_disk(photo_dir) == {keys["recent"], keys["agreed"]}
        for name in ("old", "old_agreed"):
            row = fresh(db, aged[name])
            assert (row.photo_state, row.photo_key, row.verdict) == ("deleted", None, "PASS")
        [entry] = audit(db, "photos.purge")
        assert (entry.actor, entry.target, entry.after) == ("cli", "photos", {"deleted": 2})

    def test_a_withdrawn_photo_falls_back_to_the_review_period(self, db, aged) -> None:
        row = fresh(db, aged["agreed"])
        row.training_consent = False  # as a withdrawal leaves it
        db.commit()
        assert retention.purge(db) == 3
        assert fresh(db, aged["agreed"]).photo_state == "deleted"

    def test_every_run_is_one_entry_even_with_nothing_due(self, db, aged) -> None:
        retention.purge(db)
        assert retention.purge(db) == 0
        assert [e.after for e in audit(db, "photos.purge")] == [{"deleted": 2}, {"deleted": 0}]

    def test_the_daily_job_runs_once_a_day(self, db, aged) -> None:
        assert retention.purge_if_due() == 2
        assert retention.purge_if_due() is None  # a day hasn't passed
        assert retention.purge_if_due(force=True) == 0
        for entry in audit(db, "photos.purge"):
            entry.at = NOW - timedelta(hours=25)
        db.commit()
        assert retention.purge_if_due() == 0
        assert len(audit(db, "photos.purge")) == 3

    def test_the_daily_job_waits_while_another_server_purges(self, db_engine, aged) -> None:
        with Session(db_engine) as other:
            other.execute(text(f"SELECT pg_advisory_xact_lock({retention._LOCK_KEY})"))
            assert retention.purge_if_due() is None
        assert retention.purge_if_due() == 2

    def test_a_failed_run_is_tried_again_at_the_next_look(self, monkeypatch) -> None:
        class Stop(BaseException):
            pass

        forced: list[bool] = []

        def fake(*, force: bool) -> int:
            forced.append(force)
            if len(forced) == 1:
                raise OSError("disk unplugged")
            if len(forced) == 3:
                raise Stop
            return 0

        monkeypatch.setattr(retention, "purge_if_due", fake)
        monkeypatch.setattr(retention, "FIRST_LOOK_AFTER_S", 0)
        monkeypatch.setattr(retention, "LOOK_EVERY_S", 0)
        with pytest.raises(Stop):
            asyncio.run(retention.run_daily())
        assert forced == [False, True, False]


# ---------------------------------------------------------------------------
# The CLI: purge and export (§9)
# ---------------------------------------------------------------------------


class TestTheCli:
    def test_purge_dry_run_then_for_real(self, db, aged, photo_dir) -> None:
        runner = CliRunner()
        dry = runner.invoke(cli.app, ["photos", "purge", "--dry-run"])
        assert dry.exit_code == 0, dry.output
        assert "2 photos are due for deletion" in dry.output
        assert len(on_disk(photo_dir)) == 4 and audit(db, "photos.purge") == []

        done = runner.invoke(cli.app, ["photos", "purge"])
        assert done.exit_code == 0, done.output
        assert "Deleted 2 photos" in done.output and len(on_disk(photo_dir)) == 2

    def test_export_needs_consented(self, db_engine, tmp_path) -> None:
        result = CliRunner().invoke(cli.app, ["photos", "export", "--to", str(tmp_path / "out")])
        assert result.exit_code == 1 and "--consented" in result.output

    def test_export_only_what_workers_agreed_to_and_nothing_about_who(self, db, aged,
                                                                      tmp_path) -> None:
        out = tmp_path / "out"
        result = CliRunner().invoke(cli.app, ["photos", "export", "--consented", "--to", str(out)])
        assert result.exit_code == 0, result.output
        # "agreed" only: "old_agreed" is past its period, the others never agreed
        lines = [json.loads(line) for line in (out / "manifest.jsonl").read_text().splitlines()]
        assert [line["check_id"] for line in lines] == [aged["agreed"].id]
        assert sorted(p.name for p in (out / "images").iterdir()) == [f"{aged['agreed'].id}.jpg"]
        assert (out / lines[0]["image"]).read_bytes() == jpeg()
        assert not {"worker_id", "worker_code", "worker_name", "note"} & set(lines[0])
        [entry] = audit(db, "photos.export")
        assert (entry.actor, entry.after) == ("cli", {"exported": 1, "folder": str(out.resolve())})

    def test_export_refuses_a_folder_in_use(self, db, aged, tmp_path) -> None:
        out = tmp_path / "out"
        out.mkdir()
        (out / "older.txt").write_text("x")
        result = CliRunner().invoke(cli.app, ["photos", "export", "--consented", "--to", str(out)])
        # Rich wraps long lines, and the folder's path is long
        assert result.exit_code == 1 and "isn't an empty folder" in " ".join(result.output.split())
        assert audit(db, "photos.export") == []

    def test_a_withdrawal_stops_earlier_photos_being_exported(self, client, bearer, db, store,
                                                              tmp_path) -> None:
        """§9's test: withdraw through the app's endpoint, then export."""
        answer(client, bearer, 1, True)
        worker = db.scalar(select(Worker).where(Worker.worker_code == WORKER))
        add_check(db, worker=worker, age_days=1, consent=True, store=store)
        answer(client, bearer, 1, False)
        result = CliRunner().invoke(cli.app, ["photos", "export", "--consented", "--to",
                                              str(tmp_path / "out")])
        assert result.exit_code == 0, result.output
        assert (tmp_path / "out" / "manifest.jsonl").read_text() == ""

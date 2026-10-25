"""The admin API, end to end, against a real PostgreSQL (TEST_DATABASE_URL).

Each test checks one promise in docs/admin-api.md. Skipped when
TEST_DATABASE_URL is not set; CI always sets it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import pytest
from argon2 import PasswordHasher
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session
from typer.testing import CliRunner

import cli
import src.auth.crypto as crypto
from src.admin import runtime, users
from src.admin.metrics import write_metrics
from src.api import deps, main
from src.api.errors import ApiError
from src.auth import service
from src.auth.ratelimit import ip_limiter
from src.db.models import AuditEntry, AuthSession, Role, Site, User, Worker
from src.settings import settings
from tests.conftest import TEST_DB_URL

ADMIN_EMAIL = "admin@example.com"
SUP_EMAIL = "sup@example.com"
PASSWORD = "correct horse battery"
API = "/api/v1/admin"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fast_hashing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(crypto, "_hasher", PasswordHasher(time_cost=1, memory_cost=8,
                                                          parallelism=1))


@pytest.fixture
def db_engine(seeded_db, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "database_url", TEST_DB_URL)
    ip_limiter.reset()
    yield seeded_db
    ip_limiter.reset()


@pytest.fixture
def db(db_engine) -> Iterator[Session]:
    with Session(db_engine, expire_on_commit=False) as session:
        yield session


@pytest.fixture
def main_site(db: Session) -> Site:
    return db.scalar(select(Site).where(Site.key == "main"))


@pytest.fixture
def other_site(db: Session) -> Site:
    site = Site(key="pune_tower_b", display_name="Pune Tower B")
    db.add(site)
    db.commit()
    return site


@pytest.fixture
def people(db: Session, main_site: Site) -> dict[str, User]:
    """An admin, and a supervisor at the main site."""
    return {
        "admin": service.create_user(db, ADMIN_EMAIL, "Admin", Role.ADMIN, PASSWORD),
        "sup": service.create_user(db, SUP_EMAIL, "Sup", Role.SUPERVISOR, PASSWORD,
                                   site_id=main_site.id),
    }


def _client(fake_detector, monkeypatch) -> TestClient:
    monkeypatch.setattr(main, "init_detector", lambda model_name=None, device=None: fake_detector)
    monkeypatch.setattr(deps, "_detector", fake_detector)
    return TestClient(main.app, base_url="https://testserver")


class Browser:
    """A signed-in dashboard user: a client, and the CSRF header for writes."""

    def __init__(self, client: TestClient, email: str) -> None:
        self.client = client
        response = client.post("/api/v1/auth/login", json={"email": email, "password": PASSWORD})
        assert response.status_code == 200, response.text
        self.csrf = {"X-CSRF-Token": response.json()["csrf_token"]}

    def get(self, path: str, **kw):
        return self.client.get(f"{API}{path}", **kw)

    def post(self, path: str, body: dict | None = None):
        return self.client.post(f"{API}{path}", json=body or {}, headers=self.csrf)

    def patch(self, path: str, body: dict):
        return self.client.patch(f"{API}{path}", json=body, headers=self.csrf)

    def put(self, path: str, body: dict):
        return self.client.put(f"{API}{path}", json=body, headers=self.csrf)


@pytest.fixture
def admin(db_engine, people, fake_detector, monkeypatch) -> Iterator[Browser]:
    with _client(fake_detector, monkeypatch) as client:
        yield Browser(client, ADMIN_EMAIL)


@pytest.fixture
def sup(db_engine, people, fake_detector, monkeypatch) -> Iterator[Browser]:
    with _client(fake_detector, monkeypatch) as client:
        yield Browser(client, SUP_EMAIL)


def _job(browser: Browser, key: str) -> dict:
    return next(j for j in browser.get("/jobs").json() if j["key"] == key)


def _audit(db: Session, action: str) -> list[AuditEntry]:
    db.expire_all()
    return list(db.scalars(select(AuditEntry).where(AuditEntry.action == action)
                           .order_by(AuditEntry.id)))


def _add_worker(db: Session, code: str, site: Site) -> Worker:
    worker, _ = service.create_worker(db, code, f"Worker {code}", site_id=site.id)
    return worker


# ---------------------------------------------------------------------------
# Roles (§1)
# ---------------------------------------------------------------------------

ADMIN_ONLY = [
    ("POST", "/sites"), ("PATCH", "/sites/main"), ("POST", "/jobs"),
    ("PATCH", "/jobs/welder"), ("PUT", "/jobs/welder/requirements"), ("PUT", "/job-order"),
    ("GET", "/settings"), ("PATCH", "/settings"), ("GET", "/users"), ("POST", "/users"),
    ("PATCH", "/users/1"), ("POST", "/users/1/reset-password"), ("GET", "/audit"),
]


class TestRoles:
    @pytest.mark.parametrize(("method", "path"), ADMIN_ONLY)
    def test_supervisors_are_refused_admin_routes(self, sup: Browser, method: str,
                                                 path: str) -> None:
        response = sup.client.request(method, f"{API}{path}", json={}, headers=sup.csrf)
        assert response.status_code == 403
        assert response.json()["code"] == "forbidden"

    @pytest.mark.parametrize("path", ["/sites", "/items", "/jobs", "/workers"])
    def test_supervisors_may_read(self, sup: Browser, path: str) -> None:
        assert sup.get(path).status_code == 200

    def test_signed_in_user_carries_their_site(self, admin: Browser, sup: Browser) -> None:
        """docs/auth-api.md §6: site key for a supervisor, null for an admin."""
        assert admin.client.get("/api/v1/auth/me").json()["user"]["site"] is None
        assert sup.client.get("/api/v1/auth/me").json()["user"]["site"] == "main"


# ---------------------------------------------------------------------------
# Sites (§3)
# ---------------------------------------------------------------------------


class TestSites:
    def test_admin_sees_every_site_with_active_worker_counts(
        self, admin: Browser, db: Session, main_site: Site, other_site: Site
    ) -> None:
        _add_worker(db, "W-1", main_site)
        retired = _add_worker(db, "W-2", main_site)
        retired.is_active = False
        db.commit()
        listed = {s["key"]: s for s in admin.get("/sites").json()}
        assert set(listed) == {"main", "pune_tower_b"}
        assert listed["main"]["worker_count"] == 1
        assert listed["main"]["created_at"].endswith("Z")

    def test_supervisor_sees_only_their_site(self, sup: Browser, other_site: Site) -> None:
        assert [s["key"] for s in sup.get("/sites").json()] == ["main"]

    def test_create_and_rename(self, admin: Browser, db: Session) -> None:
        created = admin.post("/sites", {"key": "plant_7", "display_name": " Plant  7 "})
        assert created.status_code == 201
        assert created.json()["site"]["display_name"] == "Plant 7"
        renamed = admin.patch("/sites/plant_7", {"display_name": "Plant Seven"})
        assert renamed.json()["site"]["display_name"] == "Plant Seven"
        assert [e.after for e in _audit(db, "site.update")] == [{"display_name": "Plant Seven"}]

    @pytest.mark.parametrize("key", ["P7", "7plant", "p", "plant-7"])
    def test_bad_keys(self, admin: Browser, key: str) -> None:
        response = admin.post("/sites", {"key": key, "display_name": "x"})
        assert (response.json()["code"], response.json()["field"]) == ("invalid_key", "key")

    def test_duplicate_key(self, admin: Browser) -> None:
        assert admin.post("/sites", {"key": "main", "display_name": "x"}).json()["code"] == \
            "already_exists"

    def test_site_in_use_cannot_be_deactivated(self, admin: Browser, db: Session,
                                               other_site: Site) -> None:
        worker = _add_worker(db, "W-9", other_site)
        refused = admin.patch("/sites/pune_tower_b", {"is_active": False})
        assert refused.status_code == 409
        assert refused.json()["code"] == "site_in_use"
        worker.is_active = False
        db.commit()
        assert admin.patch("/sites/pune_tower_b", {"is_active": False}).json()["site"][
            "is_active"] is False

    def test_unknown_site_is_404(self, admin: Browser) -> None:
        assert admin.patch("/sites/nowhere", {"display_name": "x"}).status_code == 404


# ---------------------------------------------------------------------------
# Items (§4)
# ---------------------------------------------------------------------------


class TestItems:
    def test_accuracy_comes_from_the_models_metrics_file(
        self, admin: Browser, fake_detector, tmp_path: Path
    ) -> None:
        weights = tmp_path / "m_ppe_best.pt"
        weights.write_bytes(b"")
        write_metrics(weights, "m", {"hardhat": 0.909, "boots": 0.613}, source="test")
        fake_detector.weights_path = weights
        items = {i["key"]: i for i in admin.get("/items").json()}
        assert items["hardhat"]["detector_ap50"] == 0.909
        assert items["goggles"]["detector_ap50"] is None  # no figure in the file
        assert items["face_shield"] == {"key": "face_shield", "display_name": "Face shield",
                                        "detectable": False, "detector_ap50": None}

    def test_no_metrics_file_means_no_accuracy(self, admin: Browser, fake_detector,
                                               tmp_path: Path) -> None:
        fake_detector.weights_path = tmp_path / "none.pt"
        assert all(i["detector_ap50"] is None for i in admin.get("/items").json())


# ---------------------------------------------------------------------------
# Jobs and the matrix (§5)
# ---------------------------------------------------------------------------

SCAFFOLDER = {"key": "scaffolder", "display_name": "Scaffolder", "requirements": [
    {"item": "hardhat", "mode": "required"}, {"item": "harness", "mode": "required"},
    {"item": "boots", "mode": "pending"}]}


class TestJobs:
    def test_create_goes_last_and_reaches_the_app(self, admin: Browser, db: Session) -> None:
        response = admin.post("/jobs", SCAFFOLDER)
        assert response.status_code == 201
        body = response.json()
        assert body["job"]["version"] == 1
        assert body["job"]["position"] == 6
        assert body["applies_within_s"] == 30
        assert body["warnings"] == [{
            "code": "gemini_only_item", "item": "harness",
            "message": "Harness can only be judged by Gemini, whose accuracy on it hasn't "
                       "been measured."}]
        app_jobs = admin.client.get("/api/v1/jobs").json()
        assert app_jobs["scaffolder"] == {"required": ["hardhat", "harness"],
                                          "pending": ["boots"]}
        assert _audit(db, "job.create")[0].after["requirements"] == SCAFFOLDER["requirements"]

    @pytest.mark.parametrize(("requirements", "code", "field"), [
        ([{"item": "flippers", "mode": "required"}], "unknown_item", "requirements[0].item"),
        ([{"item": "hardhat", "mode": "required"}, {"item": "hardhat", "mode": "pending"}],
         "invalid", "requirements[1].item"),
        ([{"item": "boots", "mode": "pending"}], "no_required_item", "requirements"),
        ([{"item": "boots", "mode": "optional"}], "invalid", "requirements[0].mode"),
    ])
    def test_create_errors_point_at_the_entry(self, admin: Browser, requirements: list,
                                              code: str, field: str) -> None:
        body = admin.post("/jobs", {"key": "diver", "display_name": "Diver",
                                    "requirements": requirements}).json()
        assert (body["code"], body["field"]) == (code, field)

    def test_duplicate_job_key(self, admin: Browser) -> None:
        assert admin.post("/jobs", {**SCAFFOLDER, "key": "welder"}).json()["code"] == \
            "already_exists"

    def test_rename_bumps_the_version(self, admin: Browser, db: Session) -> None:
        response = admin.patch("/jobs/welder", {"version": 1, "display_name": "Welder (hot)"})
        assert response.json()["job"]["version"] == 2
        assert _audit(db, "job.update")[0].before == {"display_name": "Welder"}

    def test_stale_version_is_refused_with_the_current_job(self, admin: Browser) -> None:
        admin.patch("/jobs/welder", {"version": 1, "display_name": "Welder (hot)"})
        stale = admin.patch("/jobs/welder", {"version": 1, "is_active": False})
        assert stale.status_code == 409
        assert stale.json()["code"] == "version_conflict"
        assert stale.json()["current"]["display_name"] == "Welder (hot)"
        assert stale.json()["current"]["version"] == 2

    def test_last_active_job_cannot_be_deactivated(self, admin: Browser) -> None:
        keys = [j["key"] for j in admin.get("/jobs").json()]
        for key in keys[:-1]:
            assert admin.patch(f"/jobs/{key}", {"version": 1, "is_active": False}).status_code \
                == 200
        last = admin.patch(f"/jobs/{keys[-1]}", {"version": 1, "is_active": False})
        assert last.json()["code"] == "last_active_job"

    def test_replace_requirements(self, admin: Browser, db: Session) -> None:
        reqs = [{"item": "hardhat", "mode": "required"}, {"item": "safety_vest",
                                                          "mode": "required"}]
        response = admin.put("/jobs/painter/requirements", {"version": 1, "requirements": reqs})
        assert response.json()["job"]["requirements"] == reqs
        assert response.json()["job"]["version"] == 2
        entry = _audit(db, "job.requirements")[0]
        assert entry.after == {"requirements": reqs}
        assert admin.client.get("/api/v1/jobs").json()["painter"]["required"] == \
            ["hardhat", "safety_vest"]

    def test_saving_the_same_list_changes_nothing(self, admin: Browser, db: Session) -> None:
        current = _job(admin, "painter")
        again = admin.put("/jobs/painter/requirements",
                          {"version": 1, "requirements": current["requirements"]})
        assert again.json()["job"]["version"] == 1
        assert _audit(db, "job.requirements") == []

    def test_gemini_off_is_warned_on_save(self, admin: Browser) -> None:
        admin.patch("/settings", {"gemini_verify": False})
        warnings = admin.put("/jobs/painter/requirements", {"version": 1, "requirements": [
            {"item": "hardhat", "mode": "required"}]}).json()["warnings"]
        assert [w["code"] for w in warnings] == ["gemini_verify_off"]

    def test_reorder_keeps_versions(self, admin: Browser) -> None:
        keys = [j["key"] for j in admin.get("/jobs").json()]
        reordered = admin.put("/job-order", {"keys": list(reversed(keys))}).json()
        assert [j["key"] for j in reordered] == list(reversed(keys))
        assert all(j["version"] == 1 for j in reordered)
        assert list(admin.client.get("/api/v1/jobs").json()) == list(reversed(keys))

    def test_reorder_must_name_every_job_once(self, admin: Browser) -> None:
        keys = [j["key"] for j in admin.get("/jobs").json()]
        missing = admin.put("/job-order", {"keys": keys[:-1]}).json()
        assert (missing["code"], missing["field"]) == ("invalid", "keys")
        doubled = admin.put("/job-order", {"keys": [keys[0], *keys]}).json()
        assert doubled["field"] == "keys[1]"


# ---------------------------------------------------------------------------
# Runtime settings (§6)
# ---------------------------------------------------------------------------


class TestSettings:
    def test_each_setting_describes_itself(self, admin: Browser) -> None:
        listed = {s["name"]: s for s in admin.get("/settings").json()}
        assert set(listed) == {"confidence_threshold", "gemini_verify",
                               "gemini_min_confidence", "worker_auth"}
        threshold = listed["confidence_threshold"]
        assert (threshold["min"], threshold["max"], threshold["confirm"]) == (0.25, 0.80, True)
        assert listed["worker_auth"]["choices"] == ["optional", "required"]
        assert all(not s["overridden"] for s in listed.values())

    def test_merge_patch_sets_and_clears(self, admin: Browser, db: Session) -> None:
        body = admin.patch("/settings", {"gemini_min_confidence": 0.8}).json()
        saved = {s["name"]: s for s in body["settings"]}["gemini_min_confidence"]
        assert (saved["value"], saved["overridden"], saved["updated_by"]) == \
            (0.8, True, ADMIN_EMAIL)
        assert saved["updated_at"].endswith("Z")
        assert body["applies_within_s"] == 30
        cleared = admin.patch("/settings", {"gemini_min_confidence": None}).json()
        assert {s["name"]: s for s in cleared["settings"]}["gemini_min_confidence"][
            "overridden"] is False
        assert [(e.before, e.after) for e in _audit(db, "settings.update")] == [
            ({"value": None}, {"value": 0.8}), ({"value": 0.8}, {"value": None})]

    @pytest.mark.parametrize(("patch", "code", "field"), [
        ({"confidence_threshold": 0.05}, "out_of_range", "confidence_threshold"),
        ({"confidence_threshold": True}, "invalid", "confidence_threshold"),
        ({"gemini_verify": "no"}, "invalid", "gemini_verify"),
        ({"worker_auth": "sometimes"}, "invalid", "worker_auth"),
        ({"model": "yolo26l"}, "invalid", "model"),
    ])
    def test_bad_values(self, admin: Browser, patch: dict, code: str, field: str) -> None:
        body = admin.patch("/settings", patch).json()
        assert (body["code"], body["field"]) == (code, field)

    def test_all_or_nothing(self, admin: Browser) -> None:
        admin.patch("/settings", {"gemini_min_confidence": 0.9, "confidence_threshold": 2.0})
        assert all(not s["overridden"] for s in admin.get("/settings").json())

    def test_warnings(self, admin: Browser) -> None:
        warnings = admin.patch("/settings", {"worker_auth": "required", "gemini_verify": False,
                                             "confidence_threshold": 0.5}).json()["warnings"]
        assert {(w["code"], w["setting"]) for w in warnings} == {
            ("worker_auth_required", "worker_auth"),
            ("gemini_verify_switched_off", "gemini_verify"),
            ("threshold_changed", "confidence_threshold")}

    def test_settings_reach_the_checks_at_once(self, admin: Browser,
                                               sample_image_bytes: bytes) -> None:
        """Saved in the dashboard, used by the next check on this server."""
        admin.patch("/settings", {"worker_auth": "required", "confidence_threshold": 0.6})
        assert runtime.get("confidence_threshold") == 0.6
        refused = admin.client.post(
            "/api/v1/compliance/check", data={"worker_id": "W-1", "job_category": "welder"},
            files={"image": ("w.png", sample_image_bytes, "image/png")})
        assert refused.status_code == 401


# ---------------------------------------------------------------------------
# Dashboard users (§7)
# ---------------------------------------------------------------------------


class TestUsers:
    def test_create_supervisor_with_a_temporary_password(self, admin: Browser,
                                                         fake_detector, monkeypatch) -> None:
        response = admin.post("/users", {"email": "New@Example.com", "display_name": "New",
                                         "role": "supervisor", "site": "main"})
        assert response.status_code == 201
        body = response.json()
        assert body["user"]["email"] == "new@example.com"
        assert (body["user"]["site"], body["user"]["must_change_password"]) == ("main", True)
        with _client(fake_detector, monkeypatch) as other:
            login = other.post("/api/v1/auth/login", json={
                "email": "new@example.com", "password": body["temporary_password"]})
            assert login.json()["user"]["must_change_password"] is True

    @pytest.mark.parametrize(("body", "status", "code"), [
        ({"role": "supervisor"}, 422, "invalid"),
        ({"role": "admin", "site": "main"}, 422, "invalid"),
        ({"role": "supervisor", "site": "nowhere"}, 404, "not_found"),
        ({"role": "supervisor", "site": "main", "email": SUP_EMAIL}, 409, "already_exists"),
    ])
    def test_create_errors(self, admin: Browser, body: dict, status: int, code: str) -> None:
        response = admin.post("/users", {"email": "x@example.com", "display_name": "X", **body})
        assert (response.status_code, response.json()["code"]) == (status, code)

    def test_deactivated_site_is_invalid(self, admin: Browser, db: Session,
                                         other_site: Site) -> None:
        other_site.is_active = False
        db.commit()
        body = admin.post("/users", {"email": "x@example.com", "display_name": "X",
                                     "role": "supervisor", "site": "pune_tower_b"}).json()
        assert (body["code"], body["field"]) == ("invalid", "site")

    def test_cannot_change_own_role_or_deactivate_self(self, admin: Browser,
                                                       people: dict) -> None:
        me = people["admin"].id
        for change in ({"role": "supervisor", "site": "main"}, {"is_active": False}):
            assert admin.patch(f"/users/{me}", change).json()["code"] == "cannot_change_self"
        assert admin.patch(f"/users/{me}", {"display_name": "Me"}).status_code == 200

    def test_move_promote_and_deactivate(self, admin: Browser, sup: Browser, db: Session,
                                         people: dict, other_site: Site) -> None:
        sup_id = people["sup"].id
        moved = admin.patch(f"/users/{sup_id}", {"site": "pune_tower_b"}).json()["user"]
        assert moved["site"] == "pune_tower_b"
        promoted = admin.patch(f"/users/{sup_id}", {"role": "admin"}).json()["user"]
        assert (promoted["role"], promoted["site"]) == ("admin", None)
        admin.patch(f"/users/{sup_id}", {"is_active": False})
        assert sup.client.get("/api/v1/auth/me").status_code == 401  # sessions ended
        assert [e.after for e in _audit(db, "user.update")][-1] == {"is_active": False}

    def test_supervisor_needs_a_site_when_demoted(self, admin: Browser, db: Session) -> None:
        other = service.create_user(db, "boss2@example.com", "Boss", Role.ADMIN, PASSWORD)
        body = admin.patch(f"/users/{other.id}", {"role": "supervisor"}).json()
        assert (body["code"], body["field"]) == ("invalid", "site")

    def test_last_admin_rule(self, db: Session, people: dict) -> None:
        """Unreachable over HTTP (the caller is itself an active admin), so call it."""
        with pytest.raises(ApiError) as caught:
            users.update_user(db, people["admin"].id, {"is_active": False},
                              me=people["sup"], actor="t", ip=None)
        assert caught.value.code == "last_admin"

    def test_reset_password(self, admin: Browser, sup: Browser, people: dict,
                            db: Session) -> None:
        me = people["admin"].id
        assert admin.post(f"/users/{me}/reset-password").json()["code"] == "cannot_change_self"
        temporary = admin.post(f"/users/{people['sup'].id}/reset-password").json()[
            "temporary_password"]
        assert len(temporary) >= 12
        assert sup.client.get("/api/v1/auth/me").status_code == 401
        assert _audit(db, "user.reset_password")[0].after is None


# ---------------------------------------------------------------------------
# Workers (§8)
# ---------------------------------------------------------------------------


class TestWorkers:
    def test_admin_must_choose_a_site(self, admin: Browser) -> None:
        body = admin.post("/workers", {"worker_id": "W-1", "display_name": "One"}).json()
        assert (body["code"], body["field"]) == ("invalid", "site")

    def test_supervisor_adds_to_their_own_site(self, sup: Browser, db: Session) -> None:
        body = sup.post("/workers", {"worker_id": "w-1", "display_name": "One"}).json()
        assert body["worker"]["worker_id"] == "W-1"
        assert (body["worker"]["site"], body["worker"]["created_by"]) == ("main", SUP_EMAIL)
        assert len(body["one_time_pin"]) == 6
        entry = _audit(db, "worker.create")[0]
        assert (entry.actor, entry.site) == (SUP_EMAIL, "main")
        assert "pin" not in json.dumps(entry.after).lower()

    def test_supervisor_cannot_add_elsewhere(self, sup: Browser, other_site: Site) -> None:
        response = sup.post("/workers", {"worker_id": "W-2", "display_name": "Two",
                                         "site": "pune_tower_b"})
        assert (response.status_code, response.json()["code"]) == (403, "forbidden")

    def test_taken_id_is_409_even_from_another_site(self, sup: Browser, db: Session,
                                                    other_site: Site) -> None:
        _add_worker(db, "W-7", other_site)
        assert sup.post("/workers", {"worker_id": "W-7", "display_name": "x"}).json()[
            "code"] == "already_exists"

    def test_list_filters_and_pages(self, admin: Browser, db: Session, main_site: Site,
                                    other_site: Site) -> None:
        for code in ("W-1", "W-2", "W-3"):
            _add_worker(db, code, main_site)
        _add_worker(db, "P-1", other_site)
        first = admin.get("/workers", params={"limit": 2}).json()
        assert [w["worker_id"] for w in first["items"]] == ["P-1", "W-1"]
        assert first["total"] == 4
        second = admin.get("/workers", params={"limit": 2, "cursor": first["next_cursor"]}).json()
        assert [w["worker_id"] for w in second["items"]] == ["W-2", "W-3"]
        assert second["next_cursor"] is None
        at_site = admin.get("/workers", params={"site": "pune_tower_b"}).json()
        assert [w["worker_id"] for w in at_site["items"]] == ["P-1"]
        assert admin.get("/workers", params={"search": "worker w-2"}).json()["total"] == 1

    def test_locked_and_active_filters(self, admin: Browser, db: Session,
                                       main_site: Site) -> None:
        locked = _add_worker(db, "W-1", main_site)
        retired = _add_worker(db, "W-2", main_site)
        locked.locked_until = service.utcnow() + timedelta(minutes=10)
        retired.is_active = False
        db.commit()
        assert [w["worker_id"] for w in admin.get("/workers", params={"locked": True})
                .json()["items"]] == ["W-1"]
        assert [w["worker_id"] for w in admin.get("/workers", params={"active": False})
                .json()["items"]] == ["W-2"]

    def test_supervisor_sees_nothing_of_other_sites(self, sup: Browser, db: Session,
                                                    main_site: Site, other_site: Site) -> None:
        _add_worker(db, "W-1", main_site)
        _add_worker(db, "P-1", other_site)
        listed = sup.get("/workers", params={"site": "pune_tower_b"}).json()
        assert [w["worker_id"] for w in listed["items"]] == ["W-1"]  # site ignored
        assert sup.get("/workers/P-1").status_code == 404
        assert sup.patch("/workers/P-1", {"display_name": "x"}).status_code == 404
        assert sup.post("/workers/P-1/reset-pin").status_code == 404

    def test_only_admins_move_workers(self, admin: Browser, sup: Browser, db: Session,
                                      main_site: Site, other_site: Site) -> None:
        _add_worker(db, "W-1", main_site)
        assert sup.patch("/workers/W-1", {"site": "pune_tower_b"}).json()["code"] == "forbidden"
        moved = admin.patch("/workers/W-1", {"site": "pune_tower_b"}).json()["worker"]
        assert moved["site"] == "pune_tower_b"
        entry = _audit(db, "worker.update")[0]
        assert (entry.before, entry.after, entry.site) == (
            {"site": "main"}, {"site": "pune_tower_b"}, "pune_tower_b")

    def test_deactivate_ends_sessions_and_reset_pin(self, sup: Browser, db: Session,
                                                    main_site: Site) -> None:
        worker, pin = service.create_worker(db, "W-1", "One", site_id=main_site.id)
        token = sup.client.post("/api/v1/auth/worker/login",
                                json={"worker_id": "W-1", "pin": pin}).json()["token"]
        sup.patch("/workers/W-1", {"is_active": False})
        db.expire_all()
        assert db.scalar(select(AuthSession.revoked_at).where(
            AuthSession.worker_id == worker.id)) is not None
        assert sup.client.get("/api/v1/auth/worker/me",
                              headers={"Authorization": f"Bearer {token}"}).status_code == 401
        new_pin = sup.post("/workers/W-1/reset-pin").json()["one_time_pin"]
        assert len(new_pin) == 6 and new_pin != pin

    def test_cli_made_worker_has_no_creator(self, admin: Browser, db_engine) -> None:
        result = CliRunner().invoke(cli.app, ["auth", "create-worker", "--worker-id", "C-1",
                                              "--name", "From CLI", "--site", "main"])
        assert result.exit_code == 0, result.output
        assert admin.get("/workers/C-1").json()["created_by"] is None


# ---------------------------------------------------------------------------
# Audit log (§9)
# ---------------------------------------------------------------------------


class TestAudit:
    def test_entries_newest_first_with_filters_and_paging(self, admin: Browser,
                                                          db: Session) -> None:
        admin.post("/sites", {"key": "a_site", "display_name": "A"})
        admin.post("/sites", {"key": "b_site", "display_name": "B"})
        admin.patch("/settings", {"gemini_min_confidence": 0.8})
        everything = admin.get("/audit").json()
        assert [e["action"] for e in everything["items"]][:3] == [
            "settings.update", "site.create", "site.create"]
        assert everything["items"][0]["at"].endswith("Z")
        sites_only = admin.get("/audit", params={"action": "site.create", "limit": 1}).json()
        assert sites_only["total"] == 2
        assert sites_only["items"][0]["target"] == "site:b_site"
        older = admin.get("/audit", params={"action": "site.create", "limit": 1,
                                            "cursor": sites_only["next_cursor"]}).json()
        assert older["items"][0]["target"] == "site:a_site"
        assert admin.get("/audit", params={"site": "a_site"}).json()["total"] == 1

    def test_date_range(self, admin: Browser, db: Session) -> None:
        admin.post("/sites", {"key": "a_site", "display_name": "A"})
        newest = admin.get("/audit").json()["items"][0]["at"]
        assert admin.get("/audit", params={"to": newest}).json()["total"] == \
            admin.get("/audit").json()["total"] - 1

    def test_no_secrets_anywhere(self, admin: Browser, sup: Browser, db: Session,
                                 people: dict) -> None:
        sup.post("/workers", {"worker_id": "W-1", "display_name": "One"})
        sup.post("/workers/W-1/reset-pin")
        admin.post(f"/users/{people['sup'].id}/reset-password")
        dumped = json.dumps(admin.get("/audit", params={"limit": 200}).json()).lower()
        for secret in ("one_time_pin", "temporary_password", "password_hash", "pin_hash",
                       "$argon2"):
            assert secret not in dumped

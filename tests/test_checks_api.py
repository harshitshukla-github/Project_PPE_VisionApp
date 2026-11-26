"""The dashboard's checks endpoints, end to end (docs/checks-api.md §4).

Against a real PostgreSQL (TEST_DATABASE_URL); skipped without it, as the
other database tests are. Checks are written straight into the table, except
in the end-to-end test, which makes one through the app's endpoint.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from argon2 import PasswordHasher
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import select, update
from sqlalchemy.orm import Session
from typer.testing import CliRunner

import cli
import src.auth.crypto as crypto
from src.admin import checks as admin_checks
from src.api import deps, main
from src.auth import service
from src.auth.ratelimit import ip_limiter
from src.checks.photos import FolderPhotoStore
from src.db.models import AuditEntry, Check, Role, Site, User, Worker
from src.settings import settings
from tests.conftest import TEST_DB_URL, make_detection

ADMIN_EMAIL = "admin@example.com"
SUP_EMAIL = "sup@example.com"
PASSWORD = "correct horse battery"
API = "/api/v1/admin"
T0 = datetime(2026, 11, 20, 8, 0, tzinfo=timezone.utc)

# Every key of a check in docs/checks-api.md §2; the list leaves out two (§4.1).
CHECK_KEYS = {
    "id", "at", "worker_id", "worker_name", "site", "signed_in", "job", "job_name",
    "job_version", "required", "pending", "verdict", "verdict_source", "missing_equipment",
    "violations_detected", "unverifiable_equipment", "detections", "verification",
    "explanation", "model_used", "confidence_threshold", "gemini_verify",
    "gemini_min_confidence", "photo", "training_consent", "review",
}
LIST_KEYS = CHECK_KEYS - {"detections", "verification"}


def jpeg(colour: tuple[int, int, int] = (10, 120, 200)) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (40, 20), colour).save(out, "JPEG")
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

    def delete(self, path: str):
        return self.client.delete(f"{API}{path}", headers=self.csrf)


def _browser(email: str, fake_detector, monkeypatch) -> Iterator[Browser]:
    monkeypatch.setattr(main, "init_detector", lambda model_name=None, device=None: fake_detector)
    monkeypatch.setattr(deps, "_detector", fake_detector)
    with TestClient(main.app, base_url="https://testserver") as client:
        yield Browser(client, email)


@pytest.fixture
def admin(db_engine, people, fake_detector, monkeypatch) -> Iterator[Browser]:
    yield from _browser(ADMIN_EMAIL, fake_detector, monkeypatch)


@pytest.fixture
def sup(db_engine, people, fake_detector, monkeypatch) -> Iterator[Browser]:
    yield from _browser(SUP_EMAIL, fake_detector, monkeypatch)


@pytest.fixture
def store(photo_dir: Path) -> FolderPhotoStore:
    return FolderPhotoStore(photo_dir)


def add_worker(db: Session, code: str, site: Site, name: str | None = None) -> Worker:
    worker, _ = service.create_worker(db, code, name or f"Worker {code}", site_id=site.id)
    return worker


def add_check(db: Session, *, site: Site | None, worker: Worker | None = None,
              code: str | None = None, at: datetime = T0, verdict: str = "PASS",
              signed_in: bool = True, job: str = "general_labor",
              photo: bytes | None = None, store: FolderPhotoStore | None = None,
              review: tuple[str, User] | None = None) -> Check:
    """A stored check, as src.checks.record writes one."""
    key = store.save(photo, "image/jpeg") if photo is not None else None
    row = Check(
        at=at, worker_id=worker.id if worker else None,
        worker_code=code or (worker.worker_code if worker else "NOBODY-1"),
        signed_in=signed_in, site_id=site.id if site else None,
        job_key=job, job_name=job.replace("_", " ").capitalize(), job_version=1,
        required=["hardhat", "safety_vest", "boots"], pending=[],
        verdict=verdict, verdict_source="detector", confidence_avg=0.87,
        missing_equipment=[], violations_detected=[], unverifiable_equipment=[],
        detections=[{"class": "hardhat", "confidence": 0.92, "bbox": [10.0, 10.0, 100.0, 200.0]}],
        verification=None, explanation=None, model_used="yolo11l", inference_time_ms=23.4,
        gemini_invoked=False, confidence_threshold=0.45, gemini_verify=True,
        gemini_min_confidence=0.75, photo_state="kept" if key else "not_stored", photo_key=key,
        photo_type="image/jpeg", photo_width=40, photo_height=20,
        photo_bytes=len(photo) if photo else 1000,
    )
    if review:
        row.review_outcome, row.reviewed_by, row.reviewed_at = review[0], review[1].id, at
    db.add(row)
    db.commit()
    return row


def audit(db: Session, action: str) -> list[AuditEntry]:
    db.expire_all()
    return list(db.scalars(select(AuditEntry).where(AuditEntry.action == action)
                           .order_by(AuditEntry.id)))


def fresh(db: Session, check: Check) -> Check:
    db.expire_all()
    return db.get(Check, check.id)


def ids(response) -> list[int]:
    assert response.status_code == 200, response.text
    return [item["id"] for item in response.json()["items"]]


# ---------------------------------------------------------------------------
# Who sees what (§1)
# ---------------------------------------------------------------------------


@pytest.fixture
def three_sites(db: Session, people, main_site: Site, other_site: Site) -> dict[str, Check]:
    """A check at the main site, one at another, and one with no site."""
    here = add_worker(db, "W-0001", main_site)
    there = add_worker(db, "W-0002", other_site)
    return {
        "here": add_check(db, site=main_site, worker=here),
        "there": add_check(db, site=other_site, worker=there, at=T0 + timedelta(minutes=1)),
        "nowhere": add_check(db, site=None, code="nobody-1", signed_in=False,
                             at=T0 + timedelta(minutes=2)),
    }


class TestWhoSeesWhat:
    def test_an_admin_sees_every_check_with_no_site_included(self, admin, three_sites) -> None:
        listed = ids(admin.get("/checks"))
        assert listed == [three_sites[k].id for k in ("nowhere", "there", "here")]

    def test_a_supervisor_sees_only_their_sites(self, sup, three_sites) -> None:
        assert ids(sup.get("/checks")) == [three_sites["here"].id]
        # whatever `site` says
        assert ids(sup.get("/checks", params={"site": "pune_tower_b"})) == [three_sites["here"].id]
        assert sup.get("/checks").json()["total"] == 1

    @pytest.mark.parametrize("which", ["there", "nowhere"])
    @pytest.mark.parametrize("path", ["", "/photo"])
    def test_another_sites_check_is_not_found_for_a_supervisor(self, sup, three_sites, which,
                                                               path) -> None:
        response = sup.get(f"/checks/{three_sites[which].id}{path}")
        assert (response.status_code, response.json()["code"]) == (404, "not_found")

    @pytest.mark.parametrize("which", ["there", "nowhere"])
    def test_nor_can_a_supervisor_review_it(self, sup, db, three_sites, which) -> None:
        response = sup.post(f"/checks/{three_sites[which].id}/review", {"outcome": "compliant"})
        assert (response.status_code, response.json()["code"]) == (404, "not_found")
        assert fresh(db, three_sites[which]).review_outcome is None

    def test_a_supervisor_cant_delete_photos(self, sup, three_sites) -> None:
        response = sup.delete("/workers/W-0001/photos")
        assert (response.status_code, response.json()["code"]) == (403, "forbidden")

    def test_signing_in_is_needed(self, db_engine, three_sites, fake_detector,
                                  monkeypatch) -> None:
        monkeypatch.setattr(main, "init_detector",
                            lambda model_name=None, device=None: fake_detector)
        monkeypatch.setattr(deps, "_detector", fake_detector)
        with TestClient(main.app, base_url="https://testserver") as anonymous:
            for path in ("/checks", f"/checks/{three_sites['here'].id}",
                         f"/checks/{three_sites['here'].id}/photo"):
                assert anonymous.get(f"{API}{path}").status_code == 401


# ---------------------------------------------------------------------------
# The list (§4.1)
# ---------------------------------------------------------------------------


@pytest.fixture
def many(db: Session, people, main_site: Site, other_site: Site) -> dict[str, Check]:
    w1 = add_worker(db, "W-0001", main_site)
    w2 = add_worker(db, "W-0002", other_site)
    minute = timedelta(minutes=1)
    return {
        "c1": add_check(db, site=main_site, worker=w1, verdict="PASS"),
        "c2": add_check(db, site=main_site, worker=w1, verdict="NEEDS_REVIEW", at=T0 + minute,
                        review=("compliant", people["sup"])),
        "c3": add_check(db, site=other_site, worker=w2, verdict="FAIL", at=T0 + 2 * minute),
        "c4": add_check(db, site=None, code="nobody-1", signed_in=False,
                        verdict="NEEDS_REVIEW", at=T0 + 3 * minute),
        # typed into an app without login, naming W-0001
        "c5": add_check(db, site=main_site, worker=w1, code=" w-0001 ", signed_in=False,
                        job="welder", at=T0 + 4 * minute),
    }


def z(moment: datetime) -> str:
    return moment.isoformat().replace("+00:00", "Z")


class TestTheList:
    @pytest.mark.parametrize(("query", "expected"), [
        ({}, ["c5", "c4", "c3", "c2", "c1"]),
        ({"verdict": "NEEDS_REVIEW"}, ["c4", "c2"]),
        ({"verdict": "NEEDS_REVIEW", "reviewed": "false"}, ["c4"]),
        ({"reviewed": "true"}, ["c2"]),
        ({"signed_in": "false"}, ["c5", "c4"]),
        ({"worker": "w-0001"}, ["c5", "c2", "c1"]),
        ({"worker": "NOBODY-1"}, ["c4"]),
        ({"job": "welder"}, ["c5"]),
        ({"site": "main"}, ["c5", "c2", "c1"]),
        ({"site": "nowhere_known"}, []),
        # from is inclusive, to exclusive
        ({"from": z(T0 + timedelta(minutes=1)), "to": z(T0 + timedelta(minutes=3))}, ["c3", "c2"]),
    ])
    def test_filters(self, admin, many, query, expected) -> None:
        response = admin.get("/checks", params=query)
        assert ids(response) == [many[name].id for name in expected]
        assert response.json()["total"] == len(expected)

    def test_checks_without_a_login_since_a_date_for_the_switch_to_required(self, admin,
                                                                           many) -> None:
        """§4.1: the count the dashboard shows next to worker_auth."""
        since = admin.get("/checks", params={"signed_in": "false",
                                             "from": z(T0 + timedelta(minutes=4)), "limit": 1})
        assert since.json()["total"] == 1

    def test_items_leave_out_detections_and_verification(self, admin, many) -> None:
        for item in admin.get("/checks").json()["items"]:
            assert set(item) == LIST_KEYS

    def test_pages_newest_first_without_gaps(self, admin, db, people, main_site) -> None:
        worker = add_worker(db, "W-0003", main_site)
        # two at the same moment: the newer ID comes first, and paging keeps both
        made = [add_check(db, site=main_site, worker=worker, at=T0 + timedelta(minutes=m))
                for m in (0, 1, 1, 2, 3)]
        seen, cursor = [], None
        while True:
            response = admin.get("/checks", params={"limit": 2, **({"cursor": cursor} if cursor
                                                                    else {})})
            body = response.json()
            assert body["total"] == 5
            seen += [item["id"] for item in body["items"]]
            cursor = body["next_cursor"]
            if cursor is None:
                break
        assert seen == [made[i].id for i in (4, 3, 2, 1, 0)]

    @pytest.mark.parametrize(("query", "field"), [
        ({"verdict": "MAYBE"}, "verdict"),
        ({"cursor": "not-a-cursor"}, "cursor"),
        ({"cursor": "WzEsMl0"}, "cursor"),  # [1, 2]: valid JSON, not a check cursor
    ])
    def test_a_bad_filter_is_invalid(self, admin, many, query, field) -> None:
        response = admin.get("/checks", params=query)
        assert response.status_code == 422
        assert (response.json()["code"], response.json()["field"]) == ("invalid", field)


# ---------------------------------------------------------------------------
# One check and its photo (§4.2)
# ---------------------------------------------------------------------------


class TestOneCheck:
    def test_has_every_field_of_the_contract(self, admin, db, people, main_site) -> None:
        worker = add_worker(db, "W-0001", main_site, name="Asha Verma")
        check = add_check(db, site=main_site, worker=worker)
        body = admin.get(f"/checks/{check.id}").json()
        assert set(body) == CHECK_KEYS
        assert (body["worker_id"], body["worker_name"], body["site"]) == ("W-0001", "Asha Verma",
                                                                         "main")
        assert body["at"] == "2026-11-20T08:00:00Z"
        assert body["photo"] == {"state": "not_stored", "width": 40, "height": 20,
                                 "bytes": 1000, "type": "image/jpeg"}
        assert body["review"] is None

    def test_an_id_that_names_nobody_has_no_name_or_site(self, admin, db, people) -> None:
        check = add_check(db, site=None, code="nobody-1", signed_in=False)
        body = admin.get(f"/checks/{check.id}").json()
        assert (body["worker_id"], body["worker_name"], body["site"], body["signed_in"]) == (
            "nobody-1", None, None, False)

    def test_no_such_check(self, admin) -> None:
        response = admin.get("/checks/999999")
        assert (response.status_code, response.json()["code"]) == (404, "not_found")


class TestThePhoto:
    def test_is_the_stored_file_and_stays_out_of_caches(self, admin, sup, db, people,
                                                        main_site, store) -> None:
        photo = jpeg()
        check = add_check(db, site=main_site, worker=add_worker(db, "W-0001", main_site),
                          photo=photo, store=store)
        for browser in (admin, sup):
            response = browser.get(f"/checks/{check.id}/photo")
            assert response.status_code == 200
            assert response.content == photo
            assert response.headers["content-type"] == "image/jpeg"
            assert response.headers["cache-control"] == "private, no-store"
            # inline, and no file name that could say whose photo it is
            assert response.headers["content-disposition"] == "inline"

    def test_never_stored_is_not_found(self, admin, db, people, main_site) -> None:
        check = add_check(db, site=main_site)
        response = admin.get(f"/checks/{check.id}/photo")
        assert (response.status_code, response.json()["code"]) == (404, "not_found")

    def test_deleted_is_gone(self, admin, db, people, main_site) -> None:
        check = add_check(db, site=main_site)
        check.photo_state = "deleted"
        db.commit()
        response = admin.get(f"/checks/{check.id}/photo")
        assert (response.status_code, response.json()["code"]) == (410, "photo_deleted")


# ---------------------------------------------------------------------------
# Reviewing (§4.3)
# ---------------------------------------------------------------------------


@pytest.fixture
def waiting(db: Session, people, main_site: Site) -> Check:
    return add_check(db, site=main_site, worker=add_worker(db, "W-0001", main_site),
                     verdict="NEEDS_REVIEW")


class TestReview:
    def test_is_recorded_beside_the_verdict(self, sup, db, waiting) -> None:
        response = sup.post(f"/checks/{waiting.id}/review",
                            {"outcome": "compliant", "note": "  Gloves in his left hand.  "})
        assert response.status_code == 200, response.text
        body = response.json()["check"]
        assert set(body) == CHECK_KEYS
        assert body["verdict"] == "NEEDS_REVIEW"  # what the system decided stays
        review = body["review"]
        assert (review["outcome"], review["note"], review["by"]) == (
            "compliant", "Gloves in his left hand.", SUP_EMAIL)
        assert review["at"].endswith("Z")

        [entry] = audit(db, "check.review")
        assert (entry.actor, entry.target, entry.site) == (SUP_EMAIL, f"check:{waiting.id}", "main")
        assert (entry.before, entry.after) == (
            None, {"outcome": "compliant", "note": "Gloves in his left hand."})

    @pytest.mark.parametrize("verdict", ["PASS", "FAIL"])
    def test_any_verdict_can_be_reviewed(self, sup, db, people, main_site, verdict) -> None:
        check = add_check(db, site=main_site, verdict=verdict)
        response = sup.post(f"/checks/{check.id}/review", {"outcome": "not_compliant"})
        assert response.json()["check"]["review"]["outcome"] == "not_compliant"

    def test_once_a_second_reviewer_gets_the_first_review(self, admin, sup, db, waiting) -> None:
        assert admin.post(f"/checks/{waiting.id}/review", {"outcome": "compliant"}).status_code == 200
        for browser in (sup, admin):  # an admin too, without replace
            response = browser.post(f"/checks/{waiting.id}/review",
                                    {"outcome": "not_compliant", "note": "Mine"})
            assert (response.status_code, response.json()["code"]) == (409, "already_reviewed")
            current = response.json()["current"]
            assert (current["review"]["outcome"], current["review"]["by"]) == ("compliant",
                                                                               ADMIN_EMAIL)
        assert fresh(db, waiting).review_outcome == "compliant"
        assert len(audit(db, "check.review")) == 1

    def test_an_admin_can_replace_a_review(self, admin, sup, db, waiting) -> None:
        sup.post(f"/checks/{waiting.id}/review", {"outcome": "compliant", "note": "Fine"})
        response = admin.post(f"/checks/{waiting.id}/review",
                              {"outcome": "not_compliant", "replace": True})
        assert response.status_code == 200
        assert response.json()["check"]["review"] == {
            "outcome": "not_compliant", "note": None, "by": ADMIN_EMAIL,
            "at": response.json()["check"]["review"]["at"]}
        first, second = audit(db, "check.review")
        assert second.before == first.after == {"outcome": "compliant", "note": "Fine"}
        assert (second.actor, second.after) == (ADMIN_EMAIL,
                                                {"outcome": "not_compliant", "note": None})

    def test_a_supervisor_cant_send_replace(self, sup, db, waiting) -> None:
        response = sup.post(f"/checks/{waiting.id}/review",
                            {"outcome": "compliant", "replace": True})
        assert response.status_code == 422
        assert (response.json()["code"], response.json()["field"]) == ("invalid", "replace")
        assert fresh(db, waiting).review_outcome is None

    @pytest.mark.parametrize(("body", "field"), [
        ({}, "outcome"),
        ({"outcome": "maybe"}, "outcome"),
        ({"outcome": "compliant", "note": "x" * 501}, "note"),
    ])
    def test_invalid_reviews(self, sup, db, waiting, body, field) -> None:
        response = sup.post(f"/checks/{waiting.id}/review", body)
        assert response.status_code == 422
        assert (response.json()["code"], response.json()["field"]) == ("invalid", field)
        assert audit(db, "check.review") == []

    def test_a_note_of_500_characters_or_only_spaces(self, sup, admin, db, waiting) -> None:
        response = sup.post(f"/checks/{waiting.id}/review",
                            {"outcome": "compliant", "note": "x" * 500})
        assert len(response.json()["check"]["review"]["note"]) == 500
        response = admin.post(f"/checks/{waiting.id}/review",
                              {"outcome": "compliant", "note": "   ", "replace": True})
        assert response.json()["check"]["review"]["note"] is None

    def test_needs_the_csrf_token(self, sup, db, waiting) -> None:
        response = sup.client.post(f"{API}/checks/{waiting.id}/review",
                                   json={"outcome": "compliant"})
        assert (response.status_code, response.json()["code"]) == (403, "csrf_failed")


# ---------------------------------------------------------------------------
# Deleting a worker's photos (§4.4)
# ---------------------------------------------------------------------------


@pytest.fixture
def photos(db: Session, people, main_site: Site, other_site: Site,
           store: FolderPhotoStore) -> dict[str, Check]:
    asha = add_worker(db, "W-0001", main_site)
    ravi = add_worker(db, "W-0002", other_site)
    return {
        "asha_1": add_check(db, site=main_site, worker=asha, photo=jpeg(), store=store,
                            review=("compliant", people["sup"])),
        "asha_2": add_check(db, site=main_site, worker=asha, photo=jpeg((200, 0, 0)), store=store,
                            verdict="FAIL", at=T0 + timedelta(minutes=1)),
        "asha_none": add_check(db, site=main_site, worker=asha, at=T0 + timedelta(minutes=2)),
        "ravi": add_check(db, site=other_site, worker=ravi, photo=jpeg(), store=store),
    }


def on_disk(folder: Path) -> set[str]:
    return {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}


class TestDeletingPhotos:
    def test_deletes_the_files_and_keeps_the_checks(self, admin, db, photos, photo_dir) -> None:
        keys = {name: c.photo_key for name, c in photos.items()}
        response = admin.delete("/workers/w-0001/photos")
        assert (response.status_code, response.json()) == (200, {"deleted": 2})

        assert on_disk(photo_dir) == {keys["ravi"]}  # gone from disk, not only from the records
        for name in ("asha_1", "asha_2"):
            row = fresh(db, photos[name])
            assert (row.photo_state, row.photo_key) == ("deleted", None)
            assert row.photo_deleted_at is not None
            got = admin.get(f"/checks/{row.id}/photo")
            assert (got.status_code, got.json()["code"]) == (410, "photo_deleted")
        # the verdicts and reviews stay
        assert fresh(db, photos["asha_1"]).review_outcome == "compliant"
        assert fresh(db, photos["asha_2"]).verdict == "FAIL"
        assert fresh(db, photos["asha_none"]).photo_state == "not_stored"
        assert fresh(db, photos["ravi"]).photo_state == "kept"

        [entry] = audit(db, "worker.photos_delete")
        assert (entry.actor, entry.target, entry.site) == (ADMIN_EMAIL, "worker:W-0001", "main")
        assert (entry.before, entry.after) == (None, {"deleted": 2})

    def test_again_deletes_nothing_and_says_so(self, admin, db, photos) -> None:
        admin.delete("/workers/W-0001/photos")
        assert admin.delete("/workers/W-0001/photos").json() == {"deleted": 0}
        assert [e.after for e in audit(db, "worker.photos_delete")] == [{"deleted": 2},
                                                                        {"deleted": 0}]

    def test_no_such_worker(self, admin, photos) -> None:
        response = admin.delete("/workers/W-9999/photos")
        assert (response.status_code, response.json()["code"]) == (404, "not_found")

    def test_a_file_that_cant_be_deleted_stops_it_and_keeps_what_was_done(
            self, admin, db, photos, photo_dir, monkeypatch) -> None:
        class FailsSecond(FolderPhotoStore):
            calls = 0

            def delete(self, key: str) -> None:
                FailsSecond.calls += 1
                if FailsSecond.calls == 2:
                    raise PermissionError("in use")
                super().delete(key)

        monkeypatch.setattr(admin_checks, "photo_store", lambda: FailsSecond(photo_dir))
        with pytest.raises(PermissionError):
            admin.delete("/workers/W-0001/photos")

        first, second = fresh(db, photos["asha_1"]), fresh(db, photos["asha_2"])
        assert (first.photo_state, second.photo_state) == ("deleted", "kept")
        assert (photo_dir / second.photo_key).exists()
        assert [e.after for e in audit(db, "worker.photos_delete")] == [{"deleted": 1}]

        monkeypatch.setattr(admin_checks, "photo_store", lambda: FolderPhotoStore(photo_dir))
        assert admin.delete("/workers/W-0001/photos").json() == {"deleted": 1}  # the rest

    def test_from_the_cli(self, db_engine, db, photos, photo_dir) -> None:
        runner = CliRunner()
        declined = runner.invoke(cli.app, ["photos", "delete", "--worker", "w-0001"], input="n\n")
        assert declined.exit_code == 1
        assert "Delete 2 photos of W-0001 (Worker W-0001)?" in declined.output
        assert fresh(db, photos["asha_1"]).photo_state == "kept"

        done = runner.invoke(cli.app, ["photos", "delete", "--worker", "w-0001", "--yes"])
        assert done.exit_code == 0, done.output
        assert "Deleted 2 photos of W-0001" in done.output
        assert on_disk(photo_dir) == {fresh(db, photos["ravi"]).photo_key}
        [entry] = audit(db, "worker.photos_delete")
        assert (entry.actor, entry.ip, entry.after) == ("cli", None, {"deleted": 2})

        again = runner.invoke(cli.app, ["photos", "delete", "--worker", "W-0001"])
        assert (again.exit_code, "has no stored photos" in again.output) == (0, True)

        unknown = runner.invoke(cli.app, ["photos", "delete", "--worker", "W-9999", "--yes"])
        assert (unknown.exit_code, "No worker W-9999." in unknown.output) == (1, True)


# ---------------------------------------------------------------------------
# End to end: a check from the app, as the dashboard sees it
# ---------------------------------------------------------------------------


def test_a_check_from_the_app_reaches_the_dashboard_as_stored(admin, db, people, photo_dir,
                                                              fake_detector) -> None:
    fake_detector.detections = [make_detection("hardhat", 0.92),
                                make_detection("safety_vest", 0.88), make_detection("boots", 0.8)]
    _, issued = service.create_worker(db, "W-0042", "Asha Verma")
    client = admin.client
    token = client.post("/api/v1/auth/worker/login",
                        json={"worker_id": "W-0042", "pin": issued}).json()["token"]
    bearer = {"Authorization": f"Bearer {token}"}
    assert client.post("/api/v1/auth/worker/pin", headers=bearer,
                       json={"current_pin": issued, "new_pin": "730154"}).status_code == 204
    db.execute(update(Worker).where(Worker.worker_code == "W-0042")
               .values(consent_seen_version=1))
    db.commit()

    exif = Image.Exif()
    exif[0x8825] = {1: "N", 2: (18.0, 31.0, 12.5)}
    out = io.BytesIO()
    Image.new("RGB", (40, 20), (10, 120, 200)).save(out, "JPEG", exif=exif.tobytes())
    answer = client.post("/api/v1/compliance/check", headers=bearer,
                         data={"job_category": "general_labor"},
                         files={"image": ("photo.jpg", out.getvalue(), "image/jpeg")}).json()

    check = admin.get(f"/checks/{answer['check_id']}").json()
    for field in ("verdict", "verdict_source", "detections", "missing_equipment",
                  "violations_detected", "unverifiable_equipment", "explanation", "model_used"):
        assert check[field] == answer[field], field
    assert (check["worker_id"], check["worker_name"], check["site"], check["signed_in"]) == (
        "W-0042", "Asha Verma", "main", True)
    assert (check["job"], check["photo"]["state"]) == ("general_labor", "kept")
    assert check["at"][:19] == answer["timestamp"][:19]

    photo = admin.get(f"/checks/{answer['check_id']}/photo")
    assert photo.status_code == 200
    assert dict(Image.open(io.BytesIO(photo.content)).getexif().get_ifd(0x8825)) == {}
    assert check["photo"]["bytes"] == len(photo.content)

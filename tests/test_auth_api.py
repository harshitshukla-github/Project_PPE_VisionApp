"""Login, end to end, against a real PostgreSQL (TEST_DATABASE_URL).

Each test checks one promise in docs/auth-api.md. Skipped when
TEST_DATABASE_URL is not set; CI always sets it.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta

import pytest
from argon2 import PasswordHasher
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select, update
from sqlalchemy.orm import Session
from typer.testing import CliRunner

import cli
import src.auth.crypto as crypto
from src.api import deps, main
from src.auth import service
from src.auth.crypto import token_hash
from src.auth.deps import require_user
from src.auth.errors import AuthError, auth_error_handler
from src.auth.ratelimit import ip_limiter
from src.db.models import AuthSession, Role, Site, User, Worker
from src.settings import settings
from tests.conftest import TEST_DB_URL

ISSUED = "W-0042"
NEW_PIN = "730154"
ADMIN_EMAIL = "admin@example.com"
ADMIN_PASSWORD = "correct horse battery"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fast_hashing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cheap Argon2 parameters, so dozens of logins take milliseconds."""
    monkeypatch.setattr(crypto, "_hasher", PasswordHasher(time_cost=1, memory_cost=8,
                                                          parallelism=1))


@pytest.fixture
def db_engine(seeded_db, monkeypatch: pytest.MonkeyPatch):
    """The test database, used by the API as its DATABASE_URL."""
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
    """An API client over https, so the Secure session cookie is sent back."""
    monkeypatch.setattr(main, "init_detector", lambda model_name=None, device=None: fake_detector)
    monkeypatch.setattr(deps, "_detector", fake_detector)
    with TestClient(main.app, base_url="https://testserver") as test_client:
        yield test_client


@pytest.fixture
def issued_pin(db: Session) -> str:
    """Worker W-0042, created with a one-time PIN."""
    _, pin = service.create_worker(db, ISSUED, "Asha Verma")
    return pin


@pytest.fixture
def worker_token(client: TestClient, issued_pin: str) -> str:
    """A signed-in worker who has replaced the issued PIN with NEW_PIN."""
    token = _worker_login(client, ISSUED, issued_pin).json()["token"]
    response = client.post("/api/v1/auth/worker/pin", headers=_bearer(token),
                           json={"current_pin": issued_pin, "new_pin": NEW_PIN})
    assert response.status_code == 204
    return token


@pytest.fixture
def admin(db: Session) -> User:
    return service.create_user(db, ADMIN_EMAIL, "Harshit", Role.ADMIN, ADMIN_PASSWORD)


@pytest.fixture
def csrf(client: TestClient, admin: User) -> str:
    """Sign the admin in; the client now holds the session cookie."""
    response = _user_login(client, ADMIN_EMAIL, ADMIN_PASSWORD)
    assert response.status_code == 200
    return response.json()["csrf_token"]


def _main_site(db: Session) -> int:
    """The site migration 0003 creates; every test database has it."""
    return db.scalar(select(Site.id).where(Site.key == "main"))


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _worker_login(client: TestClient, worker_id: str, pin: str):
    return client.post("/api/v1/auth/worker/login", json={"worker_id": worker_id, "pin": pin})


def _user_login(client: TestClient, email: str, password: str):
    return client.post("/api/v1/auth/login", json={"email": email, "password": password})


def _check(client: TestClient, image: bytes, headers: dict | None = None, worker_id: str | None
           = "W-FORM", job: str = "general_labor"):
    data = {"job_category": job}
    if worker_id is not None:
        data["worker_id"] = worker_id
    return client.post("/api/v1/compliance/check", headers=headers or {}, data=data,
                       files={"image": ("worker.png", image, "image/png")})


# ---------------------------------------------------------------------------
# Worker login
# ---------------------------------------------------------------------------


class TestWorkerLogin:
    def test_login_returns_a_token_and_asks_for_a_new_pin(
        self, client: TestClient, issued_pin: str
    ) -> None:
        body = _worker_login(client, ISSUED, issued_pin).json()
        assert len(body["token"]) == 43
        assert body["token_type"] == "bearer"
        assert body["must_change_pin"] is True
        assert body["worker"] == {"worker_id": ISSUED, "display_name": "Asha Verma"}

    def test_timestamps_are_utc_with_z(self, client: TestClient, issued_pin: str,
                                       admin: User) -> None:
        """docs/auth-api.md §3: UTC with a Z, whatever PostgreSQL's time zone."""
        login = _worker_login(client, ISSUED, issued_pin).json()
        me = client.get("/api/v1/auth/worker/me", headers=_bearer(login["token"])).json()
        dashboard = _user_login(client, ADMIN_EMAIL, ADMIN_PASSWORD).json()
        dashboard_me = client.get("/api/v1/auth/me").json()
        for stamp in (login["expires_at"], me["expires_at"], dashboard["expires_at"],
                      dashboard_me["expires_at"]):
            assert stamp.endswith("Z"), stamp

    def test_session_lasts_one_shift(self, client: TestClient, issued_pin: str,
                                     db: Session) -> None:
        _worker_login(client, ISSUED, issued_pin)
        record = db.scalar(select(AuthSession))
        assert record.expires_at - record.created_at == timedelta(hours=12)

    def test_worker_id_ignores_case(self, client: TestClient, issued_pin: str) -> None:
        assert _worker_login(client, "w-0042", issued_pin).status_code == 200

    def test_wrong_pin_and_unknown_worker_look_the_same(
        self, client: TestClient, issued_pin: str
    ) -> None:
        wrong_pin = _worker_login(client, ISSUED, "000001")
        no_worker = _worker_login(client, "W-9999", issued_pin)
        assert wrong_pin.status_code == no_worker.status_code == 401
        assert wrong_pin.json() == no_worker.json() == {
            "detail": "Wrong worker ID or PIN.", "code": "invalid_credentials",
        }

    def test_database_keeps_only_a_hash_of_the_token(
        self, client: TestClient, issued_pin: str, db: Session
    ) -> None:
        token = _worker_login(client, ISSUED, issued_pin).json()["token"]
        record = db.scalar(select(AuthSession))
        assert record.token_hash == token_hash(token)
        assert token not in {record.token_hash, record.csrf_token}

    def test_fifth_wrong_pin_locks_for_15_minutes(
        self, client: TestClient, issued_pin: str
    ) -> None:
        for _ in range(4):
            assert _worker_login(client, ISSUED, "000001").status_code == 401
        locked = _worker_login(client, ISSUED, "000001")
        assert locked.status_code == 423
        assert locked.json()["code"] == "account_locked"
        assert locked.json()["retry_after_s"] == 900
        assert locked.headers["Retry-After"] == "900"
        assert _worker_login(client, ISSUED, issued_pin).status_code == 423

    def test_lock_ends_after_its_time(self, client: TestClient, issued_pin: str,
                                      db: Session) -> None:
        for _ in range(5):
            _worker_login(client, ISSUED, "000001")
        db.execute(update(Worker).values(locked_until=service.utcnow() - timedelta(seconds=1)))
        db.commit()
        assert _worker_login(client, ISSUED, issued_pin).status_code == 200

    def test_a_good_login_resets_the_count(self, client: TestClient, issued_pin: str) -> None:
        for _ in range(4):
            _worker_login(client, ISSUED, "000001")
        assert _worker_login(client, ISSUED, issued_pin).status_code == 200
        for _ in range(4):
            assert _worker_login(client, ISSUED, "000001").status_code == 401

    def test_disabled_worker_is_told_only_with_the_right_pin(
        self, client: TestClient, issued_pin: str, db: Session
    ) -> None:
        db.execute(update(Worker).values(is_active=False))
        db.commit()
        assert _worker_login(client, ISSUED, "000001").json()["code"] == "invalid_credentials"
        right = _worker_login(client, ISSUED, issued_pin)
        assert right.status_code == 403
        assert right.json()["code"] == "account_disabled"

    def test_too_many_failures_from_one_address_gets_429(
        self, client: TestClient, issued_pin: str
    ) -> None:
        for n in range(20):  # different IDs, so no single account locks first
            _worker_login(client, f"W-{n:04d}", "000001")
        blocked = _worker_login(client, ISSUED, issued_pin)
        assert blocked.status_code == 429
        assert blocked.json()["code"] == "too_many_attempts"
        assert int(blocked.headers["Retry-After"]) > 0


# ---------------------------------------------------------------------------
# Choosing a PIN
# ---------------------------------------------------------------------------


class TestPinChange:
    def test_issued_pin_allows_only_changing_it(
        self, client: TestClient, issued_pin: str, sample_image_bytes: bytes
    ) -> None:
        token = _worker_login(client, ISSUED, issued_pin).json()["token"]
        assert client.get("/api/v1/auth/worker/me", headers=_bearer(token)).status_code == 200
        blocked = _check(client, sample_image_bytes, _bearer(token))
        assert blocked.status_code == 403
        assert blocked.json()["code"] == "pin_change_required"

    def test_after_changing_the_pin_checks_use_the_signed_in_worker(
        self, client: TestClient, worker_token: str, sample_image_bytes: bytes
    ) -> None:
        response = _check(client, sample_image_bytes, _bearer(worker_token), worker_id="W-OTHER")
        assert response.status_code == 200
        assert response.json()["worker_id"] == ISSUED

    def test_new_pin_works_and_issued_pin_no_longer_does(
        self, client: TestClient, worker_token: str, issued_pin: str
    ) -> None:
        assert _worker_login(client, ISSUED, issued_pin).status_code == 401
        signed_in = _worker_login(client, ISSUED, NEW_PIN).json()
        assert signed_in["must_change_pin"] is False

    @pytest.mark.parametrize("new_pin", ["123456", "111111", "654321", "12345", "abcdef"])
    def test_weak_pins_are_refused(self, client: TestClient, issued_pin: str,
                                   new_pin: str) -> None:
        token = _worker_login(client, ISSUED, issued_pin).json()["token"]
        response = client.post("/api/v1/auth/worker/pin", headers=_bearer(token),
                               json={"current_pin": issued_pin, "new_pin": new_pin})
        assert response.status_code == 422
        assert response.json()["code"] == "weak_pin"

    def test_keeping_the_issued_pin_is_refused(self, client: TestClient,
                                               issued_pin: str) -> None:
        token = _worker_login(client, ISSUED, issued_pin).json()["token"]
        response = client.post("/api/v1/auth/worker/pin", headers=_bearer(token),
                               json={"current_pin": issued_pin, "new_pin": issued_pin})
        assert response.json()["code"] == "weak_pin"

    def test_wrong_current_pin_counts_toward_the_lockout(
        self, client: TestClient, issued_pin: str
    ) -> None:
        token = _worker_login(client, ISSUED, issued_pin).json()["token"]
        for _ in range(4):
            wrong = client.post("/api/v1/auth/worker/pin", headers=_bearer(token),
                                json={"current_pin": "000001", "new_pin": NEW_PIN})
            assert wrong.json()["code"] == "invalid_credentials"
        fifth = client.post("/api/v1/auth/worker/pin", headers=_bearer(token),
                            json={"current_pin": "000001", "new_pin": NEW_PIN})
        assert fifth.status_code == 423

    def test_changing_the_pin_signs_out_other_sessions(
        self, client: TestClient, issued_pin: str
    ) -> None:
        here = _worker_login(client, ISSUED, issued_pin).json()["token"]
        elsewhere = _worker_login(client, ISSUED, issued_pin).json()["token"]
        client.post("/api/v1/auth/worker/pin", headers=_bearer(here),
                    json={"current_pin": issued_pin, "new_pin": NEW_PIN})
        assert client.get("/api/v1/auth/worker/me", headers=_bearer(here)).status_code == 200
        assert client.get("/api/v1/auth/worker/me", headers=_bearer(elsewhere)).status_code == 401


# ---------------------------------------------------------------------------
# Worker sessions
# ---------------------------------------------------------------------------


class TestWorkerSession:
    def test_me_describes_the_session(self, client: TestClient, worker_token: str) -> None:
        body = client.get("/api/v1/auth/worker/me", headers=_bearer(worker_token)).json()
        assert body["worker"]["worker_id"] == ISSUED
        assert body["must_change_pin"] is False

    def test_logout_ends_the_session(self, client: TestClient, worker_token: str) -> None:
        assert client.post("/api/v1/auth/worker/logout",
                           headers=_bearer(worker_token)).status_code == 204
        after = client.get("/api/v1/auth/worker/me", headers=_bearer(worker_token))
        assert after.status_code == 401
        assert after.json()["code"] == "not_authenticated"
        assert after.headers["WWW-Authenticate"] == "Bearer"

    def test_logout_with_a_dead_token_is_still_204(self, client: TestClient,
                                                   db_engine) -> None:
        response = client.post("/api/v1/auth/worker/logout", headers=_bearer("not-a-token"))
        assert response.status_code == 204

    def test_expired_session_is_refused(self, client: TestClient, worker_token: str,
                                        db: Session) -> None:
        db.execute(update(AuthSession).values(expires_at=service.utcnow()))
        db.commit()
        assert client.get("/api/v1/auth/worker/me",
                          headers=_bearer(worker_token)).status_code == 401

    def test_disabling_a_worker_ends_their_sessions(self, client: TestClient,
                                                    worker_token: str, db: Session) -> None:
        db.execute(update(Worker).values(is_active=False))
        db.commit()
        assert client.get("/api/v1/auth/worker/me",
                          headers=_bearer(worker_token)).status_code == 401

    @pytest.mark.parametrize("header", ["Token abc", "Bearer", "Bearer   ", "abc"])
    def test_malformed_authorization_header_is_401(self, client: TestClient, db_engine,
                                                   header: str) -> None:
        response = client.get("/api/v1/auth/worker/me", headers={"Authorization": header})
        assert response.status_code == 401


# ---------------------------------------------------------------------------
# Compliance checks and the rollout switch
# ---------------------------------------------------------------------------


class TestCheckAuth:
    def test_optional_without_token_works_as_before(
        self, client: TestClient, sample_image_bytes: bytes
    ) -> None:
        response = _check(client, sample_image_bytes, worker_id="W-FORM")
        assert response.status_code == 200
        assert response.json()["worker_id"] == "W-FORM"

    def test_optional_without_token_still_needs_a_worker_id(
        self, client: TestClient, sample_image_bytes: bytes
    ) -> None:
        assert _check(client, sample_image_bytes, worker_id=None).status_code == 422

    def test_a_bad_token_never_falls_back_to_the_form(
        self, client: TestClient, sample_image_bytes: bytes
    ) -> None:
        response = _check(client, sample_image_bytes, _bearer("expired-or-made-up"))
        assert response.status_code == 401
        assert response.json()["code"] == "not_authenticated"

    def test_required_without_token_is_401(
        self, client: TestClient, sample_image_bytes: bytes, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "ppb_worker_auth", "required")
        response = _check(client, sample_image_bytes)
        assert response.status_code == 401
        assert response.headers["WWW-Authenticate"] == "Bearer"

    def test_required_with_token_works(
        self, client: TestClient, worker_token: str, sample_image_bytes: bytes,
        monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "ppb_worker_auth", "required")
        response = _check(client, sample_image_bytes, _bearer(worker_token), worker_id=None)
        assert response.status_code == 200
        assert response.json()["worker_id"] == ISSUED

    def test_detect_when_required(
        self, client: TestClient, worker_token: str, csrf: str, sample_image_bytes: bytes,
        monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "ppb_worker_auth", "required")
        files = {"image": ("worker.png", sample_image_bytes, "image/png")}
        with_worker = client.post("/api/v1/detect", headers=_bearer(worker_token), files=files)
        with_admin = client.post("/api/v1/detect", headers={"X-CSRF-Token": csrf}, files=files)
        admin_no_csrf = client.post("/api/v1/detect", files=files)
        client.cookies.clear()
        nobody = client.post("/api/v1/detect", files=files)
        assert (with_worker.status_code, with_admin.status_code) == (200, 200)
        assert admin_no_csrf.json()["code"] == "csrf_failed"
        assert nobody.status_code == 401


# ---------------------------------------------------------------------------
# Dashboard login
# ---------------------------------------------------------------------------


class TestDashboardLogin:
    def test_login_sets_a_locked_down_cookie(self, client: TestClient, admin: User) -> None:
        response = _user_login(client, ADMIN_EMAIL, ADMIN_PASSWORD)
        assert response.status_code == 200
        cookie = response.headers["set-cookie"].lower()
        for flag in ("ppb_session=", "httponly", "samesite=strict", "path=/api", "secure"):
            assert flag in cookie
        body = response.json()
        assert body["user"]["role"] == "admin"
        assert len(body["csrf_token"]) == 43

    def test_email_ignores_case(self, client: TestClient, admin: User) -> None:
        assert _user_login(client, "ADMIN@Example.com", ADMIN_PASSWORD).status_code == 200

    def test_wrong_password_and_unknown_email_look_the_same(
        self, client: TestClient, admin: User
    ) -> None:
        wrong = _user_login(client, ADMIN_EMAIL, "not the password")
        unknown = _user_login(client, "nobody@example.com", ADMIN_PASSWORD)
        assert wrong.status_code == unknown.status_code == 401
        assert wrong.json() == unknown.json()

    def test_fifth_wrong_password_locks(self, client: TestClient, admin: User) -> None:
        for _ in range(4):
            _user_login(client, ADMIN_EMAIL, "not the password")
        assert _user_login(client, ADMIN_EMAIL, "not the password").status_code == 423

    def test_me_returns_the_user_and_csrf_token(self, client: TestClient, csrf: str) -> None:
        body = client.get("/api/v1/auth/me").json()
        assert body["user"]["email"] == ADMIN_EMAIL
        assert body["csrf_token"] == csrf

    def test_me_without_a_cookie_is_401(self, client: TestClient, db_engine) -> None:
        response = client.get("/api/v1/auth/me")
        assert response.status_code == 401
        assert "WWW-Authenticate" not in response.headers

    def test_writes_need_the_csrf_token(self, client: TestClient, csrf: str) -> None:
        body = {"current_password": ADMIN_PASSWORD, "new_password": "another long passphrase"}
        missing = client.post("/api/v1/auth/password", json=body)
        wrong = client.post("/api/v1/auth/password", json=body,
                            headers={"X-CSRF-Token": "guess"})
        right = client.post("/api/v1/auth/password", json=body, headers={"X-CSRF-Token": csrf})
        assert missing.json()["code"] == wrong.json()["code"] == "csrf_failed"
        assert right.status_code == 204

    def test_password_change_rules(self, client: TestClient, csrf: str) -> None:
        response = client.post("/api/v1/auth/password", headers={"X-CSRF-Token": csrf},
                               json={"current_password": ADMIN_PASSWORD, "new_password": "short"})
        assert response.status_code == 422
        assert response.json()["code"] == "weak_password"

    def test_password_change_signs_out_other_sessions(
        self, client: TestClient, admin: User, fake_detector
    ) -> None:
        with TestClient(main.app, base_url="https://testserver") as other_browser:
            assert _user_login(other_browser, ADMIN_EMAIL, ADMIN_PASSWORD).status_code == 200
            csrf = _user_login(client, ADMIN_EMAIL, ADMIN_PASSWORD).json()["csrf_token"]
            client.post("/api/v1/auth/password", headers={"X-CSRF-Token": csrf},
                        json={"current_password": ADMIN_PASSWORD,
                              "new_password": "another long passphrase"})
            assert client.get("/api/v1/auth/me").status_code == 200
            assert other_browser.get("/api/v1/auth/me").status_code == 401

    def test_idle_session_ends(self, client: TestClient, csrf: str, db: Session) -> None:
        db.execute(update(AuthSession).values(
            last_seen_at=service.utcnow() - timedelta(minutes=121)))
        db.commit()
        assert client.get("/api/v1/auth/me").status_code == 401

    def test_logout_clears_the_cookie_and_ends_the_session(
        self, client: TestClient, csrf: str, db: Session
    ) -> None:
        response = client.post("/api/v1/auth/logout", headers={"X-CSRF-Token": csrf})
        assert response.status_code == 204
        assert 'ppb_session=""' in response.headers["set-cookie"] or \
            "max-age=0" in response.headers["set-cookie"].lower()
        assert db.scalar(select(AuthSession.revoked_at)) is not None

    def test_must_change_password_blocks_other_routes(
        self, client: TestClient, db: Session, monkeypatch: pytest.MonkeyPatch,
        sample_image_bytes: bytes
    ) -> None:
        service.create_user(db, "sup@example.com", "Sup", Role.SUPERVISOR,
                            "temporary passphrase", must_change_password=True,
                            site_id=_main_site(db))
        csrf = _user_login(client, "sup@example.com", "temporary passphrase").json()["csrf_token"]
        assert client.get("/api/v1/auth/me").json()["user"]["must_change_password"] is True
        monkeypatch.setattr(settings, "ppb_worker_auth", "required")
        response = client.post("/api/v1/detect", headers={"X-CSRF-Token": csrf},
                               files={"image": ("w.png", sample_image_bytes, "image/png")})
        assert response.json()["code"] == "password_change_required"


class TestRoles:
    def test_role_is_enforced(self, db_engine, db: Session) -> None:
        """No Phase 2 route is admin-only yet, so check the gate on a test route."""
        app = FastAPI()
        app.add_exception_handler(AuthError, auth_error_handler)

        @app.get("/api/admin-only")
        def admin_only(_=Depends(require_user(Role.ADMIN))) -> dict:
            return {"ok": True}

        service.create_user(db, ADMIN_EMAIL, "A", Role.ADMIN, ADMIN_PASSWORD)
        service.create_user(db, "sup@example.com", "S", Role.SUPERVISOR, ADMIN_PASSWORD,
                            site_id=_main_site(db))
        main_client = TestClient(main.app, base_url="https://testserver")
        for email, expected in ((ADMIN_EMAIL, 200), ("sup@example.com", 403)):
            main_client.cookies.clear()
            token = _user_login(main_client, email, ADMIN_PASSWORD).cookies.get("ppb_session")
            gate = TestClient(app, base_url="https://testserver")
            response = gate.get("/api/admin-only", headers={"Cookie": f"ppb_session={token}"})
            assert response.status_code == expected
            if expected == 403:
                assert response.json()["code"] == "forbidden"


# ---------------------------------------------------------------------------
# Every route is protected unless it is meant to be public
# ---------------------------------------------------------------------------

PUBLIC = {
    ("GET", "/api/v1/health"),
    ("GET", "/api/v1/model/info"),
    ("GET", "/api/v1/jobs"),
    ("POST", "/api/v1/auth/worker/login"),
    ("POST", "/api/v1/auth/login"),
    ("POST", "/api/v1/auth/worker/logout"),  # always 204, so the app can forget a token
    ("POST", "/api/v1/auth/logout"),
}


def test_every_route_needs_a_login_unless_listed_as_public(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new route that forgets its login check fails here, in CI."""
    monkeypatch.setattr(settings, "ppb_worker_auth", "required")
    routes = {
        (method.upper(), path)
        for path, operations in main.app.openapi()["paths"].items()
        for method in operations
    }
    assert PUBLIC <= routes, "a public route was renamed or removed; update PUBLIC"
    unprotected = []
    for method, path in sorted(routes - PUBLIC):
        status = client.request(method, path).status_code
        if status != 401:
            unprotected.append(f"{method} {path} -> {status}")
    assert not unprotected, f"routes answering without a login: {unprotected}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:
    runner = CliRunner()

    def test_create_worker_prints_a_one_time_pin(self, db_engine, db: Session) -> None:
        result = self.runner.invoke(cli.app, ["auth", "create-worker", "--worker-id", "w-7",
                                              "--name", "Test Worker"])
        assert result.exit_code == 0, result.output
        assert "W-7" in result.output and "One-time PIN" in result.output
        worker = db.scalar(select(Worker).where(Worker.worker_code == "W-7"))
        assert worker.must_change_pin is True

    def test_duplicate_worker_is_a_clean_error(self, db_engine, issued_pin: str) -> None:
        result = self.runner.invoke(cli.app, ["auth", "create-worker", "--worker-id", ISSUED,
                                              "--name", "Again"])
        assert result.exit_code == 1
        assert "already exists" in result.output

    def test_reset_pin_unlocks_and_signs_out(
        self, client: TestClient, worker_token: str, db: Session
    ) -> None:
        db.execute(update(Worker).values(locked_until=service.utcnow() + timedelta(minutes=10)))
        db.commit()
        result = self.runner.invoke(cli.app, ["auth", "reset-pin", "--worker-id", ISSUED])
        assert result.exit_code == 0, result.output
        db.expire_all()
        worker = db.scalar(select(Worker))
        assert worker.locked_until is None and worker.must_change_pin is True
        assert client.get("/api/v1/auth/worker/me",
                          headers=_bearer(worker_token)).status_code == 401

    def test_create_admin_asks_for_the_password_twice(self, db_engine, db: Session) -> None:
        result = self.runner.invoke(
            cli.app, ["auth", "create-admin", "--email", "Boss@Example.com", "--name", "Boss"],
            input=f"{ADMIN_PASSWORD}\n{ADMIN_PASSWORD}\n",
        )
        assert result.exit_code == 0, result.output
        user = db.scalar(select(User))
        assert (user.email, user.role) == ("boss@example.com", "admin")
        assert ADMIN_PASSWORD not in user.password_hash

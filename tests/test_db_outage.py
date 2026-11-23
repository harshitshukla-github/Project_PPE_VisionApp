"""What the server says when its database can't be reached during a request.

The dashboard shows the server's sentence for a ``503 auth_unavailable``, and
during an outage every dashboard request gets one, since each checks its
session in the database. So the sentence has to read right on a page that has
nothing to do with signing in.
"""

from __future__ import annotations

import psycopg
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine

from src.api import main
from src.auth.errors import DATABASE_UNAVAILABLE
from src.settings import settings


def _refused() -> psycopg.Connection:
    raise psycopg.OperationalError("connection refused")


@pytest.fixture
def database_down(monkeypatch: pytest.MonkeyPatch) -> Engine:
    """A configured database that refuses every connection, as during an outage."""
    engine = create_engine("postgresql+psycopg://", creator=_refused)
    monkeypatch.setattr(settings, "database_url", "postgresql://ppb_app@db.invalid/ppb")
    monkeypatch.setattr("src.db.engine.get_engine", lambda url=None: engine)
    return engine


@pytest.fixture
def client() -> TestClient:
    """No lifespan, so no model: these routes don't need one."""
    return TestClient(main.app, base_url="https://testserver")


@pytest.mark.parametrize(("method", "path", "body"), [
    ("POST", "/api/v1/auth/login", {"email": "admin@example.com", "password": "correct horse battery"}),
    ("GET", "/api/v1/auth/me", None),
    ("GET", "/api/v1/admin/sites", None),
    ("GET", "/api/v1/admin/workers", None),
    ("POST", "/api/v1/auth/worker/login", {"worker_id": "W-0042", "pin": "482913"}),
])
def test_every_request_gets_the_same_plain_answer(database_down: Engine, client: TestClient,
                                                  method: str, path: str, body: dict | None) -> None:
    # A session cookie, so the admin routes go as far as looking it up.
    response = client.request(method, path, json=body, headers={"Cookie": "ppb_session=some-session"})

    assert response.status_code == 503
    assert response.json() == {"detail": DATABASE_UNAVAILABLE, "code": "auth_unavailable"}


def test_the_sentence_fits_a_page_that_isnt_sign_in() -> None:
    """A signed-in admin on the Workers page sees it, so it can't be about signing in."""
    assert "sign" not in DATABASE_UNAVAILABLE.lower()
    assert "database" in DATABASE_UNAVAILABLE
    assert DATABASE_UNAVAILABLE.endswith("Try again shortly.")


def test_no_database_at_all_still_says_so(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Not an outage: a server with no database configured keeps its own, permanent reason."""
    monkeypatch.setattr(settings, "database_url", "")
    response = client.get("/api/v1/auth/me")

    assert response.status_code == 503
    assert response.json()["detail"] == "Sign-in is unavailable: the server has no database configured."

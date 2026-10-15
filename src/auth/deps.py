"""FastAPI dependencies that decide who is signed in.

Workers send ``Authorization: Bearer <token>``. Dashboard users send the
``ppb_session`` cookie, plus ``X-CSRF-Token`` on every write. Routes declare
what they need with :func:`require_worker`, :func:`require_user` or
:func:`worker_for_check`; ``tests/test_auth_api.py`` calls every route
without credentials, so a route that forgets fails CI.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from src.auth.crypto import tokens_match
from src.auth.errors import AuthError, not_authenticated
from src.auth.service import resolve_user_session, resolve_worker_session
from src.db.models import AuthSession, Role
from src.settings import settings

SESSION_COOKIE = "ppb_session"
CSRF_HEADER = "X-CSRF-Token"
_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


@dataclass(frozen=True)
class SignedInWorker:
    """The worker behind a request, detached from the database session."""

    session_id: int
    worker_id: str
    display_name: str


@contextmanager
def open_db() -> Iterator[Session]:
    """A database session; 503 ``auth_unavailable`` when no database is configured."""
    if not settings.database_url:
        raise AuthError(503, "auth_unavailable",
                        "Sign-in is unavailable: the server has no database configured.")
    from src.db.engine import get_engine  # noqa: PLC0415 — only when a DB is configured

    db = Session(get_engine())
    try:
        yield db
    finally:
        db.close()


def db_session() -> Iterator[Session]:
    """FastAPI dependency form of :func:`open_db`, one session per request."""
    with open_db() as db:
        yield db


def client_ip(request: Request) -> str | None:
    return request.client.host if request.client else None


def bearer_token(request: Request) -> str | None:
    """The bearer token sent, ``None`` if no Authorization header.

    A malformed header is a 401, not "no token": the client meant to sign in.
    """
    header = request.headers.get("authorization")
    if header is None:
        return None
    scheme, _, value = header.partition(" ")
    if scheme.lower() != "bearer" or not value.strip():
        raise not_authenticated(bearer=True)
    return value.strip()


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------


def require_worker(allow_pin_change: bool = False) -> Callable[..., AuthSession]:
    """A dependency returning the worker's live session, or raising 401/403.

    Args:
        allow_pin_change: Let a worker who still has the issued PIN through.
            Only the endpoints that let them replace it, see themselves or
            log out set this.
    """

    def dependency(request: Request, db: Session = Depends(db_session)) -> AuthSession:
        token = bearer_token(request)
        record = resolve_worker_session(db, token) if token else None
        if record is None:
            raise not_authenticated(bearer=True)
        if record.worker.must_change_pin and not allow_pin_change:
            raise AuthError(403, "pin_change_required",
                            "Choose your own PIN before continuing.")
        return record

    return dependency


def worker_for_check(request: Request) -> SignedInWorker | None:
    """Who is running a compliance check (docs/auth-api.md §5).

    No token: ``None`` while ``PPB_WORKER_AUTH=optional``, 401 when
    ``required``. A token that doesn't work is always a 401, never a quiet
    fallback to the form field, so the app learns its session ended.
    """
    token = bearer_token(request)
    if token is None:
        if settings.ppb_worker_auth == "required":
            raise not_authenticated(bearer=True)
        return None
    with open_db() as db:
        record = resolve_worker_session(db, token)
        if record is None:
            raise not_authenticated(bearer=True)
        if record.worker.must_change_pin:
            raise AuthError(403, "pin_change_required",
                            "Choose your own PIN before continuing.")
        return SignedInWorker(record.id, record.worker.worker_code,
                              record.worker.display_name)


# ---------------------------------------------------------------------------
# Dashboard users
# ---------------------------------------------------------------------------


def require_user(
    *roles: Role, allow_password_change: bool = False
) -> Callable[..., AuthSession]:
    """A dependency returning the dashboard session, or raising 401/403.

    Args:
        roles: Roles allowed; none given means any signed-in user.
        allow_password_change: Let a user who must change their password through.

    Writes (POST, PUT, PATCH, DELETE) must carry the session's CSRF token in
    ``X-CSRF-Token``.
    """
    allowed = {Role(r).value for r in roles}

    def dependency(request: Request, db: Session = Depends(db_session)) -> AuthSession:
        token = request.cookies.get(SESSION_COOKIE)
        record = resolve_user_session(db, token) if token else None
        if record is None:
            raise not_authenticated(bearer=False)
        if request.method not in _SAFE_METHODS and not tokens_match(
            record.csrf_token, request.headers.get(CSRF_HEADER)
        ):
            raise AuthError(403, "csrf_failed",
                            "This request is missing its security token. Reload and try again.")
        if record.user.must_change_password and not allow_password_change:
            raise AuthError(403, "password_change_required",
                            "Change your password before continuing.")
        if allowed and record.user.role not in allowed:
            raise AuthError(403, "forbidden", "You don't have access to this.")
        return record

    return dependency


def worker_or_user_when_required(request: Request) -> None:
    """For endpoints any signed-in person may use once login is required
    (``POST /detect``). With ``PPB_WORKER_AUTH=optional`` anyone may."""
    if settings.ppb_worker_auth != "required":
        return
    if bearer_token(request) is not None:
        worker_for_check(request)
        return
    if request.cookies.get(SESSION_COOKIE):
        with open_db() as db:
            require_user()(request, db)
        return
    raise not_authenticated(bearer=True)

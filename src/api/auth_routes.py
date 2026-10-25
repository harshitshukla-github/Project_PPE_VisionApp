"""Login endpoints, as specified in docs/auth-api.md.

Workers (mobile app): ``/auth/worker/*``, bearer tokens.
Dashboard users (browser): ``/auth/login``, ``/auth/me``, ``/auth/logout``,
``/auth/password``, with the ``ppb_session`` cookie.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Request, Response
from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.auth import service
from src.auth.deps import (
    SESSION_COOKIE,
    bearer_token,
    client_ip,
    db_session,
    require_user,
    require_worker,
)
from src.db.models import AuthSession, User, Worker
from src.settings import settings

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

# FastAPI gives every dependency on db_session within one request the same
# session, so the session record returned by require_worker / require_user
# belongs to the ``db`` an endpoint receives.


# ---------------------------------------------------------------------------
# Request and response bodies
# ---------------------------------------------------------------------------


class WorkerLoginRequest(BaseModel):
    worker_id: str = Field(min_length=1, max_length=64, examples=["W-0042"])
    pin: str = Field(min_length=1, max_length=32, examples=["482913"])


class WorkerInfo(BaseModel):
    worker_id: str
    display_name: str


class WorkerLoginResponse(BaseModel):
    token: str
    token_type: str = "bearer"
    expires_at: datetime
    must_change_pin: bool
    worker: WorkerInfo


class WorkerMeResponse(BaseModel):
    worker: WorkerInfo
    must_change_pin: bool
    expires_at: datetime


class PinChangeRequest(BaseModel):
    current_pin: str = Field(min_length=1, max_length=32)
    new_pin: str = Field(min_length=1, max_length=32)


class UserLoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254, examples=["admin@example.com"])
    password: str = Field(min_length=1, max_length=256)


class UserInfo(BaseModel):
    id: int
    email: str
    display_name: str
    role: str
    # The supervisor's site key; null for an admin (docs/auth-api.md §6).
    site: str | None
    must_change_password: bool


class UserSessionResponse(BaseModel):
    user: UserInfo
    csrf_token: str
    expires_at: datetime


class PasswordChangeRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=256)
    new_password: str = Field(min_length=1, max_length=256)


def _worker_info(worker: Worker) -> WorkerInfo:
    return WorkerInfo(worker_id=worker.worker_code, display_name=worker.display_name)


def _user_session(user: User, record: AuthSession) -> UserSessionResponse:
    return UserSessionResponse(
        user=UserInfo(
            id=user.id,
            email=user.email,
            display_name=user.display_name,
            role=user.role,
            site=user.site.key if user.site else None,
            must_change_password=user.must_change_password,
        ),
        csrf_token=record.csrf_token or "",
        expires_at=service.user_session_expiry(record),
    )


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------


@router.post("/worker/login", response_model=WorkerLoginResponse)
def worker_login(
    body: WorkerLoginRequest, request: Request, db: Session = Depends(db_session)
) -> WorkerLoginResponse:
    """Sign a worker in with their worker ID and PIN; returns a bearer token."""
    issued, worker = service.login_worker(
        db, body.worker_id, body.pin, client_ip(request), request.headers.get("user-agent")
    )
    logger.info(f"Worker {worker.worker_code} signed in from {client_ip(request)}")
    return WorkerLoginResponse(
        token=issued.token,
        expires_at=issued.record.expires_at,
        must_change_pin=worker.must_change_pin,
        worker=_worker_info(worker),
    )


@router.post("/worker/pin", status_code=204)
def worker_change_pin(
    body: PinChangeRequest,
    request: Request,
    record: AuthSession = Depends(require_worker(allow_pin_change=True)),
    db: Session = Depends(db_session),
) -> Response:
    """Replace the worker's PIN, and end their other sessions."""
    service.change_worker_pin(db, record, body.current_pin, body.new_pin, client_ip(request))
    return Response(status_code=204)


@router.get("/worker/me", response_model=WorkerMeResponse)
def worker_me(
    record: AuthSession = Depends(require_worker(allow_pin_change=True)),
) -> WorkerMeResponse:
    """Who this worker token belongs to, and when it expires."""
    return WorkerMeResponse(
        worker=_worker_info(record.worker),
        must_change_pin=record.worker.must_change_pin,
        expires_at=record.expires_at,
    )


@router.post("/worker/logout", status_code=204)
def worker_logout(request: Request, db: Session = Depends(db_session)) -> Response:
    """End this worker session. Always 204, so the app can forget the token
    even if the session had already ended."""
    token = bearer_token(request)
    record = service.resolve_worker_session(db, token) if token else None
    if record is not None:
        service.revoke(db, record)
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Dashboard users
# ---------------------------------------------------------------------------


@router.post("/login", response_model=UserSessionResponse)
def user_login(
    body: UserLoginRequest,
    request: Request,
    response: Response,
    db: Session = Depends(db_session),
) -> UserSessionResponse:
    """Sign a dashboard user in; sets the ``ppb_session`` cookie."""
    issued, user = service.login_user(
        db, body.email, body.password, client_ip(request), request.headers.get("user-agent")
    )
    response.set_cookie(
        SESSION_COOKIE,
        issued.token,
        max_age=int(settings.ppb_user_session_hours * 3600),
        httponly=True,
        secure=settings.ppb_cookie_secure,
        samesite="strict",
        path="/api",
    )
    logger.info(f"User {user.email} ({user.role}) signed in from {client_ip(request)}")
    return _user_session(user, issued.record)


@router.get("/me", response_model=UserSessionResponse)
def user_me(
    record: AuthSession = Depends(require_user(allow_password_change=True)),
) -> UserSessionResponse:
    """The signed-in dashboard user, with the CSRF token to use for writes."""
    return _user_session(record.user, record)


@router.post("/logout", status_code=204)
def user_logout(request: Request, db: Session = Depends(db_session)) -> Response:
    """End this dashboard session and clear the cookie. Always 204."""
    token = request.cookies.get(SESSION_COOKIE)
    record = service.resolve_user_session(db, token) if token else None
    if record is not None:
        service.revoke(db, record)
    response = Response(status_code=204)
    response.delete_cookie(SESSION_COOKIE, path="/api", httponly=True,
                           secure=settings.ppb_cookie_secure, samesite="strict")
    return response


@router.post("/password", status_code=204)
def user_change_password(
    body: PasswordChangeRequest,
    request: Request,
    record: AuthSession = Depends(require_user(allow_password_change=True)),
    db: Session = Depends(db_session),
) -> Response:
    """Replace the user's password, and end their other sessions."""
    service.change_user_password(
        db, record, body.current_password, body.new_password, client_ip(request)
    )
    return Response(status_code=204)

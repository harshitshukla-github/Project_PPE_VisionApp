"""Login errors in the shape docs/auth-api.md §3 promises.

``detail`` stays a sentence for a person, because the mobile app already
shows it; ``code`` is what the app and dashboard branch on.
"""

from __future__ import annotations

import math

from fastapi import Request
from fastapi.responses import JSONResponse


class AuthError(Exception):
    """An error a login endpoint, or a check of who is signed in, returns."""

    def __init__(
        self,
        status: int,
        code: str,
        detail: str,
        retry_after_s: float | None = None,
        bearer: bool = False,
    ) -> None:
        super().__init__(detail)
        self.status = status
        self.code = code
        self.detail = detail
        self.retry_after_s = None if retry_after_s is None else max(1, math.ceil(retry_after_s))
        self.bearer = bearer

    def to_response(self) -> JSONResponse:
        body: dict = {"detail": self.detail, "code": self.code}
        headers: dict[str, str] = {}
        if self.retry_after_s is not None:
            body["retry_after_s"] = self.retry_after_s
            headers["Retry-After"] = str(self.retry_after_s)
        if self.status == 401 and self.bearer:
            headers["WWW-Authenticate"] = "Bearer"
        return JSONResponse(body, status_code=self.status, headers=headers)


async def auth_error_handler(request: Request, exc: AuthError) -> JSONResponse:
    """FastAPI exception handler for :class:`AuthError`."""
    return exc.to_response()


async def database_unavailable_handler(request: Request, exc: Exception) -> JSONResponse:
    """A database connection failure during a request. Only login reads the
    database per request; the job matrix handles its own outages."""
    return AuthError(
        503, "auth_unavailable", "Sign-in is temporarily unavailable. Try again shortly."
    ).to_response()


def human_duration(seconds: float) -> str:
    """``840`` → ``14 minutes``; ``40`` → ``40 seconds``."""
    seconds = max(1, math.ceil(seconds))
    if seconds < 90:
        return f"{seconds} seconds"
    minutes = math.ceil(seconds / 60)
    return f"{minutes} minutes"


def not_authenticated(bearer: bool) -> AuthError:
    return AuthError(401, "not_authenticated", "You're not signed in, or your session ended.",
                     bearer=bearer)

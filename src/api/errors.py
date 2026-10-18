"""API errors in the shape the contracts promise (docs/auth-api.md §3).

``detail`` stays a sentence for a person, because the mobile app and the
dashboard show it; ``code`` is what they branch on. Raise :class:`ApiError`
from a route or dependency and the registered handler writes the response.
"""

from __future__ import annotations

import math

from fastapi import Request
from fastapi.responses import JSONResponse


class ApiError(Exception):
    """An error answered as ``{"detail": …, "code": …}``, plus optional extras."""

    def __init__(
        self,
        status: int,
        code: str,
        detail: str,
        retry_after_s: float | None = None,
        field: str | None = None,
    ) -> None:
        super().__init__(detail)
        self.status = status
        self.code = code
        self.detail = detail
        self.retry_after_s = None if retry_after_s is None else max(1, math.ceil(retry_after_s))
        self.field = field

    def body(self) -> dict:
        body: dict = {"detail": self.detail, "code": self.code}
        if self.retry_after_s is not None:
            body["retry_after_s"] = self.retry_after_s
        if self.field is not None:
            body["field"] = self.field
        return body

    def headers(self) -> dict[str, str]:
        if self.retry_after_s is not None:
            return {"Retry-After": str(self.retry_after_s)}
        return {}

    def to_response(self) -> JSONResponse:
        return JSONResponse(self.body(), status_code=self.status, headers=self.headers())


async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
    """FastAPI exception handler for :class:`ApiError` and its subclasses."""
    return exc.to_response()

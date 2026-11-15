"""Serves the built admin dashboard at /admin (AGENTS.md §15, docs/auth-api.md §6).

Same origin as the API, so the session cookie is first-party and needs no CORS
setup. The files need no login: they hold no data, and everything they ask the
API for does. The cookie's ``Path=/api`` keeps it off these requests.

The build is looked for on each request, so ``npm run build`` needs no restart,
and a missing build doesn't stop the server: the API and the mobile app don't
need the dashboard.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.exception_handlers import http_exception_handler
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response
from starlette.staticfiles import StaticFiles
from starlette.types import Scope

MOUNT_PATH = "/admin"

# A relative PPB_ADMIN_DIST is from here, not from wherever the server was
# started: under a service manager or a Docker WORKDIR the two differ, and the
# dashboard would say it hasn't been built when it has.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

NOT_BUILT = (
    "The admin dashboard hasn't been built. Run this, then reload the page:\n\n"
    "    cd admin && npm ci && npm run build\n"
)

# This page holds an admin's session. Only our own files may run in it, and no
# other site may frame it to trick a click on Deactivate or Reset password.
CONTENT_SECURITY_POLICY = "; ".join([
    "default-src 'none'",
    "script-src 'self'",
    "style-src 'self'",
    "connect-src 'self'",
    "img-src 'self'",
    "base-uri 'none'",
    "form-action 'self'",
    "frame-ancestors 'none'",
])

# On every answer under /admin/, including "not found" and "not built".
SECURITY_HEADERS = {
    "Content-Security-Policy": CONTENT_SECURITY_POLICY,
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}

# Vite names everything in assets/ after its content, so a new build means new
# names and these never change. index.html keeps its name, so it is revalidated
# each time, or a browser would hold on to an old build.
CACHE_FOREVER = "public, max-age=31536000, immutable"
CACHE_REVALIDATE = "no-cache"


class DashboardFiles(StaticFiles):
    """The files ``npm run build`` wrote, with headers fit for an admin page."""

    def __init__(self, directory: Path) -> None:
        # Joining an absolute path keeps it, so PPB_ADMIN_DIST may be either.
        super().__init__(directory=(PROJECT_ROOT / directory).resolve(), html=True,
                         check_dir=False)

    @property
    def is_built(self) -> bool:
        """Whether there is a build to serve."""
        return (Path(self.directory) / "index.html").is_file()

    async def check_config(self) -> None:
        """Do nothing: a missing build is answered per request, not raised once."""

    async def get_response(self, path: str, scope: Scope) -> Response:
        """Serve one file, or say why not. Whatever the answer, it carries the security headers."""
        response = await self._answer(path, scope)
        response.headers.update(SECURITY_HEADERS)
        return response

    async def _answer(self, path: str, scope: Scope) -> Response:
        if not self.is_built:
            return PlainTextResponse(NOT_BUILT, status_code=503)
        try:
            response = await super().get_response(path, scope)
        except HTTPException as exc:
            # A missing file or a refused method is raised, not returned. Left to rise, the app
            # would answer it without this page's headers, so give the app's answer from here.
            return await http_exception_handler(Request(scope), exc)
        in_assets = Path(path).parts[:1] == ("assets",)
        response.headers["Cache-Control"] = CACHE_FOREVER if in_assets else CACHE_REVALIDATE
        return response

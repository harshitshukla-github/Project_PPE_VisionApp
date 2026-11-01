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

from starlette.responses import PlainTextResponse, Response
from starlette.staticfiles import StaticFiles
from starlette.types import Scope

MOUNT_PATH = "/admin"

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

# Vite names everything in assets/ after its content, so a new build means new
# names and these never change. index.html keeps its name, so it is revalidated
# each time, or a browser would hold on to an old build.
CACHE_FOREVER = "public, max-age=31536000, immutable"
CACHE_REVALIDATE = "no-cache"


class DashboardFiles(StaticFiles):
    """The files ``npm run build`` wrote, with headers fit for an admin page."""

    def __init__(self, directory: Path) -> None:
        super().__init__(directory=directory, html=True, check_dir=False)

    @property
    def is_built(self) -> bool:
        """Whether there is a build to serve."""
        return (Path(self.directory) / "index.html").is_file()

    async def check_config(self) -> None:
        """Do nothing: a missing build is answered per request, not raised once."""

    async def get_response(self, path: str, scope: Scope) -> Response:
        """Serve one file, or say how to build the dashboard if there is no build."""
        if not self.is_built:
            return PlainTextResponse(NOT_BUILT, status_code=503)
        response = await super().get_response(path, scope)
        response.headers["Content-Security-Policy"] = CONTENT_SECURITY_POLICY
        response.headers["X-Content-Type-Options"] = "nosniff"
        in_assets = Path(path).parts[:1] == ("assets",)
        response.headers["Cache-Control"] = CACHE_FOREVER if in_assets else CACHE_REVALIDATE
        return response

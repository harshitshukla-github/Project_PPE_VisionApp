"""FastAPI serving the built dashboard at /admin (src/api/dashboard.py)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from loguru import logger

from src.api import dashboard, main
from src.api.dashboard import MOUNT_PATH, DashboardFiles

INDEX = '<!doctype html><title>PPB Admin</title><script src="/admin/assets/index-abc.js"></script>'
SECRET = "outside the build"


def _point_dashboard_at(monkeypatch: pytest.MonkeyPatch, directory: Path) -> None:
    monkeypatch.setattr(main.dashboard_files, "directory", directory)
    monkeypatch.setattr(main.dashboard_files, "all_directories", [directory])


@pytest.fixture
def dist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A small build with a file next to it that must never be served."""
    directory = tmp_path / "project" / "admin" / "dist"
    (directory / "assets").mkdir(parents=True)
    (directory / "index.html").write_text(INDEX, encoding="utf-8")
    (directory / "assets" / "index-abc.js").write_text("console.log(1)", encoding="utf-8")
    (tmp_path / "project" / "secret.txt").write_text(SECRET, encoding="utf-8")
    (tmp_path / "project" / "admin" / "secret.txt").write_text(SECRET, encoding="utf-8")
    _point_dashboard_at(monkeypatch, directory)
    return directory


@pytest.fixture
def client() -> TestClient:
    """No lifespan, so no model: these routes don't need one."""
    return TestClient(main.app, follow_redirects=False)


@pytest.fixture
def logged() -> Iterator[list[str]]:
    """What the server logs while a test runs. Ask for it before api_client, which starts it."""
    lines: list[str] = []
    sink = logger.add(lines.append, format="{message}")
    yield lines
    logger.remove(sink)


class TestServing:
    def test_the_page(self, dist: Path, client: TestClient) -> None:
        response = client.get("/admin/")
        assert response.status_code == 200
        assert response.text == INDEX
        assert response.headers["content-type"].startswith("text/html")

    def test_without_the_slash_redirects_so_relative_urls_work(self, dist: Path,
                                                               client: TestClient) -> None:
        response = client.get("/admin")
        assert response.status_code in (307, 308)
        assert response.headers["location"].endswith("/admin/")

    def test_a_built_asset(self, dist: Path, client: TestClient) -> None:
        response = client.get("/admin/assets/index-abc.js")
        assert response.status_code == 200
        assert "javascript" in response.headers["content-type"]

    def test_a_missing_file_is_404(self, dist: Path, client: TestClient) -> None:
        assert client.get("/admin/assets/gone.js").status_code == 404
        assert client.get("/admin/assets/").status_code == 404

    def test_it_is_read_only(self, dist: Path, client: TestClient) -> None:
        assert client.post("/admin/").status_code == 405
        assert client.head("/admin/").status_code == 200

    def test_the_api_under_admin_is_not_swallowed(self, dist: Path, client: TestClient) -> None:
        """/api/v1/admin/* is the API; only /admin/* is the dashboard."""
        response = client.get("/api/v1/admin/sites")
        assert response.headers["content-type"].startswith("application/json")
        assert response.json()["code"]  # an API error, whatever this machine's database says

    def test_the_api_schema_stays_the_api(self) -> None:
        assert not [path for path in main.app.openapi()["paths"] if path.startswith(MOUNT_PATH)]


class TestStayingInsideTheBuild:
    @pytest.mark.parametrize("path", [
        "/admin/../secret.txt",
        "/admin/..%2fsecret.txt",
        "/admin/%2e%2e/secret.txt",
        "/admin/%2e%2e%2fsecret.txt",
        "/admin/..%5csecret.txt",
        "/admin/assets/../../secret.txt",
        "/admin/assets/..%2f..%2fsecret.txt",
        "/admin/assets/%2e%2e%5c%2e%2e%5csecret.txt",
        "/admin/..%2f..%2fsecret.txt",
        "/admin//etc/passwd",
        "/admin/%00",
    ])
    def test_no_path_reaches_a_file_outside(self, dist: Path, client: TestClient,
                                            path: str) -> None:
        response = client.get(path)
        assert SECRET not in response.text
        assert response.status_code in (404, 405)

    def test_a_symlink_out_of_the_build_is_not_followed(self, dist: Path,
                                                        client: TestClient) -> None:
        try:
            (dist / "link.txt").symlink_to(dist.parent / "secret.txt")
        except OSError:
            pytest.skip("creating symlinks needs extra rights on this machine")
        assert client.get("/admin/link.txt").status_code == 404


class TestHeaders:
    def test_the_page_is_locked_down(self, dist: Path, client: TestClient) -> None:
        headers = client.get("/admin/").headers
        policy = headers["content-security-policy"]
        for rule in ("default-src 'none'", "script-src 'self'", "connect-src 'self'",
                     "frame-ancestors 'none'", "base-uri 'none'"):
            assert rule in policy
        assert "unsafe" not in policy
        assert headers["x-content-type-options"] == "nosniff"
        assert headers["referrer-policy"] == "no-referrer"

    def test_built_assets_get_the_same_headers(self, dist: Path, client: TestClient) -> None:
        headers = client.get("/admin/assets/index-abc.js").headers
        assert headers["x-content-type-options"] == "nosniff"
        assert headers["referrer-policy"] == "no-referrer"

    def test_the_page_is_revalidated_so_a_new_build_is_seen(self, dist: Path,
                                                            client: TestClient) -> None:
        first = client.get("/admin/")
        assert first.headers["cache-control"] == "no-cache"
        again = client.get("/admin/", headers={"If-None-Match": first.headers["etag"]})
        assert again.status_code == 304
        assert again.headers["cache-control"] == "no-cache"

    def test_built_assets_are_cached_for_good(self, dist: Path, client: TestClient) -> None:
        cache = client.get("/admin/assets/index-abc.js").headers["cache-control"]
        assert cache == "public, max-age=31536000, immutable"

    def test_the_api_keeps_its_own_headers(self, dist: Path, client: TestClient) -> None:
        headers = client.get("/api/v1/admin/sites").headers
        assert "content-security-policy" not in headers
        assert "referrer-policy" not in headers


class TestNotBuilt:
    @pytest.fixture
    def no_build(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        directory = tmp_path / "not-built-yet"
        _point_dashboard_at(monkeypatch, directory)
        return directory

    def test_it_says_how_to_build(self, no_build: Path, client: TestClient) -> None:
        response = client.get("/admin/")
        assert response.status_code == 503
        assert "npm run build" in response.text

    def test_a_folder_without_a_page_is_not_a_build(self, no_build: Path,
                                                    client: TestClient) -> None:
        (no_build / "assets").mkdir(parents=True)
        assert client.get("/admin/").status_code == 503

    def test_the_server_still_starts_and_answers_the_api(self, no_build: Path,
                                                         api_client: TestClient) -> None:
        assert api_client.get("/api/v1/health").status_code == 200
        assert api_client.get("/admin/").status_code == 503

    def test_building_needs_no_restart(self, no_build: Path, client: TestClient) -> None:
        assert client.get("/admin/").status_code == 503
        no_build.mkdir()
        (no_build / "index.html").write_text(INDEX, encoding="utf-8")
        assert client.get("/admin/").status_code == 200

    def test_the_log_says_where_it_looked_but_the_page_does_not(
        self, no_build: Path, logged: list[str], api_client: TestClient
    ) -> None:
        """The full path is for whoever runs the server, not for anyone who opens /admin/."""
        assert [line for line in logged if "npm run build" in line and str(no_build) in line]
        assert str(no_build) not in api_client.get("/admin/").text


class TestWhereTheBuildIs:
    @pytest.fixture
    def project(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """A project with a build, and a server started from somewhere else entirely."""
        root = tmp_path / "project"
        (root / "admin" / "dist").mkdir(parents=True)
        (root / "admin" / "dist" / "index.html").write_text(INDEX, encoding="utf-8")
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.setattr(dashboard, "PROJECT_ROOT", root.resolve())
        monkeypatch.chdir(elsewhere)
        return root.resolve()

    def test_a_relative_path_is_from_the_project_not_the_working_directory(
        self, project: Path
    ) -> None:
        files = DashboardFiles(Path("admin/dist"))
        assert files.directory == project / "admin" / "dist"
        assert files.is_built

        app = FastAPI()
        app.mount(MOUNT_PATH, files)
        response = TestClient(app).get("/admin/")
        assert (response.status_code, response.text) == (200, INDEX)

    def test_an_absolute_path_is_used_as_given(self, project: Path, tmp_path: Path) -> None:
        other = (tmp_path / "builds" / "dashboard").resolve()
        assert DashboardFiles(other).directory == other
        assert not DashboardFiles(other).is_built

    def test_the_path_is_tidied_for_the_log(self, project: Path) -> None:
        files = DashboardFiles(Path("admin/../admin/dist"))
        assert files.directory == project / "admin" / "dist"

    def test_the_server_uses_an_absolute_path(self) -> None:
        assert Path(main.dashboard_files.directory).is_absolute()

    def test_the_project_root_really_is_the_project_root(self) -> None:
        assert (dashboard.PROJECT_ROOT / "admin" / "vite.config.ts").is_file()

    def test_the_log_gives_the_full_path(self, dist: Path, logged: list[str],
                                         api_client: TestClient) -> None:
        assert [line for line in logged if line.startswith("Dashboard: serving")
                and str(dist) in line]


def test_the_build_expects_to_live_where_it_is_served() -> None:
    """Vite writes absolute asset URLs from `base`; a mismatch is a blank page."""
    config = Path("admin/vite.config.ts").read_text(encoding="utf-8")
    assert f'base: "{MOUNT_PATH}/"' in config

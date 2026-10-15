"""Tests for src/db/bootstrap.py — provisioning the admin system's database.

No test here needs a running PostgreSQL server: provisioning is exercised
against a fake connection that records the SQL it is sent.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from psycopg import sql

from src.db.bootstrap import (
    APP_ROLE,
    build_url,
    generate_password,
    mask_url,
    parse_url,
    provision,
    read_env_var,
    upsert_env_var,
)

# ---------------------------------------------------------------------------
# Fake psycopg connection
# ---------------------------------------------------------------------------


class _FakeCursor:
    def __init__(self, server: _FakeServer) -> None:
        self.server = server
        self._row: tuple | None = None

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, query: object, params: tuple = ()) -> None:
        if isinstance(query, sql.Composable):
            text = self.server.render(query)
        else:
            text = str(query)
        self.server.executed.append(text)
        if "FROM pg_roles" in text:
            self._row = (1,) if params[0] in self.server.roles else None
        elif "FROM pg_database" in text:
            owner = self.server.databases.get(params[0])
            self._row = (owner,) if owner is not None else None
        else:
            self._row = None

    def fetchone(self) -> tuple | None:
        return self._row


class _FakeServer:
    """Just enough of a PostgreSQL connection for provision()."""

    def __init__(self, roles: set[str] | None = None, databases: dict | None = None) -> None:
        self.roles = roles or set()
        self.databases = databases or {}
        self.executed: list[str] = []

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self)

    @staticmethod
    def render(query: sql.Composable) -> str:
        """Render composed SQL the way the server would see it, without a server."""
        return query.as_string()


# ---------------------------------------------------------------------------
# Passwords and URLs
# ---------------------------------------------------------------------------


class TestPasswordsAndUrls:
    def test_password_is_long_and_url_safe(self) -> None:
        password = generate_password()
        assert len(password) >= 30
        assert set(password) <= set(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
        )

    def test_passwords_are_not_repeated(self) -> None:
        assert len({generate_password() for _ in range(20)}) == 20

    def test_url_round_trips_through_parse(self) -> None:
        url = build_url("ppb_app", "p@ss:w/rd#?", "localhost", 5432, "ppb")
        parts = parse_url(url)
        assert (parts.user, parts.password, parts.host, parts.port, parts.dbname) == (
            "ppb_app", "p@ss:w/rd#?", "localhost", 5432, "ppb",
        )

    def test_url_escapes_characters_that_would_break_it(self) -> None:
        url = build_url("ppb_app", "a@b:c/d", "localhost", 5432, "ppb")
        assert url == "postgresql://ppb_app:a%40b%3Ac%2Fd@localhost:5432/ppb"

    def test_parse_accepts_sqlalchemy_driver_suffix(self) -> None:
        parts = parse_url("postgresql+psycopg://u:p@db.local:6543/x")
        assert (parts.host, parts.port, parts.dbname) == ("db.local", 6543, "x")

    @pytest.mark.parametrize(
        "url", ["mysql://u:p@h/db", "postgresql://h:5432/db", "postgresql://u:p@h:5432"]
    )
    def test_parse_rejects_unusable_urls(self, url: str) -> None:
        with pytest.raises(ValueError):
            parse_url(url)

    def test_mask_hides_only_the_password(self) -> None:
        masked = mask_url("postgresql://ppb_app:s3cret-value@localhost:5432/ppb")
        assert masked == "postgresql://ppb_app:****@localhost:5432/ppb"
        assert "s3cret" not in masked


# ---------------------------------------------------------------------------
# .env editing
# ---------------------------------------------------------------------------


class TestEnvFile:
    """Compared as bytes: text-mode reads and writes translate newlines
    differently per platform, which once let a broken implementation pass
    on Windows and fail on Linux."""

    def test_appends_with_comment_and_keeps_other_lines(self, tmp_path: Path) -> None:
        env = tmp_path / ".env"
        env.write_bytes(b"A=1\nB=2  # note\n\n\n")
        assert upsert_env_var(env, "DATABASE_URL", "postgresql://x", comment="DB line") is True
        assert env.read_bytes() == (
            b"A=1\nB=2  # note\n\n# DB line\nDATABASE_URL=postgresql://x\n"
        )

    def test_replaces_in_place_without_duplicating(self, tmp_path: Path) -> None:
        env = tmp_path / ".env"
        env.write_bytes(b"A=1\nDATABASE_URL=old\nC=3\n")
        assert upsert_env_var(env, "DATABASE_URL", "new") is False
        assert env.read_bytes() == b"A=1\nDATABASE_URL=new\nC=3\n"

    def test_replace_keeps_windows_line_endings(self, tmp_path: Path) -> None:
        env = tmp_path / ".env"
        env.write_bytes(b"A=1\r\nDATABASE_URL=old\r\nC=3\r\n")
        upsert_env_var(env, "DATABASE_URL", "new")
        assert env.read_bytes() == b"A=1\r\nDATABASE_URL=new\r\nC=3\r\n"

    def test_does_not_touch_a_variable_whose_name_only_starts_the_same(
        self, tmp_path: Path
    ) -> None:
        env = tmp_path / ".env"
        env.write_text("TEST_DATABASE_URL=keep\n", encoding="utf-8")
        upsert_env_var(env, "DATABASE_URL", "new")
        text = env.read_text(encoding="utf-8")
        assert "TEST_DATABASE_URL=keep" in text
        assert "DATABASE_URL=new" in text.splitlines()

    def test_preserves_windows_line_endings(self, tmp_path: Path) -> None:
        env = tmp_path / ".env"
        env.write_bytes(b"A=1\r\nB=2\r\n")
        upsert_env_var(env, "DATABASE_URL", "x")
        assert env.read_bytes() == b"A=1\r\nB=2\r\n\r\nDATABASE_URL=x\r\n"

    def test_creates_a_missing_file(self, tmp_path: Path) -> None:
        env = tmp_path / ".env"
        upsert_env_var(env, "DATABASE_URL", "x")
        assert env.read_bytes() == b"DATABASE_URL=x\n"

    def test_read_strips_quotes_and_inline_comments(self, tmp_path: Path) -> None:
        env = tmp_path / ".env"
        env.write_text(
            'A="quoted"\nB=plain   # comment\nexport C=exported\nD=\n', encoding="utf-8"
        )
        assert read_env_var(env, "A") == "quoted"
        assert read_env_var(env, "B") == "plain"
        assert read_env_var(env, "C") == "exported"
        assert read_env_var(env, "D") is None
        assert read_env_var(env, "MISSING") is None
        assert read_env_var(tmp_path / "absent.env", "A") is None


# ---------------------------------------------------------------------------
# Provisioning
# ---------------------------------------------------------------------------


class TestProvision:
    def test_fresh_server_gets_role_and_both_databases(self) -> None:
        server = _FakeServer()
        actions = provision(server, APP_ROLE, "pw", ["ppb", "ppb_test"])
        assert actions == [
            "created role ppb_app",
            "created database ppb",
            "created database ppb_test",
        ]
        ddl = [s for s in server.executed if not s.startswith("SELECT")]
        assert ddl == [
            "CREATE ROLE \"ppb_app\" LOGIN PASSWORD 'pw' NOSUPERUSER NOCREATEDB NOCREATEROLE",
            "CREATE DATABASE \"ppb\" OWNER \"ppb_app\" ENCODING 'UTF8'",
            "CREATE DATABASE \"ppb_test\" OWNER \"ppb_app\" ENCODING 'UTF8'",
        ]

    def test_rerun_resets_password_and_creates_nothing(self) -> None:
        server = _FakeServer(roles={APP_ROLE}, databases={"ppb": APP_ROLE, "ppb_test": APP_ROLE})
        actions = provision(server, APP_ROLE, "new-pw", ["ppb", "ppb_test"])
        assert actions[0] == "role ppb_app exists — password reset to match .env"
        assert actions[1:] == ["database ppb exists", "database ppb_test exists"]
        ddl = [s for s in server.executed if not s.startswith("SELECT")]
        assert ddl == ["ALTER ROLE \"ppb_app\" WITH LOGIN PASSWORD 'new-pw'"]

    def test_database_owned_by_someone_else_is_handed_over(self) -> None:
        server = _FakeServer(roles={APP_ROLE}, databases={"ppb": "postgres"})
        actions = provision(server, APP_ROLE, "pw", ["ppb"])
        assert "database ppb exists — owner changed to ppb_app" in actions
        assert 'ALTER DATABASE "ppb" OWNER TO "ppb_app"' in server.executed

    def test_password_is_sent_as_an_escaped_literal(self) -> None:
        """A quote in the password must not be able to end the SQL string."""
        server = _FakeServer()
        provision(server, APP_ROLE, "x' SUPERUSER --", ["ppb"])
        create_role = next(s for s in server.executed if s.startswith("CREATE ROLE"))
        assert "PASSWORD 'x'' SUPERUSER --'" in create_role

"""Provision the local PostgreSQL database the admin system runs on.

Creates one login role and two databases, owned by that role:

- ``ppb``       — the application database (job matrix, users, sessions, ...)
- ``ppb_test``  — a throwaway database the test suite may drop and recreate

Provisioning connects once as a PostgreSQL superuser, whose password is typed
by the person running it and never stored. The application role's password is
generated here and written only to ``.env``. Every step is idempotent, so the
same command sets up a new machine and repairs an existing one.

Typical usage::

    uv run python cli.py db setup     # prompts for the postgres password
    uv run python cli.py db status    # verifies .env can connect
"""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlsplit

APP_ROLE = "ppb_app"
APP_DB = "ppb"
TEST_DB = "ppb_test"

DATABASE_URL_KEY = "DATABASE_URL"
TEST_DATABASE_URL_KEY = "TEST_DATABASE_URL"

_ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=")


def generate_password(nbytes: int = 24) -> str:
    """Return a random password that needs no escaping in a URL or .env file.

    Args:
        nbytes: Bytes of randomness; the result is about 1.3x as many characters.

    Returns:
        A URL-safe random string (``A-Z a-z 0-9 - _``).
    """
    return secrets.token_urlsafe(nbytes)


def build_url(user: str, password: str, host: str, port: int, dbname: str) -> str:
    """Build a ``postgresql://`` connection URL with credentials percent-encoded.

    The plain scheme works with psql, psycopg and most tools. SQLAlchemy code
    adds its driver suffix when it reads the URL.

    Args:
        user: Role name.
        password: Role password.
        host: Server host.
        port: Server port.
        dbname: Database name.

    Returns:
        Connection URL.
    """
    return (
        f"postgresql://{quote(user, safe='')}:{quote(password, safe='')}"
        f"@{host}:{port}/{quote(dbname, safe='')}"
    )


@dataclass(frozen=True)
class UrlParts:
    """The parts of a connection URL that provisioning needs."""

    user: str
    password: str
    host: str
    port: int
    dbname: str


def parse_url(url: str) -> UrlParts:
    """Split a ``postgresql://`` URL into its parts, decoding credentials.

    Args:
        url: Connection URL, with or without a ``+driver`` suffix on the scheme.

    Returns:
        The decoded parts.

    Raises:
        ValueError: If the URL is not a PostgreSQL URL with user and database.
    """
    parts = urlsplit(url)
    if not parts.scheme.startswith("postgresql"):
        raise ValueError(f"not a PostgreSQL URL (scheme '{parts.scheme}')")
    dbname = parts.path.lstrip("/")
    if not parts.username or not dbname:
        raise ValueError("URL must name a user and a database")
    return UrlParts(
        user=unquote(parts.username),
        password=unquote(parts.password or ""),
        host=parts.hostname or "localhost",
        port=parts.port or 5432,
        dbname=unquote(dbname),
    )


def mask_url(url: str) -> str:
    """Return the URL with its password replaced, safe to print or log."""
    return re.sub(r"(://[^:/@]+:)[^@]*@", r"\1****@", url)


def read_env_var(path: Path, key: str) -> str | None:
    """Read one variable from a .env file without loading the rest.

    Args:
        path: The .env file.
        key: Variable name.

    Returns:
        The value with surrounding quotes removed, or ``None`` if the file or
        variable does not exist, or the value is empty.
    """
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _ENV_LINE.match(line)
        if match and match.group(1) == key:
            value = line.split("=", 1)[1].split(" #", 1)[0].strip().strip("\"'")
            return value or None
    return None


def upsert_env_var(path: Path, key: str, value: str, comment: str | None = None) -> bool:
    """Set ``key=value`` in a .env file, touching no other line.

    An existing assignment is replaced in place. Otherwise the variable is
    appended, preceded by ``comment`` as a ``#`` line when given. The file's
    line endings are preserved, and the file is created if it is missing.

    Args:
        path: The .env file.
        key: Variable name.
        value: New value, written unquoted.
        comment: Optional comment for a newly appended variable.

    Returns:
        ``True`` if the variable was appended, ``False`` if replaced.
    """
    # newline="" turns off Python's newline translation in both directions.
    # Without it, reading turns every \r\n into \n, so CRLF is never seen,
    # and writing on Windows turns every \n into \r\n.
    text = ""
    if path.exists():
        with path.open(encoding="utf-8", newline="") as handle:
            text = handle.read()
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines()

    for index, line in enumerate(lines):
        match = _ENV_LINE.match(line)
        if match and match.group(1) == key:
            lines[index] = f"{key}={value}"
            path.write_text(newline.join(lines) + newline, encoding="utf-8", newline="")
            return False

    while lines and not lines[-1].strip():
        lines.pop()
    if lines:
        lines.append("")
    if comment:
        lines.extend(f"# {c}" if c else "#" for c in comment.splitlines())
    lines.append(f"{key}={value}")
    path.write_text(newline.join(lines) + newline, encoding="utf-8", newline="")
    return True


def provision(conn: Any, role: str, password: str, databases: list[str]) -> list[str]:
    """Create or repair the application role and its databases.

    Must run on a superuser connection in autocommit mode, because
    ``CREATE DATABASE`` cannot run inside a transaction.

    - The role is created as a plain login role (no superuser, no CREATEDB,
      no CREATEROLE). If it already exists its password is reset, so the
      role always matches the password about to be written to ``.env``.
    - Each database is created owned by the role, or handed to it if it
      exists with another owner. Owning the database is what lets the role
      create tables in its ``public`` schema on PostgreSQL 15 and later.

    Args:
        conn: An open psycopg connection with autocommit enabled.
        role: Role name to create.
        password: Password for the role.
        databases: Database names to create, owned by ``role``.

    Returns:
        Human-readable descriptions of what changed, in order.
    """
    from psycopg import sql  # noqa: PLC0415 — keep the CLI import cheap

    actions: list[str] = []
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,))
        if cur.fetchone() is None:
            cur.execute(
                sql.SQL(
                    "CREATE ROLE {} LOGIN PASSWORD {} "
                    "NOSUPERUSER NOCREATEDB NOCREATEROLE"
                ).format(sql.Identifier(role), sql.Literal(password))
            )
            actions.append(f"created role {role}")
        else:
            cur.execute(
                sql.SQL("ALTER ROLE {} WITH LOGIN PASSWORD {}").format(
                    sql.Identifier(role), sql.Literal(password)
                )
            )
            actions.append(f"role {role} exists — password reset to match .env")

        for dbname in databases:
            cur.execute(
                "SELECT pg_get_userbyid(datdba) FROM pg_database WHERE datname = %s",
                (dbname,),
            )
            row = cur.fetchone()
            if row is None:
                cur.execute(
                    sql.SQL("CREATE DATABASE {} OWNER {} ENCODING 'UTF8'").format(
                        sql.Identifier(dbname), sql.Identifier(role)
                    )
                )
                actions.append(f"created database {dbname}")
            elif row[0] != role:
                cur.execute(
                    sql.SQL("ALTER DATABASE {} OWNER TO {}").format(
                        sql.Identifier(dbname), sql.Identifier(role)
                    )
                )
                actions.append(f"database {dbname} exists — owner changed to {role}")
            else:
                actions.append(f"database {dbname} exists")
    return actions


@dataclass(frozen=True)
class ConnectionReport:
    """What a successful connection check found."""

    server_version: str
    current_user: str
    database: str
    can_create_tables: bool


def check_connection(url: str, timeout_s: int = 5) -> ConnectionReport:
    """Connect with a URL and confirm the role can create tables there.

    The probe table is created and dropped inside a transaction that is
    rolled back, so the check leaves nothing behind.

    Args:
        url: Connection URL, as stored in ``.env``.
        timeout_s: Connection timeout in seconds.

    Returns:
        What the server reported.

    Raises:
        psycopg.Error: If the connection or the probe fails.
    """
    import psycopg  # noqa: PLC0415

    parts = parse_url(url)
    with psycopg.connect(
        host=parts.host,
        port=parts.port,
        user=parts.user,
        password=parts.password,
        dbname=parts.dbname,
        connect_timeout=timeout_s,
    ) as conn, conn.cursor() as cur:
        cur.execute("SELECT current_setting('server_version'), current_user, current_database()")
        version, user, database = cur.fetchone()
        try:
            cur.execute("CREATE TABLE ppb_setup_probe (id int)")
            can_create = True
        except psycopg.Error:
            can_create = False
        conn.rollback()
    return ConnectionReport(
        server_version=version,
        current_user=user,
        database=database,
        can_create_tables=can_create,
    )

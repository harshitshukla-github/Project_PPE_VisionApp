"""Run Alembic migrations from Python (used by the CLI and the tests).

The repo's ``alembic.ini`` serves the ``alembic`` command line, for writing
new migrations. This module builds the same configuration in code, so
``cli.py db migrate`` and the tests need no ini file and can target any URL.
"""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory

from src.db.engine import get_engine, sqlalchemy_url
from src.settings import settings

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def alembic_config(url: str | None = None) -> Config:
    """Build an Alembic config for ``url``, or ``DATABASE_URL`` if omitted.

    The URL travels in ``config.attributes`` rather than the ini options,
    because ini values are %-interpolated and a percent-encoded password
    would be corrupted.

    Args:
        url: Connection URL.

    Returns:
        A ready-to-use Alembic configuration.
    """
    cfg = _scripts_config()
    cfg.attributes["sqlalchemy_url"] = sqlalchemy_url(url or settings.database_url)
    return cfg


def _scripts_config() -> Config:
    """A config that locates the migration scripts, with no database."""
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    return cfg


def upgrade_to_head(url: str | None = None) -> None:
    """Apply every migration not yet applied to the database at ``url``."""
    command.upgrade(alembic_config(url), "head")


def head_revision() -> str | None:
    """Return the newest migration revision in the repo."""
    return ScriptDirectory.from_config(_scripts_config()).get_current_head()


def current_revision(url: str | None = None) -> str | None:
    """Return the revision the database at ``url`` is migrated to, or ``None``."""
    with get_engine(url).connect() as conn:
        return MigrationContext.configure(conn).get_current_revision()

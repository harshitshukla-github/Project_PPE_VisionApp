"""Alembic environment: connects to the database and runs migrations.

The URL comes from ``config.attributes["sqlalchemy_url"]`` when the
migration is started from Python (``src/db/migrate.py``), and otherwise
from ``DATABASE_URL`` in ``.env``.
"""

from alembic import context
from sqlalchemy import create_engine, pool

from src.db.engine import sqlalchemy_url
from src.db.models import Base
from src.settings import settings

config = context.config
target_metadata = Base.metadata


def _url() -> str:
    return config.attributes.get("sqlalchemy_url") or sqlalchemy_url(settings.database_url)


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of running it (``alembic upgrade --sql``)."""
    context.configure(
        url=_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live database."""
    engine = create_engine(_url(), poolclass=pool.NullPool)
    try:
        with engine.connect() as connection:
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                compare_type=True,
            )
            with context.begin_transaction():
                context.run_migrations()
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

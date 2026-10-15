"""SQLAlchemy engine for the admin system's PostgreSQL database.

``.env`` stores plain ``postgresql://`` URLs (what ``cli.py db setup``
writes, and what psql understands). SQLAlchemy needs the driver named, so
:func:`sqlalchemy_url` adds ``+psycopg`` before an engine is built.
"""

from __future__ import annotations

from functools import lru_cache

from sqlalchemy import Engine, create_engine

from src.settings import settings


def sqlalchemy_url(url: str) -> str:
    """Return ``url`` with the psycopg 3 driver named in its scheme.

    Args:
        url: A ``postgresql://``, ``postgres://`` or ``postgresql+driver://`` URL.

    Returns:
        The URL with a ``postgresql+psycopg://`` scheme; a URL that already
        names a driver is returned unchanged.

    Raises:
        ValueError: If ``url`` is empty or not a PostgreSQL URL.
    """
    if not url:
        raise ValueError("no database URL configured (DATABASE_URL is empty)")
    scheme, sep, rest = url.partition("://")
    if not sep:
        raise ValueError("database URL has no scheme")
    if scheme in ("postgresql", "postgres"):
        return f"postgresql+psycopg://{rest}"
    if scheme.startswith("postgresql+"):
        return url
    raise ValueError(f"not a PostgreSQL URL (scheme '{scheme}')")


@lru_cache(maxsize=4)
def get_engine(url: str | None = None) -> Engine:
    """Return a pooled engine for ``url``, or for ``DATABASE_URL`` if omitted.

    Engines are cached per URL, so every caller shares one connection pool.

    Args:
        url: Connection URL; defaults to ``settings.database_url``.

    Returns:
        A SQLAlchemy engine. No connection is opened until first use.
    """
    return create_engine(
        sqlalchemy_url(url or settings.database_url),
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
        connect_args={"connect_timeout": 5},
    )

"""Database tables for the admin system.

The Job PPE Matrix (Phase 1): which equipment each job requires.

- ``ppe_items``        the equipment the system knows about (hardhat, ...)
- ``jobs``             job categories the app offers (welder, ...)
- ``job_requirements`` one row per (job, item): ``required`` or ``pending``

Login (Phase 2, contract in ``docs/auth-api.md``):

- ``users``     dashboard accounts: admins and supervisors
- ``workers``   people who run checks from the mobile app
- ``sessions``  one row per login, for a user or a worker; tokens stored hashed

Schema changes go through Alembic (``src/db/migrations``); after editing
these classes, add a migration. ``tests/test_db_matrix.py`` fails if the
models and the migrations disagree.
"""

from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    MetaData,
    String,
    func,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Stable constraint names, so migrations can refer to them.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Declarative base with the project's constraint naming convention."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class RequirementMode(str, enum.Enum):
    """How a job treats one piece of equipment."""

    REQUIRED = "required"  # checked: detector first, then Gemini
    PENDING = "pending"  # listed for the worker to verify manually, never scored


class PpeItem(Base):
    """A piece of equipment the system can require."""

    __tablename__ = "ppe_items"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    display_name: Mapped[str] = mapped_column(String(100))
    # Whether the detector was trained on this item. When it was not, a
    # required item can only be judged by Gemini.
    detectable: Mapped[bool] = mapped_column(Boolean)

    __table_args__ = (CheckConstraint("key ~ '^[a-z][a-z0-9_]*$'", name="key_format"),)


class Job(Base):
    """A job category and the order it appears in the app."""

    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    key: Mapped[str] = mapped_column(String(64), unique=True)
    display_name: Mapped[str] = mapped_column(String(100))
    position: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    is_active: Mapped[bool] = mapped_column(Boolean, server_default=text("true"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    requirements: Mapped[list[JobRequirement]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
        order_by="JobRequirement.position",
    )

    __table_args__ = (CheckConstraint("key ~ '^[a-z][a-z0-9_]*$'", name="key_format"),)


class JobRequirement(Base):
    """One piece of equipment in one job's list."""

    __tablename__ = "job_requirements"

    job_id: Mapped[int] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), primary_key=True
    )
    item_key: Mapped[str] = mapped_column(
        ForeignKey("ppe_items.key", ondelete="RESTRICT"), primary_key=True
    )
    mode: Mapped[str] = mapped_column(String(16))
    position: Mapped[int] = mapped_column(Integer, server_default=text("0"))

    job: Mapped[Job] = relationship(back_populates="requirements")

    __table_args__ = (
        CheckConstraint("mode IN ('required', 'pending')", name="mode_valid"),
    )


class Role(str, enum.Enum):
    """What a dashboard user may do (see docs/auth-api.md §2)."""

    ADMIN = "admin"
    SUPERVISOR = "supervisor"


class User(Base):
    """A dashboard account."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(254), unique=True)
    display_name: Mapped[str] = mapped_column(String(100))
    role: Mapped[str] = mapped_column(String(16))
    password_hash: Mapped[str] = mapped_column(String(255))
    must_change_password: Mapped[bool] = mapped_column(Boolean, server_default=text("false"))
    is_active: Mapped[bool] = mapped_column(Boolean, server_default=text("true"))
    failed_attempts: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    locked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        CheckConstraint("role IN ('admin', 'supervisor')", name="role_valid"),
        CheckConstraint("email = lower(email)", name="email_lowercase"),
    )


class Worker(Base):
    """Someone who runs compliance checks from the mobile app."""

    __tablename__ = "workers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # What the API calls ``worker_id``, e.g. "W-0042". Stored upper case.
    worker_code: Mapped[str] = mapped_column(String(32), unique=True)
    display_name: Mapped[str] = mapped_column(String(100))
    pin_hash: Mapped[str] = mapped_column(String(255))
    # True while the worker still has the PIN a supervisor issued.
    must_change_pin: Mapped[bool] = mapped_column(Boolean, server_default=text("true"))
    is_active: Mapped[bool] = mapped_column(Boolean, server_default=text("true"))
    failed_attempts: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    locked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        CheckConstraint("worker_code ~ '^[A-Z0-9][A-Z0-9-]{1,31}$'", name="worker_code_format"),
    )


class AuthSession(Base):
    """One login, by a dashboard user or by a worker.

    Only a SHA-256 hash of the token is stored, so a copy of this table
    holds no session anyone could use.
    """

    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True)
    # Dashboard sessions only: echoed in X-CSRF-Token on every write.
    csrf_token: Mapped[str | None] = mapped_column(String(64))
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    worker_id: Mapped[int | None] = mapped_column(
        ForeignKey("workers.id", ondelete="CASCADE"), index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    user_agent: Mapped[str | None] = mapped_column(String(255))
    ip: Mapped[str | None] = mapped_column(String(64))

    user: Mapped[User | None] = relationship()
    worker: Mapped[Worker | None] = relationship()

    __table_args__ = (
        CheckConstraint("(user_id IS NULL) <> (worker_id IS NULL)", name="one_owner"),
    )

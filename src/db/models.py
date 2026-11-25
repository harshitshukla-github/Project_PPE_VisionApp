"""Database tables for the admin system.

The Job PPE Matrix (Phase 1): which equipment each job requires.

- ``ppe_items``        the equipment the system knows about (hardhat, ...)
- ``jobs``             job categories the app offers (welder, ...)
- ``job_requirements`` one row per (job, item): ``required`` or ``pending``

Login (Phase 2, contract in ``docs/auth-api.md``):

- ``users``     dashboard accounts: admins and supervisors
- ``workers``   people who run checks from the mobile app
- ``sessions``  one row per login, for a user or a worker; tokens stored hashed

Admin system (Phase 3, contract in ``docs/admin-api.md``):

- ``sites``         places where workers are checked; every worker and every
                    supervisor belongs to one
- ``app_settings``  values saved from the dashboard that override ``.env``
- ``audit_log``     every change made through the admin API or the CLI

Checks and review (Phase 6, contract in ``docs/checks-api.md``):

- ``checks``  every compliance check that reached a verdict, with what it was
              judged against, its photo's state, and the review beside it

Schema changes go through Alembic (``src/db/migrations``); after editing
these classes, add a migration. ``tests/test_db_matrix.py`` fails if the
models and the migrations disagree.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
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


class Site(Base):
    """A place where workers are checked: a construction site, a plant."""

    __tablename__ = "sites"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    key: Mapped[str] = mapped_column(String(64), unique=True)
    display_name: Mapped[str] = mapped_column(String(100))
    is_active: Mapped[bool] = mapped_column(Boolean, server_default=text("true"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (CheckConstraint("key ~ '^[a-z][a-z0-9_]*$'", name="key_format"),)


class Job(Base):
    """A job category and the order it appears in the app."""

    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    key: Mapped[str] = mapped_column(String(64), unique=True)
    display_name: Mapped[str] = mapped_column(String(100))
    position: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    is_active: Mapped[bool] = mapped_column(Boolean, server_default=text("true"))
    # Goes up with every change to the name, the active flag or the
    # requirements (not the position), so a stale edit can be refused.
    version: Mapped[int] = mapped_column(Integer, server_default=text("1"))
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
    # A supervisor's site; admins work across all sites and have none.
    site_id: Mapped[int | None] = mapped_column(
        ForeignKey("sites.id", ondelete="RESTRICT"), index=True
    )
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

    site: Mapped[Site | None] = relationship()

    __table_args__ = (
        CheckConstraint("role IN ('admin', 'supervisor')", name="role_valid"),
        CheckConstraint("email = lower(email)", name="email_lowercase"),
        CheckConstraint("(role = 'supervisor') = (site_id IS NOT NULL)", name="site_matches_role"),
    )


class Worker(Base):
    """Someone who runs compliance checks from the mobile app."""

    __tablename__ = "workers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # What the API calls ``worker_id``, e.g. "W-0042". Stored upper case.
    worker_code: Mapped[str] = mapped_column(String(32), unique=True)
    display_name: Mapped[str] = mapped_column(String(100))
    site_id: Mapped[int] = mapped_column(ForeignKey("sites.id", ondelete="RESTRICT"), index=True)
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
    # The photo notice (docs/checks-api.md §5): the version the worker last
    # answered (null until they do), and whether they let their photos be
    # kept longer for training. Photos are kept only once the notice is seen.
    consent_seen_version: Mapped[int | None] = mapped_column(Integer)
    training_consent: Mapped[bool] = mapped_column(Boolean, server_default=text("false"))
    consent_updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    site: Mapped[Site] = relationship()
    creator: Mapped[User | None] = relationship(foreign_keys=[created_by])

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


class AppSetting(Base):
    """A runtime setting saved from the dashboard, overriding its ``.env`` default."""

    __tablename__ = "app_settings"

    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[Any] = mapped_column(JSONB)
    updated_by: Mapped[str | None] = mapped_column(String(254))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class AuditEntry(Base):
    """One change made through the admin API or the CLI (docs/admin-api.md §9).

    ``before`` and ``after`` hold only the documented keys for the action,
    and never a password, PIN, hash or token.
    """

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    actor: Mapped[str] = mapped_column(String(254))
    action: Mapped[str] = mapped_column(String(64), index=True)
    target: Mapped[str] = mapped_column(String(160), index=True)
    site: Mapped[str | None] = mapped_column(String(64), index=True)
    before: Mapped[Any | None] = mapped_column(JSONB)
    after: Mapped[Any | None] = mapped_column(JSONB)
    ip: Mapped[str | None] = mapped_column(String(64))


class Check(Base):
    """One compliance check that reached a verdict (docs/checks-api.md §2).

    What the app was told, plus what the verdict was judged against at the
    time: the job's requirements and the three settings that decide a
    verdict. The matrix and the settings can change later; a review has to be
    judged by the rules the verdict used. Rows are never deleted; photos
    expire, and their state says so.
    """

    __tablename__ = "checks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Who and where. worker_code is the ID as the check gave it: from the
    # login, or typed into an app without one (signed_in false), in which case
    # it may match no worker, and worker_id and site_id are null.
    worker_id: Mapped[int | None] = mapped_column(ForeignKey("workers.id", ondelete="SET NULL"))
    worker_code: Mapped[str] = mapped_column(String(64))
    signed_in: Mapped[bool] = mapped_column(Boolean)
    # The worker's site when the check was made; it stays if they move.
    site_id: Mapped[int | None] = mapped_column(ForeignKey("sites.id", ondelete="RESTRICT"))

    # The job, as it was.
    job_key: Mapped[str] = mapped_column(String(64))
    job_name: Mapped[str] = mapped_column(String(100))
    job_version: Mapped[int | None] = mapped_column(Integer)
    required: Mapped[list[str]] = mapped_column(JSONB)
    pending: Mapped[list[str]] = mapped_column(JSONB)

    # The verdict, as the app got it.
    verdict: Mapped[str] = mapped_column(String(16))
    verdict_source: Mapped[str] = mapped_column(String(32))
    confidence_avg: Mapped[float] = mapped_column(Float)
    missing_equipment: Mapped[list[str]] = mapped_column(JSONB)
    violations_detected: Mapped[list[str]] = mapped_column(JSONB)
    unverifiable_equipment: Mapped[list[str]] = mapped_column(JSONB)
    detections: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)
    verification: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    explanation: Mapped[str | None] = mapped_column(Text)
    model_used: Mapped[str] = mapped_column(String(64))
    inference_time_ms: Mapped[float] = mapped_column(Float)
    gemini_invoked: Mapped[bool] = mapped_column(Boolean)

    # The three settings that decide a verdict, as they were.
    confidence_threshold: Mapped[float] = mapped_column(Float)
    gemini_verify: Mapped[bool] = mapped_column(Boolean)
    gemini_min_confidence: Mapped[float] = mapped_column(Float)

    # The photo. photo_key names it in the photo store while it is kept, and
    # only then. Its size is the upload's, after orientation: the space the
    # detections' boxes are in.
    photo_state: Mapped[str] = mapped_column(String(16))
    photo_key: Mapped[str | None] = mapped_column(String(255))
    photo_type: Mapped[str | None] = mapped_column(String(32))
    photo_width: Mapped[int | None] = mapped_column(Integer)
    photo_height: Mapped[int | None] = mapped_column(Integer)
    photo_bytes: Mapped[int | None] = mapped_column(Integer)
    photo_deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # The worker's training choice at the time of the check (§5).
    training_consent: Mapped[bool] = mapped_column(Boolean, server_default=text("false"))

    # The review, beside the verdict, which never changes (§4.3).
    review_outcome: Mapped[str | None] = mapped_column(String(16))
    review_note: Mapped[str | None] = mapped_column(String(500))
    reviewed_by: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    worker: Mapped[Worker | None] = relationship()
    site: Mapped[Site | None] = relationship()
    reviewer: Mapped[User | None] = relationship(foreign_keys=[reviewed_by])

    __table_args__ = (
        CheckConstraint("verdict IN ('PASS', 'FAIL', 'NEEDS_REVIEW')", name="verdict_valid"),
        CheckConstraint("photo_state IN ('kept', 'deleted', 'not_stored')",
                        name="photo_state_valid"),
        CheckConstraint("(photo_state = 'kept') = (photo_key IS NOT NULL)",
                        name="photo_key_when_kept"),
        CheckConstraint("review_outcome IS NULL OR review_outcome IN ('compliant', 'not_compliant')",
                        name="review_outcome_valid"),
        CheckConstraint("(review_outcome IS NULL) = (reviewed_at IS NULL)", name="review_complete"),
        # The dashboard's lists (§4.1) and the daily photo expiry (§6).
        Index("ix_checks_at", "at"),
        Index("ix_checks_site_id_at", "site_id", "at"),
        Index("ix_checks_worker_id_at", "worker_id", "at"),
        Index("ix_checks_photo_state_at", "photo_state", "at"),
    )

"""Database tables for the admin system.

Phase 1 holds the Job PPE Matrix: which equipment each job requires.

- ``ppe_items``        the equipment the system knows about (hardhat, ...)
- ``jobs``             job categories the app offers (welder, ...)
- ``job_requirements`` one row per (job, item): ``required`` or ``pending``

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

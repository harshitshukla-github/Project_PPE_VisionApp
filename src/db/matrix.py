"""The Job PPE Matrix in PostgreSQL: read it, and seed it from the YAML file.

Once seeded, the database is the source of truth and the admin dashboard
edits it there. ``configs/job_matrix.yaml`` remains only as the starting
point for a new database and as the matrix used when ``DATABASE_URL`` is
not set.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session, selectinload

from src.db.models import Job, JobRequirement, PpeItem, RequirementMode

# Every item the system knows. ``detectable`` is whether the detector was
# trained on it; the others can only be judged by Gemini.
ITEMS: tuple[tuple[str, str, bool], ...] = (
    ("hardhat", "Hardhat", True),
    ("safety_vest", "Safety vest", True),
    ("goggles", "Goggles", True),
    ("mask", "Mask", True),
    ("gloves", "Gloves", True),
    ("boots", "Boots", True),
    ("ear_protection", "Ear protection", False),
    ("face_shield", "Face shield", False),
    ("harness", "Harness", False),
)


@dataclass(frozen=True)
class SeedReport:
    """What :func:`seed_from_yaml` did."""

    items_added: int
    jobs_added: int
    skipped_reason: str | None = None


def display_name_for(key: str) -> str:
    """Turn a job key into a label: ``general_labor`` → ``General labor``."""
    return key.replace("_", " ").capitalize()


def read_matrix(engine: Engine) -> dict[str, dict[str, list[str]]]:
    """Read the active jobs and their equipment lists, in display order.

    Args:
        engine: Engine for the application database.

    Returns:
        ``{job_key: {"required": [...], "pending": [...]}}``, jobs ordered by
        position and each list in its stored order. Inactive jobs are left out.
    """
    with Session(engine) as session:
        jobs = session.scalars(
            select(Job)
            .where(Job.is_active.is_(True))
            .order_by(Job.position, Job.id)
            .options(selectinload(Job.requirements))
        ).all()
        return {
            job.key: {
                mode.value: [r.item_key for r in job.requirements if r.mode == mode.value]
                for mode in RequirementMode
            }
            for job in jobs
        }


def seed_from_yaml(engine: Engine, yaml_path: Path) -> SeedReport:
    """Fill an empty database with the items and the matrix from YAML.

    Items missing from ``ppe_items`` are always added. Jobs are added only
    when the ``jobs`` table is empty: once anyone may have edited the matrix
    in the database, re-seeding would overwrite their changes, so it is
    refused rather than merged.

    Args:
        engine: Engine for the application database.
        yaml_path: The matrix YAML, normally ``configs/job_matrix.yaml``.

    Returns:
        What was added, or why jobs were skipped.

    Raises:
        ValueError: If the YAML names an item that is not in :data:`ITEMS`,
            or lists an item as both required and pending.
    """
    from src.inference.compliance import load_job_matrix  # noqa: PLC0415 — avoids a cycle

    matrix = load_job_matrix(path=yaml_path)
    known = {key for key, _, _ in ITEMS}
    for job, req in matrix.items():
        unknown = sorted(set(req.required + req.pending) - known)
        if unknown:
            raise ValueError(f"job '{job}' names unknown items: {unknown}")
        both = sorted(set(req.required) & set(req.pending))
        if both:
            raise ValueError(f"job '{job}' lists items as both required and pending: {both}")

    with Session(engine) as session, session.begin():
        existing = set(session.scalars(select(PpeItem.key)))
        new_items = [
            PpeItem(key=key, display_name=name, detectable=detectable)
            for key, name, detectable in ITEMS
            if key not in existing
        ]
        session.add_all(new_items)

        job_count = session.scalar(select(func.count()).select_from(Job)) or 0
        if job_count:
            return SeedReport(
                items_added=len(new_items),
                jobs_added=0,
                skipped_reason=(
                    f"{job_count} jobs already in the database; it is the source of "
                    "truth now, so the YAML was not applied"
                ),
            )

        for position, (key, req) in enumerate(matrix.items()):
            ordered = [(item, RequirementMode.REQUIRED) for item in req.required] + [
                (item, RequirementMode.PENDING) for item in req.pending
            ]
            session.add(
                Job(
                    key=key,
                    display_name=display_name_for(key),
                    position=position,
                    requirements=[
                        JobRequirement(item_key=item, mode=mode.value, position=index)
                        for index, (item, mode) in enumerate(ordered)
                    ],
                )
            )
        return SeedReport(items_added=len(new_items), jobs_added=len(matrix))


def count_jobs(engine: Engine) -> int:
    """Return how many jobs (active or not) the database holds."""
    with Session(engine) as session:
        return session.scalar(select(func.count()).select_from(Job)) or 0

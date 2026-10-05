"""PPE compliance verification against the Job PPE Matrix.

Takes raw detections from :class:`~src.inference.detector.PPEDetector` and
a job category, returns a verdict with the reasoning fields the API needs.

Verdict logic — three states, in priority order:

1. **FAIL** — an explicit violation class (``no_hardhat``, ``no_vest``) was
   detected for an item this job requires. This holds even if the positive
   class was also detected: the violation box means at least one visible
   person lacks the item.
2. **NEEDS_REVIEW** — no violation was detected, but a required class was
   not found either. The model cannot confirm compliance *or* non-compliance,
   so a human decides.
3. **PASS** — every required class detected, no violations.

Classes in the job's ``pending`` list are never scored (the model has no
training data for them); they surface as ``unverifiable_equipment``.

Why NEEDS_REVIEW exists:
    Absence of a detection is not evidence of absence. ``safety_vest`` scores
    AP50 0.861, so roughly one vest in seven is missed — and before this
    state existed every one of those misses produced a confident ``FAIL``,
    telling a compliant worker they were in violation and logging it against
    them. ``FAIL`` now requires the model to have positively seen something
    wrong. The cost is that genuine violations the model under-detects land
    in ``NEEDS_REVIEW`` rather than ``FAIL`` — deliberate, because a missed
    violation gets caught by the human reviewing the queue, whereas a false
    accusation erodes trust in the whole system.

Usage::

    from src.inference.compliance import check_compliance

    result = check_compliance(detections, job_category="welder")
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from loguru import logger
from pydantic import BaseModel, Field

from src.inference.schemas import Detection, Verdict
from src.settings import settings

# Maps a violation class to the positive class it contradicts.
VIOLATION_TO_ITEM: dict[str, str] = {
    "no_hardhat": "hardhat",
    "no_vest": "safety_vest",
}


class JobRequirements(BaseModel):
    """PPE requirements for one job category."""

    required: list[str] = Field(description="Classes that must be detected to PASS.")
    pending: list[str] = Field(
        default_factory=list,
        description="Required by the job but not yet detectable by the model.",
    )


class ComplianceResult(BaseModel):
    """Outcome of a compliance check, before API envelope fields are added."""

    verdict: Verdict
    missing_equipment: list[str]
    violations_detected: list[str]
    unverifiable_equipment: list[str]
    confidence_avg: float


@lru_cache(maxsize=1)
def load_job_matrix(path: Path | None = None) -> dict[str, JobRequirements]:
    """Load and validate the Job PPE Matrix from YAML.

    Args:
        path: Matrix YAML path. Defaults to ``configs/job_matrix.yaml``.

    Returns:
        Mapping of job category → :class:`JobRequirements`.

    Raises:
        FileNotFoundError: If the matrix file doesn't exist.
    """
    path = path or settings.configs_dir / "job_matrix.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Job PPE Matrix not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    matrix = {job: JobRequirements(**req) for job, req in raw.get("jobs", {}).items()}
    logger.debug(f"Job PPE Matrix loaded: {sorted(matrix)}")
    return matrix


def check_compliance(
    detections: list[Detection],
    job_category: str,
) -> ComplianceResult:
    """Verify detections against a job's PPE requirements.

    Args:
        detections: Output of :meth:`PPEDetector.detect`.
        job_category: Key into the Job PPE Matrix (e.g. ``"welder"``).

    Returns:
        Populated :class:`ComplianceResult`.

    Raises:
        KeyError: If ``job_category`` is not in the matrix.
    """
    matrix = load_job_matrix()
    if job_category not in matrix:
        raise KeyError(
            f"Unknown job category '{job_category}'. Known: {sorted(matrix)}"
        )
    requirements = matrix[job_category]

    detected_classes = {d.class_name for d in detections}

    missing = [item for item in requirements.required if item not in detected_classes]
    violations = [
        v for v in VIOLATION_TO_ITEM
        if v in detected_classes and VIOLATION_TO_ITEM[v] in requirements.required
    ]

    # A violation class is affirmative evidence: the model positively located a
    # bare head or an uncovered torso. Absence of a required item is not — the
    # detector simply may not have found it. Those two cases used to collapse
    # into FAIL, which is how a worker wearing a vest got told they were
    # non-compliant. They are now distinct outcomes.
    if violations:
        verdict = Verdict.FAIL
    elif missing:
        verdict = Verdict.NEEDS_REVIEW
    else:
        verdict = Verdict.PASS

    confidence_avg = (
        round(sum(d.confidence for d in detections) / len(detections), 4)
        if detections else 0.0
    )

    result = ComplianceResult(
        verdict=verdict,
        missing_equipment=missing,
        violations_detected=violations,
        unverifiable_equipment=list(requirements.pending),
        confidence_avg=confidence_avg,
    )
    logger.info(
        f"Compliance [{job_category}]: {verdict.value} "
        f"(missing={missing or 'none'}, violations={violations or 'none'})"
    )
    return result

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
    wrong.

Second opinion:
    ``NEEDS_REVIEW`` from the detector is not the end. The API shows the photo
    to Gemini (:mod:`src.inference.gemini_verifier`), and
    :func:`apply_verification` turns its per-item findings into the final
    verdict: ``FAIL`` if any item is seen to be not worn, ``PASS`` if every
    undetected item is seen to be worn, and ``NEEDS_REVIEW`` only for what
    neither model could judge. Detector ``PASS`` and ``FAIL`` are never
    reopened.

Usage::

    from src.inference.compliance import apply_verification, check_compliance

    result = check_compliance(detections, job_category="welder")
    result = apply_verification(result, verification)   # verification may be None
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from loguru import logger
from pydantic import BaseModel, Field

from src.inference.schemas import Detection, GeminiVerification, ItemStatus, Verdict
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
    verdict_source: str = "detector"


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


def apply_verification(
    result: ComplianceResult,
    verification: GeminiVerification | None,
    min_confidence: float | None = None,
) -> ComplianceResult:
    """Resolve a ``NEEDS_REVIEW`` result using Gemini's look at the photo.

    Only ``NEEDS_REVIEW`` is ever changed. ``PASS`` and ``FAIL`` came from the
    detector positively seeing something, and are returned untouched.

    Each item the detector missed is sorted by Gemini's finding:

    * ``not_worn`` → affirmative evidence of a breach. One is enough for
      ``FAIL``. Items Gemini could not judge move to
      ``unverifiable_equipment`` so the worker is only told about what was
      actually seen to be missing.
    * ``worn`` → confirmed present; dropped from ``missing_equipment``.
    * ``uncertain``, absent from the reply, or below ``min_confidence`` →
      still unknown. If any remain and nothing was ``not_worn``, the verdict
      stays ``NEEDS_REVIEW`` with only those items listed.

    All missing items confirmed ``worn`` → ``PASS``.

    Args:
        result: Detector-only compliance result.
        verification: Gemini's findings, or None if the check did not run.
        min_confidence: Findings below this are treated as uncertain.
            Defaults to ``settings.ppb_gemini_verify_min_confidence``.

    Returns:
        The resolved result (a copy), or ``result`` itself if nothing applies.
    """
    if verification is None or result.verdict is not Verdict.NEEDS_REVIEW:
        return result

    threshold = (
        settings.ppb_gemini_verify_min_confidence
        if min_confidence is None else min_confidence
    )
    findings = {f.item: f for f in verification.items}

    not_worn: list[str] = []
    unknown: list[str] = []
    for item in result.missing_equipment:
        finding = findings.get(item)
        if (
            finding is None
            or finding.status is ItemStatus.UNCERTAIN
            or finding.confidence < threshold
        ):
            unknown.append(item)
        elif finding.status is ItemStatus.NOT_WORN:
            not_worn.append(item)
        # ItemStatus.WORN at or above the threshold: confirmed, nothing to list.

    if not_worn:
        verdict = Verdict.FAIL
        missing = not_worn
        unverifiable = [*result.unverifiable_equipment, *unknown]
    elif unknown:
        verdict = Verdict.NEEDS_REVIEW
        missing = unknown
        unverifiable = list(result.unverifiable_equipment)
    else:
        verdict = Verdict.PASS
        missing = []
        unverifiable = list(result.unverifiable_equipment)

    changed = verdict is not result.verdict or missing != result.missing_equipment
    resolved = result.model_copy(update={
        "verdict": verdict,
        "missing_equipment": missing,
        "unverifiable_equipment": unverifiable,
        "verdict_source": "detector+gemini" if changed else result.verdict_source,
    })
    logger.info(
        f"Gemini verification: {result.verdict.value} → {verdict.value} "
        f"(not_worn={not_worn or 'none'}, still_unknown={unknown or 'none'})"
    )
    return resolved

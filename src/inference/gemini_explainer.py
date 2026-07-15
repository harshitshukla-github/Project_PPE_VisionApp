"""Gemini 2.5 Flash explanations for FAILED compliance checks.

Cost design (agents.md §6): Gemini is invoked **only on FAIL verdicts**.
PASS responses are pure YOLO and cost nothing. Explanations are cached per
(job_category, missing set, violation set) — identical failures reuse the
same text instead of paying for a new generation (risk register §13).

The explainer degrades gracefully: if the SDK is missing, the API key is
unset, or the call fails, it returns a deterministic fallback explanation
so the compliance endpoint never breaks because of Gemini.

Usage::

    from src.inference.gemini_explainer import explain_failure

    text = explain_failure(
        job_category="welder",
        missing_equipment=["gloves"],
        violations_detected=["no_hardhat"],
    )
"""

from __future__ import annotations

from functools import lru_cache

from loguru import logger

from src.settings import settings

GEMINI_MODEL = "gemini-2.5-flash"

_PROMPT_TEMPLATE = """You are a construction site safety officer. A worker failed \
their PPE compliance check.

Job category: {job}
Missing equipment: {missing}
Violations detected: {violations}

In 2-4 sentences, explain to the worker in plain, direct language:
1. What is missing or wrong.
2. The concrete injury risk each missing item exposes them to on this job.
Do not use markdown. Do not lecture; be factual and specific."""


def _fallback_explanation(
    job_category: str,
    missing_equipment: list[str],
    violations_detected: list[str],
) -> str:
    """Deterministic explanation used when Gemini is unavailable."""
    problems: list[str] = []
    if missing_equipment:
        problems.append(f"missing required equipment: {', '.join(missing_equipment)}")
    if violations_detected:
        readable = [v.replace("no_", "not wearing ").replace("_", " ") for v in violations_detected]
        problems.append(f"violations detected: {', '.join(readable)}")
    return (
        f"Compliance check failed for job '{job_category}' — "
        f"{'; '.join(problems)}. "
        "Please put on all required PPE and retry the check."
    )


@lru_cache(maxsize=256)
def _cached_explain(job_category: str, missing_key: str, violations_key: str) -> str:
    """Cache layer — one Gemini call per unique failure signature.

    Args:
        job_category: Job key from the PPE matrix.
        missing_key: Comma-joined, sorted missing classes ('' if none).
        violations_key: Comma-joined, sorted violation classes ('' if none).

    Returns:
        Explanation text (Gemini-generated, or fallback on any error).
    """
    missing = missing_key.split(",") if missing_key else []
    violations = violations_key.split(",") if violations_key else []

    if not settings.google_api_key:
        logger.warning("GOOGLE_API_KEY not set — using fallback explanation.")
        return _fallback_explanation(job_category, missing, violations)

    try:
        from google import genai  # noqa: PLC0415 — optional dependency

        client = genai.Client(api_key=settings.google_api_key)
        prompt = _PROMPT_TEMPLATE.format(
            job=job_category,
            missing=", ".join(missing) or "none",
            violations=", ".join(violations) or "none",
        )
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        text = (response.text or "").strip()
        if not text:
            raise ValueError("Gemini returned empty text")
        logger.info(f"Gemini explanation generated for ({job_category}, {missing_key})")
        return text
    except Exception as exc:  # graceful degradation — never break the API
        logger.warning(f"Gemini call failed ({exc}) — using fallback explanation.")
        return _fallback_explanation(job_category, missing, violations)


def explain_failure(
    job_category: str,
    missing_equipment: list[str],
    violations_detected: list[str],
) -> str:
    """Generate (or reuse) a human-readable explanation for a FAIL verdict.

    Args:
        job_category: Job key from the PPE matrix.
        missing_equipment: Required classes that were not detected.
        violations_detected: Explicit violation classes detected.

    Returns:
        Explanation text suitable for showing directly to the worker.
    """
    return _cached_explain(
        job_category,
        ",".join(sorted(missing_equipment)),
        ",".join(sorted(violations_detected)),
    )

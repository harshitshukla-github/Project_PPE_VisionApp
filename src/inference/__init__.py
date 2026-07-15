"""Production inference for the PPB Vision Module.

Pipeline: image → :class:`~src.inference.detector.PPEDetector` →
:func:`~src.inference.compliance.check_compliance` → (on FAIL only)
:func:`~src.inference.gemini_explainer.explain_failure`.
"""

from src.inference.compliance import check_compliance, load_job_matrix
from src.inference.detector import PPEDetector
from src.inference.schemas import (
    ComplianceResponse,
    Detection,
    DetectionResponse,
    Verdict,
)

__all__ = [
    "PPEDetector",
    "check_compliance",
    "load_job_matrix",
    "Detection",
    "DetectionResponse",
    "ComplianceResponse",
    "Verdict",
]

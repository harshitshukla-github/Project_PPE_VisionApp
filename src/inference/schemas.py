"""Pydantic request/response models for the inference pipeline and API.

These schemas define the public contract of the PPB vision service
(agents.md §9). Keep changes backward-compatible — the mobile app and
admin panel consume these shapes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class Verdict(str, Enum):
    """Overall compliance outcome.

    ``FAIL`` requires *affirmative* evidence of a violation — a ``no_hardhat``
    or ``no_vest`` detection. Merely failing to detect a required item is not
    proof the worker lacks it; the detector misses roughly one vest in seven
    even after the co-occurrence retrain (safety_vest AP50 0.861). Reporting
    those as ``FAIL`` accuses compliant workers, which is the defect this
    third state exists to prevent.
    """

    PASS = "PASS"
    FAIL = "FAIL"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class ItemStatus(str, Enum):
    """What the second-opinion verifier concluded about one PPE item."""

    WORN = "worn"
    NOT_WORN = "not_worn"
    UNCERTAIN = "uncertain"


class ItemVerification(BaseModel):
    """Gemini's finding for one required item the detector did not find."""

    item: str = Field(description="PPE class name, e.g. 'safety_vest'.")
    status: ItemStatus
    confidence: float = Field(ge=0.0, le=1.0, description="Verifier's own confidence.")
    reason: str = Field(default="", description="One short sentence of visual evidence.")


class GeminiVerification(BaseModel):
    """Second-opinion check of the photo, run when the detector is unsure."""

    model: str = Field(description="Gemini model that looked at the photo.")
    items: list[ItemVerification]
    elapsed_ms: float = Field(description="Wall-clock time of the Gemini call.")


class Detection(BaseModel):
    """A single detected PPE item (or violation) in the image."""

    class_name: str = Field(
        alias="class", description="PPE class name from configs/data.yaml."
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Detection confidence.")
    bbox: list[float] = Field(
        min_length=4, max_length=4,
        description="Bounding box [x1, y1, x2, y2] in pixel coordinates.",
    )

    model_config = {"populate_by_name": True}


class DetectionResponse(BaseModel):
    """Response for the raw ``/detect`` endpoint."""

    detections: list[Detection]
    model_used: str = Field(description="Model variant that produced the detections.")
    inference_time_ms: float = Field(description="End-to-end inference time.")
    image_width: int
    image_height: int


class ComplianceResponse(BaseModel):
    """Response for the ``/compliance/check`` endpoint (agents.md §9)."""

    worker_id: str
    job_category: str
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC time the check was performed.",
    )
    verdict: Verdict
    confidence_avg: float = Field(
        ge=0.0, le=1.0,
        description="Mean confidence across all detections (0 if none).",
    )
    detections: list[Detection]
    missing_equipment: list[str] = Field(
        description=(
            "On FAIL: required items confirmed not worn. On NEEDS_REVIEW: "
            "required items neither the detector nor Gemini could confirm."
        )
    )
    violations_detected: list[str] = Field(
        default_factory=list,
        description="Explicit violation classes detected (e.g. no_hardhat).",
    )
    unverifiable_equipment: list[str] = Field(
        default_factory=list,
        description=(
            "Required items that could not be checked from this photo: classes "
            "the detector has no training data for, plus — on a FAIL — items "
            "Gemini could not see clearly. Verify manually; never a failure."
        ),
    )
    explanation: str | None = Field(
        default=None,
        description="Gemini-generated explanation. Present only on FAIL.",
    )
    verdict_source: str = Field(
        default="detector",
        description=(
            "'detector' when YOLO alone decided; 'detector+gemini' when the "
            "detector was unsure and Gemini's look at the photo settled it."
        ),
    )
    verification: GeminiVerification | None = Field(
        default=None,
        description=(
            "Gemini's per-item findings. Present whenever the second-opinion "
            "check ran, including when it could not settle the verdict."
        ),
    )
    model_used: str
    inference_time_ms: float
    gemini_invoked: bool = False


class HealthResponse(BaseModel):
    """Response for ``/health``."""

    status: str = "ok"
    model_loaded: bool
    model_name: str
    device: str


class ModelInfoResponse(BaseModel):
    """Response for ``/model/info``."""

    model_name: str
    weights_path: str
    class_names: dict[int, str]
    confidence_threshold: float
    iou_threshold: float
    device: str

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
    """Overall compliance outcome."""

    PASS = "PASS"
    FAIL = "FAIL"


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
        description="Required PPE classes not detected on the worker."
    )
    violations_detected: list[str] = Field(
        default_factory=list,
        description="Explicit violation classes detected (e.g. no_hardhat).",
    )
    unverifiable_equipment: list[str] = Field(
        default_factory=list,
        description=(
            "PPE required by the job but not yet detectable by the model "
            "(no training data). Verify manually."
        ),
    )
    explanation: str | None = Field(
        default=None,
        description="Gemini-generated explanation. Present only on FAIL.",
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

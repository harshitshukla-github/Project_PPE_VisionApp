"""API routes: /detect, /compliance/check, /health, /model/info.

All endpoints live under the ``/api/v1`` prefix (agents.md §9).
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from loguru import logger

from src.api.deps import get_detector
from src.inference.compliance import check_compliance, load_job_matrix
from src.inference.detector import PPEDetector
from src.inference.gemini_explainer import explain_failure
from src.inference.schemas import (
    ComplianceResponse,
    DetectionResponse,
    HealthResponse,
    ModelInfoResponse,
    Verdict,
)
from src.settings import settings

router = APIRouter(prefix="/api/v1")


async def _decode_upload(image: UploadFile) -> np.ndarray:
    """Decode an uploaded file into a BGR numpy array.

    Raises:
        HTTPException: 400 if the payload is not a decodable image.
    """
    import cv2  # noqa: PLC0415

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty image upload.")
    array = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if array is None:
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode image '{image.filename}'. Send JPEG or PNG.",
        )
    return array


@router.get("/health", response_model=HealthResponse)
def health(detector: PPEDetector = Depends(get_detector)) -> HealthResponse:
    """Service liveness + which model is loaded."""
    return HealthResponse(
        model_loaded=True,
        model_name=detector.model_name,
        device=str(detector.device),
    )


@router.get("/model/info", response_model=ModelInfoResponse)
def model_info(detector: PPEDetector = Depends(get_detector)) -> ModelInfoResponse:
    """Loaded model metadata: variant, classes, thresholds."""
    return ModelInfoResponse(
        model_name=detector.model_name,
        weights_path=str(detector.weights_path),
        class_names=detector.class_names,
        confidence_threshold=detector.confidence,
        iou_threshold=detector.iou,
        device=str(detector.device),
    )


@router.post("/detect", response_model=DetectionResponse)
async def detect(
    image: UploadFile = File(..., description="Worker photo (JPEG/PNG)."),
    detector: PPEDetector = Depends(get_detector),
) -> DetectionResponse:
    """Raw PPE detection — bounding boxes only, no compliance logic."""
    array = await _decode_upload(image)
    detections, elapsed_ms = detector.detect(array)
    height, width = array.shape[:2]
    return DetectionResponse(
        detections=detections,
        model_used=detector.model_name,
        inference_time_ms=elapsed_ms,
        image_width=width,
        image_height=height,
    )


@router.post("/compliance/check", response_model=ComplianceResponse)
async def compliance_check(
    image: UploadFile = File(..., description="Worker photo (JPEG/PNG)."),
    worker_id: str = Form(..., description="Worker identifier, e.g. 'W-0042'."),
    job_category: str = Form(..., description="Job key from the PPE matrix."),
    detector: PPEDetector = Depends(get_detector),
) -> ComplianceResponse:
    """Full compliance check: detect → verify against Job PPE Matrix →
    Gemini explanation on FAIL (PASS responses never invoke Gemini)."""
    try:
        load_job_matrix()[job_category]
    except KeyError:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown job_category '{job_category}'. "
                   f"Known: {sorted(load_job_matrix())}",
        )

    array = await _decode_upload(image)
    detections, elapsed_ms = detector.detect(array)
    result = check_compliance(detections, job_category)

    explanation: str | None = None
    gemini_invoked = False
    if result.verdict is Verdict.FAIL:
        explanation = explain_failure(
            job_category=job_category,
            missing_equipment=result.missing_equipment,
            violations_detected=result.violations_detected,
        )
        gemini_invoked = True

    logger.info(f"[{worker_id}] {job_category}: {result.verdict.value}")
    return ComplianceResponse(
        worker_id=worker_id,
        job_category=job_category,
        verdict=result.verdict,
        confidence_avg=result.confidence_avg,
        detections=detections,
        missing_equipment=result.missing_equipment,
        violations_detected=result.violations_detected,
        unverifiable_equipment=result.unverifiable_equipment,
        explanation=explanation,
        model_used=detector.model_name,
        inference_time_ms=elapsed_ms,
        gemini_invoked=gemini_invoked,
    )


@router.get("/jobs")
def list_jobs() -> dict[str, dict[str, list[str]]]:
    """List job categories and their PPE requirements (for app dropdowns)."""
    return {
        job: {"required": req.required, "pending": req.pending}
        for job, req in load_job_matrix().items()
    }

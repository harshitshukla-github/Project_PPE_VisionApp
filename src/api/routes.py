"""API routes: /detect, /compliance/check, /health, /model/info.

All endpoints live under the ``/api/v1`` prefix (agents.md §9).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from loguru import logger

from src.api.deps import get_detector
from src.inference.compliance import (
    JobRequirements,
    MatrixUnavailableError,
    apply_verification,
    check_compliance,
    load_job_matrix,
)
from src.inference.detector import PPEDetector
from src.inference.gemini_explainer import explain_failure
from src.inference.gemini_verifier import verify_items
from src.inference.schemas import (
    ComplianceResponse,
    DetectionResponse,
    HealthResponse,
    ModelInfoResponse,
    Verdict,
)
from src.settings import settings

router = APIRouter(prefix="/api/v1")


#: Extensions we will reproduce in a dump filename. Anything else becomes
#: ``.bin`` — the uploaded name is client-controlled and never trusted.
_DUMPABLE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


def _safe_suffix(filename: str | None) -> str:
    """Extract a safe file extension from a client-supplied filename.

    Only the suffix is ever taken, and only from a fixed whitelist, so a
    hostile ``filename`` (``../../etc/passwd``, absolute paths, NUL bytes)
    cannot influence where the dump lands.

    Args:
        filename: The client-supplied upload filename, if any.

    Returns:
        A whitelisted extension including the leading dot, or ``".bin"``.
    """
    suffix = Path(filename or "").suffix.lower()
    return suffix if suffix in _DUMPABLE_SUFFIXES else ".bin"


def _dump_upload(data: bytes, filename: str | None) -> Path | None:
    """Write raw upload bytes to disk for offline diagnosis.

    Captures exactly what arrived over the wire, before any decoding, so the
    dumped file can be replayed through ``cli.py probe`` with no ambiguity
    about what the client sent. No-op unless ``PPB_DEBUG_DUMP_UPLOADS=true``.

    The output name is built entirely server-side from a UTC timestamp and a
    content hash; only a whitelisted extension is taken from the client.

    Args:
        data: Raw request-body bytes for the uploaded file.
        filename: Client-supplied filename (used only for its extension).

    Returns:
        Path written, or None when dumping is disabled or the write failed.
    """
    if not settings.ppb_debug_dump_uploads:
        return None

    import hashlib  # noqa: PLC0415
    from datetime import datetime, timezone  # noqa: PLC0415

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")[:-3]
    digest = hashlib.sha256(data).hexdigest()[:8]
    out_path = settings.ppb_debug_dump_dir / f"{stamp}_{digest}{_safe_suffix(filename)}"

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)
    except OSError as exc:
        # Diagnostics must never take down a request.
        logger.error(f"[upload-dump] Failed to write {out_path}: {exc}")
        return None

    return out_path


async def _decode_upload(image: UploadFile) -> np.ndarray:
    """Decode an uploaded file into a BGR numpy array.

    When ``PPB_DEBUG_DUMP_UPLOADS=true`` the raw bytes are written to
    ``PPB_DEBUG_DUMP_DIR`` first, and the decoded dimensions are logged — the
    fastest way to tell whether a client is downscaling before upload.

    Raises:
        HTTPException: 400 if the payload is not a decodable image.
    """
    import cv2  # noqa: PLC0415

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty image upload.")

    dumped = _dump_upload(data, image.filename)

    array = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if array is None:
        if dumped:
            logger.warning(f"[upload-dump] Undecodable payload saved to {dumped}")
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode image '{image.filename}'. Send JPEG or PNG.",
        )

    if dumped:
        height, width = array.shape[:2]
        # WARNING level so an accidentally-enabled dump is impossible to miss.
        logger.warning(
            f"[upload-dump] {dumped.name} — {len(data):,} bytes, "
            f"decoded {width}x{height} px  →  "
            f"uv run python cli.py probe {dumped} --job <job>"
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
    Gemini second opinion when the detector is unsure → explanation on FAIL.

    1. YOLO detects, and the Job PPE Matrix gives a first verdict: ``PASS``,
       ``FAIL`` (a violation class was positively detected) or
       ``NEEDS_REVIEW`` (a required item was not found, which is not proof it
       is absent).
    2. On ``NEEDS_REVIEW`` the photo goes to Gemini, which judges each
       undetected item as worn, not worn or uncertain. That settles the
       verdict as ``PASS`` or ``FAIL`` where Gemini is confident, and leaves
       ``NEEDS_REVIEW`` only for what neither model could see.
    3. Any ``FAIL``, from either model, gets a Gemini-written explanation.

    This favours accuracy over cost: every check the detector cannot settle
    costs one vision call and a few seconds. If Gemini is switched off or
    unreachable, step 2 is skipped and the verdict stays ``NEEDS_REVIEW``.
    """
    matrix = _job_matrix_or_503()
    if job_category not in matrix:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown job_category '{job_category}'. Known: {sorted(matrix)}",
        )

    array = await _decode_upload(image)
    detections, elapsed_ms = detector.detect(array)
    result = check_compliance(detections, job_category)

    gemini_invoked = False
    verification = None
    if result.verdict is Verdict.NEEDS_REVIEW:
        # A vision call takes seconds; keep it off the event loop so other
        # requests are not held up behind it.
        verification = await run_in_threadpool(
            verify_items, array, job_category, result.missing_equipment
        )
        if verification is not None:
            gemini_invoked = True
            result = apply_verification(result, verification)

    explanation: str | None = None
    if result.verdict is Verdict.FAIL:
        explanation = explain_failure(
            job_category=job_category,
            missing_equipment=result.missing_equipment,
            violations_detected=result.violations_detected,
        )
        gemini_invoked = True

    logger.info(
        f"[{worker_id}] {job_category}: {result.verdict.value} "
        f"(decided by {result.verdict_source})"
    )
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
        verdict_source=result.verdict_source,
        verification=verification,
        model_used=detector.model_name,
        inference_time_ms=elapsed_ms,
        gemini_invoked=gemini_invoked,
    )


@router.get("/jobs")
def list_jobs() -> dict[str, dict[str, list[str]]]:
    """List job categories and their PPE requirements (for app dropdowns)."""
    return {
        job: {"required": req.required, "pending": req.pending}
        for job, req in _job_matrix_or_503().items()
    }


def _job_matrix_or_503() -> dict[str, JobRequirements]:
    """The job matrix, or a 503 when the database holding it can't be read.

    Guessing the requirements would be worse than refusing the check: a
    stale or default list could pass a worker the admin now requires more of.
    """
    try:
        return load_job_matrix()
    except MatrixUnavailableError as exc:
        logger.error(str(exc))
        raise HTTPException(
            status_code=503,
            detail="Job requirements are temporarily unavailable. Try again shortly.",
        ) from exc

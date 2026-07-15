"""Dependency injection for the PPB API.

The detector is expensive to construct (model load onto GPU), so a single
instance is created at startup and shared across requests via FastAPI's
dependency system.
"""

from __future__ import annotations

from loguru import logger

from src.inference.detector import PPEDetector

_detector: PPEDetector | None = None


def init_detector(model_name: str | None = None, device: str | None = None) -> PPEDetector:
    """Create the process-wide detector instance. Called once at startup.

    Args:
        model_name: Variant to serve. Defaults to ``PPB_MODEL_NAME``.
        device: Inference device. Defaults to ``PPB_DEVICE``.

    Returns:
        The initialized detector.
    """
    global _detector
    _detector = PPEDetector(model_name=model_name, device=device)
    return _detector


def get_detector() -> PPEDetector:
    """FastAPI dependency returning the shared detector.

    Raises:
        RuntimeError: If called before :func:`init_detector` (startup hook).
    """
    if _detector is None:
        logger.error("get_detector() called before init_detector()")
        raise RuntimeError("Detector not initialized — server startup incomplete.")
    return _detector

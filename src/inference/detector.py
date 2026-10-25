"""Core PPE detection — the single model I/O point for the project.

All model inference goes through :class:`PPEDetector` (agents.md §12:
"Model I/O always through src/inference/detector.py"). The detector is
model-agnostic: it loads any trained variant from ``models/`` and exposes
one ``detect()`` method.

Usage::

    from src.inference.detector import PPEDetector

    detector = PPEDetector()               # uses PPB_MODEL_NAME from .env
    detections, elapsed_ms = detector.detect("worker.jpg")
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Windows CUDA VA-fragmentation mitigation — must precede any torch import.
# Same rationale as src/training/trainer.py; harmless on Linux.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

from loguru import logger

from src.inference.schemas import Detection
from src.settings import settings

if TYPE_CHECKING:
    import numpy as np

MODELS_DIR = Path("models")


class PPEDetector:
    """Loads a trained PPE model once and runs detection on demand.

    Args:
        model_name: Variant to load, e.g. ``"yolo11l"``. Defaults to
            ``settings.ppb_model_name``.
        weights_path: Explicit checkpoint path. Overrides ``model_name``
            resolution when given.
        device: Inference device ('0', 'cpu'). Defaults to ``settings.ppb_device``.
        confidence: Minimum detection confidence. Defaults to
            ``settings.ppb_confidence_threshold``.
        iou: NMS IoU threshold (ignored by NMS-free YOLO26 variants).
            Defaults to ``settings.ppb_iou_threshold``.

    Raises:
        FileNotFoundError: If no trained checkpoint exists for the variant.
    """

    def __init__(
        self,
        model_name: str | None = None,
        weights_path: Path | None = None,
        device: str | None = None,
        confidence: float | None = None,
        iou: float | None = None,
    ) -> None:
        self.model_name = model_name or settings.ppb_model_name
        self.device = device or settings.ppb_device
        # None: follow the live `confidence_threshold` setting (docs/admin-api.md §6).
        self._fixed_confidence = confidence
        self.iou = iou if iou is not None else settings.ppb_iou_threshold
        self.weights_path = weights_path or MODELS_DIR / f"{self.model_name}_ppe_best.pt"

        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"No trained weights at {self.weights_path}.\n"
                f"  Train first: uv run python cli.py train --model {self.model_name}"
            )

        from ultralytics import YOLO  # noqa: PLC0415 — heavy import deferred

        logger.info(f"Loading {self.model_name} from {self.weights_path} on device={self.device}")
        self._model = YOLO(str(self.weights_path))
        self.class_names: dict[int, str] = dict(self._model.names)
        logger.success(f"PPEDetector ready: {self.model_name} ({len(self.class_names)} classes)")

    @property
    def confidence(self) -> float:
        """The detection threshold in use: the one passed in, else the live setting."""
        if self._fixed_confidence is not None:
            return self._fixed_confidence
        from src.admin import runtime  # noqa: PLC0415 — avoids importing the DB layer early

        return float(runtime.get("confidence_threshold"))

    def detect(self, image: "str | Path | np.ndarray | Any") -> tuple[list[Detection], float]:
        """Run PPE detection on one image.

        Args:
            image: Image path, URL, numpy BGR array, or PIL image — anything
                ultralytics ``predict`` accepts.

        Returns:
            Tuple of (detections, elapsed_ms). Detections are sorted by
            confidence, highest first.
        """
        start = time.perf_counter()
        results = self._model.predict(
            source=image,
            conf=self.confidence,
            iou=self.iou,
            imgsz=settings.ppb_img_size,
            device=self.device,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        result = results[0]
        detections: list[Detection] = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            detections.append(
                Detection(
                    class_name=self.class_names.get(class_id, str(class_id)),
                    confidence=round(float(box.conf[0]), 4),
                    bbox=[round(float(v), 1) for v in box.xyxy[0].tolist()],
                )
            )
        detections.sort(key=lambda d: d.confidence, reverse=True)

        logger.debug(
            f"detect(): {len(detections)} detections in {elapsed_ms:.1f} ms "
            f"[{', '.join(d.class_name for d in detections)}]"
        )
        return detections, round(elapsed_ms, 1)

    @property
    def image_size(self) -> int:
        """Inference image size used for prediction."""
        return settings.ppb_img_size

"""Export trained checkpoints to deployment formats.

Primary target is ONNX (cross-platform, no Python dependency at inference
time). YOLO26 variants export **without an NMS node** in the graph —
their NMS-free architecture makes the ONNX artifact simpler and faster,
especially on CPU (agents.md §2).

Usage — Python API::

    from src.export.onnx_export import export_model

    onnx_path = export_model("yolo11l")                 # FP32 ONNX
    onnx_path = export_model("yolo26n", half=True)      # FP16 ONNX

Usage — CLI::

    uv run python cli.py export --model yolo11l --format onnx
"""

from __future__ import annotations

import shutil
from pathlib import Path

from loguru import logger

from src.settings import settings

MODELS_DIR = Path("models")
EXPORTS_DIR = MODELS_DIR / "exports"

# Formats forwarded to ultralytics .export(). Values are the file suffix
# the exporter produces for that format.
SUPPORTED_FORMATS: dict[str, str] = {
    "onnx": ".onnx",
    "torchscript": ".torchscript",
    "tflite": ".tflite",
    "engine": ".engine",  # TensorRT — requires a GPU + tensorrt installed
}


def export_model(
    model_name: str,
    fmt: str = "onnx",
    half: bool = False,
    imgsz: int | None = None,
    device: str = "cpu",
) -> Path:
    """Export a trained checkpoint to a deployment format.

    Args:
        model_name: Trained variant, e.g. ``"yolo11l"``. Weights are read
            from ``models/<model_name>_ppe_best.pt``.
        fmt: Target format — one of :data:`SUPPORTED_FORMATS`.
        half: Export FP16 weights (halves file size; needs FP16-capable
            runtime). Ignored by formats that don't support it.
        imgsz: Export image size. Defaults to ``settings.ppb_img_size``.
        device: Device used for the export trace. CPU is the safe default;
            TensorRT (``engine``) requires a GPU device like ``"0"``.

    Returns:
        Path of the exported artifact under ``models/exports/``.

    Raises:
        ValueError: If ``fmt`` is not supported.
        FileNotFoundError: If the trained checkpoint doesn't exist.
    """
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format '{fmt}'. Supported: {sorted(SUPPORTED_FORMATS)}"
        )

    weights = MODELS_DIR / f"{model_name}_ppe_best.pt"
    if not weights.exists():
        raise FileNotFoundError(
            f"No trained weights at {weights}.\n"
            f"  Train first: uv run python cli.py train --model {model_name}"
        )

    from ultralytics import YOLO  # noqa: PLC0415 — heavy import deferred

    active_imgsz = imgsz or settings.ppb_img_size
    logger.info(
        f"Exporting {model_name} → {fmt} (imgsz={active_imgsz}, half={half}, device={device})"
    )

    model = YOLO(str(weights))
    produced = Path(
        model.export(format=fmt, half=half, imgsz=active_imgsz, device=device)
    )

    # Move the artifact next to its siblings with a deterministic name.
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    precision = "_fp16" if half else ""
    dest = EXPORTS_DIR / f"{model_name}_ppe{precision}{SUPPORTED_FORMATS[fmt]}"
    if produced.resolve() != dest.resolve():
        shutil.move(str(produced), dest)

    size_mb = dest.stat().st_size / 1e6
    logger.success(f"Export complete → {dest} ({size_mb:.1f} MB)")
    return dest

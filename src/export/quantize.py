"""Post-export quantization of ONNX models.

Two precision-reduction paths exist in this project:

* **FP16** — done at export time via ``export_model(..., half=True)``
  (weights stored as float16; ~50% smaller, near-zero accuracy loss on GPU).
* **INT8 dynamic** (this module) — weights stored as int8, activations
  quantized at runtime. ~75% smaller than FP32 and faster on CPU, at the
  cost of a small accuracy drop. Always re-validate an INT8 model against
  the test split before deploying it.

Usage::

    from src.export.quantize import quantize_onnx_int8

    int8_path = quantize_onnx_int8(Path("models/exports/yolo26n_ppe.onnx"))
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger


def quantize_onnx_int8(onnx_path: Path, output_path: Path | None = None) -> Path:
    """Dynamically quantize an ONNX model to INT8.

    Args:
        onnx_path: FP32 (or FP16) ONNX model produced by
            :func:`~src.export.onnx_export.export_model`.
        output_path: Destination file. Defaults to the input path with an
            ``_int8`` suffix before the extension.

    Returns:
        Path of the quantized model.

    Raises:
        FileNotFoundError: If ``onnx_path`` doesn't exist.
        ImportError: If ``onnxruntime`` is not installed.
    """
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    from onnxruntime.quantization import QuantType, quantize_dynamic  # noqa: PLC0415

    output_path = output_path or onnx_path.with_name(
        onnx_path.stem + "_int8" + onnx_path.suffix
    )

    logger.info(f"Quantizing {onnx_path.name} → INT8 (dynamic)")
    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
    )

    before_mb = onnx_path.stat().st_size / 1e6
    after_mb = output_path.stat().st_size / 1e6
    logger.success(
        f"INT8 model → {output_path} ({before_mb:.1f} MB → {after_mb:.1f} MB, "
        f"-{(1 - after_mb / before_mb) * 100:.0f}%)"
    )
    logger.warning(
        "Re-validate the INT8 model on the test split before deployment — "
        "quantization can cost accuracy."
    )
    return output_path

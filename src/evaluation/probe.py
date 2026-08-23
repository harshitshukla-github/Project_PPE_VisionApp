"""Single-image detection probe — diagnose why one photo produced its verdict.

``src/evaluation/benchmark.py`` answers "how good is the model on the test
split?". This module answers the complementary question that comes up when a
real photo produces a surprising verdict: **"what did the model actually see
in *this* image, and why did that become a FAIL?"**

The probe runs the image through the model at a near-zero confidence floor
(default 0.01), so every candidate detection is visible — including the ones
the production threshold silently discards. It then re-runs the image at
larger input sizes and with test-time augmentation to show whether the cheap
config levers would recover a missed item.

For each PPE class required by the job it prints one of four diagnoses:

    DETECTED       — found above the production threshold; working as intended
    CONFIG FIX     — missed at baseline, recovered by a larger imgsz or TTA
    THRESHOLD      — the model saw it, but below the production threshold
    DATA GAP       — never seen at any config, even at conf 0.01

That distinction is the point: ``THRESHOLD``/``CONFIG FIX`` are a one-line
settings change, ``DATA GAP`` means the model needs more training data.

Note:
    The image is decoded with ``cv2.imdecode`` on the raw file bytes — byte for
    byte what ``src/api/routes.py::_decode_upload`` does — so the probe
    reproduces the server's preprocessing exactly rather than an approximation
    of it.

Usage::

    uv run python cli.py probe photo.jpg --job general_labor
    uv run python cli.py probe photo.jpg --job welder --model yolo26l --save
"""

from __future__ import annotations

import os

# Windows CUDA VA-fragmentation mitigation — must precede any torch import.
# Same rationale as src/inference/detector.py; harmless on Linux.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.inference.schemas import Detection
from src.settings import settings

if TYPE_CHECKING:
    import numpy as np

MODELS_DIR = Path("models")
PROBE_OUTPUT_DIR = Path("runs") / "probe"

#: Confidence floor for the probe. Low enough to surface everything the model
#: considered, including detections production would discard.
PROBE_CONF_FLOOR = 0.01

#: Config variants swept, as (label, imgsz, augment). ``imgsz=None`` means
#: "use PPB_IMG_SIZE" — i.e. exactly what production runs today.
VariantSpec = tuple[str, int | None, bool]

DEFAULT_VARIANTS: tuple[VariantSpec, ...] = (
    ("baseline", None, False),
    ("imgsz 960", 960, False),
    ("imgsz 1280", 1280, False),
    ("1280 + TTA", 1280, True),
)


#: Below this confidence a detection is noise, not a near-miss. Object
#: detectors emit low-scoring boxes for most classes on most images; treating
#: those as "the model nearly saw it" would recommend a threshold that admits
#: spurious boxes for every class at once. Anything under this floor means the
#: model has no usable signal, which is a data/image problem, not a tuning one.
NOISE_FLOOR = 0.15

#: Diagnosis labels, ordered most to least severe.
DIAG_DATA_GAP = "DATA GAP"
DIAG_NOISE = "NOISE ONLY"
DIAG_THRESHOLD = "THRESHOLD"
DIAG_CONFIG_FIX = "CONFIG FIX"
DIAG_DETECTED = "DETECTED"

console = Console()


def _supports_tta(model_name: str) -> bool:
    """Whether a model variant supports test-time augmentation.

    YOLO26 is an end-to-end (NMS-free) architecture; ultralytics rejects
    ``augment=True`` for it and silently falls back to single-scale
    prediction. Reporting that fallback as a "TTA" result would be a lie, so
    the variant is dropped instead.

    Args:
        model_name: Model variant name, e.g. ``"yolo26l"``.

    Returns:
        True if ``augment=True`` is meaningful for this variant.
    """
    return not model_name.startswith("yolo26")


@dataclass
class VariantResult:
    """Detections produced by one config variant.

    Attributes:
        label: Human-readable variant name, e.g. ``"imgsz 1280"``.
        imgsz: Inference image size actually used.
        augment: Whether test-time augmentation was enabled.
        detections: All detections at the probe confidence floor.
        elapsed_ms: Wall-clock inference time for this variant.
    """

    label: str
    imgsz: int
    augment: bool
    detections: list[Detection]
    elapsed_ms: float

    def best_conf(self, class_name: str) -> float:
        """Return the highest confidence for ``class_name``, or 0.0 if absent.

        Args:
            class_name: PPE class to look up.

        Returns:
            Best confidence in ``[0.0, 1.0]``; 0.0 means never detected.
        """
        confs = [d.confidence for d in self.detections if d.class_name == class_name]
        return max(confs) if confs else 0.0


@dataclass
class ClassDiagnosis:
    """The verdict for one required PPE class.

    Attributes:
        class_name: PPE class being diagnosed.
        diagnosis: One of the ``DIAG_*`` constants.
        baseline_conf: Best confidence under the production config.
        best_conf: Best confidence across every variant tried.
        best_variant: Label of the variant that achieved ``best_conf``.
    """

    class_name: str
    diagnosis: str
    baseline_conf: float
    best_conf: float
    best_variant: str


def _warn_if_undersized(width: int, height: int) -> bool:
    """Warn when the image is too small for the model's input size.

    The model is trained and run at ``PPB_IMG_SIZE`` (640 by default). An image
    whose long edge is well under that gets upscaled into blur before the model
    ever sees it, and every confidence in the report becomes unreliable —
    typically showing up as scores that jump around unpredictably between input
    sizes rather than improving with them.

    A phone photo arriving this small means something upstream (capture,
    resize, or compression in the client) is destroying the image before
    upload, which is worth fixing before any model conclusion is drawn.

    Args:
        width: Decoded image width in pixels.
        height: Decoded image height in pixels.

    Returns:
        True if a warning was emitted.
    """
    long_edge = max(width, height)
    target = settings.ppb_img_size
    if long_edge >= target:
        return False

    console.print(Panel(
        f"This image's long edge is [bold red]{long_edge}px[/], below the model's "
        f"[bold]{target}px[/] input size — it will be upscaled "
        f"{target / long_edge:.1f}x into blur before inference.\n\n"
        "[bold]Treat every confidence below as unreliable.[/] Fix the image "
        "source first: a phone photo should arrive at 1000px+ on the long edge. "
        "If it doesn't, the client is downscaling or over-compressing before "
        "upload, and that is the bug to chase — not the model.",
        title="⚠  Image too small to judge the model",
        border_style="red",
    ))
    return True


def _resolve_weights(model_name: str) -> Path:
    """Resolve the checkpoint path for a model variant.

    Args:
        model_name: Variant name, e.g. ``"yolo11l"``.

    Returns:
        Path to ``models/<model_name>_ppe_best.pt``.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
    """
    path = MODELS_DIR / f"{model_name}_ppe_best.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"No trained weights at {path}.\n"
            f"  Train first: uv run python cli.py train --model {model_name}"
        )
    return path


def _decode_like_api(image_path: Path) -> np.ndarray:
    """Decode an image exactly as the API's upload handler does.

    Mirrors ``src/api/routes.py::_decode_upload`` — reads raw bytes and calls
    ``cv2.imdecode`` — so the probe sees the same pixels the server would.

    Args:
        image_path: Path to a JPEG/PNG on disk.

    Returns:
        BGR image array.

    Raises:
        FileNotFoundError: If ``image_path`` does not exist.
        ValueError: If the bytes are not a decodable image.
    """
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    data = image_path.read_bytes()
    array = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if array is None:
        raise ValueError(f"Could not decode '{image_path}'. Expected JPEG or PNG.")
    return array


def _run_variant(
    model: object,
    array: np.ndarray,
    spec: VariantSpec,
    device: str,
    conf_floor: float,
) -> VariantResult:
    """Run one config variant over the image.

    Args:
        model: Loaded ultralytics ``YOLO`` instance.
        array: BGR image array.
        spec: ``(label, imgsz, augment)`` variant specification.
        device: Inference device ('0', 'cpu').
        conf_floor: Confidence floor for the sweep.

    Returns:
        Populated :class:`VariantResult`.
    """
    import time  # noqa: PLC0415

    label, imgsz, augment = spec
    active_imgsz = imgsz if imgsz is not None else settings.ppb_img_size

    start = time.perf_counter()
    results = model.predict(  # type: ignore[attr-defined]
        source=array,
        conf=conf_floor,
        iou=settings.ppb_iou_threshold,
        imgsz=active_imgsz,
        device=device,
        augment=augment,
        verbose=False,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    result = results[0]
    names: dict[int, str] = dict(result.names)
    detections = [
        Detection(
            class_name=names.get(int(box.cls[0]), str(int(box.cls[0]))),
            confidence=round(float(box.conf[0]), 4),
            bbox=[round(float(v), 1) for v in box.xyxy[0].tolist()],
        )
        for box in result.boxes
    ]
    detections.sort(key=lambda d: d.confidence, reverse=True)

    return VariantResult(
        label=label,
        imgsz=active_imgsz,
        augment=augment,
        detections=detections,
        elapsed_ms=round(elapsed_ms, 1),
    )


def _diagnose(
    class_name: str,
    baseline: VariantResult,
    variants: list[VariantResult],
    prod_conf: float,
) -> ClassDiagnosis:
    """Classify why a required class did or didn't make it into the verdict.

    Args:
        class_name: PPE class to diagnose.
        baseline: Result under the production config.
        variants: All variant results, including ``baseline``.
        prod_conf: Production confidence threshold.

    Returns:
        Populated :class:`ClassDiagnosis`.
    """
    baseline_conf = baseline.best_conf(class_name)
    best_variant, best_conf = max(
        ((v.label, v.best_conf(class_name)) for v in variants),
        key=lambda pair: pair[1],
    )

    if baseline_conf >= prod_conf:
        diagnosis = DIAG_DETECTED
    elif best_conf >= prod_conf:
        diagnosis = DIAG_CONFIG_FIX
    elif best_conf >= NOISE_FLOOR:
        diagnosis = DIAG_THRESHOLD
    elif best_conf > 0.0:
        diagnosis = DIAG_NOISE
    else:
        diagnosis = DIAG_DATA_GAP

    return ClassDiagnosis(
        class_name=class_name,
        diagnosis=diagnosis,
        baseline_conf=baseline_conf,
        best_conf=best_conf,
        best_variant=best_variant,
    )


def _save_annotated(
    array: np.ndarray,
    baseline: VariantResult,
    image_path: Path,
    prod_conf: float,
) -> Path:
    """Draw every probed detection on the image and save it.

    Boxes at or above the production threshold are drawn in green; boxes below
    it — the ones production throws away — are drawn in red, which is usually
    where the answer is.

    Args:
        array: BGR image array.
        baseline: Baseline variant result supplying the boxes.
        image_path: Original image path (used for the output filename).
        prod_conf: Production confidence threshold.

    Returns:
        Path to the written annotation.
    """
    import cv2  # noqa: PLC0415

    canvas = array.copy()
    kept = (0, 200, 0)
    dropped = (0, 0, 220)

    for det in baseline.detections:
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        colour = kept if det.confidence >= prod_conf else dropped
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(
            canvas,
            f"{det.class_name} {det.confidence:.2f}",
            (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            colour,
            2,
        )

    PROBE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROBE_OUTPUT_DIR / f"{image_path.stem}_probe.jpg"
    cv2.imwrite(str(out_path), canvas)
    return out_path


def probe_image(
    image_path: Path,
    job_category: str | None = None,
    model_name: str | None = None,
    device: str | None = None,
    conf_floor: float = PROBE_CONF_FLOOR,
    variants: tuple[VariantSpec, ...] = DEFAULT_VARIANTS,
    save_annotated: bool = False,
) -> list[ClassDiagnosis]:
    """Probe one image and print a full diagnostic report.

    Args:
        image_path: Photo to analyse.
        job_category: Job key from the PPE matrix. When given, the report
            focuses on that job's required classes and shows the verdict the
            API would return. When omitted, every detected class is reported.
        model_name: Model variant. Defaults to ``PPB_MODEL_NAME``.
        device: Inference device. Defaults to ``PPB_DEVICE``.
        conf_floor: Confidence floor for the sweep.
        variants: Config variants to sweep.
        save_annotated: Write an annotated copy to ``runs/probe/``.

    Returns:
        One :class:`ClassDiagnosis` per required class (empty when
        ``job_category`` is None).

    Raises:
        FileNotFoundError: If the image or the model weights are missing.
        KeyError: If ``job_category`` is not in the PPE matrix.
    """
    from ultralytics import YOLO  # noqa: PLC0415 — heavy import deferred

    active_model = model_name or settings.ppb_model_name
    active_device = device or settings.ppb_device
    prod_conf = settings.ppb_confidence_threshold
    weights = _resolve_weights(active_model)

    array = _decode_like_api(image_path)
    height, width = array.shape[:2]

    console.print(Panel(
        f"Image: [bold cyan]{image_path.name}[/]  ({width}x{height} px)\n"
        f"Model: [bold]{active_model}[/]  |  Device: [bold]{active_device}[/]\n"
        f"Production threshold: [bold]{prod_conf}[/]  |  "
        f"Probe floor: [bold]{conf_floor}[/]",
        title="PPB — Detection Probe",
        border_style="blue",
    ))
    _warn_if_undersized(width, height)

    logger.info(f"Loading {active_model} from {weights}")
    model = YOLO(str(weights))

    # Drop TTA variants the architecture can't honour, rather than reporting
    # ultralytics' silent single-scale fallback as if it were a TTA result.
    active_variants = list(variants)
    if not _supports_tta(active_model):
        dropped = [label for label, _, augment in active_variants if augment]
        active_variants = [v for v in active_variants if not v[2]]
        if dropped:
            console.print(
                f"[dim]Skipping {', '.join(dropped)} — {active_model} is NMS-free "
                "and does not support TTA.[/]"
            )

    # Warm up CUDA/cuDNN before timing: the first predict() call pays kernel
    # autotuning and allocator setup that would otherwise be charged to the
    # baseline variant, making 640 look slower than 1280.
    model.predict(
        source=array,
        conf=conf_floor,
        imgsz=settings.ppb_img_size,
        device=active_device,
        verbose=False,
    )

    results = [
        _run_variant(model, array, spec, active_device, conf_floor)
        for spec in active_variants
    ]
    baseline = results[0]

    _print_baseline_table(baseline, prod_conf)
    _print_variant_table(results, prod_conf)

    diagnoses: list[ClassDiagnosis] = []
    if job_category:
        diagnoses = _print_job_report(job_category, baseline, results, prod_conf)

    if save_annotated:
        out_path = _save_annotated(array, baseline, image_path, prod_conf)
        console.print(
            f"\n[green]Annotated image:[/] {out_path}  "
            f"[dim](green = kept, red = below threshold)[/]"
        )

    return diagnoses


def _print_baseline_table(baseline: VariantResult, prod_conf: float) -> None:
    """Print every detection the production config considered."""
    table = Table(
        title=f"All detections at conf >= {PROBE_CONF_FLOOR} (production config)",
        header_style="bold blue",
        show_lines=False,
    )
    table.add_column("Class", style="cyan")
    table.add_column("Confidence", justify="right")
    table.add_column("Production", justify="center")
    table.add_column("BBox [x1,y1,x2,y2]", style="dim")

    if not baseline.detections:
        console.print("[yellow]No detections at all, even at conf 0.01.[/]")
        return

    for det in baseline.detections:
        survives = det.confidence >= prod_conf
        table.add_row(
            det.class_name,
            f"{det.confidence:.4f}",
            "[green]kept[/]" if survives else "[red]dropped[/]",
            str([int(v) for v in det.bbox]),
        )
    console.print(table)


def _print_variant_table(results: list[VariantResult], prod_conf: float) -> None:
    """Print best-confidence-per-class across every config variant."""
    all_classes = sorted({d.class_name for r in results for d in r.detections})
    if not all_classes:
        return

    table = Table(
        title="Best confidence per class, by config",
        header_style="bold blue",
        show_lines=False,
    )
    table.add_column("Class", style="cyan")
    for r in results:
        table.add_column(f"{r.label}\n[dim]{r.elapsed_ms:.0f} ms[/]", justify="right")

    for class_name in all_classes:
        cells: list[str] = []
        for r in results:
            conf = r.best_conf(class_name)
            if conf == 0.0:
                cells.append("[dim]—[/]")
            elif conf >= prod_conf:
                cells.append(f"[green]{conf:.3f}[/]")
            else:
                cells.append(f"[red]{conf:.3f}[/]")
        table.add_row(class_name, *cells)

    console.print(table)
    console.print(
        f"[dim]green >= production threshold ({prod_conf}), red = discarded[/]"
    )


def _print_job_report(
    job_category: str,
    baseline: VariantResult,
    results: list[VariantResult],
    prod_conf: float,
) -> list[ClassDiagnosis]:
    """Print the per-class diagnosis and the verdict the API would return.

    Args:
        job_category: Job key from the PPE matrix.
        baseline: Baseline variant result.
        results: All variant results.
        prod_conf: Production confidence threshold.

    Returns:
        One diagnosis per required class.
    """
    from src.inference.compliance import check_compliance, load_job_matrix  # noqa: PLC0415

    matrix = load_job_matrix()
    if job_category not in matrix:
        raise KeyError(
            f"Unknown job category '{job_category}'. Known: {sorted(matrix)}"
        )
    required = matrix[job_category].required

    diagnoses = [
        _diagnose(class_name, baseline, results, prod_conf)
        for class_name in required
    ]

    style = {
        DIAG_DETECTED: "green",
        DIAG_CONFIG_FIX: "yellow",
        DIAG_THRESHOLD: "yellow",
        DIAG_NOISE: "red",
        DIAG_DATA_GAP: "red",
    }
    explain = {
        DIAG_DETECTED: "found above threshold",
        DIAG_CONFIG_FIX: "recovered by a bigger imgsz / TTA",
        DIAG_THRESHOLD: "model saw it, threshold discarded it",
        DIAG_NOISE: f"only noise-level boxes (<{NOISE_FLOOR}) — no usable signal",
        DIAG_DATA_GAP: "never seen — needs training data",
    }

    table = Table(
        title=f"Diagnosis — required PPE for '{job_category}'",
        header_style="bold blue",
        show_lines=True,
    )
    table.add_column("Class", style="cyan")
    table.add_column("Baseline", justify="right")
    table.add_column("Best (any config)", justify="right")
    table.add_column("Diagnosis")
    table.add_column("Meaning", style="dim")

    for d in diagnoses:
        colour = style[d.diagnosis]
        best = (
            f"{d.best_conf:.3f} [dim]({d.best_variant})[/]"
            if d.best_conf > 0 else "[dim]—[/]"
        )
        table.add_row(
            d.class_name,
            f"{d.baseline_conf:.3f}" if d.baseline_conf > 0 else "[dim]—[/]",
            best,
            f"[{colour}]{d.diagnosis}[/]",
            explain[d.diagnosis],
        )
    console.print(table)

    # The verdict the API would actually return for this photo.
    production_dets = [d for d in baseline.detections if d.confidence >= prod_conf]
    result = check_compliance(production_dets, job_category)
    verdict_colour = "green" if result.verdict.value == "PASS" else "red"
    console.print(Panel(
        f"Verdict: [bold {verdict_colour}]{result.verdict.value}[/]\n"
        f"Missing: [bold]{result.missing_equipment or 'none'}[/]\n"
        f"Violations: [bold]{result.violations_detected or 'none'}[/]\n"
        f"Unverifiable: [dim]{result.unverifiable_equipment or 'none'}[/]",
        title="What the API would return for this photo",
        border_style=verdict_colour,
    ))

    _print_conclusion(diagnoses, prod_conf)
    return diagnoses


def _print_conclusion(diagnoses: list[ClassDiagnosis], prod_conf: float) -> None:
    """Print the actionable takeaway for the operator."""
    gaps = [d for d in diagnoses if d.diagnosis == DIAG_DATA_GAP]
    noise = [d for d in diagnoses if d.diagnosis == DIAG_NOISE]
    thresh = [d for d in diagnoses if d.diagnosis == DIAG_THRESHOLD]
    config = [d for d in diagnoses if d.diagnosis == DIAG_CONFIG_FIX]

    lines: list[str] = []
    if thresh:
        worst = min(d.best_conf for d in thresh)
        names = ", ".join(d.class_name for d in thresh)
        # Sit just under the weakest genuine detection, but never below the
        # noise floor — a threshold there would admit spurious boxes wholesale.
        suggested = max(round(worst - 0.05, 2), NOISE_FLOOR)
        lines.append(
            f"[yellow]THRESHOLD[/] — {names} detected below {prod_conf} "
            f"(weakest {worst:.3f}).\n"
            f"  Fix: lower the threshold for these classes to ~{suggested:.2f}. "
            f"No retraining needed.\n"
            f"  Check the precision cost first: a lower floor admits more false "
            f"positives for that class."
        )
    if noise:
        names = ", ".join(f"{d.class_name} ({d.best_conf:.3f})" for d in noise)
        lines.append(
            f"[red]NOISE ONLY[/] — {names}: every box is below the {NOISE_FLOOR} "
            "noise floor.\n"
            "  [bold]Do not lower the threshold to reach these.[/] Detectors emit "
            "low-scoring boxes for most classes on most images; a floor that low "
            "would admit spurious boxes for every class at once.\n"
            "  Treat this as a data/image problem, not a tuning one."
        )
    if config:
        names = ", ".join(f"{d.class_name} via {d.best_variant}" for d in config)
        lines.append(
            f"[yellow]CONFIG FIX[/] — recovered by a larger input size / TTA: {names}.\n"
            "  Fix: raise imgsz or enable augment=True for the API path."
        )
    if gaps:
        names = ", ".join(d.class_name for d in gaps)
        lines.append(
            f"[red]DATA GAP[/] — {names} not found at any config, even at conf 0.01.\n"
            "  This is a model/data problem: no config change will fix it."
        )
    if not lines:
        lines.append("[green]All required PPE detected above threshold.[/]")

    console.print(Panel(
        "\n\n".join(lines),
        title="Conclusion",
        border_style="magenta",
    ))

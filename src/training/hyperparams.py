"""Hyperparameter loading and validation for the PPB training pipeline.

Defines a Pydantic model (``TrainingConfig``) that maps 1-to-1 to the
``*_train.yaml`` files in ``configs/``. Use ``load_training_config`` to
load a YAML and get a validated, typed config object.

Usage::

    from src.training.hyperparams import load_training_config, config_path_for

    cfg = load_training_config(config_path_for("yolo26l"))
    kwargs = cfg.to_ultralytics_kwargs()
    # → pass directly to YOLO.train(**kwargs)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from src.settings import settings

# Supported model variants — used for config file name resolution
SUPPORTED_VARIANTS: tuple[str, ...] = (
    "yolo26l", "yolo26n", "yolo26m", "yolo26s", "yolo26x",
    "yolo11l", "yolo11n", "yolo11m", "yolo11s", "yolo11x",
)


class TrainingConfig(BaseModel):
    """All ultralytics training hyperparameters, validated via Pydantic.

    Each field corresponds directly to a kwarg accepted by ``YOLO.train()``.
    See: https://docs.ultralytics.com/modes/train/#arguments

    Notes:
        - ``dfl`` should be ``0.0`` for YOLO26 variants (DFL removed).
        - ``optimizer`` will be overridden by ultralytics to MuSGD for
          YOLO26 checkpoints — setting ``SGD`` here is safe.
    """

    # --- Core training ---
    epochs: int = Field(100, ge=1, le=1000, description="Total training epochs.")
    imgsz: int = Field(640, ge=32, description="Training image size (must be multiple of 32).")
    batch: int = Field(16, ge=1, description="Batch size. -1 for auto.")
    patience: int = Field(20, ge=0, description="Early stopping patience (0 = disabled).")

    # --- Optimizer ---
    optimizer: Literal["SGD", "Adam", "AdamW", "RMSProp", "auto"] = Field(
        "SGD", description="Gradient optimizer."
    )
    lr0: float = Field(0.01, gt=0.0, description="Initial learning rate.")
    lrf: float = Field(0.01, gt=0.0, description="Final LR factor (lr_final = lr0 * lrf).")
    momentum: float = Field(0.937, ge=0.0, le=1.0, description="SGD momentum.")
    weight_decay: float = Field(0.0005, ge=0.0, description="Optimizer weight decay.")

    # --- Warmup ---
    warmup_epochs: float = Field(3.0, ge=0.0, description="Warmup duration in epochs.")
    warmup_momentum: float = Field(0.8, ge=0.0, le=1.0, description="Initial warmup momentum.")

    # --- Loss component weights ---
    box: float = Field(7.5, ge=0.0, description="Box regression loss weight.")
    cls: float = Field(0.5, ge=0.0, description="Classification loss weight.")
    dfl: float = Field(
        1.5, ge=0.0,
        description="DFL loss weight. Set to 0.0 for YOLO26 (DFL removed in architecture).",
    )

    # --- Ultralytics built-in augmentation ---
    hsv_h: float = Field(0.015, ge=0.0, le=1.0, description="HSV hue augmentation fraction.")
    hsv_s: float = Field(0.7, ge=0.0, le=1.0, description="HSV saturation augmentation fraction.")
    hsv_v: float = Field(0.4, ge=0.0, le=1.0, description="HSV value augmentation fraction.")
    degrees: float = Field(0.0, ge=0.0, le=360.0, description="Rotation augmentation (degrees).")
    translate: float = Field(0.1, ge=0.0, le=1.0, description="Translation augmentation fraction.")
    scale: float = Field(0.5, ge=0.0, description="Scale augmentation gain.")
    fliplr: float = Field(0.5, ge=0.0, le=1.0, description="Horizontal flip probability.")
    mosaic: float = Field(1.0, ge=0.0, le=1.0, description="Mosaic augmentation probability.")
    mixup: float = Field(0.0, ge=0.0, le=1.0, description="Mixup augmentation probability.")

    # --- Run settings ---
    pretrained: bool = Field(True, description="Initialize from COCO pretrained weights.")
    exist_ok: bool = Field(False, description="Allow overwriting an existing run directory.")

    @field_validator("imgsz")
    @classmethod
    def _imgsz_multiple_of_32(cls, v: int) -> int:
        """YOLO architectures require image dimensions divisible by 32."""
        if v % 32 != 0:
            raise ValueError(f"imgsz must be a multiple of 32, got {v}")
        return v

    def to_ultralytics_kwargs(self) -> dict:
        """Return a dict ready to unpack into ``YOLO.train(**kwargs)``.

        Excludes fields that are passed separately by the trainer
        (``data``, ``project``, ``name``, ``device``).

        Returns:
            Dict of validated hyperparameter kwargs.
        """
        return self.model_dump()


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def config_path_for(model_name: str) -> Path:
    """Return the expected config YAML path for a given model variant.

    Args:
        model_name: Model variant name, e.g. ``"yolo26l"`` or ``"yolo11n"``.

    Returns:
        ``configs/<model_name>_train.yaml`` resolved from project root.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = settings.configs_dir / f"{model_name}_train.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"No training config for '{model_name}' at {path}.\n"
            f"  Expected: configs/{model_name}_train.yaml\n"
            f"  Supported variants: {', '.join(SUPPORTED_VARIANTS)}"
        )
    return path


def load_training_config(config_path: Path) -> TrainingConfig:
    """Load and validate a training hyperparameter YAML.

    Args:
        config_path: Path to a ``*_train.yaml`` file in ``configs/``.

    Returns:
        Validated :class:`TrainingConfig` instance.

    Raises:
        FileNotFoundError: If ``config_path`` does not exist.
        pydantic.ValidationError: If any field fails validation.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Training config not found: {config_path}")

    with config_path.open() as f:
        raw: dict = yaml.safe_load(f) or {}

    cfg = TrainingConfig(**raw)
    logger.debug(
        f"Loaded training config from {config_path.name}: "
        f"epochs={cfg.epochs}, imgsz={cfg.imgsz}, batch={cfg.batch}, "
        f"dfl={cfg.dfl}, optimizer={cfg.optimizer}"
    )
    return cfg

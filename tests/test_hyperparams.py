"""Unit tests for src/training/hyperparams.py — config loading/validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.training.hyperparams import (
    SUPPORTED_VARIANTS,
    TrainingConfig,
    config_path_for,
    load_training_config,
)


class TestTrainingConfig:
    def test_defaults_are_valid(self) -> None:
        cfg = TrainingConfig()
        assert cfg.epochs == 100
        assert cfg.imgsz == 640

    def test_imgsz_must_be_multiple_of_32(self) -> None:
        with pytest.raises(ValidationError, match="multiple of 32"):
            TrainingConfig(imgsz=417)

    def test_negative_lr_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TrainingConfig(lr0=-0.1)

    def test_to_ultralytics_kwargs_excludes_nothing_extra(self) -> None:
        kwargs = TrainingConfig().to_ultralytics_kwargs()
        assert kwargs["epochs"] == 100
        assert "data" not in kwargs  # passed separately by the trainer
        assert "device" not in kwargs


class TestConfigFiles:
    """The committed YAML configs must always load and follow architecture rules."""

    @pytest.mark.parametrize("variant", ["yolo26l", "yolo26n", "yolo11l", "yolo11n"])
    def test_committed_config_loads(self, variant: str) -> None:
        cfg = load_training_config(config_path_for(variant))
        assert cfg.epochs > 0

    @pytest.mark.parametrize("variant", ["yolo26l", "yolo26n"])
    def test_yolo26_has_dfl_disabled(self, variant: str) -> None:
        """YOLO26 removed DFL — its configs must set dfl: 0.0."""
        cfg = load_training_config(config_path_for(variant))
        assert cfg.dfl == 0.0

    @pytest.mark.parametrize("variant", ["yolo11l", "yolo11n"])
    def test_yolo11_has_dfl_active(self, variant: str) -> None:
        cfg = load_training_config(config_path_for(variant))
        assert cfg.dfl > 0.0

    def test_unknown_variant_raises_with_supported_list(self) -> None:
        with pytest.raises(FileNotFoundError) as excinfo:
            config_path_for("yolo99z")
        assert SUPPORTED_VARIANTS[0] in str(excinfo.value)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_training_config(tmp_path / "ghost.yaml")

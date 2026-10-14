"""Unit tests for src/evaluation/probe.py — the single-image diagnostic.

Covers the pure decision logic (which is where the value is) without loading
a real YOLO model, per the suite-wide rule in conftest.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.evaluation.probe import (
    DEFAULT_VARIANTS,
    DIAG_CONFIG_FIX,
    DIAG_DATA_GAP,
    DIAG_DETECTED,
    DIAG_NOISE,
    DIAG_THRESHOLD,
    NOISE_FLOOR,
    ClassDiagnosis,
    VariantResult,
    _decode_like_api,
    _diagnose,
    _print_baseline_table,
    _print_conclusion,
    _print_job_report,
    _print_variant_table,
    _resolve_weights,
    _save_annotated,
    _supports_tta,
    _warn_if_undersized,
)
from tests.conftest import make_detection

PROD_CONF = 0.45


def make_variant(
    label: str,
    confs: dict[str, float],
    imgsz: int = 640,
    augment: bool = False,
) -> VariantResult:
    """Build a VariantResult from a {class_name: confidence} mapping."""
    return VariantResult(
        label=label,
        imgsz=imgsz,
        augment=augment,
        detections=[make_detection(name, conf) for name, conf in confs.items()],
        elapsed_ms=42.0,
    )


class TestSupportsTTA:
    """YOLO26 is NMS-free and silently ignores augment=True."""

    @pytest.mark.parametrize("name", ["yolo26l", "yolo26n", "yolo26x"])
    def test_yolo26_does_not_support_tta(self, name: str) -> None:
        assert _supports_tta(name) is False

    @pytest.mark.parametrize("name", ["yolo11l", "yolo11n", "yolo11x"])
    def test_yolo11_supports_tta(self, name: str) -> None:
        assert _supports_tta(name) is True


class TestVariantResult:
    def test_best_conf_returns_highest_for_class(self) -> None:
        variant = VariantResult(
            label="v", imgsz=640, augment=False, elapsed_ms=1.0,
            detections=[
                make_detection("hardhat", 0.3),
                make_detection("hardhat", 0.9),
                make_detection("safety_vest", 0.5),
            ],
        )
        assert variant.best_conf("hardhat") == 0.9
        assert variant.best_conf("safety_vest") == 0.5

    def test_best_conf_is_zero_when_class_absent(self) -> None:
        variant = make_variant("v", {"hardhat": 0.9})
        assert variant.best_conf("safety_vest") == 0.0

    def test_best_conf_on_empty_detections(self) -> None:
        assert make_variant("v", {}).best_conf("hardhat") == 0.0


class TestDiagnose:
    """The four-way diagnosis is the core of the tool — cover every branch."""

    def test_detected_when_baseline_above_threshold(self) -> None:
        baseline = make_variant("baseline", {"safety_vest": 0.88})
        result = _diagnose("safety_vest", baseline, [baseline], PROD_CONF)
        assert result.diagnosis == DIAG_DETECTED
        assert result.baseline_conf == 0.88

    def test_config_fix_when_only_another_variant_clears_threshold(self) -> None:
        baseline = make_variant("baseline", {"safety_vest": 0.20})
        bigger = make_variant("imgsz 1280", {"safety_vest": 0.71})
        result = _diagnose("safety_vest", baseline, [baseline, bigger], PROD_CONF)
        assert result.diagnosis == DIAG_CONFIG_FIX
        assert result.best_variant == "imgsz 1280"
        assert result.best_conf == 0.71

    def test_threshold_when_seen_but_always_below_cutoff(self) -> None:
        baseline = make_variant("baseline", {"safety_vest": 0.20})
        bigger = make_variant("imgsz 1280", {"safety_vest": 0.33})
        result = _diagnose("safety_vest", baseline, [baseline, bigger], PROD_CONF)
        assert result.diagnosis == DIAG_THRESHOLD
        assert result.best_conf == 0.33

    def test_noise_only_when_every_box_is_below_the_noise_floor(self) -> None:
        """Regression: 0.04 is noise, not a near-miss.

        Diagnosing this as THRESHOLD led the tool to recommend a 0.05 cutoff,
        which would admit spurious boxes for every class at once.
        """
        baseline = make_variant("baseline", {"safety_vest": 0.041})
        tta = make_variant("1280 + TTA", {"safety_vest": 0.025})
        result = _diagnose("safety_vest", baseline, [baseline, tta], PROD_CONF)
        assert result.diagnosis == DIAG_NOISE

    def test_noise_floor_boundary_counts_as_threshold(self) -> None:
        baseline = make_variant("baseline", {"safety_vest": NOISE_FLOOR})
        result = _diagnose("safety_vest", baseline, [baseline], PROD_CONF)
        assert result.diagnosis == DIAG_THRESHOLD

    def test_just_below_noise_floor_is_noise(self) -> None:
        baseline = make_variant("baseline", {"safety_vest": NOISE_FLOOR - 0.001})
        result = _diagnose("safety_vest", baseline, [baseline], PROD_CONF)
        assert result.diagnosis == DIAG_NOISE

    def test_data_gap_when_never_detected_anywhere(self) -> None:
        baseline = make_variant("baseline", {"hardhat": 0.9})
        bigger = make_variant("imgsz 1280", {"hardhat": 0.9})
        result = _diagnose("safety_vest", baseline, [baseline, bigger], PROD_CONF)
        assert result.diagnosis == DIAG_DATA_GAP
        assert result.best_conf == 0.0

    def test_exactly_at_threshold_counts_as_detected(self) -> None:
        """Boundary: the production filter is >=, so the probe must agree."""
        baseline = make_variant("baseline", {"safety_vest": PROD_CONF})
        result = _diagnose("safety_vest", baseline, [baseline], PROD_CONF)
        assert result.diagnosis == DIAG_DETECTED

    def test_best_variant_tracks_the_winning_config(self) -> None:
        baseline = make_variant("baseline", {"mask": 0.1})
        mid = make_variant("imgsz 960", {"mask": 0.4})
        top = make_variant("1280 + TTA", {"mask": 0.6})
        result = _diagnose("mask", baseline, [baseline, mid, top], PROD_CONF)
        assert result.best_variant == "1280 + TTA"


class TestResolveWeights:
    def test_missing_weights_raise_with_actionable_message(self) -> None:
        with pytest.raises(FileNotFoundError, match="train --model"):
            _resolve_weights("yolo11-does-not-exist")


class TestDecodeLikeAPI:
    def test_decodes_valid_image(self, sample_image_bytes: bytes, tmp_path: Path) -> None:
        img_path = tmp_path / "photo.png"
        img_path.write_bytes(sample_image_bytes)
        array = _decode_like_api(img_path)
        assert array.shape == (64, 64, 3)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Image not found"):
            _decode_like_api(tmp_path / "nope.jpg")

    def test_undecodable_bytes_raise_valueerror(self, tmp_path: Path) -> None:
        bad = tmp_path / "not_an_image.jpg"
        bad.write_bytes(b"this is definitely not a JPEG")
        with pytest.raises(ValueError, match="Could not decode"):
            _decode_like_api(bad)


class TestSaveAnnotated:
    def test_writes_annotated_file(
        self, sample_image: np.ndarray, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("src.evaluation.probe.PROBE_OUTPUT_DIR", tmp_path / "probe")
        baseline = make_variant("baseline", {"hardhat": 0.9, "safety_vest": 0.2})
        out = _save_annotated(sample_image, baseline, Path("worker.jpg"), PROD_CONF)
        assert out.exists()
        assert out.name == "worker_probe.jpg"

    def test_handles_zero_detections(
        self, sample_image: np.ndarray, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("src.evaluation.probe.PROBE_OUTPUT_DIR", tmp_path / "probe")
        out = _save_annotated(sample_image, make_variant("baseline", {}), Path("x.jpg"), PROD_CONF)
        assert out.exists()


class TestReportRendering:
    """The print helpers must not blow up on edge-case inputs."""

    def test_baseline_table_with_detections(self) -> None:
        _print_baseline_table(make_variant("baseline", {"hardhat": 0.9}), PROD_CONF)

    def test_baseline_table_with_no_detections(self) -> None:
        _print_baseline_table(make_variant("baseline", {}), PROD_CONF)

    def test_variant_table_with_no_detections_anywhere(self) -> None:
        _print_variant_table([make_variant("baseline", {})], PROD_CONF)

    def test_variant_table_mixes_present_and_absent_classes(self) -> None:
        variants = [
            make_variant("baseline", {"hardhat": 0.9}),
            make_variant("imgsz 1280", {"hardhat": 0.8, "safety_vest": 0.3}),
        ]
        _print_variant_table(variants, PROD_CONF)

    def test_job_report_returns_one_diagnosis_per_required_class(self) -> None:
        baseline = make_variant("baseline", {"hardhat": 0.9})
        diagnoses = _print_job_report("general_labor", baseline, [baseline], PROD_CONF)
        assert {d.class_name for d in diagnoses} == {"hardhat", "safety_vest", "boots"}
        assert all(isinstance(d, ClassDiagnosis) for d in diagnoses)

    def test_job_report_flags_missing_vest_as_data_gap(self) -> None:
        baseline = make_variant("baseline", {"hardhat": 0.9})
        diagnoses = _print_job_report("general_labor", baseline, [baseline], PROD_CONF)
        vest = next(d for d in diagnoses if d.class_name == "safety_vest")
        assert vest.diagnosis == DIAG_DATA_GAP

    def test_job_report_all_present_is_clean(self) -> None:
        baseline = make_variant(
            "baseline", {"hardhat": 0.9, "safety_vest": 0.8, "boots": 0.7}
        )
        diagnoses = _print_job_report("general_labor", baseline, [baseline], PROD_CONF)
        assert all(d.diagnosis == DIAG_DETECTED for d in diagnoses)

    def test_unknown_job_raises_keyerror(self) -> None:
        baseline = make_variant("baseline", {"hardhat": 0.9})
        with pytest.raises(KeyError, match="astronaut"):
            _print_job_report("astronaut", baseline, [baseline], PROD_CONF)


class TestUndersizedWarning:
    def test_warns_when_long_edge_below_model_input(self) -> None:
        """The photo that started this: 183x275 is far too small to judge."""
        assert _warn_if_undersized(183, 275) is True

    def test_silent_when_image_is_large_enough(self) -> None:
        assert _warn_if_undersized(1920, 1080) is False

    def test_boundary_exactly_at_model_input_is_silent(self) -> None:
        assert _warn_if_undersized(640, 480) is False

    def test_uses_long_edge_not_area(self) -> None:
        """A tall narrow image is fine if its long edge clears the input size."""
        assert _warn_if_undersized(200, 900) is False


class TestConclusion:
    """The conclusion must never recommend a threshold inside the noise band."""

    def test_noise_conclusion_warns_against_lowering_threshold(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        diag = ClassDiagnosis("safety_vest", DIAG_NOISE, 0.041, 0.041, "baseline")
        _print_conclusion([diag], PROD_CONF)
        out = capsys.readouterr().out
        assert "Do not lower the threshold" in out

    def test_threshold_suggestion_never_dips_into_noise(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        diag = ClassDiagnosis("safety_vest", DIAG_THRESHOLD, 0.16, 0.16, "baseline")
        _print_conclusion([diag], PROD_CONF)
        out = capsys.readouterr().out.replace("\n", " ")
        assert f"~{NOISE_FLOOR:.2f}" in out

    def test_clean_result_says_so(self, capsys: pytest.CaptureFixture[str]) -> None:
        diag = ClassDiagnosis("hardhat", DIAG_DETECTED, 0.9, 0.9, "baseline")
        _print_conclusion([diag], PROD_CONF)
        assert "All required PPE detected" in capsys.readouterr().out


class TestDefaultVariants:
    def test_baseline_is_first_and_uses_production_imgsz(self) -> None:
        """The first variant must mirror production, since _diagnose treats it so."""
        label, imgsz, augment = DEFAULT_VARIANTS[0]
        assert label == "baseline"
        assert imgsz is None, "baseline must inherit PPB_IMG_SIZE, not hardcode one"
        assert augment is False

    def test_sweep_covers_larger_sizes_and_tta(self) -> None:
        assert any(imgsz and imgsz > 640 for _, imgsz, _ in DEFAULT_VARIANTS)
        assert any(augment for *_, augment in DEFAULT_VARIANTS)

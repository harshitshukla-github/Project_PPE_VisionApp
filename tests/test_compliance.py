"""Unit tests for src/inference/compliance.py — the verdict logic."""

from __future__ import annotations

import pytest

from src.inference.compliance import (
    VIOLATION_TO_ITEM,
    check_compliance,
    load_job_matrix,
)
from src.inference.schemas import Verdict
from tests.conftest import make_detection


class TestJobMatrix:
    def test_loads_all_expected_jobs(self) -> None:
        matrix = load_job_matrix()
        assert {"general_labor", "welder", "electrician", "height_work"} <= set(matrix)

    def test_every_required_class_is_detectable(self) -> None:
        """Required lists must never contain classes with no training data."""
        undetectable = {"boots", "ear_protection", "face_shield", "harness"}
        for job, req in load_job_matrix().items():
            overlap = undetectable & set(req.required)
            assert not overlap, f"{job} requires undetectable classes: {overlap}"

    def test_unknown_job_raises_keyerror(self, compliant_detections) -> None:
        with pytest.raises(KeyError, match="astronaut"):
            check_compliance(compliant_detections, "astronaut")


class TestVerdicts:
    def test_fully_equipped_worker_passes(self, compliant_detections) -> None:
        result = check_compliance(compliant_detections, "general_labor")
        assert result.verdict is Verdict.PASS
        assert result.missing_equipment == []
        assert result.violations_detected == []

    def test_missing_vest_fails(self, missing_vest_detections) -> None:
        result = check_compliance(missing_vest_detections, "general_labor")
        assert result.verdict is Verdict.FAIL
        assert result.missing_equipment == ["safety_vest"]

    def test_no_detections_fails_with_all_missing(self) -> None:
        result = check_compliance([], "general_labor")
        assert result.verdict is Verdict.FAIL
        assert set(result.missing_equipment) == {"hardhat", "safety_vest"}
        assert result.confidence_avg == 0.0

    def test_violation_fails_even_when_item_also_detected(
        self, violation_detections
    ) -> None:
        """no_hardhat in frame ⇒ FAIL, even though a hardhat is also visible.

        Multi-person scenario: one compliant worker must not mask another
        worker's violation.
        """
        result = check_compliance(violation_detections, "general_labor")
        assert result.verdict is Verdict.FAIL
        assert result.violations_detected == ["no_hardhat"]
        assert result.missing_equipment == []  # nothing missing — a violation

    def test_violation_irrelevant_to_job_is_ignored(self) -> None:
        """A no_vest box doesn't fail a job that doesn't require a vest."""
        # Hypothetical job list: none of ours lack safety_vest, so build the
        # scenario against the matrix contract instead.
        for job, req in load_job_matrix().items():
            if "safety_vest" not in req.required:
                detections = [make_detection(c) for c in req.required]
                detections.append(make_detection("no_vest"))
                result = check_compliance(detections, job)
                assert result.verdict is Verdict.PASS
                return
        pytest.skip("All current jobs require safety_vest — scenario not constructible.")

    def test_pending_classes_reported_unverifiable(self, compliant_detections) -> None:
        result = check_compliance(compliant_detections, "general_labor")
        assert "boots" in result.unverifiable_equipment

    def test_confidence_avg_is_mean(self) -> None:
        detections = [make_detection("hardhat", 0.8), make_detection("safety_vest", 0.6)]
        result = check_compliance(detections, "general_labor")
        assert result.confidence_avg == pytest.approx(0.7)


class TestViolationMapping:
    def test_violation_map_targets_exist_in_matrix_classes(self) -> None:
        assert VIOLATION_TO_ITEM == {
            "no_hardhat": "hardhat",
            "no_vest": "safety_vest",
        }

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

    def test_undetected_vest_is_review_not_failure(self, missing_vest_detections) -> None:
        """The defect this exists to prevent.

        A worker wearing a vest the model missed used to be told they were
        non-compliant. safety_vest AP50 is 0.861, so ~1 in 7 vests is missed;
        absence alone must never produce an accusation.
        """
        result = check_compliance(missing_vest_detections, "general_labor")
        assert result.verdict is Verdict.NEEDS_REVIEW
        assert result.missing_equipment == ["safety_vest"]
        assert result.violations_detected == []

    def test_no_detections_is_review_not_failure(self) -> None:
        """An empty frame is unreadable, not a violation — nobody to accuse."""
        result = check_compliance([], "general_labor")
        assert result.verdict is Verdict.NEEDS_REVIEW
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


class TestThreeStateBoundary:
    """FAIL needs affirmative evidence; absence alone only warrants review."""

    def test_violation_outranks_absence(self) -> None:
        """Both conditions at once: a positive violation decides it.

        no_hardhat is detected (affirmative) while safety_vest is merely
        absent. Real evidence of one breach outweighs uncertainty about
        another, so this is FAIL, not NEEDS_REVIEW.
        """
        result = check_compliance([make_detection("no_hardhat")], "general_labor")
        assert result.verdict is Verdict.FAIL
        assert result.violations_detected == ["no_hardhat"]
        assert "safety_vest" in result.missing_equipment

    def test_detected_violation_always_fails(self) -> None:
        """no_vest present alongside an otherwise complete kit ⇒ FAIL."""
        dets = [make_detection("hardhat"), make_detection("safety_vest"),
                make_detection("no_vest")]
        assert check_compliance(dets, "general_labor").verdict is Verdict.FAIL

    def test_partial_kit_without_violation_is_review(self) -> None:
        """Welder with half the kit and no violation box ⇒ review, not fail."""
        dets = [make_detection("hardhat"), make_detection("safety_vest")]
        result = check_compliance(dets, "welder")
        assert result.verdict is Verdict.NEEDS_REVIEW
        assert set(result.missing_equipment) == {"goggles", "gloves"}

    def test_verdict_values_are_the_three_documented_strings(self) -> None:
        """The mobile app switches on these literals — they are a contract."""
        assert {v.value for v in Verdict} == {"PASS", "FAIL", "NEEDS_REVIEW"}

    @pytest.mark.parametrize("job", sorted(load_job_matrix()))
    def test_every_job_reachable_without_false_accusation(self, job: str) -> None:
        """No job may produce FAIL from absence alone, whatever its kit."""
        result = check_compliance([], job)
        assert result.verdict is Verdict.NEEDS_REVIEW, (
            f"job '{job}' accuses on an empty frame"
        )

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

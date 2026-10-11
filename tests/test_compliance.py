"""Unit tests for src/inference/compliance.py — the verdict logic."""

from __future__ import annotations

import pytest

from src.inference.compliance import (
    VIOLATION_TO_ITEM,
    apply_verification,
    check_compliance,
    load_job_matrix,
)
from src.inference.schemas import GeminiVerification, ItemStatus, ItemVerification, Verdict
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


def _verification(**statuses: tuple[str, float]) -> GeminiVerification:
    """Build Gemini findings from item=(status, confidence) pairs."""
    return GeminiVerification(
        model="gemini-test",
        elapsed_ms=1.0,
        items=[
            ItemVerification(item=item, status=ItemStatus(status), confidence=conf)
            for item, (status, conf) in statuses.items()
        ],
    )


class TestGeminiResolution:
    """How Gemini's look at the photo settles a NEEDS_REVIEW."""

    def _review(self, job: str = "general_labor", detected: tuple[str, ...] = ("hardhat",)):
        result = check_compliance([make_detection(c) for c in detected], job)
        assert result.verdict is Verdict.NEEDS_REVIEW
        return result

    def test_vest_confirmed_worn_becomes_pass(self) -> None:
        """The original bug report, fully resolved: detector missed it, Gemini saw it."""
        resolved = apply_verification(
            self._review(), _verification(safety_vest=("worn", 0.95)), 0.75
        )
        assert resolved.verdict is Verdict.PASS
        assert resolved.missing_equipment == []
        assert resolved.verdict_source == "detector+gemini"

    def test_vest_confirmed_absent_becomes_fail(self) -> None:
        resolved = apply_verification(
            self._review(), _verification(safety_vest=("not_worn", 0.9)), 0.75
        )
        assert resolved.verdict is Verdict.FAIL
        assert resolved.missing_equipment == ["safety_vest"]
        assert resolved.verdict_source == "detector+gemini"

    def test_uncertain_stays_review(self) -> None:
        resolved = apply_verification(
            self._review(), _verification(safety_vest=("uncertain", 0.9)), 0.75
        )
        assert resolved.verdict is Verdict.NEEDS_REVIEW
        assert resolved.missing_equipment == ["safety_vest"]
        assert resolved.verdict_source == "detector"

    @pytest.mark.parametrize("status", ["worn", "not_worn"])
    def test_low_confidence_answer_changes_nothing(self, status: str) -> None:
        """A hesitant Gemini can neither clear a worker nor accuse one."""
        resolved = apply_verification(
            self._review(), _verification(safety_vest=(status, 0.6)), 0.75
        )
        assert resolved.verdict is Verdict.NEEDS_REVIEW

    def test_confidence_exactly_at_threshold_counts(self) -> None:
        resolved = apply_verification(
            self._review(), _verification(safety_vest=("worn", 0.75)), 0.75
        )
        assert resolved.verdict is Verdict.PASS

    def test_item_gemini_did_not_report_stays_unknown(self) -> None:
        review = self._review("welder", ("hardhat", "safety_vest"))
        resolved = apply_verification(review, _verification(goggles=("worn", 0.9)), 0.75)
        assert resolved.verdict is Verdict.NEEDS_REVIEW
        assert resolved.missing_equipment == ["gloves"]
        assert resolved.verdict_source == "detector+gemini"  # the list did shrink

    def test_one_not_worn_fails_and_unknowns_are_not_blamed(self) -> None:
        """Only what was seen to be missing is reported as missing."""
        review = self._review("welder", ("hardhat", "safety_vest"))
        resolved = apply_verification(
            review,
            _verification(goggles=("not_worn", 0.9), gloves=("uncertain", 0.5)),
            0.75,
        )
        assert resolved.verdict is Verdict.FAIL
        assert resolved.missing_equipment == ["goggles"]
        assert "gloves" in resolved.unverifiable_equipment

    def test_all_items_worn_passes_multi_item_job(self) -> None:
        review = self._review("welder", ("hardhat", "safety_vest"))
        resolved = apply_verification(
            review, _verification(goggles=("worn", 0.9), gloves=("worn", 0.85)), 0.75
        )
        assert resolved.verdict is Verdict.PASS

    def test_no_verification_leaves_result_untouched(self) -> None:
        review = self._review()
        assert apply_verification(review, None) is review

    def test_detector_fail_is_never_overturned(self) -> None:
        """Gemini is a second opinion on doubt, not an appeal against a violation box."""
        fail = check_compliance(
            [make_detection("hardhat"), make_detection("no_vest")], "general_labor"
        )
        assert fail.verdict is Verdict.FAIL
        assert apply_verification(fail, _verification(safety_vest=("worn", 1.0)), 0.75) is fail

    def test_detector_pass_is_never_overturned(self, compliant_detections) -> None:
        ok = check_compliance(compliant_detections, "general_labor")
        assert apply_verification(ok, _verification(safety_vest=("not_worn", 1.0)), 0.75) is ok

    def test_original_result_is_not_mutated(self) -> None:
        review = self._review()
        apply_verification(review, _verification(safety_vest=("worn", 0.95)), 0.75)
        assert review.verdict is Verdict.NEEDS_REVIEW
        assert review.missing_equipment == ["safety_vest"]

"""Tests for class-name matching in src/data/remapper.py.

An exact-match miss here is silent but destructive: the class is discarded and
every image that contained it becomes a background image, teaching the model
the object is *not* present. That is strictly worse than never adding the
dataset. The Kaggle helmet+vest set hit exactly this — its classes are spelled
``Safety-Helmet`` / ``Reflective-Jacket`` and would have lost all 36,240
annotations.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data import remapper
from src.data.remapper import _build_id_map, _load_class_maps, _normalize_class_name


class TestNormalizeClassName:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Safety-Helmet", "safety helmet"),
            ("safety_helmet", "safety helmet"),
            ("Safety   Helmet", "safety helmet"),
            ("  SAFETY-HELMET  ", "safety helmet"),
            ("Reflective-Jacket", "reflective jacket"),
            ("hardhat", "hardhat"),
            ("NO-Vest", "no vest"),
            ("a--b__c  d", "a b c d"),
        ],
    )
    def test_normalisation(self, raw: str, expected: str) -> None:
        assert _normalize_class_name(raw) == expected

    def test_non_string_input_is_coerced(self) -> None:
        assert _normalize_class_name(0) == "0"


class TestBuildIdMap:
    def test_exact_match_takes_precedence(self) -> None:
        src = {0: "helmet"}
        assert _build_id_map(src, {"helmet": 0, "Helmet": 9}) == {0: 0}

    def test_normalised_fallback_rescues_hyphenated_name(self) -> None:
        src = {0: "Reflective-Jacket"}
        assert _build_id_map(src, {"reflective jacket": 2}) == {0: 2}

    def test_unmapped_class_is_discarded(self) -> None:
        src = {0: "unicorn"}
        assert _build_id_map(src, {"helmet": 0}) == {0: None}

    def test_explicit_none_target_is_preserved(self) -> None:
        """A class mapped to null means 'deliberately drop', not 'unmapped'."""
        src = {0: "background"}
        assert _build_id_map(src, {"background": None}) == {0: None}

    def test_ambiguous_normalisation_keeps_first_mapping(self) -> None:
        src = {0: "Safety_Helmet"}
        result = _build_id_map(src, {"safety helmet": 0, "Safety-Helmet": 7})
        assert result == {0: 0}

    def test_the_kaggle_regression_case(self) -> None:
        """Both real class names must land on hardhat (0) and safety_vest (2)."""
        src = {0: "Safety-Helmet", 1: "Reflective-Jacket"}
        assert _build_id_map(src, _load_class_maps()["kaggle_ppe"]) == {0: 0, 1: 2}

    def test_no_class_silently_lost_for_kaggle(self) -> None:
        src = {0: "Safety-Helmet", 1: "Reflective-Jacket"}
        id_map = _build_id_map(src, _load_class_maps()["kaggle_ppe"])
        assert all(v is not None for v in id_map.values())


class TestRemapAllResilience:
    """A half-downloaded or broken dataset must not abort the whole batch."""

    def test_skips_dataset_with_no_data_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog
    ) -> None:
        raw = tmp_path / "raw"
        (raw / "ppe_combined").mkdir(parents=True)
        # Mid-download: the dir is non-empty but holds only the archive.
        (raw / "ppe_combined" / "roboflow.zip").write_bytes(b"partial")
        monkeypatch.setattr(remapper.settings, "data_raw_dir", raw)
        monkeypatch.setattr(remapper.settings, "data_processed_dir", tmp_path / "processed")

        remapper.remap_all(dataset_keys=["ppe_combined"])
        assert not (tmp_path / "processed").exists()

    def test_one_failure_does_not_abort_the_rest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        raw = tmp_path / "raw"
        for key in ("good", "bad"):
            d = raw / key
            d.mkdir(parents=True)
            (d / "data.yaml").write_text("names: {0: helmet}\n")
        monkeypatch.setattr(remapper.settings, "data_raw_dir", raw)

        calls: list[str] = []

        def fake_remap(key: str, class_maps: dict) -> dict:
            calls.append(key)
            if key == "bad":
                raise RuntimeError("corrupt labels")
            return {"images": 1, "kept": 1, "discarded": 0, "background": 0}

        monkeypatch.setattr(remapper, "remap_dataset", fake_remap)
        monkeypatch.setattr(remapper, "_load_class_maps", lambda: {"good": {}, "bad": {}})

        remapper.remap_all(dataset_keys=["bad", "good"])
        assert calls == ["bad", "good"], "must continue past the failing dataset"


class TestPolygonNormalisation:
    """Some sources ship segmentation polygons mixed into box label files.

    Ultralytics decides a file is segmentation if ANY row exceeds 6 fields,
    then reinterprets that file's 5-field box rows as 2-point polygons —
    silently producing garbage boxes. 28 such mixed files were found in the
    split, so polygons are collapsed to boxes to keep every row 5-field.
    """

    def test_square_polygon_becomes_its_bounding_box(self) -> None:
        from src.data.remapper import polygon_to_bbox

        # Unit square from (0.2,0.2) to (0.6,0.6)
        box = polygon_to_bbox([0.2, 0.2, 0.6, 0.2, 0.6, 0.6, 0.2, 0.6])
        assert box == pytest.approx([0.4, 0.4, 0.4, 0.4])

    def test_irregular_polygon_uses_extremes(self) -> None:
        from src.data.remapper import polygon_to_bbox

        cx, cy, w, h = polygon_to_bbox([0.1, 0.5, 0.9, 0.2, 0.4, 0.8])
        assert (cx, cy) == pytest.approx((0.5, 0.5))
        assert (w, h) == pytest.approx((0.8, 0.6))

    def test_coordinates_are_clamped_to_unit_range(self) -> None:
        from src.data.remapper import polygon_to_bbox

        cx, cy, w, h = polygon_to_bbox([-0.2, -0.1, 1.4, 0.5, 0.5, 1.9])
        assert 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0
        assert w <= 1.0 and h <= 1.0

    def test_box_rows_pass_through_untouched(self) -> None:
        from src.data.remapper import normalise_label_line

        row = ["2", "0.5", "0.5", "0.1", "0.2"]
        assert normalise_label_line(row) == row

    def test_polygon_row_is_collapsed_to_five_fields(self) -> None:
        from src.data.remapper import normalise_label_line

        row = "6 0.2 0.2 0.6 0.2 0.6 0.6 0.2 0.6".split()
        out = normalise_label_line(row)
        assert out is not None
        assert len(out) == 5
        assert out[0] == "6"

    @pytest.mark.parametrize(
        "row",
        [
            "6 0.1 0.2".split(),          # too few fields
            "6 0.1 0.2 0.3 0.4 0.5".split(),  # even count, not a valid polygon
            "6 a b c d e f".split(),      # non-numeric
            [],                            # empty
        ],
    )
    def test_unrepairable_rows_return_none(self, row: list[str]) -> None:
        from src.data.remapper import normalise_label_line

        assert normalise_label_line(row) is None

    def test_realistic_mixed_file_row_from_the_dataset(self) -> None:
        """An actual failing row from the validator output."""
        from src.data.remapper import normalise_label_line

        row = (
            "6 0.5466666671875 0.01625 0.0916666671875 0.11125 0.27 0.93125 "
            "0.9333333328125001 0.9012499999999999 0.5466666671875 0.01625"
        ).split()
        out = normalise_label_line(row)
        assert out is not None and len(out) == 5
        assert all(0.0 <= float(v) <= 1.0 for v in out[1:])

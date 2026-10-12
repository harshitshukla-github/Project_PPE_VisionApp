"""Tests for provenance tracking and split safety in src/data/splitter.py.

The manifest exists so per-source loss masking is possible later: the merged
datasets annotate different class vocabularies, and an image from a
helmet-only source must not be treated as negative evidence for vests. Two
things would quietly destroy that guarantee — filename collisions between
sources, and splitting on top of a previous run — so both are covered here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data import splitter
from src.data.splitter import (
    MANIFEST_NAME,
    ImagePair,
    _resolve_destination_names,
    collect_pairs,
    load_manifest,
    split_and_copy,
)


def make_processed(root: Path, layout: dict[str, list[str]]) -> Path:
    """Build a fake data/processed/ tree: {source: [filenames]}."""
    for source, names in layout.items():
        img_dir = root / source / "train" / "images"
        lbl_dir = root / source / "train" / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            (img_dir / name).write_bytes(b"\xff\xd8\xff")
            (lbl_dir / f"{Path(name).stem}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    return root


class TestCollectPairs:
    def test_tags_each_pair_with_its_source(self, tmp_path: Path) -> None:
        root = make_processed(tmp_path, {"kaggle_ppe": ["a.jpg"], "ppe_combined": ["b.jpg"]})
        pairs = collect_pairs(root)
        assert {p.source for p in pairs} == {"kaggle_ppe", "ppe_combined"}

    def test_skips_images_without_labels(self, tmp_path: Path) -> None:
        root = make_processed(tmp_path, {"src": ["a.jpg"]})
        (root / "src" / "train" / "images" / "orphan.jpg").write_bytes(b"\xff\xd8\xff")
        assert len(collect_pairs(root)) == 1

    def test_ignores_non_image_files(self, tmp_path: Path) -> None:
        root = make_processed(tmp_path, {"src": ["a.jpg"]})
        (root / "src" / "train" / "images" / "notes.txt").write_text("x")
        assert len(collect_pairs(root)) == 1


class TestCollisionHandling:
    def test_same_filename_from_two_sources_is_renamed(self) -> None:
        pairs = [
            ImagePair(Path("a/train/images/0001.jpg"), Path("a/train/labels/0001.txt"), "a"),
            ImagePair(Path("b/train/images/0001.jpg"), Path("b/train/labels/0001.txt"), "b"),
        ]
        resolved, renamed = _resolve_destination_names(pairs)
        names = [n for _, n in resolved]
        assert renamed == 1
        assert len(set(names)) == 2, "collision must not produce duplicate names"
        assert "b__0001.jpg" in names

    def test_unique_filenames_are_left_alone(self) -> None:
        pairs = [
            ImagePair(Path("a/train/images/x.jpg"), Path("a/train/labels/x.txt"), "a"),
            ImagePair(Path("b/train/images/y.jpg"), Path("b/train/labels/y.txt"), "b"),
        ]
        resolved, renamed = _resolve_destination_names(pairs)
        assert renamed == 0
        assert [n for _, n in resolved] == ["x.jpg", "y.jpg"]

    def test_three_way_collision_all_unique(self) -> None:
        pairs = [
            ImagePair(Path(f"{s}/train/images/0001.jpg"), Path(f"{s}/train/labels/0001.txt"), s)
            for s in ("a", "b", "c")
        ]
        resolved, renamed = _resolve_destination_names(pairs)
        assert renamed == 2
        assert len({n for _, n in resolved}) == 3

    def test_no_image_is_lost_to_collision_end_to_end(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two sources sharing every filename must still yield all images."""
        proc = make_processed(
            tmp_path / "processed",
            {"src_a": ["0001.jpg", "0002.jpg"], "src_b": ["0001.jpg", "0002.jpg"]},
        )
        monkeypatch.setattr(splitter.settings, "data_processed_dir", proc)
        out = tmp_path / "splits"
        counts = split_and_copy(collect_pairs(proc), out, 0.5, 0.25, seed=1)
        assert sum(counts.values()) == 4
        on_disk = sum(
            len(list((out / s / "images").iterdir()))
            for s in ("train", "val", "test")
            if (out / s / "images").is_dir()
        )
        assert on_disk == 4, "an image was silently overwritten"


class TestManifest:
    def test_manifest_records_source_for_every_image(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        proc = make_processed(
            tmp_path / "processed", {"kaggle_ppe": ["a.jpg", "b.jpg"], "ppe_combined": ["c.jpg"]}
        )
        monkeypatch.setattr(splitter.settings, "data_processed_dir", proc)
        out = tmp_path / "splits"
        split_and_copy(collect_pairs(proc), out, 0.34, 0.33, seed=7)

        manifest = json.loads((out / MANIFEST_NAME).read_text(encoding="utf-8"))
        assert len(manifest["images"]) == 3
        assert manifest["sources"]["kaggle_ppe"]["total"] == 2
        assert manifest["sources"]["ppe_combined"]["total"] == 1
        assert all(
            set(v) == {"source", "source_path"} for v in manifest["images"].values()
        )

    def test_manifest_keys_are_split_qualified(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        proc = make_processed(tmp_path / "processed", {"s": ["a.jpg", "b.jpg", "c.jpg"]})
        monkeypatch.setattr(splitter.settings, "data_processed_dir", proc)
        out = tmp_path / "splits"
        split_and_copy(collect_pairs(proc), out, 0.34, 0.33, seed=3)
        manifest = json.loads((out / MANIFEST_NAME).read_text(encoding="utf-8"))
        assert all("/" in k for k in manifest["images"])
        assert {k.split("/")[0] for k in manifest["images"]} <= {"train", "val", "test"}

    def test_manifest_records_seed_and_ratios(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        proc = make_processed(tmp_path / "processed", {"s": ["a.jpg", "b.jpg"]})
        monkeypatch.setattr(splitter.settings, "data_processed_dir", proc)
        out = tmp_path / "splits"
        split_and_copy(collect_pairs(proc), out, 0.5, 0.25, seed=99)
        manifest = json.loads((out / MANIFEST_NAME).read_text(encoding="utf-8"))
        assert manifest["seed"] == 99
        assert manifest["ratios"]["train"] == 0.5

    def test_load_manifest_roundtrip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        proc = make_processed(tmp_path / "processed", {"s": ["a.jpg", "b.jpg"]})
        monkeypatch.setattr(splitter.settings, "data_processed_dir", proc)
        out = tmp_path / "splits"
        split_and_copy(collect_pairs(proc), out, 0.5, 0.25, seed=1)
        assert load_manifest(out)["sources"]["s"]["total"] == 2

    def test_load_manifest_missing_is_actionable(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="data split"):
            load_manifest(tmp_path)


class TestCleanGuard:
    def test_refuses_to_split_onto_existing_data(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        proc = make_processed(tmp_path / "processed", {"s": ["a.jpg", "b.jpg"]})
        monkeypatch.setattr(splitter.settings, "data_processed_dir", proc)
        out = tmp_path / "splits"
        (out / "train" / "images").mkdir(parents=True)
        (out / "train" / "images" / "stale.jpg").write_bytes(b"old")

        assert split_and_copy(collect_pairs(proc), out, 0.5, 0.25) == {}
        assert (out / "train" / "images" / "stale.jpg").exists(), "must not delete without --clean"

    def test_clean_wipes_then_splits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        proc = make_processed(tmp_path / "processed", {"s": ["a.jpg", "b.jpg"]})
        monkeypatch.setattr(splitter.settings, "data_processed_dir", proc)
        out = tmp_path / "splits"
        (out / "train" / "images").mkdir(parents=True)
        (out / "train" / "images" / "stale.jpg").write_bytes(b"old")

        counts = split_and_copy(collect_pairs(proc), out, 0.5, 0.25, clean=True)
        assert sum(counts.values()) == 2
        assert not (out / "train" / "images" / "stale.jpg").exists()

    def test_empty_output_needs_no_clean_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        proc = make_processed(tmp_path / "processed", {"s": ["a.jpg", "b.jpg"]})
        monkeypatch.setattr(splitter.settings, "data_processed_dir", proc)
        counts = split_and_copy(collect_pairs(proc), tmp_path / "splits", 0.5, 0.25)
        assert sum(counts.values()) == 2

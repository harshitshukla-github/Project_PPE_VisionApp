"""Tests for checkpoint promotion in src/training/trainer.py.

``models/<model>_ppe_best.pt`` is what the API serves. A one-epoch smoke run
(``--run-name smoke_yolo11n``) once overwrote a 50-epoch H100 model there,
because the copy ignored the run name. These tests pin down that custom runs
never claim the served slot, and that nothing is ever overwritten outright.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.training import trainer


@pytest.fixture
def models_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = tmp_path / "models"
    monkeypatch.setattr(trainer, "MODELS_DIR", d)
    return d


def make_run(tmp_path: Path, name: str, payload: bytes) -> Path:
    run = tmp_path / "runs" / name
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "best.pt").write_bytes(payload)
    return run


class TestPromotion:
    def test_default_run_is_promoted_to_served_slot(
        self, tmp_path: Path, models_dir: Path
    ) -> None:
        run = make_run(tmp_path, "yolo11l_ppe", b"new")
        out = trainer._save_best_weights(run, "yolo11l")
        assert out == models_dir / "yolo11l_ppe_best.pt"
        assert out.read_bytes() == b"new"

    def test_custom_run_name_never_touches_served_slot(
        self, tmp_path: Path, models_dir: Path
    ) -> None:
        """The exact incident: a smoke run must leave the served model alone."""
        models_dir.mkdir()
        served = models_dir / "yolo11n_ppe_best.pt"
        served.write_bytes(b"50-epoch H100 model")

        run = make_run(tmp_path, "smoke_yolo11n", b"1-epoch smoke model")
        out = trainer._save_best_weights(run, "yolo11n", run_name="smoke_yolo11n")

        assert out == models_dir / "smoke_yolo11n_best.pt"
        assert served.read_bytes() == b"50-epoch H100 model"

    def test_missing_best_pt_returns_none(self, tmp_path: Path, models_dir: Path) -> None:
        empty = tmp_path / "runs" / "crashed"
        empty.mkdir(parents=True)
        assert trainer._save_best_weights(empty, "yolo11l") is None


class TestNeverOverwrite:
    def test_existing_served_model_is_archived_not_destroyed(
        self, tmp_path: Path, models_dir: Path
    ) -> None:
        models_dir.mkdir()
        (models_dir / "yolo11l_ppe_best.pt").write_bytes(b"old production")

        run = make_run(tmp_path, "yolo11l_ppe", b"retrained")
        trainer._save_best_weights(run, "yolo11l")

        archived = list((models_dir / "archive").glob("yolo11l_ppe_best.*.pt"))
        assert len(archived) == 1
        assert archived[0].read_bytes() == b"old production"
        assert (models_dir / "yolo11l_ppe_best.pt").read_bytes() == b"retrained"

    def test_repeated_saves_keep_every_version(
        self, tmp_path: Path, models_dir: Path
    ) -> None:
        for i in range(3):
            run = make_run(tmp_path, f"yolo11l_ppe{i}", f"v{i}".encode())
            trainer._save_best_weights(run, "yolo11l")

        archived = sorted(p.read_bytes() for p in (models_dir / "archive").iterdir())
        assert archived == [b"v0", b"v1"]
        assert (models_dir / "yolo11l_ppe_best.pt").read_bytes() == b"v2"

    def test_nothing_to_archive_on_first_save(
        self, tmp_path: Path, models_dir: Path
    ) -> None:
        run = make_run(tmp_path, "yolo11l_ppe", b"first")
        trainer._save_best_weights(run, "yolo11l")
        assert not (models_dir / "archive").exists()

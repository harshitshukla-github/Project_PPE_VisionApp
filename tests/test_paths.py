"""The project's folders are found from the project root, not the working directory.

Under a service manager or a Docker WORKDIR the server starts somewhere else,
and it must still find its ``.env``, its weights and its job matrix.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import ModuleType

import pytest

import cli
from src.evaluation import benchmark, probe
from src.export import onnx_export
from src.inference import compliance, detector
from src.paths import PROJECT_ROOT, from_project_root
from src.settings import Settings, settings
from src.training import trainer

# A value no real .env would hold, so a test can tell which file was read.
DECOY_IOU = 0.123


@pytest.fixture
def elsewhere(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Started from a folder that is not the project, with a .env of its own lying there."""
    folder = tmp_path / "elsewhere"
    folder.mkdir()
    (folder / ".env").write_text(f"PPB_IOU_THRESHOLD={DECOY_IOU}\n", encoding="utf-8")
    monkeypatch.delenv("PPB_IOU_THRESHOLD", raising=False)
    monkeypatch.chdir(folder)
    return folder


class TestProjectRoot:
    def test_it_really_is_the_project(self) -> None:
        assert PROJECT_ROOT.is_absolute()
        assert (PROJECT_ROOT / "pyproject.toml").is_file()
        assert (PROJECT_ROOT / "cli.py").is_file()

    def test_a_relative_path_is_inside_it(self) -> None:
        assert from_project_root("configs") == PROJECT_ROOT / "configs"
        assert from_project_root(Path("admin/dist")) == PROJECT_ROOT / "admin" / "dist"

    def test_an_absolute_path_is_kept(self, tmp_path: Path) -> None:
        assert from_project_root(tmp_path / "builds") == tmp_path / "builds"


class TestSettings:
    def test_the_env_file_is_the_projects_own(self, elsewhere: Path) -> None:
        assert Path(Settings.model_config["env_file"]) == PROJECT_ROOT / ".env"
        assert Settings().ppb_iou_threshold != DECOY_IOU  # not the one in the start folder

    def test_every_path_setting_is_absolute(self, elsewhere: Path) -> None:
        fresh = Settings()
        paths = {name: getattr(fresh, name) for name in Settings.model_fields
                 if isinstance(getattr(fresh, name), Path)}
        assert {"configs_dir", "data_splits_dir", "ppb_debug_dump_dir", "ppb_admin_dist"} <= set(paths)
        assert [name for name, path in paths.items() if not path.is_absolute()] == []

    def test_the_default_folders_are_inside_the_project(self) -> None:
        assert settings.configs_dir == PROJECT_ROOT / "configs"
        assert settings.data_splits_dir == PROJECT_ROOT / "data" / "splits"

    def test_a_relative_value_counts_from_the_project(
        self, elsewhere: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PPB_ADMIN_DIST", "builds/dashboard")
        assert Settings().ppb_admin_dist == PROJECT_ROOT / "builds" / "dashboard"

    def test_an_absolute_value_is_kept(
        self, elsewhere: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PPB_DEBUG_DUMP_DIR", str(tmp_path / "uploads"))
        assert Settings().ppb_debug_dump_dir == tmp_path / "uploads"


class TestModelsFolder:
    @pytest.mark.parametrize("module", [detector, trainer, onnx_export, benchmark, probe, cli],
                             ids=lambda m: m.__name__)
    def test_every_module_means_the_same_folder(self, module: ModuleType) -> None:
        assert module.MODELS_DIR == PROJECT_ROOT / "models"

    def test_the_detector_looks_inside_the_project(self, elsewhere: Path) -> None:
        with pytest.raises(FileNotFoundError) as missing:
            detector.PPEDetector(model_name="no_such_model")
        assert str(PROJECT_ROOT / "models" / "no_such_model_ppe_best.pt") in str(missing.value)


class TestFromAnotherFolder:
    def test_the_job_matrix_is_found(self, elsewhere: Path) -> None:
        assert "welder" in compliance.load_job_matrix()

    def test_db_setup_writes_the_env_file_the_settings_read(self) -> None:
        default = inspect.signature(cli.db_setup_cmd).parameters["env_file"].default.default
        assert default == Path(Settings.model_config["env_file"])

    def test_db_seed_reads_the_projects_matrix(self) -> None:
        default = inspect.signature(cli.db_seed_cmd).parameters["matrix_file"].default.default
        assert default == PROJECT_ROOT / "configs" / "job_matrix.yaml"
        assert default.is_file()

"""Tests for MLflow callback ownership in src/training/callbacks.py.

Ultralytics auto-registers its own MLflow integration whenever mlflow is
installed. Its ``on_pretrain_routine_end`` opens a run before our
``on_train_start``, which crashed training with "Run ... is already active".
It also read the tracking URI from ``os.environ`` instead of ``settings``,
which would have split one experiment across two MLflow stores.
"""

from __future__ import annotations

import importlib

import pytest

pytest.importorskip("mlflow")
pytest.importorskip("ultralytics")

import ultralytics.utils.callbacks.mlflow as umf  # noqa: E402
from ultralytics.utils import SETTINGS  # noqa: E402
from ultralytics.utils import callbacks as ucb  # noqa: E402

from src.training.callbacks import (  # noqa: E402
    _ULTRALYTICS_MLFLOW_MODULE,
    build_mlflow_callbacks,
)
from src.training.hyperparams import TrainingConfig  # noqa: E402


class FakeDetectionTrainer:
    """Stand-in trainer.

    The class name must contain "Trainer": ultralytics only attaches
    integration callbacks (MLflow included) to such instances
    (``callbacks/base.py``: ``if "Trainer" in instance.__class__.__name__``).
    """

    def __init__(self) -> None:
        self.callbacks = ucb.get_default_callbacks()


@pytest.fixture(autouse=True)
def ultralytics_mlflow_enabled(monkeypatch: pytest.MonkeyPatch):
    """Force ultralytics' MLflow integration on for these tests.

    Under pytest, ultralytics disables it at import time unless the running
    test's name contains "test_mlflow" (see the ``TESTS_RUNNING`` assert in
    ``ultralytics/utils/callbacks/mlflow.py``). Without this, the integration
    registers nothing, and every strip test passes vacuously. The module is
    reloaded with the condition satisfied, then reloaded again on teardown so
    later tests in the session see the default (disabled) state.
    """
    if SETTINGS.get("mlflow") is not True:
        pytest.skip("ultralytics MLflow integration disabled in SETTINGS on this machine")

    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_mlflow (forced integration check)")
    importlib.reload(umf)
    yield
    monkeypatch.undo()
    importlib.reload(umf)


def _registered_trainer() -> FakeDetectionTrainer:
    """Reproduce the real registration order: ours, then integrations."""
    trainer = FakeDetectionTrainer()
    for event, fn in build_mlflow_callbacks("yolo11l", TrainingConfig()).items():
        trainer.callbacks.setdefault(event, []).append(fn)
    ucb.add_integration_callbacks(trainer)
    return trainer


def _ultralytics_mlflow(trainer: FakeDetectionTrainer) -> list[str]:
    return sorted(
        event
        for event, fns in trainer.callbacks.items()
        for fn in fns
        if getattr(fn, "__module__", "") == _ULTRALYTICS_MLFLOW_MODULE
    )


class TestBuiltinMlflowIsDisabled:
    def test_precondition_ultralytics_really_registers_mlflow(self) -> None:
        """Guards the test itself: if this fails, the strip test proves nothing."""
        assert "on_pretrain_routine_end" in _ultralytics_mlflow(_registered_trainer())

    def test_pretrain_start_removes_every_builtin_mlflow_callback(self) -> None:
        trainer = _registered_trainer()
        for fn in list(trainer.callbacks["on_pretrain_routine_start"]):
            fn(trainer)
        assert _ultralytics_mlflow(trainer) == []

    def test_our_callbacks_survive_the_strip(self) -> None:
        trainer = _registered_trainer()
        for fn in list(trainer.callbacks["on_pretrain_routine_start"]):
            fn(trainer)
        ours = {
            event
            for event, fns in trainer.callbacks.items()
            for fn in fns
            if fn.__module__ == "src.training.callbacks"
        }
        assert ours == {
            "on_pretrain_routine_start",
            "on_train_start",
            "on_fit_epoch_end",
            "on_train_end",
        }

    def test_module_name_matches_real_ultralytics_module(self) -> None:
        """If ultralytics moves the module, the strip would silently no-op."""
        assert umf.__name__ == _ULTRALYTICS_MLFLOW_MODULE

"""Ultralytics training callbacks for the PPB Vision Module.

Provides two sets of callbacks:

* **Loguru callbacks** — structured per-epoch progress logging via loguru.
  Always safe to register; no external services required.

* **MLflow callbacks** — logs hyperparams, per-epoch metrics, and the best
  model artifact to an MLflow tracking server. Requires ``mlflow`` installed
  and ``MLFLOW_TRACKING_URI`` configured (via .env).

Usage::

    from src.training.callbacks import build_loguru_callbacks, build_mlflow_callbacks

    for event, fn in build_loguru_callbacks().items():
        model.add_callback(event, fn)

    for event, fn in build_mlflow_callbacks(model_name="yolo26l", cfg=my_cfg).items():
        model.add_callback(event, fn)

Ultralytics callback events (in call order during training):
    on_train_start → on_train_epoch_start → on_train_batch_end →
    on_train_epoch_end → on_fit_epoch_end → on_val_end → on_train_end
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Callable

from loguru import logger

from src.settings import settings

if TYPE_CHECKING:
    # ultralytics BaseTrainer — available at runtime, imported lazily to avoid
    # loading ultralytics at import time (keeps startup fast).
    from ultralytics.engine.trainer import BaseTrainer

    from src.training.hyperparams import TrainingConfig

# Type alias for a single callback function
_Callback = Callable[["BaseTrainer"], None]

#: Module that holds ultralytics' built-in MLflow callbacks. Matched by module
#: name so they can be stripped without importing (and thereby enabling) them.
_ULTRALYTICS_MLFLOW_MODULE = "ultralytics.utils.callbacks.mlflow"


# ---------------------------------------------------------------------------
# Loguru callbacks
# ---------------------------------------------------------------------------

def _on_train_start(trainer: "BaseTrainer") -> None:
    """Log training kick-off details."""
    args = trainer.args
    logger.info("=" * 60)
    logger.info("PPB Training — START")
    logger.info(f"  Model    : {args.model}")
    logger.info(f"  Data     : {args.data}")
    logger.info(f"  Epochs   : {args.epochs}")
    logger.info(f"  Img size : {args.imgsz}")
    logger.info(f"  Batch    : {args.batch}")
    logger.info(f"  Device   : {args.device}")
    logger.info(f"  Output   : {trainer.save_dir}")
    logger.info("=" * 60)


def _on_fit_epoch_end(trainer: "BaseTrainer") -> None:
    """Log per-epoch metrics after validation."""
    epoch = trainer.epoch + 1  # trainer.epoch is 0-indexed
    total = trainer.epochs

    metrics: dict = getattr(trainer, "metrics", {}) or {}
    fitness: float = getattr(trainer, "fitness", 0.0) or 0.0

    # Key metrics (present after validation)
    map50 = metrics.get("metrics/mAP50(B)", None)
    map50_95 = metrics.get("metrics/mAP50-95(B)", None)
    precision = metrics.get("metrics/precision(B)", None)
    recall = metrics.get("metrics/recall(B)", None)

    # ultralytics' final_eval() bumps trainer.epoch past the last epoch and
    # re-fires this event with best.pt's metrics — label it as such rather
    # than printing a nonsensical "Epoch 2/1".
    label = "Final (best.pt)" if epoch > total else f"Epoch {epoch}/{total}"
    parts: list[str] = [label]
    if map50 is not None:
        parts.append(f"mAP50={map50:.4f}")
    if map50_95 is not None:
        parts.append(f"mAP50-95={map50_95:.4f}")
    if precision is not None:
        parts.append(f"P={precision:.4f}")
    if recall is not None:
        parts.append(f"R={recall:.4f}")
    parts.append(f"fitness={fitness:.4f}")

    logger.info("  │  ".join(parts))


def _on_train_end(trainer: "BaseTrainer") -> None:
    """Log final results and best checkpoint path."""
    best: str = str(getattr(trainer, "best", "N/A"))
    metrics: dict = getattr(trainer, "metrics", {}) or {}

    map50 = metrics.get("metrics/mAP50(B)", "N/A")
    map50_95 = metrics.get("metrics/mAP50-95(B)", "N/A")

    logger.info("=" * 60)
    logger.info("PPB Training — COMPLETE")
    logger.info(f"  Best checkpoint : {best}")
    if map50 != "N/A":
        logger.info(f"  Final mAP50     : {map50:.4f}")
    if map50_95 != "N/A":
        logger.info(f"  Final mAP50-95  : {map50_95:.4f}")
    logger.info("=" * 60)


def build_loguru_callbacks() -> dict[str, _Callback]:
    """Build the loguru callback dict for ultralytics.

    Returns:
        Mapping of ultralytics event name → callback function.
        Register each with ``model.add_callback(event, fn)``.
    """
    return {
        "on_train_start": _on_train_start,
        "on_fit_epoch_end": _on_fit_epoch_end,
        "on_train_end": _on_train_end,
    }


# ---------------------------------------------------------------------------
# MLflow callbacks
# ---------------------------------------------------------------------------

def build_mlflow_callbacks(
    model_name: str,
    cfg: "TrainingConfig",
) -> dict[str, _Callback]:
    """Build MLflow logging callbacks for ultralytics.

    Creates a single MLflow run that spans the entire training job.
    Logs hyperparameters on start, metrics each epoch, and the best
    model artifact on completion.

    Note:
        Ultralytics also has a built-in MLflow integration that activates
        automatically when ``mlflow`` is installed. These callbacks provide
        *additional* control (explicit run naming, artifact logging).
        To avoid double-logging, either use these OR the built-in integration,
        not both. The built-in integration is disabled here by design — we
        manage the MLflow run lifecycle explicitly.

    Args:
        model_name: Model variant name, e.g. ``"yolo26l"``.
        cfg: Validated training config (hyperparams to log).

    Returns:
        Mapping of ultralytics event name → callback function.
        Returns an empty dict if ``mlflow`` is not installed, so training
        continues without MLflow without raising an error.
    """
    try:
        import mlflow  # noqa: PLC0415
    except ImportError:
        logger.warning("mlflow not installed — MLflow callbacks disabled.")
        return {}

    run_name = f"{model_name}_ppe_{int(time.time())}"
    _run: list = []  # mutable container so closures can share the run reference

    def on_pretrain_routine_start(trainer: "BaseTrainer") -> None:
        """Remove ultralytics' built-in MLflow integration from this trainer.

        Ultralytics auto-registers its own MLflow callbacks whenever mlflow is
        installed. Its ``on_pretrain_routine_end`` starts a run before our
        ``on_train_start`` fires, which crashes ours with "Run ... is already
        active". It also reads the tracking URI from ``os.environ`` rather than
        ``settings``, so it would log to ``runs/mlflow`` while we log to the
        configured store — splitting one experiment across two backends.

        This event fires immediately after integrations are registered and
        before any of them run, so stripping them here is exact. It is scoped
        to this trainer: no global ultralytics setting is changed or persisted.
        """
        removed = 0
        for event, fns in list(trainer.callbacks.items()):
            kept = [
                f for f in fns
                if getattr(f, "__module__", "") != _ULTRALYTICS_MLFLOW_MODULE
            ]
            removed += len(fns) - len(kept)
            trainer.callbacks[event] = kept
        if removed:
            logger.debug(
                f"Disabled {removed} built-in ultralytics MLflow callback(s); "
                "PPB callbacks own the MLflow run."
            )

    def on_train_start(trainer: "BaseTrainer") -> None:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment("ppb-vision-ppe")

        # Defensive: if something else still opened a run (e.g. a future
        # ultralytics version registers its integration under a different
        # module), attach to it rather than crashing a long training job at
        # epoch 0.
        active = mlflow.active_run()
        if active is not None:
            logger.warning(
                f"An MLflow run ({active.info.run_id}) was already active — "
                "attaching to it instead of starting a new one."
            )
            run = active
        else:
            run = mlflow.start_run(run_name=run_name)
        _run.append(run)
        logger.info(f"MLflow run started: {run.info.run_id} ({run_name})")

        # Log all hyperparams
        mlflow.log_params({
            "model_name": model_name,
            "epochs": cfg.epochs,
            "imgsz": cfg.imgsz,
            "batch": cfg.batch,
            "optimizer": cfg.optimizer,
            "lr0": cfg.lr0,
            "lrf": cfg.lrf,
            "momentum": cfg.momentum,
            "weight_decay": cfg.weight_decay,
            "box": cfg.box,
            "cls": cfg.cls,
            "dfl": cfg.dfl,
            "mosaic": cfg.mosaic,
            "mixup": cfg.mixup,
            "pretrained": cfg.pretrained,
        })

    def on_fit_epoch_end(trainer: "BaseTrainer") -> None:
        metrics: dict = getattr(trainer, "metrics", {}) or {}
        fitness: float = getattr(trainer, "fitness", 0.0) or 0.0
        epoch = trainer.epoch + 1

        log_metrics: dict[str, float] = {"fitness": fitness}
        for key, val in metrics.items():
            # Sanitise key: MLflow doesn't allow '(' or ')' in metric names
            clean_key = key.replace("(", "_").replace(")", "").replace("/", "_")
            if isinstance(val, (int, float)):
                log_metrics[clean_key] = float(val)

        mlflow.log_metrics(log_metrics, step=epoch)

    def on_train_end(trainer: "BaseTrainer") -> None:
        best = getattr(trainer, "best", None)
        if best and str(best) != "":
            mlflow.log_artifact(str(best), artifact_path="weights")
            logger.info(f"MLflow: logged best weights artifact → {best}")

        if _run:
            mlflow.end_run()
            logger.info(f"MLflow run ended: {_run[0].info.run_id}")

    return {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_train_start": on_train_start,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_end": on_train_end,
    }

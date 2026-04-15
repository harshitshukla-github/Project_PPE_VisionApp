"""Training pipeline for the PPB Vision Module.

Provides a unified trainer for YOLO26 and YOLO11 variants with
MLflow experiment tracking and loguru progress logging.

Typical usage::

    from src.training.trainer import train
    best_weights = train("yolo26l")
"""

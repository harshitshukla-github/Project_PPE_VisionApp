"""Shared settings for the PPB Vision Module.

Loads all configuration from environment variables / .env file via
pydantic-settings. Import ``settings`` from here everywhere.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings loaded from .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API keys
    roboflow_api_key: str = Field(..., alias="ROBOFLOW_API_KEY")
    google_api_key: str = Field(..., alias="GOOGLE_API_KEY")

    # Kaggle credentials (required only when downloading Kaggle datasets).
    # Two auth schemes exist; the client prefers the token and falls back to
    # the legacy username+key pair:
    #   - kaggle_api_token: current scheme, a single "KGAT_..." bearer token
    #     from kaggle.com Settings -> API. The username is introspected from
    #     the token, so kaggle_username/kaggle_key are not needed alongside it.
    #   - kaggle_username + kaggle_key: legacy scheme, from kaggle.json.
    kaggle_api_token: str = Field("", alias="KAGGLE_API_TOKEN")
    kaggle_username: str = Field("", alias="KAGGLE_USERNAME")
    kaggle_key: str = Field("", alias="KAGGLE_KEY")
    # Kaggle PPE dataset identifier: "owner/dataset-name" from kaggle.com/datasets/...
    # Defaults to the Safety Helmet & Reflective Jacket set, which labels helmets
    # and vests on the SAME images — the co-occurrence the Roboflow sources lack.
    # Override in .env only to point at a different dataset.
    kaggle_ppe_dataset: str = Field(
        "niravnaik/safety-helmet-and-reflective-jacket",
        alias="KAGGLE_PPE_DATASET",
    )

    # Model
    ppb_model_name: str = Field("yolo26l", alias="PPB_MODEL_NAME")
    ppb_confidence_threshold: float = Field(0.45, alias="PPB_CONFIDENCE_THRESHOLD")
    ppb_iou_threshold: float = Field(0.50, alias="PPB_IOU_THRESHOLD")
    ppb_device: str = Field("cpu", alias="PPB_DEVICE")
    ppb_img_size: int = Field(640, alias="PPB_IMG_SIZE")

    # Tracking
    mlflow_tracking_uri: str = Field(
        "sqlite:///runs/mlflow.db", alias="MLFLOW_TRACKING_URI"
    )

    # Logging
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    # Debug — capture raw uploads for offline diagnosis with `cli.py probe`.
    # Writes the exact bytes the API received, before any decoding, so there is
    # no ambiguity about what the client sent. Off by default: uploads are
    # worker photos, so this stores personal data on disk. Never enable in
    # production.
    ppb_debug_dump_uploads: bool = Field(False, alias="PPB_DEBUG_DUMP_UPLOADS")
    ppb_debug_dump_dir: Path = Field(
        Path("runs/uploads"), alias="PPB_DEBUG_DUMP_DIR"
    )

    # Paths (relative to project root)
    data_raw_dir: Path = Path("data/raw")
    data_processed_dir: Path = Path("data/processed")
    data_splits_dir: Path = Path("data/splits")
    configs_dir: Path = Path("configs")


settings = Settings()

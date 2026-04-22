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

    # Kaggle credentials (required only when downloading Kaggle datasets)
    kaggle_username: str = Field("", alias="KAGGLE_USERNAME")
    kaggle_key: str = Field("", alias="KAGGLE_KEY")
    # Kaggle PPE dataset identifier: "owner/dataset-name" from kaggle.com/datasets/...
    kaggle_ppe_dataset: str = Field("", alias="KAGGLE_PPE_DATASET")

    # Model
    ppb_model_name: str = Field("yolo26l", alias="PPB_MODEL_NAME")
    ppb_confidence_threshold: float = Field(0.45, alias="PPB_CONFIDENCE_THRESHOLD")
    ppb_iou_threshold: float = Field(0.50, alias="PPB_IOU_THRESHOLD")
    ppb_device: str = Field("cpu", alias="PPB_DEVICE")
    ppb_img_size: int = Field(640, alias="PPB_IMG_SIZE")

    # Tracking
    mlflow_tracking_uri: str = Field(
        "http://localhost:5000", alias="MLFLOW_TRACKING_URI"
    )

    # Logging
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    # Paths (relative to project root)
    data_raw_dir: Path = Path("data/raw")
    data_processed_dir: Path = Path("data/processed")
    data_splits_dir: Path = Path("data/splits")
    configs_dir: Path = Path("configs")


settings = Settings()

"""FastAPI application entry point for the PPB Vision service.

Run directly::

    uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000

or via the CLI (recommended — supports model/device selection)::

    uv run python cli.py serve --model yolo11l --port 8000
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from loguru import logger

from src.api.deps import init_detector
from src.api.middleware import add_middleware
from src.api.routes import router

# Read by the lifespan hook; set by cli.py serve before uvicorn starts.
MODEL_ENV_VAR = "PPB_SERVE_MODEL"
DEVICE_ENV_VAR = "PPB_SERVE_DEVICE"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model once at startup; release on shutdown."""
    model_name = os.environ.get(MODEL_ENV_VAR) or None
    device = os.environ.get(DEVICE_ENV_VAR) or None
    detector = init_detector(model_name=model_name, device=device)
    logger.success(f"PPB API ready — serving {detector.model_name} on {detector.device}")
    yield
    logger.info("PPB API shutting down.")


app = FastAPI(
    title="PPB Vision — PPE Compliance API",
    description="AI detection & verification pipeline for PPE compliance checks.",
    version="0.1.0",
    lifespan=lifespan,
)
add_middleware(app)
app.include_router(router)

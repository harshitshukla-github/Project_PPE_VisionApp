"""Where the project's own folders are, whatever folder a command was started from.

A relative path such as ``models`` or ``.env`` means "inside the project", but
Python reads it from the working directory. Under a service manager or a Docker
``WORKDIR`` the two differ, and the server then finds no ``.env``, no weights
and no job matrix. Everything that names a project folder builds it from here.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Trained weights and their metrics files (gitignored).
MODELS_DIR = PROJECT_ROOT / "models"


def from_project_root(path: Path | str) -> Path:
    """Read a relative path as inside the project; keep an absolute one as given."""
    return PROJECT_ROOT / path

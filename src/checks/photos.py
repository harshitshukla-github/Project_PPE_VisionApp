"""Where kept photos live (docs/checks-api.md §9, decided in #34: a folder first).

Everything goes through :class:`PhotoStore`, so cloud storage can replace the
folder later without touching the code that keeps, serves or deletes photos.
A photo is known by its key, a relative name that says nothing about whose
photo it is.
"""

from __future__ import annotations

import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from src.settings import settings

_SUFFIXES = {"image/jpeg": ".jpg", "image/png": ".png"}


class PhotoStore(Protocol):
    """Keeps, reads and deletes photos by key."""

    def save(self, data: bytes, content_type: str) -> str:
        """Keep a photo; return its key."""

    def read(self, key: str) -> bytes:
        """The photo's bytes. Raises ``FileNotFoundError`` if it isn't there."""

    def delete(self, key: str) -> None:
        """Remove the photo for good. Removing one that is already gone is fine."""


class FolderPhotoStore:
    """Photos as files under one folder, ``PPB_PHOTO_DIR`` by default.

    Keys are ``YYYY/MM/<random>.jpg``: random, so a file name never names a
    worker, and grouped by month, so no folder grows without bound. The
    folder is outside anything the server serves by URL; a photo leaves only
    through the API, which checks who is asking.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    def save(self, data: bytes, content_type: str) -> str:
        now = datetime.now(timezone.utc)
        key = f"{now:%Y}/{now:%m}/{secrets.token_hex(16)}{_SUFFIXES.get(content_type, '.bin')}"
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Written beside the final name and then renamed, so a half-written
        # photo never sits under a key the database points to.
        partial = path.with_name(path.name + ".part")
        partial.write_bytes(data)
        os.replace(partial, path)
        return key

    def read(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError("A photo key can't point outside the photo folder.")
        return path


def photo_store() -> PhotoStore:
    """The store in use: the folder named by ``PPB_PHOTO_DIR``."""
    return FolderPhotoStore(settings.ppb_photo_dir)

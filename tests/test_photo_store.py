"""The folder that holds kept photos (docs/checks-api.md §9)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.checks import photos
from src.checks.photos import FolderPhotoStore
from src.paths import PROJECT_ROOT
from src.settings import settings


@pytest.fixture
def store(tmp_path: Path) -> FolderPhotoStore:
    return FolderPhotoStore(tmp_path / "photos")


def test_a_photo_comes_back_as_it_went_in(store: FolderPhotoStore) -> None:
    key = store.save(b"\xff\xd8\xffphoto", "image/jpeg")
    assert store.read(key) == b"\xff\xd8\xffphoto"


def test_a_key_is_random_and_grouped_by_month(store: FolderPhotoStore) -> None:
    keys = {store.save(b"x", "image/jpeg") for _ in range(3)}
    assert len(keys) == 3
    for key in keys:
        assert re.fullmatch(r"\d{4}/\d{2}/[0-9a-f]{32}\.jpg", key)
    assert re.fullmatch(r"\d{4}/\d{2}/[0-9a-f]{32}\.png", store.save(b"x", "image/png"))


def test_no_half_written_file_is_left(store: FolderPhotoStore) -> None:
    store.save(b"x" * 1000, "image/jpeg")
    assert not list(store.root.rglob("*.part"))


def test_a_deleted_photo_is_gone_from_disk(store: FolderPhotoStore) -> None:
    key = store.save(b"x", "image/jpeg")
    store.delete(key)
    assert not (store.root / key).exists()
    with pytest.raises(FileNotFoundError):
        store.read(key)
    store.delete(key)  # already gone: fine


@pytest.mark.parametrize("key", ["../outside.jpg", "2026/11/../../../outside.jpg"])
def test_a_key_cant_reach_outside_the_folder(store: FolderPhotoStore, key: str) -> None:
    with pytest.raises(ValueError):
        store.read(key)


def test_the_default_folder_is_in_the_project_and_ignored_by_git() -> None:
    assert photos.photo_store().root == settings.ppb_photo_dir
    assert settings.ppb_photo_dir == PROJECT_ROOT / "data" / "photos"
    assert "data/photos/" in (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

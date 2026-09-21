"""Tests for Kaggle dataset slug handling in src/data/downloader.py.

A malformed slug surfaces as an opaque 404 from the Kaggle API, which reads
like a network problem rather than a config typo. That already cost this
project one silently-skipped dataset (the only source labelling helmets and
vests on the same images), so the shape is validated up front.
"""

from __future__ import annotations

import pytest

from src.data.downloader import _build_kaggle_registry, _is_valid_slug
from src.settings import settings

DEFAULT_SLUG = "niravnaik/safety-helmet-and-reflective-jacket"


class TestSlugValidation:
    @pytest.mark.parametrize(
        "slug",
        [
            DEFAULT_SLUG,
            "user/dataset",
            "a/b",
            "some-user/some_dataset.v2",
        ],
    )
    def test_well_formed_slugs_accepted(self, slug: str) -> None:
        assert _is_valid_slug(slug) is True

    @pytest.mark.parametrize(
        ("slug", "why"),
        [
            ("safety-helmet-reflective-jacket", "no owner at all"),
            ("owner/safety-helmet", "literal 'owner' placeholder"),
            ("Owner/safety-helmet", "placeholder, different case"),
            ("a/b/c", "too many segments"),
            ("/name", "empty owner"),
            ("owner-name/", "empty name"),
            ("", "empty string"),
            ("   ", "whitespace only"),
        ],
    )
    def test_malformed_slugs_rejected(self, slug: str, why: str) -> None:
        assert _is_valid_slug(slug) is False, f"should reject: {why}"

    def test_surrounding_whitespace_tolerated(self) -> None:
        assert _is_valid_slug(f"  {DEFAULT_SLUG}  ") is True


class TestRegistry:
    def test_defaults_to_the_helmet_and_vest_dataset(self) -> None:
        """The default must be a real slug — it's the co-occurrence source."""
        assert settings.kaggle_ppe_dataset == DEFAULT_SLUG
        registry = _build_kaggle_registry()
        assert len(registry) == 1
        assert registry[0].key == "kaggle_ppe"
        assert registry[0].identifier == DEFAULT_SLUG

    def test_env_override_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings, "kaggle_ppe_dataset", "someone/other-set")
        assert _build_kaggle_registry()[0].identifier == "someone/other-set"

    def test_malformed_slug_skips_rather_than_404s(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "kaggle_ppe_dataset", "owner/placeholder")
        assert _build_kaggle_registry() == []

    def test_empty_slug_skips_kaggle_entirely(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Roboflow-only runs must still work with no Kaggle config."""
        monkeypatch.setattr(settings, "kaggle_ppe_dataset", "")
        assert _build_kaggle_registry() == []


class TestArchiveVerification:
    """The SDK reports success even when extraction stopped early at MAX_PATH.

    That silently cost this project 25,269 of 44,002 images, so completeness is
    verified against the archive rather than trusted from an exit code.
    """

    @staticmethod
    def _make_zip(path: Path, names: list[str]) -> Path:
        import zipfile

        with zipfile.ZipFile(path, "w") as zf:
            for n in names:
                zf.writestr(n, b"\xff\xd8\xff")
        return path

    def test_counts_only_images_in_zip(self, tmp_path: Path) -> None:
        from src.data.downloader import count_images_in_zip

        z = self._make_zip(
            tmp_path / "a.zip",
            ["train/images/a.jpg", "train/labels/a.txt", "data.yaml", "train/images/b.PNG"],
        )
        assert count_images_in_zip(z) == 2

    def test_counts_only_images_under_images_dirs(self, tmp_path: Path) -> None:
        from src.data.downloader import count_images_on_disk

        (tmp_path / "train" / "images").mkdir(parents=True)
        (tmp_path / "train" / "labels").mkdir(parents=True)
        (tmp_path / "train" / "images" / "a.jpg").write_bytes(b"x")
        (tmp_path / "train" / "images" / "b.jpg").write_bytes(b"x")
        (tmp_path / "train" / "labels" / "a.txt").write_text("0 .5 .5 .1 .1")
        assert count_images_on_disk(tmp_path) == 2

    def test_extract_writes_every_member(self, tmp_path: Path) -> None:
        from src.data.downloader import extract_archive

        z = self._make_zip(tmp_path / "a.zip", ["train/images/a.jpg", "train/images/b.jpg"])
        dest = tmp_path / "out"
        dest.mkdir()
        assert extract_archive(z, dest) == 2
        assert (dest / "train" / "images" / "a.jpg").exists()

    def test_missing_archive_raises(self, tmp_path: Path) -> None:
        from src.data.downloader import extract_archive

        with pytest.raises(FileNotFoundError, match="Archive not found"):
            extract_archive(tmp_path / "nope.zip", tmp_path)

    def test_repairs_a_truncated_extraction(self, tmp_path: Path) -> None:
        from src.data.downloader import count_images_on_disk, verify_and_repair

        names = [f"train/images/{i:04d}.jpg" for i in range(10)]
        self._make_zip(tmp_path / "roboflow.zip", names)
        # Simulate the SDK stopping partway through.
        (tmp_path / "train" / "images").mkdir(parents=True)
        (tmp_path / "train" / "images" / "0000.jpg").write_bytes(b"\xff\xd8\xff")
        assert count_images_on_disk(tmp_path) == 1

        assert verify_and_repair("ds", tmp_path) is True
        assert count_images_on_disk(tmp_path) == 10

    def test_complete_extraction_is_left_alone(self, tmp_path: Path) -> None:
        from src.data.downloader import verify_and_repair

        self._make_zip(tmp_path / "roboflow.zip", ["train/images/a.jpg"])
        (tmp_path / "train" / "images").mkdir(parents=True)
        (tmp_path / "train" / "images" / "a.jpg").write_bytes(b"\xff\xd8\xff")
        before = (tmp_path / "train" / "images" / "a.jpg").stat().st_mtime_ns
        assert verify_and_repair("ds", tmp_path) is True
        assert (tmp_path / "train" / "images" / "a.jpg").stat().st_mtime_ns == before

    def test_no_archive_is_not_a_failure(self, tmp_path: Path) -> None:
        """construction_safety has no zip left; that must not read as broken."""
        from src.data.downloader import verify_and_repair

        assert verify_and_repair("construction_safety", tmp_path) is True

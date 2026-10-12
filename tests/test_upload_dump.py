"""Tests for the debug upload-capture flag in src/api/routes.py.

The flag writes the exact bytes the API received to disk so they can be
replayed through `cli.py probe`. It handles client-controlled filenames, so
the path-safety tests here matter as much as the behavioural ones.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.api.routes import _DUMPABLE_SUFFIXES, _dump_upload, _safe_suffix
from src.settings import settings


@pytest.fixture
def dump_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Turn dumping on, pointed at a temp directory."""
    monkeypatch.setattr(settings, "ppb_debug_dump_uploads", True)
    monkeypatch.setattr(settings, "ppb_debug_dump_dir", tmp_path / "uploads")
    return tmp_path / "uploads"


class TestSafeSuffix:
    """`filename` is client-controlled and must never steer the output path."""

    @pytest.mark.parametrize("name", ["a.jpg", "a.JPG", "photo.jpeg", "x.png"])
    def test_whitelisted_extensions_are_kept(self, name: str) -> None:
        assert _safe_suffix(name) in _DUMPABLE_SUFFIXES

    @pytest.mark.parametrize(
        "hostile",
        [
            "../../../etc/passwd",
            "/absolute/path/evil.sh",
            "C:\\Windows\\System32\\evil.exe",
            "..\\..\\evil.bat",
            "no_extension",
            "",
        ],
    )
    def test_hostile_names_collapse_to_bin(self, hostile: str) -> None:
        assert _safe_suffix(hostile) == ".bin"

    def test_none_filename_is_handled(self) -> None:
        assert _safe_suffix(None) == ".bin"

    def test_traversal_with_valid_extension_keeps_only_extension(self) -> None:
        """A traversal path ending in .jpg yields '.jpg' and nothing else."""
        assert _safe_suffix("../../../../tmp/evil.jpg") == ".jpg"


class TestDumpDisabled:
    def test_returns_none_when_flag_is_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings, "ppb_debug_dump_uploads", False)
        assert _dump_upload(b"\xff\xd8\xff", "x.jpg") is None

    def test_writes_nothing_when_flag_is_off(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "uploads"
        monkeypatch.setattr(settings, "ppb_debug_dump_uploads", False)
        monkeypatch.setattr(settings, "ppb_debug_dump_dir", target)
        _dump_upload(b"data", "x.jpg")
        assert not target.exists()


class TestDumpEnabled:
    def test_writes_bytes_verbatim(self, dump_enabled: Path) -> None:
        payload = b"\xff\xd8\xff\xe0 arbitrary bytes \x00\x01"
        out = _dump_upload(payload, "worker.jpg")
        assert out is not None
        assert out.read_bytes() == payload

    def test_creates_directory_if_absent(self, dump_enabled: Path) -> None:
        assert not dump_enabled.exists()
        _dump_upload(b"x", "a.jpg")
        assert dump_enabled.is_dir()

    def test_output_lands_inside_dump_dir(self, dump_enabled: Path) -> None:
        out = _dump_upload(b"x", "../../escape.jpg")
        assert out is not None
        assert out.resolve().parent == dump_enabled.resolve()

    def test_filename_carries_content_hash(self, dump_enabled: Path) -> None:
        payload = b"some bytes"
        out = _dump_upload(payload, "a.png")
        assert out is not None
        assert hashlib.sha256(payload).hexdigest()[:8] in out.name

    def test_identical_payloads_share_a_hash(self, dump_enabled: Path) -> None:
        a = _dump_upload(b"same", "a.jpg")
        b = _dump_upload(b"same", "b.jpg")
        assert a is not None and b is not None
        assert a.name.split("_")[1] == b.name.split("_")[1]

    def test_distinct_payloads_do_not_collide(self, dump_enabled: Path) -> None:
        a = _dump_upload(b"one", "a.jpg")
        b = _dump_upload(b"two", "a.jpg")
        assert a is not None and b is not None
        assert a.name != b.name

    def test_write_failure_returns_none_not_raise(
        self, dump_enabled: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Diagnostics must never take down a request."""
        def boom(*args: object, **kwargs: object) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(Path, "write_bytes", boom)
        assert _dump_upload(b"x", "a.jpg") is None


class TestEndpointIntegration:
    def test_upload_is_captured_end_to_end(
        self, api_client, sample_image_bytes: bytes, dump_enabled: Path
    ) -> None:
        resp = api_client.post(
            "/api/v1/detect",
            files={"image": ("worker.png", sample_image_bytes, "image/png")},
        )
        assert resp.status_code == 200
        written = list(dump_enabled.iterdir())
        assert len(written) == 1
        assert written[0].read_bytes() == sample_image_bytes

    def test_undecodable_payload_is_still_captured(
        self, api_client, dump_enabled: Path
    ) -> None:
        """The broken payload is exactly what you need to inspect."""
        resp = api_client.post(
            "/api/v1/detect",
            files={"image": ("junk.png", b"not an image", "image/png")},
        )
        assert resp.status_code == 400
        assert len(list(dump_enabled.iterdir())) == 1

    def test_empty_upload_is_rejected_before_dumping(
        self, api_client, dump_enabled: Path
    ) -> None:
        resp = api_client.post(
            "/api/v1/detect", files={"image": ("empty.png", b"", "image/png")}
        )
        assert resp.status_code == 400
        assert not dump_enabled.exists()

    def test_nothing_written_when_flag_off(
        self, api_client, sample_image_bytes: bytes, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        target = tmp_path / "uploads"
        monkeypatch.setattr(settings, "ppb_debug_dump_uploads", False)
        monkeypatch.setattr(settings, "ppb_debug_dump_dir", target)
        resp = api_client.post(
            "/api/v1/detect",
            files={"image": ("worker.png", sample_image_bytes, "image/png")},
        )
        assert resp.status_code == 200
        assert not target.exists()

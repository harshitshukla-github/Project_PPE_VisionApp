"""A kept photo loses its location and nothing else (docs/checks-api.md §2)."""

from __future__ import annotations

import io
import struct
import zlib

import cv2
import numpy as np
import pytest
from PIL import Image

from src.checks.location import JPEG_TYPE, PNG_TYPE, PhotoFormatError, content_type, remove_location

GPS = {1: "N", 2: (18.0, 31.0, 12.5), 3: "E", 4: (73.0, 51.0, 2.25)}
ROTATE_90 = 6  # EXIF orientation: shown turned a quarter clockwise
MAKE = "PhoneMaker"


def photo(fmt: str, *, gps: bool = True) -> bytes:
    """A 40x20 photo, tagged to be shown upright as 20x40, with a location if asked."""
    image = Image.new("RGB", (40, 20), (10, 120, 200))
    image.putpixel((3, 4), (250, 10, 10))  # something that would move if it were re-encoded
    exif = Image.Exif()
    exif[0x0112] = ROTATE_90
    exif[0x010F] = MAKE
    if gps:
        exif[0x8825] = GPS
    out = io.BytesIO()
    image.save(out, fmt, exif=exif.tobytes(), **({"quality": 95} if fmt == "JPEG" else {}))
    return out.getvalue()


def decoded(data: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


@pytest.mark.parametrize(("fmt", "kind"), [("JPEG", JPEG_TYPE), ("PNG", PNG_TYPE)])
class TestTheLocationGoes:
    def test_no_gps_is_left(self, fmt: str, kind: str) -> None:
        out, found = remove_location(photo(fmt))
        assert found == kind
        assert dict(Image.open(io.BytesIO(out)).getexif().get_ifd(0x8825)) == {}

    def test_every_other_tag_stays_orientation_above_all(self, fmt: str, kind: str) -> None:
        exif = Image.open(io.BytesIO(remove_location(photo(fmt))[0])).getexif()
        assert (exif.get(0x0112), exif.get(0x010F)) == (ROTATE_90, MAKE)

    def test_the_image_is_not_re_encoded(self, fmt: str, kind: str) -> None:
        original = photo(fmt)
        out, _ = remove_location(original)
        assert len(out) == len(original)  # nothing moved; only the location was zeroed
        assert np.array_equal(decoded(out), decoded(original))
        assert decoded(out).shape[:2] == (40, 20)  # still turned upright when decoded

    def test_only_the_location_bytes_change(self, fmt: str, kind: str) -> None:
        original = photo(fmt)
        out, _ = remove_location(original)
        changed = sum(a != b for a, b in zip(original, out))
        assert 0 < changed < 64

    def test_a_photo_without_a_location_comes_back_unchanged(self, fmt: str, kind: str) -> None:
        plain = photo(fmt, gps=False)
        assert remove_location(plain)[0] == plain


def _with_jpeg_segment(data: bytes, marker: int, payload: bytes) -> bytes:
    """Insert a segment straight after SOI."""
    return data[:2] + bytes((0xFF, marker)) + struct.pack(">H", len(payload) + 2) + payload + data[2:]


XMP_GPS = b'<x:xmpmeta><rdf:Description exif:GPSLatitude="18,31.2N"/></x:xmpmeta>'
XMP_PLAIN = b'<x:xmpmeta><rdf:Description xmp:CreatorTool="Camera"/></x:xmpmeta>'


class TestXmp:
    def test_a_jpeg_xmp_packet_with_a_location_is_dropped(self) -> None:
        base = photo("JPEG", gps=False)
        data = _with_jpeg_segment(base, 0xE1, b"http://ns.adobe.com/xap/1.0/\x00" + XMP_GPS)
        out, _ = remove_location(data)
        assert b"GPSLatitude" not in out
        assert out == base

    def test_a_jpeg_xmp_packet_without_one_stays(self) -> None:
        data = _with_jpeg_segment(photo("JPEG", gps=False), 0xE1,
                                  b"http://ns.adobe.com/xap/1.0/\x00" + XMP_PLAIN)
        assert remove_location(data)[0] == data

    @pytest.mark.parametrize("compressed", [False, True])
    def test_a_png_xmp_chunk_with_a_location_is_dropped(self, compressed: bool) -> None:
        base = photo("PNG", gps=False)
        text = zlib.compress(XMP_GPS) if compressed else XMP_GPS
        body = b"XML:com.adobe.xmp\x00" + bytes((int(compressed), 0)) + b"\x00\x00" + text
        chunk = struct.pack(">I", len(body)) + b"iTXt" + body + struct.pack(">I", zlib.crc32(b"iTXt" + body))
        data = base[:33] + chunk + base[33:]  # after the signature and IHDR
        out, _ = remove_location(data)
        assert out == base

    def test_an_ordinary_png_text_chunk_stays(self) -> None:
        base = photo("PNG", gps=False)
        body = b"Comment\x00made on site"
        chunk = struct.pack(">I", len(body)) + b"tEXt" + body + struct.pack(">I", zlib.crc32(b"tEXt" + body))
        data = base[:33] + chunk + base[33:]
        assert remove_location(data)[0] == data


class TestWhatIsRefused:
    """A photo that might still hold a location isn't kept at all."""

    def test_only_jpeg_and_png(self) -> None:
        image = Image.new("RGB", (8, 8))
        out = io.BytesIO()
        image.save(out, "WEBP")
        with pytest.raises(PhotoFormatError):
            remove_location(out.getvalue())

    def test_a_truncated_jpeg(self) -> None:
        with pytest.raises(PhotoFormatError):
            remove_location(photo("JPEG")[:60])

    def test_a_gps_pointer_outside_the_exif_block(self) -> None:
        data = bytearray(photo("JPEG"))
        tiff = data.index(b"Exif\x00\x00") + 6
        order = "<" if data[tiff:tiff + 2] == b"II" else ">"
        ifd0 = tiff + struct.unpack(order + "I", data[tiff + 4:tiff + 8])[0]
        for i in range(struct.unpack(order + "H", data[ifd0:ifd0 + 2])[0]):
            entry = ifd0 + 2 + 12 * i
            if struct.unpack(order + "H", data[entry:entry + 2])[0] == 0x8825:
                data[entry + 8:entry + 12] = struct.pack(order + "I", 10_000_000)
        with pytest.raises(PhotoFormatError):
            remove_location(bytes(data))


@pytest.mark.parametrize(("data", "kind"), [
    (b"\xff\xd8\xff\xe0rest", JPEG_TYPE),
    (b"\x89PNG\r\n\x1a\nrest", PNG_TYPE),
    (b"GIF89a", None),
    (b"", None),
])
def test_the_content_type_comes_from_the_file_itself(data: bytes, kind: str | None) -> None:
    assert content_type(data) == kind

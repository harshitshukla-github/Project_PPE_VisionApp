"""Removes a photo's location before it is kept (docs/checks-api.md §2).

Only the location goes: the image data and every other tag stay byte for
byte, the orientation tag above all. The detector's boxes are measured on the
photo turned upright by that tag, and browsers turn it the same way, so
dropping it would leave a photo taken sideways stored sideways and its boxes
in the wrong place.

The EXIF GPS block is emptied where it lies: its entries and the values they
point to are overwritten with zeros, and its entry count set to 0. Nothing
moves, so no offset in the rest of the metadata needs rewriting, and the
image itself is never re-encoded. XMP packets that mention GPS are dropped,
since XMP can carry a location too.

JPEG and PNG only, the two formats the app and the API accept. Anything this
can't read safely raises :class:`PhotoFormatError`, and the photo isn't kept:
a photo that might still hold a location is worse than no photo.
"""

from __future__ import annotations

import struct
import zlib

JPEG_TYPE = "image/jpeg"
PNG_TYPE = "image/png"

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_EXIF_HEADER = b"Exif\x00\x00"
# A JPEG's XMP packet, and the extension segments a large one continues in.
_XMP_HEADERS = (b"http://ns.adobe.com/xap/1.0/\x00", b"http://ns.adobe.com/xmp/extension/\x00")
_GPS_POINTER = 0x8825

# Bytes per value for each TIFF field type.
_TYPE_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 13: 4}


class PhotoFormatError(ValueError):
    """The photo isn't a JPEG or PNG this module can read safely."""


def content_type(data: bytes) -> str | None:
    """``image/jpeg`` or ``image/png`` from the file's own first bytes, else ``None``."""
    if data.startswith(b"\xff\xd8\xff"):
        return JPEG_TYPE
    if data.startswith(_PNG_SIGNATURE):
        return PNG_TYPE
    return None


def remove_location(data: bytes) -> tuple[bytes, str]:
    """Return the photo without its location, and its content type.

    Raises:
        PhotoFormatError: Not a JPEG or PNG, or metadata this can't read safely.
    """
    kind = content_type(data)
    if kind == JPEG_TYPE:
        return _jpeg(data), kind
    if kind == PNG_TYPE:
        return _png(data), kind
    raise PhotoFormatError("Only JPEG and PNG photos are kept.")


# ---------------------------------------------------------------------------
# JPEG
# ---------------------------------------------------------------------------

# Markers with no length field after them.
_STANDALONE = {0x01, *range(0xD0, 0xD8)}


def _jpeg(data: bytes) -> bytes:
    out = bytearray(data[:2])  # SOI
    pos = 2
    while pos < len(data):
        if data[pos] != 0xFF:
            raise PhotoFormatError("A JPEG segment doesn't start where expected.")
        while pos < len(data) and data[pos] == 0xFF:  # fill bytes before a marker
            pos += 1
        if pos >= len(data):
            raise PhotoFormatError("The JPEG ends inside a marker.")
        marker = data[pos]
        pos += 1
        if marker in _STANDALONE:
            out += bytes((0xFF, marker))
            continue
        if marker == 0xD9:  # EOI
            out += b"\xff\xd9"
            out += data[pos:]
            return bytes(out)
        if pos + 2 > len(data):
            raise PhotoFormatError("The JPEG ends inside a segment header.")
        (length,) = struct.unpack(">H", data[pos:pos + 2])
        end = pos + length
        if length < 2 or end > len(data):
            raise PhotoFormatError("A JPEG segment runs past the end of the file.")
        payload = data[pos + 2:end]

        if marker == 0xE1 and payload.startswith(_EXIF_HEADER):
            payload = payload[:len(_EXIF_HEADER)] + _without_gps(payload[len(_EXIF_HEADER):])
        elif marker == 0xE1 and payload.startswith(_XMP_HEADERS) and b"GPS" in payload:
            pos = end  # an XMP packet with a location: drop the whole segment
            continue

        out += bytes((0xFF, marker)) + data[pos:pos + 2] + payload
        pos = end
        if marker == 0xDA:  # start of scan: the image data follows, untouched
            out += data[pos:]
            return bytes(out)
    raise PhotoFormatError("The JPEG has no image data.")


# ---------------------------------------------------------------------------
# PNG
# ---------------------------------------------------------------------------

_TEXT_CHUNKS = (b"iTXt", b"tEXt", b"zTXt")


def _text_may_hold_location(kind: bytes, body: bytes) -> bool:
    """Whether a PNG text chunk is a metadata block that holds, or may hold, a location.

    - XMP (``XML:com.adobe.xmp``, an iTXt chunk): when it mentions GPS,
      decompressed first if the chunk is compressed.
    - The hex copies of EXIF or XMP some tools write (``Raw profile type
      exif`` / ``xmp``): always, since they can't be read without decoding.
    """
    keyword, _, rest = body.partition(b"\x00")
    if keyword in (b"Raw profile type exif", b"Raw profile type xmp"):
        return True
    if keyword != b"XML:com.adobe.xmp":
        return False
    if kind != b"iTXt" or len(rest) < 2:
        return True  # XMP somewhere it isn't expected: treat it as holding one
    compressed, text = rest[0] == 1, rest[2:]
    text = text.split(b"\x00", 2)[-1]  # after the language tag and translated keyword
    if compressed:
        try:
            text = zlib.decompress(text)
        except zlib.error as exc:
            raise PhotoFormatError("A compressed XMP chunk isn't readable.") from exc
    return b"GPS" in text


def _png(data: bytes) -> bytes:
    out = bytearray(_PNG_SIGNATURE)
    pos = len(_PNG_SIGNATURE)
    while pos < len(data):
        if pos + 8 > len(data):
            raise PhotoFormatError("The PNG ends inside a chunk header.")
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        kind = data[pos + 4:pos + 8]
        end = pos + 12 + length
        if end > len(data):
            raise PhotoFormatError("A PNG chunk runs past the end of the file.")
        body = data[pos + 8:pos + 8 + length]

        if kind == b"eXIf":
            body = _without_gps(body)
            out += struct.pack(">I", length) + kind + body + struct.pack(">I", zlib.crc32(kind + body))
        elif kind in _TEXT_CHUNKS and _text_may_hold_location(kind, body):
            pass  # drop the chunk
        else:
            out += data[pos:end]
        pos = end
        if kind == b"IEND":
            return bytes(out + data[pos:])
    raise PhotoFormatError("The PNG has no end chunk.")


# ---------------------------------------------------------------------------
# EXIF (a TIFF structure, in both formats)
# ---------------------------------------------------------------------------


def _without_gps(tiff: bytes) -> bytes:
    """The same EXIF block, the same length, with the GPS IFD emptied."""
    if len(tiff) < 8 or tiff[:2] not in (b"II", b"MM"):
        raise PhotoFormatError("The EXIF block isn't readable.")
    order = "<" if tiff[:2] == b"II" else ">"
    out = bytearray(tiff)

    def u16(at: int) -> int:
        _check(at, 2)
        return struct.unpack(order + "H", tiff[at:at + 2])[0]

    def u32(at: int) -> int:
        _check(at, 4)
        return struct.unpack(order + "I", tiff[at:at + 4])[0]

    def _check(at: int, size: int) -> None:
        if at < 0 or at + size > len(tiff):
            raise PhotoFormatError("The EXIF block points outside itself.")

    if u16(2) != 42:
        raise PhotoFormatError("The EXIF block isn't readable.")

    ifd0 = u32(4)
    gps = None
    for i in range(u16(ifd0)):
        entry = ifd0 + 2 + 12 * i
        if u16(entry) == _GPS_POINTER:
            gps = u32(entry + 8)
            break
    if gps is None:
        return tiff  # no location to remove

    count = u16(gps)
    _check(gps, 2 + 12 * count + 4)
    for i in range(count):
        entry = gps + 2 + 12 * i
        kind, n = u16(entry + 2), u32(entry + 4)
        if kind not in _TYPE_SIZES:
            raise PhotoFormatError("The GPS block has a value this can't size.")
        size = _TYPE_SIZES[kind] * n
        if size > 4:  # stored elsewhere in the block: zero it there too
            at = u32(entry + 8)
            _check(at, size)
            out[at:at + size] = bytes(size)
    # The entries, the count, and the pointer to a next IFD.
    out[gps:gps + 2 + 12 * count + 4] = bytes(2 + 12 * count + 4)
    return bytes(out)

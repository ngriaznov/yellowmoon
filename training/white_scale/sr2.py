"""The white-balance presets of an ARW file, read from the SR2 sub-IFD with our own code.

An ARW carries its black level, white level, as-shot white balance and the white balance of every preset
of the camera (Daylight, Shade, Cloudy, Tungsten, Fluorescent and the Kelvin presets) in an encrypted
directory (``docs/04`` section SR2 private data; ``crates/ym-raw/src/sr2.rs`` is the Rust reader this
follows). The path to it:

1. IFD0 tag ``DNGPrivateData`` (0xC634) holds a four-byte little-endian offset to the SR2Private directory.
2. SR2Private lists the offset (0x7200), the length (0x7201) and the cipher key (0x7221) of the block.
3. The block is XORed with a pad derived from the key; the plain bytes are a bare IFD whose value offsets
   are absolute positions in the file.

Each preset is three unsigned 16-bit multipliers ``R G B`` with green near 1024 (a white surface records
``1 / gain``), so the camera's raw white of a preset is ``[G / R, 1, G / B]`` (``preset_white``).
"""

from __future__ import annotations

import struct
from pathlib import Path

TAG_DNG_PRIVATE_DATA = 0xC634
TAG_SUB_IFD_OFFSET = 0x7200
TAG_SUB_IFD_LENGTH = 0x7201
TAG_SUB_IFD_KEY = 0x7221
TAG_WB_RGGB = 0x7313

#: Tag of every white-balance preset of the SR2 sub-IFD and the name this package gives it. The Kelvin
#: presets are named by their label.
PRESET_TAGS: dict[int, str] = {
    0x7820: "daylight",
    0x7821: "cloudy",
    0x7822: "tungsten",
    0x7823: "flash",
    0x7824: "k4500",
    0x7825: "shade",
    0x7826: "fluorescent",
    0x7827: "fluorescent_p1",
    0x7828: "fluorescent_p2",
    0x7829: "fluorescent_m1",
    0x782A: "k8500",
    0x782B: "k6000",
    0x782C: "k3200",
    0x782D: "k2500",
}

#: The presets that carry a colour temperature in their label, and that temperature.
KELVIN_PRESETS: dict[str, float] = {"k2500": 2500.0, "k3200": 3200.0, "k4500": 4500.0, "k6000": 6000.0, "k8500": 8500.0}

#: The presets whose maker's label is a typical colour temperature (not an exact one): checks, not fit input.
NAMED_PRESETS: dict[str, float] = {"daylight": 5500.0, "cloudy": 6500.0, "shade": 7500.0, "tungsten": 3200.0}

_TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}
_TYPE_FMT = {1: "B", 3: "H", 4: "I", 6: "b", 7: "B", 8: "h", 9: "i"}


class Sr2Error(ValueError):
    """The file has no readable SR2 sub-IFD."""


def apply_pad(data: bytes, key: int) -> bytes:
    """Sony's pad cipher (an XOR with a pad derived from ``key``; the same call encrypts and decrypts).

    Four pad words come from ``key = key * 48828125 + 1`` (32 bits), the fourth is then
    ``word3 << 1 | (word0 ^ word2) >> 31``; the pad is extended to 127 words with
    ``pad[p] = (pad[p-4] ^ pad[p-2]) << 1 | (pad[p-3] ^ pad[p-1]) >> 31``, every word is byte-swapped, and
    each data word (little endian, trailing bytes untouched) is XORed with a pad word that is first
    replaced by ``pad[(p + 1) % 128] ^ pad[(p + 65) % 128]``, starting at ``p = 127``.
    """
    mask = 0xFFFFFFFF
    pad = [0] * 128
    for i in range(4):
        key = (key * 48_828_125 + 1) & mask
        pad[i] = key
    pad[3] = ((pad[3] << 1) & mask) | ((pad[0] ^ pad[2]) >> 31)
    for p in range(4, 127):
        pad[p] = (((pad[p - 4] ^ pad[p - 2]) << 1) & mask) | ((pad[p - 3] ^ pad[p - 1]) >> 31)
    for i in range(127):
        w = pad[i]
        pad[i] = int.from_bytes(w.to_bytes(4, "little"), "big")
    words = len(data) // 4
    out = bytearray(data)
    for n in range(words):
        p = 127 + n
        nxt = pad[(p + 1) & 127] ^ pad[(p + 65) & 127]
        pad[p & 127] = nxt
        w = struct.unpack_from("<I", data, 4 * n)[0] ^ nxt
        struct.pack_into("<I", out, 4 * n, w)
    return bytes(out)


def _read_ifd(buf: bytes, at: int, order: str, bias: int = 0) -> dict[int, tuple[int, int, int, int]]:
    """Entries of the IFD at ``at``: tag -> (type, count, value offset in ``buf`` or -1 if inline, entry position).

    Offsets stored in the file are positions in the file; ``bias`` is subtracted to turn them into positions in
    ``buf`` (the SR2 block starts at ``bias`` in the file).
    """
    (count,) = struct.unpack_from(order + "H", buf, at)
    entries = {}
    for i in range(count):
        pos = at + 2 + 12 * i
        tag, typ, cnt = struct.unpack_from(order + "HHI", buf, pos)
        size = _TYPE_SIZE.get(typ, 1) * cnt
        if size <= 4:
            entries[tag] = (typ, cnt, -1, pos + 8)
        else:
            (off,) = struct.unpack_from(order + "I", buf, pos + 8)
            entries[tag] = (typ, cnt, off - bias, pos + 8)
    return entries


def _values(buf: bytes, entry: tuple[int, int, int, int], order: str) -> tuple[int, ...]:
    typ, cnt, off, inline = entry
    fmt = _TYPE_FMT.get(typ)
    if fmt is None:
        raise Sr2Error(f"unsupported field type {typ}")
    at = inline if off < 0 else off
    return struct.unpack_from(order + fmt * cnt, buf, at)


def read_block(path: str | Path) -> tuple[bytes, int]:
    """The decrypted SR2 sub-IFD of a file and the file offset where the block starts."""
    data = Path(path).read_bytes()
    if data[:2] not in (b"II", b"MM"):
        raise Sr2Error("not a TIFF-based file")
    order = "<" if data[:2] == b"II" else ">"
    (ifd0,) = struct.unpack_from(order + "I", data, 4)
    ifd = _read_ifd(data, ifd0, order)
    if TAG_DNG_PRIVATE_DATA not in ifd:
        raise Sr2Error("no DNGPrivateData in IFD0")
    # Four bytes, a little-endian offset whatever the file's byte order is.
    (private_at,) = struct.unpack_from("<I", data, ifd[TAG_DNG_PRIVATE_DATA][3])
    private = _read_ifd(data, private_at, "<")
    for tag in (TAG_SUB_IFD_OFFSET, TAG_SUB_IFD_LENGTH, TAG_SUB_IFD_KEY):
        if tag not in private:
            raise Sr2Error(f"SR2Private has no tag {tag:#x}")
    (offset,) = _values(data, private[TAG_SUB_IFD_OFFSET], "<")
    (length,) = _values(data, private[TAG_SUB_IFD_LENGTH], "<")
    (key,) = struct.unpack_from("<I", data, private[TAG_SUB_IFD_KEY][3])
    if not (14 <= length <= 1 << 20) or offset < 8 or offset + length > len(data):
        raise Sr2Error("SR2 block outside the file")
    return apply_pad(data[offset : offset + length], key), offset


def read_levels(path: str | Path) -> dict[str, tuple[int, ...]]:
    """The as-shot white balance (``as_shot``, R G1 G2 B) and every white-balance preset of a file
    (``daylight``, ``k6000``, ... as R G B multipliers). Presets a body does not write are absent."""
    block, offset = read_block(path)
    entries = _read_ifd(block, 0, "<", bias=offset)
    out: dict[str, tuple[int, ...]] = {}
    if TAG_WB_RGGB in entries:
        out["as_shot"] = _values(block, entries[TAG_WB_RGGB], "<")[:4]
    for tag, name in PRESET_TAGS.items():
        if tag in entries:
            v = _values(block, entries[tag], "<")
            if len(v) >= 3:
                out[name] = tuple(v[:3])
    return out


def preset_white(levels) -> tuple[float, float, float]:
    """The camera white of an ``R G B`` multiplier triple: ``[G / R, 1, G / B]``."""
    r, g, b = (float(v) for v in levels[:3])
    return (g / r, 1.0, g / b)


def read_model(path: str | Path) -> str:
    """The IFD0 Model string of a file (``ILCE-1``), empty when absent."""
    data = Path(path).read_bytes()
    order = "<" if data[:2] == b"II" else ">"
    (ifd0,) = struct.unpack_from(order + "I", data, 4)
    entries = _read_ifd(data, ifd0, order)
    if 0x0110 not in entries:
        return ""
    typ, cnt, off, inline = entries[0x0110]
    at = inline if off < 0 else off
    return data[at : at + cnt].split(b"\0")[0].decode("ascii", "replace").strip()

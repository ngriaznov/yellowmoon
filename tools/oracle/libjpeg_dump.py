#!/usr/bin/env python3
"""Decode the lossless-JPEG tiles of a Sony Lossless L file with libjpeg, as a third oracle.

Test-only. `pylibjpeg-libjpeg` wraps Thomas Richter's libjpeg (GPL-3.0 / commercial), an
implementation of ITU T.81 / T.87 that has nothing in common with LibRaw, rawler or ym-raw. It is
used here as a separate process and is never linked into Yellowmoon (docs/14-licensing.md). It
covers what LibRaw's Sony loader cannot: lossless predictors 2 to 7 and restart intervals of more
than one row.

Usage:
    python3 tools/oracle/libjpeg_dump.py FILE.ARW --out RAW.bin

Requires `pip install pylibjpeg-libjpeg numpy`. Only Sony Lossless L files (TIFF Compression 7, one
sample per pixel, tiled, four-component streams at half the tile size) are handled; for anything
else the script prints {"kind": "skipped", ...} and exits with status 3. Like the other oracle
scripts it writes the mosaic as little-endian u16 and prints one JSON object:

    {"kind": "bayer", "channels": 1, "width": W, "height": H, "libjpeg_version": "2.4.0",
     "raw_image_min": ..., "raw_image_max": ..., "raw_image_sha256": "..."}
"""

import hashlib
import json
import struct
import sys

try:
    import libjpeg
    import numpy as np
except ImportError as exc:  # pragma: no cover - callers check for the module first
    print(json.dumps({"kind": "error", "error": f"libjpeg (pylibjpeg-libjpeg) or numpy is missing: {exc}"}))
    sys.exit(1)


def read_ifds(data):
    """All directories of a little-endian TIFF as {tag: (type, count, value bytes)}."""
    if data[:4] != b"II*\0":
        raise ValueError("not a little-endian TIFF")
    seen, out = set(), []

    def walk(offset):
        if offset in seen or offset < 8 or offset + 2 > len(data):
            return
        seen.add(offset)
        (count,) = struct.unpack_from("<H", data, offset)
        entries = {}
        for i in range(count):
            tag, typ, n, value = struct.unpack_from("<HHI4s", data, offset + 2 + 12 * i)
            entries[tag] = (typ, n, value)
        out.append(entries)
        for tag in (0x014A, 0x8769):
            if tag in entries:
                _typ, n, value = entries[tag]
                if n == 1:
                    walk(struct.unpack("<I", value)[0])
                elif n < 64:
                    (table,) = struct.unpack("<I", value)
                    for k in range(n):
                        walk(struct.unpack_from("<I", data, table + 4 * k)[0])
        (nxt,) = struct.unpack_from("<I", data, offset + 2 + 12 * count)
        if nxt:
            walk(nxt)

    walk(struct.unpack_from("<I", data, 4)[0])
    return out


def value(data, entry, index=0):
    typ, n, raw = entry
    size = {3: 2, 4: 4}[typ]
    if n * size <= 4:
        blob = raw
    else:
        (offset,) = struct.unpack("<I", raw)
        blob = data[offset : offset + n * size]
    return struct.unpack_from("<H" if typ == 3 else "<I", blob, index * size)[0]


def assemble(path):
    data = open(path, "rb").read()
    for entries in read_ifds(data):
        if 0x0144 not in entries or 0x0103 not in entries:
            continue
        if value(data, entries[0x0103]) != 7 or value(data, entries.get(0x0115, (3, 1, b"\1\0\0\0"))) != 1:
            continue
        width, height = value(data, entries[0x0100]), value(data, entries[0x0101])
        tw, tl = value(data, entries[0x0142]), value(data, entries[0x0143])
        count = entries[0x0144][1]
        across = -(-width // tw)
        image = np.zeros((-(-height // tl) * tl, across * tw), dtype="<u2")
        for k in range(count):
            offset, length = value(data, entries[0x0144], k), value(data, entries[0x0145], k)
            tile = libjpeg.decode(data[offset : offset + length])
            if tile.shape != (tl // 2, tw // 2, 4):
                raise ValueError(f"tile {k} decodes to {tile.shape}, not four components at half size")
            ty, tx = divmod(k, across)
            block = image[ty * tl : (ty + 1) * tl, tx * tw : (tx + 1) * tw]
            block[0::2, 0::2] = tile[..., 0]
            block[0::2, 1::2] = tile[..., 1]
            block[1::2, 0::2] = tile[..., 2]
            block[1::2, 1::2] = tile[..., 3]
        return image[:height, :width]
    return None


def main(argv):
    args = argv[1:]
    if len(args) != 3 or args[1] != "--out":
        print(json.dumps({"kind": "error", "error": "usage: libjpeg_dump.py FILE --out RAW.bin"}))
        return 2
    try:
        image = assemble(args[0])
    except Exception as exc:  # noqa: BLE001 - report every failure as JSON
        print(json.dumps({"kind": "error", "error": f"{type(exc).__name__}: {exc}"}))
        return 1
    if image is None:
        print(json.dumps({"kind": "skipped", "reason": "not a tiled Sony Lossless L file"}))
        return 3
    data = np.ascontiguousarray(image, dtype="<u2")
    with open(args[2], "wb") as f:
        f.write(data.tobytes())
    print(
        json.dumps(
            {
                "kind": "bayer",
                "channels": 1,
                "height": int(data.shape[0]),
                "width": int(data.shape[1]),
                "libjpeg_version": getattr(libjpeg, "__version__", "?"),
                "raw_image_min": int(data.min()),
                "raw_image_max": int(data.max()),
                "raw_image_sha256": hashlib.sha256(data.tobytes()).hexdigest(),
                "input_patched": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

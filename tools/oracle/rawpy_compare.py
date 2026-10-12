#!/usr/bin/env python3
"""Dump LibRaw's undithered view of a raw file's pixels, for the decoder oracle.

Test-only. LibRaw (through rawpy) is an LGPL/CDDL program that Yellowmoon uses as an independent
oracle: it must never be linked into a shipped binary (docs/14-licensing.md). The Rust oracle
(`tools/oracle`, `crates/ym-raw/tests/oracle_decode.rs`) runs this script as a separate process.

Usage:
    python3 tools/oracle/rawpy_compare.py FILE --out RAW.bin [--meta META.json] [--no-patch]

Writes the pixels LibRaw decoded into RAW.bin as little-endian unsigned 16-bit integers and prints
one JSON object (also written to META.json when given):

    {
      "libraw_version": "0.22.1",
      "kind": "bayer" | "rgb" | "error",
      "width": raw_width, "height": raw_height,   # pixels per row / rows
      "channels": 1 | 3,                          # samples per pixel in RAW.bin
      "black_level_per_channel": [R, G, B, G2],  # LibRaw order
      "camera_white_level_per_channel": [...] | null,     # LibRaw linear_max
      "white_level": maximum,
      "camera_whitebalance": [R, G, B, G2],
      "color_desc": "RGBG", "raw_pattern": [[0, 1], [3, 2]],
      "raw_image_sha256": "...", "raw_image_min": 0, "raw_image_max": 16383,
      "input_patched": false
    }

kind "bayer": RAW.bin is `raw.raw_image`, the whole sensor readout (margins included), exactly what
LibRaw's loaders write, before any black subtraction, scaling or dithering.

kind "rgb": the file is a three-component "pseudo-raw" (Sony Lossless M or S). LibRaw's loader
`sony_ycbcr_load_raw` converts the YCbCr tiles to RGB with replicated chroma and no white-balance
undo, and keeps the result in a four-channel stack that rawpy exposes as `raw_image` (shape
height x width x 4, the fourth channel unused). RAW.bin holds the first three channels,
interleaved. LibRaw reports black 1024 and maximum 17536 for these.

LibRaw 0.22.1 reaches that loader only for TIFF Compression 6 (its YCbCr test is
`tiff_compress == 6`; compression 7 is rewritten to 6 on a path that tiled files written by the
synthetic writer do not take, and with Compression 7 it falls back to a generic lossless-JPEG
loader that leaves `raw_image` all zero). So for a raw IFD with Compression 7, photometric YCbCr
and three samples, the script decodes a temporary copy in which that one Compression value is 6.
The tile data is untouched; "input_patched" says so. --no-patch turns this off.

On failure prints {"kind": "error", "error": "..."} and exits with status 1.
"""

import hashlib
import json
import os
import struct
import sys
import tempfile

try:
    import numpy as np
    import rawpy
except ImportError as exc:  # pragma: no cover - the callers check for rawpy first
    print(json.dumps({"kind": "error", "error": f"rawpy or numpy is not installed: {exc}"}))
    sys.exit(1)


def to_list(values):
    """Plain Python numbers from a numpy array or sequence."""
    if values is None:
        return None
    return [x.item() if hasattr(x, "item") else x for x in list(values)]


def patch_ycbcr_compression(path):
    """Returns the path of a copy of the file in which every raw IFD with Compression 7,
    photometric YCbCr (6) and three samples per pixel says Compression 6 instead, or None when
    there is no such directory (or the file is not a little-endian TIFF)."""
    with open(path, "rb") as f:
        data = bytearray(f.read())
    if data[:4] != b"II*\0":
        return None
    seen = set()
    directories = []

    def walk(offset):
        if offset in seen or offset < 8 or offset + 2 > len(data):
            return
        seen.add(offset)
        (count,) = struct.unpack_from("<H", data, offset)
        if offset + 2 + 12 * count + 4 > len(data):
            return
        entries = {}
        for i in range(count):
            tag, typ, n = struct.unpack_from("<HHI", data, offset + 2 + 12 * i)
            entries[tag] = (offset + 2 + 12 * i, typ, n)
        directories.append(entries)
        for tag in (0x014A, 0x8769):
            if tag in entries:
                pos, _typ, n = entries[tag]
                if n == 1:
                    walk(struct.unpack_from("<I", data, pos + 8)[0])
                elif n < 64:
                    (table,) = struct.unpack_from("<I", data, pos + 8)
                    for k in range(n):
                        if table + 4 * k + 4 <= len(data):
                            walk(struct.unpack_from("<I", data, table + 4 * k)[0])
        (nxt,) = struct.unpack_from("<I", data, offset + 2 + 12 * count)
        if nxt:
            walk(nxt)

    walk(struct.unpack_from("<I", data, 4)[0])

    def short(entries, tag):
        if tag not in entries:
            return None
        return struct.unpack_from("<H", data, entries[tag][0] + 8)[0]

    patched = False
    for entries in directories:
        if short(entries, 0x0103) == 7 and short(entries, 0x0106) == 6 and short(entries, 0x0115) == 3:
            struct.pack_into("<H", data, entries[0x0103][0] + 8, 6)
            patched = True
    if not patched:
        return None
    handle, copy = tempfile.mkstemp(suffix=".ARW", prefix="rawpy_compare_")
    with os.fdopen(handle, "wb") as f:
        f.write(data)
    return copy


def describe(path, out_path, patch=True):
    result = {"libraw_version": ".".join(str(v) for v in rawpy.libraw_version), "input_patched": False}
    copy = patch_ycbcr_compression(path) if patch else None
    if copy:
        result["input_patched"] = True
    try:
        with rawpy.imread(copy or path) as raw:
            result["color_desc"] = raw.color_desc.decode("ascii", "replace")
            result["black_level_per_channel"] = to_list(raw.black_level_per_channel)
            result["white_level"] = int(raw.white_level)
            result["camera_white_level_per_channel"] = to_list(raw.camera_white_level_per_channel)
            result["camera_whitebalance"] = to_list(raw.camera_whitebalance)
            result["raw_type"] = str(raw.raw_type).split(".")[-1]
            pattern = raw.raw_pattern
            result["raw_pattern"] = None if pattern is None else [to_list(r) for r in pattern]

            image = np.asarray(raw.raw_image)
            if image.ndim == 3:
                data = np.ascontiguousarray(image[..., :3], dtype="<u2")
                result["kind"] = "rgb"
                result["channels"] = 3
            else:
                data = np.ascontiguousarray(image, dtype="<u2")
                result["kind"] = "bayer"
                result["channels"] = 1
            result["height"], result["width"] = int(data.shape[0]), int(data.shape[1])
            result["raw_image_min"] = int(data.min())
            result["raw_image_max"] = int(data.max())
            result["raw_image_sha256"] = hashlib.sha256(data.tobytes()).hexdigest()
            with open(out_path, "wb") as f:
                f.write(data.tobytes())
    finally:
        if copy:
            os.unlink(copy)
    return result


def main(argv):
    args = list(argv[1:])
    meta_path = None
    out_path = None
    patch = True
    positional = []
    i = 0
    while i < len(args):
        if args[i] == "--out" and i + 1 < len(args):
            out_path = args[i + 1]
            i += 2
        elif args[i] == "--meta" and i + 1 < len(args):
            meta_path = args[i + 1]
            i += 2
        elif args[i] == "--no-patch":
            patch = False
            i += 1
        else:
            positional.append(args[i])
            i += 1
    if len(positional) != 1 or out_path is None:
        print(json.dumps({"kind": "error", "error": "usage: rawpy_compare.py FILE --out RAW.bin [--meta META.json] [--no-patch]"}))
        return 2
    try:
        result = describe(positional[0], out_path, patch)
    except Exception as exc:  # noqa: BLE001 - report every LibRaw failure as JSON
        result = {"kind": "error", "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(result))
        return 1
    text = json.dumps(result, sort_keys=True)
    if meta_path:
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

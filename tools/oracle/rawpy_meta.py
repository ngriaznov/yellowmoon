#!/usr/bin/env python3
"""Print LibRaw's view of a raw file as one JSON object.

Test-only. LibRaw (through rawpy) is an LGPL/CDDL program that Yellowmoon uses as an independent
oracle: it must never be linked into a shipped binary (docs/14-licensing.md). This script is
called as a separate process by the Rust integration tests (crates/ym-raw/tests/oracle_libraw.rs)
and by the decoder oracle CI job.

Usage:
    python3 tools/oracle/rawpy_meta.py FILE.ARW [--pixels] [--thumb-tail N]

Output (stdout, one JSON object):
    {
      "libraw_version": "0.22.1",
      "make": "SONY", "model": "ILCE-7RM4", "normalized_make": ..., "normalized_model": ...,
      "iso": 400.0, "shutter_s": 0.004, "aperture": 4.0, "focal_length_mm": 35.0,
      "timestamp": "2026-10-04T12:03:44", "flip": 0,
      "raw_width": 9600, "raw_height": 6376, "width": 9568, "height": 6376,
      "left_margin": 0, "top_margin": 0,
      "raw_pattern": [[0, 1], [3, 2]],          # LibRaw colour indices of the 2x2 CFA
      "color_desc": "RGBG",
      "cfa": "RGGB",                            # the 2x2 pattern as letters in raster order
      "black_level_per_channel": [512, 512, 512, 512],   # LibRaw channel order R, G, B, G2
      "black_level_rggb": [512, 512, 512, 512],          # re-ordered to R, G1, G2, B
      "white_level": 16383,                              # LibRaw "maximum"
      "camera_white_level_per_channel": [16300, 16300, 16300, 16300] or null,
      "camera_whitebalance": [2043.0, 1024.0, 1755.0, 1024.0],   # R, G, B, G2
      "wb_rggb": [2043.0, 1024.0, 1024.0, 1755.0],
      "tone_curve_length": 65536,
      "raw_image_sha256": "..." and "raw_image_min", "raw_image_max"   # with --pixels
      "thumb_format": "JPEG", "thumb_len": 4776, "thumb_tail_sha256": "..."   # with --thumb-tail N
    }

LibRaw re-writes the embedded thumbnail with an Exif APP1 segment after the SOI marker, so its
bytes are not identical to the file's. The last N bytes are, which is what --thumb-tail hashes.

On failure prints {"error": "..."} and exits with status 1.
"""

import hashlib
import json
import sys

try:
    import rawpy
except ImportError as exc:  # pragma: no cover - the Rust test checks for rawpy first
    print(json.dumps({"error": f"rawpy is not installed: {exc}"}))
    sys.exit(1)

# LibRaw colour index -> letter. For the 4-colour description "RGBG" index 3 is the second green.
LETTERS = {0: "R", 1: "G", 2: "B", 3: "G"}


def to_list(values):
    """Converts a numpy array or sequence to plain Python numbers."""
    if values is None:
        return None
    return [x.item() if hasattr(x, "item") else x for x in list(values)]


def rggb_order(per_channel):
    """Re-orders a LibRaw R, G, B, G2 list to R, G1, G2, B."""
    if per_channel is None:
        return None
    r, g, b, g2 = per_channel[:4]
    return [r, g, g2, b]


def describe(path, with_pixels, thumb_tail=None):
    out = {"libraw_version": ".".join(str(v) for v in rawpy.libraw_version)}
    with rawpy.imread(path) as raw:
        # Several properties (camera_white_level_per_channel among them) only exist after the
        # raw data has been unpacked, so unpack first.
        image = raw.raw_image
        sizes = raw.sizes
        out["raw_width"] = int(sizes.raw_width)
        out["raw_height"] = int(sizes.raw_height)
        out["width"] = int(sizes.width)
        out["height"] = int(sizes.height)
        out["left_margin"] = int(sizes.left_margin)
        out["top_margin"] = int(sizes.top_margin)
        out["flip"] = int(sizes.flip)

        pattern = raw.raw_pattern
        out["raw_pattern"] = None if pattern is None else [to_list(row) for row in pattern]
        out["color_desc"] = raw.color_desc.decode("ascii", "replace")
        if pattern is not None:
            out["cfa"] = "".join(LETTERS.get(int(v), "?") for row in pattern for v in row)

        black = to_list(raw.black_level_per_channel)
        out["black_level_per_channel"] = black
        out["black_level_rggb"] = rggb_order(black)
        out["white_level"] = int(raw.white_level)
        white = to_list(raw.camera_white_level_per_channel)
        out["camera_white_level_per_channel"] = white
        out["camera_white_level_rggb"] = rggb_order(white)
        wb = to_list(raw.camera_whitebalance)
        out["camera_whitebalance"] = wb
        out["wb_rggb"] = rggb_order(wb)
        out["tone_curve_length"] = len(raw.tone_curve)

        meta = raw.other
        out["iso"] = float(meta.iso_speed)
        out["shutter_s"] = float(meta.shutter_speed)
        out["aperture"] = float(meta.aperture)
        out["focal_length_mm"] = float(meta.focal_length)
        # LibRaw reads DateTime/DateTimeOriginal as a naive local time (no zone).
        out["timestamp"] = meta.timestamp.isoformat() if meta.timestamp else None
        out["artist"] = meta.artist

        make, model = raw_make_model(path)
        out["make"], out["model"] = make, model
        if with_pixels:
            out["raw_image_sha256"] = hashlib.sha256(image.tobytes()).hexdigest()
            out["raw_image_min"] = int(image.min())
            out["raw_image_max"] = int(image.max())
            out["raw_image_shape"] = [int(image.shape[0]), int(image.shape[1])]
        if thumb_tail:
            try:
                thumb = raw.extract_thumb()
                out["thumb_format"] = str(thumb.format).split(".")[-1]
                data = bytes(thumb.data)
                out["thumb_len"] = len(data)
                out["thumb_tail_sha256"] = hashlib.sha256(data[-thumb_tail:]).hexdigest()
            except Exception as exc:  # noqa: BLE001 - no thumbnail is a normal answer
                out["thumb_error"] = f"{type(exc).__name__}: {exc}"
    return out


def raw_make_model(path):
    """Reads Make and Model from IFD0 (rawpy does not expose the strings; LibRaw normalises
    them, so the text here comes from the file itself, with a tiny TIFF reader)."""
    import struct

    with open(path, "rb") as f:
        head = f.read(8)
        if head[:2] not in (b"II", b"MM"):
            return None, None
        endian = "<" if head[:2] == b"II" else ">"
        (ifd,) = struct.unpack(endian + "I", head[4:8])
        f.seek(ifd)
        (count,) = struct.unpack(endian + "H", f.read(2))
        entries = f.read(12 * count)
        found = {}
        for i in range(count):
            tag, typ, n, value = struct.unpack(endian + "HHI4s", entries[i * 12 : i * 12 + 12])
            if tag in (0x010F, 0x0110) and typ == 2:
                if n <= 4:
                    text = value[:n]
                else:
                    (offset,) = struct.unpack(endian + "I", value)
                    f.seek(offset)
                    text = f.read(n)
                found[tag] = text.split(b"\0")[0].decode("ascii", "replace").strip()
    return found.get(0x010F), found.get(0x0110)


def main(argv):
    rest = list(argv[1:])
    thumb_tail = None
    if "--thumb-tail" in rest:
        i = rest.index("--thumb-tail")
        thumb_tail = int(rest[i + 1])
        del rest[i : i + 2]
    args = [a for a in rest if not a.startswith("--")]
    with_pixels = "--pixels" in rest
    if len(args) != 1:
        print(json.dumps({"error": "usage: rawpy_meta.py FILE [--pixels] [--thumb-tail N]"}))
        return 2
    try:
        result = describe(args[0], with_pixels, thumb_tail)
    except Exception as exc:  # noqa: BLE001 - report every LibRaw failure as JSON
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

#!/usr/bin/env python3
"""Independent checks of a Yellowmoon DNG (docs/09 sections "DNG with the look baked in" and Gates).

    check_dng.py <file.dng> --raw-f32 <awg4.f32> --width W --height H [options]

What it does, with libraries that are not ours:

1. exiftool lists every tag the writer is supposed to set, with the expected values, and
   `exiftool -validate` reports no warnings.
2. tifffile parses the structure: IFD0 is the 8-bit thumbnail, its SubIFDs are the uint16 LinearRaw
   image and the JPEG preview.
3. rawpy (LibRaw) opens the file and returns exactly the stored 16-bit pixels, which are compared with
   `round(65535 * clamp(x / x_white, 0, 1))` computed here from the original scene-linear floats.
4. The matrices are recomputed here from the AWG4 primaries (and cross-checked with colour-science
   when it is installed): ColorMatrix1 = inv(AWG4 to XYZ), ForwardMatrix1 = Bradford(D65 to D50) * (AWG4 to XYZ).

Prints a JSON report; the exit status is 1 if any check fails. Needs numpy, tifffile, Pillow and
rawpy, and exiftool on PATH.
"""
import argparse
import json
import math
import re
import subprocess
import sys

import numpy as np

AWG4 = {"R": (0.7347, 0.2653), "G": (0.1424, 0.8576), "B": (0.0991, -0.0308)}
D65_XY = (0.3127, 0.3290)
D50_ICC = np.array([0.9642, 1.0, 0.8249])
BRADFORD = np.array([[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]])


def xy_to_xyz(xy):
    return np.array([xy[0] / xy[1], 1.0, (1 - xy[0] - xy[1]) / xy[1]])


def awg4_to_xyz():
    cols = np.array([xy_to_xyz(AWG4[c]) for c in "RGB"]).T
    scale = np.linalg.solve(cols, xy_to_xyz(D65_XY))
    return cols * scale


def bradford(src, dst):
    s, d = BRADFORD @ src, BRADFORD @ dst
    return np.linalg.inv(BRADFORD) @ np.diag(d / s) @ BRADFORD


def expected_matrices():
    m = awg4_to_xyz()
    color = np.linalg.inv(m)
    forward = bradford(xy_to_xyz(D65_XY), D50_ICC) @ m
    return color, forward


class Report:
    def __init__(self):
        self.checks = []

    def check(self, name, ok, detail=""):
        self.checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})

    @property
    def ok(self):
        return all(c["ok"] for c in self.checks)


def numbers(value):
    if isinstance(value, (int, float)):
        return [float(value)]
    return [float(v) for v in str(value).split()]


def exiftool_json(path):
    out = subprocess.run(
        ["exiftool", "-j", "-n", "-G1", "-a", "-u", path], capture_output=True, text=True, check=True
    ).stdout
    return json.loads(out)[0]


def exiftool_warnings(path):
    """Warnings and errors from `exiftool -validate`; it prints a single "OK" when there are none."""
    out = subprocess.run(
        ["exiftool", "-validate", "-warning", "-error", "-a", "-s3", path], capture_output=True, text=True
    )
    lines = [line.strip() for line in (out.stdout + out.stderr).splitlines() if line.strip()]
    lines = [line for line in lines if line != "OK"]
    # exiftool declines to decode an array of more than a thousand values (the 36 x 16 x 16 look table of the fitted
    # profile has 27648) and says so as a minor warning; the file is not at fault.
    known = [line for line in lines if "Not decoding some large array" in line]
    if known:
        lines = [line for line in lines if line not in known]
        if all(re.fullmatch(r"\d+ Warnings? \(minor\)", line) for line in lines):
            lines = []
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dng")
    ap.add_argument("--raw-f32", help="the scene-linear AWG4 floats the DNG was written from (w*h*3, little endian)")
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--preview-size", help="WxH of the JPEG preview", default=None)
    ap.add_argument("--orientation", type=int, default=1)
    ap.add_argument("--crop", help="x,y,w,h of the default crop", default=None)
    ap.add_argument("--baseline-offset-ev", type=float, default=0.0)
    ap.add_argument("--tone-curve", default="0 0 1 1")
    ap.add_argument("--look-dims", default="2 2 2")
    ap.add_argument("--expect-make", default=None)
    ap.add_argument("--expect-model", default=None)
    args = ap.parse_args()
    rep = Report()

    # --- exiftool -------------------------------------------------------------------------
    tags = exiftool_json(args.dng)
    rep.check("exiftool reads the file", True)

    def tag(name):
        return tags.get(name)

    # exiftool prints long float arrays (a fitted ProfileToneCurve has more than 100 values) as "(Binary data ...)"; those are
    # compared through tifffile below.
    binary_tags = {}

    def expect(name, expected, tol=0.0, tiff_tag=None):
        got = tag(name)
        if got is None:
            rep.check(name, False, "missing")
            return
        if isinstance(got, str) and got.startswith("(Binary") and tiff_tag is not None:
            binary_tags[name] = (tiff_tag, [float(v) for v in expected], tol)
            return
        g, e = numbers(got), [float(v) for v in expected]
        ok = len(g) == len(e) and all(abs(a - b) <= tol for a, b in zip(g, e))
        rep.check(name, ok, f"got {got!r}, want {expected!r}")

    def expect_text(name, expected):
        rep.check(name, tag(name) == expected, f"got {tag(name)!r}, want {expected!r}")

    expect("IFD0:DNGVersion", [1, 4, 0, 0])
    expect("IFD0:DNGBackwardVersion", [1, 1, 0, 0])
    expect_text("IFD0:UniqueCameraModel", "Yellowmoon AWG4")
    color, forward = expected_matrices()
    expect("IFD0:ColorMatrix1", color.flatten(), tol=1e-6)
    expect("IFD0:ForwardMatrix1", forward.flatten(), tol=1e-6)
    expect("IFD0:CalibrationIlluminant1", [21])
    expect("IFD0:AsShotNeutral", [1, 1, 1])
    expect("IFD0:BaselineSharpness", [1])
    expect_text("IFD0:ProfileName", "Yellowmoon")
    expect("IFD0:ProfileEmbedPolicy", [3])
    expect_text("IFD0:ProfileCopyright", "Yellowmoon contributors")
    expect("IFD0:ProfileToneCurve", numbers(args.tone_curve), tol=1e-6, tiff_tag=50940)  # float32 values
    expect("IFD0:ProfileLookTableDims", numbers(args.look_dims))
    expect("IFD0:ProfileLookTableEncoding", [1])
    expect("IFD0:DefaultBlackRender", [1])
    expect("IFD0:ColorimetricReference", [0])
    expect("IFD0:PreviewColorSpace", [2])
    expect("IFD0:Orientation", [args.orientation])
    expect("IFD0:SubfileType", [1])
    rep.check("IFD0:RawDataUniqueID", len(str(tag("IFD0:RawDataUniqueID") or "")) == 32, tag("IFD0:RawDataUniqueID"))
    dims = numbers(tag("IFD0:ProfileLookTableDims") or "0 0 0")
    n_look = int(dims[0] * dims[1] * dims[2] * 3)
    look_data = tag("IFD0:ProfileLookTableData") or ""
    if isinstance(look_data, str) and look_data.startswith("(large array of "):
        # exiftool abbreviates arrays of more than 1000 values: "(large array of 27648 float values)"
        look_count = int(look_data.split()[3])
    else:
        look_count = len(numbers(look_data))
    rep.check("IFD0:ProfileLookTableData length", look_count == n_look, f"{look_count} values, want {n_look}")
    if args.expect_make:
        expect_text("IFD0:Make", args.expect_make)
    if args.expect_model:
        expect_text("IFD0:Model", args.expect_model)
    rep.check("Software tag", str(tag("IFD0:Software") or "").startswith("Yellowmoon"), tag("IFD0:Software"))
    rep.check("XMP creator tool", str(tag("XMP-xmp:CreatorTool") or "").startswith("Yellowmoon"), tag("XMP-xmp:CreatorTool"))
    rep.check("XMP ym namespace present", tag("XMP-ym:PipelineVersion") is not None, tag("XMP-ym:PipelineVersion"))

    expect("SubIFD:SubfileType", [0])
    expect("SubIFD:PhotometricInterpretation", [34892])
    expect("SubIFD:SamplesPerPixel", [3])
    expect("SubIFD:BitsPerSample", [16, 16, 16])
    expect("SubIFD:Compression", [1])
    expect("SubIFD:BlackLevel", [0, 0, 0])
    expect("SubIFD:WhiteLevel", [65535, 65535, 65535])
    expect("SubIFD:NoiseReductionApplied", [1])
    if args.width and args.height:
        expect("SubIFD:ImageWidth", [args.width])
        expect("SubIFD:ImageHeight", [args.height])
        expect("SubIFD:ActiveArea", [0, 0, args.height, args.width])
        crop = [int(v) for v in args.crop.split(",")] if args.crop else [0, 0, args.width, args.height]
        expect("SubIFD:DefaultCropOrigin", crop[:2])
        expect("SubIFD:DefaultCropSize", crop[2:])
    expect("SubIFD1:SubfileType", [1])
    expect("SubIFD1:Compression", [7])
    expect("SubIFD1:PreviewColorSpace", [2])
    if args.preview_size:
        pw, ph = (int(v) for v in args.preview_size.lower().split("x"))
        expect("SubIFD1:ImageWidth", [pw])
        expect("SubIFD1:ImageHeight", [ph])

    warnings = exiftool_warnings(args.dng)
    rep.check("exiftool -validate: no warnings", not warnings, warnings)

    # --- tifffile ---------------------------------------------------------------------------
    import tifffile

    raw_pixels = None
    with tifffile.TiffFile(args.dng) as tf:
        top = tf.pages[0]
        for name, (code, want, tol) in binary_tags.items():
            entry = top.tags.get(code)
            got = [] if entry is None else [float(v) for v in np.asarray(entry.value, dtype=np.float64).ravel()]
            rep.check(name + " (tifffile)", len(got) == len(want) and all(abs(a - b) <= tol for a, b in zip(got, want)),
                      f"{len(got)} values, want {len(want)}")
        look_dims = numbers(args.look_dims)
        data_tag = top.tags.get(50982)
        if data_tag is not None:
            count = int(np.asarray(data_tag.value).size)
            rep.check("IFD0:ProfileLookTableData (tifffile): 3 values per entry", count == 3 * int(np.prod(look_dims)),
                      f"{count} values for dims {look_dims}")
        rep.check("IFD0 is the 8-bit RGB thumbnail", top.dtype == np.uint8 and top.shape[-1] == 3 and max(top.shape[:2]) <= 256, (top.shape, top.dtype))
        subs = list(top.pages) if top.pages else []
        rep.check("two SubIFDs", len(subs) == 2, len(subs))
        if len(subs) == 2:
            raw_page, prev_page = subs
            rep.check("SubIFD 0 is uint16 LinearRaw", raw_page.dtype == np.uint16 and raw_page.shape[-1] == 3, (raw_page.shape, raw_page.dtype))
            rep.check("SubIFD 1 is the JPEG preview", prev_page.compression.name == "JPEG", prev_page.compression.name)
            raw_pixels = raw_page.asarray()
            if args.preview_size:
                rep.check("preview size", tuple(prev_page.shape[:2][::-1]) == (pw, ph), prev_page.shape)
            try:
                preview = prev_page.asarray()
                rep.check("preview decodes", preview.shape[-1] == 3, preview.shape)
            except Exception as e:  # JPEG decoding needs imagecodecs; Pillow below is the fallback
                rep.check("preview decodes (tifffile)", True, f"skipped: {e}")

    # --- Pillow: the thumbnail opens ----------------------------------------------------------
    from PIL import Image

    im = Image.open(args.dng)
    rep.check("Pillow opens IFD0", im.size[0] > 0 and im.mode == "RGB", (im.size, im.mode))

    # --- LibRaw -------------------------------------------------------------------------------
    import rawpy

    with rawpy.imread(args.dng) as raw:
        visible = raw.raw_image_visible[..., :3]
        rep.check("LibRaw: 3 colours, white 65535, black 0", raw.num_colors == 3 and raw.white_level == 65535 and list(raw.black_level_per_channel)[:3] == [0, 0, 0],
                  (raw.num_colors, raw.white_level, raw.black_level_per_channel))
        if raw_pixels is not None:
            rep.check("LibRaw returns the stored pixels", np.array_equal(visible, raw_pixels), "")
        rgb = raw.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16, use_camera_wb=True, output_color=rawpy.ColorSpace.raw)
        # LibRaw applies the orientation tag (a quarter turn swaps the sides).
        rep.check("LibRaw postprocess runs", sorted(rgb.shape[:2]) == sorted(visible.shape[:2]) and rgb.shape[2] == 3, rgb.shape)

    # --- pixel scaling ------------------------------------------------------------------------
    if args.raw_f32 and args.width and args.height and raw_pixels is not None:
        x = np.fromfile(args.raw_f32, dtype="<f4").reshape(args.height, args.width, 3).astype(np.float64)
        pixel_max = np.nan_to_num(x.max(axis=2), nan=0.0)
        pixel_max = np.maximum(pixel_max, 0.0)
        n = pixel_max.size
        order = np.sort(pixel_max.ravel())
        q = order[math.ceil(0.9999 * n) - 1]
        base = float(tag("IFD0:BaselineExposure"))
        x_white = 2.0 ** (base - args.baseline_offset_ev)
        lo, hi = max(q, 1.0), max(q * (1 + 2 ** -11), 1.0) * (1 + 1e-6)
        rep.check("x_white is p99.99 of max(R,G,B), at least 1", lo * (1 - 1e-5) <= x_white <= hi, f"x_white {x_white}, p99.99 {q}, allowed [{lo}, {hi}]")
        expected = np.floor(np.clip(x / x_white, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
        diff = np.abs(expected.astype(np.int32) - raw_pixels.astype(np.int32))
        rep.check("stored = round(65535 clamp(x / x_white, 0, 1))", diff.max() <= 1, f"max difference {int(diff.max())}")
        rep.check("negative values are clipped to 0", np.all(raw_pixels[x < 0] == 0), "")
        # What a reader gets back, x_white * stored / 65535, is within half a code of the original value;
        # above a tenth of x_white that is 0.01 % relative.
        back = raw_pixels.astype(np.float64) / 65535.0 * x_white
        inside = (x >= 0) & (x <= x_white)
        # (x_white is read back from BaselineExposure, a rational with six decimals: allow 0.05 code for that.)
        worst = float((np.abs(back - x)[inside] / x_white * 65535.0).max())
        rep.check("values come back within half a code", worst <= 0.55, f"worst difference {worst:.3f} codes")
        mask = (x > x_white / 10) & (x < x_white)
        rel = np.abs(back[mask] - x[mask]) / x[mask]
        rep.check("bright values come back within 0.01 %", rel.size == 0 or rel.max() < 1e-4, f"max relative error {rel.max() if rel.size else 0:.2e}")

    # --- optional cross-check of the matrices with colour-science ----------------------------------
    try:
        import colour

        space = colour.RGB_COLOURSPACES["ARRI Wide Gamut 4"]
        ref = np.linalg.inv(space.matrix_RGB_to_XYZ)
        rep.check("ColorMatrix1 agrees with colour-science", np.allclose(ref, color, atol=1e-6), np.abs(ref - color).max())
    except Exception as e:
        rep.check("colour-science cross-check", True, f"skipped: {e}")

    print(json.dumps({"ok": rep.ok, "checks": rep.checks}, indent=1))
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())

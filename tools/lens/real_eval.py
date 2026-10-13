#!/usr/bin/env python3
"""Real-file evaluation of the lens tables of `docs/05-raw-processing.md` sections R3 and R9.

Local evaluation only: it reads real ARW files (never committed, never copied) and writes its
numbers and crops under `testdata/lens-eval/` (git-ignored).  Needs numpy, scipy, rawpy (LibRaw is
only the pixel oracle here), pillow and exiftool on the path.

    python3 -I tools/lens/real_eval.py tables  FILE.ARW ...        # the lens tags, both copies
    python3 -I tools/lens/real_eval.py raw-ca  FILE.ARW ...        # signed radial CA of the mosaic
    python3 -I tools/lens/real_eval.py jpeg-ca FILE.ARW ...        # the same on the camera's own JPEG

The lens frame is the one the render uses (`ym info`: the camera crop, else the active area, else
the whole readout) unless `--frame x0,y0,w,h` is given.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ca_measure as cam  # noqa: E402

OUT = HERE.parents[1] / "testdata" / "lens-eval"


# ---------------------------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------------------------


def exif_tags(path: str) -> dict:
    out = subprocess.run(
        ["exiftool", "-j", "-a", "-u", "-G1", "-n", path], capture_output=True, text=True, check=True
    ).stdout
    return json.loads(out)[0]


def ints(s) -> list[int]:
    return [int(v) for v in str(s).split()]


def lens_tables(tags: dict) -> dict:
    """The three tables of both copies: `raw` is the raw sub-IFD (what ym reads: count first, 128-step
    CA), `mn` the maker note's copy (16 knots, no count, finer CA)."""

    def get(*keys):
        for k in keys:
            if k in tags:
                return ints(tags[k])
        return None

    raw = dict(
        vignetting=get("SubIFD:VignettingCorrParams", "SR2SubIFD:VignettingCorrParams"),
        ca=get("SubIFD:ChromaticAberrationCorrParams", "SR2SubIFD:ChromaticAberrationCorrParams"),
        distortion=get("SubIFD:DistortionCorrParams", "SR2SubIFD:DistortionCorrParams"),
    )
    mn = dict(
        vignetting=get("Sony:VignettingCorrParams"),
        ca=get("Sony:ChromaticAberrationCorrParams"),
        distortion=get("Sony:DistortionCorrParams"),
    )
    return dict(raw=raw, mn=mn)


def split_stored(t):
    """(count, knots) of a stored table with the leading count."""
    if not t:
        return None, []
    n = t[0]
    return n, t[1 : 1 + n]


def frame_of(path: str, tags: dict) -> tuple[int, int, int, int]:
    """The lens frame (x0, y0, w, h) in readout coordinates, as the render picks it (crop > active
    area > readout)."""
    crop = tags.get("SubIFD:SonyCropTopLeft") or tags.get("SubIFD:DefaultCropOrigin")
    size = tags.get("SubIFD:SonyCropSize") or tags.get("SubIFD:DefaultCropSize")
    if crop and size:
        x0, y0 = ints(crop)
        w, h = ints(size)
        return x0, y0, w, h
    raw = tags.get("SonyRawImageSize") or tags.get("SubIFD:SonyRawImageSize")
    if raw:
        w, h = ints(raw)
        return 0, 0, w, h
    return 0, 0, int(tags["SubIFD:ImageWidth"]), int(tags["SubIFD:ImageHeight"])


# ---------------------------------------------------------------------------------------------
# Model of the tables (the readings under test)
# ---------------------------------------------------------------------------------------------


def knots_x(n: int, offset: float = 0.5, denom: float | None = None, scale: float = 1.0):
    d = (n - 1) if denom is None else denom
    return scale * (np.arange(n) + offset) / d


def interp(xs, ys, r):
    return np.interp(r, xs, ys)


def distortion_scale(knots, r, alpha=1.0, offset=0.5, denom=None, kscale=1.0):
    """F(r): source radius = r * F(r)."""
    if not len(knots):
        return np.ones_like(r)
    f = alpha * np.asarray(knots, dtype=np.float64) * 2.0**-14 + 1.0
    return interp(knots_x(len(f), offset, denom, kscale), f, r)


def ca_k(ca, r, alpha=1.0, offset=0.5, denom=None, kscale=1.0):
    n = len(ca) // 2
    kr = alpha * np.asarray(ca[:n], dtype=np.float64) * 2.0**-21 + 1.0
    kb = alpha * np.asarray(ca[n:], dtype=np.float64) * 2.0**-21 + 1.0
    xs = knots_x(n, offset, denom, kscale)
    return interp(xs, kr, r), interp(xs, kb, r)


def predicted_shift(rho_src, ca, dist, hd, *, alpha=1.0, offset=0.5, denom=None, kscale=1.0,
                    dist_alpha=1.0, k_at_source=False):
    """Predicted raw shift (px, outward positive) of red and blue against green at the green source
    radius `rho_src` (fraction of the half diagonal `hd`), when the table is read the way R9 reads it:
    colour X sits at r_out * F * K_X, green at r_out * F, K evaluated at the output radius r_out.
    With `k_at_source`, K is evaluated at the source radius instead."""
    rho_src = np.asarray(rho_src, dtype=np.float64)
    if len(dist):
        r_out_grid = np.linspace(0.0, 1.3, 2601)
        f_grid = distortion_scale(dist, r_out_grid, dist_alpha, offset, denom, kscale)
        src_grid = r_out_grid * f_grid
        r_out = np.interp(rho_src, src_grid, r_out_grid)
    else:
        r_out = rho_src
    at = rho_src if k_at_source else r_out
    kr, kb = ca_k(ca, at, alpha, offset, denom, kscale)
    return rho_src * hd * (kr - 1.0), rho_src * hd * (kb - 1.0)


# ---------------------------------------------------------------------------------------------
# Raw data
# ---------------------------------------------------------------------------------------------


def load_mosaic(path: str, frame: tuple[int, int, int, int]):
    import rawpy

    rp = rawpy.imread(path)
    raw = rp.raw_image.astype(np.float64)
    colors = rp.raw_colors.astype(np.int32)
    x0, y0, w, h = frame
    black = np.array(rp.black_level_per_channel, dtype=np.float64)
    bl = np.zeros_like(raw)
    for c in range(4):
        bl[colors == c] = black[c]
    sub = (raw - bl)[y0 : y0 + h, x0 : x0 + w]
    col = colors[y0 : y0 + h, x0 : x0 + w]
    white = float(rp.white_level) - float(black.max())
    return sub, col, white


def fmt_profile(rows, pred=None):
    lines = []
    for r in rows:
        if not r["n"]:
            continue
        p = ""
        if pred is not None:
            p = f"  pred {float(pred(np.array([r['rho']]))[0]):+.3f}"
        lines.append(f"    rho {r['rho']:.2f} (n={r['n']:3d})  shift {r['mean_px']:+.3f} +- {r['se_px']:.3f}{p}")
    return "\n".join(lines)


def cmd_tables(paths):
    for p in paths:
        tags = exif_tags(p)
        t = lens_tables(tags)
        frame = frame_of(p, tags)
        print(f"== {Path(p).name}  frame {frame}  lens {tags.get('ExifIFD:LensModel')}  f={tags.get('ExifIFD:FocalLength')}")
        for kind in ("vignetting", "ca", "distortion"):
            n, kn = split_stored(t["raw"][kind])
            print(f"  raw  {kind:10s} n={n} {kn}")
            print(f"  mn   {kind:10s} {t['mn'][kind]}")


def cmd_raw_ca(paths, args):
    OUT.mkdir(parents=True, exist_ok=True)
    for p in paths:
        tags = exif_tags(p)
        t = lens_tables(tags)
        frame = frame_of(p, tags)
        mosaic, colors, white = load_mosaic(p, frame)
        h, w = mosaic.shape
        centre = (w / 2.0, h / 2.0)
        hd = 0.5 * math.hypot(w, h)
        res = cam.measure_mosaic(mosaic, colors, clip_level=0.97 * white, centre=centre, block_px=args["block"])
        n_ca, ca = split_stored(t["raw"]["ca"])
        n_d, dist = split_stored(t["raw"]["distortion"])
        edges = np.linspace(0.0, 1.0, 11)
        name = Path(p).stem
        print(f"== {name}  frame {w}x{h}  blocks red {len(res['red'])} blue {len(res['blue'])}", flush=True)
        out = {}
        for chan in ("red", "blue"):
            prof = cam.radial_profile(res[chan], hd, edges)
            idx = 0 if chan == "red" else 1
            pred = (lambda r, idx=idx: predicted_shift(r, ca, dist, hd)[idx]) if ca else None
            print(f"  {chan}:")
            print(fmt_profile(prof, pred), flush=True)
            out[chan] = prof
        rec = dict(
            file=name, frame=[w, h], hd=hd, tables=t, profiles=out,
            blocks={c: [[f.x, f.y, f.shift_px, f.se_px, f.gain, f.correlation] for f in res[c]] for c in res},
        )
        (OUT / f"raw_ca_{name}.json").write_text(json.dumps(rec))


def cmd_jpeg_ca(paths, args):
    """CA of the camera's own JPEG (the preview, or the full-size one with --full), in raw pixels."""
    import io

    from PIL import Image

    OUT.mkdir(parents=True, exist_ok=True)
    for p in paths:
        tags = exif_tags(p)
        t = lens_tables(tags)
        frame = frame_of(p, tags)
        w, h = frame[2], frame[3]
        hd = 0.5 * math.hypot(w, h)
        tag = "-JpgFromRaw" if args["full"] else "-PreviewImage"
        data = subprocess.run(["exiftool", "-b", tag, p], capture_output=True, check=True).stdout
        im = Image.open(io.BytesIO(data)).convert("RGB")
        a = np.asarray(im, dtype=np.float64) / 255.0
        a = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
        hj, wj, _ = a.shape
        hdj = 0.5 * math.hypot(wj, hj)
        s0 = hd / hdj
        sigma = 2.5 if wj < 3000 else 5.0
        block = 32 if wj < 3000 else 64
        res = cam.measure_rgb(a, (wj / 2.0, hj / 2.0), block_px=block, highpass_sigma_px=sigma)
        name = Path(p).stem
        print(f"== {name} {'full' if args['full'] else 'preview'} jpeg {wj}x{hj} scale {s0:.3f} blocks red {len(res['red'])} blue {len(res['blue'])}", flush=True)
        n_ca, ca = split_stored(t["raw"]["ca"])
        n_d, dist = split_stored(t["raw"]["distortion"])
        out = {}
        for chan in ("red", "blue"):
            prof = cam.radial_profile(res[chan], hdj, np.linspace(0.0, 1.0, 11))
            for r in prof:  # report in raw pixels
                if r["n"]:
                    r["mean_px"] *= s0
                    r["se_px"] *= s0
                    r["median_px"] *= s0
            idx = 0 if chan == "red" else 1
            pred = (lambda r, idx=idx: predicted_shift(r, ca, dist, hd)[idx]) if ca else None
            print(f"  {chan} (shifts in raw px):")
            print(fmt_profile(prof, pred), flush=True)
            out[chan] = prof
        rec = dict(file=name, full=args["full"], jpeg=[wj, hj], scale=s0, profiles=out)
        (OUT / f"jpeg_ca_{name}{'_full' if args['full'] else ''}.json").write_text(json.dumps(rec))


def main():
    argv = sys.argv[1:]
    if not argv:
        print(__doc__)
        return 2
    cmd, rest = argv[0], argv[1:]
    args = dict(block=96, full="--full" in rest)
    files = [a for a in rest if not a.startswith("--")]
    if cmd == "tables":
        cmd_tables(files)
    elif cmd == "raw-ca":
        cmd_raw_ca(files, args)
    elif cmd == "jpeg-ca":
        cmd_jpeg_ca(files, args)
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

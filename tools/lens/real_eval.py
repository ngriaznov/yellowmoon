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


def read_f32(path):
    """`lens_eval_dump` output: u32 w, u32 h, then the R, G, B planes as f32."""
    a = np.fromfile(path, dtype="<u4", count=2)
    w, h = int(a[0]), int(a[1])
    d = np.fromfile(path, dtype="<f4", offset=8).reshape(3, h, w)
    return np.moveaxis(d, 0, -1)


def cmd_rgb_ca(paths, args):
    """CA left in the camera-RGB images `lens_eval_dump` wrote (demosaic only, after R9 with and
    without the CA correction, after R5b, and with `--chain` the shipped chain in three orders),
    against what the mosaic itself shows (`raw-ca` output)."""
    for p in paths:  # p: the dump directory prefix, STEM:METHOD
        stem, method = p.split(":")
        base = Path(args["dump_dir"])
        meta = json.loads((base / f"{stem}.{method}.json").read_text())
        fw, fh = meta["frame"]
        cx0, cy0, cw, ch = meta["crop"]
        hd = 0.5 * math.hypot(fw, fh)
        print(f"== {stem} {method}  frame {fw}x{fh} crop {cw}x{ch} at {cx0},{cy0}", flush=True)
        out_centre = (fw / 2.0 - cx0, fh / 2.0 - cy0)
        # `demosaic` is the whole frame; the others are the cropped output of R9. `chain`,
        # `chainold` and `chainnoca` (`lens_eval_dump --chain`) are the shipped full-resolution chain
        # with R5b, with the CA left in R9 (the order before R5b) and with no CA correction.
        for kind, centre in (("demosaic", (fw / 2.0, fh / 2.0)), ("r9noca", out_centre), ("r9", out_centre), ("r9mosaic", out_centre), ("chainnoca", out_centre), ("chainold", out_centre), ("chain", out_centre)):
            if not (base / f"{stem}.{method}.{kind}.f32").exists():
                continue
            img = read_f32(base / f"{stem}.{method}.{kind}.f32")
            res = cam.measure_rgb(img.astype(np.float64), centre, clip_level=10.0, block_px=args["block"], highpass_sigma_px=5.0, max_se_px=0.25, min_energy=0.004)
            out = {}
            print(f"  {kind}: blocks red {len(res['red'])} blue {len(res['blue'])}")
            for chan in ("red", "blue"):
                prof = cam.radial_profile(res[chan], hd, np.linspace(0.0, 1.0, 11))
                out[chan] = prof
                print(f"    {chan}: " + "  ".join(f"{r['rho']:.2f}:{r['mean_px']:+.3f}" for r in prof if r["n"]), flush=True)
            (OUT / f"rgb_ca_{stem}_{method}_{kind}.json").write_text(json.dumps(out))


def vignette_gain_model(knots, rho):
    """The R3 reading of `docs/05`: the photosite is divided by G(r) = 2^(0.5 - 2^(v 2^-13 - 1)),
    knot i of n at (i + 0.5) / (n - 1), linear in G.  Returns the multiplier 1 / G(rho)."""
    v = np.asarray(knots, dtype=np.float64)
    g = 2.0 ** (0.5 - 2.0 ** (v * 2.0**-13 - 1.0))
    return 1.0 / np.maximum(np.interp(rho, knots_x(len(v)), g), 1e-4)


def cmd_vignette(paths, args):
    """The radial gain the camera's JPEG has over the raw (tone curve fitted out) against the R3
    gain of the raw sub-IFD table.  Needs `jpeg_geometry.py match` output for the distortion map."""
    import io

    from PIL import Image

    import jpeg_geometry as jg

    for p in paths:
        tags = exif_tags(p)
        t = lens_tables(tags)
        frame = frame_of(p, tags)
        mosaic, colors, white = load_mosaic(p, frame)
        h, w = mosaic.shape
        hd = 0.5 * math.hypot(w, h)
        name = Path(p).stem
        geo = json.loads((OUT / f"geo_{name}.json").read_text())
        fit = jg.fit_one(geo, "raw", False)
        zeta = fit["zeta"] if fit else 1.0
        alpha = max(fit["alpha"], 0.0) if fit else 0.0
        n_d, dist = split_stored(t["raw"]["distortion"])
        data = subprocess.run(["exiftool", "-b", "-PreviewImage", p], capture_output=True, check=True).stdout
        im = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.float64) / 255.0
        hj, wj, _ = im.shape
        hdj = 0.5 * math.hypot(wj, hj)
        s0 = hd / hdj
        jl = np.where(im[..., 1] <= 0.04045, im[..., 1] / 12.92, ((im[..., 1] + 0.055) / 1.055) ** 2.4)
        is_green = (colors == 1) | (colors == 3)
        green = cam.bilinear_green(mosaic, is_green)
        from scipy import ndimage as ndi

        lp = ndi.gaussian_filter(green, 1.0 * s0)
        yy, xx = np.mgrid[0:hj, 0:wj]
        qx, qy = xx + 0.5 - wj / 2.0, yy + 0.5 - hj / 2.0
        rj = np.hypot(qx, qy) / hdj
        rho_raw = jg.model_rho_raw(rj, None, dist, zeta, alpha, 1.0, denom=n_d - 1) if len(dist) else zeta * rj
        scale = np.where(rj > 1e-9, rho_raw / np.maximum(rj, 1e-9), zeta)
        px = w / 2.0 + qx * s0 * scale
        py = h / 2.0 + qy * s0 * scale
        raw_j = ndi.map_coordinates(lp, [py - 0.5, px - 0.5], order=1, mode="nearest") / (0.97 * white)
        # a 3 x 3 block mean to suppress registration noise
        raw_b = ndi.uniform_filter(raw_j, 9)
        jl_b = ndi.uniform_filter(jl, 9)
        ok = (raw_b > 0.01) & (raw_b < 0.85) & (jl_b > 0.003) & (jl_b < 0.85)
        # The JPEG is J = T(raw * g(rho)) with one tone curve T for the whole frame.  T is fitted
        # as a monotone curve (isotonic regression of log J on log raw + log g) and g(rho) per
        # radial band, alternating; g = 1 in the middle band.
        from scipy.optimize import isotonic_regression

        edges = np.linspace(0.0, 1.0, 11)
        x = np.log(raw_b[ok])
        y = np.log(jl_b[ok])
        rr = rho_raw[ok]
        band = np.clip(np.digitize(rr, edges) - 1, 0, 9)
        # subsample for speed
        rng = np.random.default_rng(1)
        pick = rng.random(len(x)) < min(1.0, 400000 / len(x))
        x, y, rr, band = x[pick], y[pick], rr[pick], band[pick]
        e = np.zeros(10)
        for _ in range(12):
            z = x + e[band]
            order = np.argsort(z)
            zs, ys = z[order], y[order]
            iso = isotonic_regression(ys, increasing=True).x
            # strictly increasing for the inverse
            iso = iso + 1e-9 * np.arange(len(iso))
            # T^-1(y): the z whose fitted value is y
            z_of_y = np.interp(y, iso, zs)
            new_e = e.copy()
            for k in range(10):
                m = band == k
                if m.sum() > 300:
                    new_e[k] = float(np.median(z_of_y[m] - x[m]))
            new_e -= new_e[0:3].mean()
            if np.abs(new_e - e).max() < 1e-4:
                e = new_e
                break
            e = 0.5 * e + 0.5 * new_e
        print(f"== {name}  vignetting setting {tags.get('Sony:VignettingCorrection')}")
        n_v, vig = split_stored(t["raw"]["vignetting"])
        for k in range(10):
            m = band == k
            if m.sum() < 300:
                continue
            rho_c = float(np.median(rr[m]))
            g_mod = float(vignette_gain_model(vig, np.array([rho_c]))[0]) if vig else float("nan")
            print(f"    rho {rho_c:.2f}  camera gain {math.exp(e[k]):6.3f}   table (R3) gain {g_mod:6.3f}   n={int(m.sum())}")


def cmd_noise_radius(paths, args):
    """Is the vignetting already in the raw data?  If the camera multiplied the raw by a shading
    gain g(rho), the photon-noise slope `a` of sigma^2 = a * signal + b grows with the gain; if the
    corners are simply darker, the slope is the same at every radius.  Measured on the green
    photosites of smooth blocks, per radial band, against the R3 gain of the table."""
    from scipy import ndimage as ndi

    for p in paths:
        tags = exif_tags(p)
        t = lens_tables(tags)
        frame = frame_of(p, tags)
        mosaic, colors, white = load_mosaic(p, frame)
        h, w = mosaic.shape
        hd = 0.5 * math.hypot(w, h)
        name = Path(p).stem
        g1 = (colors == 1)
        # the first green of every 2 x 2 cell, as a half-resolution plane
        ys, xs = np.nonzero(g1[::1, ::1])
        plane = np.full((h // 2, w // 2), np.nan)
        # green sites on even rows (first green) only
        sel = (colors == 1)
        gy, gx = np.nonzero(sel)
        keep = (gy % 2 == (gy[0] % 2)) & (gx % 2 == (gx[0] % 2))
        gy, gx = gy[keep], gx[keep]
        plane_h, plane_w = h // 2, w // 2
        arr = np.zeros((plane_h, plane_w))
        arr[np.minimum(gy // 2, plane_h - 1), np.minimum(gx // 2, plane_w - 1)] = mosaic[gy, gx]
        lap = ndi.convolve(arr, np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64), mode="reflect")
        b = 12  # block of 12 x 12 first-green samples (24 x 24 photosites)
        bh, bw = plane_h // b, plane_w // b
        rows = []
        for j in range(1, bh - 1):
            for i in range(1, bw - 1):
                blk = arr[j * b : (j + 1) * b, i * b : (i + 1) * b]
                l = lap[j * b + 1 : (j + 1) * b - 1, i * b + 1 : (i + 1) * b - 1]
                if blk.max() > 0.9 * white:
                    continue
                sigma = math.sqrt(math.pi / 2.0) / 6.0 * float(np.abs(l).mean())
                # smoothness: the block's own low-order variation must be small against the noise
                gy_, gx_ = np.gradient(ndi.gaussian_filter(blk, 1.5))
                trend = float(np.hypot(gy_, gx_).mean())
                cx_ = (i + 0.5) * b * 2.0
                cy_ = (j + 0.5) * b * 2.0
                rho = math.hypot(cx_ - w / 2.0, cy_ - h / 2.0) / hd
                rows.append((rho, float(blk.mean()), sigma, trend))
        a = np.array(rows)
        n_v, vig = split_stored(t["raw"]["vignetting"])
        print(f"== {name} vignetting setting {tags.get('Sony:VignettingCorrection')}  ISO {tags.get('ExifIFD:ISO')}  blocks {len(a)}")
        # smooth blocks only
        thr = np.quantile(a[:, 3], 0.6)
        a = a[a[:, 3] <= thr]
        edges = np.array([0.0, 0.25, 0.45, 0.6, 0.75, 0.9, 1.0])
        levels = np.array([60, 120, 240, 480, 960, 1920, 3840, 7680, 15360], dtype=np.float64)
        lv = np.digitize(a[:, 1], levels)
        var = a[:, 2] ** 2
        med = {}
        for k in range(len(edges) - 1):
            for j in range(1, len(levels)):
                m = (a[:, 0] >= edges[k]) & (a[:, 0] < edges[k + 1]) & (lv == j)
                if m.sum() >= 12:
                    med[(k, j)] = float(np.quantile(var[m], 0.2))
        for k in range(1, len(edges) - 1):
            ratios = [med[(k, j)] / med[(0, j)] for j in range(1, len(levels)) if (k, j) in med and (0, j) in med]
            m = (a[:, 0] >= edges[k]) & (a[:, 0] < edges[k + 1])
            rho_c = float(np.median(a[m, 0])) if m.any() else float("nan")
            g_mod = float(vignette_gain_model(vig, np.array([rho_c]))[0]) if vig else float("nan")
            if args.get("verbose"):
                per = [(int(levels[j - 1]), round(med[(k, j)] / med[(0, j)], 2)) for j in range(1, len(levels)) if (k, j) in med and (0, j) in med]
                print(f"      per level (DN above black, ratio): {per}")
            if ratios:
                print(
                    f"    rho {rho_c:.2f}  noise variance (20th percentile of smooth blocks) at equal level, ratio to the centre: {np.median(ratios):5.2f}"
                    f" (over {len(ratios)} levels)   R3 gain {g_mod:5.2f}"
                )


def cmd_sky_profile(paths, args):
    """Is the vignetting already in the raw?  The even part (left/right average) of the horizontal
    brightness profile of a sky band is free of the sky's own, nearly linear, gradient; it is the
    vignetting if the raw is uncorrected and flat if the camera has already corrected it.  Compared
    with the R3 gain of the table.  `--rows=a,b` are fractions of the height."""
    from scipy import ndimage as ndi

    rows = (0.05, 0.3)
    for a in args["rest"]:
        if a.startswith("--rows="):
            rows = tuple(float(v) for v in a.split("=", 1)[1].split(","))
    for p in paths:
        tags = exif_tags(p)
        t = lens_tables(tags)
        frame = frame_of(p, tags)
        mosaic, colors, white = load_mosaic(p, frame)
        h, w = mosaic.shape
        hd = 0.5 * math.hypot(w, h)
        is_green = (colors == 1) | (colors == 3)
        g = cam.bilinear_green(mosaic, is_green)
        y0, y1 = int(rows[0] * h), int(rows[1] * h)
        band = g[y0:y1]
        ok = (band < 0.9 * white).all(axis=0)
        prof = np.log(np.maximum(band.mean(axis=0), 1.0))
        prof = ndi.uniform_filter1d(prof, 61)
        n_v, vig = split_stored(t["raw"]["vignetting"])
        print(f"== {Path(p).stem} sky rows {y0}..{y1}, vignetting flag {tags.get('SubIFD:VignettingCorrection')}  (even part of log brightness vs table)")
        yc = 0.5 * (y0 + y1)
        out = []
        for frac in (0.1, 0.2, 0.3, 0.4, 0.45):
            xl, xr = int(w * (0.5 - frac)), int(w * (0.5 + frac))
            even = 0.5 * (prof[xl] + prof[xr]) - prof[w // 2]
            rho_l = math.hypot(xl - w / 2, yc - h / 2) / hd
            rho_r = math.hypot(xr - w / 2, yc - h / 2) / hd
            gl, gr = (float(vignette_gain_model(vig, np.array([r]))[0]) for r in (rho_l, rho_r))
            model_even = -0.5 * (math.log(gl) + math.log(gr)) + math.log(float(vignette_gain_model(vig, np.array([abs(yc - h / 2) / hd]))[0]))
            out.append((frac, even, model_even))
            print(f"    x = {0.5 - frac:.2f} / {0.5 + frac:.2f}   measured even part {even:+.3f}   if uncorrected (table) {model_even:+.3f}   rho {rho_l:.2f}/{rho_r:.2f}")


def cmd_pfm_ca(paths, args):
    """CA before and after darktable's lens module (`darktable_oracle.sh --lens-flags ca`): `paths` are
    its output directories (`lens_in.pfm.gz`, `lens_out.pfm.gz`), given as DIR:FILE.ARW so that the
    table can be shown next to the measurement."""
    import lens_io

    for spec in paths:
        d, arw = spec.split(":", 1)
        tags = exif_tags(arw)
        t = lens_tables(tags)
        n_ca, ca = split_stored(t["raw"]["ca"])
        n_d, dist = split_stored(t["raw"]["distortion"])
        img_in, img_out = lens_io.load_pair(d)
        for name, img in (("lens module input", img_in), ("lens module output", img_out)):
            h, w, _ = img.shape
            hd = 0.5 * math.hypot(w, h)
            sc = img.max()
            res = cam.measure_rgb(img / sc, (w / 2.0, h / 2.0), clip_level=0.99, block_px=96, highpass_sigma_px=5.0, min_energy=0.004)
            print(f"== {Path(arw).stem}  darktable {name}  {w}x{h}  blocks red {len(res['red'])} blue {len(res['blue'])}", flush=True)
            for chan in ("red", "blue"):
                prof = cam.radial_profile(res[chan], hd, np.linspace(0.0, 1.0, 11))
                idx = 0 if chan == "red" else 1
                pred = (lambda r, idx=idx: predicted_shift(r, ca, dist, hd)[idx]) if ca else None
                print(f"  {chan}:")
                print(fmt_profile(prof, pred), flush=True)


def main():
    argv = sys.argv[1:]
    if not argv:
        print(__doc__)
        return 2
    cmd, rest = argv[0], argv[1:]
    args = dict(block=96, full="--full" in rest, verbose="--verbose" in rest, dump_dir=str(HERE.parents[1] / "testdata" / "lens-eval" / "dump"))
    for a in rest:
        if a.startswith("--dump-dir="):
            args["dump_dir"] = a.split("=", 1)[1]
    files = [a for a in rest if not a.startswith("--")]
    if cmd == "tables":
        cmd_tables(files)
    elif cmd == "raw-ca":
        cmd_raw_ca(files, args)
    elif cmd == "jpeg-ca":
        cmd_jpeg_ca(files, args)
    elif cmd == "rgb-ca":
        cmd_rgb_ca(files, args)
    elif cmd == "pfm-ca":
        cmd_pfm_ca(files, args)
    elif cmd == "vignette":
        cmd_vignette(files, args)
    elif cmd == "sky-profile":
        args["rest"] = rest
        cmd_sky_profile(files, args)
    elif cmd == "noise-radius":
        cmd_noise_radius(files, args)
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

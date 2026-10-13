#!/usr/bin/env python3
"""Radial geometry of the camera's own JPEG against the raw mosaic of the same ARW file.

Local evaluation only (real files, never committed).  The embedded JPEG (`PreviewImage`, 1616 px
wide, or the full-size `JpgFromRaw` of newer bodies) is the camera's render of the raw data with the
corrections its settings ask for.  Matching it block by block against the raw (green plane, scaled to
the JPEG's size, no lens correction) gives, for every block of the JPEG at radius `rho_j`, the radius
`rho_raw` of the same detail in the raw frame.  That is the camera's own distortion function (up to a
zoom), independent of how anyone reads the tables:

    rho_raw = zeta * rho_j * F(zeta * rho_j)        (F: the table as read, `zeta`: the camera's zoom)

`fit` finds the table amplitude `alpha` (F = 1 + alpha * d * 2^-14) and the knot scale `kappa` that
reproduce it, for the raw sub-IFD copy of the table (what the render reads) and the maker note's copy.

    python3 -I tools/lens/jpeg_geometry.py match FILE.ARW [--full]    # writes testdata/lens-eval/geo_<stem>.json
    python3 -I tools/lens/jpeg_geometry.py fit   STEM ...
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy import optimize, signal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ca_measure as cam  # noqa: E402
import real_eval as re_  # noqa: E402

OUT = re_.OUT


def extract_jpeg(path: str, full: bool) -> np.ndarray:
    tag = "-JpgFromRaw" if full else "-PreviewImage"
    data = subprocess.run(["exiftool", "-b", tag, path], capture_output=True, check=True).stdout
    if not data:
        raise SystemExit(f"{path}: no {tag[1:]}")
    from PIL import Image
    import io

    im = Image.open(io.BytesIO(data)).convert("RGB")
    return np.asarray(im, dtype=np.float64) / 255.0


def srgb_decode(v: np.ndarray) -> np.ndarray:
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def ncc_shift(block: np.ndarray, search: np.ndarray, max_shift: int):
    """Position of `block` inside `search` (which is larger by `max_shift` on each side): the
    normalised cross-correlation peak with a quadratic sub-pixel refinement.  Returns (dx, dy,
    peak) relative to the centred position, or None."""
    bh, bw = block.shape
    b = block - block.mean()
    bn = math.sqrt(float((b * b).sum()))
    if bn < 1e-9:
        return None
    num = signal.fftconvolve(search, b[::-1, ::-1], mode="valid")
    ones = np.ones_like(b)
    s1 = signal.fftconvolve(search, ones, mode="valid")
    s2 = signal.fftconvolve(search * search, ones, mode="valid")
    var = np.maximum(s2 - s1 * s1 / b.size, 1e-12)
    ncc = num / (bn * np.sqrt(var))
    iy, ix = np.unravel_index(np.argmax(ncc), ncc.shape)
    peak = float(ncc[iy, ix])
    if not (1 <= ix < ncc.shape[1] - 1 and 1 <= iy < ncc.shape[0] - 1):
        return None

    def vertex(a, b_, c):
        den = a - 2 * b_ + c
        return 0.0 if abs(den) < 1e-12 else 0.5 * (a - c) / den

    fx = vertex(ncc[iy, ix - 1], ncc[iy, ix], ncc[iy, ix + 1])
    fy = vertex(ncc[iy - 1, ix], ncc[iy, ix], ncc[iy + 1, ix])
    return (ix + fx - max_shift, iy + fy - max_shift, peak)


def cmd_match(path: str, full: bool):
    OUT.mkdir(parents=True, exist_ok=True)
    tags = re_.exif_tags(path)
    frame = re_.frame_of(path, tags)
    mosaic, colors, white = re_.load_mosaic(path, frame)
    h, w = mosaic.shape
    jpg = extract_jpeg(path, full)
    hj, wj, _ = jpg.shape
    hd = 0.5 * math.hypot(w, h)
    hdj = 0.5 * math.hypot(wj, hj)
    s0 = hd / hdj
    print(f"{Path(path).name}: frame {w}x{h}, jpeg {wj}x{hj}, scale {s0:.4f}", flush=True)

    is_green = (colors == 1) | (colors == 3)
    green = cam.bilinear_green(mosaic, is_green)
    lp = ndi.gaussian_filter(green, 0.45 * s0)
    # the raw, sampled on the JPEG grid with no correction: p = c + (q - c_j) * s0
    yy, xx = np.mgrid[0:hj, 0:wj]
    px = w / 2.0 + (xx + 0.5 - wj / 2.0) * s0
    py = h / 2.0 + (yy + 0.5 - hj / 2.0) * s0
    raw_on_j = ndi.map_coordinates(lp, [py - 0.5, px - 0.5], order=1, mode="nearest")
    raw_on_j = np.clip(raw_on_j / (0.97 * white), 0, 1)
    raw_g = (raw_on_j) ** (1 / 2.2)
    jl = jpg[..., 1]  # the JPEG's green channel: same plane as the raw's green
    block = 40 if wj < 3000 else 96
    max_shift = int(0.04 * wj) if wj < 3000 else int(0.03 * wj / 2)
    rows = []
    cxj, cyj = wj / 2.0, hj / 2.0
    for by in range(max_shift, hj - block - max_shift + 1, block // 2):
        for bx in range(max_shift, wj - block - max_shift + 1, block // 2):
            blk = jl[by : by + block, bx : bx + block]
            if blk.std() < 0.02:
                continue
            sr = raw_g[by - max_shift : by + block + max_shift, bx - max_shift : bx + block + max_shift]
            res = ncc_shift(blk, sr, max_shift)
            if res is None:
                continue
            dx, dy, peak = res
            qx, qy = bx + block / 2.0, by + block / 2.0
            rows.append([qx, qy, dx, dy, peak, float(blk.std())])
    arr = np.array(rows)
    rec = dict(
        file=Path(path).stem, full=full, frame=[w, h], jpeg=[wj, hj], scale=s0, hd=hd, hdj=hdj,
        tables=re_.lens_tables(tags), blocks=arr.tolist(),
    )
    (OUT / f"geo_{Path(path).stem}{'_full' if full else ''}.json").write_text(json.dumps(rec))
    good = arr[arr[:, 4] > 0.6]
    print(f"  {len(arr)} blocks, {len(good)} with NCC > 0.6", flush=True)


# ---------------------------------------------------------------------------------------------


def model_rho_raw(rho_j, ca_unused, dist, zeta, alpha, kappa, offset=0.5, denom=None):
    """rho_raw = zeta rho_j F(zeta rho_j) with F read from the table."""
    r_out = zeta * rho_j
    f = re_.distortion_scale(dist, r_out, alpha, offset=offset, denom=denom, kscale=kappa)
    return r_out * f


def fit_one(rec: dict, source: str, fit_kappa: bool, min_peak=0.7, rho_min=0.15):
    arr = np.array(rec["blocks"])
    arr = arr[arr[:, 4] > min_peak]
    wj, hj = rec["jpeg"]
    s0, hd = rec["scale"], rec["hd"]
    hdj = rec["hdj"]
    cxj, cyj = wj / 2.0, hj / 2.0
    qx, qy = arr[:, 0] - cxj, arr[:, 1] - cyj
    r_j = np.hypot(qx, qy)
    ux, uy = qx / r_j, qy / r_j
    # a global translation (the JPEG's centre differs from the frame's) from the central blocks
    cen = r_j / hdj < 0.3
    tx = float(np.median(arr[cen, 2])) if cen.sum() > 5 else 0.0
    ty = float(np.median(arr[cen, 3])) if cen.sum() > 5 else 0.0
    dx, dy = arr[:, 2] - tx, arr[:, 3] - ty
    rr = np.hypot(qx + dx, qy + dy)
    rho_j = r_j / hdj
    rho_raw = rr * s0 / hd
    m = rho_j > rho_min
    rho_j, rho_raw = rho_j[m], rho_raw[m]
    t = rec["tables"]
    n_raw, dist_raw = re_.split_stored(t["raw"]["distortion"])
    if source == "mn":
        mn = t["mn"]["distortion"]
        dist = mn[:n_raw] if mn else dist_raw
    else:
        dist = dist_raw
    if not len(dist):
        return None
    denom = n_raw - 1

    def resid(p):
        zeta, alpha = p[0], p[1]
        kappa = p[2] if fit_kappa else 1.0
        return rho_raw - model_rho_raw(rho_j, None, dist, zeta, alpha, kappa, denom=denom)

    x0 = [1.0, 1.0] + ([1.0] if fit_kappa else [])
    sol = optimize.least_squares(resid, x0, loss="soft_l1", f_scale=0.003)
    r = resid(sol.x)
    rms_px = float(np.sqrt(np.mean(r**2))) * hd
    return dict(
        zeta=float(sol.x[0]), alpha=float(sol.x[1]), kappa=float(sol.x[2]) if fit_kappa else 1.0,
        rms_raw_px=rms_px, n=int(m.sum()), shift=(tx, ty),
    )


def cmd_fit(stems):
    for stem in stems:
        path = OUT / f"geo_{stem}.json"
        rec = json.loads(path.read_text())
        t = rec["tables"]
        n_raw, dist_raw = re_.split_stored(t["raw"]["distortion"])
        print(f"== {stem}  jpeg {rec['jpeg']}  frame {rec['frame']}  raw dist n={n_raw}")
        for source in ("raw", "mn"):
            for fk in (False, True):
                r = fit_one(rec, source, fk)
                if r:
                    print(
                        f"  table {source:3s} {'zeta,alpha,kappa' if fk else 'zeta,alpha      '}: zeta {r['zeta']:.4f} "
                        f"alpha {r['alpha']:.3f} kappa {r['kappa']:.3f}  rms {r['rms_raw_px']:.2f} raw px (n={r['n']})"
                    )


def main():
    a = sys.argv[1:]
    if len(a) >= 2 and a[0] == "match":
        cmd_match(a[1], "--full" in a)
    elif len(a) >= 2 and a[0] == "fit":
        cmd_fit(a[1:])
    else:
        print(__doc__)


if __name__ == "__main__":
    main()

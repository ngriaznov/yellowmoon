#!/usr/bin/env python3
"""Measures what darktable-cli's embedded-metadata lens correction does to the synthetic Sony files
of `lensgen` and records the numbers as `crates/ym-gpu/tests/cpu_data/darktable_oracle.json`.

The snapshot lets `cargo test -p ym-gpu --test cpu_lens` pin our conventions to darktable's without
darktable or any image data in CI (only measured numbers are stored; nothing of darktable is
copied). `run_oracle_suite.sh` produces the renders this script reads:

  SUITE/vignetting/flat/{lens_in,lens_out}.pfm.gz   flat field, vignetting correction only
  SUITE/distortion/vlines/{lens_in,lens_out}.pfm.gz vertical lines, distortion only
  SUITE/ca/vlines/{lens_in,lens_out}.pfm.gz         vertical lines, CA only (tags with a large CA)
  SUITE/fixtures/tags.json, SUITE/fixtures_ca/tags.json   the raw knots of the files

`lens_in` / `lens_out` are the float images (linear, white-balanced camera RGB) that go into and come
out of darktable's lens module (`darktable-cli --dump-pipe lens`), so there is no transfer curve or
colour matrix in the way.

Usage: record_oracle.py SUITE [OUT.json]

What is measured, all in darktable's own coordinates (pixel indices, the frame centre at (W/2, H/2)):

* vignetting: the radial gain `lens_out / lens_in` of the flat field, averaged in annuli of 0.0125
  in r;
* distortion and CA: the sub-pixel centroids of the vertical lines in every colour plane, paired
  with the known source line (a line every 100 px at x = 49.5 + 100 k) by the model of
  `lens_model.py` (pairs further than 1 px from the model are dropped as mismatches).

It exits 1 if the model of `lens_model.py` does not reproduce the measurement (gain 0.1%,
distortion 0.05 px rms, CA 0.4 px max), which is how the model was verified.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lens_io  # noqa: E402
import lens_model as lm  # noqa: E402

LINE_X0, LINE_PITCH, LINES = 49.5, 100.0, 30   # source lines, pixel-index coordinates


def vignetting_profile(off, on, step=0.0125, r_max=0.985):
    h, w = off.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(xx - w / 2.0, yy - h / 2.0) / math.hypot(w / 2.0, h / 2.0)
    gain = on.mean(axis=2) / off.mean(axis=2)
    out = []
    for lo in np.arange(0.0, r_max, step):
        sel = (r >= lo) & (r < lo + step)
        if sel.any():
            out.append([round(float(r[sel].mean()), 5), round(float(gain[sel].mean()), 6)])
    return out


def centroid(row, q, half=5):
    lo, hi = q - half, q + half + 1
    if lo < 1 or hi >= len(row) - 1:
        return None
    seg = row[lo:hi]
    w = np.maximum(max(row[lo - 1], row[hi]) - seg, 0.0)
    return float((w * np.arange(lo, hi)).sum() / w.sum()) if w.sum() > 1e-9 else None


def line_centroids(row):
    pk, _ = find_peaks(-row, prominence=0.03)
    return [c for c in (centroid(row, int(q)) for q in pk if 12 < q < len(row) - 12) if c is not None]


def predict_output_x(source_x, y, cor, zoom, width, height):
    """Output x (pixel index) at row y that shows `source_x`, for the darktable model with the radial
    factor `cor(r)` (before the zoom): source - c = (x_o - c) * cor(r_o / zoom) / zoom."""
    cx, cy, hd = width / 2.0, height / 2.0, math.hypot(width / 2.0, height / 2.0)

    def residual(xo):
        r = math.hypot(xo - cx, y - cy) / hd
        return cx + (xo - cx) * cor(r / zoom) / zoom - source_x

    try:
        return brentq(residual, -400.0, width + 400.0)
    except ValueError:
        return None


def cors(distortion, ca, use_distortion, use_ca):
    n = len(distortion) if len(distortion) else len(ca) // 2
    kx = lm.knots_x(n)
    ones = np.ones(n)
    f = np.asarray(distortion, dtype=np.float64) * 2.0 ** -14 + 1.0 if use_distortion else ones
    if use_ca:
        kr = np.asarray(ca[:n], dtype=np.float64) * 2.0 ** -21 + 1.0
        kb = np.asarray(ca[n:], dtype=np.float64) * 2.0 ** -21 + 1.0
    else:
        kr = kb = ones
    return [lambda r, v=f * k: float(np.interp(r, kx, v)) for k in (kr, ones, kb)]


def line_samples(on, tags, use_distortion, use_ca, channels, rows):
    """Pairs of measured output centroids and known source lines, one tuple per (row, line):
    (y, source_x, [x_out per channel])."""
    h, w = on.shape[:2]
    distortion, ca = tags.get("distortion", []), tags.get("ca", [])
    zoom = lm.darktable_zoom(w, h, distortion, ca, use_distortion, use_ca)
    cor = cors(distortion, ca, use_distortion, use_ca)
    out, worst = [], 0.0
    for j, y in enumerate(rows):
        measured = {c: line_centroids(on[y, :, c]) for c in channels}
        for k in range(LINES):
            sx = LINE_X0 + LINE_PITCH * k
            xs = []
            for c in channels:
                pred = predict_output_x(sx, y, cor[c], zoom, w, h)
                if pred is None or not measured[c]:
                    break
                m = min(measured[c], key=lambda v: abs(v - pred))
                if abs(m - pred) >= 1.0:
                    break
                xs.append(m)
                worst = max(worst, abs(m - pred))
            else:
                out.append((y, sx, xs, [predict_output_x(sx, y, cor[c], zoom, w, h) for c in channels]))
    return zoom, out, worst


def thin(samples, per_row=9):
    """Keeps about `per_row` lines per row, a different subset in every row."""
    by_row = {}
    for s in samples:
        by_row.setdefault(s[0], []).append(s)
    kept = []
    for j, (y, items) in enumerate(sorted(by_row.items())):
        step = max(len(items) // per_row, 1)
        kept.extend(items[(j * 2) % step::step])
    return kept


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    suite = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[2] / "crates/ym-gpu/tests/cpu_data/darktable_oracle.json"
    tags = json.load(open(suite / "fixtures/tags.json"))
    tags_ca = json.load(open(suite / "fixtures_ca/tags.json"))
    ok = True

    off, on = lens_io.load_pair(suite / "vignetting/flat")
    h, w = off.shape[:2]
    profile = vignetting_profile(off, on)
    rr = np.array([p[0] for p in profile])
    model = lm.vignette_gain(tags["vignetting"], rr)
    dev = np.abs(np.array([p[1] for p in profile]) / model - 1.0)
    print(f"vignetting: {len(profile)} annuli, model deviation max {dev.max():.4%}, mean {dev.mean():.4%}")
    ok &= bool(dev.max() < 1e-3)

    rows = list(range(60, h, 160))
    on_d = lens_io.load_pair(suite / "distortion/vlines")[1]
    zoom_d, samples, worst = line_samples(on_d, tags, True, False, [1], rows)
    res = np.array([s[2][0] - s[3][0] for s in samples])
    print(f"distortion: zoom {zoom_d:.6f}, {len(samples)} line positions, residual rms {np.sqrt(np.mean(res ** 2)):.4f} px, max {np.abs(res).max():.4f} px")
    ok &= bool(np.sqrt(np.mean(res ** 2)) < 0.05)
    dist_samples = [[s[0], round(s[2][0], 4), s[1]] for s in thin(samples)]

    on_c = lens_io.load_pair(suite / "ca/vlines")[1]
    zoom_c, samples_c, worst_c = line_samples(on_c, tags_ca, False, True, [0, 1, 2], rows)
    res_c = np.array([[a - b for a, b in zip(s[2], s[3])] for s in samples_c])
    print(f"ca: zoom {zoom_c:.6f}, {len(samples_c)} line positions, residual rms {np.sqrt(np.mean(res_c ** 2, axis=0)).round(3).tolist()} px, max {np.abs(res_c).max():.3f} px")
    ok &= bool(np.abs(res_c).max() < 0.4)
    ca_samples = [[s[0], s[1]] + [round(v, 4) for v in s[2]] for s in thin(samples_c)]

    snapshot = {
        "source": "darktable-cli 4.6.1 (Ubuntu 24.04), lens module, embedded metadata, algorithm 2; synthetic ILCE-7RM4 files from tools/lens/lensgen; measured by tools/lens/record_oracle.py",
        "frame": [w, h],
        "coordinates": "darktable's: pixel indices counted from the frame centre (W/2, H/2); r is over half the frame diagonal",
        "vignetting": {"tags": tags["vignetting"], "samples": profile},
        "distortion": {"tags": tags["distortion"], "zoom": round(zoom_d, 6), "samples": dist_samples},
        "ca": {"tags": tags_ca["ca"], "zoom": round(zoom_c, 6), "samples": ca_samples},
    }
    out_path.write_text(json.dumps(snapshot, separators=(",", ":")) + "\n")
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

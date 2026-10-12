#!/usr/bin/env python3
"""Independent Python implementation of the R3 / R9 lens maths of docs/05-raw-processing.md.

Used for two things:
  1. generating crates/ym-gpu/tests/cpu_data/lens_vectors.json, which the Rust tests compare with
     (`python3 -I tools/lens/lens_model.py --write-vectors`);
  2. computing the reference maps in compare_with_darktable.py.

It deliberately shares no code with the Rust implementation (NumPy's `interp` does the linear
interpolation, the formulas are transcribed from the spec) and follows darktable 4.6.1's embedded
metadata model, which `darktable_oracle.sh` + `record_oracle.py` measured to ~0.01% (vignetting) and
~0.01 px (distortion):

  * knot i of n at the normalised radius (i + 0.5) / (n - 1), linear interpolation, end values held;
  * vignetting: relative illumination G_i = 2^(0.5 - 2^(v_i * 2^-13 - 1)), the photosite is DIVIDED by
    G(r) (not squared: that is darktable's older algorithm);
  * distortion F_i = d_i * 2^-14 + 1, source radius = r * F(r); CA K_i = ca_i * 2^-21 + 1 (red then
    blue), red / blue source radius = r * F(r) * K(r), K and F taken at the output radius r.

Conventions (match crates/ym-gpu/src/cpu/lens.rs): continuous pixel coordinates (pixel i covers
[i, i+1)), frame centre (W/2, H/2), r normalised by half the frame diagonal.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator


def knots_x(n):
    """Radii of the n knots of a Sony lens tag: (i + 0.5) / (n - 1)."""
    return (np.arange(n) + 0.5) / (n - 1)


def vignette_knot_attenuation(v):
    """G_i = 2^(0.5 - 2^(v*2^-13 - 1)): the relative illumination the lens leaves at the knot."""
    v = np.asarray(v, dtype=np.float64)
    return 2.0 ** (0.5 - 2.0 ** (v * 2.0 ** -13 - 1.0))


def vignette_attenuation(knots, r):
    g = vignette_knot_attenuation(knots)
    return np.interp(r, knots_x(len(g)), g)


def vignette_gain(knots, r):
    """The multiplier of a photosite at radius r: 1 / max(G(r), 1e-4)."""
    return 1.0 / np.maximum(vignette_attenuation(knots, r), 1e-4)


def distortion_factor(knots, r):
    f = np.asarray(knots, dtype=np.float64) * 2.0 ** -14 + 1.0
    return np.interp(r, knots_x(len(f)), f)


def ca_factors(ca, r):
    """ca holds 2n values, red then blue; K = ca * 2^-21 + 1."""
    n = len(ca) // 2
    kr = np.asarray(ca[:n], dtype=np.float64) * 2.0 ** -21 + 1.0
    kb = np.asarray(ca[n:], dtype=np.float64) * 2.0 ** -21 + 1.0
    return np.interp(r, knots_x(n), kr), np.interp(r, knots_x(n), kb)


def source_positions(width, height, distortion, ca, px, py):
    """Source position (continuous) of the output position (px, py) for red, green, blue."""
    cx, cy = width / 2.0, height / 2.0
    hd = 0.5 * math.hypot(width, height)
    dx, dy = px - cx, py - cy
    r = math.hypot(dx, dy) / hd
    f = float(distortion_factor(distortion, r)) if len(distortion) else 1.0
    kr = kb = 1.0
    if len(ca):
        kr, kb = (float(k) for k in ca_factors(ca, r))
    out = {}
    for name, k in (("red", kr), ("green", 1.0), ("blue", kb)):
        s = f * k
        out[name] = [cx + dx * s, cy + dy * s]
    return out


def source_position_arrays(width, height, distortion, ca, px, py):
    """Vectorised `source_positions`: px, py are arrays of output positions; returns the arrays
    (sx, sy) per colour name."""
    cx, cy = width / 2.0, height / 2.0
    hd = 0.5 * math.hypot(width, height)
    dx, dy = px - cx, py - cy
    r = np.hypot(dx, dy) / hd
    f = distortion_factor(distortion, r) if len(distortion) else np.ones_like(r)
    kr = kb = np.ones_like(r)
    if len(ca):
        kr, kb = ca_factors(ca, r)
    out = {}
    for name, k in (("red", kr), ("green", 1.0), ("blue", kb)):
        s = f * k
        out[name] = (cx + dx * s, cy + dy * s)
    return out


def darktable_zoom(width, height, distortion, ca, use_distortion=True, use_ca=True):
    """darktable's auto-scale of the embedded-metadata correction (algorithm 2).

    darktable does not crop the corrected image, it zooms it up so that no black corner is left: the
    zoom is the largest correction factor `F(r) * K_c(r)` over r from the short half-side to the
    corner (200 samples, all three channels). The corrected output position p_o (counted from the
    frame centre) then shows the source at `p_o * cor_c(|p_o| / hd / zoom) / zoom`. Our R9 crops
    instead; the comparison scripts fit this zoom out, and the oracle snapshot test checks that this
    rule reproduces the fitted value.
    """
    n = len(distortion) if len(distortion) else len(ca) // 2
    ones = np.ones(n)
    f = np.asarray(distortion, dtype=np.float64) * 2.0 ** -14 + 1.0 if (use_distortion and len(distortion)) else ones
    kr, kb = ((np.asarray(ca[:n], dtype=np.float64) * 2.0 ** -21 + 1.0, np.asarray(ca[n:], dtype=np.float64) * 2.0 ** -21 + 1.0)
              if (use_ca and len(ca)) else (ones, ones))
    cx, cy = width / 2.0, height / 2.0
    srr = min(cx, cy) / math.hypot(cx, cy)
    xs = srr + (1.0 - srr) * np.arange(200) / 199.0
    kx = knots_x(n)
    return float(max(np.interp(xs, kx, f * k).max() for k in (kr, ones, kb)))


def make_vectors():
    cases = []
    specs = [
        dict(width=600, height=400,
             vignetting=[0, 150, 500, 1000, 1700, 2600, 3700, 5000, 6500, 8000, 9500, 11000],
             distortion=[0, 20, 60, 120, 200, 300, 420, 560, 740],
             ca=[0, 40, 90, 160, 250, 0, -40, -90, -160, -250]),
        dict(width=7008, height=4672,
             vignetting=[0, 300, -200, 1200, 2500, 4000, 6200, 9000],
             distortion=[0, -30, -100, -240, -400, -700, -1000, -1500, -2100, -2800, -3600],
             ca=[0, 100, 300, 700, 1200, 2000, 0, -100, -300, -700, -1200, -2000]),
    ]
    for spec in specs:
        w, h = spec["width"], spec["height"]
        rs = [0.0, 0.05, 0.1, 0.33, 0.5, 0.666, 0.9, 0.999, 1.0]
        case = dict(spec)
        case["knot_attenuation"] = [float(g) for g in vignette_knot_attenuation(spec["vignetting"])]
        case["attenuation"] = [[r, float(vignette_attenuation(spec["vignetting"], r))] for r in rs]
        case["gain"] = [[r, float(vignette_gain(spec["vignetting"], r))] for r in rs]
        pts = []
        for fx, fy in [(0.5, 0.5), (0.1, 0.1), (0.9, 0.2), (0.25, 0.8), (0.999, 0.999), (0.5, 0.01), (0.75, 0.5)]:
            px, py = fx * w, fy * h
            d = dict(x=px, y=py)
            d.update(source_positions(w, h, spec["distortion"], spec["ca"], px, py))
            pts.append(d)
        case["positions"] = pts
        cases.append(case)
    xs = [0.0, 0.1, 0.35, 0.4, 0.8, 1.0]
    ys = [1.0, 1.02, 1.1, 1.1, 1.4, 1.45]
    p = PchipInterpolator(xs, ys)
    pchip = dict(xs=xs, ys=ys, samples=[[x, float(p(x))] for x in np.linspace(0, 1, 41)])
    return dict(source="tools/lens/lens_model.py (NumPy linear interpolation, darktable 4.6.1 embedded-metadata model; SciPy PCHIP for the spline.rs check)", cases=cases, pchip=pchip)


if __name__ == "__main__":
    if "--write-vectors" in sys.argv:
        out = Path(__file__).resolve().parents[2] / "crates/ym-gpu/tests/cpu_data/lens_vectors.json"
        out.write_text(json.dumps(make_vectors()))
        print("wrote", out)
    else:
        print(__doc__)

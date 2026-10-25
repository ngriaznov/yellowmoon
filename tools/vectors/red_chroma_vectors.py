#!/usr/bin/env python3
"""Generate the parity vectors of the red chroma step (C12 step 1c, docs/02-colour-pipeline.md).

The step is a hue-preserving chroma gain on the reds and, on its up side, the ochres and yellows, applied to
linear AWG4 in front of the display placement and ACES 2 (crates/ym-color/src/display/red_chroma.rs). This
script is an independent numpy
implementation of the same maths, the one the investigation of 2026-10-24 used (owner decision "R1 fix looks
better than what we have currently"): it reads the neutral-axis table `YD(s)` from
crates/ym-color/tests/data/red_chroma/yd_of_s.csv (the display luminance of the neutral axis through ACES 2,
written by the investigation with ym-color's own ACES 2; the Rust test checks that ym-color rebuilds it), and
computes everything else itself: the gamut matrices from the primaries, the gamut edge table CMAX(L*, h) by
bisection, and the step. It writes a few hundred AWG4 samples across stops, hues and chromas for the Rec.709
and P3-D65 limiting gamuts and four pairs of strengths, with the step's output, to

    crates/ym-color/tests/data/red_chroma/vectors.json

The Rust test (`display::red_chroma::tests::matches_the_python_reference`) feeds the inputs to
`RedChromaStep` and requires the outputs within the stated tolerance.

Usage
-----
  python3 tools/vectors/red_chroma_vectors.py            # (re)write vectors.json
  python3 tools/vectors/red_chroma_vectors.py --check    # fail if the committed file is stale
                                                         # (compared with a 1e-8 tolerance)
  python3 tools/vectors/red_chroma_vectors.py --cmax-check
        # also compare the bisected CMAX tables with the committed cmax_*.csv (3 decimals)

Needs numpy only. Seeded: the same numpy gives the same file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "crates" / "ym-color" / "tests" / "data" / "red_chroma"
OUT = DATA / "vectors.json"
SCHEMA = "yellowmoon/red-chroma-vectors/1"
SEED = 20261024
TOLERANCE = 1e-8

# ---- constants of the step (docs/02, "Red chroma (C12 step 1c)") ---------------------------------------
LAB_WHITE = np.array([0.95047, 1.0, 1.08883])
GREY = 0.18
Y_MIN = 1e-7
S_MAX = 8.0
YD_S_MIN, YD_STEP, YD_LEN = -10.0, 0.25, 73
FADE = (-7.5, -5.5)
WINDOW = dict(a=3.0, b=31.0, e_lo=30.0, e_hi=12.0)
# The yellow window of the up side (ochres, yellows): 1 on 68..95 degrees, 0 below 62 and above 110.
WINDOW_Y = dict(a=68.0, b=95.0, e_lo=6.0, e_hi=15.0)
S0, UP, DARK, TAU = -0.5, 0.34, 0.43, 1.65
K_MAX, KNEE = 1.45, 0.8
CAP_AT, CMAX_FLOOR, RAMP_EPS, C_TINY = 0.9, 1.0, 0.015, 1e-4

# ---- gamuts, from the primaries (the same numbers as ym_color::consts) --------------------------------
D65 = (0.3127, 0.3290)
PRIMARIES = {
    "awg4": ((0.7347, 0.2653), (0.1424, 0.8576), (0.0991, -0.0308)),
    "rec709": ((0.64, 0.33), (0.30, 0.60), (0.15, 0.06)),
    "p3": ((0.680, 0.320), (0.265, 0.690), (0.150, 0.060)),
}


def _xyz(xy):
    return np.array([xy[0] / xy[1], 1.0, (1.0 - xy[0] - xy[1]) / xy[1]])


def rgb_to_xyz(name):
    r, g, b = PRIMARIES[name]
    prim = np.stack([_xyz(r), _xyz(g), _xyz(b)], axis=1)
    scale = np.linalg.solve(prim, _xyz(D65))
    return prim * scale


AWG4_TO_XYZ = rgb_to_xyz("awg4")
XYZ_TO_AWG4 = np.linalg.inv(AWG4_TO_XYZ)
GAMUT_XYZ_TO_RGB = {"rec709": np.linalg.inv(rgb_to_xyz("rec709")), "p3": np.linalg.inv(rgb_to_xyz("p3"))}

# ---- CIELAB (exact rational constants, as ym_color::deltae) -------------------------------------------
EPS, KAPPA = 216.0 / 24389.0, 24389.0 / 27.0


def xyz_to_lab(xyz):
    t = xyz / LAB_WHITE
    f = np.where(t > EPS, np.cbrt(t), (KAPPA * t + 16.0) / 116.0)
    return np.stack([116.0 * f[..., 1] - 16.0, 500.0 * (f[..., 0] - f[..., 1]), 200.0 * (f[..., 1] - f[..., 2])], -1)


def lab_to_xyz(lab):
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = fy + lab[..., 1] / 500.0
    fz = fy - lab[..., 2] / 200.0

    def inv(f):
        cubed = f * f * f
        return np.where(cubed > EPS, cubed, (116.0 * f - 16.0) / KAPPA)

    return np.stack([inv(fx), inv(fy), inv(fz)], -1) * LAB_WHITE


# ---- tables -------------------------------------------------------------------------------------------
def load_yd():
    rows = np.loadtxt(DATA / "yd_of_s.csv", delimiter=",", skiprows=1)
    assert rows.shape == (YD_LEN, 3)
    s = YD_S_MIN + YD_STEP * np.arange(YD_LEN)
    assert np.allclose(rows[:, 0], s)
    return {"rec709": rows[:, 1], "p3": rows[:, 2]}


def build_cmax(gamut):
    """CMAX(L*, h): 41 rows (L* 0..100 step 2.5) by 72 columns (hue 0..355 step 5), 40 bisection steps."""
    lg = np.arange(0, 100.1, 2.5)
    hg = np.arange(0, 360, 5.0)
    L, H = np.meshgrid(lg, hg, indexing="ij")
    m = GAMUT_XYZ_TO_RGB[gamut]
    lo = np.zeros_like(L)
    hi = np.full_like(L, 250.0)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lab = np.stack([L, mid * np.cos(np.radians(H)), mid * np.sin(np.radians(H))], -1)
        rgb = lab_to_xyz(lab) @ m.T
        ok = (rgb.min(-1) >= -1e-9) & (rgb.max(-1) <= 1 + 1e-9)
        lo = np.where(ok, mid, lo)
        hi = np.where(ok, hi, mid)
    return lo


def yd_of_s(yd, s):
    return np.interp(s, YD_S_MIN + YD_STEP * np.arange(YD_LEN), yd)


def cmax_lookup(table, L, h):
    fl = np.clip(L / 2.5, 0, table.shape[0] - 1.000001)
    i = fl.astype(int)
    a = fl - i
    fh = (h % 360.0) / 5.0
    j = fh.astype(int) % 72
    b = fh - np.floor(fh)
    j1 = (j + 1) % 72
    return (1 - a) * ((1 - b) * table[i, j] + b * table[i, j1]) + a * ((1 - b) * table[i + 1, j] + b * table[i + 1, j1])


# ---- the step -----------------------------------------------------------------------------------------
def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def hdiff(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def raised_window(h, w):
    d_lo, d_hi = hdiff(h, w["a"]), hdiff(h, w["b"])
    out = np.ones_like(h)
    out = np.where(d_lo < 0, 0.5 * (1 + np.cos(np.pi * np.clip(-d_lo / w["e_lo"], 0, 1))), out)
    out = np.where(d_hi > 0, 0.5 * (1 + np.cos(np.pi * np.clip(d_hi / w["e_hi"], 0, 1))), out)
    return out


def window(h):
    """The red window (both sides of the step)."""
    return raised_window(h, WINDOW)


def window_up(h):
    """The window of the up side: the red window and the yellow window (they never overlap)."""
    return np.maximum(window(h), raised_window(h, WINDOW_Y))


def kappa(c):
    return np.clip((20.0 / np.maximum(c, 8.0)) ** 0.65, 0.35, 1.25)


def profile(s):
    return np.where(s >= S0, UP * (s - S0), -DARK * np.tanh((S0 - s) / TAU))


def red_chroma(awg4, gamut, tables, f_up, f_dn):
    """The step on an (N, 3) array of linear AWG4; returns (output, ln k)."""
    yd_table, cmax_table = tables
    awg4 = np.asarray(awg4, float)
    xyz = awg4 @ AWG4_TO_XYZ.T
    y = np.maximum(xyz[:, 1], 1e-12)
    s_raw = np.log2(y / GREY)
    s = np.minimum(s_raw, S_MAX)
    yd = np.clip(yd_of_s(yd_table, s), 1e-7, 1.0)
    lab = xyz_to_lab(xyz / y[:, None] * yd[:, None])
    c = np.hypot(lab[:, 1], lab[:, 2])
    h = np.degrees(np.arctan2(lab[:, 2], lab[:, 1])) % 360.0
    p = profile(s)
    # The dark side is for reds only; the up side also covers the yellow window.
    w = np.where(p > 0, window_up(h), window(h))
    kap = kappa(c)
    lnf = kap * p
    c_max = np.maximum(cmax_lookup(cmax_table, lab[:, 0], h), CMAX_FLOOR)
    pos = p > 0
    raw = np.exp(np.maximum(lnf, 0))
    k0 = np.where(raw > 1, 1 + (K_MAX - 1) * np.tanh((raw - 1) / (K_MAX - 1)), raw)
    c_in = c * k0
    limit = c_max * np.exp(kappa(CAP_AT * c_max) * np.maximum(p, 0))
    knee = KNEE * limit
    c_soft = np.where(c_in <= knee, c_in, knee + (limit - knee) * np.tanh((c_in - knee) / np.maximum(limit - knee, 1e-6)))
    x = c_soft - c
    c_boost = c + np.where(x > 0, x * x / (x + RAMP_EPS * c + 1e-9), 0.0)
    ln_b = np.where(c > C_TINY, np.log(np.maximum(c_boost, 1e-9) / np.maximum(c, C_TINY)), np.log(k0))
    lnk = np.where(pos, f_up * ln_b, f_dn * lnf) * w
    lnk = lnk * smoothstep(FADE[0], FADE[1], s_raw)
    lnk = np.where(awg4 @ AWG4_TO_XYZ[1] > Y_MIN, lnk, 0.0)
    k = np.exp(lnk)
    lab2 = lab.copy()
    lab2[:, 1:] *= k[:, None]
    chroma = lab_to_xyz(lab2) / yd[:, None]
    out = (chroma * y[:, None]) @ XYZ_TO_AWG4.T
    # The Rust step returns its input bit for bit where the gain is 0.
    out = np.where((lnk == 0.0)[:, None], awg4, out)
    return out, lnk


def pixel(tables, s, h, c):
    """AWG4 of a colour at `s` stops over grey whose Lab at the display luminance has chroma c and hue h."""
    yd_table, _ = tables
    yd = np.clip(yd_of_s(yd_table, np.minimum(s, S_MAX)), 1e-7, 1.0)
    lab_l = xyz_to_lab(yd[:, None] * LAB_WHITE)[:, 0]
    hr = np.radians(h)
    lab = np.stack([lab_l, c * np.cos(hr), c * np.sin(hr)], -1)
    rel = lab_to_xyz(lab)
    y = GREY * np.exp2(s)
    return (rel / yd[:, None] * y[:, None]) @ XYZ_TO_AWG4.T


# ---- the samples --------------------------------------------------------------------------------------
STRENGTHS = [(1.0, 0.0), (1.0, 0.5), (0.5, 1.0), (0.0, 1.0)]


def make(tables_by_gamut):
    rng = np.random.default_rng(SEED)
    rng_y = np.random.default_rng(SEED + 1)
    samples = []
    for gamut in ("rec709", "p3"):
        tables = tables_by_gamut[gamut]
        for f_up, f_dn in STRENGTHS:
            n = 52
            # Stops: a grid over the working range, the crossing, the fade, and random values.
            s = np.concatenate(
                [
                    np.array([-8.5, -7.0, -6.5, -6.0, -5.0, -3.0, -1.5, -0.75, -0.5, -0.25, 0.0, 0.5, 1.0, 2.0, 3.5, 5.0, 7.0, 8.0, 9.5]),
                    rng.uniform(-7.5, 8.0, n - 19),
                ]
            )
            # Hues: mostly the window and its tails, some elsewhere.
            h = np.concatenate([rng.uniform(-45.0, 60.0, n - 10), rng.uniform(0.0, 360.0, 10)]) % 360.0
            edge = cmax_lookup(
                tables[1],
                xyz_to_lab(np.clip(yd_of_s(tables[0], np.minimum(s, S_MAX)), 1e-7, 1.0)[:, None] * LAB_WHITE)[:, 0],
                h,
            )
            c = np.maximum(rng.uniform(0.2, 1.6, n) * np.maximum(edge, 3.0), 0.05)
            # A few muted colours, where kappa is largest.
            c[:6] = rng.uniform(0.2, 6.0, 6)
            # The yellow window and its tails, from a second generator so the samples above stay as they were.
            ny = 26
            sy = np.concatenate([np.array([-3.0, -0.75, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]), rng_y.uniform(-7.5, 8.0, ny - 9)])
            hy = np.concatenate([np.array([57.0, 61.0, 62.5, 65.0, 68.0, 80.0, 95.0, 102.0, 112.0]), rng_y.uniform(52.0, 118.0, ny - 9)])
            edge_y = cmax_lookup(
                tables[1],
                xyz_to_lab(np.clip(yd_of_s(tables[0], np.minimum(sy, S_MAX)), 1e-7, 1.0)[:, None] * LAB_WHITE)[:, 0],
                hy,
            )
            cy = np.maximum(rng_y.uniform(0.2, 1.6, ny) * np.maximum(edge_y, 3.0), 0.05)
            cy[:4] = rng_y.uniform(0.2, 6.0, 4)
            s, h, c = np.concatenate([s, sy]), np.concatenate([h, hy]), np.concatenate([c, cy])
            n += ny
            awg4 = pixel(tables, s, h, c)
            out, lnk = red_chroma(awg4, gamut, tables, f_up, f_dn)
            for i in range(n):
                samples.append(
                    {
                        "gamut": gamut,
                        "f_up": f_up,
                        "f_dn": f_dn,
                        "s": float(s[i]),
                        "h": float(h[i]),
                        "c": float(c[i]),
                        "in": [float(v) for v in awg4[i]],
                        "out": [float(v) for v in out[i]],
                        "ln_k": float(lnk[i]),
                    }
                )
    return {
        "schema": SCHEMA,
        "generator": "tools/vectors/red_chroma_vectors.py",
        "tolerance": TOLERANCE,
        "note": "AWG4 in, AWG4 out; s stops over grey, h and c the Lab hue (degrees) and chroma of the colour at the display luminance of the neutral axis.",
        "samples": samples,
    }


def tables_for_all():
    yd = load_yd()
    return {g: (yd[g], build_cmax(g)) for g in ("rec709", "p3")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="fail if the committed file differs from a regenerated one")
    ap.add_argument("--cmax-check", action="store_true", help="compare the bisected CMAX with the committed CSVs")
    args = ap.parse_args()
    tables = tables_for_all()
    if args.cmax_check:
        worst = 0.0
        for g, name in (("rec709", "cmax_srgb.csv"), ("p3", "cmax_p3.csv")):
            ref = np.loadtxt(DATA / name, delimiter=",", skiprows=1)
            worst = max(worst, float(np.abs(ref - tables[g][1]).max()))
        print(f"CMAX against the CSVs: max difference {worst:.2e}")
        if worst > 6e-4:
            return 1
    doc = make(tables)
    if args.check:
        old = json.loads(OUT.read_text())
        if len(old["samples"]) != len(doc["samples"]):
            print("stale: sample count differs", file=sys.stderr)
            return 1
        worst = 0.0
        for a, b in zip(old["samples"], doc["samples"]):
            for key in ("in", "out"):
                worst = max(worst, float(np.abs(np.array(a[key]) - np.array(b[key])).max()))
        print(f"{OUT.relative_to(REPO)}: max difference {worst:.2e} over {len(doc['samples'])} samples")
        return 0 if worst <= TOLERANCE else 1
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}: {len(doc['samples'])} samples")
    return 0


if __name__ == "__main__":
    sys.exit(main())

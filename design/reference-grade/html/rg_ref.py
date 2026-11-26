#!/usr/bin/env python3
"""A numpy sketch of the maths of docs/34 for the mock-ups of design/reference-grade/ (task RG-07). NOT product code and NOT a fit: the values it is given are illustrative.

Steps, in the node's order (docs/34 section 2.3), on LogC4 codes v (H x W x 3):
  1. the global grade: three channel curves and the luma curve (as shifts in stops), then saturation (a scale of the chroma plane);
  2. the hue curves (section 4.2): (p, q) -> sigma(h) R(delta(h)) (p, q), with Rec.709 luma restored on the codes (the owner's answer to STATUS question 48, 2026-11-25);
  3. the element grades (section 4a.2): for each element, w <- w + a_e (local_e(w) - w), in the order sky, vegetation, water, ground, built, skin, subject;
  4. Strength (section 3.2): every fitted value is scaled from the identity before the render (effective()).
The soft masks a_e are the guided filter of section 4a.3 (radius 0.02 of the short side, eps 1e-3, L*/100 of the display picture as the guide) taken on the drawing's own element layers.
"""
import math

import numpy as np

# LogC4 (ARRI's published formulas, used here only to place the codes; the product's own is ym_color)
_A = (2.0**18 - 16.0) / 117.45
_B = (1023.0 - 95.0) / 1023.0
_C = 95.0 / 1023.0
_S = (7.0 * math.log(2.0) * 2.0 ** (7.0 - 14.0 * _C / _B)) / (_A * _B)
_T = (2.0 ** (14.0 * (-_C / _B) + 6.0) - 64.0) / _A
STOP = _B / 14.0   # 0.0648: one stop in codes (V5_STOP_CODE)


def encode(x):
    x = np.asarray(x, np.float64)
    return np.where(x >= _T, (np.log2(np.maximum(_A * x + 64.0, 1e-12)) - 6.0) / 14.0 * _B + _C, (x - _T) / _S)


def decode(v):
    v = np.asarray(v, np.float64)
    lin = (2.0 ** (14.0 * (v - _C) / _B + 6.0) - 64.0) / _A
    return np.where(v >= 0.0, lin, v * _S + _T)


E1 = np.array([1.0, -0.5, -0.5]) / math.sqrt(1.5)
E2 = np.array([0.0, 1.0, -1.0]) / math.sqrt(2.0)
W709 = np.array([0.2126, 0.7152, 0.0722])
ELEMENTS = ["sky", "vegetation", "water", "ground", "built", "skin", "subject"]
CAPS = {"sky": 40, "vegetation": 35, "water": 35, "ground": 30, "built": 25, "skin": 15}


def periodic_curve(points, n=1024):
    """A periodic, smooth stand-in for the Fritsch-Carlson table of section 4.2 (the sketch uses a cosine-eased interpolation: it never overshoots the given values)."""
    pts = sorted(points)
    xs = np.array([p[0] for p in pts], np.float64)
    ys = np.array([p[1] for p in pts], np.float64)
    h = np.arange(n) * 360.0 / n
    out = np.empty(n)
    k = len(xs)
    for i, hh in enumerate(h):
        j = np.searchsorted(xs, hh, side="right") - 1
        x0 = xs[j % k] + (360.0 if j < 0 else 0.0) - (360.0 if j < 0 else 0.0)
        if j < 0:
            x0 = xs[-1] - 360.0
            x1 = xs[0]
            y0, y1 = ys[-1], ys[0]
        else:
            x0 = xs[j]
            y0 = ys[j]
            if j + 1 < k:
                x1, y1 = xs[j + 1], ys[j + 1]
            else:
                x1, y1 = xs[0] + 360.0, ys[0]
        t = (hh - x0) / (x1 - x0)
        t = 0.5 - 0.5 * math.cos(math.pi * t)
        out[i] = y0 + (y1 - y0) * t
    return out


def sample_periodic(table, hue_deg):
    n = len(table)
    f = (np.mod(hue_deg, 360.0) / 360.0) * n
    i0 = np.floor(f).astype(int) % n
    i1 = (i0 + 1) % n
    w = f - np.floor(f)
    return table[i0] * (1 - w) + table[i1] * w


def pq(v):
    return v @ E1, v @ E2


def from_pq(m, p, q):
    return m[..., None] * np.ones(3) + p[..., None] * E1 + q[..., None] * E2


def effective(s, k):
    """Strength k = strength / 100 (section 3.2), on the sketch's settings dict."""
    out = {"luma": [(x, x + k * (y - x)) for x, y in s["luma"]], "sat": k * s["sat"]}
    for c in ("red", "green", "blue"):
        out[c] = [(x, x + k * (y - x)) for x, y in s[c]]
    out["hue_vs_hue"] = [(h, k * v) for h, v in s["hue_vs_hue"]]
    out["hue_vs_sat"] = [(h, 1 + k * (v - 1)) for h, v in s["hue_vs_sat"]]
    el = {}
    for e, d in s["elements"].items():
        el[e] = {"exposure_ev": k * d["exposure_ev"]}
        if e != "subject":
            el[e]["hue_deg"] = k * d["hue_deg"]
            el[e]["chroma"] = d["chroma"] ** k
    out["elements"] = el
    return out


def curve_apply(pts, x):
    xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
    return np.interp(x, xs, ys)


def grade(v, s, masks):
    """v: LogC4 codes (H, W, 3); s: effective settings; masks: dict element -> soft mask (H, W)."""
    v = v.copy()
    # 1. channel curves then luma curve (per channel, on the codes)
    for i, c in enumerate(("red", "green", "blue")):
        v[..., i] = curve_apply(s[c], v[..., i])
    v = curve_apply(s["luma"], v)
    m = v.mean(-1)
    lam = 1.0 + s["sat"] / 100.0
    p, q = pq(v)
    v = from_pq(m, p * lam, q * lam)
    # 2. hue curves with luma restored
    p, q = pq(v)
    m = v.mean(-1)
    h = np.degrees(np.arctan2(q, p))
    d = np.radians(sample_periodic(periodic_curve(s["hue_vs_hue"]), h))
    sg = sample_periodic(periodic_curve(s["hue_vs_sat"]), h)
    c, sn = np.cos(d), np.sin(d)
    p2, q2 = sg * (c * p - sn * q), sg * (sn * p + c * q)
    v2 = from_pq(m, p2, q2)
    v = v2 + ((v @ W709) - (v2 @ W709))[..., None]
    # 3. element grades
    for e in ELEMENTS:
        a = masks.get(e)
        g = s["elements"][e]
        ident = abs(g["exposure_ev"]) < 1e-9 and (e == "subject" or (abs(g["hue_deg"]) < 1e-9 and abs(g["chroma"] - 1) < 1e-9))
        if a is None or ident:
            continue
        m = v.mean(-1)
        p, q = pq(v)
        if e != "subject":
            r = math.radians(g["hue_deg"])
            cc, ss = math.cos(r) * g["chroma"], math.sin(r) * g["chroma"]
            p, q = cc * p - ss * q, ss * p + cc * q
        loc = from_pq(m + g["exposure_ev"] * STOP, p, q)
        v = v + a[..., None] * (loc - v)
    return v


# ------------------------------------------------------------------------------------------------------------------------ display (the sketch's, not the product's)
def tone(x):
    x = np.maximum(x, 0)
    return np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0, 1)


def srgb(y):
    return np.where(y <= 0.0031308, 12.92 * y, 1.055 * np.power(np.maximum(y, 1e-9), 1 / 2.4) - 0.055)


def display(lin):
    return np.clip(srgb(tone(lin)), 0, 1)


def lstar(rgb_disp):
    lin = np.where(rgb_disp <= 0.04045, rgb_disp / 12.92, ((rgb_disp + 0.055) / 1.055) ** 2.4)
    y = lin @ np.array([0.2126, 0.7152, 0.0722])
    f = np.where(y > 216 / 24389, np.cbrt(y), (24389 / 27 * y + 16) / 116)
    return 116 * f - 16


def box(a, r):
    """Mean over a (2r+1) window cut at the frame (a mean over the pixels that exist), via cumulative sums."""
    def one(x, axis):
        n = x.shape[axis]
        c = np.cumsum(x, axis=axis)
        c = np.concatenate([np.zeros_like(np.take(c, [0], axis=axis)), c], axis=axis)
        i = np.arange(n)
        lo = np.clip(i - r, 0, n); hi = np.clip(i + r + 1, 0, n)
        s = np.take(c, hi, axis=axis) - np.take(c, lo, axis=axis)
        shape = [1] * x.ndim; shape[axis] = n
        return s / (hi - lo).reshape(shape)
    return one(one(a, 0), 1)


def soft_mask(binary, guide, short_side):
    """Section 4a.3 step 2: a guided filter, radius max(1, round(0.02 short side)), eps 1e-3, guide L*/100."""
    r = max(1, int(round(0.02 * short_side)))
    I = guide; p = binary.astype(np.float64)
    mI, mp = box(I, r), box(p, r)
    cov = box(I * p, r) - mI * mp
    var = box(I * I, r) - mI * mI
    a = cov / (var + 1e-3)
    b = mp - a * mI
    return np.clip(box(a, r) * I + box(b, r), 0, 1)

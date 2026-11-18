#!/usr/bin/env python3
"""A numpy sketch of the Colour Compressor of docs/32, used only to draw the mock-ups and to try the maths before it is built.

Not product code: the product's reference is `ym_color::compress` (task CC-02), in f64 in Rust. Every formula here is the one in
docs/32 section 3, in the same order. Run `python3 compress_ref.py` to run the property checks of docs/32 section 7 on this sketch.
"""
import math

import numpy as np

# LogC4 (ARRI's published constants, as crates/ym-color/src/consts.rs holds them)
A = (2.0 ** 18 - 16.0) / 117.45
B = (1023.0 - 95.0) / 1023.0
C = 95.0 / 1023.0
S_ = (7.0 * math.log(2.0) * 2.0 ** (7.0 - 14.0 * C / B)) / (A * B)
T_ = (2.0 ** (14.0 * (0.0 - C) / B + 6.0) - 64.0) / A
STOP = B / 14.0  # one stop in LogC4 code values
GREY = (math.log2(A * 0.18 + 64.0) - 6.0) / 14.0 * B + C  # 0.2784

E1 = np.array([1.0, -0.5, -0.5]) / math.sqrt(1.5)
E2 = np.array([0.0, 1.0, -1.0]) / math.sqrt(2.0)
ONE = np.array([1.0, 1.0, 1.0])

NEUTRAL_GUARD = 0.02  # |c| below which a colour has no hue (the warp's WARP_NEUTRAL_GUARD)
HUE_RAMP_MIN = 0.05  # radians, about 2.9 degrees
RAMP_MIN = 0.02  # chroma and tone, LogC4 code values (the masks' RANGE_RAMP_MIN)
RANGE_DEFAULT = dict(hue_deg=20.0, chroma=0.05, tone_ev=1.5, softness=0.5)


def encode(x):
    x = np.asarray(x, dtype=np.float64)
    hi = (np.log2(A * np.maximum(x, T_) + 64.0) - 6.0) / 14.0 * B + C
    return np.where(x >= T_, hi, (x - T_) / S_)


def decode(e):
    e = np.asarray(e, dtype=np.float64)
    hi = (np.exp2(14.0 * (np.maximum(e, 0.0) - C) / B + 6.0) - 64.0) / A
    return np.where(e >= 0.0, hi, e * S_ + T_)


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def window(d, width, q):
    """1 inside `width`, 0 beyond `width + q`, smooth between."""
    return 1.0 - smooth((d - width) / q)


def coords(v):
    """(c1, c2, m): the LogC4 triple in the orthonormal basis e1, e2, (1,1,1)/sqrt(3). m is the mean of the three codes."""
    v = np.asarray(v, dtype=np.float64)
    return v @ E1, v @ E2, v.mean(axis=-1)


def target_from(v):
    c1, c2, m = coords(v)
    return dict(chroma=(float(c1), float(c2)), tone=float(m))


def clamp_target(t):
    a, b = t["chroma"]
    r = math.hypot(a, b)
    if r < NEUTRAL_GUARD:
        a, b = (NEUTRAL_GUARD, 0.0) if r == 0 else (a / r * NEUTRAL_GUARD, b / r * NEUTRAL_GUARD)
    elif r > 0.5:
        a, b = a / r * 0.5, b / r * 0.5
    return dict(chroma=(a, b), tone=min(max(t["tone"], 0.0), 1.0))


def weight(v, target, rng=RANGE_DEFAULT):
    """The selection weight W in [0, 1] of each pixel of v (..., 3), and the pieces the law needs."""
    t = clamp_target(target)
    ta, tb = t["chroma"]
    rt = math.hypot(ta, tb)
    c1, c2, m = coords(v)
    r = np.hypot(c1, c2)
    dot = c1 * ta + c2 * tb
    cross = ta * c2 - tb * c1
    ok = r > 0.0
    delta = np.where(ok, np.arctan2(np.where(ok, cross, 0.0), np.where(ok, dot, 1.0)), 0.0)
    sg = rng["softness"]
    wh_w = math.radians(rng["hue_deg"])
    wr_w = rng["chroma"]
    wm_w = rng["tone_ev"] * STOP
    wh = window(np.abs(delta), wh_w, max(sg * wh_w, HUE_RAMP_MIN))
    wr = window(np.abs(r - rt), wr_w, max(sg * wr_w, RAMP_MIN))
    wm = window(np.abs(m - t["tone"]), wm_w, max(sg * wm_w, RAMP_MIN))
    g = smooth(r / NEUTRAL_GUARD)
    return g * wh * wr * wm, dict(c1=c1, c2=c2, m=m, r=r, delta=delta, rt=rt, ta=ta, tb=tb, tone=t["tone"])


def compress(v, target, hue=0.0, sat=0.0, light=0.0, rng=RANGE_DEFAULT):
    """The compression of docs/32 section 3.4 on LogC4 triples v (..., 3). Returns the new triples and W."""
    v = np.asarray(v, dtype=np.float64)
    if target is None or (hue == 0.0 and sat == 0.0 and light == 0.0):
        return v.copy(), np.zeros(v.shape[:-1])
    W, p = weight(v, target, rng)
    c1, c2, m, r, delta = p["c1"], p["c2"], p["m"], p["r"], p["delta"]
    rt, tone = p["rt"], p["tone"]
    r2 = rt + (r - rt) * (1.0 - sat * W)
    if hue != 0.0:
        d2 = delta * (1.0 - hue * W)
        th = math.atan2(p["tb"], p["ta"]) + d2
        n1, n2 = r2 * np.cos(th), r2 * np.sin(th)
    else:
        k = np.where(r > 0.0, r2 / np.where(r > 0.0, r, 1.0), 1.0)
        n1, n2 = c1 * k, c2 * k
    m2 = tone + (m - tone) * (1.0 - light * W)
    out = v + (n1 - c1)[..., None] * E1 + (n2 - c2)[..., None] * E2 + (m2 - m)[..., None] * ONE
    return out, W


# ---------------------------------------------------------------------------------------------------------- property checks
def _checks():
    rs = np.random.default_rng(7)
    tgt = dict(chroma=(0.06, 0.035), tone=0.40)
    # identity at zero amounts, and with no target
    v = rs.uniform(0.0, 0.9, (5000, 3))
    assert np.array_equal(compress(v, tgt)[0], v) and np.array_equal(compress(v, None, 1, 1, 1)[0], v)
    # the fixed point: the target colour does not move, at any amount
    vt = tone_chroma_to_rgb(tgt)
    assert np.allclose(compress(vt, tgt, 1, 1, 1)[0], vt, atol=1e-12)
    # 100 % pulls a colour at the centre of the window to the target on that axis (W = 1 at the target; test a near colour)
    near = vt + 0.004 * E1 + 0.003 * ONE
    o, W = compress(near, tgt, 1, 1, 1)
    assert W > 0.99
    c1, c2, m = coords(o)
    assert math.hypot(c1 - tgt["chroma"][0], c2 - tgt["chroma"][1]) < 5e-3 and abs(m - tgt["tone"]) < 5e-3 * 2
    # no NaN or inf on grey, black, huge and negative codes
    odd = np.array([[0, 0, 0], [0.3, 0.3, 0.3], [1, 1, 1], [-0.5, -0.5, -0.5], [5, -5, 0], [1e-12, 0, 0]], dtype=float)
    o, _ = compress(odd, tgt, 1, 1, 1)
    assert np.isfinite(o).all()
    # greys do not move (W = 0 at r = 0)
    grey = np.stack([np.linspace(-0.2, 1, 50)] * 3, -1)
    assert np.allclose(compress(grey, tgt, 1, 1, 1)[0], grey, atol=1e-15)
    # continuity across the hue wrap: the target on the negative axis, a sweep through +-180 degrees stays continuous
    for ang in (0.0, 1.0, math.pi - 0.2, math.pi):
        t2 = dict(chroma=(0.07 * math.cos(ang), 0.07 * math.sin(ang)), tone=0.4)
        th = np.linspace(-math.pi, math.pi, 20001)
        v = 0.4 + 0.07 * (np.cos(th)[:, None] * E1 + np.sin(th)[:, None] * E2)
        o, _ = compress(v, t2, 1, 0.5, 0.5, dict(hue_deg=85, chroma=0.1, tone_ev=3, softness=1))
        step = np.abs(np.diff(o, axis=0)).max()
        assert step < 5e-3, (ang, step)
    # monotone along each axis at fixed other coordinates (non-decreasing; strict below 100 %)
    wide = dict(hue_deg=60, chroma=0.1, tone_ev=3, softness=0.5)
    for k in (0.3, 0.8, 1.0):
        d = np.linspace(-math.pi, math.pi, 4001)
        t3 = tgt
        ta = math.atan2(t3["chroma"][1], t3["chroma"][0])
        th = ta + d
        v = 0.40 + 0.06 * (np.cos(th)[:, None] * E1 + np.sin(th)[:, None] * E2)
        o, _ = compress(v, t3, k, 0, 0, wide)
        c1, c2, _m = coords(o)
        ang = np.unwrap(np.arctan2(c2, c1) - ta)
        assert (np.diff(ang) >= -1e-9).all()
        rr = np.linspace(0.0, 0.3, 4001)
        v = 0.40 + rr[:, None] * (math.cos(ta + 0.1) * E1 + math.sin(ta + 0.1) * E2)
        o, _ = compress(v, t3, 0, k, 0, wide)
        c1, c2, _m = coords(o)
        assert (np.diff(np.hypot(c1, c2)) >= -1e-9).all()
        mm = np.linspace(-0.2, 1.1, 4001)
        v = mm[:, None] + 0.06 * (math.cos(ta) * E1 + math.sin(ta) * E2)
        o, _ = compress(v, t3, 0, 0, k, wide)
        assert (np.diff(coords(o)[2]) >= -1e-9).all()
    print("compress_ref: all property checks pass")


def tone_chroma_to_rgb(t):
    return t["tone"] * ONE + t["chroma"][0] * E1 + t["chroma"][1] * E2


if __name__ == "__main__":
    _checks()

#!/usr/bin/env python3
"""A numpy sketch of the Colour Uniform node of docs/32, used only to draw the mock-ups and to try the maths before it is built.

Not product code: the product's reference is `ym_color::uniform` (task CU-02), in f64 in Rust. Every formula here is the one in docs/32
section 3, in the same order. The node has no range of its own: its selection is its masks (`m`, applied by `with_mask`), and the pulls carry
only two fixed guards (the neutral guard on hue and saturation, the antipode fade on hue). Run `python3 compress_ref.py` to run the property
checks of docs/32 section 8 on this sketch and print their numbers.
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
HUE_FULL = math.pi / 2  # the hue pull is full up to 90 degrees from the target's hue ...
HUE_FADE = math.pi / 3  # ... and fades to nothing over the next 60 degrees (zero from 150 degrees on)
RANGE_RAMP_MIN = 0.02  # the Colour range mask's floor on its edge (RANGE_RAMP_MIN)
RANGE_DEFAULT = dict(width=0.05, softness=0.5)  # a Colour range row's own defaults (docs/23 section 3.3)


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


def coords(v):
    """(c1, c2, m): the LogC4 triple in the orthonormal basis e1, e2, (1,1,1)/sqrt(3). m is the mean of the three codes."""
    v = np.asarray(v, dtype=np.float64)
    return v @ E1, v @ E2, v.mean(axis=-1)


def target_from(v):
    c1, c2, m = coords(v)
    return dict(chroma=(float(c1), float(c2)), tone=float(m))


def clamp_target(t):
    """The target is held off grey: chroma 0.02 or more (Saturation 2)."""
    a, b = t["chroma"]
    r = math.hypot(a, b)
    if r < NEUTRAL_GUARD:
        a, b = (NEUTRAL_GUARD, 0.0) if r == 0 else (a / r * NEUTRAL_GUARD, b / r * NEUTRAL_GUARD)
    elif r > 0.5:
        a, b = a / r * 0.5, b / r * 0.5
    return dict(chroma=(a, b), tone=min(max(t["tone"], 0.0), 1.0))


def guards(c1, c2, tgt):
    """(g, a, delta, r): the neutral guard g(r) and the antipode fade a(delta) of docs/32 section 3.2, the signed short-way-round angle, and |c|."""
    ta, tb = tgt["chroma"]
    r = np.hypot(c1, c2)
    ok = r > 0.0
    delta = np.where(ok, np.arctan2(np.where(ok, ta * c2 - tb * c1, 0.0), np.where(ok, ta * c1 + tb * c2, 1.0)), 0.0)
    g = smooth(r / NEUTRAL_GUARD)
    a = 1.0 - smooth((np.abs(delta) - HUE_FULL) / HUE_FADE)
    return g, a, delta, r


def uniform(v, target, hue=0.0, sat=0.0, light=0.0):
    """The pulls of docs/32 section 3.3 on LogC4 triples v (..., 3), with the mask at 1. Returns the new triples."""
    v = np.asarray(v, dtype=np.float64)
    if target is None or (hue == 0.0 and sat == 0.0 and light == 0.0):
        return v.copy()
    t = clamp_target(target)
    ta, tb = t["chroma"]
    rt = math.hypot(ta, tb)
    th_t = math.atan2(tb, ta)
    c1, c2, m = coords(v)
    g, a, delta, r = guards(c1, c2, t)
    r2 = rt + (r - rt) * (1.0 - sat * g)  # saturation: the distance to the target's chroma shrinks, not for neutrals
    if hue != 0.0:
        th = th_t + delta * (1.0 - hue * g * a)  # hue: the angle to the target's hue shrinks, not for neutrals, not for the far side
        n1, n2 = r2 * np.cos(th), r2 * np.sin(th)
        n1, n2 = np.where(r > 0.0, n1, 0.0), np.where(r > 0.0, n2, 0.0)
    else:
        k = np.where(r > 0.0, r2 / np.where(r > 0.0, r, 1.0), 1.0)
        n1, n2 = c1 * k, c2 * k
    m2 = t["tone"] + (m - t["tone"]) * (1.0 - light)  # lightness: everything in the mask
    return v + (n1 - c1)[..., None] * E1 + (n2 - c2)[..., None] * E2 + (m2 - m)[..., None] * ONE


def color_range(v, center, width=RANGE_DEFAULT["width"], softness=RANGE_DEFAULT["softness"]):
    """The Colour range mask row of docs/algorithms/06 section 4.3: 1 within `width` of the centre in the chroma plane, 0 beyond width + q."""
    c1, c2, _m = coords(v)
    d = np.hypot(c1 - center[0], c2 - center[1])
    q = max(softness * width, RANGE_RAMP_MIN)
    return 1.0 - smooth((d - width) / q)


def with_mask(v, node_out, m):
    """out = in + m (node(in) - in): the node's mask blend (docs/algorithms/06 section 1.1)."""
    return v + m[..., None] * (node_out - v)


# ---------------------------------------------------------------------------------------------------------- property checks
def tone_chroma_to_rgb(t):
    return t["tone"] * ONE + t["chroma"][0] * E1 + t["chroma"][1] * E2


def _checks():
    rs = np.random.default_rng(7)
    tgt = dict(chroma=(0.0875, 0.0423), tone=0.2767)  # the mock-ups' skin target, hue 25.8, saturation 9.7
    ta = math.atan2(tgt["chroma"][1], tgt["chroma"][0])
    rt = math.hypot(*tgt["chroma"])
    res = {}
    # identity at zero amounts, and with no target
    v = rs.uniform(0.0, 0.9, (5000, 3))
    assert np.array_equal(uniform(v, tgt), v) and np.array_equal(uniform(v, None, 1, 1, 1), v)
    # the fixed point: the target colour does not move, at any amount
    vt = tone_chroma_to_rgb(tgt)
    res["fixed_point_err"] = float(np.abs(uniform(vt, tgt, 1, 1, 1) - vt).max())
    assert res["fixed_point_err"] < 1e-12
    # 100 % puts a colour within 90 degrees of the target's hue, above the guard, on the target's hue, saturation and lightness
    dd = rs.uniform(-HUE_FULL, HUE_FULL, 4000)
    rr = rs.uniform(NEUTRAL_GUARD, 0.3, 4000)
    mm = rs.uniform(0.0, 1.0, 4000)
    th = ta + dd
    v = mm[:, None] * ONE + (rr * np.cos(th))[:, None] * E1 + (rr * np.sin(th))[:, None] * E2
    o = uniform(v, tgt, 1, 1, 1)
    c1, c2, m = coords(o)
    res["hundred_pct_err"] = float(max(np.abs(np.hypot(c1, c2) - rt).max(), np.abs(m - tgt["tone"]).max(), np.abs(np.arctan2(c2, c1) - ta).max()))
    assert res["hundred_pct_err"] < 1e-9
    # 50 % is halfway on each axis (inside the full zone, above the guard)
    o = uniform(v, tgt, 0.5, 0.5, 0.5)
    c1, c2, m = coords(o)
    err = max(np.abs(np.hypot(c1, c2) - (rt + (rr - rt) * 0.5)).max(), np.abs(m - (tgt["tone"] + (mm - tgt["tone"]) * 0.5)).max(), np.abs(np.arctan2(c2, c1) - (ta + dd * 0.5)).max())
    res["fifty_pct_err"] = float(err)
    assert err < 1e-12
    # no NaN or inf on grey, black, huge and negative codes
    odd = np.array([[0, 0, 0], [0.3, 0.3, 0.3], [1, 1, 1], [-0.5, -0.5, -0.5], [5, -5, 0], [1e-12, 0, 0], [1e-300, 0, 0]], dtype=float)
    assert np.isfinite(uniform(odd, tgt, 1, 1, 1)).all()
    # greys keep no tint: with Hue and Saturation on and Lightness 0 a grey ramp moves by rounding only (1e-15); with Lightness it stays grey
    grey = np.stack([np.linspace(-0.2, 1, 50)] * 3, -1)
    res['grey_move_err'] = float(np.abs(uniform(grey, tgt, 1, 1, 0) - grey).max())
    assert res['grey_move_err'] < 1e-15
    og = uniform(grey, tgt, 1, 1, 1)
    c1, c2, _ = coords(og)
    assert np.abs(c1).max() < 1e-15 and np.abs(c2).max() < 1e-15
    # a near-neutral pixel gains little chroma: the most chroma added to a pixel of chroma <= x at 100 % hue and saturation
    bound = {}
    for x in (0.001, 0.005, 0.01, 0.02):
        rr2 = np.linspace(0, x, 401)
        v = 0.3 + rr2[:, None] * (math.cos(ta + 1.0) * E1 + math.sin(ta + 1.0) * E2)
        c1, c2, _ = coords(uniform(v, tgt, 1, 1, 0))
        bound[x] = float((np.hypot(c1, c2) - rr2).max())
    res["added_chroma_at_100pct"] = bound
    # continuity: a sweep of the whole hue circle (the seam at 0/360 and the antipode of the target) has no jump; targets at 0, 90, 180, 359.99 degrees
    steps = {}
    for ang in (0.0, math.pi / 2, math.pi, math.radians(359.99)):
        t2 = dict(chroma=(0.07 * math.cos(ang), 0.07 * math.sin(ang)), tone=0.4)
        thx = np.linspace(-math.pi, math.pi, 36001)  # 0.01 degree
        v = 0.4 + 0.07 * (np.cos(thx)[:, None] * E1 + np.sin(thx)[:, None] * E2)
        o = uniform(v, t2, 1, 0.5, 0.5)
        steps[round(math.degrees(ang), 2)] = float(np.abs(np.diff(o, axis=0)).max())
        o1 = uniform(v, t2, 0.7, 0.7, 0)
        steps[round(math.degrees(ang), 2)] = max(steps[round(math.degrees(ang), 2)], float(np.abs(np.diff(o1, axis=0)).max()))
    res["max_step_per_0.01deg"] = steps
    assert max(steps.values()) < 5e-3
    # monotone along each axis at fixed other coordinates (non-decreasing; strictly below 100 %)
    for k in (0.3, 0.8, 1.0):
        d = np.linspace(-3.1, 3.1, 4001)
        thx = ta + d
        v = 0.40 + 0.06 * (np.cos(thx)[:, None] * E1 + np.sin(thx)[:, None] * E2)
        c1, c2, _ = coords(uniform(v, tgt, k, 0, 0))
        ang = np.arctan2(c2, c1) - ta
        ang = (ang + math.pi) % (2 * math.pi) - math.pi  # signed angle from the target
        assert (np.diff(ang) >= -1e-9).all(), ("hue", k)
        if k < 1.0:
            assert (np.diff(ang) > 0).all()
        rr3 = np.linspace(0.0, 0.3, 4001)
        v = 0.40 + rr3[:, None] * (math.cos(ta + 0.1) * E1 + math.sin(ta + 0.1) * E2)
        c1, c2, _ = coords(uniform(v, tgt, 0, k, 0))
        assert (np.diff(np.hypot(c1, c2)) >= -1e-12).all(), ("sat", k)
        c1, c2, _ = coords(uniform(v, tgt, k, k, 0))
        assert (np.diff(np.hypot(c1, c2)) >= -1e-12).all(), ("sat with hue", k)
        mm2 = np.linspace(-0.2, 1.1, 4001)
        v = mm2[:, None] + 0.06 * (math.cos(ta) * E1 + math.sin(ta) * E2)
        om = coords(uniform(v, tgt, 0, 0, k))[2]
        assert (np.diff(om) >= -1e-12).all(), ("light", k)
        if k < 1.0:
            assert (np.diff(om) > 0).all()
    # the Jacobian of the whole map, by central differences in LogC4 space: positive up to 90 %, never meaningfully negative at 100 %
    pts = np.concatenate([rs.uniform(-0.2, 1.1, (100000, 1)) + rs.uniform(-0.3, 0.3, (100000, 3)), rs.uniform(0, 0.02, (100000, 1)) + rs.normal(0, 0.003, (100000, 3))])
    h = 1e-6
    jac = {}
    for k in (0.5, 0.9, 0.99, 1.0):
        cols = []
        for i in range(3):
            e = np.zeros(3)
            e[i] = h
            cols.append((uniform(pts + e, tgt, k, k, k) - uniform(pts - e, tgt, k, k, k)) / (2 * h))
        J = np.stack(cols, -1)
        det = np.linalg.det(J)
        jac[k] = float(det.min())
    res["jacobian_min"] = jac
    assert jac[0.5] > 0 and jac[0.9] > 0 and jac[1.0] > -1e-6
    # the Colour range mask: 1 at its centre; the node at mask 0 is the input, bit for bit, and at mask 1 is the node
    ctr = tgt["chroma"]
    assert abs(float(color_range(vt, ctr)) - 1.0) < 1e-15
    v = rs.uniform(0.0, 0.9, (3000, 3))
    node = uniform(v, tgt, 0.6, 0.6, 0)
    assert np.array_equal(with_mask(v, node, np.zeros(3000)), v) and np.allclose(with_mask(v, node, np.ones(3000)), node, atol=1e-15)
    # a worked example (docs/32 section 3.3): a pixel 20 degrees from the target at the target's saturation +0.5 and tone +0.5 EV
    px = tgt["tone"] + 0.5 * STOP
    thx = ta + math.radians(20)
    vx = px * ONE + (0.12 * math.cos(thx)) * E1 + (0.12 * math.sin(thx)) * E2
    c = coords(uniform(vx, tgt, 0.5, 0.5, 0.5))
    c100 = coords(uniform(vx, tgt, 1, 1, 1))
    res["worked_example"] = dict(
        before=dict(hue=round(math.degrees(thx) % 360, 2), sat=0.12 * 100, ev=0.5),
        at_50=dict(hue=round(math.degrees(math.atan2(c[1], c[0])), 3), sat=round(100 * math.hypot(c[0], c[1]), 3), ev=round((c[2] - GREY) / STOP, 3)),
        at_100=dict(hue=round(math.degrees(math.atan2(c100[1], c100[0])), 3), sat=round(100 * math.hypot(c100[0], c100[1]), 3), ev=round((c100[2] - GREY) / STOP, 3)),
        target=dict(hue=round(math.degrees(ta), 3), sat=round(rt * 100, 3), ev=round((tgt["tone"] - GREY) / STOP, 3)),
    )
    print("compress_ref: all property checks pass")
    import json

    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    _checks()

"""The cine rendering (docs/02-colour-pipeline.md, C12): reference model in numpy, float64.

Scene-linear ARRI Wide Gamut 4 in, display-linear RGB out (Rec.709, Display P3 or Rec.2020 primaries, 1.0 =
the SDR white). It evaluates the model's equations and nothing else: no ARRI file, LUT value or target is read
or needed here. The 56 constants are in ``constants.json`` (the table of the spec is the same numbers).

``forward`` is the rendering. ``inverse`` is the slow, accurate reference inverse (damped fixed point, then
Newton steps with a finite-difference Jacobian, a bisection for the shared display luminance) that the
fast inverse of the product is tested against, including the nearest-reachable rule for display colours that
the rendering cannot produce.

Python is for fixtures and training only; nothing in the app or ``ym`` runs it.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------------------------- colour spaces
D65 = (0.3127, 0.3290)


def _xyz_of_xy(xy):
    return np.array([xy[0] / xy[1], 1.0, (1.0 - xy[0] - xy[1]) / xy[1]])


def primaries_to_xyz(r, g, b, white=D65):
    """The normalised primary matrix (linear RGB to XYZ) of three chromaticities and a white."""
    p = np.stack([_xyz_of_xy(r), _xyz_of_xy(g), _xyz_of_xy(b)], axis=1)
    s = np.linalg.solve(p, _xyz_of_xy(white))
    return p * s[None, :]


AWG4_TO_XYZ = primaries_to_xyz((0.7347, 0.2653), (0.1424, 0.8576), (0.0991, -0.0308))
OUTPUTS = {
    "rec709": primaries_to_xyz((0.640, 0.330), (0.300, 0.600), (0.150, 0.060)),
    "p3": primaries_to_xyz((0.680, 0.320), (0.265, 0.690), (0.150, 0.060)),
    "rec2020": primaries_to_xyz((0.708, 0.292), (0.170, 0.797), (0.131, 0.046)),
}
WHITE = _xyz_of_xy(D65)
XN, ZN = float(WHITE[0]), float(WHITE[2])

# Numeric guards and fixed settings (not fitted, not tunable).
SCENE_LIMIT = 1.0e6      # |scene channel| is held to this before anything else
Y_MIN = 1.0e-9           # smallest scene luminance the colour terms see
Y2_MIN = 1.0e-12         # smallest display luminance the gamut stage divides by
DELTA = 1.0e-4           # width of the smooth cube root
EPS_FLOOR = 1.0e-30      # keeps the luminance floor's square root away from 0
EPS_CHROMA = 1.0e-14     # keeps the chroma length away from 0
GREY = 0.18


def load_constants(path=None):
    with open(path or os.path.join(HERE, "constants.json")) as f:
        return json.load(f)


# ---------------------------------------------------------------------------------------------- pieces
def smooth_cbrt(r, delta=DELTA):
    """Odd, C-infinity cube root: r * ((1 + d^2) / (r^2 + d^2))^(1/3). Equals 1 at r = 1, ~cbrt(r) for |r| >> d."""
    return r * ((1.0 + delta * delta) / (r * r + delta * delta)) ** (1.0 / 3.0)


def soft_clip(z, w):
    """C1 quadratic soft clip: z below 1-w, reaching exactly 1 (slope 0) at 1+w, 1 above."""
    z0, z1 = 1.0 - w, 1.0 + w
    zc = np.clip(z, z0, z1)
    return np.minimum(z - (zc - z0) ** 2 / (4.0 * w), 1.0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def tone(ys, C):
    """Neutral tone curve: scene luminance (grey 0.18) to display luminance, before the gamut stage."""
    a, c, b = np.exp(C["la"]), np.exp(C["lc"]), np.exp(C["lb"])
    h = np.exp(-b * np.log(1.0 + c * np.exp(-a * np.log(ys))))      # (1 + c*Y^-a)^-b
    return soft_clip(C["kap"] * h, C["wc"])


def harmonics(cs, sn, coef):
    """c1 cos + s1 sin + ... up to order 3 of the unit direction (cs, sn); coef = [c1, s1, c2, s2, c3, s3]."""
    c2 = cs * cs - sn * sn
    s2 = 2.0 * cs * sn
    c3 = cs * c2 - sn * s2
    s3 = sn * c2 + cs * s2
    return coef[0] * cs + coef[1] * sn + coef[2] * c2 + coef[3] * s2 + coef[4] * c3 + coef[5] * s3


def hue_coefs(C, q, base):
    return [C[f"{q}_{base}_{t}{k}"] for k in (1, 2, 3) for t in ("c", "s")]


def core(ys, ca, cb, C, with_terms=False):
    """The colour terms: scene (Ys, ca, cb) to display (Y2, ca2, cb2). ca, cb are after the chroma limit."""
    chi = np.sqrt(ca * ca + cb * cb + EPS_CHROMA)
    cs, sn = ca / chi, cb / chi
    u = np.log2(ys / GREY)
    yt = tone(ys, C)
    g0 = np.exp(C["lgpk"]) / (1.0 + np.exp(C["gb"] * (np.log(ys) - C["lYc"])))
    bh = sigmoid((u - 1.5) / 1.5)
    env = chi * chi / (chi * chi + np.exp(2.0 * C["lchih"]))
    gh = env * (harmonics(cs, sn, hue_coefs(C, "gh", "b0")) + bh * harmonics(cs, sn, hue_coefs(C, "gh", "bh")))
    dt = env * (harmonics(cs, sn, hue_coefs(C, "dt", "b0")) + bh * harmonics(cs, sn, hue_coefs(C, "dt", "bh")))
    rh = env * (harmonics(cs, sn, hue_coefs(C, "rh", "b0")) + bh * harmonics(cs, sn, hue_coefs(C, "rh", "bh")))
    rp = C["rp"]
    chenv = chi ** rp / (chi ** rp + np.exp(rp * C["r_chi"]))
    lr = chenv * (C["r_dark"] * sigmoid(-(u - C["r_Yd"])) + C["r_hi"]) + rh
    g = g0 * np.exp(gh)
    cd, sd = np.cos(dt), np.sin(dt)
    ca2 = g * (ca * cd - cb * sd)
    cb2 = g * (ca * sd + cb * cd)
    y2 = yt * np.exp(lr)
    if with_terms:
        return y2, ca2, cb2, {"chi": chi, "u": u, "yt": yt, "g0": g0, "gh": gh, "dt": dt, "rh": rh, "lr": lr, "g": g}
    return y2, ca2, cb2


def luminance_floor(xyz, C):
    """Smooth max of the luminance and rho * (distance of the chromaticity from the neutral axis)."""
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    yf = C["rho"] * np.sqrt((x - XN * y) ** 2 + (z - ZN * y) ** 2 + EPS_FLOOR)
    w = C["wfrac"] * yf
    return 0.5 * (y + yf + np.sqrt((y - yf) ** 2 + w * w))


def chroma_limit(ca, cb, C):
    """C1 soft limit of the relative chroma: identity up to chi_k, tends to chi_m."""
    k, m = C["chi_k"], C["chi_m"]
    chi = np.sqrt(ca * ca + cb * cb + EPS_CHROMA)
    e = np.maximum(chi - k, 0.0)
    chil = np.minimum(chi, k) + (m - k) * np.tanh(np.minimum(e / (m - k), 20.0))
    s = chil / chi
    return ca * s, cb * s


def scene_polar(x, C):
    """Scene AWG4 to (Ys, ca, cb) after the luminance floor and the chroma limit."""
    x = np.clip(np.asarray(x, dtype=np.float64), -SCENE_LIMIT, SCENE_LIMIT)
    xyz = x @ AWG4_TO_XYZ.T
    ys = np.maximum(luminance_floor(xyz, C), Y_MIN)
    ca = smooth_cbrt(xyz[..., 0] / (XN * ys)) - 1.0
    cb = 1.0 - smooth_cbrt(xyz[..., 2] / (ZN * ys))
    ca, cb = chroma_limit(ca, cb, C)
    return ys, ca, cb


def display_xyz(x, C):
    """Scene AWG4 to the display XYZ before the gamut stage: (X2, Y2, Z2)."""
    ys, ca, cb = scene_polar(x, C)
    y2, ca2, cb2 = core(ys, ca, cb, C)
    return np.stack([XN * y2 * (1.0 + ca2) ** 3, y2, ZN * y2 * (1.0 - cb2) ** 3], -1)


def gamut_floor(r, k):
    """Soft floor of a channel relative to the luminance: r above k, k*exp((r-k)/k) below (C1, positive)."""
    return np.maximum(r, k) + k * (np.exp(np.minimum(r - k, 0.0) / k) - 1.0)


def gamut_stage(rgb, y2, C):
    """Per-channel soft floor (relative to the luminance) and soft ceiling that reaches exactly 1."""
    yd = np.maximum(y2, Y2_MIN)[..., None]
    kl, kh = np.exp(C["lklo"]), np.exp(C["lkhi"])
    v = gamut_floor(rgb / yd, kl) * yd
    return soft_clip((1.0 + kh) * v, kh)


def forward(x, C, primaries="rec709"):
    """Scene-linear AWG4 (..., 3) to display-linear RGB of the named primaries."""
    xyz2 = display_xyz(x, C)
    rgb = xyz2 @ np.linalg.inv(OUTPUTS[primaries]).T
    return gamut_stage(rgb, xyz2[..., 1], C)


# ---------------------------------------------------------------------------------------------- inverse
D_MIN = 1.0e-6           # a display channel below this is raised to it (an exact 0 has no finite pre-image)
RESIDUAL_TOL = 1.0e-9    # |residual| in (ln Y2, ca2, cb2) below which the core has been inverted
CBRT = np.cbrt


def tone_inverse(t, C):
    """The smallest scene luminance whose tone value is t (0 for t <= 0; the flat top starts at tone_inverse(1))."""
    a, c, b = np.exp(C["la"]), np.exp(C["lc"]), np.exp(C["lb"])
    wc, kap = C["wc"], C["kap"]
    t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
    z0 = 1.0 - wc
    z = np.where(t <= z0, t, z0 + 2.0 * wc * (1.0 - np.sqrt(np.maximum(1.0 - (t - z0) / wc, 0.0))))
    q = np.clip(z / kap, 1e-300, 1.0 - 1e-16) ** (1.0 / b)
    y = (c * q / (1.0 - q)) ** (1.0 / a)
    return np.where(t <= 0.0, 0.0, y)


def smooth_cbrt_inverse(q, delta=DELTA, iters=4):
    """r with smooth_cbrt(r) = q (Newton from q^3; the exact cube root up to delta^2/(3 r^2))."""
    q = np.asarray(q, dtype=np.float64)
    r = q ** 3
    for _ in range(iters):
        f = smooth_cbrt(r, delta) - q
        # derivative of r*((1+d^2)/(r^2+d^2))^(1/3):  s*(1 - 2r^2/(3(r^2+d^2)))/(r^2+d^2)^(1/3) * (1+d^2)^(1/3)
        d2 = delta * delta
        df = ((1.0 + d2) ** (1.0 / 3.0)) * (1.0 - 2.0 * r * r / (3.0 * (r * r + d2))) / (r * r + d2) ** (1.0 / 3.0)
        r = r - f / df
    return r


def ceiling_inverse(o, C):
    """v with soft_clip((1 + kh) v, kh) = o, o in (0, 1]; o = 1 gives v = 1 (the smallest)."""
    kh = np.exp(C["lkhi"])
    o = np.asarray(o, dtype=np.float64)
    z0 = 1.0 - kh
    t = 2.0 * kh * (1.0 - np.sqrt(np.maximum(1.0 - (o - z0) / kh, 0.0)))
    z = np.where(o <= z0, o, z0 + t)
    return z / (1.0 + kh)


def floor_inverse(s, k):
    """r with gamut_floor(r, k) = s (s > 0)."""
    return np.where(s >= k, s, k * (1.0 + np.log(np.maximum(s, 1e-300) / k)))


def gamut_inverse(d, C, primaries="rec709", iters=80):
    """Display RGB to the display XYZ (X2, Y2, Z2) before the gamut stage; d is first held to [D_MIN, 1]."""
    d = np.clip(np.asarray(d, dtype=np.float64), D_MIN, 1.0)
    m = OUTPUTS[primaries]
    yw = m[1]
    k = np.exp(C["lklo"])
    v = ceiling_inverse(d, C)
    lo = np.full(d.shape[:-1], np.log(1e-12))
    hi = np.full(d.shape[:-1], np.log(1e2))
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        y2 = np.exp(mid)
        g = (floor_inverse(v / y2[..., None], k) * yw).sum(-1) - 1.0
        lo = np.where(g > 0, mid, lo)
        hi = np.where(g > 0, hi, mid)
    y2 = np.exp(0.5 * (lo + hi))
    rgb = y2[..., None] * floor_inverse(v / y2[..., None], k)
    return rgb @ m.T


def _core_log(q, C):
    """q = (ln Ys, ca, cb) to (ln Y2, ca2, cb2)."""
    y2, ca2, cb2 = core(np.exp(q[..., 0]), q[..., 1], q[..., 2], C)
    return np.stack([np.log(np.maximum(y2, 1e-300)), ca2, cb2], -1)


def core_inverse(tgt, C, q0=None, fixed_iters=6, newton_iters=60, tol=RESIDUAL_TOL):
    """Solve core(Ys, ca, cb) = (Y2, ca2, cb2) for rows of tgt = (Y2, ca2, cb2).

    Returns q = (ln Ys, ca, cb) and a boolean array: True where the residual is below tol and the
    solution is a valid input of the core (chroma inside the limit, luminance above Y_MIN)."""
    tgt = np.asarray(tgt, dtype=np.float64)
    y2, ca2, cb2 = tgt[:, 0], tgt[:, 1], tgt[:, 2]
    goal = np.stack([np.log(y2), ca2, cb2], 1)
    if q0 is None:
        cha = np.hypot(ca2, cb2)
        th2 = np.arctan2(cb2, ca2)
        y = tone_inverse(np.minimum(y2, 1.0), C)
        y = np.where(y > 0, y, 1e-6)
        chi = cha / np.exp(C["lgpk"])
        th = th2.copy()
        damp = 0.7
        for _ in range(fixed_iters):
            _, _, _, tm = core(y, chi * np.cos(th), chi * np.sin(th), C, with_terms=True)
            yn = tone_inverse(np.minimum(y2 * np.exp(-tm["lr"]), 1.0), C)
            yn = np.where(yn > 0, yn, y)
            y = np.exp((1 - damp) * np.log(np.maximum(y, 1e-300)) + damp * np.log(np.maximum(yn, 1e-300)))
            chi = (1 - damp) * chi + damp * cha * np.exp(-tm["gh"]) / tm["g0"]
            th = th + damp * ((th2 - tm["dt"] - th + np.pi) % (2 * np.pi) - np.pi)
        q = np.stack([np.log(np.maximum(y, 1e-300)), chi * np.cos(th), chi * np.sin(th)], 1)
    else:
        q = np.array(q0, dtype=np.float64)

    def resid(qq):
        r = _core_log(qq, C) - goal
        return np.where(np.isfinite(r), r, 1e3)

    r = resid(q)
    cost = (r * r).sum(1)
    lam = np.full(len(q), 1e-6)
    h = 1e-6
    for _ in range(newton_iters):
        if np.all(np.abs(r).max(1) < tol * 1e-2):
            break
        J = np.empty((len(q), 3, 3))
        for j in range(3):
            dq = np.zeros(3)
            dq[j] = h
            J[:, :, j] = (_core_log(q + dq, C) - _core_log(q - dq, C)) / (2 * h)
        J = np.where(np.isfinite(J), J, 0.0)
        A = J.transpose(0, 2, 1) @ J
        g = J.transpose(0, 2, 1) @ r[..., None]
        scale = np.maximum(np.einsum("nii->n", A), 1e-30)
        step = np.zeros_like(q)
        for _try in range(1):
            Al = A + (lam * scale)[:, None, None] * np.eye(3)[None]
            step = -np.linalg.solve(Al, g)[..., 0]
        step = np.clip(step, -0.7, 0.7)
        qn = q + step
        rn = resid(qn)
        cn = (rn * rn).sum(1)
        ok = cn < cost
        q = np.where(ok[:, None], qn, q)
        r = np.where(ok[:, None], rn, r)
        cost = np.where(ok, cn, cost)
        lam = np.where(ok, np.maximum(lam * 0.3, 1e-12), np.minimum(lam * 10.0, 1e6))
    chil = np.hypot(q[:, 1], q[:, 2])
    ok = (np.abs(r).max(1) < tol) & (chil < C["chi_m"] * (1 - 1e-9)) & (np.exp(q[:, 0]) >= Y_MIN)
    return q, ok


def unlimit_chroma(ca, cb, C):
    """Inverse of chroma_limit."""
    k, m = C["chi_k"], C["chi_m"]
    chil = np.hypot(ca, cb)
    t = np.clip((chil - k) / (m - k), 0.0, 1.0 - 1e-12)
    chi = np.where(chil <= k, chil, k + (m - k) * np.arctanh(t))
    s = np.where(chil > 0, chi / np.maximum(chil, 1e-300), 1.0)
    return ca * s, cb * s


def floor_inverse_luminance(ys, x, z, C, iters=60):
    """Y with luminance_floor((x, Y, z)) = ys, the larger root (Newton from Y = ys, which is an upper bound)."""
    y = np.array(ys, dtype=np.float64)
    for _ in range(iters):
        g = luminance_floor(np.stack([x, y, z], -1), C) - ys
        hh = 1e-7 * np.maximum(np.abs(y), 1e-12)
        g2 = luminance_floor(np.stack([x, y + hh, z], -1), C) - ys
        dg = (g2 - g) / hh
        dg = np.where(np.abs(dg) < 1e-6, 1e-6, dg)
        y = y - np.clip(g / dg, -0.5 * np.abs(y) - 1e-12, 0.5 * np.abs(y) + 1e-12)
    return y


def solve_scene(tgt, C, q0=None):
    """Display (Y2, ca2, cb2) to scene AWG4 and a validity flag.

    Valid means: the core is inverted to RESIDUAL_TOL, the limited chroma is inside the part of the limiter
    that is inverted (at most chi_k + (chi_m - chi_k) tanh(3)), and the luminance floor has a root."""
    q, ok = core_inverse(tgt, C, q0=q0)
    k, m = C["chi_k"], C["chi_m"]
    chil = np.hypot(q[:, 1], q[:, 2])
    ok &= chil <= k + (m - k) * np.tanh(3.0)
    ca, cb = unlimit_chroma(q[:, 1], q[:, 2], C)
    ym = np.exp(q[:, 0])
    x = XN * ym * smooth_cbrt_inverse(1.0 + ca)
    z = ZN * ym * smooth_cbrt_inverse(1.0 - cb)
    y = floor_inverse_luminance(ym, x, z, C)
    back = luminance_floor(np.stack([x, y, z], -1), C)
    ok &= np.abs(back - ym) <= 1e-9 * ym
    out = np.stack([x, y, z], 1) @ np.linalg.inv(AWG4_TO_XYZ).T
    return out, ok, q


def inverse(d, C, primaries="rec709", nearest=True, return_info=False):
    """Display-linear RGB (N, 3) to scene-linear AWG4.

    Display colours that the rendering cannot produce are replaced (nearest=True) by the reachable colour with
    the same display luminance and hue angle and the largest chroma that has a pre-image (bisection); with
    nearest=False they come back as NaN rows."""
    d = np.asarray(d, dtype=np.float64).reshape(-1, 3)
    xyz2 = gamut_inverse(d, C, primaries)
    y2 = np.maximum(xyz2[:, 1], 1e-12)
    ca2 = CBRT(xyz2[:, 0] / (XN * y2)) - 1.0
    cb2 = 1.0 - CBRT(xyz2[:, 2] / (ZN * y2))
    out, ok, _ = solve_scene(np.stack([y2, ca2, cb2], 1), C)
    lam_used = np.ones(len(d))
    if nearest and not ok.all():
        bad = np.where(~ok)[0]
        lo = np.zeros(len(bad))
        hi = np.ones(len(bad))
        olo, _, qlo = solve_scene(np.stack([y2[bad], 0 * ca2[bad], 0 * cb2[bad]], 1), C)
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            om, okm, qm = solve_scene(np.stack([y2[bad], mid * ca2[bad], mid * cb2[bad]], 1), C, q0=qlo)
            lo = np.where(okm, mid, lo)
            hi = np.where(okm, hi, mid)
            qlo = np.where(okm[:, None], qm, qlo)
            olo = np.where(okm[:, None], om, olo)
        out[bad] = olo
        lam_used[bad] = lo
        ok = np.ones(len(d), dtype=bool)
    out[~ok] = np.nan
    if return_info:
        return out, {"ok": ok, "lam": lam_used}
    return out

#!/usr/bin/env python3
"""A numpy sketch of the Blur node's maths (docs/33 sections 3 to 5). NOT product code: the product's reference is `ym_color::blur` (BL-02), in f64 in Rust.

It is used to try every statement of the spec and to draw the mock-up pictures, on a synthetic scene (blur_demo.py). It works in linear light, as the node does
(the mock-up's scene is made in linear light, so no LogC4 decode is needed here).

One law for every type (docs/33 section 3.2):  B(p) = sum_q k(p, q) w(q) X(q) / sum_q k(p, q) w(q),   out = in + m (B - in)
with w = m ("respect mask edges", the default) or w = 1 (the mask only blends), times a highlight weight for the Lens type; k is the type's kernel.

Run `python3 -I blur_ref.py` to run the property checks of docs/33 section 8 on the sketch and print their numbers.
"""
import math
import numpy as np

SIGMA_CUT = 4.0           # a Gaussian is cut at 4 sigma (docs/33 3.4)
RADIUS_MAX = 0.04         # Radius 100 is 4 % of the picture's long edge (docs/33 2.2)
MOTION_MAX = 0.08         # Length 100 is 8 % of the long edge
ZOOM_MAX = 0.10           # Amount 100: the scale factor spans 10 % (a pixel at distance d is spread over 0.1 d)
EPS = 1e-12               # keeps the quotient finite where no weight is in reach (docs/33 3.3); it is not a threshold: the term it guards is multiplied by m


# ------------------------------------------------------------------------------------------------------------------ kernels
def gauss_kernel(sigma):
    """Pixel-integrated Gaussian cut at 4 sigma, normalised (so a sigma near 0 tends to the identity)."""
    if sigma <= 1e-6:
        return np.ones((1, 1))
    h = int(math.ceil(SIGMA_CUT * sigma))
    x = np.arange(-h - 0.5, h + 1.5)
    from math import erf
    cdf = np.array([0.5 * (1 + erf(v / (sigma * math.sqrt(2)))) for v in x])
    w = np.diff(cdf)
    w /= w.sum()
    return np.outer(w, w)


def polygon_mask(px, py, n, rot_deg, R):
    """Inside test of a regular n-gon of circumradius R (n = 0: a disc) rotated by rot_deg, for sample points (px, py)."""
    if n == 0:
        return (px * px + py * py) <= R * R
    ok = np.ones(px.shape, bool)
    rot = math.radians(rot_deg)
    apothem = R * math.cos(math.pi / n)
    for i in range(n):
        a = rot + (i + 0.5) * 2 * math.pi / n
        ok &= (px * math.cos(a) + py * math.sin(a)) <= apothem
    return ok


def _disc_quadrant(x, y, R):
    """Area of the disc of radius R inside 0 <= X <= x, 0 <= Y <= y (x, y >= 0), in closed form."""
    x, y = min(x, R), min(y, R)
    if x * x + y * y <= R * R:
        return x * y
    xc = math.sqrt(R * R - y * y)
    prim = lambda t: 0.5 * (t * math.sqrt(max(R * R - t * t, 0.0)) + R * R * math.asin(min(t / R, 1.0)))
    return xc * y + prim(x) - prim(xc)


def disc_rect_area(R, x0, x1, y0, y1):
    """Exact area of the rectangle [x0, x1] x [y0, y1] inside the disc of radius R about the origin (signed quadrants)."""
    s = lambda x, y: math.copysign(1.0, x) * math.copysign(1.0, y) * _disc_quadrant(abs(x), abs(y), R)
    return s(x1, y1) - s(x0, y1) - s(x1, y0) + s(x0, y0)


def polygon_pixel_area(cx, cy, sides, apothem):
    """Exact area of the pixel square centred (cx, cy) inside the convex polygon x nx + y ny <= apothem: the square clipped by each side in turn."""
    poly = [(cx - .5, cy - .5), (cx + .5, cy - .5), (cx + .5, cy + .5), (cx - .5, cy + .5)]
    for nx, ny in sides:
        dist = lambda p: p[0] * nx + p[1] * ny - apothem
        out = []
        for i, p in enumerate(poly):
            q = poly[(i + 1) % len(poly)]
            dp, dq = dist(p), dist(q)
            if dp <= 0:
                out.append(p)
            if (dp < 0 < dq) or (dq < 0 < dp):
                t = dp / (dp - dq)
                out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
        poly = out
        if len(poly) < 3:
            return 0.0
    twice = sum(p[0] * q[1] - q[0] * p[1] for p, q in zip(poly, poly[1:] + poly[:1]))
    return abs(twice) / 2


def aperture_kernel(R, n=0, rot_deg=0.0):
    """Coverage of each pixel by a disc (n = 0) or a regular n-gon of circumradius R: the exact area of the pixel square inside the shape (the disc in closed form, the polygon by clipping), normalised. The same maths as the Rust oracle's aperture_coverage."""
    if R <= 1e-6:
        return np.ones((1, 1))
    h = int(math.ceil(R)) + 1
    cov = np.zeros((2 * h + 1, 2 * h + 1))
    rot = math.radians(rot_deg)
    apothem = R * math.cos(math.pi / max(n, 1))
    sides = [(math.cos(rot + (i + .5) * 2 * math.pi / n), math.sin(rot + (i + .5) * 2 * math.pi / n)) for i in range(n)]
    for iy in range(2 * h + 1):
        for ix in range(2 * h + 1):
            cx, cy = ix - h, iy - h
            near = math.hypot(max(abs(cx) - .5, 0.0), max(abs(cy) - .5, 0.0))
            far = math.hypot(abs(cx) + .5, abs(cy) + .5)
            if n == 0:
                if far * far <= R * R:
                    v = 1.0
                elif near * near >= R * R:
                    v = 0.0
                else:
                    v = min(max(disc_rect_area(R, cx - .5, cx + .5, cy - .5, cy + .5), 0.0), 1.0)
            elif near > R:
                v = 0.0
            elif far <= apothem:
                v = 1.0
            else:
                v = polygon_pixel_area(cx, cy, sides, apothem)
            cov[iy, ix] = v
    return cov / cov.sum()


def line_kernel(length, angle_deg, ss=8):
    """A line of the given length (px) through the pixel at angle_deg (0 = along x, counter-clockwise on screen: y up), each pixel's share the length of line inside it."""
    if length <= 1e-6:
        return np.ones((1, 1))
    h = int(math.ceil(length / 2)) + 1
    k = np.zeros((2 * h + 1, 2 * h + 1))
    a = math.radians(angle_deg)
    n = int(length * ss) + 1
    for i in range(n):
        t = (i + 0.5) / n * length - length / 2
        x, y = t * math.cos(a), -t * math.sin(a)
        ix, iy = int(math.floor(x + 0.5)) + h, int(math.floor(y + 0.5)) + h
        k[iy, ix] += 1
    return k / k.sum()


def conv(a, k):
    """Same-size convolution with zero padding (so what is outside the frame has no weight), by FFT."""
    H, W = a.shape[:2]
    kh, kw = k.shape
    sh, sw = H + kh - 1, W + kw - 1
    fk = np.fft.rfft2(k, (sh, sw))
    def one(p):
        r = np.fft.irfft2(np.fft.rfft2(p, (sh, sw)) * fk, (sh, sw))
        return r[kh // 2: kh // 2 + H, kw // 2: kw // 2 + W]
    if a.ndim == 2:
        return one(a)
    return np.stack([one(a[..., c]) for c in range(a.shape[2])], -1)


# ------------------------------------------------------------------------------------------------------------------ the law
def luma(X):
    return X @ np.array([0.2722287, 0.6740818, 0.0536895])    # AWG4 luminance row (docs/02 C9)


def highlight_weight(X, h):
    """Lens 'Highlights' 0..1: weights bright pixels up in the average (docs/33 3.5). 1 at or below 4 x middle grey, 1 + 15 h from 16 x."""
    if h <= 0:
        return np.ones(X.shape[:2])
    ev = np.log2(np.maximum(luma(X), 1e-6) / 0.18)
    t = np.clip((ev - 2.0) / 2.0, 0, 1)
    return 1 + 15 * h * t * t * (3 - 2 * t)


def norm_conv(X, w, k):
    num = conv(X * w[..., None], k)
    den = conv(w, k)
    return num, den


def blend(X, m, num, den):
    """out = in + m (B - in) with B = num / (den + EPS), written on the difference so a tiny m with a tiny den gives a tiny change (docs/33 3.3)."""
    B = num / (den + EPS)[..., None]
    out = X + m[..., None] * ((num - X * den[..., None]) / (den + EPS)[..., None])
    out = np.where((m == 0)[..., None], X, out)
    return np.where((m == 1)[..., None], B, out)


def sample_bilinear(P, x, y):
    """Bilinear read of planes P (H, W, C) at float pixel centres (x, y); outside the frame reads 0 (no weight)."""
    H, W = P.shape[:2]
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
    fx = (x - x0)[..., None]; fy = (y - y0)[..., None]
    def at(ix, iy):
        ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        v = P[np.clip(iy, 0, H - 1), np.clip(ix, 0, W - 1)]
        return v * ok[..., None]
    return (at(x0, y0) * (1 - fx) * (1 - fy) + at(x0 + 1, y0) * fx * (1 - fy) + at(x0, y0 + 1) * (1 - fx) * fy + at(x0 + 1, y0 + 1) * fx * fy)


def radial_gather(P, centre_px, amount_frac, n=64, spin_deg=0.0):
    """Zoom (spin_deg = 0): taps at scales 1 + a t, t in (-1/2, 1/2); Spin (later): rotations by spin_deg * t. Equal weights; returns the sum of the planes."""
    H, W = P.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    cx, cy = centre_px
    vx, vy = xx - cx, yy - cy
    acc = np.zeros_like(P)
    for i in range(n):
        t = (i + 0.5) / n - 0.5
        if spin_deg == 0.0:
            s = 1 + amount_frac * t
            qx, qy = cx + vx * s, cy + vy * s
        else:
            a = math.radians(spin_deg) * t
            qx, qy = cx + vx * math.cos(a) - vy * math.sin(a), cy + vx * math.sin(a) + vy * math.cos(a)
        acc += sample_bilinear(P, qx, qy)
    return acc / n


def blur_node(X, m, kind="soft", radius=50, respect=True, blades=0, rotation=0.0, highlights=0.0, length=50, angle=0.0, amount=50, centre=(0.5, 0.5), u=None):
    """X: linear RGB (H, W, 3); m: the node's mask (H, W) in [0, 1]. Returns the node's picture. Parameters are the panel's numbers (docs/33 2.2)."""
    H, W = X.shape[:2]
    u = max(H, W) if u is None else u      # the long edge of the whole picture, not of a tile (docs/33 2.2)
    w = (m if respect else np.ones_like(m)).astype(np.float64)
    if kind == "lens":
        w = w * highlight_weight(X, highlights)
    if kind == "soft":
        R = radius / 100 * RADIUS_MAX * u
        if R <= 0:
            return X.copy()
        num, den = norm_conv(X, w, gauss_kernel(R / 2))
    elif kind == "lens":
        R = radius / 100 * RADIUS_MAX * u
        if R <= 0:
            return X.copy()
        num, den = norm_conv(X, w, aperture_kernel(R, blades, rotation))
    elif kind == "motion":
        L = length / 100 * MOTION_MAX * u
        if L <= 0:
            return X.copy()
        num, den = norm_conv(X, w, line_kernel(L, angle))
    elif kind == "zoom":
        a = amount / 100 * ZOOM_MAX
        if a <= 0:
            return X.copy()
        P = np.concatenate([X * w[..., None], w[..., None]], -1)
        S_ = radial_gather(P, (centre[0] * W - 0.5, centre[1] * H - 0.5), a)
        num, den = S_[..., :3], S_[..., 3]
    else:
        raise ValueError(kind)
    return blend(X, m, num, den)


# ------------------------------------------------------------------------------------------------------------------ checks
def _scene(rs, H=96, W=128):
    X = rs.uniform(0.02, 2.0, (H, W, 3))
    X[40:46, 60:66] = 300.0
    return X


def checks():
    rs = np.random.default_rng(3)
    X = _scene(rs)
    H, W = X.shape[:2]
    m = np.zeros((H, W)); m[:, 64:] = 1.0
    m = conv(m, gauss_kernel(2.0))
    res = []

    def line(name, val, bound, ok):
        res.append((name, val, bound, ok))
        print(f"{name:58s} {val:12.3e}  (bound {bound:.1e})  {'ok' if ok else 'FAIL'}")

    for kind, kw in (("soft", {"radius": 100}), ("lens", {"radius": 100, "blades": 6, "rotation": 10.0, "highlights": 0.5}), ("motion", {"angle": 0.0, "length": 100}), ("zoom", {"amount": 100, "centre": (0.0, 0.5)})):
        # identity at 0
        zero = {"soft": "radius", "lens": "radius", "motion": "length", "zoom": "amount"}[kind]
        o = blur_node(X, m, kind, **{**kw, zero: 0})
        line(f"{kind}: identity at amount 0 (exact)", float(np.abs(o - X).max()), 0.0, np.array_equal(o, X))
        # mask 0 everywhere gives the input's bits; no NaN
        o = blur_node(X, np.zeros_like(m), kind, **kw)
        line(f"{kind}: mask 0 everywhere gives the input (exact)", float(np.abs(o - X).max()), 0.0, np.array_equal(o, X))
        o = blur_node(np.zeros_like(X), m, kind, **kw)
        line(f"{kind}: black picture, no NaN, stays black", float(np.abs(o).max()) + (0 if np.isfinite(o).all() else 1), 1e-12, bool(np.isfinite(o).all() and np.abs(o).max() < 1e-12))
        # flat field: unchanged
        F = np.full_like(X, 0.37)
        o = blur_node(F, m, kind, **kw)
        line(f"{kind}: flat field unchanged (energy)", float(np.abs(o - F).max()), 1e-9, np.abs(o - F).max() < 1e-9)
        # (b) nothing from outside the mask: change X where m == 0 (far from the edge) wildly; the result where m == 1 must not move
        Y = X.copy(); Y[:, 56:64] = 1e4 * rs.uniform(0.5, 1.5, (H, 8, 3))
        o1 = blur_node(X, (m > 0.5).astype(float), kind, **kw)
        o2 = blur_node(Y, (m > 0.5).astype(float), kind, **kw)
        inside = (m > 0.5)
        d = float(np.abs(o1 - o2)[inside].max())
        line(f"{kind}: respect on, nothing from outside the mask reaches inside", d, 1e-9, d < 1e-9)
        o1 = blur_node(X, (m > 0.5).astype(float), kind, respect=False, **kw)
        o2 = blur_node(Y, (m > 0.5).astype(float), kind, respect=False, **kw)
        d2 = float(np.abs(o1 - o2)[inside].max())
        line(f"{kind}: respect off, the outside does bleed in (so the test bites)", d2, 1e-3, d2 > 1e-3)
        # determinism
        a = blur_node(X, m, kind, **kw); b = blur_node(X, m, kind, **kw)
        line(f"{kind}: repeat render identical", float(np.abs(a - b).max()), 0.0, np.array_equal(a, b))
    # symmetry: a mirrored picture and mask give the mirrored result (soft, lens round, motion at 0 degrees)
    for kind, kw in (("soft", {}), ("lens", {}), ("motion", {"angle": 0.0})):
        a = blur_node(X, m, kind, **kw)
        b = blur_node(X[:, ::-1], m[:, ::-1], kind, **kw)[:, ::-1]
        line(f"{kind}: mirror symmetry", float(np.abs(a - b).max()), 1e-9, np.abs(a - b).max() < 1e-9)
    # energy of a point in the interior: the sum of the blurred picture equals the sum of the picture (no mask, away from the edge)
    yy, xx = np.mgrid[0:96, 0:128]
    P = np.repeat((1000.0 * np.exp(-((yy - 48) ** 2 + (xx - 64) ** 2) / (2 * 3.0 ** 2)))[..., None], 3, -1)
    for kind, kw in (("soft", {}), ("lens", {"blades": 0}), ("motion", {"angle": 45.0}), ("zoom", {})):
        o = blur_node(P, np.ones((96, 128)), kind, **kw)
        line(f"{kind}: energy of a point (sum out / sum in - 1)", float(abs(o.sum() / P.sum() - 1)), 2e-2 if kind == "zoom" else 1e-9, abs(o.sum() / P.sum() - 1) < (2e-2 if kind == "zoom" else 1e-9))
    # tile invariance: the node on a window with its halo equals the whole frame (soft, radius 20 -> R about 13 px at 128 wide)
    X2 = rs.uniform(0.02, 2.0, (96, 128, 3)); m2 = np.ones((96, 128)); m2[:, :60] = 0.2
    whole = blur_node(X2, m2, "soft", radius=20)
    Rpx = 20 / 100 * RADIUS_MAX * 128
    Hh = int(math.ceil(SIGMA_CUT * Rpx / 2)) + 1
    x0, x1 = 30, 80
    w0, w1 = max(0, x0 - Hh), min(128, x1 + Hh)
    part = blur_node(X2[:, w0:w1], m2[:, w0:w1], "soft", radius=20, u=128)
    # the window is a frame of its own: its edges renormalise, so only the pixels at least a halo from the window's cut edges agree
    d = float(np.abs(part[:, x0 - w0:x1 - w0] - whole[:, x0:x1]).max())
    line("soft: a tile with a halo of ceil(4 sigma) + 1 equals the frame", d, 1e-9, d < 1e-9)
    return all(r[3] for r in res)


if __name__ == "__main__":
    ok = checks()
    print("all ok" if ok else "FAILED")

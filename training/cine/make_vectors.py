"""Writes crates/ym-color/tests/fixtures/cine_vectors.json: test vectors of the cine rendering (docs/02, C12).

Every number comes from evaluating the model in ``cine_ref.py`` (numpy, float64, the constants of
``constants.json``) on inputs made here. Nothing in this script reads an ARRI file, a LUT or any measured
data; the only external numbers are the 24 ColorChecker patches (Ohta's spectra under D65, a published
reflectance set) and the chromaticities of the colour spaces. Inputs are deterministic (a tiny PRNG of
our own, so numpy's version cannot change them).

    python3 training/cine/make_vectors.py [out.json]

Cases (about 2,000): neutral ramps from -10 to +12 stops, skin, the ColorChecker, AWG4 primaries and
secondaries, a saturated hue wheel (inside and beyond the spectral locus), exposure ladders, colours with
negative channels, zeros, tiny and huge values, seeded random colours. For each: the scene triple, the
display-linear Rec.709, Display P3 and Rec.2020 triples, the Rec.709 inverse (scene triple, whether the
display colour was reachable) and the forward of that inverse. A second list holds display colours for the
inverse alone (forward outputs, 8-bit colours, channels at exactly 0 and 1) for the three primaries, with
the scene triple the reference inverse gives, the chroma fraction it kept (1 when reachable) and the
display colour that scene triple renders to. ``ray_is_interval`` is false for the few display colours whose chroma ray has gaps in the reachable set; their
answer depends on the solver, so tests skip the chroma fraction there.
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cine_ref as R

REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DEFAULT_OUT = os.path.join(REPO, "crates", "ym-color", "tests", "fixtures", "cine_vectors.json")

# ColorChecker Classic, XYZ under D65 (Y of the perfect reflector = 1), from Ohta's spectral data.
COLORCHECKER = [
    ("dark skin", [0.109712, 0.097029, 0.060551]), ("light skin", [0.381343, 0.355837, 0.259375]),
    ("blue sky", [0.178577, 0.190802, 0.345406]), ("foliage", [0.101083, 0.129846, 0.066929]),
    ("blue flower", [0.25832, 0.243814, 0.453302]), ("bluish green", [0.312785, 0.427294, 0.447074]),
    ("orange", [0.364662, 0.293271, 0.059073]), ("purplish blue", [0.134167, 0.117573, 0.372345]),
    ("moderate red", [0.284599, 0.192275, 0.137525]), ("purple", [0.086816, 0.065231, 0.146935]),
    ("yellow green", [0.331993, 0.436595, 0.111929]), ("orange yellow", [0.461862, 0.431294, 0.084244]),
    ("blue", [0.084111, 0.062301, 0.299993]), ("green", [0.145013, 0.2357, 0.095197]),
    ("red", [0.201762, 0.118256, 0.051997]), ("yellow", [0.560491, 0.59638, 0.095535]),
    ("magenta", [0.294182, 0.192689, 0.30287]), ("cyan", [0.144756, 0.198667, 0.395281]),
    ("white 9.5", [0.841388, 0.887235, 0.954256]), ("neutral 8", [0.555484, 0.583852, 0.634131]),
    ("neutral 6.5", [0.340557, 0.358172, 0.39054]), ("neutral 5", [0.193107, 0.203054, 0.221558]),
    ("neutral 3.5", [0.08778, 0.092589, 0.102405]), ("black 2", [0.031867, 0.033549, 0.038162]),
]
# Skin: CIELAB (D65, white Y = 1) of light to dark skin, a* 8 to 20, b* 12 to 28 (hue 40 to 60 degrees).
SKIN_LAB = [(82, 8, 14), (76, 11, 17), (70, 13, 20), (64, 15, 22), (58, 17, 24), (52, 18, 25),
            (46, 19, 25), (40, 18, 23), (34, 16, 20), (66, 20, 28), (60, 9, 12), (50, 12, 20)]
GREY = 0.18
XYZ_TO_AWG4 = np.linalg.inv(R.AWG4_TO_XYZ)
U0, V0 = 0.19783, 0.46832     # u'v' of D65


class Rng:
    """splitmix64: the same stream on every platform and numpy version."""

    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFFFFFFFFFF

    def next(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)

    def uniform(self, lo=0.0, hi=1.0):
        return lo + (hi - lo) * ((self.next() >> 11) / float(1 << 53))


def lab_to_xyz(L, a, b):
    fy = (L + 16.0) / 116.0
    fx, fz = fy + a / 500.0, fy - b / 200.0
    f = lambda t: t ** 3 if t > 6.0 / 29.0 else 3.0 * (6.0 / 29.0) ** 2 * (t - 4.0 / 29.0)
    return np.array([R.XN * f(fx), f(fy), R.ZN * f(fz)])


def uv_to_xyz(u, v, y):
    d = 6.0 * u - 16.0 * v + 12.0
    x, yy = 9.0 * u / d, 4.0 * v / d
    return np.array([x / yy * y, y, (1.0 - x - yy) / yy * y])


def awg4(xyz):
    return XYZ_TO_AWG4 @ xyz


def build_cases():
    cases = []

    def add(name, x):
        cases.append((name, np.asarray(x, dtype=np.float64)))

    # neutral ramps: -10 to +12 stops in quarter stops, and the top of the range in 1/20 stops
    for s in np.arange(-10.0, 12.0001, 0.25):
        add(f"neutral/{s:+.2f}", [GREY * 2.0 ** s] * 3)
    for s in np.arange(8.0, 12.0001, 0.05):
        add(f"neutral-top/{s:+.2f}", [GREY * 2.0 ** s] * 3)
    # skin at five exposures
    for i, lab in enumerate(SKIN_LAB):
        for e in (-2, -1, 0, 1, 2):
            add(f"skin/{i:02d}/{e:+d}", awg4(lab_to_xyz(*lab)) * 2.0 ** e)
    # the ColorChecker at five exposures
    for name, xyz in COLORCHECKER:
        for e in (-3, -1, 0, 1, 3):
            add(f"colorchecker/{name}/{e:+d}", awg4(np.array(xyz)) * 2.0 ** e)
    # AWG4 primaries and secondaries
    prim = {"red": [1, 0, 0], "green": [0, 1, 0], "blue": [0, 0, 1], "yellow": [1, 1, 0], "cyan": [0, 1, 1], "magenta": [1, 0, 1]}
    for name, v in prim.items():
        for e in (-4, -2, 0, 2, 4):
            add(f"primary/{name}/{e:+d}", np.array(v, float) * GREY * 2.0 ** e)
    # hue wheel in u'v' around D65: 36 hues, four radii, three luminances (beyond the locus at the larger radii)
    for r in (0.03, 0.06, 0.10, 0.14):
        for h in range(0, 360, 10):
            u, v = U0 + r * math.cos(math.radians(h)), V0 + r * math.sin(math.radians(h))
            for e in (-3, 0, 3):
                add(f"wheel/r{r:.2f}/h{h:03d}/{e:+d}", awg4(uv_to_xyz(u, v, GREY * 2.0 ** e)))
    # exposure ladders of twenty saturated colours, -8 to +12 stops in whole stops
    for k in range(20):
        h = k * 18
        u, v = U0 + 0.10 * math.cos(math.radians(h)), V0 + 0.10 * math.sin(math.radians(h))
        base = awg4(uv_to_xyz(u, v, GREY))
        for e in range(-8, 13):
            add(f"ladder/h{h:03d}/{e:+d}", base * 2.0 ** e)
    # colours with negative channels (seeded)
    rng = Rng(0x59454C4C4F574D4F)
    for i in range(240):
        v = np.array([rng.uniform(-0.25, 1.0) for _ in range(3)])
        add(f"negative/{i:03d}", v * GREY * 2.0 ** rng.uniform(-3.0, 3.0))
    for i, v in enumerate([[1, 0, -0.1], [0, 1, -0.1], [-0.1, 1, 1], [-0.05, 0.5, 0.2], [0.3, -0.02, 0.01], [-1, -1, -1], [-1e-6, 0.18, 0.18], [0.18, 0.18, -1e-6]]):
        add(f"negative/edge{i}", np.array(v, float) * GREY)
    # zeros, tiny and huge values
    for name, v in [("zero", [0, 0, 0]), ("red-only", [0.18, 0, 0]), ("green-only", [0, 0.18, 0]), ("blue-only", [0, 0, 0.18]),
                    ("no-blue", [0.18, 0.18, 0]), ("tiny", [1e-12, 1e-12, 1e-12]), ("tiny2", [1e-30, 1e-30, 1e-30]),
                    ("tiny-colour", [3e-9, 1e-9, 2e-9]), ("below-floor", [2e-10, 5e-10, 1e-10]),
                    ("1e3", [1e3] * 3), ("1e4", [1e4] * 3), ("1e5", [1e5] * 3), ("1e6", [1e6] * 3), ("1e12", [1e12] * 3),
                    ("huge-colour", [1e5, 1e4, 1e3]), ("huge-red", [1e6, 0, 0]), ("huge-mixed", [1e5, -1e5, 0]),
                    ("huge-negative", [-1e12, 5, 0]), ("huge-negative-neutral", [-1e5] * 3)]:
        add(f"edge/{name}", v)
    # seeded random colours: log-uniform luminance -8 to +9 stops, chromaticity inside radius 0.12 of D65 in u'v'
    for i in range(420):
        y = GREY * 2.0 ** rng.uniform(-8.0, 9.0)
        r, h = 0.12 * math.sqrt(rng.uniform()), rng.uniform(0.0, 2.0 * math.pi)
        add(f"random/{i:03d}", awg4(uv_to_xyz(U0 + r * math.cos(h), V0 + r * math.sin(h), y)))
    return cases


def srgb_decode(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def build_inverse_inputs(cases, C):
    """Display colours for the inverse alone: (primaries, display-linear triple, note)."""
    out = []
    rng = Rng(0x1234ABCD5678EF90)
    pick = [cases[(i * 7919) % len(cases)][1] for i in range(60)]
    for p in ("rec709", "p3", "rec2020"):
        for i, x in enumerate(pick):
            out.append((p, R.forward(x, C, p), f"forward/{i:02d}"))
        for i in range(60 if p == "rec709" else 40):
            codes = np.array([int(rng.uniform(0, 256)) for _ in range(3)], dtype=np.float64)
            out.append((p, srgb_decode(codes / 255.0), f"code8/{int(codes[0])},{int(codes[1])},{int(codes[2])}"))
        for i, v in enumerate([[0, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 0.5, 0],
                               [0.5, 0, 0.5], [0, 0.5, 1], [0.9999, 0.9999, 0.9999], [0.999999, 0.999999, 0.999999], [1e-9, 1e-9, 1e-9],
                               [0.18, 0.18, 0.18], [1e-6, 0.5, 1e-6], [0.8, 0.9, 1.0], [1.0, 0.95, 0.3]]):
            out.append((p, np.array(v, dtype=np.float64), f"edge/{i:02d}"))
    return out


def ray_is_interval(d, C, primaries, steps=100):
    """True when the colours that have a pre-image along the chroma ray of display colour d form an interval [0, lambda_max]
    (checked on a grid of steps + 1 points). The nearest-reachable rule is only unambiguous for such rays."""
    xyz2 = R.gamut_inverse(np.asarray(d, dtype=np.float64)[None, :], C, primaries)[0]
    y2 = max(xyz2[1], 1e-12)
    ca2 = np.cbrt(xyz2[0] / (R.XN * y2)) - 1.0
    cb2 = 1.0 - np.cbrt(xyz2[2] / (R.ZN * y2))
    lam = np.linspace(0.0, 1.0, steps + 1)
    _, ok, _ = R.solve_scene(np.stack([np.full(steps + 1, y2), lam * ca2, lam * cb2], 1), C)
    first_bad = np.argmin(ok) if not ok.all() else steps + 1
    return bool(not ok[first_bad:].any())


def fmt(a):
    return "[" + ", ".join(f"{float(v):.17g}" for v in np.asarray(a).reshape(-1)) + "]"


def main(path):
    C = R.load_constants()
    cases = build_cases()
    X = np.stack([c[1] for c in cases])
    out = {p: R.forward(X, C, p) for p in ("rec709", "p3", "rec2020")}
    inv, info = R.inverse(out["rec709"], C, "rec709", return_info=True)
    back = R.forward(inv, C, "rec709")
    reach = info["lam"] >= 1.0
    lines = []
    for i, (name, x) in enumerate(cases):
        lines.append(
            f'    {{"name": "{name}", "x": {fmt(x)}, "rec709": {fmt(out["rec709"][i])}, "p3": {fmt(out["p3"][i])}, '
            f'"rec2020": {fmt(out["rec2020"][i])}, "inverse709": {fmt(inv[i])}, "roundtrip709": {fmt(back[i])}, '
            f'"reachable709": {"true" if reach[i] else "false"}}}'
        )
    items = build_inverse_inputs(cases, C)
    ilines = []
    nreach = 0
    worst = 0.0
    for p in ("rec709", "p3", "rec2020"):
        sel = [it for it in items if it[0] == p]
        d = np.stack([it[1] for it in sel])
        xi, inf = R.inverse(d, C, p, return_info=True)
        bk = R.forward(xi, C, p)
        for k, it in enumerate(sel):
            lam = float(inf["lam"][k])
            nreach += lam >= 1.0
            if lam >= 1.0:
                worst = max(worst, float(np.abs(bk[k] - np.clip(d[k], R.D_MIN, 1.0)).max()))
            ilines.append(
                f'    {{"primaries": "{p}", "note": "{it[2]}", "d": {fmt(d[k])}, "x": {fmt(xi[k])}, '
                f'"chroma_fraction": {lam:.17g}, "ray_is_interval": {"true" if ray_is_interval(d[k], C, p) else "false"}, "roundtrip": {fmt(bk[k])}}}'
            )
    doc = ['{']
    doc.append('  "about": "Test vectors of the cine rendering (docs/02-colour-pipeline.md, C12), made by training/cine/make_vectors.py from the model alone. x: scene-linear AWG4. rec709, p3, rec2020: display-linear RGB of that primaries (1.0 = SDR white). inverse709: the reference inverse of rec709 (scene AWG4); roundtrip709: the forward of inverse709; reachable709: the Rec.709 colour has a pre-image (false: the nearest reachable colour was used). A scene triple of the inverse is one pre-image among possibly several: compare display colours.",')
    doc.append(f'  "scene_limit": {R.SCENE_LIMIT:.17g}, "d_min": {R.D_MIN:.17g}, "residual_tol": {R.RESIDUAL_TOL:.17g},')
    doc.append('  "constants": {' + ", ".join(f'"{k}": {v:.17g}' for k, v in C.items()) + '},')
    doc.append('  "cases": [')
    doc.append(",\n".join(lines))
    doc.append('  ],')
    doc.append('  "inverse": [')
    doc.append(",\n".join(ilines))
    doc.append('  ]')
    doc.append('}')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(doc) + "\n")
    print(f"wrote {path}: {len(cases)} cases, {len(items)} inverse inputs ({nreach} reachable)")
    worst_cases = float(np.abs(back[reach] - np.clip(out["rec709"][reach], R.D_MIN, 1.0)).max())
    print(
        f"forward range: min {min(o.min() for o in out.values()):.3e} max {max(o.max() for o in out.values()):.9f}; "
        f"inverse round trip of the reachable cases: worst |display error| {worst_cases:.2e} (cases), {worst:.2e} (inverse list)"
    )
    print(f"cases whose Rec.709 colour is not reachable: {int((~reach).sum())}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT)

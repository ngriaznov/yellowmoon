#!/usr/bin/env python3
"""Writes the pictures of design/reference-grade/ into html/assets/ (task RG-07). Not product code.

EVERYTHING HERE IS SYNTHETIC: a drawn lakeside village (sky, foliage, water, ground, buildings, and a flat drawing of a figure with a skin-toned face), made in
linear light, and a drawn 'reference' of different content (a harbour at dusk) with a clear teal-and-amber grade. No film still, no photograph, no third-party
image, nobody's likeness. The 'after' pictures are the maths of docs/34 applied by rg_ref.py with ILLUSTRATIVE values: no fit was run.
Run:  python3 -I rg_demo.py   (numpy and PIL)
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rg_ref as R  # noqa: E402

A = os.path.join(HERE, "assets")
W, H = 1500, 1000
SS = 2
LABEL = {"unlabelled": 0, "sky": 1, "vegetation": 2, "water": 3, "ground": 4, "built": 5, "skin": 6, "clothing": 7}


def noise(scale, seed, shape=(H, W)):
    r = np.random.default_rng(seed)
    n = r.standard_normal((shape[0] // scale + 2, shape[1] // scale + 2))
    img = Image.fromarray(((n - n.min()) / (n.max() - n.min()) * 255).astype(np.uint8)).resize((shape[1], shape[0]), Image.BICUBIC)
    return np.asarray(img, np.float64) / 255.0


class Canvas:
    """Linear-light picture plus a label map; layers are drawn as supersampled masks."""

    def __init__(self):
        self.X = np.zeros((H, W, 3))
        self.L = np.zeros((H, W), np.int8)

    def layer(self, fn, colour, label=None, blur=0.0, texture=None, opacity=1.0):
        m = Image.new("L", (W * SS, H * SS), 0)
        fn(ImageDraw.Draw(m), SS)
        m = m.resize((W, H), Image.LANCZOS)
        if blur:
            m = m.filter(ImageFilter.GaussianBlur(blur))
        a = np.asarray(m, np.float64) / 255.0 * opacity
        c = np.asarray(colour, np.float64)
        col = c[None, None, :] if c.ndim == 1 else c
        if texture is not None:
            col = col * texture[..., None]
        self.X = self.X * (1 - a[..., None]) + a[..., None] * col
        if label is not None:
            self.L[a > 0.5] = LABEL[label]
        return a


def poly(pts):
    return lambda g, s: g.polygon([(x * s, y * s) for x, y in pts], fill=255)


def ell(x0, y0, x1, y1):
    return lambda g, s: g.ellipse([x0 * s, y0 * s, x1 * s, y1 * s], fill=255)


def rect(x0, y0, x1, y1):
    return lambda g, s: g.rectangle([x0 * s, y0 * s, x1 * s, y1 * s], fill=255)


def many(*fns):
    def f(g, s):
        for fn in fns:
            fn(g, s)
    return f


def render_scene():
    c = Canvas()
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    # sky: blue at the top, pale toward the horizon, with soft clouds
    t = np.clip(yy / 420, 0, 1)[..., None]
    sky = (1 - t) * np.array([0.085, 0.20, 0.50]) + t * np.array([0.50, 0.60, 0.74])
    cl = np.clip(noise(90, 7) * 1.7 - 0.62, 0, 1) * np.clip(1 - yy / 400, 0, 1) * 1.1
    sky = sky * (1 - cl[..., None] * 0.8) + cl[..., None] * 0.8 * np.array([0.80, 0.78, 0.76])
    c.X[:] = sky
    c.L[:] = LABEL["sky"]
    # distant hills (ground: mountain), hazy
    hz = 400 + 22 * np.sin(np.arange(W) / 140.0) + 14 * np.sin(np.arange(W) / 47.0 + 1)
    pts = [(0, 440)] + [(x, hz[x]) for x in range(0, W, 20)] + [(W, 440)]
    c.layer(poly(pts + [(W, 470), (0, 470)]), [0.12, 0.17, 0.22], "ground", blur=1.0)
    # the far meadow (foliage) from the hills down, and a ploughed field (ground) on the left
    c.layer(poly([(0, 450), (W, 450), (W, 700), (0, 760)]), [0.085, 0.17, 0.04], "vegetation", blur=0.8, texture=0.8 + 0.4 * noise(13, 23))
    c.layer(poly([(0, 640), (520, 600), (800, 640), (620, 1000), (0, 1000)]), [0.22, 0.14, 0.075], "ground", blur=0.8, texture=0.85 + 0.3 * noise(6, 33) * (0.6 + 0.4 * np.sin(yy / 9.0) ** 2))
    # a small pond far to the left: under 2 % of the picture, so the Elements tab says "Not found in this photo." for Water
    ripple = 0.9 + 0.1 * np.sin(yy / 4.0)
    c.layer(ell(20, 520, 250, 572), np.array([0.22, 0.30, 0.42]) * ripple[..., None], "water", blur=0.8)
    # buildings on the right: houses with walls, roofs, windows
    houses = [(1010, 340, 1180, 520, [0.62, 0.55, 0.42], [0.30, 0.09, 0.05]), (1160, 300, 1330, 520, [0.50, 0.30, 0.14], [0.26, 0.08, 0.04]), (1310, 360, 1500, 540, [0.58, 0.52, 0.40], [0.24, 0.07, 0.04]),
              (880, 420, 1010, 540, [0.45, 0.38, 0.28], [0.22, 0.07, 0.04])]
    for x0, y0, x1, y1, wall, roof in houses:
        c.layer(rect(x0, y0 + 40, x1, y1), wall, "built", blur=0.6, texture=0.9 + 0.1 * noise(9, int(x0)))
        c.layer(poly([(x0 - 14, y0 + 44), ((x0 + x1) / 2, y0 - 8), (x1 + 14, y0 + 44)]), roof, "built", blur=0.6)
        for k in range(3):
            wx = x0 + 22 + k * ((x1 - x0 - 44) / 3)
            c.layer(rect(wx, y0 + 80, wx + 22, y0 + 120), [0.05, 0.07, 0.10], "built", blur=0.4)
            c.layer(rect(wx - 3, y0 + 77, wx + 25, y0 + 80), [0.7, 0.66, 0.6], "built")
    # a church-like tower
    c.layer(rect(1255, 215, 1295, 330), [0.60, 0.54, 0.44], "built")
    c.layer(poly([(1248, 218), (1275, 150), (1302, 218)]), [0.20, 0.07, 0.04], "built")
    # trees: left crowns and a hedge in the middle ground (foliage), noisy
    foliage_tex = 0.7 + 0.6 * noise(7, 11)
    for cx, cy, r in ((120, 330, 150), (330, 300, 110), (520, 360, 80), (60, 120, 130), (760, 410, 55), (880, 395, 45)):
        c.layer(ell(cx - r, cy - r * 0.9, cx + r, cy + r * 0.9), [0.040, 0.105, 0.028], "vegetation", blur=1.0, texture=foliage_tex)
    for cx in (120, 330):
        c.layer(rect(cx - 9, 400, cx + 9, 560), [0.10, 0.06, 0.03], "ground")
    # meadow across the foreground right and left of the path
    c.layer(poly([(620, 1000), (760, 760), (1500, 560), (1500, 1000)]), [0.11, 0.22, 0.045], "vegetation", blur=0.6, texture=0.8 + 0.4 * noise(11, 21))
    # the path: sand and earth, a wedge from the bottom up toward the houses
    c.layer(poly([(430, 1000), (760, 720), (900, 700), (1160, 560), (1260, 560), (1130, 700), (1000, 1000)]), [0.38, 0.30, 0.19], "ground", blur=0.8, texture=0.9 + 0.2 * noise(5, 31))
    # a low stone wall along the field
    c.layer(rect(0, 618, 560, 640), [0.30, 0.28, 0.25], "built")
    return c


def render_figure(c):
    """A flat drawing: dark hair, a face, neck and hands in skin, a jacket and dark trousers. Nobody's likeness."""
    S_ = 1.5
    P = lambda pts: [(790 + (x - 775) * S_, 1040 + (y - 960) * S_) for x, y in pts]
    def E(x0, y0, x1, y1):
        (a, b), (c_, d) = P([(x0, y0), (x1, y1)])
        return ell(a, b, c_, d)
    def Rc(x0, y0, x1, y1):
        (a, b), (c_, d) = P([(x0, y0), (x1, y1)])
        return rect(a, b, c_, d)
    skin = [0.36, 0.20, 0.14]
    skin2 = [0.33, 0.18, 0.12]
    c.layer(poly(P([(640, 1000), (655, 760), (700, 640), (790, 620), (850, 650), (880, 760), (890, 1000)])), [0.045, 0.05, 0.07], "clothing")
    c.layer(poly(P([(615, 780), (640, 585), (720, 548), (800, 548), (880, 585), (905, 790), (840, 800), (800, 700), (700, 700), (680, 800)])), [0.38, 0.33, 0.24], "clothing")
    c.layer(Rc(738, 495, 792, 570), skin2, "skin")
    c.layer(many(E(695, 345, 835, 455)), [0.018, 0.012, 0.008], "clothing")
    c.layer(E(705, 372, 825, 532), skin, "skin")
    c.layer(many(E(735, 440, 748, 450), E(782, 440, 795, 450), Rc(748, 497, 782, 502)), [0.02, 0.012, 0.01])
    c.layer(poly(P([(618, 780), (640, 790), (655, 870), (630, 880)])), skin2, "skin")
    c.layer(poly(P([(880, 790), (905, 780), (915, 870), (890, 880)])), skin2, "skin")
    c.layer(poly(P([(715, 560), (735, 560), (750, 700), (725, 700)])), [0.20, 0.08, 0.05], "clothing")
    return c


def draw_reference():
    """The drawn reference, display-referred and graded teal and amber by hand: a harbour at dusk. Different content from the scene on purpose."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    img = np.zeros((H, W, 3))
    t = np.clip(yy / 520, 0, 1)[..., None]
    top, mid, hor = np.array([0.05, 0.19, 0.27]), np.array([0.20, 0.38, 0.42]), np.array([0.95, 0.60, 0.25])
    sky = np.where(t < 0.55, top + (mid - top) * (t / 0.55), mid + (hor - mid) * (np.clip(t - 0.55, 0, None) / 0.45) ** 1.5)
    img[:] = sky
    cl = np.clip(noise(100, 41) * 1.6 - 0.62, 0, 1) * np.clip(1 - np.abs(yy - 300) / 260, 0, 1)
    img = img * (1 - cl[..., None] * 0.55) + cl[..., None] * 0.55 * np.array([0.90, 0.50, 0.25]) * (0.5 + t * 0.6)
    # sea
    sea_t = np.clip((yy - 520) / 230, 0, 1)[..., None] ** 0.8
    sea = (1 - sea_t) * np.array([0.70, 0.45, 0.22]) + sea_t * np.array([0.02, 0.15, 0.19])
    streak = np.exp(-((xx - 560) / (60 + np.maximum(yy - 520, 0) * 0.5)) ** 2)[..., None] * np.clip(1 - (yy - 520) / 400, 0, 1)[..., None] * np.array([0.35, 0.2, 0.05])
    rip = 0.9 + 0.1 * np.sin(yy / 4.0 + 8 * noise(50, 51))
    sea = (sea + streak) * rip[..., None]
    pil = Image.new("RGB", (W * SS, H * SS))
    out = img
    horizon = 520
    mask = (yy >= horizon)[..., None]
    out = np.where(mask, sea, out)

    def over(fn, col, blur=0.5):
        nonlocal out
        m = Image.new("L", (W * SS, H * SS), 0)
        fn(ImageDraw.Draw(m), SS)
        m = m.resize((W, H), Image.LANCZOS)
        if blur:
            m = m.filter(ImageFilter.GaussianBlur(blur))
        a = np.asarray(m, np.float64)[..., None] / 255.0
        out = out * (1 - a) + a * np.asarray(col)
    # a headland and a town in silhouette, with amber windows
    pts = [(0, 520)] + [(x, 470 - 30 * np.sin(x / 210.0) - 10 * np.sin(x / 33.0)) for x in range(0, 700, 20)] + [(700, 520)]
    over(poly(pts), [0.03, 0.09, 0.11])
    for i, x0 in enumerate(range(760, 1500, 85)):
        h = 70 + (i * 37) % 90
        over(rect(x0, 520 - h, x0 + 70, 524), [0.04, 0.07, 0.09])
        over(poly([(x0 - 4, 522 - h), (x0 + 35, 490 - h), (x0 + 74, 522 - h)]), [0.05, 0.06, 0.08])
        for k in range(3):
            for j in range(h // 38):
                over(rect(x0 + 12 + k * 20, 520 - h + 14 + j * 34, x0 + 22 + k * 20, 520 - h + 30 + j * 34), [1.0, 0.72, 0.30], blur=0.8)
    # pier with lamps and their reflections, a boat
    over(rect(120, 600, 900, 616), [0.03, 0.06, 0.07])
    for x0 in range(160, 900, 120):
        over(rect(x0, 570, x0 + 4, 602), [0.03, 0.06, 0.07])
        over(ell(x0 - 8, 560, x0 + 12, 580), [1.0, 0.82, 0.45], blur=3)
        over(rect(x0 - 2, 620, x0 + 6, 700), [0.85, 0.55, 0.22], blur=4)
    over(poly([(1020, 640), (1180, 640), (1150, 676), (1050, 676)]), [0.02, 0.05, 0.06])
    over(rect(1098, 580, 1102, 640), [0.02, 0.05, 0.06])
    over(poly([(1102, 584), (1150, 632), (1102, 632)]), [0.90, 0.60, 0.25])
    out = np.clip(out, 0, 1)
    grain = np.random.default_rng(9).standard_normal((H, W, 1)) * 0.006
    return np.clip(out + grain, 0, 1)


# ------------------------------------------------------------------------------------------------------------------------ settings (illustrative: no fit was run)
ST = R.STOP


def knots(stops_out):
    """A curve through scene grey: stops given as (input stop, output stop) pairs, returned as (code, code) points with the ends 0 and 1."""
    pts = [(0.0, 0.0)] + [(float(R.encode(0.18 * 2.0**a)), float(R.encode(0.18 * 2.0**b))) for a, b in stops_out] + [(1.0, 1.0)]
    return pts


SETTINGS = {
    "luma": knots([(-6, -6.9), (-4, -4.8), (-2, -2.5), (-1, -1.2), (0, -0.05), (1, 1.2), (2, 2.3), (4, 4.1)]),   # a lower toe and floor, a lifted shoulder
    "red": knots([(-3, -3.75), (0, 0.12), (3, 3.75)]),     # cooler shadows (less red), warmer highlights (more red)
    "green": knots([(-3, -3.0), (0, 0.0), (3, 3.05)]),
    "blue": knots([(-3, -2.3), (0, -0.12), (3, 2.2)]),     # deeper, bluer shadows; amber highlights
    "sat": -4.0,
    "hue_vs_hue": [(0, 10.0), (60, -8.0), (120, 22.0), (180, 10.0), (240, -14.0), (300, 0.0)],
    "hue_vs_sat": [(0, 1.15), (60, 1.25), (120, 0.80), (180, 1.30), (240, 1.20), (300, 0.80)],
    "elements": {
        "sky": {"exposure_ev": -0.45, "hue_deg": -14.0, "chroma": 1.25},
        "vegetation": {"exposure_ev": -0.25, "hue_deg": 20.0, "chroma": 0.75},
        "water": {"exposure_ev": 0.0, "hue_deg": 0.0, "chroma": 1.0},
        "ground": {"exposure_ev": 0.10, "hue_deg": -6.0, "chroma": 1.30},
        "built": {"exposure_ev": 0.0, "hue_deg": -8.0, "chroma": 1.25},
        "skin": {"exposure_ev": 0.0, "hue_deg": -3.0, "chroma": 0.95},
        "subject": {"exposure_ev": 0.25},
    },
}


def save(name, rgb, q=88, size=None):
    im = Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8))
    if size:
        im = im.resize(size, Image.LANCZOS)
    im.save(os.path.join(A, name), quality=q)
    print("wrote", name)


def main():
    os.makedirs(A, exist_ok=True)
    c = render_figure(render_scene())
    lin = c.X
    disp0 = R.display(lin)
    save("scene.jpg", disp0)
    guide = R.lstar(disp0) / 100.0
    lab = c.L
    person = (lab == LABEL["skin"]) | (lab == LABEL["clothing"])
    bins = {"sky": lab == 1, "vegetation": lab == 2, "water": lab == 3, "ground": lab == 4, "built": lab == 5, "skin": lab == 6, "subject": person}
    # clothing has a label too, but the hair is dark: the subject is the person (skin and clothing), as section 4b.3 says
    masks = {e: R.soft_mask(b, guide, H) for e, b in bins.items()}
    shares = {e: float(bins[e].mean()) for e in bins}
    print({e: round(s, 3) for e, s in shares.items()})
    for e, m in masks.items():
        save(f"mask-{e}.jpg", m, q=90, size=(750, 500))
    v0 = R.encode(lin)

    def after(strength, st=SETTINGS):
        s = R.effective(st, strength / 100.0)
        v = R.grade(v0, s, masks)
        return R.display(R.decode(v))

    full = after(100)
    save("scene-fitted.jpg", full)
    save("scene-s50.jpg", after(50))
    # the global grade alone, for the notes
    nom = {k: None for k in masks}
    s = R.effective(SETTINGS, 1.0)
    save("scene-global.jpg", R.display(R.decode(R.grade(v0, s, {}))))
    ref = draw_reference()
    save("reference.jpg", ref)
    save("reference-thumb.jpg", ref, q=88, size=(300, 200))
    save("scene-thumb.jpg", disp0, q=88, size=(300, 200))
    # the mask overlay (Show mask on an element): the accent at 0.55 over the fitted picture, as the ordinary Show mask does
    acc = np.array([1.0, 0.435, 0.824])
    for e in ("sky", "vegetation", "skin"):
        o = full + 0.55 * masks[e][..., None] * (acc - full)
        save(f"show-mask-{e}.jpg", o)
    # a picture for the 'look changed' screen: the same fit seen through another look (a warmer, flatter output), to be honest about why the fit is stale
    alt = np.clip((full - 0.5) * 0.92 + 0.5 + np.array([0.035, 0.005, -0.04]), 0, 1)
    save("scene-newlook.jpg", alt)
    # an 'edited by hand' picture: Hue of the vegetation pushed further by the person (+22), the rest unchanged
    ed = json.loads(json.dumps(SETTINGS))
    ed["elements"]["vegetation"]["hue_deg"] = 32.0
    ed["elements"]["vegetation"]["chroma"] = 0.60
    save("scene-edited.jpg", after(100, ed))
    # strip thumbnails: other photos of the same shoot (the scene seen from other places, darker or warmer) and the reference
    def crop(img, box):
        return Image.fromarray((img * 255 + 0.5).astype(np.uint8)).crop(box).resize((300, 200), Image.LANCZOS)
    strip = [
        ("strip-1.jpg", crop(disp0[:, ::-1], (0, 0, 1500, 1000))),
        ("strip-2.jpg", crop(disp0, (300, 150, 1200, 750))),
        ("strip-3.jpg", crop(np.clip(disp0 * np.array([1.05, 0.98, 0.9]) , 0, 1), (0, 300, 900, 900))),
        ("strip-4.jpg", crop(ref, (0, 0, 1500, 1000))),
        ("strip-5.jpg", crop(ref[:, ::-1], (200, 100, 1100, 700))),
        ("strip-6.jpg", crop(np.clip(disp0 * 0.55, 0, 1), (500, 200, 1400, 800))),
    ]
    for n, im in strip:
        im.save(os.path.join(A, n), quality=88)
    open(os.path.join(A, "numbers.json"), "w").write(json.dumps({"share": shares}, indent=1))
    # the number the before/after sheet prints: the change of mean display luma in the skin mask and the subject mask
    d = R.lstar(full) - R.lstar(disp0)
    print("skin dL*", float((d * bins["skin"]).sum() / bins["skin"].sum()))


if __name__ == "__main__":
    main()

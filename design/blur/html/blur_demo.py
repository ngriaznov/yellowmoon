#!/usr/bin/env python3
"""Writes the pictures of design/blur/ into html/assets/ (task BL-01). Not product code.

The scene is SYNTHETIC: a drawn street at dusk with string lights and a drawn figure in a red jacket (no photograph, nobody's likeness, no third-party image),
made in linear light so that the lights are far above white, as a raw file's are. The 'after' pictures are the maths of docs/33 section 3 applied by blur_ref.py
(numpy) through the inverted Subject mask (or a Depth mask for one picture). The display transform is a simple filmic curve and the sRGB encoding; it is not the product's.
Run with the venv's numpy and PIL:  python3 -I blur_demo.py
"""
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import blur_ref as B  # noqa: E402
A = os.path.join(HERE, "assets")
W, H = 1500, 1000
rs = np.random.default_rng(20261118)


def noise(scale, seed):
    r = np.random.default_rng(seed)
    n = r.standard_normal((H // scale + 2, W // scale + 2))
    img = Image.fromarray(((n - n.min()) / (n.max() - n.min()) * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)
    return np.asarray(img, np.float64) / 255.0


def layer_rgb(draw_fn, ss=2, blur=0.0):
    """A layer drawn with PIL at ss times the size, as float RGBA in linear light (values may be above 1: the colour is given as a float triple)."""
    return None


def render_scene():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    X = np.zeros((H, W, 3))
    # sky: deep blue to teal toward the horizon
    t = np.clip(yy / 520, 0, 1)[..., None]
    X += (1 - t) * np.array([0.010, 0.025, 0.080]) + t * np.array([0.050, 0.110, 0.140])
    # foliage band across the top: noisy, dark green, with gaps of sky
    n1, n2, n3 = noise(40, 1), noise(14, 2), noise(5, 3)
    fol = np.clip((n1 * 0.55 + n2 * 0.3 + n3 * 0.25) - 0.52, 0, 1) * 4
    topmask = np.clip((330 - yy + 60 * (n1 - 0.5)) / 60, 0, 1)
    leaf = (0.008 + 0.06 * n3)[..., None] * np.array([0.35, 1.0, 0.25])
    X = X * (1 - (fol * topmask)[..., None].clip(0, 1)) + leaf * (fol * topmask)[..., None].clip(0, 1)
    # building facades left and right with windows
    img = Image.new("RGB", (W * 2, H * 2), (0, 0, 0))
    d = ImageDraw.Draw(img)
    rect_cols = []
    def rect(x0, y0, x1, y1, c):
        d.rectangle([x0 * 2, y0 * 2, x1 * 2, y1 * 2], fill=tuple(int(v) for v in c))
    # we draw layers as 8-bit masks, then scale to the colours (so values above 1 are possible)
    def layer(fn, colour, blur=0.0):
        m = Image.new("L", (W * 2, H * 2), 0)
        fn(ImageDraw.Draw(m))
        m = m.resize((W, H), Image.LANCZOS)
        if blur:
            m = m.filter(ImageFilter.GaussianBlur(blur))
        a = (np.asarray(m, np.float64) / 255.0)[..., None]
        return a, np.array(colour)[None, None, :]
    def comp(X, a, c):
        return X * (1 - a) + a * c
    # facades
    a, c = layer(lambda g: (g.polygon([(0, 150 * 2), (330 * 2, 250 * 2), (330 * 2, 640 * 2), (0, 760 * 2)], fill=255), g.polygon([(1500 * 2, 120 * 2), (1130 * 2, 230 * 2), (1130 * 2, 640 * 2), (1500 * 2, 780 * 2)], fill=255)), [0.012, 0.014, 0.020])
    X = comp(X, a, c)
    # windows (lit, warm, many colours)
    r = np.random.default_rng(5)
    def windows(g):
        for i in range(70):
            side = r.integers(0, 2)
            if side == 0:
                x0 = r.uniform(20, 300); y0 = r.uniform(230, 600)
            else:
                x0 = r.uniform(1160, 1470); y0 = r.uniform(230, 600)
            g.rectangle([x0 * 2, y0 * 2, (x0 + r.uniform(14, 26)) * 2, (y0 + r.uniform(18, 34)) * 2], fill=255)
    a, c = layer(windows, [10.0, 6.0, 2.2])
    X = comp(X, a, c)
    # street (dark wet asphalt) and a lighter pavement
    a, c = layer(lambda g: g.polygon([(0, 760 * 2), (330 * 2, 640 * 2), (1130 * 2, 640 * 2), (1500 * 2, 780 * 2), (1500 * 2, 1000 * 2), (0, 1000 * 2)], fill=255), [0.012, 0.014, 0.018])
    X = comp(X, a, c)
    # wet reflections: vertical streaks
    for k in range(40):
        x0 = r.uniform(0, 1500); y0 = r.uniform(700, 940); L = r.uniform(30, 120)
        col = [[26, 13, 4], [3, 7, 24], [24, 3, 2], [7, 20, 30]][r.integers(0, 4)]
        a, c = layer(lambda g, x0=x0, y0=y0, L=L: g.rectangle([(x0 - 2) * 2, y0 * 2, (x0 + 2) * 2, (y0 + L) * 2], fill=255), col, blur=1.5)
        X = comp(X, a * 0.7, c)
    # string lights across the street, and a few signs: very bright points (scene light, far above white)
    def strings(g):
        for row, (y_a, y_b, n) in enumerate(((180, 250, 26), (300, 360, 22), (430, 480, 18))):
            for i in range(n):
                u = i / (n - 1)
                x = 40 + u * 1420
                y = y_a + (y_b - y_a) * (1 - (2 * u - 1) ** 2) * 0 + 70 * (1 - (2 * u - 1) ** 2) * (1 if row else 0.6)
                rad = 3.5 + 2.2 * row
                g.ellipse([(x - rad) * 2, (y - rad) * 2, (x + rad) * 2, (y + rad) * 2], fill=255)
    for col, shift in (([70, 46, 20], 0), ):
        a, c = layer(strings, col)
        X = comp(X, a, c)
    # a few cool and red lights (traffic, signs)
    def pts(lst):
        def f(g):
            for (x, y, rad) in lst:
                g.ellipse([(x - rad) * 2, (y - rad) * 2, (x + rad) * 2, (y + rad) * 2], fill=255)
        return f
    a, c = layer(pts([(90, 560, 6), (1380, 540, 6), (1255, 600, 5), (180, 650, 5)]), [20, 46, 100])
    X = comp(X, a, c)
    a, c = layer(pts([(1330, 700, 7), (1395, 700, 7), (255, 720, 6), (1060, 668, 5)]), [90, 5, 4])
    X = comp(X, a, c)
    return X


def render_subject():
    """The figure: returns (RGB linear, mask 0..1) at 1500 x 1000. A red jacket with white stripes and a yellow scarf, a face and dark hair, drawn."""
    ss = 2
    def poly_mask(draw_fn):
        m = Image.new("L", (W * ss, H * ss), 0)
        draw_fn(ImageDraw.Draw(m), ss)
        return m.resize((W, H), Image.LANCZOS)
    S_ = ss
    def body(g, s):
        g.polygon([(s * 300, s * 1000), (s * 330, s * 760), (s * 430, s * 620), (s * 620, s * 600), (s * 760, s * 640), (s * 850, s * 780), (s * 880, s * 1000)], fill=255)  # shoulders and torso
        g.rectangle([s * 520, s * 520, s * 600, s * 640], fill=255)  # neck
        g.ellipse([s * 470, s * 270, s * 660, s * 560], fill=255)  # head
        g.ellipse([s * 455, s * 245, s * 675, s * 430], fill=255)  # hair mass
    mask = np.asarray(poly_mask(body), np.float64) / 255.0
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    rgb = np.zeros((H, W, 3))
    # jacket (red) with folds
    fold = 0.75 + 0.25 * np.sin(xx / 23.0 + yy / 41.0) * np.cos(yy / 57.0)
    jacket = np.array([0.62, 0.050, 0.028])[None, None, :] * fold[..., None]
    rgb[:] = jacket
    # white stripes and zip on the jacket
    for x0 in (360, 392, 760, 790):
        rgb[620:1000, x0:x0 + 8] = np.array([0.75, 0.72, 0.70])
    rgb[640:1000, 586:594] = np.array([0.55, 0.55, 0.52])
    # scarf (yellow)
    sc = poly_mask(lambda g, s: g.polygon([(s * 440, s * 600), (s * 700, s * 600), (s * 690, s * 660), (s * 540, s * 700), (s * 455, s * 665)], fill=255))
    sc = np.asarray(sc, np.float64)[..., None] / 255.0
    rgb = rgb * (1 - sc) + sc * np.array([0.9, 0.62, 0.05])
    # face
    face = poly_mask(lambda g, s: g.ellipse([s * 490, s * 340, s * 640, s * 540], fill=255))
    face = np.asarray(face, np.float64)[..., None] / 255.0
    rgb = rgb * (1 - face) + face * np.array([0.44, 0.24, 0.17])
    # hair (dark brown)
    hair = poly_mask(lambda g, s: (g.ellipse([s * 455, s * 245, s * 675, s * 430], fill=255), g.ellipse([s * 495, s * 335, s * 635, s * 520], fill=0)))
    hair = np.asarray(hair, np.float64)[..., None] / 255.0
    rgb = rgb * (1 - hair) + hair * np.array([0.020, 0.012, 0.008])
    # eyes, mouth
    feat = poly_mask(lambda g, s: (g.ellipse([s * 525, s * 400, s * 545, s * 410], fill=255), g.ellipse([s * 585, s * 400, s * 605, s * 410], fill=255), g.rectangle([s * 545, s * 480, s * 585, s * 487], fill=255)))
    feat = np.asarray(feat, np.float64)[..., None] / 255.0
    rgb = rgb * (1 - feat) + feat * np.array([0.03, 0.015, 0.012])
    # a warm rim light on the right edge of the head and shoulder
    rim = np.clip(mask - np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(5)), np.float64) / 255.0 * 0 - 0, 0, 1)
    return rgb, mask


def tone(x):
    x = np.maximum(x, 0)
    y = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
    return np.clip(y, 0, 1)


def srgb(y):
    return np.where(y <= 0.0031308, 12.92 * y, 1.055 * np.power(np.maximum(y, 1e-9), 1 / 2.4) - 0.055)


def display(X):
    return np.clip(srgb(tone(X)), 0, 1)


def save(name, X, q=88):
    Image.fromarray((display(X) * 255 + 0.5).astype(np.uint8)).save(os.path.join(A, name), quality=q)
    print("wrote", name)


def save_gray(name, m, q=88):
    Image.fromarray((np.clip(m, 0, 1) * 255 + 0.5).astype(np.uint8)).save(os.path.join(A, name), quality=q)


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def main():
    bg = render_scene()
    fg, sm = render_subject()
    sm_f = B.conv(sm, B.gauss_kernel(2.0))                     # the feathered Subject mask (a Subject row's own feather)
    X = bg * (1 - sm[..., None]) + fg * sm[..., None]
    # a gentle lens-like softness of the scene itself is not added: everything is sharp, so the blur is the node's
    m = 1 - sm_f                                                 # Subject, inverted: the background
    save("scene.jpg", X)
    save_gray("mask-subject-inverted.jpg", m)
    # the types, all through the same mask, 'respect mask edges' on
    soft_b = B.blur_node(X, m, "soft", radius=50, respect=True)
    soft_a = B.blur_node(X, m, "soft", radius=50, respect=False)
    lens_b = B.blur_node(X, m, "lens", radius=50, respect=True, blades=6, rotation=15.0, highlights=60 / 100)
    lens_round = B.blur_node(X, m, "lens", radius=50, respect=True, blades=0, rotation=0.0, highlights=0.0)
    motion_b = B.blur_node(X, m, "motion", length=45, angle=20.0, respect=True)
    zoom_b = B.blur_node(X, m, "zoom", amount=80, centre=(0.40, 0.40), respect=True)
    for n, im in (("soft-b", soft_b), ("soft-a", soft_a), ("lens-b", lens_b), ("lens-round-b", lens_round), ("motion-b", motion_b), ("zoom-b", zoom_b)):
        save(n + ".jpg", im)
    # a Depth mask instead of the inverted Subject: a depth map (the figure is near, the street recedes upward), made a gradient like the Depth row
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    z = 0.62 + 0.38 * (1 - yy / H) * 1.0                          # far at the top, nearer toward the bottom of the frame
    z = z * (1 - sm_f) + 0.10 * sm_f                               # the figure is nearest
    z = B.conv(z, B.gauss_kernel(4.0))
    md = smooth((z - 0.30) / 0.35)
    save_gray("mask-depth.jpg", md)
    depth_b = B.blur_node(X, md, "soft", radius=40, respect=True)
    save("depth-soft-b.jpg", depth_b)
    # Show mask (Overlay): the accent at 0.55 over the display picture
    for name, img, mm in (("show-mask.jpg", soft_b, m), ("show-mask-depth.jpg", depth_b, md)):
        d = display(img)
        acc = np.array([1.0, 0.435, 0.824])
        o = d + 0.55 * mm[..., None] * (acc - d)
        Image.fromarray((np.clip(o, 0, 1) * 255 + 0.5).astype(np.uint8)).save(os.path.join(A, name), quality=88)
        print("wrote", name)
    # the halo: crops of (a) and (b) around the shoulder, 3 x, with the rest of the numbers
    box = (300, 560, 540, 720)  # x0, y0, x1, y1: the left shoulder against the dark street
    for n, im in (("scene", X), ("soft-a", soft_a), ("soft-b", soft_b)):
        c = display(im)[box[1]:box[3], box[0]:box[2]]
        Image.fromarray((c * 255 + 0.5).astype(np.uint8)).resize(((box[2] - box[0]) * 3, (box[3] - box[1]) * 3), Image.LANCZOS).save(os.path.join(A, f"halo-{n}.jpg"), quality=90)
    # the numbers: the red the subject leaks into the background ring just outside the silhouette (linear R, mean over a 12 px ring, 4 px out)
    # the halo as a number: the share of a background pixel's blurred colour that comes from the figure, in the ring just outside the silhouette
    # (respect off: the kernel's weight that falls on the figure, conv(figure, kernel); respect on: none, by construction)
    k = B.gauss_kernel(0.5 * 0.5 * 0.04 * W)
    share = B.conv(sm, k)
    ring = (B.conv(sm, B.aperture_kernel(16)) > 0.02) & (sm_f < 0.02)
    mean_share = float((share * ring).sum() / ring.sum())
    p95 = float(np.percentile(share[ring], 95))
    red_a = float(((soft_a - soft_b)[..., 0] * ring).sum() / ring.sum())
    print(f"ring {int(ring.sum())} px just outside the figure; share of the blurred colour that comes from the figure: respect off mean {mean_share:.3f}, 95th percentile {p95:.3f}; respect on 0 (the figure has weight 0)")
    open(os.path.join(A, "numbers.txt"), "w").write(f"ring {int(ring.sum())} mean_share_off {mean_share:.4f} p95_off {p95:.4f}\n")


if __name__ == "__main__":
    main()

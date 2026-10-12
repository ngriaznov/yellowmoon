"""Procedural ground-truth patches for training (docs/07 "Synthetic patterns"): ours, no licence issues.

Every function returns a linear-light RGB float32 array ``[h, w, 3]`` with white = 1.0 and is a pure
function of the ``numpy.random.Generator`` passed in. Patterns are rendered at ``ss`` times the target
resolution and box-filtered down, so the ground truth is properly anti-aliased (a camera sees an
optically filtered scene; the *mosaic* is what aliases).

* :func:`zone_plate` - circular chirp reaching and exceeding Nyquist (moire and false-colour test);
* :func:`siemens_star`, :func:`edges_and_shapes` - sharp edges at all angles, thin lines;
* :func:`fine_text` - small rotated text (fonts bundled with Pillow);
* :func:`fabric` - plain and twill weaves at random angle and thread pitch (strong moire);
* :func:`star_field` - point sources on a faint sky (hot pixels, colour fringes on tiny features);
* :func:`fractal_texture` - 1/f colour noise, a stand-in for natural texture;
* :func:`random_pattern` - picks one (or blends two) at random.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _downsample(img: np.ndarray, ss: int) -> np.ndarray:
    """Box-filter an ``[h*ss, w*ss, C]`` array down by ``ss``."""
    h, w = img.shape[0] // ss, img.shape[1] // ss
    return img.reshape(h, ss, w, ss, -1).mean(axis=(1, 3))


def _coords(h: int, w: int, ss: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample coordinates in target-pixel units (pixel centres at +0.5), ``[h*ss, w*ss]`` each."""
    ys = (np.arange(h * ss) + 0.5) / ss
    xs = (np.arange(w * ss) + 0.5) / ss
    return np.meshgrid(ys, xs, indexing="ij")


def _random_color(rng: np.random.Generator, lo: float = 0.02, hi: float = 1.0) -> np.ndarray:
    """A random linear RGB colour with mixed saturation."""
    base = rng.uniform(lo, hi)
    tint = rng.uniform(0.3, 1.0, size=3)
    sat = rng.uniform(0.0, 1.0)
    c = base * ((1.0 - sat) + sat * tint / tint.max())
    return np.clip(c, 0.0, 4.0)


def zone_plate(rng: np.random.Generator, h: int, w: int, ss: int = 3) -> np.ndarray:
    """Circular zone plate: instantaneous frequency grows linearly with radius up to ``f_max`` cycles/px."""
    yy, xx = _coords(h, w, ss)
    cy, cx = rng.uniform(0.2, 0.8) * h, rng.uniform(0.2, 0.8) * w
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    r_max = math.hypot(max(cy, h - cy), max(cx, w - cx))
    f_max = rng.uniform(0.3, 0.95)
    phase = math.pi * f_max * r2 / r_max + rng.uniform(0, 2 * math.pi)
    contrast = rng.uniform(0.4, 1.0)
    v = 0.5 + 0.5 * contrast * np.cos(phase)
    c0, c1 = _random_color(rng, 0.02, 0.3), _random_color(rng, 0.3, 1.0)
    img = c0 + (c1 - c0) * v[..., None]
    if rng.random() < 0.4:  # separate chirp per channel: coloured moire fringes
        shifts = rng.uniform(0.0, 0.6, size=3)
        img = np.stack([0.5 + 0.5 * contrast * np.cos(phase * (1 + s)) for s in shifts], axis=-1)
        img = c0 + (c1 - c0) * img
    return _downsample(img, ss).astype(np.float32)


def siemens_star(rng: np.random.Generator, h: int, w: int, ss: int = 3) -> np.ndarray:
    """Radial spokes (many orientations, frequency rising towards the centre)."""
    yy, xx = _coords(h, w, ss)
    cy, cx = rng.uniform(0.3, 0.7) * h, rng.uniform(0.3, 0.7) * w
    ang = np.arctan2(yy - cy, xx - cx)
    spokes = int(rng.integers(12, 72))
    v = (np.sin(spokes * ang + rng.uniform(0, 2 * math.pi)) > 0).astype(np.float64)
    c0, c1 = _random_color(rng, 0.02, 0.2), _random_color(rng, 0.4, 1.0)
    img = c0 + (c1 - c0) * v[..., None]
    return _downsample(img, ss).astype(np.float32)


def edges_and_shapes(rng: np.random.Generator, h: int, w: int, ss: int = 3) -> np.ndarray:
    """Random filled discs, rotated rectangles and thin lines over a smooth background."""
    yy, xx = _coords(h, w, ss)
    bg0, bg1 = _random_color(rng, 0.02, 0.5), _random_color(rng, 0.02, 0.5)
    a = rng.uniform(0, 2 * math.pi)
    t = (xx / w * math.cos(a) + yy / h * math.sin(a)) * 0.5 + 0.5
    img = bg0 + (bg1 - bg0) * np.clip(t, 0, 1)[..., None]
    for _ in range(int(rng.integers(4, 14))):
        col = _random_color(rng, 0.02, 1.0)
        cy, cx = rng.uniform(0, h), rng.uniform(0, w)
        kind = rng.integers(0, 3)
        if kind == 0:  # disc
            rad = rng.uniform(1.5, 0.3 * min(h, w))
            mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= rad**2
        elif kind == 1:  # rotated rectangle
            a = rng.uniform(0, math.pi)
            u = (xx - cx) * math.cos(a) + (yy - cy) * math.sin(a)
            v = -(xx - cx) * math.sin(a) + (yy - cy) * math.cos(a)
            mask = (np.abs(u) <= rng.uniform(1.5, 0.35 * w)) & (np.abs(v) <= rng.uniform(1.5, 0.35 * h))
        else:  # thin line, 0.6..2 px wide
            a = rng.uniform(0, math.pi)
            d = np.abs((xx - cx) * math.sin(a) - (yy - cy) * math.cos(a))
            mask = d <= rng.uniform(0.3, 1.0)
        img = np.where(mask[..., None], col, img)
    return _downsample(img, ss).astype(np.float32)


def fine_text(rng: np.random.Generator, h: int, w: int, ss: int = 3) -> np.ndarray:
    """Rotated small text on a plain background (Pillow's bundled font)."""
    big = ss * 2  # render large, rotate, then box-filter
    canvas = Image.new("L", (int(w * big * 1.6), int(h * big * 1.6)), 0)
    draw = ImageDraw.Draw(canvas)
    words = ["Yellowmoon", "demosaic", "0123456789", "moire", "ALEXA", "raw", "sensor", "lorem", "ipsum", "weave"]
    y = 0.0
    while y < canvas.size[1]:
        px = rng.uniform(4.5, 14.0)  # final pixel height of this line
        try:
            font = ImageFont.load_default(size=max(6, int(px * big)))
        except TypeError:  # very old Pillow: bitmap font
            font = ImageFont.load_default()
        line = " ".join(words[int(i)] for i in rng.integers(0, len(words), size=int(rng.integers(3, 9))))
        draw.text((rng.uniform(0, 20) * big, y), line, fill=255, font=font)
        y += px * big * rng.uniform(1.1, 1.5)
    canvas = canvas.rotate(float(rng.uniform(-35, 35)), resample=Image.BICUBIC)
    cw, ch = canvas.size
    ox = int(rng.integers(0, max(1, cw - w * big)))
    oy = int(rng.integers(0, max(1, ch - h * big)))
    crop = np.asarray(canvas.crop((ox, oy, ox + w * big, oy + h * big)), dtype=np.float64) / 255.0
    v = _downsample(crop[..., None], big)[..., 0]
    fg, bg = _random_color(rng, 0.02, 0.4), _random_color(rng, 0.3, 1.0)
    if rng.random() < 0.5:
        fg, bg = bg, fg
    return (bg + (fg - bg) * v[..., None]).astype(np.float32)


def fabric(rng: np.random.Generator, h: int, w: int, ss: int = 3) -> np.ndarray:
    """Plain or twill weave at a random angle; thread pitch 1.6..6 px (fine weaves moire strongly)."""
    yy, xx = _coords(h, w, ss)
    ang = rng.uniform(0, math.pi)
    pitch_u = rng.uniform(1.6, 6.0)
    pitch_v = pitch_u * rng.uniform(0.8, 1.25)
    u = (xx * math.cos(ang) + yy * math.sin(ang)) / pitch_u
    v = (-xx * math.sin(ang) + yy * math.cos(ang)) / pitch_v
    fu, fv = np.floor(u), np.floor(v)
    if rng.random() < 0.5:
        over_warp = ((fu + fv) % 2) == 0
    else:
        over_warp = ((fu - fv) % 4) < 2
    warp_col, weft_col = _random_color(rng, 0.05, 1.0), _random_color(rng, 0.05, 1.0)
    shade_u = 0.45 + 0.55 * np.sin(math.pi * (u - fu))
    shade_v = 0.45 + 0.55 * np.sin(math.pi * (v - fv))
    img = np.where(over_warp[..., None], warp_col * shade_u[..., None], weft_col * shade_v[..., None])
    return _downsample(img, ss).astype(np.float32)


def star_field(rng: np.random.Generator, h: int, w: int, ss: int = 2) -> np.ndarray:
    """Point sources with Gaussian PSF on a faint gradient; flux follows a power law (a few very bright)."""
    hh, ww = h * ss, w * ss
    sky = rng.uniform(0.002, 0.03)
    grad = rng.uniform(0.0, 1.0) * np.linspace(0, 1, ww)[None, :] * sky
    img = np.broadcast_to((sky + grad)[..., None] * rng.uniform(0.7, 1.0, size=3), (hh, ww, 3)).copy()
    hi_n = max(8, 6 + h * w // 60)
    n = int(rng.integers(min(15, hi_n - 1), hi_n))
    for _ in range(n):
        cy, cx = rng.uniform(0, hh), rng.uniform(0, ww)
        flux = 0.05 * (1.0 - rng.random()) ** (-1.0 / 1.4)  # Pareto, alpha 1.4
        sig = rng.uniform(0.55, 1.4) * ss
        t = rng.uniform()
        col = (1 - t) * np.array([1.0, 0.72, 0.45]) + t * np.array([0.62, 0.8, 1.0])
        r = int(math.ceil(4 * sig))
        y0, y1 = max(0, int(cy) - r), min(hh, int(cy) + r + 1)
        x0, x1 = max(0, int(cx) - r), min(ww, int(cx) + r + 1)
        if y0 >= y1 or x0 >= x1:
            continue
        gy = np.arange(y0, y1)[:, None] + 0.5 - cy
        gx = np.arange(x0, x1)[None, :] + 0.5 - cx
        psf = np.exp(-(gy**2 + gx**2) / (2 * sig**2)) / (2 * math.pi * sig**2)
        img[y0:y1, x0:x1] += (flux * ss * ss * psf)[..., None] * col
    return _downsample(img, ss).astype(np.float32)


def fractal_texture(rng: np.random.Generator, h: int, w: int, beta: float | None = None) -> np.ndarray:
    """1/f^beta colour noise, passed through a soft threshold so it has edges as well as smooth areas."""
    beta = rng.uniform(1.2, 2.4) if beta is None else beta
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    f = np.sqrt(fy**2 + fx**2)
    f[0, 0] = 1.0
    amp = f ** (-beta / 2.0)
    amp[0, 0] = 0.0
    chans = []
    shared = rng.normal(size=(h, w))
    for _ in range(3):
        n = 0.7 * shared + 0.3 * rng.normal(size=(h, w))
        field_ = np.fft.ifft2(np.fft.fft2(n) * amp).real
        field_ = (field_ - field_.mean()) / (field_.std() + 1e-9)
        chans.append(field_)
    g = np.stack(chans, axis=-1)
    sharp = rng.uniform(0.5, 4.0)
    base = _random_color(rng, 0.05, 0.5)
    img = base * np.exp(0.9 * np.tanh(sharp * g * 0.6))
    return np.clip(img, 0, None).astype(np.float32)


PATTERNS: dict[str, Callable[..., np.ndarray]] = {
    "zone_plate": zone_plate,
    "siemens_star": siemens_star,
    "edges_and_shapes": edges_and_shapes,
    "fine_text": fine_text,
    "fabric": fabric,
    "star_field": star_field,
    "fractal_texture": fractal_texture,
}


def random_pattern(rng: np.random.Generator, h: int, w: int, name: str | None = None) -> tuple[np.ndarray, str]:
    """A random pattern (or blend of two, 30% of the time). Returns ``(rgb [h, w, 3], description)``."""
    names = list(PATTERNS)
    weights = np.array([1.0, 0.6, 1.0, 1.0, 1.2, 0.8, 1.2])
    if name is not None:
        return PATTERNS[name](rng, h, w), name
    first = names[int(rng.choice(len(names), p=weights / weights.sum()))]
    img = PATTERNS[first](rng, h, w)
    desc = first
    if rng.random() < 0.3:
        second = names[int(rng.choice(len(names), p=weights / weights.sum()))]
        other = PATTERNS[second](rng, h, w)
        # Smooth random mask: low-frequency fractal thresholded.
        m = fractal_texture(rng, h, w, beta=3.0)[..., 0]
        m = (m - np.median(m)) * rng.uniform(4.0, 30.0) / (m.std() + 1e-9)
        m = (1.0 / (1.0 + np.exp(-m)))[..., None]
        img = img * m + other * (1.0 - m)
        desc = f"{first}+{second}"
    return img.astype(np.float32), desc

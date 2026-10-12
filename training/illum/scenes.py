"""Synthetic scenes for the illuminant estimator (M1-06).

A scene is a small camera preview (96 x 64 pixels, black-subtracted raw RGB, white level 1.0, before any white
balance) that stands for the 1/8-resolution preview the Rust estimator reads: a random layout of reflectances
(soft-edged Voronoi cells) with per-cell brightness, a smooth shading field, a per-pixel texture, optional
specular highlights and white or grey patches, lit by one light (or two, for ``mixed``) and rendered through
the camera's curves ``S_P``. A Poisson-Gaussian noise of the body's nominal sensor (binned 8 x 8 as in the real
preview) and per-channel clipping follow; clipped pixels are masked as the estimator expects.

Per scene we also draw what the camera would record as its as-shot white balance (the true light, perturbed),
the brightness value ``Bv`` and the ISO (noise level). Every scene is a pure function of ``(seed, split, index)``.

``split`` is ``"train"``, ``"val"`` (training lights, a different seed stream) or ``"test"`` (held-out lights and
held-out reflectance subset, :mod:`training.illum.spectra`). ``refl="ces"`` renders the 99 CIE CES (evaluation
only) instead of the Apache and smooth reflectances.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from training.illum import spectra as sp

H, W = 64, 96
#: Bumped whenever the generator changes, so that cached datasets are not reused.
GENERATOR_VERSION = 4
#: Reflectance mix per split: (share of smooth random spectra, share of single patches, amplitude of the smooth multiplicative
#: perturbation of patches); the rest are convex mixes of two or three patches. Training is broader than validation and test, so
#: that the held-out scenes measure generalisation to reflectances the model has not seen, in a mix that stays fixed.
REFLECTANCE_MIX = {"train": (0.50, 0.20, 0.20), "val": (0.35, 0.30, 0.12), "test": (0.35, 0.30, 0.12)}
SPLIT_ID = {"train": 0, "val": 1, "test": 2}

#: Class frequencies of the synthetic training and test scenes.
CLASS_P = {"daylight": 0.22, "shade_overcast": 0.16, "tungsten_halogen": 0.17, "fluorescent": 0.14, "led": 0.17, "mixed": 0.14}

#: (mean, sd, lo, hi) of the brightness value ``Bv = log2(N^2/t) - log2(ISO/100)`` per class (synthetic prior).
BV_PRIOR = {
    "daylight": (9.3, 1.7, 3.0, 13.0), "shade_overcast": (7.6, 1.6, 3.0, 12.0),
    "tungsten_halogen": (3.4, 1.8, -2.0, 9.0), "fluorescent": (4.6, 1.6, -1.0, 10.0),
    "led": (4.6, 2.0, -2.0, 11.0), "mixed": (5.0, 2.4, -2.0, 12.0),
}

#: Mixed lighting: pairs of classes.
MIXED_PAIRS = (
    ("tungsten_halogen", "daylight"), ("tungsten_halogen", "shade_overcast"), ("fluorescent", "daylight"),
    ("led", "daylight"), ("tungsten_halogen", "fluorescent"), ("led", "tungsten_halogen"),
    ("led", "fluorescent"), ("daylight", "shade_overcast"),
)

#: Camera presets of the as-shot white balance (blackbody below 4000 K, D series above), kelvin.
AS_SHOT_PRESETS_K = (5500.0, 6500.0, 7500.0, 3000.0, 4000.0, 5200.0)

BINNING = np.array([16.0, 32.0, 16.0])  # raw samples per R, G, B in an 8 x 8 block of the mosaic


@dataclass
class Scene:
    """One rendered scene and its ground truth."""

    img: np.ndarray                 # (H, W, 3) float32 raw RGB, white level 1
    mask: np.ndarray                # (H, W) uint8, 1 where a channel clipped
    truth_uv: np.ndarray            # (2,) log-chroma (ln G/R, ln G/B) of the scene light
    klass: int                      # index into spectra.CLASSES
    bv: float                       # brightness value of the "metadata"
    iso: float
    as_shot_uv: np.ndarray | None   # (2,) or None (no as-shot white balance in the file)
    as_shot_kind: str
    light_names: tuple[str, ...]
    light_families: tuple[str, ...]
    mix_alpha: float = 1.0          # mean weight of the first light (1 for a single light)
    meta: dict = field(default_factory=dict)

    @property
    def truth_w(self) -> np.ndarray:
        """Camera white of the scene light, green = 1."""
        return np.array([np.exp(-self.truth_uv[0]), 1.0, np.exp(-self.truth_uv[1])])


def _smooth_field(rng: np.random.Generator, h: int = H, w: int = W, grid=(4, 6)) -> np.ndarray:
    """Smooth zero-mean unit-variance noise field (bilinear upsampling of a coarse random grid)."""
    g = rng.normal(size=(grid[0] + 1, grid[1] + 1))
    ys = np.linspace(0, grid[0], h)
    xs = np.linspace(0, grid[1], w)
    y0 = np.minimum(ys.astype(int), grid[0] - 1)
    x0 = np.minimum(xs.astype(int), grid[1] - 1)
    fy, fx = (ys - y0)[:, None], (xs - x0)[None, :]
    f = (g[y0][:, x0] * (1 - fy) * (1 - fx) + g[y0 + 1][:, x0] * fy * (1 - fx)
         + g[y0][:, x0 + 1] * (1 - fy) * fx + g[y0 + 1][:, x0 + 1] * fy * fx)
    return (f - f.mean()) / (f.std() + 1e-9)


def _pick_reflectances(rng: np.random.Generator, split: str, refl: str, n: int) -> np.ndarray:
    """``n`` reflectance spectra (n, 81) for a scene."""
    if refl == "ces":
        ces = sp.evaluation_reflectances()["CIE 2017 CES (99)"]
        return ces[rng.integers(0, ces.shape[0], size=n)]
    tr, va, te = sp.apache_patches()
    pool = {"train": tr, "val": va, "test": te}[split]
    mix = REFLECTANCE_MIX[split]
    out = np.empty((n, sp.N_WL))
    palette = None
    if rng.random() < 0.25:  # a dominant-colour scene: most patches come from a small palette
        m = int(rng.integers(2, 6))
        palette = np.stack([_one_reflectance(rng, pool, mix) for _ in range(m)])
    for i in range(n):
        if palette is not None and rng.random() < 0.8:
            out[i] = palette[int(rng.integers(palette.shape[0]))]
        else:
            out[i] = _one_reflectance(rng, pool, mix)
    return out


def _one_reflectance(rng: np.random.Generator, pool: np.ndarray, mix: tuple[float, float, float]) -> np.ndarray:
    """One reflectance: a smooth random spectrum, a rawtoaces patch or a convex mix of two or three patches (shares in ``mix``),
    the last two with a smooth multiplicative perturbation. A model must not memorise the 132 training patches."""
    p_smooth, p_patch, tilt = mix
    u = rng.random()
    if u < p_smooth:
        return sp.smooth_reflectance(rng)
    if u < p_smooth + p_patch:
        base = pool[int(rng.integers(pool.shape[0]))]
    else:
        k = int(rng.integers(2, 4))
        base = rng.dirichlet(np.ones(k)) @ pool[rng.integers(0, pool.shape[0], size=k)]
    return np.clip(base * sp.smooth_tilt(rng, tilt), 0.01, 0.98)


def _light_for(rng: np.random.Generator, klass: str, split: str) -> sp.Light:
    return sp.draw_light(rng, klass, split)


def _draw_class(rng: np.random.Generator) -> str:
    names = list(CLASS_P)
    return names[int(rng.choice(len(names), p=np.array(list(CLASS_P.values())) / sum(CLASS_P.values())))]


def draw_bv(rng: np.random.Generator, klass: str) -> float:
    """Brightness value for a class: Gaussian prior, with 8% of the scenes drawn from another class' prior."""
    k = klass if rng.random() > 0.08 else _draw_class(rng)
    mu, sd, lo, hi = BV_PRIOR[k]
    return float(np.clip(rng.normal(mu, sd), lo, hi))


def draw_iso(rng: np.random.Generator, bv: float) -> float:
    """ISO from the brightness value (dim scenes need high ISO), in third-stop steps from 100 to 51200."""
    stops = float(np.clip(8.5 - 0.8 * bv + rng.normal(0.0, 0.9), 0.0, 9.0))
    return float(100.0 * 2.0 ** (round(stops * 3.0) / 3.0))


def as_shot_uv(rng: np.random.Generator, cam: sp.Camera, truth_uv: np.ndarray) -> tuple[np.ndarray | None, str]:
    """What the camera records as its as-shot white balance: the truth, perturbed, or a preset, or nothing.

    50 % good automatic white balance (sigma 0.04 in log-chroma, about 1.5 degrees of RGB angle), 27 % off (0.10),
    8 % bad (0.25), 11 % a preset white balance unrelated to the scene (daylight, cloudy, shade, tungsten...), 4 % no
    as-shot value. These are assumptions (no camera data in V1): the as-shot error of the generated scenes has a
    median of about 2 degrees and a heavy tail.
    """
    r = rng.random()
    if r < 0.04:
        return None, "missing"
    if r < 0.54:
        return truth_uv + rng.normal(0.0, 0.04, size=2), "auto_good"
    if r < 0.81:
        return truth_uv + rng.normal(0.0, 0.10, size=2), "auto_off"
    if r < 0.89:
        return truth_uv + rng.normal(0.0, 0.25, size=2), "auto_bad"
    t = float(AS_SHOT_PRESETS_K[int(rng.integers(len(AS_SHOT_PRESETS_K)))])
    spd = sp.blackbody_at(t) if t < 4000.0 else sp.daylight_at(t)
    return sp.light_uv(cam, spd) + rng.normal(0.0, 0.01, size=2), "preset"


def render_scene(
    cam: sp.Camera,
    seed: int,
    split: str,
    index: int,
    refl: str = "default",
    klass: str | None = None,
    light: sp.Light | None = None,
    stream: int = 0,
) -> Scene:
    """Render scene ``index`` of ``split`` for ``cam``. Deterministic in ``(seed, split, index, stream)``.

    ``klass`` fixes the class (not ``mixed``) and ``light`` fixes the light (the gate lights); both otherwise
    follow the class frequencies and the split's light pool.
    """
    rng = np.random.default_rng([seed, SPLIT_ID[split], index, stream])
    if light is not None:
        cls_name = light.klass
        lights = [light]
    else:
        cls_name = klass if klass is not None else _draw_class(rng)
        if cls_name == "mixed":
            pair = MIXED_PAIRS[int(rng.integers(len(MIXED_PAIRS)))]
            lights = [_light_for(rng, pair[0], split), _light_for(rng, pair[1], split)]
        else:
            lights = [_light_for(rng, cls_name, split)]
    mixed = len(lights) == 2

    # --- layout -----------------------------------------------------------------------------------
    n_cells = int(np.round(np.exp(rng.uniform(np.log(6), np.log(40)))))
    n_neutral = int(rng.integers(1, 5)) if rng.random() < 0.5 else 0
    n_all = n_cells + n_neutral
    refl_spectra = _pick_reflectances(rng, split, refl, n_cells)
    if n_neutral:
        grey = rng.uniform(0.04, 0.9, size=n_neutral)
        refl_spectra = np.concatenate([refl_spectra, np.repeat(grey[:, None], sp.N_WL, axis=1)])

    seeds = rng.uniform(size=(n_all, 2)) * np.array([H / W, 1.0])  # (y, x) with unit width
    add_w = rng.normal(0.0, 0.015, size=n_all)
    yy, xx = np.meshgrid((np.arange(H) + 0.5) / W, (np.arange(W) + 0.5) / W, indexing="ij")
    d2 = (yy[..., None] - seeds[:, 0]) ** 2 + (xx[..., None] - seeds[:, 1]) ** 2 - add_w
    two = np.argpartition(d2, 1, axis=-1)[..., :2]
    da = np.take_along_axis(d2, two[..., :1], axis=-1)[..., 0]
    db = np.take_along_axis(d2, two[..., 1:2], axis=-1)[..., 0]
    swap = db < da
    d_first, d_second = np.where(swap, db, da), np.where(swap, da, db)
    order = np.where(swap[..., None], two[..., ::-1], two)
    tau = rng.uniform(0.0005, 0.004)
    lam = 1.0 / (1.0 + np.exp(-(d_second - d_first) / tau))  # weight of the nearest cell
    first, second = order[..., 0], order[..., 1]

    # --- spectral rendering of every cell under each light ---------------------------------------
    cell_gain = np.exp(rng.normal(0.0, 0.35, size=n_all))  # albedo / brightness variety
    comps = []
    whites = []
    intens = [1.0] if not mixed else [1.0, float(np.exp(rng.normal(0.0, 0.5)))]
    for L, inten in zip(lights, intens):
        rgb_cells = (refl_spectra * L.spd) @ cam.curves  # (n_all, 3)
        comps.append(inten * rgb_cells * cell_gain[:, None])
        whites.append(inten * cam.response(L.spd))
    if mixed:
        a0 = rng.uniform(-1.4, 1.4)
        alpha = 1.0 / (1.0 + np.exp(-(a0 + 1.2 * _smooth_field(rng))))
    else:
        alpha = np.ones((H, W))

    def cell_image(c):
        return lam[..., None] * c[first] + (1.0 - lam[..., None]) * c[second]

    img = alpha[..., None] * cell_image(comps[0])
    white_field = alpha[..., None] * whites[0]
    if mixed:
        img = img + (1.0 - alpha[..., None]) * cell_image(comps[1])
        white_field = white_field + (1.0 - alpha[..., None]) * whites[1]

    # --- shading, texture, specular highlights ------------------------------------------------------
    shade = np.exp(rng.uniform(0.1, 0.6) * _smooth_field(rng))
    texture = np.exp(rng.uniform(0.01, 0.12) * rng.normal(size=(H, W)))
    chroma_tex = np.exp(rng.uniform(0.0, 0.03) * rng.normal(size=(H, W, 3)))
    img = img * (shade * texture)[..., None] * chroma_tex
    white_field = white_field * (shade * texture)[..., None]
    if rng.random() < 0.3:
        spec = np.zeros((H, W))
        for _ in range(int(rng.integers(1, 5))):
            cy, cx = rng.uniform(0, H), rng.uniform(0, W)
            sg = rng.uniform(1.0, 4.0)
            spec += rng.uniform(0.3, 2.5) * np.exp(-0.5 * (((np.arange(H)[:, None] - cy) ** 2
                                                           + (np.arange(W)[None, :] - cx) ** 2) / sg**2))
        local = img.mean(axis=-1, keepdims=True)
        img = img + spec[..., None] * white_field / white_field.mean(axis=-1, keepdims=True) * local.mean() * 2.0

    # --- exposure -----------------------------------------------------------------------------------
    y_img = 0.25 * img[..., 0] + 0.5 * img[..., 1] + 0.25 * img[..., 2]
    target = 2.0 ** rng.uniform(-5.2, -2.4)
    if rng.random() < 0.2:
        target *= 2.0 ** rng.uniform(0.5, 2.0)  # exposed to the right: some highlights clip
    scale = target / max(float(np.median(y_img)), 1e-12)
    img = img * scale
    white_field = white_field * scale

    # truth: the (brightness-weighted) average light over the image
    truth_rgb = white_field.reshape(-1, 3).mean(axis=0)
    truth_uv = np.array([-np.log(truth_rgb[0] / truth_rgb[1]), -np.log(truth_rgb[2] / truth_rgb[1])])

    # --- noise and clipping -------------------------------------------------------------------------
    bv = draw_bv(rng, cls_name)
    iso = draw_iso(rng, bv)
    gain = (iso / 100.0) / cam.full_well_e                 # white-level fractions per electron
    var = (np.maximum(img, 0.0) * gain + (cam.read_noise_e * gain) ** 2) / BINNING
    img = img + rng.normal(size=img.shape) * np.sqrt(var)
    clipped = (img >= 1.0).any(axis=-1)
    img = np.minimum(img, 1.0)

    asu, kind = as_shot_uv(rng, cam, truth_uv)
    names = tuple(L.name for L in lights)
    fams = tuple(L.family for L in lights)
    return Scene(
        img=img.astype(np.float32), mask=clipped.astype(np.uint8), truth_uv=truth_uv,
        klass=sp.CLASS_INDEX[cls_name], bv=bv, iso=iso, as_shot_uv=asu, as_shot_kind=kind,
        light_names=names, light_families=fams, mix_alpha=float(alpha.mean()),
        meta={"median_y": float(target), "clipped_fraction": float(clipped.mean()), "n_cells": n_all, "refl": refl},
    )

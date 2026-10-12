"""Datasets of synthetic scenes: histograms, truth and side information per scene, cached as ``.npz``.

``build(profile_id, split, n, ...)`` renders ``n`` scenes (deterministic in the seed) and keeps, per scene, the
normalised histograms of :mod:`training.illum.features`, the histogram statistics, the truth, the class, the brightness
value, the ISO and the as-shot log-chroma. Optionally the previews themselves are kept (for baselines and test
vectors; 18 kB each, so only for evaluation sets).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from training.common import paths
from training.illum import features as ft
from training.illum import scenes as sc
from training.illum import spectra as sp

SEED = 20261012
CACHE = Path(os.environ.get("YM_ILLUM_CACHE", paths.OUT / "illum" / "cache"))  # separate caches for parallel runs


@dataclass
class Dataset:
    """Arrays over ``n`` scenes."""

    hist: np.ndarray            # (n, C, bins, bins) float32
    valid_fraction: np.ndarray  # (n, C)
    in_range: np.ndarray        # (n, C)
    truth_uv: np.ndarray        # (n, 2)
    klass: np.ndarray           # (n,) int
    bv: np.ndarray              # (n,)
    iso: np.ndarray             # (n,)
    as_shot_uv: np.ndarray      # (n, 2), NaN where missing
    as_shot_kind: np.ndarray    # (n,) str
    family: np.ndarray          # (n,) str: family of the first light ("mixed:a+b" for mixed)
    light: np.ndarray           # (n,) str
    imgs: np.ndarray | None = None   # (n, H, W, 3) float32
    masks: np.ndarray | None = None  # (n, H, W) uint8

    def __len__(self) -> int:
        return self.hist.shape[0]

    def subset(self, idx) -> "Dataset":
        return Dataset(*(None if a is None else a[idx] for a in (
            self.hist, self.valid_fraction, self.in_range, self.truth_uv, self.klass, self.bv, self.iso,
            self.as_shot_uv, self.as_shot_kind, self.family, self.light, self.imgs, self.masks)))


def hist_spec_for(profile_id: str, channels=("intensity", "edge"), bin_size=0.0625, n_bins=64) -> ft.HistSpec:
    """Histogram window of a camera: centred on the extent of the light locus (1500-25000 K) in log-chroma."""
    cam = sp.camera(profile_id)
    us, vs = [], []
    for t in np.arange(1500.0, 25001.0, 250.0):
        spd = sp.blackbody_at(t) if t < 4000.0 else sp.daylight_at(t)
        uv = sp.light_uv(cam, spd)
        us.append(uv[0])
        vs.append(uv[1])
    half = 0.5 * n_bins * bin_size
    cu, cv = 0.5 * (min(us) + max(us)), 0.5 * (min(vs) + max(vs))
    lo_u = round((cu - half) / bin_size) * bin_size
    lo_v = round((cv - half) / bin_size) * bin_size
    return ft.HistSpec(n_bins=n_bins, bin_size=bin_size, lo_u=float(lo_u), lo_v=float(lo_v), channels=tuple(channels))


def _one(args):
    profile_id, spec_json, seed, split, index, refl, klass, light_name, keep, stream = args
    cam = sp.camera(profile_id)
    spec = ft.HistSpec.from_json(spec_json)
    light = None
    if light_name is not None:
        light = next(L for L in sp.gate_lights() if L.name == light_name)
    s = sc.render_scene(cam, seed, split, index, refl=refl, klass=klass, light=light, stream=stream)
    h, st = ft.histograms(s.img, s.mask, spec)
    fam = s.light_families[0] if len(s.light_families) == 1 else "mixed:" + "+".join(s.light_families)
    return (h.astype(np.float32), st.valid_fraction, st.in_range, s.truth_uv, s.klass, s.bv, s.iso,
            np.full(2, np.nan) if s.as_shot_uv is None else s.as_shot_uv, s.as_shot_kind, fam,
            "+".join(s.light_names), s.img if keep else None, s.mask if keep else None)


def build(profile_id: str, split: str, n: int, spec: ft.HistSpec, refl: str = "default", klass: str | None = None,
          light_name: str | None = None, keep_images: bool = False, seed: int = SEED, stream: int = 0,
          start: int = 0, procs: int = 2, cache: bool = True, verbose: bool = True) -> Dataset:
    """Render scenes ``start .. start + n - 1`` of ``split`` and histogram them (cached on disk)."""
    tag = f"v{sc.GENERATOR_VERSION}_{profile_id}_{split}_{refl}_{klass}_{light_name}_{n}_{start}_{seed}_{stream}_{int(keep_images)}_{spec.n_bins}_{spec.bin_size}_{spec.lo_u:.4f}_{spec.lo_v:.4f}_{'-'.join(spec.channels)}"
    path = CACHE / (tag.replace(" ", "_").replace("/", "_") + ".npz")
    if cache and path.exists():
        z = np.load(path, allow_pickle=False)
        imgs = z["imgs"] if keep_images else None
        masks = z["masks"] if keep_images else None
        if keep_images:  # evaluation sets store the previews, not the histograms (4x smaller): rebuild them
            hist = np.empty((len(z["klass"]), len(spec.channels), spec.n_bins, spec.n_bins), dtype=np.float32)
            vf, inr = np.empty((len(hist), len(spec.channels))), np.empty((len(hist), len(spec.channels)))
            for i in range(len(hist)):
                h, st = ft.histograms(imgs[i], masks[i], spec)
                hist[i], vf[i], inr[i] = h, st.valid_fraction, st.in_range
        else:
            hist, vf, inr = z["hist"].astype(np.float32), z["valid_fraction"], z["in_range"]
        return Dataset(hist, vf, inr, z["truth_uv"], z["klass"], z["bv"], z["iso"],
                       z["as_shot_uv"], z["as_shot_kind"], z["family"], z["light"], imgs, masks)
    t0 = time.time()
    jobs = [(profile_id, spec.to_json(), seed, split, i, refl, klass, light_name, keep_images, stream)
            for i in range(start, start + n)]
    if procs > 1:
        with Pool(procs) as pool:
            res = pool.map(_one, jobs, chunksize=32)
    else:
        res = [_one(j) for j in jobs]
    ds = Dataset(
        hist=np.stack([r[0] for r in res]), valid_fraction=np.stack([r[1] for r in res]),
        in_range=np.stack([r[2] for r in res]), truth_uv=np.stack([r[3] for r in res]),
        klass=np.array([r[4] for r in res]), bv=np.array([r[5] for r in res]), iso=np.array([r[6] for r in res]),
        as_shot_uv=np.stack([r[7] for r in res]), as_shot_kind=np.array([r[8] for r in res]),
        family=np.array([r[9] for r in res]), light=np.array([r[10] for r in res]),
        imgs=np.stack([r[11] for r in res]) if keep_images else None,
        masks=np.stack([r[12] for r in res]) if keep_images else None)
    if verbose:
        print(f"  built {n} {split}/{refl} scenes for {profile_id} in {time.time() - t0:.0f} s", flush=True)
    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        side = dict(truth_uv=ds.truth_uv, klass=ds.klass, bv=ds.bv, iso=ds.iso, as_shot_uv=ds.as_shot_uv,
                    as_shot_kind=ds.as_shot_kind, family=ds.family, light=ds.light)
        if keep_images:
            np.savez(path, imgs=ds.imgs, masks=ds.masks, **side)
        else:  # training sets: histograms in float16 (relative error 5e-4 per bin, irrelevant for training)
            np.savez(path, hist=ds.hist.astype(np.float16), valid_fraction=ds.valid_fraction, in_range=ds.in_range, **side)
    return ds

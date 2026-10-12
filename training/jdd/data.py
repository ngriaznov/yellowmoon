"""Training data pipeline for `jdd` (M2-04): ground-truth sources and the docs/07 sample generator.

Ground truth comes from two kinds of source:

* :class:`PixelShift200Source` - the PixelShift200 dataset (Sony A7R III, 4-shot pixel shift, CC BY 4.0)
  gives full RGB at every pixel. **Code only in V1**: the dataset is not downloadable from the build
  machine; the expected on-disk layout is documented on :class:`PixelShift200` and has to be checked
  against the real download (HUMAN M2-H4).
* :class:`SyntheticSource` - procedural patterns from :mod:`training.jdd.patterns` (ours).

:func:`generate_sample` implements steps 1 to 7 of "Sample generation" in ``docs/07-ml-models.md``:

1. a linear RGB ground-truth patch (balanced camera RGB, white = 1.0, at least 2 px larger than the
   sample in each direction);
2. random exposure and white balance (gains ``g0`` vary like real files);
3. RGGB mosaic with a random CFA phase;
4. virtual-camera noise (:mod:`training.jdd.noise`): Poisson-Gaussian in DN with Tukey-lambda read
   noise, row noise and per-plane colour bias;
5. quantisation: 14-bit; p = 0.3 through the cRAW encoder and decoder (:mod:`training.jdd.craw`),
   which yields the ``qstep`` map; p = 0.1 simulated 12-bit;
6. clipping at white, clip mask; p = 0.2 push highlights over white;
7. sigma map from the *true* model, jittered by a log-uniform factor in [0.75, 1.25]; p = 0.1 clean
   mosaic with sigma = 0 ("demosaic only").

Conventions fixed here (clarifications of docs/07, recorded there):

* the network input holds *one* sigma plane and *one* qstep plane at half resolution: the RMS over the
  four photosites of the 2x2 cell, in balanced units (``* g0`` per channel). The clip plane is 1 where any
  of the four photosites is clipped;
* the qstep plane is the cRAW quantisation step and **zero for every other mode** (docs/05: "zeros for
  lossless files"); the physical step of 14-bit (1 DN) and 12-bit (4 DN) samples still enters the sigma
  model, and clipping is ``raw >= white - 4 qstep`` with that qstep (so exactly at white for them);
* the target is the ground truth *clipped at the same per-channel white* (``g0_c``) as the input, so
  the network is never asked to invent clipped content;
* units: raw values are ``(dn - black) / (white - black)`` (negatives kept, clamped at -0.05) and are
  multiplied by ``g0`` ("balanced"); white = 1.0 before gains.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage

from . import craw
from .noise import (
    BLACK_DN,
    RANGE_DN,
    WHITE_DN,
    VirtualCamera,
    apply_noise,
    noise_variance_dn2,
    sample_camera,
)
from .patterns import random_pattern

RGGB_COLOR = np.array([[0, 1], [1, 2]])  # colour index (R, G, B) of the photosite at (y % 2, x % 2)
PLANE_SLICES = ((0, 0), (0, 1), (1, 0), (1, 1))  # (dy, dx) of R, G1, G2, B inside the 2x2 cell
NEG_CLAMP = -0.05
SPLIT_SEED = 0  # whole-scene train/validation split of PixelShift200: fixed, independent of any training seed


# --------------------------------------------------------------------------------------- mosaic helpers
def mosaic_rggb(rgb: np.ndarray) -> np.ndarray:
    """``[H, W, 3]`` -> ``[H, W]`` RGGB mosaic (origin top left is a red photosite)."""
    h, w = rgb.shape[:2]
    col = RGGB_COLOR[(np.arange(h) % 2)[:, None], (np.arange(w) % 2)[None, :]]
    return np.take_along_axis(rgb, col[..., None], axis=2)[..., 0]


def pack_planes(mosaic: np.ndarray) -> np.ndarray:
    """``[H, W]`` mosaic -> ``[4, H/2, W/2]`` planes ``R, G1, G2, B``."""
    return np.stack([mosaic[dy::2, dx::2] for dy, dx in PLANE_SLICES], axis=0)


def unpack_planes(planes: np.ndarray) -> np.ndarray:
    """Inverse of :func:`pack_planes`."""
    _, h, w = planes.shape
    out = np.empty((2 * h, 2 * w), dtype=planes.dtype)
    for k, (dy, dx) in enumerate(PLANE_SLICES):
        out[dy::2, dx::2] = planes[k]
    return out


def plane_gain_vector(g0: Sequence[float]) -> np.ndarray:
    """Gains per plane ``R, G1, G2, B`` from per-colour ``g0 = (R, G, B)``."""
    return np.array([g0[0], g0[1], g0[1], g0[2]], dtype=np.float64)


def rms4(per_plane: np.ndarray) -> np.ndarray:
    """RMS over the 4 planes of a ``[4, h, w]`` array."""
    return np.sqrt(np.mean(np.square(per_plane), axis=0))


# ------------------------------------------------------------------------------------- the sample generator
@dataclass(frozen=True)
class SamplerConfig:
    """Probabilities and ranges of the sample generator (defaults are the docs/07 values)."""

    patch_half: int = 64  # half-resolution patch size of the sample (full-res output is twice that)
    p_craw: float = 0.3
    p_12bit: float = 0.1
    p_highlight: float = 0.2
    p_clean: float = 0.1
    sigma_jitter: tuple[float, float] = (0.75, 1.25)
    peak_range: tuple[float, float] = (2.0**-7, 0.98)  # 99.5th percentile of the raw signal, log-uniform
    peak_range_highlight: tuple[float, float] = (1.05, 4.0)
    gain_r_range: tuple[float, float] = (1.3, 3.2)
    gain_b_range: tuple[float, float] = (1.2, 2.6)
    cast_sigma: float = 0.12  # log-normal scene colour cast applied to the ground truth
    iso_range: tuple[float, float] = (50.0, 64000.0)
    smooth_size: int = 5  # box filter of the clean signal for the sigma map (docs/06)

    @property
    def gt_size(self) -> int:
        """Side of the ground-truth patch that :func:`generate_sample` needs (full-res px)."""
        return 2 * self.patch_half + 2


@dataclass
class Sample:
    """One training sample. ``x``: ``[7, h, w]`` network input; ``y``: ``[3, 2h, 2w]`` target."""

    x: np.ndarray
    y: np.ndarray
    meta: dict
    debug: dict = field(default_factory=dict)


def _log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def generate_sample(
    gt: np.ndarray,
    rng: np.random.Generator,
    cfg: SamplerConfig | None = None,
    *,
    camera: VirtualCamera | None = None,
    debug: bool = False,
) -> Sample:
    """Turn a ground-truth RGB patch into a ``(noisy mosaic input, clean target)`` pair.

    ``gt``: ``[H + 2, W + 2, 3]`` linear RGB (``H`` and ``W`` even, the full-res sample size); the extra
    two pixels let step 3 pick a random CFA phase. ``camera`` overrides the sampled virtual camera
    (tests). Deterministic given ``rng``.
    """
    cfg = cfg or SamplerConfig()
    gt = np.asarray(gt, dtype=np.float64)
    hh, ww = gt.shape[0] - 2, gt.shape[1] - 2
    if hh % 2 or ww % 2 or hh < 4 or ww < 4 or gt.shape[2] != 3:
        raise ValueError(f"gt must be [H+2, W+2, 3] with even H, W >= 4, got {gt.shape}")
    meta: dict = {}

    # Step 3 (phase) first: a random crop offset decides which scene pixel the red photosite sees.
    py, px = (int(v) for v in rng.integers(0, 2, size=2))
    gt = gt[py : py + hh, px : px + ww]
    meta["phase"] = [py, px]

    # Step 2: scene colour cast, white-balance gains g0, exposure.
    cast = np.exp(rng.normal(0.0, cfg.cast_sigma, size=3))
    gt = gt * cast
    g0 = np.array(
        [_log_uniform(rng, *cfg.gain_r_range), 1.0, _log_uniform(rng, *cfg.gain_b_range)], dtype=np.float64
    )
    raw_ref = gt / g0  # raw-domain colour of the scene at unit exposure
    peak_now = float(np.percentile(raw_ref.max(axis=2), 99.5))
    highlights = bool(rng.random() < cfg.p_highlight)
    target_peak = _log_uniform(rng, *(cfg.peak_range_highlight if highlights else cfg.peak_range))
    scale = target_peak / max(peak_now, 1e-6)
    scale = min(scale, 200.0 / max(float(raw_ref.max()), 1e-6))  # never blow up near-black patches
    gt = gt * scale
    meta.update(
        g0=[float(v) for v in g0], highlights=highlights, peak=float(target_peak), exposure_scale=float(scale)
    )
    scene_raw = gt / g0  # raw-domain, unclipped; 1.0 is the sensor's white

    # Step 3: RGGB mosaic of the clean raw signal, in DN above black.
    clean_mosaic = mosaic_rggb(scene_raw)
    signal_dn = clean_mosaic * RANGE_DN

    # Step 4: virtual camera noise.
    cam = camera or sample_camera(rng, cfg.iso_range)
    clean = bool(rng.random() < cfg.p_clean)
    u = float(rng.random())
    mode = "14bit"
    if not clean:
        mode = "craw" if u < cfg.p_craw else ("12bit" if u < cfg.p_craw + cfg.p_12bit else "14bit")
    noisy = signal_dn if clean else apply_noise(signal_dn, cam, rng)
    meta.update(
        clean=clean,
        mode=mode,
        iso=cam.iso,
        k_dn_per_e=cam.k_dn_per_e,
        sigma_read_dn=cam.sigma_read_dn,
        tukey_lambda=cam.tukey_lambda,
        sigma_row_dn=cam.sigma_row_dn,
        bias_dn=list(cam.bias_dn),
    )

    # Steps 5 and 6: quantise, clip at white.
    dn = np.rint(np.clip(noisy + cam.black_dn, 0.0, cam.white_dn))
    q_dn = np.ones_like(dn)  # physical quantisation step in DN: enters the noise model (q^2 / 12)
    white_dn = cam.white_dn  # code at which this file clips
    if mode == "12bit":
        # A 12-bit ADC has codes 0..4095, which are 4 * code on the 14-bit scale of the sampler.
        dn = np.minimum(np.rint(dn / 4.0), 4095.0) * 4.0
        q_dn = np.full_like(dn, 4.0)
        white_dn = 4.0 * 4095.0
    elif mode == "craw":
        dec, q = craw.roundtrip(dn.astype(np.int64), through_bytes=False)
        dn = np.clip(dec.astype(np.float64), 0.0, cam.white_dn)
        q_dn = q
    # The network's qstep plane exists only where the decoder knows the step (cRAW, docs/04). Uncompressed
    # and lossless files feed zeros (docs/05 R6), and R1 then clips only at exactly the white level, so the
    # sampler does the same: q_input_dn is zero for every mode but cRAW.
    q_input_dn = q_dn if mode == "craw" else np.zeros_like(q_dn)
    clipped = dn >= white_dn - 4.0 * q_input_dn  # docs/05 R1: raw >= white - 4 qstep
    raw_norm = np.maximum((dn - cam.black_dn) / cam.range_dn, NEG_CLAMP)
    gain_plane = plane_gain_vector(g0)[:, None, None]
    planes = pack_planes(raw_norm) * gain_plane  # balanced units, [4, h, w]

    # Step 7: sigma map from the true model, jittered.
    smooth = ndimage.uniform_filter(
        np.clip(pack_planes(signal_dn), 0.0, cam.range_dn), size=(1, cfg.smooth_size, cfg.smooth_size), mode="reflect"
    )
    q_planes = pack_planes(q_dn)
    var = noise_variance_dn2(smooth, cam, q_planes)
    sigma_planes = np.sqrt(var) / cam.range_dn * gain_plane
    jitter = _log_uniform(rng, *cfg.sigma_jitter)
    sigma = np.zeros_like(rms4(sigma_planes)) if clean else jitter * rms4(sigma_planes)
    qstep = rms4(pack_planes(q_input_dn) / cam.range_dn * gain_plane)
    clip_mask = pack_planes(clipped.astype(np.float64)).max(axis=0)
    meta["sigma_jitter"] = 1.0 if clean else float(jitter)

    x = np.concatenate([planes, sigma[None], qstep[None], clip_mask[None]], axis=0).astype(np.float32)
    target = np.clip(np.minimum(gt, g0), 0.0, None)  # clipped at the per-channel white, [H, W, 3]
    y = np.ascontiguousarray(np.moveaxis(target, 2, 0)).astype(np.float32)
    dbg = {}
    if debug:
        dbg = {
            "clean_mosaic": clean_mosaic,
            "dn": dn,
            "q_dn": q_dn,
            "q_input_dn": q_input_dn,
            "clipped": clipped,
            "sigma_true_planes": sigma_planes,
            "signal_dn": signal_dn,
            "camera": cam,
        }
    return Sample(x=x, y=y, meta=meta, debug=dbg)


# ------------------------------------------------------------------------------------------ GT sources
class SyntheticSource:
    """Procedural ground truth (:mod:`training.jdd.patterns`)."""

    name = "synthetic"

    def __init__(self, gt_size: int) -> None:
        self.gt_size = gt_size

    def gt(self, rng: np.random.Generator) -> tuple[np.ndarray, str]:
        """A fresh random pattern patch ``[gt_size, gt_size, 3]`` and its description."""
        return random_pattern(rng, self.gt_size, self.gt_size)


def merge_pixel_shift(frames: Sequence[np.ndarray], shifts: Sequence[tuple[int, int]]) -> np.ndarray:
    """Merge four RGGB mosaics shot with the sensor moved by one pixel into full RGB.

    ``frames``: four ``[H, W]`` mosaics (linear, same scale). ``shifts[k] = (dy, dx)``: in frame ``k``
    scene pixel ``(y, x)`` was read by sensor pixel ``(y + dy, x + dx)``. Every scene pixel is seen by
    one red, one blue and two green photosites over the four frames; green is averaged. Returns
    ``[H, W, 3]``; the first/last rows and columns that fall off the sensor in some frame are filled
    from the frames that do contain them (so callers should crop one pixel from each border).
    """
    h, w = frames[0].shape
    acc = np.zeros((h, w, 3))
    cnt = np.zeros((h, w, 3))
    ys, xs = np.mgrid[0:h, 0:w]
    for frame, (dy, dx) in zip(frames, shifts, strict=True):
        sy, sx = ys + dy, xs + dx
        valid = (sy >= 0) & (sy < h) & (sx >= 0) & (sx < w)
        syc, sxc = np.clip(sy, 0, h - 1), np.clip(sx, 0, w - 1)
        col = RGGB_COLOR[syc % 2, sxc % 2]
        val = frame[syc, sxc]
        for c in range(3):
            m = valid & (col == c)
            acc[..., c] += np.where(m, val, 0.0)
            cnt[..., c] += m
    return acc / np.maximum(cnt, 1)


def simulate_pixel_shift(rgb: np.ndarray, shifts: Sequence[tuple[int, int]]) -> list[np.ndarray]:
    """Inverse of :func:`merge_pixel_shift` for tests: mosaics a full-RGB image shot at each shift."""
    h, w = rgb.shape[:2]
    frames = []
    ys, xs = np.mgrid[0:h, 0:w]  # sensor pixel coordinates
    for dy, dx in shifts:
        sy, sx = np.clip(ys - dy, 0, h - 1), np.clip(xs - dx, 0, w - 1)  # scene pixel seen by sensor pixel
        col = RGGB_COLOR[ys % 2, xs % 2]
        frames.append(np.take_along_axis(rgb[sy, sx], col[..., None], axis=2)[..., 0])
    return frames


DEFAULT_PIXEL_SHIFTS = ((0, 0), (0, 1), (1, 1), (1, 0))


class PixelShift200:
    """Scene store of the PixelShift200 dataset (CC BY 4.0). **Expected layout, to be verified on download.**

    ``root/<scene>/`` holds, in order of preference:

    1. ``gt.npy`` - ``float32 [H, W, 3]`` linear camera RGB, black subtracted, scaled so the sensor white
       is 1.0 and multiplied by the as-shot white-balance gains ("balanced"); or ``gt.npz`` with the
       same array under the key ``rgb``. Produce it once per scene from the four ARW frames with
       ``python3 -m training.jdd.data --prepare <root>`` (uses :func:`merge_pixel_shift` and rawpy,
       training-time only);
    2. ``*.ARW`` (or ``.arw``) - the four pixel-shift frames, sorted by name, in shot order.

    Scenes are split deterministically by a hash of the scene name (whole scenes are held out, 10 %).
    """

    def __init__(self, root: str | Path, shifts: Sequence[tuple[int, int]] = DEFAULT_PIXEL_SHIFTS) -> None:
        self.root = Path(root)
        self.shifts = tuple(shifts)

    def scenes(self) -> list[str]:
        """Names of all scene folders that contain data in one of the two layouts."""
        if not self.root.is_dir():
            return []
        names = []
        for d in sorted(p for p in self.root.iterdir() if p.is_dir()):
            if (d / "gt.npy").exists() or (d / "gt.npz").exists() or len(self._arw_files(d)) == 4:
                names.append(d.name)
        return names

    @staticmethod
    def _arw_files(d: Path) -> list[Path]:
        return sorted(p for p in d.iterdir() if p.suffix.lower() == ".arw")

    def load_scene(self, name: str) -> np.ndarray:
        """Full-RGB ground truth ``[H, W, 3]`` float32 for one scene."""
        d = self.root / name
        if (d / "gt.npy").exists():
            return np.load(d / "gt.npy").astype(np.float32)
        if (d / "gt.npz").exists():
            return np.load(d / "gt.npz")["rgb"].astype(np.float32)
        files = self._arw_files(d)
        if len(files) != 4:
            raise FileNotFoundError(f"{d}: need gt.npy, gt.npz or four ARW frames, found {len(files)} ARW files")
        mosaics, wb = zip(*(load_arw_mosaic(f) for f in files), strict=True)
        rgb = merge_pixel_shift(mosaics, self.shifts)[1:-1, 1:-1]
        return (rgb * np.asarray(wb[0])).astype(np.float32)


def load_arw_mosaic(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Read one ARW with rawpy (LibRaw, training time only): normalised RGGB mosaic and as-shot gains (R, G, B)."""
    import rawpy

    with rawpy.imread(str(path)) as raw:
        mosaic = raw.raw_image_visible.astype(np.float64)
        pattern = raw.raw_pattern
        if not np.array_equal(pattern, np.array([[0, 1], [3, 2]])) and not np.array_equal(
            pattern, np.array([[0, 1], [1, 2]])
        ):
            raise ValueError(f"{path}: unexpected CFA pattern {pattern.tolist()}, expected RGGB")
        black = float(np.mean(raw.black_level_per_channel))
        white = float(raw.white_level)
        wb = raw.camera_whitebalance
        gains = (wb[0] / wb[1], 1.0, wb[2] / wb[1])
    return np.clip((mosaic - black) / (white - black), 0.0, None), gains


def scene_is_validation(name: str, val_fraction: float = 0.1, seed: int = SPLIT_SEED) -> bool:
    """Deterministic whole-scene hold-out: hash of ``seed:name`` below ``val_fraction``."""
    h = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64 < val_fraction


class PixelShift200Source:
    """Random crops of PixelShift200 scenes (train or validation split).

    ``seed`` is the **split** seed, not the training seed: it must be the same for training, validation and
    every evaluation of every run, otherwise scenes held out in one run are trained on in another.
    """

    name = "pixelshift200"

    def __init__(
        self,
        root: str | Path,
        gt_size: int,
        split: str = "train",
        val_fraction: float = 0.1,
        seed: int = SPLIT_SEED,
    ) -> None:
        self.store = PixelShift200(root)
        self.gt_size = gt_size
        names = self.store.scenes()
        want_val = split == "val"
        self.scene_names = [n for n in names if scene_is_validation(n, val_fraction, seed) == want_val]
        self._cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.scene_names)

    def gt(self, rng: np.random.Generator) -> tuple[np.ndarray, str]:
        """A random crop ``[gt_size, gt_size, 3]`` of a random scene."""
        if not self.scene_names:
            raise RuntimeError("no PixelShift200 scenes found for this split")
        name = self.scene_names[int(rng.integers(len(self.scene_names)))]
        if name not in self._cache:
            if len(self._cache) >= 4:
                self._cache.pop(next(iter(self._cache)))
            self._cache[name] = self.store.load_scene(name)
        img = self._cache[name]
        y = int(rng.integers(0, img.shape[0] - self.gt_size + 1))
        x = int(rng.integers(0, img.shape[1] - self.gt_size + 1))
        return img[y : y + self.gt_size, x : x + self.gt_size], f"pixelshift200/{name}"


# ----------------------------------------------------------------------------------------- torch dataset
class JddDataset:
    """Map-style dataset of generated samples (``torch.utils.data.Dataset`` compatible).

    Sample ``i`` is a pure function of ``(seed, i)`` (``numpy`` ``default_rng([seed, i])``), so the data
    are identical across runs, worker counts and machines. ``sources`` is a list of
    ``(source, weight)``; each sample draws its source from the weights.
    """

    def __init__(
        self,
        n: int,
        seed: int,
        cfg: SamplerConfig | None = None,
        sources: Sequence[tuple[object, float]] | None = None,
        cache: bool = False,
    ) -> None:
        self.n = n
        self.cache = cache  # memoise samples (they are pure functions of the index): tiny overfit sets
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.seed = seed
        self.cfg = cfg or SamplerConfig()
        self.sources = list(sources) if sources else [(SyntheticSource(self.cfg.gt_size), 1.0)]
        w = np.array([s[1] for s in self.sources], dtype=np.float64)
        self.weights = w / w.sum()

    def __len__(self) -> int:
        return self.n

    def sample(self, i: int, *, debug: bool = False) -> Sample:
        """The numpy sample ``i`` (with its metadata)."""
        rng = np.random.default_rng([self.seed, int(i)])
        k = int(rng.choice(len(self.sources), p=self.weights))
        gt, desc = self.sources[k][0].gt(rng)
        s = generate_sample(gt, rng, self.cfg, debug=debug)
        s.meta["source"] = desc
        return s

    def __getitem__(self, i: int):
        import torch

        if self.cache and i in self._cache:
            x, y = self._cache[i]
        else:
            s = self.sample(i)
            x, y = s.x, s.y
            if self.cache:
                self._cache[i] = (x, y)
        return torch.from_numpy(x), torch.from_numpy(y)


def prepare_pixelshift_scenes(root: str | Path) -> int:
    """Convert every scene of ARW frames under ``root`` to ``gt.npy`` (balanced linear RGB). Returns the count."""
    store = PixelShift200(root)
    done = 0
    for name in store.scenes():
        d = Path(root) / name
        if (d / "gt.npy").exists() or (d / "gt.npz").exists():
            continue
        np.save(d / "gt.npy", store.load_scene(name))
        done += 1
    return done


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="PixelShift200 helpers")
    ap.add_argument("--prepare", metavar="ROOT", help="write gt.npy next to the ARW frames of every scene")
    ap.add_argument("--list", metavar="ROOT", help="list scenes and the train/val split")
    args = ap.parse_args()
    if args.prepare:
        print(f"prepared {prepare_pixelshift_scenes(args.prepare)} scenes")
    if args.list:
        names = PixelShift200(args.list).scenes()
        val = [n for n in names if scene_is_validation(n)]
        print(json.dumps({"scenes": len(names), "val": val}, indent=1))


if __name__ == "__main__":
    _main()

__all__ = [
    "BLACK_DN",
    "WHITE_DN",
    "JddDataset",
    "PixelShift200",
    "PixelShift200Source",
    "Sample",
    "SPLIT_SEED",
    "SamplerConfig",
    "SyntheticSource",
    "generate_sample",
    "merge_pixel_shift",
    "mosaic_rggb",
    "pack_planes",
    "simulate_pixel_shift",
    "unpack_planes",
]

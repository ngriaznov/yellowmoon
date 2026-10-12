"""Evaluation of `jdd` against a classical baseline on the gate sets (docs/07 "Gates (M2)", docs/13).

    python3 -m training.jdd.evaluate --weights runs/jdd-001/weights.safetensors --set synthetic
    python3 -m training.jdd.evaluate --weights W --set folder --path testdata/demosaic/kodak --srgb
    python3 -m training.jdd.evaluate --weights W --set pixelshift200 --path testdata/pixelshift200
    python3 -m training.jdd.evaluate --weights W --set synthetic --baseline mypkg.mymod:demosaic

Sets
    ``synthetic``      generated samples (noise, cRAW, clipping), the same generator as training but a
                       different seed range; reports the noisy case, which is what the app sees.
    ``folder``         clean RGB images (PNG/TIFF/NPY) mosaicked without noise: Kodak, McMaster, Gharbi
                       hard and moire sets live in ``testdata/demosaic/<set>/`` (fetched by the
                       ``demosaic-sets`` fetch script, evaluation only). ``--srgb`` linearises 8/16-bit sRGB
                       images first; ``--noise-iso`` adds virtual-camera noise at that ISO.
    ``pixelshift200``  held-out PixelShift200 scenes (validation split), with synthetic noise.

The classical baseline hook: ``--baseline NAME`` where ``NAME`` is ``bilinear`` (built in), ``menon``
(``colour_demosaicing`` if installed; BSD-3, evaluation only) or ``package.module:function`` with
``function(mosaic[H, W] float64, pattern="RGGB") -> rgb[H, W, 3]``. The gate asks for >= 2 dB over the
classical fallback in the linear-light domain.
"""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from .data import (
    SPLIT_SEED,
    JddDataset,
    PixelShift200Source,
    SamplerConfig,
    SyntheticSource,
    generate_sample,
    mosaic_rggb,
    pack_planes,
    unpack_planes,
)
from .metrics import bilinear_baseline, psnr_linear, psnr_log, summarize
from .model import JddNet
from .noise import sample_camera
from .weights_io import load_weights, write_json

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_DB = 2.0
IMAGE_SUFFIXES = {".png", ".tif", ".tiff", ".npy", ".jpg", ".jpeg"}

Baseline = Callable[[np.ndarray], np.ndarray]


# ---------------------------------------------------------------------------------------------- baselines
def baseline_bilinear(mosaic: np.ndarray, pattern: str = "RGGB") -> np.ndarray:
    """Bilinear demosaic of an RGGB mosaic ``[H, W]`` -> ``[H, W, 3]`` (float64)."""
    from scipy import ndimage

    if pattern != "RGGB":
        raise ValueError("only RGGB is supported")

    h, w = mosaic.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = ((yy % 2 == 0) & (xx % 2 == 0)).astype(np.float64)
    b = ((yy % 2 == 1) & (xx % 2 == 1)).astype(np.float64)
    g = 1.0 - r - b
    k_rb = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 4.0
    k_g = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]]) / 4.0
    out = np.empty((h, w, 3))
    for c, (mask, k) in enumerate(((r, k_rb), (g, k_g), (b, k_rb))):
        out[..., c] = ndimage.convolve(mosaic * mask, k, mode="mirror")
    return out


def baseline_menon(mosaic: np.ndarray) -> np.ndarray:
    """Menon 2007 via ``colour_demosaicing`` (optional dependency, BSD-3, evaluation only)."""
    try:
        from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise SystemExit("--baseline menon needs `pip install colour-demosaicing`") from e
    return demosaicing_CFA_Bayer_Menon2007(mosaic, "RGGB")


def get_baseline(name: str) -> Baseline:
    """Resolve a baseline name (``bilinear``, ``menon`` or ``module:function``)."""
    if name == "bilinear":
        return baseline_bilinear
    if name == "menon":
        return baseline_menon
    if ":" not in name:
        raise ValueError(f"unknown baseline {name!r}; use bilinear, menon or package.module:function")
    mod, fn = name.split(":", 1)
    func = getattr(importlib.import_module(mod), fn)
    return lambda m: np.asarray(func(m, "RGGB"), dtype=np.float64)


# ---------------------------------------------------------------------------------------------- inputs
def srgb_to_linear(v: np.ndarray) -> np.ndarray:
    """sRGB EOTF, ``v`` in [0, 1]."""
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def load_image(path: Path, srgb: bool) -> np.ndarray:
    """Read an image as float64 ``[H, W, 3]`` in [0, 1] linear light (``srgb`` linearises first)."""
    if path.suffix.lower() == ".npy":
        img = np.load(path).astype(np.float64)
    else:
        from PIL import Image

        im = Image.open(path)
        arr = np.asarray(im.convert("RGB") if im.mode not in ("RGB", "I;16") else im)
        img = arr.astype(np.float64) / (65535.0 if arr.dtype == np.uint16 else 255.0)
    return srgb_to_linear(img) if srgb else img


def clean_inputs(rgb: np.ndarray, multiple: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crop to a multiple of ``multiple`` px, mosaic (no noise): returns ``(x [7,h,w], y [3,H,W], mosaic [H,W])``."""
    h, w = (rgb.shape[0] // multiple) * multiple, (rgb.shape[1] // multiple) * multiple
    rgb = rgb[:h, :w]
    mosaic = mosaic_rggb(rgb)
    planes = pack_planes(mosaic)
    zeros = np.zeros_like(planes[0])
    # Clean, uncompressed input: sigma 0, qstep 0 (zeros for lossless files, docs/05 R6), nothing clipped.
    x = np.concatenate([planes, zeros[None], zeros[None], zeros[None]], axis=0)
    return x.astype(np.float32), np.moveaxis(rgb, 2, 0).astype(np.float32), mosaic


def folder_pairs(path: Path, srgb: bool, noise_iso: float | None, seed: int):
    """Yield ``(name, x, y, mosaic)`` for every image in ``path`` (optionally with virtual-camera noise)."""
    files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise SystemExit(f"no images in {path}")
    for i, f in enumerate(files):
        rgb = load_image(f, srgb)
        if noise_iso is None:
            x, y, mosaic = clean_inputs(rgb)
        else:  # embed the image as a mid-exposure scene through the training generator
            rng = np.random.default_rng([seed, i])
            cfg = SamplerConfig(p_clean=0.0, p_craw=0.0, p_12bit=0.0, p_highlight=0.0, peak_range=(0.5, 0.5))
            cam = sample_camera(rng, (noise_iso, noise_iso))
            h, w = (rgb.shape[0] // 8) * 8, (rgb.shape[1] // 8) * 8
            cfg = SamplerConfig(**{**cfg.__dict__, "patch_half": min(h, w) // 2 - 1})
            s = generate_sample(rgb[: cfg.gt_size, : cfg.gt_size], rng, cfg, camera=cam)
            x, y, mosaic = s.x, s.y, unpack_planes(s.x[:4].astype(np.float64))
        yield f.name, x, y, mosaic


# -------------------------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate_pairs(model: JddNet, pairs, baseline: Baseline | None, device: torch.device) -> dict:
    """PSNR of the model, the built-in bilinear residual base and (optionally) the external baseline."""
    model.eval()
    rows: dict[str, list[float]] = {}

    def add(key: str, v: torch.Tensor) -> None:
        rows.setdefault(key, []).extend(v.tolist())

    per_image = []
    for name, x_np, y_np, mosaic in pairs:
        x = torch.from_numpy(x_np)[None].to(device)
        y = torch.from_numpy(y_np)[None].to(device)
        out = model(x)
        add("model_linear", psnr_linear(out, y))
        add("model_log", psnr_log(out, y))
        add("bilinear_linear", psnr_linear(bilinear_baseline(model, x), y))
        add("bilinear_log", psnr_log(bilinear_baseline(model, x), y))
        rec = {"name": name, "model_linear": rows["model_linear"][-1], "bilinear_linear": rows["bilinear_linear"][-1]}
        if baseline is not None and mosaic.size > 1:
            base = torch.from_numpy(np.moveaxis(baseline(mosaic), 2, 0).astype(np.float32))[None].to(device)
            add("baseline_linear", psnr_linear(base, y))
            add("baseline_log", psnr_log(base, y))
            rec["baseline_linear"] = rows["baseline_linear"][-1]
        per_image.append(rec)
    res = {k: summarize(v) for k, v in rows.items()}
    ref = "baseline_linear" if "baseline_linear" in res else "bilinear_linear"
    res["reference"] = ref
    res["gain_linear_db"] = res["model_linear"]["mean"] - res[ref]["mean"]
    res["gain_log_db"] = res["model_log"]["mean"] - res[ref.rsplit("_", 1)[0] + "_log"]["mean"]
    res["gate_2db_linear"] = bool(res["gain_linear_db"] >= GATE_DB)
    res["images"] = per_image
    return res


def synthetic_pairs(n: int, seed: int, patch_half: int):
    """Generated noisy samples as evaluation pairs. The external baseline gets the noisy balanced mosaic."""
    cfg = SamplerConfig(patch_half=patch_half)
    ds = JddDataset(n, seed, cfg, [(SyntheticSource(cfg.gt_size), 1.0)])
    for i in range(n):
        s = ds.sample(i)
        yield f"synthetic-{i}", s.x, s.y, unpack_planes(s.x[:4].astype(np.float64))


def pixelshift_pairs(root: Path, n: int, seed: int, patch_half: int, split_seed: int = SPLIT_SEED):
    """Validation-split PixelShift200 crops with synthetic noise.

    ``seed`` draws the crops and the noise; ``split_seed`` picks the held-out scenes and must be the one the
    model was trained with (the default is the fixed split of ``data.SPLIT_SEED``).
    """
    cfg = SamplerConfig(patch_half=patch_half)
    src = PixelShift200Source(root, cfg.gt_size, split="val", seed=split_seed)
    if not len(src):
        raise SystemExit(f"no validation scenes under {root}")
    ds = JddDataset(n, seed + 17, cfg, [(src, 1.0)])
    for i in range(n):
        s = ds.sample(i)
        yield f"{s.meta['source']}#{i}", s.x, s.y, unpack_planes(s.x[:4].astype(np.float64))


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--set", choices=["synthetic", "folder", "pixelshift200"], default="synthetic")
    ap.add_argument("--path", type=Path, default=None, help="image folder or dataset root")
    ap.add_argument("--srgb", action="store_true", help="folder images are sRGB-encoded")
    ap.add_argument("--noise-iso", type=float, default=None, help="folder set: add virtual-camera noise at this ISO")
    ap.add_argument("--baseline", default="bilinear")
    ap.add_argument("--n", type=int, default=32, help="samples for generated sets")
    ap.add_argument("--patch-half", type=int, default=64)
    ap.add_argument("--seed", type=int, default=424242, help="seed of generated samples, crops and noise")
    ap.add_argument("--split-seed", type=int, default=SPLIT_SEED, help="PixelShift200 scene split (use the training value)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=None, help="write the JSON report here")
    args = ap.parse_args(argv)
    if args.set != "synthetic" and args.path is None:
        ap.error(f"--set {args.set} needs --path")

    model = load_weights(args.weights).to(args.device)
    baseline = get_baseline(args.baseline)
    if args.set == "synthetic":
        pairs = synthetic_pairs(args.n, args.seed, args.patch_half)
    elif args.set == "folder":
        pairs = folder_pairs(args.path, args.srgb, args.noise_iso, args.seed)
    else:
        pairs = pixelshift_pairs(args.path, args.n, args.seed, args.patch_half, args.split_seed)
    res = evaluate_pairs(model, pairs, baseline, torch.device(args.device))
    ref = res["reference"]
    print(f"{args.set}: model {res['model_linear']['mean']:.2f} dB, {ref.split('_')[0]} {res[ref]['mean']:.2f} dB "
          f"(linear, n={int(res['model_linear']['n'])}); gain {res['gain_linear_db']:+.2f} dB linear, "
          f"{res['gain_log_db']:+.2f} dB log; gate (>= {GATE_DB} dB): {'PASS' if res['gate_2db_linear'] else 'FAIL'}")
    if args.out:
        write_json(args.out, res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

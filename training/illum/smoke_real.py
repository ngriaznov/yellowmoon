"""Indicative smoke test of the estimator on real ARW files (local only, not a gate).

    python3 -m training.illum.smoke_real [--dir testdata/real/files] [--models data/models/illum]

For every ``*.ARW`` (LibRaw unpacks; no decoder of ours is involved) it builds the 1/8-resolution preview the Rust estimator
reads (black-subtracted, white level 1, **before white balance**, clipped blocks masked), takes the camera's as-shot gains,
and runs the model of the file's profile (the registry's choice; bodies without a profile use the A7R IV model) with and
without the as-shot prior. There is **no ground truth** for these files (unrecorded provenance), so the figures are the
angle between the histogram-only estimate and the as-shot white, and the same for grey world: a gross-failure check (a
wrong raw scale, a window that misses the light) and not an accuracy measurement.

When ``exiftool`` is installed the exposure settings are read as well and the model gets ``Bv = log2(N^2/t) - log2(ISO/100)`` like
the Rust estimator does (``ShotInfo::bv``), so that the class head runs on real brightness values. The synthetic scenes cover
``Bv`` from -2 to 13 (:data:`training.illum.scenes.BV_PRIOR`); real sunlit scenes are 14 to 15 (the sunny-16 rule is EV 15), so a
file outside that range is marked: the head then extrapolates.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from training.common import paths
from training.illum import baselines as bl
from training.illum import model as md


def preview_of(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(preview (h, w, 3) float32, clip mask (h, w) uint8, as-shot gains (3,) normalised to green 1)."""
    import rawpy

    with rawpy.imread(str(path)) as raw:
        v = raw.raw_image_visible.astype(np.float64)
        col = raw.raw_colors_visible.astype(np.int64)
        black = np.array(raw.black_level_per_channel, dtype=np.float64)
        white = float(raw.white_level)
        wb = np.array(raw.camera_whitebalance, dtype=np.float64)
    x = (v - black[col]) / (white - black[col])
    clipped = x >= 0.995
    h, w = (x.shape[0] // 8) * 8, (x.shape[1] // 8) * 8
    x, col, clipped = x[:h, :w], col[:h, :w], clipped[:h, :w]
    cls = np.where(col == 3, 1, col)
    out = np.zeros((h // 2, w // 2, 3))
    xr, cr, kr = x.reshape(h // 2, 2, w // 2, 2), cls.reshape(h // 2, 2, w // 2, 2), clipped.reshape(h // 2, 2, w // 2, 2)
    for c in range(3):
        sel = cr == c
        out[..., c] = (xr * sel).sum(axis=(1, 3)) / np.maximum(sel.sum(axis=(1, 3)), 1)
    mask = kr.any(axis=(1, 3))
    small = out.reshape(h // 8, 4, w // 8, 4, 3).mean(axis=(1, 3))
    mask8 = mask.reshape(h // 8, 4, w // 8, 4).any(axis=(1, 3))
    g = 0.5 * (wb[1] + wb[3]) if wb[3] > 0 else wb[1]
    return small.astype(np.float32), mask8.astype(np.uint8), np.array([wb[0] / g, 1.0, wb[2] / g])


def brightness_value(path: Path) -> float | None:
    """``Bv = log2(N^2/t) - log2(ISO/100)`` from the file's EXIF (``exiftool`` if installed), ``None`` when unavailable."""
    if shutil.which("exiftool") is None:
        return None
    out = subprocess.run(["exiftool", "-s3", "-n", "-FNumber", "-ExposureTime", "-ISO", str(path)], capture_output=True,
                         text=True, check=False).stdout.split()
    try:
        n, t, iso = (float(v) for v in out[:3])
    except ValueError:
        return None
    if min(n, t, iso) <= 0 or len(out) < 3:
        return None
    return float(np.log2(n * n / t) - np.log2(iso / 100.0))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=paths.TESTDATA / "real" / "files")
    ap.add_argument("--models", type=Path, default=paths.REPO_ROOT / "data" / "models" / "illum")
    args = ap.parse_args(argv)
    reg = json.loads(paths.REGISTRY_JSON.read_text())
    by_model = {b["model"]: b["profile"] for b in reg["bodies"]}
    files = sorted(args.dir.glob("*.ARW"))
    if not files:
        print(f"no ARW files in {args.dir}", file=sys.stderr)
        return 3
    cache: dict[str, md.IllumModel] = {}
    print(f"{'file':<22} {'profile':<16} {'hist vs as-shot':>15} {'grey-world vs as-shot':>21} {'with prior vs as-shot':>21} {'conf':>5} {'Bv':>5}  top class")
    rows = []
    for f in files:
        pid = by_model.get(f.stem.split("_")[0], reg["default_profile"])
        mp = args.models / f"{pid}.json"
        if not mp.exists():
            mp = args.models / f"{reg['default_profile']}.json"
            pid = reg["default_profile"]
        m = cache.setdefault(pid, md.IllumModel.from_json(json.loads(mp.read_text())))
        img, mask, g = preview_of(f)
        a = np.array([np.log(g[0] / g[1]), np.log(g[2] / g[1])])  # as-shot (u, v) = (ln gR/gG, ln gB/gG)
        w_as_shot = bl.w_from_uv(a)
        bv = brightness_value(f)
        hist = m.estimate(img, mask, None, bv)
        withp = m.estimate(img, mask, a, bv)
        gw = bl.grey_world(img, mask)
        e_h = float(bl.angular_error_deg(bl.w_from_uv(hist["uv"]), w_as_shot))
        e_g = float(bl.angular_error_deg(gw, w_as_shot))
        e_p = float(bl.angular_error_deg(bl.w_from_uv(withp["uv"]), w_as_shot))
        top = md.CLASSES[int(np.argmax(withp["probs"]))]
        rows.append((e_h, e_g, e_p))
        bv_text = "-" if bv is None else f"{bv:.1f}" + ("*" if not -2.0 <= bv <= 13.0 else "")
        print(f"{f.stem:<22} {pid:<16} {e_h:15.2f} {e_g:21.2f} {e_p:21.2f} {withp['confidence']:5.2f} {bv_text:>5}  {top}")
    r = np.array(rows)
    print("Bv: log2(N^2/t) - log2(ISO/100) from EXIF; * = outside the -2 to 13 range of the synthetic training scenes")
    print(f"median over {len(r)} files: histogram {np.median(r[:, 0]):.2f}, grey world {np.median(r[:, 1]):.2f}, with prior {np.median(r[:, 2]):.2f} degrees from the camera's white")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Cube++ evaluation of an illuminant model (gate: median angular error <= 3 degrees, ``docs/13``).

STATUS: BLOCKED in the environment that wrote this (``zenodo.org`` is unreachable, ``tools/fetch/cubepp.sh`` could not
download the dataset). The script is complete against the layout the dataset paper describes and is tested only on a mock
dataset made from synthetic scenes (``test_illum.py``); it has **not** been run on real Cube++ data, and the layout and
file names are assumptions to check on first use.

Expected layout under ``testdata/cubepp/`` (``tools/fetch/cubepp.sh``):

* ``PNG/<name>.png``: linear 16-bit RGB images, black level subtracted;
* ``gt.csv``: one row per image with the ground-truth illuminant, columns ``image`` and ``mean_r, mean_g, mean_b``
  (any header whose lower-cased names end in ``_r``, ``_g``, ``_b`` is accepted; the first three numeric columns are used
  otherwise), ``image`` with or without the ``.PNG`` suffix;
* a list of the test images (``--split test.txt``, one name per line) so that nothing in the training split is scored.

**Conversion per profile.** Cube++ was shot with Canon bodies, not Sony ones. The images and the ground truth are mapped
into the camera RGB of the target profile with the same 3 x 3: ``camera_to_xyz`` of the source body (``--src-matrix``, JSON
``{"camera_to_xyz": [[...], ...]}`` from the dataset's documentation or the body's colour matrix, D65) followed by
``xyz_to_camera`` of the profile, a least-squares fit of the profile's raw response (:mod:`training.illum.spectra` curves in the
published raw scale) to CIE XYZ over the 190 rawtoaces reflectances under D65. This is a Luther-type approximation (it
ignores metameric differences between the two sensors), which is why this evaluation is a gate on the estimator's
behaviour on real image statistics and not a measurement of a Sony body.

Each image is box-averaged to about 1/8 of its size (a preview), pixels at or above 0.98 of the white level in any converted
channel are masked, and the model runs without an as-shot value (Cube++ has none): the histogram-only estimate.

    python3 -m training.illum.eval_cubepp --model data/models/illum/sony-ilce-7rm4.json --src-matrix canon.json \
        --split testdata/cubepp/test.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from training.alexa import ssf_core as s
from training.common import paths
from training.illum import baselines as bl
from training.illum import evaluate as ev
from training.illum import model as md
from training.illum import spectra as sp

GATE_MEDIAN_DEG = 3.0
PREVIEW_FACTOR = 8


def read_png16(path: Path) -> np.ndarray:
    """A linear PNG as float64 RGB in [0, 1] (16-bit expected; 8-bit is scaled accordingly)."""
    try:
        import cv2  # type: ignore

        a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if a is None:
            raise OSError(f"cannot read {path}")
        a = a[..., ::-1]
    except ImportError:
        try:
            import imageio.v3 as iio  # type: ignore

            a = iio.imread(path)
        except ImportError:
            import png  # type: ignore  (pypng)

            w, h, rows, info = png.Reader(filename=str(path)).read()
            a = np.vstack([np.asarray(r, dtype=np.uint16 if info["bitdepth"] == 16 else np.uint8) for r in rows])
            a = a.reshape(h, w, info["planes"])
    a = np.asarray(a)
    scale = 65535.0 if a.dtype == np.uint16 else 255.0
    return a[..., :3].astype(np.float64) / scale


def box_downsample(img: np.ndarray, factor: int) -> np.ndarray:
    h, w, c = img.shape
    h2, w2 = h // factor, w // factor
    return img[: h2 * factor, : w2 * factor].reshape(h2, factor, w2, factor, c).mean(axis=(1, 3))


def read_ground_truth(path: Path) -> dict[str, np.ndarray]:
    """``{image name without suffix: (r, g, b)}`` from a Cube++-style ``gt.csv``."""
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    header = [h.strip().lower() for h in rows[0]]
    out = {}
    cols = []
    for suffix in ("_r", "_g", "_b"):
        cand = [i for i, h in enumerate(header) if h.endswith(suffix)]
        cand.sort(key=lambda i: "mean" not in header[i])
        if not cand:
            cols = [1, 2, 3]
            break
        cols.append(cand[0])
    for r in rows[1:]:
        if not r:
            continue
        name = Path(r[0].strip()).stem
        out[name] = np.array([float(r[c]) for c in cols])
    return out


def xyz_to_camera(profile_id: str) -> np.ndarray:
    """3 x 3 from CIE XYZ (D65, Y = 1 for the white) to the profile's raw camera RGB, least squares over 190 reflectances."""
    cam = sp.camera(profile_id)
    from training.sensor_map import fit

    d65 = s.named_illuminant("D65")
    refl = fit.training_reflectances()  # (81, 190)
    spd = refl * d65[:, None]
    xyz = spd.T @ s.CMF
    xyz /= s.CMF[:, 1] @ d65
    rgb = spd.T @ cam.curves
    m, *_ = np.linalg.lstsq(xyz, rgb, rcond=None)
    return m.T


def convert(img: np.ndarray, truth: np.ndarray, camera_to_xyz: np.ndarray, to_camera: np.ndarray):
    """Image and ground-truth illuminant from the source camera RGB to the profile's camera RGB."""
    t = to_camera @ camera_to_xyz
    return img @ t.T, t @ truth


def evaluate(model: md.IllumModel, root: Path, camera_to_xyz: np.ndarray, split: list[str] | None, limit: int | None = None) -> dict:
    gt = read_ground_truth(root / "gt.csv")
    names = sorted(gt) if split is None else [Path(n.strip()).stem for n in split if n.strip()]
    if limit:
        names = names[:limit]
    to_cam = xyz_to_camera(model.profile_id)
    errs, used = [], []
    for name in names:
        path = next((p for p in (root / "PNG" / f"{name}.png", root / "PNG" / f"{name}.PNG") if p.exists()), None)
        if path is None or name not in gt:
            continue
        img, truth = convert(read_png16(path), gt[name], camera_to_xyz, to_cam)
        prev = box_downsample(img, PREVIEW_FACTOR)
        mask = (prev >= 0.98).any(axis=-1).astype(np.uint8)
        out = model.estimate(prev.astype(np.float32), mask, None, None)
        w_true = truth / truth[1]
        errs.append(float(bl.angular_error_deg(np.array([np.exp(-out["uv"][0]), 1.0, np.exp(-out["uv"][1])]), w_true)))
        used.append(name)
    if not errs:
        raise SystemExit("no image of the split was found under " + str(root))
    e = np.array(errs)
    res = ev.summarize(e)
    res["gate_median_deg"] = GATE_MEDIAN_DEG
    res["pass"] = bool(res["median"] <= GATE_MEDIAN_DEG)
    res["images"] = used
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=paths.TESTDATA / "cubepp")
    ap.add_argument("--src-matrix", type=Path, required=True, help='JSON {"camera_to_xyz": 3x3} of the source body')
    ap.add_argument("--split", type=Path, help="text file with the names of the test images")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args(argv)
    if not (args.root / "gt.csv").exists():
        print(f"BLOCKED: {args.root / 'gt.csv'} not found; run tools/fetch/cubepp.sh on a machine that can reach zenodo.org", file=sys.stderr)
        return 3
    m = md.IllumModel.from_json(json.loads(args.model.read_text()))
    mat = np.array(json.loads(args.src_matrix.read_text())["camera_to_xyz"], dtype=np.float64)
    split = args.split.read_text().splitlines() if args.split else None
    res = evaluate(m, args.root, mat, split, args.limit)
    res.pop("images")
    print(json.dumps(res, indent=1))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

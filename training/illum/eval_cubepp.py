"""Cube++ evaluation of an illuminant model (gate: median angular error <= 3 degrees, ``docs/13``).

STATUS: runs on GitHub Actions (``.github/workflows/cubepp.yml``), whose runners reach ``zenodo.org``; the agents' cloud
containers do not. It evaluates the SimpleCube++ test split (``SimpleCube++/test``: 648 x 432 PNGs, about a fifth of
the 2234 single-light images), which is the size of the app's 1/8 preview of the cameras' 5184 x 3456 frame.

Expected layout under ``testdata/cubepp/`` (``tools/fetch/cubepp.sh``):

* ``PNG/<name>.png``: linear 16-bit RGB images, black level subtracted;
* ``gt.csv``: one row per image with the ground-truth illuminant, columns ``image`` and ``mean_r, mean_g, mean_b``
  (any header whose lower-cased names end in ``_r``, ``_g``, ``_b`` is accepted; the first three numeric columns are used
  otherwise), ``image`` with or without the ``.PNG`` suffix;
* a list of the test images (``--split test.txt``, one name per line) so that nothing in the training split is scored.

**Conversion per profile.** Cube++ was shot with Canon bodies (EOS 550D and 600D, one 18 MP sensor), not Sony ones.
The images and the ground truth are mapped into the camera RGB of the target profile with one 3 x 3. With
``--src-camera Canon_EOS_600D`` it is a least-squares fit from the source body's raw response to the profile's, over the
190 rawtoaces reflectances under blackbody and daylight lights from 2500 to 7500 K, both bodies' curves from
rawtoaces-data (Apache-2.0) in their published raw scale (:func:`camera_to_camera`). With ``--src-matrix`` it is the
source body's ``camera_to_xyz`` followed by the profile's ``xyz_to_camera``. Either is a Luther-type approximation (it
ignores metameric differences between the two sensors), which is why this evaluation is a gate on the estimator's
behaviour on real image statistics and not a measurement of a Sony body.

SimpleCube++ keeps the black level (about 2048, ``--black-level``) and each image saturates somewhere below 16384, so
pixels within 2 % of the image's own maximum are masked when that maximum is near the 14-bit white; the SpyderCube is
already cropped out (a zeroed rectangle at the bottom right), and pixels at or below the black level are masked too.

Each image is box-averaged by ``--preview-factor`` (1 for SimpleCube++, already preview-sized; 8 for full-size PNGs), a
preview pixel is masked if any pixel under it is, and the model runs without an as-shot value or a brightness (Cube++
carries neither in SimpleCube++): the histogram-only estimate, stricter than the app's. With ``--properties`` (the
dataset's ``auxiliary/test_properties.csv``) the errors are also broken down by its manual labels (place, daytime,
illumination, shadows). The four classical estimators of :mod:`training.illum.baselines` run on the same previews and
masks for scale.

    python3 -m training.illum.eval_cubepp --model data/models/illum/sony-ilce-7rm4.json \
        --root testdata/cubepp/SimpleCube++/test --src-camera Canon_EOS_600D --black-level 2048 --preview-factor 1 \
        --properties testdata/cubepp/SimpleCube++/auxiliary/test_properties.csv
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
            import png  # type: ignore  (pypng)

            w, h, rows, info = png.Reader(filename=str(path)).read()
            a = np.vstack([np.asarray(r, dtype=np.uint16 if info["bitdepth"] == 16 else np.uint8) for r in rows])
            a = a.reshape(h, w, info["planes"])
        except ImportError:
            import imageio.v3 as iio  # type: ignore

            a = iio.imread(path)  # Pillow underneath: check it kept 16 bits
            if a.dtype != np.uint16:
                raise OSError(f"{path}: read as {a.dtype}; install opencv-python-headless or pypng for 16-bit PNGs")
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


#: Lights of the camera-to-camera fit: blackbody below 4000 K, CIE daylight from 4000 K (``ssf_core.illuminant``).
FIT_CCTS = (2500, 3000, 3500, 4000, 4500, 5000, 5500, 6500, 7500)
#: SimpleCube++'s 14-bit white; an image whose maximum is above ``CLIP_SHARE`` of it is taken to clip there.
WHITE_14BIT = 16383
CLIP_SHARE = 0.85


def source_curves(name: str) -> np.ndarray:
    """A rawtoaces-data body's curves (81, 3) in their published raw scale, e.g. ``Canon_EOS_600D``."""
    from training.sensor_map import fit

    return fit.load_curves_raw(paths.CAMERA_CURVES_DIR / f"{name}_380_780_5.json")


def camera_to_camera(src: np.ndarray, profile_id: str) -> np.ndarray:
    """3 x 3 from the source curves' raw RGB to the profile's, least squares over the 190 reflectances under
    :data:`FIT_CCTS` (each light scaled to the same green response of a perfect white, so no light dominates)."""
    from training.sensor_map import fit

    dst = sp.camera(profile_id).curves
    refl = fit.training_reflectances()  # (81, 190)
    a, b = [], []
    for cct in FIT_CCTS:
        e = s.illuminant(cct)
        k = 1.0 / float(dst[:, 1] @ e)
        spd = refl * (e * k)[:, None]
        a.append(spd.T @ src)
        b.append(spd.T @ dst)
    m, *_ = np.linalg.lstsq(np.vstack(a), np.vstack(b), rcond=None)
    return m.T


def linearize(raw: np.ndarray, black: float, white: float) -> np.ndarray:
    """Raw values to [0, 1] above the black level (negative values clipped to 0)."""
    return np.clip((raw - black) / (white - black), 0.0, None)


def invalid_pixels(raw: np.ndarray, black: float, white: float) -> np.ndarray:
    """Pixels to leave out: any channel near the image's clip (when the image clips), or every channel at or below the
    black level (the cropped SpyderCube rectangle, dead pixels)."""
    top = float(raw.max())
    bad = (raw <= black).all(axis=-1)
    if top - black >= CLIP_SHARE * (white - black):
        bad |= (raw >= black + 0.98 * (top - black)).any(axis=-1)
    return bad


def read_properties(path: Path) -> dict[str, dict[str, str]]:
    """``{image name: {label: value}}`` from SimpleCube++'s ``{test/train}_properties.csv``."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return {Path(r["image"].strip()).stem: {k: (v or "").strip() for k, v in r.items() if k != "image"} for r in rows}


def convert(img: np.ndarray, truth: np.ndarray, camera_to_xyz: np.ndarray, to_camera: np.ndarray):
    """Image and ground-truth illuminant from the source camera RGB to the profile's camera RGB."""
    t = to_camera @ camera_to_xyz
    return img @ t.T, t @ truth


def evaluate(model: md.IllumModel, root: Path, to_profile: np.ndarray, split: list[str] | None,
             limit: int | None = None, black: float = 0.0, white: float | None = None, factor: int = PREVIEW_FACTOR,
             properties: dict[str, dict[str, str]] | None = None) -> dict:
    """Angular errors of the histogram-only estimate over the images of ``root`` (``gt.csv`` and ``PNG/``), with
    ``to_profile`` mapping the source camera RGB to the profile's. ``white`` defaults to the PNG's full scale."""
    gt = read_ground_truth(root / "gt.csv")
    names = sorted(gt) if split is None else [Path(n.strip()).stem for n in split if n.strip()]
    if limit:
        names = names[:limit]
    errs, used = [], []
    base = {k: [] for k in bl.BASELINES}
    for name in names:
        path = next((p for p in (root / "PNG" / f"{name}.png", root / "PNG" / f"{name}.PNG") if p.exists()), None)
        if path is None or name not in gt:
            continue
        raw = read_png16(path) * 65535.0
        top = white if white is not None else 65535.0
        bad = invalid_pixels(raw, black, top)
        img = linearize(raw, black, top) @ to_profile.T
        truth = to_profile @ gt[name]
        if factor > 1:
            prev = box_downsample(img, factor)
            mask = box_downsample(bad[..., None].astype(np.float64), factor)[..., 0] > 0
        else:
            prev, mask = img, bad
        out = model.estimate(prev.astype(np.float32), mask.astype(np.uint8), None, None)
        w_true = truth / truth[1]
        errs.append(float(bl.angular_error_deg(np.array([np.exp(-out["uv"][0]), 1.0, np.exp(-out["uv"][1])]), w_true)))
        for k, f in bl.BASELINES.items():
            base[k].append(float(bl.angular_error_deg(f(prev, mask.astype(np.uint8)), w_true)))
        used.append(name)
    if not errs:
        raise SystemExit("no image of the split was found under " + str(root))
    e = np.array(errs)
    res = ev.summarize(e)
    res["gate_median_deg"] = GATE_MEDIAN_DEG
    res["pass"] = bool(res["median"] <= GATE_MEDIAN_DEG)
    # The classical estimators on the same previews and masks, for scale (NaN where one found no pixel).
    res["baselines"] = {k: ev.summarize(np.array(v)[np.isfinite(v)]) for k, v in base.items()}
    if properties:
        labels = sorted({k for n in used for k in properties.get(n, {}) if k not in ("ds_version",)})
        res["by_label"] = {}
        for label in labels:
            groups = np.array([properties.get(n, {}).get(label, "") or "?" for n in used])
            if 1 < len(set(groups)) <= 8:
                res["by_label"][label] = ev.by_group(e, groups)
    res["images"] = used
    res["errors"] = [round(x, 3) for x in errs]
    return res


def white_check(src: np.ndarray, gt: dict[str, np.ndarray], properties: dict[str, dict[str, str]]) -> dict:
    """Is the source curves' raw scale the camera's? The median ground truth of the outdoor daylight photos against the
    curves' white under D55 to D65, as red/green and blue/green ratios (a few per cent apart is what the Sony curves
    show against their cameras, ``docs/03``)."""
    sel = [n for n, p in properties.items() if n in gt and p.get("place") == "outdoor" and p.get("daytime") == "day"]
    if not sel:
        return {}
    w = np.array([gt[n] / gt[n][1] for n in sel])
    med = np.median(w, axis=0)
    cur = {}
    for cct in (5500, 6500):
        r = src.T @ s.illuminant(cct)
        cur[f"D{cct // 100}"] = [round(float(r[0] / r[1]), 4), round(float(r[2] / r[1]), 4)]
    return {"photos": len(sel), "ground_truth_rg_bg": [round(float(med[0]), 4), round(float(med[2]), 4)], "curves": cur}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=paths.TESTDATA / "cubepp")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--src-camera", help="rawtoaces-data body of the photos, e.g. Canon_EOS_600D")
    src.add_argument("--src-matrix", type=Path, help='JSON {"camera_to_xyz": 3x3} of the source body')
    ap.add_argument("--split", type=Path, help="text file with the names of the test images")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--black-level", type=float, default=0.0)
    ap.add_argument("--white-level", type=float,
                    help="defaults to 16383 when a black level is given, else the PNG's full scale")
    ap.add_argument("--preview-factor", type=int, default=PREVIEW_FACTOR)
    ap.add_argument("--properties", type=Path, help="the dataset's {test/train}_properties.csv")
    ap.add_argument("--out", type=Path, help="also write the full result (per-image errors included) as JSON")
    args = ap.parse_args(argv)
    if not (args.root / "gt.csv").exists():
        print(f"BLOCKED: {args.root / 'gt.csv'} not found; run tools/fetch/cubepp.sh on a machine that can reach zenodo.org", file=sys.stderr)
        return 3
    m = md.IllumModel.from_json(json.loads(args.model.read_text()))
    if args.src_camera:
        curves = source_curves(args.src_camera)
        to_profile = camera_to_camera(curves, m.profile_id)
    else:
        mat = np.array(json.loads(args.src_matrix.read_text())["camera_to_xyz"], dtype=np.float64)
        to_profile = xyz_to_camera(m.profile_id) @ mat
    white = args.white_level if args.white_level is not None else (WHITE_14BIT if args.black_level else None)
    props = read_properties(args.properties) if args.properties else None
    split = args.split.read_text().splitlines() if args.split else None
    res = evaluate(m, args.root, to_profile, split, args.limit, args.black_level, white, args.preview_factor, props)
    res["profile"] = m.profile_id
    if args.src_camera and props:
        res["white_check"] = white_check(curves, read_ground_truth(args.root / "gt.csv"), props)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=1))
    res.pop("images")
    res.pop("errors")
    print(json.dumps(res, indent=1))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

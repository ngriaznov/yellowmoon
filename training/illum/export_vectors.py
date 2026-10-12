"""Test vectors for the Rust port (``crates/ym-color/tests/data/illum/``).

    python3 -m training.illum.export_vectors --model data/models/illum/sony-ilce-7rm4.json

Writes

* ``test_model.json``: a small model (16 x 16 bins, random weights, two histogram channels) in the shipped format, so that
  every stage can be compared with all of its numbers;
* ``vectors.json``: previews (float32 RGB and clip mask, base64) and the reference outputs of every stage for the test model:
  histograms, valid fractions, filter scores, the histogram-only and the fused Gaussian, the class features and
  probabilities, confidence, ``w_scene``;
* ``shipped_vectors.json``: previews and the final outputs for a shipped model (``--model``), with that model's SHA-256 so a
  retrained model makes the test fail until the vectors are regenerated;
* ``all_profiles_vectors.json``: the same previews, scaled per channel to look like the other bodies' raw data, and the final
  outputs of **every** shipped model (``data/models/illum/*.json``), each with its SHA-256, so that the Rust loader and
  estimator are checked against the reference for all 11 histogram windows, not only the A7R IV's.

Previews are 36 x 24 crops of synthetic scenes, plus hand-made edge cases (everything clipped, zero and negative pixels,
a NaN, a flat grey image, no as-shot value, no brightness value).
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np

from training.common import jsonfmt, paths
from training.illum import dataset as dsm
from training.illum import features as ft
from training.illum import model as md
from training.illum import spectra as sp

OUT = paths.REPO_ROOT / "crates" / "ym-color" / "tests" / "data" / "illum"
STAND_IN = "sony-ilce-7rm4"


def make_test_model(seed: int = 20261012, n_bins: int = 16, bin_size: float = 0.25, hidden: int = 6) -> md.IllumModel:
    """A small model with random weights (not trained; only for parity tests)."""
    rng = np.random.default_rng(seed)
    spec = ft.HistSpec(n_bins=n_bins, bin_size=bin_size, lo_u=-1.25, lo_v=-1.0, min_intensity=1 / 1024, min_edge=1 / 4096,
                       channels=("intensity", "edge"), max_samples=200)  # 200: the 36 x 24 previews are subsampled with stride 2
    f32 = lambda a: np.asarray(a, dtype=np.float32).astype(np.float64)  # noqa: E731
    filt = f32(rng.normal(0.0, 6.0, size=(2, n_bins, n_bins)))
    bias = f32(rng.normal(0.0, 1.0, size=(n_bins, n_bins)))
    k = md.n_features(2)
    head = md.Head(f32(rng.normal(0, 0.5, size=(hidden, k))), f32(rng.normal(0, 0.2, size=hidden)),
                   f32(rng.normal(0, 0.8, size=(6, hidden))), f32(rng.normal(0, 0.2, size=6)),
                   f32(rng.normal(0, 1.0, size=k)), f32(rng.uniform(0.5, 2.0, size=k)), 1.3)
    return md.IllumModel(profile_id="test-model", version="0.0.1", spec=spec, filt=filt, bias=bias, pad=0.004, head=head,
                         conf=md.ConfHead(f32(rng.normal(0, 0.5, size=k + md.N_CONF_EXTRA + 1)), f32(rng.normal(0, 1.0, size=k + md.N_CONF_EXTRA)),
                                          f32(rng.uniform(0.5, 2.0, size=k + md.N_CONF_EXTRA))),
                         default_uv=np.array([1.0, 0.4]), bv_default=6.0,
                         class_prior=np.array([0.2, 0.15, 0.15, 0.15, 0.2, 0.15]), meta={"note": "random weights for parity tests"})


def _b64(a: np.ndarray, dtype: str) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode("ascii")


def _crop(img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.ascontiguousarray(img[8:32, 30:66]), np.ascontiguousarray(mask[8:32, 30:66])


def previews(ds: dsm.Dataset, n: int) -> list[dict]:
    """``n`` crops of the dataset's scenes plus the hand-made edge cases."""
    cases = []
    for i in range(n):
        img, mask = _crop(ds.imgs[i], ds.masks[i])
        cases.append({"name": f"scene-{i}", "img": img, "mask": mask,
                      "as_shot_uv": None if np.isnan(ds.as_shot_uv[i, 0]) else [float(x) for x in ds.as_shot_uv[i]],
                      "bv": float(ds.bv[i])})
    base_img, base_mask = _crop(ds.imgs[0], ds.masks[0])
    cases.append({"name": "no-as-shot", "img": base_img, "mask": base_mask, "as_shot_uv": None, "bv": 5.0})
    cases.append({"name": "no-bv", "img": base_img, "mask": base_mask, "as_shot_uv": [0.9, 0.5], "bv": None})
    cases.append({"name": "no-mask", "img": base_img, "mask": None, "as_shot_uv": [0.9, 0.5], "bv": 7.5})
    grey = np.full((24, 36, 3), 0.1, dtype=np.float32) * np.array([0.4, 1.0, 0.7], dtype=np.float32)
    cases.append({"name": "flat-white-under-d55", "img": grey, "mask": np.zeros((24, 36), np.uint8), "as_shot_uv": [1.0, 0.4], "bv": 9.0})
    odd = base_img.copy()
    odd[0, 0] = [0.0, 0.2, 0.1]
    odd[0, 1] = [-0.01, 0.2, 0.1]
    odd[1, 0] = [np.nan, 0.2, 0.1]
    odd[1, 1] = [np.inf, 0.2, 0.1]
    cases.append({"name": "zero-negative-nan-inf-pixels", "img": odd, "mask": base_mask, "as_shot_uv": None, "bv": None})
    cases.append({"name": "all-clipped-with-as-shot", "img": base_img, "mask": np.ones((24, 36), np.uint8), "as_shot_uv": [1.0, 0.4], "bv": 5.0})
    cases.append({"name": "all-clipped-no-as-shot", "img": base_img, "mask": np.ones((24, 36), np.uint8), "as_shot_uv": None, "bv": None})
    dark = np.full((24, 36, 3), 1e-4, dtype=np.float32)
    cases.append({"name": "too-dark-everywhere", "img": dark, "mask": np.zeros((24, 36), np.uint8), "as_shot_uv": None, "bv": 1.0})
    return cases


def _arr(a) -> list:
    return np.asarray(a, dtype=np.float64).reshape(-1).tolist()


def run_case(m: md.IllumModel, c: dict, full: bool) -> dict:
    img = c["img"].astype(np.float32)
    mask = c["mask"]
    a = None if c["as_shot_uv"] is None else np.array(c["as_shot_uv"], dtype=np.float64)
    bv = c["bv"]
    out = m.estimate(img, mask, a, None if bv is None else np.float64(bv))
    exp = {"fallback": bool(out["fallback"]), "uv": _arr(out["uv"]), "cov": _arr(out["cov"]), "probs": _arr(out["probs"]),
           "confidence": float(out["confidence"]), "w_scene": [float(np.exp(-out["uv"][0])), 1.0, float(np.exp(-out["uv"][1]))]}
    if not out["fallback"]:
        exp["uv_obs"] = _arr(out["uv_obs"])
        st = out["stats"]
        exp["valid_fraction"] = _arr(st.valid_fraction)
        exp["in_range"] = _arr(st.in_range)
        exp["mass"] = _arr(st.mass)
        hist = out["hist"]
        exp["hist_sum"] = _arr(hist.sum(axis=(-2, -1)))
        exp["hist_sq_sum"] = _arr((hist**2).sum(axis=(-2, -1)))
        if full:
            exp["cov_obs"] = _arr(out["cov_obs"])
            exp["hist"] = _arr(hist)
            exp["scores"] = _arr(out["scores"])
            exp["features"] = _arr(out["features"])
    return exp


def encode(c: dict, expected: dict) -> dict:
    h, w, _ = c["img"].shape
    return {"name": c["name"], "width": w, "height": h, "rgb_f32_b64": _b64(c["img"], "<f4"),
            "clip_mask_u8_b64": None if c["mask"] is None else _b64(c["mask"], "u1"),
            "as_shot_uv": c["as_shot_uv"], "bv": c["bv"], "expected": expected}


#: The cases of ``all_profiles_vectors.json``: (preview index, use the as-shot value, use Bv, mask everything).
PROFILE_CASES = ((0, True, True, False), (0, False, True, False), (0, True, False, False), (1, True, True, False),
                 (1, False, False, False), (2, True, True, False), (2, False, True, False), (0, True, True, True))


def channel_gain(profile_id: str) -> np.ndarray:
    """float32 per-channel factors that make a preview rendered for the A7R IV look like raw data of ``profile_id``
    (the ratio of the two cameras' D55 whites): every window of the 11 models then sees its own kind of light."""
    ref = sp.camera(STAND_IN).white(sp.daylight_at(5500.0))
    own = sp.camera(profile_id).white(sp.daylight_at(5500.0))
    return (own / ref).astype(np.float32)


def profile_vectors(models_dir: Path, ds: dsm.Dataset) -> dict:
    """Previews (once) and the reference outputs of every shipped model for :data:`PROFILE_CASES`."""
    scenes = [c for c in previews(ds, 3) if c["name"].startswith("scene-")]
    out_previews = [{"name": c["name"], "width": c["img"].shape[1], "height": c["img"].shape[0],
                     "rgb_f32_b64": _b64(c["img"], "<f4"), "clip_mask_u8_b64": _b64(c["mask"], "u1"),
                     "as_shot_uv": c["as_shot_uv"], "bv": c["bv"]} for c in scenes]
    profiles = {}
    for pid in sp.profile_ids():
        model = md.IllumModel.from_json(json.loads((models_dir / f"{pid}.json").read_text()))
        gain = channel_gain(pid)
        cases = []
        for idx, use_a, use_bv, all_masked in PROFILE_CASES:
            c = scenes[idx]
            a = c["as_shot_uv"] if use_a else None  # a scene that has no as-shot value keeps it missing
            mask = np.ones_like(c["mask"]) if all_masked else c["mask"]
            case = {"img": (c["img"].astype(np.float32) * gain).astype(np.float32), "mask": mask, "as_shot_uv": a,
                    "bv": c["bv"] if use_bv else None}
            exp = run_case(model, case, False)
            exp.update({"preview": idx, "use_as_shot": use_a, "use_bv": use_bv, "all_masked": all_masked})
            cases.append(exp)
        profiles[pid] = {"model_sha256": model.payload_sha256(), "channel_gain": [float(g) for g in gain], "cases": cases}
    return {"previews": out_previews, "profiles": profiles}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=paths.REPO_ROOT / "data" / "models" / "illum" / "sony-ilce-7rm4.json")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--n", type=int, default=4, help="scene crops per vector file")
    args = ap.parse_args(argv)
    shipped = md.IllumModel.from_json(json.loads(args.model.read_text()))
    ds = dsm.build(shipped.profile_id, "test", 200, shipped.spec, keep_images=True, procs=1, cache=False)
    args.out.mkdir(parents=True, exist_ok=True)

    tm = make_test_model()
    tm = tm.rounded()
    (args.out / "test_model.json").write_text(jsonfmt.dumps(tm.to_json()))
    # the test model has a different window (16 x 16 bins): rebuild its histograms from the same previews
    cases = previews(ds, args.n)
    vec = [encode(c, run_case(tm, c, i < 5)) for i, c in enumerate(cases)]
    (args.out / "vectors.json").write_text(json.dumps({"model_sha256": tm.payload_sha256(), "cases": vec}, separators=(",", ":")) + "\n")

    keep = {"no-as-shot", "no-bv", "flat-white-under-d55", "all-clipped-with-as-shot", "zero-negative-nan-inf-pixels"}
    cases = [c for c in previews(ds, args.n) if c["name"].startswith("scene-") or c["name"] in keep]
    vec = [encode(c, run_case(shipped, c, False)) for c in cases]
    (args.out / "shipped_vectors.json").write_text(json.dumps(
        {"profile_id": shipped.profile_id, "model_sha256": shipped.payload_sha256(), "cases": vec}, separators=(",", ":")) + "\n")
    (args.out / "all_profiles_vectors.json").write_text(
        json.dumps(profile_vectors(args.model.parent, ds), separators=(",", ":")) + "\n")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

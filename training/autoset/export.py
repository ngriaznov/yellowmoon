"""Test vectors and the test model for ``crates/ym-color/tests/data/autoset/`` (docs/18 section 6.4; ``tests/autoset.rs`` must reproduce them to 1e-9).

    python3 -m training.autoset.export --model data/models/autoset/autoset.json --vectors crates/ym-color/tests/data/autoset/vectors.json
    python3 -m training.autoset.export --test-model        # writes test_model.json and vectors.json in that folder

The vectors are computed in numpy float64 from the stored float32 weights, after the file is decoded exactly as the Rust reader
decodes it (base64 little-endian float32 to float64; the constants from the JSON numbers). The document::

    {"format": "yellowmoon-autoset-vectors", "format_version": 1, "model_file": "...", "model_version": "...",
     "model_sha256": "...", "constants": {"camera_gain_ev", "ev_limit", "contrast_limit", "prior_sigma_key_target",
     "prior_sigma_contrast", "logvar_clamp", "contrast_step"},
     "cases": [{"name": "...", "input": [39 raw features], "key": m,
                "members": [[key_mean, key_logvar, contrast_mean, contrast_logvar] per member, raw outputs],
                "combined": {"key_mean", "key_var", "contrast_mean", "contrast_var"},
                "apply": {"ev_raw", "ev", "contrast_raw", "contrast", "ev_confidence", "contrast_confidence"}}, ...]}

``key_var`` and ``contrast_var`` are the mixture variances (exp of the clamped logvars); ``apply`` is section 4.4 without the
highlight guard: ``ev`` is clamped to +-3, ``contrast`` is ``contrast_raw`` rounded to 0.1 half away from zero. A case whose
contrast lies within 1e-6 of a rounding tie is skipped, so that a last-bit difference cannot flip a rounding in the port.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from training.autoset import dataset as ds
from training.autoset import model as md
from training.autoset import rows as rw
from training.autoset import synthetic
from training.autoset import train
from training.common import jsonfmt, paths

VECTOR_FORMAT = "yellowmoon-autoset-vectors"
VECTOR_FORMAT_VERSION = 1
VECTOR_SEED = 20261018
VECTOR_COUNT = 30
TEST_SEED = 20261018
TEST_SCENES = 30
TEST_STEPS = 50
TEST_VERSION = "0.0.0-test"
TIE_MARGIN = 1e-6
DATA_DIR = paths.REPO_ROOT / "crates" / "ym-color" / "tests" / "data" / "autoset"


def _tie_margin(x: float) -> float:
    """Distance of ``10 x`` from the nearest half-integer: the rounding to 0.1 is safe when it is clearly positive."""
    a = abs(x * 10.0)
    return abs((a - math.floor(a)) - 0.5)


def _case(model: md.AutosetModel, name: str, x: np.ndarray, key: float) -> dict[str, Any] | None:
    """One vector, or None when its contrast is too close to a rounding tie to be a fair test."""
    outputs = model.member_outputs(x[None, :])
    pred = md.combine(outputs)
    applied = model.apply(float(pred.key_mean[0]), float(pred.key_var[0]), float(pred.contrast_mean[0]),
                          float(pred.contrast_var[0]), key)
    if _tie_margin(applied.contrast_raw) < TIE_MARGIN:
        return None
    return {
        "name": name,
        "input": [float(v) for v in x],
        "key": float(key),
        "members": [[float(v) for v in outputs[i, 0]] for i in range(outputs.shape[0])],
        "combined": {"key_mean": float(pred.key_mean[0]), "key_var": float(pred.key_var[0]),
                     "contrast_mean": float(pred.contrast_mean[0]), "contrast_var": float(pred.contrast_var[0])},
        "apply": {"ev_raw": applied.ev_raw, "ev": applied.ev, "contrast_raw": applied.contrast_raw,
                  "contrast": applied.contrast, "ev_confidence": applied.ev_confidence,
                  "contrast_confidence": applied.contrast_confidence},
    }


def build_vectors(model: md.AutosetModel, *, model_file: str = "", seed: int = VECTOR_SEED,
                  count: int = VECTOR_COUNT) -> dict[str, Any]:
    """The vector document for ``model``: ``count`` cases, the last four being edge cases (zeros, large values, alternating)."""
    rng = np.random.default_rng(seed)
    edges = [
        ("edge-zeros", np.zeros(md.N_FEATURES), 0.0),
        ("edge-large-positive", np.full(md.N_FEATURES, 1000.0), 3.5),
        ("edge-large-negative", np.full(md.N_FEATURES, -1000.0), -6.0),
        ("edge-alternating", np.array([50.0 if i % 2 == 0 else -50.0 for i in range(md.N_FEATURES)]), 0.25),
    ]
    cases: list[dict[str, Any]] = []
    wanted = count - len(edges)
    attempts = 0
    while len(cases) < wanted:
        attempts += 1
        x = np.round(model.input_mean + model.input_std * rng.normal(0.0, 1.5, md.N_FEATURES), 6)
        key = round(float(rng.uniform(-4.0, 2.0)), 6)
        case = _case(model, f"seeded-{len(cases):02d}", x, key)
        if case is not None:
            cases.append(case)
        if attempts > 100 * count:
            raise RuntimeError("could not draw vectors clear of the contrast rounding ties")
    for name, x, key in edges:
        case = _case(model, name, x, key)
        if case is None:
            raise RuntimeError(f"edge case {name} lies on a contrast rounding tie")
        cases.append(case)
    return {
        "format": VECTOR_FORMAT,
        "format_version": VECTOR_FORMAT_VERSION,
        "model_file": model_file,
        "model_version": model.model_version,
        "model_sha256": model.payload_sha256(),
        "constants": {
            "camera_gain_ev": md.CAMERA_GAIN_EV,
            "ev_limit": md.EV_LIMIT,
            "contrast_limit": model.contrast_limit,
            "prior_sigma_key_target": model.prior_sigma_key,
            "prior_sigma_contrast": model.prior_sigma_contrast,
            "logvar_clamp": md.LOGVAR_LIMIT,
            "contrast_step": 0.1,
        },
        "cases": cases,
    }


def make_test_model() -> md.AutosetModel:
    """The test model: five members trained for :data:`TEST_STEPS` steps on synthetic rows, without early stopping."""
    read = rw.read_objects(synthetic.make_rows(TEST_SCENES, seed=TEST_SEED), label="synthetic")
    prep = ds.prepare(read)
    cfg = train.TrainConfig(steps=TEST_STEPS, holdout=0.0)
    model, _ = train.train_ensemble(prep, np.arange(prep.n), None, cfg, model_version=TEST_VERSION, log=lambda _: None)
    return model


def write_test_model(out_dir: Path = DATA_DIR) -> dict[str, Path]:
    """Writes ``test_model.json`` and ``vectors.json`` (for that model) into ``out_dir``; returns their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "test_model.json"
    vectors_path = out_dir / "vectors.json"
    model_path.write_text(jsonfmt.dumps(make_test_model().to_json()), encoding="utf-8")
    loaded = md.AutosetModel.from_json(json.loads(model_path.read_text(encoding="utf-8")))
    vectors_path.write_text(jsonfmt.dumps(build_vectors(loaded, model_file=model_path.name)), encoding="utf-8")
    return {"model": model_path, "vectors": vectors_path}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, help="a model file (docs/18 section 4.3); its vectors are written to --vectors")
    ap.add_argument("--vectors", type=Path, help="where to write the vectors of --model")
    ap.add_argument("--test-model", action="store_true", help="write the test model and its vectors to --out-dir")
    ap.add_argument("--out-dir", type=Path, default=DATA_DIR)
    args = ap.parse_args(argv)
    if args.test_model:
        paths_written = write_test_model(args.out_dir)
        for p in paths_written.values():
            print(f"wrote {p} ({p.stat().st_size} bytes)")
        return 0
    if args.model is None or args.vectors is None:
        ap.error("give --model and --vectors, or --test-model")
    model = md.AutosetModel.from_json(json.loads(args.model.read_text(encoding="utf-8")))
    args.vectors.parent.mkdir(parents=True, exist_ok=True)
    args.vectors.write_text(jsonfmt.dumps(build_vectors(model, model_file=args.model.name)), encoding="utf-8")
    print(f"wrote {args.vectors} for {args.model.name} (sha256 {model.payload_sha256()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

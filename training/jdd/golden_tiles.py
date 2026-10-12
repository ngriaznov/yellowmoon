"""Golden tiles for the Rust runtime parity tests (docs/07 "Parity and versioning").

    python3 -m training.jdd.golden_tiles --out crates/ym-ml/tests/data          # write the fixtures
    python3 -m training.jdd.golden_tiles --check crates/ym-ml/tests/data        # fail if they are stale

Fixtures (small, committed; no trained weights exist yet, so they use SEEDED RANDOM weights of two small
instances of the architecture, which exercise every code path of the CPU runtime):

* ``golden_small.safetensors`` - channels 8/16/24, 1x1 up and head convolutions (the starting layout);
* ``golden_k3.safetensors``   - channels 6/8/12, 3x3 up and head convolutions;
* ``golden_tiles.safetensors`` - 20 real generator tiles (noise, cRAW, clipping) of 4 to 48 half-res px
  with the PyTorch outputs: ``small.in.<i>`` / ``small.out.<i>`` / ``k3.in.<i>`` / ``k3.out.<i>`` (f32);
* ``golden_small.onnx``, ``golden_k3.onnx`` - the two models as ONNX graphs (opset 17) with dynamic height and
  width, so that ONNX Runtime can run every golden tile (the shipped export is static-shape: ``export.py``);
* ``golden.json`` - seeds, shapes and the measured PyTorch-vs-onnxruntime differences.

The script asserts PyTorch vs onnxruntime agree to 2e-3 on all 20 tiles (they agree to ~1e-6) before writing.
The Rust test ``crates/ym-ml/tests/parity.rs`` requires the CPU runtime to match these outputs to 2e-3.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import warnings
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from .data import SamplerConfig, SyntheticSource, generate_sample
from .export import make_random_model
from .model import JddConfig, JddNet
from .weights_io import save_weights, sha256_file

TOL = 2e-3
SMALL = {"cfg": JddConfig(channels=(8, 16, 24)), "seed": 1234, "file": "golden_small.safetensors", "key": "small"}
K3 = {"cfg": JddConfig(channels=(6, 8, 12), up_kernel=3, head_kernel=3), "seed": 99, "file": "golden_k3.safetensors", "key": "k3"}
# docs/07: "20 fixed inputs". (height, width, seed) in half-resolution px. The 32 x 32 tiles of the small model are
# the ones the static 32 x 32 ONNX graph can run; the 16-px tiles are mostly border (they test the padding paths of
# every layer), the 8 x 8 and 4 x 4 ones reach the smallest legal size (a 1 x 1 bottleneck), the others mix non-square shapes. The total stays under the 2 MB fixture budget.
TILES = {
    "small": [
        (32, 32, 101), (32, 32, 102), (32, 32, 106), (32, 32, 107), (32, 48, 103), (48, 32, 104),
        (16, 16, 108), (16, 16, 109), (8, 8, 110), (24, 40, 111), (40, 24, 112), (16, 48, 113),
    ],
    "k3": [
        (32, 32, 201), (40, 24, 202), (24, 40, 204), (16, 16, 205), (16, 16, 206), (4, 4, 207),
        (32, 48, 203), (24, 24, 208),
    ],
}
N_GOLDEN_TILES = sum(len(v) for v in TILES.values())  # 20


def make_tile(h: int, w: int, seed: int) -> np.ndarray:
    """A ``[7, h, w]`` input tile from the training generator (seeded), cropped to ``h x w`` half-res px."""
    side = max(h, w)
    cfg = SamplerConfig(patch_half=side)
    rng = np.random.default_rng([2026, seed])
    gt, _ = SyntheticSource(cfg.gt_size).gt(rng)
    return generate_sample(gt, rng, cfg).x[:, :h, :w].copy()


def build(out: Path) -> dict:
    """Write all fixtures into ``out`` and return the ``golden.json`` content."""
    out.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, torch.Tensor] = {}
    info: dict = {"tolerance": TOL, "n_tiles": N_GOLDEN_TILES, "models": {}}
    for spec in (SMALL, K3):
        model: JddNet = make_random_model(spec["seed"], spec["cfg"])
        path = out / spec["file"]
        sha = save_weights(model, path, model_version=f"golden-{spec['key']}")
        diffs = []
        shapes = []
        for i, (h, w, tseed) in enumerate(TILES[spec["key"]]):
            x = make_tile(h, w, tseed)
            with torch.no_grad():
                y = model(torch.from_numpy(x)[None])[0]
            tensors[f"{spec['key']}.in.{i}"] = torch.from_numpy(x).contiguous()
            tensors[f"{spec['key']}.out.{i}"] = y.contiguous().to(torch.float32)
            shapes.append([h, w])
            diffs.append(float(y.abs().max()))
        info["models"][spec["key"]] = {
            "seed": spec["seed"], "config": json.loads(spec["cfg"].to_json()), "weights": spec["file"],
            "sha256": sha, "tile_shapes_half_px": shapes, "max_abs_output": diffs,
        }
    # ONNX of both models (dynamic height and width: the Rust test builds one session per tile shape).
    import onnxruntime as ort

    info["onnx"] = {}
    worst = 0.0
    for spec in (SMALL, K3):
        key = spec["key"]
        model = make_random_model(spec["seed"], spec["cfg"])
        info["onnx"][key] = _export_dynamic(model, out / f"golden_{key}.onnx")
        sess = ort.InferenceSession(str(out / f"golden_{key}.onnx"), providers=["CPUExecutionProvider"])
        for i in range(len(TILES[key])):
            got = sess.run(None, {"x": tensors[f"{key}.in.{i}"].numpy()[None]})[0][0]
            worst = max(worst, float(np.abs(got - tensors[f"{key}.out.{i}"].numpy()).max()))
    if worst > TOL:
        raise AssertionError(f"PyTorch vs onnxruntime differ by {worst:.2e} > {TOL}")
    info["max_abs_diff_torch_vs_onnxruntime"] = worst
    save_file(tensors, str(out / "golden_tiles.safetensors"))
    info["tiles_sha256"] = sha256_file(out / "golden_tiles.safetensors")
    (out / "golden.json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
    return info


def _export_dynamic(model: JddNet, path: Path) -> dict:
    """ONNX (opset 17) with dynamic height and width, for parity runs on tiles of any valid shape."""
    dummy = torch.zeros(1, 7, 32, 32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model, (dummy,), str(path), opset_version=17, dynamo=False, input_names=["x"], output_names=["rgb"],
            dynamic_axes={"x": {2: "h", 3: "w"}, "rgb": {2: "H", 3: "W"}},
        )
    return {"file": path.name, "input_shape": [1, 7, "h", "w"], "sha256": sha256_file(path)}


def check(directory: Path) -> int:
    """Regenerate into a temp dir and compare with ``directory`` (values to 1e-4: summation order may differ)."""
    with tempfile.TemporaryDirectory() as td:
        build(Path(td))
        bad = []
        for name in ("golden_tiles.safetensors", SMALL["file"], K3["file"]):
            a, b = load_file(str(Path(td) / name)), load_file(str(directory / name))
            if a.keys() != b.keys():
                bad.append(f"{name}: tensor names differ")
                continue
            for k in a:
                if a[k].shape != b[k].shape or float((a[k] - b[k]).abs().max()) > 1e-4:
                    bad.append(f"{name}:{k}")
    if bad:
        print("stale fixtures:", ", ".join(bad[:8]))
        return 1
    print("golden fixtures are up to date")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--out", type=Path)
    g.add_argument("--check", type=Path)
    args = ap.parse_args(argv)
    if args.check:
        return check(args.check)
    info = build(args.out)
    total = sum(p.stat().st_size for p in args.out.iterdir() if p.is_file())
    print(f"wrote fixtures to {args.out} ({total / 1e6:.2f} MB); torch vs onnxruntime max diff "
          f"{info['max_abs_diff_torch_vs_onnxruntime']:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

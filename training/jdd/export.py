"""Export a `jdd` model: safetensors (with architecture header) and a static-shape ONNX file (docs/07).

    python3 -m training.jdd.export --weights runs/jdd-001/weights.safetensors --out runs/jdd-001/export
    python3 -m training.jdd.export --random 1234 --out testdata/ml/random-512     # seeded random weights (M0-08)

The ONNX graph takes ``x`` of shape ``[1, 7, core + 2*halo, core + 2*halo]`` (half-resolution pixels;
``core`` defaults to 512 and ``halo`` is the receptive-field halo of the network, 32 for the starting
layout) in the physical units of :mod:`training.jdd.model` and returns ``rgb`` ``[1, 3, 2*(core + 2*halo),
...]``. Opset 17, static shapes, only Conv / Mul / Sigmoid / Add / Concat / Slice / DepthToSpace (or, with
``--decompose-pixel-shuffle``, Reshape / Transpose). ConvTranspose and Resize never appear; the script
fails if they do.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import json
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .model import IN_CHANNELS, JddConfig, JddNet, check_input_shape
from .receptive_field import analyse
from .weights_io import load_weights, save_weights, sha256_file, write_json, write_safetensors

OPSET = 17
FORBIDDEN_OPS = {"ConvTranspose", "Resize", "Upsample"}


def random_init(model: JddNet, seed: int, gain: float = 0.8) -> JddNet:
    """He-normal random weights (activations keep O(1) variance) so random-weight golden tiles are informative."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            if not p.requires_grad:
                continue
            if p.dim() == 4:
                fan_in = p.shape[1] * p.shape[2] * p.shape[3]
                p.copy_(torch.randn(p.shape, generator=g) * gain * (2.0 / fan_in) ** 0.5)
            else:
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    return model.eval()


def make_random_model(seed: int, cfg: JddConfig | None = None) -> JddNet:
    """A seeded random-weight network of the given architecture."""
    cfg = cfg or JddConfig()
    torch.manual_seed(seed)
    return random_init(JddNet(cfg), seed)


@contextlib.contextmanager
def _decomposed_pixel_shuffle():
    orig = F.pixel_shuffle

    def shuffle(x: torch.Tensor, r: int) -> torch.Tensor:
        b, c, h, w = x.shape
        return x.reshape(b, c // (r * r), r, r, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // (r * r), h * r, w * r)

    F.pixel_shuffle = shuffle
    try:
        yield
    finally:
        F.pixel_shuffle = orig


def export_onnx(
    weights: str | Path | JddNet,
    out_dir: str | Path,
    core_half_px: int = 512,
    halo_half_px: int | None = None,
    *,
    check: bool = True,
    decompose_pixel_shuffle: bool = False,
    name: str = "jdd",
    model_version: str = "0",
    save_io: bool = False,
) -> dict:
    """Write ``<name>.safetensors`` and ``<name>.onnx`` into ``out_dir``; returns the report (also ``export.json``)."""
    model = load_weights(weights) if not isinstance(weights, JddNet) else weights.eval()
    rf = analyse(model)
    halo = rf.halo_half_px if halo_half_px is None else halo_half_px
    if halo < rf.reach_half_px:
        raise ValueError(f"halo {halo} half-res px is below the receptive-field reach {rf.reach_half_px}")
    size = core_half_px + 2 * halo
    check_input_shape(size, size)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    st_path = out_dir / f"{name}.safetensors"
    sha_st = save_weights(model, st_path, model_version=model_version)
    onnx_path = out_dir / f"{name}.onnx"
    g = torch.Generator().manual_seed(0)
    dummy = torch.rand(1, IN_CHANNELS, size, size, generator=g) * 0.5
    ctx = _decomposed_pixel_shuffle() if decompose_pixel_shuffle else contextlib.nullcontext()
    with ctx, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            torch.onnx.export(
                model, (dummy,), str(onnx_path), opset_version=OPSET, dynamo=False,
                input_names=["x"], output_names=["rgb"], do_constant_folding=True,
            )
        except Exception:  # legacy exporter removed in a future torch: use the torch.export one
            torch.onnx.export(
                model, (dummy,), str(onnx_path), opset_version=OPSET, dynamo=True,
                input_names=["x"], output_names=["rgb"],
            ).save(str(onnx_path))
    import onnx

    proto = onnx.load(str(onnx_path))
    onnx.checker.check_model(proto)
    ops = collections.Counter(n.op_type for n in proto.graph.node)
    bad = FORBIDDEN_OPS & set(ops)
    if bad:
        raise RuntimeError(f"forbidden ONNX ops in the exported graph: {sorted(bad)}")
    report: dict = {
        "name": name,
        "opset": OPSET,
        "input_shape": [1, IN_CHANNELS, size, size],
        "output_shape": [1, 3, 2 * size, 2 * size],
        "core_half_px": core_half_px,
        "halo_half_px": halo,
        "rf_half_px": rf.rf_half_px,
        "rf_full_px": rf.rf_full_px,
        "ops": dict(sorted(ops.items())),
        "safetensors_sha256": sha_st,
        "onnx_sha256": sha256_file(onnx_path),
    }
    if check:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        with torch.no_grad():
            want = model(dummy).numpy()
        got = sess.run(None, {"x": dummy.numpy()})[0]
        report["max_abs_diff_vs_torch"] = float(np.abs(got - want).max())
    if save_io:
        # One input tile and the PyTorch output for it: the benchmark (crates/ym-ml/examples/bench.rs) reports
        # the difference of every backend to this reference.
        with torch.no_grad():
            ref = model(dummy)
        write_safetensors({"x": dummy, "rgb": ref}, {"note": "PyTorch reference tile"}, out_dir / "io.safetensors")
        report["io_sha256"] = sha256_file(out_dir / "io.safetensors")
    write_json(out_dir / "export.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, help="weights.safetensors from a run")
    ap.add_argument("--random", type=int, default=None, metavar="SEED", help="seeded random weights instead of --weights")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--core", type=int, default=512, help="tile core, half-res px (default 512)")
    ap.add_argument("--halo", type=int, default=None, help="override the halo (half-res px)")
    ap.add_argument("--name", default="jdd")
    ap.add_argument("--version", default="0", help="model version string written to the safetensors header")
    ap.add_argument("--decompose-pixel-shuffle", action="store_true", help="Reshape/Transpose instead of DepthToSpace")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--save-io", action="store_true", help="also write io.safetensors (input tile + PyTorch output)")
    args = ap.parse_args(argv)
    if (args.weights is None) == (args.random is None):
        ap.error("give exactly one of --weights and --random")
    src = make_random_model(args.random) if args.random is not None else args.weights
    report = export_onnx(
        src, args.out, args.core, args.halo, check=not args.no_check,
        decompose_pixel_shuffle=args.decompose_pixel_shuffle, name=args.name, model_version=args.version,
        save_io=args.save_io or args.random is not None,
    )
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


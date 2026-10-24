"""Exports a trained BokehNet for the app: `bokehnet.safetensors` (f32, one file) and a parity fixture, for the Rust
runtime that comes later (`ym_ml`, like `crop_net.rs`).

Each batch norm is folded into the convolution before it (the network runs in eval mode, so a batch norm is a fixed
scale and shift per channel), and the tensors are named after the layers the runtime builds:

    stem.{weight,bias}                       3x3 convolution, stride 2, ReLU6
    blocks.<i>.expand.{weight,bias}          1x1 convolution, ReLU6 (absent when the block's expansion is 1)
    blocks.<i>.depthwise.{weight,bias}       3x3 depthwise convolution, stride 1 or 2, ReLU6
    blocks.<i>.project.{weight,bias}         1x1 convolution (no activation); plus the block's input when `residual`
    context.{weight,bias}                    1x1 convolution on the global average of the last stage, ReLU6, added to it
    decoder.<j>.skip|mix|dw|pw.{weight,bias} the decoder steps (`model.DecoderStep`), each followed by ReLU6
    head.{weight,bias}                       1x1 convolution to (mu, s), no activation

`<i>` counts the blocks over all stages from 0. The header's metadata (strings) is everything the runtime must check or
build from: `format`, `version`, `arch` (JSON: the stem width, the blocks with their channels, strides and residual
flags, the skip taps, the decoder widths), the upsampling rule, the padding rule, the encoding constants of `encode.py`
(`mid`, `k`, `stops`, `input_mean`, `input_std`, the luma weights, the median rule), and how to read the output
(`mu = log r in pixels`, `s = log beta`, `c = exp(-beta / conf_beta0)`, `b = r / half-diagonal`). The tensors are
written in sorted-name order by `safetensors`, and the metadata is sorted, so the same weights give the same bytes.

The fixture (`bokehnet_parity.json`) holds, for a few deterministic pictures, the encoded input's definition (integer
pattern, so the Rust test builds the same input exactly), the float32 PyTorch output of the *unfolded* network at
the output resolution and the folded network's, plus encode probes for the linear-input cases.

    python -m training.bokeh.export --model <run>/model.pt --out bokehnet.safetensors --fixture bokehnet_parity.json
    python -m training.bokeh.export --random --seed 1 --out ... --fixture ...   # seeded random weights, real architecture
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from training.bokeh import encode
from training.bokeh.model import ARCH, BokehNet, InvertedResidual

FORMAT = "ym-bokehnet"
VERSION = "1"
#: Sizes (height, width) of the fixture's pictures: all divisible by 32.
SIZES = ((64, 96), (96, 64), (32, 32))
LINEAR_SIZE = (64, 64)


def save_deterministic(tensors: dict[str, torch.Tensor], path: Path, meta: dict[str, str]) -> None:
    """Writes a safetensors file (8-byte little-endian header length, a JSON header padded with spaces to a multiple of
    8, the f32 data back to back) with the tensors and the metadata in sorted order. `safetensors.torch.save_file`
    does not give the same bytes twice (its header's metadata order varies between calls), so the file is written
    here; `safetensors` reads it as any other."""
    header: dict = {"__metadata__": dict(sorted(meta.items()))}
    blobs, pos = [], 0
    for name in sorted(tensors):
        arr = tensors[name].detach().cpu().contiguous().float().numpy().astype("<f4")
        data = arr.tobytes()
        header[name] = {"dtype": "F32", "shape": list(arr.shape), "data_offsets": [pos, pos + len(data)]}
        blobs.append(data)
        pos += len(data)
    raw = json.dumps(header, separators=(",", ":"), sort_keys=False).encode()
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + b"".join(blobs))


def fold(conv: torch.nn.Conv2d, bn: torch.nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    """The weight and bias of `bn(conv(x))` as one convolution."""
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    weight = conv.weight * scale.reshape(-1, 1, 1, 1)
    bias = bn.bias - bn.running_mean * scale
    if conv.bias is not None:
        bias = bias + conv.bias * scale
    return weight.detach().contiguous(), bias.detach().contiguous()


def blocks_of(model: BokehNet) -> list[InvertedResidual]:
    return [b for stage in model.stages for b in stage]


def tensors(model: BokehNet) -> dict[str, torch.Tensor]:
    """The folded, renamed tensors (module docstring)."""
    out: dict[str, torch.Tensor] = {}

    def put(name: str, seq) -> None:
        out[f"{name}.weight"], out[f"{name}.bias"] = fold(seq[0], seq[1])

    put("stem", model.stem)
    for i, block in enumerate(blocks_of(model)):
        names = (["expand"] if len(block.body) == 3 else []) + ["depthwise", "project"]
        for name, seq in zip(names, block.body):
            put(f"blocks.{i}.{name}", seq)
    put("context", model.context)
    for j, step in enumerate(model.steps):
        for name in ("skip", "mix", "dw", "pw"):
            put(f"decoder.{j}.{name}", getattr(step, name))
    out["head.weight"], out["head.bias"] = model.head.weight.detach(), model.head.bias.detach()
    return {k: v.float().contiguous() for k, v in out.items()}


def arch_json(model: BokehNet) -> str:
    a = model.arch
    blocks = []
    stride_of_stage = {i: s[3] for i, s in enumerate(a.stages)}
    for i, stage in enumerate(model.stages):
        for b, block in enumerate(stage):
            conv = block.body[0][0] if len(block.body) == 3 else None
            dw = block.body[-2][0]
            proj = block.body[-1][0]
            blocks.append({"stage": i, "expand": conv is not None, "cin": (conv.in_channels if conv else dw.in_channels),
                           "mid": dw.in_channels, "cout": proj.out_channels,
                           "stride": stride_of_stage[i] if b == 0 else 1, "residual": block.residual})
    return json.dumps({
        "stem": a.stem, "blocks": blocks, "taps": list(a.taps), "decoder": list(a.decoder),
        "skip_channels": model.skip_channels, "out_channels": a.out_channels,
    }, sort_keys=True)


def metadata(model: BokehNet, source: str) -> dict[str, str]:
    """The header's metadata (module docstring)."""
    return {
        "format": FORMAT, "version": VERSION, "arch": arch_json(model), "source": source,
        "activation": "relu6", "padding": "zero, 1 pixel for 3x3 convolutions",
        "upsample": "bilinear x2, align_corners=false (half-pixel centres, clamped at the edges)",
        "pad_multiple": str(encode.PAD_MULTIPLE), "output_stride": "2", "frame_long_px": str(encode.FRAME_LONG),
        "mid": repr(encode.MID), "k": repr(encode.K), "stops": repr(encode.STOPS),
        "input_mean": repr(encode.INPUT_MEAN), "input_std": repr(encode.INPUT_STD),
        "luma": json.dumps(list(encode.LUMA)), "median": "lower median of luma over the plane; max with 1e-3 * mean; at least 1e-12",
        "output": "channel 0 mu = log r (pixels of the 1024-px-long frame, floor 1), channel 1 s = log beta",
        "b": "b = exp(mu) / hd, hd = hypot(w, h) / 2 of the network's input in pixels",
        "conf": "c = exp(-exp(s) / conf_beta0)", "conf_beta0": repr(encode.CONF_BETA0),
        "b_min": repr(encode.B_MIN), "b_cap": repr(encode.B_CAP),
    }


def folded_forward(t: dict[str, torch.Tensor], e: torch.Tensor, arch: dict) -> torch.Tensor:
    """The exported network in plain torch ops (what the Rust runtime computes), on the encoded input `e`."""
    relu6 = lambda x: x.clamp(0, 6)  # noqa: E731
    x = relu6(F.conv2d((e - encode.INPUT_MEAN) / encode.INPUT_STD, t["stem.weight"], t["stem.bias"], stride=2, padding=1))
    skips = [x]
    blocks = arch["blocks"]
    last_of_stage = {b["stage"]: i for i, b in enumerate(blocks)}
    for i, b in enumerate(blocks):
        z = x
        if b["expand"]:
            z = relu6(F.conv2d(z, t[f"blocks.{i}.expand.weight"], t[f"blocks.{i}.expand.bias"]))
        z = relu6(F.conv2d(z, t[f"blocks.{i}.depthwise.weight"], t[f"blocks.{i}.depthwise.bias"], stride=b["stride"],
                           padding=1, groups=z.shape[1]))
        z = F.conv2d(z, t[f"blocks.{i}.project.weight"], t[f"blocks.{i}.project.bias"])
        x = x + z if b["residual"] else z
        if i == last_of_stage[b["stage"]] and b["stage"] in arch["taps"]:
            skips.append(x)
    x = x + relu6(F.conv2d(x.mean((2, 3), keepdim=True), t["context.weight"], t["context.bias"]))
    for j, skip in enumerate(reversed(skips[:-1])):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        s = relu6(F.conv2d(skip, t[f"decoder.{j}.skip.weight"], t[f"decoder.{j}.skip.bias"]))
        y = relu6(F.conv2d(torch.cat([x, s], 1), t[f"decoder.{j}.mix.weight"], t[f"decoder.{j}.mix.bias"]))
        y = relu6(F.conv2d(y, t[f"decoder.{j}.dw.weight"], t[f"decoder.{j}.dw.bias"], padding=1, groups=y.shape[1]))
        x = relu6(F.conv2d(y, t[f"decoder.{j}.pw.weight"], t[f"decoder.{j}.pw.bias"]))
    return F.conv2d(x, t["head.weight"], t["head.bias"])


def pattern_e(h: int, w: int, seed: int) -> torch.Tensor:
    """(1, 3, h, w) float32 encoded input `v / 255` from an integer pattern (the same arithmetic in the Rust test):
    `v = (x * (c + 2) * 5 + y * (3 + seed) * 3 + ((x * y) >> 4) * (7 + c) + seed * 41) % 256`, with a block (columns
    2/5 to 3/5, rows 1/3 to 2/3) of `255 - v / 4`."""
    y, x = np.mgrid[0:h, 0:w].astype(np.int64)
    out = np.empty((3, h, w), np.int64)
    for c in range(3):
        v = (x * (c + 2) * 5 + y * (3 + seed) * 3 + ((x * y) >> 4) * (7 + c) + seed * 41) % 256
        block = (x * 5 >= w * 2) & (x * 5 < w * 3) & (y * 3 >= h) & (y * 3 < h * 2)
        out[c] = np.where(block, 255 - v // 4, v)
    return torch.from_numpy(out.astype(np.float32) / np.float32(255.0)).unsqueeze(0)


def pattern_linear(h: int, w: int) -> np.ndarray:
    """(3, h, w) float64 linear RGB, exactly representable: `(1 + ((x * (c + 3) + y * 5 + c * 7) % 61)) / 16`, with
    a 12 % bright patch of 40 in the middle (so the median is not the maximum)."""
    y, x = np.mgrid[0:h, 0:w].astype(np.int64)
    out = np.empty((3, h, w), np.float64)
    for c in range(3):
        out[c] = (1 + ((x * (c + 3) + y * 5 + c * 7) % 61)) / 16.0
        patch = (x >= w // 4) & (x < w // 4 + w // 3) & (y >= h // 4) & (y < h // 4 + h // 3)
        out[c] = np.where(patch, 40.0, out[c])
    return out


def randomize(model: BokehNet, seed: int) -> None:
    """Seeded random weights of the real architecture, with batch-norm statistics that make folding matter and a head
    that is not constant: for parity work before a model is trained."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                fan = m.weight[0].numel()
                m.weight.copy_(torch.randn(m.weight.shape, generator=g) * (2.0 / fan) ** 0.5 * (0.2 if m is model.head else 1.0))
                if m.bias is not None:
                    m.bias.copy_(torch.randn(m.bias.shape, generator=g) * 0.1)
            elif isinstance(m, torch.nn.BatchNorm2d):
                n = m.num_features
                m.weight.copy_(0.8 + 0.4 * torch.rand(n, generator=g))
                m.bias.copy_(0.1 * torch.randn(n, generator=g))
                m.running_mean.copy_(0.1 * torch.randn(n, generator=g))
                m.running_var.copy_(0.5 + torch.rand(n, generator=g))


def write(model: BokehNet, out: Path, fixture: Path, source: str) -> dict:
    """Writes the weights and the fixture; returns the report. Raises when the folded network differs from the trained
    one by more than 2e-4 (activations are clamped to 6, so the folded and the trained networks agree to rounding)."""
    model = model.eval().float()
    t = tensors(model)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_deterministic(t, out, metadata(model, source))
    arch = json.loads(arch_json(model))
    cases, worst = [], 0.0
    with torch.no_grad():
        for k, (h, w) in enumerate(SIZES):
            e = pattern_e(h, w, k)
            ref, got = model(e), folded_forward(t, e, arch)
            worst = max(worst, float((ref - got).abs().max()))
            cases.append({"kind": "encoded", "height": h, "width": w, "seed": k,
                          "output": [round(float(v), 7) for v in got[0].permute(1, 2, 0).reshape(-1)]})
        h, w = LINEAR_SIZE
        lin = pattern_linear(h, w)
        e64 = encode.encode_np(lin)
        e = encode.encode_t(torch.from_numpy(lin).float().unsqueeze(0))
        ref, got = model(e), folded_forward(t, e, arch)
        worst = max(worst, float((ref - got).abs().max()))
        probes = [(0, 0), (5, 9), (h // 2, w // 2), (h - 1, w - 1)]
        cases.append({
            "kind": "linear", "height": h, "width": w,
            "encoded_probes": [{"c": c, "y": y, "x": x, "e": round(float(e64[c, y, x]), 9)} for c in range(3) for y, x in probes],
            "output": [round(float(v), 7) for v in got[0].permute(1, 2, 0).reshape(-1)],
        })
    if worst > 2e-4:
        raise SystemExit(f"the folded network differs from the trained one by {worst:.2e}")
    fixture.parent.mkdir(parents=True, exist_ok=True)
    fixture.write_text(json.dumps({"weights_sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "cases": cases}, indent=1) + "\n")
    return {"tensors": len(t), "parameters": sum(v.numel() for v in t.values()), "folded_max_diff": worst,
            "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", type=Path)
    ap.add_argument("--random", action="store_true", help="seeded random weights of the real architecture")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fixture", type=Path, required=True)
    ap.add_argument("--source", default="")
    args = ap.parse_args()
    if args.random == bool(args.model):
        raise SystemExit("give --model <model.pt> or --random")
    model = BokehNet(ARCH)
    if args.random:
        randomize(model, args.seed)
        source = args.source or f"random-seed-{args.seed}"
    else:
        state = torch.load(args.model, map_location="cpu", weights_only=False)
        model.load_state_dict(state["state_dict"])
        source = args.source or args.model.parent.name
    print(json.dumps(write(model, args.out, args.fixture, source), indent=1))


if __name__ == "__main__":
    main()

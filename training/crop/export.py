"""Exports a trained crop student for the app: `data/models/crop/cropnet.safetensors`, read by `ym_ml::crop_net`.

Each batch norm is folded into the convolution before it (the student runs in eval mode, so a batch norm is a fixed
scale and shift per channel), and the tensors are renamed to the layers `crop_net.rs` builds:

    stem.{weight,bias}                       3x3 convolution, stride 2, then SiLU
    blocks.<i>.expand.{weight,bias}          1x1 convolution, then SiLU (absent when the block's expansion is 1)
    blocks.<i>.depthwise.{weight,bias}       3x3 depthwise convolution, stride 1 or 2, then SiLU
    blocks.<i>.project.{weight,bias}         1x1 convolution (no activation); plus the block's input when the shape is kept
    neck.{weight,bias}                       1x1 convolution, then SiLU
    fc1.{weight,bias}, fc2.{weight,bias}     linear, SiLU, linear; then a sigmoid and `model.box_from_numbers`

The header's metadata says what the runtime must check (`format`, `version`, the stages, the input size). The same
run writes `crates/ym-ml/tests/data/cropnet_parity.json`: the student's answers in PyTorch for a few pictures drawn by
`pattern` (which `crop_net.rs`'s tests draw the same way), and the SHA-256 of their letterboxed bytes, so the Rust
letterbox and network are checked against these.

    python3 -m training.crop.export --model <run>/model.pt --out data/models/crop/cropnet.safetensors \
        --fixture crates/ym-ml/tests/data/cropnet_parity.json
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

from training.crop.data import letterbox
from training.crop.model import BINS, HEAD_CHANNELS, HIDDEN, STAGES, CropNet, InvertedResidual

FORMAT = "ym-cropnet"
VERSION = "1"
#: Picture sizes of the parity fixture: the app's 768 px pictures both ways round, and odd and small ones.
PATTERN_SIZES = ((768, 512), (511, 768), (768, 768), (300, 199), (97, 64))


def pattern(width: int, height: int, seed: int) -> np.ndarray:
    """A deterministic 8-bit RGB picture with smooth gradients, edges and a bright block, the same as
    `crop_net.rs`'s `pattern` (integer arithmetic only)."""
    y, x = np.mgrid[0:height, 0:width].astype(np.int64)
    out = np.empty((height, width, 3), np.uint8)
    for c in range(3):
        v = (x * (c + 1) * 7 + y * (3 + seed) + ((x * y) >> 5) * (11 + c) + seed * 37) % 256
        block = (x * 5 >= width * 2) & (x * 5 < width * 3) & (y * 3 >= height) & (y * 3 < height * 2)
        out[..., c] = np.where(block, 255 - (v >> 2), v).astype(np.uint8)
    return out


def fold(conv: torch.nn.Conv2d, bn: torch.nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    """The weight and bias of `bn(conv(x))` as one convolution."""
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    weight = conv.weight * scale.reshape(-1, 1, 1, 1)
    bias = bn.bias - bn.running_mean * scale
    if conv.bias is not None:
        bias = bias + conv.bias * scale
    return weight.detach().contiguous(), bias.detach().contiguous()


def tensors(model: CropNet) -> dict[str, torch.Tensor]:
    """The folded, renamed tensors (module docstring)."""
    f = model.features
    out = {}
    out["stem.weight"], out["stem.bias"] = fold(f[0], f[1])
    blocks = [m for m in f if isinstance(m, InvertedResidual)]
    for i, block in enumerate(blocks):
        layers = list(block.body)
        convs = [(layers[k], layers[k + 1]) for k in range(len(layers)) if isinstance(layers[k], torch.nn.Conv2d)]
        names = (["expand"] if len(convs) == 3 else []) + ["depthwise", "project"]
        for name, (conv, bn) in zip(names, convs):
            out[f"blocks.{i}.{name}.weight"], out[f"blocks.{i}.{name}.bias"] = fold(conv, bn)
    neck = len(f) - 3
    out["neck.weight"], out["neck.bias"] = fold(f[neck], f[neck + 1])
    out["fc1.weight"], out["fc1.bias"] = model.head[1].weight.detach(), model.head[1].bias.detach()
    out["fc2.weight"], out["fc2.bias"] = model.head[3].weight.detach(), model.head[3].bias.detach()
    return {k: v.float().contiguous() for k, v in out.items()}


def metadata(model: CropNet, source: str) -> dict[str, str]:
    """The header's metadata: what `crop_net.rs` checks and what the model card cites."""
    return {
        "format": FORMAT,
        "version": VERSION,
        "input_px": "256",
        "stages": json.dumps([list(s) for s in STAGES]),
        "head_channels": str(HEAD_CHANNELS),
        "hidden": str(HIDDEN),
        "head": model.kind,
        "bins": str(BINS),
        "input_mean": "0.5",
        "input_std": "0.25",
        "source": source,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fixture", type=Path, required=True)
    ap.add_argument("--source", default="", help="what the model card calls this run")
    args = ap.parse_args()
    state = torch.load(args.model, map_location="cpu", weights_only=False)
    config = state.get("config", {})
    if config.get("head", "box") != "box":
        raise SystemExit("only the box head is exported (the app's runtime has no bins decoding)")
    model = CropNet(head="box")
    model.load_state_dict(state["state_dict"])
    model.eval()
    folded = tensors(model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(folded, str(args.out), metadata(model, args.source or args.model.parent.name))

    cases = []
    worst = 0.0
    for k, (w, h) in enumerate(PATTERN_SIZES):
        pix, _ = letterbox(Image.fromarray(pattern(w, h, k)))
        x = torch.from_numpy(pix.copy()).permute(2, 0, 1)[None].float() / 255
        with torch.no_grad():
            numbers = model.numbers(x)[0].double()
            box = model(x)[0].double()
            folded_numbers = folded_forward(folded, x)[0].double()
        worst = max(worst, float((numbers - folded_numbers).abs().max()))
        cases.append({
            "width": w, "height": h, "seed": k,
            "letterbox_sha256": hashlib.sha256(pix.tobytes()).hexdigest(),
            "numbers": [round(float(v), 7) for v in numbers],
            "box": [round(float(v), 7) for v in box],
        })
    if worst > 1e-5:
        raise SystemExit(f"the folded network differs from the trained one by {worst:.2e}")
    args.fixture.parent.mkdir(parents=True, exist_ok=True)
    args.fixture.write_text(json.dumps({
        "weights_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
        "cases": cases,
    }, indent=1) + "\n")
    print(f"{args.out}: {sum(v.numel() for v in folded.values())} parameters; folded within {worst:.1e}")


def folded_forward(t: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """The exported network in plain torch ops (what `crop_net.rs` computes), for the folding check."""
    import torch.nn.functional as F

    y = F.silu(F.conv2d((x - 0.5) / 0.25, t["stem.weight"], t["stem.bias"], stride=2, padding=1))
    i, cin = 0, 24
    for expand, cout, blocks, stride in STAGES:
        for b in range(blocks):
            s = stride if b == 0 else 1
            z = y
            if f"blocks.{i}.expand.weight" in t:
                z = F.silu(F.conv2d(z, t[f"blocks.{i}.expand.weight"], t[f"blocks.{i}.expand.bias"]))
            groups = z.shape[1]
            z = F.silu(F.conv2d(z, t[f"blocks.{i}.depthwise.weight"], t[f"blocks.{i}.depthwise.bias"], stride=s,
                                padding=1, groups=groups))
            z = F.conv2d(z, t[f"blocks.{i}.project.weight"], t[f"blocks.{i}.project.bias"])
            y = y + z if (s == 1 and cin == cout) else z
            cin = cout
            i += 1
    y = F.silu(F.conv2d(y, t["neck.weight"], t["neck.bias"]))
    y = F.silu(F.linear(y.flatten(1), t["fc1.weight"], t["fc1.bias"]))
    return torch.sigmoid(F.linear(y, t["fc2.weight"], t["fc2.bias"]))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Converts the Depth Anything V2-Small checkpoint into the safetensors file the Rust runtime reads (D-02, `docs/23`).

Python is for converting weights and for test fixtures only (`AGENTS.md` rule 6): this script runs in CI
(`.github/workflows/models.yml`, job `depth`), never in the app and never in `ym`. The result is published as an asset of a
Yellowmoon release and fetched by the build (`tools/fetch/depth.sh`); it is not committed (too big for git).

    python3 -m training.depth.export --upstream /path/to/Depth-Anything-V2 --out build/depth

What it does:

1. Downloads `depth_anything_v2_vits.pth` from Hugging Face at the pinned revision (`CKPT_URL`) and refuses a file whose
   SHA-256 or size is not the pinned one. The pin is the LFS object id Hugging Face states for that revision (API, the
   `X-Linked-ETag` header and the LFS pointer agree, read 2026-11-07); this script checks the real bytes on its first run.
2. Builds the upstream model on the CPU (`DepthAnythingV2(**MODEL_CONFIG)` from the checkout at `UPSTREAM_COMMIT`, whose
   `run.py` holds the same configuration) and loads the checkpoint into it with `strict=True`, which proves that the
   checkpoint and the configuration match. It also compares every tensor name and shape with `expected_tensors()`, the
   list written out below from the upstream code, so that the names the Rust runtime reads cannot drift unnoticed.
3. Keeps every tensor of the checkpoint, under the name it has there, as f16 (239 tensors, 24,785,089 parameters). Nothing is
   dropped: `pretrained.mask_token` is not used when inference passes no mask, but it is 384 numbers and keeping the file a
   plain copy of the checkpoint is worth more.
4. Writes the safetensors file with this script, not with the `safetensors` writer: that writer serialises the
   `__metadata__` map in a hash order that changes from process to process, and the file must be byte-identical on every
   run (`docs/algorithms/06` section 6.4). The header carries the configuration the runtime needs, read from the model the
   upstream code built (not typed in), and `check_config` fails when it is not the model `docs/23` describes.
5. Reads the file back with `safetensors` (a different implementation of the format), compares it bit for bit with what
   was written, then runs the upstream model on a fixed synthetic picture with the original f32 weights and with the f16
   weights widened to f32, and reports how far the f16 storage moves the depth map (`export_report.json`).
6. Prints the parameter count, the file size and the SHA-256.

Tensor names (those of the checkpoint, which are the attribute paths of the upstream modules; `N` is 0 to 11, `K` is 1 to 4):

    pretrained.cls_token                  [1, 1, 384]
    pretrained.pos_embed                  [1, 1370, 384]     the class token's and the 37 x 37 patch positions' embeddings
    pretrained.mask_token                 [1, 384]           unused at inference
    pretrained.patch_embed.proj.{weight,bias}   [384, 3, 14, 14], [384]
    pretrained.blocks.N.norm1.{weight,bias}     [384]
    pretrained.blocks.N.attn.qkv.{weight,bias}  [1152, 384], [1152]   q, k, v stacked in that order, 6 heads of 64
    pretrained.blocks.N.attn.proj.{weight,bias} [384, 384], [384]
    pretrained.blocks.N.ls1.gamma               [384]        LayerScale
    pretrained.blocks.N.norm2.{weight,bias}     [384]
    pretrained.blocks.N.mlp.fc1.{weight,bias}   [1536, 384], [1536]
    pretrained.blocks.N.mlp.fc2.{weight,bias}   [384, 1536], [384]
    pretrained.blocks.N.ls2.gamma               [384]
    pretrained.norm.{weight,bias}               [384]        the final LayerNorm, applied to each of the four taps
    depth_head.projects.{0..3}.{weight,bias}    1 x 1 convolutions 384 -> 48, 96, 192, 384
    depth_head.resize_layers.0.{weight,bias}    ConvTranspose2d 48 -> 48, kernel 4, stride 4   [48, 48, 4, 4]
    depth_head.resize_layers.1.{weight,bias}    ConvTranspose2d 96 -> 96, kernel 2, stride 2   [96, 96, 2, 2]
    (resize_layers.2 is the identity: no tensors)
    depth_head.resize_layers.3.{weight,bias}    Conv2d 384 -> 384, kernel 3, stride 2, padding 1
    depth_head.scratch.layerK_rn.weight         3 x 3 convolutions to 64 channels, no bias
    depth_head.scratch.refinenetK.out_conv.{weight,bias}                   1 x 1, 64 -> 64
    depth_head.scratch.refinenetK.resConfUnit{1,2}.conv{1,2}.{weight,bias} 3 x 3, 64 -> 64 (refinenet4's resConfUnit1 is in the file and unused)
    depth_head.scratch.output_conv1.{weight,bias}                          3 x 3, 64 -> 32
    depth_head.scratch.output_conv2.0.{weight,bias}                        3 x 3, 32 -> 32
    depth_head.scratch.output_conv2.2.{weight,bias}                        1 x 1, 32 -> 1

Versions are pinned in `training/depth/requirements.txt` and in `models.yml` (torch, torchvision, safetensors, the upstream commit).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
import urllib.request
from pathlib import Path

#: Hugging Face `depth-anything/Depth-Anything-V2-Small` (Apache-2.0 on the model card), `main` on 2026-11-07, last changed 2024-07-08.
CKPT_REPO = "depth-anything/Depth-Anything-V2-Small"
CKPT_REVISION = "03876f8651c73a60fe4c2c48294e09fcb6838fcf"
CKPT_URL = f"https://huggingface.co/{CKPT_REPO}/resolve/{CKPT_REVISION}/depth_anything_v2_vits.pth"
#: SHA-256 and size of the file at `CKPT_URL`: the LFS object id and size Hugging Face states for that revision (read 2026-11-07
#: from the model API, from the `X-Linked-ETag` and `X-Linked-Size` headers of the resolve URL and from the LFS pointer). The file
#: itself is hashed by `fetch_checkpoint` on the first run in CI; a mismatch stops the run.
CKPT_SHA256 = "715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378"
CKPT_BYTES = 99_218_434
#: github.com/DepthAnything/Depth-Anything-V2 `main` on 2026-11-07 (there are no tags). Apache-2.0 (the LICENSE file at that commit).
UPSTREAM_COMMIT = "a561b849ebae10a6f5ef49e26c83cbbcd36c71bf"
UPSTREAM_URL = "https://github.com/DepthAnything/Depth-Anything-V2"

#: `model_configs['vits']` of the upstream `run.py` (and the same three values in its README):
#:     'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
MODEL_CONFIG = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}
#: `self.intermediate_layer_idx` of `DepthAnythingV2.__init__` in the upstream `dpt.py`: 'vits': [2, 5, 8, 11] (zero-based block indices).
INTERMEDIATE_LAYERS = [2, 5, 8, 11]
#: `NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` in `DepthAnythingV2.image2tensor`.
PIXEL_MEAN = [0.485, 0.456, 0.406]
PIXEL_STD = [0.229, 0.224, 0.225]
#: `infer_image(raw_image, input_size=518)`: the short side, and `ensure_multiple_of=14` (the patch size).
INPUT_SIZE = 518
PATCH_SIZE = 14

#: The published file and the version of its layout (`format` and `version` in the header).
OUT_NAME = "depth_anything_v2_vits.f16.safetensors"
FORMAT = "yellowmoon-depth"
VERSION = "1"

#: What `expected_tensors()` counts (derived from the upstream code by hand and checked against it in `test_depth.py`; the first
#: CI run checks the checkpoint against it too). Hugging Face and the paper say "24.8 M parameters".
EXPECTED_TENSORS = 239
EXPECTED_PARAMETERS = 24_785_089


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, read in 1 MiB pieces."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_tensors() -> dict[str, tuple[int, ...]]:
    """Every tensor name of the checkpoint and its shape, written out from the upstream code (no torch needed).

    The encoder is `DINOv2(model_name='vits')` of `dinov2.py` (`vit_small(patch_size=14, img_size=518, init_values=1.0, ...)`), the head
    is `DPTHead(384, features=64, out_channels=[48, 96, 192, 384])` of `dpt.py` with `use_bn=False` and `use_clstoken=False`.
    """
    dim, hidden, patches = 384, 4 * 384, 37 * 37
    features = MODEL_CONFIG["features"]
    out_channels = MODEL_CONFIG["out_channels"]
    shapes: dict[str, tuple[int, ...]] = {
        "pretrained.cls_token": (1, 1, dim),
        "pretrained.pos_embed": (1, patches + 1, dim),
        "pretrained.mask_token": (1, dim),
        "pretrained.patch_embed.proj.weight": (dim, 3, PATCH_SIZE, PATCH_SIZE),
        "pretrained.patch_embed.proj.bias": (dim,),
    }
    for n in range(12):
        prefix = f"pretrained.blocks.{n}."
        for name, shape in (
            ("norm1.weight", (dim,)),
            ("norm1.bias", (dim,)),
            ("attn.qkv.weight", (3 * dim, dim)),
            ("attn.qkv.bias", (3 * dim,)),
            ("attn.proj.weight", (dim, dim)),
            ("attn.proj.bias", (dim,)),
            ("ls1.gamma", (dim,)),
            ("norm2.weight", (dim,)),
            ("norm2.bias", (dim,)),
            ("mlp.fc1.weight", (hidden, dim)),
            ("mlp.fc1.bias", (hidden,)),
            ("mlp.fc2.weight", (dim, hidden)),
            ("mlp.fc2.bias", (dim,)),
            ("ls2.gamma", (dim,)),
        ):
            shapes[prefix + name] = shape
    shapes["pretrained.norm.weight"] = (dim,)
    shapes["pretrained.norm.bias"] = (dim,)
    for k, channels in enumerate(out_channels):
        shapes[f"depth_head.projects.{k}.weight"] = (channels, dim, 1, 1)
        shapes[f"depth_head.projects.{k}.bias"] = (channels,)
    # resize_layers: ConvTranspose2d weights are [in, out, kh, kw]; layer 2 is nn.Identity (no tensors).
    shapes["depth_head.resize_layers.0.weight"] = (out_channels[0], out_channels[0], 4, 4)
    shapes["depth_head.resize_layers.0.bias"] = (out_channels[0],)
    shapes["depth_head.resize_layers.1.weight"] = (out_channels[1], out_channels[1], 2, 2)
    shapes["depth_head.resize_layers.1.bias"] = (out_channels[1],)
    shapes["depth_head.resize_layers.3.weight"] = (out_channels[3], out_channels[3], 3, 3)
    shapes["depth_head.resize_layers.3.bias"] = (out_channels[3],)
    for k, channels in enumerate(out_channels, start=1):
        shapes[f"depth_head.scratch.layer{k}_rn.weight"] = (features, channels, 3, 3)  # bias=False
    for k in range(1, 5):
        prefix = f"depth_head.scratch.refinenet{k}."
        shapes[prefix + "out_conv.weight"] = (features, features, 1, 1)
        shapes[prefix + "out_conv.bias"] = (features,)
        for unit in (1, 2):
            for conv in (1, 2):
                shapes[f"{prefix}resConfUnit{unit}.conv{conv}.weight"] = (features, features, 3, 3)
                shapes[f"{prefix}resConfUnit{unit}.conv{conv}.bias"] = (features,)
    shapes["depth_head.scratch.output_conv1.weight"] = (features // 2, features, 3, 3)
    shapes["depth_head.scratch.output_conv1.bias"] = (features // 2,)
    shapes["depth_head.scratch.output_conv2.0.weight"] = (32, features // 2, 3, 3)
    shapes["depth_head.scratch.output_conv2.0.bias"] = (32,)
    shapes["depth_head.scratch.output_conv2.2.weight"] = (1, 32, 1, 1)
    shapes["depth_head.scratch.output_conv2.2.bias"] = (1,)
    return shapes


def parameter_count(shapes: dict[str, tuple[int, ...]]) -> int:
    """Number of scalars in a shape table."""
    total = 0
    for shape in shapes.values():
        count = 1
        for extent in shape:
            count *= extent
        total += count
    return total


def fetch_checkpoint(cache: Path) -> Path:
    """Downloads the checkpoint into `cache` unless a file with the pinned hash is already there."""
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "depth_anything_v2_vits.pth"
    if path.exists() and sha256_file(path) == CKPT_SHA256:
        print(f"checkpoint: cached, {path.stat().st_size:,} bytes, SHA-256 {CKPT_SHA256}")
        return path
    part = path.with_name(path.name + ".part")
    print(f"checkpoint: downloading {CKPT_URL}")
    with urllib.request.urlopen(CKPT_URL, timeout=300) as response, part.open("wb") as handle:  # noqa: S310 (a fixed https URL)
        while block := response.read(1 << 20):
            handle.write(block)
    got = sha256_file(part)
    if got != CKPT_SHA256 or part.stat().st_size != CKPT_BYTES:
        sys.exit(
            f"error: {CKPT_URL} gave {part.stat().st_size} bytes with SHA-256 {got}; "
            f"expected {CKPT_BYTES} bytes and {CKPT_SHA256}. The file changed, or the download was cut off."
        )
    part.replace(path)
    print(f"checkpoint: {path.stat().st_size:,} bytes, SHA-256 {got}")
    return path


def use_upstream(path: Path) -> None:
    """Puts the checkout of the upstream code on `sys.path` after checking that it is at `UPSTREAM_COMMIT` (when it is a git checkout)."""
    if not (path / "depth_anything_v2" / "dpt.py").is_file():
        sys.exit(f"error: {path} is not a checkout of {UPSTREAM_URL} (no depth_anything_v2/dpt.py); pass --upstream or set DEPTH_ANYTHING_V2_SRC")
    if (path / ".git").exists():
        head = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        if head != UPSTREAM_COMMIT:
            sys.exit(f"error: {path} is at {head}, the pinned commit is {UPSTREAM_COMMIT}")
    sys.path.insert(0, str(path))


def build_upstream_model(checkpoint: Path | None):
    """The upstream `DepthAnythingV2` on the CPU, in eval mode; with `checkpoint` its weights are loaded with `strict=True`."""
    import torch
    from depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(**MODEL_CONFIG)
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def write_safetensors(path: Path, tensors: dict, metadata: dict[str, str]) -> None:
    """Writes f16 tensors as a safetensors file, byte for byte the same on every run.

    Layout (https://github.com/huggingface/safetensors): an 8-byte little-endian header length, the JSON header
    (`__metadata__` first, then every tensor by name with `dtype`, `shape`, `data_offsets`), padded with spaces to a
    multiple of 8 bytes, then the tensor bytes in the order of the names. Tensors are stored little-endian f16.
    """
    import numpy as np

    header: dict = {"__metadata__": dict(sorted(metadata.items()))}
    offset = 0
    blobs = []
    for name in sorted(tensors):
        array = np.ascontiguousarray(tensors[name].numpy()).astype("<f2", copy=False)
        blob = array.tobytes()
        header[name] = {"dtype": "F16", "shape": list(array.shape), "data_offsets": [offset, offset + len(blob)]}
        offset += len(blob)
        blobs.append(blob)
    raw = json.dumps(header, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    raw += b" " * (-len(raw) % 8)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(raw)))
        handle.write(raw)
        for blob in blobs:
            handle.write(blob)


def build_config(model) -> dict:
    """The configuration of the network, read from the model the upstream code built (not typed in), so that a mismatch fails here."""
    encoder = model.pretrained
    head = model.depth_head
    block = encoder.blocks[0]
    scratch = head.scratch
    fusion = scratch.refinenet1
    return {
        "model": "depth_anything_v2_vits",
        "input": {
            "short_side_px": INPUT_SIZE,
            "multiple_of": PATCH_SIZE,
            "resize": "upstream Resize(keep_aspect_ratio=True, ensure_multiple_of=14, resize_method='lower_bound', cv2.INTER_CUBIC) of the float64 RGB picture divided by 255",
            "pixel_mean": PIXEL_MEAN,
            "pixel_std": PIXEL_STD,
            "layout": "NCHW, f32, (x - mean) / std per channel",
        },
        "encoder": {
            "type": "DINOv2 ViT-S/14",
            "patch_size": encoder.patch_size,
            "in_chans": encoder.patch_embed.proj.in_channels,
            "embed_dim": encoder.embed_dim,
            "depth": len(encoder.blocks),
            "num_heads": block.attn.num_heads,
            "mlp_ratio": block.mlp.fc1.out_features // encoder.embed_dim,
            "mlp_activation": "gelu (exact, erf)" if type(block.mlp.act).__name__ == "GELU" and block.mlp.act.approximate == "none" else type(block.mlp.act).__name__,
            "qkv_bias": block.attn.qkv.bias is not None,
            "layer_scale": type(block.ls1).__name__ == "LayerScale",
            "layer_norm_eps": encoder.norm.eps,
            "num_register_tokens": encoder.num_register_tokens,
            "pos_embed_grid": [37, 37],
            "pos_embed_interpolation": {
                "mode": "bicubic",
                "antialias": bool(encoder.interpolate_antialias),
                "offset": encoder.interpolate_offset,
                "rule": "patch grid (h0, w0) -> scale_factor = ((h0 + offset) / 37, (w0 + offset) / 37) on the 37 x 37 grid, torch interpolate(scale_factor=..., recompute_scale_factor=None), class token kept as it is; the grid is used as it is when it is 37 x 37 (518 x 518)",
            },
            "intermediate_layers": list(model.intermediate_layer_idx[model.encoder]),
            "tap_norm": "pretrained.norm (a LayerNorm) is applied to each tapped block output; class token dropped",
            "attention": "plain softmax(q k^T * head_dim^-0.5) v, q scaled before the product (xformers off)",
        },
        "head": {
            "type": "DPTHead",
            "features": scratch.layer1_rn.out_channels,
            "out_channels": [project.out_channels for project in head.projects],
            "use_clstoken": head.use_clstoken,
            "use_bn": False,
            "resize_layers": [
                {"type": "ConvTranspose2d", "kernel": 4, "stride": 4},
                {"type": "ConvTranspose2d", "kernel": 2, "stride": 2},
                {"type": "Identity"},
                {"type": "Conv2d", "kernel": 3, "stride": 2, "padding": 1},
            ],
            "fusion": {
                "interpolation": "bilinear",
                "align_corners": bool(fusion.align_corners),
                "order": "refinenet4(layer4_rn, size=layer3_rn) -> refinenet3(+layer3_rn, size=layer2_rn) -> refinenet2(+layer2_rn, size=layer1_rn) -> refinenet1(+layer1_rn, scale_factor=2)",
                "residual_unit": "relu, conv3x3, relu, conv3x3, add input",
            },
            "output": "output_conv1 (3x3, 64 -> 32), bilinear align_corners=True to (patch_h * 14, patch_w * 14), output_conv2 (3x3 32 -> 32, relu, 1x1 32 -> 1, relu), then relu and squeeze",
            "output_meaning": "relative inverse depth (affine-invariant disparity), larger is nearer; confirm on the oracle fixtures",
        },
    }


def check_config(config: dict) -> None:
    """Fails when the model the upstream code built is not the one `docs/23` describes."""
    encoder, head = config["encoder"], config["head"]
    expect = {
        "patch_size": (encoder["patch_size"], 14),
        "embed_dim": (encoder["embed_dim"], 384),
        "depth": (encoder["depth"], 12),
        "num_heads": (encoder["num_heads"], 6),
        "mlp_ratio": (encoder["mlp_ratio"], 4),
        "mlp_activation": (encoder["mlp_activation"], "gelu (exact, erf)"),
        "qkv_bias": (encoder["qkv_bias"], True),
        "layer_scale": (encoder["layer_scale"], True),
        "layer_norm_eps": (encoder["layer_norm_eps"], 1e-6),
        "num_register_tokens": (encoder["num_register_tokens"], 0),
        "pos_embed interpolation antialias": (encoder["pos_embed_interpolation"]["antialias"], False),
        "pos_embed interpolation offset": (encoder["pos_embed_interpolation"]["offset"], 0.1),
        "intermediate_layers": (encoder["intermediate_layers"], INTERMEDIATE_LAYERS),
        "features": (head["features"], 64),
        "out_channels": (head["out_channels"], [48, 96, 192, 384]),
        "use_clstoken": (head["use_clstoken"], False),
        "fusion align_corners": (head["fusion"]["align_corners"], True),
    }
    wrong = [f"{name}: {got!r}, expected {want!r}" for name, (got, want) in expect.items() if got != want]
    if wrong:
        sys.exit("error: the model is not Depth Anything V2-Small as the runtime spec says:\n  " + "\n  ".join(wrong))


def synthetic_tensor(torch, height: int = 518, width: int = 784):
    """A fixed 1 x 3 x 518 x 784 input with structure at several scales, normalised like the upstream transform; no random numbers."""
    ys = torch.linspace(0.0, 1.0, height, dtype=torch.float64)
    xs = torch.linspace(0.0, 1.0, width, dtype=torch.float64)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    red = 0.5 + 0.3 * torch.sin(14.0 * xx + 3.0 * yy) + 0.2 * torch.cos(60.0 * yy)
    green = 0.5 + 0.3 * torch.sin(9.0 * yy - 2.0 * xx) * torch.cos(23.0 * xx)
    blue = 0.5 + 0.25 * torch.cos(5.0 * (xx - 0.5) ** 2 * 40.0) + 0.2 * (xx > 0.5).double() * (yy < 0.5).double()
    picture = torch.stack([red, green, blue]).clamp(0.0, 1.0)
    mean = torch.tensor(PIXEL_MEAN, dtype=torch.float64).view(3, 1, 1)
    std = torch.tensor(PIXEL_STD, dtype=torch.float64).view(3, 1, 1)
    return ((picture - mean) / std).float()[None]


def relative_rms(a, b, torch) -> tuple[float, float]:
    """(RMS of a - b divided by the RMS of b, largest absolute difference)."""
    diff = (a - b).double()
    return float(diff.pow(2).mean().sqrt() / b.double().pow(2).mean().sqrt()), float(diff.abs().max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path("build/depth"), help="output directory (default build/depth)")
    parser.add_argument("--cache", type=Path, default=None, help="where the checkpoint is kept (default OUT/cache)")
    parser.add_argument("--upstream", type=Path, default=os.environ.get("DEPTH_ANYTHING_V2_SRC"), help=f"a checkout of {UPSTREAM_URL} at the pinned commit (default $DEPTH_ANYTHING_V2_SRC)")
    parser.add_argument("--skip-f16-check", action="store_true", help="skip the run that measures what f16 storage costs")
    args = parser.parse_args()
    if args.upstream is None:
        sys.exit("error: --upstream (or DEPTH_ANYTHING_V2_SRC) must name a checkout of the upstream code")

    import numpy as np
    import torch
    import safetensors
    from safetensors import safe_open

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    use_upstream(Path(args.upstream))
    ckpt = fetch_checkpoint(args.cache or out / "cache")

    print(f"versions: torch {torch.__version__}, safetensors {safetensors.__version__}, numpy {np.__version__}, upstream commit {UPSTREAM_COMMIT}")
    model = build_upstream_model(ckpt)  # load_state_dict(strict=True) raises on any mismatch
    config = build_config(model)
    check_config(config)

    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    shapes = expected_tensors()
    if parameter_count(shapes) != EXPECTED_PARAMETERS or len(shapes) != EXPECTED_TENSORS:
        sys.exit("error: expected_tensors() disagrees with EXPECTED_TENSORS and EXPECTED_PARAMETERS; fix the constants")
    wrong = sorted(set(state) ^ set(shapes))
    wrong += [f"{name}: {tuple(tensor.shape)}, expected {shapes[name]}" for name, tensor in state.items() if name in shapes and tuple(tensor.shape) != shapes[name]]
    if wrong:
        sys.exit("error: the checkpoint does not have the tensors the runtime reads:\n  " + "\n  ".join(wrong[:20]))
    total = sum(t.numel() for t in state.values())
    if total != EXPECTED_PARAMETERS:
        sys.exit(f"error: the checkpoint has {total:,} parameters; expected {EXPECTED_PARAMETERS:,}")

    f16 = {}
    worst_relative = 0.0
    largest = 0.0
    for name, tensor in sorted(state.items()):
        if tensor.dtype != torch.float32:
            sys.exit(f"error: {name} is {tensor.dtype}, expected float32")
        if not torch.isfinite(tensor).all():
            sys.exit(f"error: {name} has values that are not finite")
        narrow = tensor.to(torch.float16)
        if not torch.isfinite(narrow).all():
            sys.exit(f"error: {name} does not fit in f16")
        f16[name] = narrow.contiguous()
        largest = max(largest, float(tensor.abs().max()))
        nonzero = tensor != 0
        if nonzero.any():
            relative = (narrow.float() - tensor).abs()[nonzero] / tensor.abs()[nonzero]
            relative = relative[tensor.abs()[nonzero] > 6.2e-5]  # f16's smallest normal number; below it the grid is absolute
            if relative.numel():
                worst_relative = max(worst_relative, float(relative.max()))

    metadata = {
        "format": FORMAT,
        "version": VERSION,
        "dtype": "F16",
        "model": "depth_anything_v2_vits",
        "source_url": CKPT_URL,
        "source_revision": CKPT_REVISION,
        "source_sha256": CKPT_SHA256,
        "source_license": "Apache-2.0, Depth Anything V2 authors (the model card and the repository)",
        "upstream_url": UPSTREAM_URL,
        "upstream_commit": UPSTREAM_COMMIT,
        "kept": "every tensor of the checkpoint under its own name",
        "config": json.dumps(config, sort_keys=True, separators=(",", ":")),
    }
    target = out / OUT_NAME
    write_safetensors(target, f16, metadata)

    # Read it back with the safetensors library: the same names, shapes and bits, and the same header.
    with safe_open(str(target), framework="pt") as handle:
        if dict(handle.metadata()) != metadata:
            sys.exit("error: the header read back differs from the one written")
        if sorted(handle.keys()) != sorted(f16):
            sys.exit("error: the tensor names read back differ from the ones written")
        for name in f16:
            back = handle.get_tensor(name)
            if back.dtype != torch.float16 or back.shape != f16[name].shape or not torch.equal(back.view(torch.int16), f16[name].view(torch.int16)):
                sys.exit(f"error: {name} read back differs from what was written")
    digest = sha256_file(target)
    size = target.stat().st_size

    report = {
        "file": OUT_NAME,
        "bytes": size,
        "sha256": digest,
        "tensors": len(f16),
        "parameters": total,
        "largest_abs_value": largest,
        "worst_relative_f16_rounding": worst_relative,
        "source_sha256": CKPT_SHA256,
        "upstream_commit": UPSTREAM_COMMIT,
        "torch": torch.__version__,
        "safetensors": safetensors.__version__,
    }
    if not args.skip_f16_check:
        picture = synthetic_tensor(torch)
        with torch.no_grad():
            before = model(picture)
            model.load_state_dict({name: t.float() for name, t in f16.items()}, strict=True)
            after = model(picture)
        rel, largest_diff = relative_rms(after, before, torch)
        report["f16_cost_on_a_synthetic_picture"] = {"depth": {"relative_rms": rel, "largest_abs": largest_diff, "depth_rms": float(before.double().pow(2).mean().sqrt())}}
        print(f"f16 cost, depth: RMS of the difference is {rel:.2e} of the map's RMS, largest difference {largest_diff:.2e}")
    (out / "export_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"kept {len(f16)} tensors: {total:,} parameters")
    print(f"largest weight {largest:.3f}; worst relative f16 rounding {worst_relative:.2e}")
    print(f"wrote {target}: {size:,} bytes ({size / 1e6:.1f} MB)")
    print(f"{digest}  {OUT_NAME}")


if __name__ == "__main__":
    main()

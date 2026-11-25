#!/usr/bin/env python3
"""Converts the EoMT small (COCO panoptic) checkpoint into the safetensors file the Rust runtime reads (RG-03, `docs/34` section 4b.1).

Python is for converting weights and for test fixtures only (`AGENTS.md` rule 6): this script runs in CI
(`.github/workflows/models.yml`, job `eomt`), never in the app and never in `ym`. The result is published as an asset of a
Yellowmoon release and fetched by the build (`tools/fetch/eomt.sh`); it is not committed (too big for git).

    python3 -m training.eomt.export --out build/eomt

What it does:

1. Downloads `config.json`, `preprocessor_config.json` and `model.safetensors` of `tue-mps/coco_panoptic_eomt_small_640_2x` from
   Hugging Face at the pinned revision and refuses a file whose SHA-256 or size is not the pinned one (the checkpoint's pin is
   the LFS object id Hugging Face states for that revision; the two small files' pins were computed on the files the POC used).
2. Builds `EomtForUniversalSegmentation` from them on the CPU and loads the checkpoint (no missing and no unexpected key), then
   compares every tensor name and shape with `expected_tensors()`, written out below from the model's code, so that the names the
   Rust runtime reads cannot drift unnoticed. The configuration must be the one `docs/34` describes: **patch 16**, a 40 x 40
   grid of positions (the shape of `embeddings.position_embeddings.weight` and of the patch convolution decide, not the training
   YAML's `vit_small_patch14_reg4_dinov2`), and `attn_mask_probs` all zero (see below).
3. Keeps every tensor of the checkpoint, under the name it has there, as f16 (244 tensors, 23,976,079 parameters). Nothing is
   dropped: `criterion.empty_weight` is the loss's class weight, used only in training, but keeping the file a plain copy of the
   checkpoint is worth more (the runtime lists it as unused).
4. Writes the safetensors file with this script, not with the `safetensors` writer: that writer serialises the `__metadata__` map
   in an order that changes from process to process, and the file must be byte-identical on every run. The header carries the
   configuration the runtime needs, read from the model the library built (not typed in), the class names of `id2label`, the
   pre-processing and the post-processing the reference uses, and the source, its licence and the versions that made it.
5. Reads the file back with `safetensors` (a different implementation of the format) and compares it bit for bit with what was
   written, then runs the reference model on a fixed synthetic 640 x 640 picture with the original f32 weights and with the f16
   weights widened to f32, and reports how far the f16 storage moves the outputs (`export_report.json`).
6. Prints the parameter count, the file size and the SHA-256.

About `attn_mask_probs`. The model's forward pass computes a "masked attention" for the last three blocks only when
`attn_mask_probs[block] > 0` (or when training). The checkpoint was trained with the mask annealed to zero, so its three values
are exactly 0 and **inference never masks anything**: the three blocks that carry the queries attend to everything, and no
intermediate prediction is made. The Rust runtime reads the same three numbers from the file and refuses a file where one is not 0.

Tensor names (those of the checkpoint; `N` is 0 to 11, `M` is 0 or 1):

    embeddings.cls_token                              [1, 1, 384]
    embeddings.register_tokens                        [1, 4, 384]
    embeddings.patch_embeddings.projection.{weight,bias}   [384, 3, 16, 16], [384]
    embeddings.position_embeddings.weight             [1600, 384]     the 40 x 40 patch positions (none for the class and register tokens)
    layers.N.norm1.{weight,bias}, layers.N.norm2.{weight,bias}          [384]
    layers.N.attention.{q,k,v}_proj.{weight,bias}     [384, 384], [384]    6 heads of 64
    layers.N.attention.out_proj.{weight,bias}         [384, 384], [384]
    layers.N.layer_scale1.lambda1, layers.N.layer_scale2.lambda1       [384]
    layers.N.mlp.fc1.{weight,bias}                    [1536, 384], [1536]
    layers.N.mlp.fc2.{weight,bias}                    [384, 1536], [384]
    layernorm.{weight,bias}                           [384]
    query.weight                                      [200, 384]     the learned queries, entering before block 9
    class_predictor.{weight,bias}                     [134, 384], [134]    133 classes and "none"
    mask_head.fc{1,2,3}.{weight,bias}                 [384, 384], [384]
    upscale_block.block.M.conv1.{weight,bias}         [384, 384, 2, 2], [384]    ConvTranspose2d, kernel 2, stride 2
    upscale_block.block.M.conv2.weight                [384, 1, 3, 3]             depthwise, no bias
    upscale_block.block.M.layernorm2d.{weight,bias}   [384]
    attn_mask_probs                                   [3]            all 0
    criterion.empty_weight                            [134]          training only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import urllib.request
from pathlib import Path

REPO = "tue-mps/coco_panoptic_eomt_small_640_2x"
REVISION = "10f5326794a8ce7e0e975791033ae1c13aec3767"
BASE_URL = f"https://huggingface.co/{REPO}/resolve/{REVISION}"
#: file name -> (SHA-256, size in bytes). The checkpoint's hash is Hugging Face's LFS object id for the revision.
SOURCE_FILES = {
    "config.json": ("de222958bcd41d63e57059ffb42bfedbe6dddc6f8ac61529af8558e86667d2cc", 6426),
    "preprocessor_config.json": ("8088c2597646d2a84effdfd11682ce70fd5e950c787fa86b4f4d9be706ddeef7", 664),
    "model.safetensors": ("516b0b23acace639cc27739593881dda1c0010b3bec06aae138b615717fb8855", 95_927_796),
}
CKPT_SHA256 = SOURCE_FILES["model.safetensors"][0]
UPSTREAM_URL = "https://github.com/tue-mps/eomt"

OUT_NAME = "eomt_coco_panoptic_small.f16.safetensors"
FORMAT = "yellowmoon-eomt"
VERSION = "1"

HIDDEN = 384
LAYERS = 12
HEADS = 6
MLP = 4 * HIDDEN
PATCH = 16
GRID = 40
IMAGE = PATCH * GRID
REGISTERS = 4
QUERIES = 200
QUERY_BLOCKS = 3
UPSCALE_BLOCKS = 2
LABELS = 133
EXPECTED_TENSORS = 244
EXPECTED_PARAMETERS = 23_976_079


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, read in 1 MiB pieces."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_tensors() -> dict[str, tuple[int, ...]]:
    """Every tensor of the checkpoint and its shape, from the model's code (`modeling_eomt.py`)."""
    c = HIDDEN
    shapes: dict[str, tuple[int, ...]] = {
        "embeddings.cls_token": (1, 1, c),
        "embeddings.register_tokens": (1, REGISTERS, c),
        "embeddings.patch_embeddings.projection.weight": (c, 3, PATCH, PATCH),
        "embeddings.patch_embeddings.projection.bias": (c,),
        "embeddings.position_embeddings.weight": (GRID * GRID, c),
        "layernorm.weight": (c,),
        "layernorm.bias": (c,),
        "query.weight": (QUERIES, c),
        "class_predictor.weight": (LABELS + 1, c),
        "class_predictor.bias": (LABELS + 1,),
        "attn_mask_probs": (QUERY_BLOCKS,),
        "criterion.empty_weight": (LABELS + 1,),
    }
    for n in range(LAYERS):
        p = f"layers.{n}"
        for norm in ("norm1", "norm2"):
            shapes[f"{p}.{norm}.weight"] = (c,)
            shapes[f"{p}.{norm}.bias"] = (c,)
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            shapes[f"{p}.attention.{proj}.weight"] = (c, c)
            shapes[f"{p}.attention.{proj}.bias"] = (c,)
        shapes[f"{p}.layer_scale1.lambda1"] = (c,)
        shapes[f"{p}.layer_scale2.lambda1"] = (c,)
        shapes[f"{p}.mlp.fc1.weight"] = (MLP, c)
        shapes[f"{p}.mlp.fc1.bias"] = (MLP,)
        shapes[f"{p}.mlp.fc2.weight"] = (c, MLP)
        shapes[f"{p}.mlp.fc2.bias"] = (c,)
    for fc in ("fc1", "fc2", "fc3"):
        shapes[f"mask_head.{fc}.weight"] = (c, c)
        shapes[f"mask_head.{fc}.bias"] = (c,)
    for m in range(UPSCALE_BLOCKS):
        p = f"upscale_block.block.{m}"
        shapes[f"{p}.conv1.weight"] = (c, c, 2, 2)
        shapes[f"{p}.conv1.bias"] = (c,)
        shapes[f"{p}.conv2.weight"] = (c, 1, 3, 3)
        shapes[f"{p}.layernorm2d.weight"] = (c,)
        shapes[f"{p}.layernorm2d.bias"] = (c,)
    return shapes


def parameter_count(shapes: dict[str, tuple[int, ...]]) -> int:
    """The number of values in all the tensors."""
    total = 0
    for shape in shapes.values():
        n = 1
        for d in shape:
            n *= d
        total += n
    return total


def fetch_sources(cache: Path) -> Path:
    """Downloads the three files into `cache` unless files with the pinned hashes are already there; returns `cache`."""
    cache.mkdir(parents=True, exist_ok=True)
    for name, (sha, size) in SOURCE_FILES.items():
        path = cache / name
        if path.exists() and sha256_file(path) == sha:
            print(f"source: {name} cached, {path.stat().st_size:,} bytes, SHA-256 {sha}")
            continue
        part = path.with_name(path.name + ".part")
        url = f"{BASE_URL}/{name}"
        print(f"source: downloading {url}")
        with urllib.request.urlopen(url, timeout=300) as response, part.open("wb") as handle:  # noqa: S310 (a fixed https URL)
            while block := response.read(1 << 20):
                handle.write(block)
        got = sha256_file(part)
        if got != sha or part.stat().st_size != size:
            sys.exit(
                f"error: {url} gave {part.stat().st_size} bytes with SHA-256 {got}; expected {size} bytes and {sha}. "
                "The file changed, or the download was cut off."
            )
        part.replace(path)
        print(f"source: {name}, {size:,} bytes, SHA-256 {got}")
    return cache


def load_reference(folder: Path):
    """The reference `EomtForUniversalSegmentation` on the CPU in eval mode, built from `folder` (no missing, no unexpected key)."""
    import torch
    from transformers import EomtForUniversalSegmentation

    model, info = EomtForUniversalSegmentation.from_pretrained(str(folder), output_loading_info=True, dtype=torch.float32)
    if info["missing_keys"] or info["unexpected_keys"] or info.get("mismatched_keys"):
        sys.exit(f"error: the checkpoint and the model disagree: {info}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def write_safetensors(path: Path, tensors: dict, metadata: dict[str, str]) -> None:
    """Writes f16 tensors (numpy arrays) as a safetensors file, byte for byte the same on every run.

    Layout (https://github.com/huggingface/safetensors): an 8-byte little-endian header length, the JSON header
    (`__metadata__` first, then every tensor by name with `dtype`, `shape`, `data_offsets`), padded with spaces to a
    multiple of 8 bytes, then the tensor bytes in the order of the names. Tensors are stored little-endian f16.
    """
    import numpy as np

    header: dict = {"__metadata__": dict(sorted(metadata.items()))}
    offset = 0
    blobs = []
    for name in sorted(tensors):
        array = np.ascontiguousarray(tensors[name]).astype("<f2", copy=False)
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


def build_config(model, preprocessor: dict) -> dict:
    """What the runtime needs, read from the model the library built, and the reference's pre- and post-processing."""
    c = model.config
    embeddings = model.embeddings
    return {
        "hidden_size": c.hidden_size,
        "num_hidden_layers": c.num_hidden_layers,
        "num_attention_heads": c.num_attention_heads,
        "mlp_ratio": c.mlp_ratio,
        "hidden_act": c.hidden_act,
        "layer_norm_eps": c.layer_norm_eps,
        "layerscale_value": c.layerscale_value,
        "use_swiglu_ffn": c.use_swiglu_ffn,
        "patch_size": c.patch_size,
        "image_size": c.image_size,
        "grid_size": list(model.grid_size),
        "num_positions": embeddings.position_embeddings.num_embeddings,
        "num_register_tokens": c.num_register_tokens,
        "num_queries": c.num_queries,
        "num_blocks": c.num_blocks,
        "num_upscale_blocks": c.num_upscale_blocks,
        "num_labels": len(c.id2label),
        "attn_mask_probs": [float(v) for v in model.attn_mask_probs.tolist()],
        "id2label": [c.id2label[i] for i in range(len(c.id2label))],
        "preprocess": {
            "image_mean": preprocessor["image_mean"],
            "image_std": preprocessor["image_std"],
            "rescale_factor": preprocessor["rescale_factor"],
            "shortest_edge": preprocessor["size"]["shortest_edge"],
            "longest_edge": preprocessor["size"]["longest_edge"],
            "resample": preprocessor["resample"],
            "pad": "bottom and right, with 0 in 8-bit values, before the normalisation",
        },
        "panoptic": {"score_threshold": 0.8, "mask_threshold": 0.5, "overlap_mask_area_threshold": 0.8, "stuff_class_ids": [80, 132]},
    }


def check_config(config: dict) -> None:
    """Fails unless the model is the one `docs/34` section 4b.1 describes (and the runtime implements)."""
    want = {
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "mlp_ratio": 4,
        "hidden_act": "gelu",
        "layer_norm_eps": 1e-6,
        "layerscale_value": 1e-5,
        "use_swiglu_ffn": False,
        "patch_size": PATCH,
        "image_size": IMAGE,
        "grid_size": [GRID, GRID],
        "num_positions": GRID * GRID,
        "num_register_tokens": REGISTERS,
        "num_queries": QUERIES,
        "num_blocks": QUERY_BLOCKS,
        "num_upscale_blocks": UPSCALE_BLOCKS,
        "num_labels": LABELS,
        "attn_mask_probs": [0.0] * QUERY_BLOCKS,
    }
    wrong = [f"{k}: {config.get(k)!r}, expected {v!r}" for k, v in want.items() if config.get(k) != v]
    if wrong:
        sys.exit("error: the model is not the one the runtime implements:\n  " + "\n  ".join(wrong))


def synthetic_picture(torch):
    """A fixed 640 x 640 normalised input: smooth colour ramps and a few hard edges, no random numbers."""
    ys = torch.linspace(0.0, 1.0, IMAGE).view(IMAGE, 1)
    xs = torch.linspace(0.0, 1.0, IMAGE).view(1, IMAGE)
    r = (0.2 + 0.6 * xs + 0.0 * ys).expand(IMAGE, IMAGE)
    g = (0.8 - 0.5 * ys + 0.2 * torch.sin(12.0 * xs)).expand(IMAGE, IMAGE)
    b = (0.3 + 0.4 * ys * xs).expand(IMAGE, IMAGE)
    picture = torch.stack([r, g, b]).clamp(0.0, 1.0)
    picture[:, 300:420, 200:360] = torch.tensor([0.8, 0.2, 0.2]).view(3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return ((picture - mean) / std)[None]


def relative_rms(a, b, torch) -> tuple[float, float]:
    """The RMS of `a - b` over the RMS of `b`, and the largest absolute difference."""
    diff = (a.double() - b.double())
    return float(diff.pow(2).mean().sqrt() / b.double().pow(2).mean().sqrt()), float(diff.abs().max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path("build/eomt"), help="output directory (default build/eomt)")
    parser.add_argument("--cache", type=Path, default=None, help="where the three source files are kept (default OUT/cache)")
    parser.add_argument("--skip-f16-check", action="store_true", help="skip the run that measures what f16 storage costs")
    args = parser.parse_args()

    import numpy as np
    import safetensors
    import torch
    import transformers
    from safetensors import safe_open

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    cache = fetch_sources(args.cache or out / "cache")

    print(f"versions: torch {torch.__version__}, transformers {transformers.__version__}, safetensors {safetensors.__version__}, numpy {np.__version__}")
    model = load_reference(cache)
    preprocessor = json.loads((cache / "preprocessor_config.json").read_text(encoding="utf-8"))
    config = build_config(model, preprocessor)
    check_config(config)

    state = {}
    with safe_open(str(cache / "model.safetensors"), framework="pt") as handle:
        for name in handle.keys():
            state[name] = handle.get_tensor(name)
    shapes = expected_tensors()
    if parameter_count(shapes) != EXPECTED_PARAMETERS or len(shapes) != EXPECTED_TENSORS:
        sys.exit("error: expected_tensors() disagrees with EXPECTED_TENSORS and EXPECTED_PARAMETERS; fix the constants")
    wrong = sorted(set(state) ^ set(shapes))
    wrong += [f"{name}: {tuple(t.shape)}, expected {shapes[name]}" for name, t in state.items() if name in shapes and tuple(t.shape) != shapes[name]]
    if wrong:
        sys.exit("error: the checkpoint does not have the tensors the runtime reads:\n  " + "\n  ".join(wrong[:20]))
    total = sum(t.numel() for t in state.values())
    if total != EXPECTED_PARAMETERS:
        sys.exit(f"error: the checkpoint has {total:,} parameters; expected {EXPECTED_PARAMETERS:,}")
    if state["attn_mask_probs"].abs().max() != 0:
        sys.exit("error: attn_mask_probs is not all zero: inference would mask attention, which the runtime does not implement")

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
        "model": "eomt_coco_panoptic_small",
        "source_url": f"https://huggingface.co/{REPO}",
        "source_revision": REVISION,
        "source_sha256": CKPT_SHA256,
        "source_license": "MIT, Mobile Perception Systems Lab at TU/e (the model card's licence tag and the repository's LICENSE); backbone DINOv2, Apache-2.0",
        "upstream_url": UPSTREAM_URL,
        "reference_implementation": f"transformers {transformers.__version__} (EomtForUniversalSegmentation, EomtImageProcessor)",
        "kept": "every tensor of the checkpoint under its own name",
        "config": json.dumps(config, sort_keys=True, separators=(",", ":")),
    }
    target = out / OUT_NAME
    write_safetensors(target, {k: v.numpy() for k, v in f16.items()}, metadata)

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
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "safetensors": safetensors.__version__,
    }
    if not args.skip_f16_check:
        picture = synthetic_picture(torch)
        with torch.no_grad():
            before = model(pixel_values=picture)
            model.load_state_dict({name: t.float() for name, t in f16.items() if name != "attn_mask_probs"}, strict=False)
            after = model(pixel_values=picture)
        cls_rel, cls_max = relative_rms(after.class_queries_logits, before.class_queries_logits, torch)
        msk_rel, msk_max = relative_rms(after.masks_queries_logits, before.masks_queries_logits, torch)
        report["f16_cost_on_a_synthetic_picture"] = {
            "class_logits": {"relative_rms": cls_rel, "largest_abs": cls_max},
            "mask_logits": {"relative_rms": msk_rel, "largest_abs": msk_max},
        }
        print(f"f16 cost, class logits: relative RMS {cls_rel:.2e}, largest difference {cls_max:.2e}")
        print(f"f16 cost, mask logits: relative RMS {msk_rel:.2e}, largest difference {msk_max:.2e}")
    (out / "export_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"kept {len(f16)} tensors: {total:,} parameters")
    print(f"largest weight {largest:.3f}; worst relative f16 rounding {worst_relative:.2e}")
    print(f"wrote {target}: {size:,} bytes ({size / 1e6:.1f} MB)")
    print(f"{digest}  {OUT_NAME}")


if __name__ == "__main__":
    main()

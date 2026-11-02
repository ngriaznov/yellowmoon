#!/usr/bin/env python3
"""Converts Meta's SAM 2.1 *small* checkpoint into the safetensors file the Rust runtime reads (N-16, `docs/07`).

Python is for converting weights and for test fixtures only (`AGENTS.md` rule 6): this script runs in CI
(`.github/workflows/models.yml`), never in the app and never in `ym`. The result is published as an asset of a
Yellowmoon release and fetched by the build (`tools/fetch/sam2.sh`); it is not committed (too big for git).

    python3 -m training.sam2.export --out build/sam2            # downloads and checks the checkpoint into build/sam2/cache

What it does:

1. Downloads `sam2.1_hiera_small.pt` from the URL in the README of github.com/facebookresearch/sam2 and refuses a file
   whose SHA-256 is not the pinned one (`CKPT_SHA256`, computed on 2026-11-02 from the file at that URL).
2. Builds Meta's model on the CPU with `sam2.build_sam.build_sam2` (the `sam2` package at the pinned commit
   `SAM2_COMMIT`) and loads the checkpoint into it, which proves that the checkpoint and the configuration match.
3. Keeps the image path only: the image encoder (Hiera-S trunk and FPN neck), the prompt encoder, the mask decoder and
   `no_mem_embed` (added to the lowest-resolution feature map in the image predictor). Everything of the video model
   (memory attention, memory encoder, object pointers, temporal position encodings) is dropped, and the tensor names
   are kept as they are in the checkpoint.
4. Stores every kept tensor as f16 in a safetensors file whose header carries the configuration the runtime needs
   (image size, Hiera-S stages, windows, global attention, the neck, the prompt encoder, the decoder and the
   multimask rules, the pixel mean and standard deviation). The file is written by this script, not by the
   `safetensors` writer: that writer serialises the `__metadata__` map in a hash order that changes from process to
   process, and the file must be byte-identical on every run (`docs/algorithms/06` section 6.4).
5. Reads the file back with `safetensors` (a different implementation of the format) and compares it bit for bit with
   what was written, then runs Meta's encoder on a fixed synthetic picture with the original f32 weights and with the
   f16 weights widened to f32, and reports how far the f16 storage moves the maps (`export_report.json`).
6. Prints the parameter count, the file size and the SHA-256.

Versions are pinned in `training/sam2/requirements.txt` and in `models.yml` (torch, torchvision, safetensors, the
`sam2` commit).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import urllib.request
from pathlib import Path

#: The README of github.com/facebookresearch/sam2 links the SAM 2.1 checkpoints at this address (read 2026-11-02).
CKPT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
#: SHA-256 and size of the file at `CKPT_URL`, computed on 2026-11-02 (the file's Last-Modified is 2024-09-29).
CKPT_SHA256 = "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38"
CKPT_BYTES = 184_416_285
#: github.com/facebookresearch/sam2 `main` on 2026-11-02 (there are no tags). `setup.py` says version 1.0.
SAM2_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
#: The configuration `build_sam2` composes (relative to the `sam2` package).
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"

#: The published file and the version of its layout (`format` and `version` in the header).
OUT_NAME = "sam2.1_hiera_small.f16.safetensors"
FORMAT = "yellowmoon-sam2"
VERSION = "1"

#: Tensor names that are kept: the image path of the image predictor, nothing of the video model.
KEEP_PREFIXES = ("image_encoder.", "sam_prompt_encoder.", "sam_mask_decoder.")
KEEP_NAMES = ("no_mem_embed",)
#: What the pinned checkpoint holds, as counted from its pickle on 2026-11-02 (519 tensors, 46,060,610 parameters in all).
EXPECTED_KEPT_TENSORS = 359
EXPECTED_KEPT_PARAMETERS = 38_538_833
EXPECTED_ALL_TENSORS = 519
EXPECTED_ALL_PARAMETERS = 46_060_610


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, read in 1 MiB pieces."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_checkpoint(cache: Path) -> Path:
    """Downloads the checkpoint into `cache` unless a file with the pinned hash is already there."""
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "sam2.1_hiera_small.pt"
    if path.exists() and sha256_file(path) == CKPT_SHA256:
        print(f"checkpoint: cached, {path.stat().st_size:,} bytes, SHA-256 {CKPT_SHA256}")
        return path
    part = path.with_name(path.name + ".part")
    print(f"checkpoint: downloading {CKPT_URL}")
    with urllib.request.urlopen(CKPT_URL, timeout=120) as response, part.open("wb") as handle:  # noqa: S310 (a fixed https URL)
        while block := response.read(1 << 20):
            handle.write(block)
    got = sha256_file(part)
    if got != CKPT_SHA256 or part.stat().st_size != CKPT_BYTES:
        sys.exit(
            f"error: {CKPT_URL} gave {part.stat().st_size} bytes with SHA-256 {got}; "
            f"expected {CKPT_BYTES} bytes and {CKPT_SHA256}. Meta changed the file, or the download was cut off."
        )
    part.replace(path)
    print(f"checkpoint: {path.stat().st_size:,} bytes, SHA-256 {got}")
    return path


def is_kept(name: str) -> bool:
    """True for a tensor of the image path."""
    return name.startswith(KEEP_PREFIXES) or name in KEEP_NAMES


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


def build_config(model, cfg, transforms) -> dict:
    """The configuration of the image path, read from the model Meta's code built (not typed in), so that a mismatch fails here."""
    trunk = model.image_encoder.trunk
    neck = model.image_encoder.neck
    prompt = model.sam_prompt_encoder
    decoder = model.sam_mask_decoder
    transformer = decoder.transformer
    rate = transformer.embedding_dim // transformer.final_attn_token_to_image.q_proj.out_features
    blocks = [
        {
            "index": index,
            "dim_in": block.dim,
            "dim_out": block.dim_out,
            "num_heads": block.attn.num_heads,
            "window_size": block.window_size,  # 0 is global attention
            "query_pool": block.q_stride is not None,  # 2 x 2 max pooling of the queries at the start of the block
            "input_projection": block.dim != block.dim_out,  # the block has `proj` (a linear layer on the shortcut)
        }
        for index, block in enumerate(trunk.blocks)
    ]
    return {
        "model": "sam2.1_hiera_small",
        "image_size": model.image_size,
        "backbone_stride": model.backbone_stride,
        "image_embedding_size": list(prompt.image_embedding_size),
        "low_res_mask_size": 4 * prompt.image_embedding_size[0],
        "logit_clamp": 32.0,
        "mask_threshold": 0.0,
        "pixel_mean": list(transforms.mean),
        "pixel_std": list(transforms.std),
        "resize": {"mode": "bilinear", "antialias": True, "keep_aspect_ratio": False, "note": "torchvision Resize((1024, 1024)) on a tensor"},
        "trunk": {
            "type": "Hiera",
            "embed_dim": cfg.model.image_encoder.trunk.embed_dim,
            "num_heads_first_stage": cfg.model.image_encoder.trunk.num_heads,
            "stages": list(cfg.model.image_encoder.trunk.stages),
            "stage_ends": list(trunk.stage_ends),
            "window_spec": list(trunk.window_spec),
            "global_att_blocks": list(trunk.global_att_blocks),
            "q_pool_blocks": list(trunk.q_pool_blocks),
            "q_stride": list(trunk.q_stride),
            "window_pos_embed_bkg_spatial_size": list(trunk.window_pos_embed_bkg_spatial_size),
            "channel_list": list(trunk.channel_list),
            "patch_embed": {"kernel": 7, "stride": 4, "padding": 3, "in_chans": 3},
            "mlp_ratio": 4.0,
            "mlp_activation": "gelu (exact, erf)",
            "layer_norm_eps": trunk.blocks[0].norm1.eps,
            "return_interm_layers": trunk.return_interm_layers,
            "blocks": blocks,
        },
        "neck": {
            "type": "FpnNeck",
            "d_model": neck.d_model,
            "backbone_channel_list": list(neck.backbone_channel_list),
            "fpn_top_down_levels": list(neck.fpn_top_down_levels),
            "fpn_interp_model": neck.fpn_interp_model,
            "fuse_type": neck.fuse_type,
            "scalp": model.image_encoder.scalp,
            "position_encoding": {
                "type": "PositionEmbeddingSine",
                "num_pos_feats": 2 * neck.position_encoding.num_pos_feats,
                "temperature": neck.position_encoding.temperature,
                "normalize": neck.position_encoding.normalize,
                "scale": neck.position_encoding.scale,
            },
            "note": "outputs after scalp: strides 4, 8, 16; conv_s0 and conv_s1 of the decoder project strides 4 and 8; no_mem_embed is added to stride 16 (directly_add_no_mem_embed)",
        },
        "directly_add_no_mem_embed": model.directly_add_no_mem_embed,
        "use_high_res_features_in_sam": model.use_high_res_features_in_sam,
        "prompt_encoder": {
            "embed_dim": prompt.embed_dim,
            "input_image_size": list(prompt.input_image_size),
            "mask_in_chans": 16,
            "num_point_embeddings": prompt.num_point_embeddings,
            "note": "point labels: 0 negative, 1 positive, 2 and 3 the box corners, -1 padding (not_a_point_embed)",
        },
        "mask_decoder": {
            "transformer_dim": decoder.transformer_dim,
            "num_multimask_outputs": decoder.num_multimask_outputs,
            "num_mask_tokens": decoder.num_mask_tokens,
            "transformer": {
                "depth": transformer.depth,
                "embedding_dim": transformer.embedding_dim,
                "num_heads": transformer.num_heads,
                "mlp_dim": transformer.mlp_dim,
                "attention_downsample_rate": rate,
            },
            "upscaling_activation": "gelu (exact, erf)",
            "layer_norm_2d_eps": decoder.output_upscaling[1].eps,
            "iou_head_depth": 3,
            "iou_head_hidden_dim": 256,
            "iou_prediction_use_sigmoid": decoder.iou_prediction_head.sigmoid_output,
            "pred_obj_scores": decoder.pred_obj_scores,
            "pred_obj_scores_mlp": True,
            "use_high_res_features": decoder.use_high_res_features,
        },
        "multimask": {
            "multimask_output_in_sam": model.multimask_output_in_sam,
            "multimask_min_pt_num": model.multimask_min_pt_num,
            "multimask_max_pt_num": model.multimask_max_pt_num,
            "multimask_output_for_tracking": model.multimask_output_for_tracking,
            "dynamic_multimask_via_stability": decoder.dynamic_multimask_via_stability,
            "dynamic_multimask_stability_delta": decoder.dynamic_multimask_stability_delta,
            "dynamic_multimask_stability_thresh": decoder.dynamic_multimask_stability_thresh,
            "note": "the three dynamic values are the ones sam2.build_sam.build_sam2 sets with apply_postprocessing=True (its default); a single positive point is decoded with multimask_output on, anything else with it off",
        },
    }


def check_config(config: dict) -> None:
    """Fails when the model Meta's code built is not the one `docs/algorithms/06` section 6.1 describes."""
    trunk, neck, decoder = config["trunk"], config["neck"], config["mask_decoder"]
    expect = {
        "image_size": (config["image_size"], 1024),
        "embed_dim": (trunk["embed_dim"], 96),
        "stages": (trunk["stages"], [1, 2, 11, 2]),
        "stage_ends": (trunk["stage_ends"], [0, 2, 13, 15]),
        "global_att_blocks": (trunk["global_att_blocks"], [7, 10, 13]),
        "window_spec": (trunk["window_spec"], [8, 4, 14, 7]),
        "channel_list": (trunk["channel_list"], [768, 384, 192, 96]),
        "neck d_model": (neck["d_model"], 256),
        "fpn_top_down_levels": (neck["fpn_top_down_levels"], [2, 3]),
        "fpn_interp_model": (neck["fpn_interp_model"], "nearest"),
        "scalp": (neck["scalp"], 1),
        "decoder depth": (decoder["transformer"]["depth"], 2),
        "decoder heads": (decoder["transformer"]["num_heads"], 8),
        "decoder mlp_dim": (decoder["transformer"]["mlp_dim"], 2048),
        "decoder downsample rate": (decoder["transformer"]["attention_downsample_rate"], 2),
        "dynamic stability thresh": (config["multimask"]["dynamic_multimask_stability_thresh"], 0.98),
        "dynamic stability delta": (config["multimask"]["dynamic_multimask_stability_delta"], 0.05),
    }
    wrong = [f"{name}: {got!r}, expected {want!r}" for name, (got, want) in expect.items() if got != want]
    if wrong:
        sys.exit("error: the model is not SAM 2.1 small as the runtime spec says:\n  " + "\n  ".join(wrong))


def synthetic_picture(torch):
    """A fixed 1 x 3 x 1024 x 1024 input with structure at several scales, normalised like Meta's transform; no random numbers."""
    axis = torch.linspace(0.0, 1.0, 1024, dtype=torch.float64)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    red = 0.5 + 0.3 * torch.sin(14.0 * xx + 3.0 * yy) + 0.2 * torch.cos(60.0 * yy)
    green = 0.5 + 0.3 * torch.sin(9.0 * yy - 2.0 * xx) * torch.cos(23.0 * xx)
    blue = 0.5 + 0.25 * torch.cos(5.0 * (xx - 0.5) ** 2 * 40.0) + 0.2 * (xx > 0.5).double() * (yy < 0.5).double()
    picture = torch.stack([red, green, blue]).clamp(0.0, 1.0).float()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return ((picture - mean) / std)[None]


def encoder_maps(model, torch, picture) -> dict:
    """The three maps Meta's image predictor keeps: the embedding (with `no_mem_embed`) and the two high-resolution maps."""
    with torch.no_grad():
        out = model.forward_image(picture)
        fpn = out["backbone_fpn"]
        embed = fpn[-1] + model.no_mem_embed.view(1, -1, 1, 1)
    return {"image_embed": embed, "high_res_0": fpn[0], "high_res_1": fpn[1]}


def relative_rms(a, b, torch) -> tuple[float, float]:
    """(RMS of a - b divided by the RMS of b, largest absolute difference)."""
    diff = (a - b).double()
    return float(diff.pow(2).mean().sqrt() / b.double().pow(2).mean().sqrt()), float(diff.abs().max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path("build/sam2"), help="output directory (default build/sam2)")
    parser.add_argument("--cache", type=Path, default=None, help="where the checkpoint is kept (default OUT/cache)")
    parser.add_argument("--skip-f16-check", action="store_true", help="skip the encoder run that measures what f16 storage costs")
    args = parser.parse_args()

    import numpy as np
    import torch
    import safetensors
    from safetensors import safe_open

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    ckpt = fetch_checkpoint(args.cache or out / "cache")

    from omegaconf import OmegaConf
    from hydra import compose
    from sam2.build_sam import build_sam2
    from sam2.utils.transforms import SAM2Transforms

    print(f"versions: torch {torch.__version__}, safetensors {safetensors.__version__}, numpy {np.__version__}, sam2 commit {SAM2_COMMIT}")
    model = build_sam2(SAM2_CONFIG, str(ckpt), device="cpu")  # loads the checkpoint with strict=True semantics (it raises on a mismatch)
    model.eval()
    cfg = compose(config_name=SAM2_CONFIG)
    OmegaConf.resolve(cfg)
    config = build_config(model, cfg, SAM2Transforms(resolution=model.image_size, mask_threshold=0.0))
    check_config(config)

    state = torch.load(ckpt, map_location="cpu", weights_only=True)["model"]
    total = sum(t.numel() for t in state.values())
    if len(state) != EXPECTED_ALL_TENSORS or total != EXPECTED_ALL_PARAMETERS:
        sys.exit(f"error: the checkpoint has {len(state)} tensors and {total:,} parameters; expected {EXPECTED_ALL_TENSORS} and {EXPECTED_ALL_PARAMETERS:,}")
    kept = {name: tensor for name, tensor in sorted(state.items()) if is_kept(name)}
    dropped = sorted(name for name in state if not is_kept(name))
    kept_parameters = sum(t.numel() for t in kept.values())
    if len(kept) != EXPECTED_KEPT_TENSORS or kept_parameters != EXPECTED_KEPT_PARAMETERS:
        sys.exit(f"error: kept {len(kept)} tensors and {kept_parameters:,} parameters; expected {EXPECTED_KEPT_TENSORS} and {EXPECTED_KEPT_PARAMETERS:,}")

    f16 = {}
    worst_relative = 0.0
    largest = 0.0
    for name, tensor in kept.items():
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
            relative = ((narrow.float() - tensor).abs()[nonzero] / tensor.abs()[nonzero])
            relative = relative[tensor.abs()[nonzero] > 6.2e-5]  # f16's smallest normal number; below it the grid is absolute
            if relative.numel():
                worst_relative = max(worst_relative, float(relative.max()))

    metadata = {
        "format": FORMAT,
        "version": VERSION,
        "dtype": "F16",
        "model": "sam2.1_hiera_small",
        "source_url": CKPT_URL,
        "source_sha256": CKPT_SHA256,
        "source_license": "Apache-2.0, Meta Platforms, Inc. and affiliates",
        "sam2_commit": SAM2_COMMIT,
        "kept": "image_encoder.*, sam_prompt_encoder.*, sam_mask_decoder.*, no_mem_embed",
        "dropped": "memory_attention, memory_encoder, maskmem_tpos_enc, no_mem_pos_enc, no_obj_ptr, no_obj_embed_spatial, mask_downsample, obj_ptr_proj, obj_ptr_tpos_proj",
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
        "parameters": kept_parameters,
        "parameters_in_checkpoint": total,
        "dropped_tensors": len(dropped),
        "dropped_parameters": total - kept_parameters,
        "largest_abs_value": largest,
        "worst_relative_f16_rounding": worst_relative,
        "source_sha256": CKPT_SHA256,
        "sam2_commit": SAM2_COMMIT,
        "torch": torch.__version__,
        "safetensors": safetensors.__version__,
    }
    if not args.skip_f16_check:
        picture = synthetic_picture(torch)
        before = encoder_maps(model, torch, picture)
        model.load_state_dict({name: t.float() for name, t in f16.items()}, strict=False)
        after = encoder_maps(model, torch, picture)
        report["f16_cost_on_a_synthetic_picture"] = {}
        for key in before:
            rel, largest_diff = relative_rms(after[key], before[key], torch)
            report["f16_cost_on_a_synthetic_picture"][key] = {"relative_rms": rel, "largest_abs": largest_diff}
            print(f"f16 cost, {key}: RMS of the difference is {rel:.2e} of the map's RMS, largest difference {largest_diff:.2e}")
    (out / "export_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"kept {len(f16)} of {len(state)} tensors: {kept_parameters:,} of {total:,} parameters (the {len(dropped)} dropped tensors are the video model)")
    print(f"largest weight {largest:.3f}; worst relative f16 rounding {worst_relative:.2e}")
    print(f"wrote {target}: {size:,} bytes ({size / 1e6:.1f} MB)")
    print(f"{digest}  {OUT_NAME}")


if __name__ == "__main__":
    main()

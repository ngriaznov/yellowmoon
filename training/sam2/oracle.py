#!/usr/bin/env python3
"""Fixtures for the SAM 2 gate of N-17 (`docs/algorithms/06-masks.md` section 6.5, `docs/13`).

Python is for test fixtures only (`AGENTS.md` rule 6): this script runs in CI (`.github/workflows/models.yml`) and
writes data that the Rust tests of `ym_ml::sam2` read; neither the app nor `ym` nor any shipped crate runs it.

    python3 -m training.sam2.oracle --weights build/sam2/sam2.1_hiera_small.f16.safetensors \\
        --checkpoint build/sam2/cache/sam2.1_hiera_small.pt --photos build/sam2/photos --out build/sam2/fixtures

For each of the 20 public photos of `oracle_photos.json` (CC0 or public domain; URL, licence, author and SHA-256 are
listed there, and the photos are not committed) it:

1. downloads the photo (or finds it in `--photos`) and checks its SHA-256;
2. makes the 8-bit picture N-17's runtime starts from: EXIF orientation applied, RGB, the long edge resized to
   1024 px with Pillow's Lanczos filter (the colour profile is ignored: the picture is taken as sRGB);
3. runs Meta's official `SAM2ImagePredictor` (the `sam2` package at the pinned commit, SAM 2.1 small, CPU, f32, one
   thread, `torch.use_deterministic_algorithms`, no autocast) on it, **with the weights of the published safetensors
   file widened to f32**, so that the fixtures are what the shipped weights give in Meta's code and N-17's gate
   measures our runtime, not the f16 storage;
4. does the same with Meta's original f32 checkpoint and writes the difference (`report.json`: the encoder's maps,
   the logits, the IoU of the masks), so that the cost of the f16 storage is known and reported, never hidden.

One `.npz` per photo (written uncompressed with fixed timestamps, so the bytes do not depend on the time; all floats
are f32), `<id>.npz`:

    picture            u8   [H, W, 3]       the 8-bit picture (long edge 1024)
    input_tensor       f32  [1, 3, 1024, 1024]   Meta's tensor: resize to 1024 x 1024 (not keeping the aspect ratio,
                                                 torchvision bilinear with antialiasing), then (x - mean) / std
    image_embed        f32  [1, 256, 64, 64]     the image embedding, `no_mem_embed` already added
    high_res_feat_0    f32  [1, 32, 256, 256]    the decoder's high-resolution map at stride 4 (after `conv_s0`)
    high_res_feat_1    f32  [1, 64, 128, 128]    and at stride 8 (after `conv_s1`)
    prompt{k}_points   f32  [N, 2]     k = 1, 2, 3; the points as fractions (x, y) of the picture, 0 to 1 (N = 0 for a box)
    prompt{k}_labels   i32  [N]        1 positive, 0 negative
    prompt{k}_box      f32  [4] or [0] x0, y0, x1, y1 as fractions
    prompt{k}_multimask i32 [1]        1 when the decoder ran with multimask_output on
    prompt{k}_low_res_logits f32 [C, 256, 256]   the decoder's logits clamped to +-32, C = 3 (multimask) or 1
    prompt{k}_iou      f32  [C]
    prompt{k}_object_score f32 [1]     the decoder's object score logit
    prompt{k}_chosen   i32  [1]        the mask taken: the highest predicted IoU
    prompt{k}_mask     u8   [H, W]     the chosen mask at the picture's size, 1 where the upsampled logit is above 0

Prompt 1 is one positive click (decoded with multimask on), prompt 2 a positive and a negative click (multimask off, with
Meta's dynamic fall-back to the best multimask output when the single mask is unstable), prompt 3 a box (multimask off).
The coordinates are in `oracle_photos.json`. Point and box fractions times 1024 are what Meta's `transform_coords`
makes of them (`normalize_coords=False`).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHOTOS_JSON = HERE / "oracle_photos.json"
#: Wikimedia asks for a User-Agent that names the tool and a way to reach its owner (https://meta.wikimedia.org/wiki/User-Agent_policy).
USER_AGENT = "yellowmoon-oracle/1 (https://github.com/ngriaznov/yellowmoon; fixtures for the SAM 2 runtime test)"
PICTURE_LONG_EDGE = 1024
FIXTURE_VERSION = "1"
ZIP_TIME = (2026, 11, 2, 0, 0, 0)
PLACEHOLDER = "TBD-after-first-run"


def sha256_bytes(data: bytes) -> str:
    """SHA-256 of a byte string."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, read in 1 MiB pieces."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha1_bytes(data: bytes) -> str:
    """SHA-1 of a byte string (Wikimedia Commons states the SHA-1 of every file)."""
    return hashlib.sha1(data).hexdigest()  # noqa: S324 (Commons' own hash of the file, used as a second check, not for security)


def check_photo(photo: dict, data: bytes) -> str:
    """The SHA-256 of a photo's bytes, after checking them against the list; stops the run when they do not match.

    A photo is pinned by `sha256`. While that is still the placeholder (the list was written where Wikimedia could not
    be reached), the file must at least have the SHA-1 that Commons states (`commons_sha1`), and the SHA-256 found is
    printed and kept in `manifest.json` of the fixtures so that it can be copied into the list.
    """
    got = sha256_bytes(data)
    want = photo.get("sha256", PLACEHOLDER)
    if want == PLACEHOLDER:
        if sha1_bytes(data) != photo["commons_sha1"]:
            sys.exit(f"error: {photo['url']} does not have the SHA-1 {photo['commons_sha1']} that Commons states for it (the file changed)")
        print(f"  {photo['id']}: SHA-256 is not pinned yet (Commons' SHA-1 matches); it is {got}")
    elif got != want:
        sys.exit(f"error: {photo['url']} has SHA-256 {got}, the list pins {want}. The file on Wikimedia Commons changed; check its page and its licence.")
    return got


def fetch_photo(photo: dict, directory: Path) -> Path:
    """The photo's file in `directory`, downloaded when it is not there; its SHA-256 is checked against the list.

    Wikimedia answers 429 to clients that go too fast: the download waits for `Retry-After` (at least 20 s) and tries
    again, and photos are fetched 3 s apart.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{photo['id']}.jpg"
    if path.exists():
        try:
            check_photo(photo, path.read_bytes())
            return path
        except SystemExit:
            path.unlink()  # a damaged copy in the cache: fetch it again
    last = None
    for attempt in range(8):
        request = urllib.request.Request(photo["url"], headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 (https URL from the pinned list)
                data = response.read()
            break
        except urllib.error.HTTPError as error:
            last = error
            if error.code not in (429, 503):
                sys.exit(f"error: {photo['url']} answered {error.code}")
            header = error.headers.get("Retry-After", "")
            wait = min(900, max(20, int(header))) if header.isdigit() else 30
            print(f"  {photo['id']}: HTTP {error.code}, waiting {wait} s (try {attempt + 1} of 8)")
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError) as error:
            last = error
            print(f"  {photo['id']}: {error}, waiting 20 s (try {attempt + 1} of 8)")
            time.sleep(20)
    else:
        sys.exit(f"error: {photo['url']} could not be fetched: {last}")
    check_photo(photo, data)
    path.write_bytes(data)
    time.sleep(3)
    return path


def load_picture(path: Path):
    """The 8-bit picture of a photo: uint8 [H, W, 3], long edge 1024."""
    import numpy as np
    from PIL import Image, ImageOps

    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    width, height = image.size
    scale = PICTURE_LONG_EDGE / max(width, height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = image.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.uint8).copy()


def write_npz(path: Path, arrays: dict) -> None:
    """An uncompressed .npz whose bytes do not depend on the time (numpy's own writer stamps the current time)."""
    import numpy as np

    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.ascontiguousarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", date_time=ZIP_TIME)
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            archive.writestr(info, buffer.getvalue())


def build_models(checkpoint: Path, weights: Path):
    """(the model with the exported f16 weights widened to f32, the model with Meta's original f32 weights)."""
    import torch
    from safetensors.torch import load_file
    from sam2.build_sam import build_sam2

    config = "configs/sam2.1/sam2.1_hiera_s.yaml"
    original = build_sam2(config, str(checkpoint), device="cpu")
    shipped = build_sam2(config, str(checkpoint), device="cpu")
    stored = load_file(str(weights))
    result = shipped.load_state_dict({name: tensor.float() for name, tensor in stored.items()}, strict=False)
    if result.unexpected_keys:
        sys.exit(f"error: the weights file has tensors the model does not: {result.unexpected_keys[:5]}")
    video = ("memory_attention.", "memory_encoder.", "maskmem_tpos_enc", "no_mem_pos_enc", "no_obj_ptr", "no_obj_embed_spatial", "mask_downsample.", "obj_ptr_proj.", "obj_ptr_tpos_proj.")
    missing = [key for key in result.missing_keys if not key.startswith(video)]
    if missing:
        sys.exit(f"error: the weights file lacks tensors of the image path: {missing[:5]}")
    for model in (original, shipped):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return shipped, original


class Run:
    """Meta's image predictor on one picture, and the object score of every decoder call (read with a hook)."""

    def __init__(self, model, predictor_class):
        self.model = model
        self.predictor = predictor_class(model)  # mask_threshold 0, no hole filling: Meta's defaults
        self.object_score = None
        model.sam_mask_decoder.register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        self.object_score = output[3].detach().float().reshape(-1)[0].item()

    def set_image(self, picture):
        import torch

        with torch.no_grad():
            self.predictor.set_image(picture)
            tensor = self.predictor._transforms(picture)[None].float()  # the tensor set_image made, made again from the same picture
        features = self.predictor._features
        return {
            "input_tensor": tensor.numpy(),
            "image_embed": features["image_embed"].detach().float().numpy(),
            "high_res_feat_0": features["high_res_feats"][0].detach().float().numpy(),
            "high_res_feat_1": features["high_res_feats"][1].detach().float().numpy(),
        }

    def predict(self, points, labels, box, multimask):
        import numpy as np

        masks, iou, low_res = self.predictor.predict(
            point_coords=points if len(points) else None,
            point_labels=labels if len(points) else None,
            box=box if len(box) else None,
            multimask_output=multimask,
            return_logits=True,
            normalize_coords=False,
        )
        chosen = int(np.argmax(iou))
        return {
            "low_res_logits": np.asarray(low_res, dtype=np.float32),
            "iou": np.asarray(iou, dtype=np.float32),
            "object_score": np.asarray([self.object_score], dtype=np.float32),
            "chosen": np.asarray([chosen], dtype=np.int32),
            "mask": (np.asarray(masks[chosen]) > 0.0).astype(np.uint8),
        }


def prompts_of(photo: dict) -> list[dict]:
    """The three prompts of a photo as arrays (fractions of the picture)."""
    import numpy as np

    spec = photo["prompts"]
    empty_points = np.zeros((0, 2), dtype=np.float32)
    empty_labels = np.zeros((0,), dtype=np.int32)
    empty_box = np.zeros((0,), dtype=np.float32)
    pair = spec["positive_negative"]
    return [
        {
            "points": np.asarray([spec["point"]], dtype=np.float32),
            "labels": np.asarray([1], dtype=np.int32),
            "box": empty_box,
            "multimask": True,
        },
        {
            "points": np.asarray([pair["positive"], pair["negative"]], dtype=np.float32),
            "labels": np.asarray([1, 0], dtype=np.int32),
            "box": empty_box,
            "multimask": False,
        },
        {"points": empty_points, "labels": empty_labels, "box": np.asarray(spec["box"], dtype=np.float32), "multimask": False},
    ]


def iou_of(a, b) -> float:
    """Intersection over union of two boolean masks (1.0 when both are empty)."""
    import numpy as np

    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


def relative_rms(a, b) -> float:
    """RMS of a - b divided by the RMS of b, in f64."""
    import numpy as np

    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(b**2)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", type=Path, required=True, help="the exported sam2.1_hiera_small.f16.safetensors")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Meta's sam2.1_hiera_small.pt (the original f32 weights)")
    parser.add_argument("--photos", type=Path, default=Path("build/sam2/photos"), help="where the photos are kept (downloaded when missing)")
    parser.add_argument("--out", type=Path, default=Path("build/sam2/fixtures"), help="output directory")
    parser.add_argument("--only", default=None, help="comma-separated photo ids (a quick local run)")
    args = parser.parse_args()

    import numpy as np
    import torch
    import torchvision
    import PIL

    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    listing = json.loads(PHOTOS_JSON.read_text(encoding="utf-8"))
    photos = listing["photos"]
    if args.only:
        wanted = set(args.only.split(","))
        photos = [photo for photo in photos if photo["id"] in wanted]
    if len(photos) != 20 and not args.only:
        sys.exit(f"error: {PHOTOS_JSON.name} lists {len(photos)} photos, the gate needs 20")

    print(f"versions: torch {torch.__version__}, torchvision {torchvision.__version__}, numpy {np.__version__}, Pillow {PIL.__version__}")
    shipped, original = build_models(args.checkpoint, args.weights)
    shipped_run = Run(shipped, SAM2ImagePredictor)
    original_run = Run(original, SAM2ImagePredictor)
    args.out.mkdir(parents=True, exist_ok=True)

    files = {}
    report = {"photos": {}, "worst": {}}
    worst = {"encoder_relative_rms": 0.0, "logit_max_abs": 0.0, "logit_mean_abs": 0.0, "min_mask_iou_256": 1.0, "chosen_differs": 0}
    for photo in photos:
        path = fetch_photo(photo, args.photos)
        picture = load_picture(path)
        height, width = picture.shape[:2]
        arrays = {"picture": picture}
        arrays.update(shipped_run.set_image(picture))
        reference = original_run.set_image(picture)
        entry = {
            "picture_size": [height, width],
            "f16_vs_f32_encoder_relative_rms": {key: relative_rms(arrays[key], reference[key]) for key in ("image_embed", "high_res_feat_0", "high_res_feat_1")},
            "prompts": [],
        }
        worst["encoder_relative_rms"] = max(worst["encoder_relative_rms"], *entry["f16_vs_f32_encoder_relative_rms"].values())
        for index, prompt in enumerate(prompts_of(photo), start=1):
            got = shipped_run.predict(prompt["points"], prompt["labels"], prompt["box"], prompt["multimask"])
            ref = original_run.predict(prompt["points"], prompt["labels"], prompt["box"], prompt["multimask"])
            arrays[f"prompt{index}_points"] = prompt["points"]
            arrays[f"prompt{index}_labels"] = prompt["labels"]
            arrays[f"prompt{index}_box"] = prompt["box"]
            arrays[f"prompt{index}_multimask"] = np.asarray([1 if prompt["multimask"] else 0], dtype=np.int32)
            for key, value in got.items():
                arrays[f"prompt{index}_{key}"] = value
            same_shape = got["low_res_logits"].shape == ref["low_res_logits"].shape
            difference = np.abs(got["low_res_logits"] - ref["low_res_logits"]) if same_shape else None
            low_got = got["low_res_logits"][int(got["chosen"][0])] > 0
            low_ref = ref["low_res_logits"][int(ref["chosen"][0])] > 0 if same_shape else None
            record = {
                "prompt": index,
                "multimask": prompt["multimask"],
                "object_score": float(got["object_score"][0]),
                "chosen": int(got["chosen"][0]),
                "mask_area_fraction": float(got["mask"].mean()),
                "f16_vs_f32_logit_max_abs": None if difference is None else float(difference.max()),
                "f16_vs_f32_logit_mean_abs": None if difference is None else float(difference.mean()),
                "f16_vs_f32_mask_iou_256": None if low_ref is None else iou_of(low_got, low_ref),
                "f16_vs_f32_chosen_differs": None if not same_shape else int(got["chosen"][0]) != int(ref["chosen"][0]),
            }
            if difference is not None:
                worst["logit_max_abs"] = max(worst["logit_max_abs"], record["f16_vs_f32_logit_max_abs"])
                worst["logit_mean_abs"] = max(worst["logit_mean_abs"], record["f16_vs_f32_logit_mean_abs"])
                worst["min_mask_iou_256"] = min(worst["min_mask_iou_256"], record["f16_vs_f32_mask_iou_256"])
                worst["chosen_differs"] += int(record["f16_vs_f32_chosen_differs"])
            else:
                worst["chosen_differs"] += 1
            entry["prompts"].append(record)
        target = args.out / f"{photo['id']}.npz"
        write_npz(target, arrays)
        files[target.name] = {"bytes": target.stat().st_size, "sha256": sha256_file(target)}
        report["photos"][photo["id"]] = entry
        print(
            f"{photo['id']:12} {width}x{height}  "
            + "  ".join(f"p{r['prompt']}: score {r['object_score']:+.2f} area {r['mask_area_fraction']:.3f}" for r in entry["prompts"])
        )
    report["worst"] = worst
    report["note"] = (
        "f16_vs_f32: Meta's code with the published f16 weights widened to f32 (what the fixtures hold) against Meta's original f32 checkpoint. "
        "The encoder numbers are the RMS of the difference over the RMS of the original map; the logits are the decoder's, clamped to +-32; "
        "the IoU is of the chosen masks at logit > 0 at 256 x 256."
    )
    (args.out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "fixture_version": FIXTURE_VERSION,
        "weights": args.weights.name,
        "weights_sha256": sha256_file(args.weights),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "picture_long_edge_px": PICTURE_LONG_EDGE,
        "photos": {
            photo["id"]: {"title": photo["title"], "license": photo["license"], "page": photo["page"], "sha256": sha256_file(args.photos / f"{photo['id']}.jpg")}
            for photo in photos
        },
        "files": files,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    total = sum(info["bytes"] for info in files.values())
    print(f"wrote {len(files)} fixtures, {total / 1e6:.0f} MB, to {args.out}")
    print(
        "f16 against f32 weights, worst over all photos and prompts: "
        f"encoder RMS {worst['encoder_relative_rms']:.2e} of the map's RMS, logits max {worst['logit_max_abs']:.3f} / mean {worst['logit_mean_abs']:.4f}, "
        f"mask IoU at least {worst['min_mask_iou_256']:.4f}, {worst['chosen_differs']} choices differ"
    )


if __name__ == "__main__":
    main()

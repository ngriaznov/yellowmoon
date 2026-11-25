#!/usr/bin/env python3
"""Writes the parity fixture of the EoMT small runtime (RG-03, `docs/34` sections 4b.1 and 10): what the original model gives for a few
small pictures, for `crates/ym-ml/tests/eomt.rs` to compare with.

Test tooling only (`AGENTS.md` rule 6): Python never runs in the app, in `ym` or in a shipped crate. The reference is Hugging Face's
`EomtForUniversalSegmentation` and `EomtImageProcessor` (a port of the authors' code, github.com/tue-mps/eomt, MIT), on the CPU in f32
with one thread, loaded with **the f16 weights of `training/eomt/export.py` widened to f32**, so that the comparison measures our
runtime and not the f16 storage. The weights' own cost (f16 against the checkpoint's f32) is measured by `export.py` and reported
in `data/models/eomt/MODEL_CARD.md`.

    python3 tools/eomt/parity_fixture.py \\
        --source build/eomt/cache --weights build/eomt/eomt_coco_panoptic_small.f16.safetensors \\
        --out crates/ym-ml/tests/data/eomt

`--source` is the folder with `config.json` and `preprocessor_config.json` of the checkpoint (`export.py` fills `OUT/cache`). The
three pictures are made here, procedurally and without random numbers beyond a fixed seed (a landscape, a street with a person
and a shore), so the fixture has no photograph and no licence question; they are written as PNG files next to the JSON. Their
sizes differ (3:2, 2:3 and square) so that the fit and the padding are exercised.

What is written, per picture (`parity.json`; the sizes of the sets are chosen so that the whole fixture is a few hundred KB):

* `input_samples`: the 8-bit values of the model's input (after the resize and the padding, before the normalisation, recovered
  from the processor's `pixel_values`) at every `row_stride`-th row and `col_stride`-th column, all three channels;
* `queries`, `class_logits`: the class logits (134 values) of 16 of the 200 queries: every query the post-processing keeps, then
  the queries with the largest probability of a class other than "none";
* `mask_samples`: the mask logits of the same queries on the 160 x 160 grid at a 10 x 10 set of points;
* `kept`: the queries the post-processing keeps with their class and score;
* `classes_png`: the label map of the post-processing at the fit size, as an 8-bit PNG (class ids 0 to 132, 133 for "unlabelled"),
  made by the processor's `post_process_panoptic_segmentation` with `stuff_classes = range(80, 133)` as the POC did.

With `--photos DIR --photos-out DIR2` the same is made for the pictures of DIR (any image files), for a local check on real
photographs; the photos are copied to DIR2 as PNG and nothing of them should be committed unless their licence allows it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROW_STRIDE = 37
COL_STRIDE = 41
QUERY_COUNT = 16
MASK_OFFSET = 8
MASK_STRIDE = 16
MASK_COUNT = 10
NONE_CLASS = 133
STUFF_CLASSES = list(range(80, 133))
IMAGE_PX = 640


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, read in 1 MiB pieces."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------------------------------------------------------
# The pictures
# ---------------------------------------------------------------------------------------------------------------------------


def _noise(rng, shape, amount):
    import numpy as np

    return rng.normal(0.0, amount, shape).astype(np.float32)


def _blend(base, mask, color, softness=0.0):
    """Paints `color` into `base` where `mask` (0..1 float) is set; `base` is float32 H x W x 3."""
    import numpy as np

    m = mask[..., None]
    base[...] = base * (1.0 - m) + np.asarray(color, np.float32) * m


def _shape_mask(size, draw_fn, blur=0.0):
    from PIL import Image, ImageDraw, ImageFilter
    import numpy as np

    img = Image.new("L", size, 0)
    draw_fn(ImageDraw.Draw(img))
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    return np.asarray(img, np.float32) / 255.0


def scene_meadow(width=320, height=213):
    """A landscape: a graded sky with clouds, hills, a textured meadow and a tree."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(11)
    img = np.zeros((height, width, 3), np.float32)
    ys = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    horizon = int(height * 0.52)
    sky = np.stack([70 + 120 * ys, 130 + 90 * ys, 215 + 30 * ys], axis=-1).repeat(width, axis=1)
    img[:] = sky
    for cx, cy, rx, ry in [(70, 30, 38, 11), (110, 42, 30, 8), (230, 24, 44, 12), (270, 50, 28, 8)]:
        m = _shape_mask((width, height), lambda d, cx=cx, cy=cy, rx=rx, ry=ry: d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255), blur=5)
        _blend(img, m * 0.9, (250, 250, 252))
    hills = _shape_mask((width, height), lambda d: d.polygon([(0, horizon), (50, horizon - 26), (120, horizon - 8), (190, horizon - 34), (260, horizon - 10), (width, horizon - 22), (width, horizon + 4), (0, horizon + 4)], fill=255), blur=1.2)
    _blend(img, hills, (78, 104, 92))
    meadow = np.zeros((height, width), np.float32)
    meadow[horizon:, :] = 1.0
    grass = np.zeros((height, width, 3), np.float32)
    grass[...] = (74, 138, 52)
    stripes = (np.sin(np.arange(height, dtype=np.float32)[:, None] * 0.7) * 0.5 + 0.5)[..., None]
    grass += stripes * np.array([10, 18, 6], np.float32) + _noise(rng, (height, width, 3), 9.0)
    _blend(img, meadow, grass[0, 0])
    img[horizon:] = np.where(meadow[horizon:, :, None] > 0, grass[horizon:], img[horizon:])
    trunk = _shape_mask((width, height), lambda d: d.rectangle([232, horizon - 10, 240, horizon + 42], fill=255), blur=0.6)
    _blend(img, trunk, (92, 62, 40))
    for cx, cy, r in [(236, horizon - 34, 26), (218, horizon - 20, 20), (254, horizon - 22, 21), (236, horizon - 54, 17)]:
        m = _shape_mask((width, height), lambda d, cx=cx, cy=cy, r=r: d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255), blur=1.0)
        leaf = np.array([34, 96, 40], np.float32) + _noise(rng, (height, width, 3), 12.0)
        img[...] = img * (1 - m[..., None]) + leaf * m[..., None]
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")


def scene_street(width=213, height=320):
    """A portrait street: sky, a brick facade with windows, pavement, road and a person."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(23)
    img = np.zeros((height, width, 3), np.float32)
    img[...] = (150, 190, 235)
    facade_top = int(height * 0.14)
    ground = int(height * 0.66)
    facade = np.zeros((height, width, 3), np.float32)
    facade[...] = (158, 82, 64)
    brick_rows = (np.arange(height)[:, None] // 6) % 2
    facade += (np.sin((np.arange(width)[None, :] + 3 * brick_rows) * 0.9)[..., None] * 6.0).astype(np.float32)
    facade += _noise(rng, (height, width, 3), 7.0)
    img[facade_top:ground] = facade[facade_top:ground]
    for row in range(3):
        for col in range(4):
            x0, y0 = 14 + col * 48, facade_top + 14 + row * 56
            m = _shape_mask((width, height), lambda d, x0=x0, y0=y0: d.rectangle([x0, y0, x0 + 30, y0 + 38], fill=255))
            _blend(img, m, (226, 218, 200))
            m = _shape_mask((width, height), lambda d, x0=x0, y0=y0: d.rectangle([x0 + 3, y0 + 3, x0 + 27, y0 + 35], fill=255))
            _blend(img, m, (54, 70, 96))
    pavement = int(height * 0.76)
    img[ground:pavement] = np.array([176, 172, 166], np.float32) + _noise(rng, (pavement - ground, width, 3), 5.0)
    img[pavement:] = np.array([72, 72, 76], np.float32) + _noise(rng, (height - pavement, width, 3), 6.0)
    for x in range(10, width, 46):
        m = _shape_mask((width, height), lambda d, x=x: d.rectangle([x, pavement + 24, x + 24, pavement + 27], fill=255))
        _blend(img, m, (230, 230, 220))
    px = 104
    body = _shape_mask((width, height), lambda d: d.rounded_rectangle([px - 17, ground - 58, px + 17, ground + 4], 8, fill=255), blur=0.8)
    _blend(img, body, (196, 52, 48))
    legs = _shape_mask((width, height), lambda d: d.rectangle([px - 13, ground + 2, px + 13, ground + 52], fill=255), blur=0.8)
    _blend(img, legs, (40, 44, 62))
    head = _shape_mask((width, height), lambda d: d.ellipse([px - 11, ground - 86, px + 11, ground - 56], fill=255), blur=0.8)
    _blend(img, head, (224, 172, 140))
    hair = _shape_mask((width, height), lambda d: d.pieslice([px - 12, ground - 88, px + 12, ground - 60], 180, 360, fill=255), blur=0.8)
    _blend(img, hair, (50, 34, 26))
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")


def scene_shore(width=256, height=256):
    """A square shore: sky, sea with wave lines, sand and rocks."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(37)
    img = np.zeros((height, width, 3), np.float32)
    ys = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    horizon, shore = int(height * 0.38), int(height * 0.62)
    sky = np.stack([95 + 110 * ys, 150 + 80 * ys, 225 + 20 * ys], axis=-1).repeat(width, axis=1)
    img[:] = sky
    m = _shape_mask((width, height), lambda d: d.ellipse([60, 20, 150, 52], fill=255), blur=6)
    _blend(img, m * 0.85, (252, 252, 252))
    sea = np.zeros((shore - horizon, width, 3), np.float32)
    rows = np.arange(shore - horizon, dtype=np.float32)[:, None]
    sea[...] = np.array([30, 105, 140], np.float32)
    sea += (np.sin(rows * 1.3 + np.arange(width, dtype=np.float32)[None, :] * 0.05)[..., None] * 9.0)
    sea += _noise(rng, sea.shape, 4.0)
    img[horizon:shore] = sea
    sand = np.array([212, 188, 140], np.float32) + _noise(rng, (height - shore, width, 3), 8.0)
    img[shore:] = sand
    surf = _shape_mask((width, height), lambda d: d.rectangle([0, shore - 4, width, shore + 2], fill=255), blur=2.0)
    _blend(img, surf * 0.9, (240, 244, 245))
    for cx, cy, rx, ry in [(60, 214, 26, 15), (92, 224, 16, 10), (200, 206, 30, 17)]:
        m = _shape_mask((width, height), lambda d, cx=cx, cy=cy, rx=rx, ry=ry: d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255), blur=1.0)
        rock = np.array([108, 104, 98], np.float32) + _noise(rng, (height, width, 3), 10.0)
        img[...] = img * (1 - m[..., None]) + rock * m[..., None]
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")


def _shrunk(make, factor=0.75):
    """The scene at `factor` of its drawn size (smoother, and a smaller PNG)."""
    from PIL import Image

    def run():
        image = make()
        return image.resize((round(image.size[0] * factor), round(image.size[1] * factor)), Image.BICUBIC)

    return run


SCENES = {"meadow": _shrunk(scene_meadow), "street": _shrunk(scene_street), "shore": _shrunk(scene_shore)}


# ---------------------------------------------------------------------------------------------------------------------------
# The reference
# ---------------------------------------------------------------------------------------------------------------------------


def load_reference(source: Path, weights: Path):
    """The reference model (f32, CPU, eval) with the f16 weights of `weights` widened to f32, and its image processor."""
    import torch
    from safetensors import safe_open
    from transformers import EomtForUniversalSegmentation, EomtImageProcessor

    model = EomtForUniversalSegmentation.from_pretrained(str(source), dtype=torch.float32)
    state = {}
    with safe_open(str(weights), framework="pt") as handle:
        for name in handle.keys():
            state[name] = handle.get_tensor(name).float()
    missing, unexpected = model.load_state_dict(state, strict=False)
    # The model's buffers (position ids) are not in the file; everything else must be.
    if [m for m in missing if "position_ids" not in m] or unexpected:
        sys.exit(f"error: the weights do not fit the model: missing {missing}, unexpected {unexpected}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    processor = EomtImageProcessor.from_pretrained(str(source))
    return model, processor


def reference_run(model, processor, image):
    """Runs the reference on a PIL picture; returns a dict of numpy arrays and the fit size."""
    import numpy as np
    import torch
    from transformers.image_transforms import get_size_with_aspect_ratio

    width, height = image.size
    fit = get_size_with_aspect_ratio((height, width), IMAGE_PX, IMAGE_PX)
    inputs = processor(images=image, return_tensors="pt")
    with torch.inference_mode():
        out = model(**inputs)
    pixel = inputs["pixel_values"][0].numpy()
    mean = np.array(processor.image_mean, np.float32)[:, None, None]
    std = np.array(processor.image_std, np.float32)[:, None, None]
    levels = np.rint((pixel * std + mean) * 255.0).astype(np.int64)  # 3 x 640 x 640, whole numbers by construction
    panoptic = processor.post_process_panoptic_segmentation(out, target_sizes=[fit], stuff_classes=STUFF_CLASSES)[0]
    seg = panoptic["segmentation"].numpy().astype(np.int64)
    classes = np.full(seg.shape, NONE_CLASS, np.int64)
    for segment in panoptic["segments_info"]:
        classes[seg == segment["id"]] = segment["label_id"]
    probs = out.class_queries_logits[0].softmax(-1)
    score, label = probs.max(-1)
    keep = (label != NONE_CLASS) & (score > 0.8)
    return {
        "levels": levels,
        "class_logits": out.class_queries_logits[0].numpy(),
        "mask_logits": out.masks_queries_logits[0].numpy(),
        "classes": classes.astype(np.uint8),
        "kept": [(int(q), int(label[q]), float(score[q])) for q in range(len(score)) if bool(keep[q])],
        "probs": probs.numpy(),
        "fit": fit,
    }


def pick_queries(run) -> list[int]:
    """The kept queries, then the queries with the largest probability of a class other than "none"; `QUERY_COUNT` in all, sorted."""
    import numpy as np

    kept = [q for q, _, _ in run["kept"]]
    others = np.argsort(-run["probs"][:, :NONE_CLASS].max(-1), kind="stable")
    chosen = list(kept)
    for q in others:
        if len(chosen) >= QUERY_COUNT:
            break
        if int(q) not in chosen:
            chosen.append(int(q))
    return sorted(chosen[:QUERY_COUNT]) if len(kept) <= QUERY_COUNT else sorted(kept)


def num(x) -> float:
    """A float32 as a JSON number with 7 significant digits (a float32 holds about 7.2: the rounding is at most 5e-8 relative)."""
    return float(f"{float(x):.7g}")


def describe(name, image, run, png_name, classes_name):
    """The JSON record of one picture."""
    import numpy as np

    queries = pick_queries(run)
    rows = range(0, IMAGE_PX, ROW_STRIDE)
    cols = range(0, IMAGE_PX, COL_STRIDE)
    samples = [int(run["levels"][c, y, x]) for y in rows for x in cols for c in range(3)]
    points = [MASK_OFFSET + MASK_STRIDE * i for i in range(MASK_COUNT)]
    mask = run["mask_logits"]
    return {
        "name": name,
        "png": png_name,
        "classes_png": classes_name,
        "width": image.size[0],
        "height": image.size[1],
        "fit": list(run["fit"]),
        "input_samples": samples,
        "queries": queries,
        "class_logits": [[num(v) for v in run["class_logits"][q]] for q in queries],
        "mask_samples": [[num(mask[q, y, x]) for y in points for x in points] for q in queries],
        "kept": [{"query": q, "label": label, "score": num(score)} for q, label, score in run["kept"]],
        "labels_present": sorted(int(v) for v in np.unique(run["classes"])),
    }


def write_set(out: Path, images: dict, model, processor, meta: dict) -> None:
    """Writes the pictures, the reference label maps and `parity.json` into `out`."""
    from PIL import Image
    import numpy as np

    out.mkdir(parents=True, exist_ok=True)
    records = []
    for name, image in images.items():
        png = f"{name}.png"
        classes_png = f"{name}.classes.png"
        image.save(out / png, optimize=True)
        run = reference_run(model, processor, image)
        Image.fromarray(run["classes"], "L").save(out / classes_png, optimize=True)
        record = describe(name, image, run, png, classes_png)
        records.append(record)
        counts = np.bincount(run["classes"].ravel(), minlength=NONE_CLASS + 1)
        top = ", ".join(f"{int(i)}:{int(counts[i])}" for i in np.argsort(-counts)[:5] if counts[i])
        print(f"{name}: {image.size[0]}x{image.size[1]} fit {run['fit']}, {len(run['kept'])} kept queries, classes (id:pixels) {top}")
    document = {
        "schema": 1,
        "row_stride": ROW_STRIDE,
        "col_stride": COL_STRIDE,
        "mask_offset": MASK_OFFSET,
        "mask_stride": MASK_STRIDE,
        "mask_count": MASK_COUNT,
        **meta,
        "images": records,
    }
    text = json.dumps(document, separators=(",", ":"), sort_keys=True)
    (out / "parity.json").write_text(text + "\n", encoding="utf-8")
    print(f"wrote {out / 'parity.json'}: {len(text) / 1e3:.0f} KB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path, required=True, help="the folder with config.json and preprocessor_config.json")
    parser.add_argument("--weights", type=Path, required=True, help="the f16 safetensors file of export.py")
    parser.add_argument("--out", type=Path, default=Path("crates/ym-ml/tests/data/eomt"), help="where the fixture goes")
    parser.add_argument("--photos", type=Path, default=None, help="a folder of pictures for a local check on real photographs")
    parser.add_argument("--photos-out", type=Path, default=None, help="where the photographs' fixture goes (not the repository)")
    args = parser.parse_args()

    import torch
    import transformers
    from PIL import Image

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    model, processor = load_reference(args.source, args.weights)
    config = json.loads((args.source / "config.json").read_text(encoding="utf-8"))
    meta = {
        "reference": f"transformers {transformers.__version__}, torch {torch.__version__}, CPU, f32, one thread",
        "weights_sha256": sha256_file(args.weights),
        "id2label": [config["id2label"][str(i)] for i in range(len(config["id2label"]))],
    }
    write_set(args.out, {name: make() for name, make in SCENES.items()}, model, processor, meta)
    if args.photos:
        if not args.photos_out:
            sys.exit("error: --photos needs --photos-out")
        photos = {p.stem: Image.open(p).convert("RGB") for p in sorted(args.photos.iterdir()) if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
        write_set(args.photos_out, photos, model, processor, meta)


if __name__ == "__main__":
    main()

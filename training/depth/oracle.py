#!/usr/bin/env python3
"""Fixtures for the Depth Anything V2 gate of D-03 (`docs/23`, `docs/13`).

Python is for test fixtures only (`AGENTS.md` rule 6): this script runs in CI (`.github/workflows/models.yml`, job `depth`)
and writes data that the Rust tests of `ym_ml::depth` read; neither the app nor `ym` nor any shipped crate runs it.

    python3 -m training.depth.oracle --upstream /path/to/Depth-Anything-V2 \\
        --weights build/depth/depthnything_v2_vits.f16.safetensors \\
        --checkpoint build/depth/cache/depthnything_v2_vits.pth --photos build/depth/photos --out build/depth/fixtures

For each of the 20 public photos of `oracle_photos.json` (CC0 or public domain; URL, licence, author and SHA-256 are listed
there, and the photos are not committed) it:

1. downloads the photo (or finds it in `--photos`) and checks its SHA-256;
2. makes the 8-bit picture the runtime starts from: EXIF orientation applied, RGB, the long edge resized to 1024 px with
   Pillow's Lanczos filter (the same picture the SAM 2 fixtures use; the colour profile is ignored: the picture is taken as sRGB);
3. makes the network input with the upstream code's own transform (`Resize(keep_aspect_ratio=True, ensure_multiple_of=14,
   resize_method='lower_bound', cv2.INTER_CUBIC)` on the float64 RGB picture divided by 255, `NormalizeImage`, `PrepareForNet`),
   which makes the short side 518 and the long side the nearest multiple of 14;
4. runs the upstream model (`depthnything_v2` at the pinned commit, CPU, f32, one thread, `torch.use_deterministic_algorithms`,
   oneDNN off, no autocast, no xformers) **with the weights of the published safetensors file widened to f32**, so that the
   fixtures are what the shipped weights give in the upstream code and D-03's gate measures our runtime, not the f16 storage;
5. does the same with the original f32 checkpoint and writes the difference (`report.json`), so that the cost of the f16
   storage is known and reported, never hidden.

The format is the one `crates/ym-ml/tests/depth.rs` reads. One `.npz` per photo, `<id>.npz`, uncompressed (`ZIP_STORED`) with
fixed timestamps so that the bytes do not depend on the time; **every array is float32**, in PyTorch's layouts. The network
input is `h x w` (`h = 14 ph`, `w = 14 pw`), `N = ph * pw` patches; the sizes of each file are in `manifest.json`.

Every photo has:

    picture       f32 [H0, W0, 3]    the 8-bit picture (long edge 1024) divided by 255, RGB: what the Rust resize takes
    input         f32 [1, 3, h, w]   upstream's `image2tensor` of that picture (OpenCV resize, normalisation)
    depth         f32 [1, h, w]      `model(input)`: the network's output, relative inverse depth (larger is nearer), after
                                     its final ReLU, before `infer_image`'s resize back to the photo

The photos marked `"intermediates": true` in `oracle_photos.json` (one landscape, one portrait and one nearly square photo, so that
the positional-embedding resize is checked for three grid shapes) also have every stage of the run on `input`, taken with
forward hooks on the unmodified upstream modules:

    embedded      f32 [1, 1 + N, 384]   `prepare_tokens_with_masks(input)`: class token and patch tokens plus the positions
    block_00 .. block_11
                  f32 [1, 1 + N, 384]   the stream after each block (before the final norm)
    tap_0 .. tap_3
                  f32 [1, N, 384]       `get_intermediate_layers(input, [2, 5, 8, 11])`: the four taps after `pretrained.norm`,
                                        class token dropped
    layer_1 .. layer_4
                  f32 [1, C, ., .]      `depth_head.resize_layers[k]` output (after `projects[k]`): C = 48, 96, 192, 384
    path_4 .. path_1
                  f32 [1, 64, ., .]     `scratch.refinenet4 .. 1` outputs; path_1 is the fused output before the head
    output_conv1  f32 [1, 32, 8 ph, 8 pw]   `scratch.output_conv1(path_1)`, before the bilinear resize to (h, w)

A full trace is about 150 MB per photo, so the other photos have the three arrays above only. `manifest.json` is
`{"photos": {...}, "depth_only": {...}, ...}`: `photos` holds the photos with a full trace (the keys the Rust test iterates), `depth_only`
the rest (picture, input and depth; for the cut gates and the f16 report); each entry has the sizes, the file, the licence and
the arrays' dtypes and shapes. `report.json` has the f16-against-f32 numbers of every photo and the sign hints.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHOTOS_JSON = HERE / "oracle_photos.json"
#: Wikimedia asks for a User-Agent that names the tool and a way to reach its owner (https://meta.wikimedia.org/wiki/User-Agent_policy).
USER_AGENT = "yellowmoon-oracle/1 (https://github.com/ngriaznov/yellowmoon; fixtures for the Depth Anything V2 runtime test)"
PICTURE_LONG_EDGE = 1024
FIXTURE_VERSION = "1"
ZIP_TIME = (2026, 11, 7, 0, 0, 0)
PLACEHOLDER = "TBD-after-first-run"
PHOTO_COUNT = 20

#: The blocks whose outputs are stored for a full trace: all twelve (the taps are 2, 5, 8 and 11).
BLOCK_OUTPUTS = tuple(range(12))
#: What each array is, for `manifest.json`. The shapes are in the manifest per file. (See the module docstring.)
ARRAY_DOC = {
    "picture": "f32 [H0, W0, 3] the 8-bit RGB picture divided by 255, long edge 1024",
    "input": "f32 [1, 3, h, w] upstream resize (cv2 cubic, aspect kept, short side 518, multiples of 14) and normalisation",
    "depth": "f32 [1, h, w] network output on input (relative inverse depth, larger is nearer; after the final ReLU)",
    "embedded": "f32 [1, 1 + N, 384] prepare_tokens_with_masks(input): class token, patch tokens, positions",
    "block_NN": "f32 [1, 1 + N, 384] stream after block NN (00 to 11), before the final norm",
    "tap_K": "f32 [1, N, 384] K = 0..3: blocks 2, 5, 8, 11 after pretrained.norm, class token dropped",
    "layer_K": "f32 [1, C, ., .] K = 1..4: depth_head.resize_layers[K - 1] output, C = 48, 96, 192, 384",
    "path_K": "f32 [1, 64, ., .] K = 4..1: scratch.refinenetK output (path_1 is the fused output before the head)",
    "output_conv1": "f32 [1, 32, 8 ph, 8 pw] scratch.output_conv1(path_1), before the bilinear resize to (h, w)",
}
#: The keys of a full trace besides the three every photo has.
TRACE_KEYS = (
    ["embedded"] + [f"block_{n:02d}" for n in BLOCK_OUTPUTS] + [f"tap_{k}" for k in range(4)]
    + [f"layer_{k}" for k in range(1, 5)] + [f"path_{k}" for k in (4, 3, 2, 1)] + ["output_conv1"]
)
BASE_KEYS = ("picture", "input", "depth")


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

    A photo is pinned by `sha256`. While that is still the placeholder (the list was written where Wikimedia's upload server
    could not be reached), the file must at least have the SHA-1 that Commons states (`commons_sha1`), and the SHA-256 found is
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


#: EXIF orientation tag.
EXIF_ORIENTATION_TAG = 0x0112


def upright(image):
    """The image turned the way its EXIF orientation says (1 to 8; anything else, or no tag, leaves it as it is).

    This is `ImageOps.exif_transpose` without its last step: that function writes the whole EXIF block again, and stops with
    ZeroDivisionError on a photo that has a rational tag with a zero denominator (Pillow 11.0.0). Only the orientation tag is
    read here and the EXIF is never written, so a flaw in any other tag cannot matter.
    """
    from PIL import Image

    value = image.getexif().get(EXIF_ORIENTATION_TAG, 1)
    orientation = value if isinstance(value, int) and 1 <= value <= 8 else 1
    method = {
        2: Image.Transpose.FLIP_LEFT_RIGHT,
        3: Image.Transpose.ROTATE_180,
        4: Image.Transpose.FLIP_TOP_BOTTOM,
        5: Image.Transpose.TRANSPOSE,
        6: Image.Transpose.ROTATE_270,
        7: Image.Transpose.TRANSVERSE,
        8: Image.Transpose.ROTATE_90,
    }.get(orientation)
    return image.copy() if method is None else image.transpose(method)


def load_picture(path: Path):
    """The 8-bit picture of a photo: uint8 [H, W, 3], long edge 1024."""
    import numpy as np
    from PIL import Image

    with Image.open(path) as opened:
        image = upright(opened).convert("RGB")
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


def describe_arrays(arrays: dict) -> dict:
    """The index entry of a fixture: every array's dtype (`u8`, `f32`) and shape, by name."""
    import numpy as np

    names = {np.dtype(np.float32): "f32"}
    result = {}
    for name in sorted(arrays):
        array = np.asarray(arrays[name])
        if array.dtype not in names:
            raise ValueError(f"{name} has dtype {array.dtype}; fixtures hold f32 only (crates/ym-ml/tests/depth.rs reads nothing else)")
        result[name] = {"dtype": names[array.dtype], "shape": list(array.shape)}
    return result


def upstream_size(width: int, height: int, input_size: int = 518, multiple: int = 14) -> tuple[int, int]:
    """(width, height) of the network input for a picture of this size: the upstream rule, written out without torch or OpenCV.

    `Resize(width=518, height=518, keep_aspect_ratio=True, ensure_multiple_of=14, resize_method='lower_bound')` of the upstream
    `util/transform.py`: the scale that makes the short side at least 518, each side rounded to the nearest multiple of 14
    (halves go to the even multiple, as `numpy.round` does), and raised to the next multiple when that is below 518.
    A 6000 x 4000 photo gives 784 x 518.
    """
    scale_height = input_size / height
    scale_width = input_size / width
    if scale_width > scale_height:
        scale_height = scale_width
    else:
        scale_width = scale_height

    def constrain(value: float) -> int:
        size = round(value / multiple) * multiple  # Python's round() is half to even, like numpy's
        if size < input_size:
            size = -(-value // multiple) * multiple  # ceil
        return int(size)

    return constrain(scale_width * width), constrain(scale_height * height)


def upstream_tensor(picture, size: int = 518):
    """The upstream network input for an RGB uint8 picture: f32 [1, 3, Hn, Wn], made by the upstream transform itself.

    This is `DepthAnythingV2.image2tensor` of the pinned commit, without its device move and with the picture already RGB.
    """
    import cv2
    import numpy as np
    from torchvision.transforms import Compose

    from depthnything_v2.util.transform import NormalizeImage, PrepareForNet, Resize

    transform = Compose(
        [
            Resize(
                width=size,
                height=size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method="lower_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )
    image = picture.astype(np.float64) / 255.0
    return transform({"image": image})["image"][None].astype(np.float32)


def relative_rms(a, b) -> float:
    """RMS of a - b divided by the RMS of b, in f64."""
    import numpy as np

    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(b**2)))


def rank_correlation(a, b) -> float:
    """Spearman's rank correlation of two arrays (ordinal ranks, ties broken by position), in f64; 1.0 for the same order."""
    import numpy as np

    def ranks(values):
        order = np.argsort(values.reshape(-1), kind="stable")
        result = np.empty(order.size, dtype=np.float64)
        result[order] = np.arange(order.size, dtype=np.float64)
        return result

    ra, rb = ranks(a), ranks(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denominator = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return 1.0 if denominator == 0.0 else float((ra * rb).sum() / denominator)


class Probe:
    """Forward hooks on the upstream model that keep the stages of one run (the keys of the module docstring)."""

    def __init__(self, model):
        self.model = model
        self.enabled = False
        self.arrays: dict = {}
        self._taps: list = []
        encoder = model.pretrained
        head = model.depth_head
        encoder.blocks[0].register_forward_pre_hook(lambda _module, inputs: self._store("embedded", inputs[0]))  # prepare_tokens_with_masks(input)
        for index in BLOCK_OUTPUTS:
            encoder.blocks[index].register_forward_hook(self._keep(f"block_{index:02d}"))
        encoder.norm.register_forward_hook(self._tap)
        for k, layer in enumerate(head.resize_layers, start=1):
            layer.register_forward_hook(self._keep(f"layer_{k}"))
        for k in (4, 3, 2, 1):
            getattr(head.scratch, f"refinenet{k}").register_forward_hook(self._keep(f"path_{k}"))
        head.scratch.output_conv1.register_forward_hook(self._keep("output_conv1"))

    def _store(self, name, tensor):
        if self.enabled:
            self.arrays[name] = tensor.detach().float().contiguous().numpy().copy()

    def _keep(self, name):
        return lambda _module, _inputs, output: self._store(name, output)

    def _tap(self, _module, _inputs, output):
        # `pretrained.norm` is applied to the four taps in order, once each, and nowhere else in `forward`.
        if self.enabled:
            self._taps.append(output[:, 1:].detach().float().contiguous().numpy().copy())

    def run(self, torch, tensor, trace: bool):
        """(the depth map f32 [1, h, w], the stages or {}) of the model on a tensor [1, 3, h, w]."""
        self.enabled = trace
        self.arrays = {}
        self._taps = []
        with torch.no_grad():
            depth = self.model(tensor)
        self.enabled = False
        arrays = {}
        if trace:
            arrays = dict(self.arrays)
            if len(self._taps) != 4:
                sys.exit(f"error: pretrained.norm ran {len(self._taps)} times, expected 4 (the four taps)")
            for index, tap in enumerate(self._taps):
                arrays[f"tap_{index}"] = tap
            missing = [key for key in TRACE_KEYS if key not in arrays]
            if missing:
                sys.exit(f"error: the run did not produce {missing}")
        return depth.detach().float().contiguous().numpy().copy(), arrays


def build_models(checkpoint: Path, weights: Path):
    """(the model with the exported f16 weights widened to f32, the model with the original f32 weights)."""
    from safetensors.torch import load_file

    from training.depth import export

    original = export.build_upstream_model(checkpoint)
    shipped = export.build_upstream_model(None)
    stored = load_file(str(weights))
    shipped.load_state_dict({name: tensor.float() for name, tensor in stored.items()}, strict=True)
    return shipped, original


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--upstream", type=Path, default=os.environ.get("DEPTH_ANYTHING_V2_SRC"), help="a checkout of the upstream code at the pinned commit (default $DEPTH_ANYTHING_V2_SRC)")
    parser.add_argument("--weights", type=Path, required=True, help="the exported depthnything_v2_vits.f16.safetensors")
    parser.add_argument("--checkpoint", type=Path, required=True, help="the original depthnything_v2_vits.pth (f32 weights)")
    parser.add_argument("--photos", type=Path, default=Path("build/depth/photos"), help="where the photos are kept (downloaded when missing)")
    parser.add_argument("--out", type=Path, default=Path("build/depth/fixtures"), help="output directory")
    parser.add_argument("--only", default=None, help="comma-separated photo ids (a quick local run)")
    args = parser.parse_args()
    if args.upstream is None:
        sys.exit("error: --upstream (or DEPTH_ANYTHING_V2_SRC) must name a checkout of the upstream code")

    import cv2
    import numpy as np
    import torch
    import torchvision
    import PIL

    from training.depth import export

    export.use_upstream(Path(args.upstream))
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.mkldnn.enabled = False  # oneDNN picks its convolution kernels (and so its summation order) by CPU model

    listing = json.loads(PHOTOS_JSON.read_text(encoding="utf-8"))
    photos = listing["photos"]
    if args.only:
        wanted = set(args.only.split(","))
        photos = [photo for photo in photos if photo["id"] in wanted]
    if len(photos) != PHOTO_COUNT and not args.only:
        sys.exit(f"error: {PHOTOS_JSON.name} lists {len(photos)} photos, the gate needs {PHOTO_COUNT}")

    print(f"versions: torch {torch.__version__}, torchvision {torchvision.__version__}, numpy {np.__version__}, Pillow {PIL.__version__}, OpenCV {cv2.__version__}")
    shipped, original = build_models(args.checkpoint, args.weights)
    probe = Probe(shipped)
    args.out.mkdir(parents=True, exist_ok=True)

    files = {}
    report: dict = {"photos": {}, "worst": {}}
    worst = {"f16_depth_relative_rms": 0.0, "f16_min_rank_correlation": 1.0}
    traced: dict = {}
    depth_only: dict = {}
    for photo in photos:
        path = fetch_photo(photo, args.photos)
        picture = load_picture(path)
        height, width = picture.shape[:2]
        tensor = upstream_tensor(picture)
        h, w = int(tensor.shape[2]), int(tensor.shape[3])
        if (w, h) != upstream_size(width, height):
            sys.exit(f"error: the upstream transform made {(w, h)} for a {width} x {height} picture; upstream_size() says {upstream_size(width, height)}")
        full = bool(photo.get("intermediates", False))

        depth, stages = probe.run(torch, torch.from_numpy(tensor), full)
        with torch.no_grad():
            reference = original(torch.from_numpy(tensor)).float().numpy()

        arrays = {"picture": (picture.astype(np.float32) / np.float32(255.0)), "input": tensor, "depth": depth}
        arrays.update(stages)
        flat, ref_flat = depth[0], reference[0]
        entry = {
            "picture_size": [height, width],
            "network_input_size": [h, w],
            "patches": (h // 14) * (w // 14),
            "full_trace": full,
            "depth_min": float(flat.min()),
            "depth_max": float(flat.max()),
            "depth_mean": float(flat.mean()),
            "depth_top_third_mean": float(flat[: h // 3].mean()),
            "depth_bottom_third_mean": float(flat[-(h // 3) :].mean()),
            "f16_vs_f32_depth_relative_rms": relative_rms(flat, ref_flat),
            "f16_vs_f32_depth_max_abs": float(np.abs(flat - ref_flat).max()),
            "f16_vs_f32_rank_correlation": rank_correlation(flat, ref_flat),
        }
        worst["f16_depth_relative_rms"] = max(worst["f16_depth_relative_rms"], entry["f16_vs_f32_depth_relative_rms"])
        worst["f16_min_rank_correlation"] = min(worst["f16_min_rank_correlation"], entry["f16_vs_f32_rank_correlation"])
        target = args.out / f"{photo['id']}.npz"
        write_npz(target, arrays)
        info = {"bytes": target.stat().st_size, "sha256": sha256_file(target), "arrays": describe_arrays(arrays)}
        files[target.name] = info
        record = {
            "file": target.name,
            "title": photo["title"],
            "license": photo["license"],
            "page": photo["page"],
            "categories": photo["categories"],
            "photo_sha256": sha256_file(args.photos / f"{photo['id']}.jpg"),
            "picture_size": entry["picture_size"],
            "network_input_size": entry["network_input_size"],
            "patches": entry["patches"],
            "full_trace": full,
        }
        (traced if full else depth_only)[photo["id"]] = record
        report["photos"][photo["id"]] = entry
        print(
            f"{photo['id']:14} {width}x{height} -> {w}x{h}  depth {entry['depth_min']:.2f}..{entry['depth_max']:.2f}  "
            f"top {entry['depth_top_third_mean']:.2f} bottom {entry['depth_bottom_third_mean']:.2f}  f16 rms {entry['f16_vs_f32_depth_relative_rms']:.1e}"
        )
    report["worst"] = worst
    report["note"] = (
        "f16_vs_f32: the upstream code with the published f16 weights widened to f32 (what the fixtures hold) against the original f32 checkpoint, "
        "both on `input`; relative RMS of the depth map, and Spearman's rank correlation (ordinal ranks). "
        "depth_top_third_mean and depth_bottom_third_mean are for finding out which way the depth runs (larger is nearer): in an outdoor "
        "photo with sky at the top, the bottom third should be the larger."
    )
    (args.out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "fixture_version": FIXTURE_VERSION,
        "weights": args.weights.name,
        "weights_sha256": sha256_file(args.weights),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "upstream_commit": export.UPSTREAM_COMMIT,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "opencv": cv2.__version__,
        "picture_long_edge_px": PICTURE_LONG_EDGE,
        "input_short_side_px": export.INPUT_SIZE,
        "patch_size_px": export.PATCH_SIZE,
        "array_doc": ARRAY_DOC,
        "photos": traced,
        "depth_only": depth_only,
        "files": files,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    total = sum(info["bytes"] for info in files.values())
    print(f"wrote {len(files)} fixtures ({len(traced)} with a full trace), {total / 1e6:.0f} MB, to {args.out}")
    print(
        "f16 against f32 weights, worst over all photos: "
        f"depth relative RMS {worst['f16_depth_relative_rms']:.2e}, rank correlation at least {worst['f16_min_rank_correlation']:.5f}"
    )

if __name__ == "__main__":
    main()

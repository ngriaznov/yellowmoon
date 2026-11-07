#!/usr/bin/env python3
"""The DIODE ordering gate of D-06 (`docs/algorithms/06-masks.md` section 8.8; Gate D of `docs/15-milestones.md`).

Evaluation tooling only (`AGENTS.md` rules 6 and 7): it runs in CI (`.github/workflows/depth-eval.yml`, job `diode`) and on a
developer's machine; neither the app nor `ym` runs it, and only this script reaches the network (to fetch DIODE).

    python3 -m training.depth.eval_diode --upstream /path/to/Depth-Anything-V2 --weights data/models/depth/depth_anything_v2_vits.f16.safetensors \\
        --depth-maps target/release/examples/depth_maps --scratch build/depth-eval --out build/depth-eval/out [--every 8] [--accept-new]

It measures how well the model's relative inverse depth orders near and far on DIODE's validation split (MIT; indoor and outdoor
scenes; 771 pictures of 1024 x 768), for **our Rust runtime** and for **the upstream PyTorch model** on the same pictures and the
same pairs, and gates on the difference. The pair metric is written in section 8.8; the code below is that text. In order:

1. downloads `val.tar.gz` from the DIODE project (https://diode-dataset.s3.amazonaws.com/val.tar.gz, 2.6 GB) into `--scratch`,
   checks its SHA-256 against the pin in `diode.sha256` (a placeholder pin makes the run print the hash and stop with exit code 3,
   as `tools/fetch/depth.sh` does; `--accept-new` prints it and goes on), and extracts it;
2. in chunks (the pictures as `.npy` are 9 MB each, so they are written, used and removed 32 at a time): writes the 8-bit
   picture divided by 255 as f32 `H x W x 3`; runs the upstream model (the pinned upstream code on the CPU in f32 with the shipped
   f16 weights widened to f32, from `oracle.upstream_tensor`, upstream's own `image2tensor`); runs the `depth_maps` example
   (`crates/ym-ml/examples/depth_maps.rs`: `prepare_input` and `infer` as the app does);
3. scores both on the same pairs (`sample_pairs`, `image_score`) and writes `report.json` and `summary.md` into `--out`.

The picture the Rust runtime and PyTorch start from is the same array of 8-bit values. Their resizes differ slightly (ours is
f32, upstream's OpenCV on f64); that is part of what the gate measures. The exit code is 1 when the gate fails (reports are
written first), 3 when the archive's pin is a placeholder, 0 otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIN_FILE = HERE / "diode.sha256"
ARCHIVE_NAME = "val.tar.gz"
#: DIODE's own link (https://diode-dataset.org, "Download": validation split); the licence is MIT.
ARCHIVE_URL = "https://diode-dataset.s3.amazonaws.com/val.tar.gz"
#: Content-Length that the server states for the archive (read 2026-11-07); a sanity check before the hash.
ARCHIVE_BYTES = 2_774_625_282
USER_AGENT = "yellowmoon-eval/1 (https://github.com/ngriaznov/yellowmoon; DIODE ordering gate of the depth runtime)"
PLACEHOLDER = "TBD-after-first-run"

#: Section 8.8: pairs per picture, draws at most, the smallest ratio of the two depths, and the gate.
PAIRS_PER_IMAGE = 1000
MAX_DRAWS = 20_000
MIN_RATIO = 1.05
GATE_POINTS = 0.5
#: What the authors report for ViT-S on DA-2K's annotated pairs (arXiv:2406.09414): shown beside our numbers for scale only.
DA2K_VITS_PERCENT = 95.3
CHUNK = 32


# ------------------------------------------------------------------------------------------------------------------------
# The metric (section 8.8). numpy only.
# ------------------------------------------------------------------------------------------------------------------------


def sample_pairs(depth, seed: int, pairs: int = PAIRS_PER_IMAGE, draws: int = MAX_DRAWS, min_ratio: float = MIN_RATIO):
    """The pairs of one picture: `(a, b)`, two int arrays of flat indices into `depth` (row-major), at most `pairs` long.

    `depth` is the ground truth with 0 (or a non-finite value) where it is not valid. `draws` pairs of positions are drawn at once
    from the valid pixels with `numpy.random.Generator(PCG64(seed))`: `rng.integers(0, n, size=(draws, 2))`, `n` the number of
    valid pixels, taken in row-major order. A draw is kept when its two pixels are different and the larger depth is at least
    `min_ratio` times the smaller one (`max >= min_ratio * min`, in f64); the first `pairs` kept draws, in draw order, are the
    result (fewer when fewer than `pairs` of the `draws` qualify, none when the picture has under two valid pixels).
    """
    import numpy as np

    flat = np.asarray(depth, dtype=np.float64).reshape(-1)
    valid = np.flatnonzero(np.isfinite(flat) & (flat > 0.0))
    if valid.size < 2:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    rng = np.random.Generator(np.random.PCG64(seed))
    draw = rng.integers(0, valid.size, size=(draws, 2))
    a, b = valid[draw[:, 0]], valid[draw[:, 1]]
    ga, gb = flat[a], flat[b]
    keep = (a != b) & (np.maximum(ga, gb) >= min_ratio * np.minimum(ga, gb))
    return a[keep][:pairs], b[keep][:pairs]


def bilinear_at(values, flat_index, height: int, width: int):
    """The bilinear resize of `values` (`h x w`) to `height x width`, evaluated at the pixels `flat_index` (row-major in the
    large grid), as f64.

    The centre of output pixel `i` is at `(i + 0.5) * (h / height) - 0.5` in the source, clamped to `[0, h - 1]` (the usual
    half-pixel rule, no antialiasing); the blend is `v0 + (v1 - v0) * t`, which returns a flat area's value exactly (so equal `d`
    stay equal, and the tie rule means something). Both runtimes' maps go through this one function.
    """
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    h, w = values.shape
    index = np.asarray(flat_index)
    ys, xs = index // width, index % width
    fy = np.clip((ys + 0.5) * (h / height) - 0.5, 0.0, h - 1.0)
    fx = np.clip((xs + 0.5) * (w / width) - 0.5, 0.0, w - 1.0)
    y0, x0 = np.floor(fy).astype(np.int64), np.floor(fx).astype(np.int64)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    ty, tx = fy - y0, fx - x0
    top = values[y0, x0] + (values[y0, x1] - values[y0, x0]) * tx
    bottom = values[y1, x0] + (values[y1, x1] - values[y1, x0]) * tx
    return top + (bottom - top) * ty


def pair_outcomes(gt_a, gt_b, d_a, d_b):
    """Per pair: 1.0 when the pixel that is nearer by ground truth (the smaller depth) has the larger `d`, 0.5 when the two `d` are
    equal, 0.0 otherwise."""
    import numpy as np

    gt_a, gt_b, d_a, d_b = (np.asarray(x, dtype=np.float64) for x in (gt_a, gt_b, d_a, d_b))
    d_near = np.where(gt_a < gt_b, d_a, d_b)
    d_far = np.where(gt_a < gt_b, d_b, d_a)
    return np.where(d_near > d_far, 1.0, np.where(d_near == d_far, 0.5, 0.0))


def image_outcomes(depth, d_map, a, b):
    """`pair_outcomes` of the pairs `(a, b)` (flat indices into the `depth` grid) for the map `d_map` (any size, resized with
    `bilinear_at`)."""
    import numpy as np

    flat = np.asarray(depth, dtype=np.float64).reshape(-1)
    height, width = np.asarray(depth).shape[:2]
    return pair_outcomes(flat[a], flat[b], bilinear_at(d_map, a, height, width), bilinear_at(d_map, b, height, width))


def image_score(outcomes):
    """The fraction right of one picture: the mean of its outcomes, or `None` when it has no pair."""
    return float(outcomes.mean()) if len(outcomes) else None


def mean_of(values) -> float | None:
    """Mean of the values that are not `None` (`None` when there are none)."""
    kept = [v for v in values if v is not None]
    return sum(kept) / len(kept) if kept else None


def summarise(rows: list[dict]) -> dict:
    """Totals of the per-picture rows: for all, indoor and outdoor, the mean score of each runtime, their difference in
    percentage points (rust minus torch), the standard error of that difference over pictures, and the gate."""
    totals: dict = {}
    for name, keep in (("all", lambda r: True), ("indoors", lambda r: r["split"] == "indoors"), ("outdoor", lambda r: r["split"] == "outdoor")):
        chosen = [r for r in rows if keep(r) and r["pairs"] > 0]
        if not chosen:
            totals[name] = {"images": 0}
            continue
        rust = [r["score_rust"] for r in chosen]
        torch_ = [r["score_torch"] for r in chosen]
        diffs = [100.0 * (x - y) for x, y in zip(rust, torch_)]
        n = len(diffs)
        mean_diff = sum(diffs) / n
        variance = sum((x - mean_diff) ** 2 for x in diffs) / (n - 1) if n > 1 else 0.0
        totals[name] = {
            "images": n,
            "pairs": sum(r["pairs"] for r in chosen),
            "rust_percent": 100.0 * sum(rust) / n,
            "torch_percent": 100.0 * sum(torch_) / n,
            "difference_points": mean_diff,
            "difference_standard_error_points": (variance / n) ** 0.5,
            "pairs_where_the_runtimes_differ_percent": 100.0 * sum(r["outcome_differs"] for r in chosen) / n,
        }
    all_row = totals["all"]
    gate = {"limit_points": GATE_POINTS, "images": all_row.get("images", 0)}
    gate["passed"] = bool(all_row.get("images", 0) > 0 and abs(all_row["difference_points"]) <= GATE_POINTS)
    return {"totals": totals, "gate": gate, "da2k_vits_percent_reported_by_the_authors": DA2K_VITS_PERCENT}


def markdown(report: dict) -> str:
    """The short summary that the workflow appends to the job summary."""
    lines = [
        "### DIODE ordering: Rust runtime against the upstream PyTorch model",
        "",
        f"Validation split, {report['images_scored']} of {report['images_found']} pictures (every {report['every']}), up to {PAIRS_PER_IMAGE} pairs each "
        f"with a depth ratio of at least {MIN_RATIO}. Share of pairs ordered correctly (equal `d` counts half), mean over pictures.",
        "",
        "| Set | Pictures | Pairs | Rust | PyTorch | Rust minus PyTorch (points) | Standard error | Pairs ordered differently |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("all", "indoors", "outdoor"):
        t = report["totals"][name]
        if not t["images"]:
            lines.append(f"| {name} | 0 | | | | | | |")
            continue
        lines.append(
            f"| {name} | {t['images']} | {t['pairs']} | {t['rust_percent']:.2f} % | {t['torch_percent']:.2f} % | {t['difference_points']:+.3f} | "
            f"{t['difference_standard_error_points']:.3f} | {t['pairs_where_the_runtimes_differ_percent']:.2f} % |"
        )
    gate = report["gate"]
    lines += [
        "",
        f"**Gate:** the difference on all pictures is at most {gate['limit_points']} points: **{'passed' if gate['passed'] else 'FAILED'}**.",
        "",
        f"For scale only: the authors report {report['da2k_vits_percent_reported_by_the_authors']} % of the pairs of DA-2K (annotated point pairs of photographs, "
        "a different set and a different metric) for ViT-S; the two are not compared.",
    ]
    if "timing" in report:
        t = report["timing"]
        lines += ["", f"Rust runtime on this machine ({report['machine']['cpus']} CPUs): median {t['median_ms']:.0f} ms, mean {t['mean_ms']:.0f} ms per picture (preparation and network)."]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------------------------------------
# The data.
# ------------------------------------------------------------------------------------------------------------------------


def pin_of(path: Path, name: str) -> str:
    """The pin of `name` in a `<sha256>  <name>` file (empty when the file or the line is missing)."""
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1] == name:
                return parts[0]
    return ""


def is_hash(text: str) -> bool:
    """True for 64 lower-case hex digits."""
    return len(text) == 64 and all(c in "0123456789abcdef" for c in text)


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, read in 1 MiB pieces."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_archive(scratch: Path, accept_new: bool, archive: Path | None) -> Path:
    """The DIODE validation archive, downloaded into `scratch` unless it is there (or given), and checked against its pin.

    A pin that is not 64 hex digits makes the run print the SHA-256 found and stop with exit code 3, unless `accept_new` is set:
    then it prints it and goes on. A pin that does not match stops the run with exit code 1 (and the file is removed when it was
    downloaded by this run).
    """
    pin = pin_of(PIN_FILE, ARCHIVE_NAME)
    path = archive or scratch / ARCHIVE_NAME
    downloaded = False
    if not path.is_file():
        scratch.mkdir(parents=True, exist_ok=True)
        print(f"downloading {ARCHIVE_URL}", flush=True)
        part = path.with_suffix(".part")
        digest = hashlib.sha256()
        request = urllib.request.Request(ARCHIVE_URL, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=300) as response, part.open("wb") as handle:  # noqa: S310 (the fixed project URL)
            for block in iter(lambda: response.read(1 << 20), b""):
                handle.write(block)
                digest.update(block)
        part.replace(path)
        downloaded = True
        got = digest.hexdigest()
    else:
        got = sha256_file(path)
    size = path.stat().st_size
    print(f"{path.name}: {size:,} bytes, SHA-256 {got}", flush=True)
    if size != ARCHIVE_BYTES:
        print(f"warning: the server stated {ARCHIVE_BYTES:,} bytes; the file has {size:,}", file=sys.stderr)
    if not is_hash(pin):
        print(f"{PIN_FILE.name} has no SHA-256 pin yet (found '{pin or 'nothing'}'). The line to write into it:\n{got}  {ARCHIVE_NAME}", flush=True)
        if not accept_new:
            print("stopping: run again with --accept-new to go on with this file, after checking where it came from", file=sys.stderr)
            sys.exit(3)
    elif got != pin:
        if downloaded:
            path.unlink()
        sys.exit(f"error: {ARCHIVE_NAME} has SHA-256 {got}, the pin is {pin}. The file at {ARCHIVE_URL} changed, or the download was cut off.")
    return path


def extract(archive: Path, target: Path) -> None:
    """Unpacks the archive into the empty folder `target` with the system `tar` (which drops absolute paths and `..`)."""
    marker = target / ".extracted"
    if marker.is_file():
        return
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    subprocess.run(["tar", "-xzf", str(archive), "-C", str(target), "--no-same-owner"], check=True)
    marker.write_text(sha256_file(archive) + "\n", encoding="utf-8")


def list_pictures(root: Path) -> list[dict]:
    """The pictures of the extracted split, sorted by path (POSIX, relative to `root`): `{"path", "split", "png", "depth", "mask"}`.

    A picture is `<id>.png` with `<id>_depth.npy` (H x W x 1, metres) and `<id>_depth_mask.npy` (H x W) beside it; `split` is
    `indoors` or `outdoor`, from the folder names. The index of a picture in this list is its seed (section 8.8).
    """
    found = []
    for png in sorted(root.rglob("*.png"), key=lambda p: p.relative_to(root).as_posix()):
        depth = png.with_name(png.stem + "_depth.npy")
        mask = png.with_name(png.stem + "_depth_mask.npy")
        if not (depth.is_file() and mask.is_file()):
            continue
        parts = png.relative_to(root).parts
        split = "indoors" if "indoors" in parts else "outdoor" if "outdoor" in parts else "other"
        found.append({"path": png.relative_to(root).as_posix(), "split": split, "png": png, "depth": depth, "mask": mask})
    return found


# ------------------------------------------------------------------------------------------------------------------------
# The two runtimes.
# ------------------------------------------------------------------------------------------------------------------------


def load_torch_model(upstream: Path, weights: Path):
    """The upstream model on the CPU with the shipped f16 weights widened to f32 (what `oracle.py` calls `shipped`)."""
    import torch
    from safetensors.torch import load_file

    from training.depth import export

    export.use_upstream(upstream)
    model = export.build_upstream_model(None)
    model.load_state_dict({name: tensor.float() for name, tensor in load_file(str(weights)).items()}, strict=True)
    torch.backends.mkldnn.enabled = False  # as oracle.py: oneDNN picks its kernels by CPU model
    torch.set_num_threads(os.cpu_count() or 1)
    return model


def torch_map(torch, model, picture_u8):
    """The upstream map (f32 `h x w`) for an 8-bit RGB picture, from upstream's own `image2tensor` transform."""
    from training.depth import oracle

    tensor = oracle.upstream_tensor(picture_u8)
    with torch.no_grad():
        return model(torch.from_numpy(tensor))[0].float().numpy()


def run_rust(depth_maps: Path, data_root: Path, inputs: Path, outputs: Path) -> list[dict]:
    """Runs the `depth_maps` example on a folder; returns its JSON lines."""
    done = subprocess.run([str(depth_maps), str(inputs), str(outputs), str(data_root)], check=True, capture_output=True, text=True)
    return [json.loads(line) for line in done.stdout.splitlines() if line.startswith("{")]


def machine() -> dict:
    """A few lines about the runner, for the report."""
    cpu = ""
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {"cpu": cpu, "cpus": os.cpu_count()}


# ------------------------------------------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--upstream", type=Path, default=os.environ.get("DEPTH_ANYTHING_V2_SRC"), help="a checkout of the upstream code at the pinned commit (default $DEPTH_ANYTHING_V2_SRC)")
    parser.add_argument("--weights", type=Path, default=Path("data/models/depth/depth_anything_v2_vits.f16.safetensors"))
    parser.add_argument("--depth-maps", type=Path, default=Path("target/release/examples/depth_maps"), help="the built Rust example")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="the folder that holds models/depth/ (for the Rust example)")
    parser.add_argument("--scratch", type=Path, default=Path("build/depth-eval"), help="where the archive and its contents go")
    parser.add_argument("--archive", type=Path, default=None, help="an archive that is already on disk (default SCRATCH/val.tar.gz, downloaded when missing)")
    parser.add_argument("--out", type=Path, default=Path("build/depth-eval/out"))
    parser.add_argument("--every", type=int, default=1, help="use every N-th picture of the sorted list (a smoke run); seeds stay the pictures' indices in the whole list")
    parser.add_argument("--accept-new", action="store_true", help="go on when the pin is still a placeholder (the hash is printed)")
    args = parser.parse_args()
    if args.upstream is None:
        sys.exit("error: --upstream (or DEPTH_ANYTHING_V2_SRC) must name a checkout of the upstream code")
    if args.every < 1:
        sys.exit("error: --every must be at least 1")

    import numpy as np
    import torch
    from PIL import Image

    archive = fetch_archive(args.scratch, args.accept_new, args.archive)
    root = args.scratch / "diode"
    extract(archive, root)
    pictures = list_pictures(root)
    if not pictures:
        sys.exit(f"error: no <id>.png with <id>_depth.npy and <id>_depth_mask.npy found under {root}")
    counts = {s: sum(1 for p in pictures if p["split"] == s) for s in ("indoors", "outdoor", "other")}
    print(f"{len(pictures)} pictures: {counts}", flush=True)
    chosen = [(index, p) for index, p in enumerate(pictures) if index % args.every == 0]

    model = load_torch_model(Path(args.upstream), args.weights)
    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    milliseconds: list[float] = []
    started = time.time()
    for first in range(0, len(chosen), CHUNK):
        batch = chosen[first : first + CHUNK]
        work = Path(tempfile.mkdtemp(prefix="chunk-", dir=args.scratch))
        try:
            inputs, outputs = work / "in", work / "out"
            inputs.mkdir()
            torch_maps = {}
            for index, p in batch:
                with Image.open(p["png"]) as opened:
                    u8 = np.asarray(opened.convert("RGB"), dtype=np.uint8)
                name = f"{index:05d}"
                np.save(inputs / f"{name}.npy", (u8.astype(np.float32) / np.float32(255.0)))
                torch_maps[name] = torch_map(torch, model, u8)
            for line in run_rust(args.depth_maps, args.data_root, inputs, outputs):
                milliseconds.append(line["ms"])
            for index, p in batch:
                name = f"{index:05d}"
                gt = np.load(p["depth"]).astype(np.float64)[..., 0]
                mask = np.load(p["mask"]) > 0
                gt = np.where(mask & np.isfinite(gt) & (gt > 0), gt, 0.0)
                a, b = sample_pairs(gt, index)
                rust_map = np.load(outputs / f"{name}.depth.npy")
                out_rust = image_outcomes(gt, rust_map, a, b)
                out_torch = image_outcomes(gt, torch_maps[name], a, b)
                rows.append(
                    {
                        "index": index,
                        "path": p["path"],
                        "split": p["split"],
                        "valid_pixels": int((gt > 0).sum()),
                        "pairs": int(len(a)),
                        "score_rust": image_score(out_rust),
                        "score_torch": image_score(out_torch),
                        "outcome_differs": float(np.mean(out_rust != out_torch)) if len(a) else 0.0,
                    }
                )
        finally:
            shutil.rmtree(work, ignore_errors=True)
        print(f"{len(rows)} of {len(chosen)} pictures, {time.time() - started:.0f} s", flush=True)

    report = {
        "metric": "docs/algorithms/06-masks.md section 8.8",
        "pairs_per_image": PAIRS_PER_IMAGE,
        "max_draws": MAX_DRAWS,
        "min_depth_ratio": MIN_RATIO,
        "every": args.every,
        "images_found": len(pictures),
        "images_scored": sum(1 for r in rows if r["pairs"] > 0),
        "archive_sha256": sha256_file(archive),
        "weights_sha256": sha256_file(args.weights),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "machine": machine(),
        "elapsed_seconds": round(time.time() - started),
    }
    report.update(summarise(rows))
    if milliseconds:
        ordered = sorted(milliseconds)
        report["timing"] = {"median_ms": ordered[len(ordered) // 2], "mean_ms": sum(ordered) / len(ordered)}
    report["images"] = rows
    (args.out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    text = markdown(report)
    (args.out / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    if not report["gate"]["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Write ``MODEL_CARD.md`` for a `jdd` run (docs/07 "Parity and versioning", docs/14 rule 5).

    python3 -m training.jdd.model_card --run runs/jdd-001 --out data/models/jdd/MODEL_CARD.md

The card lists purpose, architecture, parameters, receptive field and halo, every training dataset with
its licence, the metrics of the run's evaluation, known failure cases, date and the git SHA of the
training code. A run made with ``--dry-run`` (or whose config is flagged so) is stamped "DRY RUN" and
must never be shipped. Nothing is invented: metrics that were not measured are marked "not evaluated".
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path

from .model import JddConfig, JddNet, count_flops, count_parameters
from .receptive_field import analyse
from .weights_io import load_weights, read_metadata, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[2]

LICENCES = {
    "pixelshift200": ("PixelShift200 (Sony A7R III, 4-shot pixel shift)", "CC BY 4.0", "main ground truth; attribution required (also in NOTICE)"),
    "synthetic": ("Procedural synthetic patterns (zone plates, text, fabrics, star fields, textures)", "ours, Apache-2.0", "moire and false-colour robustness"),
    "noise": ("Virtual-camera noise sampler", "physics and the published ELD relations (method only; ELD data not used)", "noise synthesis"),
}
NOT_USED = "DND, SID, ELD data, Cube++ (evaluation only or other model), any 'found' images: never used for these weights."
KNOWN_FAILURE_CANDIDATES = [
    "Zipper and false colour on near-Nyquist periodic detail (zone plates, fine fabrics): check in blind review of 30 crops.",
    "cRAW 'blondie' stripes: compare against LibRaw + classical on 10 A7R cRAW samples.",
    "Clipped highlights: the network passes clipped values through; reconstruction is stage R5, before this network.",
    "Hi-ISO files that the camera already filtered (docs/06 filtered-file check): sigma may be under-estimated.",
]


def git_info() -> dict[str, str]:
    """Commit SHA of the repository and whether ``training/jdd`` has uncommitted changes."""
    def run(*a: str) -> str:
        return subprocess.run(["git", *a], cwd=REPO_ROOT, capture_output=True, text=True, check=False).stdout.strip()

    sha = run("rev-parse", "HEAD") or "unknown"
    dirty = bool(run("status", "--porcelain", "--", "training/jdd"))
    return {"sha": sha, "dirty": "yes" if dirty else "no"}


def render(run_dir: Path, version: str, date: str | None = None) -> str:
    """Build the card text for ``run_dir`` (needs ``weights.safetensors``; uses config/eval/summary when present)."""
    weights = run_dir / "weights.safetensors"
    model: JddNet = load_weights(weights)
    cfg_json = json.loads((run_dir / "config.json").read_text()) if (run_dir / "config.json").exists() else {}
    ev = json.loads((run_dir / "eval.json").read_text()) if (run_dir / "eval.json").exists() else None
    summ = json.loads((run_dir / "summary.json").read_text()) if (run_dir / "summary.json").exists() else {}
    meta = read_metadata(weights)
    rf = analyse(model)
    fl = count_flops(model, 64, 64)
    git = git_info()
    date = date or dt.datetime.now(dt.UTC).strftime("%Y-%m-%d")
    dry = bool(cfg_json.get("dry_run")) or "dry" in cfg_json.get("name", "")
    arch: JddConfig = model.cfg
    data = cfg_json.get("data", {})
    used = []
    if data.get("pixelshift200_weight", 0) > 0:
        used.append("pixelshift200")
    if data.get("synthetic_weight", 1.0) > 0 or not used:
        used.append("synthetic")
    used.append("noise")

    lines = [f"# jdd model card ({version})", ""]
    if dry:
        lines += ["> **DRY RUN. These weights come from a 50-step CPU smoke test on synthetic data. They are not a release candidate and must not be committed to `data/models/`.**", ""]
    lines += [
        "## Purpose", "",
        "Joint demosaic and denoise (and cRAW de-banding) of Sony Bayer mosaics, run on 1:1 tiles at export. "
        "Input: four packed Bayer planes balanced with the initial white-balance gains, a sigma map, a "
        "quantisation-step map and a clip mask, at half resolution. Output: linear camera RGB at full "
        "resolution (docs/07-ml-models.md).", "",
        "## Architecture", "",
        "| Item | Value |", "|---|---|",
        f"| Layout | U-Net, 3 levels, channels {'/'.join(map(str, arch.channels))}, {arch.convs_per_level} x 3x3 conv + SiLU per level, strided-conv downsampling, pixel-shuffle upsampling ({arch.up_kernel}x{arch.up_kernel} conv), skip concat, {arch.head_kernel}x{arch.head_kernel} head to 12 channels + pixel shuffle x2, added to a fixed bilinear demosaic |",
        f"| Parameters | {count_parameters(model):,} |",
        f"| Receptive field | {rf.rf_half_px} half-res px = {rf.rf_full_px} full-res px (limit 128) |",
        f"| Tile halo | {rf.halo_full_px} full-res px = {rf.halo_half_px} half-res px (crop, not blend) |",
        f"| FLOPs per output pixel | {fl['per_output_pixel'] / 1e3:.1f} k (budget 250 k) |",
        "| Tile size | multiple of 4 half-res px; ONNX is static-shape (opset 17) |",
        f"| Weights file | `{weights.name}`, SHA-256 `{sha256_file(weights)}`, version `{meta.get('model_version', version)}` |", "",
        "## Training data and licences", "",
        "| Dataset | Licence | Use |", "|---|---|---|",
    ]
    for key in used:
        name, lic, use = LICENCES[key]
        lines.append(f"| {name} | {lic} | {use} |")
    lines += ["", f"Not used: {NOT_USED}", ""]
    lines += ["## Training procedure", ""]
    if cfg_json:
        t, o = cfg_json.get("train", {}), cfg_json.get("optim", {})
        lines += [
            f"Seed {cfg_json.get('seed')}, {t.get('steps')} steps, batch {t.get('batch_size')}, patch {data.get('patch_half')} half-res px, "
            f"AdamW lr {o.get('lr')}, warm-up {o.get('warmup_steps')} steps then cosine, device `{summ.get('device', cfg_json.get('device'))}`"
            + (f", {summ['train_seconds']} s" if "train_seconds" in summ else "")
            + ". Loss: L1 on log(1 + 16x) + 0.2 x (1 - MS-SSIM) + gradient-domain L1 (docs/07).", "",
        ]
    else:
        lines += ["Configuration not available for this run.", ""]
    lines += ["## Metrics", ""]
    if ev:
        lines += [
            "Validation set: synthetic samples from the training generator (noise, cRAW, clipping), held-out seed range. "
            "PSNR in dB, linear peak 1.0 and log domain peak log(1 + 16).", "",
            "| Metric | Model | Bilinear |", "|---|---|---|",
            f"| PSNR linear | {ev['model_linear']['mean']:.2f} | {ev['bilinear_linear']['mean']:.2f} |",
            f"| PSNR log | {ev['model_log']['mean']:.2f} | {ev['bilinear_log']['mean']:.2f} |", "",
        ]
    else:
        lines += ["Not evaluated.", ""]
    lines += [
        "Gate sets (docs/07: Kodak, McMaster, Gharbi hard and moire, held-out PixelShift200 scenes; >= 2 dB over the "
        "classical fallback): not evaluated for this run. Run `python3 -m training.jdd.evaluate` on each set and "
        "append the numbers here.", "",
        "## Known failure cases", "",
        "None recorded from evidence yet. Candidates to check at the gate (docs/07):", "",
    ]
    lines += [f"- {c}" for c in KNOWN_FAILURE_CANDIDATES]
    lines += [
        "", "## Provenance", "",
        f"- Date: {date}",
        f"- Training code: git `{git['sha']}`, uncommitted changes in `training/jdd`: {git['dirty']}",
        "- Card generated by `training/jdd/model_card.py`",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None, help="default: <run>/MODEL_CARD.md")
    ap.add_argument("--version", default="0")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (default: today, UTC)")
    args = ap.parse_args(argv)
    text = render(args.run, args.version, args.date)
    out = args.out or (args.run / "MODEL_CARD.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

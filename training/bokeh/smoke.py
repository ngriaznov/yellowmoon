"""The whole pipeline on a handful of photos, on the CPU, for plumbing and for looking at the synthesis. No Modal.

    python -m training.bokeh.smoke --work <folder outside the repository> --n 50 --steps 40

Steps (each skips what is already in `--work`): download `n` Open Images validation photos listed CC BY 2.0 that have
instance masks (the official mirror), read their masks, build tiles, render preview sheets of the synthesised inputs and
labels (`synth-sheet-*.png`), train a few steps, score the model and the classical baseline on synthetic scenes
(`run/metrics.json`), and export the weights and the parity fixture, for the trained model and for seeded random weights
(`export/`). Everything lands in `--work`, never in the repository. A few steps on a CPU teach the network little: this is
a plumbing check, and the numbers it prints are not results.
"""

import argparse
import json
from pathlib import Path

import torch

from training.bokeh import export, sources
from training.bokeh import train as trainer
from training.bokeh.model import BokehNet
from training.crop import openimages

SEG_URL = sources.SEG_URL.format(subset="validation")


def fetch(work: Path, n: int, seed: int = 1, digit: str = "0") -> list[dict]:
    """`n` photos and their masks into `work/photos`, `work/masks`; returns the manifest rows."""
    manifest_path = work / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    lists = work / "lists"
    lists.mkdir(parents=True, exist_ok=True)
    listing = openimages.read_list("validation", lists / "validation-images.csv")
    seg = lists / "validation-segmentation.csv"
    if not seg.exists():
        seg.write_bytes(sources.open_url(SEG_URL).read())
    with seg.open(encoding="utf-8", newline="") as f:
        first = {line.split(",")[1] for line in f if line[0] == digit}
    rows = (r for r in openimages.rows(listing) if r["ImageID"] in first)
    chosen = openimages.select(rows, n, seed)
    got = openimages.download(chosen, work / "photos", workers=16)
    with seg.open(encoding="utf-8", newline="") as f:
        idx = sources.mask_index(f, {r["id"] for r in got})
    names = {m["path"] for v in idx.values() for m in v}
    zip_path = lists / f"validation-masks-{digit}.zip"
    if not zip_path.exists():
        zip_path.write_bytes(sources.open_url(sources.MASKS_URL.format(subset="validation", c=digit)).read())
    count = sources.fetch_mask_pack("validation", digit, names, work / "masks" / digit, local_zip=zip_path)
    for r in got:
        r["masks"] = idx.get(r["id"], [])
    manifest_path.write_text(json.dumps(got, indent=1, ensure_ascii=False))
    print(f"{len(got)} photos, {count} masks")
    return got


def build_tiles(work: Path, rows: list[dict]) -> int:
    out = work / "tiles"
    if (out / "shard-00000.json").exists():
        return len(sources.TileSet(out))
    packs = sources.MaskPacks(work / "masks")
    items, report = [], {}
    for r in rows:
        pngs = [p for m in r["masks"] if (p := packs.png(m["path"])) is not None]
        tiles, rep = sources.photo_tiles(work / "photos" / "images" / f"{r['id']}.jpg", r["id"], pngs)
        report[r["id"]] = rep
        items += tiles
    sources.write_shard(out / "shard-00000", items)
    (work / "tile-report.json").write_text(json.dumps(report, indent=1))
    reasons = {k: sum(1 for v in report.values() if v["reason"] == k) for k in {v["reason"] for v in report.values()}}
    print(f"{len(items)} tiles from {reasons.get('ok', 0)} of {len(rows)} photos; reasons: {reasons}")
    return len(items)


def previews(work: Path) -> list[Path]:
    from training.bokeh import previews as pv

    return pv.render(work / "tiles", work, sheets=3, rows=6, seed=100)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--stage", choices=["data", "previews", "train", "export", "all"], default="all")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    args.work.mkdir(parents=True, exist_ok=True)
    rows = fetch(args.work, args.n)
    build_tiles(args.work, rows)
    if args.stage in ("previews", "all"):
        previews(args.work)
    run = args.work / "run"
    if args.stage in ("train", "all") and not (run / "metrics.json").exists():
        config = {"sample": 192, "batch": 4, "kernel_r": 20, "r_max_px": 14.0, "device": "cpu", "workers": 1, "amp": False,
                  "steps": args.steps, "warmup": max(2, args.steps // 10), "lr": 3e-3, "eval_every_min": 0.0,
                  "final_val_samples": 24, "baseline_n": 12, "val_share": 0.15, "ckpt_every_min": 5.0}
        trainer.train(args.work / "tiles", run, config)
    if args.stage in ("export", "all"):
        model = BokehNet()
        model.load_state_dict(torch.load(run / "model.pt", map_location="cpu", weights_only=False)["state_dict"])
        print(json.dumps(export.write(model, args.work / "export" / "bokehnet.safetensors", args.work / "export" / "bokehnet_parity.json", "smoke")))
        rnd = BokehNet()
        export.randomize(rnd, 1)
        print(json.dumps(export.write(rnd, args.work / "export" / "random.safetensors", args.work / "export" / "random_parity.json", "random-seed-1")))


if __name__ == "__main__":
    main()

"""The whole pipeline on a handful of photos, on the CPU, for plumbing and for looking at the synthesis. No Modal.

    python -m training.bokeh.smoke --work <folder outside the repository> --n 50

Steps (each skips what is already in `--work`): download `n` Open Images validation photos listed CC BY 2.0 that have
instance masks (the official mirror), read their masks, build tiles, render preview sheets of the synthesised inputs and
labels, train a few steps, export the weights and the parity fixture, and score the model and the classical baseline on
synthetic scenes. Everything lands in `--work`, never in the repository.
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from training.bokeh import sources
from training.crop import openimages

LIST_URL = "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv"
SEG_URL = sources.SEG_URL.format(subset="validation")
ZIP_URL = sources.MASKS_URL


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
        zip_path.write_bytes(sources.open_url(ZIP_URL.format(subset="validation", c=digit)).read())
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
    kept = sum(1 for v in report.values() if v["reason"] == "ok")
    print(f"{len(items)} tiles from {kept} of {len(rows)} photos; "
          f"reasons: { {k: sum(1 for v in report.values() if v['reason'] == k) for k in {v['reason'] for v in report.values()}} }")
    return len(items)


def previews(work: Path, rows_per_sheet: int = 6, sheets: int = 3) -> list[Path]:
    from training.bokeh.synth import SynthConfig, TileBatch, preview_grid, synthesize

    ts = sources.TileSet(work / "tiles")
    cfg = SynthConfig()
    paths = []
    for k in range(sheets):
        gen = torch.Generator().manual_seed(100 + k)
        idx = [ts.train[(k * rows_per_sheet + j) % len(ts.train)] for j in range(rows_per_sheet)]
        got = [ts.get(i) for i in idx]
        tb = TileBatch(torch.from_numpy(np.stack([g[0] for g in got])), torch.from_numpy(np.stack([g[1] for g in got])),
                       torch.from_numpy(np.stack([g[2] for g in got])), torch.tensor([g[3] for g in got]))
        t0 = time.time()
        sy = synthesize(tb, cfg, gen)
        print(f"sheet {k}: {time.time() - t0:.1f} s; modes {sy.info['mode'].tolist()} night {sy.info['night'].tolist()}")
        path = work / f"synth-sheet-{k}.png"
        Image.fromarray(preview_grid(sy).numpy()).save(path)
        paths.append(path)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--stage", choices=["fetch", "tiles", "previews", "all"], default="all")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    args.work.mkdir(parents=True, exist_ok=True)
    rows = fetch(args.work, args.n)
    if args.stage in ("tiles", "previews", "all"):
        build_tiles(args.work, rows)
    if args.stage in ("previews", "all"):
        previews(args.work)


if __name__ == "__main__":
    main()

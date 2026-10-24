"""The `prepare` step's work, without Modal (`modal_app.py` wraps each function in a call): list the photos the crop
pipeline already holds, index their instance masks, build the tiles chunk by chunk, write the credits.

Layout of the crop volume (read only; `training/crop/modal_app.py`): `data/<set>/manifest.json` (a list of
`{id, subset, licence, author, author_url, source_url, title}`) and `data/<set>/images/<id>.jpg`. Layout of the bokeh
volume: `index/photos.json`, `masks/<c>.bin|.json`, `tiles/shard-<k>.bin|.json`, `tiles/credits.tsv.gz`,
`tiles/tiles.json`, `tiles/reports/<k>.json`.

Work is chunked and each chunk writes its output last and whole (`sources.write_shard`), so a rerun skips what is done.
"""

import gzip
import io
import json
import multiprocessing
import os
import random
import time
from pathlib import Path

from training.bokeh import sources
from training.crop import openimages

CROP_SET = "train-20000-s1"
PRE_SET = "oi-96000-s2"
TILE_CHUNK = 1000
HEX = "0123456789abcdef"


def crop_photos(crop_root: Path, name: str = CROP_SET) -> list[dict]:
    """The manifest rows of the crop set whose image file is on the volume, listed as CC BY 2.0, sorted by id."""
    manifest = json.loads((crop_root / "data" / name / "manifest.json").read_text())
    have = {p[:-4] for p in os.listdir(crop_root / "data" / name / "images") if p.endswith(".jpg")}
    return sorted((r for r in manifest if r["id"] in have and r.get("licence") == sources.CC_BY_2), key=lambda r: r["id"])


def extra_photos(crop_root: Path, taken: set[str], n: int, seed: int) -> list[dict]:
    """`n` more rows from the pretraining set's manifest (their images are not on the volume in a usable size: the
    pretraining arrays are 256 x 256, so these are downloaded from the mirror), none already in `taken`."""
    if n <= 0:
        return []
    rows = [r for r in json.loads((crop_root / "pre" / PRE_SET / "manifest.json").read_text())
            if r["id"] not in taken and r.get("licence") == sources.CC_BY_2]
    rows.sort(key=lambda r: r["id"])
    return sorted(random.Random(seed).sample(rows, min(n, len(rows))), key=lambda r: r["id"])


def build_index(crop_root: Path, vol: Path, photos: int, extra: int, seed: int, seg_lines) -> dict:
    """Writes `index/photos.json` and returns its summary. `photos` limits the crop set's photos (0 = all on the
    volume), `extra` adds photos to download from the mirror. `seg_lines` is a callable giving the segmentation CSV's
    text lines (a stream of the 807 MB file, or a small file in tests)."""
    path = vol / "index" / "photos.json"
    if path.exists():
        doc = json.loads(path.read_text())
        if doc["params"] == {"photos": photos, "extra": extra, "seed": seed}:
            return doc["summary"]
        raise RuntimeError(f"{path} was made with other parameters {doc['params']}; use another volume or delete it")
    rows = [{**r, "source": "crop"} for r in crop_photos(crop_root)]
    if photos and photos < len(rows):
        rows = sorted(random.Random(seed).sample(rows, photos), key=lambda r: r["id"])
    rows += [{**r, "source": "mirror"} for r in extra_photos(crop_root, {r["id"] for r in rows}, extra, seed)]
    with seg_lines() as lines:
        idx = sources.mask_index(lines, {r["id"] for r in rows})
    for r in rows:
        r["masks"] = idx.get(r["id"], [])
    per_char = {c: sum(1 for r in rows for m in r["masks"] if m["path"].startswith(c)) for c in HEX}
    summary = {"photos": len(rows), "from_crop_volume": sum(1 for r in rows if r["source"] == "crop"),
               "with_masks": sum(1 for r in rows if r["masks"]), "masks": sum(per_char.values()),
               "chunks": -(-len(rows) // TILE_CHUNK), "masks_per_archive": per_char}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"params": {"photos": photos, "extra": extra, "seed": seed}, "summary": summary, "photos": rows}))
    os.replace(tmp, path)
    return summary


def load_index(vol: Path) -> dict:
    return json.loads((vol / "index" / "photos.json").read_text())


def mask_archive(vol: Path, c: str, subset: str = "train", local_zip: Path | None = None) -> int:
    """The masks of the indexed photos whose ids start with `c`, from the official archive into `masks/<c>`."""
    names = {m["path"] for r in load_index(vol)["photos"] for m in r["masks"] if m["path"].startswith(c)}
    return sources.fetch_mask_pack(subset, c, names, vol / "masks" / c, local_zip=local_zip)


def _one(args):
    row, jpeg_path, pngs, tile, min_long_edge = args
    try:
        data = Path(jpeg_path).read_bytes() if jpeg_path else openimages.fetch(
            openimages.IMAGE_URL.format(subset=row["subset"], image_id=row["id"]))
        return row["id"], *sources.photo_tiles(data, row["id"], pngs, tile, min_long_edge)
    except Exception as e:  # noqa: BLE001 - a photo that fails to read or decode is left out and reported
        return row["id"], [], {"reason": f"error: {type(e).__name__}"}


def tile_chunk(vol: Path, crop_root: Path, k: int, workers: int = 2, tile: int = sources.TILE,
               min_long_edge: int = sources.MIN_LONG_EDGE) -> dict:
    """Builds chunk `k`'s tiles into `tiles/shard-<k>`; skips a chunk that is already there."""
    prefix = vol / "tiles" / f"shard-{k:05d}"
    t0 = time.time()
    if prefix.with_suffix(".json").exists():
        return {"k": k, "skipped": True, "tiles": len(json.loads(prefix.with_suffix(".json").read_text())["items"]),
                "seconds": 0.0}
    rows = load_index(vol)["photos"][k * TILE_CHUNK:(k + 1) * TILE_CHUNK]
    packs = sources.MaskPacks(vol / "masks")
    jobs = []
    for r in rows:
        pngs = [p for m in r["masks"] if (p := packs.png(m["path"])) is not None]
        path = crop_root / "data" / CROP_SET / "images" / f"{r['id']}.jpg" if r["source"] == "crop" else None
        jobs.append((r, str(path) if path else None, pngs, tile, min_long_edge))
    if workers > 1:
        with multiprocessing.get_context("fork").Pool(workers) as pool:
            done = pool.map(_one, jobs, chunksize=8)
    else:
        done = [_one(j) for j in jobs]
    items = [t for _, tiles, _ in done for t in tiles]
    report = {pid: rep for pid, _, rep in done}
    sources.write_shard(prefix, items, tile=tile)
    rep_path = vol / "tiles" / "reports" / f"{k:05d}.json"
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.write_text(json.dumps(report))
    reasons: dict[str, int] = {}
    for rep in report.values():
        reasons[rep["reason"]] = reasons.get(rep["reason"], 0) + 1
    return {"k": k, "skipped": False, "photos": len(rows), "tiles": len(items), "reasons": reasons,
            "seconds": round(time.time() - t0, 2)}


def finalize(vol: Path) -> dict:
    """`tiles/tiles.json` (counts) and `tiles/credits.tsv.gz` (author, page and title of every photo that has a tile,
    in the format of `training/crop/CREDITS.tsv.gz`)."""
    doc = load_index(vol)
    used: set[str] = set()
    tiles = with_cands = 0
    for meta in sorted((vol / "tiles").glob("shard-*.json")):
        for r in json.loads(meta.read_text())["items"]:
            used.add(r["photo"])
            tiles += 1
            with_cands += bool(r["cands"])
    rows = sources.credits_rows(doc["photos"], used)
    buf = io.StringIO()
    buf.write("image_id\tuse\tauthor\tauthor_url\tsource_url\ttitle\n")
    for row in rows:
        buf.write("\t".join(f.replace("\t", " ").replace("\n", " ") for f in row) + "\n")
    # mtime 0: the same bytes for the same rows
    with gzip.GzipFile(filename="", mode="wb", fileobj=(raw := io.BytesIO()), mtime=0) as g:
        g.write(buf.getvalue().encode("utf-8"))
    (vol / "tiles" / "credits.tsv.gz").write_bytes(raw.getvalue())
    summary = {"tiles": tiles, "photos": len(used), "tiles_with_instances": with_cands, "index": doc["summary"]}
    (vol / "tiles" / "tiles.json").write_text(json.dumps(summary, indent=1))
    return summary


def probe(crop_root: Path, n: int = 40, seed: int = 1, tile: int = sources.TILE, min_long_edge: int = sources.MIN_LONG_EDGE) -> dict:
    """What `check` reports: does the crop volume hold the photos, can they be read and filtered, how long it takes."""
    rows = crop_photos(crop_root)
    out: dict = {"photos_on_volume": len(rows), "manifest_rows": len(json.loads((crop_root / "data" / CROP_SET / "manifest.json").read_text()))}
    pre = crop_root / "pre" / PRE_SET
    out["pretraining_manifest"] = (pre / "manifest.json").exists()
    out["pretraining_arrays"] = len(list((pre / "arrays").glob("*"))) if (pre / "arrays").exists() else 0
    sample = random.Random(seed).sample(rows, min(n, len(rows)))
    t0 = time.time()
    kept = tiles = 0
    sizes = []
    for r in sample:
        data = (crop_root / "data" / CROP_SET / "images" / f"{r['id']}.jpg").read_bytes()
        got, rep = sources.photo_tiles(data, r["id"], [], tile, min_long_edge)
        kept += bool(got)
        tiles += len(got)
        sizes.append(len(data))
    dt = time.time() - t0
    out.update({"sampled": len(sample), "photos_with_tiles": kept, "tiles": tiles,
                "pass_rate": round(kept / max(1, len(sample)), 3), "seconds_per_photo": round(dt / max(1, len(sample)), 3),
                "mean_jpeg_kb": round(sum(sizes) / max(1, len(sizes)) / 1024, 1)})
    return out

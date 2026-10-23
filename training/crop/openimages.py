"""Open Images V7 photos for the crop pilot: a seeded selection of CC BY 2.0 images with their attribution, and
their download.

Open Images' image list (`<subset>-images...-with-rotation.csv`, CC BY 4.0) carries each image's licence, author and
Flickr page; every row checked so far is CC BY 2.0, and rows with any other licence are skipped all the same. Images
whose stored pixels need a rotation (the `Rotation` column is not 0) are skipped too, so the pixels are upright as
stored. The images come from the CVDF mirror (`open-images-dataset` bucket), already resized by Open Images.

    python3 -m training.crop.openimages --subset validation --n 24 --seed 1 --out testdata/crop/smoke
"""

import argparse
import csv
import json
import random
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LISTS = {
    "train": "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv",
    "validation": "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv",
}
IMAGE_URL = "https://s3.amazonaws.com/open-images-dataset/{subset}/{image_id}.jpg"
CC_BY_2 = "https://creativecommons.org/licenses/by/2.0/"
AGENT = {"User-Agent": "yellowmoon-crop-pilot/0.1"}


def fetch(url: str, attempts: int = 4) -> bytes:
    """The bytes at `url`, retried with a growing pause."""
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=AGENT), timeout=120) as r:
                return r.read()
        except Exception:  # noqa: BLE001 - network errors of any kind are retried
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** (attempt + 1))
    raise AssertionError("unreachable")


def select(rows, n: int, seed: int) -> list[dict]:
    """`n` upright CC BY 2.0 rows of an Open Images image list, chosen with `seed` (the same rows for the same list,
    `n` and seed), with the fields the attribution manifest needs."""
    keep = ("ImageID", "Subset", "License", "Author", "AuthorProfileURL", "OriginalLandingURL", "Title")
    eligible = [
        {k: r[k] for k in keep} for r in rows
        if r["License"] == CC_BY_2 and r.get("Rotation", "") in ("0", "0.0")
    ]
    eligible.sort(key=lambda r: r["ImageID"])
    chosen = random.Random(seed).sample(eligible, min(n, len(eligible)))
    chosen.sort(key=lambda r: r["ImageID"])
    return [
        {
            "id": r["ImageID"],
            "subset": r["Subset"],
            "licence": r["License"],
            "author": r["Author"],
            "author_url": r["AuthorProfileURL"],
            "source_url": r["OriginalLandingURL"],
            "title": r["Title"],
        }
        for r in chosen
    ]


def read_list(subset: str, cache: Path) -> Path:
    """Open Images' image list for `subset`, downloaded once into `cache` (the train list is 640 MB)."""
    if not (cache.exists() and cache.stat().st_size > 0):
        cache.parent.mkdir(parents=True, exist_ok=True)
        part = cache.with_suffix(".part")
        with urllib.request.urlopen(urllib.request.Request(LISTS[subset], headers=AGENT), timeout=600) as r:
            with part.open("wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
        part.rename(cache)
    return cache


def rows(path: Path):
    """The rows of a downloaded image list, one at a time."""
    with path.open(encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)


def download(chosen: list[dict], out: Path, workers: int = 32) -> list[dict]:
    """Downloads the chosen images into `out/images/<id>.jpg` (skipping files already there) and returns the rows
    whose image arrived; the others are left out of the manifest."""
    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)

    def one(row):
        path = images / f"{row['id']}.jpg"
        if path.exists() and path.stat().st_size > 0:
            return row
        try:
            data = fetch(IMAGE_URL.format(subset=row["subset"], image_id=row["id"]))
        except Exception:  # noqa: BLE001 - a missing image is left out, not fatal
            return None
        if data[:2] != b"\xff\xd8":
            return None
        path.write_bytes(data)
        return row

    with ThreadPoolExecutor(workers) as pool:
        got = [r for r in pool.map(one, chosen) if r is not None]
    return got


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--subset", choices=sorted(LISTS), default="train")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    listing = read_list(args.subset, args.out / f"{args.subset}-images.csv")
    chosen = select(rows(listing), args.n, args.seed)
    got = download(chosen, args.out)
    (args.out / "manifest.json").write_text(json.dumps(got, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(got)} of {len(chosen)} images in {args.out / 'images'}")


if __name__ == "__main__":
    main()

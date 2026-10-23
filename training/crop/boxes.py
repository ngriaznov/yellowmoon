"""Pretraining data for the crop student's backbone (`pretrain.py`): Open Images photos with the boxes people drew
around their objects, turned into maps of where each kind of thing is and the set of classes present.

Every photo of Open Images' boxable train set carries boxes around the objects of 600 classes (annotations CC BY 4.0,
photos listed as CC BY 2.0): the same licences as the crop pilot's photos, and no model in the chain. Each photo gets
`CHANNELS` maps of `MAP` x `MAP` cells over the letterboxed square, each cell the share of it covered by boxes of that
kind (any object, people, faces, animals, vehicles, buildings), and the classes present as bits. Needs no torch, so it
runs in Modal's small CPU image.

    python3 -m training.crop.boxes index --subset validation --n 64 --chunk 32 --exclude <crop manifest.json> \
        --lists testdata/crop/lists --out testdata/crop/pre
    python3 -m training.crop.boxes chunk --root testdata/crop/pre --k 0
"""

import argparse
import csv
import io
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from training.crop import openimages
from training.crop.data import SIZE, letterbox

#: The map channels; "object" is every box, the others the classes under these names in Open Images' hierarchy.
CHANNELS = ("object", "person", "face", "animal", "vehicle", "building")
GROUPS = {
    "person": ("Person",),
    "face": ("Human face", "Human head"),
    "animal": ("Animal",),
    "vehicle": ("Vehicle",),
    "building": ("Building",),
}
#: Cells per side of the maps (stride 16 on the 256 px square).
MAP = SIZE // 16
#: Each cell is rasterised at this many points per side to measure its coverage.
RASTER = 4
BASE = "https://storage.googleapis.com/openimages/2018_04"
BOXES = {
    "train": f"{BASE}/train/train-annotations-bbox.csv",
    "validation": f"{BASE}/validation/validation-annotations-bbox.csv",
}
CLASSES = f"{BASE}/class-descriptions-boxable.csv"
HIERARCHY = f"{BASE}/bbox_labels_600_hierarchy.json"


def download(url: str, path: Path) -> Path:
    """`url` saved at `path` once (kept for later runs)."""
    if not (path.exists() and path.stat().st_size > 0):
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(urllib.request.Request(url, headers=openimages.AGENT), timeout=1200) as r:
            with part.open("wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
        part.rename(path)
    return path


def class_names(path: Path) -> dict[str, str]:
    """Open Images' boxable classes, `{mid: display name}` in the file's order."""
    with path.open(encoding="utf-8", newline="") as f:
        return {row[0]: row[1] for row in csv.reader(f) if len(row) >= 2}


def channel_classes(hierarchy: dict, names: dict[str, str]) -> dict[str, set[str]]:
    """For each channel but "object", the classes it covers: the named classes and everything under them (the
    hierarchy's `Subcategory` links, not its `Part` links)."""
    by_name = {v: k for k, v in names.items()}

    def under(node: dict) -> set[str]:
        out = {node["LabelName"]}
        for child in node.get("Subcategory", []):
            out |= under(child)
        return out

    def find(node: dict, mid: str) -> dict | None:
        if node["LabelName"] == mid:
            return node
        for key in ("Subcategory", "Part"):
            for child in node.get(key, []):
                if (hit := find(child, mid)) is not None:
                    return hit
        return None

    out = {}
    for channel, wanted in GROUPS.items():
        mids: set[str] = set()
        for name in wanted:
            node = find(hierarchy, by_name[name])
            mids |= under(node) if node is not None else {by_name[name]}
        out[channel] = mids
    return out


def read_boxes(path: Path, wanted: set[str]) -> dict[str, list[list]]:
    """`{image id: [[mid, x0, x1, y0, y1], ...]}` for the images in `wanted`, from a box annotation file."""
    out: dict[str, list[list]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        head = next(reader)
        i_id, i_label = head.index("ImageID"), head.index("LabelName")
        i_box = [head.index(k) for k in ("XMin", "XMax", "YMin", "YMax")]
        for row in reader:
            if row[i_id] in wanted:
                out.setdefault(row[i_id], []).append([row[i_label], *(float(row[i]) for i in i_box)])
    return out


def box_maps(boxes: list[list], content: np.ndarray, channels: dict[str, set[str]], size: int = MAP) -> np.ndarray:
    """(len(CHANNELS), size, size) uint8: the share of each cell of the letterboxed square that boxes of each channel
    cover (255 = all of it). `content` is the photo's box in the square (`data.letterbox`)."""
    n = size * RASTER
    centres = (np.arange(n) + 0.5) / n
    out = np.zeros((len(CHANNELS), n, n), bool)
    cx0, cy0, cx1, cy1 = (float(v) for v in content)
    for mid, x0, x1, y0, y1 in boxes:
        sx0, sx1 = cx0 + x0 * (cx1 - cx0), cx0 + x1 * (cx1 - cx0)
        sy0, sy1 = cy0 + y0 * (cy1 - cy0), cy0 + y1 * (cy1 - cy0)
        cols = (centres >= sx0) & (centres < sx1)
        rows = (centres >= sy0) & (centres < sy1)
        hit = rows[:, None] & cols[None, :]
        for c, name in enumerate(CHANNELS):
            if name == "object" or mid in channels[name]:
                out[c] |= hit
    share = out.reshape(len(CHANNELS), size, RASTER, size, RASTER).mean(axis=(2, 4))
    return np.round(share * 255).astype(np.uint8)


def class_bits(boxes: list[list], index: dict[str, int]) -> np.ndarray:
    """The classes present among `boxes`, as packed bits over `index`'s order."""
    hot = np.zeros(len(index), np.uint8)
    for b in boxes:
        if b[0] in index:
            hot[index[b[0]]] = 1
    return np.packbits(hot)


def index(subset: str, n: int, seed: int, chunk: int, exclude: set[str], lists: Path, out: Path) -> int:
    """Selects `n` upright CC BY 2.0 photos of `subset` that are not in `exclude` (the crop set), reads their boxes,
    and writes `out/manifest.json` (attribution), `classes.json`, `channels.json` and one `chunks/<k>.json` per
    `chunk` photos (each photo's manifest row and boxes). Returns the number of chunks."""
    if (out / "chunks.json").exists():
        return json.loads((out / "chunks.json").read_text())["chunks"]
    listing = openimages.read_list(subset, lists / f"{subset}-images.csv")
    rows = (r for r in openimages.rows(listing) if r["ImageID"] not in exclude)
    chosen = openimages.select(rows, n, seed)
    boxes = read_boxes(download(BOXES[subset], lists / f"{subset}-annotations-bbox.csv"), {r["id"] for r in chosen})
    chosen = [r for r in chosen if r["id"] in boxes]
    names = class_names(download(CLASSES, lists / "class-descriptions-boxable.csv"))
    hierarchy = json.loads(download(HIERARCHY, lists / "bbox_labels_600_hierarchy.json").read_text())
    channels = channel_classes(hierarchy, names)
    out.mkdir(parents=True, exist_ok=True)
    (out / "classes.json").write_text(json.dumps(list(names)))
    (out / "channels.json").write_text(json.dumps({k: sorted(v) for k, v in channels.items()}))
    (out / "manifest.json").write_text(json.dumps(chosen, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "chunks").mkdir(exist_ok=True)
    count = 0
    for k in range(0, len(chosen), chunk):
        part = [{**r, "boxes": boxes[r["id"]]} for r in chosen[k:k + chunk]]
        (out / "chunks" / f"{k // chunk:05d}.json").write_text(json.dumps(part))
        count += 1
    (out / "chunks.json").write_text(json.dumps({"chunks": count, "photos": len(chosen)}))
    return count


def build_chunk(root: Path, k: int, workers: int = 32) -> int:
    """Downloads chunk `k`'s photos and writes `arrays/<k>/` (`images.npy`, `maps.npy`, `classes.npy`, `ids.json`);
    photos that fail to arrive are left out. Returns the photos written."""
    out = root / "arrays" / f"{k:05d}"
    if (out / "ids.json").exists():
        return len(json.loads((out / "ids.json").read_text()))
    part = json.loads((root / "chunks" / f"{k:05d}.json").read_text())
    names = json.loads((root / "classes.json").read_text())
    where = {m: i for i, m in enumerate(names)}
    channels = {c: set(v) for c, v in json.loads((root / "channels.json").read_text()).items()}

    def one(row):
        try:
            data = openimages.fetch(openimages.IMAGE_URL.format(subset=row["subset"], image_id=row["id"]))
            with Image.open(io.BytesIO(data)) as img:
                img.draft("RGB", (2 * SIZE, 2 * SIZE))
                pix, content = letterbox(img)
        except Exception:  # noqa: BLE001 - a photo that fails to arrive or decode is left out
            return None
        return row["id"], pix, box_maps(row["boxes"], content, channels), class_bits(row["boxes"], where)

    with ThreadPoolExecutor(workers) as pool:
        got = [g for g in pool.map(one, part) if g is not None]
    tmp = out.with_name(out.name + ".part")
    tmp.mkdir(parents=True, exist_ok=True)
    empty = {"images": (SIZE, SIZE, 3), "maps": (len(CHANNELS), MAP, MAP), "classes": ((len(names) + 7) // 8,)}
    for i, key in enumerate(empty, start=1):
        np.save(tmp / f"{key}.npy", np.stack([g[i] for g in got]) if got else np.zeros((0, *empty[key]), np.uint8))
    (tmp / "ids.json").write_text(json.dumps([g[0] for g in got]))
    tmp.rename(out)
    return len(got)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("index")
    a.add_argument("--subset", choices=sorted(BOXES), default="train")
    a.add_argument("--n", type=int, required=True)
    a.add_argument("--seed", type=int, default=2)
    a.add_argument("--chunk", type=int, default=2000)
    a.add_argument("--exclude", type=Path, help="a crop manifest.json whose photos are left out")
    a.add_argument("--lists", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("chunk")
    b.add_argument("--root", type=Path, required=True)
    b.add_argument("--k", type=int, required=True)
    args = ap.parse_args()
    if args.cmd == "index":
        exclude = {r["id"] for r in json.loads(args.exclude.read_text())} if args.exclude else set()
        print(index(args.subset, args.n, args.seed, args.chunk, exclude, args.lists, args.out), "chunks")
    else:
        print(build_chunk(args.root, args.k), "photos")


if __name__ == "__main__":
    main()

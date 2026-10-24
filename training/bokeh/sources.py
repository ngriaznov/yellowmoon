"""The bokeh network's sharp sources: Open Images photos and their instance masks, cut into tiles.

**Photos.** The crop pilot already downloaded Open Images photos (listed CC BY 2.0, at most 1024 px on the long side,
the CVDF mirror) into the Modal volume `yellowmoon-crop`: `data/train-20000-s1/images/<id>.jpg` with
`data/train-20000-s1/manifest.json` (author, Flickr page, licence per photo). They are read from there, not
downloaded again. (The 96,000 pretraining photos on that volume are stored only as 256 x 256 letterboxed arrays,
`pre/oi-96000-s2/arrays`, too small for a sample of 512 px at the 1024-px scale the network runs at; `index_extra`
lists their ids so that a second run can download more photos from the mirror.)

**Masks.** Open Images' instance segmentation (CC BY 4.0): `<subset>-annotations-object-segmentation.csv` lists one
row per instance (`MaskPath`, `ImageID`, `LabelName`, `PredictedIoU`), and the masks are binary PNGs, the size of the
photo's original or a scaled copy of it, in 16 zip archives named by the first hex digit of the image id
(`train-masks-0.zip` ... `-f.zip`, 385 MB each; the official mirror, storage.googleapis.com/openimages/v5).

**Tiles.** Each photo that passes the sharpness filter gives up to two 512 x 512 tiles at its native scale (so the
pixel scale is the one the network runs at), chosen where the strong edges are sharp. The filter is the research
note's mitigation of the recipe's main label noise (a source's own bokeh is labelled "in focus"):
`classical.patch_fractions` reads the blur of every edge in 32-px blocks (a disc radius above 2.5 px is soft); a tile is
kept when at least `SHARP_MIN` of its blocks are sharp-textured and at most `SOFT_MAX` soft-textured, which drops
the shallow-depth-of-field photos (natural bokeh and the flat backdrops that go with it). Each tile is stored as a JPEG (quality 96, no chroma
subsampling) with its instance-id map as a PNG, in shards (`shard-00000.bin` and `.json`), and the ids of the
instances that may be cut out (between 1 % and 60 % of the tile, at least 75 % of the instance inside it).
"""

import csv
import hashlib
import io
import json
import os
import shutil
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from training.bokeh import classical

TILE = 512
#: The frame's long edge must be at least this many pixels (the mirror's photos are at most 1024).
MIN_LONG_EDGE = 700
MAX_CANDS = 8
#: A tile needs this share of sharp-textured 32-px blocks and at most this share of soft-textured ones.
SHARP_MIN = 0.75
SOFT_MAX = 0.04
TILES_PER_PHOTO = 2
VAL_SHARE = 0.04
CC_BY_2 = "https://creativecommons.org/licenses/by/2.0/"
MASKS_URL = "https://storage.googleapis.com/openimages/v5/{subset}-masks/{subset}-masks-{c}.zip"
SEG_URL = "https://storage.googleapis.com/openimages/v5/{subset}-annotations-object-segmentation.csv"
AGENT = {"User-Agent": "yellowmoon-bokeh/0.1"}


# --- masks ----------------------------------------------------------------------------------------------------


def open_url(url: str, attempts: int = 4):
    """A response object for `url`, retried with a growing pause."""
    for attempt in range(attempts):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=AGENT), timeout=300)
        except Exception:  # noqa: BLE001 - network errors of any kind are retried
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** (attempt + 1))
    raise AssertionError("unreachable")


def mask_index(lines, wanted: set[str], min_iou: float = 0.6) -> dict[str, list[dict]]:
    """`{image id: [{"path", "label"}, ...]}` for the images in `wanted`, from the lines of a segmentation CSV (a file
    object or an iterator of text lines). Instances with a predicted IoU between 0 and `min_iou` are dropped (0 means
    "not predicted": the masks were drawn by hand)."""
    out: dict[str, list[dict]] = {}
    reader = csv.reader(lines)
    head = next(reader)
    i_path, i_id, i_label, i_iou = (head.index(k) for k in ("MaskPath", "ImageID", "LabelName", "PredictedIoU"))
    for row in reader:
        if row[i_id] in wanted:
            iou = float(row[i_iou] or 0)
            if iou == 0 or iou >= min_iou:
                out.setdefault(row[i_id], []).append({"path": row[i_path], "label": row[i_label]})
    return out


def stream_lines(url_or_path: str):
    """Text lines of a local file or a URL, streamed (the train CSV is 807 MB)."""
    if url_or_path.startswith("http"):
        resp = open_url(url_or_path)
        return io.TextIOWrapper(resp, encoding="utf-8", newline="")
    return open(url_or_path, encoding="utf-8", newline="")


def fetch_mask_pack(subset: str, c: str, names: set[str], out: Path, local_zip: Path | None = None) -> int:
    """Reads the masks of `names` (mask file names starting with the hex digit `c`) out of the archive
    `<subset>-masks-<c>.zip` into `out` (`.bin` with the PNGs back to back and `.json` with their offsets). The
    archive is downloaded to a temporary file and deleted; `local_zip` uses a copy that is already there. Returns
    the number of masks written. Skips work already done."""
    meta = out.with_suffix(".json")
    if meta.exists():
        return len(json.loads(meta.read_text()))
    out.parent.mkdir(parents=True, exist_ok=True)
    wanted = sorted(n for n in names if n.startswith(c))
    index: dict[str, list[int]] = {}
    tmp_dir = tempfile.mkdtemp(prefix="masks-")
    try:
        path = local_zip
        if path is None:
            path = Path(tmp_dir) / "archive.zip"
            with open_url(MASKS_URL.format(subset=subset, c=c)) as r, path.open("wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
        with zipfile.ZipFile(path) as z, out.with_suffix(".bin.part").open("wb") as f:
            have = set(z.namelist())
            for n in wanted:
                if n in have:
                    data = z.read(n)
                    index[n] = [f.tell(), len(data)]
                    f.write(data)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    os.replace(out.with_suffix(".bin.part"), out.with_suffix(".bin"))
    meta.write_text(json.dumps(index))
    return len(index)


class MaskPacks:
    """Read access to the packs `fetch_mask_pack` wrote, in `folder` (`<c>.bin` and `<c>.json`)."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.index: dict[str, tuple[str, int, int]] = {}
        for meta in sorted(self.folder.glob("*.json")):
            for name, (off, ln) in json.loads(meta.read_text()).items():
                self.index[name] = (meta.stem, off, ln)
        self._open: dict[str, np.memmap] = {}

    def png(self, name: str) -> bytes | None:
        hit = self.index.get(name)
        if hit is None:
            return None
        stem, off, ln = hit
        if stem not in self._open:
            self._open[stem] = np.memmap(self.folder / f"{stem}.bin", dtype=np.uint8, mode="r")
        return bytes(self._open[stem][off:off + ln])


def instance_map(size: tuple[int, int], masks: list[bytes]) -> np.ndarray:
    """(h, w) uint8 ids (1-based, 0 = none) of the instances painted largest first, so a small instance is not hidden
    by a big one over it. `size` is (w, h) of the photo; masks of another size are resized (nearest)."""
    w, h = size
    decoded = []
    for data in masks:
        with Image.open(io.BytesIO(data)) as im:
            m = im.convert("L")
            if m.size != (w, h):
                m = m.resize((w, h), Image.NEAREST)
            decoded.append(np.asarray(m) > 127)
    order = sorted(range(len(decoded)), key=lambda i: -int(decoded[i].sum()))[:255]
    ids = np.zeros((h, w), np.uint8)
    for rank, i in enumerate(order, start=1):
        ids[decoded[i]] = rank
    return ids


# --- tiles ----------------------------------------------------------------------------------------------------


def gray_of(rgb: np.ndarray) -> np.ndarray:
    """(h, w, 3) uint8 sRGB -> (h, w) float in [0, 1], the Rec. 601 luma of the encoded values."""
    f = rgb.astype(np.float64) / 255.0
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def tile_candidates(gray: np.ndarray, tile: int = TILE, stride: int | None = None, n: int = TILES_PER_PHOTO) -> list[dict]:
    """Up to `n` non-overlapping tile positions of a photo, best first, each with its patch statistics
    (`classical.patch_fractions`): a tile is kept when at least `SHARP_MIN` of its 32-px blocks are sharp-textured and at
    most `SOFT_MAX` soft-textured, and it is not black or blown out. That drops photos with natural bokeh, which would
    otherwise be labelled "in focus" (the recipe's main label noise), and the flat backdrops that often go with it."""
    h, w = gray.shape
    if h < tile or w < tile:
        return []
    stride = stride or tile // 4  # 128 px for a 512-px tile
    edge, radius = classical.edge_radii(gray, None)
    found = []
    for y in sorted({*range(0, h - tile + 1, stride), h - tile}):
        for x in sorted({*range(0, w - tile + 1, stride), w - tile}):
            sl = (slice(y, y + tile), slice(x, x + tile))
            tg = gray[sl]
            if tg.mean() < 0.03 or tg.mean() > 0.97:
                continue
            f = classical.patch_fractions(edge[sl], radius[sl])
            if f["sharp"] < SHARP_MIN or f["soft"] > SOFT_MAX:
                continue
            found.append({"y": y, "x": x, **f, "score": f["sharp"] - 2 * f["soft"]})
    found.sort(key=lambda c: (-c["score"], c["y"], c["x"]))
    kept: list[dict] = []
    for c in found:
        if all(abs(c["y"] - k["y"]) >= tile // 2 or abs(c["x"] - k["x"]) >= tile // 2 for k in kept):
            kept.append(c)
        if len(kept) == n:
            break
    return kept


def candidate_ids(tile_ids: np.ndarray, full_ids: np.ndarray) -> list[int]:
    """Ids of the instances that may be cut out of a tile (module docstring), largest first, at most `MAX_CANDS`."""
    area_tile = np.bincount(tile_ids.ravel(), minlength=256)
    area_full = np.bincount(full_ids.ravel(), minlength=256)
    total = tile_ids.size
    ok = [(int(area_tile[i]), i) for i in range(1, 256)
          if 0.01 * total <= area_tile[i] <= 0.6 * total and area_tile[i] >= 0.75 * area_full[i]]
    return [i for _, i in sorted(ok, reverse=True)[:MAX_CANDS]]


def encode_jpeg(rgb: np.ndarray, quality: int = 96) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=quality, subsampling=0)
    return buf.getvalue()


def encode_png(ids: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(ids, "L").save(buf, "PNG", optimize=True)
    return buf.getvalue()


def photo_tiles(jpeg: bytes | Path, photo_id: str, mask_pngs: list[bytes], tile: int = TILE,
                min_long_edge: int = MIN_LONG_EDGE) -> tuple[list[dict], dict]:
    """The tiles of one photo: a list of `{"jpg", "png", "meta"}` (empty when the photo fails the filter) and a
    short report (`reason`, the whole photo's edge statistics)."""
    with Image.open(jpeg if isinstance(jpeg, Path) else io.BytesIO(jpeg)) as im:
        rgb = np.asarray(im.convert("RGB"))
    h, w = rgb.shape[:2]
    if max(h, w) < min_long_edge:
        return [], {"reason": "small"}
    gray = gray_of(rgb)
    cands = tile_candidates(gray, tile)
    if not cands:
        return [], {"reason": "soft_or_flat"}
    full_ids = instance_map((w, h), mask_pngs) if mask_pngs else np.zeros((h, w), np.uint8)
    hd = float(np.hypot(w, h) / 2)
    out = []
    for c in cands:
        sl = (slice(c["y"], c["y"] + tile), slice(c["x"], c["x"] + tile))
        tile_ids = full_ids[sl]
        out.append({
            "jpg": encode_jpeg(np.ascontiguousarray(rgb[sl])),
            "png": encode_png(np.ascontiguousarray(tile_ids)),
            "meta": {"photo": photo_id, "x": c["x"], "y": c["y"], "frame": [w, h], "hd": round(hd, 3),
                     "sharp": round(c["sharp"], 4), "soft": round(c["soft"], 4), "flat": round(c["flat"], 4),
                     "cands": candidate_ids(tile_ids, full_ids)},
        })
    return out, {"reason": "ok", "tiles": len(out)}


# --- shards ---------------------------------------------------------------------------------------------------


def write_shard(prefix: Path, items: list[dict], tile: int = TILE) -> int:
    """Writes `<prefix>.bin` (each item's JPEG then PNG) and `<prefix>.json` (`tile` and, per item, its meta and the
    offsets). Written through temporary files and renamed, the JSON last, so a shard exists only when whole."""
    prefix.parent.mkdir(parents=True, exist_ok=True)
    rows, pos = [], 0
    bin_tmp = prefix.with_suffix(".bin.part")
    with bin_tmp.open("wb") as f:
        for it in items:
            row = dict(it["meta"])
            row["jpg"] = [pos, len(it["jpg"])]
            f.write(it["jpg"])
            pos += len(it["jpg"])
            row["png"] = [pos, len(it["png"])]
            f.write(it["png"])
            pos += len(it["png"])
            rows.append(row)
    os.replace(bin_tmp, prefix.with_suffix(".bin"))
    json_tmp = prefix.with_suffix(".json.part")
    json_tmp.write_text(json.dumps({"tile": tile, "items": rows}))
    os.replace(json_tmp, prefix.with_suffix(".json"))
    return len(rows)


def is_val(photo_id: str, share: float = VAL_SHARE) -> bool:
    """Whether a photo is in the validation split: a hash of its id, stable whatever the set's size. Both tiles of a
    photo go the same way."""
    return int(hashlib.sha256(photo_id.encode()).hexdigest()[:8], 16) / 2**32 < share


class TileSet:
    """All the shards in a folder: `len`, `get(i)` and the train and validation index lists."""

    def __init__(self, folder: Path, val_share: float = VAL_SHARE):
        self.folder = Path(folder)
        self.rows: list[tuple[str, dict]] = []
        self.tile = TILE
        for meta in sorted(self.folder.glob("shard-*.json")):
            doc = json.loads(meta.read_text())
            self.tile = doc["tile"]
            self.rows += [(meta.stem, r) for r in doc["items"]]
        self.train = [i for i, (_, r) in enumerate(self.rows) if not is_val(r["photo"], val_share)]
        self.val = [i for i, (_, r) in enumerate(self.rows) if is_val(r["photo"], val_share)]
        self._maps: dict[str, np.memmap] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def get(self, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """(rgb (T, T, 3) uint8, ids (T, T) uint8, cands (MAX_CANDS,) int64 with 0 padding, hd)."""
        shard, r = self.rows[i]
        if shard not in self._maps:
            self._maps[shard] = np.memmap(self.folder / f"{shard}.bin", dtype=np.uint8, mode="r")
        blob = self._maps[shard]
        (jo, jl), (po, pl) = r["jpg"], r["png"]
        with Image.open(io.BytesIO(bytes(blob[jo:jo + jl]))) as im:
            rgb = np.asarray(im.convert("RGB"))
        with Image.open(io.BytesIO(bytes(blob[po:po + pl]))) as im:
            ids = np.asarray(im)
        cands = np.zeros(MAX_CANDS, np.int64)
        cands[:len(r["cands"])] = r["cands"]
        return rgb, ids, cands, float(r["hd"])


def credits_rows(manifest: list[dict], used: set[str]) -> list[list[str]]:
    """Rows of a credits file like `training/crop/CREDITS.tsv.gz` (image_id, use, author, author_url, source_url,
    title) for the photos in `used`, from a photo manifest (`openimages.select`'s rows)."""
    return [[r["id"], "bokeh", r["author"], r["author_url"], r["source_url"], r["title"]]
            for r in sorted(manifest, key=lambda r: r["id"]) if r["id"] in used]

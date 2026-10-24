"""Shared test material: small synthetic tiles and photos (no network, no real photographs)."""

import io

import numpy as np
import pytest
import torch
from PIL import Image
from scipy import ndimage

from training.bokeh import sources
from training.bokeh.synth import TileBatch

torch.set_num_threads(2)


def blocks_photo(w: int, h: int, seed: int, n: int = 240, soft: float = 0.0) -> np.ndarray:
    """An 8-bit RGB picture of many small hard-edged rectangles (sharp edges everywhere), blurred by `soft` pixels
    when asked (then its edges are soft)."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 0.5)
    for _ in range(n):
        x0, y0 = rng.integers(0, w - 6), rng.integers(0, h - 6)
        img[y0:y0 + rng.integers(3, 18), x0:x0 + rng.integers(3, 18)] = rng.random(3)
    if soft:
        img = ndimage.gaussian_filter(img, (soft, soft, 0))
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def fake_tiles(b: int, t: int, seed: int = 0, with_cands: int = 2) -> TileBatch:
    """`b` tiles of `t` px: textured backgrounds with up to three elliptical instances (ids 1..3 in `ids`)."""
    rng = np.random.default_rng(seed)
    rgb = np.zeros((b, t, t, 3), np.uint8)
    ids = np.zeros((b, t, t), np.uint8)
    cands = np.zeros((b, 8), np.int64)
    yy, xx = np.mgrid[0:t, 0:t]
    for i in range(b):
        base = ndimage.gaussian_filter(rng.random((t, t, 3)), (1.0, 1.0, 0)) * 0.6 + ndimage.gaussian_filter(rng.random((t, t, 1)), (6, 6, 0)) * 1.4
        base = np.clip(base / base.max(), 0, 1)
        for k in range(3 if i < with_cands else 0):
            cx, cy = rng.integers(t // 4, 3 * t // 4, 2)
            rx, ry = rng.integers(t // 8, t // 4, 2)
            m = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 < 1
            base[m] = np.clip(rng.random(3) * 0.8 + 0.2 + 0.2 * rng.random((m.sum(), 3)), 0, 1)
            ids[i][m] = k + 1
            cands[i, k] = k + 1
        rgb[i] = (base * 255).astype(np.uint8)
    return TileBatch(torch.from_numpy(rgb), torch.from_numpy(ids), torch.from_numpy(cands), torch.full((b,), 0.5 * float(np.hypot(t, t))))


def jpeg_of(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG", quality=95, subsampling=0)
    return buf.getvalue()


def mask_png(mask: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray((mask * 255).astype(np.uint8), "L").convert("1").save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture()
def tile_folder(tmp_path):
    """A folder with one shard of small tiles (64 px, two each) from 12 photos of 100 x 100, made by the real tile
    builder. A square mask (36..64) lies inside every tile of every photo, so each tile has one instance to cut out."""
    items = []
    for i in range(12):
        photo = blocks_photo(100, 100, i, n=120)
        mask = np.zeros((100, 100), bool)
        mask[36:64, 36:64] = True
        tiles, _ = sources.photo_tiles(jpeg_of(photo), f"{i:016x}", [mask_png(mask)], tile=64, min_long_edge=100)
        items += tiles
    sources.write_shard(tmp_path / "shard-00000", items, tile=64)
    return tmp_path

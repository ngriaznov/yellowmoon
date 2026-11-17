#!/usr/bin/env python3
"""Dev tool (docs/27 S-09, S-10): contact sheets of the face masks, for a person to look at.

    cargo run --release -p ym-retouch --features mediapipe --example face_masks_dump -- OUT_DIR PHOTO.jpg...
    python tools/mediapipe/contact_sheet.py --masks OUT_DIR --out SHEET_DIR

`face_masks_dump` writes one 8-bit PNG per mask (the size of the photo) and `faces.json`. This script makes one PNG per face,
a row of panels around the face, all on the same crop of the photo:

  1  the photo
  2  skin without ears (green over the photo; the part the tools may change)
  3  eyes (red) and the inside of the mouth (blue)
  4  the under-eye table (yellow)
  5  the non-uniform-skin table (magenta; even skin leaves this alone)
  6  the photo with the skin's edge drawn (a line where the mask crosses 50 %), to judge the edge

Nothing is committed: the sheets are looked at and thrown away (AGENTS.md: no photos in the repository). Dev tooling only;
`--scale N` of `face_masks_dump` enlarges the photo first, and the script enlarges it the same way.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PANEL = 420
MARGIN = 0.45


def load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path))


def overlay(base: np.ndarray, mask: np.ndarray | None, colour, strength=0.65) -> np.ndarray:
    out = base.astype(np.float32)
    if mask is not None:
        a = (mask.astype(np.float32) / 255.0)[..., None] * strength
        out = out * (1 - a) + np.array(colour, np.float32) * a
    return out.clip(0, 255).astype(np.uint8)


def edge(base: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    out = base.copy()
    if mask is None:
        return out
    m = mask > 127
    border = m ^ (np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))
    out[border] = (255, 40, 40)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--masks", type=Path, required=True, help="the directory face_masks_dump wrote")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    faces = json.loads((args.masks / "faces.json").read_text())
    for f in faces:
        photo = Image.open(f["photo"]).convert("RGB")
        n = f["scale"]
        if n > 1:
            photo = photo.resize((photo.width * n, photo.height * n), Image.BILINEAR)
        img = np.asarray(photo)
        h, w = img.shape[:2]
        x, y, rw, rh = f["rect"]
        side = max(rw, rh) * (1 + 2 * MARGIN)
        cx, cy = x + rw / 2, y + rh / 2
        x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
        x1, y1 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
        crop = lambda a: None if a is None else a[y0:y1, x0:x1]
        masks = {k: crop(load(args.masks / v)) for k, v in f["masks"].items()}
        base = crop(img)
        panels = [
            base,
            overlay(base, masks.get("skin"), (0, 220, 60)),
            overlay(overlay(base, masks.get("eyes"), (255, 30, 30), 0.8), masks.get("mouth"), (40, 80, 255), 0.8),
            overlay(base, masks.get("under_eye"), (255, 230, 0)),
            overlay(base, masks.get("non_uniform"), (255, 0, 220)),
            edge(base, masks.get("skin")),
        ]
        scale = PANEL / base.shape[0]
        tiles = [Image.fromarray(p).resize((int(base.shape[1] * scale), PANEL), Image.BILINEAR) for p in panels]
        sheet = Image.new("RGB", (sum(t.width for t in tiles) + 4 * (len(tiles) - 1), PANEL), (20, 20, 20))
        ox = 0
        for t in tiles:
            sheet.paste(t, (ox, 0))
            ox += t.width + 4
        name = f"{Path(f['photo']).stem}_{f['face']}.png"
        ImageDraw.Draw(sheet).text((6, 6), f"{name}  {rw}x{rh}px  skin {f['skin_ms']:.0f} ms", fill=(255, 255, 255))
        sheet.save(args.out / name)
        print(args.out / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

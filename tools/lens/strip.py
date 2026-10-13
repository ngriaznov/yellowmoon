#!/usr/bin/env python3
"""Side-by-side crops of the three R9 variants of `crates/ym-core/examples/lens_eval_dump.rs`
(`--r9`): no CA correction, R9 with the CA after the demosaic (the previous chain), and the CA
corrected on the mosaic (R5b).  Local evaluation only (real files); writes a PNG under
`testdata/lens-eval/`.

    python3 -I tools/lens/strip.py DUMP_DIR STEM [METHOD] [--at=x,y] [--size=96] [--zoom=4]

Without `--at` the crop is the 96 px block, at least 55% of the half diagonal from the centre, where
the red plane is displaced most (from `raw_ca_<stem>.json`).  The images are camera RGB, balanced by
the as-shot gains, shown with a 99.5th-percentile white and a 2.2 gamma: only the fringes matter.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "testdata" / "lens-eval"


def read_f32(path):
    a = np.fromfile(path, dtype="<u4", count=2)
    w, h = int(a[0]), int(a[1])
    return np.moveaxis(np.fromfile(path, dtype="<f4", offset=8).reshape(3, h, w), 0, -1)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--"))
    dump, stem = Path(args[0]), args[1]
    method = args[2] if len(args) > 2 else "adaptive"
    size, zoom = int(opts.get("size", 96)), int(opts.get("zoom", 4))
    meta = json.loads((dump / f"{stem}.{method}.json").read_text())
    fw, fh = meta["frame"]
    cx0, cy0, cw, ch = meta["crop"]
    if "at" in opts:
        x, y = (int(v) for v in opts["at"].split(","))
    else:
        rec = json.loads((OUT / f"raw_ca_{stem}.json").read_text())
        hd = 0.5 * np.hypot(fw, fh)
        best = None
        for bx, by, shift, se, gain, corr in rec["blocks"]["red"]:
            rho = np.hypot(bx - fw / 2, by - fh / 2) / hd
            if rho < 0.55 or se > 0.1 or abs(corr) < 0.9:
                continue
            ox, oy = bx - cx0 - size // 2, by - cy0 - size // 2
            if ox < 0 or oy < 0 or ox + size > cw or oy + size > ch:
                continue  # outside the auto crop
            if best is None or abs(shift) > abs(best[0]):
                best = (shift, bx, by)
        x, y = int(best[1] - cx0 - size // 2), int(best[2] - cy0 - size // 2)
        print(f"crop at {x},{y} (block shift {best[0]:+.2f} px)")
    panels = []
    for kind in ("r9noca", "r9", "r9mosaic"):
        img = read_f32(dump / f"{stem}.{method}.{kind}.f32")
        panels.append(img[y : y + size, x : x + size])
    luma = 0.3 * panels[0][..., 0] + 0.55 * panels[0][..., 1] + 0.15 * panels[0][..., 2]
    white = max(float(np.percentile(luma, 99.5)), 1e-6)
    tiles = []
    for p in panels:
        v = np.clip(p / white, 0, 1) ** (1 / 2.2)
        im = Image.fromarray((v * 255 + 0.5).astype(np.uint8)).resize((size * zoom, size * zoom), Image.NEAREST)
        tiles.append(im)
    sheet = Image.new("RGB", (size * zoom * 3 + 8, size * zoom), (40, 40, 40))
    for i, t in enumerate(tiles):
        sheet.paste(t, (i * (size * zoom + 4), 0))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"strip_{stem}_{method}.png"
    sheet.save(path)
    print(path)


if __name__ == "__main__":
    main()

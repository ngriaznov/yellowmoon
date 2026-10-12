#!/usr/bin/env python3
"""Generates menon_fixtures.json: Bayer mosaics and the output of colour-demosaicing's Menon (2007)
DDFAPD implementation (BSD-3-Clause, Copyright 2015 Colour Developers), so that
crates/ym-gpu/tests/cpu_demosaic.rs can check that our Rust port matches the reference.

Usage: pip3 install colour-demosaicing && python3 -I gen_menon_fixtures.py
Deterministic (seeded). Re-run only when the test scene changes."""
import json
import numpy as np
from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007, masks_CFA_Bayer

W, H = 24, 20
rng = np.random.default_rng(20261012)


def scene():
    img = np.zeros((H, W, 3))
    yy, xx = np.mgrid[0:H, 0:W]
    img[..., 0] = 0.25 + 0.4 * xx / W
    img[..., 1] = 0.30 + 0.3 * yy / H
    img[..., 2] = 0.60 - 0.3 * (xx + yy) / (W + H)
    for _ in range(7):
        x0, y0 = rng.integers(0, W - 6), rng.integers(0, H - 6)
        w, h = rng.integers(4, 12), rng.integers(4, 10)
        img[y0:y0 + h, x0:x0 + w] = rng.uniform(0.05, 0.95, 3)
    # a slanted edge and a thin line
    img[(xx - 2 * yy) > 6] = img[(xx - 2 * yy) > 6][:, ::-1] * 0.8 + 0.1
    img[:, W // 2, :] = [0.9, 0.2, 0.1]
    img += rng.normal(0, 0.01, img.shape)
    return np.clip(img, 0, 1).astype(np.float32).astype(np.float64)


out = []
for pattern in ["RGGB", "GRBG", "GBRG", "BGGR"]:
    img = scene()
    masks = np.stack(masks_CFA_Bayer((H, W), pattern), axis=-1)
    cfa = (img * masks).sum(axis=-1).astype(np.float32).astype(np.float64)
    entry = {
        "pattern": pattern,
        "width": W,
        "height": H,
        "cfa": [float(f"{v:.7g}") for v in cfa.ravel()],
        "menon": [float(f"{v:.7g}") for v in demosaicing_CFA_Bayer_Menon2007(cfa, pattern).ravel()],
        "menon_no_refine": [
            float(f"{v:.7g}") for v in demosaicing_CFA_Bayer_Menon2007(cfa, pattern, refining_step=False).ravel()
        ],
    }
    out.append(entry)

with open("menon_fixtures.json", "w") as f:
    json.dump({"source": "colour-demosaicing 0.2.7 demosaicing_CFA_Bayer_Menon2007", "cases": out}, f)
print("wrote menon_fixtures.json", sum(len(e["cfa"]) for e in out), "mosaic samples")

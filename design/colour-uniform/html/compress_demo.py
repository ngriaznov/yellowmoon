#!/usr/bin/env python3
"""Draws the three pictures of the mock-ups from the drawing portrait-before.png with the sketch of docs/32 (compress_ref.py).

The node has no range of its own: the first pick also adds a Colour range mask row centred on the picked colour (docs/32 section 2.3), and the picture
is the node blended through that mask, `in + m (node(in) - in)`.

The drawing is sRGB; for this sketch it is read as scene-linear light (display gamma removed, scaled so skin sits near middle grey) in place of
AWG4, which is near enough for a mock-up. Writes portrait-before.jpg, portrait-after.jpg and portrait-affected.jpg (the node's mask, the seeded Colour range row, in the mask
accent, as Show mask draws it) and prints the target and the numbers the panel shows.
"""
import json
import math
import os

import numpy as np
from PIL import Image

import compress_ref as R

HERE = os.path.dirname(os.path.abspath(__file__))
PICK = (600, 590)  # the click, in the 1500 x 1000 picture
ROW = R.RANGE_DEFAULT  # the seeded Colour range row: width 0.05 (the slider reads 39), softness 0.5
AMOUNTS = (0.6, 0.6, 0.0)


def to_logc(srgb):
    lin = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4) * 0.8
    return R.encode(lin)


def to_srgb(code):
    lin = np.clip(R.decode(code) / 0.8, 0.0, 1.0)
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)


def main():
    img = np.asarray(Image.open(os.path.join(HERE, "assets", "portrait-before.png")).convert("RGB"), dtype=np.float64) / 255.0
    v = to_logc(img)
    x, y = PICK
    tgt = R.target_from(v[y - 3 : y + 4, x - 3 : x + 4].reshape(-1, 3).mean(axis=0))
    flat = v.reshape(-1, 3)
    m = R.color_range(flat, tgt["chroma"], **ROW)  # the seeded row's centre is the picked chroma
    out = R.with_mask(flat, R.uniform(flat, tgt, *AMOUNTS), m).reshape(v.shape)
    W = m.reshape(v.shape[:2])
    a, b = tgt["chroma"]
    info = dict(
        chroma=[round(a, 4), round(b, 4)],
        hue_deg=round(math.degrees(math.atan2(b, a)) % 360, 1),
        saturation=round(math.hypot(a, b) * 100, 1),
        tone=round(tgt["tone"], 4),
        lightness_ev=round((tgt["tone"] - R.GREY) / R.STOP, 2),
        rgb=[int(round(c * 255)) for c in to_srgb(R.tone_chroma_to_rgb(tgt))],
        affected_share_of_picture=round(float((W > 0.5).mean()), 3),
        row_width=ROW["width"],
        row_width_slider=round(100 * math.sqrt((ROW["width"] - 0.005) / 0.295)),
    )
    print(json.dumps(info))
    save = lambda arr, name: Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8)).save(os.path.join(HERE, "assets", name), quality=88)
    save(img, "portrait-before.jpg")
    save(to_srgb(out), "portrait-after.jpg")
    accent = np.array([1.0, 0.435, 0.824])  # the mask accent #FF6FD2
    shown = img + 0.55 * W[..., None] * (accent - img)
    save(shown, "portrait-affected.jpg")
    with open(os.path.join(HERE, "assets", "target.json"), "w") as f:
        json.dump(info, f)


if __name__ == "__main__":
    main()

"""Pillow's Lanczos resize of float32 images, as an independent reference for ym-export's resize.

Usage: resize_ref.py <in.f32> <width> <height> <out_width> <out_height> <out.f32>
The files hold interleaved RGB float32 samples (little endian); each channel is resized on its own as a
mode "F" image with Image.LANCZOS (separable Lanczos-3, support scaled by the reduction factor).
"""
import sys

import numpy as np
from PIL import Image

src, w, h, ow, oh, dst = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
data = np.fromfile(src, dtype="<f4").reshape(h, w, 3)
channels = []
for c in range(3):
    im = Image.fromarray(np.ascontiguousarray(data[:, :, c]), mode="F")
    channels.append(np.asarray(im.resize((ow, oh), Image.LANCZOS)))
np.stack(channels, axis=2).astype("<f4").tofile(dst)

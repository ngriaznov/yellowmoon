#!/usr/bin/env python3
"""Draws the stand-in pictures for the Depth mockups.

NOT the model's output. A made-up depth map for the owner's photo, built from the picture's
layout alone: the rows of the photo (sky and hills far, the field in the middle, the grass near),
the mountain cut out from the Object mockup's overlay, a little of the photo's own texture
and a soft blur, as a real depth map is soft. White is near, as in the app's Show depth.

It also computes the numbers the pages print (Pick depth's band, the Focus plane band) from the
stand-in with the rules of docs/23 section 4.8, so the pictures and the cards agree.
"""
import json, os
import numpy as np
from PIL import Image, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, 'html', 'assets')
SRC = os.path.join(HERE, '..', 'nodes', 'html', 'assets')
W, H = 1332, 888
os.makedirs(ASSETS, exist_ok=True)

photo = Image.open(os.path.join(SRC, 'photo-graded.jpg')).convert('RGB').resize((W, H), Image.LANCZOS)
photo.save(os.path.join(ASSETS, 'photo.jpg'), quality=84)
lum = np.asarray(photo.convert('L'), dtype=np.float64) / 255.0

# nearness 0..1 by photo row (y as a share of the height): sky, hills, trees, fields, grass
ys = np.array([0.00, 0.40, 0.50, 0.505, 0.55, 0.60, 0.63, 0.64, 0.74, 0.86, 1.00])
nn = np.array([0.00, 0.02, 0.06, 0.20, 0.24, 0.32, 0.36, 0.50, 0.66, 0.84, 1.00])
y = (np.arange(H) + 0.5) / H
col = np.interp(y, ys, nn)
near = np.tile(col[:, None], (1, W))

# the mountain is one far plane, a little nearer than the sky
mtn = np.asarray(Image.open(os.path.join(SRC, 'mtn-overlay.png')).convert('RGBA'))[..., 3].astype(np.float64)
mtn = np.clip(mtn / 64.0, 0, 1)
mtn = np.asarray(Image.fromarray((mtn * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2)), dtype=np.float64) / 255.0
near = near * (1 - mtn) + 0.06 * mtn

# trees stand in front of the hills: dark pixels in the tree band come forward
band = np.exp(-((y - 0.575) / 0.05) ** 2)[:, None]
dark = np.clip((0.30 - lum) / 0.30, 0, 1)
near = near + 0.14 * band * dark

# the grass has texture: nearer where it is brighter or darker than its neighbours
tex = np.asarray(Image.fromarray((lum * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(6)), dtype=np.float64) / 255.0
gr = np.clip((y - 0.78) / 0.22, 0, 1)[:, None]
near = near + 0.07 * gr * (lum - tex) * 4

dm = Image.fromarray((np.clip(near, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2.2))
near = np.asarray(dm, dtype=np.float64) / 255.0

# rank normalisation (docs/23 4.8): r = 1 for the nearest. d = 1 - r is the card's scale, 0 near, 1 far.
flat = near.ravel()
order = np.argsort(flat, kind='stable')
rank = np.empty_like(flat)
rank[order] = np.arange(flat.size) / (flat.size - 1)
r = rank.reshape(near.shape)
d = 1.0 - r


def grey(a):
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8), 'L').convert('RGB')


# the Show depth picture: grey, white for near (the whole map, not cut), shown over the photo at full strength
grey(r).save(os.path.join(ASSETS, 'depth-grey.jpg'), quality=84)


def smooth(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def band(near_, far_, soft):
    q = max(0.25 * soft, 0.02)
    lo = smooth((d - (near_ - q)) / (2 * q)) if near_ > 0 else (d >= 0).astype(float)
    hi = smooth(((far_ + q) - d) / (2 * q)) if far_ < 1 else (d <= 1).astype(float)
    return lo * hi


def overlay(name, m, rgb=(255, 111, 210), amax=118):
    a = np.zeros((H, W, 4), np.uint8)
    a[..., 0], a[..., 1], a[..., 2] = rgb
    a[..., 3] = (np.clip(m, 0, 1) * amax).astype(np.uint8)
    Image.fromarray(a, 'RGBA').save(os.path.join(ASSETS, name), optimize=True)


def pick(px, py):
    """Pick depth: the band around the clicked depth, a fifth of the photo wide, two decimals."""
    c = float(d[int(py * H), int(px * W)])
    lo = min(max(c - 0.1, 0.0), 0.8)
    return round(lo, 2), round(lo + 0.2, 2)


nums = {}
# default band: Near 0, Far 50, Softness 50
overlay('mask-default.png', band(0.0, 0.5, 0.5))
nums['default'] = [0, 50, 50]

# a Pick depth click on the ploughed field
fx, fy = 0.46, 0.66
n_, f_ = pick(fx, fy)
overlay('mask-picked.png', band(n_, f_, 0.5))
nums['picked'] = [round(n_ * 100), round(f_ * 100), 50, fx, fy]

# Focus plane: the camera focused on the bush in the middle of the field (the AF box)
bx, by = 0.605, 0.745
n2, f2 = pick(bx, by)
nums['focus'] = [round(n2 * 100), round(f2 * 100), 50, bx, by]

# Focus plane with the point at the exact centre of the frame (what Sony writes when it does not know)
cx, cy = 0.5, 0.5
n3, f3 = pick(cx, cy)
overlay('mask-centre.png', band(n3, f3, 0.5))
nums['centre'] = [round(n3 * 100), round(f3 * 100), 50, cx, cy]

# the far plane (sky, hills, mountain): Near 55, Far 100, and the Subject matte (the mountain, with the edge a depth map lacks)
nums["far"] = [55, 100, 50]
# the far plane without the mountain: far map minus the clean matte (hard edge, as the Subject matte gives)
mt = np.asarray(Image.open(os.path.join(SRC, 'mtn-overlay.png')).convert('RGBA'))[..., 3].astype(np.float64)
mt = np.clip(mt / 64.0, 0, 1)
far = band(0.55, 1.0, 0.5)
overlay('mask-far-minus-subject.png', far * (1 - mt))  # with Subject subtracted: the edge is the matte's
# a faint cyan for the subtracted subject on the combined page is drawn in CSS from the same file

# the Show depth page uses the grey map as it is
json.dump(nums, open(os.path.join(ASSETS, 'numbers.json'), 'w'))
print(nums)

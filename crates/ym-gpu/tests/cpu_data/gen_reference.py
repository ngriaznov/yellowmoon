#!/usr/bin/env python3
"""Generates reference.bin / reference.json: inputs and outputs of independent NumPy
implementations of the CPU reference stages that have no published reference code to compare with,
so that crates/ym-gpu/tests/cpu_reference.rs can check the Rust stages against something other than
themselves:

  * R5 highlight reconstruction (docs/05 steps 1-4, all four Bayer phases),
  * R6 bilateral denoise in the (Y, R-Y, B-Y) space,
  * R7 colour guided filter (He et al.) on R-Y and B-Y, guided by Y,
  * R8 Richardson-Lucy with the space-variant Gaussian PSF and the damping of docs/05.

The implementations below are written from the spec text in plain loops (slow, obvious, no shared
code with the Rust). Usage: python3 -I gen_reference.py   (deterministic; needs numpy)
The .bin file holds little-endian float32 arrays back to back; the .json file lists them by name
with their offset (in floats) and shape.
"""
import json
from pathlib import Path

import numpy as np

LUMA = np.array([0.2126, 0.7152, 0.0722])
PATTERNS = {"RGGB": [[0, 1], [1, 2]], "GRBG": [[1, 0], [2, 1]], "GBRG": [[1, 2], [0, 1]], "BGGR": [[2, 1], [1, 0]]}


def mirror(i, n):
    """Index mirrored about the edge samples without repeating them (-1 -> 1, n -> n - 2)."""
    if n <= 1:
        return 0
    p = 2 * (n - 1)
    j = i % p
    return p - j if j >= n else j


# ---------------------------------------------------------------------------------------------
# R5
# ---------------------------------------------------------------------------------------------

def r5(mos, mask, pattern):
    """docs/05 R5: superpixels, cube-root opposed estimates, per-channel global chroma offsets,
    write-back as max(original, estimate + offset). Returns the new mosaic, the offsets and the
    number of reference superpixels per channel."""
    H, W = mos.shape
    sw, sh = W // 2, H // 2
    col = PATTERNS[pattern]
    val = np.zeros((3, sh, sw))
    cnt = np.zeros((3, sh, sw))
    clip = np.zeros((3, sh, sw), bool)
    for j in range(sh):
        for i in range(sw):
            for dy in (0, 1):
                for dx in (0, 1):
                    x, y = 2 * i + dx, 2 * j + dy
                    c = col[y & 1][x & 1]
                    val[c, j, i] += mos[y, x]
                    cnt[c, j, i] += 1
                    clip[c, j, i] |= bool(mask[y, x])
    val /= cnt

    def opposed(i, j, c):
        s, n = 0.0, 0
        for jj in range(max(j - 1, 0), min(j + 2, sh)):
            for ii in range(max(i - 1, 0), min(i + 2, sw)):
                for k in range(3):
                    if k != c and not clip[k, jj, ii]:
                        s += np.cbrt(val[k, jj, ii])
                        n += 1
        return None if n == 0 else (s / n) ** 3

    own = clip.any(0)
    acc, count = np.zeros(3), np.zeros(3, int)
    for j in range(sh):
        for i in range(sw):
            if own[j, i]:
                continue
            for c in range(3):
                if clip[c, max(j - 1, 0):j + 2, max(i - 1, 0):i + 2].any():
                    y = opposed(i, j, c)
                    if y is not None:
                        acc[c] += val[c, j, i] - y
                        count[c] += 1
    offset = np.where(count > 0, acc / np.maximum(count, 1), 0.0)
    out = mos.astype(np.float64).copy()
    for j in range(sh):
        for i in range(sw):
            if not own[j, i]:
                continue
            est = [None] * 3
            for c in range(3):
                if clip[c, j, i]:
                    y = opposed(i, j, c)
                    est[c] = None if y is None else y + offset[c]
            for dy in (0, 1):
                for dx in (0, 1):
                    x, y = 2 * i + dx, 2 * j + dy
                    if mask[y, x]:
                        c = col[y & 1][x & 1]
                        if est[c] is not None:
                            out[y, x] = max(out[y, x], est[c])
    return out, offset, count


# ---------------------------------------------------------------------------------------------
# R6 bilateral, R7 guided, R8 Richardson-Lucy (planar CHW float images)
# ---------------------------------------------------------------------------------------------

def opp(img):
    y = (LUMA[:, None, None] * img).sum(0)
    return y, img[0] - y, img[2] - y


def unopp(y, cr, cb):
    r, b = cr + y, cb + y
    return np.stack([r, (y - LUMA[0] * r - LUMA[2] * b) / LUMA[1], b], 0)


def box_mean(a, r):
    H, W = a.shape
    out = np.zeros((H, W))
    for y in range(H):
        for x in range(W):
            out[y, x] = a[max(0, y - r):min(H, y + r + 1), max(0, x - r):min(W, x + r + 1)].mean()
    return out


def guided(img, sigma, color):
    """docs/05 R7 Colour: radius round(2 + 6 C / 100), eps = sigma^2 (C / 50)^2."""
    r = int(np.floor(2 + 6 * color / 100 + 0.5))
    eps = np.maximum(sigma ** 2 * (color / 50) ** 2, 1e-10)
    y, cr, cb = opp(img.astype(np.float64))
    mean_i, corr_i = box_mean(y, r), box_mean(y * y, r)
    var = np.maximum(corr_i - mean_i * mean_i, 0)
    filtered = []
    for p in (cr, cb):
        mean_p, corr_ip = box_mean(p, r), box_mean(y * p, r)
        a = (corr_ip - mean_i * mean_p) / (var + eps)
        b = mean_p - a * mean_i
        filtered.append(box_mean(a, r) * y + box_mean(b, r))
    return unopp(y, filtered[0], filtered[1])


def bilateral(img, sigma, strength=1.0, radius=2, spatial=1.5):
    """docs/05 R6 Denoise: 5x5 window, spatial sigma 1.5, h = strength * sqrt(6) * sigma (centre pixel)."""
    o = np.stack(opp(img.astype(np.float64)), 0)
    _, H, W = o.shape
    out = o.copy()
    for y in range(H):
        for x in range(W):
            h = strength * np.sqrt(6.0) * sigma[y, x]
            if h <= 1e-12:
                continue
            num, den = np.zeros(3), 0.0
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    q = o[:, mirror(y + dy, H), mirror(x + dx, W)]
                    w = np.exp(-(dx * dx + dy * dy) / (2 * spatial ** 2)) * np.exp(-((q - o[:, y, x]) ** 2).sum() / (2 * h * h))
                    num += w * q
                    den += w
            out[:, y, x] = num / den
    return unopp(out[0], out[1], out[2])


def richardson_lucy(img, amount, rho, sigma_noise):
    """docs/05 R8: n = round(A / 30) iterations on the luminance, Gaussian PSF with sigma = rho (1 + r^2 / 4)
    (r over half the diagonal), kernels truncated at ceil(3 * 1.25 rho) and renormalised per pixel, blur =
    vertical(horizontal(.)) with the kernel of the pixel written, damped ratio 1 + w (d - e) / e."""
    n_iter = int(np.clip(np.floor(amount / 30 + 0.5), 0, 10))
    H, W = img.shape[1:]
    y = (LUMA[:, None, None] * img.astype(np.float64)).sum(0)
    d = np.maximum(y, 0)
    radius = int(np.ceil(3 * 1.25 * rho))
    yy, xx = np.mgrid[0:H, 0:W]
    r2 = ((xx + 0.5 - W / 2) ** 2 + (yy + 0.5 - H / 2) ** 2) / ((W * W + H * H) / 4.0)
    sig = rho * (1 + 0.25 * r2)
    ks = np.arange(-radius, radius + 1)
    kern = np.exp(-0.5 * ks[None, None, :] ** 2 / sig[..., None] ** 2)
    kern /= kern.sum(-1, keepdims=True)

    def blur(src):
        tmp = np.zeros_like(src)
        for j in range(H):
            for i in range(W):
                tmp[j, i] = (kern[j, i] * src[j, [mirror(i + k, W) for k in ks]]).sum()
        out = np.zeros_like(src)
        for j in range(H):
            idx = [mirror(j + k, H) for k in ks]
            for i in range(W):
                out[j, i] = (kern[j, i] * tmp[idx, i]).sum()
        return out

    u = d.copy()
    for _ in range(n_iter):
        e = blur(u)
        resid = d - e
        w = resid ** 2 / (resid ** 2 + sigma_noise ** 2) if sigma_noise > 0 else np.ones_like(resid)
        c = blur(1 + w * resid / np.maximum(e, 1e-4))
        u = np.maximum(u * c, 0)
    return img.astype(np.float64) + (u - d)[None]


# ---------------------------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(20261012)
    arrays, meta = [], {}

    def add(name, a, **extra):
        a = np.asarray(a, dtype=np.float32)
        meta[name] = dict(shape=list(a.shape), **extra)
        arrays.append((name, a))

    # R5: three blown-out sources of different colours; per-channel clip levels as after the gains.
    H, W = 28, 32
    yy, xx = np.mgrid[0:H, 0:W]
    scene = np.zeros((H, W, 3))
    for cx, cy, rad, col, peak in [(9, 9, 6.0, [1.0, 0.65, 0.35], 2.2), (23, 17, 5.5, [0.4, 0.7, 1.0], 2.0), (8, 22, 3.5, [1.0, 0.2, 0.8], 1.8)]:
        f = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * rad ** 2))
        for c in range(3):
            scene[..., c] += peak * col[c] * f
    scene += 0.05 + 0.02 * rng.standard_normal(scene.shape)
    clip_level = np.array([1.9, 1.0, 1.5])
    for pattern in PATTERNS:
        col = PATTERNS[pattern]
        mos = np.zeros((H, W))
        mask = np.zeros((H, W))
        for y in range(H):
            for x in range(W):
                c = col[y & 1][x & 1]
                v = scene[y, x, c]
                if v >= clip_level[c]:
                    mos[y, x], mask[y, x] = clip_level[c], 1
                else:
                    mos[y, x] = v
        out, offset, count = r5(mos, mask, pattern)
        add(f"r5_{pattern}_mosaic", mos, width=W, height=H, pattern=pattern)
        add(f"r5_{pattern}_mask", mask)
        add(f"r5_{pattern}_out", out)
        add(f"r5_{pattern}_offsets", np.concatenate([offset, count]))

    # R6 / R7 / R8 on one noisy planar image with structure.
    H, W = 27, 33
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.stack([0.4 + 0.2 * np.sin(xx / 5.0) + 0.3 * (xx > 16),
                    0.5 + 0.2 * np.cos(yy / 4.0) + 0.2 * ((xx + yy) % 11 < 3),
                    0.3 + 0.2 * np.sin((xx + yy) / 7.0)], 0)
    img = np.clip(img + 0.03 * rng.standard_normal(img.shape), 0, None)
    sigma = 0.01 + 0.005 * rng.random((H, W))
    add("rgb", img, width=W, height=H)
    add("sigma", sigma)
    for c in (30, 100):
        add(f"guided_{c}", guided(img, sigma, c), color=c)
    add("bilateral", bilateral(img, sigma))
    for amount, rho, noise in ((90, 0.8, 0.01), (150, 1.3, 0.0)):
        add(f"rl_{amount}", richardson_lucy(img, amount, rho, noise), amount=amount, radius=rho, noise=noise)

    here = Path(__file__).resolve().parent
    offset = 0
    for name, a in arrays:
        meta[name]["offset"] = offset
        offset += a.size
    (here / "reference.bin").write_bytes(b"".join(a.astype("<f4").tobytes() for _, a in arrays))
    (here / "reference.json").write_text(json.dumps({"source": "gen_reference.py: independent NumPy implementations of docs/05 R5-R8", "arrays": meta}, indent=1))
    print("wrote reference.bin", offset * 4, "bytes,", len(arrays), "arrays")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""A second, independent estimator of the lateral CA (red and blue against green), by cross-correlation.

`ca_measure.py` fits the shift by Gauss-Newton on high-passed planes.  This one shares no code with it:
every block of a band-passed pair of planes is cross-correlated by FFT, the peak gets a parabolic
sub-pixel fit, and the 2-D displacement is projected on the unit vector away from the optical centre.
Positive is outward; the tangential component is reported as a noise floor (a lateral CA has none).
It exists so that a finding of `real_eval.py` can be checked by a method that cannot share its
mistakes (RA-03c review).  Local evaluation only (real files, never committed).

    python3 -I tools/lens/ca_xcorr.py selftest
    python3 -I tools/lens/ca_xcorr.py rgb DUMP.f32 META.json [--full]     # a `lens_eval_dump` image
    python3 -I tools/lens/ca_xcorr.py mosaic FILE.ARW                     # the mosaic itself, own-colour bilinear planes

`DUMP.f32` is `u32 w, u32 h` and the R, G, B planes as little-endian `f32`; `META.json` is the
`{"frame": [w, h], "crop": [x, y, w, h]}` file next to it; `--full` for the uncropped demosaic.
Output: per radial band (r = distance over half the frame diagonal) the median radial shift in px, and
the rms over r >= 0.25.  Blocks whose correlation peak is below 0.45, or at the edge of the +-4 px
search window, are dropped.  Checked on synthetic shifts: within about 5% of the true value up to
2.5 px.
"""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy import ndimage as ndi

EDGES = (0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.2)
SEARCH_PX = 4
MIN_PEAK = 0.45


def load_f32(path: str) -> np.ndarray:
    """`lens_eval_dump` output as a (3, h, w) array."""
    w, h = (int(v) for v in np.fromfile(path, dtype="<u4", count=2))
    return np.fromfile(path, dtype="<f4", offset=8, count=w * h * 3).reshape(3, h, w)


def bandpass(plane: np.ndarray) -> np.ndarray:
    """Keeps the edges: a difference of Gaussians (1 px and 6 px)."""
    return ndi.gaussian_filter(plane, 1.0) - ndi.gaussian_filter(plane, 6.0)


def _subpixel(m1: float, m0: float, p1: float) -> float:
    den = m1 - 2.0 * m0 + p1
    return 0.0 if den == 0.0 else 0.5 * (m1 - p1) / den


def measure(img: np.ndarray, centre: tuple[float, float], half_diagonal: float, block: int = 64,
            r_min: float = 0.15, r_max: float = 1.2) -> dict[int, np.ndarray]:
    """Per block (radius, radial shift, tangential shift) of red (key 0) and blue (key 2) against green."""
    _, h, w = img.shape
    green = bandpass(img[1])
    window = np.outer(np.hanning(block), np.hanning(block)).astype(np.float32)
    pad = 2 * block
    out: dict[int, list[tuple[float, float, float]]] = {0: [], 2: []}
    for c in (0, 2):
        plane = bandpass(img[c])
        for y0 in range(0, h - block + 1, block):
            for x0 in range(0, w - block + 1, block):
                cx, cy = x0 + block / 2 - centre[0], y0 + block / 2 - centre[1]
                rad = float(np.hypot(cx, cy))
                if not r_min <= rad / half_diagonal <= r_max:
                    continue
                a = plane[y0:y0 + block, x0:x0 + block] * window
                b = green[y0:y0 + block, x0:x0 + block] * window
                ea, eb = float((a * a).sum()), float((b * b).sum())
                if ea < 1e-9 or eb < 1e-9:
                    continue
                cc = np.fft.irfft2(np.fft.rfft2(a, (pad, pad)) * np.conj(np.fft.rfft2(b, (pad, pad))), (pad, pad))
                cc = np.roll(cc, (SEARCH_PX, SEARCH_PX), axis=(0, 1))[:2 * SEARCH_PX + 1, :2 * SEARCH_PX + 1]
                iy, ix = np.unravel_index(np.argmax(cc), cc.shape)
                if iy in (0, 2 * SEARCH_PX) or ix in (0, 2 * SEARCH_PX):
                    continue
                if cc[iy, ix] / np.sqrt(ea * eb) < MIN_PEAK:
                    continue
                dx = ix - SEARCH_PX + _subpixel(cc[iy, ix - 1], cc[iy, ix], cc[iy, ix + 1])
                dy = iy - SEARCH_PX + _subpixel(cc[iy - 1, ix], cc[iy, ix], cc[iy + 1, ix])
                ux, uy = cx / rad, cy / rad
                out[c].append((rad / half_diagonal, dx * ux + dy * uy, -dx * uy + dy * ux))
    return {c: np.array(v) for c, v in out.items()}


def summarise(res: dict[int, np.ndarray]) -> dict[str, dict]:
    """Median radial shift per band, rms over r >= 0.25, tangential rms (the noise floor)."""
    rows = {}
    for c, name in ((0, "red"), (2, "blue")):
        a = res[c]
        bands = []
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            m = (a[:, 0] >= lo) & (a[:, 0] < hi)
            bands.append((lo, int(m.sum()), float(np.median(a[m, 1])) if m.any() else float("nan")))
        sel = a[:, 0] >= 0.25
        rows[name] = dict(bands=bands, rms=float(np.sqrt(np.mean(a[sel, 1] ** 2))),
                          tangential_rms=float(np.sqrt(np.mean(a[sel, 2] ** 2))), n=int(sel.sum()))
    return rows


def show(rows: dict[str, dict]) -> None:
    for name, r in rows.items():
        bands = " ".join(f"{lo:.2f}:{m:+.2f}" for lo, n, m in r["bands"])
        print(f"  {name}: rms(r>=0.25) {r['rms']:.3f}  tangential {r['tangential_rms']:.3f}  n {r['n']}   {bands}")


def own_colour_planes(path: str) -> np.ndarray:
    """The mosaic of an ARW as three full planes, each colour interpolated from its own photosites only
    (bilinear), so that nothing pulls one colour towards another."""
    import rawpy

    raw = rawpy.imread(path)
    data = raw.raw_image_visible.astype(np.float32)
    colours = raw.raw_colors_visible
    planes = []
    for want in ((0,), (1, 3), (2,)):
        mask = np.isin(colours, want).astype(np.float32)
        kernel = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]] if want == (1, 3) else [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
                          dtype=np.float32) / 4.0
        num = ndi.convolve(data * mask, kernel, mode="mirror")
        den = ndi.convolve(mask, kernel, mode="mirror")
        planes.append(num / np.maximum(den, 1e-6))
    return np.stack(planes)


def selftest() -> int:
    """A known radial shift (red outward by `amp * (r / r_max)^1.5`, blue inward by half of it) on a
    filtered noise texture comes back within 8% (relative) or 0.03 px."""
    rng = np.random.default_rng(7)
    h, w = 1200, 1800
    tex = ndi.gaussian_filter(rng.standard_normal((h, w)).astype(np.float32), 1.5)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2, h / 2
    dx, dy = xx - cx, yy - cy
    rad = np.hypot(dx, dy)
    hd = 0.5 * np.hypot(w, h)
    ux, uy = dx / np.maximum(rad, 1e-6), dy / np.maximum(rad, 1e-6)
    worst = 0.0
    for amp in (0.0, 1.0, 2.5):
        s = amp * (rad / hd) ** 1.5
        red = ndi.map_coordinates(tex, [yy - s * uy, xx - s * ux], order=3, mode="nearest")
        blue = ndi.map_coordinates(tex, [yy + 0.5 * s * uy, xx + 0.5 * s * ux], order=3, mode="nearest")
        rows = summarise(measure(np.stack([red, tex, blue]), (cx, cy), hd))
        for name, sign in (("red", 1.0), ("blue", -0.5)):
            for lo, n, got in rows[name]["bands"]:
                if n < 5 or lo + 0.15 > 1.0:
                    continue
                mid = lo + 0.075
                want = sign * amp * mid**1.5
                err = abs(got - want)
                worst = max(worst, min(err / max(abs(want), 1e-9), err / 0.03 if err > 0.03 else 0.0))
                if err > max(0.08 * abs(want), 0.03):
                    print(f"selftest FAILED: {name} amp {amp} band {lo:.2f}: got {got:+.3f}, want {want:+.3f}")
                    return 1
    print("selftest ok")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[1] == "selftest":
        return selftest()
    if len(argv) >= 4 and argv[1] == "rgb":
        meta = json.load(open(argv[3]))
        fw, fh = meta["frame"]
        x0, y0 = meta["crop"][:2]
        centre = (fw / 2, fh / 2) if "--full" in argv else (fw / 2 - x0, fh / 2 - y0)
        show(summarise(measure(load_f32(argv[2]), centre, 0.5 * float(np.hypot(fw, fh)))))
        return 0
    if len(argv) >= 3 and argv[1] == "mosaic":
        img = own_colour_planes(argv[2])
        _, h, w = img.shape
        show(summarise(measure(img, (w / 2, h / 2), 0.5 * float(np.hypot(w, h)))))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))

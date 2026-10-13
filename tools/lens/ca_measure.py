#!/usr/bin/env python3
"""Signed radial shift of red and blue against green (lateral chromatic aberration), measured on a
Bayer mosaic or on full RGB planes.  An evaluation tool for `docs/05-raw-processing.md` section R9
("Real-file check"); it is not part of the product and shares no code with it.

Model.  Around a point `p` at radius `rho` from the optical centre, a colour plane `X` (red or blue)
is the green image displaced along the radius by `s` pixels (positive: outward) and scaled:

    X(p) = a * G(p - s * u) + b,        u = unit vector from the centre to p

`a`, `b` and `s` are fitted by Gauss-Newton over a block of sites; the fit is repeated for every block
of the frame and the signed shifts are averaged per radial band, weighted by their information.  On
a mosaic the fit runs on the sites of the colour itself (R or B) and `G` is the 4-neighbour
bilinear green at the R/B sites (true samples at the green sites), sampled at `p - s u` by a cubic
spline: no colour-difference demosaic is involved, so the answer is not pulled towards green by the
demosaic's own edge logic (which a demosaiced measurement is).  `selftest` checks the estimator on
synthetic mosaics with a known radial displacement.

A shift is defined as (feature radius in colour X) - (feature radius in green) along the radius.  The
table reading of `docs/05` section R9 predicts `rho_g * hd * (K_X(r) - 1)` for it.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi


def bilinear_green(mosaic: np.ndarray, is_green: np.ndarray) -> np.ndarray:
    """Green at every site: the sample at green sites, the mean of the four neighbours elsewhere."""
    m = mosaic.astype(np.float64)
    p = np.pad(m, 1, mode="reflect")
    nb = 0.25 * (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:])
    return np.where(is_green, m, nb)


def _bilinear_sparse(mosaic: np.ndarray, is_site: np.ndarray) -> np.ndarray:
    """A plane known only at the `is_site` sites (one site per 2 x 2 cell, as red and blue are in a
    Bayer mosaic), filled everywhere by bilinear interpolation between them."""
    m = np.where(is_site, mosaic, 0.0).astype(np.float64)
    wgt = is_site.astype(np.float64)
    k = np.array([[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]])
    num = ndi.convolve(m, k, mode="reflect")
    den = ndi.convolve(wgt, k, mode="reflect")
    return np.where(is_site, mosaic, num / np.maximum(den, 1e-12))


def _sample(win: np.ndarray, qy: np.ndarray, qx: np.ndarray) -> np.ndarray:
    return ndi.map_coordinates(win, [qy, qx], order=3, mode="nearest", prefilter=True)


@dataclass
class BlockFit:
    """The fit of one block."""

    shift_px: float  # signed radial shift of the colour against green (positive: outward)
    se_px: float  # standard error of the shift
    gain: float  # a
    correlation: float  # between the colour and the shifted green
    radius_px: float  # distance of the block centre from the optical centre
    n_sites: int
    x: float = 0.0  # block centre in the frame (continuous coordinates)
    y: float = 0.0


def fit_block(
    values: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    green_win: np.ndarray,
    win_origin: tuple[int, int],
    centre: tuple[float, float],
    iterations: int = 6,
    with_offset: bool = True,
) -> BlockFit | None:
    """Fits `X(p) = a G(p - s u) + b` over the sites `(px, py)` (continuous coordinates, pixel `i`
    covering `[i, i + 1)`) with values `values`.  `green_win` is a window of the full-resolution green
    whose pixel `(0, 0)` is the frame pixel `win_origin = (x0, y0)`."""
    cx, cy = centre
    dx, dy = px - cx, py - cy
    rr = np.hypot(dx, dy)
    ok = rr > 1e-6
    if ok.sum() < 64:
        return None
    ux, uy = dx / np.maximum(rr, 1e-9), dy / np.maximum(rr, 1e-9)
    x0, y0 = win_origin
    s, a, b = 0.0, 1.0, 0.0
    se = float("inf")
    corr = 0.0
    for it in range(iterations):
        qx = px - s * ux - 0.5 - x0
        qy = py - s * uy - 0.5 - y0
        g = _sample(green_win, qy, qx)
        dg = _sample(green_win, qy + 0.5 * uy, qx + 0.5 * ux) - _sample(green_win, qy - 0.5 * uy, qx - 0.5 * ux)
        if it == 0:
            if with_offset:
                gm = g - g.mean()
                vm = values - values.mean()
            else:
                gm, vm = g, values
            den = float((gm * gm).sum())
            if den < 1e-18:
                return None
            a = float((gm * vm).sum() / den)
            b = float(values.mean() - a * g.mean()) if with_offset else 0.0
        # d g(p - s u) / d s = -dg
        cols = [g, -a * dg] if not with_offset else [g, np.ones_like(g), -a * dg]
        jac = np.stack(cols, axis=1)
        e = values - (a * g + b)
        try:
            delta, *_ = np.linalg.lstsq(jac, e, rcond=None)
        except np.linalg.LinAlgError:
            return None
        a += float(delta[0])
        if with_offset:
            b += float(delta[1])
        s += float(delta[-1])
        s = float(np.clip(s, -6.0, 6.0))
        if abs(delta[-1]) < 1e-4:
            break
    qx = px - s * ux - 0.5 - x0
    qy = py - s * uy - 0.5 - y0
    g = _sample(green_win, qy, qx)
    dg = _sample(green_win, qy + 0.5 * uy, qx + 0.5 * ux) - _sample(green_win, qy - 0.5 * uy, qx - 0.5 * ux)
    cols = [g, -a * dg] if not with_offset else [g, np.ones_like(g), -a * dg]
    jac = np.stack(cols, axis=1)
    e = values - (a * g + b)
    dof = max(len(values) - jac.shape[1], 1)
    sigma2 = float((e * e).sum() / dof)
    jtj = jac.T @ jac
    try:
        cov = np.linalg.inv(jtj) * sigma2
    except np.linalg.LinAlgError:
        return None
    se = float(np.sqrt(max(cov[-1, -1], 0.0)))
    vm, gm = values - values.mean(), g - g.mean()
    d = float(np.sqrt((vm * vm).sum() * (gm * gm).sum()))
    corr = float((vm * gm).sum() / d) if d > 0 else 0.0
    return BlockFit(
        s, se, a, corr, float(np.hypot(px.mean() - cx, py.mean() - cy)), len(values),
        float(px.mean()), float(py.mean()),
    )


def measure_mosaic(
    mosaic: np.ndarray,
    colors: np.ndarray,
    clip_level: float,
    centre: tuple[float, float],
    frame_origin: tuple[int, int] = (0, 0),
    block_px: int = 96,
    min_corr: float = 0.5,
    max_se_px: float = 0.25,
    margin_px: int = 24,
    highpass_sigma_px: float = 5.0,
) -> dict[str, list[BlockFit]]:
    """Measures the shift of red and of blue against green over every block of a mosaic.

    `mosaic` is the *frame* (the lens frame the radii refer to; black already subtracted), `colors`
    the colour index of every site (0 red, 1 green, 2 blue, 3 second green), `centre` the optical
    centre in frame coordinates.  `frame_origin` is only used to report positions.  Blocks with a
    clipped sample are left out.
    """
    h, w = mosaic.shape
    is_green = (colors == 1) | (colors == 3)
    green = bilinear_green(mosaic, is_green)
    planes = {"green": green}
    for name, code in (("red", 0), ("blue", 2)):
        planes[name] = _bilinear_sparse(mosaic, colors == code)
    if highpass_sigma_px > 0:
        planes = {k: v - ndi.gaussian_filter(v, highpass_sigma_px) for k, v in planes.items()}
    clipped = mosaic >= clip_level
    out: dict[str, list[BlockFit]] = {"red": [], "blue": []}
    pad = 12
    for by in range(margin_px, h - margin_px - block_px + 1, block_px):
        for bx in range(margin_px, w - margin_px - block_px + 1, block_px):
            sl = (slice(by, by + block_px), slice(bx, bx + block_px))
            if clipped[by - 4 : by + block_px + 4, bx - 4 : bx + block_px + 4].any():
                continue
            gwin = planes["green"][by - pad : by + block_px + pad, bx - pad : bx + block_px + pad]
            if np.ptp(gwin) < 1e-9:
                continue
            for name, code in (("red", 0), ("blue", 2)):
                sel = colors[sl] == code
                ys, xs = np.nonzero(sel)
                vals = planes[name][sl][sel]
                px = xs + bx + 0.5
                py = ys + by + 0.5
                fit = fit_block(vals, px, py, gwin, (bx - pad, by - pad), centre, with_offset=highpass_sigma_px <= 0)
                if fit is None or not np.isfinite(fit.shift_px):
                    continue
                if fit.se_px > max_se_px or abs(fit.correlation) < min_corr:
                    continue
                out[name].append(fit)
    return out


def measure_rgb(
    rgb: np.ndarray,
    centre: tuple[float, float],
    clip_level: float = 0.995,
    block_px: int = 64,
    min_corr: float = 0.5,
    max_se_px: float = 0.25,
    margin_px: int = 16,
    highpass_sigma_px: float = 5.0,
    min_energy: float = 0.0,
) -> dict[str, list[BlockFit]]:
    """The same measurement on full `h x w x 3` planes (a render, a decoded JPEG).  The high-pass
    sigma is in the pixels of `rgb`."""
    h, w, _ = rgb.shape
    out: dict[str, list[BlockFit]] = {"red": [], "blue": []}
    pad = 12
    rgb_raw = rgb
    if highpass_sigma_px > 0:
        rgb = np.stack([rgb[..., c] - ndi.gaussian_filter(rgb[..., c], highpass_sigma_px) for c in range(3)], axis=-1)
    ys0, xs0 = np.mgrid[0:block_px, 0:block_px]
    for by in range(margin_px, h - margin_px - block_px + 1, block_px):
        for bx in range(margin_px, w - margin_px - block_px + 1, block_px):
            blk = rgb_raw[by - 4 : by + block_px + 4, bx - 4 : bx + block_px + 4]
            if (blk >= clip_level).any():
                continue
            gwin = rgb[by - pad : by + block_px + pad, bx - pad : bx + block_px + pad, 1]
            if np.ptp(gwin) < 1e-9 or float(gwin.std()) < min_energy:
                continue
            px = (xs0 + bx + 0.5).ravel().astype(np.float64)
            py = (ys0 + by + 0.5).ravel().astype(np.float64)
            for name, ch in (("red", 0), ("blue", 2)):
                vals = rgb[by : by + block_px, bx : bx + block_px, ch].ravel().astype(np.float64)
                fit = fit_block(
                    vals, px, py, gwin.astype(np.float64), (bx - pad, by - pad), centre,
                    with_offset=highpass_sigma_px <= 0,
                )
                if fit is None or not np.isfinite(fit.shift_px):
                    continue
                if fit.se_px > max_se_px or abs(fit.correlation) < min_corr:
                    continue
                out[name].append(fit)
    return out


def radial_profile(
    fits: list[BlockFit], half_diagonal_px: float, edges: np.ndarray | None = None
) -> list[dict]:
    """Information-weighted mean signed shift per radial band (radii as fractions of the half
    diagonal).  Each band reports the weighted mean, its standard error (the larger of the formal one
    and the scatter-based one), the median and the block count."""
    if edges is None:
        edges = np.linspace(0.0, 1.0, 11)
    rho = np.array([f.radius_px / half_diagonal_px for f in fits])
    s = np.array([f.shift_px for f in fits])
    se = np.array([max(f.se_px, 0.02) for f in fits])
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (rho >= lo) & (rho < hi)
        if not m.any():
            rows.append(dict(lo=float(lo), hi=float(hi), n=0))
            continue
        w = 1.0 / se[m] ** 2
        mean = float((w * s[m]).sum() / w.sum())
        formal = float(1.0 / np.sqrt(w.sum()))
        n = int(m.sum())
        scatter = float(np.sqrt((w * (s[m] - mean) ** 2).sum() / w.sum() / max(n - 1, 1))) if n > 1 else formal
        rows.append(
            dict(
                lo=float(lo),
                hi=float(hi),
                n=n,
                rho=float((w * rho[m]).sum() / w.sum()),
                mean_px=mean,
                se_px=max(formal, scatter),
                median_px=float(np.median(s[m])),
            )
        )
    return rows


# --------------------------------------------------------------------------------------------
# Self-test on synthetic mosaics with a known radial displacement
# --------------------------------------------------------------------------------------------


def synthetic_scene(h: int, w: int, rng: np.random.Generator, n_edges: int = 400):
    """A callable `f(x, y)` (continuous coordinates): a sum of blurred straight edges in random
    orientations plus a few bars, normalised to 0..1.  Evaluated at the exact site positions of each
    colour, so the mosaic is aliased the way a real sensor's is."""
    cx = rng.uniform(0, w, n_edges)
    cy = rng.uniform(0, h, n_edges)
    th = rng.uniform(0, np.pi, n_edges)
    amp = rng.uniform(-1, 1, n_edges)
    wid = rng.uniform(1.2, 2.5, n_edges)
    ln = rng.uniform(40, 160, n_edges)

    def f(x, y):
        out = np.zeros_like(x, dtype=np.float64)
        for i in range(n_edges):
            c, s = np.cos(th[i]), np.sin(th[i])
            dxp = x - cx[i]
            dyp = y - cy[i]
            across = dxp * c + dyp * s
            along = -dxp * s + dyp * c
            env = 0.5 * (1 + np.tanh((ln[i] - np.abs(along)) / 6.0))
            out += amp[i] * env * np.tanh(across / wid[i])
        return 0.5 + 0.25 * np.tanh(out * 0.7)

    return f


def synthetic_mosaic(
    h: int,
    w: int,
    red_shift_fn,
    blue_shift_fn,
    seed: int = 1,
    noise: float = 0.0,
    gains: tuple[float, float, float] = (0.8, 1.0, 0.7),
    colour_drift: float = 0.0,
):
    """RGGB mosaic of a synthetic scene whose red and blue planes are the green image displaced
    outward by `red_shift_fn(rho)` and `blue_shift_fn(rho)` pixels (rho: fraction of the half diagonal)."""
    rng = np.random.default_rng(seed)
    f = synthetic_scene(h, w, rng)
    yy, xx = np.mgrid[0:h, 0:w]
    px, py = xx + 0.5, yy + 0.5
    cx, cy = w / 2.0, h / 2.0
    hd = 0.5 * np.hypot(w, h)
    dx, dy = px - cx, py - cy
    rr = np.hypot(dx, dy)
    ux, uy = dx / np.maximum(rr, 1e-9), dy / np.maximum(rr, 1e-9)
    colors = np.zeros((h, w), dtype=np.int32)
    colors[0::2, 0::2] = 0
    colors[0::2, 1::2] = 1
    colors[1::2, 0::2] = 3
    colors[1::2, 1::2] = 2
    mosaic = np.zeros((h, w), dtype=np.float64)
    for code, shift_fn, gain in ((0, red_shift_fn, gains[0]), (2, blue_shift_fn, gains[2])):
        m = colors == code
        rho = rr[m] / hd
        sh = shift_fn(rho)
        drift = 1.0 + colour_drift * np.sin(2 * np.pi * px[m] / (1.3 * w) + 3.0 * code) * np.cos(2 * np.pi * py[m] / (1.7 * h))
        mosaic[m] = gain * drift * f(px[m] - sh * ux[m], py[m] - sh * uy[m]) + 0.02
    g = (colors == 1) | (colors == 3)
    mosaic[g] = gains[1] * f(px[g], py[g]) + 0.02
    if noise > 0:
        mosaic += rng.normal(0, noise, mosaic.shape)
    return mosaic, colors, (cx, cy), hd


def selftest() -> int:
    """Checks the estimator on synthetic mosaics with a known radial shift: a linear and a bumpy
    profile for red and blue, no shift, and no shift with a 30% colour drift.  The estimator reads
    red and blue 6 to 20% low (the bilinear green it compares with is blurred at the red and blue
    photosites), so a band passes when it is within 0.04 px plus 25% of the true shift."""
    h, w = 800, 1200
    cases = [
        ("red +0.8*rho, blue -0.5*rho", lambda r: 0.8 * r, lambda r: -0.5 * r, 0.0),
        ("red 1.5*rho^2, blue +0.6*sin(3 rho)", lambda r: 1.5 * r**2, lambda r: 0.6 * np.sin(3 * r), 0.0),
        ("no shift", lambda r: 0 * r, lambda r: 0 * r, 0.0),
        ("no shift, colour drift 30%", lambda r: 0 * r, lambda r: 0 * r, 0.3),
    ]
    failures = 0
    for name, fr, fb, drift in cases:
        mosaic, colors, centre, hd = synthetic_mosaic(h, w, fr, fb, seed=3, noise=0.002, colour_drift=drift)
        res = measure_mosaic(mosaic, colors, clip_level=10.0, centre=centre, block_px=96, max_se_px=0.3)
        for chan, fn in (("red", fr), ("blue", fb)):
            prof = radial_profile(res[chan], hd, np.linspace(0.2, 0.9, 8))
            bad = []
            for r in prof:
                if r["n"] < 4:
                    continue
                want = float(fn(np.array([r["rho"]]))[0])
                if abs(r["mean_px"] - want) > 0.04 + 0.25 * abs(want):
                    bad.append((round(r["rho"], 2), round(r["mean_px"], 3), round(want, 3)))
            failures += len(bad)
            used = [(round(r["rho"], 2), round(r["mean_px"], 3), r["n"]) for r in prof if r["n"] >= 4]
            print(f"{name:40s} {chan:5s} {'FAIL ' + str(bad) if bad else 'ok'}   bands {used}")
    print("selftest:", "ok" if not failures else f"{failures} bands out of tolerance")
    return 0 if not failures else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        sys.exit(selftest())
    print(__doc__)

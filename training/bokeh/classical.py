"""A classical blur estimator, in numpy: the baseline the network must beat, and the source photos' sharpness filter.

The re-blur gradient ratio of Zhuo and Sim (Pattern Recognition 2011), written from the paper (their reference port is
GPL-3.0 and is not read): the image is blurred again with a Gaussian of `sigma_r` = 1 px, and at an edge the ratio of
the gradient magnitudes `R = |grad I| / |grad (G * I)|` gives the blur of a Gaussian-blurred step edge,
`sigma0 = sigma_r / sqrt(R^2 - 1)`. A disc of radius `r` has the same second moment as a Gaussian of `sigma = r / 2`,
so the disc radius read at the edge is `2 * sigma0`. The sparse values are spread over the picture by normalised
convolution with Gaussians of growing width (a cascade from coarse to fine) with the edge strength as weight, and
the density of nearby edges gives a confidence. The spec's detector (`docs/21` section 7) adds a joint bilateral
spread guided by the colours and a ball search; this baseline has neither, which is stated wherever its numbers are
reported. Its known blind spot (`research/bokeh-map-2026-10-24.md` section 1): a defocused light is a disc with a sharp
rim, so the edge ratio reads the rim as sharp.

`edge_stats` is what the source filter uses (`sources.py`): the share of strong edges that are soft says whether a
photo (or a tile of it) has natural bokeh of its own, the main label noise of the whole recipe.
"""

import numpy as np
from scipy import ndimage

SIGMA_R = 1.0
#: Edge pixels: gradient magnitude above this and above `NOISE_K` times the picture's 75th percentile of it.
MIN_GRADIENT = 0.02
NOISE_K = 3.0
R_CLIP_PX = 60.0
#: A strong edge reads as soft when its disc radius is above this many pixels (`b_min` is 2.5 px at 1024 px long).
SOFT_RADIUS_PX = 2.5


def gradient_magnitude(g: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(g)
    return np.hypot(gx, gy)


def edge_radii(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For a 2-D float picture in [0, ~1] (display-like or log-encoded values): a boolean map of edge pixels and the
    disc radius in pixels read at each pixel (meaningful at the edge pixels only)."""
    gray = np.asarray(gray, np.float64)
    g0 = gradient_magnitude(gray)
    g1 = gradient_magnitude(ndimage.gaussian_filter(gray, SIGMA_R))
    thr = max(MIN_GRADIENT, NOISE_K * float(np.percentile(g0, 75)))
    local_max = g0 >= ndimage.maximum_filter(g0, size=3)
    edge = (g0 > thr) & local_max & (g1 > 1e-9)
    ratio = np.where(edge, g0 / np.maximum(g1, 1e-9), 1.0)
    sigma0 = SIGMA_R / np.sqrt(np.maximum(ratio * ratio - 1.0, 1e-4))
    return edge, np.minimum(2.0 * sigma0, R_CLIP_PX)


def edge_stats(gray: np.ndarray) -> dict:
    """`density` (edge pixels per pixel), `strong` (their number among the strongest third) and `soft` (the share of
    those whose disc radius is above `SOFT_RADIUS_PX`), for the sharpness filter."""
    edge, radius = edge_radii(gray)
    n = int(edge.sum())
    if n == 0:
        return {"density": 0.0, "strong": 0, "soft": 1.0}
    g0 = gradient_magnitude(np.asarray(gray, np.float64))[edge]
    strong = g0 >= np.percentile(g0, 67)
    soft = float((radius[edge][strong] > SOFT_RADIUS_PX).mean())
    return {"density": n / gray.size, "strong": int(strong.sum()), "soft": soft}


def spread(values: np.ndarray, weights: np.ndarray, widths=(64.0, 32.0, 16.0, 8.0, 4.0, 2.0)) -> tuple[np.ndarray, np.ndarray]:
    """Normalised convolution with Gaussians from the widest to the narrowest: each finer level keeps its own estimate
    where it has enough weight and the coarser one elsewhere. Returns the dense map and the weight density at the
    narrowest width that has any."""
    dense = np.full(values.shape, np.nan)
    den_fine = np.zeros(values.shape)
    fallback = float(np.average(values, weights=weights)) if weights.sum() > 0 else 0.0
    for sigma in widths:
        num = ndimage.gaussian_filter(values * weights, sigma, mode="nearest")
        den = ndimage.gaussian_filter(weights, sigma, mode="nearest")
        level = np.where(den > 1e-4, num / np.maximum(den, 1e-12), np.nan)
        dense = np.where(np.isnan(dense) | (den > 3e-3), level, dense)
        den_fine = np.where(den > 3e-3, den, den_fine)
    return np.where(np.isnan(dense), fallback, dense), den_fine


def estimate(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The baseline's disc radius map (pixels) and confidence in [0, 1] for a 2-D picture."""
    edge, radius = edge_radii(gray)
    g0 = gradient_magnitude(np.asarray(gray, np.float64))
    weights = np.where(edge, np.minimum(1.0, g0 / (4 * max(MIN_GRADIENT, 1e-6))), 0.0)
    dense, den = spread(radius, weights)
    return dense, 1.0 - np.exp(-den / 0.01)


def gray_from_encoded(e: np.ndarray) -> np.ndarray:
    """A (3, H, W) encoded plane (`encode.encode_t`) as the 2-D picture the estimator reads: the channel mean."""
    return np.asarray(e, np.float64).mean(0)

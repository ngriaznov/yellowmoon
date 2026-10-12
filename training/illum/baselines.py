"""Simple illuminant estimators that the FFCC model has to beat, and the angular error.

All work on the same preview as the model (black-subtracted raw RGB, clipped pixels masked) and return the
camera white ``w = [r, 1, b]`` (green = 1) of the light.
"""

from __future__ import annotations

import numpy as np

from training.illum import features as ft


def w_from_uv(uv) -> np.ndarray:
    """Camera white ``[r, 1, b]`` from log-chroma ``(u, v) = (ln G/R, ln G/B)`` (works on (..., 2))."""
    uv = np.asarray(uv, dtype=np.float64)
    return np.stack([np.exp(-uv[..., 0]), np.ones(uv.shape[:-1]), np.exp(-uv[..., 1])], axis=-1)


def uv_from_w(w) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    return np.stack([-np.log(w[..., 0] / w[..., 1]), -np.log(w[..., 2] / w[..., 1])], axis=-1)


def angular_error_deg(w_est, w_true) -> np.ndarray:
    """Angle between two camera whites, degrees (the standard colour-constancy error)."""
    a = np.asarray(w_est, dtype=np.float64)
    b = np.asarray(w_true, dtype=np.float64)
    cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def angular_error_uv_deg(uv_est, uv_true) -> np.ndarray:
    return angular_error_deg(w_from_uv(uv_est), w_from_uv(uv_true))


def _valid_pixels(img, mask, floor):
    img = np.asarray(img, dtype=np.float64)
    valid = (np.zeros(img.shape[:2], dtype=bool) if mask is None else (np.asarray(mask) != 0))
    sel = (~valid) & (img > floor).all(axis=-1)
    return img[sel], sel


def grey_world(img, mask, floor=1.0 / 1024.0) -> np.ndarray:
    px, _ = _valid_pixels(img, mask, floor)
    if px.shape[0] == 0:
        return np.array([np.nan, 1.0, np.nan])
    m = px.mean(axis=0)
    return m / m[1]


def white_patch(img, mask, floor=1.0 / 1024.0, q=99.0) -> np.ndarray:
    px, _ = _valid_pixels(img, mask, floor)
    if px.shape[0] == 0:
        return np.array([np.nan, 1.0, np.nan])
    m = np.percentile(px, q, axis=0)
    return m / m[1]


def shades_of_grey(img, mask, p=6.0, floor=1.0 / 1024.0) -> np.ndarray:
    px, _ = _valid_pixels(img, mask, floor)
    if px.shape[0] == 0:
        return np.array([np.nan, 1.0, np.nan])
    m = (px**p).mean(axis=0) ** (1.0 / p)
    return m / m[1]


def grey_edge(img, mask, floor=1.0 / 4096.0) -> np.ndarray:
    valid = (np.ones(np.asarray(img).shape[:2], dtype=bool) if mask is None else (np.asarray(mask) == 0))
    edge, has = ft.edge_image(np.asarray(img, dtype=np.float64), valid)
    e = edge[has & (edge > floor).all(axis=-1)]
    if e.shape[0] == 0:
        return np.array([np.nan, 1.0, np.nan])
    m = e.mean(axis=0)
    return m / m[1]


BASELINES = {
    "grey_world": grey_world,
    "white_patch": white_patch,
    "shades_of_grey_p6": shades_of_grey,
    "grey_edge": grey_edge,
}


def fuse_gaussian(uv_a, var_a, uv_b, var_b):
    """Product of two isotropic Gaussians on (u, v): precision-weighted mean."""
    wa, wb = 1.0 / var_a, 1.0 / var_b
    return (wa * np.asarray(uv_a) + wb * np.asarray(uv_b)) / (wa + wb)

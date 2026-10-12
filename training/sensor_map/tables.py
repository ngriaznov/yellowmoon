"""Precomputed tables of a camera profile: ALEXA white ``w_A``, camera locus, CIE xy, Duv scale.

All tables sit on a uniform 10 K grid (``GRID_STEP_K``) and are stored as
``{"start_k", "step_k", "values"}``. Spectra come from :mod:`training.sensor_map.lights`
(``ssf_core`` conventions), integrated on the 380-780 nm / 5 nm grid:

* ``w_A(T)``: the ALEXA estimate's response to the light, ``S_A^T E``, divided by its green
  component, as ``[r, 1, b]``. Blackbody 1500-25000 K and CIE D series 4000-25000 K.
* camera locus: ``(ln R/G, ln B/G)`` of the Sony response ``S_P^T E`` to blackbody 1500-4000 K
  and CIE D 4000-25000 K (the rawtoaces rule: blackbody below 4000 K, D series from 4000 K).
  **``S_P`` must be in the camera's own raw scale** (the published Weta curves, clipped at 0, not
  normalised per channel: ``fit.sony_curves_raw``): the scene white at runtime comes from raw data
  (as-shot gains, Pick white, the illuminant estimator), so the locus is the raw white of the light.
  The curves of the maps and ``w_A`` tables are scale-free (every response is divided by the white
  response) and stay peak-normalised; see ``fit.load_curves_raw`` and docs/02 C1.
* ``xy``: CIE 1931 xy of the same lights, from ``CMF^T E`` on the same 5 nm grid.
* ``duv_scale``: CIE 1960 uv distance per unit of log-chroma distance **across** the locus, as a
  function of temperature (100 K grid per branch), measured on lights with a smooth green excess /
  deficit (see :func:`duv_scale_at`). It varies by about 50 % from 2000 K to 20000 K, so a single
  number would misread tint by 25 % at the ends. Measured with the raw-scale curves like the locus
  (the log-chroma distance it converts is a distance in the raw locus space).
"""

from __future__ import annotations

import numpy as np

from training.alexa import ssf_core as s
from training.sensor_map import lights as lt

GRID_STEP_K = 10
W_A_BLACKBODY_K = (1500, 25000)
W_A_DAYLIGHT_K = (4000, 25000)
LOCUS_BLACKBODY_K = (1500, 4000)
LOCUS_DAYLIGHT_K = (4000, 25000)
DECIMALS = 9  # table values are rounded to 1e-9 when written


def _grid(kind: str, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]:
    return (lt.blackbody_grid if kind == "blackbody" else lt.daylight_grid)(lo, hi, GRID_STEP_K)


def require_raw_scale(S_P: np.ndarray) -> np.ndarray:
    """Return ``S_P`` unchanged, or raise if it looks peak-normalised per channel.

    The locus tables are only valid for the camera's own raw channel balance (the curves as
    published, ``fit.sony_curves_raw``). Curves whose three channels all peak at 1 are what
    ``ssf_core.load_cameras`` returns; building a locus from them is the defect of docs/02 C1 (red
    off by x1.6, blue by x1.2). A measured camera never has three equal peaks (Weta's red peaks at
    0.57-0.62 of its green), so equal peaks identify the mistake.
    """
    peaks = np.asarray(S_P, dtype=float).max(axis=0)
    if np.allclose(peaks, peaks[0], rtol=0.0, atol=1e-9):
        raise ValueError(
            "locus tables need the camera curves in the published raw scale (fit.sony_curves_raw), "
            f"but all three channels peak at {peaks[0]:.6g}: these look peak-normalised per channel"
        )
    return S_P


def responses(E: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Raw channel responses ``S^T E`` of light(s) ``E`` ((81,) or (n, 81)) -> (3,) or (n, 3)."""
    return E @ S


def white_normalised(resp: np.ndarray) -> np.ndarray:
    """Divide by the green component (axis -1): green becomes 1."""
    return resp / resp[..., 1:2]


def log_chroma(resp: np.ndarray) -> np.ndarray:
    """``(ln R/G, ln B/G)`` of responses (n, 3) -> (n, 2)."""
    w = white_normalised(resp)
    return np.log(w[..., [0, 2]])


def xy_of(E: np.ndarray) -> np.ndarray:
    """CIE xy of SPD(s) (n, 81) -> (n, 2), 5 nm integration with the 1931 2-degree observer."""
    XYZ = E @ s.CMF
    return XYZ[..., :2] / XYZ.sum(axis=-1, keepdims=True)


def uv1960_of_xy(xy: np.ndarray) -> np.ndarray:
    """CIE 1960 (u, v) of xy chromaticities (n, 2) -> (n, 2)."""
    x, y = xy[..., 0], xy[..., 1]
    d = -2 * x + 12 * y + 3
    return np.stack([4 * x / d, 6 * y / d], axis=-1)


# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------
def w_a_tables(S_A: np.ndarray) -> dict[str, dict]:
    """``w_A`` grids (floats, unrounded): {"blackbody": {...}, "daylight": {...}}, values (n, 3)."""
    out = {}
    for kind, (lo, hi) in (("blackbody", W_A_BLACKBODY_K), ("daylight", W_A_DAYLIGHT_K)):
        ts, es = _grid(kind, lo, hi)
        out[kind] = {"start_k": lo, "step_k": GRID_STEP_K, "values": white_normalised(responses(es, S_A))}
    return out


def locus_tables(S_P: np.ndarray) -> dict[str, dict]:
    """Camera log-chroma and CIE xy grids (floats, unrounded) for the locus.

    ``S_P`` is the camera's curves in the raw scale (``fit.sony_curves_raw``, :func:`require_raw_scale`).
    """
    require_raw_scale(S_P)
    chroma, xy = {}, {}
    for kind, (lo, hi) in (("blackbody", LOCUS_BLACKBODY_K), ("daylight", LOCUS_DAYLIGHT_K)):
        ts, es = _grid(kind, lo, hi)
        chroma[kind] = {"start_k": lo, "step_k": GRID_STEP_K, "values": log_chroma(responses(es, S_P))}
        xy[kind] = {"start_k": lo, "step_k": GRID_STEP_K, "values": xy_of(es)}
    return {"chroma": chroma, "xy": xy}


def tint_offsets(
    S_P: np.ndarray,
    base: np.ndarray,
    base_minus: np.ndarray,
    base_plus: np.ndarray,
    tinted: np.ndarray,
) -> tuple[float, float]:
    """Perpendicular displacements caused by a tinted light: (camera log-chroma, CIE 1960 uv).

    ``base_minus`` and ``base_plus`` are the same kind of light a little cooler and warmer; they
    give the locus tangent at ``base`` in both spaces. ``tinted`` is the base light with a
    spectral tilt. Returns the signed components of ``tinted - base`` along the locus normal
    (tangent rotated by -90 degrees) in log-chroma space and in uv space.
    """
    def pu(E):
        return log_chroma(responses(E, S_P)), uv1960_of_xy(xy_of(E[None, :]))[0]

    p0, u0 = pu(base)
    pm, um = pu(base_minus)
    pp, up = pu(base_plus)
    pt, ut = pu(tinted)

    def normal(a, b):
        t = (b - a) / np.linalg.norm(b - a)
        return np.array([t[1], -t[0]])

    return float((pt - p0) @ normal(pm, pp)), float((ut - u0) @ normal(um, up))


# Tint model for the Duv scale: a smooth green excess or deficit (lights.tint_spd).
DUV_TINT_AMOUNTS = (-0.1, -0.05, 0.05, 0.1)
DUV_SCALE_STEP_K = 100
DUV_SCALE_RANGES_K = {"blackbody": LOCUS_BLACKBODY_K, "daylight": LOCUS_DAYLIGHT_K}


def duv_scale_at(S_P: np.ndarray, kind: str, t_k: float, amounts=DUV_TINT_AMOUNTS) -> float:
    """CIE-uv-per-log-chroma scale across the locus at ``t_k`` (median over the tint amounts).

    The locus tangent comes from lights ``max(10 K, 3 %)`` either side, clipped to the kind's range,
    which keeps the D series' 1e-4 staircase out of the direction.
    """
    make = lt.blackbody_spd if kind == "blackbody" else lt.daylight_spd
    lo_k, hi_k = DUV_SCALE_RANGES_K[kind]
    h = max(10.0, 0.03 * t_k)
    base, lo, hi = make(t_k), make(max(t_k - h, lo_k)), make(min(t_k + h, hi_k))
    ratios = []
    for a in amounts:
        dp, du = tint_offsets(S_P, base, lo, hi, lt.tint_spd(base, a))
        ratios.append(abs(du) / abs(dp))
    return float(np.median(ratios))


def duv_scale_tables(S_P: np.ndarray) -> dict[str, dict]:
    """``{"blackbody": grid, "daylight": grid}`` of the Duv scale, 100 K step, scalar ``values``.

    ``S_P`` is the camera's curves in the raw scale (``fit.sony_curves_raw``, :func:`require_raw_scale`).
    """
    require_raw_scale(S_P)
    out = {}
    for kind, (lo, hi) in DUV_SCALE_RANGES_K.items():
        ts = np.arange(lo, hi + 1, DUV_SCALE_STEP_K, dtype=float)
        out[kind] = {"start_k": lo, "step_k": DUV_SCALE_STEP_K,
                     "values": np.array([duv_scale_at(S_P, kind, t) for t in ts])}
    return out


def class_w_a(S_A: np.ndarray, klass: str, exclude: tuple[str, ...] = ()) -> list[dict]:
    """``w_A`` of the fluorescent or LED reference lights as ``[{"light", "cct_k", "mired", "w"}]``."""
    out = []
    for L in lt.reference_lights(exclude)[klass]:
        w = white_normalised(responses(L.spd, S_A))
        out.append({"light": L.name, "cct_k": L.cct_k, "mired": L.mired, "w": w})
    return out


def rounded(a, decimals: int = DECIMALS):
    """Round a float or array for output; returns Python floats / nested lists."""
    return np.round(np.asarray(a, dtype=float), decimals).tolist()

"""Labels of ``autoset`` (docs/18 sections 6.1 to 6.3): the exposure label, the contrast label and the key target.

Exposure (section 6.1). The G3 fit of the auto-exposure study: the quantile tables of our scene-linear luminance and of the camera
JPEG are matched over the 0.02 to 0.98 quantiles (``quantile_curve_from_tables``, the monotone cumulative maximum), and
``camera_gain_ev`` reads the scene level at which the curve reaches mid grey 0.18; ``L = log2(0.18 / x)``. Section 6.1's
mid-tone band is applied as follows, and nowhere else:

* **band rule.** Only table points whose JPEG linear luminance lies in ``[0.18 * 2**-2, 0.18 * 2**+1]`` (0.045 to 0.36, that is
  -2 to +1 stop around grey) enter the curve. When fewer than :data:`MIN_BAND_POINTS` points remain, the unrestricted 0.02 to
  0.98 table is used instead (``ExposureFit.band`` says which). On a dense table the two curves bracket 0.18 with the same two
  points, so ``L`` is the same as the study's G3 (``camera_gain_ev`` only reads the bracketing pair); the band changes ``L``
  only where the table has a gap that reaches beyond the band, and changes the residual term whenever points lie outside it.
* **variance.** ``V_L = 0.05**2 * f + min(r**2, 1)`` (EV squared), ``f = 4`` when the file is flagged (DRO or HDR not off, or a
  non-Standard creative style; unknown values, which are ``None``, do not flag), else 1. ``r`` is the rms residual of
  ``log2(jpeg)`` about its least-squares line over the points used, divided by the line's slope, so that it is in scene stops.

Contrast (section 6.2). For each contrast of the grid, the spreads ``p90 - p10`` and ``p75 - p25`` of our display luma, against the
same spreads of the camera JPEG. The squared difference is piecewise quadratic in ``c`` when the spreads are interpolated linearly
between grid points, so its minimum is found exactly on each segment. ``c`` is clamped to +-40. The variance is
``(r / g)**2`` in contrast units, where ``r`` is the distance between the two pairs of spreads at the minimum and
``g = |d spreads / dc|`` there (the speed of the spreads per contrast unit); it is clamped to ``[4, 1600]``.

Key target (section 6.3). ``key_target = L + key``: where the key should sit after exposure, with the same variance as ``L``
(``dataset.prepare`` computes it).
"""

from __future__ import annotations

import importlib.util
import math
import sys
from dataclasses import dataclass
from types import ModuleType

import numpy as np

from training.autoset.rows import CONTRAST_GRID, Row
from training.common import paths


def _load_study_metrics() -> ModuleType:
    """``tools/eval/metrics.py``, the auto-exposure study's maths, loaded by path (it imports only numpy and the standard library).

    Loaded by file name rather than imported, because the study keeps its helpers in a flat directory (``import metrics``).
    """
    name = "ym_tools_eval_metrics"
    if name in sys.modules:
        return sys.modules[name]
    path = paths.REPO_ROOT / "tools" / "eval" / "metrics.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the study's metrics from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_m = _load_study_metrics()

#: The quantile grid of the tables (``linspace(0.001, 0.999, 999)``) and its 0.02 to 0.98 part; ``tools/eval/study.py``
#: defines both as ``QUANTILE_GRID`` and ``INNER``.
GRID = np.linspace(0.001, 0.999, 999)
INNER = (GRID >= 0.02) & (GRID <= 0.98)
GRAY_LEVEL = _m.MID_GREY  # 0.18, the anchor of camera_gain_ev
BAND_STOPS = (-2.0, 1.0)
BAND_LO = GRAY_LEVEL * 2.0 ** BAND_STOPS[0]
BAND_HI = GRAY_LEVEL * 2.0 ** BAND_STOPS[1]
MIN_BAND_POINTS = 20
BASE_SD_EV = 0.05
FLAGGED_VARIANCE_FACTOR = 4.0
RESIDUAL_VAR_CAP_EV2 = 1.0
CONTRAST_LABEL_LIMIT = 40.0
CONTRAST_VAR_FLOOR = 4.0
CONTRAST_VAR_CAP = 1600.0
#: Positions of p10, p25, p75 and p90 in the contrast table's probabilities (``CONTRAST_P``).
_P10, _P25, _P75, _P90 = 1, 2, 4, 5


def is_flagged(meta: dict) -> bool:
    """True when the camera's JPEG was made with DRO or HDR on, or with a non-Standard creative style (section 6.1)."""
    dro, hdr, style = meta.get("dro"), meta.get("hdr"), meta.get("creative_style")
    return (dro not in (None, "off")) or (hdr not in (None, "off")) or (style not in (None, "standard"))


def is_normal(meta: dict) -> bool:
    """A photo the camera exposed normally: exposure compensation known and under 0.3 EV, and not flagged (section 8, gate 3)."""
    ec = meta.get("ec")
    if isinstance(ec, bool) or not isinstance(ec, (int, float)) or not math.isfinite(ec) or abs(ec) >= 0.3:
        return False
    return not is_flagged(meta)


@dataclass(frozen=True)
class ExposureFit:
    """The exposure label before the variance: the level, which points were used, and the fit's quality."""

    level_ev: float  # L (NaN when the curve does not reach mid grey within the table's range)
    band: bool  # True when the mid-tone band was used; False when fewer than MIN_BAND_POINTS points remained
    points: int  # table points used
    slope: float  # slope of the least-squares line of log2(jpeg) on log2(scene) over those points (NaN when undefined)
    residual_ev: float  # rms residual of that line in scene stops (inf when the line is not usable)


def _line_residual(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    """``(residual in scene stops, slope)`` of the least-squares line of ``log2 ys`` on ``log2 xs``."""
    keep = (xs > 0) & (ys > 0)
    if int(keep.sum()) < 3:
        return math.inf, math.nan
    lx = np.log2(xs[keep])
    ly = np.log2(ys[keep])
    dx = lx - lx.mean()
    sxx = float(dx @ dx)
    if not sxx > 0.0:
        return math.inf, math.nan
    slope = float(dx @ (ly - ly.mean())) / sxx
    resid = ly - (ly.mean() + slope * dx)
    rms = math.sqrt(float(resid @ resid) / len(resid))
    if not slope > 1e-3:
        return math.inf, slope
    return rms / slope, slope


def exposure_fit(row: Row) -> ExposureFit | None:
    """The G3 level of a file with the band rule (see the module docstring); None when the file has no ``g3`` tables."""
    if row.g3 is None:
        return None
    xs = row.g3.scene_q[INNER]
    ys = row.g3.jpeg_q[INNER]
    band = (ys >= BAND_LO) & (ys <= BAND_HI)
    use_band = int(band.sum()) >= MIN_BAND_POINTS
    sel = band if use_band else np.ones(xs.shape, dtype=bool)
    curve = _m.quantile_curve_from_tables(xs[sel], ys[sel])
    level = float(_m.camera_gain_ev(curve))
    residual, slope = _line_residual(curve[0], curve[1])
    return ExposureFit(level, use_band, int(sel.sum()), slope, residual)


def exposure_label(row: Row) -> tuple[float, float]:
    """``(L, V_L)``: the exposure label of a file and its variance (EV squared); ``(nan, nan)`` when there is none."""
    fit = exposure_fit(row)
    if fit is None or not math.isfinite(fit.level_ev):
        return math.nan, math.nan
    base = BASE_SD_EV**2 * (FLAGGED_VARIANCE_FACTOR if is_flagged(row.meta) else 1.0)
    return fit.level_ev, base + min(fit.residual_ev**2, RESIDUAL_VAR_CAP_EV2)


@dataclass(frozen=True)
class ContrastLabel:
    """The contrast label before the summary: the minimiser, its variance and the two quantities the variance is made of."""

    contrast: float  # c, clamped to +-40
    variance: float  # clamped to [CONTRAST_VAR_FLOOR, CONTRAST_VAR_CAP]
    residual: float  # distance between our spreads and the camera's at c (display luma, both spreads together)
    speed: float  # |d spreads / dc| at c, per contrast unit (0 when the spreads do not move)


def contrast_fit(row: Row) -> ContrastLabel | None:
    """The contrast label of a file (section 6.2); None when the file has no contrast table."""
    cf = row.contrast_fit
    if cf is None:
        return None
    q = cf.ours
    spreads = np.stack([q[:, _P90] - q[:, _P10], q[:, _P75] - q[:, _P25]], axis=1)
    target = np.array([cf.jpeg[_P90] - cf.jpeg[_P10], cf.jpeg[_P75] - cf.jpeg[_P25]], dtype=np.float64)
    best: tuple[float, int, float, float] | None = None
    for i in range(len(CONTRAST_GRID) - 1):
        d = spreads[i + 1] - spreads[i]
        e = spreads[i] - target
        dd = float(d @ d)
        u = 0.0 if dd <= 0.0 else min(max(-float(e @ d) / dd, 0.0), 1.0)
        v = e + u * d
        dist2 = float(v @ v)
        if best is None or dist2 < best[0]:
            best = (dist2, i, u, dd)
    assert best is not None  # the grid has at least two points
    dist2, i, u, dd = best
    step = CONTRAST_GRID[i + 1] - CONTRAST_GRID[i]
    c = min(max(CONTRAST_GRID[i] + step * u, -CONTRAST_LABEL_LIMIT), CONTRAST_LABEL_LIMIT)
    residual = math.sqrt(dist2)
    speed = math.sqrt(dd) / step
    variance = (residual / speed) ** 2 if speed > 0.0 else CONTRAST_VAR_CAP
    variance = min(max(variance, CONTRAST_VAR_FLOOR), CONTRAST_VAR_CAP)
    return ContrastLabel(float(c), float(variance), residual, speed)


def contrast_label(row: Row) -> tuple[float, float]:
    """``(c, V_c)``: the contrast label and its variance; ``(nan, nan)`` when the file has no contrast table."""
    fit = contrast_fit(row)
    if fit is None:
        return math.nan, math.nan
    return fit.contrast, fit.variance

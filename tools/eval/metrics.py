"""Pure numpy helpers of the auto-exposure study (`tools/eval/autoexp.py`, docs/02 "Auto exposure", Evaluation).

Nothing here touches a file or a process, so every function is unit tested (`test_metrics.py`). The pieces:

* colour: sRGB and Adobe RGB (1998) decoding of an 8-bit JPEG to linear luminance, the AWG4 `Y` row;
* a replica of `ym_color::exposure::auto_exposure` (steps 1 to 5 of the spec: the camera's gain, the highlight guard and
  the reported key statistic) with switches for the what-if variants of the previous metering rule (other target, other
  statistic, flat weights, shrinking towards a fixed gain);
* tone-curve estimation by quantile matching: the camera's own raw-to-JPEG curve (no registration needed) and our
  display transform's curve, read from the pictures themselves;
* the exposure placed by a display transform (ACES 2 neutral axis through OpenColorIO, when it is installed).

Conventions: luminances are linear light, "EV" is log2 of a ratio, a positive error means *we* are brighter than the
reference.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np

# --------------------------------------------------------------------------------------------------------------
# Colour
# --------------------------------------------------------------------------------------------------------------

#: Luminance weights `ym_color::exposure::jpeg_median_luminance` uses for an sRGB JPEG (the gate's own numbers).
SRGB_Y_ROW = (0.2126, 0.7152, 0.0722)
#: Adobe RGB (1998) `Y` row (the Sony previews flagged `InteropIndex R03` are in this space).
ADOBE_RGB_Y_ROW = (0.29734, 0.62736, 0.07529)
#: Adobe RGB (1998) transfer: a pure power, 563/256.
ADOBE_RGB_GAMMA = 563.0 / 256.0

D65_XY = (0.3127, 0.3290)
AWG4_PRIMARIES = ((0.7347, 0.2653), (0.1424, 0.8576), (0.0991, -0.0308))
MID_GREY = 0.18
#: `ym_color::exposure::CAMERA_GAIN_EV`: the interim `EV_auto` of every photo (docs/02 Auto exposure, decision 2026-10-14), the
#: median gain of the cameras' own JPEG engines over the 12 local files (this study measures it as G3's reference).
CAMERA_GAIN_EV = 1.11

#: The meter calibration of ISO 2720 for reflected light and the constants of ISO 12232's saturation-based speed.
#: A camera that meters an average (18 %) scene exactly puts the grey at `K q / 78` of the saturation exposure.
METER_K = 12.5
LENS_Q = 0.65
SAT_SPEED_CONSTANT = 78.0


def srgb_to_linear(code: np.ndarray | float) -> np.ndarray:
    """IEC 61966-2-1 decoding, code value 0 to 1 to linear light."""
    v = np.asarray(code, dtype=np.float64)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def adobe_rgb_to_linear(code: np.ndarray | float) -> np.ndarray:
    """Adobe RGB (1998) decoding (a pure power of 563/256)."""
    return np.power(np.clip(np.asarray(code, dtype=np.float64), 0.0, 1.0), ADOBE_RGB_GAMMA)


COLOUR_SPACES: dict[str, tuple[Callable[[np.ndarray], np.ndarray], tuple[float, float, float]]] = {
    "srgb": (srgb_to_linear, SRGB_Y_ROW),
    "adobergb": (adobe_rgb_to_linear, ADOBE_RGB_Y_ROW),
}


def jpeg_luminance(rgb8: np.ndarray, colour_space: str = "srgb", rows_per_chunk: int = 512) -> np.ndarray:
    """Linear luminance of an 8-bit `(H, W, 3)` image as `(H, W)` float32 (a chunked pass: the preview of a 61 MP
    camera is big)."""
    if rgb8.ndim != 3 or rgb8.shape[2] != 3 or rgb8.dtype != np.uint8:
        raise ValueError("an 8-bit (H, W, 3) image is needed")
    decode, row = COLOUR_SPACES[colour_space]
    lut = decode(np.arange(256) / 255.0)
    out = np.empty(rgb8.shape[:2], dtype=np.float32)
    for r0 in range(0, rgb8.shape[0], rows_per_chunk):
        chunk = lut[rgb8[r0 : r0 + rows_per_chunk]]
        out[r0 : r0 + rows_per_chunk] = chunk @ np.asarray(row)
    return out


def display_luminance_u16(rgb16: np.ndarray, rows_per_chunk: int = 256) -> np.ndarray:
    """Linear Rec.709 luminance of a 16-bit sRGB `(H, W, 3)` image (our PNG or TIFF export) as float32."""
    if rgb16.dtype != np.uint16 or rgb16.ndim != 3 or rgb16.shape[2] != 3:
        raise ValueError("a 16-bit (H, W, 3) image is needed")
    out = np.empty(rgb16.shape[:2], dtype=np.float32)
    for r0 in range(0, rgb16.shape[0], rows_per_chunk):
        lin = srgb_to_linear(rgb16[r0 : r0 + rows_per_chunk] / 65535.0)
        out[r0 : r0 + rows_per_chunk] = lin @ np.asarray(SRGB_Y_ROW)
    return out


def _xy_to_xyz(xy: tuple[float, float]) -> np.ndarray:
    x, y = xy
    return np.array([x / y, 1.0, (1.0 - x - y) / y])


def rgb_to_xyz_matrix(primaries: Sequence[tuple[float, float]], white: tuple[float, float]) -> np.ndarray:
    """Column-vector RGB to XYZ matrix of a gamut (white has `Y = 1`)."""
    prim = np.array([_xy_to_xyz(p) for p in primaries]).T
    scale = np.linalg.solve(prim, _xy_to_xyz(white))
    return prim * scale


def awg4_y_row() -> np.ndarray:
    """The `Y` row of AWG4 to XYZ, `[0.2545, 0.7815, -0.0360]` (`ym_color::gamut::awg4_luminance_row`)."""
    return rgb_to_xyz_matrix(AWG4_PRIMARIES, D65_XY)[1]


# --------------------------------------------------------------------------------------------------------------
# Reductions
# --------------------------------------------------------------------------------------------------------------


def box_mean(a: np.ndarray, k: int) -> np.ndarray:
    """Mean of `k x k` blocks of an `(H, W)` or `(H, W, C)` array; a remainder at the right and bottom is dropped.

    The photo's 1/8 preview (`docs/01`) bins 8 x 8 photosites per colour, so `box_mean(rgb, 8)` of the full-resolution
    scene-linear image is its stand-in (it has the same noise averaging, not the same demosaic).
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    h, w = a.shape[0] // k * k, a.shape[1] // k * k
    if h == 0 or w == 0:
        raise ValueError("the image is smaller than one block")
    a = a[:h, :w]
    shape = (h // k, k, w // k, k) + a.shape[2:]
    return a.reshape(shape).mean(axis=(1, 3), dtype=np.float64).astype(np.float32)


def resize_area(y: np.ndarray, width: int, height: int) -> np.ndarray:
    """Area-average resample of a linear `(H, W)` float array to `(height, width)` (PIL's box filter on mode F)."""
    from PIL import Image

    img = Image.fromarray(np.ascontiguousarray(y, dtype=np.float32), mode="F")
    return np.asarray(img.resize((width, height), Image.Resampling.BOX), dtype=np.float32)


def centre_weights(height: int, width: int, sigma: float = 0.35) -> np.ndarray:
    """`exp(-r^2 / (2 sigma^2))`, `r` the distance from the centre over the half diagonal (`docs/02` step 3)."""
    py = (np.arange(height) + 0.5 - 0.5 * height)[:, None]
    px = (np.arange(width) + 0.5 - 0.5 * width)[None, :]
    r_sq = (px * px + py * py) / (0.25 * (width * width + height * height))
    return np.exp(-r_sq / (2.0 * sigma * sigma))


# --------------------------------------------------------------------------------------------------------------
# The auto-exposure replica
# --------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MeterConfig:
    """The constants of `docs/02` Auto exposure, with switches for the what-if variants.

    `rule` is `camera_gain` (the spec since 2026-10-14: `EV_auto` is `camera_gain_ev` before the highlight guard, whatever the
    scene and whatever the exposure compensation, which the raw already holds) or `metered` (the previous rule,
    `log2(target / statistic) + ec_weight * EC`, kept for the what-if studies of `study.py`). `stat` is `geomean` (the spec's
    reported statistic), `mean` (arithmetic), `median` or `pNN` (a plain percentile of the unmasked luminances, for example
    `p90`); `centre_sigma = None` makes the weights flat.
    """

    rule: str = "camera_gain"
    camera_gain_ev: float = CAMERA_GAIN_EV
    target: float = 0.18
    centre_sigma: float | None = 0.35
    low_quantile: float = 0.01
    high_quantile: float = 0.99
    ec_weight: float = 0.5
    ev_limit: float = 3.0
    guard_level: float = 8.0
    guard_fraction: float = 0.005
    guard_max_reduction_ev: float = 1.0
    floor: float = 1e-4
    stat: str = "geomean"


#: The previous rule (metering to 0.18 with half the exposure compensation), for the what-if studies.
PREVIOUS_RULE = MeterConfig(rule="metered")


@dataclass(frozen=True)
class Meter:
    """The result of metering, like `ym_color::exposure::AutoExposure`."""

    ev_auto: float
    ev_before_guard: float
    guard_reduction_ev: float
    statistic: float
    unmasked_pixels: int
    metered_pixels: int


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    v, w = values[order], weights[order]
    cum = np.cumsum(w)
    return float(v[min(int(np.searchsorted(cum, 0.5 * cum[-1])), len(v) - 1)])


def meter(y: np.ndarray, ec_ev: float = 0.0, mask: np.ndarray | None = None, cfg: MeterConfig | None = None) -> Meter:
    """Steps 1 to 5 of the spec on an `(H, W)` luminance image at EV 0 (a replica of `auto_exposure`).

    With the default rule the exposure is the camera's gain, lowered by the highlight guard when the picture would clip; the
    key statistic is computed and reported and does not move it, and `ec_ev` is not used (the raw already holds the
    compensation). Only the previous rule (`MeterConfig(rule="metered")`) meters to the target and adds half of `ec_ev`.

    The exact Rust code meters the camera's 1/8 preview through the fused matrix; here `y` is the AWG4 luminance
    of the DNG that `ym render --exposure 0` writes (so it has gone through C7, sharpening and the demosaic).
    `tools/eval/autoexp.py` reports how far this replica is from the `ym` value for every file.
    """
    cfg = cfg or MeterConfig()
    if cfg.rule not in ("camera_gain", "metered"):
        raise ValueError(f"unknown rule {cfg.rule!r}")
    y = np.asarray(y, dtype=np.float64)
    h, w = y.shape
    ok = np.isfinite(y)
    if mask is not None:
        ok &= ~mask.astype(bool)
    n = int(ok.sum())
    ec = ec_ev if math.isfinite(ec_ev) else 0.0
    compensation = cfg.ec_weight * ec
    gain = min(max(cfg.camera_gain_ev, -cfg.ev_limit), cfg.ev_limit)
    if n == 0:
        ev = gain if cfg.rule == "camera_gain" else min(max(compensation, -cfg.ev_limit), cfg.ev_limit)
        return Meter(ev, ev, 0.0, 0.0, 0, 0)
    yv = np.maximum(y[ok], cfg.floor)
    sorted_y = np.sort(yv)
    low = float(np.quantile(sorted_y, cfg.low_quantile))
    high = float(np.quantile(sorted_y, cfg.high_quantile))
    wts = centre_weights(h, w, cfg.centre_sigma) if cfg.centre_sigma else np.ones((h, w))
    wv = wts[ok]
    band = (yv >= low) & (yv <= high)
    if not band.any():
        band = np.ones_like(band)
    yb, wb = yv[band], wv[band]
    if cfg.stat == "geomean":
        stat = math.exp(float((wb * np.log(yb)).sum() / wb.sum()))
    elif cfg.stat == "mean":
        stat = float((wb * yb).sum() / wb.sum())
    elif cfg.stat == "median":
        stat = _weighted_median(yb, wb)
    elif cfg.stat.startswith("p") and cfg.stat[1:].replace(".", "", 1).isdigit():
        stat = float(np.quantile(sorted_y, float(cfg.stat[1:]) / 100.0))
    else:
        raise ValueError(f"unknown statistic {cfg.stat!r}")
    if cfg.rule == "camera_gain":
        ev_before_guard = gain
    else:
        ev_before_guard = min(max(math.log2(cfg.target / stat) + compensation, -cfg.ev_limit), cfg.ev_limit)
    allowed = math.floor(cfg.guard_fraction * n)
    threshold = float(sorted_y[n - 1 - min(allowed, n - 1)])
    ev_guard = math.log2(cfg.guard_level / threshold)
    ev_auto = max(ev_guard, ev_before_guard - cfg.guard_max_reduction_ev) if ev_before_guard > ev_guard else ev_before_guard
    return Meter(ev_auto, ev_before_guard, ev_before_guard - ev_auto, stat, n, int(band.sum()))


def nominal_grey_fraction(k: float = METER_K, q: float = LENS_Q, sat: float = SAT_SPEED_CONSTANT) -> float:
    """Where a camera that meters an average scene exactly puts the 18 % grey, as a fraction of the white level.

    ISO 2720 meters reflected light with `L = K N^2 / (t S)`; ISO 12232 defines the saturation-based speed
    `S_sat = 78 / H_sat` with `H = q L t / N^2`. Eliminating the exposure: grey sits at `K q / 78` of saturation
    (0.104 with the nominal constants, 3.3 stops below the white level). A nominal figure: Sony's ISO figures are
    not guaranteed to be saturation-based, so the study reports it as a reference line and never as a measurement.
    """
    return k * q / sat


# --------------------------------------------------------------------------------------------------------------
# Tone curves by quantile matching
# --------------------------------------------------------------------------------------------------------------

DEFAULT_PROBABILITIES = np.linspace(0.02, 0.98, 97)


def quantile_curve(x: np.ndarray, y: np.ndarray, probabilities: np.ndarray = DEFAULT_PROBABILITIES) -> tuple[np.ndarray, np.ndarray]:
    """The monotone map that sends the distribution of `x` onto that of `y`: `(q_x(p), q_y(p))` for each `p`.

    If `y = f(x)` pixel by pixel for a non-decreasing `f` and both images show the same scene, `q_y(p) = f(q_x(p))`
    exactly, so the curve is found without registering the images (different resolution, crop and orientation do not
    matter, only that they show about the same area). Ties and flat parts are kept (non-decreasing).
    """
    qx = np.quantile(np.asarray(x, dtype=np.float64).ravel(), probabilities)
    qy = np.quantile(np.asarray(y, dtype=np.float64).ravel(), probabilities)
    return np.maximum.accumulate(qx), np.maximum.accumulate(qy)


def quantile_curve_from_tables(qx: np.ndarray, qy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`quantile_curve` for two quantile tables that were taken at the same probabilities already."""
    return np.maximum.accumulate(np.asarray(qx, dtype=np.float64)), np.maximum.accumulate(np.asarray(qy, dtype=np.float64))


def _log_interp(x: float, xs: np.ndarray, ys: np.ndarray) -> float:
    """Piecewise linear in log-log between points; linear extrapolation from the two outermost points."""
    keep = (xs > 0) & (ys > 0)
    lx, ly = np.log(xs[keep]), np.log(ys[keep])
    if len(lx) < 2 or not (x > 0):
        return math.nan
    # strictly increasing abscissae only
    order = np.argsort(lx, kind="stable")
    lx, ly = lx[order], ly[order]
    uniq = np.concatenate([[True], np.diff(lx) > 1e-12])
    lx, ly = lx[uniq], ly[uniq]
    if len(lx) < 2:
        return math.nan
    t = math.log(x)
    if t <= lx[0]:
        slope = (ly[1] - ly[0]) / (lx[1] - lx[0])
        return math.exp(ly[0] + slope * (t - lx[0]))
    if t >= lx[-1]:
        slope = (ly[-1] - ly[-2]) / (lx[-1] - lx[-2])
        return math.exp(ly[-1] + slope * (t - lx[-1]))
    return math.exp(float(np.interp(t, lx, ly)))


def curve_at(curve: tuple[np.ndarray, np.ndarray], x: float) -> float:
    """The curve's output at input `x` (log-log interpolation)."""
    return _log_interp(x, curve[0], curve[1])


def curve_inverse_at(curve: tuple[np.ndarray, np.ndarray], y: float) -> float:
    """The input at which the curve reaches `y` (log-log interpolation; NaN when `y` is outside the data's range by
    more than 2 stops, where an extrapolation would be a guess)."""
    xs, ys = curve
    pos = ys[ys > 0]
    if len(pos) == 0 or y < pos.min() / 4.0 or y > pos.max() * 4.0:
        return math.nan
    return _log_interp(y, ys, xs)


def camera_gain_ev(curve: tuple[np.ndarray, np.ndarray], anchor: float = MID_GREY) -> float:
    """The camera engine's scene-referred gain in stops: `log2(anchor / x)` for the raw level `x` its JPEG renders as
    `anchor` in linear light (ISO 12232's convention puts the grey at 0.18). It is the exposure the camera's own
    rendering applies to the raw data, free of the display transform and of the shape of the camera's tone curve
    (only the one point of the curve at the anchor is used)."""
    x = curve_inverse_at(curve, anchor)
    return math.log2(anchor / x) if x > 0 else math.nan


def log_slope(curve: tuple[np.ndarray, np.ndarray], x: float, half_width_ev: float = 1.0) -> float:
    """`d log y / d log x` of the curve at `x`, over plus and minus `half_width_ev` stops (contrast of the curve; 1 for
    a tone curve that is a pure gain)."""
    lo, hi = x * 2.0**-half_width_ev, x * 2.0**half_width_ev
    ylo, yhi = curve_at(curve, lo), curve_at(curve, hi)
    if not (ylo > 0 and yhi > 0):
        return math.nan
    return math.log(yhi / ylo) / math.log(hi / lo)


# --------------------------------------------------------------------------------------------------------------
# Error measures
# --------------------------------------------------------------------------------------------------------------


def ev_error(ours: float, reference: float) -> float:
    """`log2(ours / reference)`; positive when we are brighter (`ym_color::exposure::exposure_error_ev`)."""
    if not (ours > 0 and reference > 0):
        return math.nan
    return math.log2(ours / reference)


def pass_rate(errors: Sequence[float], tolerance_ev: float = 0.5) -> tuple[int, int]:
    """`(within, evaluated)`: errors that are finite and at most `tolerance_ev` in size, and errors that are finite."""
    finite = [e for e in errors if math.isfinite(e)]
    return sum(1 for e in finite if abs(e) <= tolerance_ev), len(finite)


def summary(errors: Sequence[float]) -> dict[str, float]:
    """Median, mean, standard deviation, RMS and the extremes of the finite errors."""
    a = np.array([e for e in errors if math.isfinite(e)], dtype=np.float64)
    if a.size == 0:
        return {k: math.nan for k in ("n", "median", "mean", "std", "rms", "min", "max")}
    return {
        "n": float(a.size),
        "median": float(np.median(a)),
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "rms": float(np.sqrt((a**2).mean())),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def shrunk_ev(ev_auto: float, anchor_ev: float, weight: float) -> float:
    """`anchor + weight (ev_auto - anchor)`: the what-if that keeps part of the camera's own placement (weight 1 is
    the current algorithm, 0 a fixed gain)."""
    return anchor_ev + weight * (ev_auto - anchor_ev)


# --------------------------------------------------------------------------------------------------------------
# The display transform's placement of mid grey
# --------------------------------------------------------------------------------------------------------------

ACES_WHITE_XY = (0.32168, 0.33767)


def aces2_neutral_display_y(scene_values: Sequence[float]) -> np.ndarray | None:
    """Display-linear luminance (100 nit = 1) of neutral scene-linear values through the ACES 2.0 SDR 100 nit
    Rec.709-limited output transform, computed by OpenColorIO's built-in transform (the same one
    `tools/vectors/aces2_vectors.py` uses for the vectors our own port is tested against). `None` when
    OpenColorIO is not installed. A neutral AWG4 value `s` is the neutral AP0 value `s` (both white points map to
    each other), so the ACES input is `[s, s, s]`."""
    try:
        import PyOpenColorIO as ocio  # type: ignore[import-not-found]
    except ImportError:
        return None
    t = ocio.BuiltinTransform()
    t.setStyle("ACES-OUTPUT - ACES2065-1_to_CIE-XYZ-D65 - SDR-100nit-REC709_2.0")
    config = ocio.Config.CreateRaw()
    cpu = config.getProcessor(t).getDefaultCPUProcessor()
    rgb = np.repeat(np.asarray(scene_values, dtype=np.float32)[:, None], 3, axis=1).copy()
    cpu.applyRGB(rgb)
    return rgb[:, 1].astype(np.float64)  # XYZ output: Y is the second component


def with_target(cfg: MeterConfig, target: float) -> MeterConfig:
    """`cfg` with another metering target (a convenience for the sweeps)."""
    return replace(cfg, target=target)

"""The arithmetic of the auto-exposure study: per-file metrics from the reduced pictures, and the sweeps over them.

Everything takes plain numpy arrays and numbers (`autoexp.py` reads the files, renders with `ym` and builds them), so it
is unit tested on synthetic pictures with known tone curves (`test_study.py`).

A *reduction* of one photo is a dict of what the study needs from the three pictures:

* `y0_ds`: the scene-linear AWG4 luminance of our render at EV 0 (the DNG `ym render --format dng --exposure 0`
  writes: C1 to C11, before the display transform), 8 x 8 block means (the stand-in of the 1/8 preview);
* `y0_q` (optional): the same luminance area-resampled to the size of the camera JPEG, as a table of quantiles over
  `QUANTILE_GRID`. The camera curve is read by quantile matching, which is only exact when both pictures are at the
  same spatial scale (a block mean is narrower than a 3x downsample, which is narrower than the full-size picture;
  at 8 x 8 against a 3.7x downsample the camera gain reads 0.1 EV low on a real photo and up to 0.2 EV low on the
  synthetic charts, where the matched scale is exact to 0.02). Without it the 8 x 8 means are used (the unit tests);
* `disp_ds`: the linear luminance of our sRGB render at the automatic exposure (16-bit TIFF), 8 x 8 block means;
* `disp_median`: the median of the display luminance at full resolution;
* `j_q`, `j_median`: the camera JPEG's linear luminance as a table of quantiles over `QUANTILE_GRID` and its median.

Terms: *G1* is the gate as `ym_color::exposure` implements it, *G2* the same comparison on our display-referred render,
*G3* the DRT-neutral, tone-curve-anchored comparison (`docs/02` Auto exposure, Evaluation).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import metrics as m
import numpy as np

#: Probabilities of the quantile tables (0.1 % steps; the camera curve uses the inner 2 to 98 %).
QUANTILE_GRID = np.linspace(0.001, 0.999, 999)
INNER = (QUANTILE_GRID >= 0.02) & (QUANTILE_GRID <= 0.98)

#: Mid-grey placements (display-linear luminance of scene 0.18) the sweep tries. ACES 2 is measured (0.100); the
#: others are parameters, not measurements of any product.
PLACEMENTS = (0.10, 0.12, 0.14, 0.16, 0.18, 0.20)

#: Scene-brightness statistics tried as "the camera's meter".
METER_STATS = ("geomean", "mean", "median", "p75", "p90", "p95", "p99")


def _median_of_table(table: np.ndarray) -> float:
    """The median of a distribution given as a quantile table over `QUANTILE_GRID`."""
    return float(np.interp(0.5, QUANTILE_GRID, table))


def file_metrics(red: Mapping[str, Any], ev_auto: float, ec_ev: float = 0.0) -> dict[str, Any]:
    """The per-file numbers of the study from one reduction and the `ym` automatic exposure.

    Positive errors mean our picture is brighter than the camera's JPEG, or our exposure is higher than the camera's
    engine gain.
    """
    y0 = np.asarray(red["y0_ds"], dtype=np.float64)
    y0_median = float(np.median(y0))
    j_median = float(red["j_median"])
    jq = np.asarray(red["j_q"], dtype=np.float64)
    out: dict[str, Any] = {"ev_auto": ev_auto, "ec_ev": ec_ev}

    # G1: the gate as implemented: the median of our scene-linear luminance at EV_auto against the JPEG's.
    out["scene_median_ev0"] = y0_median  # the raw scene's key as shot (our scene-linear median at EV 0)
    out["scene_median_at_auto"] = y0_median * 2.0**ev_auto
    out["jpeg_median"] = j_median
    out["g1_gate_ev"] = m.ev_error(out["scene_median_at_auto"], j_median)

    # G2: the same on our display-referred render.
    out["display_median"] = float(red["disp_median"])
    out["g2_display_ev"] = m.ev_error(out["display_median"], j_median)

    # G3: the camera's engine gain from its own tone curve, against our exposure. Both quantile tables are taken at
    # the scale of the JPEG (`y0_q`); the 8 x 8 means only stand in when a reduction has no `y0_q`.
    xq = np.asarray(red["y0_q"], dtype=np.float64) if "y0_q" in red else np.quantile(y0, QUANTILE_GRID)
    cam_curve = m.quantile_curve_from_tables(xq[INNER], jq[INNER])
    out["camera_gain_ev"] = m.camera_gain_ev(cam_curve)
    out["g3_scene_ev"] = ev_auto - out["camera_gain_ev"]
    out["camera_contrast_slope"] = m.log_slope(cam_curve, 0.18 * 2.0 ** -out["camera_gain_ev"]) if math.isfinite(out["camera_gain_ev"]) else math.nan
    # The identity-curve gain at the median pair (what G1 implicitly assumes the camera's tone curve to be).
    out["camera_gain_at_median_ev"] = m.ev_error(j_median, y0_median)

    # Our display transform's own curve on this photo, from its pixels: scene (at EV_auto) to display.
    d = np.asarray(red["disp_ds"], dtype=np.float64)
    scene_at_auto = y0 * 2.0**ev_auto
    disp_curve = m.quantile_curve(scene_at_auto, d, QUANTILE_GRID[INNER])
    out["display_of_grey"] = m.curve_at(disp_curve, 0.18)
    out["display_curve_slope"] = m.log_slope(disp_curve, 0.18)

    # The scene statistics at EV 0 (replica meter, no clamp): they feed the meter-emulation table.
    stats = {}
    for name in METER_STATS:
        r = m.meter(red["y0_ds"], 0.0, None, m.MeterConfig(rule="metered", stat=name, ev_limit=99.0, guard_max_reduction_ev=0.0, guard_fraction=1.0))
        stats[name] = r.statistic
    out["stats_ev0"] = stats
    flat = m.meter(red["y0_ds"], 0.0, None, m.MeterConfig(rule="metered", centre_sigma=None, ev_limit=99.0))
    out["geomean_flat_ev0"] = flat.statistic
    out["scene_contrast_ev"] = math.log2(stats["mean"] / stats["geomean"]) if stats["geomean"] > 0 else math.nan
    out["scene_range_ev"] = math.log2(float(np.quantile(y0, 0.99)) / max(float(np.quantile(y0, 0.01)), 1e-4))
    return out


def metrics_at(entry: Mapping[str, Any], ev_prime: float) -> dict[str, float]:
    """G1, G2 and G3 of one file if its exposure were `ev_prime` instead of the automatic one.

    G1 and G3 move one for one with the exposure. G2 needs our display curve: the display luminance at the median is
    read from the curve at the scene median shifted by the change (the curve is a log-log interpolation of the
    quantile curve of the render itself, extended along its end slopes).
    """
    delta = ev_prime - entry["ev_auto"]
    disp_curve = entry["_display_curve"]
    scene_median = entry["scene_median_at_auto"] * 2.0**delta
    disp_median = m.curve_at(disp_curve, scene_median) if abs(delta) > 1e-12 else entry["display_median"]
    return {
        "g1": entry["g1_gate_ev"] + delta,
        "g2": m.ev_error(disp_median, entry["jpeg_median"]),
        "g3": entry["g3_scene_ev"] + delta,
    }


def attach_curves(entry: dict[str, Any], red: Mapping[str, Any]) -> None:
    """Adds the display curve (kept out of the JSON) the sweeps need."""
    y0 = np.asarray(red["y0_ds"], dtype=np.float64)
    d = np.asarray(red["disp_ds"], dtype=np.float64)
    entry["_display_curve"] = m.quantile_curve(y0 * 2.0 ** entry["ev_auto"], d, QUANTILE_GRID[INNER])
    entry["_y0_ds"] = y0


def tally(errors: Sequence[float], tolerance_ev: float = 0.5) -> dict[str, float]:
    """Pass count, evaluated count and the distribution of a metric over files."""
    within, evaluated = m.pass_rate(errors, tolerance_ev)
    s = m.summary(errors)
    s.update({"within": within, "evaluated": evaluated})
    return s


def metric_table(entries: Sequence[Mapping[str, Any]], tolerance_ev: float = 0.5) -> dict[str, dict[str, float]]:
    """G1, G2 and G3 over files: tallies."""
    return {
        "g1_gate_ev": tally([e["g1_gate_ev"] for e in entries], tolerance_ev),
        "g2_display_ev": tally([e["g2_display_ev"] for e in entries], tolerance_ev),
        "g3_scene_ev": tally([e["g3_scene_ev"] for e in entries], tolerance_ev),
    }


def offset_sweep(entries: Sequence[Mapping[str, Any]], offsets: Iterable[float], tolerance_ev: float = 0.5) -> list[dict[str, Any]]:
    """What a constant change of exposure does to each metric: per offset, the median error and the pass count."""
    rows = []
    for delta in offsets:
        per = [metrics_at(e, e["ev_auto"] + delta) for e in entries]
        row: dict[str, Any] = {"offset_ev": delta}
        for key in ("g1", "g2", "g3"):
            errs = [p[key] for p in per]
            within, evaluated = m.pass_rate(errs, tolerance_ev)
            row[f"{key}_median"] = m.summary(errs)["median"]
            row[f"{key}_within"] = within
            row[f"{key}_evaluated"] = evaluated
        rows.append(row)
    return rows


def best_offset(entries: Sequence[Mapping[str, Any]], key: str) -> float:
    """The constant exposure offset that makes the median error of a metric zero (`key`: g1, g2 or g3)."""
    errs = [metrics_at(e, e["ev_auto"])[key] for e in entries]
    return -float(np.median([e for e in errs if math.isfinite(e)]))


def placement_sweep(entries: Sequence[Mapping[str, Any]], placements: Sequence[float] = PLACEMENTS, tolerance_ev: float = 0.5) -> list[dict[str, Any]]:
    """G2 if the display transform placed scene 0.18 at `g` instead of where ours does.

    The display luminance is scaled by `g / D(0.18)` with `D(0.18)` this render's own value (the shape of the
    transform is kept; only its mid-grey placement moves). `g = 0.10` is ACES 2.
    """
    rows = []
    for g in placements:
        errs = []
        for e in entries:
            scaled = e["display_median"] * g / e["display_of_grey"]
            errs.append(m.ev_error(scaled, e["jpeg_median"]))
        within, evaluated = m.pass_rate(errs, tolerance_ev)
        row = {"placement": g, "median_ev": m.summary(errs)["median"], "within": within, "evaluated": evaluated}
        rows.append(row)
    return rows


def variant_ev(entry: Mapping[str, Any], cfg: m.MeterConfig, base: m.MeterConfig | None = None) -> float:
    """The exposure a variant of the algorithm would choose, by the replica.

    A variant of the previous rule (`cfg.rule == "metered"`, the what-if sweeps of the metering) has no `ym` value any more
    (`ym` uses the camera's gain), so it is the replica alone: it matched `ym` to 0.03 EV when the rule was ym's. A variant of
    the current rule moves the `ym` value by what the replica says it changes (so the replica's small bias against `ym`
    cancels).
    """
    y0 = entry["_y0_ds"]
    ec = entry["ec_ev"]
    if cfg.rule == "metered":
        return m.meter(y0, ec, None, cfg).ev_auto
    base = base or m.MeterConfig()
    return entry["ev_auto"] + m.meter(y0, ec, None, cfg).ev_auto - m.meter(y0, ec, None, base).ev_auto


def variant_row(entries: Sequence[Mapping[str, Any]], evs: Sequence[float], tolerance_ev: float = 0.5) -> dict[str, Any]:
    """Medians and pass counts of G1, G2 and G3 for a list of exposures, one per file."""
    per = [metrics_at(e, ev) for e, ev in zip(entries, evs)]
    row: dict[str, Any] = {}
    for key in ("g1", "g2", "g3"):
        errs = [p[key] for p in per]
        within, evaluated = m.pass_rate(errs, tolerance_ev)
        s = m.summary(errs)
        row.update({f"{key}_median": s["median"], f"{key}_std": s["std"], f"{key}_within": within, f"{key}_evaluated": evaluated})
    return row


def target_sweep(entries: Sequence[Mapping[str, Any]], targets: Sequence[float], tolerance_ev: float = 0.5) -> list[dict[str, Any]]:
    """The metering target (0.18 in the spec) varied, everything else as is."""
    rows = []
    for t in targets:
        cfg = m.with_target(m.PREVIOUS_RULE, t)
        row = {"target": t}
        row.update(variant_row(entries, [variant_ev(e, cfg) for e in entries], tolerance_ev))
        rows.append(row)
    return rows


def shrink_sweep(entries: Sequence[Mapping[str, Any]], anchor_ev: float, weights: Sequence[float], tolerance_ev: float = 0.5) -> list[dict[str, Any]]:
    """`ev = anchor + w (ev_previous - anchor)`: partly keeping the camera's placement. The unshrunk exposure is the previous
    rule's (metering to 0.18, `ev_previous_rule`, a replica value; `ym` no longer chooses it, a file without it falls back to
    `ev_auto`): w = 1 is that rule, 0 a fixed gain, which is what `ym` does now with the anchor at the camera gain."""
    rows = []
    for w in weights:
        row = {"weight": w, "anchor_ev": anchor_ev}
        evs = [m.shrunk_ev(e.get("ev_previous_rule", e["ev_auto"]), anchor_ev, w) for e in entries]
        row.update(variant_row(entries, evs, tolerance_ev))
        rows.append(row)
    return rows


def statistic_sweep(entries: Sequence[Mapping[str, Any]], tolerance_ev: float = 0.5) -> list[dict[str, Any]]:
    """Other meters: each statistic normalised to 0.18 after the same percentile band (the statistic is the only
    change, so a statistic with a different typical value than the geometric mean has an offset; see `best_offset`)."""
    rows = []
    for name in METER_STATS:
        cfg = m.MeterConfig(rule="metered", stat=name)
        evs = [variant_ev(e, cfg) for e in entries]
        row = {"statistic": name, "flat_weights": False}
        row.update(variant_row(entries, evs, tolerance_ev))
        rows.append(row)
    flat = m.MeterConfig(rule="metered", centre_sigma=None)
    row = {"statistic": "geomean", "flat_weights": True}
    row.update(variant_row(entries, [variant_ev(e, flat) for e in entries], tolerance_ev))
    rows.append(row)
    return rows


def meter_emulation(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Which statistic of the raw scene does the camera's own AE hold constant?

    A camera that meters statistic `S` and exposes it at a fixed fraction `r` of the white level, plus its exposure
    compensation, leaves `log2(0.18 / S) + EC` constant over scenes (EV 0 luminance, so a fixed gain from the raw
    level). The standard deviation of that quantity over the given files, in stops, says how well each statistic
    predicts the camera's decision; the smallest is the best emulation. The caller passes only the files the camera
    metered (a program mode, no flash).
    """
    rows = []
    for name in METER_STATS:
        vals = [math.log2(m.MID_GREY / e["stats_ev0"][name]) + e["ec_ev"] for e in entries]
        rows.append({"statistic": name, "mean_ev": float(np.mean(vals)) if vals else math.nan, "std_ev": float(np.std(vals, ddof=1)) if len(vals) > 1 else math.nan, "files": len(vals)})
    rows.sort(key=lambda r: (math.isnan(r["std_ev"]), r["std_ev"]))
    return rows


def contrast_variant(entry: Mapping[str, Any], gamma: float) -> float:
    """A closed-form stand-in for G3: the camera's tone curve taken as a power `gamma` about the 0.18 anchor, so that
    the JPEG's median `J` stands for the scene level `0.18 (J / 0.18)^(1/gamma)`; the camera gain is then
    `log2(0.18 / x_median) + log2(J / 0.18) / gamma` and the error is `EV_auto` minus it. `gamma = 1` is G1 exactly.
    Needs only the two medians, so a gate can compute it without the quantile matching."""
    x_median = entry["scene_median_at_auto"] / 2.0 ** entry["ev_auto"]
    j = entry["jpeg_median"]
    if not (x_median > 0 and j > 0 and gamma > 0):
        return math.nan
    return entry["ev_auto"] - (math.log2(m.MID_GREY / x_median) + math.log2(j / m.MID_GREY) / gamma)


def fit_contrast(entries: Sequence[Mapping[str, Any]], grid: Sequence[float] = tuple(1.0 + 0.05 * i for i in range(21))) -> dict[str, Any]:
    """The contrast `gamma` for which the closed-form variant agrees best (least squares) with the quantile-matched G3,
    and how well: the residual RMS next to that of plain G1."""
    g3 = np.array([e["g3_scene_ev"] for e in entries], dtype=np.float64)
    ok = np.isfinite(g3)
    best = (math.inf, math.nan)
    for gamma in grid:
        v = np.array([contrast_variant(e, gamma) for e in entries])
        rms = float(np.sqrt(np.mean((v[ok] - g3[ok]) ** 2))) if ok.any() else math.nan
        if rms < best[0]:
            best = (rms, gamma)
    return {"gamma": best[1], "rms_ev": best[0], "g1_rms_ev": float(np.sqrt(np.mean((np.array([contrast_variant(e, 1.0) for e in entries])[ok] - g3[ok]) ** 2))) if ok.any() else math.nan}


def contrast_sweep(entries: Sequence[Mapping[str, Any]], gammas: Sequence[float], tolerance_ev: float = 0.5) -> list[dict[str, Any]]:
    """Pass counts and spread of the closed-form variant for some contrasts."""
    rows = []
    g3 = np.array([e["g3_scene_ev"] for e in entries], dtype=np.float64)
    for gamma in gammas:
        v = [contrast_variant(e, gamma) for e in entries]
        within, evaluated = m.pass_rate(v, tolerance_ev)
        s = m.summary(v)
        ok = np.isfinite(g3) & np.isfinite(np.array(v))
        rows.append({"gamma": gamma, "median": s["median"], "std": s["std"], "within": within, "evaluated": evaluated, "max_abs_diff_from_g3": float(np.max(np.abs(np.array(v)[ok] - g3[ok]))) if ok.any() else math.nan})
    return rows

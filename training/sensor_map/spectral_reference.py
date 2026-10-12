"""Reference values for the Rust spectral engine (M1-02) and the interpolation study.

    python3 -m training.sensor_map.spectral_reference            # write the JSON
    python3 -m training.sensor_map.spectral_reference --study    # print the study for all 11 profiles

Writes ``crates/ym-color/tests/data/spectral_reference.json``: values computed straight from the
spectra with the ``ssf_core`` conventions (the "exact" fields), the values the documented table
interpolation gives from the shipped ``profile.json`` (the "interpolated" fields, which the Rust
runtime must reproduce to 1e-9 because it is the same arithmetic on the same numbers), and the
largest difference between the two on a dense off-grid sample (the "tolerances" block).

Everything in the file is a number derived from spectra; no SPD or other CIE dataset is stored.

Why off-grid values cannot match ``ssf_core`` to 1e-6
-----------------------------------------------------
A 10 K table interpolated linearly cannot reproduce the spectral value between grid points to 1e-6:

* Blackbody: the curvature of ``ln w`` in kelvin is large at low temperature. Linear in kelvin gives
  7e-5 at 1500 K; linear in mired on ``ln w`` (geometric interpolation of w, linear interpolation of
  log-chroma) is nearly exact because ln w is almost linear in mired (Wien), and brings the worst
  case to a few 1e-6 at 1500 K and below 1e-6 above ~2000 K.
* Daylight: colour-science builds the CIE D series with M1 and M2 rounded to three decimals (as the
  CIE recommends). The spectrum, and so ``w_A`` and the locus, is a staircase in T with steps of
  about 7e-5 (R) and 1.6e-4 (B) around a smooth curve. No interpolation of a 10 K table can follow it, and
  a finer grid does not help (the steps stay); the error is bounded by the step size, ~2e-4.

The tolerances block states what is achievable; the Rust test should use them for the
"exact" comparison off-grid and 1e-9 for the "interpolated" comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

import colour
import numpy as np

from training.alexa import ssf_core as s
from training.common import jsonfmt, paths
from training.sensor_map import fit
from training.sensor_map import lights as lt
from training.sensor_map import runtime_ref as rr
from training.sensor_map import tables as tb
from training.sensor_map.registry_data import PROFILES

REFERENCE_PROFILE = "sony-ilce-7rm4"
SEED = 20261012
OFF_GRID_BB_K = (1500.5, 1734.3, 2345.0, 2850.0, 3000.7, 3456.7, 3999.9)
OFF_GRID_D_K = (4000.1, 4321.5, 5003.0, 5250.0, 6504.0, 6750.0, 7000.5, 9876.0, 12345.6, 19999.0, 24999.5)
OFF_GRID_BB_ALSO_HIGH_K = (5003.0, 6504.0, 9876.0)  # blackbody w_A is tabulated up to 25000 K
GRID_POINTS_BB_K = (1500, 1510, 2000, 2500, 3000, 3500, 3990, 4000, 5000, 6500, 10000, 25000)
GRID_POINTS_D_K = (4000, 4010, 4500, 5000, 5500, 6000, 6500, 7000, 8000, 9000, 10000, 12000, 15000, 20000, 24990, 25000)
SAFETY = 2.0  # stated tolerance = measured worst case * SAFETY, rounded up to 1 significant digit


# ----------------------------------------------------------------------------
# Exact values from spectra
# ----------------------------------------------------------------------------
def spd(kind: str, t_k: float) -> np.ndarray:
    return lt.blackbody_spd(t_k) if kind == "blackbody" else lt.daylight_spd(t_k)


def exact_w_a(S_A: np.ndarray, kind: str, t_k: float) -> np.ndarray:
    return tb.white_normalised(tb.responses(spd(kind, t_k), S_A))


def exact_chroma(S_P: np.ndarray, kind: str, t_k: float) -> np.ndarray:
    return tb.log_chroma(tb.responses(spd(kind, t_k), S_P))


def exact_xy(kind: str, t_k: float) -> np.ndarray:
    return tb.xy_of(spd(kind, t_k)[None, :])[0]


def camera_rgb(S_P: np.ndarray, E: np.ndarray) -> np.ndarray:
    r = tb.responses(E, S_P)
    return r / r[1]


def true_cct_duv(E: np.ndarray) -> tuple[float, float]:
    """(Ohno 2013 CCT, Duv with the magenta-positive sign) of an SPD."""
    xy = tb.xy_of(E[None, :])[0]
    uv = colour.UCS_to_uv(colour.XYZ_to_UCS(colour.xy_to_XYZ(xy)))
    cct, duv_cie = colour.temperature.uv_to_CCT_Ohno2013(uv)
    return float(cct), float(-duv_cie)


# ----------------------------------------------------------------------------
# Study
# ----------------------------------------------------------------------------
def dense_temperatures(kind: str, n: int = 1500) -> np.ndarray:
    rng = np.random.default_rng(SEED + (0 if kind == "blackbody" else 1))
    lo, hi = (1500.0, 4000.0) if kind == "blackbody" else (4000.0, 25000.0)
    return np.sort(rng.uniform(lo, hi, n))


def interp_error_study(profile: rr.Profile, S_P: np.ndarray, S_A: np.ndarray, n: int = 1500) -> dict:
    """Largest |interpolated - exact| over dense random temperatures, per table and range.

    Also evaluates the alternative of interpolating linearly in kelvin, to show why mired is used.
    """
    out: dict = {}
    for kind in ("blackbody", "daylight"):
        ts = dense_temperatures(kind, n)
        spds = np.stack([spd(kind, t) for t in ts])
        ex_w = tb.white_normalised(tb.responses(spds, S_A))
        ex_c = tb.log_chroma(tb.responses(spds, S_P))
        gw = profile.w_a_blackbody if kind == "blackbody" else profile.w_a_daylight
        gc = profile.chroma_blackbody if kind == "blackbody" else profile.chroma_daylight
        ip_w = np.stack([gw.geom_mired(t) for t in ts])
        ip_c = np.stack([gc.lerp_mired(t) for t in ts])
        lin_w = np.stack([_lerp_kelvin(gw, t) for t in ts])
        lin_c = np.stack([_lerp_kelvin(gc, t) for t in ts])
        ranges = [("all", ts >= 0)]
        if kind == "blackbody":
            ranges += [("1500-2000 K", ts < 2000), ("2000-4000 K", ts >= 2000)]
        else:
            ranges += [("4000-11000 K", ts <= 11000), ("11000-25000 K", ts > 11000)]
        for name, sel in ranges:
            out[f"{kind}/{name}"] = {
                "w_a_mired_geometric": float(np.abs(ip_w - ex_w)[sel].max()),
                "w_a_kelvin_linear": float(np.abs(lin_w - ex_w)[sel].max()),
                "locus_mired_linear": float(np.abs(ip_c - ex_c)[sel].max()),
                "locus_kelvin_linear": float(np.abs(lin_c - ex_c)[sel].max()),
            }
    return out


def _lerp_kelvin(grid: rr.Grid, t_k: float) -> np.ndarray:
    t = min(max(float(t_k), grid.start_k), grid.end_k)
    i = min(max(int(np.floor((t - grid.start_k) / grid.step_k)), 0), grid.n - 2)
    a = (t - (grid.start_k + i * grid.step_k)) / grid.step_k
    return (1 - a) * grid.values[i] + a * grid.values[i + 1]


def _up(x: float) -> float:
    """Round up to one significant digit (1.3e-5 -> 2e-5)."""
    if x <= 0:
        return 0.0
    e = int(np.floor(np.log10(x)))
    m = int(np.ceil(round(x / 10.0**e, 9)))
    return float(f"{m}e{e}")


# ----------------------------------------------------------------------------
# JSON
# ----------------------------------------------------------------------------
def _f(a) -> list:
    return np.asarray(a, dtype=float).tolist()


def build(profile_path: Path | None = None) -> dict:
    paths.require_sources()
    profile_path = profile_path or (paths.PROFILES_DIR / REFERENCE_PROFILE / "profile.json")
    prof_doc = json.loads(profile_path.read_text())
    profile = rr.Profile.from_dict(prof_doc)
    prof_meta = next(p for p in PROFILES if p.id == prof_doc["id"])
    S_P = fit.sony_curves_raw()[prof_meta.curves_name]  # the camera's own raw scale: what the locus is built from
    S_A = fit.alexa_curves()

    # --- w_A and locus at grid points and off grid ---------------------------
    def w_case(kind, t, grid_point):
        exact = exact_w_a(S_A, kind, t)
        interp = (profile.w_a_blackbody if kind == "blackbody" else profile.w_a_daylight).geom_mired(t)
        return {"cct_k": t, "exact": _f(exact), "interpolated": _f(interp)}

    def c_case(kind, t):
        exact = exact_chroma(S_P, kind, t)
        grid = profile.chroma_blackbody if kind == "blackbody" else profile.chroma_daylight
        xyg = profile.xy_blackbody if kind == "blackbody" else profile.xy_daylight
        return {
            "cct_k": t,
            "ln_chroma_exact": _f(exact),
            "ln_chroma_interpolated": _f(grid.lerp_mired(t)),
            "xy_exact": _f(exact_xy(kind, t)),
            "xy_interpolated": _f(xyg.lerp_mired(t)),
        }

    w_ref = {
        "grid_points": {
            "blackbody": [w_case("blackbody", float(t), True) for t in GRID_POINTS_BB_K],
            "daylight": [w_case("daylight", float(t), True) for t in GRID_POINTS_D_K],
        },
        "off_grid": {
            "blackbody": [w_case("blackbody", t, False) for t in OFF_GRID_BB_K + OFF_GRID_BB_ALSO_HIGH_K],
            "daylight": [w_case("daylight", t, False) for t in OFF_GRID_D_K],
        },
    }
    c_ref = {
        "profile": prof_doc["id"],
        "grid_points": {
            "blackbody": [c_case("blackbody", float(t)) for t in GRID_POINTS_BB_K if t <= 4000],
            "daylight": [c_case("daylight", float(t)) for t in GRID_POINTS_D_K],
        },
        "off_grid": {
            "blackbody": [c_case("blackbody", t) for t in OFF_GRID_BB_K],
            "daylight": [c_case("daylight", t) for t in OFF_GRID_D_K],
        },
    }

    # --- dense study for the stated tolerances -------------------------------
    study = interp_error_study(profile, S_P, S_A)
    worst = lambda key, sel: max(study[k][key] for k in study if k.split("/")[0] == sel and k.endswith("/all"))
    tol = {
        "blackbody_w_a": _up(SAFETY * worst("w_a_mired_geometric", "blackbody")),
        "daylight_w_a": _up(SAFETY * worst("w_a_mired_geometric", "daylight")),
        "blackbody_locus": _up(SAFETY * worst("locus_mired_linear", "blackbody")),
        "daylight_locus": _up(SAFETY * worst("locus_mired_linear", "daylight")),
    }
    # blackbody above 2000 K is much tighter than the 1500-2000 K corner; state it separately
    tol["blackbody_w_a_above_2000k"] = _up(SAFETY * study["blackbody/2000-4000 K"]["w_a_mired_geometric"])
    tol["blackbody_locus_above_2000k"] = _up(SAFETY * study["blackbody/2000-4000 K"]["locus_mired_linear"])

    # --- CCT / Duv ------------------------------------------------------------
    cct_cases = _cct_duv_cases(profile, S_P)

    # --- sensor-map blend --------------------------------------------------------
    blend = _blend_cases(profile)

    # --- class lists of w_A -----------------------------------------------------
    class_w = []
    for klass in ("fluorescent", "led"):
        mireds = [e["mired"] for e in prof_doc["alexa_white"][klass]]
        for m in (140.0, mireds[0], 0.5 * (mireds[0] + mireds[1]), 245.0, 300.0, mireds[-1], 400.0):
            class_w.append({"class": klass, "mired": m, "w": _f(rr.interp_class_list(prof_doc["alexa_white"][klass], m, "w"))})

    return {
        "format": 1,
        "purpose": "Reference values for the Rust spectral engine (M1-02). Generated by "
                   "`python3 -m training.sensor_map.spectral_reference`; do not edit by hand.",
        "sources": {
            "profile": f"data/cameras/profiles/{prof_doc['id']}/profile.json",
            "profile_sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
            "alexa_curves": "data/alexa/alexa_alev3_estimate_380_780_5.json",
            "alexa_curves_sha256": hashlib.sha256(paths.ALEXA_CURVES_JSON.read_bytes()).hexdigest(),
            "colour_science": colour.__version__,
            "conventions": "training/alexa/ssf_core.py: 380-780 nm at 5 nm, blackbody below 4000 K, CIE D series "
                           "(M1, M2 rounded to 3 decimals) from 4000 K with CCT correction 1.4388/1.4380",
            "camera_curve_scale": "camera_locus, cct_duv: the camera's curves in the published raw scale (data/cameras/sony/*.json "
                                  "clipped at 0, not normalised per channel; fit.sony_curves_raw). alexa_w_a: the ALEXA "
                                  "estimate, peak-normalised by construction. camera_rgb values are therefore raw whites "
                                  "(R/G, 1, B/G), not peak-normalised ones (docs/02 C1).",
            "camera_curves_sha256": hashlib.sha256((paths.SONY_CURVES_DIR / prof_meta.curves_file).read_bytes()).hexdigest(),
        },
        "interpolation": {
            "grid": "values[i] is at T_i = start_k + i * step_k; queries are clamped to the table",
            "weight": "a = (1/T - 1/T_i) / (1/T_(i+1) - 1/T_i), T_i <= T <= T_(i+1)",
            "w_a": "geometric: w = w_i^(1-a) * w_(i+1)^a per channel (green stays 1)",
            "locus": "linear: p = (1-a) * p_i + a * p_(i+1) for log-chroma and for xy",
            "class_lists": "linear in mired between bracketing entries, clamped at the ends (w and M)",
            "table_choice": "blackbody tables below 4000 K, daylight tables from 4000 K",
            "cct_duv": "see runtime_ref.rgb_to_cct_duv: nearest point on each branch's chord polyline, "
                       "branch with the smaller distance (daylight on ties), CCT = 1e6 / mired of the foot point, "
                       "Duv = signed distance * duv_scale(CCT), positive toward magenta; inverse: cct_duv_to_rgb",
        },
        "tolerances": {
            "on_grid_abs": 1e-6,
            "interpolated_abs": 1e-9,
            "off_grid_vs_exact_abs": tol,
            "dense_study_max_abs": {k: {m: float(v) for m, v in d.items()} for k, d in study.items()},
            "note": "Off-grid, the table cannot follow the exact spectral values to 1e-6: low-temperature blackbody "
                    "curvature (1500-2000 K) and the 1e-4 staircase that colour-science's three-decimal M1/M2 "
                    "rounding puts into the CIE D series. Use the 'interpolated' fields (1e-9) to test the port "
                    "and 'off_grid_vs_exact_abs' to test the physics. A finer grid fixes the blackbody corner "
                    "but not the daylight staircase.",
        },
        "alexa_w_a": w_ref,
        "camera_locus": c_ref,
        "cct_duv": cct_cases,
        "sensor_map_blend": blend,
        "class_w_a": class_w,
    }


def _locus_light(cct_k: float) -> np.ndarray:
    """The scene-light model at ``cct_k`` (blackbody below 4000 K, D series from 4000 K)."""
    return lt.blackbody_spd(cct_k) if cct_k < 4000.0 else lt.daylight_spd(cct_k)


def _readout(profile: rr.Profile, S_P: np.ndarray, E: np.ndarray) -> dict:
    """Table readout of a light next to what CIE colorimetry says, relative to the same locus."""
    rgb = camera_rgb(S_P, E)
    cct, duv = rr.rgb_to_cct_duv(profile, rgb)
    cie_cct, cie_duv = true_cct_duv(E)
    _, locus_duv = true_cct_duv(_locus_light(cct))  # CIE Duv of the locus light the readout landed on
    return {
        "camera_rgb": rgb, "cct_k": cct, "duv": duv, "cie_cct_k": cie_cct,
        "cie_duv_magenta_positive": cie_duv,
        "cie_duv_vs_locus_magenta_positive": cie_duv - locus_duv,
    }


def duv_accuracy(profile: rr.Profile, S_P: np.ndarray) -> dict:
    """Error of the Duv readout on tinted lights the scale was not calibrated on.

    Bases: off-grid temperatures; tilts: three shapes x +-3 %, +-8 %, +-15 % (the calibration used one
    shape and +-5 %, +-10 %). Reports the error of ``duv`` against CIE Duv measured from the same locus
    light, in absolute CIE Duv units and relative to the CIE value.
    """
    abs_err, rel_err, cct_err = [], [], []
    for kind, t in (("blackbody", 2345.0), ("blackbody", 3456.7), ("daylight", 5003.0),
                    ("daylight", 6504.0), ("daylight", 9876.0)):
        base = spd(kind, t)
        for centre, width in ((545.0, 45.0), (520.0, 60.0), (570.0, 40.0)):
            for a in (-0.15, -0.08, -0.03, 0.03, 0.08, 0.15):
                r = _readout(profile, S_P, lt.tint_spd(base, a, centre, width))
                err = r["duv"] - r["cie_duv_vs_locus_magenta_positive"]
                abs_err.append(abs(err))
                rel_err.append(abs(err) / max(abs(r["cie_duv_vs_locus_magenta_positive"]), 1e-9))
                cct_err.append(abs(r["cct_k"] - t))
    return {
        "lights": len(abs_err),
        "duv_abs_error_max": float(max(abs_err)),
        "duv_abs_error_median": float(np.median(abs_err)),
        "duv_rel_error_median": float(np.median(rel_err)),
        "duv_rel_error_p90": float(np.percentile(rel_err, 90)),
        "cct_shift_k_max": float(max(cct_err)),
        "note": "tinted lights (smooth green excess / deficit); CCT shift is how far the camera-space foot point "
                "moves from the generating temperature, not an error: the tilt changes the light",
    }


def _cct_duv_cases(profile: rr.Profile, S_P: np.ndarray) -> dict:
    cases = []

    def add(name, E, generating_cct_k=None):
        r = _readout(profile, S_P, E)
        cases.append({
            "name": name,
            "camera_rgb": _f(r["camera_rgb"]),
            "cct_k": r["cct_k"],
            "duv": r["duv"],
            "generating_cct_k": generating_cct_k,
            "cie_cct_k": r["cie_cct_k"],
            "cie_duv_magenta_positive": r["cie_duv_magenta_positive"],
            "cie_duv_vs_locus_magenta_positive": r["cie_duv_vs_locus_magenta_positive"],
        })

    for t in (2345.0, 3456.7):
        add(f"blackbody {t:g} K", lt.blackbody_spd(t), t)
    for t in (5003.0, 6504.0, 9876.0):
        add(f"CIE D series {t:g} K", lt.daylight_spd(t), t)
    add("CIE illuminant A", s.named_illuminant("A"))
    add("CIE illuminant D65", s.named_illuminant("D65"))
    add("CIE FL2", s.named_illuminant("FL2"))
    add("CIE LED-B3", s.named_illuminant("LED-B3"))
    for amount, t, kind in ((0.15, 3456.7, "blackbody"), (-0.15, 3456.7, "blackbody"),
                            (0.10, 6504.0, "daylight"), (-0.10, 6504.0, "daylight")):
        base = spd(kind, t)
        add(f"{kind} {t:g} K, green {'excess' if amount > 0 else 'deficit'} {abs(amount):g}", lt.tint_spd(base, amount), t)

    inverse = []
    for cct, duv in ((2345.0, 0.0), (3456.7, 0.004), (3456.7, -0.004), (5003.0, 0.0), (6504.0, 0.003),
                     (6504.0, -0.003), (9876.0, 0.0), (1600.0, 0.002), (20000.0, 0.0)):
        rgb = rr.cct_duv_to_rgb(profile, cct, duv)
        cct2, duv2 = rr.rgb_to_cct_duv(profile, rgb)
        inverse.append({"cct_k": cct, "duv": duv, "camera_rgb": _f(rgb),
                        "roundtrip_cct_k": cct2, "roundtrip_duv": duv2})
    return {"cases": cases, "inverse_cases": inverse, "duv_accuracy": duv_accuracy(profile, S_P)}


def _blend_cases(profile: rr.Profile) -> list[dict]:
    out = []
    for probs, mired in (
        ({"tungsten": 1.0}, 350.877),
        ({"daylight": 1.0}, 184.0),
        ({"daylight": 1.0}, 80.0),                       # clamp at the daylight end
        ({"fluorescent": 1.0}, 300.0),
        ({"led": 1.0}, 330.0),
        ({"daylight": 0.5, "tungsten": 0.25, "led": 0.25}, 240.0),
        ({"fluorescent": 0.7, "led": 0.3}, 600.0),       # clamp at the long end
    ):
        m = rr.sensor_map(profile, probs, mired)
        out.append({"probs": probs, "mired": mired, "m": _f(m), "row_sums": _f(m.sum(axis=1))})
    return out


# ----------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", action="store_true", help="print the interpolation study for all 11 profiles")
    ap.add_argument("--out", type=Path, default=paths.SPECTRAL_REFERENCE_JSON)
    args = ap.parse_args(argv)
    paths.require_sources()
    warnings.simplefilter("ignore", category=colour.utilities.ColourUsageWarning)

    if args.study:
        S_A = fit.alexa_curves()
        for prof_meta in PROFILES:
            doc = json.loads((paths.PROFILES_DIR / prof_meta.id / "profile.json").read_text())
            study = interp_error_study(rr.Profile.from_dict(doc), fit.sony_curves_raw()[prof_meta.curves_name], S_A)
            print(prof_meta.id)
            for k, v in study.items():
                print(f"  {k:<22} " + "  ".join(f"{m}={x:.2e}" for m, x in v.items()))
        return 0

    ref = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = jsonfmt.dumps(ref)
    args.out.write_text(text)
    print(f"wrote {args.out.relative_to(paths.REPO_ROOT)} ({len(text) / 1024:.0f} kB)")
    print("stated tolerances:", json.dumps(ref["tolerances"]["off_grid_vs_exact_abs"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

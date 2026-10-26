"""Canon and Nikon curves in rawtoaces-data, measured against what Yellowmoon asks of a profile.

Research for ``research/canon-nikon-curves-2026-10-19.md``. Nothing here feeds a shipped profile.
Inputs: the pinned rawtoaces-data (``tools/fetch/spectral.sh``: 22 Canon and 11 Nikon curves next to
the 11 Sony ones), ``colour_matrices.json`` here (the published D65 colour matrices of Canon and
Nikon bodies, numbers only) and ``training/white_scale/colour_matrices.json`` (the same for the
Sony registry bodies). Three checks:

``emulation``  The measure of ``docs/03`` "Why measured profiles matter" (``sony_profiles.csv``):
               ΔE00 between the ALEXA rendering and a body's rendering through a per-light 3x3,
               median / p95 over 138 colours under 9 lights, with the map fitted on the body's own
               curves, on the A7R IV's (today's default) and on every other measured curve
               (``training.sensor_map.evaluate.map_error``, the research objective).
``matrix``     How each measured curve agrees with its body's published matrix: the D65 white
               (R/G and B/G) of the curves in their published scale against the matrix's, and the
               ΔE00 under D65 of the 24 ColorChecker and 99 CES colours when the published matrix
               turns the curves' white-balanced camera RGB into XYZ, next to the same error with the
               curves' own best white-preserving 3x3 (fitted in Lab on those colours, so a floor).
               A curve that describes its camera well is rendered by the camera's matrix about as
               well as the Sony curves are by theirs.
``siblings``   The same-sensor check of ``docs/03`` (rows normalised to sum 1, largest
               per-coefficient difference, limit 0.02) for every Canon and Nikon body of
               ``colour_matrices.json`` against the measured bodies of its make.
``gate``       The M1 sensor-map gate of ``docs/13`` (``training.sensor_map.gate``: Lab-optimal
               tables refitted without the three gate LEDs, 99 CES under the 9 held-out lights,
               per light median <= 0.5 and p95 <= 1.2) run on every curve as if it were a profile,
               and ``training.sensor_map.gate_diagnostics`` for the headroom: the Lab 3x3 and a
               second-order root-polynomial fitted at each gate light itself (no table blend).

    python3 -m training.other_makes.check                 # writes research/canon-nikon-curves/*.csv
    python3 -m training.other_makes.check --skip-emulation
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from training.alexa import ssf_core as s
from training.common import paths
from training.sensor_map import evaluate as ev
from training.sensor_map import fit
from training.sensor_map import gate as gt
from training.sensor_map import gate_diagnostics as gd
from training.sensor_map import lights as lt

HERE = Path(__file__).resolve().parent
MATRICES_JSON = HERE / "colour_matrices.json"
SONY_MATRICES_JSON = paths.REPO_ROOT / "training" / "white_scale" / "colour_matrices.json"
OUT_DIR = paths.REPO_ROOT / "research" / "canon-nikon-curves"
MAKES = ("Canon", "Nikon", "Sony")
SIBLING_LIMIT = 0.02  # docs/03

# Curve name ("Make Model" of the rawtoaces-data header) -> body name in the matrix files, where they differ.
MATRIX_NAME = {
    "Canon Digital Rebel XTi": "EOS 400D",
    "Canon EOS 200D II": "EOS 250D",
    "Canon EOS R5m2": "EOS R5 Mark II",
    "Canon EOS R6m2": "EOS R6 Mark II",
}


# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------
def curves_raw() -> dict[str, np.ndarray]:
    """Every Canon, Nikon and Sony curve of rawtoaces-data in its published scale (clipped at 0)."""
    out = {}
    for path in sorted(paths.CAMERA_CURVES_DIR.glob("*_380_780_5.json")):
        hdr = json.loads(path.read_text())["header"]
        if hdr["manufacturer"] in MAKES:
            out[f"{hdr['manufacturer']} {hdr['model']}"] = fit.load_curves_raw(path)
    return out


def curve_headers() -> dict[str, dict]:
    """The header of every Canon, Nikon and Sony curve file (creator, lab, equipment, date)."""
    out = {}
    for path in sorted(paths.CAMERA_CURVES_DIR.glob("*_380_780_5.json")):
        hdr = json.loads(path.read_text())["header"]
        if hdr["manufacturer"] in MAKES:
            out[f"{hdr['manufacturer']} {hdr['model']}"] = hdr
    return out


def matrices() -> dict[str, np.ndarray]:
    """``{"Canon EOS R5": XYZ -> camera (3, 3), ...}``: Canon and Nikon bodies (with ``also``), Sony registry bodies."""
    out = {}
    doc = json.loads(MATRICES_JSON.read_text())
    for make, bodies in doc["bodies"].items():
        for body, rec in bodies.items():
            m = np.array(rec["xyz_to_camera"], dtype=float)
            for name in [body, *rec.get("also", [])]:
                out[f"{make} {name}"] = m
    for rec in json.loads(SONY_MATRICES_JSON.read_text())["bodies"].values():
        out[f"Sony {rec['model']}"] = np.array(rec["xyz_to_camera"], dtype=float)
    return out


def matrix_name(curve: str) -> str:
    """The body name in :func:`matrices` of a curve."""
    make = curve.split(" ", 1)[0]
    return f"{make} {MATRIX_NAME[curve]}" if curve in MATRIX_NAME else curve


def make_of(name: str) -> str:
    return name.split(" ", 1)[0]


# ----------------------------------------------------------------------------
# 1. Emulation error
# ----------------------------------------------------------------------------
def emulation_pairs(names: list[str]) -> dict[tuple[str, str], tuple[float, float, str, float]]:
    """``(target, stand-in) -> (median, p95, worst light, its median)`` for every pair of ``names``."""
    ctx = ev.context()
    cams = s.load_cameras()
    return {(t, y): ev.summary(ev.map_error(cams[t], cams[y], ctx)) for t in names for y in names}


def weta_group(names: list[str], make: str) -> list[str]:
    """The Weta-measured curves of ``make`` (the recent bodies; the 2016-17 legacy files left out)."""
    hdrs = curve_headers()
    return [n for n in names if make_of(n) == make and hdrs[n].get("document_creator") == "wetadigital"]


def make_default(pairs: dict, names: list[str], make: str) -> tuple[str, float, float]:
    """The Weta-measured body of ``make`` whose map serves the other Weta bodies of that make best.

    Returns the body, the median over the other bodies of their median ΔE00 through its map, and the
    same median through the A7R IV's map (today's default), for comparison.
    """
    group = weta_group(names, make)
    score = {y: float(np.median([pairs[(t, y)][0] for t in group if t != y])) for y in group}
    best = min(score, key=score.get)
    return best, score[best], float(np.median([pairs[(t, ev.REF_CAMERA)][0] for t in group]))


def emulation_rows(pairs: dict, names: list[str]) -> tuple[list[dict], dict[str, tuple[str, float, float]]]:
    cams = s.load_cameras()
    defaults = {mk: make_default(pairs, names, mk) for mk in ("Canon", "Nikon")}
    rows = []
    for t in names:
        mk = make_of(t)
        same = [y for y in names if make_of(y) == mk and y != t]
        best_same = min(same, key=lambda y: pairs[(t, y)][0])
        row = {
            "camera": t,
            "own_median": pairs[(t, t)][0], "own_p95": pairs[(t, t)][1],
            "7RM4_median": pairs[(t, ev.REF_CAMERA)][0], "7RM4_p95": pairs[(t, ev.REF_CAMERA)][1],
            "7RM4_worst_light": pairs[(t, ev.REF_CAMERA)][2],
            "rfse_vs_7RM4_%": s.rfse(cams[ev.REF_CAMERA], cams[t], band=(400, 700)),
            "rfse_vs_alexa_%": s.rfse(fit.alexa_curves(), cams[t], band=(400, 700)),
            "best_same_make": best_same,
            "best_same_make_median": pairs[(t, best_same)][0], "best_same_make_p95": pairs[(t, best_same)][1],
            "rfse_vs_best_same_make_%": s.rfse(cams[best_same], cams[t], band=(400, 700)),
        }
        if mk in defaults:
            d = defaults[mk][0]
            row["make_default"] = d
            row["make_default_median"] = pairs[(t, d)][0]
            row["make_default_p95"] = pairs[(t, d)][1]
        rows.append(row)
    return rows, defaults


# ----------------------------------------------------------------------------
# 2. Curves against the published matrix
# ----------------------------------------------------------------------------
def _lab(XYZ: np.ndarray, white: np.ndarray) -> np.ndarray:
    r = XYZ / white
    f = np.where(r > (6 / 29) ** 3, np.cbrt(r), r / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def matrix_check(S_raw: np.ndarray, M: np.ndarray, per_channel_normalised: bool) -> dict:
    """The D65 white and the colour error of the published matrix ``M`` (XYZ -> camera) on curves ``S_raw``."""
    E = s.named_illuminant("D65")
    sets = s.test_reflectances()
    R = np.concatenate([sets["ColorChecker (Ohta, 24)"], sets["CIE 2017 CES (99)"]], axis=1)
    norm = s.CMF[:, 1] @ E
    XYZ_w = (s.CMF.T @ E) / norm
    XYZ_t = ((E[:, None] * R).T @ s.CMF) / norm             # (N, 3)
    lab_t = _lab(XYZ_t, XYZ_w)
    rgb_n = s.camera_rgb_wb(S_raw, E, R)                     # (N, 3), white -> 1

    cam_w = M @ XYZ_w                                        # the matrix's raw white of D65
    A_pub = np.linalg.inv(M) @ np.diag(cam_w)                # balanced camera RGB -> XYZ, white preserving
    de_pub = s.delta_E00(_lab(rgb_n @ A_pub.T, XYZ_w), lab_t)

    def build(p):
        A = np.empty((3, 3))
        A[:, :2] = p.reshape(3, 2)
        A[:, 2] = XYZ_w - A[:, 0] - A[:, 1]
        return A

    def res(p):
        return (_lab(rgb_n @ build(p).T, XYZ_w) - lab_t).ravel()

    A_own = build(least_squares(res, A_pub[:, :2].ravel(), method="lm").x)
    de_own = s.delta_E00(_lab(rgb_n @ A_own.T, XYZ_w), lab_t)

    out = {
        "pub_matrix_dE_median": float(np.median(de_pub)), "pub_matrix_dE_p95": float(np.percentile(de_pub, 95)),
        "own_3x3_dE_median": float(np.median(de_own)), "own_3x3_dE_p95": float(np.percentile(de_own, 95)),
        "white_R/G_diff_%": float("nan"), "white_B/G_diff_%": float("nan"),
    }
    if not per_channel_normalised:
        w_curve = (E @ S_raw) / (E @ S_raw)[1]
        w_mat = cam_w / cam_w[1]
        out["white_R/G_diff_%"] = float(100 * (w_curve[0] / w_mat[0] - 1))
        out["white_B/G_diff_%"] = float(100 * (w_curve[2] / w_mat[2] - 1))
    return out


def matrix_rows() -> list[dict]:
    raw, hdrs, mats = curves_raw(), curve_headers(), matrices()
    rows = []
    for name, S_raw in raw.items():
        m = mats.get(matrix_name(name))
        hdr = hdrs[name]
        row = {"camera": name, "creator": hdr.get("document_creator"), "matrix_body": matrix_name(name) if m is not None else ""}
        if m is not None:
            # A file whose three channels all peak at exactly 1 lost the channel balance: no white check.
            per_channel = bool(np.allclose(S_raw.max(axis=0), 1.0))
            row.update(matrix_check(S_raw, m, per_channel))
        rows.append(row)
    return rows


# ----------------------------------------------------------------------------
# 3. Same-sensor candidates
# ----------------------------------------------------------------------------
def row_normalised(m: np.ndarray) -> np.ndarray:
    return m / m.sum(axis=1, keepdims=True)


def sibling_rows() -> list[dict]:
    mats = matrices()
    measured = {matrix_name(c): c for c in curves_raw()}
    rows = []
    for body, m in mats.items():
        mk = make_of(body)
        if mk not in ("Canon", "Nikon"):
            continue
        cands = [b for b in measured if make_of(b) == mk and b != body]
        diffs = {b: float(np.abs(row_normalised(m) - row_normalised(mats[b])).max()) for b in cands}
        nearest = min(diffs, key=diffs.get)
        rows.append({
            "body": body,
            "measured": measured.get(body, ""),
            "nearest_measured": measured[nearest],
            "max_abs_diff": diffs[nearest],
            "within_limit": diffs[nearest] <= SIBLING_LIMIT,
        })
    return sorted(rows, key=lambda r: r["body"])


def sibling_calibration_rows(pairs: dict, names: list[str]) -> list[dict]:
    """Every same-make pair of measured bodies: matrix difference against curve difference.

    ``extra_median`` is what borrowing costs, averaged over both directions: the median ΔE00 to the
    ALEXA rendering through the other body's map minus the median through its own.
    """
    mats, cams = matrices(), s.load_cameras()
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if make_of(a) != make_of(b) or matrix_name(a) not in mats or matrix_name(b) not in mats:
                continue
            rows.append({
                "make": make_of(a), "a": a, "b": b,
                "matrix_max_abs_diff": float(np.abs(row_normalised(mats[matrix_name(a)]) - row_normalised(mats[matrix_name(b)])).max()),
                "rfse_%": s.rfse(cams[a], cams[b], band=(400, 700)),
                "extra_median": 0.5 * (pairs[(a, b)][0] - pairs[(a, a)][0] + pairs[(b, a)][0] - pairs[(b, b)][0]),
            })
    return rows


# ----------------------------------------------------------------------------
# 4. The sensor-map gate
# ----------------------------------------------------------------------------
def gate_curves(S_P: np.ndarray, ctx: ev.EvalContext) -> dict:
    """``training.sensor_map.gate.run_profile``'s gate column for curves that are not a shipped profile."""
    tables = fit.fit_tables(S_P, exclude=lt.GATE_LED)
    per_light = {}
    for L in lt.gate_lights():
        M = fit.blend_map(tables, {L.klass: 1.0}, L.mired)
        per_light[L.name] = {"gate": gt._stats(ev.render_error_with_matrix(S_P, M, L.spd, L.cct_k, ctx.r_ces, ctx))}
    return per_light


def gate_rows(names: list[str]) -> list[dict]:
    ctx = ev.context()
    cams = s.load_cameras()
    rows = []
    for name in names:
        per_light = gate_curves(cams[name], ctx)
        worst_med = max(per_light, key=lambda k: per_light[k]["gate"]["median"])
        worst_p95 = max(per_light, key=lambda k: per_light[k]["gate"]["p95"])
        over = gt.gate_status(per_light)
        rows.append({
            "camera": name,
            "worst_median": per_light[worst_med]["gate"]["median"], "worst_median_light": worst_med,
            "worst_p95": per_light[worst_p95]["gate"]["p95"], "worst_p95_light": worst_p95,
            "lights_over": " ".join(over),
            "pass_except_LED-V2": not [n for n in over if n != "LED-V2"],
            **{f"{n}_{k}": per_light[n]["gate"][k] for n in per_light for k in ("median", "p95")},
        })
    return rows


def headroom_rows(names: list[str]) -> list[dict]:
    """Worst light over the 9 gate lights of the 3x3 and the root-polynomial fitted at each light itself."""
    cams = s.load_cameras()
    rows = []
    for name in names:
        diag = gd.diagnose_curves(cams[name])
        row = {"camera": name}
        for fam in ("lab", "rootpoly"):
            row[f"{fam}_worst_median"] = max(r[fam][0] for r in diag)
            row[f"{fam}_worst_p95"] = max(r[fam][1] for r in diag)
            row[f"{fam}_lights_over"] = " ".join(
                r["light"] for r in diag if not (r[fam][0] <= gt.MEDIAN_MAX and r[fam][1] <= gt.P95_MAX))
        rows.append(row)
    return rows


# ----------------------------------------------------------------------------
def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print("wrote", path.relative_to(paths.REPO_ROOT))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--skip-emulation", action="store_true", help="only the matrix and sibling checks (seconds, not minutes)")
    args = ap.parse_args(argv)
    paths.require_sources()

    _write(args.out / "matrix.csv", matrix_rows())
    _write(args.out / "siblings.csv", sibling_rows())
    names = sorted(curves_raw())
    _write(args.out / "gate.csv", gate_rows(names))
    _write(args.out / "headroom.csv", headroom_rows(names))
    if not args.skip_emulation:
        pairs = emulation_pairs(names)
        rows, defaults = emulation_rows(pairs, names)
        _write(args.out / "emulation.csv", rows)
        _write(args.out / "emulation_pairs.csv", [
            {"target": t, **{y: pairs[(t, y)][0] for y in names}} for t in names
        ])
        _write(args.out / "sibling_calibration.csv", sibling_calibration_rows(pairs, names))
        _write(args.out / "make_defaults.csv", [
            {"make": mk, "default": d, "median_of_medians": med, "7RM4_median_of_medians": ref,
             "own_median_of_medians": float(np.median([pairs[(t, t)][0] for t in weta_group(names, mk)]))}
            for mk, (d, med, ref) in defaults.items()
        ])
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Evaluation of the sensor map: port of ``research/alexa-ssf/scripts/sony_profiles.py``.

"Own" and "A7R IV" numbers of ``docs/03`` and ``docs/13`` ("Camera profiles"): the ΔE00 between
the ALEXA rendering of a scene and the rendering of the same scene from a Sony body through a
per-light 3x3, median / p95 over 138 colours (CIE 99 CES, ColorChecker (Ohta) 24, NIST CQS
VS 15) under 9 lights. The 3x3 is fitted at the test light itself on the 190 training
reflectances, with the body's own curves ("own") or the A7R IV curves (stand-in).

Both images go through ARRI's matrix at the light's CCT (``ssf_core.interpolate_matrix``).

    python3 -m training.sensor_map.evaluate           # reproduces research/.../sony_profiles.csv

Also holds the pieces the gate reuses: :func:`render_error_with_matrix`.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from training.alexa import ssf_core as s
from training.common import paths
from training.sensor_map import fit

REF_CAMERA = "Sony ILCE-7RM4"
RESEARCH_CSV = paths.REPO_ROOT / "research" / "alexa-ssf" / "out" / "sony_profiles.csv"


@dataclass(frozen=True)
class EvalContext:
    """Everything the evaluation needs besides the cameras."""

    alexa: np.ndarray          # S_A (81, 3)
    arri_ccts: np.ndarray      # (17,)
    arri_mats: np.ndarray      # (17, 3, 3) native -> AP0
    r_train: np.ndarray        # (81, 190)
    r_test: np.ndarray         # (81, 138): CES 99 + ColorChecker 24 + NIST CQS VS 15
    r_ces: np.ndarray          # (81, 99)
    illums: dict[str, np.ndarray]  # the 9 research test lights


@lru_cache(maxsize=1)
def context() -> EvalContext:
    paths.require_sources()
    sets = s.test_reflectances()
    ccts, mats = s.load_arri_matrices()
    return EvalContext(
        alexa=fit.alexa_curves(),
        arri_ccts=ccts,
        arri_mats=mats,
        r_train=fit.training_reflectances(),
        r_test=np.concatenate([v for v in sets.values()], axis=1),
        r_ces=sets["CIE 2017 CES (99)"],
        illums=s.test_illuminants(),
    )


def render_error_with_matrix(
    S_true: np.ndarray,
    M: np.ndarray,
    E: np.ndarray,
    cct_k: float,
    R: np.ndarray,
    ctx: EvalContext,
) -> np.ndarray:
    """ΔE00 per reflectance between the ALEXA rendering and the Sony rendering through ``M``.

    ALEXA: ``S_A`` white-balanced RGB of ``R`` under ``E``. Sony: ``S_true`` white-balanced RGB
    mapped by ``M``. Both are taken through ARRI's matrix at ``cct_k`` to AP0, then to Lab
    with the ACES white (``ssf_core.ap0_to_Lab``).
    """
    M_arri = s.interpolate_matrix(ctx.arri_ccts, ctx.arri_mats, cct_k)
    target = s.ap0_to_Lab(s.camera_rgb_wb(ctx.alexa, E, R) @ M_arri.T)
    mapped = s.camera_rgb_wb(S_true, E, R) @ M.T
    return s.delta_E00(s.ap0_to_Lab(mapped @ M_arri.T), target)


def map_error(S_true: np.ndarray, S_fit: np.ndarray, ctx: EvalContext) -> dict[str, np.ndarray]:
    """``sony_profiles.map_error``: per light, X's RGB through a map fitted on ``S_fit`` at that light."""
    res = {}
    for name, E in ctx.illums.items():
        M = fit.fit_map(S_fit, ctx.alexa, E, ctx.r_train)
        res[name] = render_error_with_matrix(S_true, M, E, s.illuminant_cct(E), ctx.r_test, ctx)
    return res


def summary(res: dict[str, np.ndarray]) -> tuple[float, float, str, float]:
    """(median, p95) over all lights and colours, and the light with the worst median."""
    allv = np.concatenate(list(res.values()))
    worst = max(res, key=lambda k: np.median(res[k]))
    return float(np.median(allv)), float(np.percentile(allv, 95)), worst, float(np.median(res[worst]))


def sony_profiles_rows(ctx: EvalContext | None = None) -> list[dict]:
    """The rows of ``sony_profiles.csv`` (without the estimated-curve columns), 11 Sony bodies."""
    ctx = context() if ctx is None else ctx
    cams = s.load_cameras()
    sony = sorted(k for k in cams if k.startswith("Sony ") and k in fit.sony_curves())
    rows = []
    for name in sony:
        S = cams[name]
        own = summary(map_error(S, S, ctx))
        ref = summary(map_error(S, cams[REF_CAMERA], ctx))
        rows.append({
            "camera": name,
            "rfse_vs_7RM4_%": round(s.rfse(cams[REF_CAMERA], S, band=(400, 700)), 2),
            "own_median": round(own[0], 3), "own_p95": round(own[1], 3),
            "7RM4_median": round(ref[0], 3), "7RM4_p95": round(ref[1], 3),
            "7RM4_worst_light": ref[2], "7RM4_worst_median": round(ref[3], 3),
        })
    return rows


NUMERIC = ("rfse_vs_7RM4_%", "own_median", "own_p95", "7RM4_median", "7RM4_p95", "7RM4_worst_median")


def compare_rows(rows: list[dict], research_csv: Path = RESEARCH_CSV) -> dict:
    """Differences against the published CSV: per column the largest absolute difference."""
    with open(research_csv, newline="") as fh:
        ref = {r["camera"]: r for r in csv.DictReader(fh)}
    out = {"max_abs_diff": {c: 0.0 for c in NUMERIC}, "worst_light_mismatches": [], "cameras": len(rows)}
    for row in rows:
        r = ref[row["camera"]]
        for c in NUMERIC:
            out["max_abs_diff"][c] = max(out["max_abs_diff"][c], abs(float(r[c]) - row[c]))
        if r["7RM4_worst_light"] != row["7RM4_worst_light"]:
            out["worst_light_mismatches"].append((row["camera"], r["7RM4_worst_light"], row["7RM4_worst_light"]))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=paths.OUT / "sensor_map" / "sony_profiles.csv")
    ap.add_argument("--tol", type=float, default=0.02, help="allowed |difference| (docs/13: 0.02)")
    args = ap.parse_args(argv)

    rows = sony_profiles_rows()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{'camera':<18} {'own med/p95':>14} {'A7R IV med/p95':>16}")
    for r in rows:
        print(f"{r['camera']:<18} {r['own_median']:>6.3f}/{r['own_p95']:<7.3f} {r['7RM4_median']:>7.3f}/{r['7RM4_p95']:<7.3f}")
    cmp = compare_rows(rows)
    print("max |ours - research sony_profiles.csv| per column:")
    print(json.dumps(cmp["max_abs_diff"], indent=1))
    if cmp["worst_light_mismatches"]:
        print("worst-light mismatches:", cmp["worst_light_mismatches"])
    ok = max(cmp["max_abs_diff"].values()) <= args.tol and not cmp["worst_light_mismatches"]
    print("wrote", args.out)
    print("PASS" if ok else "FAIL", f"(tolerance {args.tol})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

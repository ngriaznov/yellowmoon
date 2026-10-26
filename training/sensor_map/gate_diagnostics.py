"""Why the sensor-map gate numbers are what they are: capacity of the map families, per gate light.

    python3 -m training.sensor_map.gate_diagnostics [--profile sony-ilce-7rm4]

For each gate light (docs/02 §C2), fitted on the 190 training reflectances **at the gate light
itself** (so no table interpolation is involved) and scored on the 99 CES, ΔE00 median / p95 through
ARRI's matrix at the light's CCT:

``lab``          the shipped fit since the decision of 2026-10-12: white-preserving 3x3 minimising the CIE
                 Lab error after ARRI's matrix (``fit.fit_map_lab``)
``weighted_rgb`` the superseded M1-02 fit: white-preserving 3x3, weights Y^(-2/3), linear least squares
                 (``fit.fit_map_weighted_rgb``; the start of the Lab fit)
``free3x3``      unconstrained 3x3, same weights (does NOT keep white white; reference only)
``rootpoly``     unconstrained second-order root-polynomial (6 terms), same weights (reference only)

Only ``lab`` is what the product ships. The other columns say how much headroom another objective or a
non-linear map would have, and whether the gate threshold (median <= 0.5, p95 <= 1.2) is reachable at
all for a light. Nothing here changes the gate.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from training.alexa import ssf_core as s
from training.sensor_map import evaluate as ev
from training.sensor_map import fit
from training.sensor_map import lights as lt
from training.sensor_map.registry_data import PROFILES


def _root_poly(x: np.ndarray) -> np.ndarray:
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    return np.stack([r, g, b, np.sqrt(np.abs(r * g)), np.sqrt(np.abs(g * b)), np.sqrt(np.abs(r * b))], axis=1)


def diagnose(profile_id: str) -> list[dict]:
    prof = next(p for p in PROFILES if p.id == profile_id)
    return diagnose_curves(fit.sony_curves()[prof.curves_name])


def diagnose_curves(S: np.ndarray) -> list[dict]:
    """:func:`diagnose` for any peak-normalised curves ``S`` (81, 3), not only a shipped profile's."""
    ctx = ev.context()
    rows = []
    for L in lt.gate_lights():
        E = L.spd
        src_tr = s.camera_rgb_wb(S, E, ctx.r_train)
        dst_tr = s.camera_rgb_wb(ctx.alexa, E, ctx.r_train)  # for the reference families
        src_te = s.camera_rgb_wb(S, E, ctx.r_ces)
        dst_te = s.camera_rgb_wb(ctx.alexa, E, ctx.r_ces)
        M_arri = s.interpolate_matrix(ctx.arri_ccts, ctx.arri_mats, L.cct_k)
        lab_te = s.ap0_to_Lab(dst_te @ M_arri.T)
        w = np.clip(s.target_XYZ(E, ctx.r_train)[:, 1], 1e-3, None) ** (-2 / 3)

        def score(pred_rgb: np.ndarray) -> tuple[float, float]:
            d = s.delta_E00(s.ap0_to_Lab(pred_rgb @ M_arri.T), lab_te)
            return float(np.median(d)), float(np.percentile(d, 95))

        M_weighted = fit.fit_map_weighted_rgb(S, ctx.alexa, E, ctx.r_train)
        M_lab = fit.fit_map_lab(S, ctx.alexa, E, L.cct_k, ctx.r_train)
        P3, *_ = np.linalg.lstsq(src_tr * w[:, None], dst_tr * w[:, None], rcond=None)
        Pr, *_ = np.linalg.lstsq(_root_poly(src_tr) * w[:, None], dst_tr * w[:, None], rcond=None)
        rows.append({
            "light": L.name,
            "lab": score(src_te @ M_lab.T),
            "weighted_rgb": score(src_te @ M_weighted.T),
            "free3x3": score(src_te @ P3),
            "rootpoly": score(_root_poly(src_te) @ Pr),
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="sony-ilce-7rm4")
    args = ap.parse_args(argv)
    rows = diagnose(args.profile)
    print(f"{args.profile}: fitted at the gate light itself, median/p95 dE00 on the 99 CES (gate: 0.5 / 1.2)")
    print(f"  {'light':<18} {'lab (shipped)':>13} {'weighted_rgb':>13} {'free3x3':>13} {'rootpoly':>13}")
    for r in rows:
        print(f"  {r['light']:<18} " + " ".join(f"{r[k][0]:5.3f}/{r[k][1]:5.3f}  " for k in ("lab", "weighted_rgb", "free3x3", "rootpoly")))
    return 0


if __name__ == "__main__":
    sys.exit(main())

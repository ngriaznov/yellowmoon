"""Why the sensor-map gate numbers are what they are: capacity of the map families, per gate light.

    python3 -m training.sensor_map.gate_diagnostics [--profile sony-ilce-7rm4]

For each gate light (docs/02 §C2), fitted on the 190 training reflectances **at the gate light
itself** (so no table interpolation is involved) and scored on the 99 CES, ΔE00 median / p95 through
ARRI's matrix at the light's CCT:

``spec``        the shipped fit: white-preserving 3x3, weights Y^(-2/3), linear least squares
``lab``         white-preserving 3x3 minimising the CIE Lab error after ARRI's matrix directly
``free3x3``     unconstrained 3x3, same weights (does NOT keep white white; reference only)
``rootpoly``    unconstrained second-order root-polynomial (6 terms), same weights (reference only)

Only ``spec`` is what the product ships. The other columns say how much headroom a different
objective or a non-linear map would have, and whether the gate threshold (median <= 0.5,
p95 <= 1.2) is reachable at all for a light. Nothing here changes the gate, which uses ``spec``.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from scipy.optimize import least_squares

from training.alexa import ssf_core as s
from training.sensor_map import evaluate as ev
from training.sensor_map import fit
from training.sensor_map import lights as lt
from training.sensor_map.registry_data import PROFILES


def _wp(p: np.ndarray) -> np.ndarray:
    M = np.zeros((3, 3))
    M[:, :2] = p.reshape(3, 2)
    M[:, 2] = 1 - M[:, 0] - M[:, 1]
    return M


def _root_poly(x: np.ndarray) -> np.ndarray:
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    return np.stack([r, g, b, np.sqrt(np.abs(r * g)), np.sqrt(np.abs(g * b)), np.sqrt(np.abs(r * b))], axis=1)


def diagnose(profile_id: str) -> list[dict]:
    ctx = ev.context()
    prof = next(p for p in PROFILES if p.id == profile_id)
    S = fit.sony_curves()[prof.curves_name]
    rows = []
    for L in lt.gate_lights():
        E = L.spd
        src_tr = s.camera_rgb_wb(S, E, ctx.r_train)
        dst_tr = s.camera_rgb_wb(ctx.alexa, E, ctx.r_train)
        src_te = s.camera_rgb_wb(S, E, ctx.r_ces)
        dst_te = s.camera_rgb_wb(ctx.alexa, E, ctx.r_ces)
        M_arri = s.interpolate_matrix(ctx.arri_ccts, ctx.arri_mats, L.cct_k)
        lab_tr = s.ap0_to_Lab(dst_tr @ M_arri.T)
        lab_te = s.ap0_to_Lab(dst_te @ M_arri.T)
        w = np.clip(s.target_XYZ(E, ctx.r_train)[:, 1], 1e-3, None) ** (-2 / 3)

        def score(pred_rgb: np.ndarray) -> tuple[float, float]:
            d = s.delta_E00(s.ap0_to_Lab(pred_rgb @ M_arri.T), lab_te)
            return float(np.median(d)), float(np.percentile(d, 95))

        M_spec = fit.fit_map(S, ctx.alexa, E, ctx.r_train)
        lab_fit = least_squares(lambda p: (s.ap0_to_Lab((src_tr @ _wp(p).T) @ M_arri.T) - lab_tr).ravel(),
                                M_spec[:, :2].ravel(), method="lm")
        P3, *_ = np.linalg.lstsq(src_tr * w[:, None], dst_tr * w[:, None], rcond=None)
        Pr, *_ = np.linalg.lstsq(_root_poly(src_tr) * w[:, None], dst_tr * w[:, None], rcond=None)
        rows.append({
            "light": L.name,
            "spec": score(src_te @ M_spec.T),
            "lab": score(src_te @ _wp(lab_fit.x).T),
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
    print(f"  {'light':<18} {'spec':>12} {'lab':>12} {'free3x3':>12} {'rootpoly':>12}")
    for r in rows:
        print(f"  {r['light']:<18} " + " ".join(f"{r[k][0]:5.3f}/{r[k][1]:5.3f}  " for k in ("spec", "lab", "free3x3", "rootpoly")))
    return 0


if __name__ == "__main__":
    sys.exit(main())

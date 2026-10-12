"""How much does a per-camera profile matter, and can estimated curves fill the gaps?

Experiment A (wrong profile): files from Sony body X are mapped to the ALEXA
target with a per-light 3x3 fitted on the A7R IV curves instead of X's own
curves. Error is ΔE00 against the ALEXA rendering of the same scene.

Experiment B (estimated curves): for the 11 Sony bodies with measured curves
(Weta, rawtoaces-data), compare the published matrix-based estimates of
Solomatov & Akkaynak (2023) with the measurement, and repeat experiment A with
the map fitted on the estimate. Their data has no licence, so it is used here
for evaluation only and nothing from it is redistributed.
"""

from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ssf_core as s  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_DIR = sys.argv[1] if len(sys.argv) > 1 else ""
OUT = os.path.join(ROOT, "out")

cams = s.load_cameras()
sony = sorted(k for k in cams if k.startswith("Sony "))
REF = "Sony ILCE-7RM4"

# ALEXA target curves (our estimate) and ARRI's matrices for the Lab conversion.
alexa = np.loadtxt(os.path.join(OUT, "alexa_alev3_estimate_380_780_5.csv"), delimiter=",", skiprows=1)[:, 1:4]
alexa = alexa / alexa.max(axis=0, keepdims=True)
ccts, mats = s.load_arri_matrices()

R_train = s.training_reflectances()
R_test = np.concatenate([v for v in s.test_reflectances().values()], axis=1)
illums = s.test_illuminants()


def fit_map(S_src, E):
    """White-preserving 3x3: source WB RGB -> ALEXA WB RGB, fitted on training patches."""
    src = s.camera_rgb_wb(S_src, E, R_train)
    dst = s.camera_rgb_wb(alexa, E, R_train)
    # Weighted linear LS with rows summing to 1 (white stays white); weights
    # approximate perceptual (Lab-like) error, as in ssf_core.fit_matrix_linear.
    w = np.clip(s.target_XYZ(E, R_train)[:, 1], 1e-3, None) ** (-2 / 3)
    A = np.stack([src[:, 0] - src[:, 2], src[:, 1] - src[:, 2]], axis=1) * w[:, None]
    M = np.zeros((3, 3))
    for r in range(3):
        m, *_ = np.linalg.lstsq(A, (dst[:, r] - src[:, 2]) * w, rcond=None)
        M[r] = [m[0], m[1], 1 - m[0] - m[1]]
    return M


def map_error(S_true, S_fit):
    """ΔE00 per illuminant: X's RGB through a map fitted on S_fit vs the ALEXA rendering."""
    res = {}
    for name, E in illums.items():
        M_arri = s.interpolate_matrix(ccts, mats, s.illuminant_cct(E))
        target = s.ap0_to_Lab(s.camera_rgb_wb(alexa, E, R_test) @ M_arri.T)
        mapped = s.camera_rgb_wb(S_true, E, R_test) @ fit_map(S_fit, E).T
        res[name] = s.delta_E00(s.ap0_to_Lab(mapped @ M_arri.T), target)
    return res


def summary(res):
    allv = np.concatenate(list(res.values()))
    worst = max(res, key=lambda k: np.median(res[k]))
    return float(np.median(allv)), float(np.percentile(allv, 95)), worst, float(np.median(res[worst]))


def load_pred(name):
    model = name.split(" ", 1)[1]
    path = os.path.join(PRED_DIR, f"predicted-Sony-{model}.csv")
    if not PRED_DIR or not os.path.exists(path):
        return None
    tab = np.loadtxt(path, delimiter=",", skiprows=1)
    S = np.stack([np.interp(s.WL, tab[:, 0], tab[:, c], left=0.0, right=0.0) for c in (1, 2, 3)], axis=1)
    return S / S.max(axis=0, keepdims=True)


rows = []
for name in sony:
    S = cams[name]
    own = summary(map_error(S, S))
    ref = summary(map_error(S, cams[REF]))
    row = {
        "camera": name,
        "rfse_vs_7RM4_%": round(s.rfse(cams[REF], S, band=(400, 700)), 2),
        "own_median": round(own[0], 3), "own_p95": round(own[1], 3),
        "7RM4_median": round(ref[0], 3), "7RM4_p95": round(ref[1], 3),
        "7RM4_worst_light": ref[2], "7RM4_worst_median": round(ref[3], 3),
    }
    P = load_pred(name)
    if P is not None:
        est = summary(map_error(S, P))
        row.update({
            "est_rfse_%": round(s.rfse(P, S, band=(400, 700)), 2),
            "est_median": round(est[0], 3), "est_p95": round(est[1], 3),
        })
    rows.append(row)
    print(json.dumps(row))

with open(os.path.join(OUT, "sony_profiles.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) + [k for k in ("est_rfse_%", "est_median", "est_p95") if k not in rows[0]])
    w.writeheader()
    w.writerows(rows)

"""Estimate ALEXA (ALEV III, SUP 3.x) spectral sensitivities from ARRI's
published CCT-dependent native -> ACES AP0 matrices, and run diagnostics.

Outputs (out/):
  alexa_alev3_estimate_380_780_5.csv   recommended curve + ensemble band
  alexa_alev3_estimate_380_780_5.json  same, rawtoaces-data schema
  variants.csv                         every ensemble member's curve
  diagnostics.json                     all numbers quoted in the README
"""

import csv
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ssf_core as sc  # noqa: E402

OUT = os.path.join(sc.ROOT, "out")
CAMS = sc.load_cameras()
NAMES = list(CAMS)
CCTS, ARRI = sc.load_arri_matrices()
R_TRAIN = sc.training_reflectances()
R_TEST = sc.test_reflectances()
CES = R_TEST["CIE 2017 CES (99)"]
ILLUMS = sc.test_illuminants()

RECOMMENDED = {"prior": "all", "K": 4, "lam": 1.0, "conv": "rta"}
PRIORS = {
    "all": NAMES,
    "no_sony": [n for n in NAMES if not n.startswith("Sony")],
    "no_d21": [n for n in NAMES if n != "ARRI D21"],
}


def estimate(prior_names, K, lam, conv, mats=ARRI, n_starts=4):
    prior = sc.Prior([CAMS[n] for n in prior_names], K)
    est = sc.Estimator(prior, CCTS, mats, R_TRAIN, convention=conv,
                       objective="consistency", lam=lam)
    S, res = est.fit(n_starts=n_starts)
    return S, res


def matrix_rms(S, mats, conv="rta"):
    """RMS per coefficient between the matrices S would get and `mats`."""
    own = sc.camera_matrices(S, CCTS, R_TRAIN, convention=conv)
    return float(np.sqrt(np.mean((own - mats) ** 2)))


def run_variant(args):
    pname, K, lam, conv = args
    S, _ = estimate(PRIORS[pname], K, lam, conv)
    return {"prior": pname, "K": K, "lam": lam, "conv": conv,
            "curve": S, "matrix_rms": matrix_rms(S, ARRI, conv)}


def loo_typicality(name):
    """Held-out camera: estimate from its own matrices, return residual + RFSE."""
    truth = CAMS[name]
    pub = sc.camera_matrices(truth, CCTS, R_TRAIN)
    S, _ = estimate([n for n in NAMES if n != name], RECOMMENDED["K"],
                    RECOMMENDED["lam"], "rta", mats=pub, n_starts=2)
    return {"camera": name, "matrix_rms": matrix_rms(S, pub),
            "rfse": sc.rfse(S, truth)}


def white_preserving_3x3(x, y):
    """LS white-preserving 3x3 mapping rows of x onto rows of y."""
    A = np.stack([x[:, 0] - x[:, 2], x[:, 1] - x[:, 2]], axis=1)
    M = np.zeros((3, 3))
    for r in range(3):
        m, *_ = np.linalg.lstsq(A, y[:, r] - x[:, 2], rcond=None)
        M[r] = [m[0], m[1], 1 - m[0] - m[1]]
    return M


def root_poly2(x):
    """2nd-order root-polynomial expansion (exposure-invariant), 6 terms."""
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    return np.stack([r, g, b, np.sqrt(np.abs(r * g)), np.sqrt(np.abs(g * b)),
                     np.sqrt(np.abs(r * b))], axis=1)


def mapping_check(S_sony, S_alexa):
    """How much of Sony -> ALEXA a simple map captures, per test illuminant.

    Fitted on the 190 training reflectances, scored on the 99 CIE CES, both
    images rendered through ARRI's matrix at the illuminant's CCT.
    """
    E_ref = sc.illuminant(5500, "rta")
    M_global = white_preserving_3x3(sc.camera_rgb_wb(S_sony, E_ref, R_TRAIN),
                                    sc.camera_rgb_wb(S_alexa, E_ref, R_TRAIN))
    out = {}
    for name, E in ILLUMS.items():
        M_arri = sc.interpolate_matrix(CCTS, ARRI, sc.illuminant_cct(E))
        xs_tr = sc.camera_rgb_wb(S_sony, E, R_TRAIN)
        ya_tr = sc.camera_rgb_wb(S_alexa, E, R_TRAIN)
        xs = sc.camera_rgb_wb(S_sony, E, CES)
        ya = sc.camera_rgb_wb(S_alexa, E, CES)
        Lab_true = sc.ap0_to_Lab(ya @ M_arri.T)
        M_ill = white_preserving_3x3(xs_tr, ya_tr)
        P, *_ = np.linalg.lstsq(root_poly2(xs_tr), ya_tr, rcond=None)
        preds = {
            "no mapping (Sony RGB used as ALEXA RGB)": xs,
            "one global 3x3 (fitted at 5500 K)": xs @ M_global.T,
            "3x3 per illuminant": xs @ M_ill.T,
            "root-polynomial per illuminant": root_poly2(xs) @ P,
        }
        out[name] = {}
        for k, y in preds.items():
            d = sc.delta_E00(Lab_true, sc.ap0_to_Lab(y @ M_arri.T))
            out[name][k] = {"median": float(np.median(d)),
                            "p95": float(np.percentile(d, 95)),
                            "max": float(np.max(d))}
    return out


def colorimetric_ranking(S_est):
    """Mean ΔE00 to scene colour when ARRI's matrices are applied to each camera."""
    cams = dict(CAMS)
    cams["ALEXA estimate (this work)"] = S_est
    mean_cam = np.mean(np.stack(list(CAMS.values())), axis=0)
    cams["Mean of 52 cameras"] = mean_cam / mean_cam.max(axis=0)
    out = {}
    for name, S in cams.items():
        d = sc.colorimetric_dE(S, CCTS, ARRI, CES, ILLUMS)
        out[name] = float(np.mean(np.concatenate(list(d.values()))))
    return dict(sorted(out.items(), key=lambda kv: kv[1]))


def write_outputs(rec, ens, variants, diag):
    wl = sc.WL.astype(int)
    with open(os.path.join(OUT, "alexa_alev3_estimate_380_780_5.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["wavelength_nm", "R", "G", "B",
                    "R_min", "G_min", "B_min", "R_max", "G_max", "B_max",
                    "R_std", "G_std", "B_std"])
        for i, l in enumerate(wl):
            w.writerow([l, *np.round(rec[i], 6), *np.round(ens.min(0)[i], 6),
                        *np.round(ens.max(0)[i], 6), *np.round(ens.std(0)[i], 6)])
    doc = {
        "header": {
            "schema_version": "0.1.0",
            "manufacturer": "ARRI",
            "model": "ALEXA (ALEV III, SUP 3.x) - ESTIMATE",
            "description": (
                "Estimated camera spectral sensitivity, NOT a measurement. Fitted so that "
                "the IDT matrices these curves would receive (Academy P-2013-001-style "
                "CIE Lab fit, rawtoaces 190-patch training set) reproduce ARRI's 2013 "
                "ALEXA native->ACES AP0 matrices at 17 CCTs, under a 4-component PCA "
                "prior built from the 52 Apache-2.0 curves in rawtoaces-data. Each "
                "channel normalised to peak 1; relative channel gains are not "
                "identifiable from white-balanced matrices and are not needed for "
                "white-balanced emulation. See README for accuracy and caveats."
            ),
            "document_creator": "alexa-ssf-estimate",
            "document_creation_date": "2026-10-11",
            "measurement_equipment": None,
            "license": "Apache-2.0 (see NOTICE: derived from AMPAS-licensed aces-dev data)",
        },
        "spectral_data": {
            "units": "relative",
            "index": {"main": ["R", "G", "B"]},
            "data": {"main": {str(l): [round(float(v), 6) for v in rec[i]]
                              for i, l in enumerate(wl)}},
        },
    }
    with open(os.path.join(OUT, "alexa_alev3_estimate_380_780_5.json"), "w") as fh:
        json.dump(doc, fh, indent=2)
    with open(os.path.join(OUT, "variants.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        head = ["prior", "K", "lam", "conv", "matrix_rms", "channel"] + [str(l) for l in wl]
        w.writerow(head)
        for v in variants:
            for c, ch in enumerate("RGB"):
                w.writerow([v["prior"], v["K"], v["lam"], v["conv"],
                            round(v["matrix_rms"], 6), ch, *np.round(v["curve"][:, c], 6)])
    with open(os.path.join(OUT, "diagnostics.json"), "w") as fh:
        json.dump(diag, fh, indent=2)


def peak_nm(S):
    return {ch: int(sc.WL[np.argmax(S[:, c])]) for c, ch in enumerate("RGB")}


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    grid = [(p, K, lam, conv) for p in PRIORS for K in (3, 4)
            for lam in (0.1, 1.0) for conv in ("rta", "planck")]
    with Pool(2) as pool:
        variants = pool.map(run_variant, grid)
        typ = pool.map(loo_typicality, NAMES)

    rec_v = next(v for v in variants if v["prior"] == RECOMMENDED["prior"]
                 and v["K"] == RECOMMENDED["K"] and v["lam"] == RECOMMENDED["lam"]
                 and v["conv"] == RECOMMENDED["conv"])
    rec = rec_v["curve"]
    ens = np.stack([v["curve"] for v in variants])
    spread = [sc.rfse(v["curve"], rec) for v in variants]

    typ_rms = np.array([t["matrix_rms"] for t in typ])
    arri_rms = rec_v["matrix_rms"]
    pct = float(100 * np.mean(typ_rms <= arri_rms))
    similar = sorted(typ, key=lambda t: abs(t["matrix_rms"] - arri_rms))[:5]

    sony = CAMS["Sony ILCE-7RM4"]
    d21 = CAMS["ARRI D21"]
    diag = {
        "recommended_variant": RECOMMENDED,
        "peaks_nm": {"estimate": peak_nm(rec), "Sony ILCE-7RM4": peak_nm(sony),
                     "ARRI D21": peak_nm(d21)},
        "ensemble": {
            "n_variants": len(variants),
            "rfse_vs_recommended_pct": {"median": float(np.median(spread)),
                                        "max": float(np.max(spread))},
            "matrix_rms_per_variant": [
                {k: v[k] for k in ("prior", "K", "lam", "conv", "matrix_rms")}
                for v in variants],
        },
        "rfse_between_curves_pct": {
            "estimate_vs_Sony_ILCE-7RM4": sc.rfse(rec, sony),
            "estimate_vs_ARRI_D21": sc.rfse(rec, d21),
            "Sony_ILCE-7RM4_vs_ARRI_D21": sc.rfse(sony, d21),
        },
        "typicality": {
            "arri_matrix_rms": arri_rms,
            "loo_matrix_rms_median": float(np.median(typ_rms)),
            "loo_matrix_rms_p90": float(np.percentile(typ_rms, 90)),
            "arri_percentile_among_held_out_cameras": pct,
            "held_out_cameras_with_most_similar_residual": similar,
            "all_held_out": typ,
        },
        "colorimetric_mean_dE00_with_ARRI_matrices": colorimetric_ranking(rec),
        "sony_7rm4_to_alexa_estimate_mapping_dE00": mapping_check(sony, rec),
    }
    write_outputs(rec, ens, variants, diag)
    print(json.dumps({k: diag[k] for k in ("peaks_nm", "ensemble", "rfse_between_curves_pct")}, indent=1)[:3000])
    t = diag["typicality"]
    print("typicality:", {k: t[k] for k in list(t)[:4]})
    print("similar:", t["held_out_cameras_with_most_similar_residual"])
    rank = list(diag["colorimetric_mean_dE00_with_ARRI_matrices"].items())
    for i, (k, v) in enumerate(rank):
        if i < 5 or "ALEXA" in k or "D21" in k or "7RM4" in k or "Mean" in k:
            print(i + 1, k, round(v, 2))
    print(json.dumps(diag["sony_7rm4_to_alexa_estimate_mapping_dE00"], indent=1)[:4000])

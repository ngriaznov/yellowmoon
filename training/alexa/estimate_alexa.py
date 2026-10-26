"""Estimate ALEXA (ALEV III, SUP 3.x) spectral sensitivities from ARRI's
published CCT-dependent native -> ACES AP0 matrices, and run diagnostics.

The numerics are those of the published estimate (research/alexa-ssf/README.md); the
inputs come from testdata/sources (tools/fetch/spectral.sh) and the outputs go to
testdata/out/alexa/estimate/ (git-ignored), not over the committed data/alexa/ files.
Compare with ``python3 -m training.alexa.compare_estimate``. ~15 min on 4 cores.

    python3 -m training.alexa.estimate_alexa              # the shipped curve: estimate + hue push
    python3 -m training.alexa.estimate_alexa --no-push    # the estimate of 2026-10-11 (before the push)

The hue push (owner decision 2026-10-25, "Move ALEXA estimate")
---------------------------------------------------------------
Yellowmoon rendered photographs of ochre walls, oranges and reds more crimson than the camera's own
picture. The ALEXA curve is a place where a small part of that gap can be closed: ARRI's matrices fix
the curve only up to what a 4-component prior and 17 matrices can tell, and the family of curves that
agree with them about equally well moves the hue of warm colours by more than a degree. The push picks,
inside that family, the curve that renders orange, red and ochre reflectances ``PUSH["deg"]`` degrees of
CIELAB hue toward yellow (less crimson), and no further than the matrices allow:

* the reflectances: the training reflectances whose Lab hue is between 25 and 95 degrees with
  chroma above 12 under the un-pushed estimate (47 of 190), rendered at 5200 K (the rawtoaces
  illuminant) through ARRI's matrix at 5200 K;
* the residual: ``weight * (median hue shift of those reflectances - deg)`` is added to the
  estimator's residuals (consistency with the 17 matrices, the prior's penalty, non-negativity);
* the stopping rule: the push is the largest step of 0.5 degrees whose matrix RMS stays within the
  24-variant ensemble's maximum (0.0093 per coefficient; the un-pushed estimate has 0.0067). Steps of
  0.5, 1.0, 1.5, 2.0 and 3.0 degrees give 0.0070, 0.0081, 0.0092, 0.0112 and 0.0166 (docs/03), so
  ``deg`` is 1.5. A run whose matrix RMS exceeds the limit stops without writing the files;
* seeded and offline: ``Estimator.fit`` draws its extra start from a generator seeded with 0.

Outputs (testdata/out/alexa/estimate/, or estimate_nopush/ with ``--no-push``):
  alexa_alev3_estimate_380_780_5.csv   recommended curve + ensemble band (the band is that of the
                                       24 variants, which are all un-pushed)
  alexa_alev3_estimate_380_780_5.json  the recommended curve, rawtoaces-data schema
  variants.csv                         every ensemble member's curve (un-pushed)
  diagnostics.json                     all numbers quoted in the README
"""

import argparse
import csv
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

from training.alexa import ssf_core as sc
from training.common import paths

OUT = str(paths.OUT / "alexa" / "estimate")
OUT_NOPUSH = str(paths.OUT / "alexa" / "estimate_nopush")
paths.require_sources()
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


# --------------------------------------------------------------------------
# Hue push (owner decision 2026-10-25, "Move ALEXA estimate")
# --------------------------------------------------------------------------
PUSH = {
    "deg": 1.5,                   # median Lab hue shift asked for on the warm reflectances (toward yellow)
    "scene_cct_k": 5200.0,        # light (rawtoaces illuminant) and ARRI matrix of the measurement
    "hue_window_deg": (25.0, 95.0),  # Lab hue of the un-pushed rendering: red, orange, ochre
    "min_chroma": 12.0,           # Lab chroma of the un-pushed rendering
    "weight": 10.0,               # weight of the hue residual against the matrix residuals
    "n_starts": 2,                # least-squares starts: zero and one seeded draw (Estimator.fit, seed 0)
}
DATE_NOPUSH = "2026-10-11"
DATE_PUSH = "2026-10-25"
DESCRIPTION = (
    "Estimated camera spectral sensitivity, NOT a measurement. Fitted so that "
    "the IDT matrices these curves would receive (Academy P-2013-001-style "
    "CIE Lab fit, rawtoaces 190-patch training set) reproduce ARRI's 2013 "
    "ALEXA native->ACES AP0 matrices at 17 CCTs, under a 4-component PCA "
    "prior built from the 52 Apache-2.0 curves in rawtoaces-data. Each "
    "channel normalised to peak 1; relative channel gains are not "
    "identifiable from white-balanced matrices and are not needed for "
    "white-balanced emulation. See README for accuracy and caveats."
)
DESCRIPTION_PUSH = (
    DESCRIPTION[:DESCRIPTION.index(" See README")]
    + " The curves are moved by a hue push (owner decision 2026-10-25): inside the family of "
    "curves that agree with ARRI's matrices, the one that renders orange, red and ochre "
    "reflectances {deg} degrees of CIELAB hue toward yellow, with matrix RMS {rms:.4f} per "
    "coefficient (the 24-variant ensemble spans {lo:.4f} to {hi:.4f}; without the push "
    "{unpushed:.4f}). See README for accuracy and caveats."
)


def _lab_hue_chroma(S, ref):
    """Lab hue (deg) and chroma of the training reflectances rendered through S, for the push's light."""
    S = np.clip(S, 0, None)
    S = S / S.max(axis=0, keepdims=True)
    lab = sc.ap0_to_Lab(sc.camera_rgb_wb(S, ref["E"], R_TRAIN) @ ref["M_arri"].T)
    return np.degrees(np.arctan2(lab[:, 2], lab[:, 1])) % 360, np.hypot(lab[:, 1], lab[:, 2])


def push_reference(S0, push=PUSH):
    """The warm reflectances, the light and ARRI's matrix of the push, from the un-pushed estimate S0."""
    ref = {"E": sc.illuminant(push["scene_cct_k"], "rta"),
           "M_arri": sc.interpolate_matrix(CCTS, ARRI, push["scene_cct_k"])}
    ref["h0"], ref["C0"] = _lab_hue_chroma(S0, ref)
    lo, hi = push["hue_window_deg"]
    ref["sel"] = (ref["h0"] > lo) & (ref["h0"] < hi) & (ref["C0"] > push["min_chroma"])
    return ref


def hue_shift(S, ref):
    """Median Lab hue shift (deg, positive toward yellow) and chroma shift of the warm reflectances."""
    h, C = _lab_hue_chroma(S, ref)
    dh = (h - ref["h0"] + 180) % 360 - 180
    return float(np.median(dh[ref["sel"]])), float(np.median((C - ref["C0"])[ref["sel"]]))


class PushedEstimator(sc.Estimator):
    """The estimator's residuals plus ``weight * (median hue shift - deg)`` on the warm reflectances."""

    def __init__(self, *args, ref, deg, weight, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref, self.deg, self.weight = ref, deg, weight

    def residuals(self, a):
        r = super().residuals(a)
        dh, _ = hue_shift(self.prior.curves(a), self.ref)
        return np.concatenate([r, [self.weight * (dh - self.deg)]])


def estimate_pushed(S0, prior_names, K, lam, conv, push=PUSH):
    """The recommended estimate with the hue push; ``S0`` is the same estimate without it.

    Returns (curve, info) where info holds the achieved shifts, the matrix RMS and the number of
    reflectances in the push.
    """
    ref = push_reference(S0, push)
    prior = sc.Prior([CAMS[n] for n in prior_names], K)
    est = PushedEstimator(prior, CCTS, ARRI, R_TRAIN, convention=conv, objective="consistency", lam=lam,
                          ref=ref, deg=push["deg"], weight=push["weight"])
    S, _ = est.fit(n_starts=push["n_starts"])
    dh, dC = hue_shift(S, ref)
    return S, {"reflectances": int(ref["sel"].sum()), "hue_shift_deg": dh, "chroma_shift": dC,
               "matrix_rms": matrix_rms(S, ARRI, conv)}


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


def colorimetric_ranking(S_est, S_unpushed=None):
    """Mean ΔE00 to scene colour when ARRI's matrices are applied to each camera."""
    cams = dict(CAMS)
    cams["ALEXA estimate (this work)"] = S_est
    if S_unpushed is not None:
        cams["ALEXA estimate before the hue push"] = S_unpushed
    mean_cam = np.mean(np.stack(list(CAMS.values())), axis=0)
    cams["Mean of 52 cameras"] = mean_cam / mean_cam.max(axis=0)
    out = {}
    for name, S in cams.items():
        d = sc.colorimetric_dE(S, CCTS, ARRI, CES, ILLUMS)
        out[name] = float(np.mean(np.concatenate(list(d.values()))))
    return dict(sorted(out.items(), key=lambda kv: kv[1]))


def write_outputs(rec, ens, variants, diag, out_dir, description, date):
    wl = sc.WL.astype(int)
    with open(os.path.join(out_dir, "alexa_alev3_estimate_380_780_5.csv"), "w", newline="") as fh:
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
            "description": description,
            "document_creator": "alexa-ssf-estimate",
            "document_creation_date": date,
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
    with open(os.path.join(out_dir, "alexa_alev3_estimate_380_780_5.json"), "w") as fh:
        json.dump(doc, fh, indent=2)
    with open(os.path.join(out_dir, "variants.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        head = ["prior", "K", "lam", "conv", "matrix_rms", "channel"] + [str(l) for l in wl]
        w.writerow(head)
        for v in variants:
            for c, ch in enumerate("RGB"):
                w.writerow([v["prior"], v["K"], v["lam"], v["conv"],
                            round(v["matrix_rms"], 6), ch, *np.round(v["curve"][:, c], 6)])
    with open(os.path.join(out_dir, "diagnostics.json"), "w") as fh:
        json.dump(diag, fh, indent=2)


def peak_nm(S):
    return {ch: int(sc.WL[np.argmax(S[:, c])]) for c, ch in enumerate("RGB")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-push", action="store_true",
                    help="write the estimate of 2026-10-11, without the hue push (to estimate_nopush/)")
    ap.add_argument("--push-deg", type=float, default=PUSH["deg"],
                    help="hue push in degrees toward yellow (default %(default)s; the run refuses a push whose "
                         "matrix RMS exceeds the ensemble's maximum)")
    ap.add_argument("--out", default=None, help="output folder (default testdata/out/alexa/estimate[_nopush])")
    args = ap.parse_args(argv)
    push_on = not args.no_push
    push = dict(PUSH, deg=args.push_deg)
    out_dir = args.out or (OUT if push_on else OUT_NOPUSH)
    os.makedirs(out_dir, exist_ok=True)

    grid = [(p, K, lam, conv) for p in PRIORS for K in (3, 4)
            for lam in (0.1, 1.0) for conv in ("rta", "planck")]
    with Pool(min(4, os.cpu_count() or 1)) as pool:
        variants = pool.map(run_variant, grid)
        typ = pool.map(loo_typicality, NAMES)

    rec_v = next(v for v in variants if v["prior"] == RECOMMENDED["prior"]
                 and v["K"] == RECOMMENDED["K"] and v["lam"] == RECOMMENDED["lam"]
                 and v["conv"] == RECOMMENDED["conv"])
    unpushed = rec_v["curve"]
    ens = np.stack([v["curve"] for v in variants])
    ens_rms = [v["matrix_rms"] for v in variants]
    spread = [sc.rfse(v["curve"], unpushed) for v in variants]

    rec, push_info, description, date = unpushed, None, DESCRIPTION, DATE_NOPUSH
    if push_on:
        rec, info = estimate_pushed(unpushed, PRIORS[RECOMMENDED["prior"]], RECOMMENDED["K"],
                                    RECOMMENDED["lam"], RECOMMENDED["conv"], push)
        limit = max(ens_rms)
        print(f"hue push {push['deg']} deg: achieved {info['hue_shift_deg']:+.2f} deg, chroma "
              f"{info['chroma_shift']:+.2f}, {info['reflectances']} reflectances, matrix RMS "
              f"{info['matrix_rms']:.4f} (ensemble {min(ens_rms):.4f} to {limit:.4f})", flush=True)
        if info["matrix_rms"] > limit:
            raise SystemExit(f"stopping rule: matrix RMS {info['matrix_rms']:.4f} exceeds the ensemble's maximum "
                             f"{limit:.4f}; use a smaller --push-deg (nothing written)")
        push_info = {**push, **info, "ensemble_matrix_rms_min": min(ens_rms), "ensemble_matrix_rms_max": limit,
                     "unpushed_matrix_rms": rec_v["matrix_rms"],
                     "rfse_pushed_vs_unpushed_pct": sc.rfse(rec, unpushed),
                     "max_abs_difference_per_channel": [float(x) for x in np.abs(rec - unpushed).max(axis=0)],
                     "outside_ensemble_band": {
                         ch: int(np.sum((np.round(rec[:, c], 6) < np.round(ens.min(0)[:, c], 6) - 1e-9)
                                        | (np.round(rec[:, c], 6) > np.round(ens.max(0)[:, c], 6) + 1e-9)))
                         for c, ch in enumerate("RGB")}}
        description = DESCRIPTION_PUSH.format(deg=push["deg"], rms=info["matrix_rms"], lo=min(ens_rms), hi=limit,
                                              unpushed=rec_v["matrix_rms"])
        date = DATE_PUSH

    typ_rms = np.array([t["matrix_rms"] for t in typ])
    arri_rms = matrix_rms(rec, ARRI, RECOMMENDED["conv"])
    pct = float(100 * np.mean(typ_rms <= arri_rms))
    similar = sorted(typ, key=lambda t: abs(t["matrix_rms"] - arri_rms))[:5]

    sony = CAMS["Sony ILCE-7RM4"]
    d21 = CAMS["ARRI D21"]
    diag = {
        "recommended_variant": RECOMMENDED,
        "hue_push": push_info,
        "peaks_nm": {"estimate": peak_nm(rec), "Sony ILCE-7RM4": peak_nm(sony),
                     "ARRI D21": peak_nm(d21)},
        "ensemble": {
            "n_variants": len(variants),
            "rfse_vs_unpushed_recommended_pct": {"median": float(np.median(spread)),
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
        "colorimetric_mean_dE00_with_ARRI_matrices": colorimetric_ranking(rec, unpushed if push_on else None),
        "sony_7rm4_to_alexa_estimate_mapping_dE00": mapping_check(sony, rec),
    }
    write_outputs(rec, ens, variants, diag, out_dir, description, date)
    print(json.dumps({k: diag[k] for k in ("peaks_nm", "ensemble", "rfse_between_curves_pct")}, indent=1)[:3000])
    t = diag["typicality"]
    print("typicality:", {k: t[k] for k in list(t)[:4]})
    print("similar:", t["held_out_cameras_with_most_similar_residual"])
    rank = list(diag["colorimetric_mean_dE00_with_ARRI_matrices"].items())
    for i, (k, v) in enumerate(rank):
        if i < 5 or "ALEXA" in k or "D21" in k or "7RM4" in k or "Mean" in k:
            print(i + 1, k, round(v, 2))
    print(json.dumps(diag["sony_7rm4_to_alexa_estimate_mapping_dE00"], indent=1)[:4000])
    print("wrote", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

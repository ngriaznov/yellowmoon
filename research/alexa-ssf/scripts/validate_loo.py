"""Leave-one-camera-out validation of the matrix-to-curve estimator.

For each of the 52 measured cameras in rawtoaces-data:
  1. make the 17 CCT-dependent IDT matrices that camera would get
     (same CCT grid as ARRI's table),
  2. estimate its curves from those matrices alone, with a prior built from
     the other 51 cameras,
  3. score the estimate against the measured curves.

Writes out/loo_results.csv.
"""

import csv
import itertools
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ssf_core as sc  # noqa: E402

OUT = os.path.join(sc.ROOT, "out")
KS = (2, 3, 4)
LAMS = (0.1, 1.0)
CASES = [  # (convention used to make the matrices, convention assumed when estimating)
    ("rta", "rta"),      # recipe known
    ("planck", "rta"),   # wrong illuminant model assumed
]

CAMS = sc.load_cameras()
NAMES = list(CAMS)
CCTS, _ = sc.load_arri_matrices()
R_TRAIN = sc.training_reflectances()
R_TEST = sc.test_reflectances()["CIE 2017 CES (99)"]
ILLUMS = sc.test_illuminants()


def run_one(name):
    truth = CAMS[name]
    others = [CAMS[n] for n in NAMES if n != name]
    rows = []
    for make_conv, est_conv in CASES:
        pub = sc.camera_matrices(truth, CCTS, R_TRAIN, convention=make_conv)
        for K in KS:
            prior = sc.Prior(others, K)
            mean = prior.mean / prior.mean.max(axis=0)
            dE_mean = sc.emulation_dE(truth, mean, CCTS, pub, R_TEST, ILLUMS)
            for lam in LAMS:
                est = sc.Estimator(prior, CCTS, pub, R_TRAIN, convention=est_conv,
                                   objective="consistency", lam=lam)
                S, _ = est.fit()
                dE = sc.emulation_dE(truth, S, CCTS, pub, R_TEST, ILLUMS)
                row = {
                    "camera": name, "make_conv": make_conv, "est_conv": est_conv,
                    "K": K, "lam": lam,
                    "rfse_380_780": sc.rfse(S, truth),
                    "rfse_400_700": sc.rfse(S, truth, (400, 700)),
                    "rfse_mean_prior": sc.rfse(mean, truth),
                    "dE00_median_all": float(np.median(np.concatenate(list(dE.values())))),
                    "dE00_p95_all": float(np.percentile(np.concatenate(list(dE.values())), 95)),
                    "dE00_median_mean_prior": float(np.median(np.concatenate(list(dE_mean.values())))),
                }
                for k, v in dE.items():
                    row[f"dE00_median[{k}]"] = float(np.median(v))
                rows.append(row)
    return rows


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    with Pool(2) as pool:
        all_rows = list(itertools.chain.from_iterable(pool.map(run_one, NAMES)))
    path = os.path.join(OUT, "loo_results.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)
    print("wrote", path, len(all_rows), "rows")

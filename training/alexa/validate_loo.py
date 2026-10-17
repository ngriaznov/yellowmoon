"""Leave-one-camera-out validation of the matrix-to-curve estimator.

For each of the 52 measured cameras in rawtoaces-data:
  1. make the 17 CCT-dependent IDT matrices that camera would get
     (same CCT grid as ARRI's table),
  2. estimate its curves from those matrices alone, with a prior built from
     the other 51 cameras,
  3. score the estimate against the measured curves.

Writes ``testdata/out/alexa/loo_results.csv`` (git-ignored); ``check_loo`` compares it
with the published numbers.

The numerics are those of the published estimate (research/alexa-ssf/README.md). Inputs
come from testdata/sources, the output goes under testdata/out, and there are two options: ``--jobs`` (worker count; results do not depend on it) and
``--quick`` (only the recommended variant K=4, lam=1, which is what ``check_loo`` scores).

    python3 -m training.alexa.validate_loo            # full grid, ~5 min on 2 cores
    python3 -m training.alexa.validate_loo --quick    # recommended variant only
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
from multiprocessing import Pool

import numpy as np

from training.alexa import ssf_core as sc
from training.common import paths

OUT = paths.OUT / "alexa"
KS = (2, 3, 4)
LAMS = (0.1, 1.0)
RECOMMENDED = (4, 1.0)  # K, lam of the shipped curve estimate
CASES = [  # (convention used to make the matrices, convention assumed when estimating)
    ("rta", "rta"),      # recipe known
    ("planck", "rta"),   # wrong illuminant model assumed
]

paths.require_sources()
CAMS = sc.load_cameras()
NAMES = list(CAMS)
CCTS, _ = sc.load_arri_matrices()
R_TRAIN = sc.training_reflectances()
R_TEST = sc.test_reflectances()["CIE 2017 CES (99)"]
ILLUMS = sc.test_illuminants()

# Set by main(); module-level so forked and spawned workers see the same grid.
_KS = KS
_LAMS = LAMS


def run_one(name: str) -> list[dict]:
    truth = CAMS[name]
    others = [CAMS[n] for n in NAMES if n != name]
    rows = []
    for make_conv, est_conv in CASES:
        pub = sc.camera_matrices(truth, CCTS, R_TRAIN, convention=make_conv)
        for K in _KS:
            prior = sc.Prior(others, K)
            mean = prior.mean / prior.mean.max(axis=0)
            dE_mean = sc.emulation_dE(truth, mean, CCTS, pub, R_TEST, ILLUMS)
            for lam in _LAMS:
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


def main(argv: list[str] | None = None) -> int:
    global _KS, _LAMS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1), help="worker processes")
    ap.add_argument("--quick", action="store_true", help="only the recommended variant (K=4, lam=1)")
    ap.add_argument("--out", type=str, default=str(OUT / "loo_results.csv"), help="output CSV")
    args = ap.parse_args(argv)

    if args.quick:
        _KS, _LAMS = (RECOMMENDED[0],), (RECOMMENDED[1],)
    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"leave-one-out over {len(NAMES)} cameras, K={_KS}, lam={_LAMS}, jobs={args.jobs}", flush=True)
    with Pool(args.jobs) as pool:
        all_rows = list(itertools.chain.from_iterable(pool.map(run_one, NAMES)))
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)
    print("wrote", out_path, len(all_rows), "rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())

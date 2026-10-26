"""What the hue push costs on cameras whose curves are known (owner decision 2026-10-25).

The leave-one-out recipe of ``validate_loo`` (case rta/rta, K=4, lam=1, one start) run twice per
held-out camera, without and with the hue push of ``estimate_alexa`` (the push's reflectances are
chosen from the camera's own un-pushed estimate, the push is ``--push-deg`` degrees, default ``PUSH["deg"]``). Both estimates
are scored against the measured curves. This is **not a gate**: the push leaves the curve that agrees
best with the matrices on purpose, so a pushed estimate of a known camera is expected to be farther
from it than the plain one (``docs/03`` *The ALEXA estimate*, ``research/alexa-ssf/README.md``). It
also reports how far the measured camera's own warm-colour hue is from its plain estimate, which
says how large a push is next to the estimator's typical hue error.

    python3 -m training.alexa.loo_push [--jobs 3] [--only TEXT]        # about 45 min on 2 cores

Writes ``testdata/out/alexa/loo_push.csv`` and prints the summary.
"""

from __future__ import annotations

import argparse
import csv
import functools
import os
import sys
from multiprocessing import Pool

import numpy as np

from training.alexa import estimate_alexa as ea
from training.alexa import ssf_core as sc
from training.common import paths

OUT = paths.OUT / "alexa" / "loo_push.csv"
R_TEST = ea.R_TEST["CIE 2017 CES (99)"]
ILLUMS = ea.ILLUMS


def one(name: str, deg: float) -> dict:
    truth = ea.CAMS[name]
    others = [ea.CAMS[n] for n in ea.NAMES if n != name]
    pub = sc.camera_matrices(truth, ea.CCTS, ea.R_TRAIN, convention="rta")
    prior = sc.Prior(others, 4)
    plain, _ = sc.Estimator(prior, ea.CCTS, pub, ea.R_TRAIN, convention="rta", objective="consistency", lam=1.0).fit()
    ref = ea.push_reference(plain)
    pushed, _ = ea.PushedEstimator(
        prior, ea.CCTS, pub, ea.R_TRAIN, convention="rta", objective="consistency", lam=1.0,
        ref=ref, deg=deg, weight=ea.PUSH["weight"],
    ).fit(n_starts=1)
    row = {"camera": name, "n_warm": int(ref["sel"].sum())}
    for tag, S in (("plain", plain), ("push", pushed)):
        dE = np.concatenate(list(sc.emulation_dE(truth, S, ea.CCTS, pub, R_TEST, ILLUMS).values()))
        row.update({
            f"rfse_{tag}": sc.rfse(S, truth),
            f"dE_med_{tag}": float(np.median(dE)),
            f"dE_p95_{tag}": float(np.percentile(dE, 95)),
            f"rms_{tag}": ea.matrix_rms(S, pub),
            f"hue_shift_{tag}": ea.hue_shift(S, ref)[0],
        })
    row["hue_shift_truth"] = ea.hue_shift(truth, ref)[0]  # the measured camera against its plain estimate
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=min(3, os.cpu_count() or 1), help="worker processes")
    ap.add_argument("--only", default=None, help="only cameras whose name contains this text")
    ap.add_argument("--push-deg", type=float, default=ea.PUSH["deg"], help="the push in degrees (default %(default)s, the shipped one)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    names = [n for n in ea.NAMES if not args.only or args.only in n]
    with Pool(args.jobs) as pool:
        rows = pool.map(functools.partial(one, deg=args.push_deg), names, chunksize=1)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def col(k: str) -> np.ndarray:
        return np.array([r[k] for r in rows])

    for tag in ("plain", "push"):
        print(f"{tag:<5} leave-one-out, {len(rows)} cameras: RFSE median {np.median(col('rfse_' + tag)):.2f} "
              f"p90 {np.percentile(col('rfse_' + tag), 90):.2f} | emulation dE00 median {np.median(col('dE_med_' + tag)):.3f} "
              f"| p95 median {np.median(col('dE_p95_' + tag)):.2f} | matrix RMS median {np.median(col('rms_' + tag)):.4f}")
    d = col("rfse_push") - col("rfse_plain")
    ht = col("hue_shift_truth")
    print(f"the push brings the RFSE down for {int((d < 0).sum())} of {len(rows)} cameras (median change {np.median(d):+.2f}, "
          f"largest {d.max():+.2f})")
    print(f"measured camera against its plain estimate, warm-colour hue (deg): median {np.median(ht):+.2f}, p10 "
          f"{np.percentile(ht, 10):+.2f}, p90 {np.percentile(ht, 90):+.2f}; within 0.5 deg for {int((np.abs(ht) < 0.5).sum())} "
          f"of {len(rows)}, largest {np.abs(ht).max():.2f}")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

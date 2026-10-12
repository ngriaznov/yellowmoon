"""Compare a leave-one-camera-out run with the published numbers.

Reads ``testdata/out/alexa/loo_results.csv`` (written by ``validate_loo``), takes the
shipped variant (K=4, lam=1) and compares its medians with the numbers published in
``research/alexa-ssf/README.md`` and in ``docs/15`` (M1-03). Exits 0 when every
number is within ``--tol`` (default 0.1: percentage points for RFSE, ΔE00 for the
emulation errors) and the ``docs/13`` ALEXA-curve gate holds; non-zero otherwise.

    python3 -m training.alexa.validate_loo && python3 -m training.alexa.check_loo
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from training.common import paths

CSV_DEFAULT = paths.OUT / "alexa" / "loo_results.csv"
VARIANT = {"K": 4, "lam": 1.0}  # the shipped curve estimate

# Published (research/alexa-ssf/README.md, "Validation" table). (label, case, statistic, value, headline)
PUBLISHED = [
    ("RFSE median, illuminant model known (%)", "known", "rfse_median", 2.2, True),
    ("Emulation dE00 median, illuminant model known", "known", "emu_median", 0.23, True),
    ("RFSE p90, illuminant model known (%)", "known", "rfse_p90", 3.6, False),
    ("Emulation dE00 p95 (median over cameras), known", "known", "emu_p95_median", 0.63, False),
    ("RFSE median, wrong illuminant model (%)", "wrong", "rfse_median", 2.6, False),
    ("RFSE p90, wrong illuminant model (%)", "wrong", "rfse_p90", 4.0, False),
    ("Emulation dE00 median, wrong illuminant model", "wrong", "emu_median", 0.26, False),
    ("Emulation dE00 p95 (median over cameras), wrong", "wrong", "emu_p95_median", 0.75, False),
    ("RFSE median, average curve (no fitting) (%)", "known", "rfse_prior_median", 4.4, False),
    ("Emulation dE00 median, average curve", "known", "emu_prior_median", 1.27, False),
]
# docs/13 gates for the ALEXA curve (upper bounds).
GATES = [
    ("RFSE median <= 3 %", "known", "rfse_median", 3.0),
    ("Emulation dE00 median <= 0.3", "known", "emu_median", 0.3),
]
CASES = {"known": ("rta", "rta"), "wrong": ("planck", "rta")}
N_CAMERAS = 52


def load_stats(csv_path: Path) -> dict[str, dict[str, float]]:
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    stats: dict[str, dict[str, float]] = {}
    for case, (make, est) in CASES.items():
        sel = [r for r in rows
               if r["make_conv"] == make and r["est_conv"] == est
               and int(r["K"]) == VARIANT["K"] and float(r["lam"]) == VARIANT["lam"]]
        if len(sel) != N_CAMERAS:
            raise SystemExit(f"{csv_path}: expected {N_CAMERAS} cameras for case '{case}' "
                             f"(K={VARIANT['K']}, lam={VARIANT['lam']}), found {len(sel)}")

        def col(name: str) -> np.ndarray:
            return np.array([float(r[name]) for r in sel])

        stats[case] = {
            "rfse_median": float(np.median(col("rfse_380_780"))),
            "rfse_p90": float(np.percentile(col("rfse_380_780"), 90)),
            "emu_median": float(np.median(col("dE00_median_all"))),
            "emu_p95_median": float(np.median(col("dE00_p95_all"))),
            "rfse_prior_median": float(np.median(col("rfse_mean_prior"))),
            "emu_prior_median": float(np.median(col("dE00_median_mean_prior"))),
        }
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=CSV_DEFAULT, help="loo_results.csv from validate_loo")
    ap.add_argument("--tol", type=float, default=0.1, help="allowed absolute difference (default 0.1)")
    args = ap.parse_args(argv)

    if not args.csv.exists():
        print(f"missing {args.csv}; run python3 -m training.alexa.validate_loo first", file=sys.stderr)
        return 2
    stats = load_stats(args.csv)

    failed = 0
    print(f"leave-one-out, K={VARIANT['K']} lam={VARIANT['lam']}, {N_CAMERAS} cameras, tolerance +/-{args.tol}")
    print(f"{'quantity':<52} {'run':>8} {'published':>10} {'diff':>8}  status")
    for label, case, stat, pub, headline in PUBLISHED:
        got = stats[case][stat]
        diff = got - pub
        ok = abs(diff) <= args.tol
        failed += 0 if ok else 1
        tag = "headline" if headline else "published"
        print(f"{label:<52} {got:8.3f} {pub:10.2f} {diff:+8.3f}  {'ok' if ok else 'FAIL'} ({tag})")
    for label, case, stat, bound in GATES:
        got = stats[case][stat]
        ok = got <= bound
        failed += 0 if ok else 1
        print(f"gate {label:<47} {got:8.3f} {bound:10.2f} {'':>8}  {'ok' if ok else 'FAIL'} (docs/13)")

    if failed:
        print(f"{failed} check(s) failed", file=sys.stderr)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

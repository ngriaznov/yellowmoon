"""Compare a regenerated ALEXA curve (``estimate_alexa``) with the shipped one.

    python3 -m training.alexa.estimate_alexa      # ~10 min
    python3 -m training.alexa.compare_estimate

Exits non-zero when any sample differs by more than ``--tol`` (default 2e-5, i.e. a few
units of the 6-decimal rounding plus BLAS-level differences between machines).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from training.common import paths

REGENERATED = paths.OUT / "alexa" / "estimate" / "alexa_alev3_estimate_380_780_5.json"


def load(path: Path) -> np.ndarray:
    doc = json.loads(path.read_text())
    data = doc["spectral_data"]["data"]["main"]
    return np.array([data[k] for k in sorted(data, key=float)], dtype=float)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regenerated", type=Path, default=REGENERATED)
    ap.add_argument("--shipped", type=Path, default=paths.ALEXA_CURVES_JSON)
    ap.add_argument("--tol", type=float, default=2e-5)
    args = ap.parse_args(argv)
    if not args.regenerated.exists():
        print(f"missing {args.regenerated}; run python3 -m training.alexa.estimate_alexa first", file=sys.stderr)
        return 2
    a, b = load(args.regenerated), load(args.shipped)
    if a.shape != b.shape:
        print(f"shape differs: {a.shape} vs {b.shape}", file=sys.stderr)
        return 1
    d = float(np.max(np.abs(a - b)))
    print(f"max |regenerated - shipped| = {d:.3e} (tolerance {args.tol:g})")
    return 0 if d <= args.tol else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Cross-check vectors for the ARRI colour step (docs/13: "ARRI colour step", docs/02: Tests).

OpenColorIO ships built-in transforms for ARRI's published formulas (curve and gamut matrix, no ARRI
data files): `ARRI_LOGC4_to_ACES2065-1` (LogC4 / AWG4 to AP0) and
`ARRI_ALEXA-LOGC-EI800-AWG_to_ACES2065-1` (LogC3 EI 800 / AWG3 to AP0). This script pushes seeded
random code values through both with the float32 CPU processor (optimisation off) and writes inputs and outputs to
crates/ym-color/tests/data/lut/arri_builtin_vectors.json. `crates/ym-color/tests/arri_vectors.rs`
compares our f64 chain against them. OpenColorIO evaluates the log curves with float32 exp/pow
approximations (about 2.5e-4 relative even with optimisation off), so this is a gross-error check at
1e-3; the tight (1e-13) cross-check is against colour-science in float64 (see tests/logc.rs).

Usage:  python3 tools/vectors/lut_arri_builtin_vectors.py [--check]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import PyOpenColorIO as ocio

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "crates" / "ym-color" / "tests" / "data" / "lut" / "arri_builtin_vectors.json"
SEED = 20261013
STYLES = {
    "logc4_awg4_to_ap0": ("ARRI_LOGC4_to_ACES2065-1", 0.12, 0.98),
    "logc3_ei800_awg3_to_ap0": ("ARRI_ALEXA-LOGC-EI800-AWG_to_ACES2065-1", 0.15, 0.90),
}


def run(style: str, pts: np.ndarray) -> np.ndarray:
    cfg = ocio.Config.CreateRaw()
    t = ocio.BuiltinTransform()
    t.setStyle(style)
    # No optimisation: the default processor adds lossy fast-math approximations on top of float32.
    proc = cfg.getProcessor(t).getOptimizedCPUProcessor(ocio.OPTIMIZATION_NONE)
    px = np.ascontiguousarray(pts.astype(np.float32)).copy()
    proc.applyRGB(px)
    return px.astype(np.float64)


def build() -> dict:
    rng = np.random.default_rng(SEED)
    cases = {}
    for name, (style, lo, hi) in STYLES.items():
        pts = np.round(rng.uniform(lo, hi, (400, 3)), 9)
        grey = np.round(np.linspace(lo, hi, 25)[:, None] * np.ones(3), 9)
        pts = np.concatenate([pts, grey])
        cases[name] = {"style": style, "inputs": pts.tolist(), "outputs": run(style, pts).tolist()}
    return {
        "generator": "tools/vectors/lut_arri_builtin_vectors.py",
        "ocio_version": ocio.GetVersion(),
        "note": "OpenColorIO float32 results of its built-in ARRI transforms (formulas only).",
        "cases": cases,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    doc = build()
    text = json.dumps(doc, separators=(",", ":")) + "\n"
    if args.check:
        same = OUT.exists() and all(
            (old := json.loads(OUT.read_text())["cases"].get(k)) is not None
            and old["inputs"] == v["inputs"]
            and np.allclose(np.array(old["outputs"]), np.array(v["outputs"]), rtol=1e-5, atol=1e-6)
            for k, v in doc["cases"].items()
        )
        if not same:
            print("DIFFERENT: arri_builtin_vectors.json", file=sys.stderr)
            return 1
        print("arri builtin vectors are up to date")
        return 0
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(ROOT)} (OCIO {ocio.GetVersion()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

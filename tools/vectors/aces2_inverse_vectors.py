#!/usr/bin/env python3
"""Generate inverse ACES 2.0 output-transform test vectors with OpenColorIO.

Writes crates/ym-color/tests/data/aces2/srgb_inverse.json: display-linear Rec.709 RGB (1.0 is the
100 nit SDR white) and the linear ACES2065-1 (AP0) triple that OpenColorIO's built-in ACES 2.0
transform "ACES-OUTPUT - ACES2065-1_to_CIE-XYZ-D65 - SDR-100nit-REC709_2.0" returns when it is run
in the inverse direction (float32 CPU renderer, no optimisation). The Rec.709 triple goes in as CIE
XYZ (D65) through the Rec.709 primaries matrix, which is the form that built-in takes.

Only the sRGB output is generated: it is the one the DNG export inverts (`docs/09`, "What a DNG
carries"), and the inverse of the other outputs differs from it in the limiting primaries only.

Inputs (all deterministic):
  neutrals     60 display greys from 0.0005 to 0.99, log-spaced
  white_ramp   display greys from 0.99 up to 1.0 (where the scene value runs away; compared
               loosely, relative)
  cube         a 9 x 9 x 9 cube of display values from 0 to 0.98
  random       seeded in-gamut display colours, log-uniform in luminance
  boundary     colours on and next to the gamut boundary (one or two channels at 0)

Usage
-----
  python3 tools/vectors/aces2_inverse_vectors.py            # (re)write the committed file
  python3 tools/vectors/aces2_inverse_vectors.py --check    # fail if the committed file is stale
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import PyOpenColorIO as ocio

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "crates" / "ym-color" / "tests" / "data" / "aces2" / "srgb_inverse.json"
SCHEMA = "yellowmoon/aces2-inverse-vectors/1"
BUILTIN = "ACES-OUTPUT - ACES2065-1_to_CIE-XYZ-D65 - SDR-100nit-REC709_2.0"

D65 = (0.3127, 0.3290)
REC709 = ((0.640, 0.330), (0.300, 0.600), (0.150, 0.060), D65)


def xy_to_xyz(xy):
    x, y = xy
    return np.array([x / y, 1.0, (1.0 - x - y) / y])


def rgb_to_xyz_matrix(gamut):
    """Column-vector RGB -> XYZ matrix (white = Y 1) from primaries and white."""
    r, g, b, w = gamut
    prim = np.array([xy_to_xyz(r), xy_to_xyz(g), xy_to_xyz(b)]).T
    s = np.linalg.solve(prim, xy_to_xyz(w))
    return prim * s


def inputs() -> tuple[np.ndarray, list[dict]]:
    rows: list[np.ndarray] = []
    groups: list[dict] = []

    def add(name: str, values: np.ndarray) -> None:
        groups.append({"name": name, "start": sum(len(r) for r in rows), "count": len(values)})
        rows.append(values)

    g = np.geomspace(0.0005, 0.99, 60)
    add("neutrals", np.stack([g, g, g], axis=1))

    w = np.array([0.99, 0.992, 0.994, 0.996, 0.998, 0.999, 0.9995, 1.0])
    add("white_ramp", np.stack([w, w, w], axis=1))

    axis = np.linspace(0.0, 0.98, 9)
    cube = np.array([[r, gr, b] for r in axis for gr in axis for b in axis])
    add("cube", cube)

    rng = np.random.default_rng(20261014)
    lum = np.exp(rng.uniform(np.log(0.002), np.log(0.9), 400))
    chroma = rng.uniform(0.0, 1.0, (400, 3))
    chroma /= chroma.max(axis=1, keepdims=True)
    # Scale so the brightest channel is `lum` of the way up, then mix toward grey by a random amount.
    colours = chroma * lum[:, None]
    mix = rng.uniform(0.0, 1.0, (400, 1))
    grey = np.repeat(colours.mean(axis=1, keepdims=True), 3, axis=1)
    add("random", np.clip(grey + (colours - grey) * mix, 0.0, 0.97))

    edge = []
    for level in (0.02, 0.1, 0.3, 0.6, 0.9):
        for i in range(3):
            c = np.zeros(3)
            c[i] = level
            edge.append(c.copy())
            c2 = np.full(3, level)
            c2[i] = 0.0
            edge.append(c2)
            c3 = np.full(3, level * 0.5)
            c3[i] = level
            edge.append(c3)
    add("boundary", np.array(edge))
    return np.concatenate(rows, axis=0), groups


def build() -> dict:
    if tuple(int(p) for p in ocio.__version__.split(".")[:2]) < (2, 4):
        sys.exit(f"OpenColorIO 2.4 or newer is needed (found {ocio.__version__})")
    display, groups = inputs()
    to_xyz = rgb_to_xyz_matrix(REC709)
    xyz = (display @ to_xyz.T).astype(np.float32)

    cfg = ocio.Config.CreateRaw()
    transform = ocio.BuiltinTransform(style=BUILTIN)
    transform.setDirection(ocio.TRANSFORM_DIR_INVERSE)
    proc = cfg.getProcessor(transform).getDefaultCPUProcessor()
    ap0 = xyz.copy()
    for row in ap0:
        proc.applyRGB(row)

    return {
        "schema": SCHEMA,
        "output": "srgb",
        "description": "Inverse of the sRGB export: display-linear Rec.709 (1.0 = 100 nit white) to linear AP0",
        "generator": {
            "script": "tools/vectors/aces2_inverse_vectors.py",
            "note": "inputs are float64 display values written as float32; expected values are OpenColorIO float32 CPU output of the built-in run in the inverse direction",
        },
        "ocio": {
            "version": ocio.__version__,
            "builtin": BUILTIN,
            "direction": "inverse",
            "optimization": "none",
            "precision": "float32",
        },
        "peak_luminance_nits": 100,
        "count": int(len(display)),
        "groups": groups,
        "input_display_linear": [[float(np.float32(v)) for v in row] for row in display],
        "expected_ap0": [[float(v) for v in row] for row in ap0],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="fail if the committed file is stale")
    args = ap.parse_args()
    doc = build()
    if args.check:
        old = json.loads(OUT.read_text())
        a = np.array(old["expected_ap0"])
        b = np.array(doc["expected_ap0"])
        ok = old["count"] == doc["count"] and np.allclose(a, b, rtol=1e-4, atol=1e-6)
        print("stale" if not ok else "up to date")
        return 0 if ok else 1
    OUT.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    print(f"wrote {OUT.relative_to(REPO)} ({doc['count']} vectors, OpenColorIO {ocio.__version__})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

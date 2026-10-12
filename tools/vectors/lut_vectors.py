#!/usr/bin/env python3
"""Generate the LUT parity vectors for ym-color (docs/13: "LUT path", docs/02: Tests).

Writes, under crates/ym-color/tests/data/lut/:

  * synthetic .cube files made here (our own functions; no ARRI or other third-party LUT data), and
  * lut_vectors.json: for every .cube, a list of inputs and the outputs of OpenColorIO's
    *tetrahedral* interpolation (PyOpenColorIO >= 2.4, float32 CPU processor).

The Rust test `crates/ym-color/tests/lut.rs` parses the same .cube files with ym-color and checks its
f64 tetrahedral interpolation against these outputs (gate: at most 1 code value at 10 bits, i.e.
1/1023 of full scale; the measured maximum is reported by the test).

Usage:  python3 tools/vectors/lut_vectors.py [--check]

Deterministic: fixed seed, fixed function definitions, values quantised to the digits written to the
files. `--check` regenerates in memory and compares with what is on disk (exit 1 on any difference; the OCIO
outputs are compared to 1e-6 because float32 code paths differ between CPUs), so CI can keep the
committed files honest. Requires numpy and PyOpenColorIO.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import PyOpenColorIO as ocio

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "crates" / "ym-color" / "tests" / "data" / "lut"
SEED = 20261012


# --------------------------------------------------------------------------------------------------
# Synthetic transforms (all ours). They are smooth but strongly non-linear and mix the channels, so
# tetrahedral and trilinear interpolation give visibly different answers between grid points.
# --------------------------------------------------------------------------------------------------


def warp(rgb: np.ndarray) -> np.ndarray:
    """Gamma, a channel-mixing matrix, a luminance-dependent saturation and a gentle ripple.

    The saturation term and the products make the function non-separable (cross terms between the
    channels), which is what makes tetrahedral and trilinear interpolation differ. Input roughly in [0, 1].
    """
    x = np.clip(rgb, 0.0, 1.0)
    g = np.power(x, [0.8, 1.1, 1.35])
    m = np.array([[0.82, 0.14, 0.04], [0.06, 0.88, 0.06], [0.02, 0.18, 0.80]])
    y = g @ m.T
    y = y + 0.12 * np.stack([x[..., 1] * x[..., 2], x[..., 0] * x[..., 2], -x[..., 0] * x[..., 1]], axis=-1)
    lum = y.mean(axis=-1, keepdims=True)
    y = lum + (y - lum) * (1.0 + 0.6 * np.sin(np.pi * np.clip(lum, 0.0, 1.0)))
    ripple = 0.035 * np.sin(2.0 * np.pi * (x[..., [1, 2, 0]] * 1.5 + 0.1))
    return y + ripple


def ripple(rgb: np.ndarray) -> np.ndarray:
    """Higher-frequency features for the 33-point cube."""
    x = np.clip(rgb, 0.0, 1.0)
    base = np.stack(
        [
            0.9 * x[..., 0] ** 0.9 + 0.1 * x[..., 1] * x[..., 2],
            0.05 + 0.9 * x[..., 1] ** 1.2 - 0.05 * x[..., 0],
            0.95 * x[..., 2] ** 0.85 + 0.05 * np.sin(3.0 * np.pi * x[..., 0]),
        ],
        axis=-1,
    )
    lum = x @ np.array([0.3, 0.5, 0.2])
    return base + 0.04 * np.sin(8.0 * lum)[..., None] * np.array([1.0, -0.7, 0.4])


def extreme(rgb: np.ndarray) -> np.ndarray:
    """Used on a 2-point cube: only the eight corners matter. Values leave [0, 1]."""
    return np.stack([rgb[..., 0] * 1.7 - 0.5, rgb[..., 1] * rgb[..., 2] + 0.3 * rgb[..., 0], 1.0 - rgb[..., 0] * rgb[..., 1]], axis=-1)


def shaper(x: np.ndarray) -> np.ndarray:
    """Per-channel 1D shaper over [-0.05, 1.1] (log-like, different per channel). Output in [0, 1]."""
    t = np.clip((x + 0.05) / 1.15, 0.0, 1.0)
    return np.stack([np.log1p(7.0 * t[..., 0]) / np.log(8.0), t[..., 1] ** 0.7, 1.0 - (1.0 - t[..., 2]) ** 1.6], axis=-1)


def grid(n: int, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """All grid coordinates of an n^3 cube, red fastest, shape (n^3, 3)."""
    axis = [np.linspace(lo[c], hi[c], n) for c in range(3)]
    b, g, r = np.meshgrid(axis[2], axis[1], axis[0], indexing="ij")
    return np.stack([r.ravel(), g.ravel(), b.ravel()], axis=-1)


def q(a: np.ndarray, digits: int) -> np.ndarray:
    """Quantise to the number of decimals written to the file, so file and vectors agree exactly."""
    return np.round(a, digits)


def fmt_row(v: np.ndarray, digits: int) -> str:
    return " ".join(f"{x:.{digits}f}" for x in v)


# --------------------------------------------------------------------------------------------------
# File writers
# --------------------------------------------------------------------------------------------------


def iridas_cube(title: str, n: int, lo, hi, data: np.ndarray, digits: int) -> str:
    lines = [f'TITLE "{title}"', f"LUT_3D_SIZE {n}", f"DOMAIN_MIN {fmt_row(np.array(lo), 4)}", f"DOMAIN_MAX {fmt_row(np.array(hi), 4)}"]
    lines += [fmt_row(row, digits) for row in data]
    return "\n".join(lines) + "\n"


def resolve_cube(n1: int, r1, n3: int, r3, data1: np.ndarray, data3: np.ndarray, digits: int) -> str:
    lines = [
        "# synthetic 1D shaper + 3D cube (Resolve dialect), Yellowmoon test data",
        f"LUT_1D_SIZE {n1}",
        f"LUT_3D_SIZE {n3}",
        f"LUT_1D_INPUT_RANGE {r1[0]} {r1[1]}",
        f"LUT_3D_INPUT_RANGE {r3[0]} {r3[1]}",
    ]
    lines += [fmt_row(row, digits) for row in data1]
    lines += [fmt_row(row, digits) for row in data3]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------------------
# Test points
# --------------------------------------------------------------------------------------------------


def test_points(rng: np.random.Generator, n: int, lo: np.ndarray, hi: np.ndarray, count: int = 400) -> np.ndarray:
    span = hi - lo
    pts = [lo + rng.random((count, 3)) * span]
    # Grid corners, centre and the cube diagonal.
    corners = np.array([[lo[0] if i & 1 == 0 else hi[0], lo[1] if i & 2 == 0 else hi[1], lo[2] if i & 4 == 0 else hi[2]] for i in range(8)])
    pts.append(corners)
    pts.append(lo + span * np.linspace(0.0, 1.0, 21)[:, None] * np.ones(3))
    # Points exactly on grid nodes and on cell faces/edges/diagonals (tie cases of the tetrahedron choice).
    cell = span / (n - 1)
    node = lo + np.round(rng.random((20, 3)) * (n - 1)) * cell
    pts.append(node)
    base = lo + np.floor(rng.random((30, 3)) * (n - 1)) * cell
    f = rng.random(30)[:, None]
    for tie in ([1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]):
        pts.append(base + cell * f * np.array(tie, dtype=float))
    # Slightly and well outside the domain: both must clamp.
    pts.append(lo - 0.05 * span * rng.random((25, 3)))
    pts.append(hi + 0.3 * span * rng.random((25, 3)))
    return np.concatenate(pts, axis=0)


def ocio_apply(path: Path, pts: np.ndarray) -> np.ndarray:
    cfg = ocio.Config.CreateRaw()
    ft = ocio.FileTransform()
    ft.setSrc(str(path))
    ft.setInterpolation(ocio.INTERP_TETRAHEDRAL)
    proc = cfg.getProcessor(ft).getDefaultCPUProcessor()
    px = np.ascontiguousarray(pts.astype(np.float32).reshape(-1, 3)).copy()
    proc.applyRGB(px)  # in place, float32
    return px.astype(np.float64)


def same_cases(a: list, b: list, tol: float = 1e-6) -> bool:
    """Inputs must match exactly; OCIO outputs only to `tol` (float32 code paths differ between CPUs)."""
    if len(a) != len(b):
        return False
    for ca, cb in zip(a, b):
        if ca["file"] != cb["file"] or ca["inputs"] != cb["inputs"]:
            return False
        if not np.allclose(np.array(ca["outputs"]), np.array(cb["outputs"]), rtol=0.0, atol=tol):
            return False
    return True


def build() -> tuple[dict[str, str], dict]:
    rng = np.random.default_rng(SEED)
    files: dict[str, str] = {}

    # name -> (n, lo, hi, function, digits, title)
    specs = {
        "identity_9.cube": (9, [0, 0, 0], [1, 1, 1], lambda x: x, 8, "identity 9"),
        "warp_17.cube": (17, [0, 0, 0], [1, 1, 1], warp, 8, "warp 17"),
        "ripple_33.cube": (33, [0, 0, 0], [1, 1, 1], ripple, 6, "ripple 33"),
        "domain_9.cube": (9, [-0.1, -0.2, 0.0], [1.5, 1.2, 2.0], lambda x: warp((x - np.array([-0.1, -0.2, 0.0])) / np.array([1.6, 1.4, 2.0])), 8, "domain 9"),
        "extreme_2.cube": (2, [0, 0, 0], [1, 1, 1], extreme, 8, "extreme 2"),
    }
    meta = []
    for name, (n, lo, hi, fn, digits, title) in specs.items():
        lo_a, hi_a = np.array(lo, dtype=float), np.array(hi, dtype=float)
        data = q(fn(grid(n, lo_a, hi_a)), digits)
        files[name] = iridas_cube(title, n, lo, hi, data, digits)
        meta.append((name, n, lo_a, hi_a))

    # Resolve dialect: a 1D shaper followed by a 3D cube.
    n1, r1 = 64, (-0.05, 1.1)
    n3, r3 = 9, (0.0, 1.0)
    xs = np.linspace(r1[0], r1[1], n1)[:, None] * np.ones(3)
    data1 = q(shaper(xs), 8)
    data3 = q(warp(grid(n3, np.zeros(3), np.ones(3))), 8)
    files["shaper_3d_9.cube"] = resolve_cube(n1, r1, n3, r3, data1, data3, 8)
    meta.append(("shaper_3d_9.cube", n3, np.full(3, r1[0]), np.full(3, r1[1])))

    return files, {"meta": meta, "rng": rng}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="compare with the files on disk instead of writing")
    args = ap.parse_args()

    files, info = build()
    rng: np.random.Generator = info["rng"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_written: list[Path] = []
    for name, text in files.items():
        path = OUT_DIR / name
        if args.check:
            if not path.exists() or path.read_text() != text:
                print(f"DIFFERENT: {name}", file=sys.stderr)
                return 1
        else:
            path.write_text(text)
            tmp_written.append(path)

    cases = []
    for name, n, lo, hi in info["meta"]:
        path = OUT_DIR / name
        if args.check and not path.exists():
            print(f"MISSING: {name}", file=sys.stderr)
            return 1
        pts = q(test_points(rng, n, lo, hi), 9)
        out = ocio_apply(path, pts)
        cases.append({"file": name, "inputs": pts.tolist(), "outputs": out.tolist()})

    doc = {
        "generator": "tools/vectors/lut_vectors.py",
        "ocio_version": ocio.GetVersion(),
        "interpolation": "tetrahedral",
        "note": "Outputs are OpenColorIO float32 CPU results for FileTransform(INTERP_TETRAHEDRAL) on our own synthetic .cube files.",
        "cases": cases,
    }
    text = json.dumps(doc, separators=(",", ":")) + "\n"
    out_path = OUT_DIR / "lut_vectors.json"
    if args.check:
        if not out_path.exists() or not same_cases(json.loads(out_path.read_text())["cases"], json.loads(text)["cases"]):
            print("DIFFERENT: lut_vectors.json", file=sys.stderr)
            return 1
        print("lut vectors are up to date")
        return 0
    out_path.write_text(text)
    for c in cases:
        print(f"{c['file']:20s} {len(c['inputs'])} points")
    print(f"wrote {len(files)} cubes and {out_path.relative_to(ROOT)} (OCIO {ocio.GetVersion()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

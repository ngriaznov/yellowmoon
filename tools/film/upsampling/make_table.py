#!/usr/bin/env python3
"""Generates Yellowmoon's spectral upsampling table (docs/24 section 3.3): data/film/upsampling/.

    python3 tools/film/upsampling/make_table.py [--out DIR] [--cie DIR] [--quiet]

For every cell of a 256 x 256 grid over the CIE 1931 chromaticity square it finds the three coefficients (c0, c1, c2) of the
sigmoid-of-a-quadratic spectrum of Jakob and Hanika

    s(l) = sigmoid(c0 t^2 + c1 t + c2),   t = (l - 380 nm) / 400 nm,   sigmoid(z) = 1/2 + z / (2 sqrt(1 + z^2)),

whose chromaticity under the CIE 1931 2 degree observer is the cell's (x, y). Written from the paper (Jakob and Hanika, "A Low-
Dimensional Function Space for Efficient Spectral Upsampling", Eurographics 2019) and the CIE tables in ``data/film/cie/``. No
other input is read: no film profile, no table of anyone else (docs/24 section 3.1). Nothing here runs in the app or in ``ym``.

Method, as the paper does it, with one change: the paper fits a reflectance under an illuminant to an RGB triple, three equations
for three unknowns. We fit an emission to a chromaticity, two equations, so a third condition is needed: the luminance. Each cell
gets a target luminance Y_t(x, y), smooth and falling from 0.5 at the white point to near 0 at the spectral locus (the sigmoid is
bounded by 1, so saturated colours cannot be bright), and the three coefficients solve XYZ(s) = Y_t (x/y, 1, z/y) in CIELAB, with a
Levenberg-Marquardt iteration and an analytic Jacobian. The cells are visited in order of their distance from the white point and
each starts from its nearest solved neighbour. A cell that does not converge at Y_t is retried at a fraction of it; one that
never converges is left for the inpainting. Cells outside the locus are not tried.

Unreached cells (outside the locus, and the few inside it that no sigmoid spectrum reaches) get their coefficients from a Laplace fill
of the solved cells (a sparse linear solve), and flag 1. Y_s is filled the same way, which keeps it positive (maximum principle).

Deterministic: fixed visiting order, no random numbers, plain float64, one sparse direct solve. Two runs on one machine give the same
bytes; another machine's BLAS may differ in the last bit of a float64, which float32 storage hides almost always. The SHA-256 of the
inputs and outputs is written to table.json and to SHA256SUMS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

#: Version of this generator, written to table.json. Change it when the output could change.
GENERATOR_VERSION = "1.0.0"

REPO = Path(__file__).resolve().parents[3]

#: The grid: cells at x = ix / (N - 1), y = iy / (N - 1), so the corners are exact and bilinear lookup needs no half-cell shift.
N = 256
#: The wavelength range and step of the table's spectra (the profiles' grid) and the t normalisation.
LAMBDA_MIN_NM = 380
LAMBDA_MAX_NM = 780
LAMBDA_STEP_NM = 5
#: Target luminance at the white point and the exponent of its fall towards the locus: Y_t = Y_WHITE (1 - r)^FALL + Y_FLOOR.
Y_WHITE = 0.5
Y_FALL = 1.5
Y_FLOOR = 0.004
#: Fractions of Y_t tried in turn when a cell does not converge at Y_t.
Y_RETRY = (1.0, 0.7, 0.5, 0.3, 0.15, 0.07, 0.03, 0.01, 0.003)
#: A fit is accepted when the CIELAB residual is below this (about 1e-9 in xy).
LAB_TOL = 1e-7
LM_MAX_ITER = 200
#: Passes over the cells inside the locus: a cell that failed may start from a neighbour that a later cell of the same pass solved.
MAX_PASSES = 4
#: Error bound the checks use for a solved cell (docs/24 section 3.3, the task's acceptance).
XY_BOUND = 1e-4

FLAG_SOLVED = 0
FLAG_INPAINTED = 1


def load_observer(cie_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """The 1931 2 degree observer at 1 nm (rows: wavelength, xbar, ybar, zbar) and the 5 nm samples 380..780 by decimation."""
    rows = []
    for line in (cie_dir / "CIE_xyz_1931_2deg.csv").read_text(encoding="utf-8").splitlines():
        fields = line.split(",")
        if len(fields) == 4:
            rows.append([float(f) for f in fields])
    fine = np.array(rows, dtype=np.float64)
    lam = fine[:, 0]
    keep = (lam >= LAMBDA_MIN_NM) & (lam <= LAMBDA_MAX_NM) & ((lam - LAMBDA_MIN_NM) % LAMBDA_STEP_NM == 0)
    return fine, fine[keep]


class Observer:
    """The 81 samples of the observer, normalised so that a flat spectrum of 1 has XYZ = (1, 1, 1) up to the tables' own ~1e-4."""

    def __init__(self, cie_dir: Path):
        self.fine, coarse = load_observer(cie_dir)
        self.lam = coarse[:, 0]
        self.t = (self.lam - LAMBDA_MIN_NM) / (LAMBDA_MAX_NM - LAMBDA_MIN_NM)
        self.cmf = coarse[:, 1:4]
        #: Sum of ybar: the divisor that makes Y of a flat 1 equal to 1.
        self.norm = float(self.cmf[:, 1].sum())
        self.m = self.cmf / self.norm
        self.white_xyz = self.m.sum(axis=0)
        #: Basis of the quadratic, (81, 3).
        self.basis = np.stack([self.t**2, self.t, np.ones_like(self.t)], axis=1)

    def spectrum(self, c: np.ndarray) -> np.ndarray:
        """s(l) for coefficients c, shape (..., 3) -> (..., 81)."""
        z = c @ self.basis.T
        return 0.5 + z / (2.0 * np.sqrt(1.0 + z * z))

    def xyz(self, c: np.ndarray) -> np.ndarray:
        """XYZ of the spectrum, in the units where a flat 1 is (1, 1, 1)."""
        return self.spectrum(c) @ self.m

    def xy(self, c: np.ndarray) -> np.ndarray:
        """Chromaticity of the spectrum, shape (..., 2)."""
        xyz = self.xyz(c)
        total = xyz.sum(axis=-1, keepdims=True)
        return xyz[..., :2] / total


# ---------------------------------------------------------------------------------------------------------------- CIELAB

_EPS = 216.0 / 24389.0
_KAPPA = 24389.0 / 27.0


def _f(t: np.ndarray) -> np.ndarray:
    return np.where(t > _EPS, np.cbrt(t), (_KAPPA * t + 16.0) / 116.0)


def _fp(t: np.ndarray) -> np.ndarray:
    return np.where(t > _EPS, 1.0 / (3.0 * np.cbrt(np.maximum(t, _EPS)) ** 2), _KAPPA / 116.0)


def lab_and_jacobian(xyz: np.ndarray, white: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CIELAB of one XYZ (3,) and its 3 x 3 Jacobian with respect to XYZ."""
    r = xyz / white
    f = _f(r)
    fp = _fp(r) / white
    lab = np.array([116.0 * f[1] - 16.0, 500.0 * (f[0] - f[1]), 200.0 * (f[1] - f[2])])
    jac = np.array(
        [
            [0.0, 116.0 * fp[1], 0.0],
            [500.0 * fp[0], -500.0 * fp[1], 0.0],
            [0.0, 200.0 * fp[1], -200.0 * fp[2]],
        ]
    )
    return lab, jac


def lab_only(xyz: np.ndarray, white: np.ndarray) -> np.ndarray:
    """CIELAB of one XYZ (3,)."""
    f = _f(xyz / white)
    return np.array([116.0 * f[1] - 16.0, 500.0 * (f[0] - f[1]), 200.0 * (f[1] - f[2])])


# ---------------------------------------------------------------------------------------------------------------- the solve


def solve_cell(obs: Observer, c_start: np.ndarray, target_xyz: np.ndarray) -> tuple[np.ndarray, float]:
    """Levenberg-Marquardt on the CIELAB difference between XYZ(c) and ``target_xyz``. Returns the coefficients and the residual."""
    white = obs.white_xyz
    target_lab = lab_only(target_xyz, white)
    basis = obs.basis
    m = obs.m
    c = c_start.copy()
    damping = 1e-3

    def evaluate(cc: np.ndarray):
        z = basis @ cc
        root = np.sqrt(1.0 + z * z)
        s = 0.5 + z / (2.0 * root)
        xyz = s @ m
        lab, dlab = lab_and_jacobian(xyz, white)
        res = lab - target_lab
        return z, root, xyz, res, dlab

    z, root, xyz, res, dlab = evaluate(c)
    cost = float(res @ res)
    for _ in range(LM_MAX_ITER):
        if cost < LAB_TOL**2:
            break
        ds = 0.5 / (root**3)
        dxyz = m.T @ (ds[:, None] * basis)
        jac = dlab @ dxyz
        jtj = jac.T @ jac
        grad = jac.T @ res
        improved = False
        for _ in range(12):
            try:
                step = np.linalg.solve(jtj + damping * np.diag(np.diag(jtj) + 1e-12), -grad)
            except np.linalg.LinAlgError:
                damping *= 10.0
                continue
            trial = c + step
            z2, root2, xyz2, res2, dlab2 = evaluate(trial)
            cost2 = float(res2 @ res2)
            if np.isfinite(cost2) and cost2 < cost:
                c, z, root, xyz, res, dlab, cost = trial, z2, root2, xyz2, res2, dlab2, cost2
                damping = max(damping * 0.3, 1e-12)
                improved = True
                break
            damping *= 4.0
        if not improved:
            break
    return c, float(np.sqrt(cost))


# ---------------------------------------------------------------------------------------------------------------- the grid


def locus_polygon(obs: Observer) -> np.ndarray:
    """The spectral locus 380..780 nm at 1 nm as (x, y) vertices, closed by the purple line. Convex."""
    rows = obs.fine[(obs.fine[:, 0] >= LAMBDA_MIN_NM) & (obs.fine[:, 0] <= LAMBDA_MAX_NM)]
    xyz = rows[:, 1:4]
    return xyz[:, :2] / xyz.sum(axis=1, keepdims=True)


def inside_polygon(px: np.ndarray, py: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Even-odd rule, vectorised over points."""
    inside = np.zeros(px.shape, dtype=bool)
    n = len(poly)
    for k in range(n):
        x1, y1 = poly[k]
        x2, y2 = poly[(k + 1) % n]
        crosses = (y1 > py) != (y2 > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
        inside ^= crosses & (px < x_at)
    return inside


def relative_radius(px: np.ndarray, py: np.ndarray, poly: np.ndarray, white_xy: np.ndarray) -> np.ndarray:
    """r in [0, 1]: distance from the white point over the distance to the locus along the same ray (the locus is convex)."""
    ang_v = np.arctan2(poly[:, 1] - white_xy[1], poly[:, 0] - white_xy[0])
    rad_v = np.hypot(poly[:, 0] - white_xy[0], poly[:, 1] - white_xy[1])
    order = np.argsort(ang_v, kind="stable")
    ang_s = np.concatenate([ang_v[order] - 2 * np.pi, ang_v[order], ang_v[order] + 2 * np.pi])
    rad_s = np.concatenate([rad_v[order]] * 3)
    ang = np.arctan2(py - white_xy[1], px - white_xy[0])
    boundary = np.interp(ang, ang_s, rad_s)
    return np.clip(np.hypot(px - white_xy[0], py - white_xy[1]) / boundary, 0.0, 1.0)


def target_luminance(r: np.ndarray) -> np.ndarray:
    """The luminance a cell is fitted at, from its relative radius."""
    return Y_WHITE * (1.0 - r) ** Y_FALL + Y_FLOOR


def solve_grid(obs: Observer, quiet: bool = False, n: int = N):
    """Solves every cell inside the locus that a sigmoid spectrum reaches. Returns (coeffs[N,N,3], Y_s[N,N], solved[N,N], inside[N,N])."""
    axis = np.arange(n) / (n - 1)
    gx, gy = np.meshgrid(axis, axis)  # [iy][ix]
    poly = locus_polygon(obs)
    inside = inside_polygon(gx, gy, poly)
    white_xy = (obs.white_xyz[:2] / obs.white_xyz.sum())
    r = relative_radius(gx, gy, poly, white_xy)
    y_target = target_luminance(r)

    coeffs = np.zeros((n, n, 3))
    y_s = np.zeros((n, n))
    solved = np.zeros((n, n), dtype=bool)

    iy0 = int(round(white_xy[1] * (n - 1)))
    ix0 = int(round(white_xy[0] * (n - 1)))
    cells = np.argwhere(inside)
    dist = (cells[:, 0] - iy0) ** 2 + (cells[:, 1] - ix0) ** 2
    order = np.lexsort((cells[:, 1], cells[:, 0], dist))
    cells = cells[order]
    neighbours = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
    started = time.time()
    failures = 0
    for pass_number in range(MAX_PASSES):
        before = int(solved.sum())
        failures = 0
        for count, (iy, ix) in enumerate(cells):
            if solved[iy, ix]:
                continue
            x, y = gx[iy, ix], gy[iy, ix]
            if y <= 0.0:
                continue
            start = None
            best = None
            for dy, dx in neighbours:
                jy, jx = iy + dy, ix + dx
                if 0 <= jy < n and 0 <= jx < n and solved[jy, jx]:
                    d = (jy - iy0) ** 2 + (jx - ix0) ** 2
                    if best is None or d < best:
                        best, start = d, coeffs[jy, jx]
            if start is None:
                if pass_number == 0 and count == 0:
                    start = np.zeros(3)
                else:
                    failures += 1
                    continue
            ok = False
            for fraction in Y_RETRY:
                yt = y_target[iy, ix] * fraction
                target = yt * np.array([x / y, 1.0, (1.0 - x - y) / y])
                c, res = solve_cell(obs, start, target)
                if res < 10 * LAB_TOL:
                    ok = True
                    break
            if ok:
                coeffs[iy, ix] = c
                solved[iy, ix] = True
                y_s[iy, ix] = obs.xyz(c)[1]
            else:
                failures += 1
            if not quiet and count % 4000 == 0:
                print(f"  pass {pass_number + 1}: {count}/{len(cells)} cells, {time.time() - started:.0f} s", file=sys.stderr)
        if int(solved.sum()) == before:
            break
    return coeffs, y_s, solved, inside, failures


def laplace_fill(values: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Fills ``values`` (N, N, k) where ``known`` is False by solving the discrete Laplace equation with the known cells as fixed
    values and zero-gradient edges of the square. Exact sparse direct solve, so deterministic."""
    n = known.shape[0]
    index = -np.ones((n, n), dtype=np.int64)
    unknown = np.argwhere(~known)
    index[~known] = np.arange(len(unknown))
    rows, cols, vals = [], [], []
    rhs = np.zeros((len(unknown), values.shape[2]))
    for k, (iy, ix) in enumerate(unknown):
        diag = 0
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            jy, jx = iy + dy, ix + dx
            if not (0 <= jy < n and 0 <= jx < n):
                continue
            diag += 1
            if known[jy, jx]:
                rhs[k] += values[jy, jx]
            else:
                rows.append(k)
                cols.append(index[jy, jx])
                vals.append(-1.0)
        rows.append(k)
        cols.append(k)
        vals.append(float(diag))
    a = sparse.csc_matrix((vals, (rows, cols)), shape=(len(unknown), len(unknown)))
    out = values.copy()
    solution = np.column_stack([spsolve(a, rhs[:, j]) for j in range(values.shape[2])])
    out[~known] = solution
    return out


def build(cie_dir: Path, quiet: bool = False, n: int = N) -> dict:
    """Runs the whole generator and returns the arrays and the numbers."""
    obs = Observer(cie_dir)
    started = time.time()
    coeffs, y_s, solved, inside, failures = solve_grid(obs, quiet, n)
    solve_seconds = time.time() - started
    if not quiet:
        print(f"  inpainting {int((~solved).sum())} cells", file=sys.stderr)
    # What is stored is float32. A solved cell whose float32 coefficients miss the chromaticity by more than XY_BOUND (steep spectra
    # near the locus have coefficients in the tens of thousands) is given up to the inpainting rather than shipped with that error.
    axis = np.arange(n) / (n - 1)
    gx, gy = np.meshgrid(axis, axis)
    stored = coeffs.astype(np.float32).astype(np.float64)
    stored_xy = obs.xy(stored)
    stored_err = np.hypot(stored_xy[..., 0] - gx, stored_xy[..., 1] - gy)
    too_far = solved & (stored_err > XY_BOUND)
    solved = solved & ~too_far
    packed = np.concatenate([coeffs, np.log(np.where(solved, y_s, 1.0))[..., None]], axis=2)
    filled = laplace_fill(packed, solved)
    out_coeffs = np.where(solved[..., None], coeffs, filled[..., :3]).astype(np.float32).astype(np.float64)
    # Y_s is the Y of the stored spectrum, computed as the engine would, for every cell. For a solved cell it equals the solve's;
    # for an inpainted one it is the real Y of the filled spectrum, so that Y_colour / Y_s * s(l) has exactly Y_colour there too.
    check_y = obs.xyz(out_coeffs)[..., 1].astype(np.float32).astype(np.float64)
    reached = obs.xy(out_coeffs)
    err = np.hypot(reached[..., 0] - gx, reached[..., 1] - gy)
    y_final = check_y
    flags = np.where(solved, FLAG_SOLVED, FLAG_INPAINTED).astype(np.uint8)
    return {
        "obs": obs,
        "coeffs": out_coeffs,
        "y_s": y_final,
        "flags": flags,
        "error": err,
        "inside": inside,
        "solved": solved,
        "failures": failures,
        "float32_demoted": int(too_far.sum()),
        "solve_seconds": solve_seconds,
    }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def serialise(result: dict) -> dict[str, bytes]:
    """The data files as bytes: little-endian float32 [y][x][4] = (c0, c1, c2, Y_s), uint8 flags, float32 error."""
    table = np.concatenate([result["coeffs"], result["y_s"][..., None]], axis=2).astype("<f4")
    return {
        "coeffs.f32": table.tobytes(),
        "flags.u8": result["flags"].astype(np.uint8).tobytes(),
        "error.f32": result["error"].astype("<f4").tobytes(),
    }


def table_json(result: dict, files: dict[str, bytes], cie_dir: Path) -> dict:
    """The format record."""
    inputs = {"CIE_xyz_1931_2deg.csv": sha256_file(cie_dir / "CIE_xyz_1931_2deg.csv")}
    generator = Path(__file__)
    return {
        "name": "Yellowmoon spectral upsampling table",
        "generator": "tools/film/upsampling/make_table.py",
        "generator_version": GENERATOR_VERSION,
        "generator_sha256": sha256_file(generator),
        "form": "s(lambda) = sigmoid(c0 t^2 + c1 t + c2); sigmoid(z) = 0.5 + z / (2 sqrt(1 + z^2)); t = (lambda - 380) / 400",
        "grid": {
            "size": int(result["coeffs"].shape[0]),
            "x_range": [0.0, 1.0],
            "y_range": [0.0, 1.0],
            "node_position": f"x = ix / {int(result['coeffs'].shape[0]) - 1}, y = iy / {int(result['coeffs'].shape[0]) - 1} (nodes, corners included)",
            "layout": "coeffs.f32 is [iy][ix][4], iy the outer index, little-endian float32, row-major",
            "channels": ["c0", "c1", "c2", "Y_s"],
        },
        "wavelengths_nm": {"min": LAMBDA_MIN_NM, "max": LAMBDA_MAX_NM, "step": LAMBDA_STEP_NM, "count": 81},
        "t_normalisation": {"offset_nm": LAMBDA_MIN_NM, "span_nm": LAMBDA_MAX_NM - LAMBDA_MIN_NM},
        "observer": {
            "name": "CIE 1931 2 degree",
            "source": "data/film/cie/CIE_xyz_1931_2deg.csv (1 nm, 360..830)",
            "resampling": "decimation at 5 nm steps, 380..780 (exact, the source has 1 nm steps)",
            "normalisation": "XYZ = sum_j s(l_j) cmf(l_j) / sum_j ybar(l_j); a flat spectrum of 1 has XYZ = (1, 1, 1) to 1e-4",
        },
        "Y_s": "Y of s(lambda) in the normalisation above; the engine's spectrum is Y_colour / Y_s * s(lambda)",
        "flags": {"file": "flags.u8", "layout": "[iy][ix] uint8", "0": "solved", "1": "inpainted (outside the locus, or not reached)"},
        "error": {"file": "error.f32", "layout": "[iy][ix] little-endian float32", "meaning": "distance in xy between the cell and the chromaticity of its stored spectrum"},
        "luminance_target": {"y_white": Y_WHITE, "fall_exponent": Y_FALL, "floor": Y_FLOOR, "retry_fractions": list(Y_RETRY)},
        "solver": {"method": "Levenberg-Marquardt in CIELAB, analytic Jacobian, warm start from the nearest solved neighbour", "lab_tolerance": LAB_TOL},
        "inputs_sha256": inputs,
        "outputs_sha256": {name: sha256_bytes(data) for name, data in sorted(files.items())},
        "solved_cells": int(result["solved"].sum()),
        "inpainted_cells": int((~result["solved"]).sum()),
        "cells_inside_locus": int(result["inside"].sum()),
    }


def load_table(folder: Path) -> tuple[np.ndarray, np.ndarray]:
    """Reads ``coeffs.f32`` (float64 [N, N, 4]) and ``flags.u8`` ([N, N]) back."""
    raw = np.fromfile(folder / "coeffs.f32", dtype="<f4")
    n = int(round((raw.size / 4) ** 0.5))
    return raw.reshape(n, n, 4).astype(np.float64), np.fromfile(folder / "flags.u8", dtype=np.uint8).reshape(n, n)


def lookup_spectrum(obs: Observer, table: np.ndarray, x: float, y: float, y_colour: float = 1.0, mode: str = "spectra") -> np.ndarray:
    """The engine's lookup, in float64 (docs/24 section 3.3): the 81-sample spectrum for chromaticity (x, y) and luminance Y_colour.

    ``spectra``: bilinear mix of the four node spectra, each divided by its Y_s; ``coeffs``: bilinear mix of (c0, c1, c2), then the
    spectrum of that, divided by its own Y."""
    n = table.shape[0]
    u, v = x * (n - 1), y * (n - 1)
    ix, iy = min(int(np.floor(u)), n - 2), min(int(np.floor(v)), n - 2)
    fx, fy = u - ix, v - iy
    weights = ((ix, iy, (1 - fx) * (1 - fy)), (ix + 1, iy, fx * (1 - fy)), (ix, iy + 1, (1 - fx) * fy), (ix + 1, iy + 1, fx * fy))
    if mode == "spectra":
        total = np.zeros(len(obs.lam))
        for jx, jy, w in weights:
            node = table[jy, jx]
            total += w * obs.spectrum(node[:3]) / node[3]
        return y_colour * total
    c = sum(w * table[jy, jx, :3] for jx, jy, w in weights)
    s = obs.spectrum(c)
    return y_colour * s / float(s @ obs.m[:, 1])


def write_outputs(result: dict, out_dir: Path, cie_dir: Path) -> None:
    """Writes the data files, table.json and SHA256SUMS. ATTRIBUTION.md and LICENSE are written by hand and left alone."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files = serialise(result)
    for name, data in files.items():
        (out_dir / name).write_bytes(data)
    meta = table_json(result, files, cie_dir)
    (out_dir / "table.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = []
    for path in sorted(p for p in out_dir.iterdir() if p.is_file() and p.name not in ("SHA256SUMS", "ATTRIBUTION.md")):
        lines.append(f"{sha256_file(path)}  {path.name}")
    (out_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPO / "data" / "film" / "upsampling")
    parser.add_argument("--cie", type=Path, default=REPO / "data" / "film" / "cie")
    parser.add_argument("--size", type=int, default=N, help="cells per side; the shipped table is 256 (other sizes are for the study of the error against the size)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    started = time.time()
    result = build(args.cie, args.quiet, args.size)
    write_outputs(result, args.out, args.cie)
    if not args.quiet:
        print(
            f"wrote {args.out}: {int(result['solved'].sum())} solved, {int((~result['solved']).sum())} inpainted cells, "
            f"{time.time() - started:.0f} s",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

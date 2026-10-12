"""Sensor-map fit (docs/02 §C2): white-balanced Sony RGB -> white-balanced ALEXA RGB.

For one light ``E`` (with its ARRI-matrix CCT ``cct_k``):

1. Sony responses ``c_s = S_P . (E o R)`` and ALEXA responses ``c_a = S_A . (E o R)`` of the
   190 rawtoaces training reflectances ``R``, each divided by its own response to the white
   (``R = 1``);
2. the white-preserving 3x3 ``M`` (rows ``[m0, m1, 1 - m0 - m1]``, so ``M . [1,1,1] = [1,1,1]``)
   that minimises the CIE Lab error of the two renderings after ARRI's matrix at ``cct_k``
   to AP0 (Lab with the ACES white), summed over the 190 patches
   (:func:`fit_map_lab`, the **shipped** objective, decision 2026-10-12);
3. one ``(mired, M)`` per reference light, grouped in four classes and sorted by mired.

The previous objective, the linear least squares with weights ``Y^(-2/3)`` (an approximation of
the Lab error, ``sony_profiles.py::fit_map``), stays available as :func:`fit_map_weighted_rgb`:
it is the start of the Lab fit and what ``evaluate.py`` uses to reproduce
``research/alexa-ssf/out/sony_profiles.csv`` bit for bit.

Runtime blend (reference implementation: :func:`blend_map`): ``M(mired) = sum_k p_k M_k(mired)``
with linear interpolation in mired inside a class and clamping at its ends. Unchanged by the
choice of objective.

    python3 -m training.sensor_map.fit [--profile sony-ilce-7rm4] [--gate] [--objective lab|weighted_rgb]
    python3 -m training.sensor_map.fit --stability

prints the fitted tables' summary (the profile files are written by ``build_profiles``), or the
reproducibility study of the Lab fit.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import least_squares

from training.alexa import ssf_core as s
from training.common import paths
from training.sensor_map import lights as lt


# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def alexa_curves() -> np.ndarray:
    """ALEXA estimate S_A, (81, 3), 380-780 nm at 5 nm, each channel peak-normalised.

    Read from the shipped ``data/alexa/alexa_alev3_estimate_380_780_5.json`` (rawtoaces-data
    format, ``main`` R, G, B arrays), the same values as the research CSV.
    """
    doc = json.loads(paths.ALEXA_CURVES_JSON.read_text())
    data = {float(k): v for k, v in doc["spectral_data"]["data"]["main"].items()}
    arr = np.array([data[w] for w in s.WL], dtype=float)
    return arr / arr.max(axis=0, keepdims=True)


def load_curves_raw(path) -> np.ndarray:
    """One Weta curve file, **in the scale of the published data**: (81, 3), 380-780 nm at 5 nm.

    The ``main`` R, G, B columns of the rawtoaces-data JSON, clipped at 0 (measurement noise below
    zero; the same clip ``ssf_core.load_cameras`` and ``ym_color::spectral::SpectralCurves::clipped_at_zero``
    apply) and **not normalised**. This is the camera's raw channel balance, and it is the only scale
    in which a camera white (the response ``S^T E`` of a light, divided by its green component) equals
    what the raw data of that light looks like: the channels of a real sensor are not equally
    sensitive, and the raw white balance (as-shot gains, Pick white, the illuminant estimator, all
    working on the raw mosaic) lives in that ratio.

    The exact scale of the published files: one common factor per camera, chosen so that the green
    channel peaks at exactly 1.0 (all 11 Sony files); red peaks at 0.5675-0.6212 and blue at
    0.7805-0.8678 of that, so the channel ratios are preserved. Only ratios between channels enter
    the locus (``tables.log_chroma`` divides by green), so the common factor is immaterial.

    ``ssf_core.load_cameras`` instead divides each channel by its own peak (red x1.6-1.8, blue x1.15-1.3).
    That is harmless wherever every response is divided by the response to the white of the same
    curves (the sensor-map fit, the ALEXA ``w_A`` ratios) and wrong for anything compared with raw data
    (the locus and the Duv scale, defect of docs/02 C1 found in review).
    """
    doc = json.loads(Path(path).read_text())
    data = {float(k): v for k, v in doc["spectral_data"]["data"]["main"].items()}
    arr = np.array([data[w] for w in s.WL], dtype=float)
    return np.clip(arr, 0.0, None)


@lru_cache(maxsize=1)
def sony_curves_raw() -> dict[str, np.ndarray]:
    """The 11 shipped Sony curves in the published scale: ``{"Sony ILCE-7RM4": (81, 3), ...}``.

    Read from the shipped ``data/cameras/sony/*.json`` (the files the runtime reads through
    ``profile.spectral_curves`` and whose SHA-256 is in ``provenance.curves_source``) with
    :func:`load_curves_raw`: clipped at 0, **not peak-normalised**. Use these for everything that is
    compared with raw camera data (``tables.locus_tables``, ``tables.duv_scale_tables``); the maps and
    ``w_A`` tables keep using :func:`sony_curves`.
    """
    out = {}
    for path in sorted(paths.SONY_CURVES_DIR.glob("Sony_*_380_780_5.json")):
        doc = json.loads(path.read_text())
        hdr = doc["header"]
        out[f"{hdr['manufacturer']} {hdr['model']}"] = load_curves_raw(path)
    return out


@lru_cache(maxsize=1)
def sony_curves() -> dict[str, np.ndarray]:
    """The measured Sony curves {"Sony ILCE-7RM4": (81, 3), ...} from rawtoaces-data, **peak-normalised per channel**.

    Loaded with ``ssf_core.load_cameras`` (clip at 0, peak-normalise per channel), restricted to
    the 11 bodies whose files ship in ``data/cameras/sony/``. Valid for the sensor-map fit and the
    ALEXA ``w_A`` tables (every response there is divided by the response to the white of the same
    curves, so the per-channel scale cancels) and for the research validation; **not** for the camera
    locus, which needs :func:`sony_curves_raw`.
    """
    shipped = {p.name for p in paths.SONY_CURVES_DIR.glob("Sony_*_380_780_5.json")}
    cams = s.load_cameras()
    out = {}
    for name, arr in cams.items():
        if not name.startswith("Sony "):
            continue
        fname = f"Sony_{name.split(' ', 1)[1]}_380_780_5.json"
        if fname in shipped:
            out[name] = arr
    return out


@lru_cache(maxsize=1)
def training_reflectances() -> np.ndarray:
    """rawtoaces 190-patch training set, (81, 190)."""
    return s.training_reflectances()


@lru_cache(maxsize=1)
def arri_matrices() -> tuple[np.ndarray, np.ndarray]:
    """ARRI's camera-native -> AP0 matrices: (ccts (17,), matrices (17, 3, 3))."""
    ccts, mats = s.load_arri_matrices()
    ccts.setflags(write=False)
    mats.setflags(write=False)
    return ccts, mats


# ----------------------------------------------------------------------------
# The fit
# ----------------------------------------------------------------------------
OBJECTIVE_LAB = "lab"
OBJECTIVE_WEIGHTED_RGB = "weighted_rgb"
OBJECTIVES = (OBJECTIVE_LAB, OBJECTIVE_WEIGHTED_RGB)

#: What ``profile.json`` records as ``provenance.sensor_map_objective`` (the shipped objective).
SHIPPED_OBJECTIVE_TEXT = "lab-optimal white-preserving 3x3 (docs/02 §C2, decision 2026-10-12)"

#: Fixed solver options of the Lab fit. MINPACK Levenberg-Marquardt with the analytic Jacobian;
#: tolerances at the floating-point floor (scipy requires them above machine epsilon for "lm").
LAB_SOLVER = {"method": "lm", "xtol": 1e-15, "ftol": 1e-15, "gtol": 1e-15, "max_nfev": 500}
#: Gauss-Newton polish after the solver (analytic Jacobian): at most this many steps, stopping when the
#: largest coefficient change is below ``LAB_POLISH_TOL``. The cost is flat to rounding noise for
#: coefficient changes of ~1e-8, so the solver alone stops anywhere in that valley (up to ~3e-9 from
#: different starts); the polish follows the gradient to the bottom. With it, the 6 coefficients of all
#: 495 fits (11 bodies x 45 lights, ``--stability``) agree to ~1e-14 whether the start is the fixed one, a random
#: perturbation of 1e-3 or the identity, and a repeat run is bit-identical.
LAB_POLISH_STEPS = 8
LAB_POLISH_TOL = 1e-13

_D = 6 / 29  # CIE Lab linear-segment constant


def solver_description() -> str:
    """Solver and library versions for ``provenance.fit_solver``."""
    opts = ", ".join(f"{k}={v!r}" for k, v in LAB_SOLVER.items() if k != "method")
    return (f"scipy {scipy.__version__} optimize.least_squares(method={LAB_SOLVER['method']!r}, analytic Jacobian, "
            f"{opts}) from the Y^(-2/3) weighted-RGB solution, then up to {LAB_POLISH_STEPS} Gauss-Newton steps "
            f"(stop below {LAB_POLISH_TOL:g}); numpy {np.__version__}")


def fit_map_weighted_rgb(S_src: np.ndarray, S_dst: np.ndarray, E: np.ndarray, R_train: np.ndarray) -> np.ndarray:
    """White-preserving 3x3 by weighted linear least squares in RGB: the research objective.

    Weights ``Y^(-2/3)`` approximate the Lab error (as in ``ssf_core.fit_matrix_linear``). Identical
    arithmetic to ``sony_profiles.fit_map`` (which fixed ``S_dst`` to the ALEXA estimate), so the
    numbers reproduce bit for bit; ``evaluate.py`` relies on that. Since 2026-10-12 this is no longer
    the shipped objective but the start of :func:`fit_map_lab`.
    """
    src = s.camera_rgb_wb(S_src, E, R_train)
    dst = s.camera_rgb_wb(S_dst, E, R_train)
    w = np.clip(s.target_XYZ(E, R_train)[:, 1], 1e-3, None) ** (-2 / 3)
    A = np.stack([src[:, 0] - src[:, 2], src[:, 1] - src[:, 2]], axis=1) * w[:, None]
    M = np.zeros((3, 3))
    for r in range(3):
        m, *_ = np.linalg.lstsq(A, (dst[:, r] - src[:, 2]) * w, rcond=None)
        M[r] = [m[0], m[1], 1 - m[0] - m[1]]
    return M


def _white_preserving(p: np.ndarray) -> np.ndarray:
    """3x3 with rows ``[p[2r], p[2r+1], 1 - p[2r] - p[2r+1]]`` from the 6 free coefficients."""
    M = np.empty((3, 3))
    M[:, :2] = p.reshape(3, 2)
    M[:, 2] = 1 - M[:, 0] - M[:, 1]
    return M


def _lab_fprime(t: np.ndarray) -> np.ndarray:
    """Derivative of the CIE Lab nonlinearity ``f`` (``ssf_core._f``), continuous at ``(6/29)^3``."""
    return np.where(t > _D**3, (1 / 3) * np.cbrt(np.maximum(t, 1e-12)) ** -2, 1 / (3 * _D**2))


def fit_map_lab(
    S_src: np.ndarray,
    S_dst: np.ndarray,
    E: np.ndarray,
    cct_k: float,
    R_train: np.ndarray,
    start: np.ndarray | None = None,
) -> np.ndarray:
    """Lab-optimal white-preserving 3x3 taking ``S_src`` white-balanced RGB to ``S_dst`` white-balanced RGB.

    Minimises ``sum_n |Lab(M_arri M c_s[n]) - Lab(M_arri c_a[n])|^2`` over the 6 free coefficients of
    ``M`` (rows ``[m0, m1, 1 - m0 - m1]``), where ``c_s``, ``c_a`` are the white-balanced renderings of
    the training reflectances under ``E``, ``M_arri`` is ARRI's native -> AP0 matrix interpolated at
    ``cct_k`` (linear in mired, clamped to 2000-11000 K), and Lab uses the ACES white
    (``ssf_core.ap0_to_Lab``). Same formulation as the ``lab`` column of ``gate_diagnostics``.

    ``cct_k`` is the light's generating temperature for blackbody and D-series lights and its Ohno (2013)
    CCT for F and LED lights (:attr:`training.sensor_map.lights.Light.cct_k`).

    Deterministic: the start is the :func:`fit_map_weighted_rgb` solution (or ``start``, a 3x3, in
    studies), the solver and its options are fixed (:data:`LAB_SOLVER`, then a Gauss-Newton polish,
    :data:`LAB_POLISH_STEPS`), and the result does not depend on the start to ~1e-14 (``--stability``).
    Raises ``RuntimeError`` if the solver does not converge.
    """
    ccts, mats = arri_matrices()
    M_arri = s.interpolate_matrix(ccts, mats, cct_k)
    src = s.camera_rgb_wb(S_src, E, R_train)
    lab_target = s.ap0_to_Lab(s.camera_rgb_wb(S_dst, E, R_train) @ M_arri.T)
    C = s.AP0_TO_XYZ @ M_arri  # white-balanced ALEXA-native RGB -> XYZ
    # dXYZ/dp, (n, 3, 6), constant in p: d XYZ[n, k] / d p[2r + j] = C[k, r] (src[n, j] - src[n, 2])
    n = src.shape[0]
    dXYZ = np.zeros((n, 3, 6))
    for r in range(3):
        for j in range(2):
            dXYZ[:, :, 2 * r + j] = C[:, r][None, :] * (src[:, j] - src[:, 2])[:, None]

    def residuals(p: np.ndarray) -> np.ndarray:
        return (s.XYZ_to_Lab(src @ _white_preserving(p).T @ C.T) - lab_target).ravel()

    def jacobian(p: np.ndarray) -> np.ndarray:
        XYZ = src @ _white_preserving(p).T @ C.T
        fp = _lab_fprime(XYZ / s.ACES_WHITE_XYZ) / s.ACES_WHITE_XYZ  # (n, 3)
        J = np.zeros((n, 3, 3))  # dLab/dXYZ
        J[:, 0, 1] = 116 * fp[:, 1]
        J[:, 1, 0] = 500 * fp[:, 0]
        J[:, 1, 1] = -500 * fp[:, 1]
        J[:, 2, 1] = 200 * fp[:, 1]
        J[:, 2, 2] = -200 * fp[:, 2]
        return np.einsum("nij,njk->nik", J, dXYZ).reshape(-1, 6)

    M0 = fit_map_weighted_rgb(S_src, S_dst, E, R_train) if start is None else start
    p0 = M0[:, :2].ravel()
    sol = least_squares(residuals, p0, jac=jacobian, **LAB_SOLVER)
    if sol.status <= 0:
        raise RuntimeError(f"Lab fit did not converge (status {sol.status}: {sol.message})")
    p = sol.x
    cost = 0.5 * float(np.sum(residuals(p) ** 2))
    if cost > 0.5 * float(np.sum(residuals(p0) ** 2)) * (1 + 1e-12):
        raise RuntimeError("Lab fit ended above its starting cost")  # cannot happen for a descent method
    for _ in range(LAB_POLISH_STEPS):
        dp, *_ = np.linalg.lstsq(jacobian(p), -residuals(p), rcond=None)
        if np.max(np.abs(dp)) > 1e-4:  # the solver is at the minimum; a larger step means something is off
            break
        new_cost = 0.5 * float(np.sum(residuals(p + dp) ** 2))
        if new_cost > cost * (1 + 1e-12):  # a polish step must not make things worse (beyond rounding noise)
            break
        p, cost = p + dp, new_cost
        if np.max(np.abs(dp)) < LAB_POLISH_TOL:
            break
    return _white_preserving(p)


@dataclass(frozen=True)
class MapEntry:
    """One fitted map: the light it was fitted at, its mired coordinate and the 3x3."""

    light: str
    mired: float
    m: np.ndarray  # (3, 3), rows sum to 1


def fit_tables(
    S_P: np.ndarray,
    S_A: np.ndarray | None = None,
    exclude: tuple[str, ...] = (),
    R_train: np.ndarray | None = None,
    objective: str = OBJECTIVE_LAB,
) -> dict[str, list[MapEntry]]:
    """Fit every reference light of every class; each class is sorted by (mired, name).

    ``objective`` is ``"lab"`` (the shipped :func:`fit_map_lab`) or ``"weighted_rgb"`` (the superseded
    :func:`fit_map_weighted_rgb`, kept for comparison and for reproducing the research numbers).
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
    S_A = alexa_curves() if S_A is None else S_A
    R_train = training_reflectances() if R_train is None else R_train

    def one(L: lt.Light) -> np.ndarray:
        if objective == OBJECTIVE_LAB:
            return fit_map_lab(S_P, S_A, L.spd, L.cct_k, R_train)
        return fit_map_weighted_rgb(S_P, S_A, L.spd, R_train)

    return {klass: [MapEntry(L.name, L.mired, one(L)) for L in lights]
            for klass, lights in lt.reference_lights(exclude).items()}


# ----------------------------------------------------------------------------
# Runtime blend (reference implementation of docs/02 "Runtime")
# ----------------------------------------------------------------------------
def interp_class(entries: list[MapEntry], mired: float) -> np.ndarray:
    """``M_k(mired)``: linear in mired between the bracketing entries, clamped at the ends."""
    if mired <= entries[0].mired:
        return entries[0].m
    if mired >= entries[-1].mired:
        return entries[-1].m
    mireds = np.array([e.mired for e in entries])
    i = int(np.searchsorted(mireds, mired, side="right"))  # entries[i-1].mired <= mired < entries[i].mired
    lo, hi = entries[i - 1], entries[i]
    a = (mired - lo.mired) / (hi.mired - lo.mired)
    return (1 - a) * lo.m + a * hi.m


def blend_map(tables: dict[str, list[MapEntry]], probs: dict[str, float], mired: float) -> np.ndarray:
    """``M_map = sum_k p_k M_k(mired)``. ``probs`` must sum to 1 over the classes it names."""
    total = sum(probs.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"class probabilities sum to {total}, expected 1")
    return sum(p * interp_class(tables[k], mired) for k, p in probs.items() if p != 0.0)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def stability_study(seed: int = 20261012) -> dict:
    """How reproducible the Lab fit is: all 11 bodies x all reference lights (495 fits).

    Compares the fit from the fixed start with a repeat run, with a start perturbed by N(0, 1e-3) per
    entry (seeded) and with the identity as start. Returns the largest coefficient differences.
    """
    rng = np.random.default_rng(seed)
    R, S_A = training_reflectances(), alexa_curves()
    refs = [L for lights in lt.reference_lights().values() for L in lights]
    out = {"fits": 0, "repeat": 0.0, "perturbed_1e-3": 0.0, "identity_start": 0.0}
    for S_P in sony_curves().values():
        for L in refs:
            M = fit_map_lab(S_P, S_A, L.spd, L.cct_k, R)
            out["fits"] += 1
            out["repeat"] = max(out["repeat"], float(np.abs(M - fit_map_lab(S_P, S_A, L.spd, L.cct_k, R)).max()))
            Mp = fit_map_lab(S_P, S_A, L.spd, L.cct_k, R, start=M + rng.normal(0, 1e-3, (3, 3)))
            out["perturbed_1e-3"] = max(out["perturbed_1e-3"], float(np.abs(M - Mp).max()))
            Mi = fit_map_lab(S_P, S_A, L.spd, L.cct_k, R, start=np.eye(3))
            out["identity_start"] = max(out["identity_start"], float(np.abs(M - Mi).max()))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="sony-ilce-7rm4", help="profile id (sony-ilce-7rm4, ...)")
    ap.add_argument("--gate", action="store_true", help="fit without LED-B3, LED-RGB1, LED-V2 (gate refit)")
    ap.add_argument("--objective", choices=OBJECTIVES, default=OBJECTIVE_LAB, help="default: the shipped Lab objective")
    ap.add_argument("--stability", action="store_true", help="reproducibility study of the Lab fit (11 bodies x 45 lights)")
    args = ap.parse_args(argv)
    paths.require_sources()
    if args.stability:
        res = stability_study()
        print(f"{res['fits']} Lab fits; largest coefficient difference between runs: repeat {res['repeat']:.1e}, "
              f"start perturbed by 1e-3 {res['perturbed_1e-3']:.1e}, identity start {res['identity_start']:.1e}")
        return 0

    from training.sensor_map.registry_data import PROFILES

    prof = next(p for p in PROFILES if p.id == args.profile)
    S_P = sony_curves()[prof.curves_name]
    tables = fit_tables(S_P, exclude=lt.GATE_LED if args.gate else (), objective=args.objective)
    np.set_printoptions(precision=6, suppress=True)
    for klass, entries in tables.items():
        print(f"{klass}: {len(entries)} maps, mired {entries[0].mired:.2f} .. {entries[-1].mired:.2f}")
        for e in entries:
            row_sum_err = float(np.abs(e.m.sum(axis=1) - 1).max())
            print(f"  {e.light:<18} mired {e.mired:8.3f}  max|row sum - 1| {row_sum_err:.1e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Sensor-map fit (docs/02 §C2): white-balanced Sony RGB -> white-balanced ALEXA RGB.

Port of ``research/alexa-ssf/scripts/sony_profiles.py::fit_map``. For one light ``E``:

1. Sony responses ``c_s = S_P . (E o R)`` and ALEXA responses ``c_a = S_A . (E o R)`` of the
   190 rawtoaces training reflectances ``R``, each divided by its own response to the white
   (``R = 1``);
2. the white-preserving 3x3 ``M`` (rows sum to 1, so ``M . [1,1,1] = [1,1,1]``) that minimises
   the weighted squared error ``|M c_s - c_a|^2`` with weights ``Y^(-2/3)`` (Y of the scene colour,
   an approximation of CIE Lab error);
3. one ``(mired, M)`` per reference light, grouped in four classes and sorted by mired.

Runtime blend (reference implementation: :func:`blend_map`): ``M(mired) = sum_k p_k M_k(mired)``
with linear interpolation in mired inside a class and clamping at its ends.

    python3 -m training.sensor_map.fit [--profile sony-ilce-7rm4] [--gate]

prints the fitted tables' summary (the profile files are written by ``build_profiles``).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

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


@lru_cache(maxsize=1)
def sony_curves() -> dict[str, np.ndarray]:
    """The measured Sony curves {"Sony ILCE-7RM4": (81, 3), ...} from rawtoaces-data.

    Loaded with ``ssf_core.load_cameras`` (clip at 0, peak-normalise per channel), restricted to
    the 11 bodies whose files ship in ``data/cameras/sony/``.
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


# ----------------------------------------------------------------------------
# The fit
# ----------------------------------------------------------------------------
def fit_map(S_src: np.ndarray, S_dst: np.ndarray, E: np.ndarray, R_train: np.ndarray) -> np.ndarray:
    """White-preserving 3x3 taking ``S_src`` white-balanced RGB to ``S_dst`` white-balanced RGB.

    Identical arithmetic to ``sony_profiles.fit_map`` (which fixed ``S_dst`` to the ALEXA
    estimate), so the numbers reproduce bit for bit.
    """
    src = s.camera_rgb_wb(S_src, E, R_train)
    dst = s.camera_rgb_wb(S_dst, E, R_train)
    # Weights approximate perceptual (Lab-like) error, as in ssf_core.fit_matrix_linear.
    w = np.clip(s.target_XYZ(E, R_train)[:, 1], 1e-3, None) ** (-2 / 3)
    A = np.stack([src[:, 0] - src[:, 2], src[:, 1] - src[:, 2]], axis=1) * w[:, None]
    M = np.zeros((3, 3))
    for r in range(3):
        m, *_ = np.linalg.lstsq(A, (dst[:, r] - src[:, 2]) * w, rcond=None)
        M[r] = [m[0], m[1], 1 - m[0] - m[1]]
    return M


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
) -> dict[str, list[MapEntry]]:
    """Fit every reference light of every class; each class is sorted by (mired, name)."""
    S_A = alexa_curves() if S_A is None else S_A
    R_train = training_reflectances() if R_train is None else R_train
    tables: dict[str, list[MapEntry]] = {}
    for klass, lights in lt.reference_lights(exclude).items():
        tables[klass] = [MapEntry(L.name, L.mired, fit_map(S_P, S_A, L.spd, R_train)) for L in lights]
    return tables


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
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="sony-ilce-7rm4", help="profile id (sony-ilce-7rm4, ...)")
    ap.add_argument("--gate", action="store_true", help="fit without LED-B3, LED-RGB1, LED-V2 (gate refit)")
    args = ap.parse_args(argv)
    paths.require_sources()

    from training.sensor_map.registry_data import PROFILES

    prof = next(p for p in PROFILES if p.id == args.profile)
    S_P = sony_curves()[prof.curves_name]
    tables = fit_tables(S_P, exclude=lt.GATE_LED if args.gate else ())
    np.set_printoptions(precision=6, suppress=True)
    for klass, entries in tables.items():
        print(f"{klass}: {len(entries)} maps, mired {entries[0].mired:.2f} .. {entries[-1].mired:.2f}")
        for e in entries:
            row_sum_err = float(np.abs(e.m.sum(axis=1) - 1).max())
            print(f"  {e.light:<18} mired {e.mired:8.3f}  max|row sum - 1| {row_sum_err:.1e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

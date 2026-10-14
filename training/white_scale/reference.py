"""The two references for a body's white, and the fit of the white scale (docs/03).

A **white** is camera RGB with green = 1 in the camera's raw scale (docs/02 C1). A profile's locus gives
the white of every blackbody and daylight colour temperature in the scale of the curves the profile was
measured with. A body that borrows a profile has raw data in its own scale; the white scale ``s = [r, 1, b]``
maps the body's white to the profile's, ``w_profile = w_body (.) s``.

Reference (a), **presets**: the camera's own white-balance presets for 2500, 3200, 4500, 6000 and 8500 K
are the camera's raw whites of those temperatures. The scale that makes them read at their temperatures on the
profile's locus, in least squares over log chroma, is the mean of ``ln(locus) - ln(preset)`` per channel
(green is 1 by definition).

Reference (b), **matrix**: the body's published colour matrix at D65 (XYZ to camera RGB) gives the body's raw
white of D65, ``M . XYZ_D65`` with green = 1. The scale that lands it on the profile's D65 (daylight 6504 K)
is their ratio. Two numbers, two conditions: no residual at D65; the presets are the check.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from training.common import paths
from training.sensor_map import runtime_ref as rr
from training.white_scale import sr2

D65_K = 6504.0
#: CIE D65 as XYZ from its chromaticity (x 0.3127, y 0.3290), Y = 1.
D65_XYZ = np.array([0.3127 / 0.3290, 1.0, (1.0 - 0.3127 - 0.3290) / 0.3290])
MATRICES_JSON = Path(__file__).resolve().parent / "colour_matrices.json"
IDENTITY = (1.0, 1.0, 1.0)


def load_profile(profile_id: str) -> rr.Profile:
    """The runtime reference of a profile file (``data/cameras/profiles/<id>/profile.json``)."""
    return rr.Profile.load(paths.PROFILES_DIR / profile_id / "profile.json")


def locus_white(profile: rr.Profile, kelvin: float) -> np.ndarray:
    """The profile's camera white of a colour temperature (blackbody below 4000 K, CIE daylight from there)."""
    return rr.cct_duv_to_rgb(profile, kelvin, 0.0)


def angle_deg(a, b) -> float:
    """The angle between two whites (as 3-vectors), degrees."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1.0, 1.0))))


def scaled(white, scale) -> np.ndarray:
    """``white (.) scale``, green = 1."""
    w = np.asarray(white, dtype=float) * np.asarray(scale, dtype=float)
    return w / w[1]


def read_cct(profile: rr.Profile, white) -> float:
    """The colour temperature the profile's locus reads for a white."""
    return float(rr.rgb_to_cct_duv(profile, np.asarray(white, dtype=float))[0])


def load_matrices() -> dict[int, np.ndarray]:
    """The published D65 matrices (XYZ to camera RGB) by Sony model ID."""
    doc = json.loads(MATRICES_JSON.read_text())
    return {int(k): np.array(v["xyz_to_camera"], dtype=float) for k, v in doc["bodies"].items()}


def matrix_white(xyz_to_camera) -> np.ndarray:
    """The raw white of D65 from a published matrix (green = 1)."""
    cam = np.asarray(xyz_to_camera, dtype=float) @ D65_XYZ
    return cam / cam[1]


def normalised_rows(m) -> np.ndarray:
    """A matrix with every row scaled to sum 1 (the form of the sibling check of docs/03)."""
    m = np.asarray(m, dtype=float)
    return m / m.sum(axis=1, keepdims=True)


def matrix_difference(a, b) -> float:
    """The largest per-coefficient difference of two matrices after row normalisation."""
    return float(np.abs(normalised_rows(a) - normalised_rows(b)).max())


def presets_to_whites(levels: dict[str, tuple[int, ...]], names) -> dict[str, np.ndarray]:
    """The camera whites of the named presets of a file."""
    return {n: np.array(sr2.preset_white(levels[n])) for n in names if n in levels}


@dataclass(frozen=True)
class Fit:
    """A white scale and how well it does against the presets of the files it was judged on."""

    scale: tuple[float, float, float]
    mean_angle_before_deg: float
    mean_angle_after_deg: float
    max_angle_after_deg: float
    daylight_k_before: float
    daylight_k_after: float
    per_preset_k_after: dict[str, float]


def fit_presets(profile: rr.Profile, levels: dict[str, tuple[int, ...]]) -> tuple[float, float, float]:
    """Reference (a): the scale that puts the Kelvin presets on the profile's locus (least squares, log chroma)."""
    names = [n for n in sr2.KELVIN_PRESETS if n in levels]
    if len(names) < 3:
        raise ValueError(f"need at least 3 Kelvin presets, the file has {len(names)}")
    whites = presets_to_whites(levels, names)
    delta = np.mean(
        [np.log(locus_white(profile, sr2.KELVIN_PRESETS[n])) - np.log(whites[n]) for n in names], axis=0
    )
    delta[1] = 0.0
    s = np.exp(delta)
    return (float(s[0]), 1.0, float(s[2]))


def matrix_scale(profile: rr.Profile, xyz_to_camera) -> tuple[float, float, float]:
    """Reference (b): the scale that lands the body's matrix white of D65 on the profile's D65."""
    s = locus_white(profile, D65_K) / matrix_white(xyz_to_camera)
    return (float(s[0]), 1.0, float(s[2]))


def judge(profile: rr.Profile, levels: dict[str, tuple[int, ...]], scale) -> Fit:
    """Residuals of a scale against the Kelvin presets (angle to the locus at the label) and the Daylight preset
    (the reading in Kelvin)."""
    names = [n for n in sr2.KELVIN_PRESETS if n in levels]
    whites = presets_to_whites(levels, list(sr2.KELVIN_PRESETS) + list(sr2.NAMED_PRESETS))

    def mean_angle(s) -> tuple[float, float]:
        a = [angle_deg(scaled(whites[n], s), locus_white(profile, sr2.KELVIN_PRESETS[n])) for n in names]
        return float(np.mean(a)), float(np.max(a))

    before, _ = mean_angle(IDENTITY)
    after, worst = mean_angle(scale)
    day = whites["daylight"]
    return Fit(
        scale=tuple(float(x) for x in scale),
        mean_angle_before_deg=before,
        mean_angle_after_deg=after,
        max_angle_after_deg=worst,
        daylight_k_before=read_cct(profile, day),
        daylight_k_after=read_cct(profile, scaled(day, scale)),
        per_preset_k_after={n: read_cct(profile, scaled(w, scale)) for n, w in whites.items()},
    )

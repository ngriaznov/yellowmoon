"""Spectral inputs of the synthetic scenes: lights, camera bodies and the reflectance bank.

All spectra live on the 380-780 nm / 5 nm grid (81 samples) of ``training.alexa.ssf_core``.

* **Lights** reuse the conventions of ``training.sensor_map.lights``: blackbody below 4000 K (Planck at the
  generating temperature), CIE D series from 4000 K with the rawtoaces CCT correction, CIE FL and LED spectra
  from colour-science (CIE data, CC BY-SA 4.0, read at generation time only). The CCT used for ARRI's matrix
  lookup is the generating temperature for blackbody and D lights and the Ohno (2013) CCT for the others.
* **Camera bodies** use the published Weta curves of the 11 measured Sony bodies **in the published raw scale**
  (``fit.sony_curves_raw``), so the as-shot white balance of a file is what a real raw file of that light has.
  Bodies without curves use the A7R IV (``docs/03``): the A1 (model ID 384) is one of them.
* **Reflectances**: the 190 rawtoaces training patches (Apache-2.0) plus flat neutrals. The evaluation scenes also
  use the 99 CIE CES (CIE 224:2017, CC BY-SA 4.0), the 24 Ohta ColorChecker patches and the 15 NIST CQS patches;
  anything made from those stays in ``testdata/`` (never committed).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

from training.alexa import ssf_core as s
from training.sensor_map import fit
from training.sensor_map import lights as lt

WL = s.WL
N_WL = WL.size
CLASSES = lt.CLASSES  # ("daylight", "tungsten", "fluorescent", "led")


# ----------------------------------------------------------------------------------------------------
# Lights
# ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SceneLight:
    """One scene light (possibly a spatial mixture of two) with the labels of its ground truth."""

    token: str  # file-name token: "D6500", "bb2850", "F11", "LEDB3", "mixed"
    name: str  # SPD name for humans: "CIE D 6500 K", "Planckian 2850 K", "CIE FL11", "CIE LED-B3"
    kind: str  # "blackbody" | "cie_daylight" | "cie_fl" | "cie_led" | "mixed"
    klass: str  # one of CLASSES, or "mixed"
    cct_k: float  # CCT used for the ARRI lookup
    cct_convention: str  # "generating temperature" | "Ohno 2013 CCT of the SPD"
    spd: np.ndarray = field(repr=False, compare=False)  # (81,), peak 1 (a mixture: the mean of its two component SPDs)
    class_probs: tuple[tuple[str, float], ...] = ()
    ev100: float = 12.0  # brightness of the 18 % grey (EV at ISO 100) for the exposure metadata only
    components: tuple[str, ...] = ()  # for a mixed light: the tokens of the two lights (left, right)
    component_spds: tuple[np.ndarray, ...] = field(default=(), repr=False, compare=False)  # their SPDs, as mixed


def _light(token, name, kind, base: lt.Light, convention: str, ev100: float) -> SceneLight:
    return SceneLight(
        token=token, name=name, kind=kind, klass=base.klass, cct_k=float(base.cct_k),
        cct_convention=convention, spd=np.asarray(base.spd, dtype=float),
        class_probs=((base.klass, 1.0),), ev100=ev100,
    )


_GEN = "generating temperature"
_OHNO = "Ohno 2013 CCT of the SPD"


def single_lights() -> dict[str, SceneLight]:
    """The eight single lights of the default set (``docs`` task: blackbody 2850/3200 K, D 5500/6500/7500 K, F11,
    LED-B3, LED-V2). ``ev100`` is a plausible brightness for the exposure metadata (sunlight, shade, indoor)."""
    out = [
        _light("bb2850", "Planckian 2850 K", "blackbody", lt.blackbody_light(2850.0, "tungsten"), _GEN, 6.0),
        _light("bb3200", "Planckian 3200 K", "blackbody", lt.blackbody_light(3200.0, "tungsten"), _GEN, 7.0),
        _light("D5500", "CIE D 5500 K", "cie_daylight", lt.daylight_light(5500.0, "daylight"), _GEN, 14.0),
        _light("D6500", "CIE D 6500 K", "cie_daylight", lt.daylight_light(6500.0, "daylight"), _GEN, 13.0),
        _light("D7500", "CIE D 7500 K", "cie_daylight", lt.daylight_light(7500.0, "daylight"), _GEN, 12.0),
        _light("F11", "CIE FL11", "cie_fl", lt.named_light("FL11", "fluorescent"), _OHNO, 8.0),
        _light("LEDB3", "CIE LED-B3", "cie_led", lt.named_light("LED-B3", "led"), _OHNO, 9.0),
        _light("LEDV2", "CIE LED-V2", "cie_led", lt.named_light("LED-V2", "led"), _OHNO, 8.0),
    ]
    return {L.token: L for L in out}


#: The mixed-light scene: tungsten on the left, daylight on the right, a smooth blend across the middle third.
MIXED_LEFT = "bb2850"
MIXED_RIGHT = "D6500"


def mixed_light(left: SceneLight | None = None, right: SceneLight | None = None) -> SceneLight:
    """The reference light of the mixed scene: the equal mixture of its two lights (the image-wide average).

    The two lights are first scaled to the same CIE Y (a peak-normalised blackbody carries half the luminance of
    daylight, a spiky fluorescent a quarter), so both ends of the scene are equally bright and the middle is the plain
    mean of the two scaled SPDs, which is also the reference light: a white patch there is 1.0 under it. White balance
    and the ARRI matrix of the truth are those of this mixture; every patch also records its local mixture weight, so
    a consumer can compare with the local light instead.
    """
    lights = single_lights()
    left = left or lights[MIXED_LEFT]
    right = right or lights[MIXED_RIGHT]
    y = lambda spd: float(s.CMF[:, 1] @ spd)  # noqa: E731
    left_spd, right_spd = left.spd / y(left.spd), right.spd / y(right.spd)
    spd = 0.5 * left_spd + 0.5 * right_spd
    return SceneLight(
        token="mixed", name=f"mixture of {left.name} and {right.name}", kind="mixed", klass="mixed",
        cct_k=float(s.illuminant_cct(spd)), cct_convention=_OHNO, spd=spd,
        class_probs=((left.klass, 0.5), (right.klass, 0.5)), ev100=8.0, components=(left.token, right.token),
        component_spds=(left_spd, right_spd),
    )


def all_lights() -> dict[str, SceneLight]:
    out = single_lights()
    out["mixed"] = mixed_light(out[MIXED_LEFT], out[MIXED_RIGHT])
    return out


def cie_xy(spd: np.ndarray) -> list[float]:
    """CIE 1931 xy of an SPD."""
    import colour

    xy = colour.XYZ_to_xy(s.CMF.T @ spd)
    return [float(xy[0]), float(xy[1])]


# ----------------------------------------------------------------------------------------------------
# Camera bodies
# ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Body:
    """A camera body of the synthetic set.

    ``pitch_um`` follows from the public sensor format and pixel count (arithmetic, not a measurement). The read
    noise values and the conversion-gain switch ISO are **assumptions** that set the synthetic noise truth: plausible
    magnitudes, not measurements (``training/synth/sensor.py``).
    """

    model: str  # EXIF Model, ARW file name token: "ILCE-7RM4"
    name: str  # "Sony A7R IV"
    model_id: int  # Sony maker-note model ID (docs/03)
    curves: str  # key of fit.sony_curves_raw(): the measured body whose curves are used
    own_curves: bool  # False for a stand-in (the A1 uses the A7R IV curves)
    pitch_um: float  # pixel pitch
    read_noise_low_e: float  # input-referred read noise below the switch ISO, electrons (assumed)
    read_noise_high_e: float  # above it (assumed)
    switch_iso: int | None  # conversion-gain switch ISO (None: single gain)
    lens: str | None  # LensModel (None for a fixed lens)
    focal_mm: float
    typical_variant: str  # the variant this set writes for the body in the cross recipes (the decoder must handle it on every
    # body; of these bodies only the A1 writes lossless L in reality, research/notes/sony-arw-formats.md)
    white_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)  # the registry's white scale of the body (docs/03): the
    # factor that takes the body's raw white to its profile's, w_profile = w_body * white_scale. The scenes are rendered
    # through the profile's curves, so a stand-in body's raw is the render divided by it and its as-shot gains are
    # multiplied by it; make_scenes does both, so the pipeline's own scaling lands on the render again.


def registry_white_scale(model_id: int) -> tuple[float, float, float]:
    """The white scale of a body in ``data/cameras/registry.json`` (identity for a body the registry does not list)."""
    import json

    from training.common import paths

    for body in json.loads(paths.REGISTRY_JSON.read_text())["bodies"]:
        if body["model_id"] == model_id:
            r, g, b = body["white_scale"]
            return (float(r), float(g), float(b))
    return (1.0, 1.0, 1.0)


BODIES: dict[str, Body] = {
    b.model: dataclasses.replace(b, white_scale=registry_white_scale(b.model_id))
    for b in (
        Body("ILCE-7RM4", "Sony A7R IV", 375, "Sony ILCE-7RM4", True, 3.76, 3.4, 1.8, 320,
             "FE 24-70mm F2.8 GM", 35.0, "lossless-l"),
        Body("ILCE-7M3", "Sony A7 III", 363, "Sony ILCE-7M3", True, 5.9, 2.9, 1.5, 640,
             "FE 28-70mm F3.5-5.6 OSS", 50.0, "uncompressed14"),
        Body("ILCE-7SM3", "Sony A7S III", 383, "Sony ILCE-7SM3", True, 8.4, 3.0, 1.2, 1600,
             "FE 24-105mm F4 G OSS", 50.0, "lossless-l"),
        Body("ILCE-6400", "Sony A6400", 371, "Sony ILCE-6400", True, 3.9, 3.2, 3.2, None,
             "E 18-135mm F3.5-5.6 OSS", 35.0, "craw"),
        Body("DSC-RX100M4", "Sony RX100 IV", 341, "Sony DSC-RX100M4", True, 2.4, 3.8, 3.8, None,
             None, 12.0, "craw"),
        Body("ILCE-1", "Sony A1", 384, "Sony ILCE-7RM4", False, 4.15, 3.3, 1.7, 500,
             "FE 50mm F1.2 GM", 50.0, "lossless-l"),
    )
}

#: Areal full-well density assumed for every synthetic body, electrons per square micrometre (an assumption of the
#: synthetic truth, deliberately not the 2000 of ``ym_noise::priors::ClassAssumptions``, so that tests of the
#: estimator are not circular).
FULL_WELL_E_PER_UM2 = 1800.0


def camera_curves(body: Body) -> np.ndarray:
    """(81, 3) published-raw-scale curves of the body (a stand-in's for bodies without their own)."""
    return fit.sony_curves_raw()[body.curves]


def alexa_curves() -> np.ndarray:
    """(81, 3) ALEXA estimate S_A, each channel peak-normalised (only ratios enter the reference)."""
    return fit.alexa_curves()


# ----------------------------------------------------------------------------------------------------
# Reflectance bank
# ----------------------------------------------------------------------------------------------------
class Bank:
    """An ordered set of named reflectance spectra; scenes refer to entries by index."""

    def __init__(self) -> None:
        self.names: list[str] = []
        self.sources: list[str] = []
        self._rows: list[np.ndarray] = []
        self._index: dict[str, int] = {}

    def add(self, name: str, spectrum: np.ndarray, source: str) -> int:
        """Adds a spectrum (81 values, clipped at 0) and returns its index; a name added twice is one entry."""
        if name in self._index:
            return self._index[name]
        v = np.clip(np.asarray(spectrum, dtype=float), 0.0, None)
        assert v.shape == (N_WL,), v.shape
        self._index[name] = len(self._rows)
        self.names.append(name)
        self.sources.append(source)
        self._rows.append(v)
        return self._index[name]

    def flat(self, rho: float) -> int:
        """Index of the flat (spectrally neutral) reflectance ``rho``."""
        return self.add(f"neutral-{rho:g}", np.full(N_WL, float(rho)), "flat neutral")

    def __getitem__(self, name: str) -> int:
        return self._index[name]

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def spectra(self) -> np.ndarray:
        """(K, 81)."""
        return np.stack(self._rows)


def training_patches() -> list[tuple[str, np.ndarray]]:
    """The 190 rawtoaces training reflectances as ``("rta-001", spectrum)`` (Apache-2.0)."""
    R = s.training_reflectances()
    return [(f"rta-{i + 1:03d}", R[:, i]) for i in range(R.shape[1])]


def evaluation_patches() -> list[tuple[str, np.ndarray]]:
    """The evaluation reflectances (CC BY-SA / CIE data, ``testdata/`` only): 99 CES, 24 Ohta, 15 NIST CQS."""
    sets = s.test_reflectances()
    out: list[tuple[str, np.ndarray]] = []
    for key, prefix in (("CIE 2017 CES (99)", "ces"), ("ColorChecker (Ohta, 24)", "cc"), ("NIST CQS VS (15)", "vs")):
        arr = sets.get(key)
        if arr is None:
            continue
        out += [(f"{prefix}-{i + 1:03d}", arr[:, i]) for i in range(arr.shape[1])]
    return out


def _lab_hue_chroma(spectra: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CIE Lab (ACES white) lightness, chroma and hue angle (degrees) of reflectances under D65, (N,) each."""
    e = lt.daylight_light(6500.0).spd
    xyz = s.target_XYZ(e, spectra.T)
    lab = s.XYZ_to_Lab(xyz)
    return lab[:, 0], np.hypot(lab[:, 1], lab[:, 2]), np.degrees(np.arctan2(lab[:, 2], lab[:, 1])) % 360.0


def pick_hue_patches(candidates: list[tuple[str, np.ndarray]], n: int = 8) -> list[int]:
    """Indices (into ``candidates``) of the most saturated mid-tone patch in each of ``n`` hue sectors, sector
    order by hue angle starting at 0 degrees (red-magenta first). Deterministic: ties go to the lower index."""
    spec = np.stack([c[1] for c in candidates])
    L, C, H = _lab_hue_chroma(spec)
    ok = (L > 25.0) & (L < 80.0) & (spec.max(axis=1) <= 1.0)
    picks = []
    for k in range(n):
        lo, hi = 360.0 * k / n, 360.0 * (k + 1) / n
        mask = ok & (H >= lo) & (H < hi)
        idx = np.flatnonzero(mask)
        picks.append(int(idx[np.argmax(C[idx])]) if idx.size else int(np.argmax(np.where(ok, C, -1))))
    return picks

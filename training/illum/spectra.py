"""Spectral inputs of the synthetic scenes: lights, reflectances and camera models.

Everything is sampled on the 380-780 nm / 5 nm grid (81 samples) of ``training.alexa.ssf_core``.

**Lights** (``docs/02`` section Illuminant estimate; classes of ``docs/07``):

=================  ==============================================================================
class              sources
=================  ==============================================================================
daylight           CIE D series, 4200-6500 K
shade_overcast     CIE D series, 6500-12000 K
tungsten_halogen   blackbody, 2000-3400 K
fluorescent        CIE FL1-FL12, FL3.1-FL3.15 (``colour.SDS_ILLUMINANTS``) and generated
                   tri-phosphor / halophosphate spectra
led                CIE LED-B1..B5, LED-BH1, LED-RGB1, LED-V1, LED-V2 and generated phosphor-converted
                   white and RGB LED spectra
mixed              two of the above, mixed with a smooth spatial weight
=================  ==============================================================================

CIE data (CC BY-SA 4.0, ``docs/14``, pending HUMAN M1-H4) is read from colour-science at training time
only; it never enters the repository. The blackbody and D-series curves are formulas.

**Light splits.** Three disjoint pools: ``train``, ``val`` (calibration of the class temperature and the confidence
scale) and ``test`` (the gates). ``test`` holds the nine gate lights of the sensor-map gate (blackbody 2850 K,
D 5250 K and 6750 K, FL3.1, FL3.8, FL3.15, LED-B3, LED-RGB1, LED-V2) plus more CIE fluorescent and LED spectra, bands
of blackbody and D-series temperatures, and generated lights from a disjoint seed stream; ``val`` holds other bands
and spectra. A model never trains on a ``val`` or ``test`` light, so both measure generalisation to unseen lights.

**Reflectances.** Training: 132 of the 190 rawtoaces training patches (Apache-2.0) and smooth random
spectra generated here. Validation: 20 more patches. Test: 38 more patches and smooth spectra from another seed stream.
Evaluation only: the 99 CIE CES, the 24 Ohta ColorChecker and the 15 NIST CQS VS patches (CES 99 is
never used for training).

**Cameras.** The curves are the camera's published channel balance (``fit.sony_curves_raw``: clipped at
zero, *not* peak-normalised per channel), the same scale as the profile ``locus`` and as raw data
(``docs/02`` C1, defect found in the M1-07 review).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from training.alexa import ssf_core as s
from training.sensor_map import fit
from training.sensor_map import lights as lt
from training.sensor_map.registry_data import PROFILES

WL = s.WL  # (81,) nm
N_WL = WL.size

CLASSES = ("daylight", "shade_overcast", "tungsten_halogen", "fluorescent", "led", "mixed")
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}

# ----------------------------------------------------------------------------
# Held-out lights
# ----------------------------------------------------------------------------
#: Temperature bands (kelvin) of the blackbody and D-series lights that only the test (or the val) split uses.
TEST_BLACKBODY_BANDS_K = ((2250.0, 2350.0), (2750.0, 2950.0), (3150.0, 3250.0), (3550.0, 3650.0))
TEST_DAYLIGHT_BANDS_K = (
    (4450.0, 4550.0), (5150.0, 5350.0), (5900.0, 6100.0), (6650.0, 6850.0),
    (7400.0, 7600.0), (8900.0, 9100.0), (10400.0, 10600.0),
)
VAL_BLACKBODY_BANDS_K = ((2450.0, 2550.0), (3350.0, 3450.0))
VAL_DAYLIGHT_BANDS_K = (
    (4950.0, 5050.0), (5650.0, 5750.0), (6250.0, 6350.0), (8100.0, 8300.0), (9400.0, 9600.0), (11000.0, 11200.0),
)
#: CIE fluorescent and LED spectra that only the test (or the val) split uses (the nine gate lights are in the test sets).
TEST_FLUORESCENT = ("FL2", "FL5", "FL9", "FL3.1", "FL3.5", "FL3.8", "FL3.12", "FL3.15")
VAL_FLUORESCENT = ("FL4", "FL3.6", "FL3.13")
TEST_LED = ("LED-B3", "LED-BH1", "LED-RGB1", "LED-V2")
VAL_LED = ("LED-B5",)
GATE_LIGHT_NAMES = tuple(L.name for L in lt.gate_lights())

ALL_FLUORESCENT = tuple(f"FL{i}" for i in range(1, 13)) + tuple(f"FL3.{i}" for i in range(1, 16))
ALL_LED = lt.LED
SPLIT_SEED_STREAM = {"train": 3, "val": 5, "test": 7}  # seed streams of the generated lights


def _bands(split: str, kind: str):
    if split == "test":
        return TEST_BLACKBODY_BANDS_K if kind == "blackbody" else TEST_DAYLIGHT_BANDS_K
    if split == "val":
        return VAL_BLACKBODY_BANDS_K if kind == "blackbody" else VAL_DAYLIGHT_BANDS_K
    return TEST_BLACKBODY_BANDS_K + VAL_BLACKBODY_BANDS_K if kind == "blackbody" else TEST_DAYLIGHT_BANDS_K + VAL_DAYLIGHT_BANDS_K


def _cie_pool(all_names, test_names, val_names, split: str):
    if split == "test":
        return [n for n in all_names if n in test_names]
    if split == "val":
        return [n for n in all_names if n in val_names]
    return [n for n in all_names if n not in test_names and n not in val_names]


def _in_bands(t: float, bands) -> bool:
    return any(lo <= t <= hi for lo, hi in bands)


def sample_temperature(rng: np.random.Generator, lo: float, hi: float, bands, split: str) -> float:
    """Uniform temperature in ``[lo, hi]``; the test and val splits draw only inside their ``bands``, train outside
    (``bands`` is the split's own list for test and val, and the union of both for train)."""
    if split in ("test", "val"):
        inside = [(max(a, lo), min(b, hi)) for a, b in bands if min(b, hi) > max(a, lo)]
        if not inside:  # no held-out band in this range: use the whole range (kept for completeness)
            return float(rng.uniform(lo, hi))
        widths = np.array([b - a for a, b in inside])
        k = int(rng.choice(len(inside), p=widths / widths.sum()))
        return float(rng.uniform(*inside[k]))
    for _ in range(1000):
        t = float(rng.uniform(lo, hi))
        if not _in_bands(t, bands):
            return t
    raise RuntimeError("could not draw a training temperature")  # pragma: no cover


# ----------------------------------------------------------------------------
# Light SPDs
# ----------------------------------------------------------------------------
@lru_cache(maxsize=2)
def _grid_spds(kind: str) -> tuple[float, np.ndarray]:
    """Blackbody (1500-4000 K) or CIE D-series (4000-25000 K) SPDs every 10 K (cached, peak-normalised)."""
    lo, hi = (1500, 4000) if kind == "blackbody" else (4000, 25000)
    ts, es = (lt.blackbody_grid if kind == "blackbody" else lt.daylight_grid)(lo, hi, 10)
    return float(lo), es


def blackbody_at(t_k: float) -> np.ndarray:
    """Blackbody SPD at ``t_k`` (1500-4000 K), interpolated linearly between the 10 K grid spectra."""
    return _interp_spd("blackbody", t_k)


def daylight_at(t_k: float) -> np.ndarray:
    """CIE D-series SPD at ``t_k`` (4000-25000 K), interpolated linearly between the 10 K grid spectra."""
    return _interp_spd("daylight", t_k)


def _interp_spd(kind: str, t_k: float) -> np.ndarray:
    lo, es = _grid_spds(kind)
    x = (t_k - lo) / 10.0
    i = int(np.clip(np.floor(x), 0, es.shape[0] - 2))
    f = float(np.clip(x - i, 0.0, 1.0))
    v = (1.0 - f) * es[i] + f * es[i + 1]
    return v / v.max()


@lru_cache(maxsize=None)
def _named(name: str) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return s.named_illuminant(name)


def smooth_tilt(rng: np.random.Generator, amplitude: float) -> np.ndarray:
    """A smooth positive multiplicative spectral perturbation (a few low cosines, mean ~1)."""
    x = (WL - WL[0]) / (WL[-1] - WL[0])
    f = np.zeros_like(x)
    for k in range(1, 4):
        f += rng.normal(0.0, amplitude / k) * np.cos(np.pi * k * x + rng.uniform(0, 2 * np.pi))
    return np.exp(f)


def gaussian(center_nm: float, fwhm_nm: float) -> np.ndarray:
    sigma = fwhm_nm / 2.3548200450309493
    return np.exp(-0.5 * ((WL - center_nm) / sigma) ** 2)


@lru_cache(maxsize=1)
def _planckian_uv() -> np.ndarray:
    """CIE 1960 (u, v) of the Planckian locus 1500-25000 K every 50 K (blackbody below 4000 K, D series above)."""
    from training.sensor_map import tables as tb

    ts = np.arange(1500.0, 25001.0, 50.0)
    es = np.stack([blackbody_at(t) if t < 4000.0 else daylight_at(t) for t in ts])
    return tb.uv1960_of_xy(tb.xy_of(es))


def duv_of(spd: np.ndarray) -> float:
    """Distance of the light's chromaticity from the locus in CIE 1960 uv (unsigned; CIE colorimetry, fit time only)."""
    from training.sensor_map import tables as tb

    uv = tb.uv1960_of_xy(tb.xy_of(spd[None]))[0]
    return float(np.linalg.norm(_planckian_uv() - uv, axis=1).min())


#: Generated LED and fluorescent spectra are accepted only within this distance of the locus. The 12 CIE F, 15 FL3 and
#: 9 LED reference spectra are all within 0.0075; commercial white lamps are within about 0.012.
MAX_GENERATED_DUV = 0.012


def _generate_white(rng: np.random.Generator, make) -> np.ndarray:
    for _ in range(500):
        spd = make(rng)
        if duv_of(spd) <= MAX_GENERATED_DUV:
            return spd
    raise RuntimeError("could not generate a near-white light")  # pragma: no cover


def generated_led(rng: np.random.Generator) -> np.ndarray:
    """A random near-white LED spectrum: phosphor-converted white (blue pump plus broad phosphor), 3-LED or RGB(A)."""
    return _generate_white(rng, _raw_led)


def generated_fluorescent(rng: np.random.Generator) -> np.ndarray:
    """A random near-white fluorescent spectrum: mercury lines on a halophosphate continuum, or tri-phosphor bands."""
    return _generate_white(rng, _raw_fluorescent)


def _raw_led(rng: np.random.Generator) -> np.ndarray:
    kind = rng.choice(["pc_white", "pc_white", "rgb", "pc_two_phosphor"])
    if kind == "pc_white":
        blue = gaussian(rng.uniform(438, 462), rng.uniform(14, 26))
        phos = gaussian(rng.uniform(535, 600), rng.uniform(75, 125))
        w = rng.uniform(0.25, 2.0)
        spd = blue + w * phos * (rng.uniform(0.6, 1.0))
    elif kind == "pc_two_phosphor":
        blue = gaussian(rng.uniform(438, 462), rng.uniform(14, 26))
        green = gaussian(rng.uniform(515, 545), rng.uniform(60, 90))
        red = gaussian(rng.uniform(600, 650), rng.uniform(50, 90))
        spd = blue + rng.uniform(0.2, 1.2) * green + rng.uniform(0.2, 1.5) * red
    else:
        b = gaussian(rng.uniform(445, 475), rng.uniform(18, 30))
        g = gaussian(rng.uniform(515, 550), rng.uniform(25, 40))
        r = gaussian(rng.uniform(605, 640), rng.uniform(15, 30))
        spd = rng.uniform(0.4, 1.4) * b + rng.uniform(0.4, 1.4) * g + rng.uniform(0.5, 1.6) * r
    spd = spd + 0.01
    return spd / spd.max()


def _raw_fluorescent(rng: np.random.Generator) -> np.ndarray:
    hg = (405.0, 436.0, 546.0, 578.0)
    hg_amp = np.array([0.05, 0.25, 0.7, 0.12]) * rng.uniform(0.6, 1.4, size=4)
    spd = sum(a * gaussian(c, 6.0) for a, c in zip(hg_amp, hg))
    if rng.random() < 0.5:  # halophosphate: broad blue and orange-red bands
        spd = spd + rng.uniform(0.3, 0.9) * gaussian(rng.uniform(470, 490), rng.uniform(60, 90))
        spd = spd + rng.uniform(0.5, 1.2) * gaussian(rng.uniform(570, 600), rng.uniform(100, 140))
    else:  # tri-phosphor: blue band, Tb green line, Eu red line
        spd = spd + rng.uniform(0.2, 0.6) * gaussian(rng.uniform(445, 455), rng.uniform(35, 55))
        spd = spd + rng.uniform(0.6, 1.2) * gaussian(rng.uniform(542, 548), rng.uniform(6, 12))
        spd = spd + rng.uniform(0.8, 1.6) * gaussian(rng.uniform(608, 614), rng.uniform(6, 12))
        spd = spd + rng.uniform(0.0, 0.15) * gaussian(rng.uniform(570, 650), rng.uniform(80, 140))
    spd = spd + 0.005
    return spd / spd.max()


@dataclass(frozen=True)
class Light:
    """One light of a scene."""

    spd: np.ndarray        # (81,), peak 1
    klass: str             # one of CLASSES except "mixed"
    family: str            # blackbody | daylight | fluor_cie | led_cie | fluor_gen | led_gen
    name: str
    cct_k: float | None    # generating or nominal temperature where there is one


def draw_light(rng: np.random.Generator, klass: str, split: str) -> Light:
    """One light of ``klass`` (not ``mixed``) from the ``split`` pool (``train``, ``val`` or ``test``)."""
    if klass == "daylight":
        t = sample_temperature(rng, 4200.0, 6500.0, _bands(split, "daylight"), split)
        return Light(_tilt(daylight_at(t), rng, 0.03), klass, "daylight", f"D{t:.0f}K", t)
    if klass == "shade_overcast":
        t = sample_temperature(rng, 6500.0, 12000.0, _bands(split, "daylight"), split)
        return Light(_tilt(daylight_at(t), rng, 0.03), klass, "daylight", f"D{t:.0f}K", t)
    if klass == "tungsten_halogen":
        lo = 2000.0 if rng.random() < 0.08 else 2300.0
        t = sample_temperature(rng, lo, 3400.0, _bands(split, "blackbody"), split)
        return Light(_tilt(blackbody_at(t), rng, 0.02), klass, "blackbody", f"BB{t:.0f}K", t)
    if klass == "fluorescent":
        pool = _cie_pool(ALL_FLUORESCENT, TEST_FLUORESCENT, VAL_FLUORESCENT, split)
        if rng.random() < 0.6 and pool:
            name = pool[int(rng.integers(len(pool)))]
            return Light(_named(name), klass, "fluor_cie", name, None)
        return Light(_gen_split(rng, generated_fluorescent, split), klass, "fluor_gen", "gen-FL", None)
    if klass == "led":
        pool = _cie_pool(ALL_LED, TEST_LED, VAL_LED, split)
        if rng.random() < 0.5 and pool:
            name = pool[int(rng.integers(len(pool)))]
            return Light(_named(name), klass, "led_cie", name, None)
        return Light(_gen_split(rng, generated_led, split), klass, "led_gen", "gen-LED", None)
    raise ValueError(f"unknown light class {klass!r}")


def _tilt(spd: np.ndarray, rng: np.random.Generator, amplitude: float) -> np.ndarray:
    v = spd * smooth_tilt(rng, amplitude)
    return v / v.max()


def _gen_split(rng: np.random.Generator, make, split: str) -> np.ndarray:
    """Generated spectra: each split draws from its own seed stream, disjoint from the others."""
    sub = np.random.default_rng([int(rng.integers(1 << 30)), SPLIT_SEED_STREAM[split]])
    return make(sub)


def named_light(name: str, klass: str) -> Light:
    """A CIE light by name (the gate lights)."""
    return Light(_named(name), klass, "fluor_cie" if klass == "fluorescent" else "led_cie", name, None)


def gate_lights() -> list[Light]:
    """The nine gate lights of the sensor-map gate as :class:`Light` (classes as the gate uses)."""
    out = []
    for L in lt.gate_lights():
        fam = {"tungsten": "blackbody", "daylight": "daylight", "fluorescent": "fluor_cie", "led": "led_cie"}[L.klass]
        klass = {"tungsten": "tungsten_halogen", "daylight": "daylight", "fluorescent": "fluorescent", "led": "led"}[L.klass]
        out.append(Light(L.spd, klass, fam, L.name, L.cct_k if fam in ("blackbody", "daylight") else None))
    return out


# ----------------------------------------------------------------------------
# Reflectances
# ----------------------------------------------------------------------------
#: Seed of the fixed split of the 190 rawtoaces patches into training, validation and test subsets.
REFLECTANCE_SPLIT_SEED = 20261012
N_TEST_PATCHES = 38
N_VAL_PATCHES = 20


@lru_cache(maxsize=1)
def apache_patches() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(train (132, 81), val (20, 81), test (38, 81)) subsets of the rawtoaces 190 training patches (Apache-2.0)."""
    r = fit.training_reflectances().T  # (190, 81)
    perm = np.random.default_rng(REFLECTANCE_SPLIT_SEED).permutation(r.shape[0])
    test = r[perm[:N_TEST_PATCHES]]
    val = r[perm[N_TEST_PATCHES:N_TEST_PATCHES + N_VAL_PATCHES]]
    train = r[perm[N_TEST_PATCHES + N_VAL_PATCHES:]]
    return train, val, test


@lru_cache(maxsize=1)
def evaluation_reflectances() -> dict[str, np.ndarray]:
    """Evaluation-only reflectance sets (n, 81): CES 99, Ohta ColorChecker 24, NIST CQS VS 15."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sets = s.test_reflectances()
    return {k: v.T.copy() for k, v in sets.items()}


def smooth_reflectance(rng: np.random.Generator) -> np.ndarray:
    """A random smooth reflectance spectrum in [0.01, 0.98]: Gaussian bumps or a pigment-like edge."""
    if rng.random() < 0.55:
        spec = np.full(N_WL, rng.uniform(0.02, 0.45))
        for _ in range(int(rng.integers(1, 4))):
            spec = spec + rng.uniform(0.1, 0.8) * gaussian(rng.uniform(380, 780), rng.uniform(40, 220))
    else:
        lam0 = rng.uniform(420, 700)
        width = rng.uniform(8, 60)
        lo, hi = rng.uniform(0.02, 0.2), rng.uniform(0.5, 0.95)
        edge = 1.0 / (1.0 + np.exp(-(WL - lam0) / width))
        if rng.random() < 0.5:
            edge = 1.0 - edge
        spec = lo + (hi - lo) * edge
    return np.clip(spec, 0.01, 0.98)


# ----------------------------------------------------------------------------
# Cameras
# ----------------------------------------------------------------------------
#: Nominal full-well capacity (electrons at base ISO) and read noise per body, used only to size the noise of
#: the synthetic scenes. Typical values for the sensor class, not measurements (the V1 data rule).
NOMINAL_FULL_WELL_E = {
    "sony-ilce-7m3": 80000.0, "sony-ilce-7m4": 70000.0, "sony-ilce-7cm2": 70000.0,
    "sony-ilce-7rm2": 50000.0, "sony-ilce-7rm3": 52000.0, "sony-ilce-7rm4": 52000.0,
    "sony-ilce-7sm2": 90000.0, "sony-ilce-7sm3": 90000.0, "sony-ilce-9": 80000.0,
    "sony-ilce-6400": 40000.0, "sony-dsc-rx100m4": 12000.0,
}
NOMINAL_READ_NOISE_E = 3.0


@dataclass(frozen=True)
class Camera:
    """A camera profile's curves in the published raw scale and its nominal noise parameters."""

    profile_id: str
    display_name: str
    curves: np.ndarray       # (81, 3), clipped at 0, green peak 1.0
    full_well_e: float
    read_noise_e: float

    def response(self, spd: np.ndarray) -> np.ndarray:
        """Raw RGB response ``S^T E`` of spectra (81,) or (n, 81)."""
        return spd @ self.curves

    def white(self, spd: np.ndarray) -> np.ndarray:
        """Camera RGB of the light, green = 1 (what a white surface records)."""
        r = self.response(spd)
        return r / r[..., 1:2]


@lru_cache(maxsize=None)
def camera(profile_id: str) -> Camera:
    """The camera model of a shipped profile."""
    prof = next(p for p in PROFILES if p.id == profile_id)
    curves = fit.sony_curves_raw()[prof.curves_name]
    return Camera(profile_id, prof.display_name, curves, NOMINAL_FULL_WELL_E[profile_id], NOMINAL_READ_NOISE_E)


def profile_ids() -> list[str]:
    return [p.id for p in PROFILES]


def light_uv(cam: Camera, spd: np.ndarray) -> np.ndarray:
    """The light's log-chroma ``(u, v) = (ln G/R, ln G/B)`` for the camera."""
    w = cam.white(spd)
    return np.array([-np.log(w[0]), -np.log(w[2])])

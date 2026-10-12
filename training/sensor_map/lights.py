"""Reference lights for the sensor-map fit, the gate and the tables.

Every light is a 380-780 nm SPD at 5 nm (81 samples), peak-normalised, built with the
conventions of ``training.alexa.ssf_core`` (the rawtoaces convention: blackbody below
4000 K, CIE D series from 4000 K, CCT correction 1.4388/1.4380 for the D series).
SPDs come from colour-science at fit time and are never shipped (``docs/14``); only the
fitted coefficients are.

Classes and their reference lights (``docs/02`` §C2):

========== ==============================================================
tungsten    blackbody 2000-4000 K every 250 K
daylight    CIE D series 4000-11000 K every 500 K
fluorescent CIE F1-F12
led         CIE LED-B1..B5, LED-BH1, LED-RGB1, LED-V1, LED-V2
========== ==============================================================

The mired coordinate of a light is ``1e6 / cct_k``. ``cct_k`` is the generating temperature for
blackbody and D-series lights (known by construction) and the Ohno (2013) CCT of the SPD
for fluorescent and LED lights, the same function the ARRI-matrix lookup of the research
scripts uses.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache

import colour
import numpy as np

from training.alexa import ssf_core as s

CLASSES = ("daylight", "tungsten", "fluorescent", "led")  # order used in profile.json

TUNGSTEN_K = tuple(range(2000, 4001, 250))
DAYLIGHT_K = tuple(range(4000, 11001, 500))
FLUORESCENT = tuple(f"FL{i}" for i in range(1, 13))
LED = ("LED-B1", "LED-B2", "LED-B3", "LED-B4", "LED-B5", "LED-BH1", "LED-RGB1", "LED-V1", "LED-V2")

# Gate (docs/02 §C2 "Gate lights"): held out of the gate fit.
GATE_BLACKBODY_K = 2850
GATE_DAYLIGHT_K = (5250, 6750)
GATE_FLUORESCENT = ("FL3.1", "FL3.8", "FL3.15")
GATE_LED = ("LED-B3", "LED-RGB1", "LED-V2")  # also removed from the LED class for the gate refit


@dataclass(frozen=True)
class Light:
    """One reference or test light."""

    name: str          # "blackbody-2850K", "daylight-5250K", "FL3.1", "LED-B3"
    klass: str         # one of CLASSES
    cct_k: float       # CCT used for the mired axis (see module docstring)
    spd: np.ndarray    # (81,), peak = 1

    @property
    def mired(self) -> float:
        return 1e6 / self.cct_k


# ----------------------------------------------------------------------------
# SPD generators (ssf_core conventions)
# ----------------------------------------------------------------------------
def blackbody_spd(cct_k: float) -> np.ndarray:
    """Planckian radiator, peak-normalised (``ssf_core.illuminant(T, 'planck')``)."""
    return s.illuminant(float(cct_k), "planck")


def daylight_spd(cct_k: float) -> np.ndarray:
    """CIE D series at ``cct_k`` with the rawtoaces CCT correction, peak-normalised.

    Same code path as ``ssf_core.illuminant(T, 'rta')`` for T >= 4000, including colour-science's
    rounding of M1 and M2 to three decimals (as the CIE recommends). That rounding makes the
    spectrum a staircase in T at the 1e-4 level; it is part of the reference and is documented
    in ``training/sensor_map/README.md``. The formula's domain is 4000-25000 K after the
    correction; 25000 K is 14 K past it, colour-science warns, and the polynomial is smooth there.
    """
    return _d_series_checked(cct_k)


def _d_series(cct_k: float) -> np.ndarray:
    xy = colour.temperature.CCT_to_xy_CIE_D(float(cct_k) * 1.4388 / 1.4380)
    sd = colour.sd_CIE_illuminant_D_series(xy)
    v = sd.copy().align(s.SHAPE).values
    return v / v.max()


def ohno_cct_k(spd: np.ndarray) -> float:
    """Ohno (2013) CCT of an SPD, as used for the ARRI matrix lookup in the research scripts."""
    return s.illuminant_cct(spd)


@lru_cache(maxsize=None)
def _grid(kind: str, start_k: int, stop_k: int, step_k: int) -> tuple[np.ndarray, np.ndarray]:
    ts = np.arange(start_k, stop_k + 1, step_k, dtype=float)
    make = blackbody_spd if kind == "blackbody" else _d_series_checked
    es = np.stack([make(t) for t in ts])
    ts.setflags(write=False)
    es.setflags(write=False)
    return ts, es


def _d_series_checked(cct_k: float) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=colour.utilities.ColourUsageWarning)
        return _d_series(cct_k)


def blackbody_grid(start_k: int, stop_k: int, step_k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """(temperatures (n,), SPDs (n, 81)) of blackbody radiators on a K grid, inclusive."""
    return _grid("blackbody", start_k, stop_k, step_k)


def daylight_grid(start_k: int, stop_k: int, step_k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """(temperatures (n,), SPDs (n, 81)) of CIE D-series lights on a K grid, inclusive."""
    return _grid("daylight", start_k, stop_k, step_k)


# ----------------------------------------------------------------------------
# Named and class lights
# ----------------------------------------------------------------------------
def named_light(name: str, klass: str) -> Light:
    spd = s.named_illuminant(name)
    return Light(name=name, klass=klass, cct_k=ohno_cct_k(spd), spd=spd)


def blackbody_light(cct_k: float, klass: str = "tungsten") -> Light:
    return Light(f"blackbody-{cct_k:g}K", klass, float(cct_k), blackbody_spd(cct_k))


def daylight_light(cct_k: float, klass: str = "daylight") -> Light:
    return Light(f"daylight-{cct_k:g}K", klass, float(cct_k), _d_series_checked(cct_k))


def _sorted(lights: list[Light]) -> list[Light]:
    return sorted(lights, key=lambda L: (L.mired, L.name))


def reference_lights(exclude: tuple[str, ...] = ()) -> dict[str, list[Light]]:
    """The reference lights per class, each list sorted by (mired, name).

    ``exclude`` removes lights by name (the gate removes ``GATE_LED``).
    """
    out: dict[str, list[Light]] = {
        "tungsten": [blackbody_light(t) for t in TUNGSTEN_K],
        "daylight": [daylight_light(t) for t in DAYLIGHT_K],
        "fluorescent": [named_light(n, "fluorescent") for n in FLUORESCENT],
        "led": [named_light(n, "led") for n in LED],
    }
    for k in out:
        out[k] = _sorted([L for L in out[k] if L.name not in exclude])
    return {k: out[k] for k in CLASSES}


def gate_lights() -> list[Light]:
    """The nine held-out gate lights, in the order of ``docs/02``."""
    lights = [blackbody_light(GATE_BLACKBODY_K, "tungsten")]
    lights += [daylight_light(t, "daylight") for t in GATE_DAYLIGHT_K]
    lights += [named_light(n, "fluorescent") for n in GATE_FLUORESCENT]
    lights += [named_light(n, "led") for n in GATE_LED]
    return lights


def tint_spd(base: np.ndarray, amount: float, center_nm: float = 545.0, width_nm: float = 45.0) -> np.ndarray:
    """A light with a smooth green excess (amount > 0) or deficit (< 0) added to ``base``.

    Used only to test the Duv readout away from the locus. Not a physical light class.
    """
    g = np.exp(-0.5 * ((s.WL - center_nm) / width_nm) ** 2)
    v = base * (1.0 + amount * g)
    return v / v.max()

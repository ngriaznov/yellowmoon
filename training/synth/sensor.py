"""The synthetic sensor: exposure, CFA sampling, Poisson-Gaussian noise, quantisation and clipping.

Model (all in digital numbers, DN, of the stored sample scale)::

    x      noise-free signal above black (DN), from the camera RGB of the scene times the exposure scale
    n_e    ~ Poisson(min(x, x_sat) / a)                       photo-electrons; a = conversion gain, DN per electron
    DN     = round(black + a * n_e + N(0, b)),  clipped to [0, white]
    var(DN | x) = a * x + b  (+ 1/12 from the rounding; the clip makes the top of the range a clipped Gaussian)

with ``a = (white - black) / FWC * ISO / 100`` and ``b = (a * read_noise_e)^2``. ``FWC`` is the full-well capacity
in electrons: an assumed areal density times the pixel area of the body (``spectra.FULL_WELL_E_PER_UM2``,
``Body.pitch_um``). ``read_noise_e`` drops from ``read_noise_low_e`` to ``read_noise_high_e`` at the body's
conversion-gain switch ISO. Both are **synthetic assumptions that define the truth**, never measurements.

The pixel saturates in electrons, at 5 % above the full well that ISO 100 maps to the white level:
``x_sat = 1.05 * (white - black) * ISO / 100`` DN (at least the ISO 100 value). At ISO 100 that is just above the
white level, so a bright pixel clips as a clipped Gaussian; at higher ISO the ADC range holds only a fraction of the
well and every pixel above it is at the white level, with no noise tail below it, as on a real body.

The exposure is the camera's meter: the green response of the 18 % grey patch is placed at ``GREY_FRACTION`` of
``white - black`` (times ``2^EC``), whatever the ISO, so a higher ISO means fewer electrons for the same raw
value, as on a real body at a matched exposure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from training.synth import spectra as sp

#: Where the metered 18 % grey lands in the raw data (green, fraction of white - black). A real body at EC 0 puts
#: mid-grey a little over 3 stops below the white level.
GREY_FRACTION = 0.12
SATURATION = 1.05


@dataclass(frozen=True)
class Mode:
    """A raw sample format: bit depth and the levels as decoded (``ym_raw::decode`` scale)."""

    bits: int
    black: int
    white: int
    shift: int  # right shift from the 14-bit scale of the SR2 levels to this mode's scale

    @property
    def span(self) -> int:
        return self.white - self.black


MODE_14 = Mode(14, 512, 16383, 0)
MODE_12 = Mode(12, 128, 4095, 2)
MODES = {14: MODE_14, 12: MODE_12}

CFA_PATTERNS = {  # channel index (0 R, 1 G, 2 B) at (row & 1, col & 1)
    "RGGB": ((0, 1), (1, 2)),
    "GRBG": ((1, 0), (2, 1)),
    "GBRG": ((1, 2), (0, 1)),
    "BGGR": ((2, 1), (1, 0)),
}


@dataclass(frozen=True)
class NoiseModel:
    """The known noise truth of one file."""

    iso: int
    bits: int
    gain_dn_per_e: float  # a
    read_noise_e: float
    full_well_e: float
    regime: str  # "single gain" | "low gain" | "high gain"
    black: int
    white: int

    @property
    def a_dn(self) -> float:
        return self.gain_dn_per_e

    @property
    def b_dn2(self) -> float:
        return (self.gain_dn_per_e * self.read_noise_e) ** 2

    @property
    def a_norm(self) -> float:
        """Shot-noise gain in normalised units ((DN - black) / (white - black)): ``var = a_norm * x + b_norm``."""
        return self.a_dn / (self.white - self.black)

    @property
    def b_norm(self) -> float:
        return self.b_dn2 / (self.white - self.black) ** 2

    def to_json(self) -> dict:
        return {
            "model": "Poisson-Gaussian: var(DN) = a * (DN - black) + b + 1/12 (rounding); clipped at white",
            "synthetic": True,
            "note": "synthetic assumptions that define the truth, not measurements of any camera",
            "iso": self.iso,
            "bits": self.bits,
            "a_dn_per_electron": self.a_dn,
            "b_dn2": self.b_dn2,
            "a_normalised": self.a_norm,
            "b_normalised": self.b_norm,
            "read_noise_electrons": self.read_noise_e,
            "full_well_electrons_at_this_iso": self.full_well_e,
            "gain_regime": self.regime,
            "row_noise_sigma_dn": 0.0,
            "plane_offsets_dn": [0.0, 0.0, 0.0, 0.0],
            "quantisation_step_dn": 1.0,
            "black": self.black,
            "white": self.white,
            "saturation_dn_above_black": SATURATION * (self.white - self.black) * max(self.iso, 100) / 100.0,
        }


def noise_model(body: sp.Body, iso: int, mode: Mode) -> NoiseModel:
    """Noise truth of ``body`` at ``iso`` in sample format ``mode``."""
    fwc_100 = sp.FULL_WELL_E_PER_UM2 * body.pitch_um**2  # electrons at ISO 100
    gain = mode.span / fwc_100 * max(iso, 100) / 100.0
    if body.switch_iso is None:
        regime, read = "single gain", body.read_noise_low_e
    elif iso >= body.switch_iso:
        regime, read = "high gain", body.read_noise_high_e
    else:
        regime, read = "low gain", body.read_noise_low_e
    return NoiseModel(iso, mode.bits, gain, read, mode.span / gain, regime, mode.black, mode.white)


def exposure_scale(grey_green_response: float, mode: Mode, ev_comp: float = 0.0) -> float:
    """DN (above black) per unit of camera response: puts the grey patch's green at ``GREY_FRACTION * 2^EC`` of range."""
    return GREY_FRACTION * (2.0**ev_comp) * mode.span / float(grey_green_response)


def cfa_mosaic(canvas: np.ndarray, cfa: str) -> np.ndarray:
    """(H, W) mosaic of an (H, W, 3) RGB image for a CFA pattern name; float32."""
    h, w, _ = canvas.shape
    pat = CFA_PATTERNS[cfa]
    out = np.empty((h, w), dtype=np.float32)
    for dy in (0, 1):
        for dx in (0, 1):
            out[dy::2, dx::2] = canvas[dy::2, dx::2, pat[dy][dx]]
    return out


def sample(clean: np.ndarray, mode: Mode, nm: NoiseModel, rng: np.random.Generator, band_rows: int = 256) -> np.ndarray:
    """Poisson-Gaussian samples of a noise-free mosaic (DN above black, float) as ``uint16`` DN with black added.

    Deterministic for a given generator state and band size (the draw order is band by band)."""
    h, w = clean.shape
    out = np.empty((h, w), dtype=np.uint16)
    sat = SATURATION * mode.span * max(nm.iso, 100) / 100.0
    sigma = float(np.sqrt(nm.b_dn2))
    for r0 in range(0, h, band_rows):
        x = np.clip(clean[r0 : r0 + band_rows].astype(np.float64), 0.0, sat)
        ne = rng.poisson(x / nm.a_dn).astype(np.float64)
        y = nm.a_dn * ne + rng.normal(0.0, sigma, size=x.shape) + mode.black
        out[r0 : r0 + band_rows] = np.clip(np.rint(y), 0, mode.white).astype(np.uint16)
    return out


def clipped_fraction(mosaic: np.ndarray, mode: Mode) -> float:
    """Fraction of samples at the white level."""
    return float(np.mean(mosaic >= mode.white))

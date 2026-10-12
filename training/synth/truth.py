"""Ground truth: the ALEXA-estimate reference and the 'Sony through ARRI matrix' baseline.

Conventions are those of ``training.alexa.ssf_core`` and ``docs/02`` section C3 to C6 (the same arithmetic the
sensor-map gate uses, written to AWG4 instead of AP0):

1. Responses ``S^T (E o R)`` of a reflectance ``R`` under the light ``E``; for the reference ``S`` is the ALEXA
   estimate ``S_A``.
2. White balance to the light: divide each channel by the response to a perfect white (``R = 1``) under ``E``.
   (Auto white balance: C3 and C4 cancel, the scene white lands on ``[1, 1, 1]``.) In a mixed scene the white is
   the reference light's; a patch under another mixture of the lights has its own local white too.
3. ARRI's camera-native to AP0 matrix at the light's CCT, interpolated in mired between the bracketing entries of
   the 17-CCT table (clamped to 2000 to 11000 K), converted to AWG3 with the exact ``inv(AWG3 to AP0)`` (CAT02 from
   D65 to the ACES white), then AWG3 to AWG4 (same white, no adaptation).
4. LogC4 of the linear AWG4 values.

Values are **reflectance-referred**: exposure EV 0 means a perfect white diffuser (R = 1) under the light lands on
``[1, 1, 1]`` and the 18 % grey on ``[0.18, 0.18, 0.18]`` (LogC4 0.2784). Real renders add their own auto exposure,
so compare after normalising to the grey patch. Patches under extra irradiance (highlights, shadows, the ladder)
carry their factor ``m``; the highlight values are the **unclipped** truth.

The baseline ("Sony through ARRI matrix", no sensor map) feeds the Sony camera RGB, white-balanced to the light the
same way, straight into the same ARRI matrix and gamut steps, as if the Sony were an ALEXA.
"""

from __future__ import annotations

import numpy as np

from training.alexa import ssf_core as s
from training.sensor_map import fit
from training.synth import spectra as sp

# --- gamuts: derived from the primaries in f64, as ym-color does (docs/02 Constants) -------------------
_D65 = np.array([0.3127, 0.3290])
_ACES_WHITE = np.array([0.32168, 0.33767])
_AWG3_PRIMARIES = np.array([[0.6840, 0.3130], [0.2210, 0.8480], [0.0861, -0.1020]])
_AWG4_PRIMARIES = np.array([[0.7347, 0.2653], [0.1424, 0.8576], [0.0991, -0.0308]])
_AP0_PRIMARIES = np.array([[0.7347, 0.2653], [0.0, 1.0], [0.0001, -0.0770]])


def _npm(primaries: np.ndarray, white: np.ndarray) -> np.ndarray:
    import colour

    return colour.normalised_primary_matrix(primaries, white)


def gamut_matrices() -> dict[str, np.ndarray]:
    """``awg3_to_xyz``, ``awg4_to_xyz``, ``awg3_to_awg4`` and ``awg3_to_ap0`` (CAT02), exact in f64."""
    import colour

    awg3 = _npm(_AWG3_PRIMARIES, _D65)
    awg4 = _npm(_AWG4_PRIMARIES, _D65)
    ap0 = _npm(_AP0_PRIMARIES, _ACES_WHITE)
    cat = colour.adaptation.matrix_chromatic_adaptation_VonKries(
        colour.xy_to_XYZ(_D65), colour.xy_to_XYZ(_ACES_WHITE), "CAT02"
    )
    return {
        "awg3_to_xyz": awg3,
        "awg4_to_xyz": awg4,
        "awg3_to_awg4": np.linalg.inv(awg4) @ awg3,
        "awg3_to_ap0": np.linalg.inv(ap0) @ cat @ awg3,
    }


def native_to_awg3(cct_k: float) -> np.ndarray:
    """ARRI camera-native (balanced) to linear AWG3 at ``cct_k`` (mired interpolation, clamped to the table)."""
    ccts, mats = fit.arri_matrices()
    m_ap0 = s.interpolate_matrix(ccts, mats, float(cct_k))
    return np.linalg.inv(gamut_matrices()["awg3_to_ap0"]) @ m_ap0


# --- LogC4 (docs/02): ARRI LogC4 specification, 2022 ------------------------------------------------------
_A = (2.0**18 - 16.0) / 117.45
_B = (1023.0 - 95.0) / 1023.0
_C = 95.0 / 1023.0
_S = 7.0 * np.log(2.0) * 2.0 ** (7.0 - 14.0 * _C / _B) / (_A * _B)
_T = (2.0 ** (14.0 * (-_C / _B) + 6.0) - 64.0) / _A


def logc4_encode(x):
    """LogC4 of linear AWG4 (scalar or array)."""
    x = np.asarray(x, dtype=np.float64)
    safe = np.maximum(_A * x + 64.0, 1e-300)
    return np.where(x >= _T, (np.log2(safe) - 6.0) / 14.0 * _B + _C, (x - _T) / _S)


def responses(spectra: np.ndarray, light: np.ndarray, curves: np.ndarray) -> np.ndarray:
    """(K, 3) responses ``S^T (E o R_k)`` of K reflectances (unnormalised)."""
    return (spectra * light[None, :]) @ curves


def white_of(light: np.ndarray, curves: np.ndarray) -> np.ndarray:
    """Response to a perfect white, (3,)."""
    return light @ curves


def camera_white(light: np.ndarray, curves: np.ndarray) -> np.ndarray:
    """The camera RGB of the light with green = 1 (``w_true``; the raw white of a neutral surface)."""
    w = white_of(light, curves)
    return w / w[1]


def mixture_response(table: np.ndarray, mixture) -> np.ndarray:
    """Response of a reflectance mixture ``[(bank index, weight), ...]`` from a (K, 3) table."""
    return sum(wt * table[i] for i, wt in mixture)


def awg_values(balanced: np.ndarray, cct_k: float) -> dict:
    """AWG3, AWG4 (linear) and LogC4 of white-balanced native RGB (3,), after ARRI's matrix at ``cct_k``."""
    g = gamut_matrices()
    awg3 = native_to_awg3(cct_k) @ balanced
    awg4 = g["awg3_to_awg4"] @ awg3
    return {
        "native_balanced": balanced.tolist(),
        "awg3": awg3.tolist(),
        "awg4": awg4.tolist(),
        "logc4": logc4_encode(awg4).tolist(),
    }


class LightTruth:
    """References for one scene light (or one mixture) and one camera body.

    ``cct_k`` is the CCT of the reference light for the ARRI lookup. ``local`` gives, for a mixed scene, the pair of
    (left, right) SPDs and the per-probe weight is passed to :meth:`probe_values`.
    """

    def __init__(self, light: sp.SceneLight, body: sp.Body, bank_spectra: np.ndarray, left_right=None):
        self.light, self.body = light, body
        self.S_p = sp.camera_curves(body)
        self.S_a = sp.alexa_curves()
        self.bank = bank_spectra
        self.E = light.spd
        self.left_right = left_right  # (E_left, E_right) or None
        self.w_true = camera_white(self.E, self.S_p)
        self.w_alexa = camera_white(self.E, self.S_a)
        self.white_p = white_of(self.E, self.S_p)
        self.white_a = white_of(self.E, self.S_a)
        self._tab_p = self._tables(self.S_p)
        self._tab_a = self._tables(self.S_a)

    def _tables(self, curves):
        if self.left_right is None:
            return (responses(self.bank, self.E, curves),)
        return tuple(responses(self.bank, e, curves) for e in self.left_right)

    def camera_table(self) -> np.ndarray:
        """(K, 3) camera responses of the bank under the (reference) light (left light of a mixture: use ``tables``)."""
        return self._tab_p[0]

    def tables(self, which: str = "camera") -> tuple[np.ndarray, ...]:
        return self._tab_p if which == "camera" else self._tab_a

    @staticmethod
    def _local(tabs, mixture, s_weight):
        if len(tabs) == 1:
            return mixture_response(tabs[0], mixture)
        return (1.0 - s_weight) * mixture_response(tabs[0], mixture) + s_weight * mixture_response(tabs[1], mixture)

    def local_response(self, which: str, mixture, s_weight: float = 0.0) -> np.ndarray:
        """Response of a reflectance mixture under the light at a place with right-light weight ``s_weight``
        (``which``: "camera" or "alexa"); unnormalised and before irradiance."""
        return self._local(self.tables(which), mixture, s_weight)

    def local_white(self, curves, s_weight):
        """White response under the local mixture (mixed scenes)."""
        if self.left_right is None:
            return white_of(self.E, curves)
        el, er = self.left_right
        return white_of((1.0 - s_weight) * el + s_weight * er, curves)

    def probe_values(self, mixture, m: float, s_weight: float = 0.0) -> dict:
        """Truth of one probe: camera response (relative, before exposure), ALEXA reference and Sony baseline."""
        cam = m * self._local(self._tab_p, mixture, s_weight)
        alx = m * self._local(self._tab_a, mixture, s_weight)
        cct = self.light.cct_k
        out = {
            "camera_response": cam.tolist(),
            "alexa": awg_values(alx / self.white_a, cct),
            "sony_through_arri": awg_values(cam / self.white_p, cct),
        }
        if self.left_right is not None:
            out["alexa_local_white"] = awg_values(alx / self.local_white(self.S_a, s_weight), cct)
        return out

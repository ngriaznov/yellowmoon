"""The embedded camera JPEG: a plausible sRGB rendering of the scene, **synthetic and not Sony's rendering**.

It exists so that the auto-exposure comparison machinery (preview extraction, median luminance in linear light,
``docs/13`` auto-exposure gate) has something to compare with. It says nothing about Sony's tone curve:

1. the noise-free raw image (what the sensor would record), clipped at the white level;
2. the as-shot white balance (the file's SR2 gains, i.e. the true gains perturbed like an automatic white balance),
   clipped at the balanced white, so a clipped highlight goes white;
3. a colour matrix fitted on the 190 training reflectances under the true light, camera to linear sRGB (D65);
4. a fixed gain of 2.133 (``PREVIEW_GAIN``), a soft shoulder above 0.8 that maps the gain to 1.0, the sRGB transfer curve
   and a mild S-curve (``v - 0.12 sin(2 pi v) / (2 pi)``).

The gain is what makes the camera engine Sony-like. ``tools/eval/autoexp.py`` reads a camera's gain (the raw level at
which the JPEG shows middle grey, 0.18 in linear light) by quantile matching against our own scene picture, and reads
this engine **+0.045 EV above its own gain** (std 0.03 over the 37 scenes; the cause is not isolated, the C7 low-light
toe of our picture and the noise at the lower level are the suspects). It reads real cameras with the same method, so the
gain was calibrated on the measured value, not on the analytic one: ``known_gain_ev(2.133)`` is +1.0653 EV, and the
estimator reads the tier's median at +1.11 EV, the median gain of the cameras' own JPEG engines on the 12 local files
(``ym_color::exposure::CAMERA_GAIN_EV``, ``docs/02`` Auto exposure, decision of 2026-10-14). ``EV_auto`` is that same
constant, so the auto-exposure gate (G3, ``tools/eval/autoexp.py --gate``) passes on this tier by construction. The earlier
gain of 1.5 was +0.5575 EV (and was read at +0.55). ``known_gain_ev(gain)`` solves the tone function for the raw level that
reads 0.18.

The JPEG is baseline, 4:2:0, quality 90, starts ``FF D8 FF DB`` like Sony previews (without JFIF) and carries a
``COM`` segment that says it is synthetic.
"""

from __future__ import annotations

import io

import numpy as np

from training.alexa import ssf_core as s

JPEG_QUALITY = 90
#: The fixed gain of the synthetic camera engine (linear sRGB, balanced white = 1). `known_gain_ev(2.133)` is +1.0653 EV, which
#: `tools/eval/autoexp.py` reads at +1.11 EV (its +0.045 EV reading bias on this tier, see the module docstring).
PREVIEW_GAIN = 2.133
#: What the estimator of `tools/eval/autoexp.py` must read as the median gain of the tier: `ym_color::exposure::CAMERA_GAIN_EV`.
TARGET_MEASURED_GAIN_EV = 1.11
#: The engine's own gain in stops, `known_gain_ev(PREVIEW_GAIN)` rounded to 4 decimals (`autoexp.SYNTH_GAIN_EV`).
KNOWN_GAIN_EV = 1.0653
COMMENT = b"Yellowmoon synthetic test file: this preview is a made-up sRGB rendering, not a camera rendering."


def camera_to_srgb_matrix(light_spd: np.ndarray, curves: np.ndarray) -> np.ndarray:
    """White-preserving 3x3 from white-balanced camera RGB to linear sRGB (D65), fitted on the 190 patches."""
    import colour

    r190 = s.training_reflectances()
    rgb = s.camera_rgb_wb(curves, light_spd, r190)
    xyz = s.target_XYZ(light_spd, r190)  # adapted to the ACES white
    m_ap0 = s.fit_matrix_linear(rgb, xyz)  # camera -> AP0
    to_xyz = s.AP0_TO_XYZ @ m_ap0  # camera -> XYZ (ACES white)
    d65 = colour.xy_to_XYZ(np.array([0.3127, 0.3290]))
    cat = colour.adaptation.matrix_chromatic_adaptation_VonKries(s.ACES_WHITE_XYZ, d65, "CAT02")
    srgb = colour.RGB_COLOURSPACES["sRGB"].matrix_XYZ_to_RGB
    m = srgb @ cat @ to_xyz
    return m / m.sum(axis=1, keepdims=True)  # rows sum to 1: white maps to white exactly


def srgb_oetf(v: np.ndarray) -> np.ndarray:
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.0031308, 12.92 * v, 1.055 * np.power(v, 1.0 / 2.4) - 0.055)


def tone(lin: np.ndarray) -> np.ndarray:
    """Linear sRGB (balanced, white = 1) to display code values in [0, 1]."""
    x = PREVIEW_GAIN * np.clip(lin, 0.0, None)
    shoulder = 0.8 + 0.2 * np.tanh((x - 0.8) / 0.2)
    x = np.where(x > 0.8, shoulder, x)
    v = srgb_oetf(x)
    return np.clip(v - 0.12 * np.sin(2.0 * np.pi * v) / (2.0 * np.pi), 0.0, 1.0)


def known_gain_ev(gain: float | None = None) -> float:
    """The synthetic engine's own gain in stops, `log2(0.18 / x)` for the balanced raw level `x` (white = 1) at which
    `tone` shows 0.18 in linear light (ISO 12232's grey, the anchor `tools/eval/autoexp.py` reads the camera gain at).

    Exact for a grey pixel (the matrix rows sum to 1, so `lin` is the raw level); the 8-bit quantisation is ignored (it is
    1 code at most). `known_gain_ev(1.5)` is 0.5575, `known_gain_ev(2.133)` is 1.0653.
    """
    from scipy.optimize import brentq

    global PREVIEW_GAIN
    saved = PREVIEW_GAIN
    PREVIEW_GAIN = saved if gain is None else gain
    try:
        def linear_code(x: float) -> float:
            v = float(tone(np.array([x]))[0])
            return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4

        x = brentq(lambda x: linear_code(x) - 0.18, 1e-3, 1.0, xtol=1e-14)
    finally:
        PREVIEW_GAIN = saved
    return float(np.log2(0.18 / x))


def render(raw_norm: np.ndarray, gains: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 from the noise-free raw image normalised to the range (1.0 = white level, unclipped)."""
    raw = np.clip(raw_norm, 0.0, 1.0)
    bal = np.clip(raw * gains[None, None, :], 0.0, 1.0)
    lin = bal @ matrix.T
    return np.rint(tone(lin) * 255.0).astype(np.uint8)


def block_mean(img: np.ndarray, factor: int) -> np.ndarray:
    """Box average by an integer factor (the full-size scene is five times the preview)."""
    if factor == 1:
        return img
    h, w, c = img.shape
    return img.reshape(h // factor, factor, w // factor, factor, c).mean(axis=(1, 3))


def encode_jpeg(rgb8: np.ndarray, comment: bytes = COMMENT) -> bytes:
    """Baseline JPEG that starts ``FF D8 FF DB`` (no JFIF) with a COM segment after the first quantisation table."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb8, "RGB").save(buf, "JPEG", quality=JPEG_QUALITY, subsampling=2, optimize=False, progressive=False)
    return restructure(buf.getvalue(), comment)


def restructure(jpeg: bytes, comment: bytes) -> bytes:
    """Drops APPn and COM segments before the scan and inserts ``comment`` after the first DQT segment."""
    assert jpeg[:2] == b"\xff\xd8"
    out = bytearray(b"\xff\xd8")
    pos = 2
    inserted = False
    while pos < len(jpeg):
        assert jpeg[pos] == 0xFF, f"marker expected at {pos}"
        marker = jpeg[pos + 1]
        length = int.from_bytes(jpeg[pos + 2 : pos + 4], "big")
        seg = jpeg[pos : pos + 2 + length]
        if marker == 0xDA:  # start of scan: the entropy-coded data and the end marker follow verbatim
            out += jpeg[pos:]
            break
        if 0xE0 <= marker <= 0xEF or marker == 0xFE:
            pos += 2 + length
            continue
        out += seg
        if marker == 0xDB and not inserted:
            out += b"\xff\xfe" + (len(comment) + 2).to_bytes(2, "big") + comment
            inserted = True
        pos += 2 + length
    assert inserted and out[:4] == b"\xff\xd8\xff\xdb"
    return bytes(out)

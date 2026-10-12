"""ELD-style virtual-camera noise for the training sampler (docs/07 step 4, docs/06 "Training sampler").

A *virtual camera* is a random draw of the physical parameters of a 14-bit Sony-like sensor:

* overall gain ``K`` (DN per electron), tied to ISO: ``K = k0 * ISO / 100``; ``log K`` is uniform over
  the A1's ISO range plus or minus one stop (ISO 50..64000);
* read noise ``sigma_read`` with ``log sigma_read`` linear in ``log K`` plus Gaussian scatter, drawn
  from a **Tukey-lambda** distribution (heavy tails) scaled to that standard deviation;
* row noise ``sigma_row`` (one offset per sensor row), also linear in ``log K`` with scatter;
* per-CFA-plane **colour bias** (a constant offset in DN per plane, the black-level error ELD found
  to be the most valuable single term);
* Poisson shot noise on the electron count; optional dual conversion gain.

Ranges come from physics for 14-bit Sony sensors and from the *shape* of the relations published in the
ELD paper (Wei et al., CVPR 2020: ``log sigma`` linear in ``log K``, Tukey-lambda lambda around 0, row
noise a small fraction of read noise). ELD's released parameter files are evaluation-only and are never
loaded here. All randomness comes from a ``numpy.random.Generator`` handed in by the caller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

BLACK_DN = 512.0
WHITE_DN = 16383.0
RANGE_DN = WHITE_DN - BLACK_DN
K0_DN_PER_E = 0.25  # DN per electron at ISO 100 (about 63 ke- full scale on the 14-bit range)
ISO_MIN = 50.0
ISO_MAX = 64000.0
TUKEY_LAMBDA_RANGE = (-0.14, 0.13)


def tukey_lambda_std(lam: float) -> float:
    """Standard deviation of the Tukey lambda distribution with shape ``lam`` (valid for ``lam > -0.5``)."""
    if abs(lam) < 1e-6:
        return math.pi / math.sqrt(3.0)
    a = 1.0 / (1.0 + 2.0 * lam)
    b = math.exp(2.0 * math.lgamma(lam + 1.0) - math.lgamma(2.0 * lam + 2.0))
    return math.sqrt(2.0 / (lam * lam) * (a - b))


def sample_tukey_lambda(rng: np.random.Generator, lam: float, size: tuple[int, ...] | int, std: float = 1.0) -> np.ndarray:
    """Zero-mean samples of a Tukey-lambda distribution scaled to standard deviation ``std``."""
    u = rng.uniform(1e-7, 1.0 - 1e-7, size=size)
    if abs(lam) < 1e-6:
        q = np.log(u / (1.0 - u))
    else:
        q = (u**lam - (1.0 - u) ** lam) / lam
    return q * (std / tukey_lambda_std(lam))


@dataclass(frozen=True)
class VirtualCamera:
    """One draw of the sensor parameters; all noise terms are in DN on the 14-bit scale."""

    iso: float
    k_dn_per_e: float  # gain K: shot-noise variance in DN^2 = K * signal_dn
    sigma_read_dn: float  # standard deviation of the (Tukey-lambda) read noise
    tukey_lambda: float
    sigma_row_dn: float
    bias_dn: tuple[float, float, float, float]  # per CFA plane (R, G1, G2, B)
    black_dn: float = BLACK_DN
    white_dn: float = WHITE_DN
    meta: dict = field(default_factory=dict, compare=False)

    @property
    def range_dn(self) -> float:
        """White minus black, the normalisation divisor."""
        return self.white_dn - self.black_dn

    @property
    def a_norm(self) -> float:
        """Shot-noise slope in normalised units (docs/06's ``a``): ``K / (white - black)``."""
        return self.k_dn_per_e / self.range_dn


def sample_camera(rng: np.random.Generator, iso_range: tuple[float, float] = (ISO_MIN, ISO_MAX)) -> VirtualCamera:
    """Draw a virtual camera (docs/07 step 4)."""
    log_iso = rng.uniform(math.log(iso_range[0]), math.log(iso_range[1]))
    iso = math.exp(log_iso)
    k0 = K0_DN_PER_E * 2.0 ** rng.uniform(-0.5, 0.5)
    k = k0 * iso / 100.0
    # Read noise: log sigma_read = slope * log K + intercept + scatter (ELD form), sigma_read(K=1) in 1.5..4.5 DN.
    slope_r = rng.uniform(0.45, 0.65)
    sigma_read_k1 = math.exp(rng.uniform(math.log(1.5), math.log(4.5)))
    sigma_read = math.exp(slope_r * math.log(k) + math.log(sigma_read_k1) + rng.normal(0.0, 0.15))
    # Optional dual conversion gain: read noise drops above a break ISO.
    dual = rng.random() < 0.5
    if dual:
        iso_break = math.exp(rng.uniform(math.log(250.0), math.log(1600.0)))
        if iso >= iso_break:
            sigma_read *= rng.uniform(0.45, 0.75)
    sigma_read = float(np.clip(sigma_read, 0.3, 60.0))
    lam = float(rng.uniform(*TUKEY_LAMBDA_RANGE))
    # Row noise: a small fraction of read noise (ELD: ~0.16 at K = 1.5); some bodies have none.
    if rng.random() < 0.2:
        sigma_row = 0.0
    else:
        ratio = math.exp(rng.uniform(math.log(0.02), math.log(0.2)) + rng.normal(0.0, 0.15))
        sigma_row = float(sigma_read * ratio)
    bias_scale = rng.uniform(0.3, 3.0)
    bias = tuple(float(b) for b in rng.normal(0.0, bias_scale, size=4))
    return VirtualCamera(
        iso=float(iso),
        k_dn_per_e=float(k),
        sigma_read_dn=sigma_read,
        tukey_lambda=lam,
        sigma_row_dn=float(sigma_row),
        bias_dn=bias,  # type: ignore[arg-type]
        meta={"dual_gain": bool(dual)},
    )


def cfa_site_index(h: int, w: int) -> np.ndarray:
    """``[h, w]`` array with the RGGB plane index (0 R, 1 G1, 2 G2, 3 B) of every photosite."""
    yy = (np.arange(h) % 2)[:, None]
    xx = (np.arange(w) % 2)[None, :]
    return yy * 2 + xx


def apply_noise(signal_dn: np.ndarray, cam: VirtualCamera, rng: np.random.Generator) -> np.ndarray:
    """Add Poisson-Gaussian-Tukey, row and colour-bias noise to a clean signal.

    ``signal_dn``: ``[H, W]`` clean signal above black, in DN (float, may exceed the range: clipping
    happens later). Returns the noisy value above black, not yet quantised or clipped. The photosite
    layout is RGGB with the origin at the top left, so plane ``(y % 2) * 2 + (x % 2)``.
    """
    h, w = signal_dn.shape
    electrons = rng.poisson(np.maximum(signal_dn, 0.0) / cam.k_dn_per_e).astype(np.float64)
    out = electrons * cam.k_dn_per_e
    if cam.sigma_read_dn > 0.0:
        out += sample_tukey_lambda(rng, cam.tukey_lambda, (h, w), cam.sigma_read_dn)
    if cam.sigma_row_dn > 0.0:
        out += rng.normal(0.0, cam.sigma_row_dn, size=(h, 1))
    out += np.asarray(cam.bias_dn)[cfa_site_index(h, w)]
    return out


def noise_variance_dn2(signal_dn: np.ndarray, cam: VirtualCamera, q_dn: np.ndarray | float) -> np.ndarray:
    """The true model (docs/06): ``K * s + sigma_read^2 + sigma_row^2 + q^2 / 12`` in DN^2."""
    s = np.maximum(signal_dn, 0.0)
    return cam.k_dn_per_e * s + cam.sigma_read_dn**2 + cam.sigma_row_dn**2 + np.square(q_dn) / 12.0

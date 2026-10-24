"""The one fixed transform from the lens plane to the bokeh network's input, and the two constants that turn the
network's answer into the spec's units. The app (`ym_ml`, Rust) implements exactly this; `export.py` writes the
constants into the weights file's header and a fixture that the Rust tests reproduce.

Input to the transform: the lens plane (`docs/21` section 3) as scene-linear camera RGB, any overall scale, after
a box reduction to a long edge of `FRAME_LONG` (1024) pixels (the plane is 1024 to 2047 px long; never enlarged).

    Y  = 0.2126 R + 0.7152 G + 0.0722 B                      per pixel
    m  = max(lower median of Y over the whole plane, 1e-3 * mean(Y), 1e-12)
    v  = RGB * (MID / m)                                      the plane's median luminance becomes 0.18
    e  = log2(1 + max(v, 0) / K) / STOPS                      per channel; K = MID / 32, STOPS = 16
    x  = (e - INPUT_MEAN) / INPUT_STD                         the network's input (`model.py` does this step)

There is no per-image tone curve: one gain (the median) and one fixed curve. A plane that is twice as bright gives
the same input. The curve is linear near black (no noise blown up there) and logarithmic above, so a clipped light
at 100 times the median still lies inside the range (about 0.9). The lower median of `n` values is the element at
index (n - 1) // 2 of the sorted values, so both runtimes pick the same pixel.

The network answers per pixel (output at half the input's resolution):

    mu = log(r)    r the disc radius in pixels of the reduced plane, floored at `R_FLOOR_PX`
    s  = log(beta) beta the Laplace scale of the error of mu (a relative error: beta 0.25 is about 25 %)

The app turns that into the spec's units (`docs/21` section 7) with the plane's half-diagonal `hd` in pixels
(`hypot(w, h) / 2` of the reduced plane, before it is padded to a multiple of 32; the padding is replicated edge
pixels and the output is cropped back):  b = exp(mu) / hd  and  c = exp(-exp(s) / CONF_BETA0).
"""

import numpy as np
import torch

#: The long edge the network is fed (pixels). Training sources are at most this long too.
FRAME_LONG = 1024
#: `docs/21` section 4: the smallest blur that matters and the cap, in half-diagonal units.
B_MIN = 0.004
B_CAP = 0.03
#: Disc radii below this many pixels are all "sharp": the training target is `log(max(r, R_FLOOR_PX))`.
R_FLOOR_PX = 1.0
#: The network's input is divisible by this (stride 32 plus the half-resolution output).
PAD_MULTIPLE = 32

LUMA = (0.2126, 0.7152, 0.0722)
MID = 0.18
K = MID / 32.0
STOPS = 16.0
MEDIAN_FLOOR_OF_MEAN = 1e-3
INPUT_MEAN = 0.30
INPUT_STD = 0.15
#: confidence c = exp(-beta / CONF_BETA0)
CONF_BETA0 = 0.25


def luma_t(rgb: torch.Tensor) -> torch.Tensor:
    """(N, 3, H, W) -> (N, 1, H, W) luminance."""
    w = rgb.new_tensor(LUMA).view(1, 3, 1, 1)
    return (rgb * w).sum(1, keepdim=True)


def lower_median_t(y: torch.Tensor) -> torch.Tensor:
    """(N, 1, H, W) -> (N,): the lower median of each image (`torch.median` returns the lower one)."""
    return y.flatten(1).median(dim=1).values


def encode_t(rgb: torch.Tensor, gain: torch.Tensor | None = None) -> torch.Tensor:
    """The module docstring's `e` for (N, 3, H, W) linear RGB. `gain` (N,) multiplies `v` after the median step:
    training jitters it, because a training crop's median is not the whole frame's. Returns float32."""
    rgb = rgb.float()
    y = luma_t(rgb)
    m = torch.maximum(lower_median_t(y), y.flatten(1).mean(1) * MEDIAN_FLOOR_OF_MEAN).clamp(min=1e-12)
    v = rgb * (MID / m).view(-1, 1, 1, 1)
    if gain is not None:
        v = v * gain.view(-1, 1, 1, 1)
    return torch.log2(1.0 + v.clamp(min=0.0) / K) / STOPS


def encode_np(rgb: np.ndarray) -> np.ndarray:
    """The same in float64 for one (3, H, W) image; used by the fixtures and the tests."""
    rgb = np.asarray(rgb, np.float64)
    y = LUMA[0] * rgb[0] + LUMA[1] * rgb[1] + LUMA[2] * rgb[2]
    flat = np.sort(y.ravel())
    m = max(flat[(flat.size - 1) // 2], y.mean() * MEDIAN_FLOOR_OF_MEAN, 1e-12)
    v = np.maximum(rgb * (MID / m), 0.0)
    return np.log2(1.0 + v / K) / STOPS


def normalise(e: torch.Tensor) -> torch.Tensor:
    """The network's input from `e`."""
    return (e - INPUT_MEAN) / INPUT_STD


def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    """sRGB-encoded [0, 1] to linear light (the IEC 61966-2-1 curve)."""
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055).clamp(min=0) ** 2.4)


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    """Linear light to sRGB-encoded (for previews)."""
    x = x.clamp(0, 1)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x.clamp(min=1e-9) ** (1 / 2.4) - 0.055)


def radius_and_confidence(out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The network's (N, 2, h, w) output -> (r in pixels, beta, c)."""
    r = torch.exp(out[:, 0].float())
    beta = torch.exp(out[:, 1].float().clamp(-6, 3))
    return r, beta, torch.exp(-beta / CONF_BETA0)

"""Loss of `jdd` (docs/07): L1 on ``log(1 + 16 x)`` + 0.2 x (1 - MS-SSIM) on the same + gradient-domain L1.

All three terms act on the log-encoded images, so shadows count as much as highlights (the network's
outputs feed a log-based colour pipeline). The encoding is extended oddly to negative values
(``sign(x) * log(1 + 16 |x|)``), which keeps the gradient alive for small negative noise excursions.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_GAIN = 16.0
MSSSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
# Largest value the log-encoded images take: clip level g0_max (~3.2) -> log(1 + 16 * 4) with headroom.
LOG_DATA_RANGE = math.log1p(LOG_GAIN * 4.0)


def log_encode(x: torch.Tensor) -> torch.Tensor:
    """``sign(x) * log(1 + 16 |x|)``."""
    return torch.sign(x) * torch.log1p(LOG_GAIN * x.abs())


def gradient_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean absolute difference of the horizontal and vertical finite differences."""
    dx = (pred[..., :, 1:] - pred[..., :, :-1]) - (target[..., :, 1:] - target[..., :, :-1])
    dy = (pred[..., 1:, :] - pred[..., :-1, :]) - (target[..., 1:, :] - target[..., :-1, :])
    return 0.5 * (dx.abs().mean() + dy.abs().mean())


def _gaussian_kernel(window: int, sigma: float, channels: int, like: torch.Tensor) -> torch.Tensor:
    ax = torch.arange(window, dtype=like.dtype, device=like.device) - (window - 1) / 2.0
    g = torch.exp(-(ax**2) / (2.0 * sigma**2))
    g = g / g.sum()
    k = (g[:, None] * g[None, :])[None, None]
    return k.expand(channels, 1, window, window).contiguous()


def _ssim_cs(
    x: torch.Tensor, y: torch.Tensor, kernel: torch.Tensor, data_range: float
) -> tuple[torch.Tensor, torch.Tensor]:
    c = x.shape[1]
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2

    def filt(t: torch.Tensor) -> torch.Tensor:
        return F.conv2d(t, kernel, groups=c)

    mu_x, mu_y = filt(x), filt(y)
    sxx = filt(x * x) - mu_x**2
    syy = filt(y * y) - mu_y**2
    sxy = filt(x * y) - mu_x * mu_y
    cs = (2.0 * sxy + c2) / (sxx + syy + c2)
    lum = (2.0 * mu_x * mu_y + c1) / (mu_x**2 + mu_y**2 + c1)
    return (lum * cs).mean(), cs.mean()


def ms_ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    data_range: float = LOG_DATA_RANGE,
    window: int = 11,
    sigma: float = 1.5,
    weights: tuple[float, ...] = MSSSIM_WEIGHTS,
) -> torch.Tensor:
    """Multi-scale SSIM (Wang et al. 2003) of two ``[B, C, H, W]`` images, in ``(-inf, 1]``, 1 = identical.

    The number of scales shrinks automatically for small images (a scale is used while the image is at
    least ``window`` pixels on each side); the remaining weights are renormalised.
    """
    h, w = x.shape[-2:]
    levels = 1
    while levels < len(weights) and min(h, w) // (2**levels) >= window:
        levels += 1
    wts = torch.tensor(weights[:levels], dtype=x.dtype, device=x.device)
    wts = wts / wts.sum()
    kernel = _gaussian_kernel(window, sigma, x.shape[1], x)
    result = x.new_ones(())
    for i in range(levels):
        ssim, cs = _ssim_cs(x, y, kernel, data_range)
        term = F.relu(ssim if i == levels - 1 else cs)
        result = result * term ** wts[i]
        if i < levels - 1:
            x, y = F.avg_pool2d(x, 2), F.avg_pool2d(y, 2)
    return result


class JddLoss(nn.Module):
    """``L1(log) + w_ssim (1 - MS-SSIM(log)) + w_grad GradL1(log)``; returns ``(total, parts)``."""

    def __init__(self, w_ssim: float = 0.2, w_grad: float = 0.5) -> None:
        super().__init__()
        self.w_ssim = w_ssim
        self.w_grad = w_grad

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:  # noqa: D102
        a, b = log_encode(pred), log_encode(target)
        l1 = (a - b).abs().mean()
        ssim = ms_ssim(a, b) if self.w_ssim > 0 else a.new_ones(())
        grad = gradient_l1(a, b) if self.w_grad > 0 else a.new_zeros(())
        total = l1 + self.w_ssim * (1.0 - ssim) + self.w_grad * grad
        parts = {"l1_log": float(l1.detach()), "ms_ssim": float(ssim.detach()), "grad_l1": float(grad.detach())}
        return total, parts

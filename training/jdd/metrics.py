"""PSNR in the linear and in the log domain, and the built-in bilinear baseline (docs/07 evaluation)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .loss import LOG_GAIN, log_encode
from .model import PLANE_SCALE, JddNet

LINEAR_PEAK = 1.0
LOG_PEAK = math.log1p(LOG_GAIN * LINEAR_PEAK)


def _psnr(mse: torch.Tensor, peak: float) -> torch.Tensor:
    return 10.0 * torch.log10(peak**2 / mse.clamp_min(1e-12))


def psnr_linear(pred: torch.Tensor, target: torch.Tensor, peak: float = LINEAR_PEAK) -> torch.Tensor:
    """Per-image PSNR (dB) of ``[B, C, H, W]`` linear images; negatives in ``pred`` are clipped to 0."""
    mse = (pred.clamp_min(0.0) - target).square().mean(dim=(1, 2, 3))
    return _psnr(mse, peak)


def psnr_log(pred: torch.Tensor, target: torch.Tensor, peak: float = LOG_PEAK) -> torch.Tensor:
    """Per-image PSNR (dB) in the ``log(1 + 16 x)`` domain (peak = value of x = 1)."""
    mse = (log_encode(pred.clamp_min(0.0)) - log_encode(target)).square().mean(dim=(1, 2, 3))
    return _psnr(mse, peak)


@torch.no_grad()
def bilinear_baseline(model: JddNet, x: torch.Tensor) -> torch.Tensor:
    """Bilinear demosaic of the planes in ``x`` (the network's own residual base), no denoising."""
    planes = x[:, 0:4] * model.cfg.plane_scale
    return F.pixel_shuffle(model.bilinear(planes), 2) * (1.0 / model.cfg.plane_scale)


def summarize(values: list[float]) -> dict[str, float]:
    """Mean, median and count of a list of dB values."""
    t = torch.tensor(values, dtype=torch.float64)
    return {"mean": float(t.mean()), "median": float(t.median()), "n": float(len(values))}


__all__ = ["PLANE_SCALE", "bilinear_baseline", "psnr_linear", "psnr_log", "summarize"]

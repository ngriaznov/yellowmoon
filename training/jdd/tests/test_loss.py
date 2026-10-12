import numpy as np
import torch
from scipy import ndimage

from training.jdd.loss import JddLoss, gradient_l1, log_encode, ms_ssim, _gaussian_kernel, _ssim_cs


def test_log_encode_is_odd_monotone_and_matches_formula():
    x = torch.linspace(-1, 3, 41)
    y = log_encode(x)
    assert torch.allclose(log_encode(-x), -y)
    assert (y[1:] > y[:-1]).all()
    assert abs(float(log_encode(torch.tensor(1.0))) - np.log1p(16.0)) < 1e-6


def test_ms_ssim_identity_and_degradation():
    g = torch.Generator().manual_seed(0)
    a = torch.rand(2, 3, 96, 96, generator=g) * 3
    assert abs(float(ms_ssim(a, a)) - 1.0) < 1e-5
    noisy = a + 0.5 * torch.randn(a.shape, generator=g)
    worse = a + 1.5 * torch.randn(a.shape, generator=g)
    assert float(ms_ssim(a, noisy)) > float(ms_ssim(a, worse))
    assert float(ms_ssim(a, noisy)) < 0.99
    assert float(ms_ssim(a[..., :32, :32], a[..., :32, :32])) > 0.999  # few scales on small crops


def test_ssim_matches_scipy_reference_single_scale():
    rng = np.random.default_rng(1)
    a = rng.random((64, 64))
    b = np.clip(a + 0.1 * rng.standard_normal((64, 64)), 0, 1)
    k = _gaussian_kernel(11, 1.5, 1, torch.zeros(1))
    ssim, _ = _ssim_cs(torch.from_numpy(a)[None, None].float(), torch.from_numpy(b)[None, None].float(), k, 1.0)

    def filt(x):  # scipy gaussian with the same support (radius 5), valid region only
        return ndimage.gaussian_filter(x, 1.5, truncate=5 / 1.5, mode="constant")[5:-5, 5:-5]

    mu_a, mu_b = filt(a), filt(b)
    saa, sbb, sab = filt(a * a) - mu_a**2, filt(b * b) - mu_b**2, filt(a * b) - mu_a * mu_b
    c1, c2 = 0.01**2, 0.03**2
    ref = ((2 * mu_a * mu_b + c1) * (2 * sab + c2) / ((mu_a**2 + mu_b**2 + c1) * (saa + sbb + c2))).mean()
    assert abs(float(ssim) - ref) < 2e-4


def test_gradient_l1():
    a = torch.rand(1, 3, 16, 16)
    assert float(gradient_l1(a, a)) == 0
    assert float(gradient_l1(a, a + 5.0)) < 1e-6  # constant offset has no gradient
    assert float(gradient_l1(a, a.flip(-1))) > 0.01


def test_loss_is_differentiable_finite_and_zero_at_target():
    loss = JddLoss()
    t = torch.rand(2, 3, 48, 48) * 2
    p = (t + 0.1 * torch.randn_like(t)).requires_grad_(True)
    val, parts = loss(p, t)
    val.backward()
    assert torch.isfinite(val) and p.grad.abs().sum() > 0 and set(parts) == {"l1_log", "ms_ssim", "grad_l1"}
    z, _ = loss(t.clone(), t)
    assert float(z) < 1e-5
    neg, _ = loss(-t.clone(), t)  # negative predictions are handled
    assert torch.isfinite(neg)

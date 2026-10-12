import math

import numpy as np

from training.jdd import noise
from training.jdd.noise import VirtualCamera


def cam(**kw):
    base = dict(iso=800.0, k_dn_per_e=2.0, sigma_read_dn=4.0, tukey_lambda=-0.1, sigma_row_dn=1.5, bias_dn=(1.0, -0.5, 0.5, -1.0))
    base.update(kw)
    return VirtualCamera(**base)


def test_tukey_lambda_std_formula_and_sampling():
    rng = np.random.default_rng(0)
    for lam in (-0.14, 0.0, 0.13):
        x = noise.sample_tukey_lambda(rng, lam, 400_000, std=3.0)
        assert abs(x.mean()) < 0.05
        assert abs(x.std() - 3.0) / 3.0 < 0.03
    assert abs(noise.tukey_lambda_std(0.0) - math.pi / math.sqrt(3)) < 1e-9
    # heavier tails for negative lambda at equal std
    a = noise.sample_tukey_lambda(rng, -0.14, 400_000)
    b = noise.sample_tukey_lambda(rng, 0.13, 400_000)
    assert np.mean(np.abs(a) > 4 * a.std()) > np.mean(np.abs(b) > 4 * b.std())


def test_noise_model_matches_empirical_variance():
    """var = K*s + sigma_read^2 + sigma_row^2 (docs/06), per CFA plane after removing colour bias."""
    rng = np.random.default_rng(1)
    c = cam()
    for level in (50.0, 800.0, 6000.0):
        sig = np.full((512, 512), level)
        out = noise.apply_noise(sig, c, rng)
        res = out - level
        idx = noise.cfa_site_index(512, 512)
        tol = 5 * math.sqrt((c.k_dn_per_e * level + c.sigma_read_dn**2) / (512 * 512 / 4) + c.sigma_row_dn**2 / 256)
        for p in range(4):
            sel = res[idx == p]
            assert abs(sel.mean() - c.bias_dn[p]) < tol  # colour bias is the per-plane mean offset
        # row noise is shared along a row: remove the bias per plane, then compare total variance
        plane0 = res[idx == 0]
        want = c.k_dn_per_e * level + c.sigma_read_dn**2 + c.sigma_row_dn**2
        assert abs(plane0.var() / want - 1) < 0.05, (level, plane0.var(), want)
        assert abs(noise.noise_variance_dn2(np.array(level), c, 0.0) / want - 1) < 1e-12


def test_row_noise_is_constant_along_rows():
    rng = np.random.default_rng(2)
    c = cam(sigma_read_dn=0.0, k_dn_per_e=1e-9, bias_dn=(0, 0, 0, 0), sigma_row_dn=2.0)
    out = noise.apply_noise(np.zeros((64, 200)), c, rng)
    assert np.allclose(out, out[:, :1], atol=1e-6)  # identical along each row
    assert 1.2 < out[:, 0].std() < 3.0


def test_sample_camera_ranges_and_determinism():
    cams = [noise.sample_camera(np.random.default_rng(i)) for i in range(300)]
    isos = np.array([c.iso for c in cams])
    assert isos.min() >= noise.ISO_MIN and isos.max() <= noise.ISO_MAX
    assert isos.max() / isos.min() > 100
    k = np.array([c.k_dn_per_e / c.iso for c in cams])
    assert k.max() / k.min() < 2.1  # K tied to ISO up to the +-0.5 stop sensor spread
    assert all(noise.TUKEY_LAMBDA_RANGE[0] <= c.tukey_lambda <= noise.TUKEY_LAMBDA_RANGE[1] for c in cams)
    assert any(c.sigma_row_dn == 0 for c in cams) and any(c.sigma_row_dn > 0 for c in cams)
    # log sigma_read grows with log K (slope between 0.3 and 0.8 on average)
    lk = np.log([c.k_dn_per_e for c in cams])
    ls = np.log([c.sigma_read_dn for c in cams])
    assert 0.3 < np.polyfit(lk, ls, 1)[0] < 0.8
    assert noise.sample_camera(np.random.default_rng(5)) == noise.sample_camera(np.random.default_rng(5))

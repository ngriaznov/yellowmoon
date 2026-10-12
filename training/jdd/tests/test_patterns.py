import numpy as np
import pytest

from training.jdd import patterns


@pytest.mark.parametrize("name", list(patterns.PATTERNS))
def test_pattern_valid_deterministic_and_nontrivial(name):
    f = patterns.PATTERNS[name]
    a = f(np.random.default_rng(3), 48, 56)
    b = f(np.random.default_rng(3), 48, 56)
    c = f(np.random.default_rng(4), 48, 56)
    assert a.shape == (48, 56, 3) and a.dtype == np.float32
    assert np.isfinite(a).all() and a.min() >= 0
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.std() > 1e-3


def test_zone_plate_has_content_near_nyquist():
    rng = np.random.default_rng(0)
    hi = []
    for s in range(6):
        img = patterns.zone_plate(np.random.default_rng(s), 128, 128)
        g = img.mean(axis=2)
        spec = np.abs(np.fft.fft2(g - g.mean()))
        fy = np.abs(np.fft.fftfreq(128))[:, None]
        fx = np.abs(np.fft.fftfreq(128))[None, :]
        hi.append(spec[(np.maximum(fy, fx) > 0.3)].sum() / spec.sum())
    assert max(hi) > 0.05
    del rng


def test_random_pattern_covers_all_names():
    seen = set()
    for s in range(120):
        _, desc = patterns.random_pattern(np.random.default_rng(s), 32, 32)
        seen.update(desc.split("+"))
    assert seen == set(patterns.PATTERNS)

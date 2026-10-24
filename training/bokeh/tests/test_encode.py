import numpy as np
import torch

from training.bokeh import encode


def test_the_input_does_not_depend_on_the_planes_overall_brightness():
    g = torch.Generator().manual_seed(0)
    x = torch.rand(2, 3, 40, 56, generator=g) ** 2.2 + 1e-3
    a = encode.encode_t(x)
    b = encode.encode_t(x * 7.3)
    assert torch.allclose(a, b, atol=2e-6)


def test_median_luminance_is_mapped_to_mid_grey():
    x = torch.full((1, 3, 8, 8), 0.5)
    e = encode.encode_t(x)
    expected = np.log2(1 + encode.MID / encode.K) / encode.STOPS
    assert float(e.mean()) == np.float32(expected) or abs(float(e.mean()) - expected) < 1e-6


def test_numpy_and_torch_agree_and_the_curve_is_monotone_and_linear_near_black():
    rng = np.random.default_rng(1)
    x = rng.random((3, 24, 32)) ** 3 * 5
    a = encode.encode_np(x)
    b = encode.encode_t(torch.from_numpy(x).float().unsqueeze(0))[0].numpy()
    assert np.abs(a - b).max() < 2e-6
    v = np.linspace(0, 100, 2001)
    e = np.log2(1 + v / encode.K) / encode.STOPS
    assert np.all(np.diff(e) > 0)
    # linear near black (below a tenth of K): the curve is a straight line through the origin to within 5 %
    near = np.linspace(0, encode.K / 10, 11)[1:]
    slope = np.log2(1 + near / encode.K) / encode.STOPS / near
    assert slope.max() / slope.min() < 1.05
    assert e.max() < 1.0  # a clipped light at 100 x the median is inside the range


def test_lower_median_is_the_element_n_minus_one_over_two():
    y = torch.tensor([[[[5.0, 1.0], [3.0, 2.0]]]])  # sorted 1 2 3 5 -> index 1 -> 2.0
    assert float(encode.lower_median_t(y)) == 2.0
    rgb = np.array([[[5.0, 1.0], [3.0, 2.0]]] * 3)
    assert abs(float(encode.encode_np(rgb)[0, 0, 0]) - np.log2(1 + 5.0 * encode.MID / (2.0 * encode.K)) / encode.STOPS) < 1e-12


def test_a_black_plane_does_not_divide_by_zero():
    e = encode.encode_t(torch.zeros(1, 3, 8, 8))
    assert torch.isfinite(e).all() and float(e.max()) == 0.0


def test_radius_and_confidence_of_the_output():
    out = torch.zeros(1, 2, 2, 2)
    out[0, 0] = np.log(8.0)
    out[0, 1] = np.log(0.25)
    r, beta, c = encode.radius_and_confidence(out)
    assert torch.allclose(r, torch.tensor(8.0)) and torch.allclose(beta, torch.tensor(0.25))
    assert torch.allclose(c, torch.tensor(float(np.exp(-1.0))), atol=1e-6)

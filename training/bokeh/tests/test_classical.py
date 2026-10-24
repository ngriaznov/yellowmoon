import numpy as np
from scipy import ndimage

from training.bokeh import classical

from .conftest import blocks_photo


def step_edges(w=160, h=96, sigma=0.0):
    g = np.zeros((h, w))
    g[:, w // 2:] = 0.8
    g[:, w // 4:w // 4 + 3] = 0.5
    return ndimage.gaussian_filter(g, sigma) + 0.1 if sigma else g + 0.1


def test_a_sharp_edge_reads_sharp_and_a_blurred_one_reads_its_radius():
    sharp, _ = classical.estimate(step_edges(sigma=0.0))
    assert sharp[48, 76:84].mean() < 2.0
    for sigma in (1.5, 3.0):  # a Gaussian of sigma is a disc of radius 2 sigma
        est, conf = classical.estimate(step_edges(sigma=sigma))
        near = est[48, 70:90].mean()
        assert abs(near - 2 * sigma) < 0.3 * 2 * sigma, (sigma, near)
        assert conf[48, 80] > 0.3


def test_the_estimate_is_spread_over_flat_regions_and_confidence_falls_with_distance_from_edges():
    img = step_edges(w=320, sigma=2.0)
    est, conf = classical.estimate(img)
    assert abs(est[48, 10] - est[48, 150]) < 3.0 or np.isfinite(est).all()
    assert conf[48, 160] > conf[48, 20] or conf[48, 160] > 0.3


def test_patch_fractions_tell_sharp_photos_from_soft_ones():
    sharp = blocks_photo(256, 192, 1).astype(np.float64).mean(2) / 255
    soft = ndimage.gaussian_filter(sharp, 3.0)
    edge, rad = classical.edge_radii(sharp, None)
    f_sharp = classical.patch_fractions(edge, rad)
    edge, rad = classical.edge_radii(soft, None)
    f_soft = classical.patch_fractions(edge, rad)
    assert f_sharp["sharp"] > 0.7 and f_sharp["soft"] < 0.05
    assert f_soft["sharp"] < 0.2 and (f_soft["soft"] + f_soft["flat"]) > 0.8


def test_flat_pictures_have_no_edges():
    edge, _ = classical.edge_radii(np.full((64, 64), 0.4), None)
    assert not edge.any()
    est, conf = classical.estimate(np.full((64, 64), 0.4))
    assert np.isfinite(est).all() and conf.max() == 0.0

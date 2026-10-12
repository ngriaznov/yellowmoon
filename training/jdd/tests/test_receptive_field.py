import pytest
import torch
import torch.nn as nn

from training.jdd.export import make_random_model
from training.jdd.model import JddConfig, JddNet
from training.jdd.receptive_field import (
    RF_LIMIT_FULL_PX,
    analyse,
    assert_within_limit,
    footprint_1d,
    probe_footprint,
)

CONFIGS = [JddConfig(channels=(4, 6, 8)), JddConfig(channels=(4, 6, 8), up_kernel=3, head_kernel=3)]


@pytest.mark.parametrize("cfg", CONFIGS)
def test_analytic_footprint_equals_gradient_probe(cfg):
    """The graph-derived footprint equals the brute-force gradient support, for every phase."""
    net = make_random_model(3, cfg).double().eval()
    size = 160
    for phase_y in range(4):
        for phase_x in range(0, 4, 3):
            oy_half, ox_half = 80 + phase_y, 80 + phase_x
            # one full-res pixel inside the half-res cell (second row/column of the shuffle)
            oy, ox = 2 * oy_half + 1, 2 * ox_half
            rows, cols = probe_footprint(net, oy, ox, size)
            assert rows == footprint_1d(net, oy), (phase_y, phase_x)
            assert cols == footprint_1d(net, ox), (phase_y, phase_x)


def test_default_receptive_field_numbers():
    rf = analyse(JddNet())
    assert rf.rf_half_px == 47  # docs/07: "about 47 half-res px"
    assert rf.rf_full_px == 94
    assert rf.halo_full_px == 64  # docs/01, docs/05, docs/07: 64 full-res px for the starting layout
    assert rf.halo_half_px == 32
    assert rf.halo_full_px % 16 == 0
    assert_within_limit(rf)


def test_halo_covers_worst_one_sided_reach_and_spec_formula():
    for cfg in CONFIGS + [JddConfig()]:
        rf = analyse(JddNet(cfg))
        assert 2 * rf.reach_half_px <= rf.halo_full_px
        assert rf.halo_full_px >= rf.spec_halo_full_px


def test_limit_assertion_fires():
    big = JddNet(JddConfig(channels=(4, 6, 8), convs_per_level=4, up_kernel=3, head_kernel=3))
    with pytest.raises(AssertionError):
        assert_within_limit(analyse(big))
    assert analyse(big).rf_full_px > RF_LIMIT_FULL_PX


def test_unknown_operation_is_rejected():
    class Bad(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Conv2d(7, 3, 3, padding=1)

        def forward(self, x):
            return torch.nn.functional.avg_pool2d(self.c(x), 2)

    with pytest.raises(NotImplementedError):
        footprint_1d(Bad(), 10)

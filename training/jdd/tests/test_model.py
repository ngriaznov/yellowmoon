import numpy as np
import pytest
import torch
import torch.nn as nn

from training.jdd.data import mosaic_rggb, pack_planes
from training.jdd.evaluate import baseline_bilinear
from training.jdd.export import make_random_model
from training.jdd.model import (
    JddConfig,
    JddNet,
    bilinear_weights,
    check_input_shape,
    count_flops,
    count_parameters,
)
from training.jdd.receptive_field import analyse


def test_shapes_and_budget():
    net = JddNet().eval()
    y = net(torch.rand(2, 7, 32, 48))
    assert y.shape == (2, 3, 64, 96)
    assert 0.2e6 <= count_parameters(net) <= 1.0e6  # docs/07: 0.2-1 M parameters
    assert count_flops(net, 64, 64)["per_output_pixel"] <= 250e3  # docs/07 budget


def test_no_forbidden_layers():
    for net in (JddNet(), JddNet(JddConfig(up_kernel=3, head_kernel=3))):
        for m in net.modules():
            assert not isinstance(m, (nn.ConvTranspose2d, nn.Upsample, nn.UpsamplingBilinear2d, nn.UpsamplingNearest2d))


def test_input_shape_check():
    check_input_shape(64, 32)
    with pytest.raises(ValueError):
        check_input_shape(30, 32)


def test_bilinear_kernel_matches_direct_bilinear():
    rng = np.random.default_rng(0)
    rgb = rng.random((40, 48, 3))
    mosaic = mosaic_rggb(rgb)
    planes = torch.from_numpy(pack_planes(mosaic))[None].float()
    conv = nn.Conv2d(4, 12, 3, 1, 1, bias=False)
    conv.weight.data.copy_(bilinear_weights())
    out = nn.functional.pixel_shuffle(conv(planes), 2)[0].permute(1, 2, 0).detach().numpy()
    ref = baseline_bilinear(mosaic)  # independent masked-convolution bilinear
    # Interior only: the two differ in how they pad the border.
    assert np.abs(out[4:-4, 4:-4] - ref[4:-4, 4:-4]).max() < 1e-5


def test_head_starts_near_zero_so_net_starts_as_bilinear():
    net = JddNet().eval()
    x = torch.rand(1, 7, 32, 32) * 0.3
    out = net(x)
    base = nn.functional.pixel_shuffle(net.bilinear(x[:, :4] * 0.25), 2) * 4.0
    assert (out - base).abs().max() < 0.05 * float(base.abs().max()) + 1e-3


@pytest.mark.parametrize("cfg", [JddConfig(channels=(8, 12, 16)), JddConfig(channels=(6, 8, 10), up_kernel=3, head_kernel=3)])
def test_halo_makes_tiling_seamless(cfg):
    """Crop-not-blend tiling with the computed halo reproduces the whole-image result (docs/05)."""
    net = make_random_model(5, cfg).eval()
    rf = analyse(net)
    halo = rf.halo_half_px
    core = 16
    nty, ntx = 3, 3
    h, w = core * nty, core * ntx
    x = torch.rand(1, 7, h, w) * 0.5
    xp = nn.functional.pad(x, (halo, halo, halo, halo), mode="replicate")
    with torch.no_grad():
        whole = net(xp)[..., 2 * halo : 2 * (halo + h), 2 * halo : 2 * (halo + w)]
        tiled = torch.zeros_like(whole)
        for ty in range(nty):
            for tx in range(ntx):
                y0, x0 = ty * core, tx * core
                tile = xp[..., y0 : y0 + core + 2 * halo, x0 : x0 + core + 2 * halo]
                o = net(tile)[..., 2 * halo : 2 * (halo + core), 2 * halo : 2 * (halo + core)]
                tiled[..., 2 * y0 : 2 * (y0 + core), 2 * x0 : 2 * (x0 + core)] = o
    assert (whole - tiled).abs().max() < 1e-4


def test_too_small_halo_breaks_seams():
    net = make_random_model(5, JddConfig(channels=(8, 12, 16))).eval()
    halo = 8  # well below the reach of 26 half-res px
    core, n = 16, 3
    x = torch.rand(1, 7, core * n, core * n) * 0.5
    xp = nn.functional.pad(x, (halo, halo, halo, halo), mode="replicate")
    with torch.no_grad():
        whole = net(xp)[..., 2 * halo : -2 * halo, 2 * halo : -2 * halo]
        tile = xp[..., core : core + core + 2 * halo, core : core + core + 2 * halo]
        o = net(tile)[..., 2 * halo : 2 * (halo + core), 2 * halo : 2 * (halo + core)]
    assert (whole[..., 2 * core : 4 * core, 2 * core : 4 * core] - o).abs().max() > 1e-4

"""The synthesis: kernels, determinism, and that the labels are the blur that was applied."""

import numpy as np
import pytest
import torch

from training.bokeh import kernels as K
from training.bokeh import synth as S
from training.bokeh.encode import R_FLOOR_PX

from .conftest import fake_tiles

DISC_ONLY = (1.0, 0, 0, 0, 0, 0, 0, 0)


@pytest.mark.parametrize("radius", [3.0, 9.0, 18.0])
def test_every_kernel_kind_has_the_wanted_radius_sums_to_one_and_is_centred(radius):
    g = torch.Generator().manual_seed(0)
    n = 240
    p = K.sample_params(n, g, "cpu")
    k = K.make_kernels(torch.full((n,), radius), p, radius_cells=28)
    cx, cy, r = K.moments(k)
    assert torch.allclose(k.sum((1, 2)), torch.ones(n), atol=1e-5)
    assert float((r - radius).abs().max()) < 0.03 * radius
    assert float(cx.abs().max()) < 0.1 and float(cy.abs().max()) < 0.1
    assert float(k.min()) >= 0.0
    assert set(p["kind"].tolist()) == set(range(len(K.KINDS)))


def test_a_uniform_disc_has_the_area_of_its_radius_and_a_tiny_radius_is_a_delta():
    p = K.sample_params(2, torch.Generator().manual_seed(1), "cpu", DISC_ONLY)
    k = K.make_kernels(torch.tensor([10.0, 0.3]), {**p, "ellip": torch.zeros(2)}, radius_cells=16)
    area = float((k[0] > 0.5 * k[0].max()).sum())
    assert abs(area - np.pi * 100) / (np.pi * 100) < 0.05
    assert float(k[1, 16, 16]) == 1.0 and float(k[1].sum()) == 1.0


def test_the_fft_convolution_of_a_point_is_the_kernel():
    p = K.sample_params(1, torch.Generator().manual_seed(2), "cpu")
    k = K.make_kernels(torch.tensor([6.0]), p, radius_cells=12)
    img = torch.zeros(1, 1, 80, 80)
    img[0, 0, 40, 40] = 1.0
    out = K.conv_fft(img, k.unsqueeze(1))
    assert torch.allclose(out[0, 0, 28:53, 28:53], k[0], atol=1e-5)


def test_the_same_seed_gives_the_same_batch_and_another_seed_does_not():
    cfg = S.SynthConfig(sample=64, kernel_r=14, r_max_px=10.0)
    tb = fake_tiles(4, 64, seed=3)
    a = S.synthesize(tb, cfg, torch.Generator().manual_seed(5))
    b = S.synthesize(tb, cfg, torch.Generator().manual_seed(5))
    c = S.synthesize(tb, cfg, torch.Generator().manual_seed(6))
    assert torch.equal(a.lin, b.lin) and torch.equal(a.r, b.r) and torch.equal(a.w, b.w)
    assert not torch.equal(a.lin, c.lin)


def test_batches_are_sane_over_many_seeds():
    cfg = S.SynthConfig(sample=64, kernel_r=14, r_max_px=10.0)
    tb = fake_tiles(6, 80, seed=4)  # tiles larger than the sample: a random crop
    zero, big, n = 0, 0, 0
    for seed in range(8):
        sy = S.synthesize(tb, cfg, torch.Generator().manual_seed(seed))
        assert sy.lin.shape == (6, 3, 64, 64) and sy.r.shape == (6, 1, 64, 64)
        assert torch.isfinite(sy.lin).all() and float(sy.lin.min()) >= 0 and float(sy.lin.max()) <= 2.3
        assert float(sy.r.min()) >= 0 and float(sy.r.max()) <= cfg.r_max_px + 1e-3
        assert float(sy.w.min()) >= 0.25 - 1e-6 and float(sy.w.max()) <= 1.0 + 1e-6
        assert torch.allclose(sy.logr, torch.log(sy.r.clamp(min=R_FLOOR_PX)))
        zero += int((sy.r < R_FLOOR_PX).sum())
        big += int((sy.r > 4).sum())
        n += sy.r.numel()
    assert 0.15 < zero / n < 0.85 and big / n > 0.03  # both in-focus and clearly blurred pixels occur


@pytest.fixture()
def no_ellipticity(monkeypatch):
    orig = S.sample_params

    def patched(n, gen, dev, weights=None):
        p = orig(n, gen, dev) if weights is None else orig(n, gen, dev, weights)
        return {**p, "ellip": torch.zeros(n)}

    monkeypatch.setattr(S, "sample_params", patched)


def _edge_radius(profile: np.ndarray) -> float:
    """The disc radius of the blur that turned a step into `profile`: 2 * the standard deviation of its derivative."""
    lsf = np.diff(profile)
    lsf = lsf / lsf.sum()
    x = np.arange(lsf.size) + 0.5
    mean = (lsf * x).sum()
    return 2 * float(np.sqrt((lsf * (x - mean) ** 2).sum()))


@pytest.mark.parametrize("radius", [4.0, 8.0])
def test_the_label_is_the_blur_that_was_applied_to_a_step_edge(no_ellipticity, radius):
    m, c = 16, 128
    img = torch.full((1, 3, c, c), 0.2)
    img[..., c // 2:] = 1.0
    out, label, weight = S.render_layers(
        img, torch.ones(1, 1, c, c), torch.tensor([[radius]]), torch.full((1, c, c), radius), [], torch.zeros(1, 0), [],
        torch.zeros(1, 0), torch.Generator().manual_seed(0), m, kind_weights=K.KIND_WEIGHTS)
    row = out[0, 1, c // 2, c // 2 - 3 * m // 2:c // 2 + 3 * m // 2 + 1].numpy()
    # a step edge (0.2 to 1.0) blurred by the kernel the pipeline drew: its width in the label's own definition
    assert abs(_edge_radius((row - 0.2) / 0.8) - radius) < 0.12 * radius  # kernels differ in shape: within 12 %
    assert torch.allclose(label, torch.tensor(radius)) and float(weight.min()) > 0.99


def test_a_disc_kernel_blurs_a_step_to_exactly_the_labelled_radius(no_ellipticity):
    radius, m, c = 8.0, 16, 128
    img = torch.full((1, 3, c, c), 0.2)
    img[..., c // 2:] = 1.0
    out, label, _ = S.render_layers(
        img, torch.ones(1, 1, c, c), torch.tensor([[radius]]), torch.full((1, c, c), radius), [], torch.zeros(1, 0), [],
        torch.zeros(1, 0), torch.Generator().manual_seed(0), m, kind_weights=DISC_ONLY)
    row = out[0, 1, c // 2, c // 2 - 24:c // 2 + 25].numpy()
    assert abs(_edge_radius((row - 0.2) / 0.8) - radius) < 0.03 * radius
    assert abs(float(label[0, 0, 5, 5]) - radius) < 1e-6


def test_a_sharp_subject_in_front_of_a_blurred_background(no_ellipticity):
    m, c = 16, 128
    bg = torch.zeros(1, 3, c, c)
    bg[..., ::8, :] = 1.0  # horizontal lines
    sub_a = torch.zeros(1, 1, c, c)
    sub_a[..., 48:80, 48:80] = 1.0
    sub_c = sub_a * torch.tensor([0.9, 0.3, 0.1]).view(1, 3, 1, 1)
    out, label, weight = S.render_layers(
        bg, torch.ones(1, 1, c, c), torch.tensor([[10.0]]), torch.full((1, c, c), 10.0), [(sub_c, sub_a)], torch.zeros(1, 1), [],
        torch.zeros(1, 0), torch.Generator().manual_seed(1), m, kind_weights=DISC_ONLY)
    # the subject: its own colour exactly, not touched by the background's blur; label 0
    assert torch.allclose(out[0, :, 52:76, 52:76], sub_c[0, :, 52:76, 52:76], atol=1e-4)
    assert float(label[0, 0, 60, 60]) == 0.0 and float(weight[0, 0, 60, 60]) > 0.99
    # the background: the lines are smeared by a disc of radius 10 (contrast far below the original's), label 10
    assert float(label[0, 0, 10, 10]) == 10.0
    assert float(out[0, 1, 8:40, 10].max() - out[0, 1, 8:40, 10].min()) < 0.35
    # the ambiguous band at the subject's blurred-alpha edge is lighter in the loss than the surfaces
    assert float(weight.min()) >= 0.25 - 1e-6


def test_an_out_of_focus_light_is_a_disc_of_the_labelled_radius(no_ellipticity):
    radius, m, c = 9.0, 20, 128
    bg = torch.full((1, 3, c, c), 0.01)
    lamp = torch.zeros(1, 3, c, c)
    lamp[0, :, 64, 64] = 500.0
    out, label, _ = S.render_layers(
        bg, torch.ones(1, 1, c, c), torch.tensor([[0.0]]), torch.zeros(1, c, c), [], torch.zeros(1, 0), [lamp],
        torch.tensor([[radius]]), torch.Generator().manual_seed(2), m, kind_weights=DISC_ONLY)
    ball = out[0, 1] - 0.01
    assert abs(float(ball.sum()) - 500.0) < 0.5  # the energy is kept
    ys, xs = np.mgrid[0:c, 0:c]
    b = ball.numpy()
    cy, cx = (b * ys).sum() / b.sum(), (b * xs).sum() / b.sum()
    r_meas = np.sqrt(2 * (b * ((ys - cy) ** 2 + (xs - cx) ** 2)).sum() / b.sum())
    assert abs(r_meas - radius) < 0.03 * radius
    assert float(label[0, 0, 64, 64]) == radius  # the ball's pixels are labelled with its radius
    assert float(label[0, 0, 5, 5]) == 0.0  # the dark ground away from it is in focus


def test_a_blurred_foreground_layer_lets_the_sharp_background_show_through(no_ellipticity):
    m, c = 16, 128
    bg = torch.rand(1, 3, c, c, generator=torch.Generator().manual_seed(3))
    fg_a = torch.zeros(1, 1, c, c)
    fg_a[..., 40:90, 40:90] = 1.0
    fg_c = fg_a.expand(1, 3, c, c) * 0.9
    out, label, _ = S.render_layers(
        bg, torch.ones(1, 1, c, c), torch.tensor([[0.0]]), torch.zeros(1, c, c), [(fg_c, fg_a)], torch.tensor([[-7.0]]), [],
        torch.zeros(1, 0), torch.Generator().manual_seed(4), m, kind_weights=DISC_ONLY)
    assert torch.allclose(out[0, :, 10:20, 10:20], bg[0, :, 10:20, 10:20], atol=1e-4)  # untouched far from the layer
    assert float(label[0, 0, 65, 65]) == 7.0 and float(label[0, 0, 10, 10]) == 0.0
    # on the nominal edge of a straight, blurred layer its alpha is one half: half its colour plus half the background
    expected = 0.5 * 0.9 + 0.5 * float(bg[0, 0, 65, 40])
    assert abs(float(out[0, 0, 65, 40]) - expected) < 0.02


def test_ramp_bins_label_the_field_and_blur_each_pixel_with_its_own_radius(no_ellipticity):
    m, c = 16, 128
    bg = torch.zeros(1, 3, c, c)
    bg[..., ::6, :] = 1.0
    ramp = torch.linspace(0, 1, c).view(1, 1, c).expand(1, c, c)  # radius 1 .. 9 along x
    r_lo, r_hi, nb = 1.0, 9.0, 4
    field = r_lo + (r_hi - r_lo) * ramp
    u = ((field - r_lo) / (r_hi - r_lo)).clamp(0, 1) * (nb - 1)
    w = torch.stack([(1 - (u - j).abs()).clamp(0, 1) for j in range(nb)], 1)
    assert torch.allclose(w.sum(1), torch.ones(1, c, c), atol=1e-6)  # a partition of unity
    bg_r = torch.tensor([[r_lo + (r_hi - r_lo) * j / (nb - 1) for j in range(nb)]])
    out, label, _ = S.render_layers(bg, w, bg_r, field, [], torch.zeros(1, 0), [], torch.zeros(1, 0),
                                    torch.Generator().manual_seed(5), m, kind_weights=DISC_ONLY)
    assert torch.allclose(label[0, 0], field[0], atol=1e-5)
    contrast = lambda x0: float(out[0, 1, 30:60, x0].max() - out[0, 1, 30:60, x0].min())  # noqa: E731
    assert contrast(110) < contrast(30)  # more blur on the right, less contrast


# --- the NaN of the first Modal run: a kernel with a negative lobe -----------------------------------------------------


def _only_dark(monkeypatch, dark: float = 0.75):
    """Every kernel the synthesis draws is a dark-rim kernel with the strongest rim."""
    orig = S.sample_params

    def patched(n, gen, dev, *weights):
        p = orig(n, gen, dev, *weights)
        return {**p, "kind": torch.full((n,), K.KINDS.index("dark")), "dark": torch.full((n,), dark)}

    monkeypatch.setattr(S, "sample_params", patched)


def test_the_dark_rim_kernel_that_made_the_first_run_nan_is_finite_and_not_negative():
    # The parameters of the kernel that turned a whole sample of the first run into NaN: a dark rim of 0.745 on a
    # radius of 0.697 px. Its antialiased rim cells were negative, so its second moment was negative and the square
    # root in `moments` NaN; the fixed-point iteration of `make_kernels` then spread the NaN over the kernel.
    p = K.sample_params(1, torch.Generator().manual_seed(0), "cpu", (0, 0, 0, 1.0, 0, 0, 0, 0))
    p.update(dark=torch.tensor([0.7451964]), ellip=torch.tensor([0.0675503]))
    k = K.make_kernels(torch.tensor([0.6974374]), p, radius_cells=12)
    assert torch.isfinite(k).all() and float(k.min()) >= 0.0
    assert float(k.sum()) == pytest.approx(1.0, abs=1e-5)
    assert abs(float(K.moments(k)[2]) - 0.6974374) < 0.02


def test_no_kernel_is_negative_or_not_finite_at_any_radius_for_the_extreme_parameters():
    radii = torch.tensor([0.0, 0.3, 0.5, 0.51, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0, 2.8, 4.0, 9.0, 20.0])
    for dark in (0.3, 0.694, 0.75):
        for ellip in (0.0, 0.08):
            n = radii.numel() * len(K.KINDS)
            radius = radii.repeat(len(K.KINDS))
            p = K.sample_params(n, torch.Generator().manual_seed(1), "cpu")
            p.update(kind=torch.arange(len(K.KINDS)).repeat_interleave(radii.numel()), dark=torch.full((n,), dark),
                     ellip=torch.full((n,), ellip), bright=torch.full((n,), 1.2), ring_amp=torch.full((n,), 0.3),
                     rim=torch.full((n,), 0.05), round=torch.full((n,), 0.5))
            k = K.make_kernels(radius, p, radius_cells=28)
            assert torch.isfinite(k).all(), (dark, ellip)
            assert float(k.min()) >= 0.0, (dark, ellip)
            assert torch.allclose(k.sum((1, 2)), torch.ones(n), atol=1e-4), (dark, ellip)


def test_the_label_radius_of_a_kernel_with_a_negative_lobe_is_not_nan():
    k = torch.zeros(1, 9, 9)
    k[0, 4, 4] = 1.2
    k[0, 4, 3] = k[0, 4, 5] = k[0, 3, 4] = k[0, 5, 4] = -0.05
    assert torch.isfinite(K.moments(k)[2]).all()


def test_a_background_with_sub_pixel_bins_of_dark_kernels_renders_without_a_nan(monkeypatch):
    _only_dark(monkeypatch)
    m, c, nb = 12, 64, 4
    bg = torch.rand(2, 3, c, c, generator=torch.Generator().manual_seed(3))
    field = torch.linspace(0, 2.1, c).view(1, 1, c).expand(2, c, c)  # a ramp from sharp to 2 px: bins at 0, 0.7, 1.4, 2.1
    u = (field / 2.1).clamp(0, 1) * (nb - 1)
    w = torch.stack([(1 - (u - j).abs()).clamp(0, 1) for j in range(nb)], 1)
    bg_r = torch.tensor([[0.0, 0.7, 1.4, 2.1]]).expand(2, nb)
    for seed in range(4):
        out, label, weight = S.render_layers(bg, w, bg_r, field, [], torch.zeros(2, 0), [], torch.zeros(2, 0),
                                             torch.Generator().manual_seed(seed), m)
        assert torch.isfinite(out).all() and torch.isfinite(label).all() and torch.isfinite(weight).all()


def test_whole_batches_of_dark_kernels_are_finite_over_many_seeds(monkeypatch):
    _only_dark(monkeypatch)
    cfg = S.SynthConfig(sample=64, kernel_r=14, r_max_px=10.0, p_ramp_batch=0.5)
    tb = fake_tiles(6, 64, seed=8)
    for seed in range(10):
        sy = S.synthesize(tb, cfg, torch.Generator().manual_seed(seed))
        for name, t in (("lin", sy.lin), ("r", sy.r), ("w", sy.w), ("logr", sy.logr)):
            assert torch.isfinite(t).all(), (seed, name)


def test_colorize_treats_a_nan_as_zero_instead_of_indexing_with_it():
    v = torch.tensor([[[[float("nan"), 0.0], [0.05, float("inf")]]]])
    col = S.colorize(v, 0.05)
    assert col.shape == (1, 3, 2, 2) and torch.isfinite(col).all()
    assert torch.equal(col[..., 0, 0], col[..., 0, 1])  # NaN = 0


def test_a_radius_range_that_would_make_the_log_uniform_draw_nan_is_refused():
    for lo, hi in ((0.0, 10.0), (-1.0, 10.0), (3.0, 2.0)):
        with pytest.raises(ValueError, match="r_min_px"):
            S.SynthConfig(sample=64, kernel_r=14, r_min_px=lo, r_max_px=hi)
    S.SynthConfig(sample=64, kernel_r=14, r_min_px=1.0, r_max_px=1.0)  # a single radius is fine

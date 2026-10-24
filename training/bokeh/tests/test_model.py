"""BokehNet: shapes, size, the operations it uses, and the export (folding, bytes, fixture)."""

import hashlib
import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open

from training.bokeh import encode, export
from training.bokeh.model import BokehNet, macs_per_pixel, parameters


@pytest.fixture(scope="module")
def net():
    return BokehNet().eval()


def test_output_is_two_channels_at_half_resolution_for_any_size_divisible_by_32(net):
    for h, w in ((32, 32), (64, 96), (160, 224)):
        with torch.no_grad():
            out = net(torch.rand(1, 3, h, w))
        assert out.shape == (1, 2, h // 2, w // 2)
    with pytest.raises(ValueError, match="divisible by 32"):
        net(torch.rand(1, 3, 50, 64))


def test_size_and_cost_are_in_the_planned_range(net):
    n = parameters(net)
    assert 1_400_000 < n < 3_000_000
    macs = macs_per_pixel(net, (128, 128))
    assert macs <= 3500  # about 2.4 GMAC for a 1024 x 683 plane


def test_the_start_is_about_one_and_a_half_pixels_with_a_wide_scale(net):
    with torch.no_grad():
        out = net(torch.rand(1, 3, 64, 64) * 0.5)
    assert abs(float(out[0, 0].mean()) - np.log(1.5)) < 0.1 and abs(float(out[0, 1].mean()) - np.log(0.7)) < 0.1


def test_only_operations_the_rust_runtime_implements(net):
    allowed = (torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU6, torch.nn.Sequential, torch.nn.ModuleList)
    for m in net.modules():
        if len(list(m.children())) == 0:
            assert isinstance(m, allowed), type(m)
        if isinstance(m, torch.nn.Conv2d):
            assert m.kernel_size in ((1, 1), (3, 3)) and m.stride in ((1, 1), (2, 2))
            assert m.padding == (m.kernel_size[0] // 2,) * 2 and m.dilation == (1, 1)
            assert m.groups == 1 or m.groups == m.in_channels == m.out_channels or m.kernel_size == (3, 3)
            assert m.bias is None or m is net.head


def test_the_bilinear_step_is_the_half_pixel_rule_with_edge_clamping():
    x = torch.rand(1, 1, 3, 4)
    got = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)[0, 0]
    ref = np.zeros((6, 8))
    xs = x[0, 0].numpy()
    for i in range(6):
        for j in range(8):
            fy, fx = np.clip((i + 0.5) / 2 - 0.5, 0, 2), np.clip((j + 0.5) / 2 - 0.5, 0, 3)
            y0, x0 = int(np.floor(fy)), int(np.floor(fx))
            y1, x1 = min(y0 + 1, 2), min(x0 + 1, 3)
            wy, wx = fy - y0, fx - x0
            ref[i, j] = (xs[y0, x0] * (1 - wx) + xs[y0, x1] * wx) * (1 - wy) + (xs[y1, x0] * (1 - wx) + xs[y1, x1] * wx) * wy
    assert np.abs(got.numpy() - ref).max() < 1e-6


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    d = tmp_path_factory.mktemp("export")
    model = BokehNet()
    export.randomize(model, 7)
    report = export.write(model, d / "a.safetensors", d / "a.json", "test")
    export.write(model, d / "b.safetensors", d / "b.json", "test")
    return d, model, report


def test_the_same_weights_give_the_same_bytes_and_fixture(exported):
    d, _, report = exported
    assert (d / "a.safetensors").read_bytes() == (d / "b.safetensors").read_bytes()
    assert (d / "a.json").read_text() == (d / "b.json").read_text()
    assert report["sha256"] == hashlib.sha256((d / "a.safetensors").read_bytes()).hexdigest()
    assert json.loads((d / "a.json").read_text())["weights_sha256"] == report["sha256"]


def test_the_folded_network_matches_the_trained_one(exported):
    _, _, report = exported
    assert report["folded_max_diff"] < 2e-4  # write() refuses beyond this
    assert 1_400_000 < report["parameters"] < 3_000_000


def test_the_file_reads_back_with_its_architecture_in_the_header(exported):
    d, model, _ = exported
    with safe_open(d / "a.safetensors", "pt") as f:
        meta = f.metadata()
        names = set(f.keys())
        assert meta["format"] == export.FORMAT and meta["version"] == export.VERSION
        for key in ("arch", "upsample", "padding", "mid", "k", "stops", "input_mean", "input_std", "luma", "median", "output", "b", "conf",
                    "pad_multiple", "frame_long_px"):
            assert key in meta
        arch = json.loads(meta["arch"])
        assert len(arch["blocks"]) == sum(s[2] for s in model.arch.stages)
        assert all(f.get_tensor(n).dtype == torch.float32 for n in names)
        assert "head.weight" in names and "blocks.0.depthwise.bias" in names and "decoder.3.pw.weight" in names
        assert not any("running" in n or "bn" in n for n in names)  # batch norms are folded away
    assert float(meta["mid"]) == encode.MID and float(meta["conf_beta0"]) == encode.CONF_BETA0


def test_the_fixture_has_the_input_definitions_and_the_outputs(exported):
    d, _, _ = exported
    doc = json.loads((d / "a.json").read_text())
    kinds = [c["kind"] for c in doc["cases"]]
    assert kinds.count("encoded") == len(export.SIZES) and kinds.count("linear") == 1
    for c in doc["cases"]:
        assert len(c["output"]) == (c["height"] // 2) * (c["width"] // 2) * 2
    lin = [c for c in doc["cases"] if c["kind"] == "linear"][0]
    # the probes are the float64 encoding of the integer pattern: the Rust test reproduces them
    e = encode.encode_np(export.pattern_linear(lin["height"], lin["width"]))
    for p in lin["encoded_probes"]:
        assert abs(e[p["c"], p["y"], p["x"]] - p["e"]) < 1e-8
    assert float(export.pattern_e(32, 32, 0)[0, 0, 0, 1]) == pytest.approx(((1 * 2 * 5 + 0 + 0 + 0) % 256) / 255.0)


def test_random_weights_are_seeded_and_not_constant():
    a, b, c = BokehNet(), BokehNet(), BokehNet()
    export.randomize(a, 1)
    export.randomize(b, 1)
    export.randomize(c, 2)
    x = export.pattern_e(64, 64, 0)
    with torch.no_grad():
        ya, yb, yc = a.eval()(x), b.eval()(x), c.eval()(x)
    assert torch.equal(ya, yb) and not torch.equal(ya, yc)
    assert float(ya[0, 0].std()) > 1e-3

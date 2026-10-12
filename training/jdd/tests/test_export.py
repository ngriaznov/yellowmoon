import json

import numpy as np
import onnx
import pytest
import torch
from safetensors.torch import load_file

from training.jdd.export import export_onnx, make_random_model
from training.jdd.model import JddConfig, JddNet
from training.jdd.weights_io import load_weights, read_metadata, save_weights

SMALL = JddConfig(channels=(8, 16, 24))


def test_safetensors_roundtrip_and_header(tmp_path):
    m = make_random_model(1, SMALL)
    sha = save_weights(m, tmp_path / "w.safetensors", model_version="t1")
    assert len(sha) == 64
    meta = read_metadata(tmp_path / "w.safetensors")
    assert meta["format"] == "ym-jdd" and meta["model_version"] == "t1" and meta["in_channels"] == "7"
    assert JddConfig.from_json(meta["config"]) == SMALL
    assert int(meta["rf_half_px"]) == 47 and int(meta["halo_half_px"]) == 32
    m2 = load_weights(tmp_path / "w.safetensors")
    x = torch.rand(1, 7, 32, 32)
    assert torch.equal(m(x), m2(x))
    tensors = load_file(str(tmp_path / "w.safetensors"))
    assert "enc0.conv1.weight" in tensors and "bilinear.weight" in tensors and "head.bias" in tensors
    assert all(t.dtype == torch.float32 for t in tensors.values())
    # same weights, same bytes (deterministic file)
    sha2 = save_weights(m2, tmp_path / "w2.safetensors", model_version="t1")
    assert sha == sha2


@pytest.mark.parametrize("decompose", [False, True])
def test_onnx_export_static_shape_ops_and_parity(tmp_path, decompose):
    m = make_random_model(2, SMALL)
    rep = export_onnx(m, tmp_path, core_half_px=16, decompose_pixel_shuffle=decompose)
    size = 16 + 2 * 32
    assert rep["input_shape"] == [1, 7, size, size] and rep["output_shape"] == [1, 3, 2 * size, 2 * size]
    assert rep["opset"] == 17 and rep["max_abs_diff_vs_torch"] < 2e-3
    ops = set(rep["ops"])
    assert not ({"ConvTranspose", "Resize", "Upsample"} & ops)
    assert {"Conv", "Mul", "Sigmoid", "Concat", "Add"} <= ops
    assert ("Transpose" in ops) == decompose and ("DepthToSpace" in ops) != decompose
    proto = onnx.load(str(tmp_path / "jdd.onnx"))
    dims = [d.dim_value for d in proto.graph.input[0].type.tensor_type.shape.dim]
    assert dims == [1, 7, size, size]  # static
    assert any(o.version == 17 for o in proto.opset_import)
    assert json.loads((tmp_path / "export.json").read_text())["core_half_px"] == 16


def test_export_rejects_halo_below_receptive_field(tmp_path):
    with pytest.raises(ValueError):
        export_onnx(make_random_model(3, SMALL), tmp_path, core_half_px=16, halo_half_px=8)


def test_random_models_are_seeded():
    a, b = make_random_model(7, SMALL), make_random_model(7, SMALL)
    c = make_random_model(8, SMALL)
    x = torch.rand(1, 7, 16, 16)
    assert torch.equal(a(x), b(x)) and not torch.equal(a(x), c(x))
    # informative outputs: random weights must move the result away from plain bilinear
    base = torch.nn.functional.pixel_shuffle(a.bilinear(x[:, :4] * 0.25), 2) * 4
    assert (a(x) - base).abs().mean() > 1e-3
    assert isinstance(a, JddNet) and np.isfinite(a(x).detach().numpy()).all()

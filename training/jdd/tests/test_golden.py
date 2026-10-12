from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from training.jdd import golden_tiles
from training.jdd.weights_io import load_weights

FIXTURES = Path(__file__).resolve().parents[3] / "crates" / "ym-ml" / "tests" / "data"


def test_fixtures_are_small_and_complete():
    names = {
        "golden_small.safetensors", "golden_k3.safetensors", "golden_tiles.safetensors",
        "golden_small.onnx", "golden_k3.onnx", "golden.json",
    }
    if not FIXTURES.exists():
        pytest.skip("fixtures not generated yet")
    assert names <= {p.name for p in FIXTURES.iterdir()}
    assert sum(p.stat().st_size for p in FIXTURES.iterdir() if p.is_file()) < 2_000_000


def test_committed_tiles_match_a_fresh_pytorch_run():
    if not FIXTURES.exists():
        pytest.skip("fixtures not generated yet")
    tiles = load_file(str(FIXTURES / "golden_tiles.safetensors"))
    counts = {}
    for key, wfile in (("small", "golden_small.safetensors"), ("k3", "golden_k3.safetensors")):
        model = load_weights(FIXTURES / wfile)
        i = 0
        while f"{key}.in.{i}" in tiles:
            with torch.no_grad():
                y = model(tiles[f"{key}.in.{i}"][None])[0]
            assert (y - tiles[f"{key}.out.{i}"]).abs().max() < 1e-5, (key, i)
            i += 1
        counts[key] = i
    # docs/07: "20 fixed inputs"
    assert counts == {"small": len(golden_tiles.TILES["small"]), "k3": len(golden_tiles.TILES["k3"])}
    assert sum(counts.values()) == 20 == golden_tiles.N_GOLDEN_TILES
    # the tiles are real generator output: shapes vary and some of them contain clipped photosites
    shapes = {tuple(tiles[f"small.in.{i}"].shape[1:]) for i in range(counts["small"])}
    assert len(shapes) >= 5
    clip = [float(tiles[f"{k}.in.{i}"][6].mean()) for k in counts for i in range(counts[k])]
    assert max(clip) > 0.0


def test_onnxruntime_agrees_with_the_golden_outputs_on_all_tiles():
    import onnxruntime as ort

    if not FIXTURES.exists():
        pytest.skip("fixtures not generated yet")
    tiles = load_file(str(FIXTURES / "golden_tiles.safetensors"))
    n = 0
    for key in ("small", "k3"):
        sess = ort.InferenceSession(str(FIXTURES / f"golden_{key}.onnx"), providers=["CPUExecutionProvider"])
        i = 0
        while f"{key}.in.{i}" in tiles:
            got = sess.run(None, {"x": tiles[f"{key}.in.{i}"].numpy()[None]})[0][0]
            assert np.abs(got - tiles[f"{key}.out.{i}"].numpy()).max() <= 2e-3, (key, i)
            i += 1
            n += 1
    assert n == 20


def test_fixtures_are_regenerated_identically():
    if not FIXTURES.exists():
        pytest.skip("fixtures not generated yet")
    assert golden_tiles.check(FIXTURES) == 0

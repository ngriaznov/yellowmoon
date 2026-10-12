import numpy as np
import torch

from training.jdd import evaluate as E
from training.jdd import model_card, sample_sheet
from training.jdd.export import make_random_model
from training.jdd.model import JddConfig
from training.jdd.weights_io import save_weights, write_json

SMALL = JddConfig(channels=(8, 16, 24))


def test_bilinear_baseline_and_hook(tmp_path):
    rng = np.random.default_rng(0)
    rgb = rng.random((32, 40, 3))
    x, y, mosaic = E.clean_inputs(rgb)
    assert x.shape == (7, 16, 20) and y.shape == (3, 32, 40)
    assert (x[4] == 0).all() and (x[5] == 0).all() and (x[6] == 0).all()  # sigma, qstep (zeros: lossless), clip
    out = E.baseline_bilinear(mosaic)
    assert out.shape == (32, 40, 3)
    # the hook resolves package.module:function
    hook = E.get_baseline("training.jdd.evaluate:baseline_bilinear")
    assert np.allclose(hook(mosaic), out)
    assert E.get_baseline("bilinear")(mosaic).shape == (32, 40, 3)


def test_evaluate_reports_gain_and_gate(tmp_path):
    model = make_random_model(1, SMALL)
    pairs = list(E.synthetic_pairs(3, 5, 16))
    res = E.evaluate_pairs(model, pairs, None, torch.device("cpu"))
    assert res["reference"] == "bilinear_linear" and "gain_linear_db" in res and isinstance(res["gate_2db_linear"], bool)
    # a folder of clean images with an external baseline
    folder = tmp_path / "set"
    folder.mkdir()
    for i in range(2):
        np.save(folder / f"im{i}.npy", np.random.default_rng(i).random((48, 48, 3)))
    res2 = E.evaluate_pairs(model, E.folder_pairs(folder, False, None, 0), E.baseline_bilinear, torch.device("cpu"))
    assert res2["reference"] == "baseline_linear" and len(res2["images"]) == 2
    # an untrained residual net starting at bilinear must be close to the external bilinear baseline
    assert abs(res2["bilinear_linear"]["mean"] - res2["baseline_linear"]["mean"]) < 1.5


def test_noisy_sets_hand_the_balanced_mosaic_to_the_external_baseline(tmp_path):
    """Generated sets used to skip the external baseline (zero mosaic) and silently fall back to bilinear."""
    model = make_random_model(1, SMALL)
    calls = []

    def spy(mosaic, pattern="RGGB"):
        calls.append(mosaic.shape)
        return E.baseline_bilinear(mosaic)

    import sys
    import types

    mod = types.ModuleType("ym_spy_baseline")
    mod.spy = spy
    sys.modules["ym_spy_baseline"] = mod
    try:
        base = E.get_baseline("ym_spy_baseline:spy")
        pairs = list(E.synthetic_pairs(2, 7, 16))
        # the mosaic is the packed input planes laid out again: full resolution, same values
        name, x, y, mosaic = pairs[0]
        assert mosaic.shape == (32, 32) and np.allclose(mosaic[0::2, 0::2], x[0]) and np.allclose(mosaic[1::2, 1::2], x[3])
        res = E.evaluate_pairs(model, pairs, base, torch.device("cpu"))
    finally:
        del sys.modules["ym_spy_baseline"]
    assert calls == [(32, 32), (32, 32)]
    assert res["reference"] == "baseline_linear" and "baseline_linear" in res


def test_evaluate_cli_writes_report(tmp_path):
    save_weights(make_random_model(1, SMALL), tmp_path / "w.safetensors")
    rc = E.main(["--weights", str(tmp_path / "w.safetensors"), "--n", "2", "--patch-half", "16", "--out", str(tmp_path / "r.json")])
    assert rc == 0 and (tmp_path / "r.json").exists()


def test_model_card_has_required_sections_and_marks_dry_runs(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    save_weights(make_random_model(1, SMALL), run / "weights.safetensors", model_version="x")
    write_json(run / "config.json", {"name": "jdd-dry-run", "dry_run": True, "seed": 1, "device": "cpu",
                                     "data": {"patch_half": 32, "synthetic_weight": 1.0},
                                     "train": {"steps": 50, "batch_size": 4}, "optim": {"lr": 0.002, "warmup_steps": 5}})
    text = model_card.render(run, "x", "2026-10-12")
    for section in ("## Purpose", "## Architecture", "## Training data and licences", "## Metrics", "## Known failure cases", "## Provenance"):
        assert section in text
    assert "DRY RUN" in text and "Not evaluated" in text and "CC BY 4.0" not in text.split("Training data")[1].split("Not used")[0]
    assert "2026-10-12" in text and "Training code: git" in text


def test_sample_sheet_renders(tmp_path):
    out = sample_sheet.render(tmp_path / "s.png", 5, 16, 1)
    assert out.exists() and out.stat().st_size > 10_000


def test_evaluate_cli_needs_a_path_for_folder_sets(tmp_path):
    import pytest

    save_weights(make_random_model(1, SMALL), tmp_path / "w.safetensors")
    with pytest.raises(SystemExit) as e:  # argparse error, not an AttributeError on None
        E.main(["--weights", str(tmp_path / "w.safetensors"), "--set", "folder"])
    assert e.value.code == 2

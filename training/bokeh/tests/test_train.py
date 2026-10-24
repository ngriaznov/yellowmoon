"""The training loop and the validation numbers, on tiny tiles and the CPU."""

import json

import numpy as np
import pytest
import torch

from training.bokeh import train
from training.bokeh.encode import B_MIN

SMALL = {"sample": 64, "batch": 4, "kernel_r": 10, "r_max_px": 7.0, "device": "cpu", "workers": 0, "amp": False,
         "eval_every_min": 0.0, "final_val_samples": 8, "baseline_n": 4, "val_samples": 4, "warmup": 2, "val_share": 0.3,
         "ckpt_every_min": 1000.0, "steps": 4, "lr": 3e-3, "seed": 3}


def test_the_laplace_loss_is_the_negative_log_likelihood_and_weights_count():
    mu, s, y = torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2), torch.full((1, 1, 2, 2), 2.0)
    w = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    # |2 - 0| * exp(0) + 0 = 2 on the two weighted pixels
    assert float(train.laplace_nll(mu, s, y, w)) == pytest.approx(2.0)
    s2 = torch.full_like(s, np.log(2.0))
    assert float(train.laplace_nll(mu, s2, y, w)) == pytest.approx(2.0 / 2 + np.log(2.0))
    # a wide scale costs less when the error is large, a narrow one when it is small
    assert float(train.laplace_nll(mu, s2, y, w)) < float(train.laplace_nll(mu, s, y, w))
    assert float(train.laplace_nll(y, s, y, w)) == pytest.approx(0.0)


def test_the_learning_rate_follows_a_cosine_over_the_progress_after_a_warm_up():
    cfg = {**train.DEFAULTS, "lr": 1.0, "warmup": 10}
    assert train.lr_at(cfg, 0, 0.0) == pytest.approx(0.1 * 1.0 * (0.02 + 0.98))
    assert train.lr_at(cfg, 50, 0.0) == pytest.approx(1.0)
    assert train.lr_at(cfg, 50, 0.5) == pytest.approx(0.02 + 0.98 * 0.5)
    assert train.lr_at(cfg, 50, 1.0) == pytest.approx(0.02) and train.lr_at(cfg, 50, 3.0) == pytest.approx(0.02)


def test_the_metrics_count_what_the_spec_asks_for():
    acc = train.Acc()
    b_true = np.array([[0.001, 0.001, 0.001, 0.001, 0.01, 0.01, 0.02, 0.02]])
    b_hat = np.array([[0.001, 0.0095, 0.0095, 0.002, 0.012, 0.0, 0.02, 0.03]])
    conf = np.array([[0.9, 0.9, 0.1, 0.9, 0.9, 0.9, 0.2, 0.9]])
    acc.add(b_hat, conf, b_true, np.ones_like(b_true))
    r = acc.result()
    assert r["pixels"]["neg"] == 4 and r["pixels"]["fp"] == 2 and r["pixels"]["fp_g"] == 1  # two predicted >= 2 b_min, one with confidence
    assert r["fp_rate"] == pytest.approx(0.5) and r["fp_rate_gated"] == pytest.approx(0.25)
    assert r["pixels"]["pos"] == 4 and r["pixels"]["pos_g"] == 3
    assert r["mae_b"] == pytest.approx((0.002 + 0.01 + 0.0 + 0.01) / 4, abs=1e-5)  # results are rounded to 5 decimals
    assert r["mae_b_gated"] == pytest.approx((0.002 + 0.01 + 0.01) / 3, abs=1e-5)
    assert r["coverage"] == pytest.approx(0.75)
    assert r["recall_2bmin"] == pytest.approx(3 / 4) and 2 * B_MIN == pytest.approx(0.008)


def test_pixels_with_a_low_weight_are_left_out_of_the_numbers():
    acc = train.Acc()
    acc.add(np.array([[0.05, 0.05]]), np.ones((1, 2)), np.array([[0.001, 0.02]]), np.array([[0.3, 0.3]]))
    assert acc.result()["pixels"]["neg"] == 0 and acc.result()["pixels"]["pos"] == 0


def test_calibration_compares_the_predicted_scale_with_the_error():
    acc = train.Acc()
    rng = np.random.default_rng(0)
    beta = rng.uniform(0.1, 1.0, (1, 80000))
    err = rng.laplace(0, beta)  # errors drawn with exactly the claimed scale
    ones = np.ones_like(beta)
    acc.add(ones * 0.01, ones, ones * 0.01, ones, beta=beta, logerr=np.abs(err))
    assert acc.result()["ece"] < 0.05
    bad = train.Acc()
    bad.add(ones * 0.01, ones, ones * 0.01, ones, beta=beta, logerr=np.abs(err) * 3)
    assert bad.result()["ece"] > 0.2


def test_a_few_steps_run_write_everything_and_are_finite(tile_folder, tmp_path):
    out = tmp_path / "run"
    logs = []
    beats = []
    m = train.train(tile_folder, out, SMALL, heartbeat=beats.append, log=logs.append)
    assert m["steps"] == 4 and 1_400_000 < m["parameters"] < 3_000_000 and m["macs_per_pixel"] > 2000
    for f in ("model.pt", "config.json", "metrics.json", "ckpt.pt", "samples/val-predictions.png"):
        assert (out / f).exists(), f
    final = json.loads((out / "metrics.json").read_text())["final"]
    assert final["samples"] == 8 and final["model"]["pixels"]["pos"] > 0
    assert set(final["baseline_subset"]) == {"samples", "model", "classical"}
    assert np.isfinite(final["model"]["mae_b"]) and beats  # the heartbeat was called at the checkpoint
    state = torch.load(out / "model.pt", weights_only=False)
    assert all(torch.isfinite(v).all() for v in state["state_dict"].values() if v.dtype.is_floating_point)


def test_a_rerun_continues_from_the_checkpoint(tile_folder, tmp_path):
    out = tmp_path / "run"
    train.train(tile_folder, out, SMALL, log=lambda *_: None)
    logs = []
    m = train.train(tile_folder, out, {**SMALL, "steps": 6}, log=logs.append)
    assert any("resumed at step 4" in str(x) for x in logs) and m["steps"] == 6


def test_it_refuses_to_run_without_a_limit(tile_folder, tmp_path):
    with pytest.raises(ValueError, match="max_minutes or steps"):
        train.train(tile_folder, tmp_path / "r", {**SMALL, "steps": None, "max_minutes": None})


def test_the_half_resolution_targets_average_the_log_radius_and_weaken_mixed_blocks():
    from training.bokeh.synth import Synth

    r = torch.ones(1, 1, 4, 4)
    r[..., :2] = 8.0
    r[0, 0, 0, 2] = 8.0  # the block at rows 0-1, columns 2-3 now holds 8 and 1 together: mixed
    w = torch.ones(1, 1, 4, 4)
    sy = Synth(lin=torch.zeros(1, 3, 4, 4), r=r, w=w, hd=torch.ones(1))
    y2, w2, r2 = train.half_targets(sy)
    assert y2.shape == (1, 1, 2, 2)
    assert float(y2[0, 0, 0, 0]) == pytest.approx(np.log(8.0)) and float(y2[0, 0, 1, 1]) == pytest.approx(0.0)
    assert float(w2[0, 0, 0, 0]) == 1.0 and float(w2[0, 0, 0, 1]) == pytest.approx(0.3)

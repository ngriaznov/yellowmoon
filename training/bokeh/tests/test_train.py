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


# --- the guards: a run with a NaN in it must stop, or skip the step, and never go on to train NaNs ---------------------


def _flaky_synthesis(monkeypatch, bad_calls, sample: int = 1):
    """`train.synthesize` that puts a NaN into the lens plane of one sample on the calls whose number is in `bad_calls`
    (1-based; `None` = every call). Returns the list of calls made."""
    real = train.synthesize
    calls = []

    def flaky(tb, cfg, gen):
        sy = real(tb, cfg, gen)
        calls.append(1)
        if bad_calls is None or len(calls) in bad_calls:
            sy.lin[sample, :, 5:9, 5:9] = float("nan")
        return sy

    monkeypatch.setattr(train, "synthesize", flaky)
    return calls


def _records(logs) -> list[dict]:
    return [json.loads(x) for x in logs if str(x).startswith("{")]


def test_nonfinite_names_the_tensors_that_hold_a_nan_or_an_infinity():
    named = {"a": torch.ones(3), "b": torch.tensor([1.0, float("nan")]), "c": torch.tensor([float("inf")]), "ids": torch.arange(3)}
    assert train.nonfinite(named) == ["b", "c"] and train.nonfinite({}) == []
    assert train.bad_rows(torch.ones(3, 2, 2), torch.tensor([[[1.0, 1.0]], [[1.0, float("nan")]], [[1.0, 1.0]]])) == [1]


def test_too_many_skipped_steps_in_a_row_or_too_large_a_share_end_the_run():
    c = train.SkipCounter()
    for _ in range(train.MAX_SKIPS_IN_A_ROW):
        c.skipped(steps_taken=0)  # 20 in a row are tolerated
    with pytest.raises(train.TrainingDiverged, match="^training diverged: 21 steps in a row"):
        c.skipped(steps_taken=0)
    c = train.SkipCounter()
    for _ in range(19):
        c.skipped(steps_taken=100)  # before step 200 only the run-length rule counts ...
        c.applied()
    with pytest.raises(train.TrainingDiverged, match="of the .* steps so far were skipped"):
        c.skipped(steps_taken=250)  # ... and past it, 20 skipped of 270 (7 %) is more than 2 %
    ok = train.SkipCounter()
    for k in range(5):  # 5 of 1000 steps (0.5 %) is fine
        ok.skipped(steps_taken=200 * (k + 1))
        ok.applied()
    assert ok.total == 5
    early = train.SkipCounter()
    for _ in range(10):
        early.skipped(steps_taken=0)  # before step 200 only the run-length rule counts
        early.applied()


def test_the_early_check_wants_a_finite_loss_of_at_most_five():
    train.check_early([0.9, 0.8, 0.7])
    train.check_early([4.9])
    for bad in ([], [float("nan")], [0.5, 9.0, 9.0]):
        with pytest.raises(train.TrainingDiverged, match="mean loss of steps 25 to 100"):
            train.check_early(bad)
    with pytest.raises(train.TrainingDiverged, match="non-finite mae_b, ece"):
        train.check_scores({"mae_b": float("nan"), "fp_rate": 0.0, "ece": float("inf")}, 7)
    train.check_scores({"mae_b": 0.01, "fp_rate": 0.0}, 7)


def test_the_divergence_error_says_so_and_survives_crossing_a_process_boundary():
    import pickle

    e = train.TrainingDiverged("the weights are not finite")
    assert isinstance(e, RuntimeError) and str(e) == "training diverged: the weights are not finite"
    assert str(pickle.loads(pickle.dumps(e))) == str(e)  # Modal rebuilds an exception from its message


def test_a_batch_with_a_nan_is_skipped_and_the_model_goes_on_unharmed(tile_folder, tmp_path, monkeypatch):
    _flaky_synthesis(monkeypatch, bad_calls={2})
    logs = []
    m = train.train(tile_folder, tmp_path / "run", {**SMALL, "steps": 6}, log=logs.append)
    assert m["steps"] == 6 and m["skipped_steps"] == 1  # six steps were made, and one more batch was thrown away
    skipped = [r for r in _records(logs) if "skipped" in r]
    assert len(skipped) == 1 and skipped[0]["skipped"] == "non-finite loss" and skipped[0]["step"] == 1
    assert skipped[0]["samples"] == [1] and "synthesis lin" in skipped[0]["where"] and "encoded input" in skipped[0]["where"]
    assert skipped[0]["draw"] == 1 and skipped[0]["skipped_total"] == 1
    state = torch.load(tmp_path / "run" / "model.pt", weights_only=False)["state_dict"]
    # the NaN batch went through the forward pass, which updates the batch-norm statistics: they were put back
    assert all(torch.isfinite(v).all() for v in state.values() if v.dtype.is_floating_point)
    assert np.isfinite(json.loads((tmp_path / "run" / "metrics.json").read_text())["final"]["model"]["mae_b"])


def test_a_skipped_step_leaves_the_optimizer_and_the_batch_norm_counts_as_if_it_never_happened(tile_folder, tmp_path, monkeypatch):
    _flaky_synthesis(monkeypatch, bad_calls={1})
    train.train(tile_folder, tmp_path / "a", {**SMALL, "steps": 3, "final_val_samples": 4, "baseline_n": 0}, log=lambda *_: None)
    a = torch.load(tmp_path / "a" / "ckpt.pt", weights_only=False)
    assert a["step"] == 3 and a["skipped"] == 1
    assert all(torch.isfinite(v).all() for v in a["model"].values() if v.dtype.is_floating_point)
    moments = [t for per_param in a["opt"]["state"].values() for t in per_param.values() if torch.is_tensor(t)]
    assert moments and all(torch.isfinite(t).all() for t in moments)  # nothing of the NaN batch reached Adam
    counts = [v for k, v in a["model"].items() if k.endswith("num_batches_tracked")]
    assert counts and all(int(v) == 3 for v in counts)  # the forward pass of the thrown-away batch was undone


def test_a_gradient_that_is_not_finite_is_skipped_before_the_optimizer_sees_it(tile_folder, tmp_path, monkeypatch):
    real = torch.nn.utils.clip_grad_norm_
    calls = []

    def clip(params, max_norm, *a, **k):
        total = real(params, max_norm, *a, **k)
        calls.append(1)
        return torch.tensor(float("inf")) if len(calls) == 2 else total

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    logs = []
    m = train.train(tile_folder, tmp_path / "run", {**SMALL, "steps": 4}, log=logs.append)
    assert m["steps"] == 4 and m["skipped_steps"] == 1
    skipped = [r for r in _records(logs) if "skipped" in r]
    assert skipped[0]["skipped"] == "non-finite gradient" and skipped[0]["where"] == ["gradient"]
    state = torch.load(tmp_path / "run" / "model.pt", weights_only=False)["state_dict"]
    assert all(torch.isfinite(v).all() for v in state.values() if v.dtype.is_floating_point)


def test_a_run_whose_every_batch_is_bad_stops_after_twenty_one_steps_and_writes_nothing_to_resume_from(tile_folder, tmp_path, monkeypatch):
    calls = _flaky_synthesis(monkeypatch, bad_calls=None)
    logs = []
    with pytest.raises(RuntimeError, match="^training diverged: 21 steps in a row were skipped"):
        train.train(tile_folder, tmp_path / "run", {**SMALL, "steps": 500}, log=logs.append)
    assert len(calls) == 21
    last = _records(logs)[-1]
    assert list(last) == ["diverged"] and last["diverged"].startswith("training diverged: 21 steps in a row")
    assert not (tmp_path / "run" / "model.pt").exists() and not (tmp_path / "run" / "ckpt.pt").exists()
    assert [r["draw"] for r in _records(logs) if "skipped" in r][:3] == [0, 1, 2]  # every retry is another draw


def test_the_run_stops_at_the_early_check_when_the_loss_is_not_what_a_fresh_network_scores(tile_folder, tmp_path, monkeypatch):
    monkeypatch.setattr(train, "EARLY_STEPS", (2, 4))
    monkeypatch.setattr(train, "EARLY_MAX_LOSS", 1e-6)
    logs = []
    with pytest.raises(train.TrainingDiverged, match="mean loss of steps 2 to 4"):
        train.train(tile_folder, tmp_path / "run", {**SMALL, "steps": 50}, log=logs.append)
    assert _records(logs)[-1]["diverged"].startswith("training diverged: the mean loss") and not (tmp_path / "run" / "model.pt").exists()


def test_a_nan_score_in_an_evaluation_stops_the_run(tile_folder, tmp_path, monkeypatch):
    nan_scores = {"model": {"mae_b": float("nan"), "fp_rate": 0.0, "fp_rate_gated": 0.0, "ece": float("nan")}}
    monkeypatch.setattr(train, "evaluate", lambda *a, **k: nan_scores)
    with pytest.raises(train.TrainingDiverged, match="evaluation at step 1 returned a non-finite mae_b, ece"):
        train.train(tile_folder, tmp_path / "run", {**SMALL, "steps": 50, "eval_every_min": 1e-9}, log=lambda *_: None)


def test_weights_that_are_not_finite_are_never_evaluated_drawn_or_checkpointed(tile_folder, tmp_path):
    from training.bokeh.model import BokehNet

    net = BokehNet()
    with torch.no_grad():
        net.head.bias[0] = float("nan")
    sheet = tmp_path / "sheet.png"
    with pytest.raises(train.TrainingDiverged, match="weights to evaluate are not finite"):
        train.evaluate(net, tile_folder, {**train.DEFAULTS, **SMALL}, 4, 7, torch.device("cpu"), sheet=sheet)  # was a device-side assert
    assert not sheet.exists()
    # a checkpoint of NaNs is not resumed from either (a run of that name would otherwise train on, for the whole budget)
    out = tmp_path / "run"
    train.train(tile_folder, out, SMALL, log=lambda *_: None)
    state = torch.load(out / "ckpt.pt", weights_only=False)
    with torch.no_grad():
        next(iter(state["model"].values())).fill_(float("nan"))
    torch.save(state, out / "ckpt.pt")
    with pytest.raises(train.TrainingDiverged, match="weights in the checkpoint are not finite"):
        train.train(tile_folder, out, {**SMALL, "steps": 8}, log=lambda *_: None)

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


def _blocks():
    """A 4 x 4 picture whose 2 x 2 blocks are: all 8 px (top left), all 1 px (bottom right), 8 px with one pixel of 1 px
    (top right: mixed, and the sharp pixel is one of weight 1), and 8 px with a pixel of 1 px whose weight is low (bottom left)."""
    from training.bokeh.synth import Synth

    r = torch.ones(1, 1, 4, 4)
    r[..., :2] = 8.0
    r[0, 0, 0, 2] = 8.0  # the block at rows 0-1, columns 2-3 now holds 8 and 1 together: mixed
    r[0, 0, 3, 0] = 1.0  # bottom left: three of 8 px and one of 1 px
    w = torch.ones(1, 1, 4, 4)
    w[0, 0, 3, 0] = 0.3  # ... and that one is an ambiguous pixel
    return Synth(lin=torch.zeros(1, 3, 4, 4), r=r, w=w, hd=torch.ones(1))


def test_the_half_resolution_targets_of_bokeh_2_average_the_log_radius_and_weaken_mixed_blocks():
    y2, w2, r2 = train.half_targets(_blocks(), sharp_wins=False)
    assert y2.shape == (1, 1, 2, 2)
    assert float(y2[0, 0, 0, 0]) == pytest.approx(np.log(8.0)) and float(y2[0, 0, 1, 1]) == pytest.approx(0.0)
    assert float(y2[0, 0, 0, 1]) == pytest.approx(np.log(8.0) * 0.25)  # the average of 8, 1, 1, 1 in the log
    assert float(w2[0, 0, 0, 0]) == 1.0 and float(w2[0, 0, 0, 1]) == pytest.approx(0.3)


def test_in_a_mixed_block_the_sharp_pixel_wins_when_it_is_a_firm_one():
    y2, w2, r2 = train.half_targets(_blocks())  # the default
    assert float(y2[0, 0, 0, 0]) == pytest.approx(np.log(8.0)) and float(w2[0, 0, 0, 0]) == 1.0  # a pure block is as before
    assert float(y2[0, 0, 0, 1]) == pytest.approx(0.0) and float(r2[0, 0, 0, 1]) == pytest.approx(1.0)  # 8, 1, 1, 1: the smallest
    assert float(w2[0, 0, 0, 1]) == pytest.approx(0.7)
    # a block whose only sharp pixel has a low weight is not decided by it: the firm ones (the 8 px) are
    assert float(y2[0, 0, 1, 0]) == pytest.approx(np.log(8.0)) and float(w2[0, 0, 1, 0]) == pytest.approx(0.7 * (3 + 0.3) / 4)
    assert float(y2[0, 0, 1, 1]) == pytest.approx(0.0)


def test_a_thin_sharp_line_over_a_blur_stays_sharp_at_half_resolution():
    from training.bokeh.synth import Synth

    r = torch.full((1, 1, 8, 8), 12.0)
    r[..., 3, :] = 0.0  # a one-pixel line across
    w = torch.ones(1, 1, 8, 8)
    sy = Synth(lin=torch.zeros(1, 3, 8, 8), r=r, w=w, hd=torch.ones(1))
    old, _, _ = train.half_targets(sy, sharp_wins=False)
    new, wn, _ = train.half_targets(sy)
    assert float(old[0, 0, 1, 0]) == pytest.approx(0.5 * np.log(12.0))  # absorbed: half way to the blur
    assert float(new[0, 0, 1, 0]) == pytest.approx(0.0) and float(wn[0, 0, 1, 0]) == pytest.approx(0.7)  # held as sharp
    assert float(new[0, 0, 0, 0]) == pytest.approx(np.log(12.0))  # the blur beside it is untouched


def test_a_block_with_a_thin_structure_counts_more_when_asked():
    from training.bokeh.synth import THIN, Synth

    r = torch.full((1, 1, 8, 8), 12.0)
    r[..., 3, :] = 0.0
    cls = torch.zeros(1, 1, 8, 8, dtype=torch.uint8)
    cls[..., 3, :] = THIN
    sy = Synth(lin=torch.zeros(1, 3, 8, 8), r=r, w=torch.ones(1, 1, 8, 8), hd=torch.ones(1), cls=cls)
    _, w1, _ = train.half_targets(sy)
    _, w2, _ = train.half_targets(sy, thin_boost=2.0)
    assert float(w2[0, 0, 1, 0]) == pytest.approx(2 * float(w1[0, 0, 1, 0]))  # the block on the line
    assert float(w2[0, 0, 0, 0]) == float(w1[0, 0, 0, 0])  # the blur beside it
    assert torch.equal(train.half_targets(sy, thin_boost=2.0)[0], train.half_targets(sy)[0])  # the target is not changed
    _, w3, _ = train.half_targets(Synth(lin=sy.lin, r=r, w=sy.w, hd=sy.hd), thin_boost=2.0)  # no classes (bokeh-2's synthesis): nothing to boost
    assert torch.equal(w3, w1)


def test_the_decision_loss_is_a_cross_entropy_of_the_probability_that_b_reaches_twice_b_min():
    hd = torch.tensor([500.0])
    x = float(np.log(2 * B_MIN * 500.0))  # the threshold in the outputs' units
    w = torch.ones(1, 1, 2, 2)
    y = torch.tensor([[[[x + 1.0, x + 1.0], [x - 1.0, x - 1.0]]]])  # two blocks that are blurred, two that are not
    s = torch.full((1, 1, 2, 2), float(np.log(0.3)))
    sure = train.decision_nll(y.clone(), s, y, w, hd)  # the output is the truth, narrow: P is high where the label is blurred, low where not
    wrong = train.decision_nll(torch.flip(y, dims=(2,)), s, y, w, hd)
    assert float(sure) < 0.1 < float(wrong)
    # a block whose own Laplace has half its mass above the threshold costs log 2
    assert float(train.decision_nll(torch.full((1, 1, 1, 1), x), torch.zeros(1, 1, 1, 1), torch.full((1, 1, 1, 1), x + 1), torch.ones(1, 1, 1, 1), hd)) == pytest.approx(np.log(2), abs=1e-4)


def test_the_group_numbers_count_thin_flat_and_the_blur_bands():
    g = train.Groups()
    b_true = np.array([[0.001, 0.001, 0.009, 0.01, 0.02, 0.03, 0.001, 0.001]])
    b_hat = np.array([[0.009, 0.001, 0.009, 0.002, 0.02, 0.001, 0.001, 0.02]])
    p = np.array([[0.9, 0.1, 0.8, 0.1, 0.95, 0.2, 0.1, 0.9]])
    w = np.ones_like(b_true)
    thin = np.array([[True, True, False, False, False, False, False, False]])
    flat = np.array([[False, False, False, False, False, False, True, True]])
    g.add(b_hat, p, b_true, w, thin, flat)
    r = g.result()
    assert r["thin_false_blur_raw"] == pytest.approx(0.5) and r["thin_false_blur_p70"] == pytest.approx(0.5)
    assert r["flat_false_blur_raw"] == pytest.approx(0.5) and r["flat_false_blur_p70"] == pytest.approx(0.5)
    # blurred truth in [2 b_min, 0.012): 0.009 and 0.01: raw hits the first only, p the first only; above: 0.02 hit, 0.03 missed
    assert r["mid_recall_raw"] == pytest.approx(0.5) and r["mid_recall_p70"] == pytest.approx(0.5)
    assert r["high_recall_raw"] == pytest.approx(0.5) and r["high_recall_p70"] == pytest.approx(0.5)
    assert r["blocks"] == {"thin": 2, "flat": 2, "mid": 2, "high": 2, "truth": 4}


def test_v3_masks_pick_a_sharp_thin_structure_over_a_blur_and_the_far_inside_of_a_sharp_flat_layer():
    from training.bokeh.synth import FLAT, THIN, Synth

    r = torch.zeros(1, 1, 16, 16)
    r[..., 8:, :] = 10.0
    cls = torch.zeros(1, 1, 16, 16, dtype=torch.uint8)
    cls[0, 0, 6, 2] = THIN  # a sharp thin pixel, with a 10 px blur within 4 px: counts
    cls[0, 0, 1, 1] = THIN  # a sharp thin pixel with no blur near: does not
    cls[0, 0, 4:8, 12:16] = FLAT  # a flat layer, sharp, whose weight was cut
    w = torch.ones(1, 1, 16, 16)
    w[0, 0, 4:8, 12:16] = 0.1
    sy = Synth(lin=torch.zeros(1, 3, 16, 16), r=r, w=w, hd=torch.ones(1), cls=cls)
    thin, flat = train.v3_masks(sy)
    assert thin.shape == (1, 8, 8) and bool(thin[0, 3, 1]) and not bool(thin[0, 0, 0])
    assert bool(flat[0, 2, 6]) and bool(flat[0, 3, 7]) and not bool(flat[0, 0, 0])
    z = Synth(lin=sy.lin, r=r, w=w, hd=torch.ones(1))  # the bokeh-2 synthesis: no classes
    assert not bool(torch.stack(train.v3_masks(z)).any())


def test_the_final_evaluation_reports_the_groups_and_bokeh_2s_own_validation(tile_folder, tmp_path):
    m = train.train(tile_folder, tmp_path / "run", SMALL, log=lambda *_: None)
    final = m["final"]
    assert {"thin_false_blur_p70", "flat_false_blur_p70", "mid_recall_p70", "precision_p70", "recall_p70"} <= set(final["groups"])
    assert set(final["legacy"]) == {"model", "groups"} and final["legacy"]["model"]["pixels"]["pos"] > 0  # same samples as bokeh-2's final numbers
    assert "fp_rate_p70" in final["real_sharp"]["model"]
    old = train.train(tile_folder, tmp_path / "old", {**SMALL, "synth": 2, "sharp_wins": False}, log=lambda *_: None)
    assert "legacy" not in old["final"] and old["final"]["groups"]["blocks"]["thin"] == 0


def test_a_run_can_start_from_the_weights_of_another_and_the_decision_loss_trains(tile_folder, tmp_path):
    first = tmp_path / "first"
    train.train(tile_folder, first, SMALL, log=lambda *_: None)
    logs = []
    out = tmp_path / "second"
    m = train.train(tile_folder, out, {**SMALL, "init_from": str(first / "model.pt"), "decision_loss": 0.3, "steps": 1, "lr": 1e-9, "warmup": 1}, log=logs.append)
    assert any("started from the weights of" in str(x) for x in logs) and m["steps"] == 1
    a = torch.load(first / "model.pt", weights_only=False)["state_dict"]
    b = torch.load(out / "model.pt", weights_only=False)["state_dict"]
    for k, v in a.items():
        if v.dtype.is_floating_point and "running_" not in k:
            assert torch.allclose(v, b[k], atol=1e-4), k  # one step at a tiny learning rate: the same network (batch-norm statistics move)
    # a rerun of the second run resumes from its own checkpoint, not from the weights it started from
    logs2 = []
    train.train(tile_folder, out, {**SMALL, "init_from": str(first / "model.pt"), "steps": 2, "lr": 1e-9, "warmup": 1}, log=logs2.append)
    assert any("resumed at step 1" in str(x) for x in logs2) and not any("started from" in str(x) for x in logs2)


def test_the_synthesis_config_follows_the_run_config():
    cfg = {**train.DEFAULTS}
    assert train.synth_config(cfg).thin_layers != (1.0,) and len(train.synth_config(cfg).mode_probs) == 5
    assert train.synth_config({**cfg, "synth": 2}).thin_layers == (1.0,) and len(train.synth_config({**cfg, "synth": 2}).mode_probs) == 4
    assert len(train.synth_config(cfg, legacy=True).mode_probs) == 4


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


class _HostRefusing(torch.Tensor):
    """Stands in for a tensor on a GPU, which CUDA-less machines do not have: like one, it cannot become a numpy array
    before `.cpu()` has brought it back (the error is CUDA's own wording)."""

    def numpy(self, *args, **kwargs):
        raise TypeError("can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.")

    def cpu(self, *args, **kwargs):
        return self.as_subclass(torch.Tensor)


class _OffHostNet(torch.nn.Module):
    """A network "on another device": its weights live on the `meta` device (so `next(parameters()).device` is not the
    CPU) and it records where its input was. The outputs it returns refuse `.numpy()` like a GPU's."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(1, device="meta"))
        self.input_devices: list[torch.device] = []

    def forward(self, x):
        self.input_devices.append(x.device)
        n, _, h, w = x.shape
        return torch.zeros(n, 2, h // 2, w // 2).as_subclass(_HostRefusing)


def test_the_real_sharp_check_runs_the_network_where_its_weights_are_and_brings_the_answers_back(tile_folder):
    """Was a Modal run's last step: the input stayed on the CPU while the model was on the GPU, and the outputs were
    turned into numpy arrays without `.cpu()`. Without a GPU the other device is `meta`."""
    net = _OffHostNet()
    ts = train.sources.TileSet(tile_folder, SMALL["val_share"])
    pool = ts.val or ts.train
    out = train.real_sharp(net, ts, pool, 3, baseline_n=2)  # the baseline reads the CPU copy, whatever the network's device
    assert net.input_devices and all(d == torch.device("meta") for d in net.input_devices)
    assert out["tiles"] == len(net.input_devices) and set(out) == {"tiles", "model", "classical"}
    assert 0.0 <= out["model"]["fp_rate_gated"] <= out["model"]["fp_rate"] <= 1.0

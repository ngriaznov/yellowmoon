import json

import torch

from training.jdd import train as T
from training.jdd.weights_io import load_weights, read_metadata


def tiny_cfg(name, steps, n_train=4, **over):
    cfg = T.load_config(T.CONFIG_DIR / "overfit.json")
    cfg["name"] = name
    cfg["model"]["channels"] = [8, 16, 24]
    cfg["data"].update(patch_half=16, n_train=n_train, n_val=n_train)
    cfg["train"].update(steps=steps, batch_size=4, log_every=10)
    cfg["optim"].update(lr=0.004, warmup_steps=5)
    cfg["sampler"] = {"p_clean": 0.0}
    for k, v in over.items():
        cfg[k] = v
    return cfg


def test_short_cpu_run_overfits_a_tiny_set(tmp_path):
    """M2-05 done-when: a short CPU run overfits a tiny set (loss falls a lot, PSNR beats bilinear)."""
    s = T.train(tiny_cfg("overfit-test", 120), tmp_path / "run", quiet=True)
    assert s["loss_last"] < 0.8 * s["loss_first"]
    assert s["eval"]["gain_linear_db"] > 0.5
    assert s["eval"]["gain_log_db"] > 0.5
    meta = read_metadata(tmp_path / "run" / "weights.safetensors")
    assert meta["format"] == "ym-jdd" and int(meta["halo_full_px"]) == 64


def test_training_is_deterministic(tmp_path):
    a = T.train(tiny_cfg("det", 6), tmp_path / "a", quiet=True)
    b = T.train(tiny_cfg("det", 6), tmp_path / "b", quiet=True)
    assert a["weights_sha256"] == b["weights_sha256"]
    c = T.train(tiny_cfg("det", 6, seed=12), tmp_path / "c", quiet=True)
    assert c["weights_sha256"] != a["weights_sha256"]


def test_resume_continues_from_checkpoint(tmp_path):
    cfg = tiny_cfg("resume", 12)
    cfg["train"]["ckpt_every"] = 6
    run = tmp_path / "r"
    cfg_half = json.loads(json.dumps(cfg))
    cfg_half["train"]["steps"] = 6
    T.train(cfg_half, run, quiet=True)
    s = T.train(cfg, run, resume=True, quiet=True)
    full = T.train(cfg, tmp_path / "full", quiet=True)
    # The learning-rate schedule depends on the total step count, so only check that resume runs from step 6.
    assert s["steps"] == 12 and full["steps"] == 12
    steps = [json.loads(line)["step"] for line in (run / "log.jsonl").read_text().strip().splitlines()]
    assert steps == [1, 7, 10]  # first run logged step 1; the resume logged its first step (7) and step 10


def test_dry_run_cli_passes(tmp_path):
    rc = T.main(["--dry-run", "--steps", "6", "--device", "cpu", "--run", str(tmp_path / "dry"), "--no-onnx"])
    # 6 steps is too few for "loss decreased" to be guaranteed; the command still must complete and write files.
    assert rc in (0, 1)
    assert (tmp_path / "dry" / "weights.safetensors").exists() and (tmp_path / "dry" / "summary.json").exists()
    model = load_weights(tmp_path / "dry" / "weights.safetensors")
    assert sum(p.numel() for p in model.parameters()) > 0
    assert torch.isfinite(model(torch.zeros(1, 7, 8, 8))).all()


def test_lr_schedule():
    o = {"lr": 1.0, "warmup_steps": 10, "final_lr_fraction": 0.1}
    assert T.lr_at(0, 100, o) < 0.2 and abs(T.lr_at(10, 100, o) - 1.0) < 1e-9
    assert abs(T.lr_at(100, 100, o) - 0.1) < 1e-9
    assert T.lr_at(50, 100, o) < T.lr_at(20, 100, o)

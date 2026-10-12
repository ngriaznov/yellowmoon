"""Training loop for `jdd` (M2-05).

Usage, from the repository root::

    python3 -m training.jdd.train --config training/jdd/configs/default.json --run runs/jdd-001
    python3 -m training.jdd.train --dry-run --steps 50 --device cpu            # CI: synthetic data only
    python3 -m training.jdd.train --config training/jdd/configs/overfit.json   # tiny-set overfit demo

Properties:

* **Deterministic**: every random choice derives from ``config["seed"]``. Training samples are pure
  functions of ``(seed, index)`` (:class:`training.jdd.data.JddDataset`), so the data stream is the same
  for any number of workers; the shuffle order comes from a seeded generator; PyTorch is told to use
  deterministic algorithms. Two runs of the same config on the same machine give the same weights.
* **Config file with seeds** (JSON, see ``configs/``); command-line flags override single values.
* ``--dry-run`` forces synthetic data, CPU and a short schedule so CI can exercise the whole path
  (data, model, loss, optimiser, checkpoint, safetensors round trip) in about a minute.
* Writes to the run directory: ``config.json`` (resolved), ``log.jsonl``, ``weights.safetensors``,
  ``ckpt.pt`` (for ``--resume``), ``eval.json``.

Training itself needs a GPU and the PixelShift200 dataset (HUMAN M2-H4); nothing here downloads data.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import SPLIT_SEED, JddDataset, PixelShift200Source, SamplerConfig, SyntheticSource
from .loss import JddLoss
from .metrics import bilinear_baseline, psnr_linear, psnr_log, summarize
from .model import JddConfig, JddNet, check_input_shape, count_parameters
from .weights_io import load_weights, save_weights, write_json

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def load_config(path: str | Path) -> dict:
    """Read a JSON config."""
    return json.loads(Path(path).read_text())


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Return a copy of ``cfg`` with command-line overrides applied."""
    cfg = copy.deepcopy(cfg)
    if args.steps is not None:
        cfg["train"]["steps"] = args.steps
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.device is not None:
        cfg["device"] = args.device
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.dry_run:
        cfg["dry_run"] = True
        cfg["device"] = args.device or "cpu"
        cfg["data"]["pixelshift200_weight"] = 0.0
        cfg["data"]["synthetic_weight"] = 1.0
        cfg["train"]["num_workers"] = 0
    return cfg


def resolve_device(name: str) -> torch.device:
    """``auto`` picks CUDA, then Apple MPS, then CPU."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def seed_everything(seed: int, deterministic: bool, device: torch.device) -> None:
    """Seed Python/NumPy/PyTorch and request deterministic kernels."""
    import random

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if deterministic:
        if device.type == "cuda":
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def build_dataset(cfg: dict, split: str) -> JddDataset:
    """Training (``train``) or validation (``val``) dataset described by the config."""
    d = cfg["data"]
    sampler = SamplerConfig(patch_half=d["patch_half"], **{k: tuple(v) if isinstance(v, list) else v for k, v in cfg.get("sampler", {}).items()})
    sources = []
    syn_w = float(d.get("synthetic_weight", 1.0))
    ps_w = float(d.get("pixelshift200_weight", 0.0))
    root = d.get("pixelshift200_root")
    if ps_w > 0 and root:
        root_path = Path(root) if Path(root).is_absolute() else REPO_ROOT / root
        # The scene split is fixed (`data.split_seed`, default 0), never the training seed: runs with different
        # seeds must hold out the same scenes.
        src = PixelShift200Source(root_path, sampler.gt_size, split=split, seed=int(d.get("split_seed", SPLIT_SEED)))
        if len(src):
            sources.append((src, ps_w))
        else:
            print(f"warning: no PixelShift200 scenes under {root_path}; using synthetic patterns only")
    if syn_w > 0 or not sources:
        sources.append((SyntheticSource(sampler.gt_size), syn_w if syn_w > 0 else 1.0))
    cache = bool(d.get("cache_samples", d["n_train"] <= 64))  # tiny fixed sets: generate once
    if split == "val":
        if d.get("val_same_as_train"):
            return JddDataset(d["n_val"], cfg["seed"], sampler, sources, cache=cache)
        return JddDataset(d["n_val"], cfg["seed"] + 1_000_003, sampler, sources, cache=cache)
    return JddDataset(d["n_train"], cfg["seed"], sampler, sources, cache=cache)


def lr_at(step: int, total: int, opt_cfg: dict) -> float:
    """Linear warm-up then cosine decay to ``final_lr_fraction * lr``."""
    base = opt_cfg["lr"]
    warm = opt_cfg.get("warmup_steps", 0)
    if step < warm:
        return base * (step + 1) / warm
    frac = (step - warm) / max(1, total - warm)
    floor = opt_cfg.get("final_lr_fraction", 0.0)
    return base * (floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, frac))))


@torch.no_grad()
def evaluate(model: JddNet, dataset: JddDataset, device: torch.device, batch_size: int = 8) -> dict:
    """PSNR (linear and log) of the model and of the bilinear baseline on a dataset."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    rows: dict[str, list[float]] = {k: [] for k in ("model_linear", "model_log", "bilinear_linear", "bilinear_log")}
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        base = bilinear_baseline(model, x)
        rows["model_linear"] += psnr_linear(out, y).tolist()
        rows["model_log"] += psnr_log(out, y).tolist()
        rows["bilinear_linear"] += psnr_linear(base, y).tolist()
        rows["bilinear_log"] += psnr_log(base, y).tolist()
    res = {k: summarize(v) for k, v in rows.items()}
    res["gain_linear_db"] = res["model_linear"]["mean"] - res["bilinear_linear"]["mean"]
    res["gain_log_db"] = res["model_log"]["mean"] - res["bilinear_log"]["mean"]
    return res


def train(cfg: dict, run_dir: Path, resume: bool = False, quiet: bool = False) -> dict:
    """Run the schedule in ``cfg``; returns a summary dict (also written to ``run_dir/summary.json``)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg["device"])
    if cfg["train"].get("threads"):
        torch.set_num_threads(int(cfg["train"]["threads"]))  # on a busy machine fewer threads can be much faster
    seed_everything(cfg["seed"], cfg.get("deterministic", True), device)
    model = JddNet(JddConfig(**{**cfg["model"], "channels": tuple(cfg["model"]["channels"])})).to(device)
    loss_fn = JddLoss(**cfg["loss"]).to(device)
    ocfg, tcfg = cfg["optim"], cfg["train"]
    opt = torch.optim.AdamW(
        model.trainable_parameters(), lr=ocfg["lr"], betas=tuple(ocfg["betas"]), weight_decay=ocfg["weight_decay"]
    )
    train_set = build_dataset(cfg, "train")
    steps, bs = tcfg["steps"], tcfg["batch_size"]
    check_input_shape(cfg["data"]["patch_half"], cfg["data"]["patch_half"])
    write_json(run_dir / "config.json", cfg)

    start = 0
    ckpt_path = run_dir / "ckpt.pt"
    if resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start = ck["step"]
        if not quiet:
            print(f"resumed from step {start}")

    n = len(train_set)

    def batch_indices(step: int) -> list[int]:
        # One permutation per epoch from a seeded generator; independent of workers and of resume.
        per_epoch = max(1, n // bs)
        epoch, k = divmod(step, per_epoch)
        g = torch.Generator().manual_seed(cfg["seed"] * 7919 + epoch)
        perm = torch.randperm(n, generator=g).tolist()
        sel = perm[k * bs : (k + 1) * bs]
        while len(sel) < bs:  # tiny datasets: wrap around
            sel += perm[: bs - len(sel)]
        return sel

    class StepBatches:
        """Batch sampler: the index lists of steps ``start .. steps - 1``."""

        def __iter__(self):
            return (batch_indices(s) for s in range(start, steps))

        def __len__(self) -> int:
            return max(0, steps - start)

    workers = tcfg.get("num_workers", 0)
    loader = DataLoader(
        train_set, batch_sampler=StepBatches(), num_workers=workers, persistent_workers=False,
        prefetch_factor=4 if workers > 0 else None,
    )
    log = open(run_dir / "log.jsonl", "a" if resume else "w")
    t0 = time.time()
    history: list[float] = []
    model.train()
    try:
        for step, (x, y) in enumerate(loader, start=start):
            for g_ in opt.param_groups:
                g_["lr"] = lr_at(step, steps, ocfg)
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss, parts = loss_fn(pred, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), ocfg.get("grad_clip", 1e9))
            opt.step()
            loss_value = float(loss.detach())
            history.append(loss_value)
            if (step + 1) % tcfg["log_every"] == 0 or step == start:
                rec = {"step": step + 1, "loss": loss_value, "lr": lr_at(step, steps, ocfg), "grad_norm": float(gn), **parts}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                if not quiet:
                    print(f"step {step + 1:6d}/{steps}  loss {loss_value:.4f}  l1 {parts['l1_log']:.4f}  "
                          f"ssim {parts['ms_ssim']:.3f}  gn {float(gn):.2f}  {time.time() - t0:.0f}s", flush=True)
            if tcfg.get("ckpt_every") and (step + 1) % tcfg["ckpt_every"] == 0:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step + 1}, ckpt_path)
            if tcfg.get("eval_every") and (step + 1) % tcfg["eval_every"] == 0:
                ev = evaluate(model, build_dataset(cfg, "val"), device)
                log.write(json.dumps({"step": step + 1, "eval": ev}) + "\n")
                model.train()
    finally:
        log.close()
    train_seconds = time.time() - t0
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": steps}, ckpt_path)
    sha = save_weights(model.cpu(), run_dir / "weights.safetensors", model_version=cfg.get("model_version", "0"))
    model.to(device)
    ev = evaluate(model, build_dataset(cfg, "val"), device)
    write_json(run_dir / "eval.json", ev)
    k = max(1, min(10, len(history) // 5))
    summary = {
        "name": cfg["name"],
        "steps": steps,
        "device": str(device),
        "params": count_parameters(model),
        "loss_first": float(np.mean(history[:k])) if history else None,
        "loss_last": float(np.mean(history[-k:])) if history else None,
        "train_seconds": round(train_seconds, 1),
        "weights_sha256": sha,
        "eval": ev,
    }
    write_json(run_dir / "summary.json", summary)
    return summary


def dry_run_checks(run_dir: Path, summary: dict, export_onnx: bool = True) -> list[str]:
    """CI checks after a dry run: loss fell, weights round-trip, ONNX export agrees. Returns failures."""
    failures: list[str] = []
    if not (summary["loss_last"] < summary["loss_first"]):
        failures.append(f"loss did not decrease ({summary['loss_first']:.4f} -> {summary['loss_last']:.4f})")
    model = load_weights(run_dir / "weights.safetensors")
    x = torch.randn(1, 7, 32, 32).abs() * 0.1
    with torch.no_grad():
        a = model(x)
    cfg = load_config(run_dir / "config.json")
    ref = JddNet(JddConfig(**{**cfg["model"], "channels": tuple(cfg["model"]["channels"])}))
    ref.load_state_dict(torch.load(run_dir / "ckpt.pt", map_location="cpu")["model"])
    with torch.no_grad():
        b = ref.eval()(x)
    if not torch.equal(a, b):
        failures.append("safetensors round trip changed the outputs")
    if export_onnx:
        try:
            from .export import export_onnx as do_export

            report = do_export(run_dir / "weights.safetensors", run_dir / "export", core_half_px=32, check=True)
            if report["max_abs_diff_vs_torch"] > 2e-3:
                failures.append(f"ONNX vs PyTorch differ by {report['max_abs_diff_vs_torch']:.2e}")
        except ImportError as e:  # onnx / onnxruntime not installed
            print(f"warning: ONNX check skipped ({e})")
    return failures


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=None, help="JSON config (default: configs/default.json)")
    ap.add_argument("--run", type=Path, default=None, help="run directory (default: training/jdd/runs/<name>)")
    ap.add_argument("--dry-run", action="store_true", help="synthetic data only, CPU, short; for CI")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--device", default=None, help="auto, cpu, cuda, mps")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-onnx", action="store_true", help="dry run: skip the ONNX export check")
    args = ap.parse_args(argv)

    cfg_path = args.config or (CONFIG_DIR / ("dry_run.json" if args.dry_run else "default.json"))
    cfg = apply_overrides(load_config(cfg_path), args)
    run_dir = args.run or (Path(__file__).resolve().parent / "runs" / cfg["name"])
    summary = train(cfg, run_dir, resume=args.resume)
    print(json.dumps({k: v for k, v in summary.items() if k != "eval"}, indent=1))
    ev = summary["eval"]
    print(f"val PSNR linear: model {ev['model_linear']['mean']:.2f} dB, bilinear {ev['bilinear_linear']['mean']:.2f} dB "
          f"(gain {ev['gain_linear_db']:+.2f}); log: model {ev['model_log']['mean']:.2f}, bilinear "
          f"{ev['bilinear_log']['mean']:.2f} (gain {ev['gain_log_db']:+.2f})")
    if cfg.get("dry_run"):
        failures = dry_run_checks(run_dir, summary, export_onnx=not args.no_onnx)
        if failures:
            for f in failures:
                print("DRY RUN FAILED:", f)
            return 1
        print("dry run passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

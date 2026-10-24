"""Trains BokehNet on synthetic defocus rendered on the fly (`synth.py`) from the tiles of `sources.py`.

    python -m training.bokeh.train --tiles <tiles folder> --out <run folder> --steps 30 --batch 4 --sample 192 \
        --device cpu --workers 0

**Loss.** Laplace negative log-likelihood of the log radius: for each pixel of the half-resolution output,
`w * (|y - mu| * exp(-s) + s)` with `y` the target `log(max(r, 1 px))` (average of a 2 x 2 block) and `w` the label's
weight (lowered at the ambiguous edges between layers, and where a block mixes very different radii). The network
therefore says how sure it is: a textureless region gets a wide scale instead of a confident wrong value. The
confidence the app uses is `c = exp(-exp(s) / 0.25)` (`encode.py`).

**Schedule.** Training stops after `--max-minutes` (or `--steps`) and the cosine learning-rate schedule runs over
*elapsed time*, not over steps, so it ends at the same place whatever the GPU's speed (a run with an unknown
throughput finishes properly instead of being cut at a high learning rate). AdamW, 300 warm-up steps, an EMA of the
weights (decay 0.999) is what is evaluated and exported, bf16 autocast on a GPU, gradient clipping at 1. A checkpoint
(model, EMA, optimiser, step, elapsed time) is written every `ckpt_every_min` minutes; a rerun with the same `--out`
continues from it, and `heartbeat(seconds)` is called then so that the budget ledger knows how long the run lived.

**Guards** (the first Modal run had NaN losses from step 50 on and trained, for 100 minutes and $2.49, a network of NaNs: one
NaN kernel in the synthesis, `kernels.py`, and nothing noticed). A step whose loss is not finite is skipped before the
backward pass: no optimiser step, no EMA update, the batch-norm statistics put back, and a log line says which input was
not finite (`{"step", "skipped", "where", "samples", "draw"}`). A step whose gradient norm is not finite is skipped the
same way (the clipping would otherwise write NaN into every weight). The run is stopped with `TrainingDiverged` (a
`RuntimeError` that starts "training diverged: ") when more than `MAX_SKIPS_IN_A_ROW` steps in a row were skipped, when more
than `MAX_SKIP_SHARE` of the steps were after step `SKIP_SHARE_AFTER`, when a weight (or the EMA) is not finite, when an
evaluation returns a non-finite score, and as soon as step `EARLY_STEPS[1]` is reached with a mean loss over steps
`EARLY_STEPS` that is not finite or above `EARLY_MAX_LOSS` (a healthy start is about 1: the run would have stopped after one
minute). The log then ends with `{"diverged": ...}`, and no checkpoint with non-finite weights is ever written.

**Validation** (`evaluate`): on held-out tiles (a hash of the photo id, never trained on) with fixed seeds, on pixels
whose label is unambiguous (weight at least 0.6), in the spec's units `b = r / half-diagonal`:

- `mae_b`: mean |b_hat - b| where the truth is at least `b_min` (0.004); `mae_b_gated` the same where the confidence is
  at least 0.5, with its `coverage`;
- `fp_rate`: the share of pixels with truth below `b_min` where the prediction is at least `2 * b_min`, and
  `fp_rate_gated` with the confidence gate (the spec's rule is precision over recall: a false positive smears detail);
- `recall_2bmin`: of the pixels with truth at least `2 * b_min`, the share predicted at least `2 * b_min` (gated too);
- calibration: for ten bins of the predicted scale beta, the mean beta against the mean |log error| (a Laplace scale
  is the expected absolute error), and `ece` the weighted mean gap between them;
- the same numbers for the classical baseline (`classical.py`) on the first `baseline_n` samples, next to the
  model's numbers on the same samples;
- `real_sharp`: the false-positive rate on *real* sharp photographs, the go/no-go's second test: the held-out tiles
  themselves (they passed the sharpness filter, so the truth is "in focus"), no synthesis, the share of pixels the
  network (and the baseline) puts at `b >= 2 * b_min`, ungated and gated.
"""

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from training.bokeh import classical, sources
from training.bokeh.encode import B_MIN, encode_t, radius_and_confidence
from training.bokeh.model import BokehNet, macs_per_pixel
from training.bokeh.synth import SynthConfig, TileBatch, synthesize

DEFAULTS = {
    "seed": 1, "sample": 512, "batch": 24, "lr": 2e-3, "weight_decay": 0.01, "warmup": 300, "ema": 0.999,
    "max_minutes": None, "steps": None, "device": "cuda", "workers": 3, "amp": True, "gain_jitter_stops": 0.6,
    "eval_every_min": 12.0, "ckpt_every_min": 5.0, "val_samples": 192, "final_val_samples": 576, "baseline_n": 48,
    "kernel_r": 44, "r_max_px": 34.0, "cand_share": 0.34, "val_share": sources.VAL_SHARE,
}
TYPES = {"max_minutes": float, "steps": int}
GATE = 0.5
#: When a run is lost (module docstring, "Guards").
MAX_SKIPS_IN_A_ROW = 20
MAX_SKIP_SHARE = 0.02
SKIP_SHARE_AFTER = 200
EARLY_STEPS = (25, 100)
EARLY_MAX_LOSS = 5.0


# --- guards -------------------------------------------------------------------------------------------------------


class TrainingDiverged(RuntimeError):
    """The run is lost; stop it now instead of paying for the rest of it. The message starts "training diverged: "."""

    def __init__(self, why: str):
        # Idempotent: an exception that crosses a process boundary (Modal) is rebuilt from its message.
        super().__init__(why if why.startswith("training diverged: ") else f"training diverged: {why}")


def nonfinite(named: dict[str, torch.Tensor]) -> list[str]:
    """The names of the floating-point tensors that hold a NaN or an infinity (one device sync for all of them)."""
    named = {k: t for k, t in named.items() if t.is_floating_point()}
    if not named:
        return []
    good = torch.stack([torch.isfinite(t).all() for t in named.values()]).tolist()
    return [k for k, ok in zip(named, good) if not ok]


def require_finite(model: torch.nn.Module, what: str, step: int | None = None) -> None:
    """Raises `TrainingDiverged` when a weight or a batch-norm statistic of `model` is not finite."""
    bad = nonfinite(dict(model.state_dict()))
    if bad:
        at = "" if step is None else f" at step {step}"
        raise TrainingDiverged(f"the {what} are not finite{at} ({len(bad)} tensors, the first is {bad[0]})")


def check_scores(scores: dict, step: int) -> None:
    """Raises `TrainingDiverged` when an evaluation returned a NaN or an infinity."""
    bad = [k for k, v in scores.items() if isinstance(v, float) and not math.isfinite(v)]
    if bad:
        raise TrainingDiverged(f"the evaluation at step {step} returned a non-finite {', '.join(bad)}")


def check_early(losses: list[float]) -> None:
    """The run's first check: the mean loss of steps `EARLY_STEPS` must be finite and at most `EARLY_MAX_LOSS`."""
    mean = float(np.mean(losses)) if losses else float("nan")
    if not math.isfinite(mean) or mean > EARLY_MAX_LOSS:
        raise TrainingDiverged(f"the mean loss of steps {EARLY_STEPS[0]} to {EARLY_STEPS[1]} is {mean:.4g} "
                               f"(a healthy start is about 1, the limit is {EARLY_MAX_LOSS:g})")


class SkipCounter:
    """Counts the steps skipped for a non-finite value and says when there are too many (module docstring)."""

    def __init__(self, skipped: int = 0):
        self.total = skipped
        self.in_a_row = 0

    def applied(self) -> None:
        """A step went through."""
        self.in_a_row = 0

    def skipped(self, steps_taken: int) -> None:
        """A step was skipped; `steps_taken` optimiser steps were made before it. Raises `TrainingDiverged` when too many
        steps in a row, or too large a share of all, were skipped."""
        self.total += 1
        self.in_a_row += 1
        tried = steps_taken + self.total
        if self.in_a_row > MAX_SKIPS_IN_A_ROW:
            raise TrainingDiverged(f"{self.in_a_row} steps in a row were skipped for a non-finite value")
        if tried > SKIP_SHARE_AFTER and self.total > MAX_SKIP_SHARE * tried:
            raise TrainingDiverged(f"{self.total} of the {tried} steps so far were skipped for a non-finite value "
                                   f"(more than {MAX_SKIP_SHARE:.0%})")


# --- data ---------------------------------------------------------------------------------------------------------


class BatchStream(torch.utils.data.IterableDataset):
    """Batches of tiles for a DataLoader (`batch_size=None`). Training batches draw tiles at random (a third of them
    from the tiles that have an instance to cut out, so every batch has donors for sprites); validation batches walk
    the held-out tiles in a fixed order."""

    def __init__(self, folder: Path, batch: int, seed: int, split: str, cand_share: float, start: int = 0,
                 val_share: float = sources.VAL_SHARE):
        self.folder, self.batch, self.seed, self.split, self.cand_share, self.start = folder, batch, seed, split, cand_share, start
        self.val_share = val_share

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        ts = sources.TileSet(self.folder, self.val_share)
        pool = ts.train if self.split == "train" else ts.val
        if not pool:
            raise RuntimeError(f"no {self.split} tiles in {self.folder}")
        with_cands = [i for i in pool if ts.rows[i][1]["cands"]]
        rng = np.random.default_rng([self.seed, wid, self.start])
        k = wid
        while True:
            if self.split == "train":
                n_c = int(round(self.batch * self.cand_share)) if with_cands else 0
                idx = list(rng.choice(with_cands, n_c)) + list(rng.choice(pool, self.batch - n_c))
            else:
                idx = [pool[(k * self.batch + j) % len(pool)] for j in range(self.batch)]
            got = [ts.get(int(i)) for i in idx]
            yield TileBatch(torch.from_numpy(np.stack([g[0] for g in got])), torch.from_numpy(np.stack([g[1] for g in got])),
                            torch.from_numpy(np.stack([g[2] for g in got])), torch.tensor([g[3] for g in got], dtype=torch.float32))
            k += nw


def _identity(x):
    return x


def loader(folder: Path, batch: int, seed: int, split: str, workers: int, cand_share: float, start: int = 0,
           val_share: float = sources.VAL_SHARE):
    ds = BatchStream(folder, batch, seed, split, cand_share, start, val_share)
    return torch.utils.data.DataLoader(ds, batch_size=None, num_workers=workers, prefetch_factor=4 if workers else None,
                                       persistent_workers=bool(workers), collate_fn=_identity)


def synth_config(config: dict) -> SynthConfig:
    return SynthConfig(sample=config["sample"], kernel_r=config["kernel_r"], r_max_px=config["r_max_px"])


def half_targets(sy) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(log r, weight, r in pixels) at the network's output resolution (2 x 2 blocks). The weight is cut to 30 % where
    a block mixes radii more than a factor of e^0.35 apart."""
    logr = sy.logr
    avg = F.avg_pool2d(logr, 2)
    spread = F.max_pool2d(logr, 2) + F.max_pool2d(-logr, 2)
    w = F.avg_pool2d(sy.w, 2) * torch.where(spread < 0.35, 1.0, 0.3)
    return avg, w, torch.exp(avg)


def laplace_nll(mu: torch.Tensor, s: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    s = s.clamp(-4.0, 2.0)
    return (w * ((y - mu).abs() * torch.exp(-s) + s)).sum() / w.sum().clamp(min=1.0)


# --- metrics ------------------------------------------------------------------------------------------------------


class Acc:
    """Accumulates the validation numbers over batches (module docstring)."""

    def __init__(self):
        self.n = dict.fromkeys(("pos", "pos_g", "neg", "fp", "fp_g", "big", "hit", "hit_g"), 0)
        self.err = self.err_g = self.rel = 0.0
        self.beta: list[np.ndarray] = []
        self.logerr: list[np.ndarray] = []

    def add(self, b_hat, conf, b_true, w, beta=None, logerr=None) -> None:
        """All arguments are numpy arrays of one shape (the batch's output pixels), in half-diagonal units."""
        ok = w >= 0.6
        gate = conf >= GATE
        pos = ok & (b_true >= B_MIN)
        neg = ok & (b_true < B_MIN)
        big = ok & (b_true >= 2 * B_MIN)
        err = np.abs(b_hat - b_true)
        self.n["pos"] += int(pos.sum())
        self.n["pos_g"] += int((pos & gate).sum())
        self.err += float(err[pos].sum())
        self.err_g += float(err[pos & gate].sum())
        self.rel += float((err[pos] / b_true[pos]).sum())
        self.n["neg"] += int(neg.sum())
        fp = b_hat >= 2 * B_MIN
        self.n["fp"] += int((neg & fp).sum())
        self.n["fp_g"] += int((neg & fp & gate).sum())
        self.n["big"] += int(big.sum())
        self.n["hit"] += int((big & fp).sum())
        self.n["hit_g"] += int((big & fp & gate).sum())
        if beta is not None:
            self.beta.append(beta[ok][::13])
            self.logerr.append(logerr[ok][::13])

    def result(self) -> dict:
        n = self.n
        out = {
            "pixels": {k: v for k, v in n.items()},
            "mae_b": self.err / max(1, n["pos"]),
            "mae_b_gated": self.err_g / max(1, n["pos_g"]),
            "coverage": n["pos_g"] / max(1, n["pos"]),
            "rel_err": self.rel / max(1, n["pos"]),
            "fp_rate": n["fp"] / max(1, n["neg"]),
            "fp_rate_gated": n["fp_g"] / max(1, n["neg"]),
            "recall_2bmin": n["hit"] / max(1, n["big"]),
            "recall_2bmin_gated": n["hit_g"] / max(1, n["big"]),
        }
        if self.beta:
            beta, le = np.concatenate(self.beta), np.concatenate(self.logerr)
            order = np.argsort(beta)
            bins = np.array_split(order, 10)
            table = [{"beta": float(beta[b].mean()), "abs_log_err": float(le[b].mean())} for b in bins if len(b)]
            out["calibration"] = table
            out["ece"] = float(sum(abs(t["beta"] - t["abs_log_err"]) * len(b) for t, b in zip(table, bins)) / len(beta))
        return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in out.items()}


def _round(x):
    if isinstance(x, float):
        return round(x, 5)
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_round(v) for v in x]
    return x


def real_sharp(model: BokehNet, ts: sources.TileSet, pool: list[int], n: int, baseline_n: int) -> dict:
    """False positives on real sharp tiles (module docstring): shares of pixels at `b >= 2 b_min`. The network runs on the
    device its weights are on (a GPU in a training run); the tiles, the baseline and the numbers stay on the CPU."""
    from training.bokeh.encode import srgb_to_linear

    device = next(model.parameters()).device
    picks = pool[::max(1, len(pool) // max(1, n))][:n]
    cnt = {k: [0, 0, 0] for k in ("model", "classical")}  # pixels, predicted blurred, predicted blurred with confidence
    for j, i in enumerate(picks):
        rgb, _, _, hd = ts.get(i)
        e = encode_t(srgb_to_linear(torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255))  # on the CPU
        with torch.no_grad():
            r, _, conf = radius_and_confidence(model(e.to(device)))
        r, conf = r[0].cpu(), conf[0].cpu()
        maps = {"model": ((r / hd).numpy(), conf.numpy())}
        if j < baseline_n:
            rad, cf = classical.estimate(classical.gray_from_encoded(e[0].double().numpy()))
            h2, w2 = rad.shape[0] // 2, rad.shape[1] // 2
            half = lambda a: a[:h2 * 2, :w2 * 2].reshape(h2, 2, w2, 2).mean((1, 3))  # noqa: E731
            maps["classical"] = (half(rad) / hd, half(cf))
        for key, (b_hat, c) in maps.items():
            blurred = b_hat >= 2 * B_MIN
            cnt[key][0] += b_hat.size
            cnt[key][1] += int(blurred.sum())
            cnt[key][2] += int((blurred & (c >= GATE)).sum())
    return {"tiles": len(picks), **{k: {"fp_rate": v[1] / max(1, v[0]), "fp_rate_gated": v[2] / max(1, v[0])}
                                    for k, v in cnt.items() if v[0]}}


def evaluate(model: BokehNet, folder: Path, config: dict, n: int, seed: int, device, baseline_n: int = 0,
             sheet: Path | None = None) -> dict:
    """The validation numbers of `n` held-out samples (module docstring). The baseline runs on the first `baseline_n`."""
    from PIL import Image

    require_finite(model, "weights to evaluate")  # never score (or draw a sheet of) a network of NaNs
    scfg, batch = synth_config(config), config["batch"]
    ts = sources.TileSet(folder, config["val_share"])
    pool = ts.val or ts.train
    acc, acc_sub, acc_base = Acc(), Acc(), Acc()
    model.eval()
    done = 0
    k = 0
    with torch.no_grad():
        while done < n:
            idx = [pool[(k * batch + j) % len(pool)] for j in range(batch)]
            got = [ts.get(i) for i in idx]
            tb = TileBatch(torch.from_numpy(np.stack([g[0] for g in got])), torch.from_numpy(np.stack([g[1] for g in got])),
                           torch.from_numpy(np.stack([g[2] for g in got])), torch.tensor([g[3] for g in got], dtype=torch.float32))
            gen = torch.Generator(device=device).manual_seed(seed * 1_000_003 + k)
            sy = synthesize(tb.to(device), scfg, gen)
            e = encode_t(sy.lin)
            out = model(e)
            r_hat, beta, conf = radius_and_confidence(out)
            y2, w2, r2 = half_targets(sy)
            hd = sy.hd.view(-1, 1, 1)
            take = min(batch, n - done)
            sl = slice(0, take)
            b_hat, b_true = (r_hat[:, :, :] / hd)[sl].cpu().numpy(), (r2[:, 0] / hd)[sl].cpu().numpy()
            logerr = (torch.log(r_hat.clamp(min=1e-3)) - y2[:, 0]).abs()[sl].cpu().numpy()
            args = (b_hat, conf[sl].cpu().numpy(), b_true, w2[:, 0][sl].cpu().numpy())
            acc.add(*args, beta=beta[sl].cpu().numpy(), logerr=logerr)
            if done < baseline_n:
                m = min(take, baseline_n - done)
                acc_sub.add(*(a[:m] for a in args), beta=beta[:m].cpu().numpy(), logerr=logerr[:m])
                e_np = e[:m].double().cpu().numpy()
                for j in range(m):
                    rad, cf = classical.estimate(classical.gray_from_encoded(e_np[j]))
                    h2, w_ = rad.shape[0] // 2, rad.shape[1] // 2
                    rad2 = rad[:h2 * 2, :w_ * 2].reshape(h2, 2, w_, 2).mean((1, 3))
                    cf2 = cf[:h2 * 2, :w_ * 2].reshape(h2, 2, w_, 2).mean((1, 3))
                    acc_base.add((rad2 / hd[j].item())[None], cf2[None], b_true[j:j + 1], args[3][j:j + 1])
            if sheet is not None and k == 0:
                from training.bokeh.synth import colorize

                bmax = 0.05
                lab = colorize(r2 / hd.view(-1, 1, 1, 1), bmax)
                pred = colorize(r_hat.unsqueeze(1) / hd.view(-1, 1, 1, 1), bmax) * (0.25 + 0.75 * conf.unsqueeze(1))
                inp = F.avg_pool2d(e.clamp(0, 1), 2)
                rows = torch.cat([inp, lab, pred], 3)[:6]
                sheet.parent.mkdir(parents=True, exist_ok=True)
                img = (torch.cat(list(rows), 1).permute(1, 2, 0).clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
                Image.fromarray(img).save(sheet)
            done += take
            k += 1
    out = {"samples": done, "model": acc.result()}
    if baseline_n:
        out["real_sharp"] = real_sharp(model, ts, pool, min(48, len(pool)), min(16, baseline_n))
    if baseline_n:
        out["baseline_subset"] = {"samples": min(baseline_n, done), "model": acc_sub.result(),
                                  "classical": {kk: v for kk, v in acc_base.result().items() if kk not in ("calibration", "ece")}}
    return out


# --- the loop -----------------------------------------------------------------------------------------------------


def lr_at(config: dict, step: int, progress: float) -> float:
    warm = min(1.0, (step + 1) / config["warmup"])
    return config["lr"] * warm * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))))


def train(tiles: Path, out: Path, config: dict, heartbeat=None, log=print) -> dict:
    """Trains one model; writes `model.pt`, `config.json`, `metrics.json` and sample sheets into `out`; returns the
    metrics. Continues from `out/ckpt.pt` when it exists. Raises `TrainingDiverged` when the run is lost (module
    docstring, "Guards"); the log then ends with a `{"diverged": ...}` line."""
    try:
        return _train(tiles, out, config, heartbeat, log)
    except TrainingDiverged as e:
        log(json.dumps({"diverged": str(e)}))
        raise


def bad_rows(*tensors: torch.Tensor) -> list[int]:
    """Indices of the batch rows (first dimension) in which any of the tensors holds a NaN or an infinity."""
    return torch.stack([~torch.isfinite(t).flatten(1).all(1) for t in tensors]).any(0).nonzero().flatten().tolist()


def _train(tiles: Path, out: Path, config: dict, heartbeat, log) -> dict:
    config = {**DEFAULTS, **config}
    if not (config["max_minutes"] or config["steps"]):
        raise ValueError("give max_minutes or steps")
    torch.manual_seed(config["seed"])
    device = torch.device(config["device"])
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # the input size never changes
    out.mkdir(parents=True, exist_ok=True)
    model = BokehNet().to(device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": config["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=config["lr"], betas=(0.9, 0.99))
    step, elapsed_before, history, skipped_before = 0, 0.0, [], 0
    ckpt = out / "ckpt.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        opt.load_state_dict(state["opt"])
        step, elapsed_before, history = state["step"], state["elapsed"], state["history"]
        skipped_before = state.get("skipped", 0)
        require_finite(model, "weights in the checkpoint", step)  # a diverged run's checkpoint must not be resumed
        require_finite(ema, "EMA weights in the checkpoint", step)
        log(f"resumed at step {step} after {elapsed_before / 60:.1f} minutes")
    use_amp = bool(config["amp"]) and device.type == "cuda"
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    scfg = synth_config(config)
    stream = loader(tiles, config["batch"], config["seed"], "train", config["workers"], config["cand_share"], start=step,
                    val_share=config["val_share"])
    it = iter(stream)
    limit = config["max_minutes"] * 60 if config["max_minutes"] else None
    t0 = time.time()
    last_ckpt = last_eval = time.time()
    losses: list[float] = []
    early: list[float] = []
    skips = SkipCounter(skipped_before)
    stats = list(model.buffers())  # the batch-norm statistics (and their step counts)
    model.train()

    def elapsed() -> float:
        return elapsed_before + time.time() - t0

    def progress() -> float:
        p = 0.0
        if config["steps"]:
            p = step / config["steps"]
        if limit:
            p = max(p, elapsed() / limit)
        return p

    def save_ckpt() -> None:
        require_finite(model, "weights", step)  # a checkpoint of NaNs would be resumed from, and trained further
        require_finite(ema, "EMA weights", step)
        tmp = ckpt.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step,
                    "elapsed": elapsed(), "history": history, "config": config, "skipped": skips.total}, tmp)
        tmp.replace(ckpt)
        if heartbeat:
            heartbeat(elapsed())

    def skip(reason: str, saved: list[torch.Tensor], where: list[str], rows: list[int], draw: int) -> None:
        """Leaves the step undone: the batch-norm statistics the forward pass changed are put back, the gradients dropped.
        Logs it and counts it (`SkipCounter`); weights that are not finite end the run at once."""
        for b, old in zip(stats, saved):
            b.copy_(old)
        opt.zero_grad(set_to_none=True)
        log(json.dumps({"step": step, "skipped": reason, "where": where, "samples": rows, "draw": draw,
                        "skipped_total": skips.total + 1}))
        require_finite(model, "weights", step)
        skips.skipped(step)

    while progress() < 1.0:
        tb = next(it).to(device)
        draw = step + skips.total  # a skipped step moves the seed on too: the retry is another batch
        gen = torch.Generator(device=device).manual_seed(config["seed"] * 1_000_003 + 7_919 * 1_000 + draw)
        with torch.no_grad():
            sy = synthesize(tb, scfg, gen)
            jit = config["gain_jitter_stops"]
            gain = torch.exp2((torch.rand(tb.rgb.shape[0], generator=gen, device=device) * 2 - 1) * jit)
            e = encode_t(sy.lin, gain)
            y2, w2, _ = half_targets(sy)
        if device.type == "cuda":
            e = e.contiguous(memory_format=torch.channels_last)
        for g in opt.param_groups:
            g["lr"] = lr_at(config, step, progress())
        saved = [b.clone() for b in stats]
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            out_t = model(e)
        mu, s = out_t[:, :1].float(), out_t[:, 1:].float()
        loss = laplace_nll(mu, s, y2, w2)
        loss_v = loss.item()
        if not math.isfinite(loss_v):  # before the backward pass: nothing of this batch reaches the weights
            named = {"synthesis lin": sy.lin, "synthesis label radius": sy.logr, "synthesis weight": sy.w, "encoded input": e,
                     "target": y2, "target weight": w2, "output mu": mu, "output s": s}
            # (the outputs of a whole batch are NaN when one input is: batch norm shares its statistics, so name the inputs first)
            skip("non-finite loss", saved, nonfinite(named) or ["loss"], bad_rows(sy.lin, e, y2, w2) or bad_rows(mu, s), draw)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # The returned norm is the one before clipping. A NaN or infinite norm would make the clipping write NaN into every gradient.
        if not bool(torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))):
            skip("non-finite gradient", saved, ["gradient"], [], draw)
            continue
        opt.step()
        step += 1
        skips.applied()
        d = min(config["ema"], (1 + step) / (10 + step))
        with torch.no_grad():
            for pe, pm in zip(ema.state_dict().values(), model.state_dict().values()):
                if pe.dtype.is_floating_point:
                    pe.mul_(d).add_(pm.detach(), alpha=1 - d)
                else:
                    pe.copy_(pm)
        losses.append(loss_v)
        if EARLY_STEPS[0] <= step <= EARLY_STEPS[1]:
            early.append(loss_v)
        if step % 25 == 0:
            rec = {"step": step, "minutes": round(elapsed() / 60, 2), "loss": round(float(np.mean(losses)), 4),
                   "lr": round(opt.param_groups[0]["lr"], 6)}
            if skips.total:
                rec["skipped"] = skips.total
            history.append(rec)
            log(json.dumps(rec))
            losses = []
            require_finite(model, "weights", step)
        if step == EARLY_STEPS[1]:
            check_early(early)
        now = time.time()
        if now - last_ckpt > config["ckpt_every_min"] * 60:
            save_ckpt()
            last_ckpt = now
        if config["eval_every_min"] and now - last_eval > config["eval_every_min"] * 60:
            m = evaluate(ema, tiles, config, config["val_samples"], 7, device)
            history.append({"step": step, "eval": {k: m["model"][k] for k in ("mae_b", "fp_rate", "fp_rate_gated", "ece")}})
            log(json.dumps(history[-1]))
            check_scores(history[-1]["eval"], step)
            model.train()
            last_eval = time.time()
    save_ckpt()
    sheet = out / "samples" / "val-predictions.png"
    final = evaluate(ema, tiles, config, config["final_val_samples"], 11, device, baseline_n=config["baseline_n"], sheet=sheet)
    check_scores(final["model"], step)
    torch.save({"state_dict": ema.state_dict(), "config": config}, out / "model.pt")
    final_ts = sources.TileSet(tiles, config["val_share"])
    metrics = {
        "parameters": sum(p.numel() for p in ema.parameters()), "macs_per_pixel": round(macs_per_pixel(copy.deepcopy(ema).cpu().float())),
        "steps": step, "skipped_steps": skips.total, "minutes": round(elapsed() / 60, 2),
        "tiles": {"train": len(final_ts.train), "val": len(final_ts.val)},
        "final": _round(final), "history": history[-40:],
    }
    (out / "config.json").write_text(json.dumps(config, indent=1))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    log(json.dumps({k: v for k, v in metrics.items() if k != "history"}))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tiles", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            ap.add_argument(f"--{k.replace('_', '-')}", type=lambda x: x.lower() in ("1", "true", "yes"), default=v)
        else:
            ap.add_argument(f"--{k.replace('_', '-')}", type=type(v) if v is not None else TYPES[k], default=v)
    args = ap.parse_args()
    config = {k: getattr(args, k) for k in DEFAULTS}
    train(args.tiles, args.out, config)


if __name__ == "__main__":
    main()

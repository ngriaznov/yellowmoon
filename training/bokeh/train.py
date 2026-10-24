"""Trains BokehNet on synthetic defocus rendered on the fly (`synth.py`) from the tiles of `sources.py`.

    python -m training.bokeh.train --tiles <tiles folder> --out <run folder> --steps 30 --batch 4 --sample 192 \
        --device cpu --workers 0

**Loss.** Laplace negative log-likelihood of the log radius: for each pixel of the half-resolution output,
`w * (|y - mu| * exp(-s) + s)` with `y` the target `log(max(r, 1 px))` (average of a 2 x 2 block) and `w` the label's
weight (lowered at the ambiguous edges between layers, and where a block mixes very different radii). The network
therefore says how sure it is: a textureless region gets a wide scale instead of a confident wrong value. The
confidence the app uses is `c = exp(-exp(s) / 0.25)` (`encode.py`); that is how sure the *radius* is, and whether a pixel is out of
focus is `encode.detection_probability` (README, "The decision"). `decision_loss > 0` adds a cross-entropy of that probability
against the label `b >= 2 b_min` (`decision_nll`; no new output: it is a function of `mu` and `s`); it is off by default.

**The half-resolution target** (`half_targets`, v3). A 2 x 2 block whose pixels have radii more than a factor of `e^0.35` apart
is "mixed" (a thin line over a blur, the edge between a subject and its background). bokeh-2 trained such a block to
the *average* log radius at 30 % weight, so a one-pixel twig in a blurred background was learnt as a mild blur and the
output absorbed it. Now a mixed block takes the *smallest* radius among its firm pixels (label weight at least a half):
**the sharp one wins**, at 70 % weight. This is the rule the spec already has downstream (L1 never smears a sample that
is sharper than the pixel's own blur, and a false blur costs detail where a false sharp costs a little reshaping), and
it makes the map hold a thin sharp structure as sharp, at the price of a sharp region reaching one cell (2 px of the
1024-px frame) into the blur beside it. `sharp_wins: false` is the bokeh-2 rule.

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
- `groups` (v3): the numbers for what the new synthesis adds, with the runtime's decision (`P(b >= 2 b_min) >= 0.7`,
  `encode.detection_probability`) and the plain `b >= 2 b_min`: `thin_false_blur`, the share of blocks on a sharp thin
  structure over a blur that are called blurred; `flat_false_blur`, the same for the far interior of a sharp featureless
  layer; `mid` and `high` the recall of blurred blocks of `b` from `2 b_min` to 0.012 and above; `decision` the
  precision and recall of the decision over all firm blocks. With `legacy`, the validation of bokeh-2's recipe (the old synthesis and the old target), which
  puts this run's `model` next to bokeh-2's;
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
from training.bokeh.encode import B_DECIDE, B_MIN, DECISION_P, detection_probability, encode_t, radius_and_confidence
from training.bokeh.model import BokehNet, macs_per_pixel
from training.bokeh.synth import FLAT, THIN, SynthConfig, TileBatch, synthesize

DEFAULTS = {
    "seed": 1, "sample": 512, "batch": 24, "lr": 2e-3, "weight_decay": 0.01, "warmup": 300, "ema": 0.999,
    "max_minutes": None, "steps": None, "device": "cuda", "workers": 3, "amp": True, "gain_jitter_stops": 0.6,
    "eval_every_min": 12.0, "ckpt_every_min": 5.0, "val_samples": 192, "final_val_samples": 576, "baseline_n": 48,
    "kernel_r": 44, "r_max_px": 34.0, "cand_share": 0.34, "val_share": sources.VAL_SHARE,
    # v3: the recipe (3, or 2 for bokeh-2's), the half-resolution target rule, the weight of the extra decision loss (0 = off),
    # and the weights (a model.pt of an earlier run) to start from instead of a fresh network
    "synth": 3, "sharp_wins": True, "thin_boost": 2.0, "decision_loss": 0.0, "init_from": None,
}
TYPES = {"max_minutes": float, "steps": int, "init_from": str}
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


def synth_config(config: dict, legacy: bool = False) -> SynthConfig:
    """The synthesis the run trains on (`config["synth"]`: 3 is v3, 2 is bokeh-2's recipe); `legacy` forces bokeh-2's."""
    kw = {"sample": config["sample"], "kernel_r": config["kernel_r"], "r_max_px": config["r_max_px"]}
    return SynthConfig(**kw) if legacy or config["synth"] < 3 else SynthConfig.v3(**kw)


#: Blocks whose pixels' log radii are further apart than this are mixed.
MIXED_SPREAD = 0.35


def half_targets(sy, sharp_wins: bool = True, thin_boost: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(log r, weight, r in pixels) at the network's output resolution (2 x 2 blocks; module docstring, "The
    half-resolution target"). A pure block averages its log radius and takes the mean weight. A mixed block (radii more than
    `e^0.35` apart) takes the smallest log radius among its pixels of weight at least 0.5 at 70 % of the mean weight when
    `sharp_wins`, else (and when it has no such pixel) the average at 30 %: bokeh-2's rule. With `sharp_wins`, a block that
    holds a pixel of a thin structure (v3) counts `thin_boost` times: they are a few pixels in a hundred."""
    logr, w = sy.logr, sy.w
    avg = F.avg_pool2d(logr, 2)
    spread = F.max_pool2d(logr, 2) + F.max_pool2d(-logr, 2)
    wb = F.avg_pool2d(w, 2)
    mixed = spread >= MIXED_SPREAD
    if not sharp_wins:
        return avg, wb * torch.where(mixed, 0.3, 1.0), torch.exp(avg)
    firm = w >= 0.5
    smallest = -F.max_pool2d(-torch.where(firm, logr, torch.full_like(logr, 1e4)), 2)
    has_firm = F.max_pool2d(firm.float(), 2) > 0
    y = torch.where(mixed & has_firm, smallest, avg)
    weight = wb * torch.where(mixed, torch.where(has_firm, 0.7, 0.3), 1.0)
    if thin_boost != 1.0 and sy.cls is not None:
        weight = weight * torch.where(F.max_pool2d((sy.cls == THIN).float(), 2) > 0, thin_boost, 1.0)
    return y, weight, torch.exp(y)


def laplace_nll(mu: torch.Tensor, s: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    s = s.clamp(-4.0, 2.0)
    return (w * ((y - mu).abs() * torch.exp(-s) + s)).sum() / w.sum().clamp(min=1.0)


def decision_nll(mu: torch.Tensor, s: torch.Tensor, y: torch.Tensor, w: torch.Tensor, hd: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy of the runtime's decision `P(b >= 2 b_min)` (`encode.detection_probability`, a function of
    the two outputs only) against the block's label `b >= 2 b_min`. Optional (`decision_loss`): it trains the tails of the
    Laplace, which the log-likelihood fits only on average."""
    p = detection_probability(mu, s.clamp(-4.0, 2.0), hd.view(-1, 1, 1, 1)).clamp(1e-4, 1 - 1e-4)
    z = (y >= torch.log(B_DECIDE * hd.view(-1, 1, 1, 1))).float()
    return (w * -(z * torch.log(p) + (1 - z) * torch.log(1 - p))).sum() / w.sum().clamp(min=1.0)


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


class Groups:
    """The v3 numbers (module docstring, `groups`). Arrays are numpy of one shape (the batch's output blocks); `b_hat` and
    `b_true` are in half-diagonal units, `p` is `detection_probability`, `thin` and `flat` are the blocks of a sharp thin
    structure over a blur and of the far interior of a sharp featureless layer."""

    BANDS = (("mid", B_DECIDE, 0.012), ("high", 0.012, 1e9))

    def __init__(self):
        self.n: dict[str, int] = {}

    def _inc(self, key: str, v) -> None:
        self.n[key] = self.n.get(key, 0) + int(v)

    def add(self, b_hat, p, b_true, w, thin, flat) -> None:
        calls = {"raw": b_hat >= B_DECIDE, "p": p >= DECISION_P}
        ok = w >= 0.6
        truth = b_true >= B_DECIDE
        for name, call in calls.items():
            self._inc(f"thin_{name}", (thin & call).sum())
            self._inc(f"flat_{name}", (flat & call).sum())
            self._inc(f"tp_{name}", (ok & truth & call).sum())
            self._inc(f"fp_{name}", (ok & ~truth & call).sum())
            for band, lo, hi in self.BANDS:
                self._inc(f"{band}_{name}", (ok & (b_true >= lo) & (b_true < hi) & call).sum())
        self._inc("thin", thin.sum())
        self._inc("flat", flat.sum())
        self._inc("truth", (ok & truth).sum())
        for band, lo, hi in self.BANDS:
            self._inc(band, (ok & (b_true >= lo) & (b_true < hi)).sum())

    def result(self) -> dict:
        n, out = self.n, {}
        for name in ("raw", "p"):
            tag = "p70" if name == "p" else "raw"
            out[f"thin_false_blur_{tag}"] = n.get(f"thin_{name}", 0) / max(1, n.get("thin", 0))
            out[f"flat_false_blur_{tag}"] = n.get(f"flat_{name}", 0) / max(1, n.get("flat", 0))
            for band, _, _ in self.BANDS:
                out[f"{band}_recall_{tag}"] = n.get(f"{band}_{name}", 0) / max(1, n.get(band, 0))
            tp, fp = n.get(f"tp_{name}", 0), n.get(f"fp_{name}", 0)
            out[f"precision_{tag}"] = tp / max(1, tp + fp)
            out[f"recall_{tag}"] = tp / max(1, n.get("truth", 0))
        out["blocks"] = {k: n.get(k, 0) for k in ("thin", "flat", "mid", "high", "truth")}
        return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in out.items()}


def v3_masks(sy) -> tuple[torch.Tensor, torch.Tensor]:
    """The blocks (B, h, w) of a sharp thin structure over a blur (a firm pixel of a thin layer, label sharp, a label
    of 3 px or more within 4 px of it) and of the far interior of a sharp featureless layer (all four pixels: its weight
    was cut), for `Groups`. All false for the bokeh-2 synthesis."""
    if sy.cls is None:
        z = torch.zeros(sy.r.shape[0], sy.r.shape[2] // 2, sy.r.shape[3] // 2, dtype=torch.bool, device=sy.r.device)
        return z, z
    thin_px = (sy.cls == THIN) & (sy.r < 1.0) & (sy.w >= 0.9) & (F.max_pool2d(sy.r, 9, 1, 4) >= 3.0)
    flat_px = (sy.cls == FLAT) & (sy.r < 1.0) & (sy.w < 0.5)
    return (F.max_pool2d(thin_px.float(), 2) > 0)[:, 0], (F.avg_pool2d(flat_px.float(), 2) == 1)[:, 0]


def real_sharp(model: BokehNet, ts: sources.TileSet, pool: list[int], n: int, baseline_n: int) -> dict:
    """False positives on real sharp tiles (module docstring): shares of pixels at `b >= 2 b_min`. The network runs on the
    device its weights are on (a GPU in a training run); the tiles, the baseline and the numbers stay on the CPU."""
    from training.bokeh.encode import srgb_to_linear

    device = next(model.parameters()).device
    picks = pool[::max(1, len(pool) // max(1, n))][:n]
    cnt = {k: [0, 0, 0] for k in ("model", "classical")}  # pixels, predicted blurred, predicted blurred with confidence
    dec_hits = 0
    for j, i in enumerate(picks):
        rgb, _, _, hd = ts.get(i)
        e = encode_t(srgb_to_linear(torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255))  # on the CPU
        with torch.no_grad():
            raw = model(e.to(device))
            r, _, conf = radius_and_confidence(raw)
            p_dec = detection_probability(raw[:, 0], raw[:, 1], hd)[0].cpu()
        r, conf = r[0].cpu(), conf[0].cpu()
        maps = {"model": ((r / hd).numpy(), conf.numpy())}
        dec_hits += int((p_dec >= DECISION_P).sum())
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
    out = {"tiles": len(picks), **{k: {"fp_rate": v[1] / max(1, v[0]), "fp_rate_gated": v[2] / max(1, v[0])}
                                   for k, v in cnt.items() if v[0]}}
    if "model" in out:
        out["model"]["fp_rate_p70"] = dec_hits / max(1, cnt["model"][0])
    return out


def evaluate(model: BokehNet, folder: Path, config: dict, n: int, seed: int, device, baseline_n: int = 0,
             sheet: Path | None = None, legacy: bool = False) -> dict:
    """The validation numbers of `n` held-out samples (module docstring). The baseline runs on the first `baseline_n`.
    `legacy` scores on bokeh-2's recipe (its synthesis and its half-resolution target), the same samples bokeh-2's own
    final evaluation used."""
    from PIL import Image

    require_finite(model, "weights to evaluate")  # never score (or draw a sheet of) a network of NaNs
    scfg, batch = synth_config(config, legacy), config["batch"]
    sharp_wins = bool(config.get("sharp_wins", True)) and not legacy
    ts = sources.TileSet(folder, config["val_share"])
    pool = ts.val or ts.train
    acc, acc_sub, acc_base, groups = Acc(), Acc(), Acc(), Groups()
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
            y2, w2, r2 = half_targets(sy, sharp_wins)  # no boost: the numbers must not depend on how the loss weighs
            hd = sy.hd.view(-1, 1, 1)
            take = min(batch, n - done)
            sl = slice(0, take)
            b_hat, b_true = (r_hat[:, :, :] / hd)[sl].cpu().numpy(), (r2[:, 0] / hd)[sl].cpu().numpy()
            p_dec = detection_probability(out[:, 0], out[:, 1], hd)
            thin_blk, flat_blk = v3_masks(sy)
            groups.add(b_hat, p_dec[sl].cpu().numpy(), b_true, w2[:, 0][sl].cpu().numpy(), thin_blk[sl].cpu().numpy(), flat_blk[sl].cpu().numpy())
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
    out = {"samples": done, "model": acc.result(), "groups": groups.result()}
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
    elif config["init_from"]:
        # the weights (the EMA of an earlier run, `model.pt`) are the start; the optimiser, the schedule and the step count are new
        start = torch.load(config["init_from"], map_location=device, weights_only=False)["state_dict"]
        model.load_state_dict(start)
        ema.load_state_dict(start)
        require_finite(model, "weights to start from")
        log(f"started from the weights of {config['init_from']}")
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
            y2, w2, _ = half_targets(sy, config["sharp_wins"], config["thin_boost"])
        if device.type == "cuda":
            e = e.contiguous(memory_format=torch.channels_last)
        for g in opt.param_groups:
            g["lr"] = lr_at(config, step, progress())
        saved = [b.clone() for b in stats]
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            out_t = model(e)
        mu, s = out_t[:, :1].float(), out_t[:, 1:].float()
        loss = laplace_nll(mu, s, y2, w2)
        if config["decision_loss"]:
            loss = loss + config["decision_loss"] * decision_nll(mu, s, y2, w2, sy.hd)
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
            history[-1]["eval"].update({k: m["groups"][k] for k in ("thin_false_blur_p70", "flat_false_blur_p70", "mid_recall_p70",
                                                                    "precision_p70", "recall_p70")})
            log(json.dumps(history[-1]))
            check_scores(history[-1]["eval"], step)
            model.train()
            last_eval = time.time()
    save_ckpt()
    sheet = out / "samples" / "val-predictions.png"
    final = evaluate(ema, tiles, config, config["final_val_samples"], 11, device, baseline_n=config["baseline_n"], sheet=sheet)
    check_scores(final["model"], step)
    check_scores(final["groups"], step)
    if config["synth"] >= 3:  # bokeh-2's own validation (same tiles, seed and samples), to put this run's numbers next to its
        legacy = evaluate(ema, tiles, config, config["final_val_samples"], 11, device, legacy=True)
        check_scores(legacy["model"], step)
        final["legacy"] = {"model": legacy["model"], "groups": legacy["groups"]}
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

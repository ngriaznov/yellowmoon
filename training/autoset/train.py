"""Training of the ``autoset`` ensemble (docs/18 section 6.3) and its command line.

    python3 -m training.autoset.train --rows rows.jsonl --out data/models/autoset/autoset.json --model-version 1.0.0 [--metrics m.json]
    python3 -m training.autoset.train --rows rows.jsonl --kfold 5 --metrics m.json

Objective, per member: the Gaussian negative log-likelihood of each head against its label, with the label variance added to the
predicted variance (``exp(clip(logvar, +-20))``), the two heads summed, plus an L2 penalty on the weight matrices (not the biases).
Full-batch Adam (lr 1e-2, 3000 steps; written out here, optax is not a dependency). Member ``i`` is initialised from seed
``1000 + i``. Early stopping keeps the checkpoint with the lowest held-out NLL, evaluated every 25 steps, and stops after 40
evaluations without a new best; the held-out 15 % of scenes is chosen with a fixed seed, so a file and its augmentations never
split. Standardisation uses the training rows only, rounded to the float32 values the file stores, so the model trains with the
numbers it ships with. The model is rounded to float32 storage before it is evaluated.

``--kfold K`` is the evaluation: K folds by scene, each with its own ensemble, predictions pooled over the folds and summarised by
``evaluation.summarise``. Without ``--out`` the final model is not written. Everything is float64; JAX runs on the CPU.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from training.autoset import dataset as ds
from training.autoset import evaluation as ev
from training.autoset import model as md  # enables float64 in JAX on import
from training.autoset import rows as rw
from training.common import jsonfmt

Targets = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


@dataclass(frozen=True)
class TrainConfig:
    """The hyper-parameters of section 6.3 (defaults) and the seeds of the scene split and of the members."""

    members: int = md.N_MEMBERS
    steps: int = 3000
    lr: float = 1e-2
    weight_decay: float = 1e-3
    eval_every: int = 25
    patience: int = 40
    holdout: float = 0.15
    folds: int = 5
    seed: int = 20261018
    member_seed_base: int = 1000


@dataclass(frozen=True)
class MemberReport:
    """How one member trained: the steps run, the early-stopping checkpoint and the held-out NLL there (None without holdout)."""

    seed: int
    steps: int
    best_step: int
    best_holdout_nll: float | None
    final_loss: float


# ------------------------------------------------------------------------------------------------------------------
# The objective and Adam (JAX, float64)
# ------------------------------------------------------------------------------------------------------------------


def _head_nll(mu, logvar, y, label_var, mask):
    """Weighted mean Gaussian NLL of one head (``mask`` holds the row weights, 0 without a label); the label variance adds to the
    predicted one."""
    var = jnp.exp(jnp.clip(logvar, -md.LOGVAR_LIMIT, md.LOGVAR_LIMIT)) + label_var
    nll = 0.5 * (jnp.log(var) + (y - mu) ** 2 / var)
    return jnp.sum(mask * nll) / jnp.maximum(jnp.sum(mask), 1.0)


def _nll(params, z, targets):
    yk, vk, mk, yc, vc, mc = targets
    out = md.jax_forward(params, z)
    return _head_nll(out[:, 0], out[:, 1], yk, vk, mk) + _head_nll(out[:, 2], out[:, 3], yc, vc, mc)


def _adam_init(params):
    zeros_m = jax.tree_util.tree_map(jnp.zeros_like, params)
    zeros_v = jax.tree_util.tree_map(jnp.zeros_like, params)
    return zeros_m, zeros_v, jnp.zeros((), dtype=jnp.float64)


def _adam_step(params, grads, state, lr, b1=0.9, b2=0.999, eps=1e-8):
    m, v, t = state
    t = t + 1.0
    m = jax.tree_util.tree_map(lambda a, g: b1 * a + (1.0 - b1) * g, m, grads)
    v = jax.tree_util.tree_map(lambda a, g: b2 * a + (1.0 - b2) * g * g, v, grads)
    c1 = 1.0 - b1**t
    c2 = 1.0 - b2**t
    params = jax.tree_util.tree_map(lambda p, a, s: p - lr * (a / c1) / (jnp.sqrt(s / c2) + eps), params, m, v)
    return params, (m, v, t)


def _make_functions(weight_decay: float) -> tuple[Callable, Callable]:
    """The jitted full-batch Adam step (returns the loss including the penalty) and the held-out NLL."""

    def total(params, z, targets):
        penalty = sum(jnp.sum(w * w) for w, _ in params)
        return _nll(params, z, targets) + weight_decay * penalty

    @jax.jit
    def step(params, state, lr, z, targets):
        loss, grads = jax.value_and_grad(total)(params, z, targets)
        params, state = _adam_step(params, grads, state, lr)
        return params, state, loss

    val = jax.jit(_nll)
    return step, val


def init_member(seed: int) -> list[md.Layer]:
    """Weights of one member: normal with standard deviation ``1 / sqrt(fan_in)`` (0.1 of that for the output layer), zero biases."""
    rng = np.random.default_rng(seed)
    layers = []
    for li, (out_dim, in_dim) in enumerate(md.LAYER_SHAPES):
        scale = (0.1 if li == len(md.LAYER_SHAPES) - 1 else 1.0) / math.sqrt(in_dim)
        layers.append((rng.normal(0.0, scale, size=(out_dim, in_dim)), np.zeros(out_dim)))
    return layers


def train_member(seed: int, z_tr, t_tr: tuple, z_va, t_va, cfg: TrainConfig, step, val):
    """One member: full-batch Adam with early stopping on the held-out NLL (no holdout: the last step). Returns numpy layers."""
    params = [(jnp.asarray(w), jnp.asarray(b)) for w, b in init_member(seed)]
    state = _adam_init(params)
    best_params, best_nll, best_step, stale = params, math.inf, 0, 0
    loss = math.nan
    it = 0
    for it in range(1, cfg.steps + 1):
        params, state, loss = step(params, state, cfg.lr, z_tr, t_tr)
        if it % cfg.eval_every and it != cfg.steps:
            continue
        if z_va is None:
            continue
        nll = float(val(params, z_va, t_va))
        if nll < best_nll:
            best_nll, best_params, best_step, stale = nll, params, it, 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    if z_va is None:
        best_params, best_step = params, it
    arrays = [(np.asarray(w, dtype=np.float64), np.asarray(b, dtype=np.float64)) for w, b in best_params]
    report = MemberReport(seed, it, best_step, None if z_va is None else best_nll, float(loss))
    return arrays, report


def scene_weights(scene: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Per-row weights that give every scene the same total weight among the rows with a label (0 for the others).

    A scene with nine near-identical files (one body's raw modes) or with its augmentations counts as much as a scene of one
    file: the loss is a mean over scenes, not over rows (docs/18 section 6.3).
    """
    w = np.zeros(len(scene), dtype=np.float64)
    if not ok.any():
        return w
    labels, counts = np.unique(scene[ok], return_counts=True)
    per = dict(zip(labels.tolist(), counts.tolist()))
    w[ok] = [1.0 / per[s] for s in scene[ok].tolist()]
    return w


def _targets(prep: ds.Prepared, idx: np.ndarray) -> Targets:
    """Label arrays for the rows ``idx``; rows without a label carry 0 (and variance 1) with weight 0. The weights balance the
    scenes (:func:`scene_weights`); the loss divides by their sum."""
    ok_k = prep.key_ok[idx]
    ok_c = prep.con_ok[idx]
    scene = prep.scene[idx]
    return (
        np.where(ok_k, prep.key_target[idx], 0.0), np.where(ok_k, prep.key_var[idx], 1.0), scene_weights(scene, ok_k),
        np.where(ok_c, prep.contrast[idx], 0.0), np.where(ok_c, prep.contrast_var[idx], 1.0), scene_weights(scene, ok_c),
    )


def train_ensemble(prep: ds.Prepared, train_idx: np.ndarray, holdout_idx: np.ndarray | None, cfg: TrainConfig, *,
                   model_version: str, log: Callable[[str], None] = print) -> tuple[md.AutosetModel, list[MemberReport]]:
    """Trains ``cfg.members`` members on the labelled rows ``train_idx``; early stopping on ``holdout_idx`` when it has rows."""
    train_idx = train_idx[(prep.key_ok | prep.con_ok)[train_idx]]
    if len(train_idx) == 0:
        raise ValueError("no labelled training rows")
    X = prep.X[train_idx]
    # the statistics are stored as float32 (section 4.3): train with those numbers, so that the file is what was trained
    mean = np.asarray(X.mean(axis=0), dtype=np.float32).astype(np.float64)
    std = np.asarray(np.maximum(X.std(axis=0), md.STD_FLOOR), dtype=np.float32).astype(np.float64)
    z_tr = jnp.asarray((X - mean) / std)
    t_tr = tuple(jnp.asarray(a) for a in _targets(prep, train_idx))
    z_va = t_va = None
    if holdout_idx is not None and len(holdout_idx):
        holdout_idx = holdout_idx[(prep.key_ok | prep.con_ok)[holdout_idx]]
        if len(holdout_idx):
            z_va = jnp.asarray((prep.X[holdout_idx] - mean) / std)
            t_va = tuple(jnp.asarray(a) for a in _targets(prep, holdout_idx))
    step, val = _make_functions(cfg.weight_decay)
    members, reports = [], []
    for i in range(cfg.members):
        started = time.time()
        arrays, report = train_member(cfg.member_seed_base + i, z_tr, t_tr, z_va, t_va, cfg, step, val)
        members.append(arrays)
        reports.append(report)
        held = "" if report.best_holdout_nll is None else f" held-out NLL {report.best_holdout_nll:.4f}"
        log(f"  {model_version}: member {i} (seed {report.seed}) best step {report.best_step} of {report.steps}{held}"
            f" ({time.time() - started:.1f} s)")
    model = md.AutosetModel(model_version=model_version, input_mean=mean, input_std=std, members=members).rounded()
    return model, reports


# ------------------------------------------------------------------------------------------------------------------
# The two modes
# ------------------------------------------------------------------------------------------------------------------


def run_kfold(prep: ds.Prepared, cfg: TrainConfig, log: Callable[[str], None] = print) -> dict:
    """K folds by scene: a fresh ensemble per fold, predictions for the held-out fold, pooled over the folds."""
    n = prep.n
    pred = {name: np.full(n, np.nan) for name in ("key_mean", "key_var", "contrast_mean", "contrast_var")}
    evaluated = np.zeros(n, dtype=bool)
    folds = []
    scenes = ds.labelled_scenes(prep)
    for f, (train_scenes, test_scenes) in enumerate(ds.kfold_splits(scenes, cfg.folds, cfg.seed)):
        fit_scenes, hold_scenes = ds.holdout_split(train_scenes, cfg.holdout, cfg.seed)
        model, reports = train_ensemble(prep, ds.rows_of(prep, fit_scenes), ds.rows_of(prep, hold_scenes), cfg,
                                        model_version=f"kfold-{f}", log=log)
        test_rows = ds.rows_of(prep, test_scenes)
        p = model.predict(prep.X[test_rows])
        pred["key_mean"][test_rows] = p.key_mean
        pred["key_var"][test_rows] = p.key_var
        pred["contrast_mean"][test_rows] = p.contrast_mean
        pred["contrast_var"][test_rows] = p.contrast_var
        evaluated[test_rows] = True
        folds.append({"fold": f, "train_scenes": len(fit_scenes), "holdout_scenes": len(hold_scenes),
                      "test_scenes": len(test_scenes), "members": [asdict(r) for r in reports]})
        log(f"fold {f}: {len(test_scenes)} test scenes, {len(test_rows)} rows")
    return {"pred": pred, "evaluated": evaluated, "folds": folds}


def run_final(prep: ds.Prepared, cfg: TrainConfig, *, model_version: str, log: Callable[[str], None] = print) -> tuple:
    """The shipped model: trained on 85 % of the scenes, early stopping on the other 15 %."""
    scenes = ds.labelled_scenes(prep)
    fit_scenes, hold_scenes = ds.holdout_split(scenes, cfg.holdout, cfg.seed)
    model, reports = train_ensemble(prep, ds.rows_of(prep, fit_scenes), ds.rows_of(prep, hold_scenes), cfg,
                                    model_version=model_version, log=log)
    p = model.predict(prep.X)
    pred = {"key_mean": p.key_mean, "key_var": p.key_var, "contrast_mean": p.contrast_mean, "contrast_var": p.contrast_var}
    evaluated = np.zeros(prep.n, dtype=bool)
    evaluated[ds.rows_of(prep, hold_scenes)] = True
    summary = {"train_scenes": len(fit_scenes), "holdout_scenes": len(hold_scenes), "members": [asdict(r) for r in reports]}
    return model, pred, evaluated, summary


# ------------------------------------------------------------------------------------------------------------------
# Command line
# ------------------------------------------------------------------------------------------------------------------


def _describe(read: rw.ReadResult, prep: ds.Prepared) -> str:
    lines = [f"rows read {read.lines}, usable {len(read.rows)}, rejected {sum(read.rejected.values())} {read.rejected}"]
    lines.append("counts " + ", ".join(f"{k} {v}" for k, v in prep.counts.items()))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=Path, required=True, help="the dataset rows (JSON lines, docs/18 section 6)")
    ap.add_argument("--out", type=Path, help="write the model file here (not written when omitted)")
    ap.add_argument("--model-version", default="1.0.0")
    ap.add_argument("--metrics", type=Path, help="write the metrics JSON here")
    ap.add_argument("--kfold", type=int, default=0, help="run the k-fold evaluation by scene with K folds")
    ap.add_argument("--members", type=int, default=md.N_MEMBERS)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--holdout", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=20261018, help="the seed of the scene split")
    args = ap.parse_args(argv)
    if args.out is None and not args.kfold:
        ap.error("nothing to do: give --out (train the model) and/or --kfold K (evaluate)")
    cfg = TrainConfig(members=args.members, steps=args.steps, lr=args.lr, weight_decay=args.weight_decay,
                      holdout=args.holdout, folds=args.kfold or TrainConfig.folds, seed=args.seed)

    read = rw.read_rows(args.rows)
    prep = ds.prepare(read)
    print(_describe(read, prep), flush=True)
    doc: dict = {"config": asdict(cfg), "rows": {"file": str(args.rows.name), "lines": read.lines,
                                                  "usable": len(read.rows), "rejected": read.rejected,
                                                  "examples": read.examples},
                 "data": prep.counts, "sources": ev.source_summary(prep)}
    if args.kfold:
        result = run_kfold(prep, cfg)
        doc["evaluation"] = ev.summarise(prep, result["pred"], result["evaluated"], mode="kfold", folds=result["folds"])
    if args.out is not None:
        model, pred, evaluated, summary = run_final(prep, cfg, model_version=args.model_version)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(jsonfmt.dumps(model.to_json()), encoding="utf-8")
        print(f"wrote {args.out} ({model.param_count_per_member()} parameters per member, {len(model.members)} members, "
              f"sha256 {model.payload_sha256()})", flush=True)
        final = ev.summarise(prep, pred, evaluated, mode="final", folds=None)
        final["final_model"] = summary
        final["model"] = {"model_version": model.model_version, "sha256": model.payload_sha256(),
                          "parameters_per_member": model.param_count_per_member(), "members": len(model.members)}
        if "evaluation" not in doc:
            doc["evaluation"] = final
        doc["final_model"] = final
    if args.metrics is not None:
        ev.write_metrics(args.metrics, doc)
        print(f"wrote {args.metrics}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

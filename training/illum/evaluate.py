"""Metrics of the illuminant estimator on a dataset of synthetic scenes (angular error, classes, calibration)."""

from __future__ import annotations

import numpy as np

from training.illum import baselines as bl
from training.illum import dataset as dsm
from training.illum import model as md

GATE_MEDIAN_DEG = 2.0


def run(model: md.IllumModel, ds: dsm.Dataset, use_prior: bool = True, use_bv: bool = True, batch: int = 512) -> dict:
    """Model outputs for every scene of ``ds`` (arrays over scenes)."""
    outs = []
    for s in range(0, len(ds), batch):
        sl = slice(s, s + batch)
        a = ds.as_shot_uv[sl] if use_prior else None
        b = ds.bv[sl] if use_bv else None
        outs.append(model.estimate_from_hist(ds.hist[sl].astype(np.float64), ds.valid_fraction[sl], a, b))
    return {k: np.concatenate([o[k] for o in outs]) for k in outs[0] if k not in ("scores", "features")}


def angular(res: dict, ds: dsm.Dataset, key: str = "uv") -> np.ndarray:
    return bl.angular_error_uv_deg(res[key], ds.truth_uv)


def summarize(e: np.ndarray) -> dict:
    e = np.asarray(e)
    return {"n": int(e.size), "median": float(np.median(e)), "mean": float(np.mean(e)),
            "p90": float(np.percentile(e, 90)), "p95": float(np.percentile(e, 95)), "max": float(np.max(e))}


def by_group(e: np.ndarray, groups: np.ndarray) -> dict:
    return {str(g): summarize(e[groups == g]) for g in np.unique(groups)}


def class_metrics(probs: np.ndarray, klass: np.ndarray) -> dict:
    pred = probs.argmax(-1)
    acc = float((pred == klass).mean())
    nll = float(-np.log(np.maximum(probs[np.arange(len(klass)), klass], 1e-12)).mean())
    k = probs.shape[-1]
    conf = np.zeros((k, k), dtype=int)
    np.add.at(conf, (klass, pred), 1)
    # four profile classes: daylight+shade, tungsten, fluorescent, led (mixed excluded)
    fold = np.array([0, 0, 1, 2, 3, 4])
    p4 = np.zeros((len(klass), 5))
    for i in range(k):
        p4[:, fold[i]] += probs[:, i]
    k4 = fold[klass]
    m = klass != 5
    acc4 = float((p4[m, :4].argmax(-1) == k4[m]).mean())
    return {"accuracy": acc, "nll": nll, "accuracy_4class_no_mixed": acc4, "confusion": conf.tolist(),
            "per_class_recall": [float(conf[i, i] / max(conf[i].sum(), 1)) for i in range(k)]}


def auc(score: np.ndarray, positive: np.ndarray) -> float:
    """Area under the ROC curve of ``score`` for the boolean ``positive`` (Mann-Whitney, ties by rank order)."""
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(len(score))
    n_pos = int(positive.sum())
    n_neg = len(score) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[positive].sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg))


def calibration(conf: np.ndarray, err_deg: np.ndarray, bins: int = 5) -> dict:
    """Reliability of the confidence as ``P(error < 2 degrees)``, with its Brier score, the Brier score of the base rate and the AUC."""
    hit = err_deg < GATE_MEDIAN_DEG
    edges = np.quantile(conf, np.linspace(0, 1, bins + 1))
    rows = []
    for i in range(bins):
        m = (conf >= edges[i]) & (conf <= edges[i + 1] if i == bins - 1 else conf < edges[i + 1])
        if m.any():
            rows.append({"mean_confidence": float(conf[m].mean()), "fraction_within_2deg": float(hit[m].mean()), "n": int(m.sum())})
    brier = float(((conf - hit) ** 2).mean())
    rate = float(hit.mean())
    return {"bins": rows, "brier": brier, "brier_base_rate": rate * (1 - rate), "auc": auc(conf, hit),
            "mean_confidence": float(conf.mean()), "fraction_within_2deg": rate}


def full_report(model: md.IllumModel, ds: dsm.Dataset) -> dict:
    """Angular errors with and without the as-shot prior, per class and per light family, classes, calibration."""
    out = {}
    for name, use_prior in (("with_prior", True), ("histogram_only", False)):
        res = run(model, ds, use_prior=use_prior)
        e = angular(res, ds)
        out[name] = {"all": summarize(e), "by_class": by_group(e, np.array(md.CLASSES)[ds.klass]),
                     "by_family": by_group(e, ds.family), "by_as_shot": by_group(e, ds.as_shot_kind)}
        out[name]["calibration"] = calibration(res["confidence"], e)
        if use_prior:
            out[name]["classes"] = class_metrics(res["probs"], ds.klass)
    return out

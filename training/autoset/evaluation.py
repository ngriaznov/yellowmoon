"""Metrics of ``autoset`` (docs/18 sections 6.3 and 8), for the k-fold evaluation and for the held-out scenes of the final model.

Errors are signed as ``model - label`` and summarised as absolute values. The EV model is ``key_target - key`` (the mean of the
exposure head minus the file's key), its label is ``level = key_target - key`` as well, so a file's EV error is the error of its
key target. Reported:

* ``ev.heldout_files`` and ``ev.normal_files`` (gate 3: normal scenes, |EV - L| median and the share within 0.5 EV);
  ``ev.constant_*`` is the constant rule ``EV = 1.11`` on the same rows;
* ``augmentation.rescue_vs_label`` (gate 4: augmentations, 0.3 EV) and ``augmentation.equivariance`` (gate 2:
  ``|EV_model(aug) - (EV_model(base) - k)|``, 0.05 EV), per ``k`` as well;
* ``contrast`` (the clamped mean of the contrast head against the contrast label, and the constant 0 of the rule before autoset);
* ``calibration`` (the share of labels within one and two predictive standard deviations);
* ``failures`` (the ten largest EV errors of held-out files, for the model card).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from training.autoset import dataset as ds
from training.autoset import model as md
from training.common import jsonfmt


def _tag(tol: float) -> str:
    return f"{tol:g}".replace(".", "p").replace("-", "m")


def stats(err: np.ndarray, tolerances: tuple[float, ...] = (0.3, 0.5)) -> dict[str, Any]:
    """Count, mean absolute error, median, 90th percentile, maximum, bias (mean signed error) and the share within each tolerance."""
    err = np.asarray(err, dtype=np.float64)
    err = err[np.isfinite(err)]
    out: dict[str, Any] = {"n": int(err.size)}
    if err.size == 0:
        return out
    a = np.abs(err)
    out.update({
        "mae": float(a.mean()),
        "median_abs": float(np.median(a)),
        "p90_abs": float(np.percentile(a, 90)),
        "max_abs": float(a.max()),
        "bias": float(err.mean()),
    })
    for tol in tolerances:
        out[f"within_{_tag(tol)}"] = float(np.mean(a <= tol))
    return out


def equivariance_errors(key_mean: np.ndarray, prep: ds.Prepared, aug: np.ndarray) -> np.ndarray:
    """``EV_model(aug) - (EV_model(base) - k)`` for the augmentation rows ``aug``; zero when the model ignores the scaling."""
    ev_model = np.asarray(key_mean, dtype=np.float64) - prep.key
    return ev_model[aug] - (ev_model[prep.base[aug]] - prep.aug_k[aug])


def _mean_sd(x: np.ndarray) -> dict[str, float | None]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": None, "sd": None}
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)) if x.size > 1 else 0.0}


def summarise(prep: ds.Prepared, pred: dict[str, np.ndarray], evaluated: np.ndarray, *, mode: str,
              folds: list | None = None) -> dict[str, Any]:
    """The metrics of the rows marked ``evaluated`` (``pred`` holds the ensemble's combined outputs for every row)."""
    ev_model = np.asarray(pred["key_mean"], dtype=np.float64) - prep.key
    level = prep.level
    base = ~prep.is_aug
    key_rows = evaluated & prep.key_ok
    files = np.nonzero(key_rows & base)[0]
    normal = files[prep.normal[files]]
    constant = md.CAMERA_GAIN_EV
    ev_block = {
        "heldout_files": stats(ev_model[files] - level[files]),
        "normal_files": stats(ev_model[normal] - level[normal]),
        "constant_heldout_files": stats(constant - level[files]),
        "constant_normal_files": stats(constant - level[normal]),
    }

    aug = np.nonzero(key_rows & prep.is_aug)[0]
    rescue = ev_model[aug] - level[aug]
    equiv = equivariance_errors(pred["key_mean"], prep, aug)
    by_k = {}
    for k in sorted(set(prep.aug_k[aug].tolist())):
        sel = prep.aug_k[aug] == k
        by_k[f"{k:g}"] = {
            "rescue_vs_label": stats(rescue[sel], tolerances=(0.3,)),
            "equivariance": stats(equiv[sel], tolerances=(0.05,)),
        }
    augmentation = {
        "rows": int(len(aug)),
        "rescue_vs_label": stats(rescue, tolerances=(0.3,)),
        "equivariance": stats(equiv, tolerances=(0.05,)),
        "by_k": by_k,
    }

    crows = np.nonzero(evaluated & prep.con_ok & base)[0]
    mu = np.clip(np.asarray(pred["contrast_mean"], dtype=np.float64)[crows], -md.CONTRAST_LIMIT, md.CONTRAST_LIMIT)
    label = prep.contrast[crows]
    is_normal = prep.normal[crows]
    tol = (5.0, 10.0)
    contrast = {
        "rows": int(len(crows)),
        "model_vs_label": stats(mu - label, tol),
        "constant_zero_vs_label": stats(-label, tol),
        "normal_model_vs_label": stats(mu[is_normal] - label[is_normal], tol),
        "normal_constant_zero_vs_label": stats(-label[is_normal], tol),
    }

    calibration = {}
    if len(files):
        resid = prep.key_target[files] - np.asarray(pred["key_mean"], dtype=np.float64)[files]
        sd = np.sqrt(np.asarray(pred["key_var"], dtype=np.float64)[files] + prep.key_var[files])
        calibration["key_within_1_sd"] = float(np.mean(np.abs(resid) <= sd))
        calibration["key_within_2_sd"] = float(np.mean(np.abs(resid) <= 2.0 * sd))
    if len(crows):
        resid_c = prep.contrast[crows] - np.asarray(pred["contrast_mean"], dtype=np.float64)[crows]
        sd_c = np.sqrt(np.asarray(pred["contrast_var"], dtype=np.float64)[crows] + prep.contrast_var[crows])
        calibration["contrast_within_1_sd"] = float(np.mean(np.abs(resid_c) <= sd_c))
        calibration["contrast_within_2_sd"] = float(np.mean(np.abs(resid_c) <= 2.0 * sd_c))

    failures = []
    if len(files):
        err = ev_model[files] - level[files]
        for i in files[np.argsort(-np.abs(err), kind="stable")[:10]]:
            failures.append({
                "file": prep.file[i], "body": prep.body[i], "level_ev": float(level[i]),
                "ev_model": float(ev_model[i]), "error_ev": float(ev_model[i] - level[i]),
                "flagged": bool(prep.flagged[i]), "normal": bool(prep.normal[i]),
            })

    labels = {
        "level_ev": _mean_sd(level[files]),
        "contrast": _mean_sd(prep.contrast[crows]),
        "flagged_share": float(prep.flagged[files].mean()) if len(files) else None,
        "normal_share": float(prep.normal[files].mean()) if len(files) else None,
    }
    return {
        "mode": mode,
        "evaluated_files": int(len(files)),
        "ev": ev_block,
        "augmentation": augmentation,
        "contrast": contrast,
        "calibration": calibration,
        "label_summary": labels,
        "failures": failures,
        "folds": folds,
    }


def source_summary(prep: ds.Prepared) -> list[dict[str, Any]]:
    """Files, scenes and bodies per (source, licence) of the rows, for the model card."""
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for i in np.nonzero(~prep.is_aug)[0]:
        key = (prep.source[i] or "unknown", prep.licence[i] or "unknown")
        g = groups.setdefault(key, {"files": 0, "scenes": set(), "bodies": set()})
        g["files"] += 1
        g["scenes"].add(int(prep.scene[i]))
        if prep.body[i]:
            g["bodies"].add(prep.body[i])
    return [{"source": s, "licence": lic, "files": g["files"], "scenes": len(g["scenes"]), "bodies": sorted(g["bodies"])}
            for (s, lic), g in sorted(groups.items())]


def _plain(x: Any) -> Any:
    """Python values for the JSON writer: numpy scalars and arrays become Python ones, NaN and infinity become null."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return _plain(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        f = float(x)
        return f if math.isfinite(f) else None
    if x is None or isinstance(x, str):
        return x
    return str(x)


def write_metrics(path: Path, doc: dict) -> None:
    """Writes the metrics document as deterministic JSON (``training.common.jsonfmt``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(jsonfmt.dumps(_plain(doc)), encoding="utf-8")

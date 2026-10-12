"""Study (review of M1-06): what a robust as-shot prior would change. Not part of the shipped estimator.

    python3 -m training.illum.robust_prior_study [--profile sony-ilce-7rm4 ...] [--n 3000]

``docs/02`` fixes the as-shot prior as one Gaussian with sigma 0.05 on ``(u, v)``. That is right when the camera's automatic white
balance is right, and wrong when the photographer shot with a preset or a Kelvin value that has nothing to do with the light:
the shipped model then follows the preset (``training.illum.report``: on the 11 % of synthetic scenes with an unrelated preset the
median error is 5.9 degrees with the prior and 1.9 degrees without it, on the same scenes).

This script applies, to the *shipped* filter and without retraining anything, the fusion of a two-component prior,

    p(a | light) = (1 - eps) N(light, sigma_p^2 I) + eps N(light, sigma_b^2 I)        (sigma_p = 0.05 as in the spec)

so that the posterior weight of the narrow component falls when the histogram estimate is far from the as-shot value (a Mahalanobis
gate); the fused mean is the mixture of the two Gaussian fusions. ``eps`` and ``sigma_b`` are chosen on the validation scenes
(held-out lights and patches), then reported on the test scenes. The shipped numbers (``sigma_p`` only) come first.

Result and the decision it asks for: ``docs/02`` (As built, M1-06 review). The mixture is a change of the prior's form, not of its
sigma, and it moves the shipped class and confidence heads' inputs (they were trained on the single-Gaussian fusion), so it is a
decision for the orchestrator and Mykyta, not something to ship from a review.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from training.common import paths
from training.illum import baselines as bl
from training.illum import dataset as dsm
from training.illum import model as md

GRID_EPS = (0.05, 0.1, 0.2, 0.3)
GRID_SIGMA_B = (0.2, 0.3, 0.5)
KINDS = ("auto_good", "auto_off", "auto_bad", "preset")


def _fuse_gaussian(mu: np.ndarray, cov: np.ndarray, a: np.ndarray, sigma: float) -> np.ndarray:
    lam = np.linalg.inv(cov)
    post = np.linalg.inv(lam + np.eye(2) / sigma**2)
    return np.einsum("nij,nj->ni", post, np.einsum("nij,nj->ni", lam, mu) + a / sigma**2)


def _log_normal(d: np.ndarray, cov: np.ndarray) -> np.ndarray:
    q = np.einsum("ni,nij,nj->n", d, np.linalg.inv(cov), d)
    return -0.5 * q - 0.5 * np.log(np.linalg.det(cov)) - np.log(2.0 * np.pi)


def fuse_mixture(mu: np.ndarray, cov: np.ndarray, a: np.ndarray, eps: float, sigma_b: float, sigma_p: float = md.PRIOR_SIGMA) -> np.ndarray:
    """Mean of the two-component fusion. ``eps = 0`` is the shipped single Gaussian."""
    narrow = _fuse_gaussian(mu, cov, a, sigma_p)
    if eps <= 0.0:
        return narrow
    eye = np.eye(2)[None]
    d = mu - a
    l_narrow = np.log(1.0 - eps) + _log_normal(d, cov + sigma_p**2 * eye)
    l_broad = np.log(eps) + _log_normal(d, cov + sigma_b**2 * eye)
    w = 1.0 / (1.0 + np.exp(l_broad - l_narrow))
    return w[:, None] * narrow + (1.0 - w)[:, None] * _fuse_gaussian(mu, cov, a, sigma_b)


def _arrays(model: md.IllumModel, ds: dsm.Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu, cov, _ = model.gaussian_fit(model.filter_scores(ds.hist.astype(np.float64)))
    has = ~np.isnan(ds.as_shot_uv[:, 0])
    return mu, cov, np.where(has[:, None], ds.as_shot_uv, 0.0), has


def errors(arrays: tuple, ds: dsm.Dataset, eps: float, sigma_b: float) -> np.ndarray:
    """Angular errors of the fused estimate for ``arrays = _arrays(model, ds)`` (the filter runs once per dataset)."""
    mu, cov, a, has = arrays
    uv = np.where(has[:, None], fuse_mixture(mu, cov, a, eps, sigma_b), mu)
    return bl.angular_error_uv_deg(uv, ds.truth_uv)


def summary(e: np.ndarray, ds: dsm.Dataset) -> str:
    s = f"median {np.median(e):.2f}  p90 {np.percentile(e, 90):.2f}  p95 {np.percentile(e, 95):.2f}  max {e.max():.1f} |"
    for kind in KINDS:
        k = ds.as_shot_kind == kind
        if k.any():
            s += f" {kind} {np.median(e[k]):.2f}/{np.percentile(e[k], 90):.1f}"
    return s


def run(profile_id: str, n: int, models_dir: Path) -> dict:
    model = md.IllumModel.from_json(json.loads((models_dir / f"{profile_id}.json").read_text()))
    val = dsm.build(profile_id, "val", n, model.spec, procs=2, cache=False, verbose=False)
    test = dsm.build(profile_id, "test", n, model.spec, procs=2, cache=False, verbose=False)
    a_val, a_test = _arrays(model, val), _arrays(model, test)
    # the pick: the smallest mean of p90 and p95 on the validation scenes
    best = min(((e, s) for e in GRID_EPS for s in GRID_SIGMA_B),
               key=lambda p: np.mean(np.percentile(errors(a_val, val, *p), (90, 95))))
    print(f"{profile_id}: {n} validation and {n} test scenes; picked on validation: eps {best[0]}, sigma_b {best[1]}")
    out = {"profile": profile_id, "eps": best[0], "sigma_b": best[1]}
    rows = (("histogram only", bl.angular_error_uv_deg(a_test[0], test.truth_uv)),
            ("shipped: Gaussian sigma 0.05", errors(a_test, test, 0.0, 0.3)),
            (f"mixture eps {best[0]} sigma_b {best[1]}", errors(a_test, test, *best)))
    for name, e in rows:
        print(f"  {name:<34} {summary(e, test)}")
        out[name] = {"median": float(np.median(e)), "p90": float(np.percentile(e, 90)), "p95": float(np.percentile(e, 95))}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", nargs="*", default=["sony-ilce-7rm4"])
    ap.add_argument("--n", type=int, default=3000, help="scenes per split")
    ap.add_argument("--models", type=Path, default=paths.REPO_ROOT / "data" / "models" / "illum")
    args = ap.parse_args(argv)
    for pid in args.profile:
        run(pid, args.n, args.models)
    return 0


if __name__ == "__main__":
    sys.exit(main())

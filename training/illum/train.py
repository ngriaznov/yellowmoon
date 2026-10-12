"""Training of the illuminant model (M1-06): JAX for the FFCC filter, numpy and JAX for the class head.

    python3 -m training.illum.train --profile sony-ilce-7rm4 [--steps 3000] [--n-train 8000]

Stage 1 minimises, with Adam on minibatches, the FFCC losses (Barron and Tsai 2017: a cross-entropy of the softmax
map against the truth and the Gaussian negative log-likelihood of the fitted ``(mu, Sigma)``) for the histogram-only
estimate and for the estimate fused with the as-shot prior, plus an L1-like error term and the FFCC regularisers on
the filter and the bias (total variation and L2). Stage 2 fits the class head on the features of the frozen
stage 1 model; stage 3 calibrates the class temperature and the confidence scale on the validation scenes.

Scenes are a pure function of ``(seed, split, index)``; the filter starts at zero, the head from a seeded
initialisation, the minibatch order is seeded.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from training.common import jsonfmt, paths
from training.illum import baselines as bl
from training.illum import dataset as dsm
from training.illum import evaluate as ev
from training.illum import features as ft
from training.illum import model as md
from training.illum import spectra as sp

MODEL_VERSION = "1.0.0"


@dataclass
class Config:
    steps: int = 3000
    batch: int = 128
    lr: float = 0.03
    f_scale: float = 10.0          # the optimiser sees F / f_scale
    w_ce: float = 1.0
    w_nll_obs: float = 1.0
    w_nll_post: float = 1.0
    w_dist: float = 20.0           # weight of the L1-like distance (log-chroma), both estimates
    lam_tv_f: float = 2e-3
    lam_tv_b: float = 2e-3
    lam_l2_f: float = 1e-4
    lam_l2_b: float = 1e-5
    pad0: float = 0.0016           # initial variance floor (0.04^2)
    aug_shift_bins: int = 0        # random integer shift of the histogram (and truth, as-shot) of up to +-this many bins
    eval_every: int = 250
    seed: int = 1
    head_hidden: int = 24
    head_steps: int = 2500
    head_lr: float = 0.01
    head_l2: float = 1e-4


# ------------------------------------------------------------------------------------------------------
# Stage 1: filter
# ------------------------------------------------------------------------------------------------------
def _gauss_moments(p, n, bin_size, lo_u, lo_v, pad):
    """Mean (log-chroma) and covariance of a softmax map (N, n, n) (jax version of IllumModel.gaussian_fit)."""
    k = jnp.arange(n, dtype=jnp.float32) - 0.5 * (n - 1)
    pi, pj = p.sum(-1), p.sum(-2)
    mi, mj = pi @ k, pj @ k
    eii, ejj = pi @ (k * k), pj @ (k * k)
    eij = jnp.einsum("nij,i,j->n", p, k, k)
    b2 = bin_size * bin_size
    c00 = (eii - mi * mi) * b2 + pad
    c11 = (ejj - mj * mj) * b2 + pad
    c01 = (eij - mi * mj) * b2
    mu = jnp.stack([lo_u + (mi + 0.5 * (n - 1)) * bin_size, lo_v + (mj + 0.5 * (n - 1)) * bin_size], -1)
    return mu, c00, c11, c01


def _nll(mu, c00, c11, c01, y):
    d = y - mu
    det = c00 * c11 - c01 * c01
    q = (c11 * d[:, 0] ** 2 - 2 * c01 * d[:, 0] * d[:, 1] + c00 * d[:, 1] ** 2) / det
    return 0.5 * q + 0.5 * jnp.log(det), d


def make_loss(spec: ft.HistSpec, cfg: Config, prior_sigma: float):
    n, b = spec.n_bins, spec.bin_size

    def loss_fn(params, x, y, a, a_ok):
        F = params["F"] * cfg.f_scale
        N = x.shape[0]
        xf = jnp.fft.rfft2(x)
        ff = jnp.fft.rfft2(F)
        z = jnp.fft.irfft2((xf * ff[None]).sum(1), s=(n, n)) + params["B"]
        logp = jax.nn.log_softmax(z.reshape(N, n * n), axis=-1).reshape(N, n, n)
        p = jnp.exp(logp)
        # cross-entropy against the bilinear splat of the truth
        fu = jnp.clip((y[:, 0] - spec.lo_u) / b, 0.0, n - 1.001)
        fv = jnp.clip((y[:, 1] - spec.lo_v) / b, 0.0, n - 1.001)
        iu, iv = jnp.floor(fu).astype(jnp.int32), jnp.floor(fv).astype(jnp.int32)
        tu, tv = fu - iu, fv - iv
        rows = jnp.arange(N)
        ce = -((1 - tu) * (1 - tv) * logp[rows, iu, iv] + tu * (1 - tv) * logp[rows, iu + 1, iv]
               + (1 - tu) * tv * logp[rows, iu, iv + 1] + tu * tv * logp[rows, iu + 1, iv + 1])
        pad = jnp.exp(params["log_pad"])
        mu, c00, c11, c01 = _gauss_moments(p, n, b, spec.lo_u, spec.lo_v, pad)
        nll_obs, d_obs = _nll(mu, c00, c11, c01, y)
        dist_obs = jnp.sqrt((d_obs**2).sum(-1) + 1e-6)
        # fuse with the as-shot prior
        det = c00 * c11 - c01 * c01
        l00, l11, l01 = c11 / det, c00 / det, -c01 / det
        wp = 1.0 / prior_sigma**2
        t00, t11, t01 = l00 + wp, l11 + wp, l01
        dt = t00 * t11 - t01 * t01
        p00, p11, p01 = t11 / dt, t00 / dt, -t01 / dt
        r0 = l00 * mu[:, 0] + l01 * mu[:, 1] + wp * a[:, 0]
        r1 = l01 * mu[:, 0] + l11 * mu[:, 1] + wp * a[:, 1]
        mu_post = jnp.stack([p00 * r0 + p01 * r1, p01 * r0 + p11 * r1], -1)
        nll_post, d_post = _nll(mu_post, p00, p11, p01, y)
        dist_post = jnp.sqrt((d_post**2).sum(-1) + 1e-6)
        m = a_ok.astype(jnp.float32)
        data = (cfg.w_ce * ce + cfg.w_nll_obs * nll_obs + cfg.w_dist * dist_obs).mean() \
            + ((cfg.w_nll_post * nll_post + cfg.w_dist * dist_post) * m).sum() / jnp.maximum(m.sum(), 1.0)

        def tv(t):
            return (jnp.sqrt((t - jnp.roll(t, 1, -1)) ** 2 + 1e-6).mean() + jnp.sqrt((t - jnp.roll(t, 1, -2)) ** 2 + 1e-6).mean())
        reg = (cfg.lam_tv_f * tv(F) + cfg.lam_tv_b * tv(params["B"]) + cfg.lam_l2_f * (F**2).mean()
               + cfg.lam_l2_b * (params["B"] ** 2).mean())
        aux = {"ce": ce.mean(), "nll_obs": nll_obs.mean(), "dist_obs": dist_obs.mean(),
               "dist_post": (dist_post * m).sum() / jnp.maximum(m.sum(), 1.0)}
        return data + reg, aux

    return loss_fn


def adam_init(params):
    z = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"m": z, "v": jax.tree_util.tree_map(jnp.zeros_like, params), "t": jnp.zeros((), jnp.int32)}


def adam_step(params, grads, state, lr, b1=0.9, b2=0.999, eps=1e-8):
    t = state["t"] + 1
    m = jax.tree_util.tree_map(lambda m_, g: b1 * m_ + (1 - b1) * g, state["m"], grads)
    v = jax.tree_util.tree_map(lambda v_, g: b2 * v_ + (1 - b2) * g * g, state["v"], grads)
    c1, c2 = 1 - b1**t, 1 - b2**t
    new = jax.tree_util.tree_map(lambda p_, m_, v_: p_ - lr * (m_ / c1) / (jnp.sqrt(v_ / c2) + eps), params, m, v)
    return new, {"m": m, "v": v, "t": t}


def params_to_model(params, spec, cfg: Config, profile_id: str, head: md.Head | None = None) -> md.IllumModel:
    F = np.asarray(params["F"], dtype=np.float64) * cfg.f_scale
    cam = sp.camera(profile_id)
    default_uv = sp.light_uv(cam, sp.daylight_at(5500.0))
    if head is None:
        k = md.n_features(len(spec.channels))
        head = md.Head(np.zeros((cfg.head_hidden, k)), np.zeros(cfg.head_hidden), np.zeros((6, cfg.head_hidden)),
                       np.zeros(6), np.zeros(k), np.ones(k), 1.0)
    return md.IllumModel(
        profile_id=profile_id, version=MODEL_VERSION, spec=spec, filt=F, bias=np.asarray(params["B"], dtype=np.float64),
        pad=float(np.exp(params["log_pad"])), head=head, conf=md.ConfHead(np.zeros(md.n_features(len(spec.channels)) + md.N_CONF_EXTRA + 1), np.zeros(md.n_features(len(spec.channels)) + md.N_CONF_EXTRA), np.ones(md.n_features(len(spec.channels)) + md.N_CONF_EXTRA)), default_uv=default_uv,
        bv_default=6.0, class_prior=np.ones(6) / 6.0)


def train_filter(spec, ds_train, ds_val, cfg: Config, profile_id: str, log=print):
    n = spec.n_bins
    C = ds_train.hist.shape[1]
    params = {"F": jnp.zeros((C, n, n), jnp.float32), "B": jnp.zeros((n, n), jnp.float32),
              "log_pad": jnp.asarray(np.log(cfg.pad0), jnp.float32)}
    loss_fn = make_loss(spec, cfg, md.PRIOR_SIGMA)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    @jax.jit
    def step(params, state, lr, x, y, a, a_ok):
        (loss, aux), g = grad_fn(params, x, y, a, a_ok)
        params, state = adam_step(params, g, state, lr)
        return params, state, loss, aux

    state = adam_init(params)
    rng = np.random.default_rng(cfg.seed)
    N = len(ds_train)
    a_all = np.nan_to_num(ds_train.as_shot_uv, nan=0.0).astype(np.float32)
    ok_all = ~np.isnan(ds_train.as_shot_uv[:, 0])
    y_all = ds_train.truth_uv.astype(np.float32)
    perm = rng.permutation(N)
    pos = 0
    t0 = time.time()
    history = []
    for it in range(cfg.steps):
        if pos + cfg.batch > N:
            perm, pos = rng.permutation(N), 0
        idx = perm[pos:pos + cfg.batch]
        pos += cfg.batch
        lr = cfg.lr * 0.5 * (1 + np.cos(np.pi * it / cfg.steps)) * min(1.0, (it + 1) / 50)
        xb, yb, ab = ds_train.hist[idx], y_all[idx], a_all[idx]
        if cfg.aug_shift_bins > 0:
            # A gain on the raw channels is a shift of the log-chroma histogram and of the truth and the as-shot value.
            sh = rng.integers(-cfg.aug_shift_bins, cfg.aug_shift_bins + 1, size=(len(idx), 2))
            xb = np.stack([np.roll(x, (int(a), int(b)), axis=(-2, -1)) for x, (a, b) in zip(xb, sh)])
            yb = yb + sh * np.float32(spec.bin_size)
            ab = ab + sh * np.float32(spec.bin_size)
        params, state, loss, aux = step(params, state, jnp.float32(lr), xb, yb, ab, ok_all[idx])
        if it % cfg.eval_every == 0 or it == cfg.steps - 1:
            m = params_to_model(params, spec, cfg, profile_id)
            ve = ev.run(m, ds_val, use_prior=False, use_bv=False)
            vp = ev.run(m, ds_val, use_prior=True, use_bv=False)
            e0, e1 = ev.angular(ve, ds_val), ev.angular(vp, ds_val)
            rec = {"it": it, "loss": float(loss), "val_median_hist": float(np.median(e0)), "val_p90_hist": float(np.percentile(e0, 90)),
                   "val_median_prior": float(np.median(e1)), "val_p90_prior": float(np.percentile(e1, 90)),
                   "pad": float(np.exp(params["log_pad"])), "sec": time.time() - t0}
            history.append(rec)
            log(f"  it {it:5d} loss {rec['loss']:.4f}  val median hist {rec['val_median_hist']:.2f} p90 {rec['val_p90_hist']:.2f}  "
                f"prior {rec['val_median_prior']:.2f} p90 {rec['val_p90_prior']:.2f}  pad {rec['pad']:.5f}  {rec['sec']:.0f}s")
    return params, history


# ------------------------------------------------------------------------------------------------------
# Stage 2 and 3: class head and calibration
# ------------------------------------------------------------------------------------------------------
def head_features(model: md.IllumModel, ds: dsm.Dataset, use_prior: bool, use_bv: bool) -> np.ndarray:
    feats = []
    for s in range(0, len(ds), 512):
        sl = slice(s, s + 512)
        out = model.estimate_from_hist(ds.hist[sl].astype(np.float64), ds.valid_fraction[sl],
                                       ds.as_shot_uv[sl] if use_prior else None, ds.bv[sl] if use_bv else None)
        feats.append(out["features"])
    return np.concatenate(feats)


def train_head(model: md.IllumModel, ds_train, ds_val, cfg: Config, log=print) -> md.Head:
    rng = np.random.default_rng(cfg.seed + 100)
    # features: the real thing, plus copies without the as-shot value and without Bv (so that the head copes with files lacking them)
    fs = [head_features(model, ds_train, True, True), head_features(model, ds_train, False, True),
          head_features(model, ds_train, True, False)]
    ys = [ds_train.klass] * 3
    X = np.concatenate(fs)
    y = np.concatenate(ys)
    mean = X.mean(0)
    scale = np.maximum(X.std(0), 1e-3)
    Z = ((X - mean) / scale).astype(np.float32)
    Xv = ((head_features(model, ds_val, True, True) - mean) / scale).astype(np.float32)
    yv = ds_val.klass
    k, h = md.n_features(len(model.spec.channels)), cfg.head_hidden
    key = jax.random.PRNGKey(cfg.seed)
    k1, k2 = jax.random.split(key)
    params = {"w1": jax.random.normal(k1, (h, k)) / np.sqrt(k), "b1": jnp.zeros(h),
              "w2": 0.1 * jax.random.normal(k2, (6, h)), "b2": jnp.zeros(6)}

    def fwd(p, z):
        return jnp.tanh(z @ p["w1"].T + p["b1"]) @ p["w2"].T + p["b2"]

    def loss(p, z, yy):
        lg = fwd(p, z)
        nll = -jax.nn.log_softmax(lg)[jnp.arange(z.shape[0]), yy].mean()
        return nll + cfg.head_l2 * ((p["w1"] ** 2).sum() + (p["w2"] ** 2).sum())

    gfn = jax.jit(jax.value_and_grad(loss))
    state = adam_init(params)
    best, best_nll = params, 1e9
    for it in range(cfg.head_steps):
        idx = rng.integers(0, Z.shape[0], size=1024)
        l, g = gfn(params, Z[idx], y[idx])
        lr = cfg.head_lr * 0.5 * (1 + np.cos(np.pi * it / cfg.head_steps))
        params, state = adam_step(params, g, state, lr)
        if it % 100 == 0 or it == cfg.head_steps - 1:
            lv = np.asarray(fwd(params, Xv))
            pv = md.softmax(lv)
            nll = float(-np.log(np.maximum(pv[np.arange(len(yv)), yv], 1e-12)).mean())
            if nll < best_nll:
                best, best_nll = {k_: np.asarray(v) for k_, v in params.items()}, nll
            if it % 500 == 0:
                log(f"  head it {it:5d} train {float(l):.3f} val nll {nll:.3f} val acc {(pv.argmax(-1) == yv).mean():.3f}")
    log(f"  head best val nll {best_nll:.3f}")
    p = {k_: np.asarray(v, dtype=np.float64) for k_, v in best.items()}
    return md.Head(p["w1"], p["b1"], p["w2"], p["b2"], mean, scale, 1.0)


def confidence_inputs(model: md.IllumModel, ds: dsm.Dataset, use_prior: bool) -> tuple[np.ndarray, np.ndarray]:
    """Confidence features and the labels (angular error under 2 degrees) of ``ds`` with or without the as-shot prior."""
    xs, hits = [], []
    for s in range(0, len(ds), 512):
        sl = slice(s, s + 512)
        a = ds.as_shot_uv[sl] if use_prior else np.full((len(ds.as_shot_uv[sl]), 2), np.nan)
        out = model.estimate_from_hist(ds.hist[sl].astype(np.float64), ds.valid_fraction[sl], a, ds.bv[sl])
        xs.append(np.concatenate([out["features"], md.conf_extras(out["uv_obs"], out["cov_obs"], out["cov"], a)], axis=-1))
        hits.append(bl.angular_error_uv_deg(out["uv"], ds.truth_uv[sl]) < ev.GATE_MEDIAN_DEG)
    return np.concatenate(xs), np.concatenate(hits)


def fit_confidence(model: md.IllumModel, ds_val: dsm.Dataset, l2: float = 1e-3, log=print) -> None:
    """Logistic regression of "error under 2 degrees" on the head features and three extras, on the validation scenes (held-out
    lights and patches), once with and once without the as-shot value, so that the confidence is honest for both (in place)."""
    from scipy.optimize import minimize

    parts = [confidence_inputs(model, ds_val, up) for up in (True, False)]
    x = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts]).astype(np.float64)
    mean, scale = x.mean(0), np.maximum(x.std(0), 1e-3)
    z = (x - mean) / scale

    def nll(w):
        s = z @ w[:-1] + w[-1]
        return np.mean(np.logaddexp(0.0, s) - y * s) + l2 * np.sum(w[:-1] ** 2)

    w = minimize(nll, np.zeros(z.shape[1] + 1), method="L-BFGS-B").x
    model.conf = md.ConfHead(w, mean, scale)
    p = model.conf.probability(x)
    log(f"  confidence: logistic on {z.shape[1]} features, validation Brier {np.mean((p - y) ** 2):.4f} "
        f"(base rate {y.mean() * (1 - y.mean()):.4f})")


def calibrate(model: md.IllumModel, ds_val: dsm.Dataset, log=print) -> None:
    """Class temperature on the validation scenes, then the confidence head (in place)."""
    f = head_features(model, ds_val, True, True)
    lg = model.head.logits(f)
    yv = ds_val.klass
    best_t, best = 1.0, 1e9
    for t in np.exp(np.linspace(np.log(0.4), np.log(4.0), 61)):
        p = md.softmax(lg / t)
        nll = float(-np.log(np.maximum(p[np.arange(len(yv)), yv], 1e-12)).mean())
        if nll < best:
            best_t, best = float(t), nll
    model.head.temperature = best_t
    log(f"  calibration: class temperature {best_t:.3f} (val nll {best:.3f})")
    fit_confidence(model, ds_val, log=log)


def class_prior_of(ds: dsm.Dataset) -> np.ndarray:
    c = np.bincount(ds.klass, minlength=6).astype(np.float64)
    return c / c.sum()


def train_profile(profile_id: str, cfg: Config, n_train: int, n_val: int, procs: int = 2, log=print,
                  channels=("intensity", "edge")) -> tuple[md.IllumModel, dict]:
    spec = dsm.hist_spec_for(profile_id, channels=channels)
    log(f"{profile_id}: histogram window u0 {spec.lo_u:.4f} v0 {spec.lo_v:.4f} bin {spec.bin_size}")
    ds_tr = dsm.build(profile_id, "train", n_train, spec, procs=procs)
    ds_va = dsm.build(profile_id, "val", n_val, spec, procs=procs)
    params, history = train_filter(spec, ds_tr, ds_va, cfg, profile_id, log=log)
    m = params_to_model(params, spec, cfg, profile_id)
    m.class_prior = class_prior_of(ds_tr)
    m.head = train_head(m, ds_tr, ds_va, cfg, log=log)
    calibrate(m, ds_va, log=log)
    m.meta = {"config": asdict(cfg), "n_train": n_train, "n_val": n_val, "dataset_seed": dsm.SEED}
    return m.rounded(), {"history": history}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="sony-ilce-7rm4")
    ap.add_argument("--steps", type=int, default=Config.steps)
    ap.add_argument("--batch", type=int, default=Config.batch)
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-val", type=int, default=1500)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--channels", default="intensity,edge")
    ap.add_argument("--out", type=Path, default=paths.OUT / "illum" / "models")
    args = ap.parse_args(argv)
    cfg = Config(steps=args.steps, batch=args.batch)
    model, info = train_profile(args.profile, cfg, args.n_train, args.n_val, args.procs, channels=tuple(args.channels.split(",")))
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"{args.profile}.json"
    out.write_text(jsonfmt.dumps(model.to_json()))
    print("wrote", out, "params", model.param_count())
    return 0


if __name__ == "__main__":
    sys.exit(main())

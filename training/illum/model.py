"""The illuminant model: FFCC-style filter, Gaussian fit, as-shot prior, class head. Numpy reference.

``ym_color::illum`` is a port of this file (``docs/02`` section Illuminant estimate); the two agree to 1e-9 on
the committed test vectors. All arithmetic is float64; the stored tensors are float32 (exactly representable).

Pipeline for one preview (``estimate``):

1. ``features.histograms``: normalised log-chroma histograms ``X_c`` (``C`` channels, ``n x n`` bins).
2. Filter (Barron and Tsai 2017, FFCC): ``Z = sum_c  X_c (*) F_c  +  B`` with a circular convolution
   (``(X (*) F)[i, j] = sum_{a, b} X[a, b] F[(i - a) mod n, (j - b) mod n]``, evaluated with FFTs). ``Z[i, j]`` scores
   the light whose log-chroma is ``(lo_u + i * bin_size, lo_v + j * bin_size)``.
3. ``P = softmax(Z)``; the histogram-only estimate is the mean ``mu_obs`` of ``P`` (in log-chroma) and the
   covariance ``Sigma_obs = Cov(P) + pad * I`` (``pad`` is a learned variance floor).
4. As-shot prior: a Gaussian ``N(a, sigma_p^2 I)`` on ``(u, v)`` with ``a`` the camera's as-shot white balance
   (``sigma_p = 0.05``) multiplies the estimate: ``Sigma_post = (Sigma_obs^-1 + I / sigma_p^2)^-1``,
   ``mu_post = Sigma_post (Sigma_obs^-1 mu_obs + a / sigma_p^2)``. Without an as-shot value ``post = obs``.
5. ``w_scene = [exp(-u), 1, exp(-v)]`` of ``mu_post``; confidence = a logistic regression on the class-head features and three
   more (:func:`conf_extras`: the Mahalanobis distance between the histogram estimate and the as-shot value, and the
   log variances of the fused and of the histogram-only Gaussian), fitted on validation scenes with held-out lights so that it is
   the probability that the angular error is under 2 degrees.
6. Class head: 25 features (:func:`head_features`) -> standardise -> ``tanh`` layer -> logits -> softmax over
   :data:`CLASSES`.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

from training.illum import features as ft

CLASSES = ("daylight", "shade_overcast", "tungsten_halogen", "fluorescent", "led", "mixed")
FORMAT = "yellowmoon-illum"
FORMAT_VERSION = 1
N_FEATURES = 11 + 7 * 2  # head features for two histogram channels; in general 11 + 7 * channels (:func:`n_features`)
PRIOR_SIGMA = 0.05
N_CONF_EXTRA = 3
TENSOR_ORDER = ("filter", "bias", "w1", "b1", "w2", "b2", "feature_mean", "feature_scale", "conf_w", "conf_mean", "conf_scale")


@dataclass
class Head:
    """Two-layer class head: standardise, tanh layer, linear layer, temperature softmax."""

    w1: np.ndarray            # (hidden, n_features)
    b1: np.ndarray            # (hidden,)
    w2: np.ndarray            # (classes, hidden)
    b2: np.ndarray            # (classes,)
    feature_mean: np.ndarray  # (n_features,)
    feature_scale: np.ndarray  # (n_features,)
    temperature: float = 1.0

    def logits(self, f: np.ndarray) -> np.ndarray:
        z = (f - self.feature_mean) / self.feature_scale
        return np.tanh(z @ self.w1.T + self.b1) @ self.w2.T + self.b2

    def probs(self, f: np.ndarray) -> np.ndarray:
        return softmax(self.logits(f) / self.temperature, axis=-1)


@dataclass
class ConfHead:
    """Logistic confidence: ``sigmoid(w . standardise([head features, extras]) + b)``."""

    w: np.ndarray       # (n_features + N_CONF_EXTRA + 1,): weights, then the bias
    mean: np.ndarray    # (n_features + N_CONF_EXTRA,)
    scale: np.ndarray   # (n_features + N_CONF_EXTRA,)

    def probability(self, x: np.ndarray) -> np.ndarray:
        z = ((x - self.mean) / self.scale) @ self.w[:-1] + self.w[-1]
        return 1.0 / (1.0 + np.exp(-z))


def conf_extras(mu_obs: np.ndarray, cov_obs: np.ndarray, cov_post: np.ndarray, as_shot_uv: np.ndarray) -> np.ndarray:
    """The three extra confidence features ``(..., 3)``: Mahalanobis distance of the histogram estimate from the as-shot value
    (0 without one), ``ln`` of the mean variance of the fused Gaussian and of the histogram-only Gaussian."""
    a = np.asarray(as_shot_uv, dtype=np.float64)
    has = ~np.isnan(a[..., 0])
    d0 = np.where(has, mu_obs[..., 0] - np.where(has, a[..., 0], 0.0), 0.0)
    d1 = np.where(has, mu_obs[..., 1] - np.where(has, a[..., 1], 0.0), 0.0)
    s0, s1, s01 = cov_obs[..., 0, 0], cov_obs[..., 1, 1], cov_obs[..., 0, 1]
    det = s0 * s1 - s01 * s01
    q = (s1 * d0 * d0 - 2.0 * s01 * d0 * d1 + s0 * d1 * d1) / det
    maha = np.sqrt(np.maximum(q, 0.0))
    return np.stack([maha, np.log(0.5 * (cov_post[..., 0, 0] + cov_post[..., 1, 1])),
                     np.log(0.5 * (s0 + s1))], axis=-1)


@dataclass
class IllumModel:
    """One trained model (one camera profile)."""

    profile_id: str
    version: str
    spec: ft.HistSpec
    filt: np.ndarray                # (C, n, n)
    bias: np.ndarray                # (n, n)
    pad: float                      # variance floor added to the fitted covariance (log-chroma^2)
    head: Head
    conf: ConfHead
    default_uv: np.ndarray          # (2,) light used when there is nothing to estimate from (D55 for the camera)
    bv_default: float
    class_prior: np.ndarray         # (6,)
    prior_sigma: float = PRIOR_SIGMA
    meta: dict = field(default_factory=dict)

    # ---- parameter count -----------------------------------------------------------------------
    def param_count(self) -> int:
        h = self.head
        return int(self.filt.size + self.bias.size + h.w1.size + h.b1.size + h.w2.size + h.b2.size
                   + h.feature_mean.size + h.feature_scale.size + self.conf.w.size + self.conf.mean.size
                   + self.conf.scale.size + 2)  # + pad, temperature

    # ---- stages ---------------------------------------------------------------------------------
    def filter_scores(self, x: np.ndarray) -> np.ndarray:
        """``Z`` for histograms ``x`` of shape ``(..., C, n, n)`` -> ``(..., n, n)``."""
        fx = np.fft.fft2(x, axes=(-2, -1))
        ff = np.fft.fft2(self.filt, axes=(-2, -1))
        return np.fft.ifft2((fx * ff).sum(axis=-3), axes=(-2, -1)).real + self.bias

    def gaussian_fit(self, z: np.ndarray):
        """Softmax and moment fit of the score map ``z (..., n, n)``.

        Returns ``(mu_obs (..., 2), cov_obs (..., 2, 2), p_max (...))``; mean and covariance are in log-chroma.
        """
        n = self.spec.n_bins
        p = softmax(z.reshape(*z.shape[:-2], n * n), axis=-1).reshape(z.shape)
        k = np.arange(n, dtype=np.float64) - 0.5 * (n - 1)
        pi = p.sum(axis=-1)  # marginal over j -> (..., n) along i
        pj = p.sum(axis=-2)
        mi = pi @ k
        mj = pj @ k
        eii = pi @ (k * k)
        ejj = pj @ (k * k)
        eij = np.einsum("...ij,i,j->...", p, k, k)
        b = self.spec.bin_size
        cov = np.empty(z.shape[:-2] + (2, 2))
        cov[..., 0, 0] = (eii - mi * mi) * b * b + self.pad
        cov[..., 1, 1] = (ejj - mj * mj) * b * b + self.pad
        cov[..., 0, 1] = cov[..., 1, 0] = (eij - mi * mj) * b * b
        mu = np.stack([self.spec.lo_u + (mi + 0.5 * (n - 1)) * b, self.spec.lo_v + (mj + 0.5 * (n - 1)) * b], axis=-1)
        return mu, cov, p.max(axis=(-2, -1))

    def fuse(self, mu_obs: np.ndarray, cov_obs: np.ndarray, as_shot_uv: np.ndarray | None):
        """Multiply the estimate with the as-shot Gaussian prior. ``as_shot_uv (..., 2)`` may hold NaN for 'none'."""
        if as_shot_uv is None:
            return mu_obs.copy(), cov_obs.copy()
        a = np.asarray(as_shot_uv, dtype=np.float64)
        has = ~np.isnan(a[..., 0])
        a0 = np.where(has[..., None], a, 0.0)
        s0, s1, s01 = cov_obs[..., 0, 0], cov_obs[..., 1, 1], cov_obs[..., 0, 1]
        det = s0 * s1 - s01 * s01
        lam = np.empty_like(cov_obs)  # precision of the histogram estimate
        lam[..., 0, 0], lam[..., 1, 1], lam[..., 0, 1] = s1 / det, s0 / det, -s01 / det
        lam[..., 1, 0] = lam[..., 0, 1]
        wp = 1.0 / (self.prior_sigma**2)
        tot = lam.copy()
        tot[..., 0, 0] += wp
        tot[..., 1, 1] += wp
        t0, t1, t01 = tot[..., 0, 0], tot[..., 1, 1], tot[..., 0, 1]
        dt = t0 * t1 - t01 * t01
        cov_post = np.empty_like(cov_obs)
        cov_post[..., 0, 0], cov_post[..., 1, 1], cov_post[..., 0, 1] = t1 / dt, t0 / dt, -t01 / dt
        cov_post[..., 1, 0] = cov_post[..., 0, 1]
        rhs = np.einsum("...ij,...j->...i", lam, mu_obs) + wp * a0
        mu_post = np.einsum("...ij,...j->...i", cov_post, rhs)
        return (np.where(has[..., None], mu_post, mu_obs), np.where(has[..., None, None], cov_post, cov_obs))

    # ---- class head -----------------------------------------------------------------------------
    def moments(self, x: np.ndarray) -> np.ndarray:
        """Per channel ``(mean_u, mean_v, std_u, std_v, corr, entropy)`` of the histograms ``(..., C, n, n)``.

        An empty channel gives zeros.
        """
        n = self.spec.n_bins
        b = self.spec.bin_size
        k = np.arange(n, dtype=np.float64) - 0.5 * (n - 1)
        pi, pj = x.sum(axis=-1), x.sum(axis=-2)
        mass = pi.sum(axis=-1)
        ok = mass > 0
        m = np.where(ok, mass, 1.0)
        mi, mj = pi @ k / m, pj @ k / m
        vi = np.maximum(pi @ (k * k) / m - mi * mi, 0.0)
        vj = np.maximum(pj @ (k * k) / m - mj * mj, 0.0)
        eij = np.einsum("...ij,i,j->...", x, k, k) / m
        si, sj = np.sqrt(vi), np.sqrt(vj)
        corr = np.where((si > 0) & (sj > 0), (eij - mi * mj) / np.maximum(si * sj, 1e-300), 0.0)
        ent = -np.where(x > 0, x * np.log(np.where(x > 0, x, 1.0)), 0.0).sum(axis=(-2, -1)) / m
        ent = ent + np.log(m)  # entropy of x / mass
        out = np.stack([self.spec.lo_u + (mi + 0.5 * (n - 1)) * b, self.spec.lo_v + (mj + 0.5 * (n - 1)) * b,
                        si * b, sj * b, corr, ent], axis=-1)
        return np.where(ok[..., None], out, 0.0)

    def head_features(self, x, valid_fraction, mu_obs, cov_obs, mu_post, p_max, as_shot_uv, bv) -> np.ndarray:
        """The 25 features of the class head (see :data:`FEATURE_NAMES`). Batch-capable; ``as_shot_uv`` and ``bv``
        may hold NaN for 'not available'."""
        a = np.asarray(as_shot_uv, dtype=np.float64)
        has_a = ~np.isnan(a[..., 0])
        a_u = np.where(has_a, a[..., 0], mu_obs[..., 0])
        a_v = np.where(has_a, a[..., 1], mu_obs[..., 1])
        bv = np.asarray(bv, dtype=np.float64)
        has_bv = ~np.isnan(bv)
        det = cov_obs[..., 0, 0] * cov_obs[..., 1, 1] - cov_obs[..., 0, 1] ** 2
        mom = self.moments(x)  # (..., C, 6)
        cols = [
            mu_post[..., 0], mu_post[..., 1], a_u, a_v, has_a.astype(np.float64),
            np.where(has_a, mu_obs[..., 0] - a_u, 0.0), np.where(has_a, mu_obs[..., 1] - a_v, 0.0),
            0.5 * np.log(det), np.where(has_bv, bv, self.bv_default), has_bv.astype(np.float64), np.log(p_max),
        ]
        for c in range(x.shape[-3]):
            m = mom[..., c, :]
            cols += [m[..., 0] - mu_post[..., 0], m[..., 1] - mu_post[..., 1], m[..., 2], m[..., 3], m[..., 4],
                     m[..., 5], valid_fraction[..., c]]
        return np.stack(cols, axis=-1)

    # ---- whole estimate ---------------------------------------------------------------------------
    def estimate_from_hist(self, x, valid_fraction, as_shot_uv=None, bv=None) -> dict:
        """Everything the estimator outputs for histograms ``x (..., C, n, n)`` (batch-capable)."""
        batch = x.shape[:-3]
        a = np.full(batch + (2,), np.nan) if as_shot_uv is None else np.asarray(as_shot_uv, dtype=np.float64)
        b = np.full(batch, np.nan) if bv is None else np.asarray(bv, dtype=np.float64)
        z = self.filter_scores(x)
        mu_obs, cov_obs, p_max = self.gaussian_fit(z)
        mu_post, cov_post = self.fuse(mu_obs, cov_obs, a)
        feats = self.head_features(x, valid_fraction, mu_obs, cov_obs, mu_post, p_max, a, b)
        probs = self.head.probs(feats)
        conf = self.conf.probability(np.concatenate([feats, conf_extras(mu_obs, cov_obs, cov_post, a)], axis=-1))
        return {"uv": mu_post, "cov": cov_post, "uv_obs": mu_obs, "cov_obs": cov_obs, "probs": probs,
                "confidence": conf, "features": feats, "scores": z}

    def estimate(self, img: np.ndarray, mask: np.ndarray | None, as_shot_uv=None, bv=None) -> dict:
        """Estimate from a preview (the reference of ``ym_color::illum::IlluminantModel::estimate``)."""
        x, st = ft.histograms(img, mask, self.spec)
        if st.mass.sum() <= 0:
            return self.fallback(as_shot_uv)
        out = self.estimate_from_hist(x, st.valid_fraction, as_shot_uv, bv)
        out["hist"] = x
        out["stats"] = st
        out["fallback"] = False
        return out

    def fallback(self, as_shot_uv=None) -> dict:
        """No usable pixels: the as-shot value if there is one, else the camera's D55, with confidence 0."""
        has = as_shot_uv is not None and not np.isnan(np.asarray(as_shot_uv)[0])
        uv = np.asarray(as_shot_uv, dtype=np.float64) if has else self.default_uv.copy()
        return {"uv": uv, "cov": np.eye(2) * self.prior_sigma**2, "probs": self.class_prior.copy(),
                "confidence": 0.0, "fallback": True}

    # ---- JSON -----------------------------------------------------------------------------------
    def tensors(self) -> dict[str, np.ndarray]:
        h = self.head
        return {"filter": self.filt, "bias": self.bias, "w1": h.w1, "b1": h.b1, "w2": h.w2, "b2": h.b2,
                "feature_mean": h.feature_mean, "feature_scale": h.feature_scale, "conf_w": self.conf.w,
                "conf_mean": self.conf.mean, "conf_scale": self.conf.scale}

    def payload_sha256(self) -> str:
        """SHA-256 of the float32 little-endian bytes of the tensors in :data:`TENSOR_ORDER`."""
        t = self.tensors()
        m = hashlib.sha256()
        for name in TENSOR_ORDER:
            m.update(np.ascontiguousarray(t[name], dtype="<f4").tobytes())
        return m.hexdigest()

    def to_json(self) -> dict:
        t = self.tensors()
        h = self.head
        doc = {
            "format": FORMAT, "format_version": FORMAT_VERSION, "model": "illum", "model_version": self.version,
            "profile_id": self.profile_id,
            "histogram": self.spec.to_json(),
            "fit": {"pad_variance": float(self.pad), "prior_sigma": float(self.prior_sigma),
                    "default_uv": [float(v) for v in self.default_uv]},
            "confidence": {"kind": "logistic", "n_features": int(self.conf.mean.size),
                           "meaning": "probability that the angular error of w_scene is under 2 degrees"},
            "head": {"classes": list(CLASSES), "n_features": int(h.feature_mean.size), "hidden": int(h.w1.shape[0]),
                     "temperature": float(h.temperature), "bv_default": float(self.bv_default),
                     "class_prior": [float(v) for v in self.class_prior]},
            "tensors": {name: {"shape": list(t[name].shape), "dtype": "f32le",
                               "data_b64": base64.b64encode(np.ascontiguousarray(t[name], dtype="<f4").tobytes()).decode("ascii")}
                        for name in TENSOR_ORDER},
            "tensor_order": list(TENSOR_ORDER),
            "param_count": self.param_count(),
            "sha256": self.payload_sha256(),
            "meta": self.meta,
        }
        return doc

    @staticmethod
    def from_json(doc: dict) -> "IllumModel":
        if doc.get("format") != FORMAT or doc.get("format_version") != FORMAT_VERSION:
            raise ValueError("not an illuminant model file this reader understands")
        t = {}
        for name in TENSOR_ORDER:
            d = doc["tensors"][name]
            t[name] = np.frombuffer(base64.b64decode(d["data_b64"]), dtype="<f4").astype(np.float64).reshape(d["shape"])
        hd = doc["head"]
        head = Head(t["w1"], t["b1"], t["w2"], t["b2"], t["feature_mean"], t["feature_scale"], float(hd["temperature"]))
        m = IllumModel(
            profile_id=doc["profile_id"], version=doc["model_version"], spec=ft.HistSpec.from_json(doc["histogram"]),
            filt=t["filter"], bias=t["bias"], pad=float(doc["fit"]["pad_variance"]), head=head,
            conf=ConfHead(t["conf_w"], t["conf_mean"], t["conf_scale"]),
            default_uv=np.array(doc["fit"]["default_uv"], dtype=np.float64), bv_default=float(hd["bv_default"]),
            class_prior=np.array(hd["class_prior"], dtype=np.float64), prior_sigma=float(doc["fit"]["prior_sigma"]),
            meta=doc.get("meta", {}))
        if m.payload_sha256() != doc["sha256"]:
            raise ValueError("model payload does not match its sha256")
        return m

    def rounded(self) -> "IllumModel":
        """The model as it is after a round trip through float32 storage (what ships)."""
        return IllumModel.from_json(json.loads(json.dumps(self.to_json())))


def softmax(a: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(a - a.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


FEATURE_NAMES = (
    "u_post", "v_post", "as_shot_u", "as_shot_v", "as_shot_valid", "obs_minus_as_shot_u", "obs_minus_as_shot_v",
    "half_log_det_cov_obs", "bv", "bv_valid", "log_p_max",
) + tuple(f"{ch}_{n}" for ch in ("intensity", "edge")
          for n in ("mean_u_rel", "mean_v_rel", "std_u", "std_v", "corr", "entropy", "valid_fraction"))
assert len(FEATURE_NAMES) == N_FEATURES


def n_features(n_channels: int) -> int:
    """Number of class-head features for a model with ``n_channels`` histogram channels."""
    return 11 + 7 * n_channels

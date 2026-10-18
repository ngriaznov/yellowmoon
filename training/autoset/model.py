"""The ``autoset`` ensemble (docs/18 sections 4.2 to 4.4): the file format, the two forward passes, the combination and the rule.

``ym_color::autoset`` (AS-03) is the Rust port. ``export.py`` writes the vectors it must reproduce to 1e-9. The weights are stored
as little-endian float32 and read back as float64, so every number here is what the Rust reader sees. The forward pass exists
twice on purpose: in JAX (training, float64) and in numpy (export and evaluation); the tests check that they agree.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
from dataclasses import dataclass

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update("jax_enable_x64", True)

FORMAT = "yellowmoon-autoset"
FORMAT_VERSION = 1
N_FEATURES = 39
FEATURE_NAMES = (
    "key_bv", "bv_known",
    "q01", "q02", "q05", "q10", "q20", "q30", "q40", "q50", "q60", "q70", "q80", "q90", "q95", "q98", "q99", "q995", "q999",
    "mq50", "mq90", "mq98", "mq995", "mq999",
    "clip_headroom", "clipped_share", "centre_minus_key", "top_minus_bottom", "bright_share", "dark_share",
    "p_daylight", "p_tungsten", "p_fluorescent", "p_led",
    "mired", "duv", "light_confidence", "log_iso", "flash",
)
OUTPUTS = ("key_target", "key_logvar", "contrast", "contrast_logvar")
#: ``(out, in)`` of the three layers of every member (section 4.2): 39 -> 32 -> 32 -> 4.
LAYER_SHAPES = ((32, N_FEATURES), (32, 32), (4, 32))
N_MEMBERS = 5
LOGVAR_LIMIT = 20.0
CONTRAST_LIMIT = 30.0
PRIOR_SIGMA_KEY = 0.75
PRIOR_SIGMA_CONTRAST = 10.0
CAMERA_GAIN_EV = 1.11
EV_LIMIT = 3.0
#: Standardisation floor: a feature with no spread in the training rows is divided by this, not by zero.
STD_FLOOR = 1e-3

assert len(FEATURE_NAMES) == N_FEATURES

Layer = tuple[np.ndarray, np.ndarray]


class AutosetError(ValueError):
    """A model file this reader refuses (format, feature list, shape, checksum or a non-finite value)."""


# ------------------------------------------------------------------------------------------------------------------
# Tensors and JSON (section 4.3)
# ------------------------------------------------------------------------------------------------------------------


def _encode(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode("ascii")


def _tensor_doc(a: np.ndarray) -> dict:
    return {"shape": [int(s) for s in a.shape], "data_b64": _encode(a)}


def _decode(text: str, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        raw = base64.b64decode(text, validate=True)
    except (ValueError, TypeError) as exc:
        raise AutosetError(f"{name}: not base64") from exc
    count = int(np.prod(shape))
    if len(raw) != 4 * count:
        raise AutosetError(f"{name}: expected {count} float32 values, got {len(raw) // 4}")
    arr = np.frombuffer(raw, dtype="<f4").astype(np.float64).reshape(shape)
    if not np.all(np.isfinite(arr)):
        raise AutosetError(f"{name}: a value is not finite")
    return arr


def _tensor_from_doc(doc: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    if not isinstance(doc, dict) or doc.get("shape") != list(shape) or not isinstance(doc.get("data_b64"), str):
        raise AutosetError(f"{name}: shape {list(shape)} and data_b64 are required")
    return _decode(doc["data_b64"], shape, name)


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise AutosetError(f"{name} is not a finite number")
    return float(value)


def _vector(value: object, n: int, name: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != n:
        raise AutosetError(f"{name} must be a list of {n} numbers")
    return np.array([_number(v, name) for v in value], dtype=np.float64)


def _positive(value: object, name: str) -> float:
    x = _number(value, name)
    if not x > 0.0:
        raise AutosetError(f"{name} must be positive")
    return x


def _f32(x: float) -> float:
    """The float32 value nearest to ``x``, as a float (what the file stores for every constant)."""
    return float(np.float32(x))


# ------------------------------------------------------------------------------------------------------------------
# The model
# ------------------------------------------------------------------------------------------------------------------


@dataclass(eq=False)
class AutosetModel:
    """An ensemble of MLPs with standardised inputs (docs/18 section 4)."""

    model_version: str
    input_mean: np.ndarray  # (39,)
    input_std: np.ndarray  # (39,)
    members: list[list[Layer]]  # per member: three (w (out, in), b (out,)) layers
    prior_sigma_key: float = PRIOR_SIGMA_KEY
    prior_sigma_contrast: float = PRIOR_SIGMA_CONTRAST
    contrast_limit: float = CONTRAST_LIMIT

    # ---- sizes and checksum ----------------------------------------------------------------------------
    def param_count_per_member(self) -> int:
        return sum(w.size + b.size for w, b in self.members[0])

    def param_count(self) -> int:
        return sum(w.size + b.size for layers in self.members for w, b in layers)

    def tensors(self) -> list[tuple[str, np.ndarray]]:
        """Every weight tensor in checksum order: members, then layers, ``w`` before ``b``."""
        out = []
        for mi, layers in enumerate(self.members):
            for li, (w, b) in enumerate(layers):
                out.append((f"members.{mi}.layers.{li}.w", w))
                out.append((f"members.{mi}.layers.{li}.b", b))
        return out

    def payload_sha256(self) -> str:
        """SHA-256 of the little-endian float32 bytes of :meth:`tensors`, in order (section 4.3)."""
        h = hashlib.sha256()
        for _, a in self.tensors():
            h.update(np.ascontiguousarray(a, dtype="<f4").tobytes())
        return h.hexdigest()

    # ---- JSON --------------------------------------------------------------------------------------------
    def to_json(self) -> dict:
        """The document of section 4.3 (every number is stored as float32, the checksum covers the weights)."""
        return {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "model_version": self.model_version,
            "feature_names": list(FEATURE_NAMES),
            "input_mean": [_f32(v) for v in self.input_mean],
            "input_std": [_f32(v) for v in self.input_std],
            "activation": "tanh",
            "outputs": list(OUTPUTS),
            "prior_sigma": {"key_target": _f32(self.prior_sigma_key), "contrast": _f32(self.prior_sigma_contrast)},
            "contrast_limit": _f32(self.contrast_limit),
            "members": [{"layers": [{"w": _tensor_doc(w), "b": _tensor_doc(b)} for w, b in layers]} for layers in self.members],
            "sha256": self.payload_sha256(),
        }

    @classmethod
    def from_json(cls, doc: object) -> AutosetModel:
        """Reads a model file; refuses a wrong format, feature list, shape, non-finite value or checksum (``AutosetError``)."""
        if not isinstance(doc, dict) or doc.get("format") != FORMAT or doc.get("format_version") != FORMAT_VERSION:
            raise AutosetError("not an autoset model file this reader understands")
        if doc.get("feature_names") != list(FEATURE_NAMES):
            raise AutosetError("feature names differ from the 39 features of docs/18 section 3")
        if doc.get("activation") != "tanh" or doc.get("outputs") != list(OUTPUTS):
            raise AutosetError("activation or outputs differ from docs/18 section 4.3")
        if not isinstance(doc.get("model_version"), str):
            raise AutosetError("model_version is required")
        mean = _vector(doc.get("input_mean"), N_FEATURES, "input_mean")
        std = _vector(doc.get("input_std"), N_FEATURES, "input_std")
        if np.any(std <= 0.0):
            raise AutosetError("input_std must be positive")
        prior = doc.get("prior_sigma")
        if not isinstance(prior, dict):
            raise AutosetError("prior_sigma is required")
        members_doc = doc.get("members")
        if not isinstance(members_doc, list) or not members_doc:
            raise AutosetError("the model has no members")
        members: list[list[Layer]] = []
        for mi, member in enumerate(members_doc):
            layers_doc = member.get("layers") if isinstance(member, dict) else None
            if not isinstance(layers_doc, list) or len(layers_doc) != len(LAYER_SHAPES):
                raise AutosetError(f"member {mi} must have {len(LAYER_SHAPES)} layers")
            layers: list[Layer] = []
            for li, (layer, (out_dim, in_dim)) in enumerate(zip(layers_doc, LAYER_SHAPES, strict=True)):
                name = f"members[{mi}].layers[{li}]"
                if not isinstance(layer, dict):
                    raise AutosetError(f"{name} is not an object")
                layers.append((_tensor_from_doc(layer.get("w"), (out_dim, in_dim), f"{name}.w"),
                               _tensor_from_doc(layer.get("b"), (out_dim,), f"{name}.b")))
            members.append(layers)
        model = cls(
            model_version=doc["model_version"],
            input_mean=mean,
            input_std=std,
            members=members,
            prior_sigma_key=_positive(prior.get("key_target"), "prior_sigma.key_target"),
            prior_sigma_contrast=_positive(prior.get("contrast"), "prior_sigma.contrast"),
            contrast_limit=_positive(doc.get("contrast_limit"), "contrast_limit"),
        )
        if model.payload_sha256() != doc.get("sha256"):
            raise AutosetError("model payload does not match its sha256")
        return model

    def rounded(self) -> AutosetModel:
        """The model as it is after a round trip through the file (float32 storage): what ships and what is evaluated."""
        return AutosetModel.from_json(json.loads(json.dumps(self.to_json())))

    # ---- forward passes ----------------------------------------------------------------------------------
    def standardise(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=np.float64) - self.input_mean) / self.input_std

    def member_outputs(self, X: np.ndarray) -> np.ndarray:
        """Raw outputs ``(members, n, 4)`` in :data:`OUTPUTS` order (numpy, float64; logvars not clamped)."""
        z = self.standardise(X)
        outs = []
        for layers in self.members:
            h = z
            for w, b in layers[:-1]:
                h = np.tanh(h @ w.T + b)
            w, b = layers[-1]
            outs.append(h @ w.T + b)
        return np.stack(outs)

    def predict(self, X: np.ndarray) -> Prediction:
        return combine(self.member_outputs(X))

    def apply(self, key_mean: float, key_var: float, contrast_mean: float, contrast_var: float, key: float) -> Applied:
        """Section 4.4 with this model's prior sigmas and contrast limit: the exposure in full, the contrast shrunk."""
        return apply_rule(key_mean, key_var, contrast_mean, contrast_var, key, prior_key=self.prior_sigma_key,
                          prior_contrast=self.prior_sigma_contrast, contrast_limit=self.contrast_limit)


def jax_forward(params: list[Layer], z):
    """The MLP of section 4.2 in JAX (float64): tanh hidden layers, a linear output. ``z`` is the standardised input ``(n, 39)``."""
    h = z
    for w, b in params[:-1]:
        h = jnp.tanh(h @ w.T + b)
    w, b = params[-1]
    return h @ w.T + b


# ------------------------------------------------------------------------------------------------------------------
# Combination (section 4.2) and the rule (section 4.4)
# ------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Prediction:
    """The ensemble's combined outputs for a batch: the means and the mixture variances of both heads."""

    key_mean: np.ndarray
    key_var: np.ndarray
    contrast_mean: np.ndarray
    contrast_var: np.ndarray


def combine(outputs: np.ndarray) -> Prediction:
    """Combines raw member outputs ``(members, n, 4)``: the mean of the means, and the mean of the variances plus the population
    variance of the means (the mixture variance). A logvar is clamped to +-20 before ``exp``."""
    outputs = np.asarray(outputs, dtype=np.float64)
    key_mu, con_mu = outputs[..., 0], outputs[..., 2]
    key_v = np.exp(np.clip(outputs[..., 1], -LOGVAR_LIMIT, LOGVAR_LIMIT))
    con_v = np.exp(np.clip(outputs[..., 3], -LOGVAR_LIMIT, LOGVAR_LIMIT))
    km = key_mu.mean(axis=0)
    cm = con_mu.mean(axis=0)
    return Prediction(
        key_mean=km,
        key_var=key_v.mean(axis=0) + ((key_mu - km) ** 2).mean(axis=0),
        contrast_mean=cm,
        contrast_var=con_v.mean(axis=0) + ((con_mu - cm) ** 2).mean(axis=0),
    )


def round_tenth(x: float) -> float:
    """``(x * 10).round() / 10`` as Rust's ``f64::round`` computes it (half away from zero, exact on the tie)."""
    v = float(x) * 10.0
    a = abs(v)
    f = math.floor(a)
    r = f + 1.0 if a - f >= 0.5 else float(f)
    return math.copysign(r, v) / 10.0


def weight(prior_sigma: float, variance: float) -> float:
    """``w = s0^2 / (s0^2 + variance)``: 1 for a certain prediction, towards 0 for a doubtful one (section 4.4)."""
    s2 = prior_sigma * prior_sigma
    return s2 / (s2 + variance)


@dataclass(frozen=True)
class Applied:
    """The rule's output for one photo, before the highlight guard and the burst median (which are not part of ``autoset``)."""

    ev_raw: float  # EV_auto before the guard, unclamped
    ev: float  # clamped to +-EV_LIMIT
    contrast_raw: float  # clamped to +-contrast_limit, not rounded
    contrast: float  # rounded to 0.1
    ev_confidence: float  # w_ev
    contrast_confidence: float  # w_c


def apply_rule(key_mean: float, key_var: float, contrast_mean: float, contrast_var: float, key: float, *,
               prior_key: float = PRIOR_SIGMA_KEY, prior_contrast: float = PRIOR_SIGMA_CONTRAST,
               contrast_limit: float = CONTRAST_LIMIT) -> Applied:
    """Section 4.4 for one photo: ``EV = clamp(mu_key - key)`` (full rescue: the exposure is the model's, ``w_ev`` is only its
    reported confidence) and ``contrast = round_0.1(clamp(w_c mu_contrast))``."""
    w_ev = weight(prior_key, key_var)
    ev_raw = key_mean - key
    ev = min(max(ev_raw, -EV_LIMIT), EV_LIMIT)
    w_c = weight(prior_contrast, contrast_var)
    c_raw = min(max(w_c * contrast_mean, -contrast_limit), contrast_limit)
    return Applied(ev_raw=ev_raw, ev=ev, contrast_raw=c_raw, contrast=round_tenth(c_raw),
                   ev_confidence=w_ev, contrast_confidence=w_c)

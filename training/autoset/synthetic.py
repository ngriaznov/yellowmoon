"""Synthetic dataset rows for the tests and for the test model (docs/18 section 6 row schema). Not training data for the shipped model.

Each scene has latent variables ``z1`` and ``z2``, standard normal. The scene luminance relative to its key is normal in log2 with
median ``mu = -0.3 + 0.25 z2`` and spread ``sigma = 1.1 exp(0.15 z1)`` (with a little jitter per file). The true key target is
``0.1 + 0.35 z2 - 0.25 z1`` plus noise, so the planted exposure label is ``L = key_target - key``; the camera's JPEG is a pure gain
``2**L`` of the scene (optionally with a shoulder above 0.5, which the mid-tone band excludes). The planted contrast ``c`` is
``clip(16 z1 - 8 z2 + noise)``; the contrast table's display-luma quantiles scale with ``exp(0.008 c)`` around grey.

Augmentations obey section 3: the key moves by ``k``, ``clip_headroom`` by ``-k``, ``clipped_share`` rises for brighter scalings,
and everything relative to the key (and ``key_bv``) is unchanged. Augmentation rows carry no label tables.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from statistics import NormalDist

import numpy as np

from training.autoset import rows as rw

SEED = 20261018
BODIES = ("ILCE-7M4", "ILCE-7RM4", "ILCE-1", "ILCE-7C")
AUG_K = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)
FEATURE_P = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99, 0.995, 0.999)
MQ_P = (0.50, 0.90, 0.98, 0.995, 0.999)
GRID = np.linspace(0.001, 0.999, 999)
_NORMAL = NormalDist()


def _ndtri(p) -> np.ndarray:
    return np.array([_NORMAL.inv_cdf(float(x)) for x in np.atleast_1d(np.asarray(p, dtype=np.float64))])


_Z_GRID = _ndtri(GRID)
_Z_FEATURE = _ndtri(FEATURE_P)
_Z_MQ = _ndtri(MQ_P)
_Z_CONTRAST = _ndtri(rw.CONTRAST_P)


@dataclass(frozen=True)
class Truth:
    """What was planted for one file: the exposure label, the key target and the contrast."""

    level: float
    key_target: float
    contrast: float


def make_dataset(n_scenes: int = 40, *, seed: int = SEED, augment: bool = True,
                 shoulder: bool = False) -> tuple[list[dict], list[Truth | None]]:
    """``(rows, truth)``: the rows as JSON-ready dicts and, parallel to them, the planted truth (None for augmentations)."""
    rng = np.random.default_rng(seed)
    out_rows: list[dict] = []
    out_truth: list[Truth | None] = []
    for s in range(n_scenes):
        z1, z2 = (float(v) for v in rng.standard_normal(2))
        mu_scene = -0.3 + 0.25 * z2
        sigma_scene = 1.1 * math.exp(0.15 * z1)
        c_scene = min(max(16.0 * z1 - 8.0 * z2 + float(rng.normal(0.0, 2.0)), -40.0), 40.0)
        body = BODIES[s % len(BODIES)]
        t0 = datetime(2024, 5, 1, 9, 0, 0) + timedelta(minutes=23 * s)
        for f in range(1 + int(rng.integers(0, 2))):
            name = f"synthetic-{s:03d}-{f}.ARW"
            sha = hashlib.sha256(f"{seed}:{s}:{f}".encode()).hexdigest()
            m = float(rng.normal(-0.5, 0.5))
            mu = mu_scene + float(rng.normal(0.0, 0.02))
            sigma = sigma_scene * math.exp(float(rng.normal(0.0, 0.02)))
            key_target = 0.1 + 0.35 * z2 - 0.25 * z1 + float(rng.normal(0.0, 0.03))
            level = key_target - m
            bv_known = bool(rng.random() < 0.9)
            key_bv = (float(rng.uniform(4.0, 12.0)) + m) if bv_known else 0.0
            headroom = 1.5 + 0.3 * z1 + float(rng.normal(0.0, 0.02))
            clipped = min(max(0.01 * math.exp(z1), 0.0), 1.0)
            probs = np.exp(rng.normal(0.0, 1.0, 4))
            probs = probs / probs.sum()
            iso = int(rng.choice([100, 200, 400, 800, 1600, 3200]))
            flash = int(rng.random() < 0.1)
            flagged = bool(rng.random() < 0.15)
            ec = float(rng.choice([0.0, 0.0, 0.0, 0.3, -0.7]))
            c_true = min(max(c_scene + float(rng.normal(0.0, 0.5)), -40.0), 40.0)
            features = [
                key_bv, float(bv_known),
                *(mu + sigma * z for z in _Z_FEATURE),
                *(mu + 0.25 + sigma * z for z in _Z_MQ),
                headroom, clipped, 0.2 * z1, 0.1 * z2,
                min(max(0.05 + 0.02 * z1, 0.0), 1.0), min(max(0.05 - 0.02 * z2, 0.0), 1.0),
                *(float(p) for p in probs),
                float(rng.uniform(120.0, 350.0)), float(rng.normal(0.0, 0.5)), float(rng.uniform(0.3, 1.0)),
                math.log2(iso / 100), float(flash),
            ]
            meta = {
                "dro": "advanced auto" if flagged else "off",
                "creative_style": "vivid" if flagged else "standard",
                "hdr": "off", "flash": flash, "metering": "multi", "ec": ec, "iso": iso,
                "exposure_time_s": 1.0 / 125.0, "f_number": 5.6,
            }
            scene_q = [0.18 * 2.0 ** (m + mu + sigma * z) for z in _Z_GRID]
            gain = 2.0**level
            jpeg_q = [gain * v for v in scene_q]
            if shoulder:
                jpeg_q = [v if v <= 0.5 else 0.5 + 0.5 * (v - 0.5) for v in jpeg_q]
            display = [0.5 + 0.2 * z * math.exp(0.1 * z1) for z in _Z_CONTRAST]
            ours = [[0.5 + (b - 0.5) * math.exp(0.008 * c) for b in display] for c in rw.CONTRAST_GRID]
            jpeg_display = [0.5 + (b - 0.5) * math.exp(0.008 * c_true) for b in display]
            base = {
                "file": name, "sha256": sha, "source": "synthetic", "licence": "Apache-2.0", "body": body,
                "capture_time": (t0 + timedelta(seconds=4 * f)).strftime(rw.TIME_FORMAT),
                "features": [float(x) for x in features], "key": m, "meta": meta,
            }
            out_rows.append({**base, "aug_k": 0.0,
                             "g3": {"grid": rw.G3_GRID, "scene_q": scene_q, "jpeg_q": jpeg_q},
                             "label_ev": level,
                             "contrast_fit": {"grid": list(rw.CONTRAST_GRID), "p": list(rw.CONTRAST_P),
                                              "ours": ours, "jpeg": jpeg_display}})
            out_truth.append(Truth(level, key_target, c_true))
            if not augment:
                continue
            for k in AUG_K:
                feats = list(base["features"])
                feats[24] = headroom - k
                feats[25] = clipped if k <= 0.0 else min(1.0, clipped + 0.02 * k)
                out_rows.append({**base, "aug_k": k, "features": feats, "key": m + k,
                                 "g3": None, "label_ev": None, "contrast_fit": None})
                out_truth.append(None)
    return out_rows, out_truth


def make_rows(n_scenes: int = 40, *, seed: int = SEED, augment: bool = True, shoulder: bool = False) -> list[dict]:
    """The rows of :func:`make_dataset` without the truth."""
    return make_dataset(n_scenes, seed=seed, augment=augment, shoulder=shoulder)[0]


def write_jsonl(rows_: list[dict], path: Path) -> None:
    """Writes one compact JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows_:
            fh.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")

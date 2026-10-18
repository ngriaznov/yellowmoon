"""The training table of ``autoset`` (docs/18 section 6): labels per row, augmentations inheriting their file's labels, and splits.

``prepare`` computes the labels of every file from its tables, checks the ``label_ev`` written by the dump against them (they must
agree to :data:`LABEL_TOLERANCE`; a disagreement is an error, not a row to drop), and gives each augmentation its file's
``key_target`` and contrast label (section 6.3: "the augmentation rows enter with the same ``key_target``").

Splits are by scene (section 6.3): every file and augmentation of a scene lands on the same side.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from training.autoset.labels import contrast_label, exposure_label, is_flagged, is_normal
from training.autoset.rows import FEATURE_COUNT, ReadResult, Row, assign_scenes

#: The dump's ``label_ev`` must reproduce the training code's level to this tolerance (docs/18 section 6).
LABEL_TOLERANCE = 1e-9


class LabelMismatch(ValueError):
    """The dump's ``label_ev`` and the training code's exposure label disagree (or only one of them exists)."""


@dataclass(eq=False)
class Prepared:
    """Arrays over all usable rows (files and augmentations, in the order of the file).

    ``key_target``, ``key_var``, ``contrast`` and ``contrast_var`` are NaN where the row has no label; ``key_ok`` and ``con_ok``
    say where they are present. ``level = key_target - key`` is the exposure the model is asked for (EV).
    """

    X: np.ndarray
    key: np.ndarray
    aug_k: np.ndarray
    is_aug: np.ndarray
    base: np.ndarray
    scene: np.ndarray
    file: list[str]
    body: list[str | None]
    source: list[str | None]
    licence: list[str | None]
    flagged: np.ndarray
    normal: np.ndarray
    ec: np.ndarray
    key_target: np.ndarray
    key_var: np.ndarray
    key_ok: np.ndarray
    contrast: np.ndarray
    contrast_var: np.ndarray
    con_ok: np.ndarray
    level: np.ndarray
    counts: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.X.shape[0])


def _agrees(label_ev: float | None, level: float) -> bool:
    if label_ev is None:
        return not math.isfinite(level)
    return math.isfinite(level) and abs(label_ev - level) <= LABEL_TOLERANCE


def prepare(read: ReadResult) -> Prepared:
    """Labels and arrays for the usable rows of ``read`` (raises :class:`LabelMismatch` when the dump and the code disagree)."""
    rows: list[Row] = read.rows
    n = len(rows)
    scene = assign_scenes(rows)
    base_of = {row.identity: i for i, row in enumerate(rows) if row.is_base}
    base = np.array([i if row.is_base else base_of[row.identity] for i, row in enumerate(rows)], dtype=np.int64)
    X = np.stack([row.features for row in rows]) if n else np.zeros((0, FEATURE_COUNT))
    key = np.array([row.key for row in rows], dtype=np.float64)
    aug_k = np.array([row.aug_k for row in rows], dtype=np.float64)
    is_aug = aug_k != 0.0

    key_target = np.full(n, math.nan)
    key_var = np.full(n, math.nan)
    contrast = np.full(n, math.nan)
    contrast_var = np.full(n, math.nan)
    mismatches: list[str] = []
    for i, row in enumerate(rows):
        if row.is_base:
            level, var = exposure_label(row)
            if row.g3 is not None and not _agrees(row.label_ev, level):
                mismatches.append(f"{row.file}:{row.line} label_ev={row.label_ev} computed={level!r}")
            if math.isfinite(level):
                key_target[i] = level + row.key  # section 6.3: key_target = L + key
                key_var[i] = var
            c, c_var = contrast_label(row)
            if math.isfinite(c):
                contrast[i] = c
                contrast_var[i] = c_var
    if mismatches:
        raise LabelMismatch(f"{len(mismatches)} label_ev values differ from the training code's level (tolerance "
                            f"{LABEL_TOLERANCE}); first: {mismatches[:3]}")
    for i in range(n):
        j = base[i]
        if j != i:
            key_target[i] = key_target[j]
            key_var[i] = key_var[j]
            contrast[i] = contrast[j]
            contrast_var[i] = contrast_var[j]

    base_meta = [rows[j].meta for j in base]
    flagged = np.array([is_flagged(m) for m in base_meta], dtype=bool)
    normal = np.array([is_normal(m) for m in base_meta], dtype=bool)
    ec = np.array([_ec(m) for m in base_meta], dtype=np.float64)
    key_ok = np.isfinite(key_target)
    con_ok = np.isfinite(contrast)
    base_row = ~is_aug
    counts = {
        "rows": n,
        "files": int(base_row.sum()),
        "augmentations": int(is_aug.sum()),
        "scenes": int(len(np.unique(scene))) if n else 0,
        "files_with_key_label": int((key_ok & base_row).sum()),
        "files_with_contrast_label": int((con_ok & base_row).sum()),
        "files_normal": int((normal & base_row).sum()),
        "files_flagged": int((flagged & base_row).sum()),
        "rows_with_key_label": int(key_ok.sum()),
        "rows_with_contrast_label": int(con_ok.sum()),
    }
    return Prepared(
        X=X, key=key, aug_k=aug_k, is_aug=is_aug, base=base, scene=scene,
        file=[row.file for row in rows], body=[row.body for row in rows],
        source=[row.source for row in rows], licence=[row.licence for row in rows],
        flagged=flagged, normal=normal, ec=ec,
        key_target=key_target, key_var=key_var, key_ok=key_ok,
        contrast=contrast, contrast_var=contrast_var, con_ok=con_ok,
        level=key_target - key, counts=counts,
    )


def _ec(meta: dict) -> float:
    ec = meta.get("ec")
    if isinstance(ec, bool) or not isinstance(ec, (int, float)):
        return math.nan
    return float(ec)


def labelled_scenes(prep: Prepared) -> np.ndarray:
    """The scenes with at least one row that has a key or a contrast label, sorted."""
    return np.unique(prep.scene[prep.key_ok | prep.con_ok])


def rows_of(prep: Prepared, scenes: np.ndarray) -> np.ndarray:
    """Indices of the rows of ``scenes`` (sorted)."""
    return np.nonzero(np.isin(prep.scene, np.asarray(scenes)))[0]


def holdout_split(scenes: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """``(train_scenes, holdout_scenes)``: about ``fraction`` of the scenes held out, chosen by a seeded permutation.

    With no scene to spare (``fraction <= 0`` or fewer than two scenes) the holdout is empty.
    """
    scenes = np.sort(np.asarray(scenes, dtype=np.int64))
    n = len(scenes)
    if fraction <= 0.0 or n < 2:
        return scenes, np.array([], dtype=np.int64)
    n_hold = min(max(1, int(round(fraction * n))), n - 1)
    perm = np.random.default_rng(seed).permutation(scenes)
    return np.sort(perm[n_hold:]), np.sort(perm[:n_hold])


def kfold_splits(scenes: np.ndarray, k: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """``k`` folds of scenes (balanced by a seeded permutation): ``(train_scenes, test_scenes)`` for each fold."""
    scenes = np.sort(np.asarray(scenes, dtype=np.int64))
    if k < 2 or k > len(scenes):
        raise ValueError(f"k-fold needs 2 <= k <= {len(scenes)} scenes, got k={k}")
    perm = np.random.default_rng(seed).permutation(scenes)
    folds = [perm[i::k] for i in range(k)]
    out = []
    for f in range(k):
        test = np.sort(folds[f])
        train = np.sort(np.concatenate([folds[g] for g in range(k) if g != f]))
        out.append((train, test))
    return out

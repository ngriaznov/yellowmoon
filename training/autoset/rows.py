"""Dataset rows of ``autoset`` (docs/18 section 6): reading, validation and grouping into scenes.

One JSON object per line. A row with ``aug_k == 0`` is a file and carries its label tables (``g3``, ``contrast_fit``,
``label_ev``). A row with ``aug_k != 0`` is the same file scaled by ``2 ** aug_k`` (section 3): it carries its own features and
``key`` (the file's key plus ``aug_k``), and no label tables; its labels are the file's (``dataset.prepare``).

Rows that cannot be used are dropped and counted by reason in ``ReadResult.rejected``. Nothing is repaired silently.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

FEATURE_COUNT = 39
G3_GRID = "linspace(0.001, 0.999, 999)"
G3_COUNT = 999
CONTRAST_GRID = (-40.0, -30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0)
CONTRAST_P = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
#: Two files of one body whose capture times are at most this far apart (chained) are one scene.
SCENE_GAP_S = 10.0
#: Files whose SHA-256 agree in this many leading hex digits are one scene (the raw.pixls.us duplicate groups).
SHA_PREFIX_LEN = 16
TIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
REQUIRED = ("file", "sha256", "aug_k", "features", "key")


@dataclass(frozen=True, eq=False)
class G3Tables:
    """The exposure label's quantile tables, both on ``linspace(0.001, 0.999, 999)``."""

    scene_q: np.ndarray  # scene-linear AWG4 luminance at EV 0, area-resized to the JPEG
    jpeg_q: np.ndarray  # the camera JPEG's linear luminance


@dataclass(frozen=True, eq=False)
class ContrastFit:
    """The contrast table (section 6.2): display-luma quantiles of our render at each grid contrast, and of the JPEG."""

    ours: np.ndarray  # (len(CONTRAST_GRID), len(CONTRAST_P))
    jpeg: np.ndarray  # (len(CONTRAST_P),)


@dataclass(frozen=True, eq=False)
class Row:
    """One validated row. ``features`` is float64 of length 39; ``capture_time`` is naive local time."""

    line: int
    file: str
    sha256: str
    source: str | None
    licence: str | None
    body: str | None
    capture_time: datetime | None
    aug_k: float
    features: np.ndarray
    key: float
    meta: dict
    g3: G3Tables | None
    label_ev: float | None
    contrast_fit: ContrastFit | None

    @property
    def is_base(self) -> bool:
        """True for a file (``aug_k == 0``), false for an augmentation of it."""
        return self.aug_k == 0.0

    @property
    def identity(self) -> tuple[str, str]:
        """What joins an augmentation to its file: the file name and its SHA-256."""
        return (self.file, self.sha256)


class RowError(ValueError):
    """A row that is not used. ``reason`` is the short tag counted in :attr:`ReadResult.rejected`."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass
class ReadResult:
    """The usable rows of one file, with what was dropped and why."""

    rows: list[Row] = field(default_factory=list)
    lines: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    examples: dict[str, list[str]] = field(default_factory=dict)

    def reject(self, reason: str, where: str) -> None:
        """Counts a dropped row; the first three places of each reason are kept as examples."""
        self.rejected[reason] = self.rejected.get(reason, 0) + 1
        seen = self.examples.setdefault(reason, [])
        if len(seen) < 3:
            seen.append(where)


# ------------------------------------------------------------------------------------------------------------------
# Field validation
# ------------------------------------------------------------------------------------------------------------------


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RowError("bad_type", f"{name} is not a number")
    return float(value)


def _finite(value: Any, name: str, reason: str) -> float:
    x = _number(value, name)
    if not math.isfinite(x):
        raise RowError(reason, f"{name} is not finite")
    return x


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise RowError("bad_type", f"{name} is not a string")
    return value


def _optional_text(value: Any, name: str) -> str | None:
    return None if value is None else _text(value, name)


def _capture_time(value: Any) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise RowError("bad_capture_time", "capture_time is not a string")
    try:
        return datetime.strptime(value, TIME_FORMAT)
    except ValueError as exc:
        raise RowError("bad_capture_time", f"capture_time {value!r} is not {TIME_FORMAT}") from exc


def _features(value: Any) -> np.ndarray:
    if not isinstance(value, list) or len(value) != FEATURE_COUNT:
        raise RowError("features_length", f"features must be a list of {FEATURE_COUNT} numbers")
    arr = np.array([_number(v, "features") for v in value], dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise RowError("non_finite_features", "a feature is not finite")
    return arr


def _table(value: Any, n: int, name: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != n:
        raise RowError("bad_g3", f"{name} must be a list of {n} numbers")
    arr = np.array([_number(v, name) for v in value], dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise RowError("non_finite_g3", f"{name} has a value that is not finite")
    return arr


def _matrix(value: Any, rows: int, cols: int, name: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != rows:
        raise RowError("bad_contrast_fit", f"{name} must have {rows} rows")
    out = np.empty((rows, cols), dtype=np.float64)
    for i, row in enumerate(value):
        out[i] = _table(row, cols, f"{name}[{i}]").astype(np.float64)
    return out


def _g3(value: Any) -> G3Tables | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise RowError("bad_g3", "g3 is not an object")
    grid = value.get("grid", G3_GRID)
    if grid != G3_GRID:
        raise RowError("bad_g3", f"g3 grid is {grid!r}, expected {G3_GRID!r}")
    return G3Tables(
        scene_q=_table(value.get("scene_q"), G3_COUNT, "g3.scene_q"),
        jpeg_q=_table(value.get("jpeg_q"), G3_COUNT, "g3.jpeg_q"),
    )


def _same_numbers(value: Any, expected: tuple[float, ...]) -> bool:
    return isinstance(value, list) and len(value) == len(expected) and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) and float(v) == e for v, e in zip(value, expected, strict=True)
    )


def _contrast(value: Any) -> ContrastFit | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise RowError("bad_contrast_fit", "contrast_fit is not an object")
    if not _same_numbers(value.get("grid", list(CONTRAST_GRID)), CONTRAST_GRID):
        raise RowError("bad_contrast_fit", "contrast_fit grid differs from the canonical grid")
    if not _same_numbers(value.get("p", list(CONTRAST_P)), CONTRAST_P):
        raise RowError("bad_contrast_fit", "contrast_fit p differs from the canonical probabilities")
    return ContrastFit(
        ours=_matrix(value.get("ours"), len(CONTRAST_GRID), len(CONTRAST_P), "contrast_fit.ours"),
        jpeg=_table(value.get("jpeg"), len(CONTRAST_P), "contrast_fit.jpeg"),
    )


def parse_row(obj: Any, line: int = 0) -> Row:
    """Validates one decoded row (raises :class:`RowError` with a reason tag when it cannot be used)."""
    if not isinstance(obj, dict):
        raise RowError("not_object", "a row is a JSON object")
    for name in REQUIRED:
        if name not in obj:
            raise RowError(f"missing_{name}", f"the row has no {name}")
    aug_k = _finite(obj["aug_k"], "aug_k", "non_finite_aug_k")
    features = _features(obj["features"])
    key = _finite(obj["key"], "key", "non_finite_key")
    meta = obj.get("meta")
    if meta is None:
        meta = {}
    if not isinstance(meta, dict):
        raise RowError("bad_meta", "meta is not an object")
    label_ev = None if obj.get("label_ev") is None else _finite(obj["label_ev"], "label_ev", "non_finite_label")
    g3 = _g3(obj.get("g3"))
    contrast_fit = _contrast(obj.get("contrast_fit"))
    if aug_k != 0.0 and (g3 is not None or contrast_fit is not None or label_ev is not None):
        raise RowError("label_on_augmentation", "an augmentation row carries label tables")
    if g3 is None and label_ev is not None:
        raise RowError("label_without_tables", "label_ev without g3 tables")
    return Row(
        line=line,
        file=_text(obj["file"], "file"),
        sha256=_text(obj["sha256"], "sha256"),
        source=_optional_text(obj.get("source"), "source"),
        licence=_optional_text(obj.get("licence"), "licence"),
        body=_optional_text(obj.get("body"), "body"),
        capture_time=_capture_time(obj.get("capture_time")),
        aug_k=aug_k,
        features=features,
        key=key,
        meta=meta,
        g3=g3,
        label_ev=label_ev,
        contrast_fit=contrast_fit,
    )


# ------------------------------------------------------------------------------------------------------------------
# Reading
# ------------------------------------------------------------------------------------------------------------------


def _settle(result: ReadResult, parsed: list[tuple[Row | None, str]]) -> ReadResult:
    """Keeps the first base row of each file, and the augmentations whose file is kept; counts the rest."""
    bases: set[tuple[str, str]] = set()
    for row, where in parsed:
        if row is None:
            continue
        if row.is_base:
            if row.identity in bases:
                result.reject("duplicate_file", where)
                continue
            bases.add(row.identity)
        result.rows.append(row)
    kept: list[Row] = []
    for row in result.rows:
        if not row.is_base and row.identity not in bases:
            result.reject("orphan_augmentation", f"{row.file}:{row.line}")
            continue
        kept.append(row)
    result.rows = kept
    return result


def read_objects(objects: Iterable[Any], *, label: str = "rows") -> ReadResult:
    """The same validation as :func:`read_rows`, for decoded objects (tests and the synthetic generator)."""
    result = ReadResult()
    parsed: list[tuple[Row | None, str]] = []
    for number, obj in enumerate(objects, start=1):
        result.lines += 1
        where = f"{label}:{number}"
        try:
            parsed.append((parse_row(obj, number), where))
        except RowError as exc:
            result.reject(exc.reason, f"{where} ({exc})")
            parsed.append((None, where))
    return _settle(result, parsed)


def read_rows(path: Path | str) -> ReadResult:
    """Reads a JSON-lines file of rows (blank lines are skipped); see the module docstring for what is dropped."""
    path = Path(path)
    result = ReadResult()
    parsed: list[tuple[Row | None, str]] = []
    with open(path, encoding="utf-8") as fh:
        for number, text in enumerate(fh, start=1):
            if not text.strip():
                continue
            result.lines += 1
            where = f"{path.name}:{number}"
            try:
                obj = json.loads(text)
            except ValueError:
                result.reject("not_json", where)
                parsed.append((None, where))
                continue
            try:
                parsed.append((parse_row(obj, number), where))
            except RowError as exc:
                result.reject(exc.reason, f"{where} ({exc})")
                parsed.append((None, where))
    return _settle(result, parsed)


# ------------------------------------------------------------------------------------------------------------------
# Scenes
# ------------------------------------------------------------------------------------------------------------------


def assign_scenes(rows: list[Row]) -> np.ndarray:
    """A scene index for every row (0, 1, ... in order of first appearance).

    Files of one body whose capture times are at most :data:`SCENE_GAP_S` apart (chained in time order) are one scene;
    files with no body or no capture time are their own scene; files whose SHA-256 share their first
    :data:`SHA_PREFIX_LEN` digits are one scene; an augmentation joins the scene of its file.
    """
    n = len(rows)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    base_index = {row.identity: i for i, row in enumerate(rows) if row.is_base}
    for i, row in enumerate(rows):
        if not row.is_base and row.identity in base_index:
            union(i, base_index[row.identity])
    timed = [i for i, row in enumerate(rows) if row.is_base and row.body is not None and row.capture_time is not None]
    timed.sort(key=lambda i: (rows[i].body, rows[i].capture_time, rows[i].file, i))
    for a, b in zip(timed, timed[1:], strict=False):
        gap = (rows[b].capture_time - rows[a].capture_time).total_seconds()
        if rows[a].body == rows[b].body and gap <= SCENE_GAP_S:
            union(a, b)
    by_prefix: dict[str, int] = {}
    for i, row in enumerate(rows):
        if row.is_base:
            prefix = row.sha256[:SHA_PREFIX_LEN]
            if prefix in by_prefix:
                union(i, by_prefix[prefix])
            else:
                by_prefix[prefix] = i
    ids: dict[int, int] = {}
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        out[i] = ids.setdefault(find(i), len(ids))
    return out

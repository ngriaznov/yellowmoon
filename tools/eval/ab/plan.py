"""The photos, the strata and the trial plan of the blind A/B (docs/18 section 8, gate 5).

Stdlib only. The plan is made from file names, folder names and a seed; no picture is read here. The page sees only
opaque picture paths (`page.py`). The map from a trial to its variants is in `plan.json`, which the page never reads.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

DEFAULT_SEED = 20261018
OTHER_STRATUM = "other"
PLAN_FORMAT = "yellowmoon-ab-plan"
#: Every picture the plan can show. `autoset` is the picture under test; the others are what it is compared with.
VARIANTS: Tuple[str, ...] = ("autoset", "rule", "camera", "lightroom", "autoset-neutral")
#: Comparison key (used in plan and score files) -> the variant that `autoset` is compared with.
COMPARISONS: Mapping[str, str] = {
    "rule": "rule",
    "camera": "camera",
    "lightroom": "lightroom",
    "neutral": "autoset-neutral",
}
#: The comparisons that decide gate 5. The neutral comparison is reported only.
GATED: Tuple[str, ...] = ("rule", "camera", "lightroom")
#: The folder under the out folder that holds the pictures; one subfolder per variant, named by its code.
PICTURES_DIR = "pictures"
JPEG_SUFFIXES = (".jpg", ".jpeg")
_UNSAFE_IN_NAME = re.compile(r"[^A-Za-z0-9_.-]")


@dataclass(frozen=True)
class Photo:
    """A raw file found in the photos folder, with the stratum its folder gives it."""

    stem: str
    stratum: str
    path: Path


@dataclass(frozen=True)
class PlanPhoto:
    """A photo that is in the test: its number (names its pictures), stem, stratum and Lightroom match."""

    number: int
    stem: str
    stratum: str
    lightroom: bool


@dataclass(frozen=True)
class Trial:
    """One comparison of one photo. `left` and `right` are variant names; the page never gets them."""

    id: int
    number: int
    stem: str
    stratum: str
    comparison: str
    left: str
    right: str


def find_photos(folder: Path) -> List[Photo]:
    """Every ARW under `folder`, at any depth, sorted by stratum and name.

    The first folder inside `folder` names the stratum of the files below it; files directly in `folder` are
    `other`. Hidden files and folders are skipped, which includes the `._` copies macOS makes on other drives.
    """
    found: List[Photo] = []
    root = str(folder)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(name for name in dirnames if not name.startswith("."))
        parts = Path(dirpath).relative_to(root).parts
        stratum = parts[0] if parts else OTHER_STRATUM
        for name in sorted(filenames):
            stem, ext = os.path.splitext(name)
            if name.startswith(".") or ext.lower() != ".arw":
                continue
            found.append(Photo(stem=stem, stratum=stratum, path=Path(dirpath) / name))
    return sorted(found, key=lambda photo: (photo.stratum, photo.stem.lower(), str(photo.path)))


def match_lightroom(photos: Sequence[Photo], folder: Path) -> Tuple[Dict[int, Path], List[str]]:
    """The Lightroom export of each photo, matched by file stem (case-insensitive) in the top level of `folder`.

    Returns a map from the photo's index in `photos` to its JPEG, and notes for the summary. A stem that more than
    one raw file has is not matched: the export could belong to either of them.
    """
    exports: Dict[str, List[Path]] = {}
    for name in sorted(os.listdir(folder)):
        path = folder / name
        stem, ext = os.path.splitext(name)
        if name.startswith(".") or ext.lower() not in JPEG_SUFFIXES or not path.is_file():
            continue
        exports.setdefault(stem.lower(), []).append(path)
    raw_counts = Counter(photo.stem.lower() for photo in photos)
    matches: Dict[int, Path] = {}
    notes: List[str] = []
    for index, photo in enumerate(photos):
        key = photo.stem.lower()
        candidates = exports.get(key)
        if not candidates:
            continue
        if raw_counts[key] > 1:
            notes.append(f"{photo.stem}: more than one raw file has this name, so it has no Lightroom match")
            continue
        if len(candidates) > 1:
            notes.append(f"{photo.stem}: {len(candidates)} JPEGs have this name, {candidates[0].name} is used")
        matches[index] = candidates[0]
    return matches, notes


def variant_code(seed: int, variant: str) -> str:
    """The folder code of a variant. It keeps the variant names out of the page's paths; it is not a secret."""
    return hashlib.sha256(f"yellowmoon-ab/{seed}/{variant}".encode()).hexdigest()[:8]


def picture_name(number: int, stem: str) -> str:
    """The file name of a picture, without its extension: the photo's number, then its stem with unsafe characters
    replaced. The number keeps two photos with the same stem apart."""
    return f"{number:04d}_{_UNSAFE_IN_NAME.sub('_', stem)}"


def picture_relpath(seed: int, variant: str, number: int, stem: str) -> str:
    """Where a picture lives, relative to the out folder, with forward slashes (the page's paths)."""
    return f"{PICTURES_DIR}/{variant_code(seed, variant)}/{picture_name(number, stem)}.jpg"


def comparisons_of(photo: PlanPhoto) -> List[str]:
    """The comparisons a photo takes part in: Lightroom only when the photo has a Lightroom export."""
    return [name for name in COMPARISONS if name != "lightroom" or photo.lightroom]


def build_trials(photos: Sequence[PlanPhoto], seed: int) -> List[Trial]:
    """Every comparison of every photo, shuffled, with the sides chosen at random.

    Within each comparison the sides are balanced exactly (half the trials have `autoset` on the left, to within one
    when the count is odd) and then shuffled, so no comparison has a side bias. The trial ids are 1, 2, ... in the order
    shown. Everything is drawn from one `random.Random(seed)`, so the same photos and seed give the same plan.
    """
    rng = random.Random(seed)
    by_comparison: Dict[str, List[PlanPhoto]] = {name: [] for name in COMPARISONS}
    for photo in photos:
        for name in comparisons_of(photo):
            by_comparison[name].append(photo)
    drafts: List[Tuple[PlanPhoto, str, str, str]] = []
    for name, variant in COMPARISONS.items():
        items = by_comparison[name]
        left_count = len(items) // 2
        if len(items) % 2 and rng.random() < 0.5:
            left_count += 1  # the odd trial goes to either side, at random, so no side is favoured
        autoset_left = [True] * left_count + [False] * (len(items) - left_count)
        rng.shuffle(autoset_left)
        for photo, left_is_autoset in zip(items, autoset_left):
            if left_is_autoset:
                left, right = "autoset", variant
            else:
                left, right = variant, "autoset"
            drafts.append((photo, name, left, right))
    rng.shuffle(drafts)
    return [
        Trial(
            id=position,
            number=photo.number,
            stem=photo.stem,
            stratum=photo.stratum,
            comparison=name,
            left=left,
            right=right,
        )
        for position, (photo, name, left, right) in enumerate(drafts, start=1)
    ]


def build_plan(photos: Sequence[PlanPhoto], seed: int, provenance: Mapping[str, object]) -> Dict[str, object]:
    """The plan document (`plan.json`): the photos, their pictures, and the trials in the order the page shows them.

    `provenance` records what made the pictures (the `ym` version, the long edge, the model's hash). The plan id is a
    hash of the design and of the provenance, so a different render or a different set of photos gets a new id and
    the page keeps the answers of each plan apart.
    """
    trials = build_trials(photos, seed)
    photo_entries: List[Dict[str, object]] = []
    for photo in photos:
        pictures = {
            variant: picture_relpath(seed, variant, photo.number, photo.stem)
            for variant in VARIANTS
            if variant != "lightroom" or photo.lightroom
        }
        photo_entries.append(
            {
                "number": photo.number,
                "stem": photo.stem,
                "stratum": photo.stratum,
                "lightroom": photo.lightroom,
                "pictures": pictures,
            }
        )
    trial_entries: List[Dict[str, object]] = []
    for trial in trials:
        trial_entries.append(
            {
                "id": trial.id,
                "number": trial.number,
                "stem": trial.stem,
                "stratum": trial.stratum,
                "comparison": trial.comparison,
                "left": trial.left,
                "right": trial.right,
                "pictures": {
                    "left": picture_relpath(seed, trial.left, trial.number, trial.stem),
                    "right": picture_relpath(seed, trial.right, trial.number, trial.stem),
                },
            }
        )
    identity = {
        "seed": seed,
        "provenance": dict(provenance),
        "photos": [[p.number, p.stem, p.stratum, p.lightroom] for p in photos],
        "trials": [[t.id, t.comparison, t.left, t.right] for t in trials],
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return {
        "format": PLAN_FORMAT,
        "version": 1,
        "plan_id": digest.hexdigest()[:12],
        "seed": seed,
        "provenance": dict(provenance),
        "variants": {variant: {"code": variant_code(seed, variant)} for variant in VARIANTS},
        "photos": photo_entries,
        "trials": trial_entries,
    }


@dataclass
class Counts:
    """How many trials of each comparison a plan holds (for the summary)."""

    by_comparison: Dict[str, int] = field(default_factory=dict)
    by_stratum: Dict[str, int] = field(default_factory=dict)


def count_plan(plan: Mapping[str, object]) -> Counts:
    """Trials per comparison and photos per stratum of a plan document."""
    counts = Counts()
    for trial in plan["trials"]:  # type: ignore[index]
        name = trial["comparison"]
        counts.by_comparison[name] = counts.by_comparison.get(name, 0) + 1
    for photo in plan["photos"]:  # type: ignore[index]
        counts.by_stratum[photo["stratum"]] = counts.by_stratum.get(photo["stratum"], 0) + 1
    return counts


def plan_to_json(plan: Mapping[str, object]) -> str:
    """The plan as the file `plan.json` is written: readable, with a final newline."""
    return json.dumps(plan, indent=2, ensure_ascii=False) + "\n"

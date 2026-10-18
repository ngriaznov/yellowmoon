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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

DEFAULT_SEED = 20261018
OTHER_STRATUM = "other"
PLAN_FORMAT = "yellowmoon-ab-plan"
#: Every picture the plan can show. `autoset` is the picture under test; the others are what it is compared with.
VARIANTS: Tuple[str, ...] = ("autoset", "rule", "camera")
#: Comparison key (used in plan and score files) -> the variant that `autoset` is compared with.
COMPARISONS: Mapping[str, str] = {
    "rule": "rule",
    "camera": "camera",
}
#: The comparisons that decide gate 5.
GATED: Tuple[str, ...] = ("rule", "camera")
#: The folder under the out folder that holds the pictures; one subfolder per variant, named by its code.
PICTURES_DIR = "pictures"
_UNSAFE_IN_NAME = re.compile(r"[^A-Za-z0-9_.-]")


@dataclass(frozen=True)
class Photo:
    """A raw file found in the photos folder, with the stratum its folder gives it."""

    stem: str
    stratum: str
    path: Path


@dataclass(frozen=True)
class PlanPhoto:
    """A photo that is in the test: its number (names its pictures), stem and stratum, and the quarter turns that show
    its camera's JPEG upright (0 to 3, from the raw file's orientation)."""

    number: int
    stem: str
    stratum: str
    camera_turns: int = 0


def picture_turns(photo: PlanPhoto, variant: str) -> int:
    """The quarter turns clockwise that show a picture of `photo` upright on the page. A render is already upright,
    because `ym` applies the raw file's orientation, so it is 0. The camera's JPEG is stored in sensor orientation, so
    it takes the turns of its raw file."""
    return photo.camera_turns if variant == "camera" else 0


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


def build_trials(photos: Sequence[PlanPhoto], seed: int) -> List[Trial]:
    """Every comparison of every photo (two trials per photo), shuffled, with the sides chosen at random.

    Within each comparison the sides are balanced exactly (half the trials have `autoset` on the left, to within one
    when the count is odd) and then shuffled, so no comparison has a side bias. The trial ids are 1, 2, ... in the order
    shown. Everything is drawn from one `random.Random(seed)`, so the same photos and seed give the same plan.
    """
    rng = random.Random(seed)
    drafts: List[Tuple[PlanPhoto, str, str, str]] = []
    for name, variant in COMPARISONS.items():
        left_count = len(photos) // 2
        if len(photos) % 2 and rng.random() < 0.5:
            left_count += 1  # the odd trial goes to either side, at random, so no side is favoured
        autoset_left = [True] * left_count + [False] * (len(photos) - left_count)
        rng.shuffle(autoset_left)
        for photo, left_is_autoset in zip(photos, autoset_left):
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
    hash of the design, of the provenance and of the quarter turns of every picture, so a different render, a
    different set of photos or a different turn gets a new id and the page keeps the answers of each plan apart.

    Each photo entry and each trial side also records the quarter turns of its picture (`turns`). The page reads them
    from `plan.js`; scoring does not need them.
    """
    trials = build_trials(photos, seed)
    by_number = {photo.number: photo for photo in photos}
    photo_entries: List[Dict[str, object]] = []
    for photo in photos:
        pictures = {variant: picture_relpath(seed, variant, photo.number, photo.stem) for variant in VARIANTS}
        turns = {variant: picture_turns(photo, variant) for variant in VARIANTS}
        photo_entries.append(
            {
                "number": photo.number,
                "stem": photo.stem,
                "stratum": photo.stratum,
                "pictures": pictures,
                "turns": turns,
            }
        )
    trial_entries: List[Dict[str, object]] = []
    for trial in trials:
        photo = by_number[trial.number]
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
                "turns": {
                    "left": picture_turns(photo, trial.left),
                    "right": picture_turns(photo, trial.right),
                },
            }
        )
    identity = {
        "seed": seed,
        "provenance": dict(provenance),
        "photos": [[p.number, p.stem, p.stratum, [picture_turns(p, v) for v in VARIANTS]] for p in photos],
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

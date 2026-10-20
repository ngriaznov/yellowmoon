"""The photos, the strata and the trial plan of the blind A/B of Apple Auto (docs/19 section 13).

Stdlib only. The plan is made from file names, folder names, the steps Apple's model answered and a seed; no picture
is read here. The page sees only opaque picture paths (`page.py`). The map from a trial to its variants is in
`plan.json`, which the page never reads.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

DEFAULT_SEED = 20261019
OTHER_STRATUM = "other"
PLAN_FORMAT = "yellowmoon-ab-plan"
#: 3: the steps of prompt version 2 (docs/19 section 4), any of which may be empty (null), and the instructions in the
#: provenance.
PLAN_VERSION = 3
#: Every picture the plan can show. `apple` (Apple Auto on) is the picture under test; the others are what it is
#: compared with: `rule` (Apple Auto off, Yellowmoon's own Auto) and `camera` (the camera's JPEG, with --with-camera).
VARIANTS: Tuple[str, ...] = ("apple", "rule", "camera")
#: The picture under test, on one side of every trial.
UNDER_TEST = "apple"
#: Comparison key (used in plan and score files) -> the variant that `apple` is compared with.
COMPARISONS: Mapping[str, str] = {
    "rule": "rule",
    "camera": "camera",
}
#: The comparisons that decide the gate of docs/19 section 13. The camera's JPEG is reported, not gated.
GATED: Tuple[str, ...] = ("rule",)
#: The numeric steps of Apple's answer (docs/19 sections 4 and 5, prompt version 2), in the order of the table, as `ym`
#: prints them (`white` is the warmth), and the range of each step.
STEP_FIELDS: Tuple[str, ...] = (
    "exposure",
    "highlights",
    "shadows",
    "blacks",
    "contrast",
    "white",
    "tint",
    "saturation",
)
STEP_RANGE: Mapping[str, Tuple[int, int]] = {name: (-3, 3) if name == "exposure" else (-2, 2) for name in STEP_FIELDS}
#: The colours of the shadows and the highlights: each colour field, the field of its strength and the colours it may
#: name (docs/19 section 5).
COLOR_FIELDS: Mapping[str, Tuple[str, Tuple[str, ...]]] = {
    "shadow_color": ("shadow_color_strength", ("none", "teal", "blue", "green", "amber", "magenta")),
    "highlight_color": ("highlight_color_strength", ("none", "warm", "gold", "cool", "pink")),
}
#: The range of a colour's strength: subtle (1) to strong (3).
STRENGTH_RANGE: Tuple[int, int] = (1, 3)
#: Every field of an answer's steps as `ym` prints them, in order: the numeric steps, then each colour and its strength.
ANSWER_FIELDS: Tuple[str, ...] = STEP_FIELDS + (
    "shadow_color",
    "shadow_color_strength",
    "highlight_color",
    "highlight_color_strength",
)
#: The folder under the out folder that holds the pictures; one subfolder per variant, named by its code.
PICTURES_DIR = "pictures"
_UNSAFE_IN_NAME = re.compile(r"[^A-Za-z0-9_.-]")

#: One field of an answer: a step or a strength (int), a colour (str), or None where the model left it empty, which
#: sets nothing (the control keeps Yellowmoon's own Auto value).
StepValue = Union[int, str, None]
#: A photo's steps, in the order of ANSWER_FIELDS.
Steps = Tuple[StepValue, ...]
NO_STEPS: Steps = (None,) * len(ANSWER_FIELDS)


@dataclass(frozen=True)
class Photo:
    """A raw file found in the photos folder, with the stratum its folder gives it."""

    stem: str
    stratum: str
    path: Path


@dataclass(frozen=True)
class PlanPhoto:
    """A photo that is in the test: its number (names its pictures), stem and stratum, the quarter turns that show its
    camera's JPEG upright (0 to 3, from the raw file's orientation), whether it has a camera picture (--with-camera),
    and the steps, scene and plan of Apple's answer."""

    number: int
    stem: str
    stratum: str
    camera_turns: int = 0
    has_camera: bool = False
    steps: Steps = NO_STEPS
    scene: str = ""
    plan: str = ""


def steps_dict(steps: Sequence[StepValue]) -> Dict[str, StepValue]:
    """The steps as the object `ym` prints (`{"exposure": 1, "highlights": null, "shadow_color": "teal", ...}`)."""
    return {name: value for name, value in zip(ANSWER_FIELDS, steps)}


def step_of(steps: Sequence[StepValue], name: str) -> StepValue:
    """One field of a photo's steps, by its name in ANSWER_FIELDS."""
    return steps[ANSWER_FIELDS.index(name)]


def changed_fields(steps: Sequence[StepValue]) -> List[str]:
    """What an answer changes, in the order of the table: each numeric step that was set and is not keep, then each
    colour that was set, is not none and has a strength. A field left empty sets nothing, and a strength without a
    colour is ignored (docs/19 section 4), so an answer with none of these leaves the picture as the rule's."""
    changed = [name for name in STEP_FIELDS if isinstance(step_of(steps, name), int) and step_of(steps, name) != 0]
    for color, (strength, _) in COLOR_FIELDS.items():
        value, amount = step_of(steps, color), step_of(steps, strength)
        if isinstance(value, str) and value != "none" and isinstance(amount, int):
            changed.append(color)
    return changed


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


def comparison_photos(photos: Sequence[PlanPhoto], comparison: str) -> List[PlanPhoto]:
    """The photos that take part in a comparison: every photo against the rule, those with a camera picture against
    the camera's JPEG."""
    if comparison == "camera":
        return [photo for photo in photos if photo.has_camera]
    return list(photos)


def build_trials(photos: Sequence[PlanPhoto], seed: int) -> List[Trial]:
    """Every comparison of every photo (one trial against the rule, and one against the camera's JPEG when the photo
    has one), shuffled, with the sides chosen at random.

    Within each comparison the sides are balanced exactly (half the trials have `apple` on the left, to within one
    when the count is odd) and then shuffled, so no comparison has a side bias. The trial ids are 1, 2, ... in the order
    shown. Everything is drawn from one `random.Random(seed)`, so the same photos and seed give the same plan.
    """
    rng = random.Random(seed)
    drafts: List[Tuple[PlanPhoto, str, str, str]] = []
    for name, variant in COMPARISONS.items():
        members = comparison_photos(photos, name)
        left_count = len(members) // 2
        if len(members) % 2 and rng.random() < 0.5:
            left_count += 1  # the odd trial goes to either side, at random, so no side is favoured
        apple_left = [True] * left_count + [False] * (len(members) - left_count)
        rng.shuffle(apple_left)
        for photo, left_is_apple in zip(members, apple_left):
            if left_is_apple:
                left, right = UNDER_TEST, variant
            else:
                left, right = variant, UNDER_TEST
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


def build_plan(
    photos: Sequence[PlanPhoto],
    seed: int,
    provenance: Mapping[str, object],
    details: Optional[Mapping[int, Mapping[str, object]]] = None,
) -> Dict[str, object]:
    """The plan document (`plan.json`): the photos, their pictures, and the trials in the order the page shows them.

    `provenance` records what made the pictures (the `ym` version, the long edge, Apple's model, macOS, the prompt and
    table versions, and which instructions the model read with their digest). The plan id is a hash of the design, of
    the provenance, of every photo's steps and of the quarter turns of every picture, so a different render, another answer, a different set of photos or a different turn gets
    a new id and the page keeps the answers of each plan apart. `details` adds fields to a photo's entry (the Auto
    values of its renders, for the diagnostics); it is not part of the id.

    Each photo entry and each trial records Apple's steps, scene and plan (scoring reads them for the diagnostics; a
    step the model left empty is `null`) and each
    trial side the quarter turns of its picture (`turns`; the page reads them from `plan.js`).
    """
    trials = build_trials(photos, seed)
    by_number = {photo.number: photo for photo in photos}
    extra = details or {}
    photo_entries: List[Dict[str, object]] = []
    for photo in photos:
        shown = [v for v in VARIANTS if v != "camera" or photo.has_camera]
        entry: Dict[str, object] = {
            "number": photo.number,
            "stem": photo.stem,
            "stratum": photo.stratum,
            "steps": steps_dict(photo.steps),
            "scene": photo.scene,
            "plan": photo.plan,
            "pictures": {variant: picture_relpath(seed, variant, photo.number, photo.stem) for variant in shown},
            "turns": {variant: picture_turns(photo, variant) for variant in shown},
        }
        entry.update(extra.get(photo.number, {}))
        photo_entries.append(entry)
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
                "steps": steps_dict(photo.steps),
                "scene": photo.scene,
                "plan": photo.plan,
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
    comparisons = [name for name in COMPARISONS if comparison_photos(photos, name)]
    identity = {
        "version": PLAN_VERSION,
        "seed": seed,
        "provenance": dict(provenance),
        "photos": [
            [p.number, p.stem, p.stratum, list(p.steps), p.has_camera, [picture_turns(p, v) for v in VARIANTS]]
            for p in photos
        ],
        "trials": [[t.id, t.comparison, t.left, t.right] for t in trials],
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return {
        "format": PLAN_FORMAT,
        "version": PLAN_VERSION,
        "plan_id": digest.hexdigest()[:12],
        "seed": seed,
        "provenance": dict(provenance),
        "under_test": UNDER_TEST,
        "comparisons": comparisons,
        "gated": [name for name in GATED if name in comparisons],
        "variants": {variant: {"code": variant_code(seed, variant)} for variant in VARIANTS},
        "photos": photo_entries,
        "trials": trial_entries,
    }


@dataclass
class Counts:
    """How many trials of each comparison a plan holds, and its photos per stratum (for the summary)."""

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

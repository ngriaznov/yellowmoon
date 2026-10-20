"""The score of the blind A/B of Apple Auto: wins, losses and ties per comparison, the Wilson bound, the strata, the
gate of docs/19 section 13 and the diagnostics that help tune the table of docs/19 section 5.

The gate compares Apple Auto with the rule (Apple Auto off). Its thresholds are provisional: they are the numbers of
the parked `autoset` test (gate 5 of the old docs/18) until the owner agrees to them.

* the pooled win share of Apple Auto is at least 60 %, ties counting as half a win;
* the Wilson 95 % lower bound of the win share is above 50 %, with ties left out of the count (so the bound is over
  the decisive trials only);
* no stratum has a win share below 45 %;
* Apple Auto's picture is not marked "needs a fix" more often than the rule's.

The comparison with the camera's JPEG (--with-camera) is reported with the same numbers but is not part of the gate.
Stdlib only.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from page import RESULTS_FORMAT
from plan import (
    ANSWER_FIELDS,
    COLOR_FIELDS,
    COMPARISONS,
    GATED,
    PLAN_FORMAT,
    STEP_FIELDS,
    UNDER_TEST,
    VARIANTS,
    StepValue,
    Steps,
    changed_fields,
    step_of,
)

SCORE_FORMAT = "yellowmoon-ab-score"
SCORE_FILE = "score.json"
ANSWERS = ("left", "right", "same")
#: 95 % two-sided normal quantile for the Wilson interval.
Z95 = 1.959963984540054
SHARE_GATE = 0.60
LOWER_GATE = 0.50
STRATUM_GATE = 0.45
_EPS = 1e-12
GATE_NAME = "docs/19 section 13"
THRESHOLDS_NOTE = "provisional thresholds, docs/19 §13"
LABELS: Mapping[str, str] = {
    "rule": "Apple Auto vs the rule",
    "camera": "Apple Auto vs the camera's JPEG",
}
#: Words for each numeric step of an answer (docs/19 section 4): -3 ... +3 for exposure, -2 ... +2 for the others.
STEP_WORDS: Mapping[str, Mapping[int, str]] = {
    "exposure": {
        -3: "much darker",
        -2: "darker",
        -1: "slightly darker",
        1: "slightly brighter",
        2: "brighter",
        3: "much brighter",
    },
    "highlights": {
        -2: "much darker highlights",
        -1: "darker highlights",
        1: "brighter highlights",
        2: "much brighter highlights",
    },
    "shadows": {-2: "much darker shadows", -1: "darker shadows", 1: "brighter shadows", 2: "much brighter shadows"},
    "blacks": {-2: "much deeper blacks", -1: "deeper blacks", 1: "lifted blacks", 2: "much lifted blacks"},
    "contrast": {-2: "much less contrast", -1: "less contrast", 1: "more contrast", 2: "much more contrast"},
    "white": {-2: "much cooler", -1: "cooler", 1: "warmer", 2: "much warmer"},
    "tint": {-2: "much greener", -1: "greener", 1: "more magenta", 2: "much more magenta"},
    "saturation": {-2: "much less saturation", -1: "less saturation", 1: "more saturation", 2: "much more saturation"},
}
#: Words for a colour's strength (docs/19 section 4).
STRENGTH_WORDS: Mapping[int, str] = {1: "subtle", 2: "medium", 3: "strong"}
#: The tones each colour field colours, for the words ("teal shadows").
COLOR_TONES: Mapping[str, str] = {"shadow_color": "shadows", "highlight_color": "highlights"}
#: The direction groups of the diagnostics, in the order they are printed (the order of the table): (label, field,
#: sign).
DIRECTIONS: Tuple[Tuple[str, str, int], ...] = (
    ("brighter", "exposure", 1),
    ("darker", "exposure", -1),
    ("darker highlights", "highlights", -1),
    ("brighter highlights", "highlights", 1),
    ("brighter shadows", "shadows", 1),
    ("darker shadows", "shadows", -1),
    ("deeper blacks", "blacks", -1),
    ("lifted blacks", "blacks", 1),
    ("more contrast", "contrast", 1),
    ("less contrast", "contrast", -1),
    ("warmer", "white", 1),
    ("cooler", "white", -1),
    ("greener", "tint", -1),
    ("more magenta", "tint", 1),
    ("more saturation", "saturation", 1),
    ("less saturation", "saturation", -1),
)


class ScoreError(Exception):
    """The results or the plan cannot be scored. A usage error: the message says what to do."""


def wilson_lower(successes: int, trials: int) -> Optional[float]:
    """The lower end of the Wilson score interval at 95 % for `successes` out of `trials`; None with no trials."""
    if trials <= 0:
        return None
    if not 0 <= successes <= trials:
        raise ValueError(f"{successes} successes out of {trials} trials")
    p = successes / trials
    z2 = Z95 * Z95
    centre = p + z2 / (2 * trials)
    spread = Z95 * math.sqrt(p * (1 - p) / trials + z2 / (4 * trials * trials))
    return max(0.0, (centre - spread) / (1 + z2 / trials))


@dataclass
class Tally:
    """Outcomes of Apple Auto in some set of trials, and the needs-a-fix flags of both pictures."""

    wins: int = 0
    losses: int = 0
    ties: int = 0
    flags_apple: int = 0
    flags_other: int = 0

    def add(self, outcome: str, flag_apple: bool, flag_other: bool) -> None:
        """Counts one trial. `outcome` is "win", "loss" or "tie" for Apple Auto."""
        if outcome == "win":
            self.wins += 1
        elif outcome == "loss":
            self.losses += 1
        else:
            self.ties += 1
        self.flags_apple += int(flag_apple)
        self.flags_other += int(flag_other)

    @property
    def trials(self) -> int:
        return self.wins + self.losses + self.ties

    @property
    def decisive(self) -> int:
        return self.wins + self.losses

    @property
    def share(self) -> Optional[float]:
        """Pooled win share with each tie counted as half a win; None with no trials."""
        if self.trials == 0:
            return None
        return (self.wins + self.ties / 2) / self.trials

    @property
    def lower(self) -> Optional[float]:
        """The Wilson 95 % lower bound of the win share, over the decisive trials (ties left out)."""
        return wilson_lower(self.wins, self.decisive)

    def fix_rate(self, apple: bool) -> Optional[float]:
        """The share of trials in which the picture needed a fix: Apple Auto's, or the other picture's."""
        if self.trials == 0:
            return None
        return (self.flags_apple if apple else self.flags_other) / self.trials

    def as_dict(self) -> Dict[str, object]:
        return {
            "trials": self.trials,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "share": self.share,
            "wilson_lower": self.lower,
        }


@dataclass
class Bucket:
    """The overall tally and the per-stratum tallies of one comparison."""

    overall: Tally = field(default_factory=Tally)
    strata: Dict[str, Tally] = field(default_factory=dict)


@dataclass(frozen=True)
class Row:
    """One answered trial: its plan entry, Apple Auto's outcome, the flags of both pictures, and Apple's steps."""

    trial: Mapping[str, object]
    outcome: str
    flag_apple: bool
    flag_other: bool
    #: Apple's steps in the order of ANSWER_FIELDS (None for a field the model left empty), or None when the plan has
    #: none for the trial.
    steps: Optional[Steps]


def pct(value: Optional[float]) -> str:
    """A share as a percentage with one decimal and a space before the sign, or "n/a"."""
    if value is None:
        return "n/a"
    return f"{100 * value:.1f} %"


def color_words(steps: Sequence[StepValue], color: str) -> str:
    """A colour of an answer in words: "teal shadows (medium)"."""
    strength = step_of(steps, COLOR_FIELDS[color][0])
    return f"{step_of(steps, color)} {COLOR_TONES[color]} ({STRENGTH_WORDS.get(strength, strength)})"  # type: ignore


def steps_words(steps: Optional[Sequence[StepValue]]) -> str:
    """What Apple's answer changes, in words, in the order of the table: "slightly brighter, darker highlights, more
    contrast, teal shadows (medium)". A field left empty or at keep or none says nothing; "no change" when nothing is
    left, "unknown" without steps."""
    if steps is None:
        return "unknown"
    parts = [
        color_words(steps, name) if name in COLOR_FIELDS else STEP_WORDS[name][step_of(steps, name)]  # type: ignore
        for name in changed_fields(steps)
    ]
    return ", ".join(parts) if parts else "no change"


def empty_words(steps: Optional[Sequence[StepValue]]) -> str:
    """The fields Apple's model left empty (they set nothing), by name, or "none"."""
    if steps is None:
        return "unknown"
    empty = [name for name, value in zip(ANSWER_FIELDS, steps) if value is None and not name.endswith("_strength")]
    return ", ".join(empty) if empty else "none"


def load_json(path: Path, what: str) -> Dict[str, object]:
    """Reads a JSON object from `path`; a usage error when it is missing, unreadable or not an object."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ScoreError(f"the {what} file {path} does not exist") from None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScoreError(f"the {what} file {path} cannot be read: {exc}") from None
    if not isinstance(doc, dict):
        raise ScoreError(f"the {what} file {path} is not a JSON object")
    return doc


def validate_plan(plan: Mapping[str, object], path: Path) -> None:
    """Checks the parts of a plan that scoring reads; a usage error names what is wrong."""
    if plan.get("format") != PLAN_FORMAT or not isinstance(plan.get("plan_id"), str):
        raise ScoreError(f"{path} is not a plan.json written by this kit")
    trials = plan.get("trials")
    if not isinstance(trials, list):
        raise ScoreError(f"{path} has no trials list")
    for trial in trials:
        if not isinstance(trial, dict) or not isinstance(trial.get("id"), int):
            raise ScoreError(f"{path} has a trial without an id")
        if trial.get("comparison") not in COMPARISONS:
            raise ScoreError(f"trial {trial.get('id')} in {path} has an unknown comparison")
        if trial.get("left") not in VARIANTS or trial.get("right") not in VARIANTS:
            raise ScoreError(f"trial {trial.get('id')} in {path} has an unknown variant")
        if UNDER_TEST not in (trial.get("left"), trial.get("right")):
            raise ScoreError(
                f"trial {trial.get('id')} in {path} does not show Apple Auto (a plan of another test of this kit?)"
            )


def _valid_step(name: str, value: object) -> bool:
    """Whether a field of the plan's steps can be read: null (left empty: nothing set), a colour's name, or a number."""
    if value is None:
        return True
    if name in COLOR_FIELDS:
        return isinstance(value, str)
    return isinstance(value, int) and not isinstance(value, bool)


def _steps_of(trial: Mapping[str, object], photos: Mapping[int, Mapping[str, object]]) -> Optional[Steps]:
    """Apple's steps for a trial, in the order of ANSWER_FIELDS: the trial's own, else its photo's, else None. A field
    that is null or missing (a plan of prompt version 1 has four) is None: nothing set."""
    source = trial.get("steps")
    if not isinstance(source, dict):
        photo = photos.get(trial.get("number"))  # type: ignore[arg-type]
        source = photo.get("steps") if isinstance(photo, dict) else None
    if not isinstance(source, dict):
        return None
    values = [source.get(name) for name in ANSWER_FIELDS]
    if not all(_valid_step(name, value) for name, value in zip(ANSWER_FIELDS, values)):
        return None
    return tuple(values)  # type: ignore[arg-type]


def trial_answers(plan: Mapping[str, object], results: Mapping[str, object]) -> List[Row]:
    """Each answered trial with Apple Auto's outcome and the flags. Refuses results of another plan."""
    if results.get("format") != RESULTS_FORMAT:
        raise ScoreError("this is not a results.json exported from the blind page")
    plan_id = results.get("plan_id")
    if not isinstance(plan_id, str):
        raise ScoreError("the results have no plan id")
    if plan_id != plan["plan_id"]:
        raise ScoreError(f"the results are from plan {plan_id}, but the plan given is {plan['plan_id']}")
    answers = results.get("answers")
    if not isinstance(answers, list):
        raise ScoreError("the results have no answers list")
    trials = {trial["id"]: trial for trial in plan["trials"]}  # type: ignore[index]
    photos = {
        photo.get("number"): photo
        for photo in plan.get("photos", [])  # type: ignore[union-attr]
        if isinstance(photo, dict)
    }
    seen = set()
    rows: List[Row] = []
    for row in answers:
        if not isinstance(row, dict):
            raise ScoreError("an answer in the results is not an object")
        trial_id = row.get("trial")
        if not isinstance(trial_id, int) or isinstance(trial_id, bool) or trial_id not in trials:
            raise ScoreError(f"trial {trial_id!r} is not in the plan")
        if trial_id in seen:
            raise ScoreError(f"trial {trial_id} is answered more than once in the results")
        answer = row.get("answer")
        if answer not in ANSWERS:
            raise ScoreError(f"trial {trial_id} has the answer {answer!r}; expected left, right or same")
        flags = row.get("flags")
        if not isinstance(flags, dict):
            flags = {}
        seen.add(trial_id)
        trial = trials[trial_id]
        apple_side = "left" if trial["left"] == UNDER_TEST else "right"
        other_side = "right" if apple_side == "left" else "left"
        if answer == "same":
            outcome = "tie"
        elif answer == apple_side:
            outcome = "win"
        else:
            outcome = "loss"
        rows.append(
            Row(
                trial=trial,
                outcome=outcome,
                flag_apple=bool(flags.get(apple_side)),
                flag_other=bool(flags.get(other_side)),
                steps=_steps_of(trial, photos),  # type: ignore[arg-type]
            )
        )
    return rows


def tally(rows: Sequence[Row]) -> Dict[str, Bucket]:
    """Outcomes of Apple Auto per comparison, overall and per stratum."""
    buckets = {name: Bucket() for name in COMPARISONS}
    for row in rows:
        bucket = buckets[str(row.trial["comparison"])]
        bucket.overall.add(row.outcome, row.flag_apple, row.flag_other)
        stratum = bucket.strata.setdefault(str(row.trial["stratum"]), Tally())
        stratum.add(row.outcome, row.flag_apple, row.flag_other)
    return buckets


def _below_stratum_gate(part: Tally) -> bool:
    """True when a stratum has trials and its win share is below the stratum gate (45 %)."""
    return bool(part.trials and (part.share or 0.0) + _EPS < STRATUM_GATE)


def gate_one(tally_all: Tally, strata: Mapping[str, Tally]) -> Tuple[str, List[str]]:
    """The verdict of one comparison and the reasons it fails: PASS, FAIL, NOT RUN (no answers yet) or NO DATA."""
    if tally_all.trials == 0:
        return "NOT RUN", ["there are no answers for this comparison"]
    if tally_all.decisive == 0:
        return "NO DATA", ["every answer is a tie, so no win share can be bounded"]
    reasons: List[str] = []
    share = tally_all.share or 0.0
    if share + _EPS < SHARE_GATE:
        reasons.append(f"the pooled win share is {pct(share)}, below 60 %")
    lower = tally_all.lower or 0.0
    if not lower > LOWER_GATE:
        reasons.append(f"the Wilson 95 % lower bound is {pct(lower)}, not above 50 %")
    for stratum, part in sorted(strata.items()):
        if _below_stratum_gate(part):
            reasons.append(f"stratum {stratum} has a win share of {pct(part.share)}, below 45 %")
    if tally_all.flags_apple > tally_all.flags_other:
        reasons.append(
            f"Apple Auto's picture needs a fix in {pct(tally_all.fix_rate(True))} of trials, "
            f"the other picture in {pct(tally_all.fix_rate(False))}"
        )
    return ("FAIL" if reasons else "PASS"), reasons


def overall_verdict(verdicts: Sequence[str]) -> str:
    """PASS when every gated comparison passes; FAIL when one fails; otherwise NOT DECIDED."""
    if any(verdict == "FAIL" for verdict in verdicts):
        return "FAIL"
    if verdicts and all(verdict == "PASS" for verdict in verdicts):
        return "PASS"
    return "NOT DECIDED"


def _group(label: str, part: Tally) -> Dict[str, object]:
    return {"label": label, **part.as_dict(), "needs_fix": {"apple": part.fix_rate(True), "other": part.fix_rate(False)}}


def diagnostics(rows: Sequence[Row], comparison: str = "rule") -> Dict[str, object]:
    """What helps tune the table of docs/19 section 5 and the prompt of section 4, over the trials of one comparison:
    Apple Auto's win share by the direction of each step (brighter, darker highlights, warmer, ...), by the colour it
    gave the shadows or the highlights, by photos where Apple changed one thing only, by each step, how often each field
    was left empty, and the losses with Apple's scene and plan. A photo with several changes counts in each of its
    directions; a field left empty changed nothing and counts in none."""
    mine = [row for row in rows if row.trial["comparison"] == comparison]
    directions: Dict[str, Tally] = {label: Tally() for label, _, _ in DIRECTIONS}
    colors: Dict[str, Tally] = {
        f"{color} {COLOR_TONES[name]}": Tally() for name, (_, names) in COLOR_FIELDS.items() for color in names[1:]
    }
    single: Dict[str, Tally] = {f"only {name}": Tally() for name in (*STEP_FIELDS, *COLOR_FIELDS)}
    single["two or more changes"] = Tally()
    by_step: Dict[Tuple[int, object, object], Tally] = {}
    empty: Dict[str, int] = {name: 0 for name in ANSWER_FIELDS if not name.endswith("_strength")}
    with_steps = 0
    losses: List[Dict[str, object]] = []
    for row in mine:
        steps = row.steps
        if steps is None:
            continue
        with_steps += 1
        for name in empty:
            if step_of(steps, name) is None:
                empty[name] += 1
        for label, name, sign in DIRECTIONS:
            value = step_of(steps, name)
            if isinstance(value, int) and value * sign > 0:
                directions[label].add(row.outcome, row.flag_apple, row.flag_other)
        changed = changed_fields(steps)
        if len(changed) == 1:
            single[f"only {changed[0]}"].add(row.outcome, row.flag_apple, row.flag_other)
        elif len(changed) > 1:
            single["two or more changes"].add(row.outcome, row.flag_apple, row.flag_other)
        for name in changed:
            index = ANSWER_FIELDS.index(name)
            if name in COLOR_FIELDS:
                colors[f"{step_of(steps, name)} {COLOR_TONES[name]}"].add(row.outcome, row.flag_apple, row.flag_other)
                key: Tuple[int, object, object] = (index, step_of(steps, name), step_of(steps, COLOR_FIELDS[name][0]))
            else:
                key = (index, step_of(steps, name), None)
            by_step.setdefault(key, Tally()).add(row.outcome, row.flag_apple, row.flag_other)
    for row in mine:
        if row.outcome != "loss":
            continue
        losses.append(
            {
                "trial": row.trial["id"],
                "stem": row.trial.get("stem"),
                "stratum": row.trial.get("stratum"),
                "scene": row.trial.get("scene", ""),
                "plan": row.trial.get("plan", ""),
                "steps": None if row.steps is None else dict(zip(ANSWER_FIELDS, row.steps)),
                "changes": steps_words(row.steps),
                "left_empty": empty_words(row.steps),
                "needs_fix": {"apple": row.flag_apple, "other": row.flag_other},
            }
        )
    losses.sort(key=lambda item: (str(item["stratum"]), str(item["stem"]), int(item["trial"])))  # type: ignore[arg-type]
    step_rows = []
    ordered = sorted(by_step.items(), key=lambda item: (item[0][0], str(item[0][1]), str(item[0][2])))
    for (index, value, strength), part in ordered:
        name = ANSWER_FIELDS[index]
        if name in COLOR_FIELDS:
            words = f"{value} {COLOR_TONES[name]}, {STRENGTH_WORDS.get(strength, strength)}"  # type: ignore[arg-type]
            label = f"{name} {value} {strength} ({words})"
            step_rows.append({**_group(label, part), "field": name, "step": value, "strength": strength})
        else:
            words = STEP_WORDS[name][value]  # type: ignore[index]
            step_rows.append({**_group(f"{name} {value:+d} ({words})", part), "field": name, "step": value})
    return {
        "comparison": comparison,
        "by_direction": [_group(label, directions[label]) for label, _, _ in DIRECTIONS],
        "by_color": [_group(label, part) for label, part in colors.items()],
        "by_single_change": [_group(label, part) for label, part in single.items()],
        "by_step": step_rows,
        "left_empty": {"trials": with_steps, "fields": empty},
        "losses": losses,
    }


def build_report(plan: Mapping[str, object], rows: Sequence[Row]) -> Dict[str, object]:
    """The score document: every comparison with its tallies, strata, needs-a-fix rates and verdict, the gate (over
    the gated comparisons only), and the diagnostics of the comparison with the rule."""
    buckets = tally(rows)
    total = len(plan["trials"])  # type: ignore[arg-type]
    present = plan.get("comparisons")
    if not isinstance(present, list):
        present = sorted({str(t["comparison"]) for t in plan["trials"]}, key=list(COMPARISONS).index)  # type: ignore
    gated = [name for name in GATED if name in present]
    comparisons: Dict[str, object] = {}
    verdicts: Dict[str, str] = {}
    for name in COMPARISONS:
        if name not in present:
            continue
        bucket = buckets[name]
        verdict, reasons = gate_one(bucket.overall, bucket.strata)
        verdicts[name] = verdict
        entry = bucket.overall.as_dict()
        entry.update(
            {
                "against": COMPARISONS[name],
                "gated": name in gated,
                "needs_fix": {
                    "apple": bucket.overall.fix_rate(True),
                    "other": bucket.overall.fix_rate(False),
                },
                "strata": {
                    stratum: {**part.as_dict(), "below_gate": _below_stratum_gate(part)}
                    for stratum, part in sorted(bucket.strata.items())
                },
                "verdict": verdict,
                "reasons": reasons,
            }
        )
        comparisons[name] = entry
    gate = overall_verdict([verdicts[name] for name in gated])
    provenance = plan.get("provenance") if isinstance(plan.get("provenance"), dict) else {}
    return {
        "format": SCORE_FORMAT,
        "version": 3,
        "plan_id": plan["plan_id"],
        "seed": plan.get("seed"),
        "provenance": provenance,
        "photos": plan.get("counts"),
        "trials": total,
        "answered": len(rows),
        "complete": len(rows) == total,
        "comparisons": comparisons,
        "diagnostics": diagnostics(rows, "rule") if "rule" in present else None,
        "gate": {
            "name": GATE_NAME,
            "provisional": True,
            "note": "provisional thresholds (the parked autoset test's gate 5); they need the owner's agreement",
            "verdict": gate,
            "rules": {
                "pooled_share_at_least": SHARE_GATE,
                "wilson_lower_above": LOWER_GATE,
                "stratum_share_at_least": STRATUM_GATE,
                "needs_fix_not_more_often_than_other": True,
            },
            "gated": gated,
        },
    }


def _model_line(provenance: Mapping[str, object]) -> Optional[str]:
    apple = provenance.get("apple_auto")
    if not isinstance(apple, dict):
        return None
    line = (
        f"Apple Auto: {apple.get('model') or 'unknown model'} · macOS {apple.get('os') or 'unknown'} · "
        f"prompt {apple.get('prompt')} · table {apple.get('table')}"
    )
    if apple.get("instructions"):
        line += f" · instructions {apple.get('instructions')} ({apple.get('instructions_sha256')})"
    return line


def _photos_line(counts: object) -> Optional[str]:
    if not isinstance(counts, dict):
        return None
    return (
        f"Photos: {counts.get('found')} found, {counts.get('in_test')} in the test; Apple Auto left "
        f"{counts.get('unchanged')} as they were (no trial) and failed on {counts.get('failed')}; "
        f"{counts.get('not_rendered')} could not be rendered."
    )


def _trials(count: object) -> str:
    """"1 trial", "2 trials"."""
    return f"{count} trial" if count == 1 else f"{count} trials"


def _group_line(group: Mapping[str, object]) -> str:
    return f"    {str(group['label']):<40}{pct(group['share']):>9}   {_trials(group['trials'])}"  # type: ignore[index]


def format_report(report: Mapping[str, object]) -> str:
    """The text the owner reads: a row per comparison, the strata, the diagnostics, the losses, the reasons of any
    failure, and the verdict of the gate."""
    comparisons: Mapping[str, Mapping[str, object]] = report["comparisons"]  # type: ignore[assignment]
    lines = [
        f"Blind A/B of Apple Auto, plan {report['plan_id']} (seed {report['seed']}): "
        f"{report['answered']} of {report['trials']} trials answered."
    ]
    for extra in (_model_line(report.get("provenance") or {}), _photos_line(report.get("photos"))):  # type: ignore
        if extra:
            lines.append(extra)
    if not report["complete"]:
        lines.append("Not every trial is answered yet, so the verdict below is provisional.")
    lines.append("")
    lines.append(
        "Share: ties count as half a win. Wilson low: the 95 % lower bound over the decisive trials (ties left out)."
    )
    header = (
        f"{'Comparison':<34}{'Trials':>7}{'Wins':>6}{'Losses':>8}{'Ties':>6}"
        f"{'Share':>9}{'Wilson low':>12}  {'Fix: Apple / other':<22}{'Gate':>10}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for name, entry in comparisons.items():
        needs = entry["needs_fix"]
        fix = f"{pct(needs['apple'])} / {pct(needs['other'])}"  # type: ignore[index]
        verdict = str(entry["verdict"]) if entry["gated"] else f"({entry['verdict']})"
        lines.append(
            f"{LABELS[name]:<34}{entry['trials']:>7}{entry['wins']:>6}{entry['losses']:>8}{entry['ties']:>6}"
            f"{pct(entry['share']):>9}{pct(entry['wilson_lower']):>12}  {fix:<22}{verdict:>10}"  # type: ignore[arg-type]
        )
    if any(not entry["gated"] for entry in comparisons.values()):
        lines.append("In brackets: reported, not part of the gate.")
    lines.append("")
    lines.append("Strata, win share (in the gate, a stratum below 45 % fails):")
    for name, entry in comparisons.items():
        lines.append(f"  {LABELS[name]}")
        strata: Mapping[str, Mapping[str, object]] = entry["strata"]  # type: ignore[assignment]
        if not strata:
            lines.append("    no answers yet")
        for stratum, part in strata.items():
            marker = "   below 45 %" if part["below_gate"] else ""
            lines.append(f"    {stratum:<22}{pct(part['share']):>9}   {_trials(part['trials'])}{marker}")  # type: ignore
    diag = report.get("diagnostics")
    if isinstance(diag, dict):
        lines.append("")
        lines.append("For tuning the table (docs/19 section 5): Apple Auto's win share against the rule, by what it did.")
        for title, key in (
            ("By direction (a photo with several changes counts in each)", "by_direction"),
            ("By the colour of the shadows or the highlights", "by_color"),
            ("Photos with one change only, and the others", "by_single_change"),
            ("By step", "by_step"),
        ):
            groups = [g for g in diag[key] if g["trials"]]
            lines.append(f"  {title}:")
            if not groups:
                lines.append("    no answers yet")
            lines.extend(_group_line(group) for group in groups)
        empty = diag.get("left_empty")
        if isinstance(empty, dict) and empty.get("trials"):
            counts = [f"{name} {count}" for name, count in empty["fields"].items() if count]
            lines.append(f"  Fields Apple's model left empty (they set nothing), of {_trials(empty['trials'])}:")
            lines.append(f"    {', '.join(counts) if counts else 'none'}")
        losses = diag["losses"]
        lines.append("")
        lines.append(f"Trials Apple Auto lost against the rule ({len(losses)}), with Apple's scene, plan and changes:")
        if not losses:
            lines.append("  none")
        for loss in losses:
            fixes = []
            if loss["needs_fix"]["apple"]:
                fixes.append("Apple's picture needs a fix")
            if loss["needs_fix"]["other"]:
                fixes.append("the rule's picture needs a fix")
            fix_text = f" [{'; '.join(fixes)}]" if fixes else ""
            plan_text = f" (plan: \"{loss['plan']}\")" if loss.get("plan") else ""
            lines.append(
                f"  trial {loss['trial']}, {loss['stratum']}/{loss['stem']}: \"{loss['scene']}\"{plan_text}; "
                f"{loss['changes']}{fix_text}"
            )
    gated = report["gate"]["gated"]  # type: ignore[index]
    failing = [
        (name, entry)
        for name, entry in comparisons.items()
        if name in gated and entry["verdict"] in ("FAIL", "NOT RUN", "NO DATA")
    ]
    if failing:
        lines.append("")
        lines.append("Why the gate is not passed:")
        for name, entry in failing:
            for reason in entry["reasons"]:  # type: ignore[union-attr]
                lines.append(f"  {LABELS[name]}: {reason}")
    gate = report["gate"]["verdict"]  # type: ignore[index]
    provisional = "" if report["complete"] else " (not every trial is answered)"
    lines.append("")
    lines.append(
        f"Thresholds ({THRESHOLDS_NOTE}): win share at least 60 % (ties as half), Wilson 95 % lower bound above 50 % "
        "(ties left out), no stratum below 45 %, Apple Auto's picture not marked \"needs a fix\" more often than the "
        "rule's. They need your agreement."
    )
    lines.append(f"Gate, Apple Auto against the rule ({THRESHOLDS_NOTE}): {gate}{provisional}")
    return "\n".join(lines)


def find_plan(results_path: Path, plan_id: str, explicit: Optional[Path], fallback_dir: Path) -> Path:
    """The plan for the results: the one given, else a plan.json with the same id next to the results or in the
    fallback folder (where `prepare` writes by default). Without one, a usage error asks for `--plan`."""
    if explicit is not None:
        if not explicit.is_file():
            raise ScoreError(f"the plan file {explicit} does not exist")
        return explicit
    checked = []
    for folder in (results_path.parent, fallback_dir):
        candidate = folder / "plan.json"
        if candidate in checked or not candidate.is_file():
            continue
        checked.append(candidate)
        try:
            doc = load_json(candidate, "plan")
        except ScoreError:
            continue
        if doc.get("plan_id") == plan_id:
            return candidate
    raise ScoreError(
        f"no plan.json with plan id {plan_id} next to the results or in {fallback_dir}; pass it with --plan PLAN_JSON"
    )


def score_results(results_path: Path, plan_path: Optional[Path], fallback_dir: Path) -> Tuple[Dict[str, object], Path]:
    """Scores a results file, writes `score.json` next to it, and returns the report and the file's path."""
    results = load_json(results_path, "results")
    plan_id = results.get("plan_id")
    if not isinstance(plan_id, str):
        raise ScoreError("the results have no plan id")
    plan_file = find_plan(results_path, plan_id, plan_path, fallback_dir)
    plan = load_json(plan_file, "plan")
    validate_plan(plan, plan_file)
    rows = trial_answers(plan, results)
    report = build_report(plan, rows)
    out = results_path.parent / SCORE_FILE
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report, out

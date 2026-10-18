"""The score of the blind A/B: wins, losses and ties per comparison, the Wilson bound, the strata and gate 5.

Gate 5 of docs/18 section 8, for each of the rule, the camera JPEG and Lightroom Auto:

* the pooled win share of autoset is at least 60 %, ties counting as half a win;
* the Wilson 95 % lower bound of the win share is above 50 %, with ties left out of the count (so the bound is over
  the decisive trials only);
* no stratum has a win share below 45 %;
* autoset's needs-a-fix rate is not higher than the other picture's.

The comparison with autoset's white preference switched off is reported and not gated: it answers whether the
preferred white helps. Stdlib only.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from page import RESULTS_FORMAT
from plan import COMPARISONS, GATED, PLAN_FORMAT, VARIANTS

SCORE_FORMAT = "yellowmoon-ab-score"
SCORE_FILE = "score.json"
ANSWERS = ("left", "right", "same")
#: 95 % two-sided normal quantile for the Wilson interval.
Z95 = 1.959963984540054
SHARE_GATE = 0.60
LOWER_GATE = 0.50
STRATUM_GATE = 0.45
_EPS = 1e-12
LABELS: Mapping[str, str] = {
    "rule": "autoset vs rule (before autoset)",
    "camera": "autoset vs camera JPEG",
    "lightroom": "autoset vs Lightroom Auto",
    "neutral": "autoset vs white preference off",
}


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
    """Outcomes of autoset in some set of trials, and the needs-a-fix flags of both pictures."""

    wins: int = 0
    losses: int = 0
    ties: int = 0
    flags_autoset: int = 0
    flags_other: int = 0

    def add(self, outcome: str, flag_autoset: bool, flag_other: bool) -> None:
        """Counts one trial. `outcome` is "win", "loss" or "tie" for autoset."""
        if outcome == "win":
            self.wins += 1
        elif outcome == "loss":
            self.losses += 1
        else:
            self.ties += 1
        self.flags_autoset += int(flag_autoset)
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

    def fix_rate(self, autoset: bool) -> Optional[float]:
        """The share of trials in which the picture needed a fix: autoset's, or the other picture's."""
        if self.trials == 0:
            return None
        return (self.flags_autoset if autoset else self.flags_other) / self.trials

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


def pct(value: Optional[float]) -> str:
    """A share as a percentage with one decimal and a space before the sign, or "n/a"."""
    if value is None:
        return "n/a"
    return f"{100 * value:.1f} %"


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
        if "autoset" not in (trial.get("left"), trial.get("right")):
            raise ScoreError(f"trial {trial.get('id')} in {path} does not show autoset")


def trial_answers(plan: Mapping[str, object], results: Mapping[str, object]) -> List[Tuple[dict, str, dict]]:
    """Each answered trial with its plan entry, its answer and its flags. Refuses results of another plan."""
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
    seen = set()
    rows: List[Tuple[dict, str, dict]] = []
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
        rows.append((trials[trial_id], answer, {"left": bool(flags.get("left")), "right": bool(flags.get("right"))}))
    return rows


def tally(rows: Sequence[Tuple[dict, str, dict]]) -> Dict[str, Bucket]:
    """Outcomes of autoset per comparison, overall and per stratum."""
    buckets = {name: Bucket() for name in COMPARISONS}
    for trial, answer, flags in rows:
        bucket = buckets[trial["comparison"]]
        autoset_side = "left" if trial["left"] == "autoset" else "right"
        other_side = "right" if autoset_side == "left" else "left"
        if answer == "same":
            outcome = "tie"
        elif answer == autoset_side:
            outcome = "win"
        else:
            outcome = "loss"
        flag_autoset = flags[autoset_side]
        flag_other = flags[other_side]
        bucket.overall.add(outcome, flag_autoset, flag_other)
        stratum = bucket.strata.setdefault(str(trial["stratum"]), Tally())
        stratum.add(outcome, flag_autoset, flag_other)
    return buckets


def _below_stratum_gate(part: Tally) -> bool:
    """True when a stratum has trials and its win share is below the stratum gate (45 %)."""
    return bool(part.trials and (part.share or 0.0) + _EPS < STRATUM_GATE)


def gate_one(name: str, tally_all: Tally, strata: Mapping[str, Tally]) -> Tuple[str, List[str]]:
    """The verdict of one comparison and the reasons it fails. "reported" for the comparisons that are not gated."""
    if name not in GATED:
        return "reported", []
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
    if tally_all.flags_autoset > tally_all.flags_other:
        reasons.append(
            f"autoset needs a fix in {pct(tally_all.fix_rate(True))} of trials, "
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


def build_report(plan: Mapping[str, object], rows: Sequence[Tuple[dict, str, dict]]) -> Dict[str, object]:
    """The score document: every comparison with its tallies, strata, needs-a-fix rates and gate verdict."""
    buckets = tally(rows)
    total = len(plan["trials"])  # type: ignore[arg-type]
    comparisons: Dict[str, object] = {}
    verdicts: Dict[str, str] = {}
    for name, variant in COMPARISONS.items():
        bucket = buckets[name]
        verdict, reasons = gate_one(name, bucket.overall, bucket.strata)
        verdicts[name] = verdict
        entry = bucket.overall.as_dict()
        entry.update(
            {
                "against": variant,
                "gated": name in GATED,
                "needs_fix": {
                    "autoset": bucket.overall.fix_rate(True),
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
    gate = overall_verdict([verdicts[name] for name in GATED])
    return {
        "format": SCORE_FORMAT,
        "version": 1,
        "plan_id": plan["plan_id"],
        "seed": plan.get("seed"),
        "trials": total,
        "answered": len(rows),
        "complete": len(rows) == total,
        "comparisons": comparisons,
        "gate": {
            "name": "docs/18 section 8, gate 5",
            "verdict": gate,
            "rules": {
                "pooled_share_at_least": SHARE_GATE,
                "wilson_lower_above": LOWER_GATE,
                "stratum_share_at_least": STRATUM_GATE,
                "needs_fix_not_higher_than_other": True,
            },
            "gated": list(GATED),
        },
    }


def format_report(report: Mapping[str, object]) -> str:
    """The table the owner reads: one row per comparison, the strata, the reasons of any failure, and the verdict."""
    comparisons: Mapping[str, Mapping[str, object]] = report["comparisons"]  # type: ignore[assignment]
    lines = [
        f"Blind A/B, plan {report['plan_id']} (seed {report['seed']}): "
        f"{report['answered']} of {report['trials']} trials answered."
    ]
    if not report["complete"]:
        lines.append("Not every trial is answered yet, so the verdict below is provisional.")
    lines.append("")
    lines.append(
        "Share: ties count as half a win. Wilson low: the 95 % lower bound over the decisive trials (ties left out)."
    )
    header = (
        f"{'Comparison':<34}{'Trials':>7}{'Wins':>6}{'Losses':>8}{'Ties':>6}"
        f"{'Share':>9}{'Wilson low':>12}  {'Fix: autoset / other':<22}{'Gate':>10}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for name, entry in comparisons.items():
        needs = entry["needs_fix"]
        fix = f"{pct(needs['autoset'])} / {pct(needs['other'])}"
        lines.append(
            f"{LABELS[name]:<34}{entry['trials']:>7}{entry['wins']:>6}{entry['losses']:>8}{entry['ties']:>6}"
            f"{pct(entry['share']):>9}{pct(entry['wilson_lower']):>12}  {fix:<22}{entry['verdict']:>10}"
        )
    lines.append("")
    lines.append("Strata, win share (in the gate, a stratum below 45 % fails):")
    for name, entry in comparisons.items():
        heading = LABELS[name] + ("" if entry["gated"] else " (reported, not gated)")
        lines.append(f"  {heading}")
        if not entry["strata"]:
            lines.append("    no answers yet")
        for stratum, part in entry["strata"].items():
            marker = "   below 45 %" if part["below_gate"] and entry["gated"] else ""
            lines.append(f"    {stratum:<22}{pct(part['share']):>9}   {part['trials']} trials{marker}")
    unresolved = ("FAIL", "NOT RUN", "NO DATA")
    failing = [(name, entry) for name, entry in comparisons.items() if entry["verdict"] in unresolved]
    if failing:
        lines.append("")
        lines.append("Why the gate is not passed:")
        for name, entry in failing:
            for reason in entry["reasons"]:
                lines.append(f"  {LABELS[name]}: {reason}")
    gate = report["gate"]["verdict"]  # type: ignore[index]
    provisional = "" if report["complete"] else " (provisional)"
    lines.append("")
    lines.append(f"Gate 5 (docs/18 section 8): {gate}{provisional}")
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

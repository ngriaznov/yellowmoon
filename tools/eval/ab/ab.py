#!/usr/bin/env python3
"""The blind A/B kit for gate 5 of docs/18 section 8 (AS-H1): `prepare` makes the test, `score` scores the answers.

    tools/eval/ab/ab.sh prepare PHOTOS_DIR [--out DIR] [--long-edge 2000] [--seed N]
                                           [--ym PATH] [--data DIR] [--jobs N]
    tools/eval/ab/ab.sh score RESULTS_JSON [--plan PLAN_JSON]

`prepare` renders each photo as autoset and as the rule before autoset, and copies the camera's own JPEG (the page turns
it upright from the raw file's orientation). It writes
`index.html` (the blind page), `plan.js` (what the page shows), `plan.json` (the key, for scoring) and the pictures
under `pictures/`. `score` reads the `results.json` the page exports, applies gate 5 and writes `score.json` next to
the results. Nothing is uploaded.

Python 3.9 or newer, standard library only. The tests:
`python3 -I -m unittest discover -s tools/eval/ab -p 'test_*.py'`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from page import write_page  # noqa: E402
from plan import (  # noqa: E402
    COMPARISONS,
    DEFAULT_SEED,
    VARIANTS,
    PlanPhoto,
    build_plan,
    count_plan,
    find_photos,
    plan_to_json,
)
from renders import (  # noqa: E402
    QUALITY,
    PhotoJob,
    Pictures,
    copy_rule_data,
    make_pictures,
    raw_turns,
    ym_environment,
)
from score import ScoreError, format_report, score_results  # noqa: E402

REPO = HERE.parents[2]
DEFAULT_OUT = REPO / "testdata" / "ab"
DEFAULT_YM = REPO / "target" / "release" / "ym"
DEFAULT_DATA = REPO / "data"
PLAN_FILE = "plan.json"
MODEL_RELPATH = Path("models") / "autoset" / "autoset.json"


class UsageError(Exception):
    """A usage error (exit 2): a folder, a program or a file is missing or wrong. The message says what to do."""


class PrepareFailed(Exception):
    """The test could not be made at all (exit 1), for example because no photo rendered in every variant."""


@dataclass(frozen=True)
class PrepareOptions:
    """What `prepare` needs. The defaults of the command line are in `build_parser`."""

    photos: Path
    out: Path
    long_edge: int
    seed: int
    ym: Path
    data: Path
    jobs: int
    scratch_root: Optional[Path] = None


@dataclass
class PrepareResult:
    """What `prepare` made: the plan, the skipped photos and the number of photos found, for the summary."""

    plan: Dict[str, object]
    skipped: List[Dict[str, object]]
    found: int


def stderr_log(message: str) -> None:
    """The default progress printer: one line per message, on stderr."""
    print(message, file=sys.stderr, flush=True)


def default_jobs() -> int:
    """Half the CPUs, at least one: each `ym` render already uses several threads."""
    return max(1, (os.cpu_count() or 2) // 2)


def ym_version(ym: Path, data: Path, scratch: Path) -> str:
    """The first line of `ym --version`, recorded in the plan so that a plan says which renderer made it."""
    home = Path(tempfile.mkdtemp(prefix="ym-version-", dir=str(scratch)))
    try:
        proc = subprocess.run(
            [str(ym), "--version"],
            env=ym_environment(data, home),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=120,
            check=False,
        )
    except OSError as exc:
        raise UsageError(f"ym could not start ({ym}): {exc}") from None
    finally:
        shutil.rmtree(home, ignore_errors=True)
    lines = proc.stdout.decode("utf-8", errors="replace").strip().splitlines()
    if proc.returncode != 0 or not lines:
        raise UsageError(f"{ym} --version failed; build ym again or pass --ym PATH")
    return lines[0].strip()


def model_sha256(data: Path) -> str:
    """The hash of the shipped automatic settings model, so that a plan says which model made the Auto pictures."""
    return hashlib.sha256((data / MODEL_RELPATH).read_bytes()).hexdigest()


def read_plan_id(path: Path) -> Optional[str]:
    """The plan id of an existing plan.json, or None when there is none or it cannot be read."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    value = doc.get("plan_id") if isinstance(doc, dict) else None
    return value if isinstance(value, str) else None


def prepare(options: PrepareOptions, log: Callable[[str], None] = stderr_log) -> PrepareResult:
    """Renders every photo in every variant, drops the photos that fail anywhere, and writes the test.

    The originals are only read. Pictures go under `options.out/pictures`, the page and the plan next to them.
    """
    photos = find_photos(options.photos)
    if not photos:
        raise UsageError(f"no ARW files in {options.photos} (folders are walked; hidden files are skipped)")
    options.out.mkdir(parents=True, exist_ok=True)
    parent = None if options.scratch_root is None else str(options.scratch_root)
    scratch = Path(tempfile.mkdtemp(prefix="ab-", dir=parent))
    try:
        provenance = {
            "ym": ym_version(options.ym, options.data, scratch),
            "long_edge": options.long_edge,
            "quality": QUALITY,
            "autoset_model_sha256": model_sha256(options.data),
        }
        rule_data = copy_rule_data(options.data, scratch)
        jobs = [
            PhotoJob(
                number=number,
                stem=photo.stem,
                stratum=photo.stratum,
                raw=photo.path,
                camera_turns=raw_turns(photo.path),
            )
            for number, photo in enumerate(photos, start=1)
        ]
        pictures = Pictures(
            ym=options.ym,
            data=options.data,
            rule_data=rule_data,
            out=options.out,
            scratch=scratch,
            seed=options.seed,
            long_edge=options.long_edge,
        )
        log(f"Rendering {len(jobs)} photos, {options.jobs} at a time")
        failures = make_pictures(pictures, jobs, options.jobs, log)
        skipped: List[Dict[str, object]] = []
        for job in jobs:
            if job.number in failures:
                reasons = sorted(failures[job.number], key=lambda pair: _variant_order(pair[0]))
                skipped.append(
                    {
                        "number": job.number,
                        "stem": job.stem,
                        "stratum": job.stratum,
                        "problems": [{"variant": variant, "reason": reason} for variant, reason in reasons],
                    }
                )
        kept = [job for job in jobs if job.number not in failures]
        if not kept:
            raise PrepareFailed("no photo rendered in every variant, so there is no test; see the reasons above")
        plan_photos = [
            PlanPhoto(number=job.number, stem=job.stem, stratum=job.stratum, camera_turns=job.camera_turns)
            for job in kept
        ]
        plan = build_plan(plan_photos, options.seed, provenance)
        plan["skipped"] = skipped
        plan_path = options.out / PLAN_FILE
        previous = read_plan_id(plan_path)
        if previous is not None and previous != plan["plan_id"]:
            log(f"note: the folder held plan {previous}; it is replaced, so answers given to it no longer match")
        plan_path.write_text(plan_to_json(plan), encoding="utf-8")
        write_page(options.out, plan)
        return PrepareResult(plan=plan, skipped=skipped, found=len(jobs))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _variant_order(variant: str) -> int:
    """The position of a variant in the plan's order, for listing the failures of a photo."""
    return VARIANTS.index(variant) if variant in VARIANTS else len(VARIANTS)


def summary_text(result: PrepareResult, out: Path, ym_name: str) -> str:
    """The summary `prepare` prints: photos, strata, trials, the skipped photos, and how to open the page."""
    plan = result.plan
    counts = count_plan(plan)
    kept = len(plan["photos"])  # type: ignore[arg-type]
    parts = [f"{name} {counts.by_comparison.get(name, 0)}" for name in COMPARISONS]
    lines = [
        f"Blind A/B written to {out}",
        f"  photos: {result.found} found, {kept} in the test, {len(result.skipped)} skipped",
        "  strata: " + ", ".join(f"{name} {count}" for name, count in sorted(counts.by_stratum.items())),
        f"  trials: {len(plan['trials'])} ({', '.join(parts)}); seed {plan['seed']}; plan {plan['plan_id']}",
    ]
    if result.skipped:
        lines.append("  skipped (left out of every trial):")
        for item in result.skipped:
            problems = "; ".join(f"{p['variant']}: {p['reason']}" for p in item["problems"])  # type: ignore[index]
            lines.append(f"    {item['stem']} ({item['stratum']}): {problems}")
    lines.append("")
    lines.append(f"Open the page:  open {shlex.quote(str(out / 'index.html'))}")
    lines.append("Answer every trial, then press Export results in the page.")
    lines.append(f"Score the answers with:  tools/eval/ab/ab.sh score RESULTS_JSON   (ym: {ym_name})")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """The command line: `prepare` and `score`, with the defaults of the plan (seed, long edge, jobs, paths)."""
    parser = argparse.ArgumentParser(
        prog="ab.sh",
        description="Blind A/B for gate 5 of docs/18 section 8: prepare the test, then score the answers.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare", help="render the pictures and write the blind page")
    prep.add_argument("photos", type=Path, metavar="PHOTOS_DIR", help="folder of ARW files; subfolders are strata")
    prep.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"output folder (default {DEFAULT_OUT})")
    prep.add_argument("--long-edge", type=int, default=2000, help="long edge of the renders in pixels (default 2000)")
    prep.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"seed of the plan (default {DEFAULT_SEED})")
    prep.add_argument("--ym", type=Path, default=DEFAULT_YM, help="the ym binary (default target/release/ym)")
    prep.add_argument("--data", type=Path, default=DEFAULT_DATA, help="the data folder (default data/)")
    prep.add_argument("--jobs", type=int, default=default_jobs(), help="renders at the same time (default: CPUs / 2)")

    score = sub.add_parser("score", help="score the results.json the page exported")
    score.add_argument("results", type=Path, metavar="RESULTS_JSON")
    score.add_argument(
        "--plan",
        type=Path,
        metavar="PLAN_JSON",
        help="the plan of the results (default: the plan.json with the same id next to them or in the default folder)",
    )
    return parser


def run_prepare(args: argparse.Namespace) -> int:
    """`ab.sh prepare`: checks the inputs, makes the test, and prints the summary."""
    if not args.photos.is_dir():
        raise UsageError(f"{args.photos} is not a folder")
    if args.long_edge < 64:
        raise UsageError("--long-edge must be at least 64")
    if args.jobs < 1:
        raise UsageError("--jobs must be at least 1")
    if not (args.ym.is_file() and os.access(args.ym, os.X_OK)):
        raise UsageError(
            f"there is no ym at {args.ym}. Build it with: cargo build -p ym-cli --release (or pass --ym PATH)"
        )
    if not (args.data / MODEL_RELPATH).is_file():
        raise UsageError(f"the data folder {args.data} has no automatic settings model; pass --data DIR")
    options = PrepareOptions(
        photos=args.photos,
        out=args.out,
        long_edge=args.long_edge,
        seed=args.seed,
        ym=args.ym,
        data=args.data,
        jobs=args.jobs,
    )
    try:
        result = prepare(options)
    except PrepareFailed as exc:
        print(f"ab.sh: {exc}", file=sys.stderr)
        return 1
    print(summary_text(result, args.out.resolve(), str(args.ym)))
    return 0


def run_score(args: argparse.Namespace) -> int:
    """`ab.sh score`: scores the results, prints the table and the gate, and writes score.json."""
    report, path = score_results(args.results, args.plan, DEFAULT_OUT)
    print(format_report(report))
    print(f"\nScore written to {path}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """The entry point. Exit codes: 0 for a completed run (the verdict is in the output), 1 when no test could be
    made, 2 for usage errors, 130 when interrupted."""
    if sys.version_info < (3, 9):
        print("ab.sh needs Python 3.9 or newer", file=sys.stderr)
        return 2
    args = build_parser().parse_args(argv)
    previous = signal.signal(signal.SIGTERM, _terminate_as_interrupt)
    try:
        if args.command == "prepare":
            return run_prepare(args)
        return run_score(args)
    except (UsageError, ScoreError) as exc:
        print(f"ab.sh: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        # A folder that cannot be read (permissions, a disconnected drive): say so instead of a traceback.
        print(f"ab.sh: {exc.strerror or exc}: {exc.filename or ''}".rstrip(": "), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("ab.sh: interrupted", file=sys.stderr)
        return 130
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)


def _terminate_as_interrupt(signum: int, frame: object) -> None:
    """SIGTERM is handled like Ctrl-C, so that `prepare` still removes its temporary copy of the data folder."""
    raise KeyboardInterrupt


if __name__ == "__main__":
    sys.exit(main())

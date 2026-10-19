#!/usr/bin/env python3
"""The blind A/B kit for Apple Auto (docs/19 section 13, AA-1): `prepare` makes the test, `score` scores the answers.

    tools/eval/ab/ab.sh prepare PHOTOS_DIR --ym PATH [--out DIR] [--long-edge 2000] [--seed N] [--jobs N]
                                           [--apple-jobs 1] [--with-camera] [--resume] [--data DIR]
                                           [--instructions FILE]
    tools/eval/ab/ab.sh score RESULTS_JSON [--plan PLAN_JSON]

`prepare` checks with `ym doctor` that Apple Auto is available, then renders each photo with Apple Auto on (`apple`)
and off (`rule`, Yellowmoon's own Auto), at the defaults: the built-in default look and every setting at Auto. Apple's
model reads the default instructions, or those of --instructions FILE (docs/19 section 2); the plan records which, with
their digest. A photo whose answer failed is left out with its reason; a photo whose answer changed nothing gives no
trial. With
--with-camera it also copies the camera's own JPEG for a second, reported comparison. It writes `index.html` (the
blind page), `plan.js` (what the page shows), `plan.json` (the key, for scoring), the pictures under `pictures/` and
`ym`'s records under `records/` (the key too). `score` reads the `results.json` the page exports, applies the gate with
its provisional thresholds and writes `score.json` next to the results. Nothing is uploaded.

Python 3.9 or newer, standard library only. The tests:
`python3 -I -m unittest discover -s tools/eval/ab -p 'test_*.py'`.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from page import write_page  # noqa: E402
from plan import (  # noqa: E402
    DEFAULT_SEED,
    PlanPhoto,
    build_plan,
    count_plan,
    find_photos,
    plan_to_json,
    steps_dict,
)
from renders import (  # noqa: E402
    QUALITY,
    Outcome,
    PhotoJob,
    PictureError,
    Pictures,
    call_ym,
    make_pictures,
    raw_turns,
    instructions_digest,
    unavailable_words,
    without_final_break,
    ym_environment,
)
from score import LABELS, ScoreError, format_report, score_results  # noqa: E402

REPO = HERE.parents[2]
DEFAULT_OUT = REPO / "testdata" / "ab"
DEFAULT_YM = REPO / "target" / "release" / "ym"
REPO_DATA = REPO / "data"
PLAN_FILE = "plan.json"
TEST_ADVISOR = "Test advisor"
DOCTOR_TIMEOUT_S = 300.0
#: The longest instructions Apple's model is given, in characters (`ym_core::apple_auto::INSTRUCTIONS_MAX_CHARS`).
INSTRUCTIONS_MAX_CHARS = 2000


class UsageError(Exception):
    """A usage error (exit 2): a folder, a program or a file is missing or wrong. The message says what to do."""


class PrepareFailed(Exception):
    """The test could not be made (exit 1): Apple Auto is not available, or no photo is in the test."""


@dataclass(frozen=True)
class PrepareOptions:
    """What `prepare` needs. The defaults of the command line are in `build_parser`."""

    photos: Path
    out: Path
    long_edge: int
    seed: int
    ym: Path
    data: Optional[Path]
    jobs: int
    apple_jobs: int = 1
    with_camera: bool = False
    allow_unavailable: bool = False
    scratch_root: Optional[Path] = None
    resume: bool = False
    #: Instructions for Apple's model instead of the default (`--instructions FILE`).
    instructions: Optional[Path] = None


@dataclass
class PrepareResult:
    """What `prepare` made: the plan, the photos found, and those that gave no trial, for the summary."""

    plan: Dict[str, object]
    found: int
    unchanged: List[Dict[str, object]] = field(default_factory=list)
    failed: List[Dict[str, object]] = field(default_factory=list)
    errors: List[Dict[str, object]] = field(default_factory=list)
    no_camera: List[Dict[str, object]] = field(default_factory=list)


def stderr_log(message: str) -> None:
    """The default progress printer: one line per message, on stderr."""
    print(message, file=sys.stderr, flush=True)


def default_jobs() -> int:
    """Half the CPUs, at least one: each `ym` render already uses several threads."""
    return max(1, (os.cpu_count() or 2) // 2)


def in_app_bundle(ym: Path) -> bool:
    """Whether `ym` is the one inside an app (`Yellowmoon.app/Contents/MacOS/ym`), which finds the app's data itself."""
    resolved = ym.resolve()
    return resolved.parent.name == "MacOS" and resolved.parent.parent.name == "Contents"


def default_data(ym: Path) -> Optional[Path]:
    """The data folder `ym` gets without --data: none for the `ym` inside an app (it uses the app's own), else the
    repository's `data/` (a release `ym` built here does not find it by itself)."""
    if in_app_bundle(ym) or not REPO_DATA.is_dir():
        return None
    return REPO_DATA


def ym_version(ym: Path, data: Optional[Path], scratch: Path) -> str:
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
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise UsageError(f"ym could not start ({ym}): {exc}") from None
    finally:
        shutil.rmtree(home, ignore_errors=True)
    lines = proc.stdout.decode("utf-8", errors="replace").strip().splitlines()
    if proc.returncode != 0 or not lines:
        raise UsageError(f"{ym} --version failed; pass the ym of the test build with --ym PATH")
    return lines[0].strip()


def apple_auto_status(ym: Path, data: Optional[Path], scratch: Path) -> Dict[str, object]:
    """`apple_auto` of `ym doctor --json` (docs/19 section 8): available, reason, model, os, prompt, table, and the
    instructions (`default` here: each `ym` call has an empty home folder) with their digest."""
    try:
        code, doc, last_error = call_ym(ym, data, scratch, ["doctor", "--json"], DOCTOR_TIMEOUT_S)
    except PictureError as exc:
        raise UsageError(f"ym doctor did not run: {exc}") from None
    status = doc.get("apple_auto") if isinstance(doc, dict) else None
    if not isinstance(status, dict):
        detail = f" ({last_error})" if last_error else ""
        raise UsageError(
            f"ym doctor gave no Apple Auto status (exit {code}){detail}. This ym is older than Apple Auto: pass the ym "
            "of the test build with --ym PATH"
        )
    return status


def check_available(status: Dict[str, object], allow_unavailable: bool, log: Callable[[str], None]) -> None:
    """Stops when Apple Auto is not available (unless `allow_unavailable`, for tests): the test would have nothing
    to compare."""
    if status.get("available") is True:
        return
    words = unavailable_words(status.get("reason"))
    if not allow_unavailable:
        raise PrepareFailed(
            f"Apple Auto is not available on this Mac ({words}). The test needs macOS 27 with Apple Intelligence on."
        )
    log(f"note: Apple Auto is not available ({words}); going on because of --allow-unavailable")


def instructions_id(status: Dict[str, object], path: Optional[Path]) -> Dict[str, object]:
    """Which instructions Apple's model reads in the test and their digest (`instructions`, `instructions_sha256`, as
    `ym` reports them): the default (as `ym doctor` reports it), or the text of `path`. A file that cannot be read, is
    not UTF-8 or is longer than the model takes is a usage error."""
    default_sha = status.get("instructions_sha256")
    if path is None:
        return {"instructions": status.get("instructions"), "instructions_sha256": default_sha}
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise UsageError(f"the instructions file {path} cannot be read: {exc}") from None
    length = len(without_final_break(text))
    if length > INSTRUCTIONS_MAX_CHARS:
        raise UsageError(
            f"the instructions in {path} are {length} characters long; at most {INSTRUCTIONS_MAX_CHARS} fit, because "
            "the model must also see the photo"
        )
    digest = instructions_digest(text, str(default_sha))
    kind = "default" if digest == default_sha else "edited"
    return {"instructions": kind, "instructions_sha256": digest}


def _photo_item(job: PhotoJob, outcome: Optional[Outcome] = None, reason: str = "") -> Dict[str, object]:
    item: Dict[str, object] = {"number": job.number, "stem": job.stem, "stratum": job.stratum}
    if outcome is not None and outcome.status in ("unchanged", "trial"):
        item["steps"] = steps_dict(outcome.steps)
        item["scene"] = outcome.scene
        item["plan"] = outcome.plan
    if reason:
        item["reason"] = reason
    return item


def read_plan_id(path: Path) -> Optional[str]:
    """The plan id of an existing plan.json, or None when there is none or it cannot be read."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    value = doc.get("plan_id") if isinstance(doc, dict) else None
    return value if isinstance(value, str) else None


def prepare(options: PrepareOptions, log: Callable[[str], None] = stderr_log) -> PrepareResult:
    """Checks that Apple Auto is available, renders every photo with Apple Auto on and off, keeps the photos whose
    answer changed the picture, and writes the test.

    The originals are only read. Pictures go under `options.out/pictures`, `ym`'s records under `options.out/records`,
    the page and the plan next to them.
    """
    photos = find_photos(options.photos)
    if not photos:
        raise UsageError(f"no ARW files in {options.photos} (folders are walked; hidden files are skipped)")
    options.out.mkdir(parents=True, exist_ok=True)
    parent = None if options.scratch_root is None else str(options.scratch_root)
    scratch = Path(tempfile.mkdtemp(prefix="ab-", dir=parent))
    try:
        version = ym_version(options.ym, options.data, scratch)
        status = apple_auto_status(options.ym, options.data, scratch)
        check_available(status, options.allow_unavailable, log)
        instructions = instructions_id(status, options.instructions)
        provenance = {
            "ym": version,
            "long_edge": options.long_edge,
            "quality": QUALITY,
            "with_camera": options.with_camera,
            "apple_auto": {
                **{key: status.get(key) for key in ("available", "reason", "model", "os", "prompt", "table")},
                **instructions,
            },
        }
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
            out=options.out,
            scratch=scratch,
            seed=options.seed,
            long_edge=options.long_edge,
            with_camera=options.with_camera,
            resume=options.resume,
            instructions=options.instructions,
            instructions_sha256=instructions["instructions_sha256"],  # type: ignore[arg-type]
        )
        log(
            f"Rendering {len(jobs)} photos with Apple Auto on and off, {options.jobs} at a time "
            f"({options.apple_jobs} waiting for Apple's model at a time)"
        )
        outcomes = make_pictures(pictures, jobs, options.jobs, options.apple_jobs, log)
        reported = {o.instructions for o in outcomes.values() if o.instructions is not None}
        wanted = (instructions["instructions"], instructions["instructions_sha256"])
        if reported - {wanted}:
            log(
                "note: ym reported other instructions than expected ("
                + ", ".join(f"{kind} {digest}" for kind, digest in sorted(reported - {wanted}))  # type: ignore[misc]
                + f"; expected {wanted[0]} {wanted[1]})"
            )
        result = PrepareResult(plan={}, found=len(jobs))
        kept: List[PlanPhoto] = []
        details: Dict[int, Dict[str, object]] = {}
        for job in jobs:
            outcome = outcomes[job.number]
            if outcome.status == "unchanged":
                result.unchanged.append(_photo_item(job, outcome, outcome.reason))
            elif outcome.status == "failed":
                result.failed.append(_photo_item(job, outcome, outcome.reason))
            elif outcome.status == "error":
                result.errors.append(_photo_item(job, outcome, outcome.reason))
            else:
                if options.with_camera and not outcome.has_camera:
                    result.no_camera.append(_photo_item(job, None, outcome.camera_problem))
                kept.append(
                    PlanPhoto(
                        number=job.number,
                        stem=job.stem,
                        stratum=job.stratum,
                        camera_turns=job.camera_turns,
                        has_camera=outcome.has_camera,
                        steps=outcome.steps,
                        scene=outcome.scene,
                        plan=outcome.plan,
                    )
                )
                details[job.number] = {"auto_values": outcome.values}
        if not kept:
            raise PrepareFailed(
                f"no photo is in the test: Apple Auto left {len(result.unchanged)} of {len(jobs)} photos as they were, "
                f"failed on {len(result.failed)}, and {len(result.errors)} could not be rendered; see the lines above"
            )
        plan = build_plan(kept, options.seed, provenance, details)
        plan["counts"] = {
            "found": len(jobs),
            "in_test": len(kept),
            "unchanged": len(result.unchanged),
            "failed": len(result.failed),
            "not_rendered": len(result.errors),
            "no_camera": len(result.no_camera),
        }
        plan["unchanged"] = result.unchanged
        plan["left_out"] = [{**item, "why": "failed"} for item in result.failed] + [
            {**item, "why": "not rendered"} for item in result.errors
        ]
        plan["no_camera"] = result.no_camera
        plan_path = options.out / PLAN_FILE
        previous = read_plan_id(plan_path)
        if previous is not None and previous != plan["plan_id"]:
            log(f"note: the folder held plan {previous}; it is replaced, so answers given to it no longer match")
        plan_path.write_text(plan_to_json(plan), encoding="utf-8")
        write_page(options.out, plan)
        result.plan = plan
        return result
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _photos_word(count: int) -> str:
    return "photo" if count == 1 else "photos"


def summary_text(result: PrepareResult, out: Path, ym_name: str) -> str:
    """The summary `prepare` prints: Apple's model, the photos, how many Apple Auto left as they were and how many
    it failed on, the strata, the trials, and how to open the page."""
    plan = result.plan
    counts = count_plan(plan)
    kept = len(plan["photos"])  # type: ignore[arg-type]
    provenance = plan["provenance"]  # type: ignore[index]
    apple = provenance["apple_auto"]  # type: ignore[index]
    found = result.found
    lines = [
        f"Blind A/B of Apple Auto written to {out}",
        f"  Apple Auto: {apple.get('model')} · macOS {apple.get('os')} · prompt {apple.get('prompt')} · "
        f"table {apple.get('table')} · instructions {apple.get('instructions')} ({apple.get('instructions_sha256')})",
        f"  ym: {provenance['ym']}",  # type: ignore[index]
        f"  photos: {found} found, {kept} in the test",
        f"  Apple Auto left {len(result.unchanged)} of {found} {_photos_word(found)} as they were: they give no trial.",
        f"  Apple Auto failed on {len(result.failed)} of {found} {_photos_word(found)}: they are left out.",
    ]
    for item in result.failed:
        lines.append(f"    {item['stratum']}/{item['stem']}: {item['reason']}")
    if result.errors:
        count = len(result.errors)
        lines.append(f"  {count} {_photos_word(count)} could not be rendered: left out.")
        for item in result.errors:
            lines.append(f"    {item['stratum']}/{item['stem']}: {item['reason']}")
    if result.no_camera:
        count = len(result.no_camera)
        lines.append(f"  {count} {_photos_word(count)} without a camera JPEG: only the trial against the rule.")
        for item in result.no_camera:
            lines.append(f"    {item['stratum']}/{item['stem']}: {item.get('reason', '')}")
    lines.append(
        "  strata (photos in the test): "
        + ", ".join(f"{name} {count}" for name, count in sorted(counts.by_stratum.items()))
    )
    parts = [f"{LABELS[name]} {count}" for name, count in counts.by_comparison.items()]
    lines.append(
        f"  trials: {len(plan['trials'])} ({', '.join(parts)}); seed {plan['seed']}; plan {plan['plan_id']}"  # type: ignore
    )
    if apple.get("model") == TEST_ADVISOR:
        lines.append(
            "  note: the answers came from the test advisor (YELLOWMOON_TEST_APPLE_AUTO), not from Apple's model"
        )
    lines.append("")
    lines.append(f"Open the page:  open {shlex.quote(str(out / 'index.html'))}")
    lines.append("Answer every trial, then press Export results in the page.")
    lines.append(f"Score the answers with:  tools/eval/ab/ab.sh score RESULTS_JSON   (ym: {ym_name})")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """The command line: `prepare` and `score`, with the defaults of the plan (seed, long edge, jobs, paths)."""
    parser = argparse.ArgumentParser(
        prog="ab.sh",
        description="Blind A/B of Apple Auto (docs/19 section 13): prepare the test, then score the answers.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare", help="render the pictures and write the blind page")
    prep.add_argument("photos", type=Path, metavar="PHOTOS_DIR", help="folder of ARW files; subfolders are strata")
    prep.add_argument(
        "--ym",
        type=Path,
        default=DEFAULT_YM,
        help="the ym of the test build, e.g. ~/Applications/Yellowmoon.app/Contents/MacOS/ym "
        "(default target/release/ym)",
    )
    prep.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"output folder (default {DEFAULT_OUT})")
    prep.add_argument("--long-edge", type=int, default=2000, help="long edge of the renders in pixels (default 2000)")
    prep.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"seed of the plan (default {DEFAULT_SEED})")
    prep.add_argument("--jobs", type=int, default=default_jobs(), help="photos at the same time (default: CPUs / 2)")
    prep.add_argument(
        "--apple-jobs",
        type=int,
        default=1,
        help="renders waiting for Apple's model at the same time (default 1: one question at a time)",
    )
    prep.add_argument(
        "--with-camera",
        action="store_true",
        help="also compare Apple Auto with the camera's JPEG (reported, not part of the gate)",
    )
    prep.add_argument(
        "--resume",
        action="store_true",
        help="keep the pictures an earlier run with the same photos, --out and --seed already made, and make the rest "
        "(a photo whose answer failed is asked again)",
    )
    prep.add_argument(
        "--data",
        type=Path,
        default=None,
        help="the data folder ym uses (default: the app's own for the ym in an app, else the repository's data/)",
    )
    prep.add_argument(
        "--instructions",
        type=Path,
        metavar="FILE",
        default=None,
        help="a text file of instructions for Apple's model instead of the default (docs/19 section 2; at most 2000 "
        "characters); the plan records which instructions were read",
    )
    prep.add_argument(
        "--allow-unavailable",
        action="store_true",
        help="go on when ym doctor says Apple Auto is not available (for tests of the kit)",
    )

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
    if args.apple_jobs < 1:
        raise UsageError("--apple-jobs must be at least 1")
    ym = args.ym.expanduser()
    if not (ym.is_file() and os.access(ym, os.X_OK)):
        raise UsageError(
            f"there is no ym at {ym}. Pass the one inside the test build, for example "
            "--ym ~/Applications/Yellowmoon.app/Contents/MacOS/ym"
        )
    data = args.data.expanduser() if args.data is not None else default_data(ym)
    if data is not None and not data.is_dir():
        raise UsageError(f"the data folder {data} does not exist; pass --data DIR, or leave it out")
    instructions = args.instructions.expanduser().resolve() if args.instructions is not None else None
    if instructions is not None and not instructions.is_file():
        raise UsageError(f"there is no instructions file at {instructions}")
    options = PrepareOptions(
        photos=args.photos.expanduser(),
        out=args.out.expanduser(),
        long_edge=args.long_edge,
        seed=args.seed,
        ym=ym,
        data=data,
        jobs=args.jobs,
        apple_jobs=args.apple_jobs,
        with_camera=args.with_camera,
        allow_unavailable=args.allow_unavailable,
        resume=args.resume,
        instructions=instructions,
    )
    try:
        result = prepare(options)
    except PrepareFailed as exc:
        print(f"ab.sh: {exc}", file=sys.stderr)
        return 1
    print(summary_text(result, options.out.resolve(), str(ym)))
    return 0


def run_score(args: argparse.Namespace) -> int:
    """`ab.sh score`: scores the results, prints the table and the gate, and writes score.json."""
    report, path = score_results(args.results.expanduser(), args.plan, DEFAULT_OUT)
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
    """SIGTERM is handled like Ctrl-C, so that `prepare` still removes its temporary folder."""
    raise KeyboardInterrupt


if __name__ == "__main__":
    sys.exit(main())

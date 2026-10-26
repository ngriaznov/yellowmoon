"""The pictures of the blind A/B of an Auto provider: the `ym` renders with the provider and with Default Auto, and the
camera's embedded JPEG (with --with-camera). Stdlib only.

Every `ym` call gets a new, empty home folder (so the user's preferences and imported looks cannot change a
picture), the `--cpu` renderer, the built-in default look and every setting at Auto. The two renders differ only in
`--auto-provider <p>` (`apple`; `<p>` is `apple`, `claude` or `agy`, `Pictures.provider`) and `--auto-provider
yellowmoon` (`rule`, Default Auto). Each photo is handled on its own:

1. the `apple` render, whose `--json` record says what the provider answered (`auto_advisor`, or `apple_auto` of an
   older `ym`: state, steps, scene, plan, and the instructions it read);
2. a photo whose answer failed is left out with its reason; a photo whose answer changes nothing (every step it set is
   keep or none; a step left empty sets nothing) gives no trial, because its picture is the rule's (docs/19 section 5:
   with every step at 0 the chain is today's Auto, bit for bit);
3. otherwise the `rule` render; when the two pictures are the same pixels after all, the photo gives no trial either;
4. with --with-camera, the camera's JPEG, copied byte for byte from the raw file. Its pixels are stored in sensor
   orientation, so the page turns it upright by the raw file's EXIF orientation (`raw_turns`), as `ym` turns the
   renders.

The `ym` records are kept under `records/` in the out folder (part of the key, like `plan.json`) so that `--resume`
can keep a render without asking the provider again. The originals are only ever read.

The crop test (`prepare --test crop`, docs/20 section 12) makes two renders of each photo, both at the defaults with
`--auto-provider yellowmoon`, so that only the framing differs (`process_crop_photo`):

1. the `crop` render, with `--auto-crop=original`, whose `--json` record says what Auto crop proposed (`auto_crop`:
   the straighten kind and angle, the frames, `keep`, the subject's source, or an `error`);
2. a photo where Auto crop failed is left out with its reason (its picture is the original framing); a photo where it
   changed nothing (`keep` and an angle of 0: the whole photo, not turned) **kept its frame** and gives no trial;
3. otherwise the `original` render, without the flag; when the two pictures are the same pixels after all, the photo
   gives no trial either.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import shutil
import struct
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from plan import (
    ANSWER_FIELDS,
    COLOR_FIELDS,
    CROP_ASPECT,
    CROP_TEST,
    NO_STEPS,
    STEP_RANGE,
    STRENGTH_RANGE,
    StepValue,
    Steps,
    changed_fields,
    picture_name,
    picture_relpath,
)

JPEG_START = b"\xff\xd8"
QUALITY = 92
RENDER_TIMEOUT_S = 1800.0
RECORDS_DIR = "records"
#: The variables of the caller's environment that reach `ym`. Every other YELLOWMOON_ variable is dropped, so the
#: caller's shell cannot change a picture; the test advisor (docs/19 section 11) passes, for tests and smoke runs, and
#: the plan then names it as the model.
PASS_THROUGH: Tuple[str, ...] = (
    "YELLOWMOON_TEST_APPLE_AUTO",
    # The two programs (docs/22 section 15.3, for tests and smoke runs) and the developer variables that choose a model
    # and an effort for one run (docs/22 D18; the plan records the model that answered).
    "YELLOWMOON_TEST_AUTO_PROVIDERS",
    "YELLOWMOON_CLAUDE_MODEL",
    "YELLOWMOON_CLAUDE_EFFORT",
    "YELLOWMOON_AGY_MODEL",
    "YELLOWMOON_AGY_EFFORT",
)
#: The providers the test can grade with (`prepare --provider`, docs/22 section 17), and what the person calls them.
PROVIDERS: Mapping[str, str] = {"apple": "Apple Auto", "claude": "Claude Code", "agy": "Antigravity"}
DEFAULT_PROVIDER = "apple"
#: The flag of `ym render` that gives each photo the crop the app's Auto would give (docs/20 section 8).
AUTO_CROP_FLAG = "--auto-crop"
#: The flags each rendered variant adds to the shared render flags. The crop test's two renders (`crop`, `original`)
#: both have Apple Auto off, so only the framing differs.
#: The `apple` render is made with `--auto-provider <the provider under test>` (`Pictures.provider`, apple by default);
#: the others use Default Auto.
RENDER_FLAGS: Mapping[str, Tuple[str, ...]] = {
    "apple": ("--auto-provider", DEFAULT_PROVIDER),
    "rule": ("--auto-provider", "yellowmoon"),
    "crop": ("--auto-provider", "yellowmoon", f"{AUTO_CROP_FLAG}={CROP_ASPECT}"),
    "original": ("--auto-provider", "yellowmoon"),
}
#: The renders made with Default Auto, whose record must say that Default Auto made them.
RULE_RENDERS: Tuple[str, ...] = ("rule", "crop", "original")
#: What Auto crop's straightener answered (docs/20 section 3), and why a photo it did not turn was not turned.
STRAIGHTEN_KINDS: Mapping[str, str] = {
    "level": "already level",
    "rotate": "not turned",
    "unsure": "no horizon or straight lines found",
}
#: Where Auto crop's subject came from (docs/20 section 3).
SUBJECT_SOURCES: Tuple[str, ...] = ("camera", "picture")
RECT_FIELDS: Tuple[str, ...] = ("x", "y", "width", "height")
#: Why Apple Auto is not available, in the words of docs/19 section 8 (`UnavailableReason::words`).
UNAVAILABLE_WORDS: Mapping[str, str] = {
    "not_a_mac": "this is not a Mac",
    "intel_mac": "this Mac has an Intel processor",
    "built_without_support": "this build of Yellowmoon was made without it",
    "macos_too_old": "it needs macOS 27 or later",
    "device_not_eligible": "this Mac can't run Apple Intelligence",
    "apple_intelligence_off": "Apple Intelligence is off in System Settings",
    "model_not_ready": "Apple Intelligence is still downloading its model",
    "unavailable": "Apple Intelligence is not available",
    "no_image_input": "Apple's model on this Mac can't look at pictures",
}
#: Why one answer failed, in the words of docs/19 section 8 (`FailureKind::words`).
FAILURE_WORDS: Mapping[str, str] = {
    "unavailable": "Apple Intelligence stopped being available",
    "rate_limited": "macOS asked Yellowmoon to wait",
    "timeout": "no answer within 30 seconds",
    "refused": "the model refused",
    "guardrail": "Apple's safety check stopped it",
    "unsupported_language": "the model does not support this language",
    "context_too_large": "the request was too large for the model",
    "assets_unavailable": "the model's files were not available",
    "concurrent_requests": "another request was running",
    "bad_answer": "the answer could not be read",
    "other": "an error",
}
#: The Auto values of a render that the plan keeps for the diagnostics (`auto` of `ym render --json`).
AUTO_VALUES: Tuple[str, ...] = (
    "exposure_ev",
    "kelvin",
    "tint",
    "highlights",
    "shadows",
    "black",
    "contrast",
    "saturation",
    "balance_shadows",
    "balance_highlights",
)
#: EXIF Orientation (IFD0 tag 0x0112) -> quarter turns clockwise that show the camera's JPEG upright. Sony cameras do
#: not write the mirrored values (2, 4, 5, 7), so those are shown as their unmirrored turns and the mirror is ignored.
ORIENTATION_TURNS: Mapping[int, int] = {1: 0, 2: 0, 3: 2, 4: 2, 5: 1, 6: 1, 7: 3, 8: 3}
ORIENTATION_TAG = 0x0112
TIFF_TYPE_SHORT = 3
TIFF_LITTLE_ENDIAN = b"II*\x00"
TIFF_BIG_ENDIAN = b"MM\x00*"
IFD_ENTRY_BYTES = 12


class PictureError(Exception):
    """A picture could not be made. The message says why; the photo is then left out of the test."""


#: Why Claude Code or Antigravity is not available, as the clause of docs/22 section 3.4 (`unavailable_words` of
#: `ym_core::auto_provider`); `{name}` is the program's name and `{version}` its version with a leading space.
PROGRAM_UNAVAILABLE_WORDS: Mapping[str, str] = {
    "signed_out": "{name} isn't signed in",
    "too_old": "{name}{version} is too old",
    "probe_failed": "{name} didn't answer its check",
    "unsafe_launcher": "{name} was found only as a script that Yellowmoon can't run safely",
    "not_installed": "{name} isn't installed",
    "checking": "Yellowmoon is still checking",
    "not_consented": "you haven't agreed to send photos to {name} yet",
}


def unavailable_words(reason: object, provider: str = DEFAULT_PROVIDER, tool_version: object = None) -> str:
    """Why a provider is not available, in words (for Apple Auto the words of docs/19 section 8, for the two programs
    the clause of docs/22 section 3.4); the wire name when it is one this kit does not know."""
    if reason is None:
        return "no reason given"
    if provider != DEFAULT_PROVIDER and str(reason) in PROGRAM_UNAVAILABLE_WORDS:
        version = f" {tool_version}" if isinstance(tool_version, str) and tool_version else ""
        return PROGRAM_UNAVAILABLE_WORDS[str(reason)].format(name=PROVIDERS.get(provider, provider), version=version)
    return UNAVAILABLE_WORDS.get(str(reason), str(reason))


#: Why an answer of Claude Code or Antigravity failed, in the words of docs/22 section 9.1 (`failure_words`); the kinds
#: they do not use read as Apple's.
PROGRAM_FAILURE_WORDS: Mapping[str, Mapping[str, str]] = {
    "claude": {
        "unavailable": "Claude Code stopped being available",
        "rate_limited": "Claude Code reported a rate limit",
        "timeout": "no answer within 90 seconds",
        "refused": "Claude refused",
        "not_signed_in": "Claude Code isn't signed in",
        "cli_failed": "Claude Code reported an error",
    },
    "agy": {
        "unavailable": "Antigravity stopped being available",
        "rate_limited": "Antigravity reported a rate limit",
        "timeout": "no answer within 120 seconds",
        "refused": "Antigravity refused",
        "not_signed_in": "Antigravity isn't signed in",
        "cli_failed": "Antigravity reported an error",
    },
}


def failure_words(kind: object, message: object = None, provider: str = DEFAULT_PROVIDER) -> str:
    """Why an answer failed, in words, with the program's message for a failure of kind `other` or `cli_failed`."""
    words = PROGRAM_FAILURE_WORDS.get(provider, {}).get(str(kind)) or FAILURE_WORDS.get(str(kind), str(kind))
    if kind in ("other", "cli_failed") and isinstance(message, str) and message.strip():
        return f"{words}: {message.strip()}"
    return words


def advisor_record(record: Mapping[str, object]) -> Optional[Mapping[str, object]]:
    """What the provider said about a photo: `auto_advisor` of a record (docs/22 section 14.1), else the deprecated
    `apple_auto` of a `ym` that has only that (Apple's view). None when no provider acted on the photo."""
    for key in ("auto_advisor", "apple_auto"):
        value = record.get(key)
        if isinstance(value, dict):
            return value
    return None


@dataclass(frozen=True)
class PhotoJob:
    """One photo to make pictures of: its number, stem and stratum, the raw file, and the quarter turns that show the
    camera's JPEG upright (`raw_turns`)."""

    number: int
    stem: str
    stratum: str
    raw: Path
    camera_turns: int = 0


@dataclass(frozen=True)
class Pictures:
    """Where and how the pictures are made."""

    ym: Path
    #: The data folder `ym` is given (`YELLOWMOON_DATA`), or None to let `ym` find its own (the app bundle's).
    data: Optional[Path]
    out: Path
    scratch: Path
    seed: int
    long_edge: int
    with_camera: bool = False
    timeout_s: float = RENDER_TIMEOUT_S
    #: Keep a picture an earlier run already made (`prepare --resume`). `ym` and the camera copy write through a
    #: temporary file and a rename, so a picture that is there is whole; it must still start like a JPEG, and a render
    #: is kept only with its record (an `apple` render only when its answer arrived).
    resume: bool = False
    #: A text file of instructions for Apple's model (`prepare --instructions`), passed to the `apple` render as
    #: `--auto-instructions`; None for the default instructions (each `ym` call has an empty home folder, so the
    #: app's own instructions never reach the test).
    instructions: Optional[Path] = None
    #: The digest of the instructions the `apple` renders must report (`instructions_sha256`): with --resume, an `apple`
    #: render made with other instructions is made again. None does not check.
    instructions_sha256: Optional[str] = None
    #: The provider under test (`prepare --provider`): the `apple` render is made with `--auto-provider <provider>`.
    provider: str = DEFAULT_PROVIDER


@dataclass
class Outcome:
    """What happened to one photo. `status` is `trial` (it is in the test), `unchanged` (Apple's answer changed
    nothing, or in the crop test the photo kept its frame: no trial), `failed` (Apple's answer or Auto crop failed:
    left out) or `error` (a render failed: left out)."""

    status: str
    reason: str = ""
    steps: Steps = NO_STEPS
    scene: str = ""
    plan: str = ""
    #: Which instructions Apple's model read (`default` or `edited`) and their digest, from the `apple` render's record.
    instructions: Optional[Tuple[str, str]] = None
    #: The Auto values of each render (`apple`, `rule`), for the diagnostics.
    values: Dict[str, Dict[str, object]] = field(default_factory=dict)
    has_camera: bool = False
    #: Why the camera's JPEG could not be copied (--with-camera); the photo then gives only the trial against the rule.
    camera_problem: str = ""
    #: In the crop test, what Auto crop proposed (the summary of `crop_answer`), when it answered.
    crop: Optional[Dict[str, object]] = None
    #: How long the provider's render took (the whole `ym` call, the picture included), in seconds; None for a render
    #: kept by --resume, which asked nothing.
    seconds: Optional[float] = None
    #: The model that answered (`model` of `auto_advisor`: for Claude Code the id the service served), when it answered.
    model: Optional[str] = None


def ym_environment(data: Optional[Path], home: Path, base: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """The environment of one `ym` call: the caller's, minus every YELLOWMOON_ variable but the test advisor's, plus
    this call's home, the data folder when one is given, and the CPU renderer."""
    source = os.environ if base is None else base
    env = {
        key: value for key, value in source.items() if not key.startswith("YELLOWMOON_") or key in PASS_THROUGH
    }
    if data is not None:
        env["YELLOWMOON_DATA"] = str(data)
    env["YELLOWMOON_HOME"] = str(home)
    env["YELLOWMOON_GPU"] = "off"
    return env


def json_object(stdout: bytes) -> Optional[Dict[str, object]]:
    """The JSON object `ym --json` printed: the whole output if it is one, else its last non-empty line, else None."""
    text = stdout.decode("utf-8", errors="replace")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict):
        return value
    for line in reversed(text.splitlines()):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None
    return None


def call_ym(
    ym: Path, data: Optional[Path], scratch: Path, args: Sequence[str], timeout_s: float
) -> Tuple[int, Optional[Dict[str, object]], str]:
    """Runs `ym` with a new empty home. Returns its exit code, the JSON object it printed (or None) and the last
    line of its stderr. Raises PictureError when `ym` cannot start or takes too long."""
    scratch.mkdir(parents=True, exist_ok=True)
    home = Path(tempfile.mkdtemp(prefix="ym-home-", dir=str(scratch)))
    try:
        try:
            proc = subprocess.run(
                [str(ym), *args],
                env=ym_environment(data, home),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise PictureError(f"ym took longer than {timeout_s:.0f} s") from None
        except OSError as exc:
            raise PictureError(f"ym could not start: {exc}") from None
    finally:
        shutil.rmtree(home, ignore_errors=True)
    lines = proc.stderr.decode("utf-8", errors="replace").strip().splitlines()
    return proc.returncode, json_object(proc.stdout), (lines[-1] if lines else "")


def run_ym(pictures: Pictures, args: Sequence[str]) -> Dict[str, object]:
    """Runs `ym` for one file and returns the JSON object it printed for it.

    Raises PictureError with the reason when `ym` cannot start, takes too long, prints no result, or reports an error.
    """
    code, record, last_error = call_ym(pictures.ym, pictures.data, pictures.scratch, args, pictures.timeout_s)
    if record is None:
        detail = f": {last_error}" if last_error else ""
        raise PictureError(f"ym printed no result (exit {code}){detail}")
    if code != 0 or record.get("status") != "ok":
        error = record.get("error")
        message = error.get("message") if isinstance(error, dict) else None
        raise PictureError(str(message or f"ym exited with status {code}"))
    return record


def record_path(pictures: Pictures, variant: str, job: PhotoJob) -> Path:
    """Where the `ym` record of a render is kept: `records/<variant>/<picture name>.json` (part of the key)."""
    return pictures.out / RECORDS_DIR / variant / f"{picture_name(job.number, job.stem)}.json"


def picture_path(pictures: Pictures, variant: str, job: PhotoJob) -> Path:
    """Where a picture of a photo is written."""
    return pictures.out / picture_relpath(pictures.seed, variant, job.number, job.stem)


def write_atomically(target: Path, blob: bytes) -> None:
    """Writes `blob` to `target` through a temporary file and a rename, so a file that is there is whole."""
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    part.write_bytes(blob)
    os.replace(part, target)


def picture_is_made(path: Path) -> bool:
    """Whether a picture is already there: a file that starts like a JPEG."""
    try:
        with path.open("rb") as handle:
            return handle.read(2) == JPEG_START
    except OSError:
        return False


def kept_record(pictures: Pictures, variant: str, job: PhotoJob) -> Optional[Dict[str, object]]:
    """With --resume, the record of a render an earlier run made, when its picture is there and the record can be used
    again: the rule's and the original framing's always, Apple's only when its answer arrived and Auto crop's only when
    it answered (a failure is asked again)."""
    if not pictures.resume or not picture_is_made(picture_path(pictures, variant, job)):
        return None
    try:
        record = json.loads(record_path(pictures, variant, job).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(record, dict):
        return None
    if variant == "apple":
        apple = advisor_record(record)
        if not (isinstance(apple, dict) and apple.get("state") == "answered"):
            return None
        if apple.get("provider", DEFAULT_PROVIDER) != pictures.provider:
            return None
        wanted = pictures.instructions_sha256
        if wanted is not None and apple.get("instructions_sha256") != wanted:
            return None
    elif record.get("auto_source") != "rule":
        return None
    elif variant == "crop" and not crop_answer(record).answered:
        return None
    return record


def instructions_flag(pictures: Pictures, variant: str) -> List[str]:
    """`--auto-instructions FILE` for an `apple` render with instructions of its own, else nothing."""
    if variant != "apple" or pictures.instructions is None:
        return []
    return ["--auto-instructions", str(pictures.instructions)]


def render_picture(
    pictures: Pictures, job: PhotoJob, variant: str, seconds: Optional[List[float]] = None
) -> Dict[str, object]:
    """Renders one photo with the provider under test (`apple`; Apple Auto unless `Pictures.provider` says otherwise)
    or with Default Auto (`rule`), or for the crop test with Auto crop (`crop`) or without (`original`), and returns
    `ym`'s record of it, which is also kept under `records/`. With --resume, a render an earlier run made is kept and
    its record returned. When `seconds` is given, the time `ym` took is appended to it (nothing for a kept render)."""
    kept = kept_record(pictures, variant, job)
    if kept is not None:
        return kept
    target = picture_path(pictures, variant, job)
    target.parent.mkdir(parents=True, exist_ok=True)
    args = [
        "render",
        "--cpu",
        "--quiet",
        "--json",
        "--format",
        "jpeg",
        "--quality",
        str(QUALITY),
        "--long-edge",
        str(pictures.long_edge),
        "--overwrite",
        "--name",
        picture_name(job.number, job.stem),
        "--out",
        str(target.parent),
        *(("--auto-provider", pictures.provider) if variant == "apple" else RENDER_FLAGS[variant]),
        *instructions_flag(pictures, variant),
        str(job.raw),
    ]
    started = time.monotonic()
    record = run_ym(pictures, args)
    if seconds is not None:
        seconds.append(time.monotonic() - started)
    if not target.is_file() or target.stat().st_size == 0:
        raise PictureError(f"ym wrote no picture to {target.name}")
    if variant in RULE_RENDERS and record.get("auto_source") != "rule":
        raise PictureError("the render with Default Auto did not use Default Auto")
    blob = (json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    write_atomically(record_path(pictures, variant, job), blob)
    return record


def auto_values(record: Mapping[str, object]) -> Dict[str, object]:
    """The Auto values in effect of a render (`auto` of `ym render --json`): exposure, white balance, highlights,
    shadows, black, contrast, saturation and the points of the shadows and highlights wheels."""
    auto = record.get("auto")
    if not isinstance(auto, dict):
        return {}
    return {key: auto.get(key) for key in AUTO_VALUES if key in auto}


def without_final_break(text: str) -> str:
    """The instructions of a file as `ym --auto-instructions` reads them: without the file's final line break
    (`\\n` or `\\r\\n`), which text editors add."""
    if text.endswith("\n"):
        text = text[:-1]
        if text.endswith("\r"):
            text = text[:-1]
    return text


def instructions_digest(text: str, default_sha256: str) -> str:
    """The digest `ym` reports for instructions read from a file (`instructions_sha256`, docs/19 section 8): the first
    12 hex digits of the SHA-256 of the text without the file's final line break, or `default_sha256` when the text is
    empty or only white space (the default instructions)."""
    text = without_final_break(text)
    if not text.strip():
        return default_sha256
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class AppleAnswer:
    """What the provider under test did for one photo, read from the `apple` render's record."""

    answered: bool
    reason: str = ""
    steps: Steps = NO_STEPS
    scene: str = ""
    plan: str = ""
    instructions: Optional[Tuple[str, str]] = None


def _step_value(name: str, value: object) -> Tuple[bool, StepValue]:
    """Reads one field of an answer's steps: (whether it is valid, its value). An empty field (`null`, or missing in the
    record of an older `ym`) is valid and sets nothing."""
    if value is None:
        return True, None
    if name in COLOR_FIELDS:
        return (isinstance(value, str) and value in COLOR_FIELDS[name][1]), value  # type: ignore[return-value]
    if isinstance(value, bool) or not isinstance(value, int):
        return False, None
    is_strength = any(name == strength for strength, _ in COLOR_FIELDS.values())
    low, high = STRENGTH_RANGE if is_strength else STEP_RANGE[name]
    return low <= value <= high, value


def instructions_of(apple: Mapping[str, object]) -> Optional[Tuple[str, str]]:
    """Which instructions Apple's model read (`default` or `edited`) and their digest, from `apple_auto` of a record, or
    None for a `ym` that does not say."""
    kind, digest = apple.get("instructions"), apple.get("instructions_sha256")
    if isinstance(kind, str) and isinstance(digest, str):
        return kind, digest
    return None


def apple_answer(record: Mapping[str, object], provider: str = DEFAULT_PROVIDER) -> AppleAnswer:
    """Reads the provider's answer from the record of an `apple` render (`auto_advisor`, docs/22 section 14.1; Apple's
    deprecated `apple_auto`, docs/19 section 8, for a `ym` that has only that). `provider` names it in the reasons.

    Without such an object the provider did not act on the photo: it was not available for the run (the record's
    `auto.apple` says why). A failure gives its reason in words; steps outside their range count as a bad answer, and so
    does an answer that set nothing (the core reports that as a failure already). A step left empty (`null`) sets
    nothing: it is kept as None.
    """
    name = PROVIDERS.get(provider, provider)
    apple = advisor_record(record)
    if not isinstance(apple, dict):
        auto = record.get("auto")
        inner = auto.get("apple") if isinstance(auto, dict) else None
        state = inner.get("state") if isinstance(inner, dict) else None
        if state == "unavailable":
            words = unavailable_words(inner.get("reason"), provider)  # type: ignore[union-attr]
            return AppleAnswer(False, f"{name} was not available ({words})")
        return AppleAnswer(False, f"{name} did not act on the photo (its state: {state or 'unknown'})")
    state = apple.get("state")
    if state == "unavailable":
        words = unavailable_words(apple.get("reason"), provider, apple.get("tool_version"))
        return AppleAnswer(False, f"{name} was not available ({words})")
    if state == "failed":
        auto = record.get("auto")
        inner = auto.get("apple") if isinstance(auto, dict) else None
        message = inner.get("message") if isinstance(inner, dict) else None
        return AppleAnswer(False, f"{name} failed ({failure_words(apple.get('reason'), message, provider)})")
    if state == "pending":
        return AppleAnswer(False, f"{name} failed (the answer had not arrived when ym stopped)")
    if state != "answered":
        return AppleAnswer(False, f"{name} did not answer (its state: {state})")
    raw_steps = apple.get("steps")
    if not isinstance(raw_steps, dict):
        return AppleAnswer(False, f"{name} failed ({FAILURE_WORDS['bad_answer']})")
    steps: List[StepValue] = []
    for field_name in ANSWER_FIELDS:
        valid, value = _step_value(field_name, raw_steps.get(field_name))
        if not valid:
            return AppleAnswer(False, f"{name} failed ({FAILURE_WORDS['bad_answer']})")
        steps.append(value)
    if all(value is None for value in steps):
        return AppleAnswer(False, f"{name} failed ({FAILURE_WORDS['bad_answer']})")
    scene, plan = apple.get("scene"), apple.get("plan")
    return AppleAnswer(
        True,
        steps=tuple(steps),
        scene=scene if isinstance(scene, str) else "",
        plan=plan if isinstance(plan, str) else "",
        instructions=instructions_of(apple),
    )


@dataclass(frozen=True)
class CropAnswer:
    """What Auto crop did for one photo, read from the `crop` render's record (`auto_crop`, docs/20 section 3).

    `answered` is False when it failed (`reason` says why; the picture is then the original framing). `kept` is True
    when it changed nothing: the whole frame kept (`keep`) and an angle of 0. `summary` is what the plan records."""

    answered: bool
    reason: str = ""
    kept: bool = False
    summary: Optional[Dict[str, object]] = None


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value  # not NaN


def crop_answer(record: Mapping[str, object]) -> CropAnswer:
    """Reads Auto crop's answer from the record of a `crop` render: `{"error": "<sentence>"}` is a failure with that
    sentence; an answer gives its straighten kind, angle, `keep`, the subject's source and the first frame (the frame
    the photo got). An answer that is missing or cannot be read is a failure too."""
    value = record.get("auto_crop")
    if not isinstance(value, dict):
        return CropAnswer(False, "Auto crop failed: ym did not report it (is this ym older than Auto crop?)")
    if "error" in value:
        message = value.get("error")
        words = message.strip() if isinstance(message, str) and message.strip() else "no reason given"
        return CropAnswer(False, f"Auto crop failed: {words}")
    unreadable = CropAnswer(False, "Auto crop failed: its answer could not be read")
    straighten, angle, keep = value.get("straighten"), value.get("angle_deg"), value.get("keep")
    subject, frames = value.get("subject"), value.get("frames")
    if straighten not in STRAIGHTEN_KINDS or not _is_number(angle) or not isinstance(keep, bool):
        return unreadable
    if subject not in SUBJECT_SOURCES or not isinstance(frames, list) or not frames:
        return unreadable
    first = frames[0]
    rect = first.get("rect") if isinstance(first, dict) else None
    if not isinstance(rect, dict) or not all(_is_number(rect.get(name)) for name in RECT_FIELDS):
        return unreadable
    if not (rect["width"] > 0 and rect["height"] > 0) or not _is_number(first.get("score")):
        return unreadable
    summary: Dict[str, object] = {
        "straighten": straighten,
        "angle_deg": angle,
        "keep": keep,
        "subject": subject,
        "rect": {name: rect[name] for name in RECT_FIELDS},
        "score": first["score"],
    }
    return CropAnswer(True, kept=keep and angle == 0, summary=summary)


def jpeg_pixel_digest(path: Path) -> Optional[str]:
    """A hash of what a JPEG decodes from: every segment but the metadata (APP0 to APP15 and comments), with the
    entropy-coded data. Two files with the same digest decode to the same pixels; the two renders of a photo differ in
    their metadata (`ym:Auto`, `ym:AppleAuto`, `ym:Settings`) even when their pixels are the same. None when the file
    cannot be read or is not a JPEG this walk understands.
    """
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if data[:2] != JPEG_START:
        return None
    digest = hashlib.sha256(JPEG_START)
    at = 2
    size = len(data)
    while at < size:
        if data[at] != 0xFF:
            return None
        while at < size and data[at] == 0xFF:
            at += 1  # fill bytes before a marker
        if at >= size:
            return None
        marker = data[at]
        at += 1
        if marker == 0xD9:  # end of image
            digest.update(b"\xff\xd9")
            return digest.hexdigest()
        if 0xD0 <= marker <= 0xD7 or marker == 0x01:  # markers without a length
            digest.update(bytes((0xFF, marker)))
            continue
        if at + 2 > size:
            return None
        length = int.from_bytes(data[at : at + 2], "big")
        if length < 2 or at + length > size:
            return None
        segment = data[at : at + length]
        at += length
        if 0xE0 <= marker <= 0xEF or marker == 0xFE:
            continue  # metadata: EXIF, XMP, the ICC profile, comments
        digest.update(bytes((0xFF, marker)) + segment)
        if marker == 0xDA:  # start of scan: the entropy-coded data runs to the next marker that is not a stuffed byte
            end = at
            while True:
                end = data.find(b"\xff", end)
                if end < 0 or end + 1 >= size:
                    return None
                following = data[end + 1]
                if following == 0x00 or 0xD0 <= following <= 0xD7:
                    end += 2
                    continue
                break
            digest.update(data[at:end])
            at = end
    return None


def same_pixels(first: Path, second: Path) -> bool:
    """Whether two JPEGs decode to the same pixels (`jpeg_pixel_digest`); False when either cannot be read."""
    one = jpeg_pixel_digest(first)
    return one is not None and one == jpeg_pixel_digest(second)


def raw_orientation(path: Path) -> int:
    """The EXIF orientation (1 to 8) that the raw file records in IFD0, or 1 when it has none, the value is out of
    range, or the file is missing, unreadable or not a TIFF file.

    An ARW is a TIFF file: the first eight bytes give the byte order and the offset of IFD0, and IFD0 holds the
    orientation as a SHORT (tag 0x0112). Only the header and IFD0 are read, and this never raises.
    """
    try:
        with open(path, "rb") as handle:
            header = handle.read(8)
            if header[:4] == TIFF_LITTLE_ENDIAN:
                order = "<"
            elif header[:4] == TIFF_BIG_ENDIAN:
                order = ">"
            else:
                return 1
            (ifd_offset,) = struct.unpack(order + "I", header[4:8])
            handle.seek(ifd_offset)
            (count,) = struct.unpack(order + "H", handle.read(2))
            entries = handle.read(count * IFD_ENTRY_BYTES)
    except (OSError, struct.error, ValueError, OverflowError):
        return 1
    for start in range(0, len(entries) - IFD_ENTRY_BYTES + 1, IFD_ENTRY_BYTES):
        tag, kind = struct.unpack_from(order + "HH", entries, start)
        if tag != ORIENTATION_TAG:
            continue
        if kind != TIFF_TYPE_SHORT:
            return 1
        (value,) = struct.unpack_from(order + "H", entries, start + 8)
        return value if value in ORIENTATION_TURNS else 1
    return 1


def raw_turns(path: Path) -> int:
    """The quarter turns clockwise that show the camera's JPEG of a raw file upright: its orientation, mapped."""
    return ORIENTATION_TURNS[raw_orientation(path)]


def camera_picture(pictures: Pictures, job: PhotoJob) -> None:
    """Copies the camera's embedded JPEG, byte for byte, from the raw file (its offset and length from `ym info`, run
    with Default Auto: the copy needs no answer)."""
    target = picture_path(pictures, "camera", job)
    if pictures.resume and picture_is_made(target):
        return
    record = run_ym(pictures, ["info", "--json", "--auto-provider", "yellowmoon", str(job.raw)])
    preview = record.get("preview")
    try:
        offset = int(preview["offset"])  # type: ignore[index]
        length = int(preview["length"])  # type: ignore[index]
    except (TypeError, KeyError, ValueError):
        raise PictureError("the file has no embedded JPEG") from None
    if offset < 0 or length <= 0:
        raise PictureError("the file has no embedded JPEG")
    with open(job.raw, "rb") as handle:
        handle.seek(offset)
        blob = handle.read(length)
    if len(blob) != length or blob[:2] != JPEG_START:
        raise PictureError("the embedded picture of the file is not a JPEG")
    write_atomically(target, blob)


def _problem(exc: BaseException) -> str:
    return str(exc) or type(exc).__name__


def process_photo(pictures: Pictures, job: PhotoJob, apple_gate: threading.Semaphore) -> Outcome:
    """Makes the pictures of one photo and says what became of it (`Outcome`). `apple_gate` limits how many renders
    wait for the provider's answer at the same time."""
    name = PROVIDERS.get(pictures.provider, pictures.provider)
    taken: List[float] = []
    try:
        with apple_gate:
            record = render_picture(pictures, job, "apple", taken)
    except (PictureError, OSError, ValueError) as exc:
        return Outcome("error", f"with {name} on: {_problem(exc)}")
    answer = apple_answer(record, pictures.provider)
    values: Dict[str, Dict[str, object]] = {"apple": auto_values(record)}
    seconds = taken[0] if taken else None
    if not answer.answered:
        return Outcome("failed", answer.reason, values=values, seconds=seconds)
    advisor = advisor_record(record) or {}
    model = advisor.get("model")
    said = {
        "steps": answer.steps,
        "scene": answer.scene,
        "plan": answer.plan,
        "instructions": answer.instructions,
        "seconds": seconds,
        "model": model if isinstance(model, str) and model else None,
    }
    if not changed_fields(answer.steps):
        why = "every step it set is keep or none"
        return Outcome("unchanged", why, values=values, **said)  # type: ignore[arg-type]
    try:
        rule = render_picture(pictures, job, "rule")
    except (PictureError, OSError, ValueError) as exc:
        why = f"with Default Auto: {_problem(exc)}"
        return Outcome("error", why, values=values, **said)  # type: ignore[arg-type]
    values["rule"] = auto_values(rule)
    if same_pixels(picture_path(pictures, "apple", job), picture_path(pictures, "rule", job)):
        return Outcome("unchanged", "the two pictures are the same", values=values, **said)  # type: ignore[arg-type]
    outcome = Outcome("trial", "", values=values, **said)  # type: ignore[arg-type]
    if pictures.with_camera:
        try:
            camera_picture(pictures, job)
            outcome.has_camera = True
        except (PictureError, OSError, ValueError) as exc:
            outcome.camera_problem = _problem(exc)
    return outcome


def process_crop_photo(pictures: Pictures, job: PhotoJob) -> Outcome:
    """Makes the pictures of one photo for the crop test, with Auto crop and without, and says what became of it."""
    try:
        record = render_picture(pictures, job, "crop")
    except (PictureError, OSError, ValueError) as exc:
        return Outcome("error", f"with Auto crop: {_problem(exc)}")
    answer = crop_answer(record)
    if not answer.answered:
        return Outcome("failed", answer.reason)
    if answer.kept:
        straighten = str((answer.summary or {}).get("straighten"))
        return Outcome("unchanged", STRAIGHTEN_KINDS.get(straighten, ""), crop=answer.summary)
    try:
        render_picture(pictures, job, "original")
    except (PictureError, OSError, ValueError) as exc:
        return Outcome("error", f"without Auto crop: {_problem(exc)}", crop=answer.summary)
    if same_pixels(picture_path(pictures, "crop", job), picture_path(pictures, "original", job)):
        return Outcome("unchanged", "the two pictures are the same", crop=answer.summary)
    return Outcome("trial", "", crop=answer.summary)


def make_pictures(
    pictures: Pictures,
    jobs: Sequence[PhotoJob],
    workers: int,
    apple_workers: int,
    log: Callable[[str], None],
    test: str = "apple",
) -> Dict[int, Outcome]:
    """Makes the pictures of every photo on a pool of `workers` threads (each thread waits on one `ym` process at a
    time), with at most `apple_workers` renders waiting for the provider's answer at once. Returns each photo's outcome.
    With `test` "crop" the pictures are the crop test's (`process_crop_photo`), which never wait for a provider."""
    outcomes: Dict[int, Outcome] = {}
    gate = threading.Semaphore(max(1, apple_workers))
    words = {"trial": "in the test", "unchanged": "unchanged", "failed": "left out", "error": "left out"}
    if test == CROP_TEST.name:
        words["unchanged"] = "kept its frame"
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        if test == CROP_TEST.name:
            pending = {pool.submit(process_crop_photo, pictures, job): job for job in jobs}
        else:
            pending = {pool.submit(process_photo, pictures, job, gate): job for job in jobs}
        for finished, future in enumerate(concurrent.futures.as_completed(pending), start=1):
            job = pending[future]
            outcome = future.result()
            outcomes[job.number] = outcome
            detail = f" ({outcome.reason})" if outcome.reason else ""
            log(f"[{finished}/{len(jobs)}] {job.stratum}/{job.stem}: {words[outcome.status]}{detail}")
    return outcomes

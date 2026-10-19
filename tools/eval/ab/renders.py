"""The pictures of the blind A/B: the `ym` renders of `autoset` and of the rule, and the camera's embedded JPEG.
Stdlib only.

Every `ym` call gets a new, empty home folder (so the user's preferences and imported looks cannot change a
picture), the `--cpu` renderer and the data folder of its variant. The `rule` variant runs on a temporary copy of
the data folder without `models/autoset`: `ym` then uses the rule of the time before automatic settings
(docs/18 section 4.4), with `--contrast 0 --white-preference 0` as docs/18 section 8 gives it. The camera's JPEG is
copied byte for byte from the raw file, and it is not re-encoded. Its pixels are stored in sensor orientation, so the
page turns it upright by the raw file's EXIF orientation (`raw_turns`), as `ym` turns the renders. The originals are
only ever read.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from plan import VARIANTS, picture_name, picture_relpath

JPEG_START = b"\xff\xd8"
QUALITY = 92
RENDER_TIMEOUT_S = 1800.0
#: EXIF Orientation (IFD0 tag 0x0112) -> quarter turns clockwise that show the camera's JPEG upright. Sony cameras do
#: not write the mirrored values (2, 4, 5, 7), so those are shown as their unmirrored turns and the mirror is ignored.
ORIENTATION_TURNS: Mapping[int, int] = {1: 0, 2: 0, 3: 2, 4: 2, 5: 1, 6: 1, 7: 3, 8: 3}
ORIENTATION_TAG = 0x0112
TIFF_TYPE_SHORT = 3
TIFF_LITTLE_ENDIAN = b"II*\x00"
TIFF_BIG_ENDIAN = b"MM\x00*"
IFD_ENTRY_BYTES = 12
#: The flags each rendered variant adds to the shared render flags.
RENDER_FLAGS: Mapping[str, Tuple[str, ...]] = {
    "autoset": (),
    "rule": ("--white-preference", "0", "--contrast", "0"),
}


class PictureError(Exception):
    """A picture could not be made. The message says why; the photo is then left out of the test."""


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
    data: Path
    rule_data: Path
    out: Path
    scratch: Path
    seed: int
    long_edge: int
    timeout_s: float = RENDER_TIMEOUT_S
    #: Keep a picture an earlier run already made (`prepare --resume`). Both `ym` and the camera copy write through a
    #: temporary file and a rename, so a picture that is there is whole; it must still start like a JPEG.
    resume: bool = False


def ym_environment(data: Path, home: Path, base: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """The environment of one `ym` call: the caller's, minus every YELLOWMOON_ variable, plus this call's data
    and home."""
    source = os.environ if base is None else base
    env = {key: value for key, value in source.items() if not key.startswith("YELLOWMOON_")}
    env["YELLOWMOON_DATA"] = str(data)
    env["YELLOWMOON_HOME"] = str(home)
    env["YELLOWMOON_GPU"] = "off"
    return env


def last_json_object(stdout: bytes) -> Optional[Dict[str, object]]:
    """The last non-empty line of `stdout` if it is a JSON object (what `ym --json` prints), else None."""
    for line in reversed(stdout.decode("utf-8", errors="replace").splitlines()):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None
    return None


def run_ym(pictures: Pictures, args: Sequence[str], data: Path) -> Dict[str, object]:
    """Runs `ym` with a new empty home, and returns the JSON object it printed for the file.

    Raises PictureError with the reason when `ym` cannot start, takes too long, prints no result, or reports an error.
    """
    pictures.scratch.mkdir(parents=True, exist_ok=True)
    home = Path(tempfile.mkdtemp(prefix="ym-home-", dir=str(pictures.scratch)))
    try:
        try:
            proc = subprocess.run(
                [str(pictures.ym), *args],
                env=ym_environment(data, home),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=pictures.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise PictureError(f"ym took longer than {pictures.timeout_s:.0f} s") from None
        except OSError as exc:
            raise PictureError(f"ym could not start: {exc}") from None
    finally:
        shutil.rmtree(home, ignore_errors=True)
    record = last_json_object(proc.stdout)
    if record is None:
        lines = proc.stderr.decode("utf-8", errors="replace").strip().splitlines()
        detail = f": {lines[-1]}" if lines else ""
        raise PictureError(f"ym printed no result (exit {proc.returncode}){detail}")
    if proc.returncode != 0 or record.get("status") != "ok":
        error = record.get("error")
        message = error.get("message") if isinstance(error, dict) else None
        raise PictureError(str(message or f"ym exited with status {proc.returncode}"))
    return record


def write_picture(pictures: Pictures, variant: str, job: PhotoJob, blob: bytes) -> None:
    """Writes the bytes of a picture (the camera's JPEG) to its place in the out folder, atomically."""
    target = pictures.out / picture_relpath(pictures.seed, variant, job.number, job.stem)
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    part.write_bytes(blob)
    os.replace(part, target)


def render_picture(pictures: Pictures, job: PhotoJob, variant: str) -> None:
    """Renders one photo in one of the two `ym` variants: autoset and the rule."""
    rel = picture_relpath(pictures.seed, variant, job.number, job.stem)
    target = pictures.out / rel
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
        *RENDER_FLAGS[variant],
        str(job.raw),
    ]
    data = pictures.rule_data if variant == "rule" else pictures.data
    run_ym(pictures, args, data)
    if not target.is_file() or target.stat().st_size == 0:
        raise PictureError(f"ym wrote no picture to {target.name}")


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
    """Copies the camera's embedded JPEG, byte for byte, from the raw file (its offset and length from `ym info`)."""
    record = run_ym(pictures, ["info", "--json", str(job.raw)], pictures.data)
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
    write_picture(pictures, "camera", job, blob)


def picture_is_made(pictures: Pictures, job: PhotoJob, variant: str) -> bool:
    """Whether the picture of `variant` is already in the out folder: a non-empty file that starts like a JPEG."""
    target = pictures.out / picture_relpath(pictures.seed, variant, job.number, job.stem)
    try:
        with target.open("rb") as handle:
            return handle.read(2) == JPEG_START
    except OSError:
        return False


def run_task(pictures: Pictures, job: PhotoJob, variant: str) -> Optional[str]:
    """Makes one picture, or keeps the one an earlier run made when resuming. Returns None on success, else the
    reason it failed."""
    if pictures.resume and picture_is_made(pictures, job, variant):
        return None
    try:
        if variant == "camera":
            camera_picture(pictures, job)
        else:
            render_picture(pictures, job, variant)
    except (PictureError, OSError, ValueError) as exc:
        return str(exc) or type(exc).__name__
    return None


def make_pictures(
    pictures: Pictures,
    jobs: Sequence[PhotoJob],
    workers: int,
    log: Callable[[str], None],
) -> Dict[int, List[Tuple[str, str]]]:
    """Makes every picture of every photo on a pool of `workers` threads (each thread waits on one `ym` process).

    Returns the failures: for each photo that failed in any variant, the (variant, reason) pairs.
    """
    failures: Dict[int, List[Tuple[str, str]]] = {}
    remaining = {job.number: len(VARIANTS) for job in jobs}
    finished = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        pending = {
            pool.submit(run_task, pictures, job, variant): (job, variant)
            for job in jobs
            for variant in VARIANTS
        }
        for future in concurrent.futures.as_completed(pending):
            job, variant = pending[future]
            problem = future.result()
            if problem is not None:
                failures.setdefault(job.number, []).append((variant, problem))
            remaining[job.number] -= 1
            if remaining[job.number] == 0:
                finished += 1
                state = "skipped" if job.number in failures else "done"
                log(f"[{finished}/{len(jobs)}] {job.stem}: {state}")
    return failures


def copy_rule_data(data: Path, parent: Path) -> Path:
    """A copy of the data folder without `models/autoset`, under `parent`. Returns the copy's path."""
    models = (data / "models").resolve()

    def ignore(directory: str, names: List[str]) -> List[str]:
        if Path(directory).resolve() == models:
            return [name for name in names if name == "autoset"]
        return []

    target = parent / "data"
    shutil.copytree(data, target, ignore=ignore)
    if (target / "models" / "autoset").exists():
        raise RuntimeError("the copy of the data folder still has models/autoset")
    return target

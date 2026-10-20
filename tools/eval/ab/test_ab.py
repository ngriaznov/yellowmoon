"""Tests of the blind A/B kit of Apple Auto: the plan, the pictures (through a fake `ym`), the page and the scoring.

No `ym`, no network and no raw files are needed:
`python3 -I -m unittest discover -s tools/eval/ab -p 'test_*.py'`.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import signal
import struct
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ab  # noqa: E402
import page  # noqa: E402
import plan as planlib  # noqa: E402
import renders  # noqa: E402
import score as scorelib  # noqa: E402

#: Names that must appear nowhere in the page or in plan.js: the variants, what makes them, and the parked kit's name.
FORBIDDEN_NAMES = ("apple", "camera", "rule", "intelligence", "autoset", "preference")
#: Ordinary words that must not appear in the visible markup (the CSS keyword `auto` is not visible text).
FORBIDDEN_VISIBLE_WORDS = (r"\brule\b", r"\bauto\b", r"\bwhite\b", r"\bapple\b", r"\bai\b")
MODEL = "AFM 3 Core"
MACOS = "27.0.1 (27A404)"
#: The digest of the default instructions as the fake `ym` reports it.
DEFAULT_SHA = "932b96907b63"


def S(
    exposure: Optional[int] = None,
    white: Optional[int] = None,
    contrast: Optional[int] = None,
    saturation: Optional[int] = None,
    **more: object,
) -> planlib.Steps:
    """A photo's steps in the order of ANSWER_FIELDS: the fields given, every other left empty (None: nothing set)."""
    named = {"exposure": exposure, "white": white, "contrast": contrast, "saturation": saturation, **more}
    unknown = set(named) - set(planlib.ANSWER_FIELDS)
    assert not unknown, unknown
    return tuple(named.get(name) for name in planlib.ANSWER_FIELDS)  # type: ignore[misc]


def visible_markup(text: str) -> str:
    """The page without its <style> and <script> blocks: what a reader sees, and what the screen reader names."""
    without_style = re.sub(r"<style\b.*?</style>", "", text, flags=re.S | re.I)
    return re.sub(r"<script\b.*?</script>", "", without_style, flags=re.S | re.I)


#: A stand-in for `ym`. It records each call (arguments, the YELLOWMOON_ variables it sees, whether the home folder is
#: empty), answers `doctor --json` (unavailable with FAKE_YM_UNAVAILABLE set to a reason), `info --json` (the embedded
#: JPEG's place from the fake raw file) and `render --json` with the JSON shapes of docs/19 section 8, and writes a
#: small JPEG whose metadata names the Auto that made it and whose scan data stands for its pixels. Per stem:
#: FAKE_YM_FAIL (the render fails), FAKE_YM_APPLE_FAIL (the answer times out), FAKE_YM_APPLE_OTHER (the answer fails
#: with Apple's message), FAKE_YM_APPLE_GONE (Apple Auto unavailable for the run), FAKE_YM_SAME_PIXELS (an answer that
#: changes no pixel), FAKE_YM_STEPS (a JSON object stem -> [exposure, white, contrast, saturation], the other fields
#: left empty, or stem -> the whole steps object; default [1, 0, 1, 0]) and FAKE_YM_SLEEP (seconds an Apple render
#: takes). It reads the instructions of `--apple-auto-instructions FILE` as `ym` does and reports their digest.
FAKE_YM = r'''#!/usr/bin/env python3
import json
import os
import sys
import time

import hashlib

args = sys.argv[1:]
log = os.environ.get("FAKE_YM_LOG")
MODEL = "AFM 3 Core"
MACOS = "27.0.1 (27A404)"
DEFAULT_SHA = "932b96907b63"
FIELDS = ["exposure", "highlights", "shadows", "blacks", "contrast", "white", "tint", "saturation", "shadow_color",
          "shadow_color_strength", "highlight_color", "highlight_color_strength"]


def instructions_id():
    if "--apple-auto-instructions" not in args:
        return {"instructions": "default", "instructions_sha256": DEFAULT_SHA}
    with open(args[args.index("--apple-auto-instructions") + 1], encoding="utf-8") as handle:
        text = handle.read()
    if text.endswith("\n"):
        text = text[:-1]
        if text.endswith("\r"):
            text = text[:-1]
    if not text.strip():
        return {"instructions": "default", "instructions_sha256": DEFAULT_SHA}
    return {"instructions": "edited", "instructions_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]}


def record(**fields):
    if log:
        with open(log, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(fields) + "\n")


def stems(name):
    return [part for part in os.environ.get(name, "").split(",") if part]


def segment(marker, payload):
    return bytes((0xFF, marker)) + (len(payload) + 2).to_bytes(2, "big") + payload


def jpeg(meta, pixels):
    return (b"\xff\xd8" + segment(0xE1, meta) + segment(0xDB, b"tables") + segment(0xDA, b"scan")
            + pixels + b"\xff\xd9")


if args == ["--version"]:
    print("ym 9.9.9-test (pipeline 0.0.0, fake)")
    sys.exit(0)
home = os.environ.get("YELLOWMOON_HOME", "")
data = os.environ.get("YELLOWMOON_DATA")
home_empty = not os.listdir(home)
seen = sorted(key for key in os.environ if key.startswith("YELLOWMOON_"))
test_advisor = os.environ.get("YELLOWMOON_TEST_APPLE_AUTO")
if args[:1] == ["doctor"]:
    record(cmd="doctor", args=args, home_empty=home_empty, seen=seen, data=data)
    reason = os.environ.get("FAKE_YM_UNAVAILABLE", "")
    if reason:
        apple = {"available": False, "reason": reason, "model": None, "os": None, "preference": True,
                 "prompt": 3, "table": 2, "instructions": "default", "instructions_sha256": DEFAULT_SHA}
    else:
        apple = {"available": True, "reason": None, "model": MODEL, "os": MACOS, "preference": True,
                 "prompt": 3, "table": 2, "instructions": "default", "instructions_sha256": DEFAULT_SHA}
    print(json.dumps({"apple_auto": apple, "version": "9.9.9"}))
    sys.exit(0)
if args[:1] == ["info"]:
    src = args[-1]
    with open(src, "rb") as handle:
        head = handle.read(256)
    start = head.find(b"FAKEARW")
    first = head[start:].split(b"\n", 1)[0].decode("ascii", "replace").split() if start >= 0 else []
    fields = dict(part.split("=", 1) for part in first[1:] if "=" in part)
    stem = os.path.splitext(os.path.basename(src))[0]
    record(cmd="info", stem=stem, args=args, home_empty=home_empty, seen=seen, data=data)
    preview = None
    if "OFFSET" in fields:
        preview = {"offset": int(fields["OFFSET"]), "length": int(fields["LENGTH"])}
    print(json.dumps({"status": "ok", "error": None, "input": src, "preview": preview}))
    sys.exit(0)
if args[:1] == ["render"]:
    src = args[-1]
    stem = os.path.splitext(os.path.basename(src))[0]
    out = args[args.index("--out") + 1]
    name = args[args.index("--name") + 1]
    mode = args[args.index("--apple-auto") + 1]
    record(cmd="render", stem=stem, args=args, mode=mode, home_empty=home_empty, seen=seen, data=data,
           start=time.monotonic())
    if stem in stems("FAKE_YM_FAIL"):
        message = "fake failure for " + stem
        print(json.dumps({"status": "error", "error": {"kind": "unsupported", "message": message}, "input": src,
                          "output": None}))
        sys.exit(1)
    ids = instructions_id()
    info = {"model": MODEL, "os": MACOS, **ids}
    apple_auto = None
    applied = [0, 0, 0, 0]
    if mode == "off":
        auto_apple = {"state": "off"}
    elif stem in stems("FAKE_YM_APPLE_FAIL"):
        auto_apple = {"state": "failed", "kind": "timeout", "message": "no answer", "info": info}
        apple_auto = {"state": "failed", "reason": "timeout", "model": MODEL, "os": MACOS, "prompt": 3, "table": 2,
                      **ids}
    elif stem in stems("FAKE_YM_APPLE_OTHER"):
        auto_apple = {"state": "failed", "kind": "other", "message": "Something odd happened", "info": info}
        apple_auto = {"state": "failed", "reason": "other", "model": MODEL, "os": MACOS, "prompt": 3, "table": 2,
                      **ids}
    elif stem in stems("FAKE_YM_APPLE_GONE"):
        auto_apple = {"state": "unavailable", "reason": "apple_intelligence_off"}
    else:
        steps = json.loads(os.environ.get("FAKE_YM_STEPS", "{}")).get(stem, [1, 0, 1, 0])
        if isinstance(steps, list):
            steps = dict(zip(["exposure", "white", "contrast", "saturation"], steps))
        named = {name: steps.get(name) for name in FIELDS}
        scene = "Scene of " + stem
        plan = "Plan for " + stem
        auto_apple = {"state": "answered", "answer": {"steps": named, "scene": scene, "plan": plan}, "info": info}
        apple_auto = {"state": "answered", "reason": None, "model": MODEL, "os": MACOS, "prompt": 3, "table": 2,
                      **ids, "steps": named, "scene": scene, "plan": plan}
        applied = [named[name] or 0 for name in ("exposure", "white", "contrast", "saturation")]
        if named["shadow_color"] not in (None, "none") and named["shadow_color_strength"]:
            applied.append(named["shadow_color"])
    source = "apple" if apple_auto is not None and apple_auto["state"] == "answered" else "rule"
    if apple_auto is not None and apple_auto["state"] == "failed":
        source = "rule"
    shown = [0, 0, 0, 0] if stem in stems("FAKE_YM_SAME_PIXELS") else applied
    pixels = ("pixels of %s %s" % (stem, shown)).encode("ascii")
    os.makedirs(out, exist_ok=True)
    target = os.path.join(out, name + ".jpg")
    with open(target, "wb") as handle:
        handle.write(jpeg(("Auto=" + source).encode("ascii"), pixels))
    if mode == "on":
        time.sleep(float(os.environ.get("FAKE_YM_SLEEP", "0")))
    record(cmd="rendered", stem=stem, mode=mode, end=time.monotonic())
    auto = {"apple": auto_apple, "contrast": 15.0 * applied[2], "exposure_ev": 0.5 + 0.3 * applied[0],
            "kelvin": 5000.0 - 100 * applied[1], "saturation": 15.0 * applied[3], "source": source, "tint": 0.0}
    obj = {"input": src, "output": target, "status": "ok", "auto": auto, "auto_source": source}
    if apple_auto is not None:
        obj["apple_auto"] = apple_auto
    obj["settings"] = {"contrast": {"kind": "manual", "value": auto["contrast"]}}
    obj["error"] = None
    print(json.dumps(obj))
    sys.exit(0)
sys.exit(2)
'''


def camera_bytes(stem: str) -> bytes:
    """The embedded JPEG the fake raw files carry for a photo."""
    return b"\xff\xd8\xff\xe0camera of " + stem.encode() + b"\xff\xd9"


def tiff_file(
    orientation: Optional[int] = 1,
    big_endian: bool = False,
    tag: int = 0x0112,
    orientation_type: int = 3,
) -> bytes:
    """A TIFF header with IFD0 at offset 8: one entry for `tag` (the Orientation by default, a SHORT unless
    `orientation_type` says otherwise) holding `orientation`, or no entry at all when `orientation` is None."""
    order = ">" if big_endian else "<"
    magic = b"MM\x00*" if big_endian else b"II*\x00"
    entries = b""
    count = 0
    if orientation is not None:
        count = 1
        value = struct.pack(order + "H", orientation) + b"\x00\x00"
        entries = struct.pack(order + "HHI", tag, orientation_type, 1) + value
    ifd = struct.pack(order + "H", count) + entries + struct.pack(order + "I", 0)
    return magic + struct.pack(order + "I", 8) + ifd


def write_arw(path: Path, jpeg: Optional[bytes], orientation: Optional[int] = 1) -> None:
    """A fake ARW: a TIFF header whose IFD0 holds the orientation (none when `orientation` is None), a line with the
    embedded JPEG's offset (256) and length that the fake `ym` reads, the JPEG at offset 256, then some raw bytes. With
    `jpeg` None the line says there is no preview."""
    path.parent.mkdir(parents=True, exist_ok=True)
    head = tiff_file(orientation).ljust(64, b" ")
    if jpeg is None:
        body = head + b"FAKEARW NOPREVIEW\n" + b"x" * 300
    else:
        line = f"FAKEARW OFFSET=256 LENGTH={len(jpeg)}\n".encode("ascii")
        body = (head + line).ljust(256, b" ") + jpeg
    path.write_bytes(body + b"RAWDATA" * 40)


def jpeg_bytes(meta: bytes, scan: bytes, tables: bytes = b"tables", extra: bytes = b"") -> bytes:
    """A small JPEG of segments: an APP1 with `meta`, a DQT with `tables`, an SOS and its scan data, then EOI."""

    def segment(marker: int, payload: bytes) -> bytes:
        return bytes((0xFF, marker)) + (len(payload) + 2).to_bytes(2, "big") + payload

    return b"\xff\xd8" + segment(0xE1, meta) + extra + segment(0xDB, tables) + segment(0xDA, b"sos") + scan + b"\xff\xd9"


class CountingReader(io.BytesIO):
    """A BytesIO that counts the bytes it hands out, to show how much of a file the code reads."""

    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        self.read_total = 0

    def read(self, size: Optional[int] = -1) -> bytes:
        chunk = super().read(size)
        self.read_total += len(chunk)
        return chunk


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_log(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def plan_photos(count: int, with_camera: bool = True) -> List[planlib.PlanPhoto]:
    """`count` photos in four strata, each with a camera picture (or none), steps that vary with the number."""
    photos = []
    for number in range(1, count + 1):
        stratum = ("portrait", "landscape", "night", "other")[number % 4]
        steps = S((number % 3) - 1, 0, 1, 0)
        photos.append(
            planlib.PlanPhoto(
                number=number,
                stem=f"DSC{number:05d}",
                stratum=stratum,
                has_camera=with_camera,
                steps=steps if planlib.changed_fields(steps) else S(1, 0, 0, 0),
                scene=f"scene {number}",
            )
        )
    return photos


class PlanTests(unittest.TestCase):
    """The photos, the strata and the trial plan."""

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="ab-plan-"))
        self.addCleanup(shutil.rmtree, self.root, True)

    def test_find_photos_takes_strata_from_folders_and_skips_hidden_files(self) -> None:
        for relpath in (
            "top.ARW",
            "portrait/a.arw",
            "portrait/deeper/b.ARW",
            "night/c.Arw",
            "portrait/notes.txt",
            "portrait/d.jpg",
            "portrait/._e.ARW",
            ".hidden/f.ARW",
        ):
            target = self.root / relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"x")
        found = planlib.find_photos(self.root)
        self.assertEqual(
            [(photo.stratum, photo.stem) for photo in found],
            [("night", "c"), ("other", "top"), ("portrait", "a"), ("portrait", "b")],
        )

    def test_the_plan_is_deterministic_for_a_seed(self) -> None:
        photos = plan_photos(40)
        first = planlib.build_trials(photos, 7)
        second = planlib.build_trials(photos, 7)
        other = planlib.build_trials(photos, 8)
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)

    def test_each_photo_has_one_trial_against_the_rule_and_one_against_its_camera_jpeg(self) -> None:
        photos = plan_photos(12)
        plan = planlib.build_plan(photos, 1, {"ym": "ym"})
        self.assertEqual(len(plan["trials"]), 2 * len(photos))  # type: ignore[arg-type]
        self.assertEqual(plan["comparisons"], ["rule", "camera"])
        self.assertEqual(plan["gated"], ["rule"])
        for photo in photos:
            mine = [t for t in plan["trials"] if t["number"] == photo.number]  # type: ignore[union-attr]
            self.assertEqual(sorted(t["comparison"] for t in mine), ["camera", "rule"], f"photo {photo.number}")
            for trial in mine:
                other = planlib.COMPARISONS[trial["comparison"]]
                self.assertEqual({trial["left"], trial["right"]}, {"apple", other})

    def test_without_camera_pictures_there_is_only_the_comparison_with_the_rule(self) -> None:
        photos = plan_photos(9, with_camera=False)
        plan = planlib.build_plan(photos, 1, {"ym": "ym"})
        self.assertEqual(len(plan["trials"]), len(photos))  # type: ignore[arg-type]
        self.assertEqual({t["comparison"] for t in plan["trials"]}, {"rule"})  # type: ignore[union-attr]
        self.assertEqual(plan["comparisons"], ["rule"])
        for entry in plan["photos"]:  # type: ignore[union-attr]
            self.assertEqual(set(entry["pictures"]), {"apple", "rule"})

    def test_a_photo_without_a_camera_picture_gives_only_the_trial_against_the_rule(self) -> None:
        photos = plan_photos(6)
        photos[2] = planlib.PlanPhoto(number=3, stem=photos[2].stem, stratum=photos[2].stratum, steps=S(1, 0, 0, 0))
        plan = planlib.build_plan(photos, 1, {"ym": "ym"})
        mine = [t["comparison"] for t in plan["trials"] if t["number"] == 3]  # type: ignore[union-attr]
        self.assertEqual(mine, ["rule"])
        self.assertEqual(len(plan["trials"]), 11)  # type: ignore[arg-type]

    def test_sides_are_balanced_within_each_comparison(self) -> None:
        photos = plan_photos(301)
        trials = planlib.build_trials(photos, 3)
        for name in planlib.COMPARISONS:
            group = [t for t in trials if t.comparison == name]
            self.assertEqual(len(group), 301)
            apple_left = sum(1 for t in group if t.left == "apple")
            self.assertLessEqual(abs(apple_left - len(group) / 2), 0.5, name)
            for t in group:
                self.assertEqual({t.left, t.right}, {"apple", planlib.COMPARISONS[name]})

    def test_the_odd_trial_of_a_comparison_goes_either_side(self) -> None:
        # One photo gives one trial per comparison, an odd count: over many seeds the trial should land on each side.
        photos = plan_photos(1)
        left_is_apple = 0
        for seed in range(100):
            for trial in planlib.build_trials(photos, seed):
                if trial.comparison == "rule" and trial.left == "apple":
                    left_is_apple += 1
        self.assertTrue(30 <= left_is_apple <= 70, left_is_apple)

    def test_trials_are_shuffled_and_numbered_from_one(self) -> None:
        photos = plan_photos(30)
        trials = planlib.build_trials(photos, 5)
        self.assertEqual([t.id for t in trials], list(range(1, len(trials) + 1)))
        numbers = [t.number for t in trials]
        self.assertNotEqual(numbers, sorted(numbers))

    def test_the_plan_id_changes_with_the_provenance(self) -> None:
        photos = plan_photos(6)
        first = planlib.build_plan(photos, 1, {"ym": "ym 1.0"})
        same = planlib.build_plan(photos, 1, {"ym": "ym 1.0"})
        later = planlib.build_plan(photos, 1, {"ym": "ym 1.1"})
        other_model = planlib.build_plan(photos, 1, {"ym": "ym 1.0", "apple_auto": {"model": "AFM 4"}})
        self.assertEqual(first["plan_id"], same["plan_id"])
        self.assertNotEqual(first["plan_id"], later["plan_id"])
        self.assertNotEqual(first["plan_id"], other_model["plan_id"])

    def test_the_plan_id_changes_when_an_answer_changes(self) -> None:
        photos = plan_photos(6)
        other = list(photos)
        other[1] = planlib.PlanPhoto(
            number=2, stem=photos[1].stem, stratum=photos[1].stratum, has_camera=True, steps=S(-2, 0, 0, 0)
        )
        self.assertNotEqual(
            planlib.build_plan(photos, 1, {"ym": "ym"})["plan_id"],
            planlib.build_plan(other, 1, {"ym": "ym"})["plan_id"],
        )

    def test_the_details_do_not_change_the_plan_id(self) -> None:
        photos = plan_photos(4)
        bare = planlib.build_plan(photos, 1, {"ym": "ym"})
        detailed = planlib.build_plan(photos, 1, {"ym": "ym"}, {1: {"auto_values": {"apple": {"contrast": 15.0}}}})
        self.assertEqual(bare["plan_id"], detailed["plan_id"])
        first = next(p for p in detailed["photos"] if p["number"] == 1)  # type: ignore[union-attr]
        self.assertEqual(first["auto_values"], {"apple": {"contrast": 15.0}})

    def test_the_plan_id_changes_when_a_camera_turn_changes(self) -> None:
        photos = plan_photos(6)
        turned = list(photos)
        turned[2] = planlib.PlanPhoto(
            number=3, stem=photos[2].stem, stratum=photos[2].stratum, camera_turns=1, has_camera=True,
            steps=photos[2].steps, scene=photos[2].scene,
        )
        upright = planlib.build_plan(photos, 1, {"ym": "ym"})
        again = planlib.build_plan(photos, 1, {"ym": "ym"})
        sideways = planlib.build_plan(turned, 1, {"ym": "ym"})
        self.assertEqual(upright["plan_id"], again["plan_id"])
        self.assertNotEqual(upright["plan_id"], sideways["plan_id"])

    def test_trials_carry_the_steps_and_scene_of_their_photo(self) -> None:
        photos = plan_photos(8)
        plan = planlib.build_plan(photos, 2, {"ym": "ym"})
        by_number = {photo.number: photo for photo in photos}
        for trial in plan["trials"]:  # type: ignore[union-attr]
            photo = by_number[trial["number"]]
            self.assertEqual(trial["steps"], planlib.steps_dict(photo.steps))
            self.assertEqual(trial["scene"], photo.scene)

    def test_no_picture_path_in_the_plan_contains_a_variant_name(self) -> None:
        photos = plan_photos(8)
        plan = planlib.build_plan(photos, 9, {"ym": "ym"})
        pattern = re.compile(r"^pictures/[0-9a-f]{8}/\d{4}_DSC\d{5}\.jpg$")
        paths: List[str] = []
        for entry in plan["photos"]:
            paths.extend(entry["pictures"].values())
        for trial in plan["trials"]:
            paths.extend(trial["pictures"].values())
        self.assertEqual(len(paths), len(planlib.VARIANTS) * len(photos) + 2 * len(plan["trials"]))
        for path in paths:
            self.assertRegex(path, pattern)
            for word in planlib.VARIANTS:
                self.assertNotIn(word, path.lower())
        self.assertEqual(len({v["code"] for v in plan["variants"].values()}), len(plan["variants"]))


class RawOrientationTests(unittest.TestCase):
    """The orientation a raw file records in IFD0, and the turns the page takes from it: synthetic TIFF headers."""

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="ab-orient-"))
        self.addCleanup(shutil.rmtree, self.root, True)

    def write(self, data: bytes, name: str = "raw.ARW") -> Path:
        path = self.root / name
        path.write_bytes(data)
        return path

    def test_little_endian_orientations_are_read(self) -> None:
        for value in (1, 3, 6, 8):
            self.assertEqual(renders.raw_orientation(self.write(tiff_file(value))), value, value)

    def test_big_endian_orientations_are_read(self) -> None:
        for value in (1, 3, 6, 8):
            path = self.write(tiff_file(value, big_endian=True))
            self.assertEqual(renders.raw_orientation(path), value, value)

    def test_the_mirrored_values_are_read_as_recorded(self) -> None:
        for value in (2, 4, 5, 7):
            self.assertEqual(renders.raw_orientation(self.write(tiff_file(value))), value, value)

    def test_a_missing_orientation_is_one(self) -> None:
        self.assertEqual(renders.raw_orientation(self.write(tiff_file(None))), 1)

    def test_another_tag_is_not_the_orientation(self) -> None:
        self.assertEqual(renders.raw_orientation(self.write(tiff_file(6, tag=0x010F))), 1)

    def test_out_of_range_values_are_one(self) -> None:
        for value in (0, 9, 255, 65535):
            self.assertEqual(renders.raw_orientation(self.write(tiff_file(value))), 1, value)

    def test_an_orientation_that_is_not_a_short_is_one(self) -> None:
        self.assertEqual(renders.raw_orientation(self.write(tiff_file(6, orientation_type=4))), 1)

    def test_a_truncated_file_is_one_until_the_orientation_entry_is_complete(self) -> None:
        full = tiff_file(6)
        entry_end = 8 + 2 + 12  # header, count and the one entry: the orientation is complete at byte 22
        for length in range(len(full)):
            expected = 6 if length >= entry_end else 1
            self.assertEqual(renders.raw_orientation(self.write(full[:length])), expected, length)

    def test_garbage_and_wrong_headers_are_one(self) -> None:
        noise = bytes(random.Random(3).randrange(256) for _ in range(500))
        good = tiff_file(6)
        cases = [
            b"",
            b"not a raw file at all",
            noise,
            b"II+\x00" + good[4:],  # wrong magic
            b"II*\x00" + b"\xff\xff\xff\xff",  # IFD0 beyond the end of the file
            b"II*\x00" + struct.pack("<I", 8) + b"\xff\xff",  # 65535 entries, none present
            b"MM\x00*" + good[4:],  # big-endian header, little-endian contents
        ]
        for data in cases:
            self.assertEqual(renders.raw_orientation(self.write(data)), 1, data[:12])

    def test_a_missing_file_a_folder_and_a_bad_name_are_one(self) -> None:
        self.assertEqual(renders.raw_orientation(self.root / "missing.ARW"), 1)
        self.assertEqual(renders.raw_orientation(self.root), 1)
        self.assertEqual(renders.raw_orientation(Path("bad\x00name.ARW")), 1)

    def test_only_the_header_and_ifd0_are_read(self) -> None:
        reader = CountingReader(tiff_file(6) + b"\x00" * (4 * 1024 * 1024))
        with mock.patch.object(renders, "open", create=True, side_effect=lambda *args, **kwargs: reader):
            self.assertEqual(renders.raw_orientation(self.root / "big.ARW"), 6)
        self.assertLess(reader.read_total, 1024)

    def test_the_mapping_turns_orientations_clockwise(self) -> None:
        self.assertEqual(
            dict(renders.ORIENTATION_TURNS),
            {1: 0, 2: 0, 3: 2, 4: 2, 5: 1, 6: 1, 7: 3, 8: 3},
        )
        self.assertEqual(sorted(renders.ORIENTATION_TURNS), list(range(1, 9)))

    def test_raw_turns_maps_the_file_orientation_to_quarter_turns(self) -> None:
        for orientation, turns in ((1, 0), (6, 1), (3, 2), (8, 3)):
            self.assertEqual(renders.raw_turns(self.write(tiff_file(orientation))), turns, orientation)
        self.assertEqual(renders.raw_turns(self.root / "missing.ARW"), 0)



class TurnTests(unittest.TestCase):
    """The turns of each picture in the plan and in `plan.js`: a render is upright, the camera's JPEG takes its raw's."""

    def photos(self) -> List[planlib.PlanPhoto]:
        return [
            planlib.PlanPhoto(number=1, stem="DSC00001", stratum="portrait", camera_turns=1, has_camera=True),
            planlib.PlanPhoto(number=2, stem="DSC00002", stratum="landscape", has_camera=True),
            planlib.PlanPhoto(number=3, stem="DSC00003", stratum="night", camera_turns=3, has_camera=True),
            planlib.PlanPhoto(number=4, stem="DSC00004", stratum="other", camera_turns=2, has_camera=True),
        ]

    def test_every_picture_of_plan_js_has_its_turns_and_renders_are_zero(self) -> None:
        photos = self.photos()
        plan = planlib.build_plan(photos, 4, {"ym": "ym"})
        text = page.plan_js(plan)
        payload = json.loads(text.split("=", 1)[1].strip().rstrip(";"))
        camera_turns = {photo.number: photo.camera_turns for photo in photos}
        by_id = {trial["id"]: trial for trial in plan["trials"]}  # type: ignore[index]
        self.assertEqual(len(payload["trials"]), len(plan["trials"]))  # type: ignore[arg-type]
        pictures = 0
        for entry in payload["trials"]:
            trial = by_id[entry["id"]]
            for side in ("left", "right"):
                picture = entry["pictures"][side]
                pictures += 1
                self.assertEqual(picture["path"], trial["pictures"][side])
                self.assertIsInstance(picture["turns"], int)
                variant = trial[side]
                expected = camera_turns[trial["number"]] if variant == "camera" else 0
                self.assertEqual(picture["turns"], expected, f"trial {trial['id']} {side} {variant}")
        self.assertEqual(pictures, 2 * len(plan["trials"]))  # type: ignore[arg-type]

    def test_plan_json_records_the_turns_of_every_picture(self) -> None:
        photos = self.photos()
        plan = planlib.build_plan(photos, 4, {"ym": "ym"})
        for entry in plan["photos"]:  # type: ignore[index]
            self.assertEqual(set(entry["turns"]), set(planlib.VARIANTS))
            self.assertEqual(entry["turns"]["apple"], 0)
            self.assertEqual(entry["turns"]["rule"], 0)
            photo = next(p for p in photos if p.number == entry["number"])
            self.assertEqual(entry["turns"]["camera"], photo.camera_turns)
        for trial in plan["trials"]:  # type: ignore[union-attr]
            self.assertEqual(set(trial["turns"]), {"left", "right"})
            for side in ("left", "right"):
                photo = next(p for p in photos if p.number == trial["number"])
                self.assertEqual(trial["turns"][side], planlib.picture_turns(photo, trial[side]))

    def test_a_plan_without_turns_still_validates_and_scores(self) -> None:
        plan, results = make_case([("rule", "a", 2, 1, 0, 0, 0), ("camera", "a", 1, 2, 0, 0, 0)])
        self.assertNotIn("turns", plan["trials"][0])  # type: ignore[index]
        scorelib.validate_plan(plan, Path("plan.json"))
        report = score_answers(plan, results)
        self.assertEqual(report["answered"], len(results["answers"]))  # type: ignore[index]


class JpegDigestTests(unittest.TestCase):
    """The pixel digest that tells two renders with the same pixels apart from two that differ."""

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="ab-jpeg-"))
        self.addCleanup(shutil.rmtree, self.root, True)

    def write(self, name: str, data: bytes) -> Path:
        path = self.root / name
        path.write_bytes(data)
        return path

    def test_the_same_pixels_with_other_metadata_have_the_same_digest(self) -> None:
        one = self.write("one.jpg", jpeg_bytes(b"ym:Auto=apple steps 1,0,1,0", b"pixels"))
        two = self.write("two.jpg", jpeg_bytes(b"ym:Auto=rule", b"pixels", extra=b"\xff\xfe\x00\x05abc"))
        self.assertIsNotNone(renders.jpeg_pixel_digest(one))
        self.assertTrue(renders.same_pixels(one, two))

    def test_other_scan_data_or_tables_differ(self) -> None:
        base = self.write("base.jpg", jpeg_bytes(b"meta", b"pixels"))
        scan = self.write("scan.jpg", jpeg_bytes(b"meta", b"pixelz"))
        tables = self.write("tables.jpg", jpeg_bytes(b"meta", b"pixels", tables=b"tablez"))
        self.assertFalse(renders.same_pixels(base, scan))
        self.assertFalse(renders.same_pixels(base, tables))

    def test_stuffed_bytes_and_restart_markers_stay_in_the_scan(self) -> None:
        first = self.write("a.jpg", jpeg_bytes(b"m", b"ab\xff\x00cd\xff\xd0ef"))
        second = self.write("b.jpg", jpeg_bytes(b"other", b"ab\xff\x00cd\xff\xd0ef"))
        third = self.write("c.jpg", jpeg_bytes(b"m", b"ab\xff\x00cd\xff\xd1ef"))
        self.assertTrue(renders.same_pixels(first, second))
        self.assertFalse(renders.same_pixels(first, third))

    def test_files_that_are_not_whole_jpegs_have_no_digest(self) -> None:
        good = jpeg_bytes(b"m", b"pixels")
        cases = [b"", b"not a jpeg", good[:-2], good[:10], b"\xff\xd8\x00\x00", b"\xff\xd8\xff\xdb\x00\x01"]
        for index, data in enumerate(cases):
            self.assertIsNone(renders.jpeg_pixel_digest(self.write(f"bad{index}.jpg", data)), data[:12])
        self.assertIsNone(renders.jpeg_pixel_digest(self.root / "missing.jpg"))
        missing = self.root / "missing.jpg"
        self.assertFalse(renders.same_pixels(missing, missing))


class AppleAnswerTests(unittest.TestCase):
    """Reading Apple's answer from the record of a render with Apple Auto on (docs/19 section 8)."""

    def answered(self, steps: object) -> Dict[str, object]:
        return {
            "auto_source": "apple",
            "apple_auto": {
                "state": "answered",
                "reason": None,
                "steps": steps,
                "scene": "A dog on a beach",
                "plan": "Warm, soft late light",
                "instructions": "edited",
                "instructions_sha256": "0123456789ab",
            },
        }

    def test_an_answer_gives_its_steps_scene_plan_and_instructions(self) -> None:
        steps = {name: None for name in planlib.ANSWER_FIELDS}
        steps.update({"exposure": -2, "highlights": -1, "blacks": 2, "white": 1, "tint": -2, "saturation": -1,
                      "shadow_color": "teal", "shadow_color_strength": 3, "highlight_color": "none"})
        answer = renders.apple_answer(self.answered(steps))
        self.assertTrue(answer.answered)
        self.assertEqual(
            answer.steps,
            S(-2, 1, None, -1, highlights=-1, blacks=2, tint=-2, shadow_color="teal", shadow_color_strength=3,
              highlight_color="none"),
        )
        self.assertEqual(answer.scene, "A dog on a beach")
        self.assertEqual(answer.plan, "Warm, soft late light")
        self.assertEqual(answer.instructions, ("edited", "0123456789ab"))

    def test_a_field_left_empty_or_missing_sets_nothing(self) -> None:
        # null is what ym prints for a field the model did not fill; a record of prompt version 1 has four fields.
        answer = renders.apple_answer(self.answered({"exposure": 1, "white": None, "contrast": 0, "saturation": 0}))
        self.assertTrue(answer.answered)
        self.assertEqual(answer.steps, S(1, None, 0, 0))
        self.assertEqual(planlib.changed_fields(answer.steps), ["exposure"])

    def test_steps_out_of_range_or_missing_are_a_bad_answer(self) -> None:
        for steps in (
            {"exposure": 4, "white": 0, "contrast": 0, "saturation": 0},
            {"exposure": 0, "white": -3, "contrast": 0, "saturation": 0},
            {"exposure": 0, "highlights": 3},
            {"exposure": 0, "white": 0, "contrast": True, "saturation": 0},
            {"exposure": 0.5, "white": 0, "contrast": 0, "saturation": 0},
            {"shadow_color": "purple", "shadow_color_strength": 1},
            {"highlight_color": "warm", "highlight_color_strength": 4},
            {"shadow_color": "teal", "shadow_color_strength": 0},
            {"shadow_color": 2},
            {name: None for name in planlib.ANSWER_FIELDS},
            {},
            None,
            [1, 0, 1, 0],
        ):
            answer = renders.apple_answer(self.answered(steps))
            self.assertFalse(answer.answered, steps)
            self.assertIn("the answer could not be read", answer.reason)

    def test_a_failure_says_why_in_words(self) -> None:
        record = {"apple_auto": {"state": "failed", "reason": "rate_limited"}}
        answer = renders.apple_answer(record)
        self.assertFalse(answer.answered)
        self.assertEqual(answer.reason, "Apple Auto failed (macOS asked Yellowmoon to wait)")

    def test_a_failure_of_kind_other_gives_apples_message(self) -> None:
        record = {
            "apple_auto": {"state": "failed", "reason": "other"},
            "auto": {"apple": {"state": "failed", "kind": "other", "message": "The model is busy"}},
        }
        self.assertEqual(renders.apple_answer(record).reason, "Apple Auto failed (an error: The model is busy)")

    def test_without_apple_auto_the_record_says_it_was_not_available(self) -> None:
        record = {"auto": {"apple": {"state": "unavailable", "reason": "model_not_ready"}}, "auto_source": "rule"}
        answer = renders.apple_answer(record)
        self.assertFalse(answer.answered)
        self.assertEqual(answer.reason, "Apple Auto was not available (Apple Intelligence is still downloading its model)")
        self.assertIn("off", renders.apple_answer({"auto": {"apple": {"state": "off"}}}).reason)

    def test_a_pending_answer_is_a_failure(self) -> None:
        answer = renders.apple_answer({"apple_auto": {"state": "pending"}})
        self.assertFalse(answer.answered)
        self.assertIn("had not arrived", answer.reason)

    def test_every_reason_of_docs_19_has_words(self) -> None:
        for reason in ("not_a_mac", "intel_mac", "built_without_support", "macos_too_old", "device_not_eligible",
                       "apple_intelligence_off", "model_not_ready", "unavailable", "no_image_input"):
            self.assertNotEqual(renders.unavailable_words(reason), reason)
        for kind in ("unavailable", "rate_limited", "timeout", "refused", "guardrail", "unsupported_language",
                     "context_too_large", "assets_unavailable", "concurrent_requests", "bad_answer", "other"):
            self.assertNotEqual(renders.failure_words(kind), kind)
        self.assertEqual(renders.unavailable_words(None), "no reason given")
        self.assertEqual(renders.unavailable_words("something_new"), "something_new")


class EnvironmentTests(unittest.TestCase):
    """What reaches `ym`: an empty home, the CPU, the data folder only when given, and only the test advisor's
    variable from the caller's YELLOWMOON_ variables."""

    def test_the_environment_keeps_only_the_test_advisor(self) -> None:
        base = {
            "PATH": "/bin",
            "YELLOWMOON_DATA": "/elsewhere",
            "YELLOWMOON_HOME": "/home/me/.yellowmoon",
            "YELLOWMOON_GPU": "on",
            "YELLOWMOON_LOOK": "x",
            "YELLOWMOON_TEST_APPLE_AUTO": "fake",
        }
        env = renders.ym_environment(None, Path("/tmp/home"), base)
        self.assertEqual(env["PATH"], "/bin")
        self.assertEqual(env["YELLOWMOON_HOME"], "/tmp/home")
        self.assertEqual(env["YELLOWMOON_GPU"], "off")
        self.assertEqual(env["YELLOWMOON_TEST_APPLE_AUTO"], "fake")
        self.assertNotIn("YELLOWMOON_DATA", env)
        self.assertNotIn("YELLOWMOON_LOOK", env)
        with_data = renders.ym_environment(Path("/data"), Path("/tmp/home"), base)
        self.assertEqual(with_data["YELLOWMOON_DATA"], "/data")

    def test_the_ym_inside_an_app_uses_the_apps_data(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="ab-bundle-"))
        self.addCleanup(shutil.rmtree, root, True)
        inside = root / "Yellowmoon.app" / "Contents" / "MacOS" / "ym"
        inside.parent.mkdir(parents=True)
        inside.write_text("")
        loose = root / "bin" / "ym"
        loose.parent.mkdir()
        loose.write_text("")
        self.assertTrue(ab.in_app_bundle(inside))
        self.assertIsNone(ab.default_data(inside))
        self.assertFalse(ab.in_app_bundle(loose))
        self.assertEqual(ab.default_data(loose), ab.REPO_DATA if ab.REPO_DATA.is_dir() else None)


class PrepareHarness(unittest.TestCase):
    """A small test set (six raw files in four strata) rendered through the fake `ym`."""

    STEMS = ("L1", "L2", "N1", "T1", "P1", "B1")

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="ab-prepare-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.ym = self.root / "bin" / "ym"
        self.ym.parent.mkdir()
        self.ym.write_text(FAKE_YM)
        os.chmod(self.ym, 0o755)
        self.data = self.root / "data"
        (self.data / "cameras").mkdir(parents=True)
        (self.data / "cameras" / "profile.json").write_text("{}")
        self.photos = self.root / "photos"
        self.scratch = self.root / "scratch"
        self.scratch.mkdir()
        self.log = self.root / "ym.log"
        self.out = self.root / "out"
        for relpath, stem in (
            ("landscape/L1.ARW", "L1"),
            ("landscape/L2.ARW", "L2"),
            ("night/N1.ARW", "N1"),
            ("T1.ARW", "T1"),
            ("portrait/P1.arw", "P1"),
            ("backlit/B1.ARW", "B1"),
        ):
            write_arw(self.photos / relpath, camera_bytes(stem))
        clean = {key: "" for key in os.environ if key.startswith("FAKE_YM_")}
        clean.update({"FAKE_YM_LOG": str(self.log)})
        env = mock.patch.dict(os.environ, clean)
        env.start()
        self.addCleanup(env.stop)
        for key in [key for key in os.environ if key.startswith("YELLOWMOON_")]:
            unset = mock.patch.dict(os.environ)
            unset.start()
            self.addCleanup(unset.stop)
            del os.environ[key]

    def options(self, **overrides: object) -> ab.PrepareOptions:
        base: Dict[str, object] = {
            "photos": self.photos,
            "out": self.out,
            "long_edge": 900,
            "seed": planlib.DEFAULT_SEED,
            "ym": self.ym,
            "data": self.data,
            "jobs": 3,
            "scratch_root": self.scratch,
        }
        base.update(overrides)
        return ab.PrepareOptions(**base)  # type: ignore[arg-type]

    def prepare(self, **overrides: object) -> ab.PrepareResult:
        messages: List[str] = []
        return ab.prepare(self.options(**overrides), log=messages.append)

    def renders_logged(self) -> List[Dict[str, object]]:
        return [c for c in read_log(self.log) if c["cmd"] == "render"]

    def raw_hashes(self) -> Dict[str, str]:
        return {str(path): sha256_of(path) for path in sorted(self.photos.rglob("*")) if path.is_file()}


class PrepareTests(PrepareHarness):
    """`prepare` through the fake `ym`: the flags, the homes, the answers, the photos left out and those unchanged."""

    def test_each_photo_is_rendered_with_apple_auto_on_and_off_at_the_defaults(self) -> None:
        self.prepare()
        calls = self.renders_logged()
        modes_by_stem: Dict[str, List[str]] = {}
        for call in calls:
            args = call["args"]
            for flag in ("--cpu", "--quiet", "--json", "--overwrite"):
                self.assertIn(flag, args)
            self.assertEqual(args[args.index("--format") + 1], "jpeg")
            self.assertEqual(args[args.index("--quality") + 1], "92")
            self.assertEqual(args[args.index("--long-edge") + 1], "900")
            for setting in ("--look", "--exposure", "--contrast", "--saturation", "--wb", "--settings", "--light"):
                self.assertNotIn(setting, args, "the defaults: every setting at Auto, the default look")
            self.assertTrue(call["home_empty"], "each render gets an empty YELLOWMOON_HOME")
            self.assertEqual(call["data"], str(self.data))
            modes_by_stem.setdefault(str(call["stem"]), []).append(str(call["mode"]))
        self.assertEqual(sorted(modes_by_stem), sorted(self.STEMS))
        for stem, modes in modes_by_stem.items():
            self.assertEqual(sorted(modes), ["off", "on"], stem)

    def test_doctor_is_asked_first_with_an_empty_home(self) -> None:
        self.prepare()
        calls = read_log(self.log)
        self.assertEqual(calls[0]["cmd"], "doctor")
        self.assertEqual(calls[0]["args"], ["doctor", "--json"])
        self.assertTrue(calls[0]["home_empty"])

    def test_the_plan_records_the_model_macos_prompt_and_table(self) -> None:
        result = self.prepare()
        apple = result.plan["provenance"]["apple_auto"]  # type: ignore[index]
        self.assertEqual(
            apple,
            {"available": True, "reason": None, "model": MODEL, "os": MACOS, "prompt": 3, "table": 2,
             "instructions": "default", "instructions_sha256": DEFAULT_SHA},
        )
        self.assertEqual(result.plan["provenance"]["ym"], "ym 9.9.9-test (pipeline 0.0.0, fake)")  # type: ignore[index]
        on_disk = json.loads((self.out / "plan.json").read_text(encoding="utf-8"))
        self.assertEqual(on_disk["provenance"]["apple_auto"]["model"], MODEL)

    def test_an_unavailable_apple_auto_stops_before_rendering(self) -> None:
        with mock.patch.dict(os.environ, {"FAKE_YM_UNAVAILABLE": "apple_intelligence_off"}):
            with self.assertRaises(ab.PrepareFailed) as caught:
                self.prepare()
        self.assertEqual(
            str(caught.exception),
            "Apple Auto is not available on this Mac (Apple Intelligence is off in System Settings). "
            "The test needs macOS 27 with Apple Intelligence on.",
        )
        self.assertEqual(self.renders_logged(), [])
        self.assertFalse((self.out / "plan.json").exists())

    def test_allow_unavailable_goes_on(self) -> None:
        with mock.patch.dict(os.environ, {"FAKE_YM_UNAVAILABLE": "not_a_mac"}):
            result = self.prepare(allow_unavailable=True)
        self.assertEqual(result.plan["provenance"]["apple_auto"]["reason"], "not_a_mac")  # type: ignore[index]
        self.assertTrue(self.renders_logged())

    def test_a_failed_answer_leaves_the_photo_out_with_its_reason(self) -> None:
        with mock.patch.dict(os.environ, {"FAKE_YM_APPLE_FAIL": "N1", "FAKE_YM_APPLE_OTHER": "B1"}):
            result = self.prepare()
        self.assertEqual(
            [(item["stem"], item["reason"]) for item in result.failed],
            [("B1", "Apple Auto failed (an error: Something odd happened)"),
             ("N1", "Apple Auto failed (no answer within 30 seconds)")],
        )
        stems_in_test = {p["stem"] for p in result.plan["photos"]}  # type: ignore[union-attr]
        self.assertNotIn("N1", stems_in_test)
        self.assertNotIn("N1", {t["stem"] for t in result.plan["trials"]})  # type: ignore[union-attr]
        rule_renders = {c["stem"] for c in self.renders_logged() if c["mode"] == "off"}
        self.assertNotIn("N1", rule_renders, "a photo whose answer failed needs no render of the rule")
        self.assertEqual(result.plan["counts"]["failed"], 2)  # type: ignore[index]
        left_out = {item["stem"]: item for item in result.plan["left_out"]}  # type: ignore[union-attr]
        self.assertEqual(left_out["N1"]["why"], "failed")

    def test_apple_auto_gone_for_the_run_leaves_the_photo_out(self) -> None:
        with mock.patch.dict(os.environ, {"FAKE_YM_APPLE_GONE": "L2"}):
            result = self.prepare()
        self.assertEqual(
            [(item["stem"], item["reason"]) for item in result.failed],
            [("L2", "Apple Auto was not available (Apple Intelligence is off in System Settings)")],
        )

    def test_an_answer_that_keeps_everything_gives_no_trial_and_no_rule_render(self) -> None:
        steps = {"T1": [0, 0, 0, 0]}
        with mock.patch.dict(os.environ, {"FAKE_YM_STEPS": json.dumps(steps)}):
            result = self.prepare()
        self.assertEqual(
            [(item["stem"], item["reason"]) for item in result.unchanged], [("T1", "every step it set is keep or none")]
        )
        self.assertNotIn("T1", {t["stem"] for t in result.plan["trials"]})  # type: ignore[union-attr]
        self.assertNotIn("T1", {c["stem"] for c in self.renders_logged() if c["mode"] == "off"})
        self.assertEqual(result.plan["counts"]["unchanged"], 1)  # type: ignore[index]

    def test_an_answer_that_changes_no_pixel_gives_no_trial(self) -> None:
        with mock.patch.dict(os.environ, {"FAKE_YM_SAME_PIXELS": "P1"}):
            result = self.prepare()
        self.assertEqual(
            [(item["stem"], item["reason"]) for item in result.unchanged], [("P1", "the two pictures are the same")]
        )
        self.assertIn("P1", {c["stem"] for c in self.renders_logged() if c["mode"] == "off"})
        self.assertNotIn("P1", {t["stem"] for t in result.plan["trials"]})  # type: ignore[union-attr]

    def test_a_render_that_fails_leaves_the_photo_out(self) -> None:
        with mock.patch.dict(os.environ, {"FAKE_YM_FAIL": "L1"}):
            result = self.prepare()
        self.assertEqual([item["stem"] for item in result.errors], ["L1"])
        self.assertIn("fake failure for L1", str(result.errors[0]["reason"]))
        self.assertIn("with Apple Auto on", str(result.errors[0]["reason"]))
        self.assertEqual(result.plan["counts"]["not_rendered"], 1)  # type: ignore[index]

    def test_the_plan_keeps_the_steps_scene_and_auto_values_of_each_photo(self) -> None:
        steps = {"L1": [-1, 2, 0, 1]}
        with mock.patch.dict(os.environ, {"FAKE_YM_STEPS": json.dumps(steps)}):
            result = self.prepare()
        entry = next(p for p in result.plan["photos"] if p["stem"] == "L1")  # type: ignore[union-attr]
        self.assertEqual(entry["steps"], planlib.steps_dict(S(-1, 2, 0, 1)))
        self.assertIsNone(entry["steps"]["highlights"], "a field the model left empty is null")
        self.assertEqual(entry["scene"], "Scene of L1")
        self.assertEqual(entry["plan"], "Plan for L1")
        self.assertAlmostEqual(entry["auto_values"]["apple"]["exposure_ev"], 0.2)
        self.assertAlmostEqual(entry["auto_values"]["rule"]["exposure_ev"], 0.5)
        self.assertEqual(entry["auto_values"]["apple"]["saturation"], 15.0)
        for trial in result.plan["trials"]:  # type: ignore[union-attr]
            if trial["stem"] == "L1":
                self.assertEqual(trial["steps"], entry["steps"])

    def test_a_colour_alone_is_a_change_and_a_colour_without_a_strength_is_none(self) -> None:
        steps = {
            "L1": {"shadow_color": "teal", "shadow_color_strength": 2},
            "T1": {"exposure": 0, "shadow_color": "blue", "highlight_color": "none", "highlight_color_strength": 3},
        }
        with mock.patch.dict(os.environ, {"FAKE_YM_STEPS": json.dumps(steps)}):
            result = self.prepare()
        self.assertIn("L1", {t["stem"] for t in result.plan["trials"]})  # type: ignore[union-attr]
        self.assertEqual(
            [(item["stem"], item["reason"]) for item in result.unchanged], [("T1", "every step it set is keep or none")]
        )

    def test_instructions_from_a_file_reach_the_apple_renders_and_the_plan(self) -> None:
        text = "Grade it like a quiet documentary."
        path = self.root / "instructions.txt"
        path.write_text(text + "\n", encoding="utf-8")
        result = self.prepare(instructions=path)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        apple = result.plan["provenance"]["apple_auto"]  # type: ignore[index]
        self.assertEqual((apple["instructions"], apple["instructions_sha256"]), ("edited", digest))
        for call in self.renders_logged():
            flag = "--apple-auto-instructions" in call["args"]  # type: ignore[operator]
            self.assertEqual(flag, call["mode"] == "on", call)
            if flag:
                args = call["args"]
                self.assertEqual(args[args.index("--apple-auto-instructions") + 1], str(path))  # type: ignore
        # The default text in a file is the default; the plan id changes with the instructions.
        default = self.root / "default.txt"
        default.write_text("  \n", encoding="utf-8")
        again = self.prepare(instructions=default)
        apple = again.plan["provenance"]["apple_auto"]  # type: ignore[index]
        self.assertEqual((apple["instructions"], apple["instructions_sha256"]), ("default", DEFAULT_SHA))
        self.assertNotEqual(again.plan["plan_id"], result.plan["plan_id"])

    def test_resume_asks_again_where_the_instructions_changed(self) -> None:
        self.prepare()
        path = self.root / "instructions.txt"
        path.write_text("Moody and dark.", encoding="utf-8")
        self.log.unlink()
        self.prepare(resume=True, instructions=path)
        modes = sorted(str(c["mode"]) for c in self.renders_logged())
        self.assertEqual(modes, ["on"] * len(self.STEMS), "every Apple render is made again, no rule render")

    def test_instructions_too_long_are_a_usage_error(self) -> None:
        path = self.root / "long.txt"
        path.write_text("x" * 2001 + "\n", encoding="utf-8")
        with self.assertRaises(ab.UsageError) as caught:
            self.prepare(instructions=path)
        self.assertIn("2001 characters long; at most 2000 fit", str(caught.exception))
        path.write_text("x" * 2000 + "\r\n", encoding="utf-8")
        self.prepare(instructions=path)

    def test_the_originals_are_never_written(self) -> None:
        before = self.raw_hashes()
        self.prepare(with_camera=True)
        self.assertEqual(self.raw_hashes(), before)

    def test_without_the_camera_there_is_one_trial_per_photo_and_no_camera_picture(self) -> None:
        result = self.prepare()
        self.assertEqual(len(result.plan["trials"]), len(self.STEMS))  # type: ignore[arg-type]
        self.assertFalse(any(c["cmd"] == "info" for c in read_log(self.log)))
        code = result.plan["variants"]["camera"]["code"]  # type: ignore[index]
        self.assertFalse((self.out / "pictures" / code).exists())

    def test_with_the_camera_its_pictures_are_byte_copies_of_the_embedded_jpeg(self) -> None:
        result = self.prepare(with_camera=True)
        self.assertEqual(len(result.plan["trials"]), 2 * len(self.STEMS))  # type: ignore[arg-type]
        for entry in result.plan["photos"]:  # type: ignore[index]
            camera = self.out / entry["pictures"]["camera"]
            self.assertEqual(camera.read_bytes(), camera_bytes(entry["stem"]))
        for call in read_log(self.log):
            if call["cmd"] == "info":
                self.assertIn("--apple-auto", call["args"])
                self.assertEqual(call["args"][call["args"].index("--apple-auto") + 1], "off")

    def test_a_photo_without_an_embedded_jpeg_gives_only_the_trial_against_the_rule(self) -> None:
        write_arw(self.photos / "landscape" / "L2.ARW", None)
        result = self.prepare(with_camera=True)
        self.assertEqual([item["stem"] for item in result.no_camera], ["L2"])
        self.assertIn("embedded JPEG", str(result.no_camera[0]["reason"]))
        mine = [t["comparison"] for t in result.plan["trials"] if t["stem"] == "L2"]  # type: ignore[union-attr]
        self.assertEqual(mine, ["rule"])
        self.assertIn("L2", {p["stem"] for p in result.plan["photos"]})  # type: ignore[union-attr]

    def test_a_raw_shot_upright_gets_one_turn_on_its_camera_picture_only(self) -> None:
        write_arw(self.photos / "portrait" / "P1.arw", camera_bytes("P1"), orientation=6)
        result = self.prepare(with_camera=True)
        turns = {entry["stem"]: entry["turns"] for entry in result.plan["photos"]}  # type: ignore[index]
        self.assertEqual(turns["P1"]["camera"], 1)
        self.assertEqual((turns["P1"]["apple"], turns["P1"]["rule"]), (0, 0))
        for stem in ("L1", "L2", "N1", "T1", "B1"):
            self.assertEqual(turns[stem]["camera"], 0, stem)
        payload = json.loads((self.out / "plan.js").read_text(encoding="utf-8").split("=", 1)[1].strip().rstrip(";"))
        shown = {entry["id"]: entry["pictures"] for entry in payload["trials"]}
        checked = 0
        for trial in result.plan["trials"]:  # type: ignore[union-attr]
            if trial["stem"] != "P1":
                continue
            for side in ("left", "right"):
                expected = 1 if trial[side] == "camera" else 0
                self.assertEqual(shown[trial["id"]][side]["turns"], expected, f"trial {trial['id']} {side}")
                checked += 1
        self.assertEqual(checked, 2 * 2, "two trials of P1, two sides each")

    def test_apple_renders_wait_for_the_model_one_at_a_time(self) -> None:
        with mock.patch.dict(os.environ, {"FAKE_YM_SLEEP": "0.15"}):
            self.prepare(jobs=4, apple_jobs=1)
        calls = read_log(self.log)
        starts = {c["stem"]: c["start"] for c in calls if c["cmd"] == "render" and c["mode"] == "on"}
        ends = {c["stem"]: c["end"] for c in calls if c["cmd"] == "rendered" and c["mode"] == "on"}
        spans = sorted((starts[stem], ends[stem]) for stem in starts)
        self.assertEqual(len(spans), len(self.STEMS))
        for (_, end), (start, _) in zip(spans, spans[1:]):
            self.assertLessEqual(end, start, "two renders with Apple Auto on overlapped")

    def test_resume_keeps_the_pictures_and_answers_already_made(self) -> None:
        first = self.prepare()
        pictures = sorted(p for p in (self.out / "pictures").rglob("*.jpg"))
        self.assertTrue(pictures)
        missing = pictures[0]
        missing.unlink()
        kept = {p: p.read_bytes() for p in pictures[1:]}
        self.log.unlink()
        second = self.prepare(resume=True)
        self.assertEqual(second.plan["plan_id"], first.plan["plan_id"])
        self.assertTrue(missing.is_file(), "the missing picture is made again")
        for path, blob in kept.items():
            self.assertEqual(path.read_bytes(), blob)
        self.assertEqual(len(self.renders_logged()), 1, "only the missing picture is rendered")
        self.log.unlink()
        self.prepare()
        self.assertEqual(len(self.renders_logged()), 2 * len(self.STEMS), "without --resume every picture is made")

    def test_resume_asks_again_where_the_answer_failed(self) -> None:
        with mock.patch.dict(os.environ, {"FAKE_YM_APPLE_FAIL": "N1"}):
            first = self.prepare()
        self.assertEqual([item["stem"] for item in first.failed], ["N1"])
        self.log.unlink()
        second = self.prepare(resume=True)
        self.assertEqual(second.failed, [])
        self.assertIn("N1", {p["stem"] for p in second.plan["photos"]})  # type: ignore[union-attr]
        self.assertEqual(
            sorted((c["stem"], c["mode"]) for c in self.renders_logged()), [("N1", "off"), ("N1", "on")]
        )

    def test_resume_remakes_a_file_that_is_not_a_jpeg_or_has_no_record(self) -> None:
        self.prepare()
        pictures = sorted(p for p in (self.out / "pictures").rglob("*.jpg"))
        pictures[0].write_bytes(b"not a jpeg")
        records = sorted((self.out / renders.RECORDS_DIR).rglob("*.json"))
        self.assertEqual(len(records), 2 * len(self.STEMS))
        records[0].unlink()
        self.log.unlink()
        self.prepare(resume=True)
        self.assertEqual(pictures[0].read_bytes()[:2], b"\xff\xd8")
        self.assertEqual(len(self.renders_logged()), 2)

    def test_the_number_of_jobs_does_not_change_the_plan(self) -> None:
        one = self.prepare(jobs=1, out=self.root / "out-one", with_camera=True)
        four = self.prepare(jobs=4, out=self.root / "out-four", with_camera=True)
        self.assertEqual(one.plan["plan_id"], four.plan["plan_id"])
        self.assertEqual(one.plan["trials"], four.plan["trials"])
        for entry in one.plan["photos"]:  # type: ignore[index]
            for rel in entry["pictures"].values():
                self.assertEqual(
                    (self.root / "out-one" / rel).read_bytes(),
                    (self.root / "out-four" / rel).read_bytes(),
                )

    def test_the_folder_gets_the_page_the_plan_the_pictures_and_the_records(self) -> None:
        result = self.prepare()
        self.assertTrue((self.out / "index.html").is_file())
        self.assertTrue((self.out / "plan.js").is_file())
        plan_doc = json.loads((self.out / "plan.json").read_text(encoding="utf-8"))
        self.assertEqual(plan_doc["plan_id"], result.plan["plan_id"])
        for trial in plan_doc["trials"]:
            for side in ("left", "right"):
                self.assertTrue((self.out / trial["pictures"][side]).is_file())
        record = json.loads((self.out / "records" / "apple" / "0001_B1.json").read_text(encoding="utf-8"))
        self.assertEqual(record["apple_auto"]["state"], "answered")

    def test_the_files_the_page_loads_have_no_variant_name_in_their_paths(self) -> None:
        self.prepare(with_camera=True)
        served = [path.relative_to(self.out).as_posix() for path in self.out.rglob("*") if path.is_file()]
        # The key: plan.json and the records name the variants; the page never loads them.
        served = [rel for rel in served if rel != "plan.json" and not rel.startswith("records/")]
        self.assertIn("index.html", served)
        self.assertIn("plan.js", served)
        self.assertEqual(len([rel for rel in served if rel.startswith("pictures/")]), len(planlib.VARIANTS) * 6)
        for rel in served:
            for word in planlib.VARIANTS:
                self.assertNotIn(word, rel.lower(), rel)

    def test_the_test_advisor_reaches_ym_and_nothing_else_of_the_callers(self) -> None:
        extra = {"YELLOWMOON_TEST_APPLE_AUTO": "fake", "YELLOWMOON_LOOK": "should-not-pass"}
        with mock.patch.dict(os.environ, extra):
            self.prepare()
        for call in read_log(self.log):
            if call["cmd"] in ("render", "doctor"):
                self.assertEqual(
                    call["seen"],
                    ["YELLOWMOON_DATA", "YELLOWMOON_GPU", "YELLOWMOON_HOME", "YELLOWMOON_TEST_APPLE_AUTO"],
                )

    def test_without_a_data_folder_ym_finds_its_own(self) -> None:
        self.prepare(data=None)
        for call in read_log(self.log):
            if call["cmd"] in ("render", "doctor"):
                self.assertIsNone(call["data"])
                self.assertNotIn("YELLOWMOON_DATA", call["seen"])

    def test_an_interrupt_removes_the_temporary_folder(self) -> None:
        with mock.patch.object(ab, "make_pictures", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.prepare()
        self.assertEqual(list(self.scratch.iterdir()), [])

    def test_the_temporary_folder_is_removed_after_a_run(self) -> None:
        self.prepare()
        self.assertEqual(list(self.scratch.iterdir()), [])

    def test_no_photo_in_the_test_is_an_error_that_gives_the_counts(self) -> None:
        steps = {stem: [0, 0, 0, 0] for stem in ("L1", "L2", "N1")}
        env = {"FAKE_YM_STEPS": json.dumps(steps), "FAKE_YM_APPLE_FAIL": "T1,P1", "FAKE_YM_FAIL": "B1"}
        with mock.patch.dict(os.environ, env):
            with self.assertRaises(ab.PrepareFailed) as caught:
                self.prepare()
        self.assertIn("Apple Auto left 3 of 6 photos as they were, failed on 2, and 1 could not be rendered",
                      str(caught.exception))


class CommandLineTests(PrepareHarness):
    """The command line: the summary, the usage errors and the exit codes."""

    def run_main(self, argv: Sequence[str]) -> Tuple[int, str, str]:
        out, err = StringIO(), StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = ab.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def prepare_argv(self, *extra: str) -> List[str]:
        return [
            "prepare", str(self.photos), "--out", str(self.out), "--ym", str(self.ym), "--data", str(self.data),
            "--jobs", "2", "--long-edge", "900", *extra,
        ]

    def test_prepare_prints_the_summary_with_both_counts(self) -> None:
        env = {"FAKE_YM_STEPS": json.dumps({"T1": [0, 0, 0, 0]}), "FAKE_YM_APPLE_FAIL": "N1"}
        with mock.patch.dict(os.environ, env):
            code, text, _ = self.run_main(self.prepare_argv())
        self.assertEqual(code, 0)
        self.assertIn(
            f"Apple Auto: {MODEL} · macOS {MACOS} · prompt 3 · table 2 · instructions default ({DEFAULT_SHA})", text
        )
        self.assertIn("photos: 6 found, 4 in the test", text)
        self.assertIn("Apple Auto left 1 of 6 photos as they were: they give no trial.", text)
        self.assertIn("Apple Auto failed on 1 of 6 photos: they are left out.", text)
        self.assertIn("night/N1: Apple Auto failed (no answer within 30 seconds)", text)
        self.assertIn("trials: 4 (Apple Auto vs the rule 4)", text)
        self.assertIn(str(self.out.resolve() / "index.html"), text)
        self.assertNotIn("test advisor", text)

    def test_the_summary_says_so_when_the_test_advisor_answered(self) -> None:
        fake = self.ym.read_text().replace('MODEL = "AFM 3 Core"', 'MODEL = "Test advisor"')
        self.ym.write_text(fake)
        code, text, _ = self.run_main(self.prepare_argv("--with-camera"))
        self.assertEqual(code, 0)
        self.assertIn("note: the answers came from the test advisor", text)
        self.assertIn("trials: 12 (Apple Auto vs the rule 6, Apple Auto vs the camera's JPEG 6)", text)

    def test_unavailable_apple_auto_exits_one_with_the_reason_in_words(self) -> None:
        with mock.patch.dict(os.environ, {"FAKE_YM_UNAVAILABLE": "macos_too_old"}):
            code, _, err = self.run_main(self.prepare_argv())
        self.assertEqual(code, 1)
        self.assertIn(
            "Apple Auto is not available on this Mac (it needs macOS 27 or later). "
            "The test needs macOS 27 with Apple Intelligence on.",
            err,
        )
        with mock.patch.dict(os.environ, {"FAKE_YM_UNAVAILABLE": "macos_too_old"}):
            code, _, _ = self.run_main(self.prepare_argv("--allow-unavailable"))
        self.assertEqual(code, 0)

    def test_a_ym_without_apple_auto_is_a_usage_error(self) -> None:
        old = self.root / "bin" / "old-ym"
        old.write_text("#!/bin/sh\nif [ \"$1\" = --version ]; then echo 'ym 0.1.9'; exit 0; fi\necho '{}'\n")
        os.chmod(old, 0o755)
        code, _, err = self.run_main(
            ["prepare", str(self.photos), "--out", str(self.out), "--ym", str(old), "--data", str(self.data)]
        )
        self.assertEqual(code, 2)
        self.assertIn("older than Apple Auto", err)

    def test_usage_errors_exit_two(self) -> None:
        empty = self.root / "no-raw-files"
        empty.mkdir()
        (empty / "notes.txt").write_text("nothing here", encoding="utf-8")
        cases = [
            ["prepare", str(self.root / "missing")],
            ["prepare", str(self.photos), "--ym", str(self.root / "no-such-ym")],
            ["prepare", str(self.photos), "--ym", str(self.ym), "--data", str(self.root / "empty")],
            ["prepare", str(empty), "--ym", str(self.ym), "--data", str(self.data)],
            ["prepare", str(self.photos), "--ym", str(self.ym), "--apple-jobs", "0"],
            ["prepare", str(self.photos), "--ym", str(self.ym), "--long-edge", "10"],
            ["score", str(self.root / "missing.json")],
        ]
        for argv in cases:
            code, _, _ = self.run_main(argv)
            self.assertEqual(code, 2, argv)

    def test_an_unreadable_folder_exits_one_with_a_message(self) -> None:
        error = PermissionError(13, "Permission denied", str(self.photos))
        with mock.patch.object(ab, "run_prepare", side_effect=error):
            code, _, err = self.run_main(["prepare", str(self.photos)])
        self.assertEqual(code, 1)
        self.assertIn("Permission denied", err)
        self.assertNotIn("Traceback", err)

    def test_sigterm_is_handled_like_ctrl_c(self) -> None:
        def terminate(*args: object, **kwargs: object) -> int:
            os.kill(os.getpid(), signal.SIGTERM)
            return 0

        with mock.patch.object(ab, "run_prepare", side_effect=terminate):
            code, _, err = self.run_main(["prepare", str(self.photos)])
        self.assertEqual(code, 130)
        self.assertIn("interrupted", err)

    def test_an_unknown_command_is_an_argument_error(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                ab.main(["bogus"])
        self.assertEqual(caught.exception.code, 2)

    def test_prepare_then_score_end_to_end(self) -> None:
        code, _, _ = self.run_main(self.prepare_argv())
        self.assertEqual(code, 0)
        plan = json.loads((self.out / "plan.json").read_text(encoding="utf-8"))
        answers = []
        for trial in plan["trials"]:
            apple_side = "left" if trial["left"] == "apple" else "right"
            answers.append({"trial": trial["id"], "answer": apple_side, "flags": {"left": False, "right": False}})
        results = {"format": page.RESULTS_FORMAT, "version": 1, "plan_id": plan["plan_id"], "answers": answers}
        downloads = self.root / "downloads"
        downloads.mkdir()
        (downloads / "results.json").write_text(json.dumps(results), encoding="utf-8")
        code, text, _ = self.run_main(
            ["score", str(downloads / "results.json"), "--plan", str(self.out / "plan.json")]
        )
        self.assertEqual(code, 0)
        self.assertIn("Gate, Apple Auto against the rule (provisional thresholds, docs/19 §13): PASS", text)
        written = json.loads((downloads / "score.json").read_text(encoding="utf-8"))
        self.assertEqual(written["provenance"]["apple_auto"]["model"], MODEL)
        self.assertEqual(written["photos"]["found"], 6)


class PageTests(unittest.TestCase):
    """The blind page: no network, no variant names, and it reads the plan from plan.js."""

    def test_the_page_makes_no_network_requests_and_loads_only_plan_js(self) -> None:
        text = page.PAGE
        self.assertNotIn("http:", text)
        self.assertNotIn("https:", text)
        self.assertNotIn("<link", text)
        self.assertIn('<script src="plan.js"></script>', text)
        self.assertNotIn("@import", text)

    def test_the_page_names_no_variant(self) -> None:
        lowered = page.PAGE.lower()
        for name in FORBIDDEN_NAMES:
            self.assertNotIn(name, lowered, name)
        visible = visible_markup(page.PAGE).lower()
        for pattern in FORBIDDEN_VISIBLE_WORDS:
            self.assertIsNone(re.search(pattern, visible), pattern)

    def test_plan_js_names_no_variant_and_carries_the_trials(self) -> None:
        photos = [planlib.PlanPhoto(number=1, stem="DSC1", stratum="other", has_camera=True, steps=S(1, 0, 0, 0),
                                    scene="A red apple on a table")]
        plan = planlib.build_plan(photos, 1, {"ym": "ym", "apple_auto": {"model": MODEL}})
        text = page.plan_js(plan).lower()
        self.assertIn("window.ym_ab_plan = {", text)
        for name in FORBIDDEN_NAMES:
            self.assertNotIn(name, text, name)
        self.assertIsNone(re.search(r"\brule\b|\bauto\b|\bwhite\b|\bscene\b|\bsteps\b", text))
        self.assertIn(plan["plan_id"], text)
        self.assertNotIn("http", text)

    def test_the_page_script_keys_results_by_plan_and_exports_results_json(self) -> None:
        self.assertIn('"ym-ab:" + plan.plan_id', page.PAGE)
        self.assertIn('link.download = "results.json"', page.PAGE)
        self.assertIn('format: "yellowmoon-ab-results"', page.PAGE)


class ScoreTests(unittest.TestCase):
    """The Wilson bound, ties, the strata, the flags, the gate's rules and the diagnostics, on hand-made answers."""

    def test_wilson_lower_bound_matches_known_values(self) -> None:
        cases = [
            (60, 100, 0.5020),
            (50, 100, 0.4038),
            (9, 10, 0.5959),
            (10, 10, 0.7225),
            (0, 10, 0.0),
            (2, 3, 0.2077),
            (145, 200, 0.6593),
        ]
        for successes, trials, expected in cases:
            got = scorelib.wilson_lower(successes, trials)
            self.assertAlmostEqual(got, expected, delta=0.0005, msg=str((successes, trials)))

    def test_wilson_lower_bound_agrees_with_the_quadratic_form(self) -> None:
        z = scorelib.Z95
        for successes, trials in ((6, 10), (55, 95), (1, 1), (3, 7), (500, 1000), (12, 20)):
            p = successes / trials
            a = trials + z * z
            b = -(2 * trials * p + z * z)
            c = trials * p * p
            root = (-b - math.sqrt(b * b - 4 * a * c)) / (2 * a)
            self.assertAlmostEqual(scorelib.wilson_lower(successes, trials), root, places=9)

    def test_wilson_rejects_impossible_counts(self) -> None:
        self.assertIsNone(scorelib.wilson_lower(0, 0))
        with self.assertRaises(ValueError):
            scorelib.wilson_lower(4, 3)

    def test_ties_count_as_half_and_the_bound_leaves_them_out(self) -> None:
        tally = scorelib.Tally(wins=2, losses=1, ties=1)
        self.assertAlmostEqual(tally.share, 0.625)
        self.assertAlmostEqual(tally.lower, scorelib.wilson_lower(2, 3))

    def test_the_gate_passes_on_a_clear_result_and_says_its_thresholds_are_provisional(self) -> None:
        plan, results = make_case([("rule", "portrait", 70, 20, 10, 1, 5)])
        report = score_answers(plan, results)
        self.assertEqual(sorted(report["comparisons"]), ["rule"])  # type: ignore[arg-type]
        self.assertEqual(report["comparisons"]["rule"]["verdict"], "PASS")  # type: ignore[index]
        self.assertEqual(report["gate"]["verdict"], "PASS")  # type: ignore[index]
        self.assertTrue(report["gate"]["provisional"])  # type: ignore[index]
        self.assertEqual(report["gate"]["name"], "docs/19 section 13")  # type: ignore[index]
        text = scorelib.format_report(report)
        self.assertIn("Gate, Apple Auto against the rule (provisional thresholds, docs/19 §13): PASS", text)
        self.assertIn("Thresholds (provisional thresholds, docs/19 §13)", text)
        self.assertNotIn("camera", text.lower())

    def test_the_camera_comparison_is_reported_but_not_gated(self) -> None:
        plan, results = make_case(
            [("rule", "portrait", 70, 20, 10, 0, 0), ("camera", "portrait", 10, 80, 10, 9, 0)]
        )
        report = score_answers(plan, results)
        camera = report["comparisons"]["camera"]  # type: ignore[index]
        self.assertEqual(camera["verdict"], "FAIL")
        self.assertFalse(camera["gated"])
        self.assertEqual(report["gate"]["gated"], ["rule"])  # type: ignore[index]
        self.assertEqual(report["gate"]["verdict"], "PASS")  # type: ignore[index]
        text = scorelib.format_report(report)
        self.assertIn("(FAIL)", text)
        self.assertIn("In brackets: reported, not part of the gate.", text)
        self.assertNotIn("Why the gate is not passed", text)

    def test_a_pooled_share_below_sixty_percent_fails(self) -> None:
        plan, results = make_case([("rule", "portrait", 55, 40, 5, 0, 0)])
        report = score_answers(plan, results)
        rule = report["comparisons"]["rule"]  # type: ignore[index]
        self.assertEqual(rule["verdict"], "FAIL")
        self.assertIn("pooled win share", " ".join(rule["reasons"]))
        self.assertEqual(report["gate"]["verdict"], "FAIL")  # type: ignore[index]
        self.assertIn("Why the gate is not passed", scorelib.format_report(report))

    def test_a_stratum_below_forty_five_percent_fails(self) -> None:
        plan, results = make_case(
            [("rule", "portrait", 80, 0, 0, 0, 0), ("rule", "night", 0, 20, 0, 0, 0)]
        )
        report = score_answers(plan, results)
        rule = report["comparisons"]["rule"]  # type: ignore[index]
        self.assertEqual(rule["verdict"], "FAIL")
        self.assertIn("stratum night", " ".join(rule["reasons"]))
        self.assertTrue(rule["strata"]["night"]["below_gate"])  # type: ignore[index]

    def test_apple_flagged_more_often_than_the_rule_fails(self) -> None:
        plan, results = make_case([("rule", "portrait", 70, 20, 10, 10, 5)])
        report = score_answers(plan, results)
        rule = report["comparisons"]["rule"]  # type: ignore[index]
        self.assertEqual(rule["verdict"], "FAIL")
        self.assertIn("needs a fix", " ".join(rule["reasons"]))

    def test_a_lower_bound_under_fifty_percent_fails_even_at_sixty_percent(self) -> None:
        plan, results = make_case([("rule", "portrait", 6, 4, 0, 0, 0)])
        report = score_answers(plan, results)
        rule = report["comparisons"]["rule"]  # type: ignore[index]
        self.assertAlmostEqual(rule["share"], 0.6)
        self.assertEqual(rule["verdict"], "FAIL")
        self.assertIn("Wilson", " ".join(rule["reasons"]))

    def test_the_boundaries_pass(self) -> None:
        plan, results = make_case([("rule", "a", 100, 0, 0, 0, 0), ("rule", "b", 45, 55, 0, 0, 0)])
        report = score_answers(plan, results)
        self.assertAlmostEqual(report["comparisons"]["rule"]["share"], 0.725)  # type: ignore[index]
        self.assertEqual(report["comparisons"]["rule"]["verdict"], "PASS")  # type: ignore[index]
        self.assertEqual(report["gate"]["verdict"], "PASS")  # type: ignore[index]

    def test_no_answers_yet_leave_the_gate_undecided(self) -> None:
        plan, results = make_case([("rule", "a", 70, 20, 10, 0, 0), ("camera", "a", 70, 20, 10, 0, 0)])
        report = score_answers(plan, drop_answers(plan, results, "rule"))
        self.assertEqual(report["comparisons"]["rule"]["verdict"], "NOT RUN")  # type: ignore[index]
        self.assertEqual(report["gate"]["verdict"], "NOT DECIDED")  # type: ignore[index]
        self.assertFalse(report["complete"])
        text = scorelib.format_report(report)
        self.assertIn("NOT DECIDED (not every trial is answered)", text)

    def test_the_diagnostics_split_the_wins_by_what_apple_changed(self) -> None:
        plan, results = make_case(
            [
                ("rule", "a", 3, 1, 0, 0, 0, (1, 0, 0, 0)),  # only exposure, brighter
                ("rule", "a", 0, 2, 0, 0, 0, (-2, 0, 1, 0)),  # darker and more contrast
                ("rule", "b", 1, 0, 1, 0, 0, (0, 1, 0, -1)),  # warmer and less saturation
            ]
        )
        diag = score_answers(plan, results)["diagnostics"]
        by_direction = {g["label"]: g for g in diag["by_direction"]}  # type: ignore[index]
        self.assertEqual(by_direction["brighter"]["trials"], 4)
        self.assertAlmostEqual(by_direction["brighter"]["share"], 0.75)
        self.assertEqual(by_direction["darker"]["trials"], 2)
        self.assertAlmostEqual(by_direction["darker"]["share"], 0.0)
        self.assertEqual(by_direction["more contrast"]["trials"], 2)
        self.assertEqual(by_direction["warmer"]["trials"], 2)
        self.assertAlmostEqual(by_direction["warmer"]["share"], 0.75)
        self.assertEqual(by_direction["less saturation"]["trials"], 2)
        self.assertEqual(by_direction["cooler"]["trials"], 0)
        single = {g["label"]: g for g in diag["by_single_change"]}  # type: ignore[index]
        self.assertEqual(single["only exposure"]["trials"], 4)
        self.assertEqual(single["two or more changes"]["trials"], 4)
        self.assertEqual(single["only white"]["trials"], 0)
        by_step = {g["label"]: g for g in diag["by_step"]}  # type: ignore[index]
        self.assertEqual(by_step["exposure +1 (slightly brighter)"]["trials"], 4)
        self.assertEqual(by_step["exposure -2 (darker)"]["trials"], 2)
        self.assertEqual(by_step["saturation -1 (less saturation)"]["trials"], 2)
        losses = diag["losses"]  # type: ignore[index]
        self.assertEqual(len(losses), 3)
        self.assertEqual(
            [loss["changes"] for loss in losses],
            ["slightly brighter", "darker, more contrast", "darker, more contrast"],
        )
        self.assertTrue(all(loss["scene"].startswith("scene of trial") for loss in losses))
        self.assertEqual(losses[0]["left_empty"], "highlights, shadows, blacks, tint, shadow_color, highlight_color")
        self.assertEqual(diag["left_empty"]["trials"], 8)  # type: ignore[index]
        self.assertEqual(diag["left_empty"]["fields"]["tint"], 8)  # type: ignore[index]
        self.assertEqual(diag["left_empty"]["fields"]["exposure"], 0)  # type: ignore[index]
        text = scorelib.format_report(score_answers(plan, results))
        self.assertIn("Fields Apple's model left empty (they set nothing), of 8 trials:", text)
        self.assertIn("For tuning the table (docs/19 section 5)", text)
        self.assertIn("Trials Apple Auto lost against the rule (3)", text)
        self.assertIn('"scene of trial', text)

    def test_the_diagnostics_read_the_steps_from_the_photo_when_a_trial_has_none(self) -> None:
        plan, results = make_case([("rule", "a", 1, 1, 0, 0, 0, (0, -2, 0, 0))])
        for trial in plan["trials"]:  # type: ignore[union-attr]
            del trial["steps"]
        plan["photos"] = [{"number": t["number"], "steps": {"exposure": 0, "white": -2, "contrast": 0, "saturation": 0}}
                          for t in plan["trials"]]  # type: ignore[union-attr]
        diag = score_answers(plan, results)["diagnostics"]
        by_direction = {g["label"]: g for g in diag["by_direction"]}  # type: ignore[index]
        self.assertEqual(by_direction["cooler"]["trials"], 2)
        self.assertEqual(diag["losses"][0]["changes"], "much cooler")  # type: ignore[index]

    def test_steps_in_words(self) -> None:
        self.assertEqual(scorelib.steps_words(S(3, -2, 2, -1)),
                         "much brighter, much more contrast, much cooler, less saturation")
        self.assertEqual(
            scorelib.steps_words(S(highlights=-1, blacks=1, tint=2, shadow_color="teal", shadow_color_strength=2,
                                   highlight_color="warm", highlight_color_strength=1)),
            "darker highlights, lifted blacks, much more magenta, teal shadows (medium), warm highlights (subtle)",
        )
        self.assertEqual(scorelib.steps_words(S(0, 0, 0, 0)), "no change")
        # A colour without a strength, none, and empty fields change nothing.
        nothing = S(shadow_color="teal", highlight_color="none", highlight_color_strength=2)
        self.assertEqual(scorelib.steps_words(nothing), "no change")
        self.assertEqual(scorelib.steps_words(None), "unknown")
        self.assertEqual(
            scorelib.empty_words(S(1, 0, 0, 0)), "highlights, shadows, blacks, tint, shadow_color, highlight_color"
        )

    def test_the_diagnostics_split_the_wins_by_the_colours_and_say_the_plan_of_a_loss(self) -> None:
        teal = S(shadow_color="teal", shadow_color_strength=2)
        both = S(1, highlights=-1, shadow_color="teal", shadow_color_strength=1, highlight_color="warm",
                 highlight_color_strength=3)
        plan, results = make_case([("rule", "a", 2, 1, 0, 0, 0, teal), ("rule", "b", 1, 1, 0, 0, 0, both)])
        for trial in plan["trials"]:  # type: ignore[union-attr]
            trial["plan"] = f"plan of trial {trial['id']}"
        diag = score_answers(plan, results)["diagnostics"]
        by_color = {g["label"]: g for g in diag["by_color"]}  # type: ignore[index]
        self.assertEqual(by_color["teal shadows"]["trials"], 5)
        self.assertAlmostEqual(by_color["teal shadows"]["share"], 0.6)
        self.assertEqual(by_color["warm highlights"]["trials"], 2)
        self.assertEqual(by_color["blue shadows"]["trials"], 0)
        single = {g["label"]: g for g in diag["by_single_change"]}  # type: ignore[index]
        self.assertEqual(single["only shadow_color"]["trials"], 3)
        self.assertEqual(single["two or more changes"]["trials"], 2)
        by_step = {g["label"]: g for g in diag["by_step"]}  # type: ignore[index]
        self.assertEqual(by_step["shadow_color teal 2 (teal shadows, medium)"]["trials"], 3)
        self.assertEqual(by_step["highlight_color warm 3 (warm highlights, strong)"]["trials"], 2)
        self.assertEqual(by_step["highlights -1 (darker highlights)"]["trials"], 2)
        by_direction = {g["label"]: g for g in diag["by_direction"]}  # type: ignore[index]
        self.assertEqual(by_direction["darker highlights"]["trials"], 2)
        losses = diag["losses"]  # type: ignore[index]
        self.assertEqual([loss["changes"] for loss in losses], [
            "teal shadows (medium)",
            "slightly brighter, darker highlights, teal shadows (subtle), warm highlights (strong)",
        ])
        self.assertEqual(losses[0]["plan"], "plan of trial 3")
        text = scorelib.format_report(score_answers(plan, results))
        self.assertIn("By the colour of the shadows or the highlights:", text)
        self.assertIn('"scene of trial 3" (plan: "plan of trial 3"); teal shadows (medium)', text)

    def test_results_of_another_plan_are_refused(self) -> None:
        plan, results = make_case([("rule", "a", 1, 0, 0, 0, 0)])
        results["plan_id"] = "otherplan"
        with self.assertRaises(scorelib.ScoreError):
            scorelib.trial_answers(plan, results)

    def test_a_plan_of_the_parked_autoset_test_is_refused(self) -> None:
        plan, _ = make_case([("rule", "a", 1, 0, 0, 0, 0)])
        for trial in plan["trials"]:  # type: ignore[union-attr]
            trial["left"], trial["right"] = "autoset", "rule"
        with self.assertRaises(scorelib.ScoreError):
            scorelib.validate_plan(plan, Path("plan.json"))

    def test_duplicate_unknown_and_invalid_answers_are_refused(self) -> None:
        plan, results = make_case([("rule", "a", 2, 0, 0, 0, 0)])
        duplicate = dict(results)
        duplicate["answers"] = [results["answers"][0], results["answers"][0]]  # type: ignore[index]
        cases = [
            duplicate,
            {**results, "answers": [{"trial": 99, "answer": "left", "flags": {}}]},
            {**results, "answers": [{"trial": 1, "answer": "maybe", "flags": {}}]},
            {**results, "format": "something-else"},
        ]
        for bad in cases:
            with self.assertRaises(scorelib.ScoreError):
                scorelib.trial_answers(plan, bad)

    def test_score_finds_the_plan_next_to_the_results_and_writes_score_json(self) -> None:
        plan, results = make_case([("rule", "a", 70, 20, 10, 0, 0), ("camera", "a", 70, 20, 10, 0, 0)])
        folder = Path(tempfile.mkdtemp(prefix="ab-score-"))
        self.addCleanup(shutil.rmtree, folder, True)
        (folder / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
        results_path = folder / "results.json"
        results_path.write_text(json.dumps(results), encoding="utf-8")
        buffer = StringIO()
        with redirect_stdout(buffer):
            code = ab.main(["score", str(results_path)])
        self.assertEqual(code, 0)
        self.assertIn("(provisional thresholds, docs/19 §13): PASS", buffer.getvalue())
        written = json.loads((folder / "score.json").read_text(encoding="utf-8"))
        self.assertEqual(written["gate"]["verdict"], "PASS")
        self.assertEqual(written["format"], "yellowmoon-ab-score")
        self.assertIn("diagnostics", written)

    def test_score_without_a_plan_asks_for_one(self) -> None:
        plan, results = make_case([("rule", "a", 1, 0, 0, 0, 0)])
        results["plan_id"] = "no-such-plan-anywhere"
        folder = Path(tempfile.mkdtemp(prefix="ab-score-"))
        self.addCleanup(shutil.rmtree, folder, True)
        results_path = folder / "results.json"
        results_path.write_text(json.dumps(results), encoding="utf-8")
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            code = ab.main(["score", str(results_path)])
        self.assertEqual(code, 2)


Spec = Tuple  # (comparison, stratum, wins, losses, ties, flags on Apple, flags on the other[, steps])


def make_case(spec: Sequence[Spec]) -> Tuple[Dict[str, object], Dict[str, object]]:
    """A plan and the results of it. Each spec line is (comparison, stratum, wins, losses, ties, flags on Apple's
    picture, flags on the other picture[, steps]); the first `flags` trials of a group carry the flag, and the steps
    (four numbers for exposure, white, contrast and saturation, or a whole steps tuple) default to S(1, 0, 0, 0). The
    plan lists the comparisons it has, as `prepare` writes it."""
    trials: List[Dict[str, object]] = []
    answers: List[Dict[str, object]] = []
    for line in spec:
        comparison, stratum, wins, losses, ties, flags_apple, flags_other = line[:7]
        steps = line[7] if len(line) > 7 else S(1, 0, 0, 0)
        if len(steps) == 4:
            steps = S(*steps)
        other = planlib.COMPARISONS[comparison]
        outcomes = ["win"] * wins + ["loss"] * losses + ["tie"] * ties
        for index, outcome in enumerate(outcomes):
            trial_id = len(trials) + 1
            apple_left = index % 2 == 0
            left, right = ("apple", other) if apple_left else (other, "apple")
            trials.append(
                {
                    "id": trial_id,
                    "number": trial_id,
                    "stem": f"DSC{trial_id:05d}",
                    "comparison": comparison,
                    "stratum": stratum,
                    "left": left,
                    "right": right,
                    "steps": planlib.steps_dict(steps),
                    "scene": f"scene of trial {trial_id}",
                }
            )
            apple_side = "left" if apple_left else "right"
            other_side = "right" if apple_left else "left"
            if outcome == "tie":
                answer = "same"
            elif outcome == "win":
                answer = apple_side
            else:
                answer = other_side
            flags = {"left": False, "right": False}
            if index < flags_apple:
                flags[apple_side] = True
            if index < flags_other:
                flags[other_side] = True
            answers.append({"trial": trial_id, "answer": answer, "flags": flags, "timestamp": "2026-10-19T00:00:00Z"})
    present = [name for name in planlib.COMPARISONS if any(t["comparison"] == name for t in trials)]
    plan: Dict[str, object] = {
        "format": planlib.PLAN_FORMAT,
        "version": planlib.PLAN_VERSION,
        "plan_id": "testplan01",
        "seed": 1,
        "comparisons": present,
        "trials": trials,
    }
    results: Dict[str, object] = {
        "format": page.RESULTS_FORMAT,
        "version": 1,
        "plan_id": "testplan01",
        "answers": answers,
    }
    return plan, results


def drop_answers(plan: Dict[str, object], results: Dict[str, object], comparison: str) -> Dict[str, object]:
    """The results without the answers to the trials of one comparison, as an export made before they are answered."""
    ids = {trial["id"] for trial in plan["trials"] if trial["comparison"] == comparison}  # type: ignore[index]
    answers = [row for row in results["answers"] if row["trial"] not in ids]  # type: ignore[index]
    return {**results, "answers": answers}


def score_answers(plan: Dict[str, object], results: Dict[str, object]) -> Dict[str, object]:
    """The score document for a plan and its results, as `score.py` builds it."""
    rows = scorelib.trial_answers(plan, results)
    return scorelib.build_report(plan, rows)


if __name__ == "__main__":
    unittest.main()

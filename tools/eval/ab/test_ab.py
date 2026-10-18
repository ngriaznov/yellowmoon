"""Tests of the blind A/B kit: the plan, the pictures (through a fake `ym`), the page and the scoring.

No `ym`, no network and no raw files are needed:
`python3 -I -m unittest discover -s tools/eval/ab -p 'test_*.py'`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import signal
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
import score as scorelib  # noqa: E402

#: Names that must appear nowhere in the page or in plan.js: the variants and their sources.
FORBIDDEN_NAMES = ("autoset", "lightroom", "camera", "neutral", "preference")
#: Ordinary words that must not appear in the visible markup (the CSS keyword `auto` is not visible text).
FORBIDDEN_VISIBLE_WORDS = (r"\brule\b", r"\bauto\b", r"\bwhite\b")


def visible_markup(text: str) -> str:
    """The page without its <style> and <script> blocks: what a reader sees, and what the screen reader names."""
    without_style = re.sub(r"<style\b.*?</style>", "", text, flags=re.S | re.I)
    return re.sub(r"<script\b.*?</script>", "", without_style, flags=re.S | re.I)

#: A stand-in for `ym`: it records each call (arguments, data folder, whether the data folder has the model, whether
#: the home folder is empty), honours --out and --name, writes a tiny JPEG, and fails for the stems in FAKE_YM_FAIL.
FAKE_YM = r'''#!/usr/bin/env python3
import json
import os
import sys

args = sys.argv[1:]
log = os.environ.get("FAKE_YM_LOG")


def record(**fields):
    if log:
        with open(log, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(fields) + "\n")


if args == ["--version"]:
    print("ym 9.9.9-test (pipeline 0.0.0, fake)")
    sys.exit(0)
home = os.environ.get("YELLOWMOON_HOME", "")
data = os.environ.get("YELLOWMOON_DATA", "")
model = os.path.isfile(os.path.join(data, "models", "autoset", "autoset.json"))
home_empty = not os.listdir(home)
if args[:1] == ["info"]:
    source = args[-1]
    with open(source, "rb") as handle:
        first = handle.readline().decode("ascii", "replace").split()
    fields = dict(part.split("=", 1) for part in first[1:] if "=" in part)
    stem = os.path.splitext(os.path.basename(source))[0]
    record(cmd="info", stem=stem, data=data, model=model, home_empty=home_empty)
    preview = None
    if "OFFSET" in fields:
        preview = {"offset": int(fields["OFFSET"]), "length": int(fields["LENGTH"])}
    print(json.dumps({"status": "ok", "error": None, "input": source, "preview": preview}))
    sys.exit(0)
if args[:1] == ["render"]:
    source = args[-1]
    stem = os.path.splitext(os.path.basename(source))[0]
    out = args[args.index("--out") + 1]
    name = args[args.index("--name") + 1]
    record(cmd="render", stem=stem, args=args, data=data, model=model, home_empty=home_empty)
    if stem in os.environ.get("FAKE_YM_FAIL", "").split(","):
        message = "fake failure for " + stem
        print(json.dumps({"status": "error", "error": {"kind": "unsupported", "message": message}, "input": source}))
        sys.exit(1)
    os.makedirs(out, exist_ok=True)
    target = os.path.join(out, name + ".jpg")
    with open(target, "wb") as handle:
        handle.write(b"\xff\xd8\xff\xe0render of " + stem.encode() + b"\xff\xd9")
    print(json.dumps({"status": "ok", "error": None, "input": source, "output": target}))
    sys.exit(0)
sys.exit(2)
'''


def camera_bytes(stem: str) -> bytes:
    """The embedded JPEG the fake raw files carry for a photo."""
    return b"\xff\xd8\xff\xe0camera of " + stem.encode() + b"\xff\xd9"


def lightroom_bytes(stem: str) -> bytes:
    """The JPEG of a Lightroom export for a photo."""
    return b"\xff\xd8\xff\xe0lightroom of " + stem.encode() + b"\xff\xd9"


def write_arw(path: Path, jpeg: Optional[bytes]) -> None:
    """A fake ARW: a header line with the embedded JPEG's offset (256) and length, then the JPEG, then some raw bytes.
    With `jpeg` None the header has no preview."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if jpeg is None:
        body = b"FAKEARW NOPREVIEW\n" + b"x" * 300
    else:
        header = f"FAKEARW OFFSET=256 LENGTH={len(jpeg)}\n".encode("ascii")
        body = header.ljust(256, b" ") + jpeg
    path.write_bytes(body + b"RAWDATA" * 40)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_log(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def variant_of_call(call: Dict[str, object]) -> str:
    """Which variant a logged `render` call made, from its flags."""
    args = call["args"]  # type: ignore[index]
    if "--contrast" in args:  # type: ignore[operator]
        return "rule"
    if "--white-preference" in args:  # type: ignore[operator]
        return "autoset-neutral"
    return "autoset"


class PlanTests(unittest.TestCase):
    """The photos, the strata, the Lightroom matches and the trial plan."""

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="ab-plan-"))
        self.addCleanup(shutil.rmtree, self.root, True)

    def photo_list(self, count: int, lightroom_every: int = 2) -> List[planlib.PlanPhoto]:
        photos = []
        for number in range(1, count + 1):
            stratum = ("portrait", "landscape", "night", "other")[number % 4]
            photos.append(
                planlib.PlanPhoto(
                    number=number,
                    stem=f"DSC{number:05d}",
                    stratum=stratum,
                    lightroom=number % lightroom_every == 0,
                )
            )
        return photos

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

    def test_lightroom_matches_by_stem_ignoring_case_and_skips_shared_stems(self) -> None:
        raw = [
            planlib.Photo("DSC0001", "portrait", self.root / "DSC0001.ARW"),
            planlib.Photo("dsc0002", "portrait", self.root / "dsc0002.ARW"),
            planlib.Photo("DSC0003", "night", self.root / "night" / "DSC0003.ARW"),
            planlib.Photo("DSC0003", "other", self.root / "DSC0003.ARW"),
            planlib.Photo("DSC0004", "other", self.root / "DSC0004.ARW"),
        ]
        export = self.root / "export"
        export.mkdir()
        for name in ("dsc0001.JPG", "DSC0002.jpeg", "DSC0003.jpg", "DSC0004.png", "._DSC0004.jpg"):
            (export / name).write_bytes(b"\xff\xd8")
        matches, notes = planlib.match_lightroom(raw, export)
        self.assertEqual(set(matches), {0, 1})
        self.assertEqual(matches[0].name, "dsc0001.JPG")
        self.assertEqual(matches[1].name, "DSC0002.jpeg")
        self.assertTrue(any("DSC0003" in note for note in notes))

    def test_the_plan_is_deterministic_for_a_seed(self) -> None:
        photos = self.photo_list(40)
        first = planlib.build_trials(photos, 7)
        second = planlib.build_trials(photos, 7)
        other = planlib.build_trials(photos, 8)
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)

    def test_every_photo_gets_its_comparisons_and_lightroom_only_when_matched(self) -> None:
        photos = self.photo_list(12, lightroom_every=3)
        trials = planlib.build_trials(photos, 1)
        for photo in photos:
            names = sorted(t.comparison for t in trials if t.number == photo.number)
            expected = ["camera", "neutral", "rule"] + (["lightroom"] if photo.lightroom else [])
            self.assertEqual(names, sorted(expected), f"photo {photo.number}")

    def test_sides_are_balanced_within_each_comparison(self) -> None:
        photos = self.photo_list(301, lightroom_every=2)
        trials = planlib.build_trials(photos, 3)
        for name in planlib.COMPARISONS:
            group = [t for t in trials if t.comparison == name]
            autoset_left = sum(1 for t in group if t.left == "autoset")
            self.assertLessEqual(abs(autoset_left - len(group) / 2), 0.5, name)
            for t in group:
                self.assertEqual({t.left, t.right}, {"autoset", planlib.COMPARISONS[name]})

    def test_the_odd_trial_of_a_comparison_goes_either_side(self) -> None:
        # One photo gives one trial per comparison, an odd count: over many seeds the trial should land on each side.
        photos = self.photo_list(1, lightroom_every=99)
        left_is_autoset = 0
        for seed in range(100):
            for trial in planlib.build_trials(photos, seed):
                if trial.comparison == "rule" and trial.left == "autoset":
                    left_is_autoset += 1
        self.assertTrue(30 <= left_is_autoset <= 70, left_is_autoset)

    def test_trials_are_shuffled_and_numbered_from_one(self) -> None:
        photos = self.photo_list(30)
        trials = planlib.build_trials(photos, 5)
        self.assertEqual([t.id for t in trials], list(range(1, len(trials) + 1)))
        numbers = [t.number for t in trials]
        self.assertNotEqual(numbers, sorted(numbers))

    def test_the_plan_id_changes_with_the_provenance(self) -> None:
        photos = self.photo_list(6)
        first = planlib.build_plan(photos, 1, {"ym": "ym 1.0"})
        same = planlib.build_plan(photos, 1, {"ym": "ym 1.0"})
        later = planlib.build_plan(photos, 1, {"ym": "ym 1.1"})
        self.assertEqual(first["plan_id"], same["plan_id"])
        self.assertNotEqual(first["plan_id"], later["plan_id"])

    def test_picture_paths_hide_the_variant_names(self) -> None:
        photos = self.photo_list(8, lightroom_every=2)
        plan = planlib.build_plan(photos, 9, {"ym": "ym"})
        pattern = re.compile(r"^pictures/[0-9a-f]{8}/\d{4}_DSC\d{5}\.jpg$")
        for entry in plan["photos"]:
            for path in entry["pictures"].values():
                self.assertRegex(path, pattern)
                for word in ("autoset", "rule", "camera", "lightroom", "neutral"):
                    self.assertNotIn(word, path.lower())
        self.assertEqual(len({v["code"] for v in plan["variants"].values()}), len(plan["variants"]))


class PrepareHarness(unittest.TestCase):
    """A small test set (five raw files in four strata, two with Lightroom exports) rendered through the fake `ym`."""

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="ab-prepare-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.ym = self.root / "bin" / "ym"
        self.ym.parent.mkdir()
        self.ym.write_text(FAKE_YM)
        os.chmod(self.ym, 0o755)
        self.data = self.root / "data"
        (self.data / "models" / "autoset").mkdir(parents=True)
        (self.data / "models" / "autoset" / "autoset.json").write_text("{}")
        (self.data / "models" / "illum").mkdir(parents=True)
        (self.data / "models" / "illum" / "illum.json").write_text("{}")
        (self.data / "cameras").mkdir()
        (self.data / "cameras" / "profile.json").write_text("{}")
        self.photos = self.root / "photos"
        self.lightroom = self.root / "lightroom"
        self.lightroom.mkdir()
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
        ):
            write_arw(self.photos / relpath, camera_bytes(stem))
        (self.lightroom / "L1.jpg").write_bytes(lightroom_bytes("L1"))
        (self.lightroom / "p1.JPEG").write_bytes(lightroom_bytes("P1"))
        env = mock.patch.dict(os.environ, {"FAKE_YM_LOG": str(self.log)})
        env.start()
        self.addCleanup(env.stop)
        env_fail = mock.patch.dict(os.environ, {"FAKE_YM_FAIL": ""})
        env_fail.start()
        self.addCleanup(env_fail.stop)

    def options(self, **overrides: object) -> ab.PrepareOptions:
        base: Dict[str, object] = {
            "photos": self.photos,
            "out": self.out,
            "lightroom": self.lightroom,
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

    def raw_hashes(self) -> Dict[str, str]:
        return {str(path): sha256_of(path) for path in sorted(self.photos.rglob("*.ARW"))}


class PrepareTests(PrepareHarness):
    """`prepare` through the fake `ym`: the flags, the data folders, the homes, the copies and the failures."""

    def test_each_variant_is_rendered_with_its_own_flags_data_and_empty_home(self) -> None:
        self.prepare()
        calls = [c for c in read_log(self.log) if c["cmd"] == "render"]
        self.assertTrue(calls)
        for call in calls:
            args = call["args"]
            for flag in ("--cpu", "--quiet", "--json", "--overwrite"):
                self.assertIn(flag, args)
            self.assertEqual(args[args.index("--format") + 1], "jpeg")
            self.assertEqual(args[args.index("--quality") + 1], "92")
            self.assertEqual(args[args.index("--long-edge") + 1], "900")
            self.assertTrue(call["home_empty"], "each render gets an empty YELLOWMOON_HOME")
            variant = variant_of_call(call)
            if variant == "rule":
                self.assertEqual(args[args.index("--white-preference") + 1], "0")
                self.assertEqual(args[args.index("--contrast") + 1], "0")
                self.assertFalse(call["model"], "the rule runs without models/autoset")
                self.assertNotEqual(call["data"], str(self.data))
            elif variant == "autoset-neutral":
                self.assertEqual(args[args.index("--white-preference") + 1], "0")
                self.assertNotIn("--contrast", args)
                self.assertTrue(call["model"])
                self.assertEqual(call["data"], str(self.data))
            else:
                self.assertNotIn("--white-preference", args)
                self.assertNotIn("--contrast", args)
                self.assertTrue(call["model"])
                self.assertEqual(call["data"], str(self.data))

    def test_the_rule_copy_is_deleted_and_the_shipped_data_is_untouched(self) -> None:
        rule_calls = [c for c in self.prepare_then_log() if c["cmd"] == "render" and variant_of_call(c) == "rule"]
        self.assertTrue(rule_calls)
        for call in rule_calls:
            self.assertFalse(Path(str(call["data"])).exists(), "the temporary copy is removed")
        self.assertTrue((self.data / "models" / "autoset" / "autoset.json").is_file())
        self.assertEqual(list(self.scratch.iterdir()), [])

    def prepare_then_log(self) -> List[Dict[str, object]]:
        self.prepare()
        return read_log(self.log)

    def test_the_originals_are_never_written(self) -> None:
        before = self.raw_hashes()
        self.prepare()
        self.assertEqual(self.raw_hashes(), before)

    def test_camera_and_lightroom_pictures_are_byte_copies(self) -> None:
        result = self.prepare()
        for entry in result.plan["photos"]:  # type: ignore[index]
            stem = entry["stem"]
            camera = self.out / entry["pictures"]["camera"]
            self.assertEqual(camera.read_bytes(), camera_bytes(stem))
            if entry["lightroom"]:
                lightroom = self.out / entry["pictures"]["lightroom"]
                self.assertEqual(lightroom.read_bytes(), lightroom_bytes(stem))
            else:
                self.assertNotIn("lightroom", entry["pictures"])

    def test_only_matched_photos_have_lightroom_comparisons(self) -> None:
        result = self.prepare()
        lightroom_stems = {t["stem"] for t in result.plan["trials"] if t["comparison"] == "lightroom"}  # type: ignore[index]
        self.assertEqual(lightroom_stems, {"L1", "P1"})
        self.assertEqual(result.lightroom_matched, 2)

    def test_a_failing_render_drops_the_photo_from_every_trial(self) -> None:
        with mock.patch.dict(os.environ, {"FAKE_YM_FAIL": "P1"}):
            result = self.prepare()
        plan = result.plan
        self.assertNotIn("P1", {p["stem"] for p in plan["photos"]})  # type: ignore[index]
        self.assertNotIn("P1", {t["stem"] for t in plan["trials"]})  # type: ignore[index]
        self.assertEqual([s["stem"] for s in result.skipped], ["P1"])
        problems = result.skipped[0]["problems"]
        self.assertIn("fake failure for P1", problems[0]["reason"])  # type: ignore[index]
        self.assertEqual(len(plan["trials"]), 4 + 4 + 1 + 4)  # type: ignore[arg-type,index]

    def test_a_photo_without_an_embedded_jpeg_is_skipped(self) -> None:
        write_arw(self.photos / "landscape" / "L2.ARW", None)
        result = self.prepare()
        skipped = {s["stem"]: s for s in result.skipped}
        self.assertIn("L2", skipped)
        self.assertIn("embedded JPEG", skipped["L2"]["problems"][0]["reason"])  # type: ignore[index]

    def test_a_lightroom_file_that_is_not_a_jpeg_drops_its_photo(self) -> None:
        (self.lightroom / "L1.jpg").write_bytes(b"not a jpeg")
        result = self.prepare()
        self.assertIn("L1", {s["stem"] for s in result.skipped})
        self.assertNotIn("L1", {t["stem"] for t in result.plan["trials"]})  # type: ignore[index]

    def test_the_number_of_jobs_does_not_change_the_plan(self) -> None:
        one = self.prepare(jobs=1, out=self.root / "out-one")
        four = self.prepare(jobs=4, out=self.root / "out-four")
        self.assertEqual(one.plan["plan_id"], four.plan["plan_id"])
        self.assertEqual(one.plan["trials"], four.plan["trials"])
        for entry in one.plan["photos"]:  # type: ignore[index]
            for rel in entry["pictures"].values():
                self.assertEqual(
                    (self.root / "out-one" / rel).read_bytes(),
                    (self.root / "out-four" / rel).read_bytes(),
                )

    def test_the_folder_gets_the_page_the_plan_and_the_pictures(self) -> None:
        result = self.prepare()
        self.assertTrue((self.out / "index.html").is_file())
        self.assertTrue((self.out / "plan.js").is_file())
        plan_doc = json.loads((self.out / "plan.json").read_text(encoding="utf-8"))
        self.assertEqual(plan_doc["plan_id"], result.plan["plan_id"])
        self.assertEqual(plan_doc["provenance"]["ym"], "ym 9.9.9-test (pipeline 0.0.0, fake)")
        self.assertEqual(
            plan_doc["provenance"]["autoset_model_sha256"],
            hashlib.sha256((self.data / "models" / "autoset" / "autoset.json").read_bytes()).hexdigest(),
        )
        for trial in plan_doc["trials"]:
            for side in ("left", "right"):
                self.assertTrue((self.out / trial["pictures"][side]).is_file())

    def test_an_interrupt_removes_the_temporary_copy_of_the_data_folder(self) -> None:
        with mock.patch.object(ab, "make_pictures", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.prepare()
        self.assertEqual(list(self.scratch.iterdir()), [])

    def test_no_photo_rendering_is_an_error_with_no_test(self) -> None:
        with mock.patch.dict(os.environ, {"FAKE_YM_FAIL": "L1,L2,N1,T1,P1"}):
            with self.assertRaises(ab.PrepareFailed):
                self.prepare()


class CommandLineTests(PrepareHarness):
    """The command line: the usage errors and the exit codes."""

    def run_main(self, argv: Sequence[str]) -> Tuple[int, str]:
        buffer = StringIO()
        with redirect_stdout(buffer):
            code = ab.main(list(argv))
        return code, buffer.getvalue()

    def test_prepare_writes_the_summary_and_exits_zero(self) -> None:
        code, text = self.run_main(
            ["prepare", str(self.photos), "--out", str(self.out), "--lightroom", str(self.lightroom),
             "--ym", str(self.ym), "--data", str(self.data), "--jobs", "2", "--long-edge", "900"]
        )
        self.assertEqual(code, 0)
        self.assertIn("photos: 5 found, 5 in the test", text)
        self.assertIn("open ", text)
        self.assertIn(str(self.out.resolve() / "index.html"), text)

    def test_usage_errors_exit_two(self) -> None:
        empty = self.root / "no-raw-files"
        empty.mkdir()
        (empty / "notes.txt").write_text("nothing here", encoding="utf-8")
        cases = [
            ["prepare", str(self.root / "missing")],
            ["prepare", str(self.photos), "--ym", str(self.root / "no-such-ym")],
            ["prepare", str(self.photos), "--ym", str(self.ym), "--data", str(self.root / "empty")],
            ["prepare", str(self.photos), "--ym", str(self.ym), "--data", str(self.data), "--lightroom", "/no/such"],
            ["prepare", str(empty), "--ym", str(self.ym), "--data", str(self.data)],
            ["score", str(self.root / "missing.json")],
        ]
        for argv in cases:
            code, _ = self.run_main(argv)
            self.assertEqual(code, 2, argv)

    def test_an_unreadable_folder_exits_one_with_a_message(self) -> None:
        error = PermissionError(13, "Permission denied", str(self.photos))
        with mock.patch.object(ab, "run_prepare", side_effect=error):
            buffer = StringIO()
            with redirect_stderr(buffer):
                code = ab.main(["prepare", str(self.photos)])
        self.assertEqual(code, 1)
        self.assertIn("Permission denied", buffer.getvalue())
        self.assertNotIn("Traceback", buffer.getvalue())

    def test_sigterm_is_handled_like_ctrl_c(self) -> None:
        def terminate(*args: object, **kwargs: object) -> int:
            os.kill(os.getpid(), signal.SIGTERM)
            return 0

        with mock.patch.object(ab, "run_prepare", side_effect=terminate):
            buffer = StringIO()
            with redirect_stderr(buffer):
                code = ab.main(["prepare", str(self.photos)])
        self.assertEqual(code, 130)
        self.assertIn("interrupted", buffer.getvalue())

    def test_an_unknown_command_is_an_argument_error(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            with redirect_stdout(StringIO()):
                ab.main(["bogus"])
        self.assertEqual(caught.exception.code, 2)


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
        photos = [planlib.PlanPhoto(number=1, stem="DSC1", stratum="other", lightroom=True)]
        plan = planlib.build_plan(photos, 1, {"ym": "ym"})
        text = page.plan_js(plan).lower()
        self.assertIn("window.ym_ab_plan = {", text)
        for name in FORBIDDEN_NAMES:
            self.assertNotIn(name, text, name)
        self.assertIsNone(re.search(r"\brule\b|\bauto\b|\bwhite\b", text))
        self.assertIn(plan["plan_id"], text)
        self.assertNotIn("http", text)

    def test_the_page_script_keys_results_by_plan_and_exports_results_json(self) -> None:
        self.assertIn('"ym-ab:" + plan.plan_id', page.PAGE)
        self.assertIn('link.download = "results.json"', page.PAGE)
        self.assertIn('format: "yellowmoon-ab-results"', page.PAGE)


class ScoreTests(unittest.TestCase):
    """The Wilson bound, ties, the strata, the flags and the gate's rules, on hand-made answers."""

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

    def test_all_gates_pass_on_a_clear_result(self) -> None:
        plan, results = make_case(
            [
                ("rule", "portrait", 70, 20, 10, 1, 5),
                ("camera", "portrait", 70, 20, 10, 1, 5),
                ("lightroom", "portrait", 70, 20, 10, 1, 5),
                ("neutral", "portrait", 40, 60, 0, 0, 0),
            ]
        )
        report = score_answers(plan, results)
        self.assertEqual(report["comparisons"]["rule"]["verdict"], "PASS")  # type: ignore[index]
        self.assertEqual(report["comparisons"]["neutral"]["verdict"], "reported")  # type: ignore[index]
        self.assertEqual(report["gate"]["verdict"], "PASS")  # type: ignore[index]

    def test_a_pooled_share_below_sixty_percent_fails(self) -> None:
        plan, results = make_case([("rule", "portrait", 55, 40, 5, 0, 0)])
        report = score_answers(plan, results)
        rule = report["comparisons"]["rule"]  # type: ignore[index]
        self.assertEqual(rule["verdict"], "FAIL")
        self.assertIn("pooled win share", " ".join(rule["reasons"]))
        self.assertEqual(report["gate"]["verdict"], "FAIL")  # type: ignore[index]

    def test_a_stratum_below_forty_five_percent_fails(self) -> None:
        plan, results = make_case(
            [("rule", "portrait", 80, 0, 0, 0, 0), ("rule", "night", 0, 20, 0, 0, 0)]
        )
        report = score_answers(plan, results)
        rule = report["comparisons"]["rule"]  # type: ignore[index]
        self.assertEqual(rule["verdict"], "FAIL")
        self.assertIn("stratum night", " ".join(rule["reasons"]))
        self.assertTrue(rule["strata"]["night"]["below_gate"])  # type: ignore[index]

    def test_autoset_flagged_more_often_than_the_other_picture_fails(self) -> None:
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
        plan, results = make_case(
            [
                ("rule", "a", 100, 0, 0, 0, 0),
                ("rule", "b", 45, 55, 0, 0, 0),
                ("camera", "a", 60, 40, 0, 0, 0),
                ("lightroom", "a", 60, 40, 0, 0, 0),
            ]
        )
        report = score_answers(plan, results)
        self.assertAlmostEqual(report["comparisons"]["rule"]["share"], 0.725)  # type: ignore[index]
        self.assertEqual(report["comparisons"]["rule"]["verdict"], "PASS")  # type: ignore[index]
        self.assertEqual(report["comparisons"]["camera"]["verdict"], "PASS")  # type: ignore[index]
        self.assertEqual(report["gate"]["verdict"], "PASS")  # type: ignore[index]

    def test_a_missing_lightroom_comparison_leaves_the_gate_undecided(self) -> None:
        plan, results = make_case(
            [("rule", "a", 70, 20, 10, 0, 0), ("camera", "a", 70, 20, 10, 0, 0)]
        )
        report = score_answers(plan, results)
        self.assertEqual(report["comparisons"]["lightroom"]["verdict"], "NOT RUN")  # type: ignore[index]
        self.assertEqual(report["gate"]["verdict"], "NOT DECIDED")  # type: ignore[index]
        self.assertTrue(report["complete"], "every planned trial is answered; the Lightroom set is simply absent")

    def test_a_failure_outranks_a_missing_comparison(self) -> None:
        plan, results = make_case([("rule", "a", 10, 90, 0, 0, 0)])
        self.assertEqual(score_answers(plan, results)["gate"]["verdict"], "FAIL")  # type: ignore[index]

    def test_the_neutral_comparison_is_reported_not_gated(self) -> None:
        plan, results = make_case(
            [
                ("rule", "a", 70, 20, 10, 0, 0),
                ("camera", "a", 70, 20, 10, 0, 0),
                ("lightroom", "a", 70, 20, 10, 0, 0),
                ("neutral", "a", 0, 100, 0, 0, 0),
            ]
        )
        report = score_answers(plan, results)
        self.assertEqual(report["comparisons"]["neutral"]["verdict"], "reported")  # type: ignore[index]
        self.assertEqual(report["gate"]["verdict"], "PASS")  # type: ignore[index]

    def test_results_of_another_plan_are_refused(self) -> None:
        plan, results = make_case([("rule", "a", 1, 0, 0, 0, 0)])
        results["plan_id"] = "otherplan"
        with self.assertRaises(scorelib.ScoreError):
            scorelib.trial_answers(plan, results)

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
        plan, results = make_case([("rule", "a", 70, 20, 10, 0, 0), ("camera", "a", 70, 20, 10, 0, 0),
                                   ("lightroom", "a", 70, 20, 10, 0, 0)])
        folder = Path(tempfile.mkdtemp(prefix="ab-score-"))
        self.addCleanup(shutil.rmtree, folder, True)
        (folder / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
        results_path = folder / "results.json"
        results_path.write_text(json.dumps(results), encoding="utf-8")
        buffer = StringIO()
        with redirect_stdout(buffer):
            code = ab.main(["score", str(results_path)])
        self.assertEqual(code, 0)
        text = buffer.getvalue()
        self.assertIn("Gate 5 (docs/18 section 8): PASS", text)
        self.assertIn("PASS", text)
        written = json.loads((folder / "score.json").read_text(encoding="utf-8"))
        self.assertEqual(written["gate"]["verdict"], "PASS")

    def test_score_without_a_plan_asks_for_one(self) -> None:
        plan, results = make_case([("rule", "a", 1, 0, 0, 0, 0)])
        results["plan_id"] = "no-such-plan-anywhere"
        folder = Path(tempfile.mkdtemp(prefix="ab-score-"))
        self.addCleanup(shutil.rmtree, folder, True)
        results_path = folder / "results.json"
        results_path.write_text(json.dumps(results), encoding="utf-8")
        with redirect_stdout(StringIO()):
            code = ab.main(["score", str(results_path)])
        self.assertEqual(code, 2)


def make_case(spec: Sequence[Tuple[str, str, int, int, int, int, int]]) -> Tuple[Dict[str, object], Dict[str, object]]:
    """A plan and the results of it. Each spec line is (comparison, stratum, wins, losses, ties, flags on autoset,
    flags on the other picture), and the first `flags` trials of a group carry the flag."""
    trials: List[Dict[str, object]] = []
    answers: List[Dict[str, object]] = []
    for comparison, stratum, wins, losses, ties, flags_autoset, flags_other in spec:
        other = planlib.COMPARISONS[comparison]
        outcomes = ["win"] * wins + ["loss"] * losses + ["tie"] * ties
        for index, outcome in enumerate(outcomes):
            trial_id = len(trials) + 1
            autoset_left = index % 2 == 0
            left, right = ("autoset", other) if autoset_left else (other, "autoset")
            trials.append(
                {"id": trial_id, "comparison": comparison, "stratum": stratum, "left": left, "right": right}
            )
            autoset_side = "left" if autoset_left else "right"
            other_side = "right" if autoset_left else "left"
            if outcome == "tie":
                answer = "same"
            elif outcome == "win":
                answer = autoset_side
            else:
                answer = other_side
            flags = {"left": False, "right": False}
            if index < flags_autoset:
                flags[autoset_side] = True
            if index < flags_other:
                flags[other_side] = True
            answers.append({"trial": trial_id, "answer": answer, "flags": flags, "timestamp": "2026-10-18T00:00:00Z"})
    plan: Dict[str, object] = {
        "format": planlib.PLAN_FORMAT,
        "version": 1,
        "plan_id": "testplan01",
        "seed": 1,
        "trials": trials,
    }
    results: Dict[str, object] = {
        "format": page.RESULTS_FORMAT,
        "version": 1,
        "plan_id": "testplan01",
        "answers": answers,
    }
    return plan, results


def score_answers(plan: Dict[str, object], results: Dict[str, object]) -> Dict[str, object]:
    """The score document for a plan and its results, as `score.py` builds it."""
    rows = scorelib.trial_answers(plan, results)
    return scorelib.build_report(plan, rows)


if __name__ == "__main__":
    unittest.main()

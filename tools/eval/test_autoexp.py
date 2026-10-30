"""Unit tests of `autoexp.py` (file finding, the cache, the checks on the renders, the reports) and, when a `ym` binary and
the synthetic set are there, one end-to-end run: python3 -I -m unittest discover -s tools/eval -p 'test_*.py'."""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
import stat
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import autoexp
import test_study

REPO = Path(__file__).resolve().parents[2]


def fake_inputs(ev_auto: float = 0.9, median: float = 0.08, seed: int = 3, jpeg_scale: float = 1.0) -> tuple[dict, dict, dict, dict]:
    """`(info, exif, extras, reduction)` of a made-up photo, enough for `autoexp._entry` and the reports."""
    red = test_study.reduction(median, ev_auto=ev_auto, seed=seed)
    if jpeg_scale != 1.0:
        red["j_q"] = np.asarray(red["j_q"]) * jpeg_scale
        red["j_median"] = red["j_median"] * jpeg_scale
    info = {
        "camera": {"body": "Sony Test", "match": "exact"},
        "exposure": {"iso": 100, "shutter_s": 0.01, "aperture": 4.0, "exposure_comp_ev": 0.0},
        "auto": {"exposure_ev": ev_auto},
    }
    exif = {"exposure_program": 3, "flash": 0, "dro": "Off"}
    return info, exif, {"ym_version": "ym test", "jpeg_colour_space": "srgb"}, red


class Sizes(unittest.TestCase):
    def test_scale_match_size_gives_the_jpegs_size_in_the_sources_orientation(self):
        self.assertEqual(autoexp.scale_match_size((3276, 4928, 3), (1080, 1616, 3)), (1616, 1080))
        # a portrait source against a landscape reference: the reference is turned
        self.assertEqual(autoexp.scale_match_size((4928, 3276), (1080, 1616)), (1080, 1616))
        # never above the source (a bigger reference would only interpolate)
        self.assertEqual(autoexp.scale_match_size((600, 900), (1080, 1616)), (900, 600))
        self.assertEqual(autoexp.scale_match_size((832, 1248), (832, 1248)), (1248, 832))


class Files(unittest.TestCase):
    def test_find_files_sorts_filters_and_skips_hidden_files(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name in ("b.ARW", "a.arw", ".hidden.ARW", "notes.txt", "C_Burst.ARW"):
                (root / name).write_bytes(b"x")
            files = autoexp.find_files([root], False, None)
            self.assertEqual([f.name for f in files], ["C_Burst.ARW", "a.arw", "b.ARW"])  # sorted by name, plain ASCII order
            self.assertEqual([f.name for f in autoexp.find_files([root], False, "burst")], ["C_Burst.ARW"])
            self.assertTrue(all(f.is_absolute() for f in files))

    def test_a_missing_input_is_a_study_error(self):
        with self.assertRaises(autoexp.StudyError):
            autoexp.find_files([Path("/nonexistent/yellowmoon-eval")], False, None)

    def test_one_per_scene_prefers_the_exact_raw_variant(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "arw").mkdir()
            (root / "scenes").mkdir()
            for scene in ("chart_a", "chart_b"):
                (root / "scenes" / f"{scene}.json").write_text("{}")
            for name in ("chart_a__craw", "chart_a__uncompressed14", "chart_a__lossless-l", "chart_b__lossless-l", "chart_b__craw", "plain"):
                (root / "arw" / f"{name}.ARW").write_bytes(b"x")
            picked = [f.name for f in autoexp.find_files([root / "arw"], True, None)]
            self.assertEqual(picked, ["chart_a__uncompressed14.ARW", "chart_b__lossless-l.ARW", "plain.ARW"])

    def test_the_sidecar_is_found_only_for_a_synthetic_name(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "arw").mkdir()
            (root / "scenes").mkdir()
            (root / "scenes" / "chart_a.json").write_text(json.dumps({"truth": {"probes": []}}))
            self.assertEqual(autoexp.synthetic_sidecar(root / "arw" / "chart_a__craw.ARW"), {"truth": {"probes": []}})
            self.assertIsNone(autoexp.synthetic_sidecar(root / "arw" / "other.ARW"))


class Reading(unittest.TestCase):
    def test_last_json_takes_the_last_object_line_or_an_indented_object(self):
        self.assertEqual(autoexp._last_json('log line\n{"a": 1}\n{"b": 2}\n'), {"b": 2})
        self.assertEqual(autoexp._last_json('{\n "a": 1,\n "b": [2, 3]\n}\n'), {"a": 1, "b": [2, 3]})
        self.assertIsNone(autoexp._last_json("nothing here"))

    def test_a_render_with_another_exposure_than_the_study_assumes_is_refused(self):
        autoexp.check_render_exposure({"settings": {"exposure": {"ev": 0.0, "kind": "manual"}}}, 0.0, "DNG")
        autoexp.check_render_exposure({"settings": {"exposure": {"ev": 0.9758425733837791}}}, 0.9758425733837791, "display")
        autoexp.check_render_exposure({}, 0.5, "display")  # an older ym without the field is not checked
        with self.assertRaises(autoexp.StudyError):
            autoexp.check_render_exposure({"settings": {"exposure": {"ev": 0.4}}}, 0.0, "DNG")
        with self.assertRaises(autoexp.StudyError):
            autoexp.check_render_exposure({"settings": {"exposure": {"ev": float("nan")}}}, 0.0, "DNG")


class FittedProfile(unittest.TestCase):
    """A `ym` DNG carries the fitted Adobe profile, whose offset is part of its `BaselineExposure`."""

    def write_profile(self, root: Path, curve, offset=-4.0) -> Path:
        path = root / "profile.json"
        path.write_text(json.dumps({"tone_curve": curve, "fit_offset_ev": offset}))
        return path

    def test_the_offset_is_read_when_the_dngs_tone_curve_is_the_profiles(self):
        curve = [[0.0, 0.0], [0.0002441, 9.1e-05], [0.5, 0.6180339887], [1.0, 1.0]]
        dng_curve = tuple(float(v) for v in np.asarray(curve, dtype=np.float32).ravel())  # the DNG stores float32
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(autoexp.fitted_profile_offset_ev(dng_curve, self.write_profile(Path(d), curve)), -4.0)
            # a binary built from another profile cannot be undone
            other = [[0.0, 0.0], [0.0002441, 9.1e-05], [0.5, 0.55], [1.0, 1.0]]
            with self.assertRaises(autoexp.StudyError):
                autoexp.fitted_profile_offset_ev(dng_curve, self.write_profile(Path(d), other))
            with self.assertRaises(autoexp.StudyError):
                autoexp.fitted_profile_offset_ev(dng_curve[:-2], self.write_profile(Path(d), curve))
            with self.assertRaises(autoexp.StudyError):
                autoexp.fitted_profile_offset_ev(dng_curve, Path(d) / "missing.json")

    def test_the_baseline_exposure_of_a_fitted_dng_has_no_display_placement(self):
        # `BaselineExposure = log2(x_white) + fit_offset_ev` of a fitted DNG (pipeline 0.8.0): a photo with x_white 16
        # (the floor) has the fit offset alone, and the scene value is stored * 16.
        self.assertFalse(hasattr(autoexp, "display_placement_ev"))
        x_white_ev = -4.0 - (-4.0) + 4.0
        self.assertAlmostEqual(2.0**x_white_ev, 16.0, places=9)

    def test_the_embedded_profile_of_the_repository_is_readable(self):
        profile = json.loads(autoexp.FITTED_PROFILE.read_text())
        flat = tuple(float(v) for v in np.asarray(profile["tone_curve"], dtype=np.float32).ravel())
        self.assertEqual(autoexp.fitted_profile_offset_ev(flat), profile["fit_offset_ev"])


class Runner(unittest.TestCase):
    def test_the_version_is_not_a_recorded_command_and_the_binary_runs_with_its_own_home(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            binary = root / "ym"
            binary.write_text('#!/bin/sh\nif [ "$1" = "--version" ]; then echo "ym 9.9 (test)"; else echo "{\\"home\\": \\"$YELLOWMOON_HOME\\"}"; fi\n')
            binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
            ym = autoexp.Ym(binary, root / "home", "own")
            self.assertEqual(ym.version(), "ym 9.9 (test)")
            self.assertEqual(ym.commands, [])
            proc = ym.run(["info", "x.ARW"])
            self.assertEqual(json.loads(proc.stdout)["home"], str(root / "home"))
            self.assertEqual(len(ym.commands), 1)
            self.assertTrue(ym.commands[0].endswith("info x.ARW"))

    def test_a_binary_that_is_rebuilt_during_the_run_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            binary = root / "ym"
            binary.write_text("#!/bin/sh\necho first\n")
            binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
            ym = autoexp.Ym(binary, root / "home", "own")
            self.assertEqual(ym.run(["info", "x.ARW"]).stdout.strip(), "first")
            # What `cargo build` does: write a new file and move it over the old one.
            new = root / "ym.new"
            new.write_text("#!/bin/sh\necho second, with another size\n")
            new.chmod(new.stat().st_mode | stat.S_IXUSR)
            os.replace(new, binary)
            with self.assertRaises(autoexp.StudyError) as caught:
                ym.run(["render", "x.ARW"])
            self.assertIn("rebuilt or replaced", str(caught.exception))
            self.assertEqual(len(ym.commands), 1)  # the refused command was not run or recorded


class Cache(unittest.TestCase):
    """`analyse(..., reuse=True)` without a `ym`: it can only use a cache that fits the file, the schema and the display."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name) / "work"
        self.arw = Path(self.tmp.name) / "IMG_0001.ARW"
        self.arw.write_bytes(b"not a real raw file")
        self.info, self.exif, self.extras, self.red = fake_inputs()

    def write_cache(self, **key_changes):
        st = self.arw.stat()
        key = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "schema": autoexp.CACHE_SCHEMA, "display": "own", **key_changes}
        (self.work / "cache").mkdir(parents=True, exist_ok=True)
        autoexp.save_cache(self.work / "cache" / "IMG_0001.npz", self.red, {"key": key, "info": self.info, "exif": self.exif, "extras": self.extras})

    def analyse(self, display="own"):
        return autoexp.analyse(None, self.arw, self.work, True, False, display)

    def test_a_cache_that_fits_is_reused(self):
        self.write_cache()
        e = self.analyse()
        self.assertEqual(e["file"], "IMG_0001.ARW")
        self.assertAlmostEqual(e["ev_auto"], 0.9)
        self.assertTrue(math.isfinite(e["g3_scene_ev"]))

    def test_a_cache_of_another_display_schema_or_file_is_not(self):
        self.write_cache()
        with self.assertRaises(autoexp.StudyError):
            self.analyse(display="arri")  # numbers made with our own rendering must not stand for another display
        self.write_cache(schema=autoexp.CACHE_SCHEMA - 1)
        with self.assertRaises(autoexp.StudyError):
            self.analyse()
        self.write_cache(size=1)
        with self.assertRaises(autoexp.StudyError):
            self.analyse()

    def test_a_damaged_cache_is_a_miss_not_a_crash(self):
        self.write_cache()
        (self.work / "cache" / "IMG_0001.npz").write_bytes(b"PK\x03\x04 not a zip")
        with self.assertRaises(autoexp.StudyError):
            self.analyse()
        self.assertTrue(issubclass(zipfile.BadZipFile, Exception))


class Reports(unittest.TestCase):
    def entries(self, with_dead_jpeg: bool = False) -> list[dict]:
        out = []
        for i, (median, ev) in enumerate([(0.05, 1.4), (0.08, 0.9), (0.15, 0.3), (0.1, 0.6)]):
            info, exif, extras, red = fake_inputs(ev, median, seed=i + 1)
            out.append(autoexp._entry(Path(f"/x/IMG_{i}.ARW"), info, exif, extras, red))
        if with_dead_jpeg:  # a black preview: no camera curve, no medians
            info, exif, extras, red = fake_inputs(0.7, 0.08, seed=9)
            red["j_q"] = np.zeros_like(red["j_q"])
            red["j_median"] = 0.0
            out.append(autoexp._entry(Path("/x/IMG_dead.ARW"), info, exif, extras, red))
        return out

    def test_the_report_and_the_files_are_built_from_the_entries(self):
        entries = self.entries()
        summary = autoexp.summarise(entries, 0.5)
        meta = {"label": "test", "ym_version": "ym test", "command_examples": ["ym info x"], "tolerance_ev": 0.5}
        text = autoexp.markdown(meta, entries, [("bad.ARW", "no preview")], summary)
        for heading in ("## Per photo", "## The three comparisons", "## Offset sweep", "## Validation of the study itself", "## Failures"):
            self.assertIn(heading, text)
        self.assertIn("| DRO |", text)
        self.assertIn("IMG_0.ARW", text)
        with tempfile.TemporaryDirectory() as d:
            autoexp.write_csv(Path(d) / "r.csv", entries)
            self.assertEqual(len((Path(d) / "r.csv").read_text().splitlines()), len(entries) + 1)
        json.dumps(autoexp._clean({"files": entries, "summary": summary}))  # serialisable, NaN and private keys handled

    def test_a_black_preview_does_not_break_the_summary_or_the_report(self):
        entries = self.entries(with_dead_jpeg=True)
        summary = autoexp.summarise(entries, 0.5)
        text = autoexp.markdown({"label": "t", "ym_version": "ym test", "command_examples": ["ym"], "tolerance_ev": 0.5}, entries, [], summary)
        self.assertIn("n/a", text)
        self.assertTrue(math.isfinite(summary["key_correlation"]["g3_scene_ev"]))  # the dead photo is left out of the fit


def g3_entry(name: str, g3: float, **kw) -> dict:
    """The few fields of an analysed photo that the gate reads."""
    e = {"file": name, "body": "Sony Test", "iso": 100, "dro": None, "jpeg_median": 0.2, "ev_auto": 1.0 + g3 if g3 is not None and math.isfinite(g3) else 1.0, "camera_gain_ev": 1.0, "g3_scene_ev": g3}
    e.update(kw)
    return e


class GuardCheck(unittest.TestCase):
    """The highlight guard is the only part of `EV_auto` that looks at the scene (`docs/02` Auto exposure, step 5)."""

    @staticmethod
    def e(name: str, ev_auto: float, replica: float, replica_reduction: float) -> dict:
        return {"file": name, "ev_auto": ev_auto, "ev_replica": replica, "guard_reduction_replica": replica_reduction}

    def test_ym_and_the_replica_agree_when_both_leave_the_camera_gain_alone(self):
        g = autoexp.guard_check([self.e("a", 1.11, 1.11, 0.0), self.e("b", 1.11, 1.11, 0.0)])
        self.assertEqual((g["lowered_by_ym"], g["lowered_by_replica"]), ([], []))
        self.assertTrue(g["agrees_with_replica"] and g["never_raised"] and g["at_most_one_stop"])
        self.assertEqual(g["largest_reduction_ev"], 0.0)

    def test_a_guarded_file_is_lowered_by_both_by_the_same_amount_and_at_most_one_stop(self):
        g = autoexp.guard_check([self.e("a", 1.11, 1.11, 0.0), self.e("b", 0.41, 0.40, 0.71)])
        self.assertEqual((g["lowered_by_ym"], g["lowered_by_replica"]), (["b"], ["b"]))
        self.assertAlmostEqual(g["largest_reduction_ev"], 0.70, places=9)
        self.assertTrue(g["agrees_with_replica"] and g["never_raised"] and g["at_most_one_stop"])

    def test_a_disagreement_a_raise_and_more_than_a_stop_are_caught(self):
        # ym lowered a file the replica says is fine
        self.assertFalse(autoexp.guard_check([self.e("a", 0.9, 1.11, 0.0)])["agrees_with_replica"])
        # the replica lowered a file ym did not
        self.assertFalse(autoexp.guard_check([self.e("a", 1.11, 0.9, 0.2)])["agrees_with_replica"])
        # the replica's value is more than 0.03 EV from ym's
        self.assertFalse(autoexp.guard_check([self.e("a", 0.50, 0.40, 0.7)])["agrees_with_replica"])
        # a raise above the camera gain, and a reduction of more than a stop
        self.assertFalse(autoexp.guard_check([self.e("a", 1.5, 1.5, 0.0)])["never_raised"])
        self.assertFalse(autoexp.guard_check([self.e("a", -0.2, -0.2, 1.31)])["at_most_one_stop"])

    def test_the_summary_and_the_report_carry_the_guard_check(self):
        entries = Reports().entries()
        summary = autoexp.summarise(entries, 0.5)
        self.assertIn("guard_check", summary)
        self.assertEqual(summary["camera_gain_ev"], 1.11)
        text = autoexp.markdown({"label": "t", "ym_version": "ym test", "command_examples": ["ym"], "tolerance_ev": 0.5}, entries, [], summary)
        self.assertIn("Highlight guard", text)
        self.assertIn("ev_previous_rule", json.dumps(autoexp._clean({"files": entries})))


class Gate(unittest.TestCase):
    """The M1-08 gate (docs/13, decision 2026-10-13): |G3| <= 0.5 EV for at least 80 % of the files."""

    def gate(self, g3s, failures=(), **kw):
        return autoexp.evaluate_gate([g3_entry(f"f{i}.ARW", g) for i, g in enumerate(g3s)], failures, **kw)

    def test_the_defaults_are_the_decision(self):
        self.assertEqual((autoexp.GATE_TOLERANCE_EV, autoexp.GATE_MIN_FRACTION), (0.5, 0.8))
        g = self.gate([0.0] * 4 + [1.0])
        self.assertEqual((g["tolerance_ev"], g["min_fraction"], g["metric"]), (0.5, 0.8, "g3_scene_ev"))

    def test_the_tolerance_bound_itself_passes_for_either_sign(self):
        self.assertEqual(self.gate([0.5, -0.5, 0.25, -0.25, 0.0])["within"], 5)
        g = self.gate([0.5000001, -0.5000001, 0.0, 0.0, 0.0])
        self.assertEqual(g["within"], 3)
        self.assertFalse(g["passed"])  # 3 of 5 is 60 %
        self.assertEqual(self.gate([0.74, -0.74, 0.0, 0.0, 0.0], tolerance_ev=0.75)["within"], 5)

    def test_the_fraction_is_at_least_not_more_than(self):
        self.assertTrue(self.gate([0.0] * 8 + [2.0] * 2)["passed"])  # 80 % exactly
        self.assertFalse(self.gate([0.0] * 7 + [2.0] * 3)["passed"])
        self.assertTrue(self.gate([0.0] * 4 + [2.0])["passed"])
        self.assertFalse(self.gate([0.0] * 3 + [2.0] * 2)["passed"])
        g = self.gate([0.0] * 28 + [2.0] * 7)  # 0.8 * 35 is not exactly 28 in floating point
        self.assertEqual((g["needed"], g["passed"]), (28, True))
        self.assertTrue(self.gate([0.0] * 37)["passed"])
        self.assertTrue(self.gate([0.0] * 5 + [2.0] * 5, min_fraction=0.5)["passed"])
        self.assertFalse(self.gate([0.0] * 9 + [2.0], min_fraction=1.0)["passed"])

    def test_a_run_without_files_does_not_pass(self):
        g = autoexp.evaluate_gate([], [])
        self.assertEqual((g["files"], g["within"], g["passed"], g["fraction"]), (0, 0, False, 0.0))

    def test_files_that_failed_or_have_no_camera_gain_count_as_misses(self):
        # 4 of 4 analysed files are within, but two more failed: 4 of 6 is below 80 %
        g = self.gate([0.0] * 4, failures=[("a.ARW", "no preview"), ("b.ARW", "ym failed")])
        self.assertEqual((g["files"], g["within"], g["needed"], g["passed"]), (6, 4, 5, False))
        self.assertEqual([u["file"] for u in g["unusable"]], ["a.ARW", "b.ARW"])
        self.assertEqual(g["unusable"][0]["why"], "no preview")
        # a G3 that is not a number (a black preview) is a miss too, and is listed with its name
        g = self.gate([0.0] * 4 + [math.nan, None])
        self.assertEqual((g["files"], g["within"], g["passed"]), (6, 4, False))
        self.assertEqual([u["file"] for u in g["unusable"]], ["f4.ARW", "f5.ARW"])
        self.assertEqual(g["outliers"], [])
        self.assertTrue(self.gate([0.0] * 4 + [math.nan])["passed"])  # 4 of 5

    def test_the_outliers_come_worst_first_with_what_the_review_needs(self):
        entries = [
            g3_entry("ok.ARW", 0.1),
            g3_entry("dark_brick.ARW", 0.74, dro="Auto", jpeg_median=0.05, body="Sony A500", iso=200),
            g3_entry("flash.ARW", -0.72, jpeg_median=0.47),
            g3_entry("tiles.ARW", 0.52),
        ]
        g = autoexp.evaluate_gate(entries)
        self.assertEqual([o["file"] for o in g["outliers"]], ["dark_brick.ARW", "flash.ARW", "tiles.ARW"])
        first = g["outliers"][0]
        self.assertEqual((first["ours"], first["dro"], first["body"], first["iso"], first["jpeg_median"]), ("brighter", "Auto", "Sony A500", 200, 0.05))
        self.assertAlmostEqual(first["g3_ev"], 0.74)
        self.assertEqual(g["outliers"][1]["ours"], "darker")
        self.assertEqual((g["within"], g["files"], g["passed"]), (1, 4, False))

    def test_the_text_and_the_markdown_say_the_statement_the_verdict_and_the_outliers(self):
        g = autoexp.evaluate_gate([g3_entry("ok.ARW", 0.1), g3_entry("far.ARW", 0.9, dro="Auto")], [("bad.ARW", "no preview")])
        text = "\n".join(autoexp.gate_lines(g))
        for part in ("within +-0.5 EV for at least 80 % of the files", "1 of 3 files within (33.3 %; 3 needed): NOT MET", "far.ARW: G3 +0.90 (we are brighter)", "DRO Auto", "counted as a miss: bad.ARW: no preview"):
            self.assertIn(part, text)
        md = "\n".join(autoexp.gate_markdown(g))
        for part in ("## Gate (M1-08, metric G3)", "**Not met: 1 of 3 files within", "| far.ARW |", "M1-H5", "- bad.ARW: no preview"):
            self.assertIn(part, md)
        met = "\n".join(autoexp.gate_lines(autoexp.evaluate_gate([g3_entry("ok.ARW", 0.0)])))
        self.assertIn(": MET", met)
        self.assertIn("no outliers", met)

    def test_the_gate_section_goes_into_the_report_only_when_there_is_a_gate(self):
        entries = Reports().entries()
        meta = {"label": "t", "ym_version": "ym test", "command_examples": ["ym"], "tolerance_ev": 0.5}
        plain = autoexp.markdown(meta, entries, [], autoexp.summarise(entries, 0.5))
        self.assertNotIn("## Gate", plain)
        summary = autoexp.summarise(entries, 0.5)
        summary["gate"] = autoexp.evaluate_gate(entries)
        self.assertIn("## Gate (M1-08, metric G3)", autoexp.markdown(meta, entries, [], summary))
        json.dumps(autoexp._clean({"summary": summary}))


class GateCommand(unittest.TestCase):
    """`main --gate` over made-up photos whose caches are in the work folder (`--reuse`, no `ym` needed): the exit codes and the files."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.photos = self.root / "photos"
        self.photos.mkdir()
        self.work = self.root / "work"

    def library(self, g3s):
        """One made-up ARW per entry of `g3s`, with a cache whose G3 is about that value (the made-up camera gain is 1 EV)."""
        for i, g3 in enumerate(g3s):
            arw = self.photos / f"IMG_{i:04d}.ARW"
            arw.write_bytes(b"not a real raw file %d" % i)
            info, exif, extras, red = fake_inputs(1.0 + g3, 0.08, seed=i + 1)
            st = arw.stat()
            key = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "schema": autoexp.CACHE_SCHEMA, "display": "own"}
            (self.work / "cache").mkdir(parents=True, exist_ok=True)
            autoexp.save_cache(self.work / "cache" / f"{arw.stem}.npz", red, {"key": key, "info": info, "exif": exif, "extras": extras})

    def run_main(self, *flags):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = autoexp.main([str(self.photos), "--reuse", "--ym", str(self.root / "no-ym"), "--out", str(self.work), "--label", "t", *flags])
        return code, out.getvalue(), err.getvalue()

    def test_a_met_gate_exits_0_and_writes_the_gate_into_the_report_and_the_json(self):
        self.library([0.0, 0.2, -0.2, 0.1, -0.1])
        code, out, _ = self.run_main("--gate")
        self.assertEqual(code, 0, out)
        self.assertIn("5 of 5 files within (100.0 %; 4 needed): MET", out)
        doc = json.loads((self.work / "results.json").read_text())
        self.assertTrue(doc["summary"]["gate"]["passed"])
        self.assertIn("## Gate (M1-08, metric G3)", (self.work / "report.md").read_text())

    def test_an_unmet_gate_exits_3_prints_the_outliers_and_still_writes_the_report(self):
        self.library([0.0, 0.1, -0.1, 0.9, -0.8])  # 3 of 5: 60 %
        code, out, _ = self.run_main("--gate", "--tolerance-ev", "0.5", "--min-fraction", "0.8")
        self.assertEqual(code, autoexp.EXIT_GATE_NOT_MET, out)
        self.assertEqual(autoexp.EXIT_GATE_NOT_MET, 3)
        self.assertIn("3 of 5 files within (60.0 %; 4 needed): NOT MET", out)
        self.assertIn("IMG_0003.ARW: G3 +0.90 (we are brighter)", out)
        self.assertIn("IMG_0004.ARW: G3 -0.80 (we are darker)", out)
        self.assertLess(out.index("IMG_0003.ARW: G3"), out.index("IMG_0004.ARW: G3"))  # worst first
        doc = json.loads((self.work / "results.json").read_text())
        gate = doc["summary"]["gate"]
        self.assertFalse(gate["passed"])
        self.assertEqual([o["file"] for o in gate["outliers"]], ["IMG_0003.ARW", "IMG_0004.ARW"])
        report = (self.work / "report.md").read_text()
        self.assertIn("**Not met: 3 of 5 files within", report)
        self.assertIn("| IMG_0003.ARW |", report)

    def test_the_threshold_options_change_the_verdict(self):
        self.library([0.0, 0.1, -0.1, 0.9, -0.8])
        self.assertEqual(self.run_main("--gate", "--min-fraction", "0.6")[0], 0)
        self.assertEqual(self.run_main("--gate", "--tolerance-ev", "1.0")[0], 0)
        self.assertEqual(self.run_main("--gate", "--tolerance", "0.25")[0], autoexp.EXIT_GATE_NOT_MET)  # the older option name still works

    def test_without_gate_the_exit_code_does_not_depend_on_the_errors(self):
        self.library([2.0, -2.0, 1.5, 0.0, 0.0])
        code, out, _ = self.run_main()
        self.assertEqual(code, 0)
        self.assertNotIn("M1-08 gate", out)
        self.assertNotIn("gate", json.loads((self.work / "results.json").read_text())["summary"])
        self.assertNotIn("## Gate", (self.work / "report.md").read_text())

    def test_a_file_that_cannot_be_analysed_counts_against_the_gate(self):
        self.library([0.0, 0.1, -0.1, 0.2])
        (self.photos / "IMG_9999.ARW").write_bytes(b"no cache for this one and no ym to render it")
        code, out, _ = self.run_main("--gate")
        self.assertEqual(code, 1, out)  # 4 of 5 is 80 %: met, but a file failed
        self.assertIn("4 of 5 files within (80.0 %; 4 needed): MET", out)
        self.assertIn("counted as a miss: IMG_9999.ARW", out)
        self.library([0.0, 0.1, -0.1])
        (self.photos / "IMG_0003.ARW").unlink()
        code, out, _ = self.run_main("--gate")
        self.assertEqual(code, autoexp.EXIT_GATE_NOT_MET, out)  # 3 of 4 is 75 %: not met, and that wins over "a file failed"

    def test_bad_thresholds_are_a_usage_error(self):
        self.library([0.0])
        for flags in (("--min-fraction", "0"), ("--min-fraction", "1.5"), ("--min-fraction", "nan"), ("--tolerance-ev", "0"), ("--tolerance-ev", "-1")):
            with self.subTest(flags), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                self.run_main("--gate", *flags)
            self.assertEqual(raised.exception.code, 2)


YM = Path(os.environ.get("YM", REPO / "target" / "debug" / "ym"))
SYNTHETIC = REPO / "testdata" / "synthetic" / "arw" / "chart__D6500__DSC-RX100M4__iso400__rggb__14b__uncompressed14.ARW"


@unittest.skipUnless(YM.exists() and SYNTHETIC.exists(), "needs a ym binary (cargo build -p ym-cli, or YM=...) and tools/fetch/synthetic.sh")
class EndToEnd(unittest.TestCase):
    def test_a_synthetic_scene_recovers_the_known_camera_gain_and_exposure(self):
        # The synthetic JPEG is full size and made from the scene with a known gain (1.11 EV, the camera gain that
        # `EV_auto` is): at the JPEG's own scale the quantile-matched gain is exact to a few hundredths (at 8 x 8 block
        # means it read 0.07 to 0.22 EV low).
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "out"
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                code = autoexp.main([str(SYNTHETIC), "--ym", str(YM), "--out", str(out), "--label", "t"])
            self.assertEqual(code, 0)
            doc = json.loads((out / "results.json").read_text())
            (entry,) = doc["files"]
            self.assertAlmostEqual(entry["camera_gain_ev"], autoexp.SYNTH_GAIN_EV, delta=0.06)
            # EV_auto is the camera gain, so G3 is zero by construction (to the estimator's error)
            self.assertEqual(entry["ev_auto"], 1.11)
            self.assertLess(abs(entry["g3_scene_ev"]), 0.06)
            self.assertEqual([p.name for p in out.glob("ym-home-*")], [])  # the empty home folder is removed
            self.assertEqual(list((out / "renders").iterdir()), [])  # and so are the big renders


if __name__ == "__main__":
    unittest.main()

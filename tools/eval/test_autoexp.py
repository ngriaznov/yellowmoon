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
            ym = autoexp.Ym(binary, root / "home", "aces2")
            self.assertEqual(ym.version(), "ym 9.9 (test)")
            self.assertEqual(ym.commands, [])
            proc = ym.run(["info", "x.ARW"])
            self.assertEqual(json.loads(proc.stdout)["home"], str(root / "home"))
            self.assertEqual(len(ym.commands), 1)
            self.assertTrue(ym.commands[0].endswith("info x.ARW"))


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
        key = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "schema": autoexp.CACHE_SCHEMA, "display": "aces2", **key_changes}
        (self.work / "cache").mkdir(parents=True, exist_ok=True)
        autoexp.save_cache(self.work / "cache" / "IMG_0001.npz", self.red, {"key": key, "info": self.info, "exif": self.exif, "extras": self.extras})

    def analyse(self, display="aces2"):
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
            self.analyse(display="arri")  # numbers made with ACES 2 must not stand for an ARRI display
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


YM = Path(os.environ.get("YM", REPO / "target" / "debug" / "ym"))
SYNTHETIC = REPO / "testdata" / "synthetic" / "arw" / "chart__D6500__DSC-RX100M4__iso400__rggb__14b__uncompressed14.ARW"


@unittest.skipUnless(YM.exists() and SYNTHETIC.exists(), "needs a ym binary (cargo build -p ym-cli, or YM=...) and tools/fetch/synthetic.sh")
class EndToEnd(unittest.TestCase):
    def test_a_synthetic_scene_recovers_the_known_camera_gain_and_exposure(self):
        # The synthetic JPEG is full size and made from the scene with a known gain (0.5575 EV): at the JPEG's own scale
        # the quantile-matched gain is exact to a few hundredths (at 8 x 8 block means it read 0.07 to 0.22 EV low).
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "out"
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                code = autoexp.main([str(SYNTHETIC), "--ym", str(YM), "--out", str(out), "--label", "t"])
            self.assertEqual(code, 0)
            doc = json.loads((out / "results.json").read_text())
            (entry,) = doc["files"]
            self.assertAlmostEqual(entry["camera_gain_ev"], autoexp.SYNTH_GAIN_EV, delta=0.06)
            self.assertLess(abs(entry["ev_auto_minus_truth_ev"]), 0.3)
            self.assertEqual([p.name for p in out.glob("ym-home-*")], [])  # the empty home folder is removed
            self.assertEqual(list((out / "renders").iterdir()), [])  # and so are the big renders


if __name__ == "__main__":
    unittest.main()

"""Checks of the white-scale tools (``python3 -m unittest training.white_scale.test_white_scale``; a few seconds).

The registry data tests are in ``crates/ym-profiles/tests/data_files.rs`` and the pipeline's in
``crates/ym-color/tests/white_scale.rs`` and ``crates/ym-core/tests/white_scale.rs``. This file checks the
Python side: our own SR2 reader (pad cipher against the vectors of ``crates/ym-raw/src/sr2.rs``, the presets of
the owner's photos when they are on disk), the two references and the fit on constructed bodies, and that the
committed ``white_scale.json`` is what ``registry.json`` carries (and, with the owner's photos, what a fresh fit gives).
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from training.common import paths
from training.sensor_map import registry_data as rd
from training.white_scale import fit as fit_mod
from training.white_scale import reference as ref
from training.white_scale import sr2

PHOTOS = paths.TESTDATA / "readme-photos" / "src"
A1_FILES = sorted(PHOTOS.glob("A1M0*.ARW"))
SYNTHETIC = paths.TESTDATA / "synthetic" / "arw"


def levels_for(profile, scale, labels=sr2.KELVIN_PRESETS):
    """The preset multipliers a body would write if its raw were the profile's divided by ``scale`` (green 1024)."""
    out = {}
    for name, kelvin in labels.items():
        w_body = ref.locus_white(profile, kelvin) / np.asarray(scale)
        out[name] = (round(1024 * w_body[1] / w_body[0]), 1024, round(1024 * w_body[1] / w_body[2]))
    return out


class PadCipher(unittest.TestCase):
    """The same vectors as the Rust test: the pad is what the cipher makes of zero data."""

    CASES = (
        (0x00000000, (0xFF189372, 0xAE835A5E, 0xD0FCE299, 0x1FAD6D8C, 0x5FC8E3D6, 0x625D6EA4), 0xAF835A5E, 0x0EF20B9B),
        (0x12345678, (0xBBE9C514, 0x7DDC32A2, 0x4E893991, 0xAFC8F2A0, 0xEAC1F80B, 0xA5298005), 0xE4E91395, 0x582AE3F3),
        (0xDEADBEEF, (0x2EF14285, 0x66865EDD, 0xE7958C33, 0x11306EA4, 0x92C99C6D, 0xEE6C61F2), 0x3260ABBC, 0x62FBD87D),
    )

    def test_the_pad_matches_the_reference_vectors(self):
        for key, first, word127, word128 in self.CASES:
            pad = sr2.apply_pad(bytes(300 * 4), key)
            word = lambda i: int.from_bytes(pad[4 * i : 4 * i + 4], "little")  # noqa: E731
            for i, want in enumerate(first):
                self.assertEqual(word(i), want, f"key {key:#x} word {i}")
            self.assertEqual(word(127), word127)
            self.assertEqual(word(128), word128)

    def test_applying_it_twice_restores_the_data_and_a_tail_is_left_alone(self):
        data = bytes(range(251)) * 3  # not a multiple of four
        once = sr2.apply_pad(data, 0xCAFEF00D)
        self.assertNotEqual(once, data)
        self.assertEqual(once[-1:], data[-1:])
        self.assertEqual(sr2.apply_pad(once, 0xCAFEF00D), data)


class Reader(unittest.TestCase):
    def test_a_synthetic_file_gives_its_as_shot_white_balance(self):
        files = sorted(SYNTHETIC.glob("chart__D6500__ILCE-7RM4__iso400__rggb__14b__lossless-l.ARW"))
        if not files:
            self.skipTest("the synthetic set is not on disk (tools/fetch/synthetic.sh)")
        side = json.loads((paths.TESTDATA / "synthetic" / "scenes" / "chart__D6500__ILCE-7RM4__iso400__rggb__14b.json").read_text())
        levels = sr2.read_levels(files[0])
        self.assertEqual(list(levels["as_shot"]), side["writer"]["sr2"]["wb_rggb"])
        self.assertEqual(sr2.read_model(files[0]), "ILCE-7RM4")
        self.assertNotIn("daylight", levels, "the synthetic files write no presets")

    def test_the_owners_a1_photos_carry_the_same_presets(self):
        if not A1_FILES:
            self.skipTest("the owner's photos are not on disk (tools/fetch/owner-photos.sh)")
        first = sr2.read_levels(A1_FILES[0])
        self.assertEqual(sr2.read_model(A1_FILES[0]), "ILCE-1")
        # What the camera wrote for its Daylight and 6000 K presets (also what the as-shot balance of these files is).
        self.assertEqual(first["daylight"], (2348, 1024, 1714))
        self.assertEqual(first["k6000"], (2530, 1024, 1573))
        self.assertEqual(first["as_shot"], (2348, 1024, 1024, 1714))
        # The Kelvin presets run from warm to cool: the red multiplier rises and the blue one falls.
        order = [first[k] for k in ("k2500", "k3200", "k4500", "k6000", "k8500")]
        self.assertEqual(sorted(r for r, _, _ in order), [r for r, _, _ in order])
        self.assertEqual(sorted((b for _, _, b in order), reverse=True), [b for _, _, b in order])
        for other in A1_FILES[1:]:
            self.assertEqual({k: v for k, v in sr2.read_levels(other).items() if k != "as_shot"}, {k: v for k, v in first.items() if k != "as_shot"})

    def test_a_file_that_is_not_an_arw_is_refused(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.ARW"
            p.write_bytes(b"not a tiff at all, just text")
            with self.assertRaises(sr2.Sr2Error):
                sr2.read_levels(p)


class Fit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = ref.load_profile(rd.DEFAULT_PROFILE)

    def test_a_body_that_is_the_profile_needs_no_scale(self):
        levels = levels_for(self.profile, (1.0, 1.0, 1.0))
        s = ref.fit_presets(self.profile, levels)
        self.assertAlmostEqual(s[0], 1.0, delta=2e-3)
        self.assertAlmostEqual(s[2], 1.0, delta=2e-3)
        fit = ref.judge(self.profile, {**levels, "daylight": levels["k6000"]}, s)
        self.assertLess(fit.mean_angle_after_deg, 0.2)

    def test_the_fit_recovers_a_known_scale_from_the_quantised_presets(self):
        for true in ((0.87, 1.0, 1.09), (1.0, 1.0, 1.2), (0.64, 1.0, 1.19), (1.05, 1.0, 0.8)):
            levels = levels_for(self.profile, true)
            levels["daylight"] = levels["k6000"]
            s = ref.fit_presets(self.profile, levels)
            # w_profile = w_body * s, and the body's raw was the profile's divided by the true scale: s is the true scale.
            self.assertAlmostEqual(s[0], true[0], delta=3e-3, msg=str(true))
            self.assertAlmostEqual(s[2], true[2], delta=3e-3, msg=str(true))
            fit = ref.judge(self.profile, levels, s)
            self.assertLess(fit.mean_angle_after_deg, 0.3)
            self.assertGreater(fit.mean_angle_before_deg, fit.mean_angle_after_deg)

    def test_the_fit_is_least_squares_in_log_chroma(self):
        levels = levels_for(self.profile, (0.9, 1.0, 1.1))
        # A bigger red multiplier on one preset (a smaller red white) moves the log-offset by a fifth of it (five
        # presets, the mean of the offsets).
        base = ref.fit_presets(self.profile, levels)
        bumped = dict(levels)
        r, g, b = bumped["k4500"]
        bumped["k4500"] = (round(r * np.exp(0.05)), g, b)
        moved = ref.fit_presets(self.profile, bumped)
        self.assertAlmostEqual(np.log(moved[0] / base[0]), 0.05 / 5, delta=2e-3)
        self.assertAlmostEqual(moved[2], base[2], delta=1e-3)

    def test_it_needs_at_least_three_kelvin_presets(self):
        levels = levels_for(self.profile, (1.0, 1.0, 1.0))
        with self.assertRaises(ValueError):
            ref.fit_presets(self.profile, {k: v for k, v in list(levels.items())[:2]})

    def test_the_matrix_reference_lands_the_matrix_white_on_the_profiles_d65(self):
        d65_profile = ref.locus_white(self.profile, ref.D65_K)
        # A matrix XYZ -> camera whose D65 white is the profile's white divided by a known scale.
        true = np.array([0.9, 1.0, 1.1])
        target_white = d65_profile / true
        m = np.diag(target_white / (np.eye(3) @ ref.D65_XYZ))  # XYZ -> camera: D65 maps to the target white
        w = ref.matrix_white(m)
        np.testing.assert_allclose(w, target_white / target_white[1], rtol=1e-12)
        s = ref.matrix_scale(self.profile, m)
        self.assertAlmostEqual(s[0], true[0], places=9)
        self.assertAlmostEqual(s[2], true[2], places=9)
        np.testing.assert_allclose(ref.scaled(w, s), d65_profile / d65_profile[1], rtol=1e-12)

    def test_matrix_differences_use_row_normalised_coefficients(self):
        m = np.array([[0.8, -0.2, -0.05], [-0.5, 1.3, 0.25], [-0.1, 0.18, 0.65]])
        self.assertAlmostEqual(ref.matrix_difference(m, 3.7 * m), 0.0, places=12)
        n = m.copy()
        n[0, 0] += 0.05
        self.assertGreater(ref.matrix_difference(m, n), 0.01)


class Registry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = json.loads(fit_mod.RESULT_JSON.read_text())
        cls.registry = json.loads(paths.REGISTRY_JSON.read_text())

    def test_the_registry_carries_the_committed_table(self):
        self.assertEqual(len(self.registry["bodies"]), len(self.table["bodies"]))
        for b in self.registry["bodies"]:
            entry = self.table["bodies"][str(b["model_id"])]
            self.assertEqual(b["white_scale"], entry["white_scale"], b["model"])
            self.assertEqual(b["white_scale_source"], entry["source"], b["model"])

    def test_every_rule_of_the_table_holds(self):
        for b in self.registry["bodies"]:
            src, s = b["white_scale_source"], b["white_scale"]
            self.assertEqual(s[1], 1.0)
            if b["match"] == "exact":
                self.assertEqual((src["kind"], s), ("own_curves", [1.0, 1.0, 1.0]), b["model"])
            elif b["match"] == "sensor_family":
                self.assertEqual((src["kind"], s), ("same_sensor", [1.0, 1.0, 1.0]), b["model"])
            else:
                self.assertIn(src["kind"], ("presets", "sibling", "none"))
                if src["kind"] == "none":
                    self.assertEqual(s, [1.0, 1.0, 1.0])
                else:
                    self.assertNotEqual(s, [1.0, 1.0, 1.0])

    def test_the_a1_meets_the_limits_and_the_matrix_reference_does_not(self):
        a1 = next(b for b in self.registry["bodies"] if b["model_id"] == 384)
        res = a1["white_scale_source"]["residual"]
        self.assertLessEqual(res["mean_preset_angle_after_deg"], fit_mod.MEAN_ANGLE_LIMIT_DEG)
        lo, hi = fit_mod.DAYLIGHT_WINDOW_K
        self.assertTrue(lo <= res["daylight_k_after"] <= hi)
        self.assertTrue(res["matrix_mean_preset_angle_deg"] > fit_mod.MEAN_ANGLE_LIMIT_DEG or not lo <= res["matrix_daylight_k"] <= hi)
        a1m2 = next(b for b in self.registry["bodies"] if b["model_id"] == 400)
        self.assertEqual(a1m2["white_scale"], a1["white_scale"])

    def test_the_siblings_published_matrices_agree_with_the_fitted_body(self):
        m = ref.load_matrices()
        self.assertLessEqual(ref.matrix_difference(m[400], m[384]), rd.SIBLING_LIMIT)

    def test_the_stand_in_profile_of_every_scaled_body_is_the_registrys_default(self):
        for b in self.registry["bodies"]:
            if b["white_scale"] != [1.0, 1.0, 1.0]:
                self.assertEqual(b["profile"], self.registry["default_profile"], "a scale is fitted to its profile's locus")

    def test_a_fresh_fit_on_the_owners_photos_gives_the_committed_table(self):
        if not A1_FILES:
            self.skipTest("the owner's photos are not on disk (tools/fetch/owner-photos.sh)")
        self.assertEqual(fit_mod.fit_all(PHOTOS), self.table)


if __name__ == "__main__":
    unittest.main()

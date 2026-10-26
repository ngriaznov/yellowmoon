"""Checks of the white-scale tools (``python3 -m unittest training.white_scale.test_white_scale``; a few seconds).

The registry data tests are in ``crates/ym-profiles/tests/data_files.rs`` and the pipeline's in
``crates/ym-color/tests/white_scale.rs`` and ``crates/ym-core/tests/white_scale.rs``. This file checks the
Python side: our own SR2 reader (pad cipher against the vectors of ``crates/ym-raw/src/sr2.rs``, the presets of
the owner's photos when they are on disk), the two references, the consistency rule for a body's files and the fit on
constructed bodies, the pins of the fetch scripts, and that the committed ``white_scale.json`` is what ``registry.json``
carries (and, with the pinned files on disk, what a fresh fit gives).
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
#: Folders where the Play Raw files of other work already sit (the fit finds the pinned files by content in any folder).
EXTRA_FOLDERS = (paths.TESTDATA / "lens-photos", paths.TESTDATA / "cropeval" / "playraw")
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


def found(name, sha="0" * 64, set_=fit_mod.PLAY_RAW):
    return fit_mod.Found(fit_mod.Pin(set_, name), Path(name), sha)


#: A stand-in body of the registry (the A7, ILCE-7) and the labels of a camera whose Daylight preset sits at 5300 K.
BODY = rd.BODY_BY_ID[306]
LABELS = {**sr2.KELVIN_PRESETS, "daylight": 5300.0}


def scales(*pairs):
    return np.array([[r, 1.0, b] for r, b in pairs])


class Consistency(unittest.TestCase):
    """The rule of docs/03: files agree within 5 % of the median; one outlier may go from three files on."""

    def test_a_single_file_is_kept_because_nothing_disagrees_with_it(self):
        sel = fit_mod.choose(scales((0.9, 1.1)))
        self.assertEqual((sel.kept, sel.excluded, sel.verdict), ((0,), (), "single"))

    def test_files_within_five_percent_agree(self):
        sel = fit_mod.choose(scales((0.900, 1.100), (0.930, 1.104), (0.870, 1.140)))
        self.assertEqual((sel.kept, sel.excluded, sel.verdict), ((0, 1, 2), (), "agree"))
        sel = fit_mod.choose(scales((0.90, 1.10), (0.97, 1.10)))  # 7 % apart, 3.6 % from their median each
        self.assertEqual(sel.verdict, "agree")

    def test_two_files_that_disagree_keep_nothing(self):
        sel = fit_mod.choose(scales((0.90, 1.10), (1.00, 1.10)))
        self.assertEqual((sel.kept, sel.verdict), ((), "disagree"))

    def test_one_outlier_may_go_from_three_files_on(self):
        for n in (3, 4, 6):
            files = [(0.90, 1.10)] * (n - 1) + [(0.95, 1.10)]
            sel = fit_mod.choose(scales(*files))
            self.assertEqual((sel.kept, sel.excluded, sel.verdict), (tuple(range(n - 1)), (n - 1,), "outlier"), n)

    def test_the_outlier_is_the_file_farthest_from_the_median_in_red_or_blue(self):
        sel = fit_mod.choose(scales((0.90, 1.10), (0.90, 1.20), (0.90, 1.10), (0.90, 1.10)))
        self.assertEqual((sel.excluded, sel.verdict), ((1,), "outlier"))

    def test_two_files_off_is_not_an_outlier(self):
        sel = fit_mod.choose(scales((0.90, 1.10), (0.90, 1.10), (0.95, 1.10), (0.85, 1.10)))
        self.assertEqual((sel.kept, sel.verdict), ((), "disagree"))
        sel = fit_mod.choose(scales((0.80, 1.10), (0.95, 1.10), (1.10, 1.10)))
        self.assertEqual((sel.kept, sel.verdict), ((), "disagree"))

    def test_the_limit_is_five_percent_in_red_and_in_blue(self):
        self.assertEqual(fit_mod.AGREE_LIMIT, 0.05)
        self.assertEqual(fit_mod.choose(scales((1.0, 1.0), (1.04, 1.0), (1.0, 1.04))).verdict, "agree")
        self.assertEqual(fit_mod.choose(scales((1.0, 1.0), (1.12, 1.0))).verdict, "disagree")
        self.assertEqual(fit_mod.choose(scales((1.0, 1.0), (1.0, 1.12))).verdict, "disagree")

    def test_the_limit_is_the_spread_of_the_bodies_with_curves_of_their_own(self):
        # docs/03: the largest unit-to-unit distances of the files of the measured bodies are 4.7 % (A7 III red).
        self.assertGreater(fit_mod.AGREE_LIMIT, 0.047)
        self.assertLess(fit_mod.AGREE_LIMIT, 0.06)

    def test_five_percent_is_seven_to_nine_mired(self):
        # The justification in docs/03: 5 % in red or blue moves a Kelvin reading by 7 to 9 mired (1 % by 1.4 to 1.7).
        profile = ref.load_profile(rd.DEFAULT_PROFILE)
        w = ref.locus_white(profile, 5500.0)
        base = 1e6 / ref.read_cct(profile, w)
        for s, lo, hi in (((1.05, 1.0, 1.0), 6.5, 7.5), ((1.0, 1.0, 1.05), 8.0, 9.5), ((1.01, 1.0, 1.0), 1.2, 1.9), ((1.0, 1.0, 1.01), 1.2, 1.9)):
            moved = abs(1e6 / ref.read_cct(profile, ref.scaled(w, s)) - base)
            self.assertTrue(lo < moved < hi, (s, moved))


class BodyFit(unittest.TestCase):
    """The fit of one body on constructed files whose presets come from a known scale."""

    @classmethod
    def setUpClass(cls):
        cls.profile = ref.load_profile(rd.DEFAULT_PROFILE)

    def items(self, *true_scales):
        return [(found(f"f{i}.ARW"), levels_for(self.profile, s, LABELS)) for i, s in enumerate(true_scales)]

    def test_files_with_one_scale_give_that_scale(self):
        r = fit_mod.evaluate_levels(BODY, self.items((0.88, 1.0, 0.96), (0.88, 1.0, 0.96)))
        self.assertEqual(r.problem, "")
        self.assertEqual(r.selection.verdict, "agree")
        self.assertAlmostEqual(r.scale[0], 0.88, delta=4e-3)
        self.assertAlmostEqual(r.scale[2], 0.96, delta=4e-3)
        self.assertEqual(r.scale[1], 1.0)
        self.assertEqual(r.spread_percent(), (0.0, 0.0))

    def test_the_scale_is_the_median_of_the_files_kept(self):
        r = fit_mod.evaluate_levels(BODY, self.items((0.88, 1.0, 0.96), (0.8812, 1.0, 0.9612), (0.8788, 1.0, 0.9588)))
        self.assertEqual(r.selection.verdict, "agree")
        per_file = sorted(f.scale[0] for f in r.files)
        self.assertEqual(r.scale[0], round(per_file[1], fit_mod.DECIMALS))

    def test_an_outlier_is_left_out_named_and_its_file_is_not_in_the_entry(self):
        r = fit_mod.evaluate_levels(BODY, self.items((0.88, 1.0, 0.96), (0.88, 1.0, 0.96), (0.88, 1.0, 0.96), (0.88, 1.0, 1.01)))
        self.assertEqual(r.selection.verdict, "outlier")
        entry = fit_mod.source_entry(r)
        self.assertEqual([f["name"] for f in entry["source"]["files"]], ["f0.ARW", "f1.ARW", "f2.ARW"])
        self.assertIn("f3.ARW", entry["source"]["note"])
        self.assertIn("outlier", entry["source"]["note"])

    def test_files_that_disagree_give_no_scale(self):
        r = fit_mod.evaluate_levels(BODY, self.items((0.88, 1.0, 0.96), (1.00, 1.0, 0.96)))
        self.assertIsNone(r.scale)
        self.assertIn("differ by up to", r.problem)

    def test_a_scale_that_misses_the_limits_is_refused(self):
        levels = levels_for(self.profile, (0.88, 1.0, 0.96), {**sr2.KELVIN_PRESETS, "daylight": 6800.0})
        r = fit_mod.evaluate_levels(BODY, [(found("f.ARW"), levels)])
        self.assertIsNotNone(r.scale)
        self.assertIn("Daylight preset", r.problem)

    def test_the_entry_has_the_shape_of_the_registry(self):
        matrix = ref.load_matrices()[375]
        for m in (None, matrix):
            r = fit_mod.evaluate_levels(BODY, self.items((0.88, 1.0, 0.96)), m)
            entry = fit_mod.source_entry(r)
            src = entry["source"]
            self.assertEqual(entry["white_scale"], list(r.scale))
            self.assertEqual(src["kind"], "presets")
            self.assertEqual(src["files"], [{"name": "f0.ARW", "sha256": "0" * 64, "set": "play-raw"}])
            res = src["residual"]
            self.assertEqual(res["file_spread_percent"], [0.0, 0.0])
            matrix_keys = {"matrix_scale", "matrix_mean_preset_angle_deg", "matrix_daylight_k"}
            self.assertEqual(matrix_keys <= set(res), m is not None)
            self.assertEqual(matrix_keys & set(res) in (set(), matrix_keys), True, "all or nothing")
            self.assertIn("One file", src["note"])
            self.assertEqual("No published colour matrix" in src["note"], m is None)


class Pins(unittest.TestCase):
    """Every file a shipped scale rests on is pinned by a fetch script of the repository."""

    def test_the_pin_tables_are_read(self):
        # The seven A1 photos of tools/fetch/owner-photos.sha256 (three landscapes, four Provence street and market photos
        # pinned in b90ec8e), each by name and SHA-256.
        pinned = {p.name: p.sha256 for p in fit_mod.owner_pins()}
        self.assertEqual(
            pinned,
            {
                "A1M01454.ARW": "55d20fa10ac3d880c0d1485a34b247b9fe137ea9888d7a39aad05e8f82df0f8e",
                "A1M01739.ARW": "aa170c3792dda23c4beb3ede12a46a2143edb7df06b7321240d368009357f700",
                "A1M01749.ARW": "629b7c36664e1458196a74d45c56e5ec55b1fc056826bfe36e5b4a4fd6fca93e",
                "A1M03159.ARW": "10bf29a430f2c1c678c0f5f62d24f63fcef91e01569bd0f5b42cba24a85e972a",
                "A1M03165.ARW": "ae4f59dc52c3ae14f3ceb4ebfcc87f3c51512bc0f681c804867e28bd9bf23945",
                "A1M03175.ARW": "4bb4f0f54c4037a9cd567f4e84b69edfafc38743554f8ddaaeaa35896ac5ffb7",
                "A1M03185.ARW": "9322b2105e841f7e678a43f74d605cbc4354d114226d59c2671c4e4f5c8a26d9",
            },
        )
        self.assertFalse(hasattr(fit_mod, "real_eval_pins"), "the local-eval files (unclear or non-commercial licences) are not pins")
        self.assertEqual({p.set for p in fit_mod.all_pins()}, {"owner-photos", "play-raw"})
        play = fit_mod.play_raw_pins()
        self.assertEqual(len(play), 32)
        self.assertEqual(len({p.sha256 for p in play}), 32)
        self.assertEqual(len({(p.set, p.name) for p in fit_mod.all_pins()}), len(fit_mod.all_pins()))

    def test_every_file_of_the_registry_is_a_pinned_file_of_a_set(self):
        registry = json.loads(paths.REGISTRY_JSON.read_text())
        pins = {(p.set, p.name): p for p in fit_mod.all_pins()}
        used = 0
        for b in registry["bodies"]:
            for f in b["white_scale_source"]["files"]:
                used += 1
                pin = pins.get((f["set"], f["name"]))
                self.assertIsNotNone(pin, f"{b['model']}: {f['name']} ({f['set']}) is pinned by no fetch script")
                if pin.sha256:
                    self.assertEqual(f["sha256"], pin.sha256, f["name"])
                if pin.model:
                    self.assertEqual(pin.model, b["model"], f["name"])
                self.assertIn(f["set"], fit_mod.SETS)
        # The A1 (7), the A7 (4), the ZV-E10 (3), the NEX-5R (2), seven bodies with one file each, the A6000 (8: the two files
        # whose post is NonCommercial or states no licence are not pinned), the A6300 (5) and the A6500 (2).
        self.assertEqual(used, 38, "the files named by the registry")


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

    def test_every_fitted_body_meets_the_rules_of_docs_03(self):
        lo, hi = fit_mod.DAYLIGHT_WINDOW_K
        fitted = [b for b in self.registry["bodies"] if b["white_scale_source"]["kind"] == "presets"]
        self.assertEqual(sorted(b["model_id"] for b in fitted), [296, 297, 306, 308, 311, 312, 318, 357, 360, 365, 374, 382, 384, 394])
        for b in fitted:
            src = b["white_scale_source"]
            res = src["residual"]
            self.assertLessEqual(max(res["file_spread_percent"]), 100 * fit_mod.AGREE_LIMIT, b["model"])
            self.assertLessEqual(res["mean_preset_angle_after_deg"], fit_mod.MEAN_ANGLE_LIMIT_DEG, b["model"])
            self.assertTrue(lo <= res["daylight_k_after"] <= hi, b["model"])
            self.assertGreaterEqual(len(src["files"]), fit_mod.MIN_FILES)
            self.assertEqual(b["match"], "default")

    def test_no_file_of_a_restricted_licence_is_named_by_the_registry(self):
        # Shipped scales rest only on files whose licence allows any use (docs/03): the Play Raw files whose post states a
        # NonCommercial licence or none are not pinned, and the entries do not name them.
        named = {f["name"] for b in self.registry["bodies"] for f in b["white_scale_source"]["files"]}
        self.assertFalse(named & {"12024__DSC1085.ARW", "35354_DSC00912.ARW", "48272_DSC03902.ARW"})
        pinned = {p.name for p in fit_mod.play_raw_pins()}
        self.assertFalse(pinned & {"12024__DSC1085.ARW", "35354_DSC00912.ARW", "48272_DSC03902.ARW"})

    def test_the_bodies_whose_only_files_are_of_a_licence_that_does_not_allow_any_use_have_no_scale(self):
        # The A300 (a CC BY-NC-SA file), A500, A5000, NEX-5N and NEX-7 (unclear licences) are listed, for their names, with `none`.
        self.assertEqual(sorted(fit_mod.DROPPED_FOR_LICENCE), [261, 274, 288, 289, 313])
        for b in self.registry["bodies"]:
            if b["model_id"] in fit_mod.DROPPED_FOR_LICENCE:
                self.assertEqual((b["white_scale"], b["white_scale_source"]["kind"], b["white_scale_source"]["files"]), ([1.0, 1.0, 1.0], "none", []), b["model"])
                self.assertIn("scale dropped 2026-10-24", b["white_scale_source"]["note"])
        sets = {f["set"] for b in self.registry["bodies"] for f in b["white_scale_source"]["files"]}
        self.assertEqual(sets, {"owner-photos", "play-raw"})

    def test_the_bodies_whose_files_disagree_are_not_in_the_registry(self):
        # The A7 II (docs/03, Consistency of a body's files: 6 files up to 10.4 % in blue) and the A230 (misses the limits).
        ids = {b["model_id"] for b in self.registry["bodies"]}
        self.assertFalse(ids & {340, 265})

    def test_a_fresh_fit_on_the_pinned_files_gives_the_committed_table(self):
        folders = [*fit_mod.DEFAULT_FOLDERS, *EXTRA_FOLDERS]
        missing = fit_mod.scan(folders).missing()
        if missing:
            self.skipTest("pinned files are not on disk: " + ", ".join(p.name for p in missing[:4]) + " ...")
        self.assertEqual(fit_mod.fit_all(folders), self.table)


if __name__ == "__main__":
    unittest.main()

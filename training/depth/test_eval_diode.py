"""Tests of the pair metric of `eval_diode.py` (section 8.8 of `docs/algorithms/06-masks.md`); numpy only, no torch, no network:

    python3 -m unittest training.depth.test_eval_diode        # from the repository root
"""

from __future__ import annotations

import unittest

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

from training.depth import eval_diode as ev


@unittest.skipIf(np is None, "numpy is not installed")
class PairSampling(unittest.TestCase):
    def scene(self):
        rng = np.random.default_rng(5)
        depth = rng.uniform(0.5, 30.0, size=(40, 50))
        depth[:5] = 0.0  # invalid pixels
        depth[10, 10] = np.nan
        return depth

    def test_same_seed_same_pairs_other_seed_other_pairs(self):
        depth = self.scene()
        a1, b1 = ev.sample_pairs(depth, 7)
        a2, b2 = ev.sample_pairs(depth, 7)
        a3, _ = ev.sample_pairs(depth, 8)
        self.assertTrue(np.array_equal(a1, a2) and np.array_equal(b1, b2))
        self.assertFalse(np.array_equal(a1, a3))
        self.assertEqual(len(a1), ev.PAIRS_PER_IMAGE)

    def test_pairs_are_valid_distinct_and_apart_by_the_ratio(self):
        depth = self.scene()
        a, b = ev.sample_pairs(depth, 3)
        flat = depth.reshape(-1)
        self.assertTrue(np.all(a != b))
        for side in (a, b):
            self.assertTrue(np.all(np.isfinite(flat[side]) & (flat[side] > 0)))
        high, low = np.maximum(flat[a], flat[b]), np.minimum(flat[a], flat[b])
        self.assertTrue(np.all(high >= ev.MIN_RATIO * low))

    def test_ratio_rule_is_inclusive_and_ties_are_not_drawn(self):
        # Two depths exactly 1.05 apart qualify; a picture of one depth has no pair at all.
        two = np.array([[1.0, 1.05]])
        a, b = ev.sample_pairs(two, 0, pairs=10)
        self.assertGreater(len(a), 0)
        self.assertEqual(set(zip(np.minimum(a, b), np.maximum(a, b))), {(0, 1)})
        a, _ = ev.sample_pairs(np.full((6, 6), 2.0), 0)
        self.assertEqual(len(a), 0)
        a, _ = ev.sample_pairs(np.array([[1.0, 1.049]]), 0)
        self.assertEqual(len(a), 0)

    def test_draw_limit_and_few_valid_pixels(self):
        depth = self.scene()
        a, _ = ev.sample_pairs(depth, 1, pairs=10_000, draws=300)
        self.assertLessEqual(len(a), 300)
        a, _ = ev.sample_pairs(np.array([[0.0, 3.0]]), 0)
        self.assertEqual(len(a), 0)


@unittest.skipIf(np is None, "numpy is not installed")
class Scoring(unittest.TestCase):
    def test_nearer_must_have_the_larger_d(self):
        # a is nearer (depth 1 against 4)
        self.assertEqual(ev.pair_outcomes([1.0], [4.0], [0.9], [0.2])[0], 1.0)
        self.assertEqual(ev.pair_outcomes([1.0], [4.0], [0.2], [0.9])[0], 0.0)
        # the same with the pair given the other way round
        self.assertEqual(ev.pair_outcomes([4.0], [1.0], [0.2], [0.9])[0], 1.0)

    def test_equal_d_counts_half(self):
        self.assertEqual(ev.pair_outcomes([1.0], [4.0], [0.0], [0.0])[0], 0.5)
        self.assertEqual(ev.image_score(ev.pair_outcomes([1.0, 1.0], [4.0, 4.0], [0.0, 1.0], [0.0, 0.0])), 0.75)

    def test_a_map_that_is_inverse_depth_is_always_right_and_its_reverse_never(self):
        depth = np.random.default_rng(2).uniform(1.0, 20.0, size=(30, 40))
        a, b = ev.sample_pairs(depth, 4)
        self.assertEqual(ev.image_score(ev.image_outcomes(depth, 1.0 / depth, a, b)), 1.0)
        self.assertEqual(ev.image_score(ev.image_outcomes(depth, depth, a, b)), 0.0)
        self.assertEqual(ev.image_score(ev.image_outcomes(depth, np.zeros_like(depth), a, b)), 0.5)
        self.assertIsNone(ev.image_score(np.empty(0)))

    def test_bilinear_is_exact_on_the_same_grid_and_on_flat_areas(self):
        d = np.random.default_rng(1).uniform(size=(6, 8)).astype(np.float32)
        everything = np.arange(48)
        self.assertTrue(np.allclose(ev.bilinear_at(d, everything, 6, 8), d.reshape(-1), atol=0, rtol=0))
        flat = np.full((5, 7), np.float32(0.3))
        got = ev.bilinear_at(flat, np.arange(12 * 14), 12, 14)
        self.assertTrue(np.all(got == np.float64(np.float32(0.3))))

    def test_bilinear_upsamples_a_ramp_linearly(self):
        ramp = np.array([[0.0, 1.0, 2.0, 3.0]])
        got = ev.bilinear_at(ramp, np.arange(8), 1, 8)  # 8 outputs over 4 inputs: centres at -0.25 (clamped), 0.25, 0.75, ...
        self.assertTrue(np.allclose(got, [0.0, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.0]))


@unittest.skipIf(np is None, "numpy is not installed")
class Totals(unittest.TestCase):
    def rows(self, delta):
        return [
            {"split": "indoors", "pairs": 10, "score_rust": 0.90 + delta, "score_torch": 0.90, "outcome_differs": 0.0},
            {"split": "outdoor", "pairs": 10, "score_rust": 0.80 + delta, "score_torch": 0.80, "outcome_differs": 0.01},
            {"split": "outdoor", "pairs": 0, "score_rust": None, "score_torch": None, "outcome_differs": 0.0},
        ]

    def test_gate(self):
        ok = ev.summarise(self.rows(0.004))
        self.assertTrue(ok["gate"]["passed"])
        self.assertEqual(ok["totals"]["all"]["images"], 2)
        self.assertAlmostEqual(ok["totals"]["all"]["difference_points"], 0.4)
        self.assertEqual(ok["totals"]["outdoor"]["images"], 1)
        self.assertFalse(ev.summarise(self.rows(-0.006))["gate"]["passed"])
        self.assertFalse(ev.summarise([])["gate"]["passed"])

    def test_markdown_mentions_the_gate_and_da2k(self):
        report = {"images_scored": 2, "images_found": 2, "every": 1, **ev.summarise(self.rows(0.0))}
        text = ev.markdown(report)
        self.assertIn("passed", text)
        self.assertIn("95.3", text)


class Pins(unittest.TestCase):
    def test_pin_file_has_the_archive_line(self):
        pin = ev.pin_of(ev.PIN_FILE, ev.ARCHIVE_NAME)
        self.assertTrue(pin == ev.PLACEHOLDER or ev.is_hash(pin), pin)


if __name__ == "__main__":
    unittest.main()

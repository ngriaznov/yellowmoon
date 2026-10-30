"""Unit tests of `metrics.py`: python3 -I -m unittest discover -s tools/eval -p 'test_*.py'."""

from __future__ import annotations

import functools
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import metrics as m


class Colour(unittest.TestCase):
    def test_srgb_decoding_matches_the_standard(self):
        self.assertAlmostEqual(float(m.srgb_to_linear(0.5)), 0.2140411, places=6)
        self.assertAlmostEqual(float(m.srgb_to_linear(0.04045)), 0.04045 / 12.92, places=12)
        self.assertEqual(float(m.srgb_to_linear(1.0)), 1.0)
        # code 118 is the 18 % grey of ISO 12232 (the SOS convention)
        self.assertAlmostEqual(float(m.srgb_to_linear(118 / 255)), 0.18, delta=0.002)

    def test_adobe_rgb_is_a_pure_power(self):
        self.assertAlmostEqual(float(m.adobe_rgb_to_linear(0.5)), 0.5**2.19921875, places=12)
        # it is darker than sRGB in the shadows and about equal in the mid-tones
        self.assertLess(float(m.adobe_rgb_to_linear(0.1)), float(m.srgb_to_linear(0.1)))
        self.assertAlmostEqual(float(m.adobe_rgb_to_linear(0.5)), float(m.srgb_to_linear(0.5)), delta=0.005)

    def test_the_awg4_luminance_row_is_the_one_of_ym_color(self):
        np.testing.assert_allclose(m.awg4_y_row(), [0.2545, 0.7815, -0.0360], atol=1e-4)
        self.assertAlmostEqual(float(m.awg4_y_row().sum()), 1.0, places=12)  # white has Y = 1

    def test_awg4_row_agrees_with_colour_science_when_it_is_installed(self):
        try:
            import colour  # type: ignore[import-not-found]
        except ImportError:
            self.skipTest("colour-science is not installed")
        ref = colour.RGB_COLOURSPACES["ARRI Wide Gamut 4"].matrix_RGB_to_XYZ[1]
        np.testing.assert_allclose(m.awg4_y_row(), ref, atol=1e-4)

    def test_jpeg_luminance_of_neutrals_and_primaries(self):
        img = np.zeros((2, 2, 3), dtype=np.uint8)
        img[0, 0] = (255, 255, 255)
        img[0, 1] = (118, 118, 118)
        img[1, 0] = (255, 0, 0)
        y = m.jpeg_luminance(img)
        self.assertAlmostEqual(float(y[0, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(y[0, 1]), 0.18, delta=0.002)
        self.assertAlmostEqual(float(y[1, 0]), 0.2126, places=5)
        self.assertEqual(float(y[1, 1]), 0.0)
        # the Adobe RGB flavour weights differently
        ya = m.jpeg_luminance(img, "adobergb")
        self.assertAlmostEqual(float(ya[1, 0]), m.ADOBE_RGB_Y_ROW[0], places=5)

    def test_jpeg_luminance_refuses_a_wrong_array(self):
        with self.assertRaises(ValueError):
            m.jpeg_luminance(np.zeros((2, 2, 3), dtype=np.float32))
        with self.assertRaises(ValueError):
            m.jpeg_luminance(np.zeros((2, 2), dtype=np.uint8))

    def test_display_luminance_of_a_16_bit_picture(self):
        img = np.zeros((1, 2, 3), dtype=np.uint16)
        img[0, 0] = 65535
        img[0, 1] = (round(0.5 * 65535),) * 3
        y = m.display_luminance_u16(img)
        self.assertAlmostEqual(float(y[0, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(y[0, 1]), 0.2140411, places=4)  # code 32768 of 65535


class Reductions(unittest.TestCase):
    def test_box_mean_averages_blocks_and_drops_the_remainder(self):
        a = np.arange(5 * 7, dtype=np.float32).reshape(5, 7)
        b = m.box_mean(a, 2)
        self.assertEqual(b.shape, (2, 3))
        self.assertAlmostEqual(float(b[0, 0]), float(a[:2, :2].mean()), places=5)
        self.assertAlmostEqual(float(b[1, 2]), float(a[2:4, 4:6].mean()), places=5)
        rgb = np.random.default_rng(1).random((16, 24, 3)).astype(np.float32)
        self.assertEqual(m.box_mean(rgb, 8).shape, (2, 3, 3))
        with self.assertRaises(ValueError):
            m.box_mean(a, 8)

    def test_resize_area_keeps_the_mean(self):
        a = np.random.default_rng(2).random((60, 90)).astype(np.float32)
        b = m.resize_area(a, 30, 20)
        self.assertEqual(b.shape, (20, 30))
        self.assertAlmostEqual(float(b.mean()), float(a.mean()), places=3)
        self.assertTrue(np.allclose(m.resize_area(np.full((40, 40), 0.3, np.float32), 8, 8), 0.3, atol=1e-6))

    def test_centre_weights(self):
        w = m.centre_weights(100, 150)
        self.assertEqual(w.shape, (100, 150))
        self.assertGreater(w[50, 75], 0.999)
        self.assertAlmostEqual(float(w[0, 0]), 0.017, delta=0.002)  # a corner, as docs/02 says
        self.assertTrue(np.allclose(w, w[::-1, ::-1]))


def _numpy_quantile(sorted_y: np.ndarray, p: float) -> float:
    """numpy's default (linear) quantile of a sorted array, written out."""
    pos = p * (len(sorted_y) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_y[lo] + (sorted_y[hi] - sorted_y[lo]) * (pos - lo)


def _flat(h: int, w: int, v: float) -> np.ndarray:
    return np.full((h, w), v)


class Meter(unittest.TestCase):
    """The replica against the cases of `crates/ym-color/tests/exposure.rs`."""

    def test_the_exposure_is_the_cameras_gain_for_scenes_of_any_key(self):
        for y in (0.0, 1e-5, 0.03, 0.09, 0.18, 0.5, 1.5):
            r = m.meter(_flat(20, 30, y))
            self.assertEqual(r.ev_auto, m.CAMERA_GAIN_EV, y)
            self.assertEqual(r.ev_before_guard, m.CAMERA_GAIN_EV)
            self.assertEqual(r.guard_reduction_ev, 0.0)
            # the key statistic is still reported
            self.assertAlmostEqual(r.statistic, max(y, 1e-4), places=12)
        self.assertEqual(m.CAMERA_GAIN_EV, 1.11)

    def test_the_previous_rule_meters_to_middle_grey_for_the_what_if_studies(self):
        for y in (0.03, 0.09, 0.18, 0.5):
            r = m.meter(_flat(20, 30, y), cfg=m.PREVIOUS_RULE)
            self.assertAlmostEqual(r.ev_before_guard, math.log2(0.18 / y), places=9)

    def test_the_exposure_compensation_is_not_a_term_and_the_previous_rule_counts_it_at_half_weight(self):
        base = m.meter(_flat(20, 20, 0.18)).ev_auto
        # the raw already holds the compensation: no term in the new rule, whatever the value
        for ec in (-2.0, 0.7, 2.0, float("nan")):
            self.assertEqual(m.meter(_flat(20, 20, 0.18), ec).ev_auto, base)
        # the previous rule: half weight, clamped
        prev = m.PREVIOUS_RULE
        b = m.meter(_flat(20, 20, 0.18), cfg=prev).ev_auto
        self.assertAlmostEqual(m.meter(_flat(20, 20, 0.18), 2.0, cfg=prev).ev_auto - b, 1.0, places=9)
        self.assertEqual(m.meter(_flat(20, 20, 1e-3), cfg=prev).ev_before_guard, 3.0)
        self.assertEqual(m.meter(_flat(20, 20, 50.0), cfg=prev).ev_before_guard, -3.0)

    def test_a_scene_exposed_two_to_the_ec_brighter_renders_two_to_the_ec_brighter(self):
        # raw * 2^EV: the exposure is the same constant, so the picture follows the raw by exactly 2^EC
        rng = np.random.default_rng(5)
        scene = np.exp(rng.normal(math.log(0.03), 0.6, (24, 36)))
        for ec in (-3.0, -1.0, 0.5, 1.0, 2.5):
            shot = scene * 2.0**ec
            a, b = m.meter(scene, 0.0), m.meter(shot, ec)
            self.assertEqual(a.ev_auto, m.CAMERA_GAIN_EV)
            self.assertEqual(b.ev_auto, a.ev_auto)
            self.assertAlmostEqual(float(np.mean(shot * 2.0**b.ev_auto) / np.mean(scene * 2.0**a.ev_auto)), 2.0**ec, places=9)

    def test_the_highlight_guard_follows_the_half_percent_rule(self):
        def with_bright(count: int, level: float) -> np.ndarray:
            y = _flat(100, 100, 0.18)
            y.ravel()[:count] = level
            return y

        # 0.5 % (50 pixels) is allowed: no change
        self.assertEqual(m.meter(with_bright(50, 6.0)).guard_reduction_ev, 0.0)
        # 51 pixels at 6 (13 after the gain): the 51st brightest must land at 8, log2(8 / 6) = +0.415
        r = m.meter(with_bright(51, 6.0))
        self.assertAlmostEqual(r.ev_auto, math.log2(8.0 / 6.0), places=5)
        self.assertAlmostEqual(r.guard_reduction_ev, m.CAMERA_GAIN_EV - math.log2(8.0 / 6.0), places=5)
        # a highlight that the gain leaves at or below 8 does not count (3.6 * 2^1.11 = 7.8)
        self.assertEqual(m.meter(with_bright(300, 3.6)).guard_reduction_ev, 0.0)
        # very bright: at most one stop
        r = m.meter(with_bright(300, 100.0))
        self.assertAlmostEqual(r.guard_reduction_ev, 1.0, places=9)
        self.assertAlmostEqual(r.ev_auto, m.CAMERA_GAIN_EV - 1.0, places=9)
        # a mask removes the pixels from the count
        masked = np.zeros((100, 100), dtype=bool)
        masked.ravel()[:300] = True
        self.assertEqual(m.meter(with_bright(300, 100.0), 0.0, masked).guard_reduction_ev, 0.0)

    def test_it_agrees_with_a_plain_loop_on_random_scenes(self):
        rng = np.random.default_rng(7)
        for _ in range(4):
            h, w = int(rng.integers(8, 20)), int(rng.integers(8, 30))
            y = np.exp(rng.normal(math.log(0.1), 1.2, (h, w)))
            r = m.meter(y, 0.7)
            flat = y.ravel()
            sorted_y = np.sort(np.maximum(flat, 1e-4))
            q = functools.partial(_numpy_quantile, sorted_y)
            lo, hi = q(0.01), q(0.99)
            sw = swl = 0.0
            for i in range(h):
                for j in range(w):
                    yy = max(y[i, j], 1e-4)
                    dx, dy = j + 0.5 - w / 2.0, i + 0.5 - h / 2.0
                    rr = math.hypot(dx, dy) / (0.5 * math.hypot(w, h))
                    if lo <= yy <= hi:
                        wt = math.exp(-rr * rr / (2 * 0.35**2))
                        sw += wt
                        swl += wt * math.log(yy)
            self.assertAlmostEqual(r.statistic, math.exp(swl / sw), places=9)

    def test_the_centre_counts_more_than_the_edges(self):
        y = _flat(60, 90, 0.05)
        y[20:40, 30:60] = 0.4
        centred = m.meter(y).statistic
        flat = m.meter(y, cfg=m.MeterConfig(centre_sigma=None)).statistic
        self.assertGreater(centred, flat)

    def test_the_first_and_last_percent_are_ignored(self):
        y = _flat(100, 100, 0.18)
        y.ravel()[:50] = 1e-3  # 0.5 % very dark
        y.ravel()[-50:] = 0.9  # 0.5 % bright
        self.assertAlmostEqual(m.meter(y).statistic, 0.18, delta=0.02)

    def test_other_statistics(self):
        y = np.concatenate([np.full(5000, 0.05), np.full(5000, 0.5)]).reshape(100, 100)
        flat = m.MeterConfig(centre_sigma=None)
        from dataclasses import replace

        geo = m.meter(y, cfg=flat).statistic
        self.assertAlmostEqual(geo, math.sqrt(0.05 * 0.5), delta=0.002)
        self.assertAlmostEqual(m.meter(y, cfg=replace(flat, stat="mean")).statistic, 0.275, delta=0.005)
        self.assertAlmostEqual(m.meter(y, cfg=replace(flat, stat="p90")).statistic, 0.5, places=6)
        with self.assertRaises(ValueError):
            m.meter(y, cfg=replace(flat, stat="nonsense"))

    def test_nothing_to_meter_gives_the_same_camera_gain(self):
        r = m.meter(_flat(4, 4, 0.2), 2.0, np.ones((4, 4), dtype=bool))
        self.assertEqual((r.ev_auto, r.unmasked_pixels), (m.CAMERA_GAIN_EV, 0))
        # the previous rule had only the compensation to go on
        r = m.meter(_flat(4, 4, 0.2), 2.0, np.ones((4, 4), dtype=bool), m.PREVIOUS_RULE)
        self.assertEqual((r.ev_auto, r.unmasked_pixels), (1.0, 0))

    def test_the_nominal_grey_fraction(self):
        self.assertAlmostEqual(m.nominal_grey_fraction(), 0.1042, places=4)
        self.assertAlmostEqual(math.log2(0.18 / m.nominal_grey_fraction()), 0.787, places=2)


class Curves(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(11)
        self.x = np.exp(self.rng.normal(math.log(0.08), 0.9, 200_000))

    def test_quantile_matching_recovers_a_global_monotone_curve_between_images_of_different_size(self):
        f = lambda v: np.clip(1.6 * v**1.15, 0, 1)
        # the second image is another (smaller) sample of the same scene: no registration, other resolution
        sub = self.rng.choice(self.x, 30_000, replace=False)
        curve = m.quantile_curve(self.x, f(sub), np.linspace(0.05, 0.95, 91))
        for probe in (0.03, 0.08, 0.2):
            self.assertAlmostEqual(m.curve_at(curve, probe) / float(f(np.array(probe))), 1.0, delta=0.03)

    def test_the_camera_gain_is_read_at_the_anchor(self):
        for gain_ev in (0.3, 0.8, 1.4):
            f = lambda v, g=gain_ev: np.clip(2.0**g * v, 0, 1)
            curve = m.quantile_curve(self.x, f(self.x), np.linspace(0.02, 0.98, 97))
            self.assertAlmostEqual(m.camera_gain_ev(curve), gain_ev, delta=0.02)

    def test_the_camera_gain_with_a_contrast_curve_is_the_gain_at_the_anchor_not_at_the_median(self):
        gain_ev = 0.7
        f = lambda v: np.clip(0.18 * (2.0**gain_ev * v / 0.18) ** 1.25, 0, 1)  # pinned at 0.18 for the gained value
        curve = m.quantile_curve(self.x, f(self.x), np.linspace(0.02, 0.98, 97))
        self.assertAlmostEqual(m.camera_gain_ev(curve), gain_ev, delta=0.03)
        self.assertAlmostEqual(m.log_slope(curve, 0.18 * 2.0**-gain_ev), 1.25, delta=0.03)
        # an identity-curve reading at the median pair is off by the contrast times the distance from the anchor
        med_gain = math.log2(float(np.median(f(self.x))) / float(np.median(self.x)))
        self.assertGreater(abs(med_gain - gain_ev), 0.1)

    def test_the_inverse_refuses_to_extrapolate_far(self):
        curve = m.quantile_curve(self.x, np.clip(self.x, 0, 1), np.linspace(0.02, 0.98, 97))
        self.assertTrue(math.isnan(m.curve_inverse_at(curve, 1e-9)))
        self.assertTrue(math.isnan(m.curve_inverse_at(curve, 1e3)))

    def test_the_log_interpolation_is_exact_for_a_power_law_and_extrapolates_along_it(self):
        xs = np.array([0.01, 0.1, 1.0])
        ys = 2.0 * xs**1.5
        self.assertAlmostEqual(m.curve_at((xs, ys), 0.03), 2.0 * 0.03**1.5, places=10)
        self.assertAlmostEqual(m.curve_at((xs, ys), 3.0), 2.0 * 3.0**1.5, places=10)
        self.assertAlmostEqual(m.curve_at((xs, ys), 0.003), 2.0 * 0.003**1.5, places=10)


class Errors(unittest.TestCase):
    def test_ev_error_is_in_stops_and_signed(self):
        self.assertEqual(m.ev_error(0.36, 0.18), 1.0)
        self.assertEqual(m.ev_error(0.09, 0.18), -1.0)
        self.assertTrue(math.isnan(m.ev_error(0.0, 0.18)))

    def test_pass_rate_counts_finite_errors(self):
        self.assertEqual(m.pass_rate([0.1, -0.5, 0.6, math.nan]), (2, 3))
        self.assertEqual(m.pass_rate([]), (0, 0))

    def test_summary(self):
        s = m.summary([1.0, 2.0, 3.0, math.nan])
        self.assertEqual((s["n"], s["median"], s["mean"], s["min"], s["max"]), (3.0, 2.0, 2.0, 1.0, 3.0))
        self.assertAlmostEqual(s["std"], 1.0)
        self.assertTrue(math.isnan(m.summary([])["median"]))

    def test_shrinking_towards_an_anchor(self):
        self.assertEqual(m.shrunk_ev(1.6, 0.8, 1.0), 1.6)
        self.assertEqual(m.shrunk_ev(1.6, 0.8, 0.0), 0.8)
        self.assertAlmostEqual(m.shrunk_ev(1.6, 0.8, 0.5), 1.2)


if __name__ == "__main__":
    unittest.main()

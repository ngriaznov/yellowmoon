"""Unit tests of `study.py` on synthetic pictures with known tone curves:
python3 -I -m unittest discover -s tools/eval -p 'test_*.py'."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import metrics as m
import study

CAMERA_GAIN_EV = 1.0


def camera(x: np.ndarray, gain_ev: float = CAMERA_GAIN_EV) -> np.ndarray:
    """A camera JPEG engine that is a pure gain (linear light), clipped at white."""
    return np.minimum(1.0, 2.0**gain_ev * x)


def display(x: np.ndarray) -> np.ndarray:
    """A made-up display transform: a Reinhard-like curve (white at infinity) that puts 0.18 at 0.10."""
    k = 1.62
    return x / (x + k) * (0.10 * (0.18 + k) / 0.18)


def reduction(median: float, sigma: float = 0.8, ev_auto: float = 0.9, seed: int = 3, gain_ev: float = CAMERA_GAIN_EV) -> dict:
    rng = np.random.default_rng(seed)
    y0 = np.exp(rng.normal(math.log(median), sigma, (120, 160))).astype(np.float32)
    jpeg = camera(rng.permutation(y0.ravel()), gain_ev).reshape(y0.shape)  # same scene, other pixel order: no registration used
    disp = display(y0 * 2.0**ev_auto)
    return {
        "y0_ds": y0,
        "disp_ds": disp.astype(np.float32),
        "disp_median": float(np.median(disp)),
        "j_q": np.quantile(jpeg, study.QUANTILE_GRID),
        "j_median": float(np.median(jpeg)),
    }


def entry(median: float = 0.08, ev_auto: float = 0.9, ec: float = 0.0, **kw) -> dict:
    red = reduction(median, ev_auto=ev_auto, **kw)
    e = study.file_metrics(red, ev_auto, ec)
    study.attach_curves(e, red)
    return e


class PerFile(unittest.TestCase):
    def test_the_camera_gain_is_recovered_without_registration(self):
        e = entry(0.08)
        self.assertAlmostEqual(e["camera_gain_ev"], CAMERA_GAIN_EV, delta=0.03)
        self.assertAlmostEqual(e["g3_scene_ev"], 0.9 - CAMERA_GAIN_EV, delta=0.03)
        self.assertAlmostEqual(e["camera_contrast_slope"], 1.0, delta=0.05)

    def test_g1_is_the_median_of_the_scene_at_the_exposure_against_the_jpeg_median(self):
        e = entry(0.08)
        y0_median = float(np.median(np.exp(np.random.default_rng(3).normal(math.log(0.08), 0.8, (120, 160)))))
        self.assertAlmostEqual(e["g1_gate_ev"], 0.9 - CAMERA_GAIN_EV, delta=0.03)  # a pure-gain camera: G1 = G3
        self.assertAlmostEqual(e["scene_median_at_auto"], y0_median * 2**0.9, delta=0.003)

    def test_a_display_transform_that_darkens_moves_g2_but_not_g1_or_g3(self):
        e = entry(0.08)
        self.assertLess(e["g2_display_ev"], e["g1_gate_ev"] - 0.5)
        self.assertAlmostEqual(e["display_of_grey"], 0.10, delta=0.003)

    def test_the_contrast_of_the_camera_curve_does_not_move_g3(self):
        # a camera whose engine is gain then contrast about 0.18 (slope 1.3): G3 reads the gain, G1 reads the median
        rng = np.random.default_rng(5)
        y0 = np.exp(rng.normal(math.log(0.03), 0.9, (120, 160))).astype(np.float32)  # a dark scene
        gain = 1.2
        cam = np.clip(0.18 * (2.0**gain * y0 / 0.18) ** 1.3, 0, 1)
        red = {"y0_ds": y0, "disp_ds": display(y0 * 2.0**1.5).astype(np.float32), "disp_median": float(np.median(display(y0 * 2.0**1.5))),
               "j_q": np.quantile(cam, study.QUANTILE_GRID), "j_median": float(np.median(cam))}
        e = study.file_metrics(red, 1.5, 0.0)
        self.assertAlmostEqual(e["camera_gain_ev"], gain, delta=0.04)
        self.assertAlmostEqual(e["camera_contrast_slope"], 1.3, delta=0.05)
        # the identity-curve reading at the median is off by the contrast times the distance from the anchor
        self.assertGreater(abs(e["camera_gain_at_median_ev"] - gain), 0.2)

    def test_statistics_of_the_scene_and_the_flat_geometric_mean(self):
        e = entry(0.08)
        s = e["stats_ev0"]
        self.assertAlmostEqual(s["geomean"], 0.08, delta=0.006)
        self.assertGreater(s["mean"], s["geomean"])
        self.assertLess(s["p75"], s["p90"] + 1e-12)
        self.assertGreater(e["scene_contrast_ev"], 0.2)  # lognormal sigma 0.8 (the arithmetic mean is higher)


class ScaleMatching(unittest.TestCase):
    """The camera curve is read by quantile matching, which needs both pictures at the same spatial scale."""

    def reductions(self):
        rng = np.random.default_rng(9)
        scene = np.exp(rng.normal(math.log(0.08), 0.9, (480, 720))).astype(np.float32)  # independent pixels: averaging narrows it
        at_jpeg = m.box_mean(scene, 3)  # the camera's preview is a downsample of the full picture
        jpeg = camera(at_jpeg)
        y0_ds = m.box_mean(scene, 8)  # the stand-in of the 1/8 preview, a coarser scale than the JPEG's
        disp = display(y0_ds * 2.0**0.9)
        base = {"y0_ds": y0_ds, "disp_ds": disp.astype(np.float32), "disp_median": float(np.median(disp)), "j_q": np.quantile(jpeg, study.QUANTILE_GRID), "j_median": float(np.median(jpeg))}
        return base, np.quantile(at_jpeg, study.QUANTILE_GRID)

    def test_the_table_at_the_jpegs_scale_reads_the_camera_gain_that_the_block_means_get_wrong(self):
        base, y0_q = self.reductions()
        matched = study.file_metrics({**base, "y0_q": y0_q}, 0.9)
        coarse = study.file_metrics(base, 0.9)  # no y0_q: the 8 x 8 means stand in
        self.assertAlmostEqual(matched["camera_gain_ev"], CAMERA_GAIN_EV, delta=0.03)
        self.assertAlmostEqual(matched["g3_scene_ev"], 0.9 - CAMERA_GAIN_EV, delta=0.03)
        self.assertGreater(abs(coarse["camera_gain_ev"] - CAMERA_GAIN_EV), 0.15)

    def test_the_other_metrics_do_not_depend_on_the_table(self):
        base, y0_q = self.reductions()
        a, b = study.file_metrics({**base, "y0_q": y0_q}, 0.9), study.file_metrics(base, 0.9)
        for key in ("g1_gate_ev", "g2_display_ev", "scene_median_at_auto", "display_of_grey"):
            self.assertEqual(a[key], b[key])


class Sweeps(unittest.TestCase):
    def setUp(self):
        self.entries = [entry(0.05, 1.4, seed=1), entry(0.08, 0.9, seed=2), entry(0.15, 0.3, seed=3), entry(0.1, 0.6, seed=4)]

    def test_metrics_at_the_automatic_exposure_is_the_stored_metric(self):
        for e in self.entries:
            at = study.metrics_at(e, e["ev_auto"])
            self.assertAlmostEqual(at["g1"], e["g1_gate_ev"], places=9)
            self.assertAlmostEqual(at["g2"], e["g2_display_ev"], places=9)
            self.assertAlmostEqual(at["g3"], e["g3_scene_ev"], places=9)

    def test_g1_and_g3_move_one_for_one_and_g2_follows_the_display_curve(self):
        e = self.entries[1]
        at = study.metrics_at(e, e["ev_auto"] + 0.5)
        self.assertAlmostEqual(at["g1"], e["g1_gate_ev"] + 0.5, places=9)
        self.assertAlmostEqual(at["g3"], e["g3_scene_ev"] + 0.5, places=9)
        # the display curve is compressive (a Reinhard shoulder), so +0.5 EV in the scene is less than +0.5 EV on the display
        self.assertTrue(0.1 < at["g2"] - e["g2_display_ev"] < 0.5)

    def test_the_offset_sweep_has_the_base_row_and_the_best_offset_centres_the_error(self):
        rows = study.offset_sweep(self.entries, [-0.5, 0.0, 0.5])
        base = next(r for r in rows if r["offset_ev"] == 0.0)
        self.assertAlmostEqual(base["g1_median"], m.summary([e["g1_gate_ev"] for e in self.entries])["median"], places=9)
        for key in ("g1", "g3"):
            off = study.best_offset(self.entries, key)
            errs = [study.metrics_at(e, e["ev_auto"] + off)[key] for e in self.entries]
            self.assertAlmostEqual(float(np.median(errs)), 0.0, places=9)

    def test_the_placement_sweep_scales_the_display_median(self):
        rows = {r["placement"]: r for r in study.placement_sweep(self.entries, (0.10, 0.20))}
        self.assertAlmostEqual(rows[0.20]["median_ev"] - rows[0.10]["median_ev"], 1.0, delta=0.02)
        # at the placement of the render itself the sweep reproduces G2
        self.assertAlmostEqual(rows[0.10]["median_ev"], m.summary([e["g2_display_ev"] for e in self.entries])["median"], delta=0.03)

    def test_shrinking_to_the_anchor_pins_the_exposure(self):
        rows = {r["weight"]: r for r in study.shrink_sweep(self.entries, 1.0, (1.0, 0.0))}
        self.assertAlmostEqual(rows[1.0]["g3_median"], m.summary([e["g3_scene_ev"] for e in self.entries])["median"], places=9)
        # a fixed gain equal to the camera's engine gain: G3 is zero for every photo
        self.assertAlmostEqual(rows[0.0]["g3_median"], 0.0, delta=0.03)
        self.assertAlmostEqual(rows[0.0]["g3_std"], 0.0, delta=0.03)
        self.assertEqual(rows[0.0]["g3_within"], rows[0.0]["g3_evaluated"])

    def test_the_target_moves_the_exposure_by_its_ratio(self):
        # entries whose exposure is neither clamped nor lowered by the guard
        rows = {r["target"]: r for r in study.target_sweep(self.entries, (0.18, 0.36))}
        self.assertAlmostEqual(rows[0.36]["g1_median"] - rows[0.18]["g1_median"], 1.0, delta=0.05)

    def test_the_statistic_sweep_lists_every_statistic_and_the_flat_weights(self):
        rows = study.statistic_sweep(self.entries)
        self.assertEqual([r["statistic"] for r in rows], list(study.METER_STATS) + ["geomean"])
        self.assertTrue(rows[-1]["flat_weights"])

    def test_the_meter_emulation_finds_the_statistic_the_camera_holds_constant(self):
        entries = [
            {"ec_ev": 0.0, "stats_ev0": {"geomean": 0.10, "mean": 0.20, "median": 0.12, "p75": 0.3, "p90": 0.5, "p95": 0.6, "p99": 0.9}},
            {"ec_ev": 0.0, "stats_ev0": {"geomean": 0.10, "mean": 0.12, "median": 0.18, "p75": 0.2, "p90": 0.4, "p95": 0.5, "p99": 0.8}},
            {"ec_ev": 0.0, "stats_ev0": {"geomean": 0.10, "mean": 0.30, "median": 0.09, "p75": 0.25, "p90": 0.6, "p95": 0.7, "p99": 0.95}},
        ]
        rows = study.meter_emulation(entries)
        self.assertEqual(rows[0]["statistic"], "geomean")
        self.assertAlmostEqual(rows[0]["std_ev"], 0.0, places=12)
        self.assertAlmostEqual(rows[0]["mean_ev"], math.log2(0.18 / 0.10), places=12)
        # the exposure compensation is part of the quantity
        entries[0]["ec_ev"] = 1.0
        shifted = {r["statistic"]: r for r in study.meter_emulation(entries)}
        self.assertGreater(shifted["geomean"]["std_ev"], 0.3)


class ContrastVariant(unittest.TestCase):
    def contrast_entry(self, median: float, ev_auto: float, seed: int, gain_ev: float = 1.1, slope: float = 1.3) -> dict:
        rng = np.random.default_rng(seed)
        y0 = np.exp(rng.normal(math.log(median), 0.7, (120, 160))).astype(np.float32)
        cam = np.clip(0.18 * (2.0**gain_ev * y0 / 0.18) ** slope, 0, 1)
        disp = display(y0 * 2.0**ev_auto)
        red = {"y0_ds": y0, "disp_ds": disp.astype(np.float32), "disp_median": float(np.median(disp)), "j_q": np.quantile(cam, study.QUANTILE_GRID), "j_median": float(np.median(cam))}
        e = study.file_metrics(red, ev_auto, 0.0)
        study.attach_curves(e, red)
        return e

    def test_gamma_one_is_the_gate_and_the_true_contrast_recovers_g3(self):
        entries = [self.contrast_entry(0.03, 1.7, 1), self.contrast_entry(0.09, 1.0, 2), self.contrast_entry(0.25, 0.4, 3)]
        for e in entries:
            self.assertAlmostEqual(study.contrast_variant(e, 1.0), e["g1_gate_ev"], places=9)
            self.assertAlmostEqual(study.contrast_variant(e, 1.3), e["g3_scene_ev"], delta=0.06)
        fit = study.fit_contrast(entries)
        self.assertAlmostEqual(fit["gamma"], 1.3, delta=0.1)
        self.assertLess(fit["rms_ev"], fit["g1_rms_ev"])
        rows = study.contrast_sweep(entries, (1.0, 1.3))
        self.assertAlmostEqual(rows[1]["max_abs_diff_from_g3"], 0.0, delta=0.06)
        self.assertGreater(rows[0]["max_abs_diff_from_g3"], 0.2)

    def test_bad_input_is_nan(self):
        e = self.contrast_entry(0.1, 1.0, 4)
        self.assertTrue(math.isnan(study.contrast_variant(e, 0.0)))


if __name__ == "__main__":
    unittest.main()

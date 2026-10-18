"""Labels: the exposure label (G3 with the band rule), the contrast label, the variances and the key target."""

from __future__ import annotations

import copy
import dataclasses
import math
import unittest

import numpy as np

from training.autoset import dataset as ds
from training.autoset import labels as lb
from training.autoset import rows as rw
from training.autoset import synthetic
from training.common import paths  # noqa: F401  (imported for its side effect: the repository root is on the path)

_m = lb._m  # the study's metrics module (tools/eval/metrics.py)


def _base_rows(n_scenes: int = 12, **kwargs) -> tuple[list[rw.Row], list[synthetic.Truth | None]]:
    objs, truth = synthetic.make_dataset(n_scenes, **kwargs)
    read = rw.read_objects(objs)
    rows = [r for r in read.rows if r.is_base]
    tr = [t for o, t in zip(objs, truth, strict=True) if o["aug_k"] == 0.0]
    return rows, tr


class Exposure(unittest.TestCase):
    def test_the_level_is_the_planted_gain_exactly(self) -> None:
        rows, truth = _base_rows(20)
        self.assertGreater(len(rows), 10)
        for r, t in zip(rows, truth, strict=True):
            level, _ = lb.exposure_label(r)
            self.assertAlmostEqual(level, t.level, places=6)

    def test_a_shoulder_above_the_band_does_not_move_the_level(self) -> None:
        rows, truth = _base_rows(12, shoulder=True)
        for r, t in zip(rows, truth, strict=True):
            fit = lb.exposure_fit(r)
            self.assertTrue(fit.band)
            self.assertAlmostEqual(fit.level_ev, t.level, places=6)

    def test_on_a_dense_table_the_band_rule_is_the_study_g3(self) -> None:
        rows, _ = _base_rows(8, shoulder=True)
        inner = lb.INNER
        for r in rows:
            full = _m.quantile_curve_from_tables(r.g3.scene_q[inner], r.g3.jpeg_q[inner])
            study = _m.camera_gain_ev(full)
            self.assertAlmostEqual(lb.exposure_fit(r).level_ev, study, places=12)

    def test_a_sparse_table_falls_back_to_the_full_fit(self) -> None:
        rows, _ = _base_rows(2)
        r = rows[0]
        jpeg = np.where(np.arange(999) < 500, 0.02, 0.9)  # nothing between 0.02 and 0.9: no point in the band
        g3 = rw.G3Tables(scene_q=r.g3.scene_q, jpeg_q=jpeg)
        fit = lb.exposure_fit(dataclasses.replace(r, g3=g3))
        self.assertFalse(fit.band)
        self.assertEqual(fit.points, int(lb.INNER.sum()))

    def test_a_flagged_file_has_four_times_the_base_variance(self) -> None:
        rows, _ = _base_rows(4)
        plain = rows[0]
        flagged = dataclasses.replace(plain, meta={**plain.meta, "dro": "advanced auto"})
        _, v_plain = lb.exposure_label(plain)
        _, v_flagged = lb.exposure_label(flagged)
        self.assertAlmostEqual(v_plain, 0.05**2 + lb.exposure_fit(plain).residual_ev**2, places=9)
        self.assertAlmostEqual(v_flagged, 4 * 0.05**2 + lb.exposure_fit(plain).residual_ev**2, places=9)

    def test_the_residual_of_a_bent_curve_is_added(self) -> None:
        rows, _ = _base_rows(2)
        r = rows[0]
        bend = 1.0 + 0.15 * np.sin(np.linspace(0.0, 6.0, 999))
        bent = dataclasses.replace(r, g3=rw.G3Tables(r.g3.scene_q, r.g3.jpeg_q * bend))
        fit = lb.exposure_fit(bent)
        self.assertTrue(math.isfinite(fit.residual_ev) and fit.residual_ev > 0.01)
        _, var = lb.exposure_label(bent)
        self.assertAlmostEqual(var, 0.05**2 + min(fit.residual_ev**2, 1.0), places=12)

    def test_the_flag_and_normal_rules(self) -> None:
        self.assertFalse(lb.is_flagged({"dro": "off", "hdr": "off", "creative_style": "standard"}))
        self.assertFalse(lb.is_flagged({"dro": None, "hdr": None, "creative_style": None}))
        self.assertTrue(lb.is_flagged({"dro": "standard"}))
        self.assertTrue(lb.is_flagged({"hdr": "on"}))
        self.assertTrue(lb.is_flagged({"creative_style": "vivid"}))
        self.assertTrue(lb.is_normal({"ec": 0.0, "dro": "off"}))
        self.assertFalse(lb.is_normal({"ec": 0.3, "dro": "off"}))
        self.assertFalse(lb.is_normal({"ec": None}))
        self.assertFalse(lb.is_normal({"ec": 0.1, "creative_style": "vivid"}))


class Contrast(unittest.TestCase):
    def test_the_planted_contrast_is_recovered_within_one_unit(self) -> None:
        rows, truth = _base_rows(20)
        for r, t in zip(rows, truth, strict=True):
            c, _ = lb.contrast_label(r)
            self.assertLessEqual(abs(c - t.contrast), 1.0)

    def test_an_exact_table_sits_on_the_variance_floor(self) -> None:
        rows, _ = _base_rows(3)
        for r in rows:
            _, var = lb.contrast_label(r)
            self.assertEqual(var, lb.CONTRAST_VAR_FLOOR)

    def test_a_mismatch_widens_the_variance(self) -> None:
        rows, _ = _base_rows(2)
        r = rows[0]
        cf = r.contrast_fit
        jpeg = cf.jpeg.copy()
        jpeg[5] += 0.2  # the camera's p90 sits well off the curve that any contrast of the grid draws
        r2 = dataclasses.replace(r, contrast_fit=rw.ContrastFit(cf.ours, jpeg))
        fit = lb.contrast_fit(r2)
        self.assertGreater(fit.residual, 0.05)
        self.assertGreater(fit.variance, lb.CONTRAST_VAR_FLOOR)

    def test_the_label_is_clamped_to_forty(self) -> None:
        rows, _ = _base_rows(2)
        r = rows[0]
        cf = r.contrast_fit
        jpeg = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0])  # both spreads are 1: wider than any row of the grid
        r2 = dataclasses.replace(r, contrast_fit=rw.ContrastFit(cf.ours, jpeg))
        c, _ = lb.contrast_label(r2)
        self.assertLessEqual(c, 40.0)
        self.assertGreaterEqual(c, -40.0)
        self.assertEqual(c, 40.0)


class Targets(unittest.TestCase):
    def test_augmentations_inherit_the_file_labels_and_their_level_moves_by_minus_k(self) -> None:
        objs, _ = synthetic.make_dataset(6)
        prep = ds.prepare(rw.read_objects(objs))
        base = ~prep.is_aug
        for i in np.nonzero(prep.is_aug)[0]:
            j = prep.base[i]
            self.assertEqual(prep.key_target[i], prep.key_target[j])
            self.assertEqual(prep.contrast[i], prep.contrast[j])
            self.assertAlmostEqual(prep.level[i], prep.level[j] - prep.aug_k[i], places=12)
        self.assertTrue(prep.key_ok[base].all())
        self.assertTrue(np.allclose(prep.key_target[base] - prep.key[base], prep.level[base]))

    def test_a_label_ev_that_disagrees_is_an_error(self) -> None:
        objs, _ = synthetic.make_dataset(2)
        bad = copy.deepcopy(objs)
        for o in bad:
            if o["aug_k"] == 0.0:
                o["label_ev"] = o["label_ev"] + 1e-6
                break
        with self.assertRaises(ds.LabelMismatch):
            ds.prepare(rw.read_objects(bad))

    def test_a_missing_label_ev_for_tables_is_an_error(self) -> None:
        objs, _ = synthetic.make_dataset(2)
        bad = copy.deepcopy(objs)
        for o in bad:
            if o["aug_k"] == 0.0:
                o["label_ev"] = None
                break
        with self.assertRaises(ds.LabelMismatch):
            ds.prepare(rw.read_objects(bad))


if __name__ == "__main__":
    unittest.main()

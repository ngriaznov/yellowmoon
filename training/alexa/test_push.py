"""The shipped ALEXA curve is the estimate plus the documented hue push (owner decision 2026-10-25).

Checks the committed ``data/alexa/`` files against the numbers the push is defined by, without
re-running the 10-minute estimate (``estimate_alexa`` + ``compare_estimate`` do that):

* the push's reflectances, chosen from the un-pushed recommended variant of ``variants.csv``;
* the shipped curve moves their median Lab hue by the push amount (toward yellow);
* the shipped curve's matrix RMS is within the ensemble's maximum (the stopping rule);
* the shipped curve is 2.2 % RFSE from the un-pushed one, and the ensemble file still holds 24 variants.

Needs the fetched sources (``tools/fetch/spectral.sh``), like the other training checks.

    python3 -m unittest training.alexa.test_push
"""

from __future__ import annotations

import csv
import json
import unittest

import numpy as np

from training.alexa import estimate_alexa as ea
from training.alexa import ssf_core as sc
from training.common import paths


def _shipped() -> np.ndarray:
    data = json.loads(paths.ALEXA_CURVES_JSON.read_text())["spectral_data"]["data"]["main"]
    return np.array([data[str(int(w))] for w in sc.WL], dtype=float)


def _variants() -> list[dict]:
    rows = list(csv.DictReader(open(paths.ALEXA_DIR / "variants.csv", newline="")))
    out: dict[tuple, dict] = {}
    for r in rows:
        key = (r["prior"], int(r["K"]), float(r["lam"]), r["conv"])
        v = out.setdefault(key, {"key": key, "matrix_rms": float(r["matrix_rms"]), "curve": np.zeros((len(sc.WL), 3))})
        v["curve"][:, "RGB".index(r["channel"])] = [float(r[str(int(w))]) for w in sc.WL]
    return list(out.values())


class ShippedPush(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shipped = _shipped()
        cls.variants = _variants()
        rec = ea.RECOMMENDED
        cls.unpushed = next(v for v in cls.variants
                            if v["key"] == (rec["prior"], rec["K"], rec["lam"], rec["conv"]))["curve"]
        cls.ref = ea.push_reference(cls.unpushed)

    def test_the_ensemble_has_24_variants_without_the_push(self):
        self.assertEqual(len(self.variants), 24)

    def test_the_push_selects_47_warm_reflectances(self):
        self.assertEqual(int(self.ref["sel"].sum()), 47)

    def test_the_shipped_curve_moves_their_hue_by_the_push_amount_toward_yellow(self):
        dh, dc = ea.hue_shift(self.shipped, self.ref)
        self.assertAlmostEqual(dh, ea.PUSH["deg"], delta=0.02)
        self.assertGreater(dc, 0.0)  # chroma does not fall
        self.assertAlmostEqual(ea.hue_shift(self.unpushed, self.ref)[0], 0.0, delta=1e-9)

    def test_the_shipped_curve_meets_the_stopping_rule(self):
        limit = max(v["matrix_rms"] for v in self.variants)
        rms = ea.matrix_rms(self.shipped, ea.ARRI)
        self.assertLessEqual(rms, limit)
        self.assertGreater(rms, min(v["matrix_rms"] for v in self.variants))

    def test_the_shipped_curve_is_2_percent_rfse_from_the_unpushed_one(self):
        self.assertAlmostEqual(sc.rfse(self.shipped, self.unpushed), 2.2, delta=0.1)

    def test_the_red_channel_is_lower_at_580_nm(self):
        i = list(sc.WL).index(580.0)
        self.assertLess(self.shipped[i, 0], 0.55)
        self.assertGreater(self.unpushed[i, 0], 0.55)


if __name__ == "__main__":
    unittest.main()

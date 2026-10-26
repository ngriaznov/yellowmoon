"""The shipped ALEXA curve is the estimate plus the documented hue push (owner decisions 2026-10-25 and 2026-10-26).

Checks the committed ``data/alexa/`` files against the numbers the push is defined by, without
re-running the 20-minute estimate (``estimate_alexa`` + ``compare_estimate`` do that):

* the push's reflectances, chosen from the un-pushed recommended variant of ``variants.csv``;
* the shipped curve moves their median Lab hue by the push amount (toward yellow);
* the stopping rule: the shipped push (4.5 degrees) is past the ensemble's maximum matrix RMS by the owner's
  decision of 2026-10-26, which ``push_decisions.beyond_matrix_limit`` records with the matrix RMS it accepted;
  the guard refuses any other push past the limit, and the push of 1.5 degrees (2026-10-25) stays inside it;
* the curve's description and ``target.json`` carry that record;
* the shipped curve is 3.7 % RFSE from the un-pushed one, and the ensemble file still holds 24 variants.

Needs the fetched sources (``tools/fetch/spectral.sh``), like the other training checks.

    python3 -m unittest training.alexa.test_push
"""

from __future__ import annotations

import csv
import json
import unittest

import numpy as np

from training.alexa import estimate_alexa as ea
from training.alexa import push_decisions as pd
from training.alexa import ssf_core as sc
from training.common import paths


def _shipped() -> np.ndarray:
    data = json.loads(paths.ALEXA_CURVES_JSON.read_text())["spectral_data"]["data"]["main"]
    return np.array([data[str(int(w))] for w in sc.WL], dtype=float)


def _variants() -> list[dict]:
    with open(paths.ALEXA_DIR / "variants.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
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
        self.assertAlmostEqual(dh, ea.PUSH["deg"], delta=0.03)
        self.assertEqual(ea.PUSH["deg"], pd.DEFAULT_PUSH_DEG)
        self.assertGreater(dc, 0.0)  # chroma does not fall
        self.assertAlmostEqual(ea.hue_shift(self.unpushed, self.ref)[0], 0.0, delta=1e-9)

    def test_the_shipped_curve_is_past_the_matrix_limit_by_the_owners_decision(self):
        limit = max(v["matrix_rms"] for v in self.variants)
        rms = ea.matrix_rms(self.shipped, ea.ARRI)
        self.assertGreater(rms, limit)
        self.assertGreater(rms, min(v["matrix_rms"] for v in self.variants))
        beyond = pd.decision(ea.PUSH["deg"])["beyond_matrix_limit"]
        self.assertIn("2026-10-26", beyond["decision"])
        self.assertEqual(round(rms, 4), beyond["matrix_rms"])
        self.assertTrue(pd.accepts(ea.PUSH["deg"], rms, limit))

    def test_the_guard_refuses_any_other_push_past_the_limit(self):
        limit = max(v["matrix_rms"] for v in self.variants)
        self.assertTrue(pd.accepts(1.5, 0.0092, limit))       # the push of 2026-10-25 is inside the limit
        self.assertFalse(pd.accepts(3.0, 0.0166, limit))      # no decision
        self.assertFalse(pd.accepts(2.0, 0.0112, limit))
        self.assertFalse(pd.accepts(4.5, 0.0245, limit))      # the decided push, but a larger matrix RMS than accepted
        self.assertFalse(pd.accepts(4.6, 0.0241, limit))      # the accepted RMS, but another push

    def test_the_earlier_estimates_stay_reproducible_behind_flags(self):
        self.assertEqual(pd.decision(1.5)["date"], "2026-10-25")
        self.assertIsNone(pd.decision(1.5)["beyond_matrix_limit"])
        self.assertEqual(pd.DATE_NOPUSH, ea.DATE_NOPUSH)
        text, date = ea.describe_push(1.5, 0.0092, 0.0064, 0.0093, 0.0067)
        self.assertEqual(date, "2026-10-25")
        self.assertIn("(owner decision 2026-10-25)", text)
        self.assertNotIn("beyond_matrix_limit", text)

    def test_the_curves_description_and_the_target_carry_the_decision(self):
        header = json.loads(paths.ALEXA_CURVES_JSON.read_text())["header"]
        rms = ea.matrix_rms(self.shipped, ea.ARRI)
        ens = [v["matrix_rms"] for v in self.variants]
        text, date = ea.describe_push(ea.PUSH["deg"], rms, min(ens), max(ens), ea.matrix_rms(self.unpushed, ea.ARRI))
        self.assertEqual(header["description"], text)
        self.assertEqual(header["document_creation_date"], date)
        self.assertIn("beyond_matrix_limit", header["description"])
        self.assertIn('owner decision 2026-10-26: "4.5°"', header["description"])
        target = json.loads(paths.ALEXA_TARGET_JSON.read_text())["provenance"]
        self.assertIn("beyond_matrix_limit", target["method"])
        self.assertIn("2026-10-26", target["method"])
        self.assertIn(f"{rms:.4f}", target["method"])

    def test_the_shipped_curve_is_3_7_percent_rfse_from_the_unpushed_one(self):
        self.assertAlmostEqual(sc.rfse(self.shipped, self.unpushed), 3.7, delta=0.1)

    def test_the_red_channel_is_lower_at_580_nm(self):
        i = list(sc.WL).index(580.0)
        self.assertLess(self.shipped[i, 0], 0.5)
        self.assertGreater(self.unpushed[i, 0], 0.55)


if __name__ == "__main__":
    unittest.main()

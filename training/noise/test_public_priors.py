"""Tests of the fitting logic of ``public_priors.py`` on synthetic estimate tables.

The corpus is not reachable in V1 builds, so the estimates here are generated from known laws
(``a`` proportional to ISO, ``b`` growing as a power of ISO, a conversion-gain step) with a few
percent of scatter. They test the fitting, selection and output logic, not the sensors.

Run from the repository root::

    python3 -m unittest training.noise.test_public_priors
    python3 -m training.noise.test_public_priors --write-fixture    # refresh crates/ym-noise/tests/data/

The fixture ``crates/ym-noise/tests/data/384-ilce-1.json`` is a *synthetic* body file (not a real
fit of A1 files); the Rust tests load it to check that the schema written here is the schema read there.
"""

from __future__ import annotations

import json
import math
import random
import sys
import unittest
from pathlib import Path

import jsonschema

from training.common import jsonfmt, paths
from training.noise import public_priors as pp

SCHEMA = paths.CAMERAS_DIR / "noise" / "noise-prior.schema.json"
FIXTURE = paths.REPO_ROOT / "crates" / "ym-noise" / "tests" / "data" / "384-ilce-1.json"

TABLE = {
    384: {"model_id": 384, "model": "ILCE-1", "switch_iso": 500, "base_iso": None},
    392: {"model_id": 392, "model": "ILCE-9M3", "switch_iso": 800, "base_iso": 250},
}

# Truth of the synthetic A1: a = A0 ISO / 100 everywhere; b(ISO) = B_LOW (ISO/100)^0.5 below the
# switch and B_HIGH (ISO/500)^1.2 above it (read noise falls at the switch).
A0 = 3.0e-5
B_LOW, B_HIGH = 4.0e-9, 1.5e-9


def true_a(gain: float) -> float:
    return A0 * gain / 100.0


def true_b(gain: float) -> float:
    return B_LOW * (gain / 100.0) ** 0.5 if gain < 500 else B_HIGH * (gain / 500.0) ** 1.2


def make_estimate(rng: random.Random, name: str, model_id: int, model: str, iso: int, base: int = 100, scatter: float = 0.04, coverage: float = 0.8, filtered: bool = False, status: str = "ok") -> dict:
    gain = max(iso, base)
    a = true_a(gain) * math.exp(rng.gauss(0, scatter))
    b = true_b(gain) * math.exp(rng.gauss(0, scatter * 1.5))
    return {
        "file": name, "status": status, "model_id": model_id, "model": model, "iso": iso,
        "decode": "lossless_l", "bit_depth": 14,
        "fitted": {"a": a, "b": b, "delta": [0.0] * 4},
        "model_fit": {"a": a, "b": b},
        "coverage": coverage, "bins_surviving": 18, "texture_fallback": False,
        "confidence": "Data", "residual_rms": 0.06,
        "filtered": {"flagged": filtered, "ratio": 0.6 if filtered else 0.26, "factor": 1.4 if filtered else 1.0, "evaluated": True},
    }


def synthetic_a1(seed: int = 7) -> list[dict]:
    """Twenty files of a synthetic A1: seven ISOs below the 500 switch (50 is an extended ISO), eight above."""
    rng = random.Random(seed)
    isos = [50, 100, 125, 160, 200, 250, 320, 400, 500, 640, 800, 1250, 2000, 3200, 6400, 12800]
    est = []
    for k, iso in enumerate(isos):
        for rep in range(2 if iso in (100, 800) else 1):
            est.append(make_estimate(rng, f"A1_{iso}_{rep}.ARW", 384, "ILCE-1", iso))
    return est


def fixture_document() -> dict:
    estimates = synthetic_a1()
    docs, _ = pp.build(estimates, TABLE, {384: "Sony A1"}, {"corpus": "synthetic estimates (training/noise/test_public_priors.py); not real files"})
    return docs["384-ilce-1.json"]


class FitTests(unittest.TestCase):
    def test_recovers_the_laws_per_regime(self):
        docs, report = pp.build(synthetic_a1(), TABLE, {384: "Sony A1"}, {"corpus": "synthetic"})
        doc = docs["384-ilce-1.json"]
        low, high = doc["regimes"]
        self.assertEqual((low["name"], low["iso_min"], low["iso_max"]), ("low-gain", 100, 499))
        self.assertEqual((high["name"], high["iso_min"], high["iso_max"]), ("high-gain", 500, None))
        # Low regime: gain ISOs 100 (ISO 50 and 100), 125, 160, 200, 250, 320, 400 -> 7 distinct.
        self.assertEqual(low["a"]["n_isos"], 7)
        self.assertAlmostEqual(low["a"]["alpha"], 1.0, delta=0.06)
        self.assertAlmostEqual(low["a"]["beta"], math.log(A0 / 100.0), delta=0.15)
        self.assertLess(low["a"]["rms_log_residual"], 0.08)
        self.assertAlmostEqual(low["b"]["gamma"], 0.5, delta=0.35)
        self.assertEqual(high["a"]["n_isos"], 8)
        self.assertAlmostEqual(high["a"]["alpha"], 1.0, delta=0.06)
        self.assertAlmostEqual(high["b"]["gamma"], 1.2, delta=0.35)
        # The prior sigmas are the docs/06 values.
        self.assertEqual(low["a"]["sigma_log"], 0.25)
        self.assertEqual(low["b"]["sigma_rel"], 0.40)
        self.assertEqual(report["without_enough_data"], [])

    def test_prediction_matches_the_truth(self):
        doc = fixture_document()
        for regime, gains in zip(doc["regimes"], ([100, 160, 320], [640, 1600, 6400])):
            for g in gains:
                a = math.exp(regime["a"]["alpha"] * math.log(g) + regime["a"]["beta"])
                b = math.exp(regime["b"]["gamma"] * math.log(g) + regime["b"]["beta"])
                self.assertAlmostEqual(a / true_a(g), 1.0, delta=0.10, msg=f"a at {g}")
                self.assertAlmostEqual(b / true_b(g), 1.0, delta=0.45, msg=f"b at {g}")

    def test_extended_low_iso_uses_the_base_gain(self):
        self.assertEqual(pp.gain_iso(50, None), 100)
        self.assertEqual(pp.gain_iso(160, None), 160)
        self.assertEqual(pp.gain_iso(100, 250), 250)
        self.assertEqual(pp.regime_of(499, 500), 0)
        self.assertEqual(pp.regime_of(500, 500), 1)
        self.assertEqual(pp.regime_of(25600, None), 0)

    def test_thresholds_on_the_number_of_isos(self):
        rng = random.Random(3)
        # Four ISOs in the low regime: no fit of a (needs 5) but b (needs 3) is written.
        est = [make_estimate(rng, f"f{i}.ARW", 384, "ILCE-1", iso) for i, iso in enumerate([100, 160, 250, 400])]
        docs, report = pp.build(est, TABLE, {}, {})
        regime = docs["384-ilce-1.json"]["regimes"][0]
        self.assertIsNone(regime["a"])
        self.assertIsNotNone(regime["b"])
        # Two ISOs: nothing; the body is listed as lacking data.
        docs, report = pp.build(est[:2], TABLE, {}, {})
        self.assertEqual(docs, {})
        self.assertEqual([b["model"] for b in report["without_enough_data"]], ["ILCE-1"])

    def test_coverage_filter_and_filtered_files_are_excluded(self):
        rng = random.Random(5)
        isos = [100, 125, 160, 200, 250, 320, 400]
        good = [make_estimate(rng, f"g{i}.ARW", 384, "ILCE-1", iso) for i, iso in enumerate(isos)]
        bad = [
            make_estimate(rng, "narrow.ARW", 384, "ILCE-1", 640, coverage=0.45),
            make_estimate(rng, "star_eater.ARW", 384, "ILCE-1", 800, filtered=True),
            make_estimate(rng, "broken.ARW", 384, "ILCE-1", 800, status="error"),
        ]
        bad[2]["error"] = "unsupported"
        docs, report = pp.build(good + bad, TABLE, {}, {})
        reasons = {e["file"]: e["reason"] for e in report["excluded"]}
        self.assertIn("narrow.ARW", reasons)
        self.assertIn("45%", reasons["narrow.ARW"])
        self.assertIn("filtered", reasons["star_eater.ARW"])
        self.assertIn("unsupported", reasons["broken.ARW"])
        listed = {f["file"] for f in docs["384-ilce-1.json"]["provenance"]["files"]}
        self.assertEqual(listed, {f"g{i}.ARW" for i in range(len(isos))})

    def test_implausible_slope_is_not_written(self):
        rng = random.Random(9)
        est = []
        for i, iso in enumerate([100, 125, 160, 200, 250, 320]):
            e = make_estimate(rng, f"f{i}.ARW", 384, "ILCE-1", iso)
            e["fitted"]["a"] = 3e-5  # flat: slope 0 against ISO
            est.append(e)
        docs, report = pp.build(est, TABLE, {}, {})
        regime = docs["384-ilce-1.json"]["regimes"][0] if docs else None
        self.assertTrue(regime is None or regime["a"] is None)
        self.assertTrue(any("slope" in n for info in report["bodies"] + report["without_enough_data"] for n in info["notes"]))

    def test_single_outlier_does_not_move_the_fit(self):
        est = synthetic_a1()
        est[5]["fitted"]["a"] *= 3.0
        docs, _ = pp.build(est, TABLE, {}, {})
        low = docs["384-ilce-1.json"]["regimes"][0]
        self.assertAlmostEqual(low["a"]["alpha"], 1.0, delta=0.1)

    def test_base_iso_of_the_body_is_respected(self):
        rng = random.Random(11)
        # A9 III: base ISO 250, switch 800; ISO 100 and 160 use the gain of 250.
        est = [make_estimate(rng, f"f{i}.ARW", 392, "ILCE-9M3", iso, base=250) for i, iso in enumerate([100, 160, 250, 320, 400, 500, 640])]
        docs, _ = pp.build(est, TABLE, {}, {})
        low = docs["392-ilce-9m3.json"]["regimes"][0]
        self.assertEqual(low["iso_min"], 250)
        self.assertEqual(low["a"]["n_isos"], 5)  # 250 (three files), 320, 400, 500, 640

    def test_file_names(self):
        self.assertEqual(pp.body_file_name(384, "ILCE-1"), "384-ilce-1.json")
        self.assertEqual(pp.body_file_name(375, "ILCE-7RM4"), "375-ilce-7rm4.json")
        self.assertEqual(pp.body_file_name(341, "DSC-RX100M4"), "341-dsc-rx100m4.json")


class OutputTests(unittest.TestCase):
    def test_document_validates_against_the_schema(self):
        schema = json.loads(SCHEMA.read_text())
        jsonschema.validate(fixture_document(), schema)

    def test_output_is_deterministic(self):
        self.assertEqual(jsonfmt.dumps(fixture_document()), jsonfmt.dumps(fixture_document()))

    def test_committed_fixture_is_current(self):
        self.assertTrue(FIXTURE.is_file(), f"{FIXTURE} is missing: run with --write-fixture")
        self.assertEqual(FIXTURE.read_text(), jsonfmt.dumps(fixture_document()), "stale fixture: run with --write-fixture")

    def test_shipped_gain_regimes_validate(self):
        schema = json.loads((paths.CAMERAS_DIR / "noise" / "gain-regimes.schema.json").read_text())
        jsonschema.validate(json.loads(pp.GAIN_REGIMES_JSON.read_text()), schema)
        for row in json.loads(pp.GAIN_REGIMES_JSON.read_text())["bodies"]:
            self.assertEqual(row["source"], "docs/06 table, to be confirmed on samples")

    def test_every_shipped_body_file_validates(self):
        schema = json.loads(SCHEMA.read_text())
        for f in sorted((paths.CAMERAS_DIR / "noise").glob("[0-9]*-*.json")):
            jsonschema.validate(json.loads(f.read_text()), schema)
            model_id = int(f.name.split("-")[0])
            self.assertEqual(json.loads(f.read_text())["model_id"], model_id)


if __name__ == "__main__":
    if "--write-fixture" in sys.argv:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(jsonfmt.dumps(fixture_document()))
        print("wrote", FIXTURE)
    else:
        unittest.main()

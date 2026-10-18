"""Test vectors (section 6.4): their formulas, determinism, the test model and the committed files under crates/."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from training.autoset import export as ex
from training.autoset import model as md
from training.autoset.tests.test_model import random_model


def _recompute(case: dict, model: md.AutosetModel) -> dict:
    """The vector's combined values and rule, computed again from its member outputs with plain loops."""
    outs = np.array(case["members"], dtype=np.float64)  # (members, 4)
    m = len(outs)
    km = sum(outs[i, 0] for i in range(m)) / m
    cm = sum(outs[i, 2] for i in range(m)) / m
    kv = sum(math.exp(min(max(outs[i, 1], -20.0), 20.0)) for i in range(m)) / m
    cv = sum(math.exp(min(max(outs[i, 3], -20.0), 20.0)) for i in range(m)) / m
    kv += sum((outs[i, 0] - km) ** 2 for i in range(m)) / m
    cv += sum((outs[i, 2] - cm) ** 2 for i in range(m)) / m
    w_ev = model.prior_sigma_key**2 / (model.prior_sigma_key**2 + kv)
    ev = min(max(1.11 + w_ev * (km - case["key"] - 1.11), -3.0), 3.0)
    w_c = model.prior_sigma_contrast**2 / (model.prior_sigma_contrast**2 + cv)
    c_raw = min(max(w_c * cm, -model.contrast_limit), model.contrast_limit)
    return {"combined": {"key_mean": km, "key_var": kv, "contrast_mean": cm, "contrast_var": cv},
            "apply": {"ev": ev, "contrast_raw": c_raw, "ev_confidence": w_ev, "contrast_confidence": w_c}}


class Vectors(unittest.TestCase):
    def setUp(self) -> None:
        self.model = random_model(seed=12, members=5)
        self.doc = ex.build_vectors(self.model, model_file="test_model.json")

    def test_the_document_has_thirty_cases_and_the_edges_last(self) -> None:
        self.assertEqual(self.doc["format"], ex.VECTOR_FORMAT)
        self.assertEqual(len(self.doc["cases"]), ex.VECTOR_COUNT)
        self.assertEqual([c["name"] for c in self.doc["cases"][-4:]],
                         ["edge-zeros", "edge-large-positive", "edge-large-negative", "edge-alternating"])
        for case in self.doc["cases"]:
            self.assertEqual(len(case["input"]), md.N_FEATURES)
            self.assertEqual(len(case["members"]), 5)
            self.assertTrue(all(len(row) == 4 for row in case["members"]))

    def test_the_model_is_named_by_its_checksum(self) -> None:
        self.assertEqual(self.doc["model_sha256"], self.model.payload_sha256())
        self.assertEqual(self.doc["constants"]["camera_gain_ev"], 1.11)
        self.assertEqual(self.doc["constants"]["ev_limit"], 3.0)

    def test_combined_and_apply_follow_the_formulas(self) -> None:
        for case in self.doc["cases"]:
            again = _recompute(case, self.model)
            for k, v in again["combined"].items():
                self.assertAlmostEqual(case["combined"][k], v, delta=1e-12 * max(1.0, abs(v)), msg=f"{case['name']} {k}")
            self.assertAlmostEqual(case["apply"]["ev"], again["apply"]["ev"], places=12)
            self.assertAlmostEqual(case["apply"]["contrast_raw"], again["apply"]["contrast_raw"], places=12)
            self.assertAlmostEqual(case["apply"]["ev_confidence"], again["apply"]["ev_confidence"], places=12)
            self.assertAlmostEqual(case["apply"]["contrast_confidence"], again["apply"]["contrast_confidence"], places=12)
            self.assertAlmostEqual(case["apply"]["contrast"], md.round_tenth(again["apply"]["contrast_raw"]), places=12)

    def test_the_stored_members_are_the_forward_pass_of_the_stored_input(self) -> None:
        for case in self.doc["cases"]:
            outs = self.model.member_outputs(np.array(case["input"])[None, :])[:, 0, :]
            np.testing.assert_allclose(np.array(case["members"]), outs, rtol=0, atol=1e-12)

    def test_the_vectors_are_deterministic(self) -> None:
        again = ex.build_vectors(self.model, model_file="test_model.json")
        self.assertEqual(json.dumps(again, sort_keys=True), json.dumps(self.doc, sort_keys=True))

    def test_the_test_model_is_written_and_its_vectors_name_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = ex.write_test_model(Path(tmp))
            loaded = md.AutosetModel.from_json(json.loads(paths["model"].read_text(encoding="utf-8")))
            vectors = json.loads(paths["vectors"].read_text(encoding="utf-8"))
        self.assertEqual(vectors["model_sha256"], loaded.payload_sha256())
        self.assertEqual(vectors["model_file"], "test_model.json")
        self.assertEqual(len(vectors["cases"]), ex.VECTOR_COUNT)
        self.assertEqual(loaded.model_version, ex.TEST_VERSION)


class Committed(unittest.TestCase):
    def test_the_committed_vectors_reproduce_from_the_committed_model(self) -> None:
        model_path = ex.DATA_DIR / "test_model.json"
        vectors_path = ex.DATA_DIR / "vectors.json"
        if not (model_path.exists() and vectors_path.exists()):
            self.skipTest("run python3 -m training.autoset.export --test-model")
        model = md.AutosetModel.from_json(json.loads(model_path.read_text(encoding="utf-8")))
        stored = json.loads(vectors_path.read_text(encoding="utf-8"))
        self.assertEqual(stored["model_sha256"], model.payload_sha256())
        fresh = ex.build_vectors(model, model_file=stored["model_file"])
        self.assertEqual(len(stored["cases"]), len(fresh["cases"]))
        for a, b in zip(stored["cases"], fresh["cases"], strict=True):
            self.assertEqual(a["name"], b["name"])
            np.testing.assert_allclose(a["input"], b["input"], rtol=0, atol=1e-12)
            np.testing.assert_allclose(a["members"], b["members"], rtol=0, atol=1e-9)
            for k in a["combined"]:
                self.assertAlmostEqual(a["combined"][k], b["combined"][k], delta=1e-9 * max(1.0, abs(b["combined"][k])))
            self.assertAlmostEqual(a["apply"]["ev"], b["apply"]["ev"], delta=1e-9)
            self.assertAlmostEqual(a["apply"]["contrast"], b["apply"]["contrast"], delta=1e-9)


if __name__ == "__main__":
    unittest.main()

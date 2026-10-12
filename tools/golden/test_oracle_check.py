"""Unit tests of the verdict logic of ``oracle_check.py`` (``python3 -m unittest discover -s tools/golden``; no files needed)."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("oracle_check", Path(__file__).with_name("oracle_check.py"))
oracle_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle_check)
judge = oracle_check.judge

MOSAIC, OURS, OTHER = "m" * 64, "o" * 64, "x" * 64


class Verdicts(unittest.TestCase):
    def test_an_exact_variant_must_equal_the_mosaic_and_ym_raw(self):
        self.assertTrue(judge("f", "uncompressed14", "ILCE-7RM4", MOSAIC, "bayer", MOSAIC, MOSAIC)["ok"])
        wrong = judge("f", "uncompressed14", "ILCE-7RM4", OTHER, "bayer", MOSAIC, MOSAIC)
        self.assertFalse(wrong["ok"])
        self.assertFalse(wrong["libraw_model_gate"])  # only Lossless L has a model gate
        # LibRaw agrees with the mosaic but ym-raw's decode differs: a ym-raw problem, still a mismatch
        self.assertFalse(judge("f", "packed12", "ILCE-7RM4", MOSAIC, "bayer", OTHER, MOSAIC)["ok"])

    def test_lossy_and_reduced_variants_are_compared_with_ym_raw_only(self):
        for variant in ("craw", "lossless-ms"):
            row = judge("f", variant, "ILCE-7RM4", OURS, "bayer", OURS, MOSAIC)
            self.assertTrue(row["ok"])
            self.assertIsNone(row["libraw_equals_mosaic"])
            self.assertFalse(judge("f", variant, "ILCE-7RM4", OTHER, "bayer", OURS, MOSAIC)["ok"])

    def test_the_lossless_l_model_gate_is_for_models_libraw_does_not_read_that_way(self):
        # LibRaw 0.22.1, src/metadata/tiff.cpp: ILCE-1, ILCE-7* and ILCE-9* only
        for model in ("ILCE-1", "ILCE-7RM4", "ILCE-7SM3", "ILCE-9M3"):
            self.assertTrue(oracle_check.lossless_l_gate(model), model)
            self.assertFalse(judge("f", "lossless-l", model, "0" * 64, "bayer", OURS, MOSAIC)["ok"], model)
        for model in ("ILCE-6400", "DSC-RX100M4"):
            self.assertFalse(oracle_check.lossless_l_gate(model), model)
            gated = judge("f", "lossless-l", model, "0" * 64, "bayer", OURS, MOSAIC)
            self.assertTrue(gated["libraw_model_gate"] and gated["ok"])
            # a body outside LibRaw's list whose file LibRaw does reproduce is an ordinary match, not a gate
            read = judge("f", "lossless-l", model, MOSAIC, "bayer", MOSAIC, MOSAIC)
            self.assertTrue(read["ok"] and not read["libraw_model_gate"])

    def test_a_file_without_a_ym_raw_hash_is_compared_with_the_mosaic_only(self):
        # written with --no-verify: no hash in the index; that must not turn every file into a mismatch
        self.assertTrue(judge("f", "lossless-l", "ILCE-7RM4", MOSAIC, "bayer", None, MOSAIC)["ok"])
        self.assertIsNone(judge("f", "lossless-l", "ILCE-7RM4", MOSAIC, "bayer", None, MOSAIC)["libraw_equals_ours"])
        self.assertTrue(judge("f", "craw", "ILCE-7RM4", OTHER, "bayer", None, MOSAIC)["ok"])  # nothing to compare
        self.assertFalse(judge("f", "lossless-l", "ILCE-7RM4", OTHER, "bayer", None, MOSAIC)["ok"])  # the mosaic is known


if __name__ == "__main__":
    unittest.main()

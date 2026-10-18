"""The model card renders from a metrics document (the end-to-end run is in test_train.py)."""

from __future__ import annotations

import unittest

from training.autoset import model_card as mc


def _stats(n: int = 10) -> dict:
    return {"n": n, "mae": 0.2, "median_abs": 0.15, "p90_abs": 0.4, "max_abs": 0.9, "bias": 0.01,
            "within_0p3": 0.8, "within_0p5": 0.9, "within_0p05": 0.5}


class Card(unittest.TestCase):
    def test_the_card_has_its_sections_and_numbers(self) -> None:
        metrics = {
            "config": {"lr": 0.01, "steps": 3000, "weight_decay": 0.001, "members": 5, "member_seed_base": 1000, "seed": 7},
            "data": {"files": 10, "files_with_key_label": 9, "files_with_contrast_label": 9, "files_normal": 6,
                     "files_flagged": 1},
            "sources": [{"source": "raw.pixls.us", "licence": "CC0-1.0", "files": 10, "scenes": 6, "bodies": ["ILCE-7M4"]}],
            "evaluation": {
                "ev": {"heldout_files": _stats(), "normal_files": _stats(6), "constant_heldout_files": _stats(),
                       "constant_normal_files": _stats(6)},
                "augmentation": {"rescue_vs_label": _stats(), "equivariance": _stats()},
                "contrast": {"model_vs_label": _stats(), "constant_zero_vs_label": _stats()},
                "calibration": {"key_within_1_sd": 0.7},
                "failures": [{"file": "a.ARW", "body": "ILCE-7M4", "level_ev": 1.0, "ev_model": -0.5, "error_ev": -1.5,
                              "flagged": True, "normal": False}],
            },
        }
        text = mc.build(metrics, None, date="2026-10-18", git_sha="0123456789abcdef")
        self.assertIn("# Model card: `autoset`, exposure and contrast", text)
        self.assertIn("git commit `0123456789ab`", text)
        for heading in ("## Purpose", "## Architecture", "## Training data and licences", "## Labels", "## Metrics",
                        "## Failure cases", "## Limits"):
            self.assertIn(heading, text)
        self.assertIn("| raw.pixls.us | CC0-1.0 | 10 | 6 | 1 |", text)
        self.assertIn("| a.ARW | ILCE-7M4 | 1.00 | -0.50 | -1.50 | yes | no |", text)
        self.assertIn("gate 3", text)

    def test_a_card_without_an_evaluation_says_so(self) -> None:
        text = mc.build({}, None, date="2026-10-18", git_sha="abc")
        self.assertIn("No evaluation in the metrics file.", text)
        self.assertIn("None recorded.", text)


if __name__ == "__main__":
    unittest.main()

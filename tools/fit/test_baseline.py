#!/usr/bin/env python3
"""Tests of tools/fit/baseline.py (docs/29, RC-06). Standard library only; run with `python3 -I tools/fit/test_baseline.py`."""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import baseline  # noqa: E402

EDGE = """soft edge before sharpening: rise 2.489 px (spec 2.51)
screen, full size            rise 1.905 px (spec 1.94), overshoot 3.45 % / undershoot 3.38 % (spec 3.0)
screen, s = 0.4              rise 2.274 px (spec 2.30), overshoot 0.43 % / undershoot 0.43 % (spec 0.3)
screen, halo 0, full size    rise 1.869 px (spec NaN), overshoot 4.79 % / undershoot 4.79 % (spec NaN)
print, full size             rise 1.101 px (spec 1.13), overshoot 21.12 % / undershoot 21.12 % (spec 20.7)
test edge_profile_screen_preset ... ok
noise sigma 0.00398 -> 0.00410 (x1.0298) with threshold 1 level(s)
noise sigma 0.00398 -> 0.00548 (x1.3786) with threshold 0 level(s)
"""


class Baseline(unittest.TestCase):
    def test_the_spec_04_edge_output_is_read(self):
        text = baseline.spec04_spec_edge(EDGE)
        self.assertIn("| screen, full size | 2.489 | 1.905 | 3.45 / 3.38 |", text)
        self.assertIn("x1.0298 with the 1-level threshold", text)

    def test_a_missing_edge_line_is_an_error_not_a_blank(self):
        with self.assertRaises(SystemExit):
            baseline.spec04_spec_edge(EDGE.replace("print, full size", "print"))

    def test_a_missing_group_p_is_said_to_be_not_run(self):
        base = {"psnr_db": {"A": 1.0, "A_fit": 1.0, "A_held": 1.0, "B": 1.0, "B_fit": 1.0, "B_held": 1.0,
                            "B2": 1.0, "P": None}}
        self.assertIn("not run: PixelShift200 absent; evaluation data, never committed", baseline.groups(base))

    def test_the_held_out_set_is_marked(self):
        self.assertEqual(baseline.split_tag("McMaster (18)"), "held out")
        self.assertEqual(baseline.split_tag("Kodak (24)"), "fit")


if __name__ == "__main__":
    unittest.main()

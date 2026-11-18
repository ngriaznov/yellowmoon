#!/usr/bin/env python3
"""Tests of tools/fit/search.py (docs/29, RC-05). Standard library only; run with `python3 -I tools/fit/test_search.py`.

A fake example stands in for the Rust ones: it reads `--eval label@name=v,...` and prints the JSON lines
the real examples print, with an objective that is a known parabola, so the driver's grid, descent,
constraint and tie-breaking rules can be checked without building anything.
"""

import contextlib
import io
import json
import os
import stat
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import search  # noqa: E402

FAKE = '''#!/usr/bin/env python3
import json, sys
args = sys.argv[1:]
evals = [a for flag, a in zip(args, args[1:]) if flag == "--eval"]
def line(label, sets):
    x = sets.get("x", 0.3)
    y = sets.get("y", 1.0)
    return {"kind": "eval", "example": "fake", "label": label, "set": sets,
            "psnr_db": {"A": 35.0 - 10.0 * (x - 0.42) ** 2 - (y - 2.0) ** 2},
            "d_psnr_db": {"B": x - 0.3}, "edge": {"overshoot_pct": 1.0 + 4.0 * x}}
print(json.dumps({"kind": "data_note", "note": "never committed", "used": [], "skipped": []}))
print(json.dumps(line("base", {})))
for e in evals:
    label, spec = e.split("@", 1)
    sets = {k: float(v) for k, v in (p.split("=") for p in spec.split(","))}
    print(json.dumps(line(label, sets)))
'''


class Fake:
    """A fake example binary in a temp folder, passed to the driver through FIT_BIN."""

    def __enter__(self):
        self.dir = tempfile.TemporaryDirectory()
        path = os.path.join(self.dir.name, "fake_example")
        with open(path, "w") as f:
            f.write(FAKE)
        os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
        self.old = os.environ.get("FIT_BIN")
        os.environ["FIT_BIN"] = path
        return self

    def __exit__(self, *exc):
        if self.old is None:
            os.environ.pop("FIT_BIN", None)
        else:
            os.environ["FIT_BIN"] = self.old
        self.dir.cleanup()


def run_driver(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        search.main(argv)
    return out.getvalue(), err.getvalue()


def result(text):
    return [json.loads(l) for l in text.splitlines() if l.startswith("{") and '"kind": "search"' in l][-1]


class SearchTests(unittest.TestCase):
    def test_axis_values_are_tidy_and_inclusive(self):
        name, vs = search.axis_values("x=0.1:0.5:0.1")
        self.assertEqual(name, "x")
        self.assertEqual(vs, [0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertEqual(search.axis_values("x=1,2.5")[1], [1.0, 2.5])
        self.assertEqual(search.tidy(1.6 * 0.9), 1.44)

    def test_grid_finds_the_parabola_and_is_deterministic(self):
        argv = ["grid", "fit_demosaic", "--objective", "psnr_db.A", "--axis", "x=0.1:0.9:0.1",
                "--axis", "y=1:3:1"]
        with Fake():
            a, _ = run_driver(argv)
            b, _ = run_driver(argv)
        self.assertEqual(a, b)
        r = result(a)
        self.assertEqual(r["best"], {"x": 0.4, "y": 2.0})
        self.assertTrue(r["taken"])
        self.assertAlmostEqual(r["best_value"], 35.0 - 10.0 * 0.02 ** 2, places=9)

    def test_a_gain_below_the_minimum_is_not_taken(self):
        # The base is x = 0.3 (the fake's default) and y = 1; start the grid on the base's own point.
        argv = ["grid", "fit_demosaic", "--objective", "psnr_db.A", "--axis", "x=0.3:0.31:0.01",
                "--axis", "y=1:1:1", "--min-gain", "5"]
        with Fake():
            r = result(run_driver(argv)[0])
        self.assertFalse(r["taken"])

    def test_a_constraint_rejects_a_candidate(self):
        argv = ["grid", "fit_demosaic", "--objective", "psnr_db.A", "--axis", "x=0.1:0.9:0.1",
                "--axis", "y=2:2:1", "--constraint", "edge.overshoot_pct<=2.0"]
        with Fake():
            r = result(run_driver(argv)[0])
        # overshoot = 1 + 4x <= 2 means x <= 0.25: the best allowed is 0.2, not the optimum 0.4.
        self.assertEqual(r["best"]["x"], 0.2)

    def test_minimize_flips_the_direction(self):
        argv = ["grid", "fit_demosaic", "--objective", "d_psnr_db.B", "--minimize", "--axis", "x=0.1:0.9:0.1",
                "--axis", "y=2:2:1"]
        with Fake():
            r = result(run_driver(argv)[0])
        self.assertEqual(r["best"]["x"], 0.1)

    def test_descent_converges_to_the_optimum_and_is_deterministic(self):
        argv = ["descent", "fit_demosaic", "--objective", "psnr_db.A", "--axis", "x=0.3:0.1",
                "--axis", "y=1:0.5", "--min-step", "0.004", "--min-gain", "1e-9"]
        with Fake():
            a, _ = run_driver(argv)
            b, _ = run_driver(argv)
        self.assertEqual(a, b)
        r = result(a)
        self.assertAlmostEqual(r["best"]["x"], 0.42, delta=0.01)
        self.assertAlmostEqual(r["best"]["y"], 2.0, delta=0.01)

    def test_descent_keeps_to_its_bounds(self):
        argv = ["descent", "fit_demosaic", "--objective", "psnr_db.A", "--axis", "x=0.3:0.1",
                "--axis", "y=2:0.5", "--bounds", "x=0:0.35", "--min-gain", "1e-9"]
        with Fake():
            r = result(run_driver(argv)[0])
        self.assertLessEqual(r["best"]["x"], 0.35)

    def test_the_data_note_is_passed_on(self):
        argv = ["grid", "fit_demosaic", "--objective", "psnr_db.A", "--axis", "x=0.3:0.3:0.1", "--axis", "y=1:1:1"]
        with Fake():
            out, _ = run_driver(argv)
        self.assertIn('"kind": "data_note"', out)
        self.assertIn("never committed", out)

    def test_lookup_reads_dotted_paths(self):
        self.assertEqual(search.lookup({"a": {"b": 2}}, "a.b"), 2)
        self.assertIsNone(search.lookup({"a": {"b": 2}}, "a.c"))
        self.assertIsNone(search.lookup({"a": "x"}, "a"))


if __name__ == "__main__":
    unittest.main()

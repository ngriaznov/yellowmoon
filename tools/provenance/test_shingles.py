"""Tests of the provenance check (stdlib only: ``python3 -I -m unittest discover -s tools/provenance``)."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("shingles", HERE / "shingles.py")
sh = importlib.util.module_from_spec(spec)
sys.modules["shingles"] = sh
spec.loader.exec_module(sh)

RUST = "fn f(a: f64, b: f64) -> f64 { let t = a * 2.5 + b; if t > 1.0 { t.sqrt() } else { t * t } } // comment\n"
RENAMED = "fn g(x: f64, y: f64) -> f64 { let u = x * 7.0 + y; if u > 9.0 { u.sqrt() } else { u * u } }\n"
OTHER = "pub struct P { pub v: Vec<f64> } impl P { pub fn n(&self) -> usize { self.v.len() } }\n"


def units(root: Path, files: dict[str, str], k: int = 8):
    for name, text in files.items():
        (root / name).write_text(text, encoding="utf-8")
    return sh.load([root], k, 16)


class Tokens(unittest.TestCase):
    def test_names_numbers_and_strings_are_normalised_and_comments_dropped(self):
        tokens = [t for t, _ in sh.tokenize('let a = 2.5 + b; // c\n/* d */ let s = "x";', "rust")]
        self.assertEqual(tokens, ["let", "ID", "=", "NUM", "+", "ID", ";", "let", "ID", "=", "STR", ";"])

    def test_python_hash_comment_and_floor_division(self):
        tokens = [t for t, _ in sh.tokenize("x = a // b  # note\n", "python")]
        self.assertEqual(tokens, ["ID", "=", "ID", "//", "ID"])

    def test_a_docstring_is_not_code(self):
        self.assertEqual([t for t, _ in sh.tokenize('"""words"""\nx = 1\n', "python")], ["ID", "=", "NUM"])


class Comparison(unittest.TestCase):
    def test_a_renamed_copy_matches_completely(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            ours = units(Path(a), {"x.rs": RENAMED})
            theirs = units(Path(b), {"y.rs": RUST})
            pairs = sh.compare(ours, theirs, 8)
            self.assertEqual(len(pairs), 1)
            self.assertAlmostEqual(pairs[0]["jaccard"], 1.0)
            self.assertTrue(any(not r["trivial"] for r in pairs[0]["runs"]))

    def test_unrelated_code_shares_nothing(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            ours = units(Path(a), {"x.rs": OTHER})
            theirs = units(Path(b), {"y.rs": RUST})
            self.assertEqual(sh.compare(ours, theirs, 8), [])
            self.assertEqual(sh.corpus_overlap(ours, theirs)["token_share"], 0.0)

    def test_boilerplate_is_trivial(self):
        loop = "fn f() { for i in 0..3 { a[i] = b[i]; } }\n"
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            ours = units(Path(a), {f"x{i}.rs": loop for i in range(3)})
            theirs = units(Path(b), {"y.rs": loop})
            runs = [r for p in sh.compare(ours, theirs, 8) for r in p["runs"]]
            self.assertTrue(runs)
            self.assertTrue(all(r["trivial"] for r in runs))

    def test_the_report_names_commits_and_prints_no_source_text(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            ours = units(Path(a), {"x.rs": RENAMED})
            theirs = units(Path(b), {"y.rs": RUST})
            result = {"other": (sh.corpus_overlap(ours, theirs), sh.compare(ours, theirs, 8))}
            text = sh.render(result, {"other": "abc123"}, 8, 16, None, 5)
            self.assertIn("abc123", text)
            self.assertNotIn("sqrt", text.replace("ID . ID", ""))  # identifiers are normalised away


if __name__ == "__main__":
    unittest.main()

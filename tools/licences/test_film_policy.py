"""Tests of the two policy checks of docs/24 section 6.3 (stdlib only: ``python3 -I -m unittest discover -s tools/licences``)."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

spec = importlib.util.spec_from_file_location("check_film_policy", HERE / "check_film_policy.py")
policy = importlib.util.module_from_spec(spec)
sys.modules["check_film_policy"] = policy
spec.loader.exec_module(policy)


def write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class Training(unittest.TestCase):
    def test_a_training_script_that_reads_the_film_data_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            write(Path(tmp), "training/film/upsample.py", "DATA = 'data/film/cie/CIE_illum_D55.csv'\n")
            problems = policy.problems(Path(tmp))
        self.assertTrue(any("training/film/upsample.py" in p and "data/film/" in p for p in problems), problems)

    def test_the_mention_is_found_in_any_text_file_of_training(self):
        for name in ("notes.md", "run.sh", "cfg.toml", "deep/dir/x.json"):
            with tempfile.TemporaryDirectory() as tmp:
                write(Path(tmp), f"training/{name}", "see ../data/film/spektrafilm/profiles\n")
                problems = policy.problems(Path(tmp))
            self.assertTrue(any(name in p for p in problems), (name, problems))

    def test_training_without_a_mention_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            write(Path(tmp), "training/film/readme.md", "data/models only; data/filmstrip is not it\n")
            problems = policy.problems(Path(tmp))
        self.assertEqual(problems, [])

    def test_binary_files_and_caches_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "training").mkdir()
            (root / "training" / "blob.bin").write_bytes(b"\xff\xfe\x00data/film/")
            write(root, "training/__pycache__/x.py", "data/film/\n")
            self.assertEqual(policy.problems(root), [])


class Name(unittest.TestCase):
    def test_the_name_in_the_interface_or_the_cli_fails_in_any_case(self):
        for rel in ("src/components/Menu.tsx", "src-tauri/src/commands.rs", "crates/ym-cli/src/cli.rs"):
            for word in ("spektrafilm", "Spektrafilm", "SPEKTRAFILM"):
                with tempfile.TemporaryDirectory() as tmp:
                    write(Path(tmp), rel, f'const LABEL = "{word} look";\n')
                    problems = policy.problems(Path(tmp))
                self.assertTrue(any(rel in p and "spektrafilm" in p for p in problems), (rel, word, problems))

    def test_the_credit_lines_and_about_may_use_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root, "src/components/About.tsx", "Film profiles by Andrea Volpato (spektrafilm)\n")
            write(root, "src/components/AboutPanel.test.tsx", "spektrafilm\n")
            write(root, "src/components/film/filmCredit.ts", "export const CREDIT = 'spektrafilm'\n")
            self.assertEqual(policy.problems(root), [])

    def test_the_allowance_is_for_the_interface_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            write(Path(tmp), "crates/ym-cli/src/About.rs", "spektrafilm\n")
            problems = policy.problems(Path(tmp))
        self.assertEqual(len(problems), 1, problems)

    def test_other_folders_are_not_looked_at(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root, "docs/24-film.md", "spektrafilm\n")
            write(root, "crates/ym-film/src/lib.rs", "// spektrafilm is credited in NOTICE\n")
            write(root, "tools/oracle/spektrafilm/run.py", "spektrafilm\n")
            self.assertEqual(policy.problems(root), [])


class Repository(unittest.TestCase):
    def test_the_repository_itself_passes(self):
        self.assertEqual(policy.problems(REPO), [])

    def test_the_command_line_exits_one_on_a_problem(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(policy.main([tmp]), 0)
            write(Path(tmp), "src/App.tsx", "spektrafilm\n")
            self.assertEqual(policy.main([tmp]), 1)


if __name__ == "__main__":
    unittest.main()

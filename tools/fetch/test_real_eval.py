"""Tests for real-eval.sh, the local-evaluation fetch script. Standard library and git only; no network.

Run:  python3 -I -m unittest discover -s tools/fetch -p 'test_*.py' -v

The script is run against a local git repository shaped like the real source (an arw/ folder with ARW files and text dumps, and
other folders that must not be fetched). That proves the script logic: the sparse checkout, the pin, the links named after the
model tag, idempotence, the protection of files it did not write, and the refusal to run in CI. It does not touch
github.com/drewnoakes/metadata-extractor-images.
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Tuple

SCRIPT = Path(__file__).resolve().parent / "real-eval.sh"


def tiff_with_model(model: str, byte_order: str = "<", padding: int = 4000) -> bytes:
    """A minimal TIFF (a header, IFD0 with one Model tag, some padding): all the script reads of an ARW file."""
    text = model.encode("ascii") + b"\0"
    e = byte_order
    head = (b"II" if e == "<" else b"MM") + struct.pack(e + "HI", 42, 8)
    ifd_size = 2 + 12 + 4
    value_offset = 8 + ifd_size
    if len(text) <= 4:
        entry = struct.pack(e + "HHI", 0x0110, 2, len(text)) + text.ljust(4, b"\0")
        tail = b""
    else:
        entry = struct.pack(e + "HHII", 0x0110, 2, len(text), value_offset)
        tail = text
    ifd = struct.pack(e + "H", 1) + entry + struct.pack(e + "I", 0)
    return head + ifd + tail + bytes((i * 7) % 251 for i in range(padding))


def git(cwd: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com",
        "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z", "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z",
    }
    return subprocess.run(["git", *args], cwd=cwd, env=env, check=True, capture_output=True, text=True).stdout.strip()


def git_ok() -> bool:
    if shutil.which("git") is None:
        return False
    out = subprocess.run(["git", "--version"], capture_output=True, text=True).stdout.split()
    try:
        major, minor = (int(x) for x in out[2].split(".")[:2])
    except (IndexError, ValueError):
        return False
    return (major, minor) >= (2, 30)


@unittest.skipUnless(git_ok() and shutil.which("bash") and shutil.which("python3"), "git 2.30+, bash and python3 are needed")
class RealEvalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="ym-real-eval-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.src = self.tmp / "source"
        self.td = self.tmp / "testdata"
        (self.src / "arw" / "metadata" / "java").mkdir(parents=True)
        (self.src / "jpg").mkdir()
        # names as in the real repository: vendor prefix, a marketing name in brackets, mixed case extensions
        (self.src / "arw" / "Sony ILCE-TEST1 (T1).arw").write_bytes(tiff_with_model("ILCE-TEST1"))
        (self.src / "arw" / "Sony NEX-T2.ARW").write_bytes(tiff_with_model("NEX-T2", ">", 9000))
        (self.src / "arw" / "Sony Short.arw").write_bytes(tiff_with_model("A1"))      # a Model value that fits in the entry
        (self.src / "arw" / "metadata" / "java" / "Sony NEX-T2.ARW.txt").write_text("dump\n")
        (self.src / "jpg" / "other.jpg").write_bytes(b"\xff\xd8not fetched")
        git(self.src, "init", "-q", "-b", "main")
        git(self.src, "config", "uploadpack.allowFilter", "true")
        git(self.src, "config", "uploadpack.allowAnySHA1InWant", "true")
        git(self.src, "add", "-A")
        git(self.src, "commit", "-q", "-m", "fixture")
        self.commit = git(self.src, "rev-parse", "HEAD")

    def run_script(self, *args: str, extra: Dict[str, str] | None = None, unset: Tuple[str, ...] = ("CI", "GITHUB_ACTIONS")) -> Tuple[int, str, str]:
        env = {k: v for k, v in os.environ.items() if k not in unset}
        env.update({
            "YM_TESTDATA": str(self.td),
            "YM_REAL_EVAL_REPO": self.src.as_uri(),
            "YM_REAL_EVAL_COMMIT": self.commit,
            "GIT_TERMINAL_PROMPT": "0",
        })
        env.update(extra or {})
        p = subprocess.run(["bash", str(SCRIPT), *args], env=env, capture_output=True, text=True, timeout=120)
        return p.returncode, p.stdout, p.stderr

    @property
    def files(self) -> Path:
        return self.td / "real" / "files"

    def names(self) -> List[str]:
        return sorted(p.name for p in self.files.iterdir()) if self.files.is_dir() else []

    # ---- the fetch ---------------------------------------------------------------------------------------------------
    def test_fetches_only_arw_and_links_by_model(self) -> None:
        code, out, err = self.run_script()
        self.assertEqual(code, 0, out + err)
        self.assertEqual(self.names(), ["A1.ARW", "ILCE-TEST1.ARW", "NEX-T2.ARW"])
        for p in self.files.iterdir():
            self.assertTrue(p.is_symlink() and p.is_file(), p)
        checkout = self.td / "real" / "metadata-extractor-images"
        self.assertTrue((checkout / "arw" / "Sony NEX-T2.ARW").is_file())
        self.assertTrue((checkout / "arw" / "metadata" / "java" / "Sony NEX-T2.ARW.txt").is_file())
        self.assertFalse((checkout / "jpg").exists(), "only arw/ is checked out")
        self.assertEqual(git(checkout, "rev-parse", "HEAD"), self.commit)
        self.assertEqual(len(git(checkout, "log", "--format=%H").splitlines()), 1, "shallow")

    def test_lists_model_and_size_and_warns_about_the_licence(self) -> None:
        code, out, err = self.run_script()
        self.assertEqual(code, 0, out + err)
        for model in ("ILCE-TEST1", "NEX-T2", "A1"):
            self.assertIn(model, out)
        size = (self.src / "arw" / "Sony ILCE-TEST1 (T1).arw").stat().st_size
        self.assertIn(f"{size / 1e6:.1f} MB", out)
        self.assertIn("3 files", out)
        self.assertIn("LOCAL EVALUATION ONLY", err)
        self.assertIn("LICENCE OF THESE FILES IS UNCLEAR", err)
        for word in ("never commit", "never use them in CI", "never redistribute"):
            self.assertIn(word, err)

    def test_second_run_needs_no_network(self) -> None:
        self.assertEqual(self.run_script()[0], 0)
        shutil.rmtree(self.src)   # the source is gone: a second run must not ask for it
        code, out, err = self.run_script()
        self.assertEqual(code, 0, out + err)
        self.assertIn("already at", out)
        self.assertEqual(self.names(), ["A1.ARW", "ILCE-TEST1.ARW", "NEX-T2.ARW"])

    def test_list_works_offline_and_reports_nothing_found_with_exit_3(self) -> None:
        code, out, err = self.run_script("--list")
        self.assertEqual(code, 3, out + err)
        self.assertIn("no ARW files", out)
        self.assertEqual(self.run_script()[0], 0)
        code, out, err = self.run_script("--list", extra={"YM_REAL_EVAL_REPO": "file:///nonexistent"})
        self.assertEqual(code, 0, out + err)
        self.assertIn("metadata-extractor-images (pinned)", out)
        self.assertIn("LOCAL EVALUATION ONLY", err)

    # ---- files the script did not write -------------------------------------------------------------------------------
    def test_a_different_file_is_left_alone_unless_forced(self) -> None:
        self.assertEqual(self.run_script()[0], 0)
        mine = self.files / "NEX-T2.ARW"
        mine.unlink()
        mine.write_bytes(b"my own file")
        code, out, err = self.run_script()
        self.assertEqual(code, 1, out + err)
        self.assertIn("left alone", err)
        self.assertEqual(mine.read_bytes(), b"my own file")
        self.assertIn("not from this script", out)
        code, out, err = self.run_script("--force")
        self.assertEqual(code, 0, out + err)
        self.assertTrue(mine.is_symlink())
        self.assertNotEqual(mine.read_bytes(), b"my own file")

    def test_an_identical_copy_is_accepted_and_not_replaced(self) -> None:
        self.files.mkdir(parents=True)
        copy = self.files / "NEX-T2.ARW"
        shutil.copyfile(self.src / "arw" / "Sony NEX-T2.ARW", copy)
        code, out, err = self.run_script()
        self.assertEqual(code, 0, out + err)
        self.assertFalse(copy.is_symlink(), "an identical regular file stays as it is")
        self.assertEqual(self.names(), ["A1.ARW", "ILCE-TEST1.ARW", "NEX-T2.ARW"])

    def test_a_link_to_a_file_that_is_gone_is_made_again(self) -> None:
        self.files.mkdir(parents=True)
        (self.files / "NEX-T2.ARW").symlink_to("../metadata-extractor-images/arw/not-there.arw")
        code, out, err = self.run_script()
        self.assertEqual(code, 0, out + err)
        self.assertTrue((self.files / "NEX-T2.ARW").is_file(), "the link points at the pinned file now")

    def test_other_files_in_the_folder_are_never_touched(self) -> None:
        self.files.mkdir(parents=True)
        other = self.files / "DSLR-A300.ARW"
        other.write_bytes(b"a file from somewhere else")
        self.assertEqual(self.run_script()[0], 0)
        self.assertEqual(other.read_bytes(), b"a file from somewhere else")
        code, out, _ = self.run_script("--force")
        self.assertEqual(code, 0)
        self.assertEqual(other.read_bytes(), b"a file from somewhere else")
        self.assertIn("DSLR-A300.ARW", out)

    # ---- refusals and failures ----------------------------------------------------------------------------------------
    def test_refuses_to_run_in_ci(self) -> None:
        for var in ("CI", "GITHUB_ACTIONS"):
            for args in ((), ("--list",)):
                code, out, err = self.run_script(*args, extra={var: "true"})
                self.assertEqual(code, 2, (var, args, out, err))
                self.assertIn("refusing to run in CI", err)
        self.assertFalse(self.td.exists(), "nothing was fetched or created")

    def test_an_unreachable_source_exits_4_and_leaves_nothing_behind(self) -> None:
        code, out, err = self.run_script(extra={"YM_REAL_EVAL_REPO": (self.tmp / "missing").as_uri()})
        self.assertEqual(code, 4, out + err)
        self.assertIn("cannot fetch", err)
        self.assertFalse((self.td / "real" / "metadata-extractor-images").exists())

    def test_a_commit_that_is_not_the_pin_is_refused(self) -> None:
        code, out, err = self.run_script(extra={"YM_REAL_EVAL_COMMIT": "0" * 40})
        self.assertEqual(code, 4, out + err)
        self.assertEqual(self.names(), [])

    def test_a_modified_checkout_is_refused(self) -> None:
        self.assertEqual(self.run_script()[0], 0)
        (self.td / "real" / "metadata-extractor-images" / "arw" / "Sony NEX-T2.ARW").write_bytes(b"edited")
        code, out, err = self.run_script()
        self.assertEqual(code, 1, out + err)
        self.assertIn("local changes", err)

    def test_usage(self) -> None:
        code, _, err = self.run_script("--bogus")
        self.assertEqual(code, 2)
        self.assertIn("unknown option", err)
        code, out, _ = self.run_script("--help")
        self.assertEqual(code, 0)
        self.assertIn("LICENCE IS UNCLEAR", out)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for demosaic_sets.py. Standard library only; no network (the "server" is a folder read through file://).

Run:  python3 -I -m unittest discover -s tools/fetch -p 'test_*.py' -v
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import demosaic_sets  # noqa: E402


def blob(seed: int, size: int = 600) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < size:
        out += hashlib.sha256(f"{seed}:{counter}".encode()).digest()
        counter += 1
    return bytes(out[:size])


COMMIT_A = "a" * 40
COMMIT_B = "b" * 40


def make_world(root: Path) -> Tuple[Dict[str, Any], Path]:
    """A fake raw host under ``root/host`` and a manifest that describes it: two small sets."""
    host = root / "host"
    sources = []
    for index, (source_id, repo, commit, directory, names) in enumerate(
        [
            ("alpha", "own/one", COMMIT_A, "sets/A", ["a1.png", "a2.png"]),
            ("beta", "own/two", COMMIT_B, "sets/B Set", ["1.tif", "2.tif", "3.tif"]),
        ]
    ):
        files = []
        for n, name in enumerate(names):
            data = blob(index * 10 + n)
            target = host / repo / commit / directory / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            files.append({"name": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        sources.append(
            {"id": source_id, "repo": repo, "commit": commit, "dir": directory, "dest": source_id, "licence_note": f"{source_id} note", "files": files}
        )
    manifest = {"raw_base": host.as_uri(), "sources": sources}
    return manifest, host


def run(argv: List[str], manifest: Dict[str, Any], testdata: Path, extra_env: Dict[str, str] | None = None) -> Tuple[int, str, str]:
    path = testdata.parent / "manifest.json"
    path.write_text(json.dumps(manifest))
    env = {"YM_TESTDATA": str(testdata), "YM_DEMOSAIC_SETS": str(path)}
    env.update(extra_env or {})
    out, err = io.StringIO(), io.StringIO()
    with mock.patch.dict(os.environ, env), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = demosaic_sets.main(argv)
    return code, out.getvalue(), err.getvalue()


class Fetch(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.manifest, self.host = make_world(self.root)
        self.testdata = self.root / "testdata"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def dest(self, source: str, name: str) -> Path:
        return self.testdata / "demosaic" / source / name

    def test_downloads_every_file_into_testdata_demosaic_and_checks_the_hash(self) -> None:
        code, out, err = run([], self.manifest, self.testdata)
        self.assertEqual(code, demosaic_sets.EXIT_OK, err)
        for source in self.manifest["sources"]:
            for entry in source["files"]:
                data = self.dest(source["id"], entry["name"]).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), entry["sha256"])
        self.assertIn("alpha: 2 of 2", out)
        self.assertIn("beta: 3 of 3", out)

    def test_a_second_run_needs_no_network(self) -> None:
        run([], self.manifest, self.testdata)
        self.manifest["raw_base"] = (self.root / "gone").as_uri()
        code, out, err = run([], self.manifest, self.testdata)
        self.assertEqual(code, demosaic_sets.EXIT_OK, err)
        self.assertIn("already there", out)

    def test_a_download_with_the_wrong_hash_is_refused_and_leaves_no_file(self) -> None:
        victim = self.host / "own/one" / COMMIT_A / "sets/A" / "a2.png"
        victim.write_bytes(victim.read_bytes()[:-1] + b"X")
        code, _out, err = run([], self.manifest, self.testdata)
        self.assertEqual(code, demosaic_sets.EXIT_MISMATCH)
        self.assertIn("a2.png", err)
        self.assertFalse(self.dest("alpha", "a2.png").exists())
        self.assertEqual(list((self.testdata / "demosaic" / "alpha").glob("*.part")), [])
        # the other files, and the other set, were still fetched
        self.assertTrue(self.dest("alpha", "a1.png").exists())
        self.assertTrue(self.dest("beta", "3.tif").exists())

    def test_a_file_that_is_not_the_pinned_one_is_left_alone_unless_forced(self) -> None:
        run([], self.manifest, self.testdata)
        mine = self.dest("alpha", "a1.png")
        mine.write_bytes(b"my own image")
        code, _out, err = run([], self.manifest, self.testdata)
        self.assertEqual(code, demosaic_sets.EXIT_MISMATCH)
        self.assertEqual(mine.read_bytes(), b"my own image")
        self.assertIn("--force", err)
        code, _out, err = run(["--force"], self.manifest, self.testdata)
        self.assertEqual(code, demosaic_sets.EXIT_OK, err)
        self.assertEqual(hashlib.sha256(mine.read_bytes()).hexdigest(), self.manifest["sources"][0]["files"][0]["sha256"])

    def test_other_files_in_the_folder_are_never_touched(self) -> None:
        run([], self.manifest, self.testdata)
        extra = self.testdata / "demosaic" / "alpha" / "notes.txt"
        extra.write_text("keep")
        run(["--force"], self.manifest, self.testdata)
        self.assertEqual(extra.read_text(), "keep")

    def test_an_unreachable_host_exits_4_and_still_tries_the_other_set(self) -> None:
        (self.host / "own/one").rename(self.root / "moved")
        code, _out, err = run([], self.manifest, self.testdata)
        self.assertEqual(code, demosaic_sets.EXIT_NETWORK)
        self.assertIn("alpha", err)
        self.assertTrue(self.dest("beta", "1.tif").exists())

    def test_the_url_is_the_pinned_commit_and_keeps_spaces_in_the_directory(self) -> None:
        seen: List[str] = []
        real = demosaic_sets.download

        def spy(url: str, *args: Any, **kwargs: Any) -> bytes:
            seen.append(url)
            return real(url, *args, **kwargs)

        with mock.patch.object(demosaic_sets, "download", spy):
            run([], self.manifest, self.testdata)
        self.assertTrue(any(f"/own/two/{COMMIT_B}/sets/B%20Set/1.tif" in url for url in seen), seen)


class List_(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.manifest, _host = make_world(self.root)
        self.testdata = self.root / "testdata"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_nothing_there_is_exit_3_without_network(self) -> None:
        self.manifest["raw_base"] = "http://127.0.0.1:9/unreachable"
        code, out, _err = run(["--list"], self.manifest, self.testdata)
        self.assertEqual(code, demosaic_sets.EXIT_NOTHING)
        self.assertIn("0 of 2", out)

    def test_a_complete_verified_set_is_exit_0(self) -> None:
        run([], self.manifest, self.testdata)
        code, out, _err = run(["--list"], self.manifest, self.testdata)
        self.assertEqual(code, demosaic_sets.EXIT_OK)
        self.assertIn("alpha: 2 of 2", out)
        self.assertIn("beta: 3 of 3", out)

    def test_a_missing_or_changed_file_is_named_and_exit_1(self) -> None:
        run([], self.manifest, self.testdata)
        (self.testdata / "demosaic" / "beta" / "2.tif").unlink()
        (self.testdata / "demosaic" / "alpha" / "a1.png").write_bytes(b"changed")
        code, out, _err = run(["--list"], self.manifest, self.testdata)
        self.assertEqual(code, demosaic_sets.EXIT_MISMATCH)
        self.assertIn("beta: 2 of 3", out)
        self.assertIn("2.tif (missing)", out)
        self.assertIn("a1.png (differs from the pinned file)", out)


class Table(unittest.TestCase):
    def table(self) -> Dict[str, Any]:
        return json.loads((HERE / "demosaic-sets.json").read_text())

    def test_the_built_in_table_pins_kodak_24_and_mcmaster_18(self) -> None:
        sources = {s["id"]: s for s in self.table()["sources"]}
        self.assertEqual(sorted(sources), ["kodak", "mcmaster"])
        self.assertEqual(len(sources["kodak"]["files"]), 24)
        self.assertEqual(len(sources["mcmaster"]["files"]), 18)
        # the folders crates/ym-gpu/tests/cpu_demosaic.rs reads
        self.assertEqual(sources["kodak"]["dest"], "kodak")
        self.assertEqual(sources["mcmaster"]["dest"], "mcmaster")

    def test_every_entry_is_pinned_by_commit_and_sha256(self) -> None:
        for source in self.table()["sources"]:
            self.assertRegex(source["commit"], r"^[0-9a-f]{40}$")
            names = [f["name"] for f in source["files"]]
            self.assertEqual(len(names), len(set(names)))
            for entry in source["files"]:
                self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")
                self.assertGreater(entry["size"], 0)
            self.assertTrue(source["licence_note"])

    def test_the_table_is_validated_before_anything_is_fetched(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        bad = self.table()
        bad["sources"][0]["files"][0]["sha256"] = "not-a-hash"
        code, _out, err = run([], bad, Path(tmp.name) / "testdata")
        self.assertEqual(code, demosaic_sets.EXIT_USAGE)
        self.assertIn("sha256", err)
        self.assertFalse((Path(tmp.name) / "testdata" / "demosaic").exists())

    def test_a_file_name_cannot_leave_its_folder(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        bad = self.table()
        bad["sources"][0]["files"][0]["name"] = "../escape.png"
        code, _out, err = run([], bad, Path(tmp.name) / "testdata")
        self.assertEqual(code, demosaic_sets.EXIT_USAGE)
        self.assertIn("name", err)

    def test_the_documentation_points_at_this_script(self) -> None:
        text = (HERE.parents[1] / "docs" / "05-raw-processing.md").read_text()
        self.assertIn("demosaic-sets.sh", text)
        cpu_test = (HERE.parents[1] / "crates" / "ym-gpu" / "tests" / "cpu_demosaic.rs").read_text()
        self.assertTrue(re.search(r"testdata/demosaic/\{kodak,mcmaster\}", cpu_test))


class Usage(unittest.TestCase):
    def test_help_and_unknown_arguments(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(demosaic_sets.main(["--help"]), demosaic_sets.EXIT_OK)
        self.assertIn("Kodak", out.getvalue())
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(demosaic_sets.main(["--bogus"]), demosaic_sets.EXIT_USAGE)


if __name__ == "__main__":
    unittest.main()

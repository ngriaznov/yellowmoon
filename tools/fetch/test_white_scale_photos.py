"""Unit tests for white_scale_photos.py. Standard library only; no network (the "server" is a folder read through file://).

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
import white_scale_photos as wsp  # noqa: E402


def blob(seed: int, size: int = 600) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < size:
        out += hashlib.sha256(f"{seed}:{counter}".encode()).digest()
        counter += 1
    return bytes(out[:size])


def make_world(root: Path) -> Tuple[Dict[str, Any], Path]:
    """A fake file store under ``root/host`` and a table that describes it: three small files."""
    host = root / "host"
    files = []
    for n, (name, model) in enumerate([("100_A.ARW", "ILCE-7"), ("101_B.ARW", "ILCE-7"), ("102_C.ARW", "ZV-E10")]):
        data = blob(n)
        target = host / "original" / f"{n}.ARW"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        files.append(
            {
                "name": name,
                "model": model,
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "url": target.as_uri(),
                "topic_url": f"https://discuss.pixls.us/t/{100 + n}",
                "author": "someone",
                "licence": "Creative Commons, By-Attribution, Share-Alike",
            }
        )
    return {"schema_version": 1, "hosts": ["files.example"], "files": files}, host


def run(argv: List[str], table: Dict[str, Any], testdata: Path) -> Tuple[int, str, str]:
    path = testdata.parent / "table.json"
    path.write_text(json.dumps(table))
    env = {"YM_TESTDATA": str(testdata), wsp.TABLE_ENV: str(path)}
    out, err = io.StringIO(), io.StringIO()
    with mock.patch.dict(os.environ, env), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = wsp.main(argv)
    return code, out.getvalue(), err.getvalue()


class Fetch(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.table, self.host = make_world(self.root)
        self.testdata = self.root / "testdata"
        self.dest = self.testdata / "white-scale" / "play-raw"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_downloads_every_file_into_testdata_white_scale_play_raw_and_checks_the_hash(self) -> None:
        code, out, err = run([], self.table, self.testdata)
        self.assertEqual(code, wsp.EXIT_OK, err)
        for entry in self.table["files"]:
            data = (self.dest / entry["name"]).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), entry["sha256"])
        self.assertIn("3 of 3", out)

    def test_a_second_run_needs_no_network(self) -> None:
        run([], self.table, self.testdata)
        for entry in self.table["files"]:
            entry["url"] = (self.root / "gone").as_uri()
        code, out, err = run([], self.table, self.testdata)
        self.assertEqual(code, wsp.EXIT_OK, err)
        self.assertIn("3 already there", out)

    def test_a_download_with_the_wrong_hash_is_refused_and_leaves_no_file(self) -> None:
        victim = self.host / "original" / "1.ARW"
        victim.write_bytes(victim.read_bytes()[:-1] + b"X")
        code, _out, err = run([], self.table, self.testdata)
        self.assertEqual(code, wsp.EXIT_MISMATCH)
        self.assertIn("101_B.ARW", err)
        self.assertFalse((self.dest / "101_B.ARW").exists())
        self.assertEqual(list(self.dest.glob("*.part")), [])
        self.assertTrue((self.dest / "100_A.ARW").exists())
        self.assertTrue((self.dest / "102_C.ARW").exists())

    def test_a_file_that_is_not_the_pinned_one_is_left_alone_unless_forced(self) -> None:
        run([], self.table, self.testdata)
        mine = self.dest / "100_A.ARW"
        mine.write_bytes(b"my own file")
        code, _out, err = run([], self.table, self.testdata)
        self.assertEqual(code, wsp.EXIT_MISMATCH)
        self.assertEqual(mine.read_bytes(), b"my own file")
        self.assertIn("--force", err)
        code, _out, err = run(["--force"], self.table, self.testdata)
        self.assertEqual(code, wsp.EXIT_OK, err)
        self.assertEqual(hashlib.sha256(mine.read_bytes()).hexdigest(), self.table["files"][0]["sha256"])

    def test_other_files_in_the_folder_are_never_touched(self) -> None:
        run([], self.table, self.testdata)
        extra = self.dest / "notes.txt"
        extra.write_text("keep")
        run(["--force"], self.table, self.testdata)
        self.assertEqual(extra.read_text(), "keep")

    def test_an_unreachable_store_exits_4(self) -> None:
        for entry in self.table["files"]:
            entry["url"] = (self.root / "gone").as_uri()
        code, _out, err = run([], self.table, self.testdata)
        self.assertEqual(code, wsp.EXIT_NETWORK)
        self.assertIn("100_A.ARW", err)


class List_(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.table, _host = make_world(self.root)
        self.testdata = self.root / "testdata"
        self.dest = self.testdata / "white-scale" / "play-raw"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_nothing_there_is_exit_3_without_network(self) -> None:
        code, out, _err = run(["--list"], self.table, self.testdata)
        self.assertEqual(code, wsp.EXIT_NOTHING)
        self.assertIn("0 of 3", out)

    def test_a_complete_verified_set_is_exit_0(self) -> None:
        run([], self.table, self.testdata)
        code, out, _err = run(["--list"], self.table, self.testdata)
        self.assertEqual(code, wsp.EXIT_OK)
        self.assertIn("3 of 3", out)

    def test_a_missing_or_changed_file_is_named_and_exit_1(self) -> None:
        run([], self.table, self.testdata)
        (self.dest / "101_B.ARW").unlink()
        (self.dest / "100_A.ARW").write_bytes(b"changed")
        code, out, _err = run(["--list"], self.table, self.testdata)
        self.assertEqual(code, wsp.EXIT_MISMATCH)
        self.assertIn("1 of 3", out)
        self.assertIn("101_B.ARW (missing)", out)
        self.assertIn("100_A.ARW (differs from the pinned file)", out)


class Table(unittest.TestCase):
    def table(self) -> Dict[str, Any]:
        return json.loads((HERE / "white-scale-photos.json").read_text())

    def test_every_entry_is_pinned_by_sha256_with_its_post_poster_and_licence(self) -> None:
        files = self.table()["files"]
        self.assertEqual(len(files), 17)
        names = [f["name"] for f in files]
        self.assertEqual(len(names), len(set(names)))
        hosts = set(self.table()["hosts"])
        for entry in files:
            self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")
            self.assertGreater(entry["size"], 0)
            self.assertRegex(entry["topic_url"], r"^https://discuss\.pixls\.us/t/\d+$")
            self.assertTrue(entry["name"].startswith(entry["topic_url"].rsplit("/", 1)[1] + "_"), entry["name"])
            self.assertTrue(entry["author"])
            self.assertIn(entry["licence"], ("Creative Commons, By-Attribution, Share-Alike", "Public Domain"))
            self.assertIn(re.match(r"https://([^/]+)/", entry["url"]).group(1), hosts)
            # The address names the file by its SHA-1, so it cannot change under the pin.
            self.assertRegex(entry["url"], r"/original/\dX/[0-9a-f]/[0-9a-f]/[0-9a-f]{40}\.ARW$")

    def test_the_built_in_table_loads(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(wsp.TABLE_ENV, None)
            self.assertEqual(len(wsp.load_table()["files"]), 17)

    def test_the_table_is_validated_before_anything_is_fetched(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        bad = self.table()
        bad["files"][0]["sha256"] = "not-a-hash"
        code, _out, err = run([], bad, Path(tmp.name) / "testdata")
        self.assertEqual(code, wsp.EXIT_USAGE)
        self.assertIn("sha256", err)
        self.assertFalse((Path(tmp.name) / "testdata" / "white-scale").exists())

    def test_a_file_name_cannot_leave_its_folder(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        bad = self.table()
        bad["files"][0]["name"] = "../escape.ARW"
        code, _out, err = run([], bad, Path(tmp.name) / "testdata")
        self.assertEqual(code, wsp.EXIT_USAGE)
        self.assertIn("name", err)

    def test_an_address_outside_the_tables_hosts_is_refused(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        bad = self.table()
        bad["files"][0]["url"] = "https://elsewhere.example/x.ARW"
        env = {"YM_TESTDATA": str(Path(tmp.name) / "testdata")}
        path = Path(tmp.name) / "t.json"
        path.write_text(json.dumps(bad))
        env[wsp.TABLE_ENV] = str(path)
        err = io.StringIO()
        with mock.patch.dict(os.environ, env), contextlib.redirect_stderr(err):
            self.assertEqual(wsp.main([]), wsp.EXIT_USAGE)
        self.assertIn("hosts", err.getvalue())


class Usage(unittest.TestCase):
    def test_help_and_unknown_arguments(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(wsp.main(["--help"]), wsp.EXIT_OK)
        self.assertIn("white scales", out.getvalue())
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(wsp.main(["--bogus"]), wsp.EXIT_USAGE)


if __name__ == "__main__":
    unittest.main()

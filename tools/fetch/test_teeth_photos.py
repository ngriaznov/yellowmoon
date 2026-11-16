"""Tests of tools/fetch/teeth-photos.sh (stdlib only: ``python3 -I -m unittest discover -s tools/fetch -p 'test_*.py'``).

The script runs inside a temporary copy of the tree it needs and downloads from local ``file://`` URLs, so nothing here touches the
network or the real data folders.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "teeth-photos.sh"
REAL_LIST = HERE / "teeth-photos.tsv"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class TeethPhotosFetch(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "repo"
        (self.root / "tools" / "fetch").mkdir(parents=True)
        shutil.copy(SCRIPT, self.root / "tools" / "fetch" / "teeth-photos.sh")
        self.photos = self.root / "testdata" / "teeth" / "photos"
        self.src = Path(self._tmp.name) / "a.jpg"
        self.src.write_bytes(b"\xff\xd8 a photo")

    def list_with(self, pin: str):
        (self.root / "tools" / "fetch" / "teeth-photos.tsv").write_text(f"# comment\npub-a\t{pin}\t{self.src.as_uri()}\tCC0, test\n")

    def run_script(self, *args: str) -> subprocess.CompletedProcess:
        env = dict(os.environ, YM_TESTDATA=str(self.root / "testdata"), YM_TEETH_RETRIES="0")
        return subprocess.run(["bash", str(self.root / "tools" / "fetch" / "teeth-photos.sh"), *args], env=env, capture_output=True, text=True, timeout=60, check=False)

    def test_a_photo_is_fetched_by_its_pin_and_verified(self):
        self.list_with(sha256(self.src.read_bytes()))
        self.assertEqual(self.run_script("--verify").returncode, 1, "nothing on disk yet")
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.photos / "pub-a.jpg").read_bytes(), self.src.read_bytes())
        self.assertFalse((self.photos / "pub-a.jpg.part").exists())
        self.assertEqual(self.run_script("--verify").returncode, 0)

    def test_a_download_with_another_hash_is_never_kept_as_the_photo(self):
        self.list_with(sha256(b"something else"))
        result = self.run_script()
        self.assertEqual(result.returncode, 1)
        self.assertFalse((self.photos / "pub-a.jpg").exists())
        self.assertTrue((self.photos / "pub-a.jpg.rejected").exists())

    def test_a_missing_pin_stops_with_exit_3(self):
        self.list_with("TBD")
        self.assertEqual(self.run_script().returncode, 3)

    def test_the_real_list_is_complete(self):
        rows = [l.split("\t") for l in REAL_LIST.read_text().splitlines() if l and not l.startswith("#")]
        self.assertGreaterEqual(len(rows), 20)
        for row in rows:
            self.assertEqual(len(row), 4, row)
            self.assertRegex(row[1], r"^[0-9a-f]{64}$")
            self.assertTrue(row[2].startswith("https://"), row)
            self.assertRegex(row[3], r"^(CC0|public domain mark), .*expect (teeth|no-teeth|blown-out)$")
        self.assertEqual(len({r[0] for r in rows}), len(rows), "unique ids")
        self.assertGreaterEqual(sum(1 for r in rows if r[3].endswith("expect teeth")), 18)


if __name__ == "__main__":
    unittest.main()

"""Tests of tools/fetch/sam2.sh (stdlib only: ``python3 -I -m unittest discover -s tools/fetch -p 'test_*.py'``).

The script runs inside a temporary copy of the tree it needs (tools/fetch/sam2.sh, data/models/sam2/SHA256SUMS,
tools/fetch/sam2.sha256) and downloads from a local directory through ``YM_SAM2_BASE_URL=file://...``, so nothing
here touches the network or the real data folders.
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
SCRIPT = HERE / "sam2.sh"
WEIGHTS = "sam2.1_hiera_small.f16.safetensors"
FIXTURES = "sam2-oracle-fixtures-1.tar.zst"
PLACEHOLDER = "TBD-after-first-run"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Sam2Fetch(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "repo"
        (self.root / "tools" / "fetch").mkdir(parents=True)
        (self.root / "data" / "models" / "sam2").mkdir(parents=True)
        shutil.copy(SCRIPT, self.root / "tools" / "fetch" / "sam2.sh")
        self.release = Path(self._tmp.name) / "release"
        self.release.mkdir()
        self.weights_path = self.root / "data" / "models" / "sam2" / WEIGHTS

    def pin(self, weights: str | None = None, fixtures: str | None = None):
        if weights is not None:
            (self.root / "data" / "models" / "sam2" / "SHA256SUMS").write_text(f"{weights}  {WEIGHTS}\n")
        if fixtures is not None:
            (self.root / "tools" / "fetch" / "sam2.sha256").write_text(f"{fixtures}  {FIXTURES}\n")

    def run_script(self, *args: str, base: Path | None = None) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        env["YM_SAM2_BASE_URL"] = (base or self.release).as_uri()
        env["YM_TESTDATA"] = str(self.root / "testdata")
        return subprocess.run(
            ["bash", str(self.root / "tools" / "fetch" / "sam2.sh"), *args], env=env, capture_output=True, text=True, timeout=60, check=False
        )

    def test_a_placeholder_pin_stops_with_exit_3_and_downloads_nothing(self):
        self.pin(weights=PLACEHOLDER)
        (self.release / WEIGHTS).write_bytes(b"weights")
        result = self.run_script()
        self.assertEqual(result.returncode, 3, result.stderr)
        self.assertIn("no SHA-256 pin", result.stderr)
        self.assertFalse(self.weights_path.exists())

    def test_a_missing_sums_file_is_a_missing_pin(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 3, result.stderr)

    def test_the_weights_are_downloaded_and_checked(self):
        data = b"sam2 weights" * 100
        (self.release / WEIGHTS).write_bytes(data)
        self.pin(weights=sha256(data))
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.weights_path.read_bytes(), data)
        self.assertFalse(self.weights_path.with_name(WEIGHTS + ".part").exists())
        # The file is on disk and right: a second run needs no network (the base URL points nowhere).
        again = self.run_script(base=Path(self._tmp.name) / "nowhere")
        self.assertEqual(again.returncode, 0, again.stderr)
        self.assertIn("matches its pin", again.stdout)

    def test_a_download_with_another_hash_is_refused_and_removed(self):
        (self.release / WEIGHTS).write_bytes(b"other bytes")
        self.pin(weights=sha256(b"the pinned bytes"))
        result = self.run_script()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("nothing was kept", result.stderr)
        self.assertFalse(self.weights_path.exists())
        self.assertFalse(self.weights_path.with_name(WEIGHTS + ".part").exists())

    def test_a_damaged_file_on_disk_is_replaced(self):
        data = b"good weights"
        (self.release / WEIGHTS).write_bytes(data)
        self.pin(weights=sha256(data))
        self.weights_path.write_bytes(b"damaged")
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.weights_path.read_bytes(), data)

    def test_a_release_without_the_asset_is_an_error(self):
        self.pin(weights=sha256(b"x"))
        result = self.run_script()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("could not download", result.stderr)

    def test_verify_downloads_nothing_and_says_what_is_wrong(self):
        data = b"weights"
        self.pin(weights=sha256(data))
        missing = self.run_script("--verify")
        self.assertEqual(missing.returncode, 1)
        self.assertIn("is missing", missing.stderr)
        self.weights_path.write_bytes(b"not the weights")
        wrong = self.run_script("--verify")
        self.assertEqual(wrong.returncode, 1)
        self.assertIn("the pin is", wrong.stderr)
        self.weights_path.write_bytes(data)
        self.assertEqual(self.run_script("--verify").returncode, 0)

    def test_accept_new_prints_the_lines_for_the_pin_files(self):
        (self.release / WEIGHTS).write_bytes(b"w")
        (self.release / FIXTURES).write_bytes(b"f")
        result = self.run_script("--accept-new")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"{sha256(b'w')}  {WEIGHTS}", result.stdout)
        self.assertIn(f"{sha256(b'f')}  {FIXTURES}", result.stdout)

    def test_an_unknown_argument_is_a_usage_error(self):
        self.assertEqual(self.run_script("--nope").returncode, 2)

    @unittest.skipUnless(shutil.which("zstd") and shutil.which("tar"), "zstd and tar are needed to build a fixture archive")
    def test_the_fixtures_are_unpacked_once(self):
        source = Path(self._tmp.name) / "src"
        source.mkdir()
        (source / "manifest.json").write_text("{}")
        (source / "dog.npz").write_bytes(b"npz")
        tarball = self.release / "fixtures.tar"
        subprocess.run(["tar", "-cf", str(tarball), "-C", str(source), "manifest.json", "dog.npz"], check=True)
        subprocess.run(["zstd", "-q", "-f", str(tarball), "-o", str(self.release / FIXTURES)], check=True)
        data = (self.release / FIXTURES).read_bytes()
        self.pin(weights=sha256(b"unused"), fixtures=sha256(data))
        result = self.run_script("--fixtures-only")
        self.assertEqual(result.returncode, 0, result.stderr)
        unpacked = self.root / "testdata" / "sam2" / "oracle-1"
        self.assertEqual((unpacked / "dog.npz").read_bytes(), b"npz")
        self.assertTrue((unpacked / "manifest.json").exists())
        self.assertFalse(self.weights_path.exists(), "--fixtures-only fetches no weights")


if __name__ == "__main__":
    unittest.main()

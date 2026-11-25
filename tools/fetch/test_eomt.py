"""Tests of tools/fetch/eomt.sh (stdlib only: ``python3 -I -m unittest discover -s tools/fetch -p 'test_*.py'``).

The script runs inside a temporary copy of the tree it needs (tools/fetch/eomt.sh and data/models/eomt/SHA256SUMS) and downloads from
a local directory through ``YM_EOMT_BASE_URL=file://...``, so nothing here touches the network or the real data folders.
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
SCRIPT = HERE / "eomt.sh"
WEIGHTS = "eomt_coco_panoptic_small.f16.safetensors"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class EomtFetch(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "repo"
        (self.root / "tools" / "fetch").mkdir(parents=True)
        (self.root / "data" / "models" / "eomt").mkdir(parents=True)
        shutil.copy(SCRIPT, self.root / "tools" / "fetch" / "eomt.sh")
        self.release = Path(self._tmp.name) / "release"
        self.release.mkdir()
        self.weights_path = self.root / "data" / "models" / "eomt" / WEIGHTS

    def pin(self, weights: str):
        (self.root / "data" / "models" / "eomt" / "SHA256SUMS").write_text(f"{weights}  {WEIGHTS}\n")

    def run_script(self, *args: str, base: Path | None = None) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        env["YM_EOMT_BASE_URL"] = (base or self.release).as_uri()
        return subprocess.run(["bash", str(self.root / "tools" / "fetch" / "eomt.sh"), *args], env=env, capture_output=True, text=True, timeout=60, check=False)

    def test_a_placeholder_pin_stops_with_exit_3_and_downloads_nothing(self):
        self.pin("TBD-after-first-run")
        (self.release / WEIGHTS).write_bytes(b"weights")
        result = self.run_script()
        self.assertEqual(result.returncode, 3, result.stderr)
        self.assertIn("no SHA-256 pin", result.stderr)
        self.assertFalse(self.weights_path.exists())

    def test_a_missing_sums_file_is_a_missing_pin(self):
        self.assertEqual(self.run_script().returncode, 3)

    def test_the_weights_are_downloaded_and_checked(self):
        data = b"eomt weights" * 100
        (self.release / WEIGHTS).write_bytes(data)
        self.pin(sha256(data))
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
        self.pin(sha256(b"the pinned bytes"))
        result = self.run_script()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("nothing was kept", result.stderr)
        self.assertFalse(self.weights_path.exists())
        self.assertFalse(self.weights_path.with_name(WEIGHTS + ".part").exists())

    def test_a_damaged_file_on_disk_is_replaced(self):
        data = b"good weights"
        (self.release / WEIGHTS).write_bytes(data)
        self.pin(sha256(data))
        self.weights_path.write_bytes(b"damaged")
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.weights_path.read_bytes(), data)

    def test_a_release_without_the_asset_is_an_error_that_names_the_cause(self):
        self.pin(sha256(b"x"))
        result = self.run_script()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("could not download", result.stderr)
        self.assertIn("published", result.stderr)

    def test_verify_downloads_nothing_and_says_what_is_wrong(self):
        data = b"weights"
        self.pin(sha256(data))
        missing = self.run_script("--verify")
        self.assertEqual(missing.returncode, 1)
        self.assertIn("is missing", missing.stderr)
        self.weights_path.write_bytes(b"not the weights")
        wrong = self.run_script("--verify")
        self.assertEqual(wrong.returncode, 1)
        self.assertIn("the pin is", wrong.stderr)
        self.weights_path.write_bytes(data)
        self.assertEqual(self.run_script("--verify").returncode, 0)

    def test_an_unknown_argument_is_a_usage_error(self):
        self.assertEqual(self.run_script("--nope").returncode, 2)


class CommittedPin(unittest.TestCase):
    """The pin file in the repository holds exactly one line for the weights, and the script, the workflow and the pin agree."""

    def test_the_weights_pin_is_a_sha256(self):
        path = HERE.parents[1] / "data" / "models" / "eomt" / "SHA256SUMS"
        lines = [line.split() for line in path.read_text().splitlines() if line.strip()]
        self.assertEqual([fields[1] for fields in lines], [WEIGHTS])
        self.assertRegex(lines[0][0], r"^[0-9a-f]{64}$")

    def test_the_script_names_the_release_and_asset_the_workflow_publishes(self):
        script = SCRIPT.read_text()
        workflow = (HERE.parents[1] / ".github" / "workflows" / "models.yml").read_text()
        for needle in ('tag="models-eomt-small-1"', f'weights_name="{WEIGHTS}"'):
            self.assertIn(needle, script)
        self.assertIn("RELEASE_TAG: models-eomt-small-1", workflow)
        self.assertIn(f"WEIGHTS: {WEIGHTS}", workflow)


if __name__ == "__main__":
    unittest.main()

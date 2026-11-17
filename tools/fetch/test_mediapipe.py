"""Tests of tools/fetch/mediapipe.sh (stdlib only: ``python3 -I -m unittest discover -s tools/fetch -p 'test_*.py'``).

The script runs inside a temporary copy of the tree it needs and downloads from a local directory through
``YM_MEDIAPIPE_BASE_URL=file://...`` (and ``file://`` URLs for the photos), so nothing here touches the network or the real data folders.
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
SCRIPT = HERE / "mediapipe.sh"
FILES = {
    "blazeface": "blaze_face_full_range.f16.safetensors",
    "face_mesh": "face_mesh_v2.f16.safetensors",
    "selfie_multiclass": "selfie_multiclass_256.f32.safetensors",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class MediapipeFetch(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "repo"
        (self.root / "tools" / "fetch").mkdir(parents=True)
        (self.root / "data" / "models" / "mediapipe").mkdir(parents=True)
        shutil.copy(SCRIPT, self.root / "tools" / "fetch" / "mediapipe.sh")
        self.release = Path(self._tmp.name) / "release"
        self.release.mkdir()
        self.models = self.root / "data" / "models" / "mediapipe"

    def pin(self, **contents: bytes):
        lines = [f"{sha256(data)}  {FILES[name]}\n" for name, data in contents.items()]
        (self.models / "SHA256SUMS").write_text("".join(lines))

    def run_script(self, *args: str, base: Path | None = None) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        env["YM_MEDIAPIPE_BASE_URL"] = (base or self.release).as_uri()
        env["YM_TESTDATA"] = str(self.root / "testdata")
        env["YM_MEDIAPIPE_RETRIES"] = "0"
        return subprocess.run(
            ["bash", str(self.root / "tools" / "fetch" / "mediapipe.sh"), *args], env=env, capture_output=True, text=True, timeout=60, check=False
        )

    def release_with(self, **contents: bytes):
        for name, data in contents.items():
            (self.release / FILES[name]).write_bytes(data)

    def test_a_missing_or_placeholder_pin_stops_with_exit_3_and_downloads_nothing(self):
        self.release_with(blazeface=b"w")
        self.assertEqual(self.run_script("--only", "blazeface").returncode, 3)
        (self.models / "SHA256SUMS").write_text(f"TBD-after-first-run  {FILES['blazeface']}\n")
        result = self.run_script("--only", "blazeface")
        self.assertEqual(result.returncode, 3, result.stderr)
        self.assertIn("no SHA-256 pin", result.stderr)
        self.assertFalse((self.models / FILES["blazeface"]).exists())

    def test_the_three_networks_are_downloaded_and_checked(self):
        data = {k: (k.encode() * 50) for k in FILES}
        self.release_with(**data)
        self.pin(**data)
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        for name, file in FILES.items():
            self.assertEqual((self.models / file).read_bytes(), data[name])
            self.assertFalse((self.models / (file + ".part")).exists())
        # Everything is on disk and right: a second run needs no network (the base URL points nowhere).
        again = self.run_script(base=Path(self._tmp.name) / "nowhere")
        self.assertEqual(again.returncode, 0, again.stderr)
        self.assertIn("matches its pin", again.stdout)

    def test_only_fetches_the_named_network(self):
        data = {k: k.encode() for k in FILES}
        self.release_with(**data)
        self.pin(**data)
        self.assertEqual(self.run_script("--only", "face_mesh").returncode, 0)
        self.assertTrue((self.models / FILES["face_mesh"]).exists())
        self.assertFalse((self.models / FILES["blazeface"]).exists())
        self.assertEqual(self.run_script("--only", "nope").returncode, 2)

    def test_a_download_with_another_hash_is_refused_and_removed(self):
        self.release_with(blazeface=b"other bytes")
        self.pin(blazeface=b"the pinned bytes")
        result = self.run_script("--only", "blazeface")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("nothing was kept", result.stderr)
        self.assertFalse((self.models / FILES["blazeface"]).exists())

    def test_a_damaged_file_on_disk_is_replaced(self):
        self.release_with(blazeface=b"good")
        self.pin(blazeface=b"good")
        (self.models / FILES["blazeface"]).write_bytes(b"damaged")
        self.assertEqual(self.run_script("--only", "blazeface").returncode, 0)
        self.assertEqual((self.models / FILES["blazeface"]).read_bytes(), b"good")

    def test_a_release_without_the_asset_is_an_error(self):
        self.pin(blazeface=b"x")
        result = self.run_script("--only", "blazeface")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("could not download", result.stderr)

    def test_verify_downloads_nothing_and_says_what_is_wrong(self):
        self.pin(blazeface=b"weights")
        missing = self.run_script("--verify", "--only", "blazeface")
        self.assertEqual(missing.returncode, 1)
        self.assertIn("is missing", missing.stderr)
        (self.models / FILES["blazeface"]).write_bytes(b"not the weights")
        self.assertIn("the pin is", self.run_script("--verify", "--only", "blazeface").stderr)
        (self.models / FILES["blazeface"]).write_bytes(b"weights")
        self.assertEqual(self.run_script("--verify", "--only", "blazeface").returncode, 0)

    def test_photos_are_fetched_by_their_pins(self):
        photo = b"\xff\xd8 a photo"
        src = Path(self._tmp.name) / "photo.jpg"
        src.write_bytes(photo)
        (self.root / "tools" / "fetch" / "mediapipe-photos.tsv").write_text(
            f"# comment\nportrait\t{sha256(photo)}\t{src.as_uri()}\tCC0\n"
        )
        result = self.run_script("--photos-only")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.root / "testdata" / "mediapipe" / "photos" / "portrait.jpg").read_bytes(), photo)
        (self.root / "tools" / "fetch" / "mediapipe-photos.tsv").write_text(f"portrait\t{sha256(b'other')}\t{src.as_uri()}\tCC0\n")
        (self.root / "testdata" / "mediapipe" / "photos" / "portrait.jpg").unlink()
        self.assertEqual(self.run_script("--photos-only").returncode, 1)

    def test_an_unknown_argument_is_a_usage_error(self):
        self.assertEqual(self.run_script("--nope").returncode, 2)


if __name__ == "__main__":
    unittest.main()

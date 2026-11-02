"""Tests of the parts of training/sam2 that need neither torch nor the network (stdlib, plus numpy and Pillow where noted):

    python3 -m unittest training.sam2.test_sam2           # from the repository root; numpy tests skip without numpy

The conversion and the oracle themselves run in CI (`.github/workflows/models.yml`); here are the file writers, the photo
list's rules, the downloader's checks and the prompts.
"""

from __future__ import annotations

import hashlib
import io
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from training.sam2 import export, oracle

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

LISTING = json.loads(oracle.PHOTOS_JSON.read_text(encoding="utf-8"))
ALLOWED_LICENCES = {"CC0-1.0", "Public domain"}


class Tensor:
    """What `write_safetensors` needs of a torch tensor."""

    def __init__(self, array):
        self.array = array

    def numpy(self):
        return self.array


def read_safetensors(path: Path):
    raw = path.read_bytes()
    (length,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + length])
    return length, header, raw[8 + length :]


@unittest.skipIf(np is None, "numpy is needed")
class Writers(unittest.TestCase):
    def test_safetensors_is_valid_aligned_sorted_and_the_same_bytes_whatever_the_metadata_order(self):
        tensors = {"b.weight": Tensor(np.arange(6, dtype=np.float16).reshape(2, 3)), "a.bias": Tensor(np.array([1.5, -2.25], dtype=np.float16))}
        metadata = {"format": "yellowmoon-sam2", "version": "1", "config": json.dumps({"k": [1, 2]})}
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp) / "a.safetensors", Path(tmp) / "b.safetensors"
            export.write_safetensors(first, tensors, metadata)
            export.write_safetensors(second, tensors, dict(reversed(list(metadata.items()))))
            self.assertEqual(first.read_bytes(), second.read_bytes())
            length, header, data = read_safetensors(first)
        self.assertEqual(length % 8, 0)
        self.assertEqual(header["__metadata__"], metadata)
        self.assertEqual(list(header)[1:], ["a.bias", "b.weight"])
        self.assertEqual(header["a.bias"], {"dtype": "F16", "shape": [2], "data_offsets": [0, 4]})
        self.assertEqual(header["b.weight"], {"dtype": "F16", "shape": [2, 3], "data_offsets": [4, 16]})
        self.assertEqual(len(data), 16)
        self.assertEqual(np.frombuffer(data[4:], dtype="<f2").reshape(2, 3).tolist(), [[0, 1, 2], [3, 4, 5]])

    def test_npz_has_the_given_keys_shapes_and_dtypes_and_the_same_bytes_every_time(self):
        arrays = {"picture": np.zeros((4, 6, 3), np.uint8), "empty": np.zeros((0, 2), np.float32), "n": np.asarray([3], np.int32)}
        with tempfile.TemporaryDirectory() as tmp:
            one, two = Path(tmp) / "one.npz", Path(tmp) / "two.npz"
            oracle.write_npz(one, arrays)
            oracle.write_npz(two, arrays)
            self.assertEqual(one.read_bytes(), two.read_bytes())
            loaded = np.load(one)
            self.assertEqual(sorted(loaded.files), sorted(arrays))
            for key, value in arrays.items():
                self.assertEqual(loaded[key].shape, value.shape)
                self.assertEqual(loaded[key].dtype, value.dtype)

    def test_iou_and_relative_rms(self):
        a = np.array([[1, 1], [0, 0]], bool)
        self.assertEqual(oracle.iou_of(a, np.array([[1, 0], [0, 0]], bool)), 0.5)
        self.assertEqual(oracle.iou_of(np.zeros((2, 2), bool), np.zeros((2, 2), bool)), 1.0)
        self.assertAlmostEqual(oracle.relative_rms(np.array([1.1, 2.2]), np.array([1.0, 2.0])), 0.1, places=6)

    def test_the_picture_has_a_long_edge_of_1024_and_keeps_the_aspect(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            for size in ((300, 200), (200, 300), (1024, 1024)):
                path = Path(tmp) / "p.jpg"
                Image.new("RGB", size, (10, 200, 30)).save(path)
                picture = oracle.load_picture(path)
                self.assertEqual(picture.dtype, np.uint8)
                self.assertEqual(picture.shape[2], 3)
                self.assertEqual(max(picture.shape[:2]), 1024)
                self.assertAlmostEqual(picture.shape[1] / picture.shape[0], size[0] / size[1], places=2)

    def test_the_three_prompts_of_a_photo(self):
        photo = LISTING["photos"][0]
        positive, pair, box = oracle.prompts_of(photo)
        self.assertEqual(positive["points"].shape, (1, 2))
        self.assertEqual(positive["labels"].tolist(), [1])
        self.assertTrue(positive["multimask"])
        self.assertEqual(pair["labels"].tolist(), [1, 0])
        self.assertFalse(pair["multimask"])
        self.assertEqual(box["box"].shape, (4,))
        self.assertEqual(box["points"].shape, (0, 2))


class Downloader(unittest.TestCase):
    def photo(self, directory: Path, data: bytes, sha256: str | None = None) -> dict:
        source = directory / "source.jpg"
        source.write_bytes(data)
        return {"id": "t", "url": source.as_uri(), "sha256": sha256 or hashlib.sha256(data).hexdigest()}

    def test_a_download_is_checked_and_kept(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(oracle.time, "sleep"):
            directory = Path(tmp)
            path = oracle.fetch_photo(self.photo(directory, b"jpeg bytes"), directory / "cache")
            self.assertEqual(path.read_bytes(), b"jpeg bytes")

    def test_a_file_with_another_hash_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(oracle.time, "sleep"):
            directory = Path(tmp)
            with self.assertRaises(SystemExit) as stop:
                oracle.fetch_photo(self.photo(directory, b"changed", sha256="0" * 64), directory / "cache")
            self.assertIn("changed", str(stop.exception))
            self.assertFalse((directory / "cache" / "t.jpg").exists())

    def test_an_unpinned_photo_needs_the_hash_commons_states(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(oracle.time, "sleep"):
            directory = Path(tmp)
            photo = self.photo(directory, b"abc", sha256=oracle.PLACEHOLDER)
            photo["commons_sha1"] = hashlib.sha1(b"abc").hexdigest()
            self.assertEqual(oracle.fetch_photo(photo, directory / "cache").read_bytes(), b"abc")
            photo["commons_sha1"] = "0" * 40
            (directory / "cache" / "t.jpg").unlink()
            with self.assertRaises(SystemExit) as stop:
                oracle.fetch_photo(photo, directory / "cache")
            self.assertIn("SHA-1", str(stop.exception))

    def test_a_damaged_cached_photo_is_fetched_again(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(oracle.time, "sleep"):
            directory = Path(tmp)
            photo = self.photo(directory, b"good")
            (directory / "cache").mkdir()
            (directory / "cache" / "t.jpg").write_bytes(b"damaged")
            self.assertEqual(oracle.fetch_photo(photo, directory / "cache").read_bytes(), b"good")


class PhotoList(unittest.TestCase):
    def test_there_are_20_photos_with_unique_ids_and_urls(self):
        photos = LISTING["photos"]
        self.assertEqual(len(photos), 20)
        self.assertEqual(len({p["id"] for p in photos}), 20)
        self.assertEqual(len({p["url"] for p in photos}), 20)

    def test_every_photo_has_clear_terms_a_commons_page_and_a_pin(self):
        for photo in LISTING["photos"]:
            with self.subTest(photo=photo["id"]):
                self.assertIn(photo["license"], ALLOWED_LICENCES)
                self.assertTrue(photo["url"].startswith("https://upload.wikimedia.org/wikipedia/commons/"), photo["url"])
                self.assertNotIn("?", photo["url"])
                self.assertTrue(photo["page"].startswith("https://commons.wikimedia.org/wiki/File:"), photo["page"])
                self.assertTrue(photo["author"].strip())
                self.assertTrue(photo["title"].endswith((".jpg", ".jpeg")))
                self.assertRegex(photo["commons_sha1"], r"^[0-9a-f]{40}$")
                self.assertRegex(photo["sha256"], r"^([0-9a-f]{64}|TBD-after-first-run)$")

    def test_the_prompts_lie_inside_the_picture_and_the_box_is_not_empty(self):
        for photo in LISTING["photos"]:
            spec = photo["prompts"]
            points = [spec["point"], spec["positive_negative"]["positive"], spec["positive_negative"]["negative"]]
            with self.subTest(photo=photo["id"]):
                for x, y in points:
                    self.assertTrue(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0, (x, y))
                x0, y0, x1, y1 = spec["box"]
                self.assertTrue(0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0, spec["box"])
                self.assertNotEqual(spec["positive_negative"]["positive"], spec["positive_negative"]["negative"])
                self.assertTrue(x0 <= spec["point"][0] <= x1 and y0 <= spec["point"][1] <= y1, "the positive click is inside the box")

    def test_the_pins_of_the_scripts_and_the_workflow_agree(self):
        workflow = (Path(__file__).resolve().parents[2] / ".github" / "workflows" / "models.yml").read_text(encoding="utf-8")
        self.assertIn(f"SAM2_COMMIT: {export.SAM2_COMMIT}", workflow)
        requirements = (Path(__file__).with_name("requirements.txt")).read_text(encoding="utf-8")
        self.assertIn(export.SAM2_COMMIT, requirements)
        self.assertIn(export.CKPT_SHA256, (Path(__file__).resolve().parents[2] / "data" / "models" / "sam2" / "MODEL_CARD.md").read_text(encoding="utf-8"))
        self.assertIn(export.OUT_NAME, workflow)
        self.assertEqual(export.EXPECTED_KEPT_PARAMETERS, 38_538_833)


if __name__ == "__main__":
    unittest.main()

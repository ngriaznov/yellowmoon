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
try:
    import PIL  # noqa: F401 (only to know whether Pillow is importable)
except ImportError:  # pragma: no cover
    PIL = None

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


def tiff_with_zero_denominator(orientation: int) -> bytes:
    """The EXIF block of a JPEG as raw bytes: Orientation and an XResolution of 1/0 (a zero denominator, as some cameras and editors write)."""
    entries = [
        struct.pack("<HHII", 0x0112, 3, 1, orientation),  # Orientation, SHORT
        struct.pack("<HHII", 0x011A, 5, 1, 8 + 2 + 2 * 12 + 4),  # XResolution, RATIONAL, its data right after the IFD
    ]
    ifd = struct.pack("<H", len(entries)) + b"".join(entries) + struct.pack("<I", 0)
    return b"Exif\x00\x00" + b"II*\x00" + struct.pack("<I", 8) + ifd + struct.pack("<II", 1, 0)


@unittest.skipIf(np is None or PIL is None, "numpy and Pillow are needed")
class Orientation(unittest.TestCase):
    """`load_picture` applies the EXIF orientation by itself: `ImageOps.exif_transpose` writes the whole EXIF block again and
    stops with ZeroDivisionError on a rational tag whose denominator is 0 (the `hyacinth` photo)."""

    SIZE = (30, 20)

    def base_image(self):
        from PIL import Image

        rng = np.random.default_rng(7)
        pixels = rng.integers(0, 256, size=(self.SIZE[1], self.SIZE[0], 3), dtype=np.uint8)
        return Image.fromarray(pixels, "RGB")

    def write(self, directory: Path, name: str, **save_arguments) -> Path:
        path = directory / name
        self.base_image().save(path, "JPEG", quality=95, **save_arguments)
        return path

    def with_orientation(self, directory: Path, orientation: int | None) -> Path:
        from PIL import Image

        exif = Image.Exif()
        if orientation is not None:
            exif[0x0112] = orientation
        return self.write(directory, f"o{orientation}.jpg", exif=exif)

    def old_picture(self, path: Path):
        """What `load_picture` made before: `ImageOps.exif_transpose`, then the same RGB conversion and Lanczos resize."""
        from PIL import Image, ImageOps

        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        scale = oracle.PICTURE_LONG_EDGE / max(image.size)
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        return np.asarray(image.resize(size, Image.Resampling.LANCZOS), dtype=np.uint8)

    def test_every_orientation_gives_the_pixels_of_exif_transpose(self):
        with tempfile.TemporaryDirectory() as tmp:
            for orientation in range(1, 9):
                with self.subTest(orientation=orientation):
                    path = self.with_orientation(Path(tmp), orientation)
                    got = oracle.load_picture(path)
                    np.testing.assert_array_equal(got, self.old_picture(path))
                    swapped = orientation >= 5
                    self.assertEqual(got.shape[:2], (1024, 683) if swapped else (683, 1024))

    def test_the_eight_orientations_differ_from_each_other_so_the_comparison_means_something(self):
        with tempfile.TemporaryDirectory() as tmp:
            pictures = [oracle.load_picture(self.with_orientation(Path(tmp), orientation)) for orientation in range(1, 9)]
        for i in range(8):
            for j in range(i + 1, 8):
                if pictures[i].shape == pictures[j].shape:
                    self.assertFalse(np.array_equal(pictures[i], pictures[j]), (i + 1, j + 1))

    def test_no_orientation_tag_or_a_value_outside_1_to_8_leaves_the_picture_as_it_is(self):
        with tempfile.TemporaryDirectory() as tmp:
            upright = oracle.load_picture(self.with_orientation(Path(tmp), 1))
            for orientation in (None, 0, 9, 65535):
                with self.subTest(orientation=orientation):
                    path = self.with_orientation(Path(tmp), orientation)
                    np.testing.assert_array_equal(oracle.load_picture(path), upright)
                    np.testing.assert_array_equal(oracle.load_picture(path), self.old_picture(path))
            plain = self.write(Path(tmp), "plain.jpg")  # no EXIF block at all
            np.testing.assert_array_equal(oracle.load_picture(plain), upright)

    def test_a_zero_denominator_in_the_exif_does_not_stop_the_load_and_the_orientation_still_applies(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            broken = self.write(directory, "broken.jpg", exif=tiff_with_zero_denominator(6))
            got = oracle.load_picture(broken)
            self.assertEqual(got.shape[:2], (1024, 683), "rotated: the 30 x 20 photo became 20 x 30")
            well_formed = self.with_orientation(directory, 6)  # the same pixels, a clean EXIF block
            np.testing.assert_array_equal(got, oracle.load_picture(well_formed))
            np.testing.assert_array_equal(got, self.old_picture(well_formed))


    def test_the_exif_is_never_written_again_whatever_the_version_of_pillow(self):
        """The zero denominator only stops older Pillow versions; this fails on any version that re-serialises the EXIF."""
        from PIL import Image

        def refuse(*_arguments, **_keywords):
            raise AssertionError("the EXIF block must not be serialised again")

        with tempfile.TemporaryDirectory() as tmp:
            path = self.with_orientation(Path(tmp), 6)
            expected = self.old_picture(path)
            with mock.patch.object(Image.Exif, "tobytes", refuse):
                got = oracle.load_picture(path)
        np.testing.assert_array_equal(got, expected)


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

    def test_pinned_hashes_are_64_lowercase_hex_characters_and_no_two_photos_share_one(self):
        pinned = [(photo["id"], photo["sha256"]) for photo in LISTING["photos"] if photo["sha256"] != oracle.PLACEHOLDER]
        self.assertTrue(pinned, "at least the first photos are pinned")
        for photo_id, digest in pinned:
            with self.subTest(photo=photo_id):
                self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertEqual(len({digest for _, digest in pinned}), len(pinned), "a hash pasted twice")

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

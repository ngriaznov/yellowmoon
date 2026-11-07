"""Tests of the parts of training/depth that need neither torch nor the network (stdlib, plus numpy and Pillow where noted):

    python3 -m unittest training.depth.test_depth         # from the repository root; numpy tests skip without numpy

The conversion and the oracle themselves run in CI (`.github/workflows/models.yml`, job `depth`); here are the file writer, the
tensor table, the size rule, the fixture index, the photo list's rules and the agreement of the pins.
"""

from __future__ import annotations

import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

from training.depth import export, oracle

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

ROOT = Path(__file__).resolve().parents[2]
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
        metadata = {"format": export.FORMAT, "version": "1", "config": json.dumps({"k": [1, 2]})}
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

    def test_f32_tensors_are_narrowed_to_f16_little_endian(self):
        tensors = {"w": Tensor(np.array([0.1, -3.0, 65504.0], dtype=np.float32))}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "w.safetensors"
            export.write_safetensors(path, tensors, {"format": export.FORMAT})
            _, header, data = read_safetensors(path)
        self.assertEqual(header["w"]["dtype"], "F16")
        self.assertEqual(np.frombuffer(data, dtype="<f2").tolist(), np.array([0.1, -3.0, 65504.0], dtype=np.float16).tolist())

    def test_npz_has_the_given_keys_shapes_and_dtypes_and_the_same_bytes_every_time(self):
        arrays = {"picture": np.zeros((4, 6, 3), np.float32), "depth": np.ones((1, 2, 3), np.float32)}
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

    def test_the_fixture_index_lists_every_array_with_dtype_and_shape(self):
        arrays = {"picture": np.zeros((4, 6, 3), np.float32), "input": np.zeros((1, 3, 518, 784), np.float32)}
        index = oracle.describe_arrays(arrays)
        self.assertEqual(index, {"input": {"dtype": "f32", "shape": [1, 3, 518, 784]}, "picture": {"dtype": "f32", "shape": [4, 6, 3]}})
        self.assertEqual(list(index), sorted(index))
        json.dumps(index)
        with self.assertRaises(ValueError):
            oracle.describe_arrays({"x": np.zeros(2, np.float64)})
        with self.assertRaises(ValueError):
            oracle.describe_arrays({"x": np.zeros(2, np.uint8)})

    def test_the_documented_keys_are_the_ones_the_rust_test_reads(self):
        """`crates/ym-ml/tests/depth.rs` reads exactly these keys; the documentation, the probe and the Rust test agree."""
        wanted = {"picture", "input", "depth", "embedded", "output_conv1"}
        wanted |= {f"block_{n:02d}" for n in range(12)} | {f"tap_{k}" for k in range(4)}
        wanted |= {f"layer_{k}" for k in range(1, 5)} | {f"path_{k}" for k in range(1, 5)}
        self.assertEqual(set(oracle.BASE_KEYS) | set(oracle.TRACE_KEYS), wanted)
        self.assertEqual(len(oracle.BASE_KEYS) + len(oracle.TRACE_KEYS), len(wanted), "no key twice")
        pattern = {"block_NN": [f"block_{n:02d}" for n in range(12)], "tap_K": [f"tap_{k}" for k in range(4)],
                   "layer_K": [f"layer_{k}" for k in range(1, 5)], "path_K": [f"path_{k}" for k in range(1, 5)]}
        documented = set()
        for name in oracle.ARRAY_DOC:
            documented.update(pattern.get(name, [name]))
        self.assertEqual(documented, wanted)
        rust = (ROOT / "crates" / "ym-ml" / "tests" / "depth.rs")
        if rust.exists():
            text = rust.read_text(encoding="utf-8")
            for key in ("picture", "input", "embedded", "output_conv1", "depth", "block_", "tap_", "layer_", "path_"):
                self.assertIn(key, text)

    def test_rank_correlation_and_relative_rms(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.assertAlmostEqual(oracle.rank_correlation(a, a * 10 + 3), 1.0)
        self.assertAlmostEqual(oracle.rank_correlation(a, -a), -1.0)
        self.assertEqual(oracle.rank_correlation(np.ones(4), np.arange(4.0)), 1.0)
        self.assertAlmostEqual(oracle.relative_rms(np.array([1.1, 2.2]), np.array([1.0, 2.0])), 0.1, places=6)


class Sizes(unittest.TestCase):
    """`upstream_size` is the upstream `Resize(...).get_size` for 518 / 14 / lower_bound written out; (width, height)."""

    def test_the_network_input_size_for_common_pictures(self):
        cases = {
            (6000, 4000): (784, 518),  # 3:2, 777 px rounds to 56 patches (55.5 goes to the even 56)
            (4000, 6000): (518, 784),
            (1024, 683): (770, 518),  # the 3:2 picture of the oracle (long edge 1024)
            (683, 1024): (518, 770),
            (1024, 768): (686, 518),
            (518, 518): (518, 518),
            (1000, 1000): (518, 518),
            (3000, 1000): (1554, 518),  # a 3:1 panorama: the upstream rule has no cap
            (300, 200): (784, 518),
        }
        for (width, height), want in cases.items():
            with self.subTest(width=width, height=height):
                self.assertEqual(oracle.upstream_size(width, height), want)

    def test_both_sides_are_multiples_of_14_and_at_least_518(self):
        for width, height in ((517, 389), (1920, 1080), (1080, 1920), (2000, 3000), (1, 1), (5000, 100)):
            w, h = oracle.upstream_size(width, height)
            self.assertEqual((w % 14, h % 14), (0, 0), (width, height))
            self.assertGreaterEqual(min(w, h), 518)


class Tables(unittest.TestCase):
    def test_the_tensor_table_has_239_tensors_and_24_785_089_parameters(self):
        shapes = export.expected_tensors()
        self.assertEqual(len(shapes), export.EXPECTED_TENSORS)
        self.assertEqual(export.parameter_count(shapes), export.EXPECTED_PARAMETERS)
        self.assertAlmostEqual(export.parameter_count(shapes) / 1e6, 24.8, places=1)  # "24.8 M parameters" on the model card

    def test_the_names_are_the_ones_the_rust_runtime_reads(self):
        shapes = export.expected_tensors()
        for name in (
            "pretrained.cls_token", "pretrained.pos_embed", "pretrained.patch_embed.proj.weight", "pretrained.blocks.11.ls2.gamma",
            "pretrained.blocks.0.attn.qkv.weight", "pretrained.norm.bias", "depth_head.projects.3.weight", "depth_head.resize_layers.0.weight",
            "depth_head.resize_layers.3.bias", "depth_head.scratch.layer4_rn.weight", "depth_head.scratch.refinenet4.resConfUnit1.conv2.bias",
            "depth_head.scratch.output_conv2.2.weight",
        ):
            self.assertIn(name, shapes)
        self.assertNotIn("depth_head.resize_layers.2.weight", shapes, "layer 2 is the identity")
        self.assertNotIn("depth_head.scratch.layer1_rn.bias", shapes, "the 3 x 3 convolutions to 64 channels have no bias")
        self.assertEqual(shapes["pretrained.pos_embed"], (1, 37 * 37 + 1, 384))
        self.assertEqual(shapes["pretrained.blocks.5.attn.qkv.weight"], (1152, 384))
        self.assertEqual(shapes["depth_head.resize_layers.0.weight"], (48, 48, 4, 4))
        self.assertEqual(shapes["depth_head.scratch.output_conv1.weight"], (32, 64, 3, 3))

    def test_the_configuration_constants_are_the_upstream_ones(self):
        self.assertEqual(export.MODEL_CONFIG, {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]})
        self.assertEqual(export.INTERMEDIATE_LAYERS, [2, 5, 8, 11])
        self.assertEqual((export.INPUT_SIZE, export.PATCH_SIZE), (518, 14))
        self.assertEqual(export.INPUT_SIZE % export.PATCH_SIZE, 0)

    def test_check_config_refuses_a_different_model(self):
        config = {
            "encoder": {"patch_size": 14, "embed_dim": 384, "depth": 12, "num_heads": 6, "mlp_ratio": 4, "mlp_activation": "gelu (exact, erf)", "qkv_bias": True,
                        "layer_scale": True, "layer_norm_eps": 1e-6, "num_register_tokens": 0, "intermediate_layers": [2, 5, 8, 11],
                        "pos_embed_interpolation": {"antialias": False, "offset": 0.1}},
            "head": {"features": 64, "out_channels": [48, 96, 192, 384], "use_clstoken": False, "fusion": {"align_corners": True}},
        }
        export.check_config(config)
        config["encoder"]["embed_dim"] = 768
        with self.assertRaises(SystemExit) as stop:
            export.check_config(config)
        self.assertIn("embed_dim", str(stop.exception))


class Pins(unittest.TestCase):
    def test_the_checkpoint_pin_is_a_sha256_of_a_revision_pinned_url(self):
        self.assertRegex(export.CKPT_SHA256, r"^[0-9a-f]{64}$")
        self.assertRegex(export.CKPT_REVISION, r"^[0-9a-f]{40}$")
        self.assertRegex(export.UPSTREAM_COMMIT, r"^[0-9a-f]{40}$")
        self.assertIn(export.CKPT_REVISION, export.CKPT_URL)
        self.assertTrue(export.CKPT_URL.endswith("/depth_anything_v2_vits.pth"))
        self.assertGreater(export.CKPT_BYTES, 90_000_000)

    def test_the_pins_of_the_scripts_and_the_workflow_agree(self):
        workflow = (ROOT / ".github" / "workflows" / "models.yml").read_text(encoding="utf-8")
        self.assertIn(f"DEPTH_COMMIT: {export.UPSTREAM_COMMIT}", workflow)
        self.assertIn(export.UPSTREAM_COMMIT, (Path(__file__).with_name("requirements.txt")).read_text(encoding="utf-8"))
        self.assertIn("RELEASE_TAG: models-depth-anything-v2-small-1", workflow)
        self.assertIn(f"WEIGHTS: {export.OUT_NAME}", workflow)
        self.assertIn("FIXTURES: depth-oracle-fixtures-1.tar.zst", workflow)
        self.assertIn(export.OUT_NAME, (ROOT / "tools" / "fetch" / "depth.sh").read_text(encoding="utf-8"))
        self.assertIn(export.OUT_NAME, (ROOT / "data" / "models" / "depth" / "SHA256SUMS").read_text(encoding="utf-8"))

    def test_the_sam_and_depth_jobs_are_selected_by_the_branch_name(self):
        workflow = (ROOT / ".github" / "workflows" / "models.yml").read_text(encoding="utf-8")
        self.assertIn("startsWith(github.ref_name, 'models/depth-')", workflow)
        self.assertIn("group: models-depth", workflow)
        self.assertIn("group: models-sam2", workflow)

    def test_the_torch_versions_of_the_two_jobs_are_the_same(self):
        workflow = (ROOT / ".github" / "workflows" / "models.yml").read_text(encoding="utf-8")
        self.assertEqual(workflow.count('TORCH: "2.5.1+cpu"'), 2)
        self.assertEqual(workflow.count('TORCHVISION: "0.20.1+cpu"'), 2)

    def test_the_metadata_header_names_what_the_rust_loader_checks(self):
        self.assertEqual(export.FORMAT, "yellowmoon-depth")
        self.assertEqual(export.VERSION, "1")
        self.assertEqual(export.OUT_NAME, "depth_anything_v2_vits.f16.safetensors")


class Downloader(unittest.TestCase):
    def photo(self, directory: Path, data: bytes, sha256: str | None = None) -> dict:
        source = directory / "source.jpg"
        source.write_bytes(data)
        return {"id": "t", "url": source.as_uri(), "sha256": sha256 or hashlib.sha256(data).hexdigest()}

    def test_a_download_is_checked_and_kept(self):
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(oracle.time, "sleep"):
            directory = Path(tmp)
            path = oracle.fetch_photo(self.photo(directory, b"jpeg bytes"), directory / "cache")
            self.assertEqual(path.read_bytes(), b"jpeg bytes")

    def test_a_file_with_another_hash_is_refused(self):
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(oracle.time, "sleep"):
            directory = Path(tmp)
            with self.assertRaises(SystemExit) as stop:
                oracle.fetch_photo(self.photo(directory, b"changed", sha256="0" * 64), directory / "cache")
            self.assertIn("changed", str(stop.exception))
            self.assertFalse((directory / "cache" / "t.jpg").exists())

    def test_an_unpinned_photo_needs_the_hash_commons_states(self):
        from unittest import mock

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


class PhotoList(unittest.TestCase):
    def test_there_are_20_photos_with_unique_ids_and_urls(self):
        photos = LISTING["photos"]
        self.assertEqual(len(photos), oracle.PHOTO_COUNT)
        self.assertEqual(len({p["id"] for p in photos}), len(photos))
        self.assertEqual(len({p["url"] for p in photos}), len(photos))

    def test_every_photo_has_clear_terms_a_commons_page_and_a_pin(self):
        for photo in LISTING["photos"]:
            with self.subTest(photo=photo["id"]):
                self.assertIn(photo["license"], ALLOWED_LICENCES)
                self.assertTrue(photo["url"].startswith("https://upload.wikimedia.org/wikipedia/commons/"), photo["url"])
                self.assertNotIn("?", photo["url"])
                self.assertTrue(photo["page"].startswith("https://commons.wikimedia.org/wiki/File:"), photo["page"])
                self.assertTrue(photo["author"].strip())
                self.assertTrue(photo["title"].lower().endswith((".jpg", ".jpeg")))
                self.assertRegex(photo["commons_sha1"], r"^[0-9a-f]{40}$")
                self.assertRegex(photo["sha256"], r"^([0-9a-f]{64}|TBD-after-first-run)$")
                self.assertTrue(photo["categories"], "each photo says what it is there for")

    def test_pinned_hashes_are_unique(self):
        pinned = [photo["sha256"] for photo in LISTING["photos"] if photo["sha256"] != oracle.PLACEHOLDER]
        self.assertEqual(len(set(pinned)), len(pinned), "a hash pasted twice")

    def test_the_gate_s_scene_types_are_all_covered(self):
        covered = {category for photo in LISTING["photos"] for category in photo["categories"]}
        for needed in ("portrait", "hair", "foliage", "sky", "glass", "interior", "shallow-focus", "flat"):
            self.assertIn(needed, covered)

    def test_three_photos_carry_the_intermediates_and_they_differ_in_shape(self):
        deep = [photo for photo in LISTING["photos"] if photo.get("intermediates")]
        self.assertEqual(len(deep), 3)
        sizes = {oracle.upstream_size(photo["width"], photo["height"]) for photo in deep}
        self.assertEqual(len(sizes), 3, "three different grids")
        orientations = {photo["width"] > photo["height"] for photo in deep}
        self.assertEqual(orientations, {True, False}, "a landscape and a portrait photo")


if __name__ == "__main__":
    unittest.main()

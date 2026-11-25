"""Tests of the parts of training/eomt/export.py that need no torch and no network (``python -m unittest training.eomt.test_eomt``)."""

from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from training.eomt import export


class ExpectedTensors(unittest.TestCase):
    def test_the_list_has_the_counts_of_the_checkpoint(self):
        shapes = export.expected_tensors()
        self.assertEqual(len(shapes), export.EXPECTED_TENSORS)
        self.assertEqual(export.parameter_count(shapes), export.EXPECTED_PARAMETERS)

    def test_the_patch_is_16_and_the_grid_40(self):
        shapes = export.expected_tensors()
        self.assertEqual(shapes["embeddings.patch_embeddings.projection.weight"], (384, 3, 16, 16))
        self.assertEqual(shapes["embeddings.position_embeddings.weight"], (1600, 384))
        self.assertEqual(export.IMAGE, 640)

    def test_every_block_has_the_same_tensors(self):
        shapes = export.expected_tensors()
        per_block = {name.split(".", 2)[2] for name in shapes if name.startswith("layers.0.")}
        for n in range(1, export.LAYERS):
            self.assertEqual({name.split(".", 2)[2] for name in shapes if name.startswith(f"layers.{n}.")}, per_block)
        self.assertEqual(len(per_block), 18)


class WriteSafetensors(unittest.TestCase):
    def tensors(self):
        rng = np.random.default_rng(3)
        return {
            "b.weight": rng.normal(size=(3, 4)).astype(np.float32),
            "a.bias": rng.normal(size=(5,)).astype(np.float32),
            "c.scalar": np.array([0.5], np.float32),
        }

    def test_the_file_is_byte_identical_on_every_run_and_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            one, two = Path(tmp) / "one", Path(tmp) / "two"
            meta = {"z": "1", "a": "2", "format": export.FORMAT}
            export.write_safetensors(one, self.tensors(), meta)
            export.write_safetensors(two, self.tensors(), dict(reversed(list(meta.items()))))
            self.assertEqual(one.read_bytes(), two.read_bytes())
            raw = one.read_bytes()
            (length,) = struct.unpack("<Q", raw[:8])
            self.assertEqual(length % 8, 0)
            header = json.loads(raw[8 : 8 + length])
            self.assertEqual(list(header["__metadata__"]), ["a", "format", "z"])
            self.assertEqual(sorted(k for k in header if k != "__metadata__"), ["a.bias", "b.weight", "c.scalar"])
            self.assertEqual(header["b.weight"]["dtype"], "F16")
            from safetensors.numpy import load_file

            back = load_file(str(one))
            for name, array in self.tensors().items():
                np.testing.assert_array_equal(back[name], array.astype(np.float16))


class CheckConfig(unittest.TestCase):
    def good(self):
        return {
            "hidden_size": 384, "num_hidden_layers": 12, "num_attention_heads": 6, "mlp_ratio": 4, "hidden_act": "gelu",
            "layer_norm_eps": 1e-6, "layerscale_value": 1e-5, "use_swiglu_ffn": False, "patch_size": 16, "image_size": 640,
            "grid_size": [40, 40], "num_positions": 1600, "num_register_tokens": 4, "num_queries": 200, "num_blocks": 3,
            "num_upscale_blocks": 2, "num_labels": 133, "attn_mask_probs": [0.0, 0.0, 0.0],
        }

    def test_the_documented_model_passes(self):
        export.check_config(self.good())

    def test_another_patch_size_or_a_masked_checkpoint_is_refused(self):
        for key, value in (("patch_size", 14), ("attn_mask_probs", [0.0, 0.5, 0.0]), ("num_register_tokens", 0)):
            config = self.good()
            config[key] = value
            with self.assertRaises(SystemExit, msg=key):
                export.check_config(config)


if __name__ == "__main__":
    unittest.main()

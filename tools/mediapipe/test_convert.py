"""Tests of tools/mediapipe/convert.py that need no network and no model file:
``python -m unittest discover -s tools/mediapipe -p 'test_*.py'`` (the ``mediapipe`` job of models.yml runs them before converting).

They check the safetensors writer (the reason the output bytes are always the same), the table of pins, and that the names in
the table are the names ``data/models/mediapipe/SHA256SUMS`` and ``tools/fetch/mediapipe.sh`` use.
"""

from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

try:
    import numpy as np

    import convert
except ImportError as error:  # numpy or tflite is not installed: nothing to test here
    raise unittest.SkipTest(f"tools/mediapipe needs its requirements ({error})") from error


class Writer(unittest.TestCase):
    def tensors(self):
        return {
            "t9": np.arange(6, dtype="<f2").reshape(2, 3),
            "t10": np.arange(4, dtype="<f4").reshape(2, 2) * 0.5,
        }

    def test_the_same_input_gives_the_same_bytes_whatever_the_order(self):
        meta = {"b": "2", "a": "1"}
        first = convert.write_safetensors(self.tensors(), meta)
        reordered = dict(reversed(list(self.tensors().items())))
        self.assertEqual(first, convert.write_safetensors(reordered, {"a": "1", "b": "2"}))

    def test_the_file_is_a_valid_safetensors_file(self):
        blob = convert.write_safetensors(self.tensors(), {"graph": "{}"})
        header_len = int.from_bytes(blob[:8], "little")
        self.assertEqual(header_len % 8, 0)
        header = json.loads(blob[8 : 8 + header_len])
        self.assertEqual(header["__metadata__"], {"graph": "{}"})
        self.assertEqual(header["t9"]["dtype"], "F16")
        self.assertEqual(header["t10"]["dtype"], "F32")
        self.assertEqual(header["t9"]["shape"], [2, 3])
        body = blob[8 + header_len :]
        start, end = header["t10"]["data_offsets"]
        self.assertEqual(np.frombuffer(body[start:end], dtype="<f4").tolist(), [0.0, 0.5, 1.0, 1.5])
        start, end = header["t9"]["data_offsets"]
        self.assertEqual(np.frombuffer(body[start:end], dtype="<f2").tolist(), [0, 1, 2, 3, 4, 5])
        # Tensors follow each other with no gap, in name order.
        self.assertEqual(header["t10"]["data_offsets"][0], 0)
        self.assertEqual(header["t10"]["data_offsets"][1], header["t9"]["data_offsets"][0])
        self.assertEqual(len(body), header["t9"]["data_offsets"][1])

    def test_safetensors_reads_it_when_it_is_installed(self):
        try:
            from safetensors import safe_open
        except ImportError:
            self.skipTest("the safetensors package is not installed")
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.safetensors"
            path.write_bytes(convert.write_safetensors(self.tensors(), {"k": "v"}))
            with safe_open(str(path), framework="np") as f:
                self.assertEqual(sorted(f.keys()), ["t10", "t9"])
                self.assertEqual(f.metadata(), {"k": "v"})
                self.assertEqual(f.get_tensor("t9").tolist(), [[0, 1, 2], [3, 4, 5]])


class Pins(unittest.TestCase):
    def test_every_model_has_real_pins_and_an_https_url(self):
        for name, spec in convert.MODELS.items():
            self.assertRegex(spec["sha256"], r"^[0-9a-f]{64}$", name)
            self.assertTrue(spec["url"].startswith("https://storage.googleapis.com/mediapipe-models/"), name)
            self.assertEqual(spec["member"] is None, spec["member_sha256"] is None, name)
            if spec["member_sha256"]:
                self.assertRegex(spec["member_sha256"], r"^[0-9a-f]{64}$", name)

    def test_the_outputs_are_the_files_that_sha256sums_and_the_fetch_script_know(self):
        sums = (REPO / "data" / "models" / "mediapipe" / "SHA256SUMS").read_text().split("\n")
        listed = {line.split()[1] for line in sums if line.strip()}
        for line in sums:
            if line.strip():
                self.assertRegex(line.split()[0], r"^[0-9a-f]{64}$")
        self.assertEqual({spec["out"] for spec in convert.MODELS.values()}, listed)
        script = (REPO / "tools" / "fetch" / "mediapipe.sh").read_text()
        for out in listed:
            self.assertIn(out, script)

    def test_a_download_with_the_wrong_hash_is_refused(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "model.tflite"
            source.write_bytes(b"some bytes")
            cache = Path(tmp) / "cache"
            cache.mkdir()
            (cache / "model.tflite").write_bytes(b"cached but wrong")
            with self.assertRaises(SystemExit) as raised:
                convert.download(source.as_uri(), "0" * 64, cache)
            self.assertIn("nothing was converted", str(raised.exception))
            good = convert.sha256_of(b"some bytes")
            self.assertEqual(convert.download(source.as_uri(), good, cache), b"some bytes")
            # Now the cache holds the good file and needs no network.
            source.unlink()
            self.assertEqual(convert.download(source.as_uri(), good, cache), b"some bytes")


if __name__ == "__main__":
    unittest.main()

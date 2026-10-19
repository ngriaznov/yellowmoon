"""Tests of the command line of ``compare_metadata.py`` (stdlib only: ``python3 -I -m unittest discover -s tools/oracle``).

The M0-04 metadata gate runs our reader as a separate process. The reader is the ``ym`` binary
(decision 2026-10-12); a fresh ``cargo build -p ym-cli`` writes ``target/debug/ym`` and nothing called ``yellowmoon``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "compare_metadata.py"
REPO = HERE.parents[1]

spec = importlib.util.spec_from_file_location("compare_metadata", SCRIPT)
compare_metadata = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare_metadata)


def fake_ym(directory: Path, name: str = "ym") -> Path:
    """A stand-in for ``ym``: ``info --json FILE`` prints one JSON line that says it was called that way."""
    path = directory / name
    path.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = info ] && [ "$2" = --json ]; then\n'
        "  printf '{\"status\": \"fake\", \"argv\": [\"%s\", \"%s\", \"%s\", \"%s\", \"%s\"]}\\n' \"$1\" \"$2\" \"$3\" \"$4\" \"$5\"\n"
        "  exit 0\n"
        "fi\n"
        "exit 3\n"
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


class Command(unittest.TestCase):
    def test_the_default_binary_is_the_one_cargo_builds(self):
        args = compare_metadata.build_parser().parse_args(["file.ARW"])
        self.assertEqual(Path(args.ym), REPO / "target" / "debug" / "ym")

    def test_ym_flag_selects_the_binary(self):
        args = compare_metadata.build_parser().parse_args(["--ym", "/x/ym", "file.ARW"])
        self.assertEqual(args.ym, "/x/ym")

    def test_the_old_flag_name_still_works(self):
        args = compare_metadata.build_parser().parse_args(["--yellowmoon", "/x/ym", "file.ARW"])
        self.assertEqual(args.ym, "/x/ym")

    def test_ours_runs_info_json_on_the_given_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = fake_ym(Path(tmp))
            result = compare_metadata.ours(str(binary), "photo.ARW")
        self.assertEqual(result["argv"], ["info", "--json", "photo.ARW", "--apple-auto", "off"])

    def test_a_missing_binary_is_a_usage_error_that_names_ym(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "nope")
            proc = subprocess.run(
                [sys.executable, "-I", str(SCRIPT), "--ym", missing, "file.ARW"], capture_output=True, text=True
            )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("ym binary not found", proc.stdout)
        self.assertNotIn("yellowmoon", proc.stdout)


if __name__ == "__main__":
    unittest.main()

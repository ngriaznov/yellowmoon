"""Tests of manifest_table.py (``python3 -I -m unittest discover -s tools/golden``; stdlib only)."""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("manifest_table", HERE / "manifest_table.py")
manifest_table = importlib.util.module_from_spec(spec)
sys.modules["manifest_table"] = manifest_table
spec.loader.exec_module(manifest_table)

MANIFEST = json.loads((HERE / "manifest.json").read_text())


class Table(unittest.TestCase):
    def test_the_default_set_has_one_row_per_file_with_body_light_iso_and_tier(self):
        entries = manifest_table.select(MANIFEST, "default")
        self.assertEqual(len(entries), 61)
        table = manifest_table.per_file(entries).splitlines()
        self.assertEqual(len(table), 2 + 61)
        self.assertTrue(table[0].startswith("| File | Body | Light | ISO |"))
        first = entries[0]
        self.assertIn(first["file"].rsplit("/", 1)[-1], table[2])
        self.assertIn(first["body"], table[2])
        self.assertIn("ours (generated)", table[2])

    def test_the_summary_counts_add_up_to_the_files(self):
        entries = manifest_table.select(MANIFEST, "default")
        rows = manifest_table.summary(entries).splitlines()[2:]
        total = sum(sum(int(c) for c in row.split("|")[2:-3]) for row in rows)
        self.assertEqual(total, len(entries))

    def test_every_entry_has_a_tier_the_table_knows(self):
        for entry in MANIFEST["entries"]:
            self.assertIn(entry["tier"], manifest_table.TIER_TEXT)


if __name__ == "__main__":
    unittest.main()

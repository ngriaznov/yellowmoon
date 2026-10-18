"""Rows: validation, reading, augmentations and scenes (``python3 -m unittest discover -s training/autoset/tests -t .``)."""

from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path


from training.autoset import rows as rw
from training.autoset import synthetic


def _obj(file: str, sha: str, *, body: str | None = "ILCE-7M4", time: str | None = "2024-05-01T10:00:00",
         k: float = 0.0, key: float = 0.0) -> dict:
    """A minimal valid row (features zero, no label tables)."""
    return {"file": file, "sha256": sha, "source": "test", "licence": "CC0-1.0", "body": body, "capture_time": time,
            "aug_k": k, "features": [0.0] * 39, "key": key, "meta": {}, "g3": None, "label_ev": None,
            "contrast_fit": None}


class Validation(unittest.TestCase):
    def setUp(self) -> None:
        self.base, _ = synthetic.make_dataset(2)
        self.row = self.base[0]

    def test_a_synthetic_row_parses(self) -> None:
        r = rw.parse_row(self.row, 1)
        self.assertEqual(r.features.shape, (39,))
        self.assertTrue(r.is_base)
        self.assertIsNotNone(r.g3)
        self.assertEqual(len(r.g3.scene_q), 999)
        self.assertEqual(r.contrast_fit.ours.shape, (9, 7))

    def test_wrong_feature_length_is_rejected(self) -> None:
        obj = copy.deepcopy(self.row)
        obj["features"] = obj["features"][:38]
        with self.assertRaises(rw.RowError) as ctx:
            rw.parse_row(obj)
        self.assertEqual(ctx.exception.reason, "features_length")

    def test_non_finite_features_and_key_are_rejected(self) -> None:
        obj = copy.deepcopy(self.row)
        obj["features"][3] = math.nan
        with self.assertRaises(rw.RowError) as ctx:
            rw.parse_row(obj)
        self.assertEqual(ctx.exception.reason, "non_finite_features")
        obj = copy.deepcopy(self.row)
        obj["key"] = math.inf
        with self.assertRaises(rw.RowError) as ctx:
            rw.parse_row(obj)
        self.assertEqual(ctx.exception.reason, "non_finite_key")

    def test_a_non_finite_value_in_a_table_is_rejected(self) -> None:
        obj = copy.deepcopy(self.row)
        obj["g3"]["jpeg_q"][10] = math.nan
        with self.assertRaises(rw.RowError) as ctx:
            rw.parse_row(obj)
        self.assertEqual(ctx.exception.reason, "non_finite_g3")

    def test_booleans_are_not_numbers(self) -> None:
        obj = copy.deepcopy(self.row)
        obj["features"][0] = True
        with self.assertRaises(rw.RowError) as ctx:
            rw.parse_row(obj)
        self.assertEqual(ctx.exception.reason, "bad_type")

    def test_the_grids_must_be_the_canonical_ones(self) -> None:
        obj = copy.deepcopy(self.row)
        obj["g3"]["grid"] = "linspace(0, 1, 999)"
        with self.assertRaises(rw.RowError) as ctx:
            rw.parse_row(obj)
        self.assertEqual(ctx.exception.reason, "bad_g3")
        obj = copy.deepcopy(self.row)
        obj["contrast_fit"]["grid"] = [-40, -30, -20, -10, 0, 10, 20, 30, 41]
        with self.assertRaises(rw.RowError) as ctx:
            rw.parse_row(obj)
        self.assertEqual(ctx.exception.reason, "bad_contrast_fit")

    def test_an_augmentation_cannot_carry_label_tables(self) -> None:
        obj = copy.deepcopy(self.row)
        obj["aug_k"] = 1.0
        with self.assertRaises(rw.RowError) as ctx:
            rw.parse_row(obj)
        self.assertEqual(ctx.exception.reason, "label_on_augmentation")

    def test_a_label_needs_its_tables(self) -> None:
        obj = copy.deepcopy(self.row)
        obj["g3"] = None
        with self.assertRaises(rw.RowError) as ctx:
            rw.parse_row(obj)
        self.assertEqual(ctx.exception.reason, "label_without_tables")

    def test_a_bad_capture_time_is_rejected(self) -> None:
        obj = copy.deepcopy(self.row)
        obj["capture_time"] = "01/05/2024 10:00"
        with self.assertRaises(rw.RowError) as ctx:
            rw.parse_row(obj)
        self.assertEqual(ctx.exception.reason, "bad_capture_time")


class Reading(unittest.TestCase):
    def test_read_keeps_augmentations_and_counts_what_it_drops(self) -> None:
        base, _ = synthetic.make_dataset(2)
        files = [o for o in base if o["aug_k"] == 0.0]
        augs = [o for o in base if o["aug_k"] != 0.0]
        orphan = copy.deepcopy(augs[0])
        orphan["file"] = "nobody.ARW"
        lines = [json.dumps(o) for o in files + augs + [orphan]] + ["{not json", "", json.dumps(files[0])]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            read = rw.read_rows(path)
        self.assertEqual(len(read.rows), len(files) + len(augs))
        self.assertEqual(read.rejected, {"orphan_augmentation": 1, "not_json": 1, "duplicate_file": 1})
        self.assertEqual(sum(1 for r in read.rows if r.is_base), len(files))

    def test_read_objects_reports_the_line_of_a_rejected_row(self) -> None:
        bad = copy.deepcopy(synthetic.make_rows(1)[0])
        bad["features"] = bad["features"][:5]
        read = rw.read_objects([bad])
        self.assertEqual(read.rows, [])
        self.assertEqual(read.rejected, {"features_length": 1})
        self.assertIn("rows:1", read.examples["features_length"][0])


class Scenes(unittest.TestCase):
    def _scenes(self, objs: list[dict]) -> list[int]:
        return rw.assign_scenes(rw.read_objects(objs).rows).tolist()

    def test_a_burst_within_ten_seconds_is_one_scene(self) -> None:
        objs = [_obj("a.ARW", "1" * 64, time="2024-05-01T10:00:00"), _obj("b.ARW", "2" * 64, time="2024-05-01T10:00:08")]
        self.assertEqual(self._scenes(objs), [0, 0])

    def test_a_longer_gap_starts_a_new_scene(self) -> None:
        objs = [_obj("a.ARW", "1" * 64, time="2024-05-01T10:00:00"), _obj("b.ARW", "2" * 64, time="2024-05-01T10:00:11")]
        self.assertEqual(self._scenes(objs), [0, 1])

    def test_chains_in_time_order_and_needs_the_same_body(self) -> None:
        objs = [
            _obj("a.ARW", "1" * 64, time="2024-05-01T10:00:00"),
            _obj("b.ARW", "2" * 64, time="2024-05-01T10:00:08"),
            _obj("c.ARW", "3" * 64, time="2024-05-01T10:00:16"),
            _obj("d.ARW", "4" * 64, body="ILCE-1", time="2024-05-01T10:00:17"),
        ]
        self.assertEqual(self._scenes(objs), [0, 0, 0, 1])

    def test_files_without_a_time_are_their_own_scenes(self) -> None:
        objs = [_obj("a.ARW", "1" * 64, time=None), _obj("b.ARW", "2" * 64, time=None)]
        self.assertEqual(self._scenes(objs), [0, 1])

    def test_identical_sha256_prefixes_join(self) -> None:
        sha = "abcdef0123456789" * 4
        objs = [_obj("a.ARW", sha, time=None), _obj("b.ARW", sha[:16] + "f" * 48, time=None)]
        self.assertEqual(self._scenes(objs), [0, 0])

    def test_augmentations_join_their_file_whatever_the_time(self) -> None:
        objs = [_obj("a.ARW", "1" * 64, time=None), _obj("a.ARW", "1" * 64, time=None, k=1.0, key=1.0),
                _obj("b.ARW", "2" * 64, time=None)]
        self.assertEqual(self._scenes(objs), [0, 0, 1])


if __name__ == "__main__":
    unittest.main()

"""Tests of blind_look.py: the randomisation, the crop windows, the page and the scoring. No `ym` and no ARW
files are needed: `python3 -I -m unittest discover -s tools/eval/demosaic -p 'test_*.py'`."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import blind_look as bl  # noqa: E402


class AssignLabels(unittest.TestCase):
    ITEMS = [f"real/photo{i:02d}" for i in range(9)] + [f"synthetic/scene{i:02d}" for i in range(61)]

    def test_balanced_and_complete(self) -> None:
        got = bl.assign_labels(self.ITEMS, "s", "menon", "adaptive_ratio")
        self.assertEqual(set(got), set(self.ITEMS))
        for labels in got.values():
            self.assertEqual(sorted(labels.values()), ["adaptive_ratio", "menon"])
            self.assertEqual(sorted(labels), ["A", "B"])
        baseline_first = sum(1 for v in got.values() if v["A"] == "menon")
        self.assertEqual(baseline_first, len(self.ITEMS) // 2)

    def test_deterministic_and_independent_of_input_order(self) -> None:
        a = bl.assign_labels(self.ITEMS, "s", "menon", "adaptive_ratio")
        b = bl.assign_labels(list(reversed(self.ITEMS)), "s", "menon", "adaptive_ratio")
        self.assertEqual(a, b)

    def test_depends_on_the_seed(self) -> None:
        a = bl.assign_labels(self.ITEMS, "one", "menon", "adaptive_ratio")
        b = bl.assign_labels(self.ITEMS, "two", "menon", "adaptive_ratio")
        self.assertNotEqual(a, b)

    def test_the_methods_must_differ(self) -> None:
        with self.assertRaises(ValueError):
            bl.assign_labels(["a"], "s", "menon", "menon")


def scene() -> np.ndarray:
    """640 x 480 float RGB in [0, 1]: a flat mid grey frame with four distinct regions."""
    rng = np.random.default_rng(7)
    img = np.full((480, 640, 3), 0.5, dtype=np.float32)
    # Edges: wide black and white bars (strong gradient only at the bar borders), top left.
    for i in range(0, 192, 32):
        img[64:192, 64 + i : 64 + i + 16] = 0.95
        img[64:192, 80 + i : 80 + i + 16] = 0.08
    # Fine texture: low-amplitude noise around mid grey, top right (no strong gradients).
    img[64:192, 384:576] += rng.normal(0.0, 0.03, size=(128, 192, 1)).astype(np.float32)
    # Saturated colour boundaries: red and blue patches, bottom left.
    img[288:416, 64:128] = (0.95, 0.05, 0.05)
    img[288:416, 128:192] = (0.05, 0.1, 0.95)
    img[288:416, 192:256] = (0.05, 0.9, 0.1)
    # Shadows: dark noisy region, bottom right.
    img[288:416, 384:576] = 0.06 + rng.normal(0.0, 0.015, size=(128, 192, 1)).astype(np.float32)
    return np.clip(img, 0.0, 1.0)


class Windows(unittest.TestCase):
    def test_cell_features_shape_and_flat_frame(self) -> None:
        flat = np.full((100, 130, 3), 0.4, dtype=np.float32)
        cells = bl.cell_features(flat)
        self.assertEqual(cells["g"].shape, (100 // bl.CELL, 130 // bl.CELL))
        self.assertTrue(np.allclose(cells["g"], 0.0))
        self.assertTrue(np.allclose(cells["strong"], 0.0))
        self.assertTrue(np.allclose(cells["sat"], 0.0))
        self.assertTrue(np.allclose(cells["lum"] / (bl.CELL * bl.CELL), 0.4, atol=1e-5))

    def test_the_windows_land_on_their_regions_and_are_disjoint(self) -> None:
        img = scene()
        cells = bl.cell_features(img)
        wh, ww = 4, 5  # 128 x 160 px
        got = bl.pick_windows(cells, wh, ww)
        self.assertEqual(set(got), set(bl.KINDS))

        def centre(kind: str) -> tuple[float, float]:
            cy, cx = got[kind]
            return (cy + wh / 2) * bl.CELL, (cx + ww / 2) * bl.CELL

        def inside(kind: str, y0: int, y1: int, x0: int, x1: int) -> bool:
            y, x = centre(kind)
            return y0 <= y <= y1 and x0 <= x <= x1

        self.assertTrue(inside("edges", 64, 192, 64, 256), f"edges at {centre('edges')}")
        self.assertTrue(inside("texture", 64, 192, 384, 576), f"texture at {centre('texture')}")
        self.assertTrue(inside("colour", 288, 416, 64, 256), f"colour at {centre('colour')}")
        self.assertTrue(inside("shadows", 288, 416, 384, 576), f"shadows at {centre('shadows')}")
        kinds = list(got)
        for i, a in enumerate(kinds):
            for b in kinds[i + 1 :]:
                (ay, ax), (by, bx) = got[a], got[b]
                self.assertTrue(abs(ay - by) >= wh or abs(ax - bx) >= ww, f"{a} and {b} overlap: {got[a]} {got[b]}")
        for cy, cx in got.values():
            self.assertGreaterEqual(cy, 1)
            self.assertGreaterEqual(cx, 1)
            self.assertLessEqual(cy + wh, cells["g"].shape[0] - 1)
            self.assertLessEqual(cx + ww, cells["g"].shape[1] - 1)

    def test_the_choice_is_deterministic(self) -> None:
        cells = bl.cell_features(scene())
        self.assertEqual(bl.pick_windows(cells, 4, 5), bl.pick_windows(cells, 4, 5))

    def test_a_flat_frame_still_gives_four_windows(self) -> None:
        flat = np.full((320, 320, 3), 0.5, dtype=np.float32)
        got = bl.pick_windows(bl.cell_features(flat), 3, 3)
        self.assertEqual(set(got), set(bl.KINDS))

    def test_a_window_that_does_not_fit_is_an_error(self) -> None:
        cells = bl.cell_features(np.zeros((64, 64, 3), dtype=np.float32))
        with self.assertRaises(ValueError):
            bl.pick_windows(cells, 5, 5)

    def test_window_size(self) -> None:
        self.assertEqual(bl.window_size_cells(6000, 4000), (320 // bl.CELL, 480 // bl.CELL))
        self.assertIsNone(bl.window_size_cells(64, 64))
        wh, ww = bl.window_size_cells(300, 200)  # type: ignore[misc]
        self.assertLessEqual(ww * bl.CELL * 3, 300)
        self.assertLessEqual(wh * bl.CELL * 3, 200)

    def test_strips_do_not_change_the_numbers(self) -> None:
        img = scene()
        whole = bl.cell_features(img)
        old = bl.STRIP
        try:
            bl.STRIP = bl.CELL
            strips = bl.cell_features(img)
        finally:
            bl.STRIP = old
        for k in whole:
            np.testing.assert_allclose(whole[k], strips[k], rtol=1e-5, atol=1e-3)


class Page(unittest.TestCase):
    def entries(self) -> dict:
        def entry(name: str) -> dict:
            return {
                "name": name,
                "width": 6000,
                "height": 4000,
                "A": "menon",
                "B": "adaptive_ratio",
                "files": {"full_A": {"path": f"images/real/{name}/A.jpg"}, "full_B": {"path": f"images/real/{name}/B.jpg"}},
                "crops": {
                    "edges": {"A": f"images/real/{name}/edges_A.png", "B": f"images/real/{name}/edges_B.png", "box_xyxy": [0, 0, 1, 1]}
                },
            }

        return {"real/ILCE-7M3": entry("ILCE-7M3"), "real/NEX-7": entry("NEX-7")}

    def test_the_page_names_no_method(self) -> None:
        page = bl.index_html(self.entries())
        for method in bl.METHODS:
            self.assertNotIn(method, page.lower())
        self.assertNotIn("menon", page.lower())
        self.assertNotIn("baseline", page.lower())
        self.assertNotIn("candidate", page.lower())
        self.assertIn("images/real/ILCE-7M3/A.jpg", page)
        self.assertIn("images/real/ILCE-7M3/B.jpg", page)
        self.assertIn('data-id="real/ILCE-7M3/full"', page)
        self.assertIn('data-id="real/ILCE-7M3/edges"', page)

    def test_identical_pairs_are_left_out_of_the_page(self) -> None:
        entries = self.entries()
        entries["real/NEX-7"]["difference"] = {"identical": True}
        entries["real/ILCE-7M3"]["difference"] = {"identical": False}
        page = bl.index_html(entries)
        self.assertIn("ILCE-7M3", page)
        self.assertNotIn("images/real/NEX-7/", page)
        self.assertIn("1 files are left out", page)

    def test_the_picks_have_the_item_in_their_id(self) -> None:
        page = bl.index_html(self.entries())
        self.assertEqual(page.count('class="picks"'), 4)  # two files, a full frame and an edges crop each


class Scoring(unittest.TestCase):
    KEY = {
        "baseline": "menon",
        "candidate": "adaptive_ratio",
        "items": {
            "real/a": {"A": "menon", "B": "adaptive_ratio"},
            "real/b": {"A": "adaptive_ratio", "B": "menon"},
            "synthetic/c": {"A": "menon", "B": "adaptive_ratio"},
        },
    }

    def test_picks_are_mapped_through_the_key(self) -> None:
        picks = {
            "real/a/full": "B",  # candidate
            "real/b/full": "B",  # baseline
            "real/b/edges": "A",  # candidate
            "synthetic/c/full": "same",
            "synthetic/c/colour": "A",  # baseline
            "real/zzz/full": "A",  # not in the key
        }
        result = bl.score(self.KEY, picks)
        t = result["tallies"]
        self.assertEqual(t["all"]["candidate"], 2)
        self.assertEqual(t["all"]["baseline"], 2)
        self.assertEqual(t["all"]["same"], 1)
        self.assertEqual(t["set:real"], {"baseline": 1, "candidate": 2, "same": 0, "sign_test_p": t["set:real"]["sign_test_p"]})
        self.assertEqual(t["kind:full"]["same"], 1)
        self.assertEqual(result["unknown_picks"], ["real/zzz/full"])
        self.assertIn("baseline menon against candidate adaptive_ratio", bl.format_score(result))

    def test_sign_test(self) -> None:
        self.assertEqual(bl.sign_test_p(0, 0), 1.0)
        self.assertAlmostEqual(bl.sign_test_p(5, 5), 1.0)
        self.assertAlmostEqual(bl.sign_test_p(10, 0), 2 * 0.5**10)
        self.assertAlmostEqual(bl.sign_test_p(0, 10), 2 * 0.5**10)


class BuildItem(unittest.TestCase):
    """`build_item` end to end with a fake renderer that writes two small PNGs."""

    def test_files_labels_and_crops(self) -> None:
        base = (scene() * 255).astype(np.uint8)
        cand = base.copy()
        cand[100:110, 100:110] = 0  # a visible difference
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            out, work = tmp_path / "out", tmp_path / "work"

            def fake(ym: Path, src: Path, method: str, out_dir: Path, home: Path) -> tuple[Path, float]:
                out_dir.mkdir(parents=True, exist_ok=True)
                path = out_dir / "photo.png"
                Image.fromarray(base if method == "menon" else cand).save(path)
                return path, 0.5

            labels = {"A": "adaptive_ratio", "B": "menon"}
            entry = bl.build_item("real/photo", Path("photo.ARW"), labels, "menon", "adaptive_ratio", Path("ym"), out, work, False, render=fake)
            self.assertEqual((entry["A"], entry["B"]), ("adaptive_ratio", "menon"))
            self.assertFalse(entry["difference"]["identical"])
            self.assertEqual(entry["difference"]["max_abs_diff_codes"], int(base[100:110, 100:110].max()))
            # A is the candidate: its full frame differs from B's in the patch.
            a = np.asarray(Image.open(out / entry["files"]["full_A"]["path"]).convert("RGB")).astype(int)
            b = np.asarray(Image.open(out / entry["files"]["full_B"]["path"]).convert("RGB")).astype(int)
            self.assertGreater(abs(a[100:110, 100:110] - b[100:110, 100:110]).mean(), 20)
            self.assertEqual(set(entry["crops"]), set(bl.KINDS))
            for kind, crop in entry["crops"].items():
                x0, y0, x1, y1 = crop["box_xyxy"]
                self.assertEqual((x0 % 2, y0 % 2), (0, 0), "windows start on even pixels (same CFA phase)")
                ca = np.asarray(Image.open(out / crop["A"]))
                self.assertEqual(ca.shape[:2], (y1 - y0, x1 - x0), kind)
                # The crop is the same window of A's full-size render (lossless).
                np.testing.assert_array_equal(ca, cand[y0:y1, x0:x1])
            # The renders are deleted unless asked for.
            self.assertFalse(list((work / "renders").rglob("*.png")))
            page = bl.index_html({"real/photo": entry})
            self.assertNotIn("adaptive", page)


if __name__ == "__main__":
    unittest.main()

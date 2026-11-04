#!/usr/bin/env python3
"""Checks of the upsampling table (docs/24 section 3.3): the committed data in data/film/upsampling/, and the generator.

    python3 -m unittest discover -s tools/film/upsampling

The generator is run at 32 x 32 cells for the determinism test (a full run takes about a minute); the committed 256 x 256 table is
checked as it is. The numbers the tests assert are the ones docs/24 section 3.3 and docs/STATUS.md quote.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_table as mt  # noqa: E402

CIE = mt.REPO / "data" / "film" / "cie"
DATA = mt.REPO / "data" / "film" / "upsampling"


def spd(obs: mt.Observer, name: str) -> np.ndarray:
    """A CIE illuminant table from ``data/film/cie/`` on the observer's wavelengths."""
    rows = np.array([[float(v) for v in line.split(",")] for line in (CIE / name).read_text(encoding="utf-8").splitlines() if line.strip()])
    return np.interp(obs.lam, rows[:, 0], rows[:, 1])


def planck(obs: mt.Observer, kelvin: float) -> np.ndarray:
    """Relative spectral power of a blackbody, by formula."""
    metres = obs.lam * 1e-9
    return 1.0 / (metres**5 * (np.exp(1.4388e-2 / (metres * kelvin)) - 1.0))


class CommittedTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.obs = mt.Observer(CIE)
        cls.table, cls.flags = mt.load_table(DATA)
        cls.meta = json.loads((DATA / "table.json").read_text(encoding="utf-8"))
        axis = np.arange(256) / 255
        cls.gx, cls.gy = np.meshgrid(axis, axis)

    def test_the_files_have_the_documented_sizes_and_hashes(self):
        self.assertEqual(self.table.shape, (256, 256, 4))
        self.assertEqual((DATA / "coeffs.f32").stat().st_size, 256 * 256 * 4 * 4)
        self.assertEqual((DATA / "flags.u8").stat().st_size, 256 * 256)
        self.assertEqual((DATA / "error.f32").stat().st_size, 256 * 256 * 4)
        for name, digest in self.meta["outputs_sha256"].items():
            self.assertEqual(mt.sha256_file(DATA / name), digest, name)
        self.assertEqual(mt.sha256_file(CIE / "CIE_xyz_1931_2deg.csv"), self.meta["inputs_sha256"]["CIE_xyz_1931_2deg.csv"])

    def test_the_generator_in_the_record_is_this_one(self):
        self.assertEqual(self.meta["generator_version"], mt.GENERATOR_VERSION)
        self.assertEqual(self.meta["generator_sha256"], mt.sha256_file(Path(mt.__file__)), "make_table.py changed: run it again")

    def test_flags_are_zero_or_one_and_the_counts_match_the_record(self):
        self.assertTrue(set(np.unique(self.flags)) <= {0, 1})
        self.assertEqual(int((self.flags == 0).sum()), self.meta["solved_cells"])
        self.assertEqual(int((self.flags == 1).sum()), self.meta["inpainted_cells"])

    def test_every_solved_cell_reproduces_its_chromaticity_within_1e_4(self):
        xy = self.obs.xy(self.table[..., :3])
        err = np.hypot(xy[..., 0] - self.gx, xy[..., 1] - self.gy)
        solved = self.flags == 0
        worst = float(err[solved].max())
        iy, ix = np.unravel_index(np.argmax(np.where(solved, err, 0.0)), err.shape)
        print(f"\n  worst solved cell: {worst:.3g} at x={ix / 255:.4f} y={iy / 255:.4f}", file=sys.stderr)
        self.assertLessEqual(worst, mt.XY_BOUND)
        stored = np.fromfile(DATA / "error.f32", dtype="<f4").reshape(256, 256)
        self.assertLess(float(np.abs(stored - err).max()), 1e-6, "error.f32 is the error of the stored spectra")

    def test_the_solved_share_of_the_locus(self):
        inside = mt.inside_polygon(self.gx, self.gy, mt.locus_polygon(self.obs))
        self.assertEqual(int(inside.sum()), self.meta["cells_inside_locus"])
        solved_in = int(((self.flags == 0) & inside).sum())
        share = solved_in / int(inside.sum())
        print(f"\n  locus area: {solved_in} of {int(inside.sum())} cells solved ({100 * share:.2f} %), "
              f"{int(inside.sum()) - solved_in} inside the locus inpainted", file=sys.stderr)
        self.assertGreater(share, 0.99)
        self.assertEqual(int(((self.flags == 0) & ~inside).sum()), 0, "no cell outside the locus is solved")

    def test_all_spectra_are_finite_and_inside_zero_one_and_y_s_is_positive(self):
        self.assertTrue(np.isfinite(self.table).all())
        self.assertTrue((self.table[..., 3] > 0).all())
        spectra = self.obs.spectrum(self.table[..., :3])
        self.assertTrue(np.isfinite(spectra).all())
        self.assertGreaterEqual(float(spectra.min()), 0.0)
        self.assertLessEqual(float(spectra.max()), 1.0)
        y = self.obs.xyz(self.table[..., :3])[..., 1]
        self.assertLess(float(np.abs(y - self.table[..., 3]).max()), 1e-5, "Y_s is the Y of the stored spectrum")

    def test_the_inpainting_stays_between_the_solved_values_and_is_smooth(self):
        solved = self.flags == 0
        for k in range(3):
            lo, hi = self.table[..., k][solved].min(), self.table[..., k][solved].max()
            filled = self.table[..., k][~solved]
            self.assertGreaterEqual(float(filled.min()), lo - 1e-3 * (hi - lo), "maximum principle of the Laplace fill")
            self.assertLessEqual(float(filled.max()), hi + 1e-3 * (hi - lo))
        spectra = self.obs.spectrum(self.table[..., :3])
        pairs = []
        for axis, (a, b) in ((1, (np.s_[:, :-1], np.s_[:, 1:])), (0, (np.s_[:-1], np.s_[1:]))):
            both = (self.flags[a] == 1) & (self.flags[b] == 1)
            pairs.append(np.abs(spectra[b] - spectra[a]).max(axis=2)[both])
        change = np.concatenate(pairs)
        # Spectra near the locus are steps sharper than the 5 nm grid, so the largest change between neighbours is about 1 (an edge
        # moves by a sample); the bulk of the inpainted area changes by far less (the median is 4e-5, the 75th percentile 5e-3, the 90th
        # 0.06 with the luminance target of F-06b; 7e-7, 1e-5 and 0.01 with the first one, which was brighter at the locus). The
        # inpainted cells are outside the locus, which stage 1 never reaches, apart from the rim of a bilinear lookup.
        self.assertLess(float(np.percentile(change, 50)), 1e-4)
        self.assertLess(float(np.percentile(change, 75)), 2e-2)
        self.assertLess(float(np.percentile(change, 90)), 0.15)

    def test_ten_white_points_have_smooth_spectra_and_the_right_chromaticity(self):
        obs = self.obs
        whites = {
            "D50": spd(obs, "CIE_std_illum_D50.csv"),
            "D55": spd(obs, "CIE_illum_D55.csv"),
            "equal energy": np.ones(81),
            "blackbody 2700 K": planck(obs, 2700),
            "blackbody 2856 K (A)": planck(obs, 2856),
            "blackbody 3200 K": planck(obs, 3200),
            "blackbody 4000 K": planck(obs, 4000),
            "blackbody 5000 K": planck(obs, 5000),
            "blackbody 6500 K": planck(obs, 6500),
        }
        points = {}
        for name, power in whites.items():
            xyz = power @ obs.m
            points[name] = (xyz[0] / xyz.sum(), xyz[1] / xyz.sum())
        points["D65 (CIE 15 chromaticity)"] = (0.31272, 0.32903)
        self.assertEqual(len(points), 10)
        for name, (x, y) in points.items():
            spectrum = mt.lookup_spectrum(obs, self.table, x, y, 1.0)
            xyz = spectrum @ obs.m
            error = float(np.hypot(xyz[0] / xyz.sum() - x, xyz[1] / xyz.sum() - y))
            curvature = float(np.abs(np.diff(spectrum, 2)).max() / spectrum.max())
            print(f"\n  {name}: chromaticity error {error:.2g}, largest second difference {curvature:.2g} of the peak", file=sys.stderr, end="")
            self.assertLess(error, 5e-5, name)
            self.assertLess(curvature, 0.01, name)
            self.assertGreater(float(spectrum.min()), 0.0, name)
            self.assertAlmostEqual(float(xyz[1]), 1.0, places=6, msg=name)

    def test_the_lookup_is_homogeneous_of_degree_one(self):
        base = mt.lookup_spectrum(self.obs, self.table, 0.40, 0.38, 1.0)
        for scale in (0.01, 3.0, 40.0):
            np.testing.assert_allclose(mt.lookup_spectrum(self.obs, self.table, 0.40, 0.38, scale), scale * base, rtol=1e-12)

    def test_mixing_spectra_beats_mixing_coefficients(self):
        rng = np.random.default_rng(7)
        solved = self.flags == 0
        quad = solved[:-1, :-1] & solved[1:, :-1] & solved[:-1, 1:] & solved[1:, 1:]
        cells = np.argwhere(quad)
        worst = {"spectra": 0.0, "coeffs": 0.0}
        for iy, ix in cells[rng.choice(len(cells), 400, replace=False)]:
            x, y = (ix + rng.random()) / 255, (iy + rng.random()) / 255
            for mode in worst:
                xyz = mt.lookup_spectrum(self.obs, self.table, x, y, 1.0, mode) @ self.obs.m
                worst[mode] = max(worst[mode], float(np.hypot(xyz[0] / xyz.sum() - x, xyz[1] / xyz.sum() - y)))
        print(f"\n  between nodes, 400 random points: mixing spectra {worst['spectra']:.2g}, mixing coefficients {worst['coeffs']:.2g}", file=sys.stderr)
        self.assertLess(worst["spectra"], 5e-4)
        self.assertLess(worst["spectra"] * 10, worst["coeffs"])


class Generator(unittest.TestCase):
    def test_two_runs_give_identical_bytes(self):
        first = mt.serialise(mt.build(CIE, True, 32))
        second = mt.serialise(mt.build(CIE, True, 32))
        self.assertEqual(first.keys(), second.keys())
        for name in first:
            self.assertEqual(first[name], second[name], name)

    def test_the_small_grid_solves_the_white_point_and_inpaints_outside_the_locus(self):
        result = mt.build(CIE, True, 32)
        self.assertEqual(result["flags"][0, 0], mt.FLAG_INPAINTED)
        self.assertEqual(result["flags"][10, 10], mt.FLAG_SOLVED)
        self.assertTrue(np.isfinite(result["coeffs"]).all())


if __name__ == "__main__":
    unittest.main()

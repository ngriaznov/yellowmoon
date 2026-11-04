#!/usr/bin/env python3
"""Numbers and a picture for the committed table (not part of the build; needs matplotlib).

    python3 tools/film/upsampling/report.py [--png table.png] [--sizes DIR64 DIR128 DIR256 DIR512]

Prints the shares, the worst errors and, when tables of other sizes are given (``make_table.py --size N --out DIR``), the error of
the lookup between nodes against the grid size. Writes a picture of the locus with the solved and inpainted cells, a map of the error,
and a few spectra.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_table as mt  # noqa: E402

CIE = mt.REPO / "data" / "film" / "cie"


def between_nodes(obs: mt.Observer, folder: Path, count: int = 2000) -> tuple[float, float, float]:
    """Worst, 99th percentile and mean chromaticity error of the spectra-mixing lookup at random points among solved nodes."""
    table, flags = mt.load_table(folder)
    n = table.shape[0]
    solved = flags == 0
    quad = solved[:-1, :-1] & solved[1:, :-1] & solved[:-1, 1:] & solved[1:, 1:]
    cells = np.argwhere(quad)
    rng = np.random.default_rng(11)
    errors = []
    for iy, ix in cells[rng.choice(len(cells), min(count, len(cells)), replace=False)]:
        x, y = (ix + rng.random()) / (n - 1), (iy + rng.random()) / (n - 1)
        xyz = mt.lookup_spectrum(obs, table, x, y, 1.0) @ obs.m
        errors.append(np.hypot(xyz[0] / xyz.sum() - x, xyz[1] / xyz.sum() - y))
    errors = np.array(errors)
    return float(errors.max()), float(np.percentile(errors, 99)), float(errors.mean())


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, default=mt.REPO / "data" / "film" / "upsampling")
    parser.add_argument("--png", type=Path)
    parser.add_argument("--sizes", type=Path, nargs="*", default=[])
    args = parser.parse_args(argv)
    obs = mt.Observer(CIE)
    table, flags = mt.load_table(args.table)
    n = table.shape[0]
    axis = np.arange(n) / (n - 1)
    gx, gy = np.meshgrid(axis, axis)
    poly = mt.locus_polygon(obs)
    inside = mt.inside_polygon(gx, gy, poly)
    xy = obs.xy(table[..., :3])
    err = np.hypot(xy[..., 0] - gx, xy[..., 1] - gy)
    solved = flags == 0
    solved_in = int((solved & inside).sum())
    print(f"cells {n}x{n}; inside the locus {int(inside.sum())}; solved {solved_in} ({100 * solved_in / inside.sum():.2f} %); "
          f"inside and inpainted {int((inside & ~solved).sum())} ({100 * (inside & ~solved).sum() / inside.sum():.2f} %)")
    print(f"solved cells: worst error {err[solved].max():.3g}, median {np.median(err[solved]):.3g}")
    miss = inside & ~solved
    print(f"inpainted cells inside the locus: median error {np.median(err[miss]):.3g}, p90 {np.percentile(err[miss], 90):.3g}, "
          f"max {err[miss].max():.3g}")
    for folder in args.sizes:
        o = mt.Observer(CIE)
        worst, p99, mean = between_nodes(o, folder)
        t, _ = mt.load_table(folder)
        print(f"grid {t.shape[0]:4d}: lookup between nodes, worst {worst:.3g}, p99 {p99:.3g}, mean {mean:.3g}")
    if args.png:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
        shade = np.zeros((n, n, 3))
        shade[inside & solved] = (0.2, 0.7, 0.3)
        shade[inside & ~solved] = (0.9, 0.2, 0.2)
        shade[~inside & ~solved] = (0.85, 0.85, 0.9)
        a = axes[0]
        a.imshow(shade, origin="lower", extent=(0, 1, 0, 1))
        a.plot(np.append(poly[:, 0], poly[0, 0]), np.append(poly[:, 1], poly[0, 1]), "k-", lw=0.8)
        a.set_title("green: solved; red: inside the locus, inpainted;\ngrey: outside the locus, inpainted")
        a.set_xlim(0, 0.85)
        a.set_ylim(0, 0.9)
        a.set_xlabel("x")
        a.set_ylabel("y")
        b = axes[1]
        shown = np.where(solved & inside, np.log10(np.maximum(err, 1e-9)), np.nan)
        im = b.imshow(shown, origin="lower", extent=(0, 1, 0, 1), vmin=-8, vmax=-4, cmap="viridis")
        b.plot(np.append(poly[:, 0], poly[0, 0]), np.append(poly[:, 1], poly[0, 1]), "k-", lw=0.8)
        b.set_xlim(0, 0.85)
        b.set_ylim(0, 0.9)
        b.set_title("log10 of the chromaticity error of solved cells")
        fig.colorbar(im, ax=b, shrink=0.7)
        c = axes[2]
        picks = {
            "white (0.3333, 0.3333)": (1 / 3, 1 / 3),
            "D65 (0.3127, 0.3290)": (0.31272, 0.32903),
            "tungsten (0.4234, 0.3990)": (0.42337, 0.39896),
            "red (0.60, 0.33)": (0.60, 0.33),
            "green (0.20, 0.70)": (0.20, 0.70),
            "blue (0.15, 0.06)": (0.15, 0.06),
            "magenta (0.40, 0.18)": (0.40, 0.18),
        }
        for label, (x, y) in picks.items():
            ix, iy = round(x * (n - 1)), round(y * (n - 1))
            node = table[iy, ix]
            c.plot(obs.lam, obs.spectrum(node[:3]), label=f"{label}{' *' if flags[iy, ix] else ''}")
        c.set_xlabel("wavelength (nm)")
        c.set_ylabel("s(lambda)")
        c.set_title("spectra of a few cells (* inpainted)")
        c.legend(fontsize=8)
        args.png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.png, dpi=90)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

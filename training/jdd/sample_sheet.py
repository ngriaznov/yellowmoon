"""Visual sample sheet of the training sample generator (M2-04 "done when").

    python3 -m training.jdd.sample_sheet --out training/jdd/out/sample_sheet.png

One row per sample: the clean target, the network input as a (bilinear) demosaiced preview, the sigma
map, the qstep map and the clip mask, with the random choices of the generator as the row title. Rows are
chosen so every branch of docs/07 appears at least once (clean, cRAW, 12-bit, highlights over white).
The image is a review aid; do not commit it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .data import JddDataset, SamplerConfig, SyntheticSource  # noqa: E402
from .evaluate import baseline_bilinear  # noqa: E402

DEFAULT_OUT = Path(__file__).resolve().parent / "out" / "sample_sheet.png"


def srgb(v: np.ndarray) -> np.ndarray:
    """sRGB OETF with clipping, for display only."""
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.0031308, 12.92 * v, 1.055 * np.power(v, 1 / 2.4) - 0.055)


def pick_samples(ds: JddDataset, n: int, search: int = 400) -> list[int]:
    """Indices covering the generator's branches first, then filling up in index order."""
    want = ["clean", "craw", "12bit", "highlights"]
    found: dict[str, int] = {}
    for i in range(search):
        m = ds.sample(i).meta
        keys = [k for k in want if k not in found and ((k == "clean" and m["clean"]) or (k == "highlights" and m["highlights"]) or m["mode"] == k)]
        for k in keys:
            found[k] = i
        if len(found) == len(want):
            break
    chosen = list(dict.fromkeys(found.values()))
    i = 0
    while len(chosen) < n:
        if i not in chosen:
            chosen.append(i)
        i += 1
    return chosen[:n]


def render(out: Path, n: int, patch_half: int, seed: int) -> Path:
    """Draw the sheet and return its path."""
    cfg = SamplerConfig(patch_half=patch_half)
    ds = JddDataset(1000, seed, cfg, [(SyntheticSource(cfg.gt_size), 1.0)])
    idx = pick_samples(ds, n)
    cols = ["target (clipped GT)", "input, bilinear preview", "sigma map", "qstep map", "clip mask"]
    fig, axes = plt.subplots(len(idx), len(cols), figsize=(2.3 * len(cols), 2.5 * len(idx)), squeeze=False)
    for r, i in enumerate(idx):
        s = ds.sample(i, debug=True)
        m = s.meta
        target = np.moveaxis(s.y, 0, 2)
        scale = 1.0 / max(float(np.percentile(target, 99.5)), 1e-6)
        planes = s.x[:4]
        mosaic = np.zeros((2 * planes.shape[1], 2 * planes.shape[2]))
        for k, (dy, dx) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
            mosaic[dy::2, dx::2] = planes[k]
        preview = baseline_bilinear(mosaic)
        panels = [
            srgb(target * scale),
            srgb(preview * scale),
            s.x[4],
            s.x[5],
            s.x[6],
        ]
        for c, panel in enumerate(panels):
            ax = axes[r][c]
            if panel.ndim == 3:
                ax.imshow(panel, interpolation="nearest")
            else:
                im = ax.imshow(panel, cmap="magma", interpolation="nearest")
                ax.set_xlabel(f"max {panel.max():.3g}", fontsize=7, labelpad=1)
                del im
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=8)
        tag = "CLEAN" if m["clean"] else m["mode"]
        axes[r][0].set_ylabel(
            f"#{i} {m['source']}\nISO {m['iso']:.0f} {tag}\npeak {m['peak']:.3g}"
            f"{' HIGHLIGHTS' if m['highlights'] else ''}\nsigma x{m['sigma_jitter']:.2f}\ng0 R{m['g0'][0]:.2f} B{m['g0'][2]:.2f}",
            fontsize=6, rotation=0, ha="right", va="center", labelpad=4,
        )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--patch-half", type=int, default=64)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args(argv)
    path = render(args.out, args.n, args.patch_half, args.seed)
    print(f"wrote {path} ({path.stat().st_size / 1e3:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

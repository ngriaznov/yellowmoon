"""Trained models side by side on the same synthetic scenes (local, CPU or GPU, no Modal): what a new run changed.

    python -m training.bokeh.compare --tiles <tiles folder> --models bokeh-2=<run>/model.pt bokeh-3=<run>/model.pt --samples 48

For every model, on v3's scenes (thin structures, featureless layers, the depth stack; the half-resolution target with the sharp
pixel winning) and on bokeh-2's own (its synthesis and target, the same samples as its final evaluation when the tiles are the
run's): `mae_b`, `rel_err`, `fp_rate`, `recall_2bmin`, `ece` and the groups of `train.py`'s validation (the decision `P >= 0.7`:
`thin_false_blur`, `flat_false_blur`, `mid_recall`, `high_recall`, `precision`, `recall`). The tiles are the training volume's `tiles/`
(fetch them, or use `smoke.py`'s folder for a rough look: with few tiles, give `--val-share 0.5` so that some are held out); a model.pt is
a run's `model.pt` (`torch.save({"state_dict": ...})`, the EMA weights). Real photographs (CUHK, the lens planes) are `eval_real.py`'s.
"""

import argparse
import json
from pathlib import Path

import torch

from training.bokeh import sources, train
from training.bokeh.model import BokehNet

COLUMNS = (("model", ("mae_b", "rel_err", "fp_rate", "recall_2bmin", "ece")),
           ("groups", ("thin_false_blur_p70", "flat_false_blur_p70", "mid_recall_p70", "high_recall_p70", "precision_p70", "recall_p70")))


def load_model(path: Path, device) -> BokehNet:
    model = BokehNet()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["state_dict"])
    return model.to(device).eval()


def compare(tiles: Path, models: dict[str, Path], samples: int, device: str = "cpu", batch: int = 4, seed: int = 11,
            val_share: float = sources.VAL_SHARE) -> dict:
    """`{model name: {"v3": {...}, "bokeh-2": {...}}}` of the columns above."""
    config = {**train.DEFAULTS, "batch": batch, "val_share": val_share}
    out: dict = {}
    for name, path in models.items():
        model = load_model(path, torch.device(device))
        out[name] = {}
        for label, legacy in (("v3", False), ("bokeh-2", True)):
            got = train.evaluate(model, tiles, config, samples, seed, torch.device(device), legacy=legacy)
            out[name][label] = {k: got[part][k] for part, keys in COLUMNS for k in keys}
    return out


def table(result: dict) -> str:
    keys = [k for _, ks in COLUMNS for k in ks]
    lines = []
    for label in ("v3", "bokeh-2"):
        lines.append(f"on {label}'s scenes" + ("  (bokeh-2's synthesis and target)" if label == "bokeh-2" else "  (thin structures, featureless layers, the depth stack)"))
        lines.append(f"  {'':<14}" + "".join(f"{k[:19]:>20}" for k in keys))
        for name, r in result.items():
            lines.append(f"  {name:<14}" + "".join(f"{r[label][k]:>20.4f}" for k in keys))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tiles", type=Path, required=True)
    ap.add_argument("--models", nargs="+", required=True, help="name=path/to/model.pt, one or more")
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--val-share", type=float, default=sources.VAL_SHARE)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    models = {a.split("=", 1)[0]: Path(a.split("=", 1)[1]) for a in args.models}
    result = compare(args.tiles, models, args.samples, args.device, val_share=args.val_share)
    print(table(result))
    if args.json:
        args.json.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

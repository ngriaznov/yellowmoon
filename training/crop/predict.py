"""The student's crops for a folder of photos: its free crop, and for each fixed aspect the frame of that aspect
closest to it (the largest-IoU frame among 6 sizes x 9 x 9 positions, as Auto crop's candidates are laid out).

    python3 -m training.crop.predict --model testdata/crop/runs/smoke/model.pt --images <folder of JPEGs> \
        --out predictions.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from training.crop.data import letterbox
from training.crop.model import CropNet

SIZES = (1.0, 0.94, 0.88, 0.82, 0.76, 0.70)
GRID = 9
ASPECTS = {"1x1": 1.0, "16x9": 16 / 9, "4x5": None}  # 4:5 follows the photo's orientation (5:4 on a landscape)


def frames(width: int, height: int, ratio: float) -> np.ndarray:
    """Candidate frames of `ratio` (width over height, in pixels) as `[x0, y0, x1, y1]` fractions of the photo."""
    if width / height > ratio:
        fw, fh = ratio * height / width, 1.0
    else:
        fw, fh = 1.0, (width / height) / ratio
    out = []
    for s in SIZES:
        w, h = fw * s, fh * s
        xs = np.linspace(0, 1 - w, GRID) if w < 1 - 1e-9 else [0.0]
        ys = np.linspace(0, 1 - h, GRID) if h < 1 - 1e-9 else [0.0]
        out += [(x, y, x + w, y + h) for y in ys for x in xs]
    return np.asarray(out)


def closest(free: np.ndarray, cands: np.ndarray) -> np.ndarray:
    """The candidate with the largest IoU with `free` (ties: the earlier one, so the larger)."""
    ix = np.clip(np.minimum(cands[:, 2], free[2]) - np.maximum(cands[:, 0], free[0]), 0, None)
    iy = np.clip(np.minimum(cands[:, 3], free[3]) - np.maximum(cands[:, 1], free[1]), 0, None)
    inter = ix * iy
    area = (cands[:, 2] - cands[:, 0]) * (cands[:, 3] - cands[:, 1])
    union = area + (free[2] - free[0]) * (free[3] - free[1]) - inter
    return cands[int(np.argmax(inter / union))]


def rect(box) -> dict:
    """`[x0, y0, x1, y1]` as the `{x, y, width, height}` rect the evaluation scripts read."""
    x0, y0, x1, y1 = (float(v) for v in box)
    return {"x": round(x0, 6), "y": round(y0, 6), "width": round(x1 - x0, 6), "height": round(y1 - y0, 6)}


def predict(model_path: Path, images: list[Path]) -> dict:
    """`{"free": {stem: {"rect"}}, "1x1": ..., "16x9": ..., "4x5": ...}` for the images."""
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    model = CropNet()
    model.load_state_dict(state["state_dict"])
    model.eval()
    out = {k: {} for k in ["free", *ASPECTS]}
    with torch.no_grad():
        for p in images:
            stem = p.stem.rsplit("_", 1)[0] if "_" in p.stem else p.stem
            with Image.open(p) as img:
                w, h = img.size
                pix, _ = letterbox(img)
            free = model(torch.from_numpy(pix.copy()).permute(2, 0, 1)[None].float() / 255)[0].numpy()
            out["free"][stem] = {"rect": rect(free)}
            for name, ratio in ASPECTS.items():
                r = ratio if ratio is not None else (4 / 5 if h > w else 5 / 4)
                out[name][stem] = {"rect": rect(closest(free, frames(w, h, r)))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = predict(args.model, sorted(args.images.glob("*.jpg")))
    args.out.write_text(json.dumps(out, indent=1))
    print(f"{len(out['free'])} photos")


if __name__ == "__main__":
    main()

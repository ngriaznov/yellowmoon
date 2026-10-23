"""Training arrays for the crop student: each photo letterboxed into a `SIZE` x `SIZE` square, the square's content
box, and the teacher's crop as fractions of the photo.

    python3 -m training.crop.data --images testdata/crop/smoke/images --labels testdata/crop/smoke/labels.json \
        --out testdata/crop/smoke/arrays
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

SIZE = 256


def letterbox(img: Image.Image, size: int = SIZE) -> tuple[np.ndarray, np.ndarray]:
    """The image scaled to fit a `size` square (long side `size`, centred, black around it), as uint8 HWC RGB, and
    its content box `[x0, y0, x1, y1]` as fractions of the square."""
    img = img.convert("RGB")
    w, h = img.size
    scale = size / max(w, h)
    cw, ch = max(1, round(w * scale)), max(1, round(h * scale))
    x0, y0 = (size - cw) // 2, (size - ch) // 2
    square = Image.new("RGB", (size, size))
    square.paste(img.resize((cw, ch), Image.BILINEAR, reducing_gap=None), (x0, y0))
    return np.asarray(square, dtype=np.uint8), np.array(
        [x0 / size, y0 / size, (x0 + cw) / size, (y0 + ch) / size], dtype=np.float32
    )


def build(images: Path, labels: dict, out: Path) -> int:
    """Writes `images.npy` (N, SIZE, SIZE, 3), `content.npy` and `target.npy` (N, 4) and `ids.json` for the
    labelled photos, in id order; returns N."""
    ids = sorted(k for k, v in labels.items() if v.get("crop") and (images / f"{k}.jpg").exists())
    out.mkdir(parents=True, exist_ok=True)
    pix = np.lib.format.open_memmap(out / "images.npy", mode="w+", dtype=np.uint8, shape=(len(ids), SIZE, SIZE, 3))
    content = np.zeros((len(ids), 4), np.float32)
    target = np.zeros((len(ids), 4), np.float32)
    for i, k in enumerate(ids):
        with Image.open(images / f"{k}.jpg") as img:
            pix[i], content[i] = letterbox(img)
        target[i] = np.clip(np.asarray(labels[k]["crop"], np.float32), 0.0, 1.0)
    pix.flush()
    np.save(out / "content.npy", content)
    np.save(out / "target.npy", target)
    (out / "ids.json").write_text(json.dumps(ids))
    return len(ids)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    n = build(args.images, json.loads(args.labels.read_text()), args.out)
    print(f"{n} photos in {args.out}")


if __name__ == "__main__":
    main()

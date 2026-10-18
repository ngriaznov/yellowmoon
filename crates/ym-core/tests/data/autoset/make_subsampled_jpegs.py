"""Writes the two small JPEG fixtures of `ym_core::autoset::camera`'s decoder test: a synthetic 96 x 64 picture
(smooth colour ramps, a hard-edged patch and a fine stripe pattern) encoded with 4:2:2 chroma, as Sony's embedded
previews are, and with 4:2:0. The pictures are ours (no photograph); the bytes are committed, so the test never needs
this script, which only records how they were made (Pillow, quality 90)."""

import sys
from pathlib import Path

from PIL import Image

W, H = 96, 64


def picture() -> Image.Image:
    img = Image.new("RGB", (W, H))
    px = img.load()
    for y in range(H):
        for x in range(W):
            r = int(255 * x / (W - 1))
            g = int(255 * y / (H - 1))
            b = int(255 * (1 - x / (W - 1)) * (1 - y / (H - 1)))
            if 20 <= x < 44 and 12 <= y < 36:
                r, g, b = 200, 40, 160
            if x >= 64 and (x // 2 + y // 3) % 2 == 0:
                r, g, b = 30, 220, 90
            px[x, y] = (r, g, b)
    return img


def main(out: Path) -> None:
    img = picture()
    img.save(out / "subsampled-422.jpg", quality=90, subsampling=1, optimize=False)
    img.save(out / "subsampled-420.jpg", quality=90, subsampling=2, optimize=False)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent)

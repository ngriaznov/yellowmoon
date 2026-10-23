"""The teacher: FrameLite (Apache-2.0, `research/crop-models-2026-10-23.md`) labels each training photo with one
free crop. Offline only; nothing of it ships.

FrameLite's inference code is not vendored: it is cloned at `FRAMELITE_COMMIT` into the Modal image (or a local
checkout named by `FRAMELITE_SRC`), and its weights and Qwen3.5-0.8B base are downloaded at the pinned revisions.

    FRAMELITE_SRC=/path/to/FrameLite python3 -m training.crop.teacher --images testdata/crop/smoke/images \
        --weights /path/to/framelite --out testdata/crop/smoke/labels.json --device cpu
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

FRAMELITE_REPO = "https://github.com/JianzeLi-114/FrameLite"
FRAMELITE_COMMIT = "e0f28dd2b6e137264a0a5a9996c1cca17c330657"
FRAMELITE_WEIGHTS = ("zlab-sjtu/FrameLite", "5c3c9aa1387ed2492f7e7816934167a720df8842")
QWEN_BASE = ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17")


def download_weights(directory: Path) -> Path:
    """FrameLite's adapter and its base model at the pinned revisions, laid out as FrameLite expects
    (`<directory>/framelite.safetensors`, `<directory>/base/`)."""
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=FRAMELITE_WEIGHTS[0], revision=FRAMELITE_WEIGHTS[1], local_dir=str(directory),
        allow_patterns=["framelite.safetensors", "inference_config.json", "LICENSE", "NOTICE"],
    )
    snapshot_download(repo_id=QWEN_BASE[0], revision=QWEN_BASE[1], local_dir=str(directory / "base"))
    return directory


class Teacher:
    """FrameLite loaded once; `label` answers one free crop per image."""

    def __init__(self, weights: Path, device: str, dtype: str):
        src = os.environ.get("FRAMELITE_SRC")
        if src:
            sys.path.insert(0, src)
        from framelite import FrameLite  # noqa: PLC0415 - FrameLite is only importable where the teacher runs

        self.model = FrameLite.from_pretrained(str(weights), device=device, dtype=dtype)

    def label(self, path: Path) -> dict:
        """FrameLite's crop of the image at `path`, as fractions `[x0, y0, x1, y1]` of the image, or why there is
        none."""
        t0 = time.time()
        try:
            r = self.model.predict(str(path))
        except Exception as e:  # noqa: BLE001 - one bad image must not stop a batch
            return {"id": path.stem, "crop": None, "error": f"{type(e).__name__}: {e}"[:200]}
        crop = r.get("bbox_normalized") if r.get("valid") else None
        return {
            "id": path.stem,
            "crop": [round(float(v), 4) for v in crop] if crop else None,
            "error": None if crop else str(r.get("error"))[:200],
            "seconds": round(time.time() - t0, 2),
        }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--download", action="store_true", help="download the pinned weights into --weights first")
    args = ap.parse_args()
    if args.download:
        download_weights(args.weights)
    teacher = Teacher(args.weights, args.device, args.dtype)
    done = json.loads(args.out.read_text()) if args.out.exists() else {}
    for path in sorted(args.images.glob("*.jpg")):
        if path.stem not in done:
            done[path.stem] = teacher.label(path)
            args.out.write_text(json.dumps(done, indent=1))
    print(f"{sum(1 for v in done.values() if v['crop'])} of {len(done)} images labelled")


if __name__ == "__main__":
    main()

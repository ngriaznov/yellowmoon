#!/usr/bin/env python3
"""Writes the parity fixtures of the Rust BlazeFace port (docs/27 S-08): `crates/ym-ml/tests/fixtures/blazeface.json`.

Development only (AGENTS.md rule 6): run by hand when the port or the photos change, in an environment with
`requirements-reference.txt`; the JSON is committed (small), the photos never are.

    python tools/mediapipe/reference.py --tflite blaze_face_full_range.tflite --photos DIR --out blazeface.json

What the file holds, all computed from Google's own `blaze_face_full_range.tflite` and Google's own code, never from ours:

  synthetic   the raw outputs of the TFLite interpreter (ai-edge-litert) for a fixed input made by a formula that the Rust
              test repeats (`synthetic_input` below): 128 anchors (every 18th) with their logit and 16 regressors, and the
              mean and largest absolute value of all of them. This is the numerical parity of the network alone.
  photos      for each photo `<id>.jpg` of tools/fetch/mediapipe-photos.tsv (fetched by `tools/fetch/mediapipe.sh --photos`; the
              SHA-256 of the file is checked): the detections of MediaPipe's own FaceDetector task (end to end), and the
              TFLite interpreter's raw answer to the letterboxed input made as MediaPipe's ImageToTensor makes it (OpenCV
              warpAffine, bilinear, no filtering, black bars, x / 127.5 - 1), as its 8 best anchors and the weighted NMS of
              those, written in numpy from MediaPipe's documented algorithm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

SIDE = 192
GRID = 48
SCORE_THRESHOLD = 0.5
NMS_IOU = 0.3


def synthetic_input() -> np.ndarray:
    """The fixed test input `[1, 192, 192, 3]`: a repeating ramp mixed with integer-hash noise, `x * 2 - 1`, as f32.

    Every step is exact in f64 and the Rust test repeats it step by step, so both sides feed the network the same bits.
    """
    n = SIDE * SIDE * 3
    i = np.arange(n, dtype=np.uint64)
    noise = ((i * np.uint64(2654435761) + np.uint64(12345)) & np.uint64(0xFFFFFFFF)).astype(np.float64) / 4294967296.0
    c = i % np.uint64(3)
    x = (i // np.uint64(3)) % np.uint64(SIDE)
    y = i // np.uint64(3 * SIDE)
    ramp = ((x + np.uint64(2) * y + np.uint64(40) * c) % np.uint64(97)).astype(np.float64) / 96.0
    v = 0.6 * ramp + 0.4 * noise
    return (v * 2.0 - 1.0).astype(np.float32).reshape(1, SIDE, SIDE, 3)


def anchors() -> np.ndarray:
    return np.array([((x + 0.5) / GRID, (y + 0.5) / GRID) for y in range(GRID) for x in range(GRID)], np.float32)


def run_net(interpreter, tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    details_in = interpreter.get_input_details()[0]
    interpreter.set_tensor(details_in["index"], tensor.astype(np.float32))
    interpreter.invoke()
    regressors = logits = None
    for out in interpreter.get_output_details():
        data = interpreter.get_tensor(out["index"])[0]
        if data.shape[-1] == 16:
            regressors = data
        else:
            logits = data[:, 0]
    return regressors, logits


def sigmoid(logit):
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -100.0, 100.0)))


def iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def decode(regressors, logits, scale, ox, oy):
    """Detections above the threshold in picture pixels (box x0 y0 x1 y1, 6 keypoints, score), then weighted NMS."""
    a = anchors()
    score = sigmoid(logits)
    dets = []
    for i in np.nonzero(score >= SCORE_THRESHOLD)[0]:
        r = regressors[i]
        xc, yc = r[0] / SIDE + a[i, 0], r[1] / SIDE + a[i, 1]
        w, h = r[2] / SIDE, r[3] / SIDE
        to_x = lambda n: (n * SIDE - ox) / scale
        to_y = lambda n: (n * SIDE - oy) / scale
        box = [to_x(xc - w / 2), to_y(yc - h / 2), to_x(xc + w / 2), to_y(yc + h / 2)]
        kps = [[to_x(r[4 + 2 * k] / SIDE + a[i, 0]), to_y(r[5 + 2 * k] / SIDE + a[i, 1])] for k in range(6)]
        dets.append((float(score[i]), box, kps))
    dets.sort(key=lambda d: (-d[0], d[1][1], d[1][0]))
    out = []
    while dets:
        best = dets[0]
        group = [d for d in dets if iou(d[1], best[1]) > NMS_IOU or d is best]
        dets = [d for d in dets if not (iou(d[1], best[1]) > NMS_IOU or d is best)]
        total = sum(d[0] for d in group)
        box = [sum(d[0] * d[1][j] for d in group) / total for j in range(4)]
        kps = [[sum(d[0] * d[2][k][j] for d in group) / total for j in range(2)] for k in range(6)]
        out.append({"score": best[0], "box": box, "keypoints": kps, "merged": len(group)})
    return out


def mediapipe_tensor(img: np.ndarray):
    """MediaPipe's ImageToTensor for the whole picture: letterbox, warpAffine bilinear (no half-pixel shift), x / 127.5 - 1."""
    import cv2

    h, w = img.shape[:2]
    scale = min(SIDE / w, SIDE / h)
    ox, oy = (SIDE - w * scale) / 2, (SIDE - h * scale) / 2
    m = np.array([[scale, 0, ox], [0, scale, oy]], np.float64)
    canvas = cv2.warpAffine(img, m, (SIDE, SIDE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).astype(np.float32)
    return (canvas / 127.5 - 1.0)[None], scale, ox, oy


def round6(v):
    if isinstance(v, (list, tuple)):
        return [round6(x) for x in v]
    if isinstance(v, dict):
        return {k: round6(x) for k, x in v.items()}
    if isinstance(v, (float, np.floating)):
        return round(float(v), 6)
    return v


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tflite", type=Path, required=True)
    parser.add_argument("--photos", type=Path, required=True, help="directory with <id>.jpg")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[2] / "tools/fetch/mediapipe-photos.tsv")
    parser.add_argument("--ids", nargs="*", default=None, help="photo ids (default: all of the manifest)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import mediapipe as mp
    from ai_edge_litert.interpreter import Interpreter
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision
    from PIL import Image

    model_bytes = args.tflite.read_bytes()
    interpreter = Interpreter(model_path=str(args.tflite))
    interpreter.allocate_tensors()

    regressors, logits = run_net(interpreter, synthetic_input())
    sample = list(range(0, GRID * GRID, 18))
    synthetic = {
        "anchors": sample,
        "logits": [float(logits[i]) for i in sample],
        "regressors": [[float(v) for v in regressors[i]] for i in sample],
        "logit_abs_mean": float(np.abs(logits).mean()),
        "logit_abs_max": float(np.abs(logits).max()),
        "regressor_abs_mean": float(np.abs(regressors).mean()),
        "regressor_abs_max": float(np.abs(regressors).max()),
    }

    manifest = {}
    for line in args.manifest.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            pid, sha, *_ = line.split("\t")
            manifest[pid] = {"sha256": sha}
    ids = list(manifest) if args.ids is None else args.ids
    detector = vision.FaceDetector.create_from_options(
        vision.FaceDetectorOptions(
            base_options=mpt.BaseOptions(model_asset_path=str(args.tflite)),
            min_detection_confidence=SCORE_THRESHOLD,
            min_suppression_threshold=NMS_IOU,
        )
    )
    photos = {}
    for pid in ids:
        path = args.photos / f"{pid}.jpg"
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != manifest[pid]["sha256"]:
            raise SystemExit(f"{path}: SHA-256 {digest}, the manifest pins {manifest[pid]['sha256']}")
        img = np.asarray(Image.open(path).convert("RGB"))
        h, w = img.shape[:2]
        result = detector.detect(mp.Image.create_from_file(str(path)))
        task = []
        for d in result.detections:
            b = d.bounding_box
            task.append(
                {
                    "box": [b.origin_x, b.origin_y, b.origin_x + b.width, b.origin_y + b.height],
                    "score": d.categories[0].score,
                    "keypoints": [[k.x * w, k.y * h] for k in d.keypoints],
                }
            )
        tensor, scale, ox, oy = mediapipe_tensor(img)
        r, l = run_net(interpreter, tensor)
        top = np.argsort(-l)[:8]
        photos[pid] = {
            "sha256": digest,
            "width": w,
            "height": h,
            "mediapipe_task": task,
            "tflite_top_anchors": [{"anchor": int(i), "logit": float(l[i]), "regressors": [float(v) for v in r[i]]} for i in top],
            "tflite_weighted_nms": decode(r, l, scale, ox, oy),
        }
        print(f"{pid}: {len(task)} face(s) from MediaPipe, {len(photos[pid]['tflite_weighted_nms'])} from the interpreter")

    detector.close()
    out = {
        "schema": 1,
        "model": "blaze_face_full_range",
        "tflite_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "mediapipe": mp.__version__ if hasattr(mp, "__version__") else "1.2.2",
        "score_threshold": SCORE_THRESHOLD,
        "nms_iou": NMS_IOU,
        "synthetic": synthetic,
        "photos": photos,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(round6(out), indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

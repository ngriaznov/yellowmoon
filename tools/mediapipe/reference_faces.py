#!/usr/bin/env python3
"""Writes the parity fixtures of the Rust Face Mesh V2 and Selfie Multiclass ports (docs/27 S-09, S-10):
`crates/ym-ml/tests/fixtures/faces.json`.

Development only (AGENTS.md rule 6): run by hand when the ports or the photos change, in an environment with
`requirements-reference.txt`; the JSON is committed (small), the photos never are.

    python tools/mediapipe/reference_faces.py --mesh face_landmarks_detector.tflite --selfie selfie_multiclass_256x256.tflite \
        --task face_landmarker.task --blaze blaze_face_full_range.tflite --photos DIR --out faces.json

`face_landmarks_detector.tflite` is the member of `face_landmarker.task` (a zip). Everything is computed from Google's own
files and Google's own code, never from ours:

  mesh_synthetic    the raw outputs of the TFLite interpreter (ai-edge-litert) for a fixed input made by a formula that the
                    Rust test repeats (`synthetic_input`): all 478 landmarks and the presence logit. The network alone.
  selfie_synthetic  the same for Selfie Multiclass: the softmax of the interpreter's scores at every 16th pixel.
  photos            for each photo `<id>.jpg` of tools/fetch/mediapipe-photos.tsv (SHA-256 checked) and each face the whole-
                    picture BlazeFace pass finds (the interpreter and `reference.py`'s decoder): the crop of MediaPipe's first
                    pass (`roi1`, cv2.warpAffine as MediaPipe's ImageToTensor), the interpreter's landmarks for it, the crop of
                    the second pass made from them (`roi2`) and its landmarks, the crop of the segmenter (`selfie_roi`, the
                    landmark box times 1.8, turned like roi2) and the interpreter's class probabilities for it (every 16th
                    pixel), and the landmarks of MediaPipe's own FaceLandmarker task (end to end, its own detector).
                    Landmarks are kept for every 4th index (0, 4, ..., 476) to keep the file small.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reference  # noqa: E402  (BlazeFace helpers of S-08)

LANDMARK_STRIDE = 4
SELFIE_STRIDE = 16
SELFIE_SCALE = 1.8
SIDE = 256


def synthetic_input(side: int = SIDE) -> np.ndarray:
    """The fixed test input `[1, side, side, 3]` in 0..1: a repeating ramp mixed with integer-hash noise, as f32.

    Every step is exact in f64 and the Rust test repeats it step by step, so both sides feed the network the same bits.
    """
    n = side * side * 3
    i = np.arange(n, dtype=np.uint64)
    noise = ((i * np.uint64(2654435761) + np.uint64(12345)) & np.uint64(0xFFFFFFFF)).astype(np.float64) / 4294967296.0
    c = i % np.uint64(3)
    x = (i // np.uint64(3)) % np.uint64(side)
    y = i // np.uint64(3 * side)
    ramp = ((x + np.uint64(2) * y + np.uint64(40) * c) % np.uint64(97)).astype(np.float64) / 96.0
    v = 0.6 * ramp + 0.4 * noise
    return v.astype(np.float32).reshape(1, side, side, 3)


def run(interpreter, tensor: np.ndarray) -> dict:
    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], tensor.astype(np.float32).reshape(1, SIDE, SIDE, 3))
    interpreter.invoke()
    return {d["name"]: interpreter.get_tensor(d["index"]).ravel().copy() for d in interpreter.get_output_details()}


def softmax(scores: np.ndarray) -> np.ndarray:
    e = np.exp(scores - scores.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def roi_matrix(center, side, rotation):
    """The matrix from crop pixels (i, j) to picture coordinates: picture = centre + R((i/N - 0.5) side, (j/N - 0.5) side)."""
    import cv2  # noqa: F401

    s, c = math.sin(rotation), math.cos(rotation)
    n = SIDE
    return np.array(
        [
            [side / n * c, -side / n * s, center[0] - 0.5 * side * c + 0.5 * side * s],
            [side / n * s, side / n * c, center[1] - 0.5 * side * s - 0.5 * side * c],
        ]
    )


def crop(img: np.ndarray, roi: dict) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    a = roi_matrix(roi["center"], roi["side"], roi["rotation"])
    m = cv2.invertAffineTransform(a)
    out = cv2.warpAffine(img, m, (SIDE, SIDE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return out.astype(np.float32) / 255.0, a


def landmarks_in_picture(outputs: dict, a: np.ndarray) -> np.ndarray:
    p = outputs["Identity"].reshape(-1, 3)
    xy = np.c_[p[:, 0], p[:, 1], np.ones(len(p))] @ a.T
    z = p[:, 2] * (np.hypot(a[0, 0], a[1, 0]))
    return np.c_[xy, z]


def roi_from_landmarks(pts: np.ndarray, scale: float) -> dict:
    b = [pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()]
    return {
        "center": [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2],
        "side": max(b[2] - b[0], b[3] - b[1]) * scale,
        "rotation": math.atan2(pts[263][1] - pts[33][1], pts[263][0] - pts[33][0]),
    }


def subset(pts: np.ndarray) -> list:
    return [[round(float(v), 3) for v in pts[i]] for i in range(0, 478, LANDMARK_STRIDE)]


def jround(v):
    """Python floats for JSON (numpy scalars are not serialisable)."""
    if isinstance(v, dict):
        return {k: jround(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jround(x) for x in v]
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def round_list(a, nd=4):
    return [round(float(v), nd) for v in np.asarray(a).ravel()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mesh", type=Path, required=True, help="face_landmarks_detector.tflite")
    parser.add_argument("--selfie", type=Path, required=True, help="selfie_multiclass_256x256.tflite")
    parser.add_argument("--task", type=Path, required=True, help="face_landmarker.task")
    parser.add_argument("--blaze", type=Path, required=True, help="blaze_face_full_range.tflite")
    parser.add_argument("--photos", type=Path, required=True, help="directory with <id>.jpg")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[2] / "tools/fetch/mediapipe-photos.tsv")
    parser.add_argument("--ids", nargs="*", default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import mediapipe as mp
    from ai_edge_litert.interpreter import Interpreter
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision
    from PIL import Image

    mesh = Interpreter(model_path=str(args.mesh))
    mesh.allocate_tensors()
    selfie = Interpreter(model_path=str(args.selfie))
    selfie.allocate_tensors()
    blaze = Interpreter(model_path=str(args.blaze))
    blaze.allocate_tensors()

    synth = run(mesh, synthetic_input())
    mesh_synthetic = {
        "landmarks": round_list(synth["Identity"], 3),
        "presence_logit": round(float(synth["Identity_1"][0]), 4),
        "landmark_abs_max": round(float(np.abs(synth["Identity"]).max()), 3),
    }
    seg = softmax(run(selfie, synthetic_input())["Identity"].reshape(SIDE, SIDE, 6))
    selfie_synthetic = {
        "stride": 16,
        "probs": round_list(seg[::16, ::16], 5),
        "class_means": round_list(seg.reshape(-1, 6).mean(0), 5),
    }

    manifest = {}
    for line in args.manifest.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            pid, sha, *_ = line.split("\t")
            manifest[pid] = sha
    ids = list(manifest) if args.ids is None else args.ids
    landmarker = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=str(args.task)),
            num_faces=8,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
        )
    )
    photos = {}
    for pid in ids:
        path = args.photos / f"{pid}.jpg"
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != manifest[pid]:
            raise SystemExit(f"{path}: SHA-256 {digest}, the manifest pins {manifest[pid]}")
        img = np.asarray(Image.open(path).convert("RGB"))
        h, w = img.shape[:2]
        tensor, scale, ox, oy = reference.mediapipe_tensor(img)
        r, l = reference.run_net(blaze, tensor)
        detections = reference.decode(r, l, scale, ox, oy)
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=img))
        task_faces = [np.array([[q.x * w, q.y * h, q.z * w] for q in f]) for f in result.face_landmarks]
        faces = []
        for d in detections:
            b, kp = d["box"], d["keypoints"]
            roi1 = {
                "center": [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2],
                "side": max(b[2] - b[0], b[3] - b[1]) * 1.5,
                "rotation": math.atan2(kp[1][1] - kp[0][1], kp[1][0] - kp[0][0]),
            }
            c1, a1 = crop(img, roi1)
            o1 = run(mesh, c1)
            p1 = landmarks_in_picture(o1, a1)
            roi2 = roi_from_landmarks(p1, 1.5)
            c2, a2 = crop(img, roi2)
            o2 = run(mesh, c2)
            p2 = landmarks_in_picture(o2, a2)
            selfie_roi = roi_from_landmarks(p2, SELFIE_SCALE)
            c3, _ = crop(img, selfie_roi)
            probs = softmax(run(selfie, c3)["Identity"].reshape(SIDE, SIDE, 6))
            task = None
            if task_faces:
                centre = p2[:, :2].mean(0)
                best = min(task_faces, key=lambda t: np.hypot(*(t[:, :2].mean(0) - centre)))
                if np.hypot(*(best[:, :2].mean(0) - centre)) < 0.5 * (b[2] - b[0]):
                    task = subset(best)
            faces.append(
                {
                    "detection": {"box": [round(v, 3) for v in b], "keypoints": [[round(v, 3) for v in k] for k in kp]},
                    "roi1": {k: ([round(x, 4) for x in v] if isinstance(v, list) else round(v, 6)) for k, v in roi1.items()},
                    "presence1": round(float(o1["Identity_1"][0]), 3),
                    "pass1": subset(p1),
                    "roi2": {k: ([round(x, 4) for x in v] if isinstance(v, list) else round(v, 6)) for k, v in roi2.items()},
                    "presence2": round(float(o2["Identity_1"][0]), 3),
                    "pass2": subset(p2),
                    "selfie_roi": {k: ([round(x, 4) for x in v] if isinstance(v, list) else round(v, 6)) for k, v in selfie_roi.items()},
                    "selfie_probs": round_list(probs[::SELFIE_STRIDE, ::SELFIE_STRIDE], 4),
                    "selfie_class_means": round_list(probs.reshape(-1, 6).mean(0), 4),
                    "task": task,
                }
            )
        photos[pid] = {"sha256": digest, "width": w, "height": h, "faces": faces, "task_faces": len(task_faces)}
        print(f"{pid}: {len(faces)} face(s) from BlazeFace, {len(task_faces)} from the task")

    landmarker.close()
    out = {
        "schema": 1,
        "landmark_stride": LANDMARK_STRIDE,
        "selfie_stride": SELFIE_STRIDE,
        "selfie_scale": SELFIE_SCALE,
        "mesh_sha256": hashlib.sha256(args.mesh.read_bytes()).hexdigest(),
        "selfie_sha256": hashlib.sha256(args.selfie.read_bytes()).hexdigest(),
        "mediapipe": getattr(mp, "__version__", "1.2.2"),
        "mesh_synthetic": mesh_synthetic,
        "selfie_synthetic": selfie_synthetic,
        "photos": photos,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(jround(out), separators=(",", ":"), sort_keys=True) + "\n")
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

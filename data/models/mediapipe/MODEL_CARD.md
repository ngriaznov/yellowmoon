# Model card: `mediapipe`, the face networks of skin retouching (BlazeFace, Face Mesh V2, Selfie Multiclass)

**S-08, 2026-11-11 (`docs/27-skin-retouch.md`, `docs/07-ml-models.md`).** Three small networks that Google publishes in MediaPipe, converted from TFLite to safetensors. They are **Google's models, not ours**: we did not train, fine-tune or prune them, and we change only the container (and, for the f16 files, nothing in the values: Google's files already store f16). They find the faces of a photo (BlazeFace, run by `ym_ml::mediapipe::blazeface` since S-08), put a 478-point mesh on each face (Face Mesh V2, S-09) and separate skin, hair, clothes and background (Selfie Multiclass, S-10) for the Skin node (`docs/27`). All three have their runtime (`ym_ml::mediapipe`: BlazeFace S-08, Face Mesh V2 S-09, Selfie Multiclass S-10).

**The weights are not in git.** `tools/fetch/mediapipe.sh` downloads them from the Yellowmoon release `models-mediapipe-faces-1` (a prerelease that is never the "latest" release) and checks the SHA-256 in `SHA256SUMS`; `release.yml` and `build.yml` do that before the app is built, so the bundle holds them in `data/models/mediapipe/`, and `ym` finds them there like the other models. The app and `ym` never download anything (`AGENTS.md` rule 7). The model card, `SHA256SUMS` and the notice below are committed. The conversion is deterministic, so the three hashes were committed before the release existed, and the `mediapipe` job of `models.yml` publishes only bytes that equal them.

## Source and licence

| | BlazeFace (full range) | Face Mesh V2 | Selfie Multiclass (256) |
|---|---|---|---|
| Original file | `blaze_face_full_range.tflite` | `face_landmarks_detector.tflite`, inside `face_landmarker.task` | `selfie_multiclass_256x256.tflite` |
| URL | `https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/1/blaze_face_full_range.tflite` | `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task` | `https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite` |
| Original SHA-256 | `3698b18f…c2d69aed2b181b` (file, 1,083,786 bytes) | `64184e22…2fe6544e0bc9ff` (the task, 3,758,596 bytes); `c7d54204…67e43046bd43d5` (the member, 2,553,590 bytes) | `c6748b12…ea23016507e0e0` (file, 16,371,837 bytes) |
| Model card | `https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Model%20Card%20(Full%20Range).pdf` (19 January 2021) | `https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf` (15 September 2022) | `https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Multiclass%20Segmentation.pdf` (10 May 2023) |
| Licence | **Apache-2.0** ("LICENSED UNDER Apache License, Version 2.0") | **Apache-2.0** (same words) | **Apache-2.0** (same words) |
| Authors on the card | Valentin Bazarevsky (card: Yury Kartynnik) | Geng Yan and Ivan Grishchenko | Adel Ahmadyan |

The full hashes are in `tools/mediapipe/convert.py` (`MODELS`), which refuses to convert a file whose SHA-256 differs. The MediaPipe repository (`https://github.com/google-ai-edge/mediapipe`) is Apache-2.0 ("Copyright 2021 The MediaPipe Authors"); the guides of the Face Landmarker and Face Detector carry no licence of their own, so the licence is the cards' (read on 2026-11-11, `research/research_notes/Skin retouching models and data/face_detection_and_mesh.md`).

Notice (Apache-2.0, section 4): this product includes the BlazeFace (full range), Face Mesh V2 and Selfie Multiclass networks, Copyright 2021 The MediaPipe Authors (Google LLC), licensed under the Apache License, Version 2.0 (`https://www.apache.org/licenses/LICENSE-2.0`), from `https://github.com/google-ai-edge/mediapipe` and `https://storage.googleapis.com/mediapipe-models/`. The networks were converted from TFLite to safetensors (the changes: container; the graph is stored in the file header); their weights were not changed and not trained further. `NOTICE` item 14 carries the same.

## Training data (Google's, as the cards state it; we use none of it)

- **BlazeFace:** "trained and evaluated on consented images of people using a mobile AR application captured with smartphone cameras", "100+K images" from front- and back-facing cameras.
- **Face Mesh V2:** smartphone captures "in a real-world environment" through an AR application; the z coordinate "is obtained from synthetic data".
- **Selfie Multiclass:** "thousands of samples with diverse identities", various expressions and rotations; the card names no dataset.

Google does not publish these images. **The owner's decision (2026-11-11, `docs/STATUS.md` question 45):** the weights are judged by their own licence, Apache-2.0, as the SAM 2.1 and Depth Anything V2 weights are (`docs/14`); the data are Google's own consented captures, undisclosed, and the owner accepted that. Yellowmoon has not downloaded, used or redistributed any of them and trains nothing here.

## What each network is

| | BlazeFace (full range) | Face Mesh V2 | Selfie Multiclass (256) |
|---|---|---|---|
| File | `blaze_face_full_range.f16.safetensors`, 1,067,186 bytes | `face_mesh_v2.f16.safetensors`, 2,448,944 bytes | `selfie_multiclass_256.f32.safetensors`, 16,321,304 bytes |
| Constants | 514,069 values, f16 | 1,199,316 values, f16 | 4,070,284 values, f32 (Google ships it in f32; we keep every bit) |
| Input | `1 x 192 x 192 x 3`, RGB, `x * 2 - 1` after a letterbox | `1 x 256 x 256 x 3`: the face cropped with a 25 % margin | `1 x 256 x 256 x 3` |
| Output | `1 x 2304 x 16` regressors (a box and six keypoints per anchor) and `1 x 2304 x 1` logits | `1 x 1 x 1 x 1434` (478 points x, y, z), `1 x 1 x 1 x 1` (face presence), `1 x 1` | `1 x 256 x 256 x 6` class scores (logits, softmax by us): background, hair, body-skin, face-skin, clothes, others |
| Operators | CONV_2D 49, DEPTHWISE_CONV_2D 40, ADD 23, RELU 47, MAX_POOL_2D 4, PAD 8, RESIZE_BILINEAR 3, RESHAPE 2 | CONV_2D 72, DEPTHWISE_CONV_2D 34, PRELU 69, ADD 34, MAX_POOL_2D 6, PAD 3, LOGISTIC 1, RESHAPE 1 | CONV_2D 67, DEPTHWISE_CONV_2D 18, ADD 25, MUL 18, SUM 6, SOFTMAX 6, TRANSPOSE 8, TRANSPOSE_CONV 1, RESIZE_BILINEAR 3, RESIZE_NEAREST_NEIGHBOR 3, RESHAPE 20 |
| Runtime in `ym-ml` | **done** (S-08) | **done** (S-09) | **done** (S-10; MUL, SUM, SOFTMAX, TRANSPOSE, TRANSPOSE_CONV and nearest resize added to `graph.rs`) |

**The file format.** Safetensors, tensors named `t<index>` after the TFLite tensor index (the f16 weights that sit behind DEQUANTIZE operators in Google's file are stored as they are, under the index of the tensor TFLite would have made from them), and the header metadata: `format` = `yellowmoon-tflite-graph-1`, `model`, `source`, `source_sha256`, `licence`, `converted_by`, and `graph`, a JSON text with the operators in execution order (type, input and output tensor indices, every builtin option), the integer constants (shapes, paddings) and the graph's inputs and outputs. `ym_ml::mediapipe::graph` reads that and runs it; it implements the operators of BlazeFace and Face Mesh V2 and refuses a graph with another when it loads.

## Conversion

`tools/mediapipe/convert.py` (Python 3.11, `tflite` 2.18.0, `numpy` 2.2.6; `tools/mediapipe/requirements.txt`), run in CI by `.github/workflows/models.yml` (job `mediapipe`). It downloads each original, refuses it unless its SHA-256 is the pinned one, reads the graph with the `tflite` FlatBuffers bindings, and writes the file with a writer of our own that sorts the tensors and the header keys, so the bytes depend on the original only (a second run gives the same bytes; the unit tests and the job check it). The job compares the three hashes with `SHA256SUMS` and publishes nothing if they differ. Python is used to convert and to make test fixtures only; nothing in the app or in `ym` runs it (`AGENTS.md` rule 6).

## Checks (BlazeFace; measured 2026-11-11 on a 4-core Linux Xeon)

`crates/ym-ml/tests/blazeface.rs` against `tests/fixtures/blazeface.json`, which `tools/mediapipe/reference.py` writes from Google's own TFLite interpreter (ai-edge-litert) and Google's own face detector task (mediapipe 1.2.2):

| Check | Bound | Measured |
|---|---|---|
| Raw network on a fixed synthetic input against the interpreter: largest logit difference | 1e-4 | 6.2e-6 |
| The same, regressors, relative to the largest regressor | 1e-5 | 8.9e-7 |
| 1 thread against 4 threads | equal bits | equal |
| Three public photos with MediaPipe's own resampling, against the interpreter on its tensor: IoU, score difference | at least 0.98, at most 0.01 | at least 0.9931, at most 0.0025 |
| The same against MediaPipe's own task (its box is whole pixels): worst edge, worst keypoint error, score difference | 2.5 px or 2 % of the box; 2 % of the box; 0.02 | 1.45 px; 0.33 %; 0.005 |
| The finder's defaults on the same photos against the task | IoU at least 0.85, same number of faces | at least 0.93, same |
| A 24 MP picture with seven faces from 49 to 390 px pasted in: faces found / planted, IoU | every face of 90 px or more once, none extra; IoU at least 0.8 | 5 of 5 (and 7 of 7 with a minimum of 30 px); IoU at least 0.877 |
| One 192 x 192 pass | measured | 12 ms on 4 threads, 20 ms on 1 |
| The finder on a 6000 x 4000 picture (36 windows) | measured | 240 ms on 4 threads, 0.8 s on 1 |

## Known limits

The cards' and ours. BlazeFace full range finds faces from about 7 % to 50 % of its frame (we measured: 10 px of 192 start to be found, 12 px reliably, and the score falls past 150 px), handles yaw to 90 degrees and roll and pitch to 45 degrees, and the card asks for about 70 % of the box inside the frame. Its card reports recall of 98.9 to 99.3 % across perceived gender, skin tone and region groups, and an average precision of 89.3 % across regions (88.1 to 89.6 %). The finder runs it on windows so that small faces in a large photo are found; a face under the minimum size (1.5 % of the longer side by default), a face turned past 90 degrees, a very small or cropped face and a face behind a hand or a mask can be missed, and drawings and statues can be found. Face Mesh V2 and Selfie Multiclass have their own limits (the mesh needs the eye line within 8 degrees of level; the segmenter's face-skin class includes the eyes, brows, lips and ears and its mean IoU is 68 for the darkest skin-tone group); S-09 and S-10 add them here when the runtimes exist.

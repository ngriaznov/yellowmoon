# Model card: `eomt`, the scene segmenter of Reference Grade (EoMT small, COCO panoptic)

**2026-11-25 (RG-03).** The weights file is `eomt_coco_panoptic_small.f16.safetensors` (47,978,758 bytes, SHA-256 in `SHA256SUMS`), converted by `training/eomt/export.py` from the authors' checkpoint `tue-mps/coco_panoptic_eomt_small_640_2x`. It is **the authors' model, not ours**: we did not train, fine-tune or prune it, and we change only the precision and the container. It runs in `ym_ml::eomt` (our own pure-Rust CPU code, `docs/34-reference-grade.md` section 4b.1, `docs/07`) when Reference Grade needs the scene of a photograph or of a reference.

**The weights are not in git.** They are too big. `tools/fetch/eomt.sh` downloads them from the Yellowmoon release `models-eomt-small-1` (a prerelease that is never the "latest" release) and checks the SHA-256 in `SHA256SUMS`; `release.yml` and `build.yml` are to do that before the app is built, so the app bundle holds them in `data/models/eomt/`, and `ym` finds them there like the other models. The app and `ym` never download anything (`AGENTS.md` rule 7). The model card, `SHA256SUMS` and the licence notice below are committed. **The release is not published yet** (the `eomt` job of `.github/workflows/models.yml` is written and has not run); the SHA-256 in `SHA256SUMS` is that of the file `export.py` made on 2026-11-25, which is byte-identical on every run, and the job refuses to publish a file with another hash.

## Purpose

For a picture the model says **which COCO panoptic class each pixel belongs to**: 133 classes, among them sky, tree, grass, sea, river, sand, road, building, wall and person. `ym_ml::eomt` turns that into a label map (one class or "unlabelled" per pixel, on the model's own 640 px grid) and `docs/34` section 4b.2 maps the classes to the **elements** that the Reference Grade node grades (sky, vegetation, water, ground, architecture, interior, person). It is used for **masks and measurements only**: nothing is trained for grading, and the model makes no grading decision. It runs once per photograph and once per reference, in the background, and the result is kept in memory for the session (`docs/34` section 11).

## Source and licence

| | |
|---|---|
| Model | EoMT, the Encoder-only Mask Transformer, **small**, the COCO panoptic checkpoint at 640 px ("2x" schedule). T. Kerssies, N. Cavagnero, A. Hermans, N. Norouzi, G. Averta, B. Leibe, G. Dubbelman, D. de Geus, "Your ViT is Secretly an Image Segmentation Model", CVPR 2025 (highlight), arXiv:2503.19108 |
| Code and checkpoint | `https://github.com/tue-mps/eomt` (the authors' code); the checkpoint is on Hugging Face as `tue-mps/coco_panoptic_eomt_small_640_2x`, in the format of the `transformers` library |
| Checkpoint files | revision `10f5326794a8ce7e0e975791033ae1c13aec3767`: `model.safetensors`, 95,927,796 bytes, SHA-256 `516b0b23acace639cc27739593881dda1c0010b3bec06aae138b615717fb8855`; `config.json`, 6,426 bytes, SHA-256 `de222958bcd41d63e57059ffb42bfedbe6dddc6f8ac61529af8558e86667d2cc`; `preprocessor_config.json`, 664 bytes, SHA-256 `8088c2597646d2a84effdfd11682ce70fd5e950c787fa86b4f4d9be706ddeef7` (the pins of `training/eomt/export.py`) |
| Code used to convert | `transformers` 5.24.1 with `torch` 2.15.0 (CPU), the versions pinned in `training/eomt/requirements.txt` |
| Licence of the code and the weights | **MIT**, Copyright (c) 2025 Mobile Perception Systems Lab at TU/e. The repository's `LICENSE` begins "MIT License / Copyright (c) 2025 Mobile Perception Systems Lab at TU/e" and the Hugging Face model card says `license: mit`. The repository states no separate licence for the weights. Read on 2026-11-24 (`research/` notes of the POC) and again on 2026-11-25 (RG-03): `LICENSE` on the repository's `master` and the Hugging Face API's tag `license:mit` for the pinned revision |
| Licence of the backbone | **Apache-2.0**: the network is a DINOv2 ViT-S/14 with four register tokens, initialised from Meta's DINOv2 weights (Meta Platforms, Inc. and affiliates, `https://github.com/facebookresearch/dinov2`, Apache License, Version 2.0; M. Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision", arXiv:2304.07193) |
| Variants not used | the DINOv3 checkpoints (`..._dinov3`) carry the DINOv3 licence and are not used; Mask2Former (CC BY-NC 4.0 weights), SegFormer (non-commercial) and the ADE20K-trained options were rejected for their terms or their cost |
| Notice | `NOTICE` item 15 carries the MIT text and the DINOv2 notice; this folder carries the notices too |

Notice (MIT): this product includes the EoMT small weights, Copyright (c) 2025 Mobile Perception Systems Lab at TU/e, from `https://github.com/tue-mps/eomt`, under the MIT licence:

> Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Notice (Apache-2.0, section 4): the backbone of the network is DINOv2 by Meta Platforms, Inc. and affiliates, licensed under the Apache License, Version 2.0 (`https://www.apache.org/licenses/LICENSE-2.0`), from `https://github.com/facebookresearch/dinov2`. The weights in this product are the EoMT authors' checkpoint converted to f16 safetensors (the changes: precision and container); they were not trained or fine-tuned further.

## Training data (the authors', as the model card and the paper state it; we use none of it)

- **COCO panoptic 2017** (the 133-class set of 80 "thing" and 53 merged "stuff" classes). COCO's terms of use (`https://cocodataset.org/#termsofuse`): the annotations belong to the COCO Consortium and are licensed under **Creative Commons Attribution 4.0**; the images come from Flickr, the COCO Consortium does not own their copyright, and their use is subject to Flickr's terms. The weights are the product of that training and carry the authors' MIT licence; **Yellowmoon has not downloaded, used or redistributed any COCO image or annotation** (the same position as for SAM's SA-1B and Depth Anything's data, `docs/14`), and the model card of the checkpoint is the authors' to keep correct. This product contains no COCO data.
- The backbone's pre-training: DINOv2 on LVD-142M, a curated set of web images assembled by Meta (arXiv:2304.07193); the weights are Meta's Apache-2.0 release.

## What we run

The whole of `EomtForUniversalSegmentation` for this checkpoint, as the `transformers` code builds it (a port of the authors' code), and nothing else:

| Part | What | Parameters |
|---|---|---|
| Embedding | a 16 × 16 convolution of stride 16 (3 to 384 channels), a class token, **4 register tokens**, a learned position table of 1600 rows (a 40 × 40 grid; the class and register tokens get no position), and the 200 learned queries | 0.97 M |
| Transformer | 12 blocks of a ViT-S: LayerNorm (epsilon 1e-6), global attention with 6 heads of 64 and biases on q, k, v and the output, LayerScale, LayerNorm, an MLP of 1536 with the exact GELU, LayerScale. **The 200 learned queries are put in front of the stream before block 9**, so blocks 9 to 11 run on 1805 tokens and blocks 0 to 8 on 1605. A final LayerNorm | 21.3 M |
| Class head | one linear layer on the 200 queries: 134 outputs (133 classes and "none") | 51 k |
| Mask head | three linear layers with GELU on the 200 queries | 0.4 M |
| Upscaling | two blocks on the 40 × 40 patch tokens: a 2 × 2 transposed convolution of stride 2, GELU, a depthwise 3 × 3 convolution with no bias, a LayerNorm over the channels; 40 to 160 px | 1.2 M |
| Output | the mask logits are the product of each query's mask embedding and every position of the 160 × 160 map: 200 × 160 × 160 | |

In all **23,976,079 parameters** in 244 tensors, stored as **f16**: 47,978,758 bytes. The file's header (`config`, a JSON string) holds the architecture, the 133 class names of `id2label`, the pre-processing and the post-processing, so that the Rust runtime refuses a file that is not what it implements, or whose class table is not its own.

**Patch 16, not 14.** The training configuration of the authors names `vit_small_patch14_reg4_dinov2` (DINOv2's patch of 14), and the open point of `docs/34` was which one the checkpoint has. The checkpoint says **16**: the patch convolution is `[384, 3, 16, 16]` and the position table has 1600 rows (40 × 40 = 640 / 16). With patch 14 the grid would be 45 × 45 for 640 px, and the table would not fit. `export.py` and the runtime both refuse any other size. The token count is 1605 (class, registers, patches) plus 200 queries from block 9.

**No attention mask at inference.** The model's forward pass builds a masked attention for the last three blocks only when `attn_mask_probs[block] > 0` (or when training); this checkpoint's three values are **exactly 0** (the mask was annealed away), so inference attends to everything and makes no intermediate prediction. The converted file keeps the three values and the runtime refuses a file where one is not 0.

**Input:** the Input's picture (the Neutral look, 8 bits, sRGB, 1024 px on the long edge), resized to fit 640 × 640 keeping its aspect ratio (the size rule of `transformers`' `get_size_with_aspect_ratio`; an antialiased triangle filter in the 8-bit integer arithmetic of PyTorch's kernel), padded at the bottom and the right with black, then ImageNet's mean and standard deviation. **Output:** 200 class logits of 134 values and 200 mask logits of 160 × 160, and from them the panoptic label map of the `transformers` post-processing with its defaults (a query is kept when its class is not "none" and its probability is above 0.8; each pixel goes to the kept query with the largest probability times its mask's sigmoid; a query is a segment when it owns more than 0.8 of its mask), on the fitted grid (at most 640 × 640); a pixel no segment holds is "unlabelled".

## Conversion

`training/eomt/export.py`, run in CI by `.github/workflows/models.yml` (job `eomt`, not run yet; the pins are in `training/eomt/requirements.txt`). It:

1. downloads the three files and refuses any whose SHA-256 or size is not the pinned one;
2. builds the model with `transformers` and loads the checkpoint (a missing, unexpected or mismatched key stops it), compares every tensor name and shape with its own list written from the model's code, and checks the configuration: patch 16, a 40 × 40 position table, 4 registers, 200 queries, **`attn_mask_probs` all zero**;
3. keeps every tensor, with its name, as f16 (the loss's class weight `criterion.empty_weight` included: the file is a plain copy of the checkpoint, and the runtime lists it as unused);
4. writes the file itself, byte for byte the same on every run (two runs on 2026-11-25 gave the same SHA-256), and reads it back with the `safetensors` library, comparing every tensor bit for bit;
5. runs the model on a fixed synthetic picture with the original f32 weights and with the f16 weights, and writes the difference to `export_report.json`.

Python is used to convert and to make fixtures only; nothing in the app or in `ym` runs it (`AGENTS.md` rule 6).

## Fixtures and checks

`tools/eomt/parity_fixture.py` runs the original model (`transformers` 5.24.1, CPU, f32, one thread) with the f16 weights widened to f32 on three small pictures made in the script (a landscape, a street with a person, a shore; 3:2, 2:3 and square), and writes `crates/ym-ml/tests/data/eomt/` (committed, 260 KB): the pictures, the original's label maps, and the class logits, mask logits and kept queries of 16 queries per picture. `crates/ym-ml/tests/eomt.rs` compares our runtime with it, and with the fixture of any folder named in `YM_EOMT_EXTRA_FIXTURE` (a `--photos` run of the script: the 11 real photographs and film stills of the local check). Without the weights the gates print `SKIPPED` and pass; `YM_REQUIRE_EOMT=1` makes that a failure.

| Gate | Bound | Measured (worst, Linux, 2026-11-25) |
|---|---|---|
| The model input against the original's (8-bit values, sampled) | at most 1 level, at most 1 % of the samples | 0 levels on the committed pictures; 1 level on 0.35 % of the samples at most (real photographs) |
| Class logits (134 per query, 16 queries per picture) | max abs 0.15, relative RMS 4e-3 | max abs 0.070, relative RMS 1.8e-3 (6.1e-3 and 2.4e-4 on the committed set) |
| Mask logits (100 points per query, logits of order 10) | max abs 1.2, relative RMS 4e-3 | max abs 0.56, relative RMS 1.7e-3 (0.018 and 2.0e-4 on the committed set) |
| Kept queries and their classes | the same set | the same on all 14 pictures; probability within 8e-3 (measured 3.2e-3) |
| **The network alone** (the picture whose input is the original's to the bit) | class logits within 5e-4, mask logits within 2e-3, the label map on every pixel | 2.5e-5, 1.0e-4, 100 % of the pixels |
| **Label map: pixels with the same class as the original's post-processing** | at least 99.9 % | **99.974 %** (the worst of 14 pictures, a committed one; 99.980 % to 100 % on the others) |
| Element map (`docs/34` section 4b.2 applied to both) | at least 99.9 % | 99.974 % (the classes that merge into one element can only raise it) |
| 1, 2 and 8 threads, a second run | equal bits (the input, the logits, the label map) | equal, on the real weights and on synthetic ones |

What the differences are: the 8-bit resize is the same integer kernel as PyTorch's and matches it on all but 0 to 0.2 % of the pixels (one level); after that, float rounding through 12 blocks of attention. They flip a pixel only on the border between two segments.

**The cost of f16 storage** against the original f32 checkpoint, measured with the original code on the 14 pictures: the label map agrees on at least **99.94 %** of the pixels (99.9998 % at best), the class logits move by at most 0.076 and the mask logits by at most 1.2 (logits of order 10; relative RMS 9.5e-4 and 1.3e-3 on the export's synthetic picture).

**Against the POC's labels** (the original f32 weights on the full-size film stills, in Python): our reference pipeline's element map equals the POC's on 99.4 % of the pixels on average over 8 stills (97.8 % to 99.9 %); the differences are the f16 weights, the 1024 px input instead of the 3840 px original, and the library version.

**Cost.** About **68 billion multiply-adds** per picture, the same for every picture (the input is always 640 × 640). On the Linux test machine (4 logical cores, our pure-Rust CPU code, f32): about **6 s on 1 thread, 3.1 s on 2 and 1.7 s on 4** for the network, and under 20 ms for the preparation of the input and the post-processing. No Apple-silicon figure exists yet (`docs/34` section 11).

## Known limits

The authors' and the data's, as they show in the label map:

- **PQ 46.7 on COCO panoptic** for this checkpoint (the authors' model zoo); it is the small model. It is good at the large regions that Reference Grade needs (sky, water, ground, buildings, people) and weaker on small things and on classes that are rare in COCO.
- **COCO's classes are coarse.** There is no separate "bush", "plant" or "lawn": plants are merged into `tree-merged` and `grass-merged`, a mountain, a hill and a rock face are `mountain-merged` or `rock-merged`, a wall of any material may be `wall-other-merged`. `potted plant` is a thing class (`docs/34` section 4b.2 puts it in vegetation).
- **A pixel can be unlabelled.** A query is kept only above a probability of 0.8, and a segment needs most of its mask to be its own, so uncertain regions (a night sky, a blown-out window, heavy haze, a dark interior) are often left unlabelled, and so are the borders between segments. The label map is a decision on a 160 × 160 mask grid raised to 640 px: its edges are soft, 4 px on that grid, and a guided filter (`docs/34` section 4a.3) refines them.
- **Sky and water at night or in haze, and anything an unlike COCO picture** (an unusual lens, a heavy colour grade, a scene of one colour) can be wrong. The segmenter reads the Input's picture in the Neutral look, which is the one the model was trained on in kind (a normally exposed sRGB picture).
- **People include their clothes**: the class `person` is the whole person; skin comes from MediaPipe (`docs/34` section 4b.3).
- **Glass, mirrors and reflections** are labelled as what they show or as a surface, unpredictably.
- **The unusual aspect ratios**: the picture is fitted to 640 px on the long side and padded; a panorama of 5:1 is 640 × 128 and its detail is low.
- **The mask grid is 160 × 160 for a 640 px input**, so a thing smaller than about 4 px of that grid (16 px of the 640 grid, 25 px of a 1024 px picture) may be missed.

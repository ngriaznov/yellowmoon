# Model card: `depth`, Depth masks (Depth Anything V2-Small)

**Draft 0, 2026-11-07 (D-01).** Nothing here is measured yet: every number marked TBD is filled in by D-02 (the weights and the oracle), D-03 (the runtime) or D-06 (the evaluation), and the card is final only after D-06. The weights file is `depth_anything_v2_vits.f16.safetensors` (about 50 MB; the size and the SHA-256 are TBD in `SHA256SUMS`), converted by `training/depth/export.py` from the authors' `depth_anything_v2_vits` checkpoint. It is **the authors' model, not ours**: we did not train, fine-tune or prune it, and we change only the precision and the container. It runs in `ym_ml::depth` (our own pure-Rust CPU code, `docs/07`, `docs/algorithms/06-masks.md` section 8) when a node has a Depth mask row (`docs/23` section 4.8).

**The weights are not in git.** They are too big. `tools/fetch/depth.sh` (D-02) downloads them from the Yellowmoon release `models-depth-anything-v2-small-1` (a prerelease that is never the "latest" release; the name is proposed) and checks the SHA-256 in `SHA256SUMS`; `release.yml` and `build.yml` do that before the app is built, so the app bundle holds them in `data/models/depth/`, and `ym` finds them there like the other models. The app and `ym` never download anything (`AGENTS.md` rule 7). The model card, `SHA256SUMS` and the licence notice below are committed.

## Purpose

For a Depth mask the model gives, for the photo, a **relative depth map**: one value per pixel of a picture of 518 px on the short side, larger for nearer (to be confirmed by the first oracle fixture, D-02). The map is made once per photo, in the background, on the Input's picture that the Object mask reads, and held in memory. A Depth row then cuts a band of it by near, far and softness, with no more network runs (`docs/algorithms/06` section 5.8). The model says **which plane**; the edge comes from the photo (Refine edges) and from the Subject, People and Object masks. It is used for **masks only**: it does not drive any retouching, and lens character does not use it (the owner, 2026-10-24).

## Source and licence

| | |
|---|---|
| Model | Depth Anything V2, the **Small** size (ViT-S, `depth_anything_v2_vits`, 24.8 million parameters in the authors' table). L. Yang, B. Kang, Z. Huang, Z. Zhao, X. Xu, J. Feng, H. Zhao, "Depth Anything V2", arXiv:2406.09414 |
| Code and checkpoint | `https://github.com/DepthAnything/Depth-Anything-V2`; the Small weights are also on Hugging Face as `depth-anything/Depth-Anything-V2-Small` |
| Checkpoint file | the file name, size and SHA-256 are TBD (D-02 pins them in `training/depth/export.py` and writes them here) |
| Code used to convert | the `depth_anything_v2` package at a commit TBD (D-02), CPU only |
| Licence | **Apache-2.0.** The authors' README says: "Depth-Anything-V2-Small model is under the Apache-2.0 license. Depth-Anything-V2-Base/Large/Giant models are under the CC-BY-NC-4.0 license." The Hugging Face card of the Small model says Apache-2.0 too. Read on 2026-11-07 (`research/depth-mask-2026-11-07.md`); D-02 reads the README sentence and the repository's `LICENSE` again at the pinned commit and records the date here. **Only the Small model is used.** The Base, Large and Giant models are non-commercial and are not shipped, not run and not used as teachers. |
| Notice | `NOTICE` item 13 names the model, the authors and the link; this folder carries the notice too |

Notice (Apache-2.0, section 4): this product includes the Depth Anything V2-Small model weights, by Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng and Hengshuang Zhao, licensed under the Apache License, Version 2.0 (`https://www.apache.org/licenses/LICENSE-2.0`), from `https://github.com/DepthAnything/Depth-Anything-V2`. The weights were converted to f16 safetensors (the changes: precision and container); they were not trained further. (D-02 checks the repository for a `NOTICE` file or a copyright line that the licence asks us to carry, and adds it here and to `NOTICE`.)

## Training data (the authors', as the paper and the README state it; we use none of it)

Depth Anything V2 is trained in two steps (arXiv:2406.09414). A large **teacher** (DINOv2-Giant, never released) is trained on **labelled synthetic images** from five sets: BlendedMVS, Hypersim, IRS, TartanAir and Virtual KITTI 2. The teacher then labels about 62 million **real, unlabelled images** (the paper lists BDD100K, Google Landmarks, ImageNet-21K, LSUN, Objects365, Open Images V7, Places365 and SA-1B), and the **student models, the Small one among them, are trained purely on those pseudo-labelled real images**, not on the synthetic sets. The sizes of the sets and their terms are in the paper and on the sets' own pages; this card does not repeat numbers it has not checked.

Two of the five synthetic sets have terms that are not Apache-2.0: **Hypersim is CC BY-SA 3.0** and **Virtual KITTI 2 is licensed "for non-commercial purposes only"** (their own pages, read on 2026-11-07). SA-1B, among the real images, is Meta's set with its own terms (the SAM 2.1 card names it too).

**The owner's decision (2026-11-07, `docs/STATUS.md` question 41): the teacher chain is accepted.** The Small weights are the authors' Apache-2.0 release; the restricted data sit behind an unreleased teacher, so the whole chain from the data to the weights stays inside one licensor's Apache-2.0 grant, as the SAM 2.1 card accepts SA-1B and SA-V behind Meta's Apache-2.0 checkpoint. Yellowmoon judges the weights by their own licence (`docs/14`, rule 5 and the Depth Anything V2-Small row), has **not downloaded, used or redistributed any of these datasets**, and trains nothing here. Models distilled from the released non-commercial Large weights (DepthART, Distill-Any-Depth) are refused: there the restriction is on the teacher's weights.

## What we run

The whole of `DepthAnythingV2` for ViT-S, as the authors' code builds it, and nothing of the other sizes or of the metric head:

| Part | What | Parameters |
|---|---|---|
| Encoder | DINOv2 ViT-S/14: a 14 × 14 patch embedding of stride 14, a class token, a positional embedding trained on 37 × 37 patches, 12 blocks (global attention, 6 heads of 64, MLP ratio 4 with GELU, LayerNorm eps 1e-6, LayerScale), the blocks 2, 5, 8 and 11 taken through the final norm | TBD |
| DPT head | four 1 × 1 projections to 48, 96, 192 and 384 channels, their resize layers (a transposed convolution of stride 4, one of stride 2, the identity, a convolution of stride 2), four 3 × 3 convolutions to 64 channels, four fusion blocks, the output head (3 × 3 to 32 channels, a bilinear resize to the input size, 3 × 3 to 32, ReLU, 1 × 1 to 1, ReLU) | TBD |

In all about **24.8 million parameters** (the authors' figure), stored as **f16**: about 49.6 MB of tensors plus a header. The tensor count and the exact size are TBD (D-02). The file's header (`config`, a JSON string) holds the architecture, so that the Rust runtime refuses a file that is not what it implements. Settings that the code defaults decide (`use_bn` false, `use_clstoken` false) are checked against the checkpoint's tensors by the conversion script (D-02).

**Input:** the Input's picture (the Neutral look, 8 bits, sRGB, 1024 px on the long edge), resized bicubically, the aspect kept, to a short side of 518 px with both sides a multiple of 14 (a 3:2 photo is 770 × 518; the long side is capped at 1036 px), then ImageNet's mean and standard deviation. **Output:** one non-negative value per input pixel. Meaning and sign: see Purpose (TBD, D-02).

## Conversion

`training/depth/export.py`, run in CI by `.github/workflows/models.yml` (a `depth` job; the pins are in `training/depth/requirements.txt`; D-02 writes the versions here). It is planned to:

1. download the checkpoint and refuse it unless its SHA-256 is the pinned one;
2. build the authors' model and load the checkpoint (a mismatch of names or shapes raises);
3. keep the tensors, with their names, as f16 (`format` `yellowmoon-depth`, `version` `1`, the config, the source URL, its SHA-256 and the commit in the header);
4. write the file itself, byte for byte the same on every run, and read it back with the `safetensors` library, comparing every tensor bit for bit;
5. run the authors' code on a fixed synthetic picture with the original f32 weights and with the f16 weights, and write the difference to `export_report.json` (a CI artefact of the run).

Python is used to convert and to make fixtures only; nothing in the app or in `ym` runs it (`AGENTS.md` rule 6).

## Fixtures and checks (proposed gates, TBD until measured)

`training/depth/oracle.py` runs the authors' code (CPU, f32, one thread) on 20 public photos (CC0 or public domain; `training/depth/oracle_photos.json` lists each with its page, licence, author and SHA-256; the photos are not committed) with the published weights widened to f32, and writes the picture, the resized tensor, the four encoder maps and the depth map. They are the release asset `depth-oracle-fixtures-1.tar.zst` (fetched into `testdata/depth/oracle-1/` by `tools/fetch/depth.sh --fixtures`) and feed the gate of D-03 (`docs/algorithms/06` section 8.6): the resize within 1e-3 largest and 1e-4 mean; the encoder's maps within 2e-3 of their RMS; the depth map within 2e-3 of its RMS and 2e-2 of its largest value; **the cut at five bands per photo with a mask IoU of at least 0.99**; a second run bit-identical; the same bits on 1, 2 and 8 threads. The cost of f16 storage against the original f32 checkpoint is measured in the same run and reported, never hidden.

| Gate | Bound | Measured |
|---|---|---|
| Resize against the oracle's tensor | 1e-3 largest, 1e-4 mean (proposed) | TBD |
| Encoder maps from the oracle's tensor | 2e-3 of the RMS (proposed) | TBD |
| Depth map from identical tensors | 2e-3 of the RMS, 2e-2 of the largest (proposed) | TBD |
| Cut IoU, 20 photos × 5 bands | at least 0.99 (proposed) | TBD |
| 1, 2 and 8 threads, a second run | equal bits | TBD |
| Cost of f16 against the f32 checkpoint | reported | TBD |
| Time for the map at 518 × 770, 4 cores Linux | measured (our estimate before measuring: 2 to 2.5 s) | TBD |
| DIODE ordering against the oracle's | within half a point (starting) | TBD |

## Known limits

The authors': the model gives a **relative** depth, with no metric scale and no fixed zero; its ordering of near and far is good (their DA-2K benchmark orders 95.3 % of the point pairs correctly for the Small model against 97.1 % for the Large one, arXiv:2406.09414), and its edges are soft at 518 px with no boundary score published for the Small model. For a mask that means:

- **Hair, fur, foliage and fine structure**: no depth model resolves them (the best boundary recall on a hair set in Depth Pro's paper is 0.173 for Depth Pro and 0.107 for Depth Anything V2). Use Refine edges and a Subject or Object row for the edge.
- **Glass, mirrors, water and windows** take the depth of what they show, or of their surface, unpredictably; translucent surfaces and scattering are ill-posed for any depth model.
- **Sky and haze** collapse to the far end whatever is in them, so a far band takes the sky with the distant hills.
- **A flat scene** (copy work, a wall) has almost no depth to order, and the rank spreads its small differences over the whole range.
- **A smoothly defocused background** (portraits at f/1.4 to f/2.8): a defocused picture lacks the edges the model relies on; the Background mask, from blur, is strongest there and Depth weakest.
- **Macro and ultra-wide pictures** have not been tested by anyone we found; D-06 tests them on public photos.
- The picture is **518 px on the short side**, about 60 times fewer pixels than a 24 MP photo (770 × 518 against 6000 × 4000): the map is coarse (a cell is about 8 px on a 6000 px export), and Refine edges does the rest.

The contact sheets of D-06 (hair, foliage, sky, glass, macro, shallow-focus backgrounds, flat copy work) are summarised here when they exist.

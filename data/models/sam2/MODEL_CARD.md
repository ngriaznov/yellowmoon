# Model card: `sam2`, Object masks (SAM 2.1 small, image path)

Version 1, 2026-11-02. `sam2.1_hiera_small.f16.safetensors` (about 77.1 MB, SHA-256 in `SHA256SUMS`), converted by `training/sam2/export.py` from Meta's `sam2.1_hiera_small.pt`. It is **Meta's model, not ours**: we did not train, fine-tune or prune it, and we only drop the parts of the video model that an image never uses. Runs in `ym_ml::sam2` (our own pure-Rust CPU code, `docs/07`, `docs/algorithms/06-masks.md` section 6) when a node has an Object mask row (`docs/23` section 4.4).

**The weights are not in git.** They are too big. `tools/fetch/sam2.sh` downloads them from the Yellowmoon release `models-sam2.1-small-1` (a prerelease that is never the "latest" release) and checks the SHA-256 in `SHA256SUMS`; `release.yml` and `build.yml` do that before the app is built, so the app bundle holds them in `data/models/sam2/`, and `ym` finds them there like the other models. The app and `ym` never download anything (`AGENTS.md` rule 7). The model card, `SHA256SUMS` and the licence notice below are committed.

## Purpose

The person clicks the photo or draws a box, and the model answers with a mask of the object there. The image encoder runs once per photo, the prompt encoder and the mask decoder run per click. The model finds "an object", not a named one. The mask is a function of the picture and of the points and the box alone, so settings that hold only the points reproduce it (`docs/23` section 4.4).

## Source and licence

| | |
|---|---|
| Model | SAM 2.1, the **small** size (`sam2.1_hiera_small`, "46" million parameters in Meta's table), N. Ravi et al., "SAM 2: Segment Anything in Images and Videos", arXiv:2408.00714 |
| Code and checkpoint | `https://github.com/facebookresearch/sam2`, © Meta Platforms, Inc. and affiliates |
| Checkpoint file | `https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt` (the link in the README), 184,416,285 bytes, SHA-256 `6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38` (computed on 2026-11-02) |
| Code used to convert | the `sam2` package at commit `2b90b9f5ceec907a1c18123530e92e794ad901a4` (`main` on 2026-11-02; no tags), CPU only |
| Licence | **Apache-2.0.** The README says: "The SAM 2 model checkpoints, SAM 2 demo code (front-end and back-end), and SAM 2 training code are licensed under Apache 2.0, however the Inter Font and Noto Color Emoji used in the SAM 2 demo code are made available under the SIL Open Font License, version 1.1." The repository's `LICENSE` is the Apache License, Version 2.0. Nothing of the demo (the fonts) is used. |
| Notice | `NOTICE` names the model, Meta and the link; this folder carries the notice too |

Notice (Apache-2.0, section 4): this product includes the SAM 2.1 small model weights, Copyright (c) Meta Platforms, Inc. and affiliates, licensed under the Apache License, Version 2.0 (`https://www.apache.org/licenses/LICENSE-2.0`), from `https://github.com/facebookresearch/sam2`. The weights were converted to f16 safetensors and cut down to the image path (the changes: precision, container, and the video parts removed); they were not trained further.

## Training data (Meta's, as Meta states it; we use none of it)

Meta trained SAM 2 on **SA-V** (the "Segment Anything Video" dataset of videos with mask annotations that Meta released with the model, described in `sav_dataset/README.md` of the repository) together with **SA-1B** (the images and masks of the first Segment Anything) and other video datasets, as the SAM 2 paper describes (arXiv:2408.00714; the 2.1 checkpoints of 2024-09-29 are trained with additional data and augmentation, as the README says). The sizes and the terms of those datasets are in the paper and on Meta's dataset pages; this card does not repeat numbers it has not checked (the paper was not opened for it). We have not downloaded, used or redistributed any of these datasets, and Yellowmoon trains nothing here: the weights are Meta's, under the licence above.

## What we run

The image path of the image predictor (`SAM2ImagePredictor`) and nothing of the video model:

| Part | Tensors (names as in Meta's checkpoint) | Parameters |
|---|---|---|
| Image encoder, Hiera-S trunk | `image_encoder.trunk.*` (patch embedding, `pos_embed`, `pos_embed_window`, 16 blocks) | 33,947,328 |
| Image encoder, FPN neck | `image_encoder.neck.*` (four 1 x 1 convolutions to 256 channels) | 369,664 |
| Prompt encoder | `sam_prompt_encoder.*` (random-Fourier positional encoding, four point embeddings, `not_a_point_embed`, `no_mask_embed`, the mask-input path, which we do not use) | 6,476 |
| Mask decoder | `sam_mask_decoder.*` (two-way transformer of depth 2, output upscaling, `conv_s0`, `conv_s1`, four hypernetwork MLPs, IoU head, object-score head) | 4,215,109 |
| `no_mem_embed` | added to the stride-16 feature map (`directly_add_no_mem_embed`) | 256 |

In all **359 tensors and 38,538,833 parameters**, stored as **f16**: 77,077,666 bytes of tensors plus a header, about 77.1 MB. Dropped from the 519 tensors and 46,060,610 parameters of the checkpoint: `memory_attention.*`, `memory_encoder.*`, `maskmem_tpos_enc`, `no_mem_pos_enc`, `no_obj_ptr`, `no_obj_embed_spatial`, `mask_downsample.*`, `obj_ptr_proj.*`, `obj_ptr_tpos_proj.*` (160 tensors, 7,521,777 parameters; all of the video model).

Architecture as in `sam2/configs/sam2.1/sam2.1_hiera_s.yaml`: input 1024 x 1024; Hiera with embedding dimension 96, one head at the first stage, stages of 1, 2, 11 and 2 blocks, channel widths 96, 192, 384 and 768 at strides 4, 8, 16 and 32, windowed attention with windows 8, 4, 14 and 7 and global attention in blocks 7, 10 and 13, queries pooled 2 x 2 at the start of stages 2 to 4, a 7 x 7 background plus 8 x 8 window positional embedding; an FPN neck of 256 channels (`fpn_top_down_levels` 2 and 3, nearest-neighbour upsampling, `scalp` 1, sine position encoding); a prompt encoder and a two-way mask decoder of 256 dimensions, 8 heads and MLP 2048, with high-resolution features (`conv_s0`, `conv_s1`), 3 + 1 mask tokens, an IoU head with a sigmoid and an object-score head. The file's header (`config`, a JSON string) holds all of it, read from the model Meta's code built, so that the Rust runtime refuses a file that is not what it implements.

## Conversion

`training/sam2/export.py`, run in CI by `.github/workflows/models.yml` (Python 3.11, CPU, torch 2.5.1, torchvision 0.20.1, safetensors 0.5.3, `sam2` at the commit above; the pins are in `training/sam2/requirements.txt`):

1. downloads the checkpoint and refuses it unless its SHA-256 is the pinned one;
2. builds Meta's model and loads the checkpoint (a mismatch of names or shapes raises);
3. keeps the 359 tensors above, with their names, as f16 (`format` `yellowmoon-sam2`, `version` `1`, the config, the source URL, its SHA-256 and the commit in the header);
4. writes the file itself, byte for byte the same on every run (the `safetensors` writer orders the header's map by hash), and reads it back with the `safetensors` library, comparing every tensor bit for bit;
5. runs Meta's encoder on a fixed synthetic picture with the original f32 weights and with the f16 weights, and writes the difference to `export_report.json` (a CI artefact of the run);

The release notes name the commit of the `models` run that made the assets, which is the version of the conversion script (`docs/07`).

Python is used to convert and to make fixtures only; nothing in the app or in `ym` runs it (`AGENTS.md` rule 6).

## Fixtures and checks

`training/sam2/oracle.py` runs Meta's `SAM2ImagePredictor` (CPU, f32, one thread, deterministic algorithms) on 20 public photos (CC0 or public domain; `training/sam2/oracle_photos.json` lists each with its Wikimedia Commons page, licence, author and SHA-256; the photos are not committed) with three prompts each (one positive click, a positive and a negative click, a box), using the published weights widened to f32, and writes the picture, Meta's input tensor, the image embedding and the high-resolution maps, and for each prompt the low-resolution logits, the IoU predictions, the object score and the chosen mask. They are the release asset `sam2-oracle-fixtures-1.tar.zst` (fetched into `testdata/sam2/oracle-1/` by `tools/fetch/sam2.sh --fixtures`) and feed the gate of N-17 (`docs/13`, `docs/algorithms/06` section 6.5): mask IoU at least 0.99 at 256 x 256, logits within 0.05 largest and 0.005 mean, the encoder's maps within 2e-3 of their RMS, a second run bit-identical, the same bits on 1, 2 and 8 threads. The same run compares Meta's code with these f16 weights against Meta's original f32 checkpoint and writes the numbers to `report.json` in the fixtures; the cost of f16 storage is reported there, never hidden.

## Known limits

Meta's: thin structures, glass and reflections, a click on a boundary between two objects, very small objects, crowded scenes where "the object" is ambiguous (a single click decodes three masks and the most likely one is taken; the Object card lets the person add or remove points or draw a box). The model was trained on video and photos of the kinds Meta's datasets hold; it may do worse on unusual material (macro, infrared, heavy stylisation). It finds regions, not names. The 1024 x 1024 resize does not keep the photo's aspect ratio (as Meta's `SAM2Transforms`), so a very wide panorama is squeezed; masks are at 256 x 256 before they are stretched back, so fine edges come from Refine edges (`docs/23` section 4.3), not from the model.

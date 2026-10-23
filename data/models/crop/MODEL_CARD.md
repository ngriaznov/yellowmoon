# Model card: `crop`, Auto crop's crop model

Version 1, 2026-10-23. `cropnet.safetensors` (4,414,072 bytes, SHA-256 in `SHA256SUMS`), exported by `training/crop/export.py` from the run `ft-pre-box-nall` of `training/crop/` (branch `auto-crop`). Runs in `ym_ml::crop_net` (`crates/ym-ml/src/crop_net.rs`), once per photo, on the CPU, when the person turns on Auto in Crop mode or `ym` is asked for `--suggest-crop` or `--auto-crop`. Licence of the weights: Apache-2.0, as the repository (`docs/14` rule 5: trained only on CC BY photos and annotations and on the labels of an Apache-2.0 model).

## Purpose

Looks at a photo's picture (768 px on its long edge, `docs/20-auto-crop.md` §4) and answers one free crop: `[x0, y0, x1, y1]` as fractions of the picture. Auto crop turns that crop into the frame of each aspect (`docs/20` §7). It does not straighten (the straightener is geometry, `docs/20` §5) and does not use the camera's focus point.

## Architecture

The picture letterboxed into 256 × 256 (Pillow's bilinear resize, reproduced to the bit in Rust), normalised `(v / 255 − 0.5) / 0.25`. A 3 × 3 stem of stride 2 (24 channels), seventeen MobileNetV2-style inverted-residual blocks in five stages (expansion 1, then 4; 24, 32, 48, 96, 160 channels; strides 1, 2, 2, 2, 2), a 1 × 1 neck to 32 channels at 8 × 8, linear layers of 256 and 4; SiLU after every layer but the projections and the last; batch norms folded into the convolutions at export. A sigmoid on the four outputs gives `p`, and `x0 = p0`, `x1 = x0 + (1 − x0) p2` (the same for y), so every answer is a box inside the picture. 1,101,852 parameters, 263 M multiply-adds, f32; about 120 ms on one core of a 2.8 GHz Xeon.

## Training

Two steps, both on Modal (L4 GPUs) from GitHub Actions, seeds fixed (GPU training is not bit-for-bit repeatable):

1. **Pretraining of the backbone** (`training/crop/boxes.py`, `pretrain.py`, run `pre-oi-96000-s2-e12`). 96,000 photos of Open Images V7's boxable train set, chosen with seed 2 among those listed as CC BY 2.0 and stored upright, none of the crop set's. Targets from the boxes people drew on them: six maps of 16 × 16 cells over the letterboxed square, each cell the share of it covered by boxes of any object, of people, faces, animals, vehicles and buildings (the classes under those names in Open Images' hierarchy), and the set of the 601 boxable classes present. Two small heads, dropped afterwards. 10 epochs (21 minutes; the time cap fitted 10 of 12), AdamW 2e-3, horizontal flips and brightness and contrast jitter. On 1,865 held-out photos the most likely class was one the photo holds 58 % of the time.
2. **Crop training** (`training/crop/train.py`, run `ft-pre-box-nall`). The backbone above, then the whole network on 11,999 other Open Images photos (seed 1 among the CC BY 2.0 upright ones), each labelled with FrameLite's free crop (FrameLite: an Apache-2.0 cropper built on Qwen3.5-0.8B, run offline as a teacher, code, weights and base model at pinned revisions in `training/crop/teacher.py`; the owner allowed it on 2026-10-23). 10,827 to train, 1,172 to validate (by a hash of each photo's id). L1 on the edges plus (1 − IoU), 40 epochs, AdamW 1e-3, flips and brightness and contrast jitter; the epoch with the best validation IoU is kept.

| Input | Licence | Use |
|---|---|---|
| Open Images V7 photos (Flickr, via the CVDF mirror), 107,999 | listed as CC BY 2.0 by their authors | pretraining (96,000) and crop training (11,999); every photo with its author, title and page in `training/crop/CREDITS.tsv.gz` |
| Open Images V7 boxes and class hierarchy | CC BY 4.0 (Google) | pretraining targets |
| FrameLite's crops of the 11,999 photos | outputs of an Apache-2.0 model (FrameLite and Qwen3.5-0.8B) | crop labels |
| The 150 held-out Play Raw photos (`testdata/cropeval/heldout/`) | CC BY-SA (pixls.us Play Raw) | **evaluation only**, never trained on |

FrameLite itself was fine-tuned on crop datasets without a licence (GAICD, CPC, FCDB); those photos never reach this model, only FrameLite's answers on our CC BY photos do. Whether a teacher's own training data matter is the owner's decision of 2026-10-23 (`docs/STATUS.md`, question 22).

## Metrics

Agreement with FrameLite's crop (mean intersection over union), against doing nothing (the whole photo):

| Set | This model | Whole photo | Same network from scratch |
|---|---|---|---|
| Open Images validation, 1,172 photos | 0.868 | 0.830 | 0.861 |
| Held-out Play Raw, 150 photos | 0.847 | 0.751 | 0.822 |

Blind pairs on the 150 held-out photos against the whole photo, judged by the agent and then unblinded (a tie counts half): the free crop won 56, lost 10 and tied 39 of the 105 pairs where it moved (72 %; FrameLite itself 67 %); the closest frame of the photo's own shape (Auto crop's Original) kept the whole photo on 37 and won 36, lost 15 and tied 62 of the other 113 (59 %). One judge who had seen the photos before: a pointer. The gate is the owner's blind A/B (`tools/eval/ab`, the crop test).

**Known weaknesses.** It never asks for more than the photo holds and rarely crops hard (its crops keep 82 % of the photo on average, FrameLite's 75 %). Its losses on the held-out photos: mountain tops in fog, a house at the frame's edge, a bridge tower and a lit tree cut, too tight on a crocus, a lizard, a windsurfer, a runner and a sunset.

## Parity

`crates/ym-ml/tests/data/cropnet_parity.json` (written by `export.py`) holds the training framework's answers for five drawn pictures and the SHA-256 of their letterboxed bytes: the Rust letterbox matches Pillow exactly and the numbers agree within 2e-4 (2.2e-7 measured).

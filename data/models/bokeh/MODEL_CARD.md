# Model card: `bokeh`, lens character's bokeh map

Version 1, 2026-10-25, model `bokeh-3`. `bokehnet.safetensors` (6,100,536 bytes, SHA-256 in `SHA256SUMS`), exported by `training/bokeh/export.py` from the run `bokeh-3` of `training/bokeh/` (branch `lens-character`). Runs in `ym_ml::bokeh_net` (`crates/ym-ml/src/bokeh_net.rs`), once per photo, on the CPU, when the lens plan reshapes bokeh (`LensPlan::needs_blur_map`: Anamorphic above 0, or Edges above 0 on a lens with swirl or cat's eye; `docs/21-lens-character.md` §7). Licence of the weights: Apache-2.0, as the repository (`docs/14` rule 5: a network trained from scratch with our own code on synthetic defocus made from CC BY 2.0 photos and CC BY 4.0 masks, no pretrained weights). If this folder is missing or damaged, the lens falls back to the classical detector of `docs/21` §7 and logs a warning.

## Purpose

Looks at the lens plane of a photo (scene-linear camera RGB, 1024 to 2047 px on its long edge) and answers, for every 2 x 2 pixels of the plane the network saw, the radius of the disc blur the photo already carries and how sure that is. It measures blur, not depth: a far wall at f/11 is sharp. Lens character uses it to find the out-of-focus areas it may reshape (it never adds blur where there is none). It does not know the focus point (the lens adds the subject anchor itself), does not find bright balls (the lens does, on the plane), and does not run on the display picture.

## Architecture

The plane is box-reduced by exact area weights to a long edge of 1024 px (a plane that is not longer is used as it is, never enlarged), encoded with one fixed transform (the lower median of luminance mapped to 0.18, then `log2(1 + v / K) / 16` per channel with `K = 0.18 / 32`, then `(e - 0.30) / 0.15`; every constant is in the weights' header), padded with replicated edge pixels to a multiple of 32, run, and the answer cropped to half resolution. A MobileNetV2-style encoder: a 3 x 3 stem of stride 2 (16 channels), twelve inverted-residual blocks in six stages to stride 32 (widths 24, 32, 64, 96, 224, 416; ReLU6), a global-context block at stride 32 (global average pooling, 1 x 1 convolution, added to every cell), a small U-Net decoder (bilinear upsampling by 2, `align_corners = false`, skips at strides 16, 8, 4 and 2) and a 1 x 1 head at half resolution with two outputs, `mu = ln r` (`r` in pixels of the 1024-px frame, floored at 1 px) and `s = ln beta` (`beta` the Laplace scale of the error of `mu`). Batch norms are folded into the convolutions at export. **1,522,122 parameters** (1,531,258 while training), **3,364 multiply-adds per input pixel** (about 2.4 G for a 1024 x 683 plane), f32. The Rust runtime reads the architecture from the file's header and refuses a file whose encoding constants, activation, strides or tensor shapes differ from what it implements.

Reading the outputs (`docs/21` §7): `b = max(exp(mu), 1 px) / hd` in half-diagonals of the reduced plane; `c = exp(-beta / 0.25)` (how sure the *radius* is: high on sharp areas, low inside strong blur, so it is **not** the decision); and **`P = P(b >= 2 b_min)`**, the tail of the Laplace(`mu`, `beta`) the network gives for `ln r` with `b_min = 0.004` (2.5 px on the 1024-px frame): the probability that a cell is out of focus. `P` is a function of the two outputs and nothing else.

Speed (the same architecture, measured on an earlier export; 4 shared cores, the dev profile with `opt-level = 1`, so a release build is several times faster): a 1024 x 683 plane takes about 1.1 s in the network, 2.2 s on one thread; the mirrored pass (below) doubles it; loading the weights 60 to 70 ms. Output bits are identical on every run and any number of threads (`f32` in a fixed order, no fused multiply-add).

## Training

On Modal (one L4 each) from GitHub Actions, seeds fixed (GPU training is not bit-for-bit repeatable). Everything is in `training/bokeh/` (`README.md` has the recipe, the validation and the go/no-go).

**Data.** Sharp photos only; the blur is synthesised with a known radius, so the label is the blur that was applied. 20,000 Open Images V7 photos (the ones the crop pipeline had put on the volume, at most 1024 px on the long side) were filtered for sharpness (a tile is kept only if at least 85 % of its 32-px blocks are sharp-textured and at most 4 % soft) and cut into 512 x 512 tiles at native scale: 30,283 tiles from 16,091 photos, 29,071 to train and 1,212 to validate (by a hash of the photo's id, so no photo is on both sides). Open Images' instance masks (10,768 of the photos have them, 29,108 masks) cut sprites out of other tiles, so outlines are realistic without a depth model. Each batch is rendered on the GPU: tiles in linear light, a stack of layers drawn back to front with blurred-alpha compositing (the background, one to four sprites, up to six layers in the stack mode, groups of small bright lights that make out-of-focus balls on dark backgrounds too), each with its own defocus radius (0 to 34 px) and kernel (disc, soft, bright, dark and onion rims, 5 to 9 blades, cat's eye, Gaussian), then a sensor (blur, exposure, colour cast, Poisson-Gaussian noise, clip, white-balance-like gains). Version 3 of the synthesis added what bokeh-2 had never seen: thin sharp structures in front of a blur (hair, twigs, wires, grass; five procedural kinds), surfaces at every depth (a stack whose radii are spread evenly in the log radius), and featureless layers (skies and walls whose blur cannot be read from the picture, weighted 0.2 far from an edge), with a half-resolution target in which a mixed block takes the sharp radius.

| Input | Licence | Use |
|---|---|---|
| Open Images V7 photos (Flickr, via the CVDF mirror), 16,091 of them in tiles | listed as CC BY 2.0 by their authors | the background and the sprites of every training scene; every photo with its author, title and page in `training/bokeh/CREDITS.tsv.gz` |
| Open Images V7 instance masks (385 MB archives, the official mirror) | CC BY 4.0 (Google) | cutting objects out for layers |
| Layout, radii, kernels, lights, noise | made here (`synth.py`, `kernels.py`, `structures.py`) | the synthesis |
| CUHK's blur detection set (704 out-of-focus photos) | research-only, no licence for training or redistribution | **local evaluation only** (below), by the owner's decision of 2026-10-24; never trained on, never uploaded to the training volume, never committed |

**Never used:** CUHK, DUT-DBD and RTF for training, SYNDOF/SYNTHIA, Hypersim, DPDD, DIV2K, ImageNet, COCO, and pretrained weights of any kind. The credits list (16,091 rows) is in the repository and not in the app bundle, as the crop model's is: the model card and `NOTICE` name the dataset and point to it.

**Runs.**

| Run | Start | Synthesis | Steps | Minutes | Cost (L4) | Used |
|---|---|---|---|---|---|---|
| `bokeh-1` | scratch | v2 | lost to a NaN (a negative second moment in the dark-rim kernel; fixed at the root and guarded in `train.py`) | 100 | $2.49 | no |
| `bokeh-2` | scratch | v2 | 14,977 | 101.2 | $2.51 | the start of `bokeh-3`; shipped nowhere |
| `bokeh-3` | `bokeh-2`'s weights (its EMA) | v3 | 10,536 (none skipped) | 101.4 | $2.51 ($2.69 with its GPU smoke run) | **this model** |

`bokeh-3`: AdamW, learning rate 1e-3 (half of the first run's, since the weights were already good) with 300 warm-up steps and a cosine schedule over elapsed time, weight decay 0.01, an EMA of the weights (0.999, the EMA is what is evaluated and exported), batch 24 at 512 x 512, bf16 autocast, gradient clipping, a Laplace negative log-likelihood of `ln r` at half resolution, seed 1. Starting from `bokeh-2` was chosen because v3's data contain bokeh-2's (about 80 % of the samples have no thin structure or featureless layer), so the 100 minutes went to what is new. The programme's whole Modal spend (indexing, masks, tiles, previews, the three GPU smoke runs, the three trainings including the lost one) is **$8.25 of the owner's $10 cap** (the programme's own ledger). Export: the folded network differs from the trained one by 3.5e-5 at most (the export refuses above 2e-4).

## Metrics

All on held-out tiles (never trained on), fixed seeds, in the spec's `b` units. `bokeh-3` final evaluation, 576 samples of the v3 synthesis (`metrics.json` of the run):

| Measure | `bokeh-3` |
|---|---|
| mean absolute error of `b` where the truth is at least `b_min` | 0.00381 (relative error 0.202) |
| sharp pixels predicted at `2 b_min` or more (`fp_rate`) | 1.56 % (0.034 % where `c >= 0.5`) |
| recall of blur of at least `2 b_min` | 92.1 % |
| calibration error of the scale (`ece`; the go/no-go limit is 0.1) | 0.032 |
| precision / recall of the decision `P >= 0.7` | 95.5 % / 86.7 % |
| recall by blur size at `P >= 0.7`: `2 b_min` to 0.012 / 0.012 and above | 63.9 % / 93.3 % |
| thin sharp structures over a blur called blurred at `P >= 0.7` (raw radius rule: 37.1 %) | 28.0 % |
| the far inside of a sharp featureless layer called blurred at `P >= 0.7` (raw radius rule: 15.1 %) | 10.1 % |

**Against `bokeh-2` on `bokeh-2`'s own validation** (its synthesis, half-resolution target, tiles, seed and samples; `final.legacy` of `bokeh-3` against `bokeh-2`'s `final.model`): mean absolute error of `b` 0.00348 against 0.00409; sharp pixels called blurred 1.00 % against 1.46 %; recall 94.1 % against 93.2 %; `ece` 0.040 against 0.036. Nothing the old recipe taught was lost. On the new groups `bokeh-2` was measured earlier on a different, smaller set of tiles (24 samples of the local smoke tiles; thin structures 11.9 %, featureless interiors 47.9 % at `P >= 0.7`), so those two numbers are not a like-for-like comparison with the table above; the real planes below are the comparison that matters.

**Against the classical detector** (`training/bokeh/classical.py`, a stripped-down Zhuo and Sim; the same 48 samples): mean absolute error of `b` 0.00488 against 0.01821; recall 89.2 % against 0.6 % (the classical detector has no estimate in a region with no edge left); sharp pixels called blurred where `c >= 0.5`: 0.009 % against 0.002 %. The go/no-go of `training/bokeh/README.md` asks for no more false positives than the baseline's: the network's absolute rate is under 0.01 %, 7e-5 above the baseline's on this subset; the owner decided on 2026-10-25 to ship on the whole of the evidence in this card.

**Real sharp tiles, no synthesis** (48 tiles): sharp pixels predicted at `2 b_min` or more 0.046 % (0.037 % at `P >= 0.7`; none where `c >= 0.5`), against `bokeh-2`'s 0.060 % and the classical detector's 0.

**Real Sony lens planes** (the 28 planes of `testdata/lens-photos/`, 1371 to 1848 px long; the ARW files are public samples, git-ignored; the lens's own rules, mirrored pass; judged by the agent: 7 photos have shallow depth of field with a clearly blurred background, 21 do not). Share of the frame called blurred (`P >= 0.7`) and "has bokeh" (`P >= 0.9`), `bokeh-2` and `bokeh-3`:

| Photo (id) | Bokeh | blurred b2 / b3 | has bokeh b2 / b3 |
|---|---|---|---|
| 50585 | yes | 88.5 / 88.0 % | 86.3 / 77.7 % |
| 56893 | yes | 72.8 / 71.0 % | 64.3 / 64.5 % |
| 26276 | yes | 64.7 / 62.4 % | 46.4 / 49.9 % |
| 27308 | yes | 59.6 / 52.6 % | 46.2 / 35.6 % |
| 28322 | yes | 53.7 / 36.3 % | 4.6 / 9.4 % |
| 14077 | yes (a backlit portrait, mild blur) | 3.9 / 5.1 % | 0 / 0 % |
| 29348 | yes (neither this network nor the classical detector sees it) | 0 / 0 % | 0 / 0 % |
| 16238 | no (patches of sky) | 14.3 / 17.7 % | 1.2 / 2.7 % |
| 55204 | no (patches of sky) | 36.9 / 6.5 % | 1.1 / 0 % |
| 60942 | no (patches of sky) | 36.6 / 14.6 % | 5.7 / 0 % |
| 61159 | no (patches of sky, a tower) | 4.6 / 6.3 % | 0 / 0.2 % |
| 27998 | no | 0.3 / 1.2 % | 0 / 0 % |
| the other 16 sharp photos | no | 0 / 0 % | 0 / 0 % |

Across the 28 photos `bokeh-3` is more specific than `bokeh-2` where it matters for the gate: the largest "has bokeh" share on a photo without bokeh falls from 5.7 % to 2.7 %, and the patches of sky of 55204 and 60942 fall from 1.1 and 5.7 % to nothing. It keeps the five photos with a clear bokeh and reads their out-of-focus areas in the same shapes as `bokeh-2` (the dog's ear and fur edge and the dandelion's stem side by side on the comparison sheets), with a little less of the mildly blurred mid-ground at `P >= 0.9`. The gate that turns these shares into "this photo has bokeh" is in "The decision" below, with the numbers of the full lens path.

**CUHK, local evaluation only, not training data** (704 `out_of_focus` photos at their own size, 500 to 640 px; the mask polarity read from the data, 66.6 % of the pixels are out of focus; the numbers are for precision, not for segmentation, since the mask's blur is mostly mild and the photos are web JPEGs): the share of the mask's out-of-focus pixels the network finds and its precision, `bokeh-2` against `bokeh-3` (the plain pass): `P >= 0.7` precision 0.995 / 0.992, recall 0.033 / 0.030; `P >= 0.9` precision 0.998 / 0.997, recall 0.015 / 0.016; mirrored pass at `P >= 0.7` precision 0.997 / 0.997, recall 0.025 / 0.022. In short, nothing changed on that set: precision above 0.99 and a recall of a few percent. Do not train toward it.

## The decision

What the lens does with the outputs (the numbers are in `docs/21` §7):

- A cell is **blurred** when `P >= 0.7`: it carries its `b`, every other cell carries 0, and each weighs `max(P, 1 - P)`, how sure the decision is. A gate on `c` would throw the blur away.
- **"The photo has bokeh"** counts the cells with `P >= 0.9`, which is the lower-quantile rule `b * exp(-1.61 beta) >= 2 b_min` and keeps the low-confidence blobs a featureless sky gets out of the count. The lens reshapes a photo when at least **5 %** of the frame is at `P >= 0.9` (`MAP_MIN_STRONG_AREA` in `crates/ym-lens/src/l1/detector.rs`) or it holds five clear balls, and at least 50 cells are measured. Measured with the full lens path (the weights of this folder, the mirrored pass, the balls rule, the anchor and the cleaning) on the 28 real planes above: the share at `P >= 0.9` is 77.7, 64.5, 50.0, 35.5 and 9.4 % on the five photos with a clear bokeh (the last, the Christmas tree, also holds 21 clear balls), 0.0 % on the backlit portrait and on 29348, 2.7 % on the sky-patch photo 16238 and 0.2 % or nothing on the other 20 photos without bokeh. Any threshold from 2.75 % to 35.5 % gives the same result (five of seven reshaped, none of 21; from 2.75 % to 9.4 % without the balls rule), and 5 % is the geometric middle of the gap between the largest share of a photo without bokeh (2.74 %) and the smallest share a photo with bokeh needs on its own (9.37 %), a factor of 1.8 from each. The old 2 % (the classical detector's) would also have reshaped 16238. The sample is small: five photos with bokeh and one sky photo at the edge. The full table is in `docs/21-lens-character.md` §7, *The gate's share*.
- **Mirrored pass** (on): the network runs a second time on the padded input flipped left to right, the answer is flipped back and the two are combined (mean `mu`, scale `mean(beta) + |mu_1 - mu_2| / 2`). It cuts the thin sharp structures read as blur (31 % to 10 % of their pixels in the synthetic scenes of `training/bokeh/README.md`, `bokeh-2`) and the blobs on a featureless sky, at the cost of a second run (1.5 s a photo against 0.8 s, median, same machine as above).
- The anchor at the focus point, the bright balls, the sharp-edge test and the cleaning of specks are not the network's; they stay the classical detector's, the way `docs/21` §7 describes.

## Known limits

- **It misses mild blur.** A blur of under about `2 b_min` (about 5 px on a 1024-px frame is the line) is called sharp by construction, and the blur in the photos it misses is mostly of that size: the backlit portrait (14077) and 29348 read 0 % "has bokeh" and the lens leaves them as they are. It is high in precision and low in recall by design.
- **More conservative on busy mid-ground than `bokeh-2`.** The Christmas tree (28322) reads as blurred over 36 % of the frame against `bokeh-2`'s 54 %, and the out-of-focus background of 27308 over 53 % against 60 %, because the v3 data teach it that texture at mid distance can be in focus. The reshaping then acts on the clearly blurred part only.
- **Low-confidence blobs on featureless sky.** A sky with faint cloud patches still gets blobs at `P >= 0.7` (up to 18 % of a frame in 16238) that carry a radius the picture cannot support; they stay out of the "has bokeh" count (the 5 % share at `P >= 0.9`) and the mirrored pass cuts them, but they are what an eye sees in the map. The lens only reshapes where the photo has bokeh, so it does not act on them alone.
- **Thin structures are not solved.** On the synthetic groups 28 % of the pixels of a sharp twig or hair over a blur are still called blurred at `P >= 0.7` (the sharp-edge test and the mirrored pass bring that down in the lens, and the real planes show no false blur on glasses wires, hair, twigs or railings), and the synthetic structures are procedural (constant colour along a strand, no texture) where real hair and branches are not.
- **The synthetic-to-real gap** is the main risk: noise, exposure and clip are modelled, a Sony lens plane is not. The tests of it are the real sharp tiles and the real planes above; the training data has no raw file in it, only JPEG photos from Flickr.
- **Natural bokeh in the sources** is filtered out of the training tiles (a sharpness filter), not removed; a photo with natural bokeh that passed makes its blurred background a "sharp" label. The self-cleaning round of `training/bokeh/README.md` is not built.
- The numbers above are from synthetic data, 48 real sharp tiles and 28 real planes judged by one agent; none of it is a blind test by the owner.

## Parity

`crates/ym-ml/tests/data/bokehnet_parity.json` (written by `export.py`) holds the training framework's folded network's answers for four planes (three integer patterns of 96 x 64, 64 x 96 and 32 x 32 and one linear plane of 64 x 64) and the encoder's probes, with the SHA-256 of the weights. `cargo test -p ym-ml --test bokeh_net` reproduces them from the committed weights: the network agrees with PyTorch's within 2.1e-5 (the limit is 1e-4), the encoder's probes within 4.7e-10 (limit 1e-6), a whole plane through `map` to the fixture's `b` (in pixels), `c` and `P` within 2.3e-5, and the probability against the training code's within 1.3e-7. Two runs and any number of threads give identical bits, on planes that need padding and on one that is reduced.

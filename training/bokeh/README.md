# training/bokeh: the learned bokeh map

Lens character (`docs/21-lens-character.md`) reshapes the blur a photo *already has*; it never adds blur. For that it needs,
for every pixel of the lens plane, the radius `b` of the disc blur the pixel already carries (in half-diagonal units of the
frame) and how sure that is. It is blur, not depth: a far wall at f/11 is sharp. No pretrained defocus-map or
blur-detection model is licence-clean (`research/bokeh-map-2026-10-24.md`), so this directory trains our own small network
(`model.py`, 1.5 M parameters, from scratch) on synthetic defocus with a known blur, made from licence-clean sharp photos.
Python only trains, exports and writes fixtures: the app and `ym` never run it (`AGENTS.md` rule 6); a Rust runtime
(`ym_ml`, like `crop_net.rs`) will run the exported weights. The weights of **bokeh-3** ship in `data/models/bokeh/` with their model card (the owner's decision of 2026-10-25;
the go/no-go below, the metrics and the known limits are in that card).

The owner's decisions of 2026-10-24: the learned route is allowed ("Bokeh map extraction can use ML if needed"), **Modal may
be used, cap $10** ("Yes, use Modal for the bokeh map, cap $10"; about $11 of the account's hard $30 limit is left), and
the research-only blur sets (CUHK, DUT-DBD, RTF) may be used for **local evaluation only** (below).

## Data and licences (`docs/14` rule 5)

| What | Source | Licence | Used for |
|---|---|---|---|
| Sharp photos | Open Images V7 photos, the ones the crop pipeline already put on the Modal volume `yellowmoon-crop` (`data/train-20000-s1/images`, at most 1024 px on the long side, the CVDF mirror), read in place, not downloaded again | listed CC BY 2.0 per photo (author and page in the volume's `manifest.json`) | the background and the sprites of every training scene |
| Instance masks | Open Images V7 segmentation, `train-annotations-object-segmentation.csv` and the 16 archives `train-masks-<c>.zip` (385 MB each, official mirror `storage.googleapis.com/openimages/v5`) | CC BY 4.0 (Google) | cutting objects out for layers: realistic outlines without a depth model |
| Everything else | made here: the layout, radii, kernels, lights, noise (`synth.py`, `kernels.py`) | ours | |

Weights trained on CC BY photos keep the repository's Apache-2.0. Every photo with a tile is credited (author, Flickr page,
title) in `tiles/credits.tsv.gz`, written by `prepare` in the format of `training/crop/CREDITS.tsv.gz`, for the model card
and `NOTICE`; the list of the 16,091 photos of the shipped model is committed as `training/bokeh/CREDITS.tsv.gz`. **Never used:** CUHK, DUT-DBD, RTF, SYNDOF/SYNTHIA, Hypersim, DPDD, DIV2K, ImageNet, COCO (research-only,
non-commercial, share-alike or unclear), and no pretrained weights of any kind. DIODE (MIT) is left out of the first run to keep
it simple; its val split is small. The crop volume's 96,000 pretraining photos exist there only as 256 x 256 letterboxed arrays, too
small for samples at the 1024-px scale, so they are not used; `prepare --extra N` can download more photos from the mirror
for a second run.

**Evaluation only (`eval_real.py`).** CUHK's blur detection set, DUT-DBD and RTF have no licence that allows training or
redistribution. The owner allows them for local evaluation (2026-10-24) and nothing else. They are never trained on, never
uploaded to the Modal volume, never committed (`testdata/bokeh-eval/` is git-ignored), never redistributed, and nothing in
the pipeline imports `eval_real.py`. See its docstring for what it reports and for the two datasets (DUT-DBD, RTF) that
you must fetch yourself.

## The frame, the units and the encoding (decided here, the app must match)

- **Scale.** The network runs on the lens plane box-reduced to a **long edge of 1024 px** (`encode.FRAME_LONG`; the plane is
  1024 to 2047 px long, never enlarged), the pixel scale of the training photos. A disc of `b = 0.004` (`b_min`) is 2.5 px
  there and `b_cap = 0.03` is 18 px.
- **Units.** The network answers in *pixels of that frame* (`mu = log r`, `r` floored at 1 px), not in half-diagonal units:
  a convolutional network sees pixels, and the half-diagonal depends on the aspect ratio. The app converts:
  `b = exp(mu) / hd`, `hd = hypot(w, h) / 2` of the reduced plane. Training labels are stored as `r` in pixels with each
  tile's `hd`, so every metric below is in the spec's `b`.
- **Confidence.** The second output `s = log beta`, `beta` the Laplace scale of the error of `mu` (a relative error: 0.25 is
  about 25 %). `c = exp(-beta / 0.25)`; the spec's rule is "`b` is 0 where `c` is low", the metrics gate at `c >= 0.5`.
- **Input.** One fixed transform (`encode.py`), no per-image tone tricks: luminance `Y` (Rec. 709 weights) of the scene-linear
  camera RGB, its lower median `m`, `v = RGB * 0.18 / m`, `e = log2(1 + v / K) / 16` per channel with `K = 0.18 / 32` (linear
  near black, logarithmic above, 100 times the median is still inside the range), then `(e - 0.30) / 0.15`. A plane
  twice as bright gives the same input. `export.py` writes every constant into the weights' header and a fixture the Rust
  tests must reproduce.
- **Fully convolutional.** Any size divisible by 32; the runtime pads the plane to that with replicated edge pixels and crops
  the half-resolution output. The stride-32 global average pooling means a pixel's answer depends on the whole plane.
- **The label** is the radius of the uniform disc with the same second moment as the kernel (`sqrt(2 * sum(k * r^2))`), whatever
  the kernel's shape (polygon, cat's eye, rings): the size a viewer reads and the spec's `b` for balls. The label is the
  blur that was *added*; a source's own blur is label noise.

## Pipeline

| Step | Module | Where |
|---|---|---|
| list photos, read the masks' names from the segmentation CSV | `prepare.build_index`, `sources.mask_index` | Modal CPU (`index_photos`) |
| fetch the masks of those photos from the 16 archives | `sources.fetch_mask_pack` | Modal CPU, 16 calls (`mask_archive`) |
| sharpness filter, cut 512 x 512 tiles at native scale, instance-id maps, shards | `sources.photo_tiles`, `prepare.tile_chunk` | Modal CPU, 1,000 photos a call (`tile_chunk`) |
| credits, counts | `prepare.finalize` | the orchestrator |
| look at the synthesis | `previews.py` | Modal CPU (`preview`), or locally |
| render scenes, train, evaluate, export | `synth.py`, `structures.py` (v3), `kernels.py`, `train.py`, `export.py` | Modal L4 (`train_run`), or locally on the CPU for a smoke run |
| the classical baseline, the sharpness measure | `classical.py` | numpy, in `train.py`'s evaluation and `sources.py` |
| budget plans and the ledger | `budget.py` | everywhere |
| local evaluation on research-only sets | `eval_real.py` | locally only |
| models side by side on the synthetic scenes | `compare.py` | locally |

**Why scenes are rendered on the GPU, not before.** A pre-rendered set of 0.5 M samples at 512 x 512 is hundreds of
gigabytes. `train_run` copies the tiles (about 128 KB each as JPEG with a 1 KB mask: roughly 4.5 GB for 36,000 tiles) to local disk
and renders each batch on the L4 in PyTorch (FFT convolution per layer), so there is no `synthesize` step; `preview` shows what the
renderer makes.

### The synthesis in one paragraph (`synth.py` has the full recipe)

Tiles to linear light, highlights near clip raised 1.4 to 16 times, then a layer stack drawn back to front with
*blurred alpha* compositing (SYNDOF's layered recipe): the background tile, 1 to 4 sprites cut out of *other* tiles with their
Open Images masks (so no hole is left behind a lifted object), and 0 to 2 groups of small bright lights (dots and patches,
warm, neutral, cool and a few saturated) that add light and are hidden by what is in front, so out-of-focus balls appear on
dark backgrounds too (up to a fifth of the samples of a batch with lights are "night", about 13 % overall: the scene is
darkened 5 to 32 times against its lights). Each layer gets a defocus radius (0 for the one in focus; 1 px to 34 px,
which is up to about `b = 0.05`, beyond `b_cap` for coverage) and a kernel from a library of disc, soft rim, bright rim, dark rim,
onion rings, 5 to 9 blades, cat's eye and Gaussian. 30 % of the samples are all in focus (plain tiles and collages), 25 % of the
batches have a tilted background plane (radius ramps across it, four radius bins blended). Then a sensor: a mild blur in
40 %, an exposure that puts the median at 3 to 30 % of clip (night 0.4 to 5 %), a colour cast, Poisson-Gaussian noise
at several levels, the clip at 1, white-balance-like gains in half of the samples (so clip levels differ per channel). A
pixel's label is the radius of the layer that contributes most to it; where no layer dominates (the band around a blurred
edge) the loss weight falls to 0.25. `tests/test_synth.py` renders a step edge, a sharp subject, a lamp, a blurred
foreground and a ramp and measures the blur back: it equals the label.

**Sample sheets** (`preview`, `smoke.py`): per row the network's input (the encoded plane), a display version of it, the label
as colour (0 to `b = 0.05`, dark where the weight is low) and, from v3, the loss weight as grey (cyan on thin structures, magenta
on featureless layers).

### v3: what the synthesis added (`SynthConfig.v3()`, `structures.py`; `SynthConfig()` is still bokeh-2's recipe, bit for bit)

The owner's review of bokeh-2 ("much better, can we improve it?") and the measurements below (CUHK, the 28 Sony planes, and bokeh-2
on v3's own scenes) pointed at three holes in the data, all in what the network never saw:

1. **Thin structures in focus in front of a blur** (hair, twigs, wires, grass, chair legs). Five kinds of procedural curves (`structures.TYPES`),
   drawn antialiased with a width that sums to its line width across the line, in the hue of a place in another tile and moved in brightness
   until they contrast with the background, as a layer with its own sharp mask in front of the other layers: in 55 % of the samples of a batch
   with one (0 to 2 such layers per batch; 40 / 45 / 15 %); one batch in five has them out of focus instead (a defocused wire in front).
   The loss weight of a pixel they dominate is at least 0.9 (a one-pixel line covers a pixel by under 0.9, and its own label is the one to learn).
2. **Surfaces at every depth.** A fifth mode, 30 % of the samples: a *stack* whose sprites' radii are spread evenly (one in each slice of the log radius)
   between 1.5 px and the background's, a few in front, the first sprite in focus half of the time; the layer count goes from 2 to 5 to 2 to 6
   (10 / 25 / 30 / 20 / 15 %). The all-in-focus mode falls from 30 to 22 %.
3. **Featureless layers** (a sky under a wavy horizon, a wall or sheet of 1 to 3 ellipses; one layer in 40 % of the batches, in 60 % of its
   samples, in focus 55 % of the time, and the whole background of 7 % of the samples). Gradient and faint clouds only, sharp antialiased
   edges. Far (more than 8 px plus the blur) from the edge of every layer the radius cannot be read from the picture: the weight there is
   0.2 and rises to 1 within reach of an edge. (The pixels there are not left out: a region that is uniform to the noise is told apart from a
   blurred wash by its appearance, and its true labels, in focus 55 % of the time, teach the network to answer "probably sharp, but not sure"
   there, which the decision rule below turns into "not blurred". At zero weight nothing would teach it, and the many smooth pixels of blurred layers
   would say "blur".)

The half-resolution target changed with it (see "The half-resolution target" in `train.py`'s docstring: **a mixed block takes the sharp
radius**), `skip_identity` leaves out the FFT of layers without blur (an exact identity), and the previews have a fourth panel (the weight).
The synthesis is about 2 to 3 times dearer on the CPU (more layers; the structures' rasterisation); its cost on the L4 is unmeasured: the
GPU smoke prints it (minutes at step 25, 50, ...).

### The sharpness filter (the recipe's main label noise)

A Flickr photo with natural bokeh would label its blurred background "in focus". `classical.patch_fractions` reads the blur
of every edge in 32-px blocks (the re-blur gradient ratio; a disc radius above 2.5 px is soft), and a tile is kept only if at
least 85 % of its blocks are sharp-textured and at most 4 % soft-textured. On 56 sample photos it kept tiles of 45; the
photos it dropped are the ones that are bokeh or flat throughout. **A leak:** the sharp subject of a shallow-depth photo can still
make a passing tile, with up to 15 % of its blocks flat, blurred background among them (labelled "in focus"). Flat but sharp
regions (sky, walls) are the cost of the rule. Not done, and the next step if label noise shows: the research note's
self-cleaning round (run the first model on the unblurred sources and drop or down-weight pixels where it already predicts blur).

## The model (`model.py`)

MobileNetV2-style encoder (stride-2 stem of 16 channels, six stages of inverted residual blocks with ReLU6 to stride 32,
widths 24, 32, 64, 96, 224, 416), a global-context block at stride 32 (global average pooling, 1x1 convolution, ReLU6, added to
every cell), a small U-Net decoder with bilinear upsampling by 2 (`align_corners=False`, no transposed convolution)
and skip connections at strides 16, 8, 4, 2, and a 1x1 head at **half resolution**. **1,522,122 parameters** (folded; 1,531,258
trained, with the batch norms), **3,364 multiply-accumulates per input pixel** (about 2.4 GMAC for a 1024 x 683 plane; the
research budgeted "about 3 k"). Operations: convolution (1x1, 3x3, 3x3 depthwise, zero padding 1, stride 1 or 2), batch norm (folded
at export), ReLU6, bilinear 2x, add, concat, global average pooling. The parameters are deliberately deep: a parameter at stride 32
costs a thousandth of a MAC per pixel, one at stride 2 a quarter.

## Training (`train.py`)

Loss: Laplace negative log-likelihood of `log r` at half resolution. bokeh-2: a 2 x 2 block averaged its log radius, and one that
mixed radii more than `e^0.35` apart counted 30 %, so a one-pixel twig in a blur was learnt as a mild blur. **v3: in a mixed block the
sharp radius wins** (the smallest radius among the pixels of weight at least a half, at 70 %; `half_targets`, `--no-sharp-wins` is the old rule).
Optional: `--decision-loss w` adds a cross-entropy of the runtime's decision probability against the label (off: CUHK and the synthetic
groups show no calibration failure it would mend). `--init-from bokeh-2` starts from a finished run's weights (its `model.pt`, EMA) with a
new optimiser, schedule and step count: the old knowledge is kept and the run learns what v3 added; give a lower `--lr` (1e-3). AdamW (lr 2e-3, 300 warm-up steps), an EMA of the weights (0.999) is what is evaluated and
exported, bf16 autocast, gradient clipping. **The cosine schedule runs over elapsed time**, not over steps: the throughput of the L4
is not measured, and a run stopped by its time limit would otherwise end at a high learning rate. A checkpoint every 5 minutes; a rerun
with the same output folder continues from it. Batch 24 at 512 x 512 (random crops of the 512 tiles are possible for smaller samples).

**The first long run (bokeh-1, 2026-10-24) was lost to a NaN, and what stops that now.** Its loss was 0.93 at step 25 and NaN from
step 50 on; it trained a network of NaNs for 100 minutes ($2.49) and died in the final sheet (`colorize` indexed a colour ramp with
a NaN: a device-side assert). Cause: the `dark` (dark-rim) kernel of `kernels.py` had a negative profile on its antialiased rim
cells when its radius was under about 3 px and its rim strength above 0.69, so a kernel of one or two pixels had a negative
second moment, the label-radius square root was NaN, and the NaN kernel (through the FFT) turned its whole sample into NaN. About
one kernel in 300 below 3 px was affected, and a batch of 24 samples draws ~100 kernels, many of them small (ramp bins, layers
of 1 to 3 px), so roughly every fourth batch held one. Fixed at the root (the dark profile is capped at the rim, coverage is never
negative, the second moment is clamped) and guarded in `train.py`: a step with a non-finite loss or gradient norm is skipped
(no optimiser step, no EMA update, batch-norm statistics put back) and logged with the input that was not finite; the run stops
(`TrainingDiverged`, "training diverged: ...") after 21 skipped steps in a row, when more than 2 % of the steps after step 200
were skipped, when a weight or an evaluation is not finite, or when the mean loss of steps 25 to 100 is NaN or above 5; no
checkpoint of NaN weights is written, resumed or exported. A run that diverges costs minutes, not the budget. Run the GPU smoke
(`modal_app::train --budget 0.8 --minutes 8 --steps 1000 --run bokeh-smoke`, about $0.3; v3's is in "Running it") before a long run: it is the full
configuration, and its log must show finite losses (that fall, from a fresh network) and no `skipped` lines. A run name that was used before resumes from that run's checkpoint,
so a new run takes a new name.

### Validation and the go/no-go

On held-out tiles (a hash of the photo id; never trained on), fixed seeds, pixels with label weight at least 0.6, in `b` units
(`metrics.json`, `final`): `mae_b` where the truth is at least `b_min` (and gated at `c >= 0.5`, with its `coverage`); `fp_rate` the share
of pixels with truth below `b_min` that are predicted at `2 * b_min` or more (and `fp_rate_gated`); `recall_2bmin`; calibration (ten bins
of the predicted scale against the mean absolute log error, and `ece`); the same numbers for the classical baseline (`classical.py`)
on the first 48 samples next to the model's on the same samples; and `real_sharp`, the false-positive rate on *real* sharp tiles
without synthesis (also with the runtime's decision, `fp_rate_p70`). v3 adds `groups` (the numbers for what the new synthesis added,
with the decision `P >= 0.7`: the share of blocks on a sharp thin structure over a blur, and in the far inside of a sharp featureless layer,
that are called blurred; the recall of blur from `2 b_min` to 0.012 and above; the decision's precision and recall) and `legacy`: bokeh-2's
validation (its synthesis, its half-resolution target, the same tiles, seed and samples as its final numbers), so that `final.legacy.model`
sits next to bokeh-2's `final.model` and a regression on the old recipe cannot hide behind the new one.

**Go** (the model may go to `data/models/bokeh/`) if: (1) it beats the classical detector on synthetic scenes with known blur (lower
`mae_b`, at least equal gated precision: `fp_rate_gated` not higher); (2) it does not raise false positives on sharp real photos
(`real_sharp.fp_rate_gated` at or under the baseline's, and small in absolute terms (the lead's call: under 1 % is suggested), then the
same measured in the Rust app on the CC0 Sony files and `testdata/lens-photos/`); (3) its confidence is calibrated (`ece` under 0.1). If not, the classical detector of `docs/21` section 7 ships and
the generator stays. **Read the baseline fairly:** it is a stripped-down Zhuo and Sim (edges only, Gaussian spread, no colour-guided
spread, no ball search); a blurred region with no edge left has no estimate of its own, so it has near-zero recall of strong blur by design.
The spec's detector (LC-07) adds a joint bilateral spread, the subject anchor and the ball search and will be harder to beat: the final
comparison is the network against that detector on the same scenes.

### Real photos with human masks: bokeh-2 on CUHK (local evaluation only, 2026-10-24)

`eval_real.py`, the 704 `out_of_focus` photos (the mask polarity read from the data: white = out of focus, 66.6 % of the pixels), at the photos' own
size (500 to 640 px; the spec's threshold `2 b_min` is a disc of 2.9 px there) and, on every fifth photo, enlarged to 1024 px (the network's scale: 4.9 px). The classical
column is `classical.py`'s stripped-down Zhuo and Sim, not the Rust detector.

| rule | accuracy | precision | recall | F1 | F1 per photo (mean ± sd) |
|---|---|---|---|---|---|
| native size, 704 photos | | | | | |
| network, `b >= 2 b_min` (= `P >= 0.5`) | 0.363 | 0.991 | 0.044 | 0.084 | 0.056 ± 0.160 |
| network, `c >= 0.5` gate | 0.334 | 1.000 | 0.0007 | 0.001 | 0.001 ± 0.013 |
| network, `P >= 0.7` | 0.356 | 0.995 | 0.033 | 0.064 | 0.044 ± 0.141 |
| network, `P >= 0.9` (= the lower-quantile rule) | 0.344 | 0.998 | 0.015 | 0.030 | 0.020 ± 0.095 |
| network, mirrored pass, `P >= 0.7` | 0.350 | 0.997 | 0.025 | 0.048 | 0.032 ± 0.121 |
| classical, `b >= 2 b_min` | 0.365 | 0.971 | 0.048 | 0.091 | 0.076 ± 0.126 |
| classical, `c >= 0.5` gate | 0.347 | 0.954 | 0.021 | 0.042 | 0.038 ± 0.057 |
| enlarged to 1024 px, 141 photos | | | | | |
| network, `b >= 2 b_min` | 0.387 | 0.971 | 0.054 | 0.102 | 0.080 ± 0.167 |
| network, `c >= 0.5` gate | 0.353 | 0.000 | 0.000 | 0.000 | 0.000 |
| network, `P >= 0.7` | 0.369 | 0.999 | 0.025 | 0.049 | 0.042 ± 0.106 |
| network, `P >= 0.9` | 0.358 | 1.000 | 0.007 | 0.014 | 0.013 ± 0.051 |
| network, mirrored pass, `P >= 0.7` | 0.365 | 1.000 | 0.017 | 0.034 | 0.029 ± 0.083 |
| classical, `b >= 2 b_min` | 0.354 | 0.956 | 0.001 | 0.002 | 0.002 ± 0.004 |

Read it as precision, not as segmentation. Accuracy is that of a detector that calls nothing out of focus (0.334; calling everything gives 0.666). The
network is almost never wrong when it says "blurred" (0.97 to 1.00) and says so for 1 to 5 % of the masks' out-of-focus pixels; against the classical baseline
that is equal at native size and 45 times more at 1024 px. Why so few, which the numbers cannot separate: CUHK's out-of-focus areas are mostly *mild* (their gradient
energy is a fifth of the focused areas', median) and below `2 b_min`; the photos are web JPEGs, with block artefacts on top of the blur that a raw lens plane
does not have and the synthesis does not make; and the network's own log radius is the floor (1 px) for 90 % of those pixels (AUC of `mu` as a detector: 0.64 against 0.79 for a
smoothed gradient energy). The `c` gate is the one clear failure: no threshold on the radius's confidence keeps the blur. The runtime's domain is a raw plane at 1024 px,
where bokeh-2's maps are good (the Sony planes), so CUHK is a check on precision and not a score to optimise; do not train toward it.

### The decision: what the runtime does with the two outputs (measured on bokeh-2, 2026-10-24)

`c = exp(-beta / 0.25)` says how sure the *radius* is. It is high on sharp areas (the radius is the floor) and low inside strong blur, so a
gate on it throws the blur away: on the 28 Sony lens planes `c >= 0.5` keeps **0.1 %** of the frame where `b >= 2 b_min` holds for 19.7 % (the classical
detector: 1.5 %); on CUHK it keeps 0.07 % of the out-of-focus pixels (recall 0.0007 against 0.044 without the gate); on synthetic scenes recall falls from 0.92 to 0.12.
Whether a pixel is out of focus is a different question, and the network's own Laplace answers it: `encode.detection_probability(mu, s, hd)` is
`P(b >= 2 b_min)`, a function of the two outputs and nothing else (no new output channel, the weights and the Rust parity tests stay valid).
`P >= 0.5` is the same as `b >= 2 b_min` (the median); `P >= 0.9` is the same as the lower-quantile rule `b * exp(-1.61 beta) >= 2 b_min`.

| rule (full-resolution synthetic v3 scenes, bokeh-2) | thin-structure pixels called blurred | featureless interior called blurred | precision | recall |
|---|---|---|---|---|
| `b >= 2 b_min` (= `P >= 0.5`) | 36.7 % | 65.9 % | 0.898 | 0.918 |
| `c >= 0.5` gate | 0.9 % | 0.1 % | 0.979 | 0.118 |
| `P >= 0.7` | 31.5 % | 46.7 % | 0.929 | 0.884 |
| `P >= 0.9` | 21.6 % | 7.1 % | 0.976 | 0.716 |
| mirrored average, `P >= 0.7` | 10.5 % | 38.3 % | 0.949 | 0.836 |
| mirrored average, `P >= 0.9` | 4.9 % | 4.3 % | 0.991 | 0.606 |

Recommended: **the map's "blurred" is `P >= 0.7`**, with `b = exp(mu) / hd` there and 0 elsewhere; a cell's weight in the bilateral spread is
`max(P, 1 - P)` (how sure the *decision* is: a certain blur counts as much as a certain sharp), not `c`; **"this photo has bokeh"**
(`LensPlanes::active`: 2 % of the frame) is counted with `P >= 0.9`, which turns the low-confidence blobs on a featureless sky into nothing (7 % against 47 %
of their pixels) at a recall that does not matter for a share of the frame. The mirrored pass (the photo flipped left to right, the two outputs combined
by the mean log radius and the scale `mean(beta) + |mu_1 - mu_2| / 2`) costs a second network run and is the cheapest thing that helps thin structures
(31 % to 10 % at `P >= 0.7`): worth it when 1 to 2 s more is acceptable. A guided filter of the map with the photo as the guide changes nothing (31.5 % to 31.2 %): the stage's
own bilateral upsampling already follows edges. The sharp-edge test (L1, "as built") is a real second line for thin structures (31 % to 14 %, to 4 % with the mirrored
pass) and costs recall next to sharp edges (0.88 to 0.80 on these scenes, dilation 2 px; 0.85 with none): keep it, and do not widen it.

## Into the app (`export.py`)

    python -m training.bokeh.export --model <runs>/bokeh-2/model.pt --out data/models/bokeh/bokehnet.safetensors \
        --fixture crates/ym-ml/tests/data/bokehnet_parity.json
    python -m training.bokeh.export --random --seed 1 --out <...>/bokehnet.safetensors --fixture <...>/bokehnet_parity.json

`train_run` already writes both into the run folder. The weights file is f32, batch norms folded, tensors named after the layers
(`stem`, `blocks.<i>.expand|depthwise|project`, `context`, `decoder.<j>.skip|mix|dw|pw`, `head`), and its **bytes are deterministic**
(`safetensors.torch.save_file` is not, so the file is written here: sorted tensors and metadata). The header's metadata says how to build the
network (`arch`: every block's channels, stride and residual flag, the taps and decoder widths), the upsampling and padding rules and every
constant of `encode.py`. The fixture holds a few deterministic pictures (an integer pattern the Rust test rebuilds exactly), the float32
outputs, and the encode probes. `--random` writes seeded random weights of the real architecture, with batch-norm statistics that make
folding matter, for parity work before training finishes. `export.write` refuses if the folded network differs from the trained one
by more than 2e-4.

## Budget (`budget.py`; `training/crop/budget.py`'s prices: L4 $0.000222/s, CPU core $0.0000131/s, memory $0.00000222/GiB/s, margin 1.25)

The whole programme stays at or under **$10** by the run's own estimate. Every entrypoint takes `--budget`.

1. **Refuse to start.** The worst case is every call it may make at its time limit; it starts only if that is at most `--budget` and
   the programme's spend so far plus it is at most $10.
2. **Reserve, then settle.** `ledger.json` on the volume: a reservation (the worst case) at the start, each finished call's actual cost
   added, a settlement at the end. A run that dies stays reserved in full until its deadline, then counts at what was recorded plus a
   bound on what could still be running (a training run: its worker's heartbeat file, written at every checkpoint). A failed
   run lets its reservation lapse after two minutes. The orchestrators are non-preemptible, and a restarted orchestrator with the same
   `--run-id` continues the same entry (and, for training, is charged for the time its dead attempt trained); the crop pilot's restart
   that began a new count and spent $5 twice cannot happen.
3. **Work is counted once.** Mask archives and tile chunks have item keys and done-markers on the volume; a rerun skips them and does
   not charge them again.
4. **Stop cleanly.** After every call the orchestrator saves the ledger and raises `BudgetExceeded` if the estimate passed the budget; `modal run`
   without `--detach` stops the app with the client. The training call is stopped by the orchestrator at its time limit, and the loop
   stops itself at `--minutes`.

| Entry | `--budget` default | Worst case (all calls at their limit) | Expected |
|---|---|---|---|
| `check` | 0.5 | $0.05 | about $0.01 |
| `prepare` (20,000 photos) | 1.5 | $1.26 | about $0.3 to $0.5 |
| `preview` | 0.3 | $0.12 | about $0.02 |
| `train --minutes 100` | 5 | $2.90 | about $2.6 |
| `train --minutes 5 --overhead-min 8` (v3's smoke) | 0.45 | $0.33 | about $0.2 |
| `results` / `fetch` | 0.05 / 0 | $0.008 / none | a cent / none |

`python -m training.bokeh.budget plan --entry train --minutes 100` prints a plan without touching Modal. Expected costs are guesses
(a training call of 100 minutes plus about 8 for setup and the final evaluation costs (100 + 8) * 60 * 0.00031 * 1.25 = $2.5, plus
$0.1 for the orchestrator); the ledger records the real ones. After the four entries above the programme has
spent about $3.2 and has reserved at most $4.3, leaving about $5.7 for a second run of 150 minutes (worst case $4.2) or more
data. The longest single training that fits `--budget 5` is 186 minutes.

**The overhead.** The time limit of a training call is `minutes + overhead`, 18 minutes by default: bokeh-2 needed 10 s to copy the tiles and
75 s for the final evaluation of 576 samples, so 18 is generous, and it is most of a short run's worst case (a 5-minute smoke would have a worst
case of $0.57). `--overhead-min` (never below 6) says less for a run whose final evaluation is small (`--final-val 96`); the plan, the call's time
limit and the heartbeat all use it. The programme's spend after bokeh-2 is $5.55 (ledger); v3's smoke and run reserve at most $0.33 and $2.90.

## Running it

Modal's client speaks gRPC, which the agents' proxy does not carry: runs start from GitHub Actions (`.github/workflows/bokeh-train.yml`;
a push to the branch `bokeh-train-run` runs `modal run -m training.bokeh.modal_app::$ENTRY $ARGS`; secrets `MODAL_TOKEN_ID` and
`MODAL_TOKEN_SECRET`; results come back on the temporary branch `bokeh-train-runs` and as an artifact). The workflow's default is
`ENTRY: results`, which spends about a cent, so a stray push is harmless. The order of a first run, one push each (set `ENTRY` and `ARGS`
in the workflow file on the branch):

| # | ENTRY | ARGS | Does |
|---|---|---|---|
| 1 | `check` | `--budget 0.5` | volumes and photos reachable? filters 40 photos, times them, tests the Open Images URLs; writes `check.json` |
| 2 | `prepare` | `--budget 1.5` | index, 16 mask archives, tiles, credits (resumable: run it again after a failure) |
| 3 | `preview` | `--budget 0.3` | four sheets of synthesised inputs and labels from the real tiles: look at them |
| 4a | `train` | `--budget 0.8 --minutes 8 --steps 1000 --run bokeh-smoke` | the GPU smoke: the full configuration for 1,000 steps (about 6 minutes, about $0.3), then the final evaluation, the sheet and the export. Its log (`runs/bokeh-smoke/log.txt`) must show finite losses that fall and no `skipped` lines |
| 4b | `train` | `--budget 5 --minutes 100 --run bokeh-2` | one L4, then the final evaluation and the export (`bokeh-1` holds the NaN checkpoint of the lost run: use a new name) |
| 5 | `results` | | metrics and the ledger |

**bokeh-3 (v3), the second round** (`ENTRY: train` for both; the cap leaves $4.45 and this round's worst case is $3.22 with the smoke):

| # | ARGS | Does |
|---|---|---|
| 0 (optional, a cent) | `ENTRY: preview`, `--budget 0.3` | four sheets of v3's scenes from the 29,000 real tiles (`previews.py` renders `SynthConfig.v3()` now): the lead looks before paying for a smoke |
| 6a | `--budget 0.45 --minutes 5 --steps 600 --overhead-min 8 --final-val 96 --init-from bokeh-2 --lr 0.001 --run bokeh-3-smoke` | the GPU smoke of the new synthesis and of the start from bokeh-2's weights (worst case $0.33, about $0.2): 600 steps or 5 minutes. Its log must show `started from the weights of`, finite losses and no `skipped` lines; the minutes at steps 25 and 50 give the speed of v3's steps (bokeh-2: 0.17 min per 25 steps); `metrics.json` has `final.groups` and `final.legacy` |
| 6b | `--budget 3 --minutes 100 --init-from bokeh-2 --lr 0.001 --run bokeh-3` | the run: one L4, bokeh-2's weights as the start (worst case $2.90, about $2.6) |

Starting from bokeh-2 is recommended: v3's data contain bokeh-2's (80 % of the samples have no thin structure or featureless layer), so what
it learnt carries over and the 100 minutes go to what is new; from scratch the same minutes would also have to learn the old recipe again, with
slower steps (more layers). The learning rate is half the first run's because the weights are already good (warm-up stays 300 steps).

**What the runs cost and gave (the programme's ledger, 2026-10-25).** `bokeh-1` (lost to the NaN) $2.49, `bokeh-2` (14,977 steps, 101.2 minutes) $2.51, `bokeh-3` (from bokeh-2's weights, 10,536 steps, 101.4 minutes, no step skipped; export SHA-256 `adf978dd...72fe`) $2.51, indexing, masks, tiles and previews $0.28, the GPU smoke runs $0.46: **$8.25 of the $10 cap**. `bokeh-3` is the model that ships: on the synthetic validation it beats the classical baseline (mean absolute error of `b` 0.00488 against 0.01821 on the same 48 samples), keeps what `bokeh-2` knew (`final.legacy`: 0.00348 against 0.00409), has an `ece` of 0.032 and calls 0.046 % of real sharp tiles blurred; on the 28 real Sony planes it is more specific than `bokeh-2` (`data/models/bokeh/MODEL_CARD.md`).

From a computer with direct internet and the two variables: `pip install modal==1.6.2`, then `modal run -m training.bokeh.modal_app::<entry> <args>`, and
`modal run -m training.bokeh.modal_app::fetch` copies the small files (weights, metrics, samples, previews, credits, ledger) into
`training/bokeh/runs/` (git-ignored). The volume is `yellowmoon-bokeh`; the crop volume is mounted read-only and never written.

**Locally, no Modal, no spend** (CPU, minutes; this is what produced the numbers in the lead's report):

    pip install -r training/bokeh/requirements.txt
    python -m training.bokeh.smoke --work <a folder outside the repository> --n 50 --steps 400 --batch 6
    python -m pytest -q training/bokeh/tests

`smoke.py` downloads about 50 Open Images validation photos with masks (CC BY 2.0, official mirror), builds tiles, renders
`synth-sheet-*.png`, trains on the CPU, and exports the trained and the random weights. A few hundred CPU steps teach little: it is a
plumbing check.

## Tests (`tests/`, pytest, CPU, about a minute)

Budget: refusal over a budget or the cap, persistence across restarts, work charged once, dead and failed runs, heartbeats, the
default plans fit. Synthesis: every kernel kind has its radius, determinism for a seed, and **the label is the blur that was applied**
(a step edge, a sharp subject, a lamp's ball with its energy, a blurred foreground, ramp bins). Model: shapes, size and cost, only
allowed operations, the bilinear rule, folding, export **bytes** (same weights, same file), fixture, the random mode. Sources: masks,
the sharpness filter, tiles, shards, the split, `prepare` end to end on a fake volume. Training: the loss, the schedule, the metrics, a
few real steps, resume. Encode and classical.

v3: the default config is bokeh-2's recipe (recorded sums of a batch); every kind of thin structure is drawn finite, deterministic, in the frame and
thin; a drawn line covers its width whatever its offset; a structure contrasts with its background; a featureless layer has no texture; a thin
structure over a blur is labelled sharp and weighted at least 0.9, and the far inside of a featureless layer at 0.2; skipping the FFT of unblurred layers
changes nothing; the stack mode spreads the radii; whole batches of dark-rim kernels and many seeds are finite (the bokeh-1 NaN); the half-resolution
targets (bokeh-2's rule, the sharp-wins rule, a thin line, the boost); the decision loss and probability; the group numbers and masks; the final evaluation
reports the groups and the legacy validation; a run starts from another's weights (and a resumed run does not); the overhead and the smoke's plan; the
evaluation on research-only sets (`test_eval_real.py`: the rules, the counts, the spread, the polarity).

## Known limits and risks

- **The synthetic-to-real gap** is the main risk: noise, exposure and clip are modelled, a Sony lens plane is not a JPEG. The only
  tests of it are `real_sharp` here and the Rust-side checks on real raws.
- **GPU speed is unmeasured** (no GPU here): the time-based schedule makes the run finish properly whatever it is. The estimate
  is 60 to 100 samples/s, so 100 minutes is 0.4 to 0.6 M samples.
- **Natural bokeh in the sources** is filtered, not removed; the self-cleaning round is not built.
- **Sprites are few**: about a fifth of the tiles of photos that have masks have an instance that fits (21 of 95 sample tiles), and only
  about half the photos have masks, so about a tenth of all tiles can be a donor. Each batch takes a third of its tiles from them, and
  the same objects recur. `prepare --extra` photos would help.
- **The Modal code could not be run here**: its building blocks are tested (`prepare`, the ledger, the plans, the import with
  `modal` 1.6.2), but the calls (`map(order_outputs=False, return_exceptions=True)`, `Volume.read_only`, `spawn().get(timeout)`) are used
  as the crop pipeline used theirs and as `modal` 1.6.2's signatures say. Run `check` first.
- The baseline is weak by design (above).
- **Mild real blur is missed** (CUHK, above): precision is high and recall on masks of mild defocus is a few percent. Not trained against: the label of
  a blur under `b_min` is "sharp" by construction, and natural bokeh in the sources is filtered, not removed (the self-cleaning round: run the first
  model over the unblurred sources and give the pixels where it already predicts blur no weight as "sharp" labels; the cost is one more forward pass per batch).
- **Thin structures and featureless regions are the weak points of bokeh-2** (v3's target): 31 % of the pixels of a sharp twig or hair over a blur are called
  blurred, and 47 % of a sharp sky's interior (`P >= 0.7`). v3's synthesis and the sharp-wins target are the fix; measured on bokeh-3 (`final.groups`): 28 % of the blocks on a thin
  structure and 10 % in a featureless layer's interior are still called blurred at `P >= 0.7` (37 % and 15 % by the radius alone), and the lens planes show no false blur on wires, hair or twigs; the synthetic structures are procedural (constant colour along a strand, no texture) and
  real hair and branches are not, so the real photographs (the lens planes) are the test, and thin structures can still be absorbed there.

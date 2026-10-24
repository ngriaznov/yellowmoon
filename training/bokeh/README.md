# training/bokeh: the learned bokeh map

Lens character (`docs/21-lens-character.md`) reshapes the blur a photo *already has*; it never adds blur. For that it needs,
for every pixel of the lens plane, the radius `b` of the disc blur the pixel already carries (in half-diagonal units of the
frame) and how sure that is. It is blur, not depth: a far wall at f/11 is sharp. No pretrained defocus-map or
blur-detection model is licence-clean (`research/bokeh-map-2026-10-24.md`), so this directory trains our own small network
(`model.py`, 1.5 M parameters, from scratch) on synthetic defocus with a known blur, made from licence-clean sharp photos.
Python only trains, exports and writes fixtures: the app and `ym` never run it (`AGENTS.md` rule 6); a Rust runtime
(`ym_ml`, like `crop_net.rs`) will run the exported weights. Nothing here ships yet: weights go to `data/models/bokeh/` with
a model card only if the model passes the go/no-go below.

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
and `NOTICE`. **Never used:** CUHK, DUT-DBD, RTF, SYNDOF/SYNTHIA, Hypersim, DPDD, DIV2K, ImageNet, COCO (research-only,
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
| render scenes, train, evaluate, export | `synth.py`, `kernels.py`, `train.py`, `export.py` | Modal L4 (`train_run`), or locally on the CPU for a smoke run |
| the classical baseline, the sharpness measure | `classical.py` | numpy, in `train.py`'s evaluation and `sources.py` |
| budget plans and the ledger | `budget.py` | everywhere |
| local evaluation on research-only sets | `eval_real.py` | locally only |

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

**Sample sheets** (`preview`, `smoke.py`): per row the network's input (the encoded plane), a display version of it, and the label
as colour (0 to `b = 0.05`, dark where the weight is low).

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

Loss: Laplace negative log-likelihood of `log r` at half resolution (a 2 x 2 block averages its log radius; a block that mixes radii
more than `e^0.35` apart counts 30 %). AdamW (lr 2e-3, 300 warm-up steps), an EMA of the weights (0.999) is what is evaluated and
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
(`modal_app::train --budget 0.8 --minutes 8 --steps 1000 --run bokeh-smoke`, about $0.3) before a long run: it is the full
configuration, and its log must show finite losses that fall. A run name that was used before resumes from that run's checkpoint,
so a new run takes a new name.

### Validation and the go/no-go

On held-out tiles (a hash of the photo id; never trained on), fixed seeds, pixels with label weight at least 0.6, in `b` units
(`metrics.json`, `final`): `mae_b` where the truth is at least `b_min` (and gated at `c >= 0.5`, with its `coverage`); `fp_rate` the share
of pixels with truth below `b_min` that are predicted at `2 * b_min` or more (and `fp_rate_gated`); `recall_2bmin`; calibration (ten bins
of the predicted scale against the mean absolute log error, and `ece`); the same numbers for the classical baseline (`classical.py`)
on the first 48 samples next to the model's on the same samples; and `real_sharp`, the false-positive rate on *real* sharp tiles
without synthesis.

**Go** (the model may go to `data/models/bokeh/`) if: (1) it beats the classical detector on synthetic scenes with known blur (lower
`mae_b`, at least equal gated precision: `fp_rate_gated` not higher); (2) it does not raise false positives on sharp real photos
(`real_sharp.fp_rate_gated` at or under the baseline's, and small in absolute terms (the lead's call: under 1 % is suggested), then the
same measured in the Rust app on the CC0 Sony files and `testdata/lens-photos/`); (3) its confidence is calibrated (`ece` under 0.1). If not, the classical detector of `docs/21` section 7 ships and
the generator stays. **Read the baseline fairly:** it is a stripped-down Zhuo and Sim (edges only, Gaussian spread, no colour-guided
spread, no ball search); a blurred region with no edge left has no estimate of its own, so it has near-zero recall of strong blur by design.
The spec's detector (LC-07) adds a joint bilateral spread, the subject anchor and the ball search and will be harder to beat: the final
comparison is the network against that detector on the same scenes.

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
| `results` / `fetch` | 0.05 / 0 | $0.008 / none | a cent / none |

`python -m training.bokeh.budget plan --entry train --minutes 100` prints a plan without touching Modal. Expected costs are guesses
(a training call of 100 minutes plus about 8 for setup and the final evaluation costs (100 + 8) * 60 * 0.00031 * 1.25 = $2.5, plus
$0.1 for the orchestrator); the ledger records the real ones. After the four entries above the programme has
spent about $3.2 and has reserved at most $4.3, leaving about $5.7 for a second run of 150 minutes (worst case $4.2) or more
data. The longest single training that fits `--budget 5` is 186 minutes.

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

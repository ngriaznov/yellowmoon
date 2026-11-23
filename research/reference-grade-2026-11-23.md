# Reference Grade: fit the grade at run time, don't train a predictor

**The goal fits Yellowmoon; the proposed build does not.** About half of the proposal exists already: grading happens in LogC4/AWG4 before the display transform, contrast pivots at 0.2784, there is no ACEScct to switch from, a DINOv2-S encoder runs inside the shipped Depth Anything V2-S in pure Rust, hue-vs-hue and hue-vs-sat curves exist in the parametric look, RGB curves can express lift, gamma and gain, and masks, the Auto "write numbers into Main, pin them at export" pattern and ledgered Modal pipelines all exist. The rest collides with project rules or has no clean data: `ort`/CoreML conflicts with the determinism rules, hardware 3D-texture filtering is banned by K1, ShotDeck bans AI training, Blu-ray rips cannot feed shipped weights, ARRI looks can never be in training, "REVEAL" is ARRI's name, the grid size is wrong, and a bilateral grid has no free binding. **We recommend "fit, don't predict", in three phases.** Phase 1, which we recommend now, fits the node's own corrector fields to the reference at run time, against the person's real chain (their Output look plus the cine rendering), on display-referred output. It needs no model, no training data, no Modal spend, no differentiable port, no WGSL change and no GPU risk. Phase 2 (a small learned initialiser on synthetic pairs) and Phase 3 (spatial adaptation) come only if Phase 1 misses its bar. Nothing is built yet.

The owner pasted the proposal on 2026-11-23 with the words: "See if we can reuse existing machinery/models and if we can't - we need whatever we need it to work. Asess the idea". This report is built from four research notes (codebase survey, methods literature, data and licences, fact-check of the proposal) and from checks of its own. Labels from the notes are kept: *verified* means fetched and read, *estimate* means computed or reasoned by us and not measured, *unverified* means not read from a primary source.

## The proposal in one paragraph

A node called "Reference Grade" takes a film still. DINOv2-S and Depth Anything V2-S features feed a trained head. The head writes about 30 parameters (lift/gamma/gain, a contrast pivot at LogC4 0.278, Oklab hue-vs-hue and hue-vs-sat splines) and a 16×16×8 bilateral grid of about 24 KB. The grade runs in LogC4/AWG4 before the display transform, which the proposal calls "REVEAL". Training goes through a frozen differentiable copy of that transform (a PyTorch port, or a 65³ LUT through `grid_sample`) on ShotDeck and Blu-ray frames. At run time an ONNX model runs through `ort`/CoreML or MLX, and a WGSL pass samples a 3D texture with hardware trilinear filtering. Four sliders override the result: Look Strength, Lighting Adaptation, Skin Protection and Depth Separation. The framing is that "switching the grading engine to AWG4/LogC4 simplifies the architecture compared to ACEScct".

## About half of it already exists

The codebase survey (`origin/main` at d0674e53, pipeline 0.16.0) gives this reuse map.

| Proposal component | Status | What the repository has |
|---|---|---|
| Grade in LogC4/AWG4 before the display transform | Exists | Nodes work in LogC4/AWG4 with no per-node colour space (`docs/23:36`). Trims run in linear AWG4, then C9, C9b and C11 in LogC4, then the mask blend. The Output look (C10) and the cine rendering (C12) come after the last node (`docs/02:15-31`). |
| ACEScct to LogC4 "switch" | Moot | ACEScct appears nowhere in `docs/` or `crates/`. |
| Contrast pivot at LogC4 0.278 | Exists | C9 Contrast pivots at grey, 0.2784 (`docs/02:434`; `MIDDLE_GRAY`, `LOGC4_STOP` in `crates/ym-color/src/consts.rs:237,240`). The shape is a fixed bend that holds black (0.0929), grey and code 1.0. It is not a free power about the pivot. |
| Lift/gamma/gain | Partly | No lift/gamma/gain control and no log wheels. But the four monotone curves (luma, R, G, B, 2 to 16 points each, LogC4) can encode any monotone lift, gamma or gain per channel (`curve.rs:33,36`). |
| Hue-vs-hue and hue-vs-sat | Partly | The parametric look has `hue_vs_hue`, `hue_vs_sat` and `luma_vs_sat` (`crates/ym-color/src/look.rs`, `docs/08:40-71`), baked on the CPU to a 65³ cube. It is a Look in the Output slot, not a node field, and its hue axis is the LogC4 chroma-plane angle, not Oklab. A node's closest match is the colour warp: up to 16 Gaussian pins in the chroma plane (`warp.rs`). |
| Masks for local grading | Exists | Object (SAM 2.1 small), Subject and People (Apple Vision), Background, Depth, shapes and key ranges, with Add, Subtract, Invert and combine (`docs/23:456-500`). There is no Face mask kind. |
| Predictor writes numbers, export pins them | Exists | Auto resolves to numbers in Main; `ym:Settings` pins them at export; albums store settings (`docs/19`, `docs/22`, `docs/31`). Auto writes only Input and Main today. |
| DINOv2-S ViT/14 encoder | Partly | It runs inside the shipped Depth Anything V2-S, pure Rust, f32, deterministic (`crates/ym-ml/src/depth/encoder.rs`, taps at blocks 2, 5, 8, 11). These are the Depth-fine-tuned weights, not Meta's raw checkpoint. No public feature accessor exists, and no standalone DINOv2 or CLIP model is in `crates/`. |
| Display transform copy for training | Partly | The cine rendering is analytic, tableless and C1-smooth, with a numpy f64 reference (`training/cine/cine_ref.py`) and a 1,919-vector fixture. A PyTorch port is mechanical but not written. |
| Bilateral grid | Missing | `bilateral.wgsl` is a denoise filter. Nothing slices a coarse grid. |
| Reference image input | Missing | No code path takes a second user image as a setting. |
| Modal training | Partly | Ledgered pipelines exist (`training/crop`, `training/bokeh`). |
| Blind A/B and ΔE tooling | Exists | `tools/eval/ab`, the golden set, `crates/ym-color/src/deltae.rs` (CIEDE2000). There is no look-match metric. |

Two facts from the survey shape everything below. First, the Output look (C10) sits between the node and the display transform. The proposal's order, "grade then DRT", leaves it out. The default is the Signature look, and a person can import an ARRI look. An ARRI LogC3-to-Rec.709 look even replaces C12 (`docs/02:589`). A grade matched to a reference must therefore be judged on the person's real chain, not on a model of it. Second, the Auto providers receive exactly one picture and answer in a fixed table of steps ("set by eye, never fitted to a provider's answers", `docs/22:855`). They cannot carry a reference.

## Where the proposal's claims fail

The fact-check read each claim against the repository and outside sources.

| Claim | Verdict | Correction |
|---|---|---|
| LogC4 grey 0.278, LogC3 0.391, ACEScct 0.4135 | Correct | Computed: 0.2783958, 0.391007, 0.41359. The pivot is already C9's `g = 0.2784`. |
| AWG4 to Oklab needs no Bradford | Correct, with caveats | AWG4 is D65. But its blue primary has y = -0.0308, so many in-gamut colours give negative LMS and the cube root must be signed. Oklab was fitted on display-referred data, not scene values far above 1. |
| Oklab hue splines beat log RGB | Partly | Oklab's hue prediction is good (the author's own tables: H RMS 0.49 against 0.59 for CAM16-UCS). But "better than log RGB" is almost empty, because log RGB has no hue axis. The real baseline is the existing chroma plane. The display transform then applies its own order-3 hue harmonics (`docs/02:654-668`), so a hue spline drawn before it does not land where it was drawn. |
| Grid of about 24 KB | Wrong | 16×16×8 is 2,048 cells. With 12 coefficients per cell (a 3×4 affine) that is 96 KiB in f32 and 48 KiB in f16. 24 KiB is the offset-only grid (3 values per cell, f32). |
| 3D texture with hardware trilinear "eliminates manual interpolation" | Wrong for this project | The hardware weight is fixed point (4 to 8 bits, vendor-specific; lavapipe reports 8). Filtering rgba32float needs the optional `float32-filterable` feature, and K6 requires no features beyond compute. It trades exact maths for a hardware approximation. |
| "REVEAL DRT" | Wrong (name) | Our transform is "the cine rendering" (C12), analytic, in `ym_color::display::cine`. REVEAL is ARRI's name for the ALEXA 35 colour science, which `docs/00:61` lists as out of scope. |
| A 65³ LUT is a fine differentiable copy | Lossy | Computed on the neutral axis only: max abs error 3.4e-4 in display linear, max ΔL* 0.21. The full-volume error near the gamut edge and the +10.4-stop kink was not run and is an estimate. The transform is analytic and ports to PyTorch directly. |
| 384 px input to ViT-S/14 | Invalid | 384 is not a multiple of 14. Valid sizes are 378, 392, 518. Encoder cost, computed: about 20 GMAC at 378², 22 at 392², 41 at 518². |
| DINOv2-S as a second backbone | Duplicates | The shipped depth encoder already holds one (about 100 GMAC per photo with the DPT head at 518×784; the 2 to 2.5 s on four threads is an estimate, Mac not measured). |
| Switching to AWG4/LogC4 "simplifies" the architecture | Moot | The pipeline is already linear AWG4 and LogC4. |

## What conflicts with the project rules

| Proposal item | Rule | Why |
|---|---|---|
| `ort`/CoreML at run time | AGENTS.md rules 6 and 8; `docs/07` convention | `ort` is not banned: `deny.toml` bans only `dng`, `rawler`, `rawloader`, `ljpeg`, `jpegxl-rs`, and `docs/14:53` allows it. But every shipped model runs in our own pure-Rust CPU runtime so that the app and `ym` give the same bits. CoreML's default compute units include the Neural Engine, which is fp16-only. Apple's own forum threads (not independent measurements by us) report CPU, GPU and ANE disagreeing. The ONNX Runtime CPU provider documents no bit-exactness across instruction sets. This conflict is our reading of the rules, not a written ban. |
| MLX | Rules 6 and 8; Linux CI | Metal-only, no official Rust binding, no determinism claim. |
| Hardware 3D-texture trilinear | K1 (`docs/17:65`: "no ... texture sampling"); parity gate 2e-4 (`docs/02:43`); `docs/13:111` | A lint fails any kernel containing `textureSample` (`crates/ym-gpu/tests/shaders_naga.rs:291`). An rgba16float LUT texture rounds by up to 2.4e-4, over the gate. |
| A new per-pixel stage inside `run_nodes` | K3; the llvmpipe rule (`docs/STATUS.md:530-531`, `docs/17:42-43`) | The colour kernel already binds all 8 storage buffers. The 2026-11-20 and 2026-11-21 llvmpipe miscompiles followed small control-flow changes in `run_nodes` and `node_mask`. The rule: change that code as little as possible, prefer host-side packing, and pass `diag-alf` before a release. |
| Oklab inside the kernel | `docs/32:186` | Rejected for Colour Uniform: "needs a transfer function and a matrix per pixel and gives no better hue on scene values than the plane the product already has". This is a decision about scene values in the kernel, not a general ban on Oklab. |
| ShotDeck frames for training | `docs/14` rule 5; ShotDeck's help page | The help page bans "training AI models with them". The full Terms of Use returned HTTP 403 to our fetch and were not read. |
| Blu-ray rips for shipped weights | `docs/14` rule 5 (project policy) | The law is unsettled (see the data table). We treat it as unusable. |
| ARRI looks in training | AGENTS.md rule 2; `docs/14` rule 9 | Rule 2 keeps ARRI files out of the repository and the bundle. With rule 9 (a tuning program and its fit inputs also stay out), a cloud trainer cannot see them. This is derived from the two rules, not stated as one sentence. |
| The name "REVEAL" | AGENTS.md rule 3; `docs/14` rule 6 | The product and UI never use ARRI's names. A factual compatibility line is fine. |
| A reference image kept per photo | AGENTS.md rule 4 | Nothing is saved per photo. Fitted values are ordinary settings, so they may be saved. The reference image is session memory only. |

## Real film data cannot feed shipped weights

The data and licence note gave each source a verdict. OK to ship means it may feed shipped Apache-2.0 weights. Eval-only means local evaluation, never committed, never trained into shipped weights.

| Source or model | Verdict | Basis |
|---|---|---|
| ShotDeck | Not usable | Help page bans AI training; Terms of Use not read (403). |
| Blu-ray rips | Not usable for shipped weights | DMCA s.1201 circumvention is separate from fair use, and the 2024 exemption covers only scholarly text and data mining. Fair-use rulings point both ways: Bartz v. Anthropic and Kadrey v. Meta (2025) found fair use on their records, while the Third Circuit in Thomson Reuters v. Ross (29 Sep 2026) found training a non-generative competing tool not fair use. We read the Ross ruling through law-firm and news summaries, not the opinion. Not legal advice. |
| MIT-Adobe FiveK, PPR10K, RAISE | Eval-only | Research-only or non-commercial terms. |
| HDR+ burst | Eval-only | CC BY-SA with a "scientific purposes" intention; whether weights are an adaptation under ShareAlike is untested. |
| Unsplash Lite | Eval-only | Training allowed "for internal business purposes" only; marketing text and legal text disagree. |
| Pexels, Pixabay | Eval-only | Licences are silent on machine learning, so nothing is granted. |
| ARRI sample footage, ASC StEM2 | Eval-only | No licence for training found; StEM2 terms not read. |
| ARRI Look Library | Not usable | All rights reserved (rule 2). |
| Wikimedia Commons CC0, public domain and CC BY files | OK to ship (unverified) | The page was not fetched. Filter by the licence field and keep an attribution list. Film stills are mostly non-free. |
| Prelinger items marked public domain | OK to ship | Low value: faded standard-definition ephemera. |
| Blender open movies (Tears of Steel) | OK to ship, with attribution | CC BY 3.0 (film) and CC BY (source footage). Actors keep portrait rights, so flag it in a model card. Few distinct looks. |
| Open Images (CC BY) | OK to ship | Already used for the crop model (`training/crop`). |
| DINOv2 ViT-S/14 | OK to ship | Apache-2.0 now; it was CC BY-NC-4.0 in the April 2023 preview and relicensed in August 2023. The repository also holds non-commercial models, so pick weights explicitly. Not recorded in `docs/14` yet. |
| DINOv3 | Not usable | Custom licence with field-of-use limits and gated weights. |
| Depth Anything V2 Small | OK to ship | Apache-2.0 (already bundled). Base, Large, Giant are CC BY-NC-4.0. |
| CLIP, SigLIP 2 | OK to ship (licence); too heavy | CLIP ViT-B/32 is about 150 M parameters. |

Real film grades therefore reach us only as run-time references that the person supplies, and as local evaluation. Both are fine under the owner's rule of 2026-10-24: research-only data may be used for local evaluation, never committed (`docs/STATUS.md:463`).

## DINOv2 features are the wrong encoder for a look

A look is mostly colour. DINO-family models are trained to be invariant to colour changes. We checked this ourselves.

- **DINO** (arXiv 2104.14294, section 3.2, implementation details, *verified*): "We follow the data augmentations of BYOL (color jittering, Gaussian blur and solarization)". The text we read does not mention grayscale.
- **DINOv2** (arXiv 2304.07193, *verified* by reading the PDF text): the paper's text, as extracted, lists no augmentations. It says it combines the DINO and iBOT objectives. So the paper alone does not confirm the claim.
- **DINOv2 code** (`dinov2/data/augmentations.py`, *verified*): `ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)` with probability 0.8, `RandomGrayscale(p=0.2)`, and solarization on the second global crop.

In short: the DINO paper names colour jittering, and the DINOv2 training code applies colour jitter and random grayscale to every crop. Training for that invariance should make a frozen embedding a poor measure of grade. That the features are *poor* for a look is our reasoning and *unverified* by any experiment. Colour statistics of the two pictures carry the look directly, and a statistics-based fit needs no encoder at all.

The depth encoder is a further trap. It holds Depth-fine-tuned weights and has no public accessor, so reusing "the DINOv2 features" is a code change plus a second 100-GMAC pass for the reference.

## Methods: assemble known pieces, avoid the non-commercial ones

The methods survey (tags there: verified from a fetched source, or from memory) finds no published method that fits existing grading controls to a reference under a fixed display transform for raw or log workflows. The closest are Exposure (Hu et al., 2018), CSRNet (He et al., 2020), Deep Preset (Ho and Zhou, 2021) and Neural Photo-Finishing (Tseng et al., 2022). So nobody has published a quality bar for this exact shape, and we must build our own evaluation.

| Method | What it gives | Licence or ship status |
|---|---|---|
| Reinhard et al. 2001 | Mean and standard deviation match in lαβ (6 numbers). A good initialiser; fails when content differs. | Published algorithm, no code to copy. |
| MKL (Pitié and Kokaram) | Closed-form 3×3 matrix plus offset. Cheap, smooth, can leave gamut. | Published algorithm. The year (2007) is from memory. |
| IDT, sliced Wasserstein (Pitié 2005/2007; Bonneel 2015) | Non-parametric maps; grain and banding artefacts; needs seeded projections. | Published algorithms. Use only as a distance term or a comparison map. |
| Exposure (Hu 2018) | White-box, readable filters trained unpaired. Lower PSNR than black-box (18.59 dB, 8.56 M parameters). | Idea only. |
| CSRNet (He 2020) | 36,489 parameters, global modulated colour map. | Idea only. |
| HDRNet (Gharbi 2017) | 16×16×8 grid of 3×4 affine transforms; single-image, not reference-conditioned. | Apache-2.0, archived 2024. |
| Image-adaptive 3D LUT, AdaInt | Blended basis LUTs; not reference-conditioned. | Apache-2.0. |
| Deep Preset (2021) | Predicts a 69-value Lightroom preset from input and reference. | Non-commercial, research only. Not usable. |
| Neural Preset (2023) | 5.15 M parameters, self-supervised, 16×16 colour matrix per image. | CC BY-NC-SA 4.0. Not usable; the idea is free. |
| SA-LUT (2025), video grading by diffusion (2025), Deep Analog (2026, an unreviewed thesis), CanonCGT (2026) | Reference-conditioned LUT prediction. | Licences not stated on the pages seen. Do not depend on them. |
| Photoreal style transfer (WCT2, PhotoWCT) | Pixel output, float-convolution nondeterminism. | Rejected: output is not a map and breaks the parity rule. |

Two points from this survey matter most. The field has converged on "predict a small colour-mapping object, apply it deterministically", which is the form our node already is. And self-supervised training on synthetic grades from our own control space is cheap, because we can synthesise unlimited pairs without any studio data.

## Recommendation: fit, don't predict

### Phase 1 (recommended now): a training-free, deterministic fit at run time

The new node's values are fitted to the reference by optimising the node's existing corrector fields. The target is the person's real chain: their Output look (C10, which may be an imported look; that is fine, because it never leaves their Mac) followed by the cine rendering. The match is measured on display-referred output. So there is no model, no training data, no Modal spend, no differentiable port, no WGSL change (our inference from the corrector's design; confirm in the task) and no GPU risk. The render path is the corrector's. The fit renders the photo from the node's input through any nodes after it and the Output, on the CPU reference path (as Auto's analysis picture is rendered), so its input is the same bits in the app and in `ym`.

**Parameter set.** About 30 values mapped to existing fields. This is our proposal, not a spec.

| Proposal concept | Existing field | Values |
|---|---|---|
| Tone (contrast about grey, toe, shoulder) | Luma curve, about 8 fixed knots at scene stops around grey | 8 |
| Per-channel lift, gamma, gain | R, G and B curves, 3 knots each | 9 |
| Global saturation | Saturation | 1 |
| Hue-vs-hue and hue-vs-sat | Warp pins: 6 fixed source hues at a fixed chroma, each with a hue shift and a saturation | 12 |

The warp acts radially and in Gaussian blobs (ranges 0.01 to 0.25, neutral guard 0.02), so it imitates hue curves but is not a pure function of hue angle. The alternative is a new per-node hue-curve field, which needs a kernel change in `run_nodes` and so runs into the llvmpipe rule. That is a decision for the owner.

**Objective.** Computed on downscaled analysis pictures of the reference and of the photo after the real chain, in display-referred CIELAB or Oklab. Oklab is fine here, because display values are bounded; the `docs/32` objection was about scene values in the kernel. Terms:

- L* quantiles: match the shape of the tone curve, and match its level only weakly;
- a* and b* means per lightness band;
- chroma per hue sector;
- optionally sliced Wasserstein, with a seeded fixed projection set;
- a regulariser toward identity;
- a gamut-clip penalty;
- a skin guard on MediaPipe skin pixels. MediaPipe is deterministic and already bundled (`ym_ml::mediapipe`, used by the Face Refinement node, behind the `mediapipe` feature of `ym-retouch`). Which of its outputs gives a skin class should be confirmed in the task. Do not use Apple Vision here: it is not reproducible across Macs.

**Optimiser.** Closed-form initialisation (quantile tone match, band means), then a seeded, fixed-iteration, derivative-free refinement. The chain is the real one run forward, so nothing needs to be differentiable. It runs in CPU f64 in a fixed order, so the app and `ym` get the same bits. The GPU render that follows is held to the existing parity tolerances (D8), as for every other node.

**Run time.** Unmeasured. The target is 1 to 2 s on an M1, using about 8 to 16 k stratified pixels. The methods note's "under 1 s on a 128-pixel thumbnail" is an unmeasured figure from the literature, not a result.

**Persistence.** The reference image is session-only (rule 4). Fitted values are ordinary settings: pinned into `ym:Settings` at export, saved in albums and presets as values.

**Interface.**

- A reference picker that accepts JPEG, PNG and TIFF, or another photo in the filmstrip. Photo-to-photo matching across a series is a strong use. The codebase has the `tiff` and `png` crates in the workspace; the decode path for a reference is a task to confirm.
- Strength, which scales the fitted deltas instantly.
- Skin protection, which re-fits.
- Fit again.
- Light, Colour and Curve tabs to override the result, as Colour Uniform has since CG-01 (`docs/32` section 12a: Compress, Light, Colour, Curve, Mask).
- Masks, as on any node.

**CLI.** `ym ... --reference <file>`.

**Reference handling.** Letterbox bars are removed by a simple deterministic rule. Untagged references are assumed sRGB, and ICC and colour tags are handled for sRGB, Display P3 and Rec.709. No HEIC: the libheif licence is LGPL.

**Changing the Output look invalidates the fit.** Show a hint, or re-fit while the reference is still in memory.

**Effort.** About the size of Colour Uniform: a core fit module, a node kind whose render is the corrector record, UI, CLI, schema, docs and tests. It needs no WGSL change.

### Phase 2 (only if Phase 1 fails its quality bar): a small learned initialiser

- Train on **synthetic** pairs: random grades from the node's own parameter family, applied through our own chain with the Signature and Neutral looks and our own synthetic `.cube` files, to licence-clean photos (CC BY Open Images, already used for crop).
- Inputs are colour statistics, not DINOv2.
- It runs as a tiny MLP in our pure-Rust runtime, with a parity test against the Python reference. The weights are committed (kilobytes to megabytes).
- Its prediction feeds the Phase 1 optimiser, so the fit stays the arbiter.
- Cost, by the data note's estimate (not measured): a parametric predictor on synthetic pairs is about $2 to $10 including reruns, on Modal rates read on 2026-11-23. It needs the owner's yes and a cap. By the ledgers of record about $2 to $3 of the $30 remains (an inference from `docs/STATUS.md` figures, and the account may have been topped up since), so the owner should confirm the balance first.

### Phase 3 (only if needed): spatial or local adaptation

- Either region-aware fits (Depth near and far, faces) that emit masked sub-nodes, or a grid pass built like Film: made once per photo frame, in its own kernel outside `run_nodes`, with manual trilinear interpolation on an f32 storage buffer, a CPU reference and a parity test.
- Costs: the K3 binding problem, the `diag-alf` release gate, a new `PIPELINE_VERSION`, and a large effort. Nodes created by a predictor are new behaviour, because Auto writes only Main and Input today.

### What we would not do

- Train on ShotDeck or Blu-ray frames.
- Use `ort`/CoreML or MLX at run time.
- Use hardware texture sampling.
- Use Oklab inside the kernel.
- Add a second DINOv2 backbone.
- Use the name "REVEAL".

## Precedents: these tools exist

- **Photoshop, Image > Adjustments > Match Color.** Adobe's help describes matching the colours of one image, layer or selection to another, in RGB mode. We confirmed this through Adobe's help pages as returned by a search; our direct fetch of the older help URL returned HTTP 403, so we did not read the page text ourselves.
- **DaVinci Resolve, Shot Match.** On the Color page, right-click the reference clip and choose Shot Match to This Clip. A third-party site that quotes Blackmagic describes it as matching colour, contrast and brightness of a clip to a selected clip; a Larry Jordan tutorial for Resolve 20 covers it too. We did not read Blackmagic's own manual. Resolve's separate "Color Match" is chart-based and is a different feature.

Both match one image to another without a trained model, which supports the product shape. Neither gives us a quality bar; we must set our own.

## Evaluation plan

- **E1, synthetic recovery (CI-safe).** Grade a CC0 or CC BY photo with a known random grade from the family, fit, and measure CIEDE2000 to the truth (`crates/ym-color/src/deltae.rs` exists).
- **E2, cross-content transfer, the honest test.**
  1. Photo A graded with G is the reference.
  2. Photo B graded with G is the ground truth.
  3. Fit B to the reference, then measure ΔE to the ground truth.
  4. Compare against three baselines: no change, Reinhard (as published, in lαβ; a Lab variant is common), and MKL. These run as pixel maps, for comparison only.
- **E3, blind A/B [HUMAN].** Run with `tools/eval/ab` on references the owner picks and keeps local. Copyrighted stills are local evaluation only, never committed or cached by CI.
- **Data limits.** Public photos only for choosing constants. The owner's photos are at most a held-out check, per the owner's rule "I won't let you tune specifically for my photos" (`docs/32:102`).
- **Thresholds** are the owner's to set. The existing A/B gate (win share at least 60 %, Wilson lower bound above 50 %) is provisional and can serve as a starting point.

## Risks

- **Content mismatch.** A night reference could darken a beach. Tone level is weighted weakly, and Strength scales the result.
- **Dependence on the look.** A fit is valid for one Output look. Changing it invalidates the fit.
- **Unknown colour encoding of the reference.** Untagged files are assumed sRGB; a wrong guess shifts the whole fit.
- **Fit time on a Mac.** Unmeasured. The Mac-only checks go to "Waiting on the owner" as usual (M0-H4).
- **Skin.** The guard limits hue and chroma moves on skin pixels. Thresholds for it come from public data.
- **Strongly local looks.** Vignettes and graded skies cannot be copied by a global grade. That is Phase 3.
- **Warp pins as hue curves.** They approximate hue-vs-hue. The E1 test shows how close.

## Decisions for the owner

1. **Approve Phase 1** (fit at run time, no training) as the approach.
2. **Reference sources:** image files, other photos in the filmstrip, or both.
3. **Hue moves:** warp pins (no kernel change), or a new hue-curve field (kernel change, llvmpipe risk).
4. **Phase 2 budget:** only later, and only if Phase 1 misses the bar. Confirm the Modal balance first.
5. **The name** stays "Reference Grade"; never "REVEAL".

**Answered 2026-11-23: parked.** Nothing is planned or built. The owner's answers for when it is picked up: reference sources are both image files and photos in the strip; hue moves use a new hue-curve field on the node instead of warp pins, so the change touches the colour kernel's node code and must pass `diag-alf` before a release.

## Sources

Repository (at `origin/main`, d0674e53): `AGENTS.md`; `docs/00`, `02`, `07`, `08`, `13`, `14`, `17`, `19`, `22`, `23`, `32`, `STATUS.md`; `crates/ym-color/src/{grade,warp,curve,look,deltae}.rs`; `crates/ym-ml/src/depth/`; `crates/ym-gpu/tests/shaders_naga.rs`; `training/cine/`; `tools/eval/ab/`.

Methods:
- Reinhard et al., Color Transfer between Images: https://users.cs.northwestern.edu/~bgooch/PDFs/ColorTransfer.pdf
- Pitié and Kokaram, colour distribution transfer: https://mural.maynoothuniversity.ie/15125/1/RD_automated.pdf
- Bonneel et al., sliced Wasserstein barycenters: https://hal.archives-ouvertes.fr/hal-00881872
- Colour transfer survey: https://arxiv.org/abs/2204.13339
- HDRNet: https://arxiv.org/abs/1707.02880
- Deep Preset: https://arxiv.org/abs/2007.10701
- Neural Preset: https://arxiv.org/abs/2303.13511
- Image-adaptive 3D LUT: https://arxiv.org/abs/2009.14468 ; AdaInt: https://arxiv.org/abs/2204.13983
- SA-LUT: https://arxiv.org/abs/2506.13465 ; video grading: https://arxiv.org/abs/2508.00548
- Exposure: https://arxiv.org/abs/1709.09602 ; CSRNet: https://arxiv.org/abs/2009.10390
- Neural Photo-Finishing: https://light.princeton.edu/publication/neural-photo-finishing/

Encoders and augmentations:
- DINO: https://arxiv.org/abs/2104.14294 (section 3.2, read in the HTML version)
- DINOv2: https://arxiv.org/abs/2304.07193 (PDF text read)
- DINOv2 augmentations code: https://raw.githubusercontent.com/facebookresearch/dinov2/main/dinov2/data/augmentations.py
- DINOv2 licence: https://github.com/facebookresearch/dinov2 ; DINOv3 licence: https://raw.githubusercontent.com/facebookresearch/dinov3/main/LICENSE.md
- Depth Anything V2: https://github.com/DepthAnything/Depth-Anything-V2

Data and law:
- ShotDeck help: https://help.shotdeck.com/content/how-can-i-use-the-images-on-shotdeck/
- DMCA s.1201 2024 recommendation: https://www.copyright.gov/1201/2024/2024_Section_1201_Registers_Recommendation.pdf
- Fair-use summaries: https://www.skadden.com/insights/publications/2025/07/fair-use-and-ai-training ; https://www.debevoise.com/insights/publications/2025/06/anthropic-and-meta-decisions-on-fair-use ; https://www.akingump.com/en/insights/ai-law-and-regulation-tracker/third-circuit-finds-ai-training-use-of-westlaw-headnotes-is-not-fair-use-thomson-reuters-enterprise-centre-gmbh-v-ross-intelligence-inc
- FiveK: https://data.csail.mit.edu/graphics/fivek/ ; PPR10K: https://github.com/csjliang/PPR10K ; HDR+: https://hdrplusdata.org/dataset.html ; Unsplash terms: https://raw.githubusercontent.com/unsplash/datasets/master/TERMS.md
- Tears of Steel: http://mango.blender.org/production/4-tb-original-4k-footage-available-as-cc-by/

Runtime and GPU:
- `ort`: https://github.com/pykeio/ort ; CoreML provider: https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html
- Apple forum threads: https://developer.apple.com/forums/thread/689135 ; https://developer.apple.com/forums/thread/672384
- MLX: https://github.com/ml-explore/mlx
- Oklab: https://bottosson.github.io/posts/oklab/
- WebGPU features: https://developer.mozilla.org/en-US/docs/Web/API/GPUSupportedFeatures
- Texture filtering precision: https://reedbeta.com/blog/texture-gathers-and-coordinate-precision
- Modal pricing: https://modal.com/pricing
- PyTorch `grid_sample`: https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.grid_sample.html

Precedents:
- Photoshop Match Color: https://helpx.adobe.com/photoshop/desktop/adjust-color/selective-color-adjustments/match-color-in-different-images.html
- Resolve Shot Match: https://davinciresolveclub.com/davinci-resolve-shot-matching/ ; https://larryjordan.com/articles/how-to-color-match-clips-in-davinci-resolve-20/

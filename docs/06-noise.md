# 06 · Noise estimation for every camera

Crate: `ym-noise`. Background and sources: `research/notes/noise-estimation.md`.

This document describes the per-photo noise model: the model itself, how it is estimated from the photo's own pixels, the priors it starts from, how photos of one session pool their estimates, and how the estimator is validated. The model gives the denoise stages a per-pixel noise σ (the bilateral denoise, R7, R8, and the thresholds of the single-frame denoise, `docs/algorithms/05-single-frame-denoise.md`) and will give the same σ to the denoise network (`docs/07-ml-models.md`) once one is trained.

V1 has no calibration shoots (no dark or flat frames for any body), so every file gets its noise model estimated from its own pixels, with priors built from public files. Nothing is stored between sessions.

## Model

Per image, in normalised units (after R1, before white balance), for CFA plane `c`:

```
var(μ) = a · (μ − δ_c) + b + σ_row² + q²/12
```

- `a`: shot-noise gain, **shared** by the four CFA planes (the gain in DN does not depend on colour; per-plane differences in published profiles are white-balance scaling).
- `δ_c`: small per-plane black-level offsets (the "colour bias" of the ELD model, the largest single gain in its ablation).
- `b`: read-noise variance.
- `σ_row`: row-noise σ (usually small on Sony).
- `q`: the quantisation step (from the decoder's `qstep` for cRAW and ARW 6.0; 1 DN otherwise, normalised).

σ map for the network: `σ(x) = sqrt(a · x̂ + b + σ_row² + q(x)²/12)`, with `x̂` a 5×5 box-smoothed version of the signal (never the noisy value), in the same balanced units as the network input (multiply by `g0` per channel).

## Estimator

Written from the published papers; no reference code is used. The input is a `PlaneSource`: a `U16Mosaic` with the black and white levels of R1, or an `F32Plane` for an already-normalised mosaic. `ym-noise` does not depend on `ym-raw`.

1. **Planes.** Split the normalised mosaic into four half-resolution planes; the plane index is the position in the 2×2 cell, `p = 2·(y & 1) + (x & 1)`, so the estimator never needs to know which colour a position carries. Work on a 2 MP crop grid (16 tiles spread across the frame) for speed.
2. **Smooth regions.** Per plane, take windows of 6×6 plane pixels (3×3 Haar blocks). The measurement is the diagonal Haar coefficient `dd`, which cancels row offsets and linear ramps; the window's energy is the detrended horizontal and vertical detail plus the mean `dd²` of the neighbouring windows. Per intensity bin keep the 40% of windows with the lowest energy (70% in clipped bins, where ranking hard on energy would pick the most clipped windows), after Foi 2008. A second pass normalises energy and coefficients by the first fit, which removes the bin-width and selection bias; further passes change `a` by less than 0.1%.
3. **Bins.** 24 bin edges from 0.002 to 0.98, log-spaced below 0.05. Per bin: the robust variance of the `dd` coefficients by MAD, `(1.4826·MAD)²`, the bin mean, and the count. A bin with fewer than 400 samples is dropped; a (bin, plane) pair needs 100 samples to give a point of its own.
4. **Clip-aware fit.** For bins near white (mean > 0.85), the (mean, variance) of the clipped Gaussian is inverted to the unclipped `(μ, σ²)` (Foi 2008 §4) so clipping does not bias `a` downward: `h(u) = uΦ(u) + φ(u)`, `g(u) = (u²+1)Φ(u) + uφ(u) − h²`; the ratio `h/√g` is monotone in `u`.
5. **MAP fit** of `(a, b, δ_c)` to the bin statistics: weighted least squares in variance with a one-sided robust loss (a Cauchy-type weight above the line and Huber below it, then Tukey's bisquare; texture can only add variance), plus the priors below as Gaussian penalties in `log a` and `b`. `δ_c` has a ridge prior (σ = 3·10⁻⁴); it is not identifiable from smooth regions alone and stays near zero unless the data pull it. `b` is weakly identified when `a·μ` dominates every surviving bin; the σ map stays right because the combination that matters is what the bins constrain.
   - The soft wall of the log-uniform prior on `b` is a reweighted quadratic in `b − bound` (the secant of the log residual through the bound), which keeps the iteration from stepping to a negative `b`.
   - The mean intercept is kept above `−a·0.002`: the model can carry a negative intercept only as a common plane offset, clamped at 2·10⁻³ of white (32 DN at 14 bit). When a prior pins `a` above what the data allow, `a` gives way, so the model's variance at the image's own levels stays right.
6. **Texture fallback.** A Chen 2015-style PCA estimate per bin on 7×7 patches (smallest eigenvalue over a confidence test: Marchenko–Pastur edge correction, Wilson–Hilferty threshold), after a variance-stabilising transform with the prior `a`, followed by a second fit. It runs when (a) fewer than 6 bins survive (busy scenes), (b) 15% or more of the points sit well above the fit, or (c) a cheap three-bin PCA probe finds the noise below 0.95 of the smooth-region fit (texture the other tests cannot see; clean images give 1.05–1.1). Probe results below 0.5 of the fit are not trusted: spatially correlated noise looks nearly noise-free to PCA and is the business of step 9.
7. **Refinement (YOND-style).** `estimate::refine(prev, noisy, denoised, prior, cfg)` re-estimates per bin from `var(noisy − denoised) − var(denoised − smoothed denoised)`, with the retained-noise correction capped at 15% of the residual variance, and refits once. It returns a new `NoiseEstimate`; row noise and the filtered-file factor are carried over from `prev`. The renders use the estimate of steps 1 to 6; refinement is exercised by the synthetic validation only.
8. **Row noise.** In smooth regions, the variance of the row offset is measured from cross-tile cross-products of second differences across rows (tile rows 0 and 2 only, for speed), so the pixel-noise part never has to be subtracted. Clamped at 0. A row σ below about half the read-noise σ is not reliably resolved and often reads as 0.
9. **Filtered-file check.** `R = Σ binned residual variance / Σ pixel residual variance`, over the cleanest quarter of the dark-half smooth windows after a plane fit, comparing 2×2-binned residuals with per-pixel ones; white noise gives 0.25. A ratio above 0.4 means the camera filtered the raw (Sony's "star eater" on long exposures, or other in-camera processing). Then σ is raised by `sqrt(pixel residual variance / model pixel variance)`, clamped to [1.3, 4]: over-estimating is the safer error. The check is skipped when the texture fallback ran or the fit has no data (texture raises `R` the same way correlated noise does); the result then says `evaluated = false`.

## Priors

| Camera | Prior on `a` | Prior on `b` |
|---|---|---|
| Bodies with a public-data prior (M2-01) | fitted `log a = α·log ISO + β` per gain regime, σ = 0.25 in log | fitted per regime, σ = 40% |
| Any other body | `a ∝ ISO` around a body-class median (full-frame 24–61 MP, APS-C, 1-inch), σ = 1 stop in log | wide: log-uniform over two decades |

The class prior is derived from physics: full-well capacity 2000 e⁻/µm² times an assumed pixel area per body class, `a = 1/FWC` at ISO 100 and `∝ ISO` above (σ one stop in `ln a`); `b` is log-uniform between `(a·1 e⁻)²` and `(a·10 e⁻)²`. The assumptions are public constants in `ym_noise::priors::ClassAssumptions`. `BodyClass::from_model` knows the NEX, ILCE-3/5xxx, A-mount and fixed-lens bodies as well as the full-frame ones.

**Gain regimes.** Sony bodies switch conversion gain at one ISO; read noise drops there. The table, from public measurements: A1 500, A1 II 500, A7 III 640, A7 IV ~400, A7 V 1000, A7R IV 320, A7R V 320, A7S III 1600, A7C 640, A7C II 400, A9 500, A9 II 640, A9 III 800 (base 250). Unknown bodies: no break, wide prior. Data overrides the table. `data/cameras/noise/gain-regimes.json` carries it; every row's `source` says "docs/06 table, to be confirmed on samples".

**ISO quirks.** Extended low ISOs (50–80) use base gain; Hi ISOs above about 16000 (A1, A7R V, A9 III), 64000 (A7 III, A7 IV) or 128000 (A7S III) look processed: widen the prior and rely on the data. Use the maker-note ISO, since EXIF ISO saturates at 65535.

### Batch pooling

Within a session, files from the same model, ISO, bit depth and compression share a pooled estimate (weighted by bin counts), updated as files finish decoding. Nothing is written to disk. Pooling widens the intensity range the bins cover and averages them; it does not make them more precise than the best file's, because files of one session share the scatter of their bins (what the scene does to a bin), which does not average away. So pooling copies of one estimate returns that estimate, and further frames of one scene stay with the single-file fit to a few percent.

`BatchPool` keys on (model, ISO, bit depth, compression), merges variance points of the same (plane, bin, kind) and refits; the filtered-file factor is per file and never pooled. A merged point has the count-weighted mean and variance, the summed count, and **the relative variance of its best member** (not `1 / N`). The merge is written `a + (b − a)·w`, which returns equal points exactly, so copies pool to the single-file fit bit for bit. `ym-core` keeps an agreement check (`POOL_AGREEMENT_FACTOR`) for members whose models disagree beyond what pooling can resolve.

Tests (`crates/ym-noise/tests/pooling.rs`):

- three frames of one ramp scene (`tests/data/pool-ramp-iso800-3-frames.json`) pool to within the single-file fits' range;
- copies of one estimate pool to that estimate (property test, fixed seed, 600 cases, with the hardest case pinned);
- frames with a common scatter of up to 9% pool inside the single-file fits' range widened by `2% + scatter / 2` (property test; the worst excursion over 600 cases is 3.2%);
- 18 whole-estimator scenes of 2 to 4 frames pool within 2.2% in `a` and 2.3% in the variance at 0.18 of the single-file fits' mean.

Known limits: a textured bin of one file raises the merged bin (the robust fit sees one point where it saw two), and with a common scatter of about 12% or more (four times the nominal error) the robust iteration no longer settles for a single file either.

## Priors from public files (M2-01)

No calibration frames in V1. `training/noise/public_priors.py` runs the estimator (without priors) over every CC0 and free-licence file in the corpus (`docs/test-photo-sources.md`, tiers 1–2), keeps estimates whose smooth-region bins cover at least 60% of the intensity range, and fits per body:

- `log a = α·log ISO + β` per gain regime, where the body has files at 5 or more ISOs;
- `b` per regime, where it has 3 or more;
- otherwise nothing, and the body falls back to the class prior.

Output: `data/cameras/noise/<model-id>-<model>.json` per body that qualifies, and `data/cameras/noise/gain-regimes.json` with the table above. These are statistics derived from the files, not the files themselves, so they ship with the app; each JSON's `provenance` records the source files.

**Training sampler** (`docs/07`): virtual cameras sampled the ELD way, with ranges set from physics (14-bit Sony sensors, ISO 100–102400) and the relations published in the ELD paper (log σ_read and log σ_row linear in log K). ELD's evaluation-only parameter files are used only in validation, never for training.

**V2:** a proper A1 calibration (dark and flat frames per ISO) replaces the public-data prior for that body and becomes the truth for the A1 gate.

## σ map

`ym_noise::sigma_map` turns the model into the σ the denoise stages read.

- `sigma_planes` returns the four balanced-plane maps: a 5×5 box on the signal, the per-plane gain `g0`, the vignetting **gain** as the multiplier R3 applied (`VignetteProfile::gain_at`, `1/G(r)` in the notation of `docs/05-raw-processing.md`), and the `qstep` map (none, packed or single).
- `sigma_quad` reduces them to the one plane `docs/07` gives the network, as the RMS over the four positions of a 2×2 cell. The training sampler must use the same reduction.
- `SigmaParams` is a 48-byte plain struct, the uniform of the device kernel.

On the device the half-resolution σ plane is made from the resident raw words in one pass (`ym_gpu::kernels::sigma`, `shaders/sigma.wgsl`, one thread per 2×2 cell), with the CPU's bits; a second kernel (`shaders/sigma_expand.wgsl`) expands the plane to a tile's full-resolution map with the rule of `full_sigma_map`. `ym_noise::sigma_map` is the reference both are tested against. The refinement of step 7 runs on the CPU only.

## Validation

| Test | Truth | Gate |
|---|---|---|
| Many bodies | noise-on-noise: add a known Poisson–Gaussian increment to CC0 and free-licence corpus files and recover the increment | increment within 15% median, 30% p90 |
| A1 absolute level | studio ISO series of the A1 (evaluation tier, local only): fit `a`, `b` on the flat grey patches and compare with the per-image estimate on the same files' textured areas | `a` within 15% median |
| Second body | ELD's released Sony A7S II parameters (evaluation-only data) on A7S II test photos | `a` within 15% median |
| Filtered files | A7R II/A7 III long exposures with and without star-eater conditions | detector flags ≥ 90% of filtered, ≤ 5% false positives |

**Status of the four gates in a V1 build (no corpus, no A1 series, no A7S II files).** Only the first has any evidence, and that is synthetic; none of the four is passed on real files.

| Gate | Status | Evidence / blocker |
|---|---|---|
| Noise on noise | synthetic only: met (see *Synthetic validation*) | increment `a` within 15% median / 30% p90 on mixed, busy and smooth procedural scenes; the real-file run needs M2-01's corpus |
| A1 absolute level | blocked | needs the A1 studio series (evaluation tier, local only); the V1 data rule forbids our own captures |
| Second body (ELD A7S II) | blocked | needs ELD's released A7S II parameters (`tools/fetch/eld-params.sh` is not written) and A7S II test photos |
| Filtered files | synthetic stand-in only: met on smooth scenes, **not met on textured scenes** | smooth scenes: 97.5% (median3), 100% (box) flagged, 0 false positives; mixed scenes 40% / 85%; busy scenes 17.5% / 57.5% to 92.5%. The gate (≥ 90% flagged, ≤ 5% false positives) holds for smooth scenes only; real star-eater captures are not available |

darktable's `noiseprofiles.json` (GPL-3.0) may be read offline for sanity checks; it is never shipped or committed.

### Synthetic validation

The corpus and the A1 series are not reachable in V1 builds, so the truth below is a known Poisson–Gaussian model on procedural scenes (`ym_noise::model::synth`, `crates/ym-noise/examples/validate_synthetic.rs`, 1536×1024 mosaics, random cameras over ISO 100–25600). The numbers are from held-out seeds (40000 and up). They do not replace the gates above, which need real files.

| Check | Result (held-out) |
|---|---|
| Absolute recovery, mixed scenes (100) | `a`: median 1.1%, p90 3.0%; σ: median 0.5%, p90 1.3% |
| Absolute recovery, smooth scenes (60) | `a`: median 1.1%, p90 2.3%; σ: median 0.5%, p90 1.1% |
| Absolute recovery, busy (textured) scenes (60) | `a`: median 5.2%, p90 24.3%, biased high; σ: median 1.8%, p90 11.3%. The 15 scenes on which the fallback ran: `a` +21.5% median, σ about +10% |
| Noise on noise, mixed (100) / busy (60) / smooth (40) | increment `a`: median 1.8% / 3.0% / 1.7%; p90 5.6% / 19.4% / 5.2% (gate: 15% / 30%) |
| Filtered detector, smooth scenes | flagged: median3 97.5%, 3×3 box (w = 0.8, 1.0) 100%, median3 then box 87.5%; white noise 0% (0 of 40) |
| Filtered detector, mixed scenes | flagged: median3 40%, box (w = 0.8, 1.0) 85%; white 0 of 39 evaluated |
| Filtered detector, busy scenes | flagged: median3 17.5%, box w = 1.0 92.5%, box w = 0.8 57.5%; white 0 of 25 evaluated |
| Clipping | steep ramp, `a = 8·10⁻³`: clip-aware +0.8% against −92.7% naive; mixed scenes with 9–41% clipped pixels: −1.9% median signed (naive −2.0%), p90 7.9% |
| Row noise | `σ_row` at 1× to 2× the read noise recovered at 0.89–0.94 of the truth (p10 0.81–0.84); at 0.5× median 0.89 but p10 0; at 0.25× it reads as 0; true 0 reads as 0 |
| Refinement, oracle-quality denoiser | σ error −1.6% (mixed), −6.7% (busy); a bilateral filter that keeps correlated noise under-reads by 12% (mixed) |

Known limits:

1. Strongly textured, low-ISO scenes still over-read `a` (fallback or not), because every window carries texture.
2. The filtered-file detector needs clean windows, so it flags well on smooth scenes and poorly on textured ones (false positives stay at 0 in all tests); filters with a binned/pixel ratio below about 0.45 are not flagged by construction.
3. `δ_c` is not identifiable from smooth regions.
4. Refinement reads low when the denoiser leaves correlated noise in place.

Real-file behaviour is not validated yet: that needs the corpus run of M2-01.

## DNG NoiseProfile

Our DNGs are denoised LinearRaw, so either omit `NoiseProfile` or write the re-estimated residual noise of the output. V1 omits it and sets `NoiseReductionApplied = 1` (`docs/09-export.md`).

## Performance

Target: estimation on the 2 MP grid < 80 ms per file on CPU.

Measured with `cargo run -p ym-noise --example bench_estimate -- 5 4` on a 4-core x86-64 machine, 50 MP synthetic mosaic, 5 runs on 4 threads:

| Machine state | Minimum | Median | Maximum | CPU time per estimate |
|---|---|---|---|---|
| shared (load average about 1.5) | 54.1 ms | 74.8 ms | 89.8 ms | 116 ms (read 3.0, windows 7.8, passes 22.0, fallback 7.7, extras 11.3 ms in the best run) |
| idle | 50.8 ms | 52.9 ms | 73.8 ms | 96 ms |

The idle figures are the estimator's cost and the loaded ones the worst case; the spread between runs is machine load. The target is for an Apple M1 (HUMAN M0-H4), so the figures are to be re-measured on the Mac.

Not on the device yet: the refinement of step 7 (target < 50 ms on the GPU, M2-02). It is CPU code, and when a kernel exists it is checked against `estimate::refine` with a parity test like the other kernels.

## Implementation notes

- **Determinism.** The result does not depend on the thread count (tested).
- **Size checks.** `U16Mosaic` and `F32Plane` check their size arithmetic, so header-sized garbage cannot overflow it.

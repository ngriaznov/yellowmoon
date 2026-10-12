# 06 · Noise estimation for every camera

Crate: `ym-noise`. Background and sources: `research/notes/noise-estimation.md`.

The denoise network needs a per-pixel noise σ. V1 has no calibration shoots (no dark or flat frames for any body), so every file gets its noise model estimated from its own pixels, with priors built from public files. Nothing is stored between sessions.

## Model

Per image, in normalised units (after R1, before white balance), for CFA plane `c`:

```
var(μ) = a · (μ − δ_c) + b + σ_row² + q²/12
```

- `a`: shot-noise gain, **shared** by the four CFA planes (the gain in DN doesn't depend on colour; per-plane differences in published profiles are white-balance scaling).
- `δ_c`: small per-plane black-level offsets (ELD's "colour bias"; the largest single gain in its ablation).
- `b`: read-noise variance.
- `σ_row`: row-noise σ (usually small on Sony).
- `q`: the quantisation step (from the decoder's `qstep` for cRAW and ARW 6.0; 1 DN otherwise, normalised).

σ map for the network: `σ(x) = sqrt(a · x̂ + b + σ_row² + q(x)²/12)`, with `x̂` a 5×5 box-smoothed version of the signal (never the noisy value), in the same balanced units as the network input (multiply by `g0` per channel).

## Estimator (write from the papers; reference code licences forbid reuse)

1. **Planes.** Split the normalised mosaic into four half-resolution planes. Work on a 2 MP crop grid (16 tiles spread across the frame) for speed.
2. **Smooth regions.** Per plane, compute a 2D Haar or Daubechies detail coefficient at each pixel and a local gradient energy; keep pixels whose local energy is in the lowest 40% for their intensity bin (Foi 2008).
3. **Bins.** 24 intensity bins from 0 to 0.98 (log-spaced below 0.05). Per bin: robust variance by MAD of detail coefficients (`(1.4826·MAD)²`), the bin mean, and the count. Drop bins with fewer than 400 samples.
4. **Clip-aware fit.** For bins near white (mean > 0.85), use the clipped-Gaussian likelihood (Foi 2008 §4) so clipping doesn't bias `a` downward.
5. **MAP fit** of `(a, b, δ_c)` to the bin statistics with weighted least squares in variance, plus the priors below as Gaussian penalties in `log a` and `b`.
6. **Texture fallback.** If fewer than 6 bins survive (busy scenes), run a Chen 2015-style PCA estimate per bin on 7×7 patches (smallest eigenvalue over a confidence test), after a variance-stabilising transform with the prior `a`, and fit again.
7. **Refinement (YOND-style).** After the first denoise of the preview, re-estimate per bin from `var(noisy − denoised) − var(denoised − smoothed denoised)` and refit once. Use the refined model for export renders.
8. **Row noise.** In smooth regions, take the median of each row's residual, then the variance of the second difference across rows; subtract the pixel-noise contribution expected from the model. Clamp at 0.
9. **Filtered-file check.** Compare the variance of 2×2-binned residuals with the per-pixel variance; white noise gives 1/4. A ratio above 0.4 means the camera filtered the raw (Sony's "star eater" on long exposures, or other in-camera processing). Then raise σ by the factor that restores the ratio to the observed spatial correlation, at least ×1.3. Over-estimating is the safer error.

## Priors

| Camera | Prior on `a` | Prior on `b` |
|---|---|---|
| Bodies with a public-data prior (M2-01) | fitted `log a = α·log ISO + β` per gain regime, σ = 0.25 in log | fitted per regime, σ = 40% |
| Any other body | `a ∝ ISO` around a body-class median (full-frame 24–61 MP, APS-C, 1-inch), σ = 1 stop in log | wide: log-uniform over two decades |

**Gain regimes.** Sony bodies switch conversion gain at one ISO; read noise drops there. Our own table (from public measurements, to be confirmed on samples): A1 500, A1 II 500, A7 III 640, A7 IV ~400, A7 V 1000, A7R IV 320, A7R V 320, A7S III 1600, A7C 640, A7C II 400, A9 500, A9 II 640, A9 III 800 (base 250). Unknown bodies: no break, wide prior. Data overrides the table.

**ISO quirks.** Extended low ISOs (50–80) use base gain; Hi ISOs above about 16000 (A1, A7R V, A9 III), 64000 (A7 III, A7 IV) or 128000 (A7S III) look processed: widen the prior and rely on the data. Use the maker-note ISO, since EXIF ISO saturates at 65535.

**Batch pooling.** Within a session, files from the same model, ISO, bit depth and compression share a pooled estimate (weighted by bin counts), updated as files finish decoding. Nothing is written to disk.

## Priors from public files (M2-01)

No calibration frames in V1. Instead, `training/noise/public_priors.py` runs the estimator (without priors) over every CC0 and free-licence file in the corpus (`docs/test-photo-sources.md`, tiers 1–2), keeps estimates whose smooth-region bins cover at least 60% of the intensity range, and fits per body:

- `log a = α·log ISO + β` per gain regime, where the body has files at 5 or more ISOs;
- `b` per regime, where it has 3 or more;
- otherwise nothing, and the body falls back to the class prior.

Output: `data/cameras/noise/<model-id>-<model>.json` per body that qualifies, and `data/cameras/noise/gain-regimes.json` with the table above. These are statistics derived from the files, not the files themselves, so they ship with the app; record the source files in each JSON's `provenance`.

**Training sampler** (`docs/07`): virtual cameras sampled the ELD way, with ranges set from physics (14-bit Sony sensors, ISO 100–102400) and the relations published in the ELD paper (log σ_read and log σ_row linear in log K). Don't load ELD's evaluation-only parameter files for training; use them only in validation.

**V2:** a proper A1 calibration (dark and flat frames per ISO) replaces the public-data prior for that body and becomes the truth for the A1 gate.

## Validation

| Test | Truth | Gate |
|---|---|---|
| Many bodies | noise-on-noise: add a known Poisson–Gaussian increment to CC0 and free-licence corpus files and recover the increment | increment within 15% median, 30% p90 |
| A1 absolute level | studio ISO series of the A1 (evaluation tier, local only): fit `a`, `b` on the flat grey patches and compare with the per-image estimate on the same files' textured areas | `a` within 15% median |
| Second body | ELD's released Sony A7S II parameters (evaluation-only data) on A7S II test photos | `a` within 15% median |
| Filtered files | A7R II/A7 III long exposures with and without star-eater conditions | detector flags ≥ 90% of filtered, ≤ 5% false positives |

darktable's `noiseprofiles.json` (GPL-3.0) may be read offline for sanity checks; never ship or commit it.

## DNG NoiseProfile

Our DNGs are denoised LinearRaw, so either omit `NoiseProfile` or write the re-estimated residual noise of the output. V1 omits it and sets `NoiseReductionApplied = 1` (`docs/09-export.md`).

## Performance

Estimation on the 2 MP grid: < 80 ms per file on CPU. Refinement: < 50 ms on GPU.

## Implementation notes (ym-noise, M2-02)

Clarifications recorded while building the estimator from the papers. None change a number above; they say how the steps were realised.

- **Units and planes.** Everything is in normalised units (after R1, before white balance); the plane index is the position in the 2×2 cell, `p = 2·(y & 1) + (x & 1)`. The estimator never needs to know which colour a position carries. `ym-noise` does not depend on `ym-raw`: the input is a `PlaneSource` (`U16Mosaic` with black/white from R1, or `F32Plane` for an already-normalised mosaic). `ym-raw` is a dev-dependency of the `estimate_arw` example only.
- **Step 2, windows.** A window is 6×6 plane pixels (3×3 Haar blocks). The measurement is the diagonal Haar coefficient `dd`, which cancels row offsets and linear ramps; the energy is the detrended horizontal and vertical detail plus the mean `dd²` of the neighbouring windows. Per intensity bin the 40% of windows with the lowest energy are kept (70% in clipped bins, where ranking hard on energy would pick the most clipped windows). A second pass normalises energy and coefficients by the first fit, which removes the bin-width and selection bias; more passes change `a` by less than 0.1%.
- **Step 3, bins.** The 24 bin edges run from 0.002 to 0.98, log-spaced below 0.05. The variance of a bin is `(1.4826·MAD)²` of the `dd` coefficients. A bin is dropped below 400 samples; a (bin, plane) pair needs 100 samples to give a point of its own.
- **Step 4, clipping.** For bins with a mean above 0.85 the (mean, variance) of the clipped Gaussian is inverted to the unclipped `(μ, σ²)` (`h(u) = uΦ(u) + φ(u)`, `g(u) = (u²+1)Φ(u) + uφ(u) − h²`, the ratio `h/√g` is monotone in `u`).
- **Step 5, fit.** Weighted least squares in variance with a one-sided robust loss (a Cauchy-type weight above the line and Huber below it, then Tukey's bisquare; texture can only add variance), `δ_c` with a ridge prior (σ = 3·10⁻⁴), `a` log-normal and `b` per the prior table. `δ_c` is not identifiable from smooth regions alone and stays near zero unless the data pull it. `b` is also weakly identified when `a·μ` dominates every surviving bin; the sigma map stays right because the combination that matters is what the bins constrain.
- **Step 6, fallback triggers.** The patch-PCA fallback (7×7 patches, variance-stabilising transform with the prior `a`, Marchenko–Pastur edge correction, Wilson–Hilferty confidence threshold) runs when (a) fewer than 6 bins survive, (b) 15% or more of the points sit well above the fit, or (c) a cheap three-bin PCA probe finds the noise below 0.95 of the smooth-region fit (texture the other tests cannot see; clean images give 1.05–1.1). Probe results below 0.5 of the fit are not trusted: spatially correlated noise looks nearly noise-free to PCA and is the filtered-file check's business.
- **Step 7, refinement.** `estimate::refine(prev, noisy, denoised, prior, cfg)` implements the YOND-style residual `var(noisy − denoised) − var(denoised − smoothed denoised)` with the retained-noise correction capped at 15% of the residual variance, then refits once. It returns a new `NoiseEstimate`; row noise and the filtered-file factor are carried over from `prev`.
- **Step 8, row noise.** The variance of the row offset is measured from cross-tile cross-products of second differences across rows (tile rows 0 and 2 only, for speed), so the pixel-noise part never has to be subtracted; clamped at 0. A row σ below about half the read-noise σ is not reliably resolved and often reads as 0.
- **Step 9, filtered-file check.** `R = Σ binned residual variance / Σ pixel residual variance` over the cleanest quarter of the dark-half smooth windows after a plane fit (0.25 for white noise). Flagged above 0.4; the factor is `sqrt(pixel residual variance / model pixel variance)` clamped to [1.3, 4]. The check is skipped when the texture fallback ran or the fit has no data (texture raises `R` the same way correlated noise does); the result then says `evaluated = false`.
- **Batch pooling.** `BatchPool` keys on (model, ISO, bit depth, compression), merges variance points weighted by count and refits; the filtered-file factor is per file and never pooled.
- **Sigma map.** `sigma_planes` returns the four balanced-plane maps (5×5 box on the signal, per-plane `g0`, vignetting gain, `qstep` map: none, packed or single) and `sigma_quad` reduces them to the one plane `docs/07` gives the network, as the RMS over the four positions of a 2×2 cell. The training sampler must use the same reduction. `SigmaParams` is a 48-byte plain struct for a GPU uniform.
- **Priors.** The class prior is derived from physics, not tuned: full-well capacity 2000 e⁻/µm² times an assumed pixel area per body class, `a = 1/FWC` at ISO 100 and `∝ ISO` above (σ one stop in `ln a`); `b` log-uniform between `(a·1 e⁻)²` and `(a·10 e⁻)²`. The assumptions are public constants in `ym_noise::priors::ClassAssumptions`. `gain-regimes.json` carries the table above; every row says `"source": "docs/06 table, to be confirmed on samples"`.
- **Determinism.** The result does not depend on the thread count (tested).
- **Review corrections (M2-02 review).** (1) The soft wall of the log-uniform prior on `b` is applied as a reweighted quadratic in `b − bound` (the secant of the log residual through the bound). The first version linearised `ln b` at the current `b`; once `b` exceeded e times the bound the step went to a negative `b` and the iteration cycled (`a` up to 40 times too large on weak data). (2) The mean intercept is kept above `−a·0.002`: the model can carry a negative intercept only as a common plane offset, and that offset is clamped at 2·10⁻³ of white (32 DN at 14 bit). Without the wall, a prior that pinned `a` above what the data allow left a model whose variance at the image's own levels was too high (+24% for a prior 25% too high on a flat field, +100% for a prior 2× too high); now `a` gives way. (3) `BodyClass::from_model` knows the NEX, ILCE-3/5xxx, A-mount and fixed-lens bodies (ILCE-5000, NEX-5N, NEX-7 and DSLR-A500 used to get the full-frame class). (4) `sigma_planes` takes the vignetting **gain** as the multiplier R3 applied (`VignetteProfile::gain_at`, `1/G(r)` in the notation of `docs/05`). (5) `U16Mosaic` and `F32Plane` check their size arithmetic (header-sized garbage used to overflow it).

## Synthetic validation (M2-02, no corpus yet)

The corpus and the A1 series are not reachable in V1 builds, so the truth below is a known Poisson–Gaussian model on procedural scenes (`ym_noise::model::synth`, `crates/ym-noise/examples/validate_synthetic.rs`, 1536×1024 mosaics, random cameras over ISO 100–25600). Final numbers are from seeds 40000 and up with a frozen binary; the tuning seeds were different. These numbers do not replace the gates in the Validation table, which need real files.

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

Known weak spots: (1) strongly textured, low-ISO scenes still over-read `a` (fallback or not), because every window carries texture; (2) the filtered-file detector needs clean windows, so it flags well on smooth scenes and poorly on textured ones (false positives stay at 0 in all tests; the detection gate is met on smooth scenes only), and filters with a binned/pixel ratio below about 0.45 are not flagged by construction; (3) `δ_c` is not identifiable from smooth regions; (4) refinement reads low when the denoiser leaves correlated noise in place. Real-file behaviour is open until M2-01's corpus run.

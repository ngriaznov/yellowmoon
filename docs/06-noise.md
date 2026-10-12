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

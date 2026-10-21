# Integration guide

This file maps the algorithms of `docs/algorithms` onto Yellowmoon: where each one sits in the raw chain, and the crates, types, units, noise model, tiling and export path it uses. Each spec repeats the parts it needs; this file is the overview and the tie-breaker for conventions. Where a spec and the code differ on conventions (units, layout, borders), this file decides.

Specs, as cited by the code: spec/01 is `01-adaptive-ratio-demosaic.md`, spec/02 `02-log-guided-demosaic.md`, spec/03 `03-diffraction-deconvolution.md`, spec/04 `04-output-sharpening.md`, spec/05 `05-single-frame-denoise.md`.

## 1. Where each algorithm goes

```
R1 normalise → R3 vignetting → R4 gains g0 → R5 highlights → R5b CA on the mosaic
→ R6 demosaic ─┬─ AdaptiveRatio   (spec/01; THE DEFAULT: 100 % tiles and exports)
               ├─ Bilinear | Menon | Lmmse   (reachable through the test-only switch only)
               └─ LogGuided       (spec/02; test-only switch; no render path uses it)
  → R6 single-frame denoise: method A (spec/05) in the working encoding
  → false-colour step (R7's colour guided filter where the luma alternates)
        [the reduced path (fit view, thumbnails) runs the bilateral denoise on the binned image;
         the bilateral denoise followed by R7 is also what the internal denoise switch can select]
→ R8 capture sharpening ─┬─ RichardsonLucy                       (default)
                         └─ Diffraction   (spec/03; internal switch, not the default)
→ R9 lens resample → colour C1…C12
→ [Lanczos-3 resize, exports] → OutputSharpen (spec/04; amount = DevelopSettings.output_sharpening,
                                default 100 %, every display-referred picture, never DNG) → C13 + dither → writer
```

R5b is the lateral CA correction on the mosaic (`docs/05-raw-processing.md` R9). Output sharpening is not in the raw chain: it runs on the display-referred picture, after C12, at the size the picture is shown or written.

| Algorithm | Rust home | Selection | Ships as |
|---|---|---|---|
| Adaptive ratio demosaic | `ym-gpu/src/cpu/demosaic/adaptive_ratio.rs`, `Method::AdaptiveRatio` | `Chain::method`; `DEFAULT_METHOD` | the default for R6 classical |
| Log-guided demosaic | `ym-gpu/src/cpu/demosaic/log_guided.rs`, `Method::LogGuided` | the test-only switch only. It meets its targets (spec/02 section 9), but the fit view and the thumbnails at a reduction of 2 or more bin the mosaic and never demosaic (spec/02 section 9.3) | exists; no render path uses it |
| Single-frame denoise (method A and the false-colour step) | `ym-gpu/src/cpu/frame_denoise.rs`, `bm_dct.rs`, `false_color.rs`, `denoise_space.rs`; `DenoiseMethod::BmDct` | `RenderOptions::denoise`, `DEFAULT_DENOISE`; methods B and C and the bilateral path stay behind the same internal switch (spec/05 *Integration*) | the default for every full-resolution render |
| Diffraction deconvolution | `ym-gpu/src/cpu/sharpen/diffraction.rs` | `CaptureSharpen::Diffraction`, chosen by the internal `PhotoSource::capture_method` (tests and evaluation tools) | not the default; Richardson–Lucy is the default of R8 (`DEFAULT_CAPTURE_METHOD`) |
| Output sharpening | `ym-gpu/src/cpu/output_sharpen.rs`; export: `ym_export::resize::sharpen_output` in `encode_to_writer_with` (`PictureStages`); views: `ym-core/src/pipeline/frame.rs` | `DevelopSettings::output_sharpening`, 0 to 200 %, default 100; the screen preset, nothing else exposed | **on by default**, every display-referred picture, never DNG, no export switch |

## 2. Types and layouts

| Yellowmoon type | Meaning | What the algorithms assume |
|---|---|---|
| `ym_gpu::cpu::Mosaic` | f32 per photosite, row-major, black-subtracted, white = 1.0, `Cfa` phase of the buffer origin, `Placement` | spec/01 and spec/02 input |
| `ym_gpu::cpu::RgbImage` | **planar** f32 (R plane, G plane, B plane) | demosaic output, R6 denoise, R7 and R8 input and output |
| `ym_export` `LinearImage` / `ImageRef` | **interleaved** f32, display-linear | spec/04 input and output (convert with `RgbImage::from_interleaved` / `to_interleaved` or run interleaved; the result is identical) |
| `ym_gpu::cpu::mirror` | mirror without repeating the edge sample, CFA parity kept | all specs use the same rule at buffer edges |
| `ym_gpu::cpu::LUMA` (Rec. 709) | luma weights of R6 to R8 | spec/03 and spec/04 default to their own weights 0.299/0.587/0.114 (see each spec's evaluation notes); `LUMA` itself does not change |

CFA phases: every algorithm accepts all four Bayer phases. spec/01 pads once at entry with a parity that puts red at even/even; spec/02 classifies sites per phase. Both are covered by the `phase_*` vectors.

## 3. Units: the 16-bit working scale

The demosaics' absolute constants are defined at a working scale of **S = 65535** relative to white:

```
x_dn = S · x                                   (mosaic, planes)
g    = S · a                                   (shot-noise gain)
r    = S · sqrt(b + sigma_row² + q²/12)        (signal-independent sigma)
σ_dn²(x_dn) = g · x_dn + r²
```

`NoiseDn { g: f64, r: f64 }` with `NoiseDn::from_model(&NoiseModel)` in `ym-gpu/src/cpu` converts the normalised model of `docs/06-noise.md`. `NoiseModel::delta` is not used. S stays fixed relative to white even if the normalisation changes. spec/03 and spec/04 state their own unit conventions: spec/04 is fully normalised, and spec/03 lists its conversions in its section 1. The single-frame denoise (spec/05) works on the encoded image, with the thresholds derived from the model in normalised units (`frame_denoise::working_noise`).

## 4. White balance around R6

| Algorithm | Input of the algorithm | What R6 does |
|---|---|---|
| AdaptiveRatio | **unbalanced** mosaic plus `gains` | divide each photosite by `g0[colour]` (exact inverse of R4), call with `gains = g0`; output is balanced |
| LogGuided | balanced mosaic | call directly; output is balanced |
| Bilinear, Menon, Lmmse | balanced mosaic | unchanged |

`demosaic_with(m, method, &inputs)` and `demosaic_whole_with` take the **balanced** mosaic, like every method, `AdaptiveRatio` included, and return the balanced image. For `AdaptiveRatio` they undo R4 themselves: each photosite is divided by `inputs.gains` of its CFA colour (`adaptive_ratio::demosaic_balanced`, an `f64` division rounded to `f32` once) and the three passes run with `gains = g0` and the chain's noise. With unit gains the division is the identity, so `demosaic(m, AdaptiveRatio)` gives the bits of the unbalanced entry point. `adaptive_ratio::demosaic` is the unbalanced entry point of spec/01 section 1.2 (the vectors use it). The `Chain`, the tile scheduler and R7's `Y_classical` get the division through `demosaic_with`. `LogGuided` is called directly and reads only the read noise of the inputs.

## 5. Halos (`ym_gpu::cpu::tiling`)

| Stage | Halo (px) | Source |
|---|---|---|
| `Demosaic(AdaptiveRatio)` | 13 (11 without shadow matching; `Method::halo()` returns 13) | spec/01 §1.5 |
| `Demosaic(LogGuided)` | 5 | spec/02 §1.4 |
| `FrameDenoise` (method A) | 23 (`bm_dct::HALO_PX`); methods B and C read the whole frame | spec/05 reading A-d |
| `FalseColor` | `max(2 r, 5)`, `r` the guided filter's radius (10 at Auto Colour) | spec/05 *False colour* |
| `Sharpen` (Diffraction) | `10 + R + r_G`, at most 25 (R = PSF kernel radius, r_G = high-pass Gaussian radius). Richardson–Lucy's 24 px cap (`SHARPEN_HALO_CAP_PX`) is not applied to this method; `Stage::Sharpen` takes the halo of its method (`CaptureSharpen::halo_px()`, `DiffractionParams::halo_px()`, 0 when inactive) | spec/03 |
| Output sharpening | 12, or 20 with halo suppression (method 2: 13 / 21; print mode up to 33 / 41) | spec/04 §1.4 |

Output sharpening is not in the raw tile walk. Callers ask `SharpenPlan::halo_px(scale)` for the exact field of the plan (screen preset 17 px at full size, 9 px for a 2048 px long edge from 6000; print preset 14 px at full size) and `OutputSharpen::max_halo_px()` for the bound of the mode (21 px screen, 41 px print). `Stage::Detail` takes `DEFAULT_METHOD.halo()` (13) for the detail term's `Y_classical`.

The scheduler aligns mosaic rectangles to even coordinates. `cpu_tiling.rs` iterates `Method::ALL`, so every demosaic method has a tiling test.

## 6. Lateral CA curves for spec/01 pass 2 (the CA-aware green refinement)

The CA-aware green refinement takes two radial curves, `shift_px` against `radius_px` from the optical centre, for red and blue. They derive from the R9 lens data (Sony tag 0x7035, `ym_gpu::cpu::lens::LensGeometry`): with `K_c(r)` the per-channel radial scale factor, `shift_c(r) = r_src(r) · (K_c(r) − 1)`. spec/01 §3.6 gives the details, including the effect of distortion on `r_src`. Without CA tags, or with `decode.chromatic_aberration = false`, the call passes `ca = None`.

`LensGeometry::ca_curves()` builds the curves, and returns `None` for a table whose knots are all zero (it moves no pixel in R9, so the CA path would only add the pass-2 smoothing; spec/01 section 7 note 4).

**The chain does not reach this path.** Lateral CA is corrected on the mosaic before the demosaic (R5b, `LensGeometry::with_ca_on_mosaic`, `docs/05-raw-processing.md` R9 "Real files"), because a green-guided demosaic absorbs about half of the CA. `Chain::ca_curves` therefore passes `None` when the lens corrects its CA on the mosaic, which every full-resolution `RawPlan` does. The CA-aware path, its vector and its tests are kept for a geometry that keeps its CA in R9 (`LensGeometry::new` alone).

## 7. Inputs from the camera registry and the metadata

| Needed by | Input | Where it comes from |
|---|---|---|
| spec/03 | pixel pitch (µm) | `sensor_width_mm` and `active_width_px` per body in `data/cameras/registry.json`, from published specifications with the source in the entry's provenance (27 of 28 bodies; the ILME-FX3 has none). Fallback: `43266.6 / (crop · √(W² + H²))` with `crop = focal_length_35mm / focal_length_mm` from `RawMetadata` (about 2 % high on APS-C). A pitch outside 0.5 to 20 µm counts as not available. If none is available, diffraction deconvolution is disabled for the photo |
| spec/03 | f-number | `RawMetadata::f_number`. Disabled when it is missing or outside 1–50 |
| spec/04 | output scale `s` | `out_long_edge / rendered_long_edge` from the export request (`scale_for_edges`): 1.0 for full-size exports, the zoom for a viewport frame, a thumbnail's size over the photo's, 1 for the 100 % view |

## 8. Settings, parity and determinism

- **Settings.** The R6 demosaic method (`RenderOptions::demosaic`), the R6 denoise method (`RenderOptions::denoise`) and the R8 method (`PhotoSource::capture_method`) are internal switches living in `RawParams`, `Chain` and `RenderOptions`, not in `settings.schema.json`: no settings field, no UI control, no CLI flag, and nothing stored per photo. Output sharpening is the exception: `DevelopSettings.output_sharpening` is in `settings.schema.json`, `SettingsPatch`, the `ym:Settings` of an export, the CLI (`--output-sharpening`) and the UI (Detail › Sharpening › Output); old settings documents read as 100.
- **The R6 method switch** is `RenderOptions::demosaic` in `ym-core` (`DEFAULT_METHOD` unless a test sets it) with a test-only environment variable, `YM_FORCE_DEMOSAIC`, behind the `force-demosaic` cargo feature; see spec/01 section 10.
- **Determinism.** All five algorithms are pure functions of their inputs and per-frame constants, with no reductions that depend on thread order. f64 stays where the specs say so. `--jobs 1` and `--jobs 4` are byte-identical.
- **Goldens.** Hashes change only when a default changes; then `golden update --note "…"` records the gate numbers in the note, and `PIPELINE_VERSION` is bumped (it is 0.5.1: method A is the default denoise, with its chroma threshold per block since 0.5.1).
- **GPU.** Every WGSL port has a CPU reference and a parity test (`AGENTS.md`). Each spec's §5 lists which parts are per-pixel and which need care: decision near-ties in spec/01 and spec/05, and the window counts of the halo limiter in spec/04. The device contracts are in `docs/17-gpu.md`.
- **Deviations and constants.** The specs describe exact behaviour. Where a port deviates (for example f32 instead of f64 in a reduction), its module doc comment says so, and it stays within the vector tolerances. A constant that is not in a spec is not added; it is raised as a question first.

## 9. Vectors

- **What a vector holds.** Each vector JSON has `{name, description, params, input, expected, tolerance}`. Rust tests load them from `crates/ym-gpu/tests/cpu_data/<algorithm>/` (`adaptive_ratio_demosaic`, `log_guided_demosaic`, `diffraction_deconvolution`, `output_sharpening`) and compare with the stated tolerance. They follow the pattern of `menon_fixtures.json`.
- **The oracle.** The vectors were produced by independent C++ reference implementations, compiled with `-ffp-contract=off` so GCC and Clang agree, and run under ASan/UBSan. The references are not committed: the vectors are the fixed oracle and are never hand-edited. The Rust ports avoid `mul_add` where a spec says an operation order matters.
- **The single-frame denoise** has no vector files; each reading of spec/05 is pinned by a unit test in `crates/ym-gpu/src/cpu/`, and the device kernels by `tests/gpu_bm_dct.rs` and `tests/gpu_false_color.rs`.

## 10. The R6 default and an unusable noise model

- **The default.** `ym_gpu::cpu::demosaic::DEFAULT_METHOD` is `Method::AdaptiveRatio` (`docs/05-raw-processing.md` R6). `RenderOptions::demosaic` starts there, `FullChain::new` uses it, and Menon, LMMSE, bilinear and log-guided stay selectable through the test-only switch. The fit view and the thumbnails bin the mosaic and do not demosaic, so only the 100 % view and exports pay the demosaic's CPU time. The tile scheduler's `Demosaic(AdaptiveRatio)` halo of 13 px applies to every default tile.
- **What a missing or unusable noise model does.** spec/01 section 4 says the algorithm refuses an invalid model (`BadNoise`), section 7 says "R6 always passes the real model", and a zero model is valid input (K = T = 1 DN, near-pure ratio interpolation). A render must never fail for a noise model, so **a model that is missing or fails `NoiseDn::is_valid` is read as the zero model, `NoiseDn::ZERO`, for that photo.**
  - `NoiseDn::or_zero` is the rule. `Chain::noise_dn` applies it to `Chain::noise` for the R6 demosaic and the R8 diffraction stage, and `RawParams::noise_dn` (`ym-core`) applies it to the model in the parameters for the chain, the diffraction plan and the choice of R8 method.
  - One level above, the renderer (`Job::params` in `pipeline/graph.rs`) replaces a photo's `NoiseModel` by the zero model (`usable_noise_model` in `pipeline/raw.rs`: `NoiseDn::is_valid` and finite black-level offsets) before it enters `RawParams`, so the σ map, the denoise, R7, R8 and R6 all read the same usable model, and it logs one `warn` ("the noise model of the photo is not usable; rendering with zero noise", no path). `FullChain::with_method` logs the same for a `RawParams` made by hand. With a `NaN` gain the σ map would be `NaN` and the frame would come out black (`crates/ym-core/tests/render_noise_model.rs`).
  - A usable model is untouched, bit for bit. The estimator of `ym-noise` never produces an unusable model, so this is a guard against corrupt data, not a path a real photo takes.
  - Known limit: the colour plan (`pipeline/color.rs`, the toe of C7) reads `estimates.noise.model` directly, so a finite but absurd read-noise variance (`b` of 1e300) still blackens the frame there (a `NaN` or negative number is harmless: it is floored at zero).
  - The chain still refuses gains that are not positive and finite (a chain with such gains has no meaningful R4), and the infallible `demosaic_with` still falls back to Menon's bits for unusable gains or noise.
- **`NoiseDn::is_valid`.** Besides finite and non-negative `g` and `r`, the variance at white, `g * S + r * r`, and `g * g` (squared by the derived constants) must be finite. A model of `a = b = 1e300` has finite `g` and `r` but no finite variance over the working range, so it is invalid: read as zero noise in a render and refused (`BadNoise`) by `adaptive_ratio::demosaic`.

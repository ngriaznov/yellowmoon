# Integration guide

> **In this repository (2026-10-13, owner's decision):** the C++ references (`reference/*.cpp`, `tools/check.sh`) are not committed. They were used as a development oracle outside the repository; the vectors in `crates/ym-gpu/tests/cpu_data/<algorithm>/` are the fixed oracle here and are never hand-edited. Where this text says `tools/reference/` or `check.sh`, read it as that external oracle.

How the four algorithms fit Yellowmoon as built on `claude/yellowmoon-v1` (commit f219537): its crates, types, units, noise model, tiling and export path. Each spec repeats the parts it needs; this file is the overview and the tie-breaker for conventions.

> **As built (RA-10, 2026-10-13).** This file was written before the code. Where it and the code differ, the code wins, and the places that changed are marked *as built*: section 1 (what is the default, what is reachable, output sharpening), section 4 (what `demosaic_with` takes), section 5 (halos), section 6 (the CA curves), section 7 (inputs that now exist) and section 8 (settings, goldens). Section 10 was added by the RA-04 and default-switch work.

## 1. Where each algorithm goes

```
R1 normalise → R3 vignetting → R4 gains g0 → R5 highlights → R5b CA on the mosaic
→ R6 demosaic ─┬─ Bilinear | Menon | Lmmse   (existing; reachable through the test-only switch only)
               ├─ AdaptiveRatio   (spec/01; THE DEFAULT: 100 % tiles, exports, Detail's Y_classical)
               └─ LogGuided       (spec/02; test-only switch; no render path uses it)
  → R6 bilateral denoise → R7 detail
→ R8 capture sharpening ─┬─ RichardsonLucy                       (existing, default)
                         └─ Diffraction   (spec/03; internal switch, not the default)
→ R9 lens resample → colour C1…C12
→ [Lanczos-3 resize, exports] → OutputSharpen (spec/04; amount = DevelopSettings.output_sharpening,
                                default 100 %, every display-referred picture, never DNG) → C13 + dither → writer
```
*(As built: the diagram above shows the stage order of the code. R5b is the lateral CA correction on the mosaic, `docs/05` R9; the original diagram had no such stage and put output sharpening in the export only, "off by default".)*

| Algorithm | Rust home | Selection | Ships as |
|---|---|---|---|
| Adaptive ratio demosaic | `ym-gpu/src/cpu/demosaic/adaptive_ratio.rs`, `Method::AdaptiveRatio` | `Chain::method`; `DEFAULT_METHOD` since 2026-10-13 (owner's decision after RA-H4) | the default for R6 classical |
| Log-guided demosaic | `ym-gpu/src/cpu/demosaic/log_guided.rs`, `Method::LogGuided` | the test-only switch only. RA-06 met its targets, but the fit view and the thumbnails at a reduction of 2 or more bin the mosaic and never demosaic (spec/02 section 9.3) | exists; no render path uses it |
| Diffraction deconvolution | `ym-gpu/src/cpu/sharpen/diffraction.rs` | `CaptureSharpen::Diffraction`, chosen by the internal `PhotoSource::capture_method` (tests and evaluation tools) | not the default; Richardson–Lucy stays until RA-H2 (open) |
| Output sharpening | `ym-gpu/src/cpu/output_sharpen.rs`; export: `ym_export::resize::sharpen_output` in `encode_to_writer_with` (`PictureStages`); views: `ym-core/src/pipeline/frame.rs` | `DevelopSettings::output_sharpening`, 0 to 200 %, default 100 (owner's decision of 2026-10-13); the screen preset, nothing else exposed | **on by default**, every display-referred picture, never DNG, no export switch. The first version of this row said `ExportOptions::output_sharpen: Option<OutputSharpen>`, `None` until RA-H3: superseded |

## 2. Types and layouts

| Yellowmoon type | Meaning | What the algorithms assume |
|---|---|---|
| `ym_gpu::cpu::Mosaic` | f32 per photosite, row-major, black-subtracted, white = 1.0, `Cfa` phase of the buffer origin, `Placement` | spec/01 and spec/02 input |
| `ym_gpu::cpu::RgbImage` | **planar** f32 (R plane, G plane, B plane) | demosaic output, R8 input/output |
| `ym_export` `LinearImage` / `ImageRef` | **interleaved** f32, display-linear | spec/04 input/output (convert with `RgbImage::from_interleaved`/`to_interleaved` or run interleaved; the result must be identical) |
| `ym_gpu::cpu::mirror` | mirror without repeating the edge sample, CFA parity kept | all specs use the same rule at buffer edges |
| `ym_gpu::cpu::LUMA` (Rec. 709) | luma weights of the existing R6–R8 | spec/03 and spec/04 default to their own weights 0.299/0.587/0.114 (see each spec's evaluation notes); don't change `LUMA` itself |

CFA phases: every algorithm accepts all four Bayer phases. spec/01 pads once at entry with a parity that puts red at even/even; spec/02 classifies sites per phase. Both are covered by the `phase_*` vectors.

## 3. Units: the 16-bit working scale

The demosaics' absolute constants are defined at a working scale of **S = 65535** relative to white:

```
x_dn = S · x                                   (mosaic, planes)
g    = S · a                                   (shot-noise gain)
r    = S · sqrt(b + sigma_row² + q²/12)        (signal-independent sigma)
σ_dn²(x_dn) = g · x_dn + r²
```

Add `NoiseDn { g: f64, r: f64 }` with `NoiseDn::from_model(&NoiseModel)` in `ym-gpu/src/cpu` (RA-02). `NoiseModel::delta` is not used. Keep S fixed relative to white even if Yellowmoon's normalisation ever changes. spec/03 and spec/04 state their own unit conventions: spec/04 is fully normalised, and spec/03 lists its conversions in its §1.

## 4. White balance around R6

| Algorithm | Input of the algorithm | What R6 does |
|---|---|---|
| AdaptiveRatio | **unbalanced** mosaic plus `gains` | divide each photosite by `g0[colour]` (exact inverse of R4), call with `gains = g0`; output is balanced |
| LogGuided | balanced mosaic | call directly; output is balanced |
| Bilinear, Menon, Lmmse | balanced mosaic (as today) | unchanged |

Put the division inside `demosaic_with` for `AdaptiveRatio` so that `Chain`, the tile scheduler and R7's `Y_classical` all get it for free.

*As built (RA-04).* Exactly that: `demosaic_with(m, method, &inputs)` and `demosaic_whole_with` take the **balanced** mosaic, like every method, for `AdaptiveRatio` too, and return the balanced image. For `AdaptiveRatio` they undo R4 themselves: each photosite is divided by `inputs.gains` of its CFA colour (`adaptive_ratio::demosaic_balanced`, an `f64` division rounded to `f32` once, as the oracle's `demosaic_balanced` does) and the three passes run with `gains = g0` and the chain's noise. With unit gains the division is the identity, so `demosaic(m, AdaptiveRatio)` gives the bits of the unbalanced entry point. `adaptive_ratio::demosaic` stays the unbalanced entry point of spec/01 section 1.2 (the vectors use it). The left column above is therefore the input of the *algorithm*; the input of `demosaic_with` is balanced for every method. `LogGuided` is called directly and reads only the read noise of the inputs. RA-03 had first built the division outside `demosaic_with`; RA-04 moved it in (spec/01 sections 9 and 10).

## 5. Halos (`ym_gpu::cpu::tiling`)

| Stage | Halo (px) | Source |
|---|---|---|
| `Demosaic(AdaptiveRatio)` | 13 (11 without shadow matching; `Method::halo()` returns 13) | spec/01 §1.5 |
| `Demosaic(LogGuided)` | 5 | spec/02 §1.4 |
| `Sharpen` (Diffraction) | `10 + R + r_G`, at most 25 (R = PSF kernel radius, r_G = high-pass Gaussian radius). Richardson–Lucy's 24 px cap (`SHARPEN_HALO_CAP_PX`) must not be applied to this method; give `Stage::Sharpen` a per-method halo | spec/03 |
| Output sharpening | 12, or 20 with halo suppression (method 2: 13 / 21; print mode up to 33 / 41) | spec/04 §1.4 |

*As built.* `Method::halo()` returns 13 for `AdaptiveRatio` and 5 for `LogGuided`. `Stage::Sharpen(CaptureSharpen)` takes the halo of its method (`CaptureSharpen::halo_px()`: Richardson–Lucy capped at 24, diffraction `10 + R + r_G` up to 25, `DiffractionParams::halo_px()`, 0 when inactive). Output sharpening is not in the raw tile walk: callers ask `SharpenPlan::halo_px(scale)` for the exact field of the plan (screen preset 17 px at full size, 9 px for a 2048 px long edge from 6000; print preset 14 px at full size) and `OutputSharpen::max_halo_px()` for the bound of the mode (21 px screen, 41 px print). `Stage::Detail` takes `DEFAULT_METHOD.halo()` (13) for the detail term's `Y_classical`.

The scheduler already aligns mosaic rectangles to even coordinates. `cpu_tiling.rs` iterates `Method::ALL`, so the new methods get tiling tests as soon as they are added there.

## 6. Lateral CA curves for spec/01 pass 2 (phase 2, RA-03b)

The CA-aware green refinement takes two radial curves, `shift_px` against `radius_px` from the optical centre, for red and blue. Derive them from the R9 lens data (Sony tag 0x7035, `ym_gpu::cpu::lens::LensGeometry`): with `K_c(r)` the per-channel radial scale factor, the curve is `shift_c(r) = r_src(r) · (K_c(r) − 1)`. spec/01 §3.6 gives the details, including the effect of distortion on `r_src`. Without CA tags, or with `decode.chromatic_aberration = false`, pass `ca = None`.

*As built (RA-03b, then the lens work of 2026-10-13).* `LensGeometry::ca_curves()` builds the curves as described, and returns `None` for a table whose knots are all zero (it moves no pixel in R9, so the CA path would only add the pass-2 smoothing; spec/01 section 7 note 4). **The chain no longer reaches this path.** Lateral CA is now corrected on the mosaic before the demosaic (R5b, `LensGeometry::with_ca_on_mosaic`, `docs/05` R9 "Real files"), because a green-guided demosaic absorbs about half of the CA and R9 then over-corrected real Sony files. `Chain::ca_curves` therefore passes `None` when the lens corrects its CA on the mosaic, which every full-resolution `RawPlan` does. The CA-aware path, its vector and its tests stay for a geometry that keeps its CA in R9 (`LensGeometry::new` alone).

## 7. Inputs that Yellowmoon doesn't have yet

| Needed by | Input | Where to get it |
|---|---|---|
| spec/03 | pixel pitch (µm) | preferred: `sensor_width_mm` and `active_width_px` (or `pixel_pitch_um`) per body in `data/cameras/registry.json`, from published specifications with the source in the entry's provenance. Fallback: `crop = focal_length_35mm / focal_length_mm` from `RawMetadata`, `pitch_um ≈ 36000 / (crop · active_width_px)` for 3:2 sensors. If neither is available, diffraction deconvolution is disabled for the photo. *As built (RA-07): registry 1.1.0 has the fields for 27 of 28 bodies (the ILME-FX3 has none and takes the fallback); the fallback uses `43266.6 / (crop · √(W² + H²))`; a pitch outside 0.5 to 20 µm counts as not available* |
| spec/03 | f-number | `RawMetadata::f_number`. Disabled when it is missing or outside 1–50 |
| spec/04 | output scale `s` | `out_long_edge / rendered_long_edge` from the export request (1.0 for full-size exports). *As built: `scale_for_edges`; for a viewport frame the zoom, for a thumbnail its size over the photo's, 1 for the 100 % view* |

## 8. Settings, parity and determinism

- **No schema changes in this work.** New switches live in `RawParams` / `Chain` / `ExportOptions`, not in `settings.schema.json`, until Mykyta decides on UI (RA-H2, RA-H3). That keeps the serde-order tests and the app↔CLI parity untouched. *As built: this holds for the R6 method (a test-only switch) and the R8 method (internal, `PhotoSource::capture_method`). Output sharpening is the exception, by the owner's decision of 2026-10-13 (RA-H3): `DevelopSettings.output_sharpening` is in `settings.schema.json`, `SettingsPatch`, the `ym:Settings` of an export, the CLI (`--output-sharpening`) and the UI (Detail › Sharpening › Output); old settings documents read as 100, and the settings tests and the app↔CLI parity tests were extended for it.*
- **The R6 method switch (RA-04)** is `RenderOptions::demosaic` in `ym-core` (an internal option of the renderer, `DEFAULT_METHOD` unless a test sets it) with a test-only environment variable behind the `force-demosaic` cargo feature; see `01-adaptive-ratio-demosaic.md` section 10.
- **Determinism.** All four algorithms are pure functions of their inputs and per-frame constants, with no reductions that depend on thread order. Keep f64 where the specs say so. `--jobs 1` and `--jobs 4` must stay byte-identical.
- **Goldens.** Hashes change only when a default changes (RA-05). Use `golden update --note "…"` with the gate numbers in the note. *As built: two defaults changed in the RA wave, the R6 method (adaptive ratio, RA-05) and the output sharpening amount (100 %, RA-08), so all 61 synthetic files move; they are re-baselined once, with `PIPELINE_VERSION` bumped, in one `golden update --note`.*
- **GPU.** Every WGSL port needs a CPU reference and a parity test (`AGENTS.md`). Each spec's §5 lists which parts are per-pixel and which need care: decision near-ties in spec/01 and the window counts of the halo limiter in spec/04.

## 9. Vectors and the C++ references

- **What a vector holds.** Each vector JSON has `{name, description, params, input, expected, tolerance}`. Rust tests load them from `crates/ym-gpu/tests/cpu_data/<algorithm>/` and compare with the stated tolerance. They follow the pattern of the existing `menon_fixtures.json`.
- **Rebuilding.** `REF_DIR=tools/reference VEC_DIR=crates/ym-gpu/tests/cpu_data tools/reference/check.sh` rebuilds each reference, runs its self-test under ASan/UBSan, regenerates the vectors and compares them. Use it, not hand edits, if a vector must change.
- **Compiler flags.** The references are compiled with `-ffp-contract=off` so GCC and Clang agree. The Rust ports should avoid `mul_add` where the specs say an operation order matters.

## 10. As built: the R6 default and an unusable noise model (2026-10-13)

- **The default.** `ym_gpu::cpu::demosaic::DEFAULT_METHOD` is `Method::AdaptiveRatio` (owner's decision after the blind look RA-H4; `docs/05` section R6, `docs/16`). `RenderOptions::demosaic` (the test-only switch) starts there, `FullChain::new` uses it, and Menon, LMMSE, bilinear and log-guided stay selectable through the switch. The fit view and the thumbnails bin the mosaic and do not demosaic, so only the 100 % view and exports pay the extra CPU time (measured in `docs/05` section R6). The tile scheduler's `Demosaic(AdaptiveRatio)` halo of 13 px applies to every default tile.
- **What a missing or unusable noise model does (decision).** The specs are silent on a render whose noise model is missing or invalid: spec 01 section 4 says the algorithm refuses it (`BadNoise`), section 7 says "R6 always passes the real model", and a zero model is valid input (K = T = 1 DN, near-pure ratio interpolation). With the adaptive ratio demosaic as the default a refusal would have failed the whole render (`RenderError::Image`), which a render must never do for a noise model. So: **a model that is missing or fails `NoiseDn::is_valid` is read as the zero model, `NoiseDn::ZERO`, for that photo.** `NoiseDn::or_zero` is that rule; `Chain::noise_dn` applies it to `Chain::noise` for the R6 demosaic and the R8 diffraction stage, and `RawParams::noise_dn` (`ym-core`) applies it to the model in the parameters for the chain, the diffraction plan and the choice of R8 method. One level above, the renderer (`Job::params` in `pipeline/graph.rs`) replaces a photo's `NoiseModel` by the zero model (`usable_noise_model`: `NoiseDn::is_valid` and finite black-level offsets) before it enters `RawParams`, so the sigma map, the denoise, R7, R8 and R6 all read the same usable model, and it logs one `warn` ("the noise model of the photo is not usable; rendering with zero noise", no path); `FullChain::with_method` logs the same for a `RawParams` made by hand. The replacement is needed for more than the demosaic: with a `NaN` gain the sigma map was `NaN` and the frame came out black, which the review of this change found (`crates/ym-core/tests/render_noise_model.rs`). A usable model is untouched, bit for bit. The estimator of `ym-noise` never produces an unusable model, so this is a guard against corrupt data, not a path a real photo takes. Known gap: the colour plan (`pipeline/color.rs`, the toe of C7) reads `estimates.noise.model` directly, so a finite but absurd read-noise variance (`b` of 1e300) still blackens the frame there (a `NaN` or negative number is harmless: it is floored at zero); it should read `usable_noise_model` too. The chain still refuses gains that are not positive and finite (a chain with such gains has no meaningful R4), and the infallible `demosaic_with` still falls back to Menon's bits for unusable gains or noise.
- **`NoiseDn::is_valid` is tighter.** Besides finite and non-negative `g` and `r`, the variance at white, `g * S + r * r`, and `g * g` (squared by the derived constants) must be finite. A model of `a = b = 1e300` has finite `g` and `r` but no finite variance over the working range and is invalid, so it is read as zero noise in a render and refused (`BadNoise`) by `adaptive_ratio::demosaic`.

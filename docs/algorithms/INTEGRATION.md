# Integration guide

> **In this repository (2026-10-13, owner's decision):** the C++ references (`reference/*.cpp`, `tools/check.sh`) are not committed. They were used as a development oracle outside the repository; the vectors in `crates/ym-gpu/tests/cpu_data/<algorithm>/` are the fixed oracle here and are never hand-edited. Where this text says `tools/reference/` or `check.sh`, read it as that external oracle.

How the four algorithms fit Yellowmoon as built on `claude/yellowmoon-v1` (commit f219537): its crates, types, units, noise model, tiling and export path. Each spec repeats the parts it needs; this file is the overview and the tie-breaker for conventions.

## 1. Where each algorithm goes

```
R1 normalise → R3 vignetting → R4 gains g0 → R5 highlights
→ R6 demosaic ─┬─ Bilinear | Menon | Lmmse                      (existing)
               ├─ AdaptiveRatio   (spec/01; export fallback, 1:1, Detail's Y_classical)
               └─ LogGuided       (spec/02; fit view, thumbnails)
  → R6 bilateral denoise → R7 detail
→ R8 capture sharpening ─┬─ RichardsonLucy                       (existing, default)
                         └─ Diffraction   (spec/03)
→ R9 lens resample → colour C1…C12
→ export: [Lanczos-3 resize] → OutputSharpen (spec/04, off by default) → C13 + dither → writer
```

| Algorithm | Rust home | Selection | Ships as |
|---|---|---|---|
| Adaptive ratio demosaic | `ym-gpu/src/cpu/demosaic/adaptive_ratio.rs`, `Method::AdaptiveRatio` | `Chain::method`; `DEFAULT_METHOD` changes only after the RA-05 gate | candidate default for R6 classical |
| Log-guided demosaic | `ym-gpu/src/cpu/demosaic/log_guided.rs`, `Method::LogGuided` | used by the fit-view and thumbnail renders if RA-06 meets its targets | preview path |
| Diffraction deconvolution | `ym-gpu/src/cpu/sharpen/diffraction.rs` | `CaptureSharpen::Diffraction`, internal switch | off until RA-H2 |
| Output sharpening | `ym-gpu/src/cpu/output_sharpen.rs`, called from `ym_export::request::encode_to_writer` | `ExportOptions::output_sharpen: Option<OutputSharpen>` | `None` until RA-H3 |

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

| Algorithm | Input | What R6 does |
|---|---|---|
| AdaptiveRatio | **unbalanced** mosaic plus `gains` | divide each photosite by `g0[colour]` (exact inverse of R4), call with `gains = g0`; output is balanced |
| LogGuided | balanced mosaic | call directly; output is balanced |
| Bilinear, Menon, Lmmse | balanced mosaic (as today) | unchanged |

Put the division inside `demosaic_with` for `AdaptiveRatio` so that `Chain`, the tile scheduler and R7's `Y_classical` all get it for free.

## 5. Halos (`ym_gpu::cpu::tiling`)

| Stage | Halo (px) | Source |
|---|---|---|
| `Demosaic(AdaptiveRatio)` | 13 (11 without shadow matching; `Method::halo()` returns 13) | spec/01 §1.5 |
| `Demosaic(LogGuided)` | 5 | spec/02 §1.4 |
| `Sharpen` (Diffraction) | `10 + R + r_G`, at most 25 (R = PSF kernel radius, r_G = high-pass Gaussian radius). Richardson–Lucy's 24 px cap (`SHARPEN_HALO_CAP_PX`) must not be applied to this method; give `Stage::Sharpen` a per-method halo | spec/03 |
| Output sharpening | 12, or 20 with halo suppression (method 2: 13 / 21; print mode up to 33 / 41) | spec/04 §1.4 |

The scheduler already aligns mosaic rectangles to even coordinates. `cpu_tiling.rs` iterates `Method::ALL`, so the new methods get tiling tests as soon as they are added there.

## 6. Lateral CA curves for spec/01 pass 2 (phase 2, RA-03b)

The CA-aware green refinement takes two radial curves, `shift_px` against `radius_px` from the optical centre, for red and blue. Derive them from the R9 lens data (Sony tag 0x7035, `ym_gpu::cpu::lens::LensGeometry`): with `K_c(r)` the per-channel radial scale factor, the curve is `shift_c(r) = r_src(r) · (K_c(r) − 1)`. spec/01 §3.6 gives the details, including the effect of distortion on `r_src`. Without CA tags, or with `decode.chromatic_aberration = false`, pass `ca = None`.

## 7. Inputs that Yellowmoon doesn't have yet

| Needed by | Input | Where to get it |
|---|---|---|
| spec/03 | pixel pitch (µm) | preferred: `sensor_width_mm` and `active_width_px` (or `pixel_pitch_um`) per body in `data/cameras/registry.json`, from published specifications with the source in the entry's provenance. Fallback: `crop = focal_length_35mm / focal_length_mm` from `RawMetadata`, `pitch_um ≈ 36000 / (crop · active_width_px)` for 3:2 sensors. If neither is available, diffraction deconvolution is disabled for the photo |
| spec/03 | f-number | `RawMetadata::f_number`. Disabled when it is missing or outside 1–50 |
| spec/04 | output scale `s` | `out_long_edge / rendered_long_edge` from the export request (1.0 for full-size exports) |

## 8. Settings, parity and determinism

- **No schema changes in this work.** New switches live in `RawParams` / `Chain` / `ExportOptions`, not in `settings.schema.json`, until Mykyta decides on UI (RA-H2, RA-H3). That keeps the serde-order tests and the app↔CLI parity untouched.
- **The R6 method switch (RA-04)** is `RenderOptions::demosaic` in `ym-core` (an internal option of the renderer, `DEFAULT_METHOD` unless a test sets it) with a test-only environment variable behind the `force-demosaic` cargo feature; see `01-adaptive-ratio-demosaic.md` section 10.
- **Determinism.** All four algorithms are pure functions of their inputs and per-frame constants, with no reductions that depend on thread order. Keep f64 where the specs say so. `--jobs 1` and `--jobs 4` must stay byte-identical.
- **Goldens.** Hashes change only when a default changes (RA-05). Use `golden update --note "…"` with the gate numbers in the note.
- **GPU.** Every WGSL port needs a CPU reference and a parity test (`AGENTS.md`). Each spec's §5 lists which parts are per-pixel and which need care: decision near-ties in spec/01 and the window counts of the halo limiter in spec/04.

## 9. Vectors and the C++ references

- **What a vector holds.** Each vector JSON has `{name, description, params, input, expected, tolerance}`. Rust tests load them from `crates/ym-gpu/tests/cpu_data/<algorithm>/` and compare with the stated tolerance. They follow the pattern of the existing `menon_fixtures.json`.
- **Rebuilding.** `REF_DIR=tools/reference VEC_DIR=crates/ym-gpu/tests/cpu_data tools/reference/check.sh` rebuilds each reference, runs its self-test under ASan/UBSan, regenerates the vectors and compares them. Use it, not hand edits, if a vector must change.
- **Compiler flags.** The references are compiled with `-ffp-contract=off` so GCC and Clang agree. The Rust ports should avoid `mul_add` where the specs say an operation order matters.

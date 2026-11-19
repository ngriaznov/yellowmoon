# 03 · Diffraction deconvolution (capture sharpening)

A second R8 capture-sharpening method beside Richardson–Lucy. It computes the diffraction blur of the lens from the f-number and the pixel pitch, and removes it from luma in one noise-weighted local least-squares step. It is selectable through an internal switch and is not the default (section 1.2).

| | |
|---|---|
| Code | `crates/ym-gpu/src/cpu/sharpen/diffraction.rs` (planner and stage, CPU reference); `shaders/diffraction.wgsl` and `kernels/diffraction.rs` (device) |
| Oracle | an independent C++ implementation outside the repository |
| Vectors | `crates/ym-gpu/tests/cpu_data/diffraction_deconvolution/*.json` (23 files) |

## 1. Purpose and placement

### 1.1 What the stage does

Stopping a lens down blurs the image by diffraction. That blur depends only on the f-number, the wavelength and the pixel pitch, so it can be computed rather than guessed. The stage:

1. **Plans the PSF once per frame.** It builds a polychromatic Airy-pattern PSF from the f-number
   `N` and the pixel pitch `p`, summed over 461–619 nm with a fixed spectral weight, truncates and
   tapers it, splats it onto the pixel grid and cuts it to a `(2R+1)²` kernel `K` with `R ≤ 7`. It
   also derives the width `σ_G` of a Gaussian that approximates `K`, used for a high-pass.
2. **Corrects luma in one step.** Luma `Y` is split into a dark layer `D` and a bright layer `B`
   (`D + B = Y`, built from local lower and upper envelopes), plus a Gaussian high-pass `H(Y)`. For
   every pixel, three coefficients are fitted by weighted least squares over a 9×9 window. The
   weights are the inverse noise variance. The fit makes the PSF-reblurred correction explain the
   residual `Y − K⊗Y`. The coefficients are constrained, noise-gated and averaged over 5×5, then
   applied to the unblurred layers.
3. **Carries the new luma into RGB.** A capped ratio (gain at most 1.5) plus an additive remainder,
   so the output luma equals the corrected luma.

The model: take `X = Y + cD·D + cB·B + cH·H(Y)` and require `K⊗X ≈ Y`. Because `K` and the Gaussian
commute, this is `cD·(K⊗D) + cB·(K⊗B) + cH·H(K⊗Y) ≈ Y − K⊗Y`, which is the system solved per pixel
(§3.8–3.9). Edges get sharper because the dark layer is pushed down (`cD < 0`) and the bright
layer up (`cB > 0`, with `cD ≤ cB` enforced). Fine texture goes through `cH ∈ [0, 1]`. Where the
residual is at the noise level, the pixel is left alone. There is no iteration.

### 1.2 Place in the Yellowmoon chain

- **Stage R8**, after R7 (colour guided filter and detail re-injection) and before R9 (lens
  resample). It runs wherever Richardson–Lucy runs: the full-resolution chain, and
  camera-processed Lossless M/S files after R7. Like Richardson–Lucy, it does not run on the binned fit view or thumbnails, and it does not run for a DNG (`docs/05-raw-processing.md` R8).
- **Selection.** R8 is `CaptureSharpen::{RichardsonLucy(SharpenParams), Diffraction(DiffractionParams)}`
  in `Chain` and `Stage::Sharpen`. Exactly one method runs (either/or, §2.5). Richardson–Lucy is the
  default (`DEFAULT_CAPTURE_METHOD`). Diffraction is reached only through the internal
  `PhotoSource::capture_method`, which tests and evaluation tools set: there is no settings field,
  no UI control and no CLI flag, and default renders do not change.
- **Fallback.** If Diffraction is selected but the stage is inactive (§3.1: no f-number, an
  f-number outside [1, 50], no usable pixel pitch), R8 runs Richardson–Lucy with its resolved
  `SharpenParams`. A `Chain` that is given an inactive Diffraction stage directly leaves the image unchanged.
- **Code.** `cpu/sharpen.rs` (`CaptureSharpen`, Richardson–Lucy), `cpu/sharpen/diffraction.rs`
  (planner, stage, `DiffractionParams`, `DiffractionPlan`), `cpu/sharpen/diffraction/eval.rs` (the
  edge, MTF and noise measures of the A/B; not used by the render path), the per-method halo in
  `cpu/tiling.rs` and the Diffraction arm of `Chain::run_tile`. In `ym-core`:
  `pipeline::raw::{pixel_pitch, diffraction_params, CaptureMethod, DEFAULT_CAPTURE_METHOD,
  RawParams::{diffraction, capture_sharpen}}` and `PhotoSource::{sensor, capture_method}`.
- **Device.** The device chain runs the stage as R8 when it is selected (§10).

### 1.3 Input and output contract

| | |
|---|---|
| Input | `RgbImage`, planar f32 (all R, then all G, then all B): camera RGB balanced with `g0` (R4), as R7 leaves it. Normalised: white = 1.0 before the gains, so a clipped photosite sits at `g0_c` (times the R3 vignetting gain). Values above 1.0 (R5) and slight negatives are allowed. Samples must be finite. |
| Output | Same size, layout, units and `Placement`. Balanced. Not clamped (negatives and values above 1 pass through). |
| Balanced or not | Balanced in, balanced out. The gains `g0` enter only the noise weight (§2.3) and the clip test (§3.4), which divides by `g0` to recover the unbalanced value. |
| What changes | Only luma. The transfer (§3.12) keeps the R:G:B ratios wherever its gain is at most 1.5 and luma is above 1 DN. |
| Position | The PSF is space-invariant: nothing depends on `Placement` or on the CFA. |

### 1.4 Working scale

The constants are tuned for a 16-bit scale. The stage works on `S = 65535`: `x_dn = S·x` on entry.
On exit it writes `out = in·k + off/S` (§3.12), so no round trip through DN is needed and identities
stay exact.

| Quantity | On the working scale | In normalised units |
|---|---|---|
| Luma noise floor `σ0` (and its minimum) | DN, at least 1 DN | `σ0/S`, at least `1/S` |
| Luma shot-noise slope `σ1` | DN | `σ1/S` |
| Inverse variance `w` | 1/DN² | `S²·w` |
| Clip threshold `T` | `0.8·S` (unbalanced) | 0.8 (unbalanced) |
| Transfer floors | 1 DN | `1/S` |

Everything else is scale-free: the moments `S_ij`, `T_i`, `Q` and the penalty mean `P̄` are all
weighted by `w`, so they are dimensionless. The gate constants, the ridge, the coefficient bounds,
the envelope constants and the ratio cap do not depend on `S`. An implementation that works in
normalised units must convert the five rows above; the result is the same up to rounding. The
CPU implementation works in DN.

### 1.5 Halo

Each output pixel depends on the input within `halo = 10 + R + r_G` pixels:

| Step | Window | Radius |
|---|---|---|
| Envelope estimate (§3.5) | 5×5 | 2 |
| Envelope means (§3.6) | 5×5 | 2 |
| PSF reblur of the layers (§3.7) | (2R+1)² | `R` |
| Gaussian high-pass of the reblurred luma (§3.7) | 2·r_G + 1 | `r_G` |
| Weighted moments (§3.8) | 9×9 | 4 |
| Coefficient smoothing (§3.10) | 5×5 | 2 |
| **Total** | | **10 + R + r_G** |

The longest chain is output ← smoothing ← moments ← `F3 = Z − G(Z)` ← `Z = K⊗(D + B)` ←
layers ← envelope means ← envelopes ← `Y`. The other chains are shorter: `H(Y)` needs `r_G`, and
the layers at the output pixel need 4.

- `R ≤ 7`. `σ_G ≤ 3` gives `floor(3.5·3 + 1) = 11`, capped at `r_G ≤ 8`. So **halo ≤ 25 px**.
- A sweep over pitches 0.8–15 µm, f/1 to f/50 and scales 1 to 1/8 gave halos from 12 to 24. The
  maximum, 24, is R 7 with r_G 7 (small pitches at small apertures, from 0.8 µm at f/14 to 2.8 µm
  at f/50). Typical values at a 3.76 µm pitch: 13 at f/2.8, 15 at f/8, 17 at f/11, 18 at f/16, 19 at
  f/22, 22 at f/50.
- The scheduler uses the per-frame value from the plan (`DiffractionParams::halo_px()`), at most
  `MAX_HALO_PX` = 25. Richardson–Lucy's 24 px cap (`SHARPEN_HALO_CAP_PX`) does **not** apply: at the
  exact halo tiles equal the whole frame bit for bit.

## 2. Parameters

### 2.1 Per-frame inputs

| Name | Type | Default | Range | Meaning |
|---|---|---|---|---|
| `pixel_pitch_x_um`, `pixel_pitch_y_um` | f64 | none | > 0 (sanity 0.5–20, §2.2) | Sensor pixel pitch in µm. The larger of the two is used. |
| `f_number` | f64 | none | [1, 50] | Lens f-number, from `RawMetadata::f_number`. |
| `scale` | f64 | 1.0 | (0, 1] | Image pixels per sensor pixel. 1 for every mosaic file. A value outside the range reads as 1. |
| `g0` | [f32; 3] | none | each > 0, finite | The R4 gains (`RawParams::g0`). |
| `luma` | [f32; 3] | [0.299, 0.587, 0.114] | sum 1 | Luma weights on balanced linear camera RGB. |
| `noise` | `NoiseModel` | none | | `RawParams::noise`, converted per §2.3. |
| `strength` | f32 | 1.0 | [0, 3] | Blend between input and corrected luma (§2.5). Clamped; NaN reads as 1. |

Rust types (`crates/ym-gpu/src/cpu/sharpen/diffraction.rs`):

```rust
/// R8 diffraction deconvolution controls (docs/algorithms/03).
pub struct DiffractionParams {
    /// Sensor pixel pitch (x, y), micrometres.
    pub pixel_pitch_um: [f64; 2],
    /// Lens f-number.
    pub f_number: f64,
    /// Image pixels per sensor pixel, (0, 1].
    pub scale: f64,
    /// Blend between input and corrected luma, 0..=3. 1 is the base behaviour.
    pub strength: f32,
}

/// Everything derived once per frame: kernel, sigma_G, radii, noise constants.
pub struct DiffractionPlan { /* see §2.7 */ }

impl DiffractionPlan {
    // Signatures only.
    /// `None` when the stage is inactive (§3.1).
    pub fn new(p: &DiffractionParams, g0: [f32; 3], noise: &NoiseDn, luma: [f32; 3]) -> Option<Self>;
    pub fn halo_px(&self) -> usize;
}
```

`NoiseDn { g, r }` is the shared working-scale noise type of `INTEGRATION.md` (§2.3).

### 2.2 Pixel pitch

The pixel pitch comes from, in this order:

1. **Camera registry (preferred).** Each entry of `bodies` in `data/cameras/registry.json` can carry
   `sensor_width_mm` and `active_width_px` (27 of its 28 bodies do; `SensorGeometry`). Both describe the
   full sensor, from the maker's published specification (sensor size and effective pixel count), with
   the source recorded in the entry (`docs/03-camera-profiles.md`). Then
   `pitch_x = pitch_y = 1000·sensor_width_mm / active_width_px` µm. An optional
   `pixel_pitch_um` field overrides this when the maker states the pitch. A camera crop mode
   (APS-C mode on a full-frame body) does not change the pitch. Example: a 35.7 mm wide sensor with
   9504 active columns has a 3.76 µm pitch.
2. **EXIF crop factor (fallback).** With `crop = focal_length_35mm / focal_length_mm` from
   `RawMetadata`, and `W × H` the frame as R8 receives it (active area in pixels):
   - for 3:2 frames, `pitch ≈ 36000 / (crop · W)` µm;
   - for any aspect ratio, `pitch ≈ 43266.6 / (crop · √(W² + H²))` µm (the 35 mm diagonal is
     43.2666 mm; this equals the 3:2 form for 3:2 frames).

   Both pitches are taken equal. `focal_length_35mm` is an integer, so the pitch carries an error of up
   to about 2 % at short focal lengths, which is small next to the kernel's radius steps.
3. **Neither available**, or the result is outside the sanity range 0.5–20 µm: the stage is
   inactive and R8 falls back to Richardson–Lucy.

**Camera-processed reduced files (Lossless M/S).** The camera has downscaled the image. With the
registry, pass the full-sensor pitch and `scale = W / active_width_px`. With the EXIF fallback, the
formula already gives the pitch of the delivered image: pass it with `scale = 1`.

**f-number.** `RawMetadata::f_number`. Missing, zero (manual lenses often write 0), NaN or outside
[1, 50]: inactive.

### 2.3 Noise weight

Yellowmoon's model per photosite, normalised and unbalanced, is
`var(μ) = a·(μ − δ_c) + b + σ_row² + q²/12`. On the working scale (integration contract):

    σ²_dn(u) = g·u + r²,   g = S·a,   r = S·√(b + σ_row² + q²/12)

This is the shared `NoiseDn { g, r }` of `INTEGRATION.md`. The per-plane offsets `δ_c` are not used.

The balanced channel values are `x_c = g0_c·u_c`, so luma is `Y = Σ a_c·x_c = Σ (a_c·g0_c)·u_c`.
The stage treats the three channel values of a pixel as one raw sample `u` scaled by

    d = a_R·g0_R + a_G·g0_G + a_B·g0_B

which is the fully correlated limit. After demosaicing and R7, the noise of the three channels at
one pixel is strongly correlated. With `Y = d·u`:

    var(Y) = d²·(g·u + r²) = (d·r)² + (d·g)·Y

So, with `s = min(2·scale, 1)` (`s = 1` at full resolution):

    σ0 = max(d · r · √s, 1 DN)
    σ1 = d · g · s
    w  = 1 / (σ0² + σ1 · max(Y, 0))          per pixel, Y in DN

Example: `g0 = (2, 1, 1.5)` and the default weights give `d = 1.356`. With `a = 2.5e-5`,
`b = 4e-8`, `σ_row = 5e-5`, `q = 0`, that is `σ0 = 18.32` DN and `σ1 = 2.222` DN (vector
`plan_p3.76_f8`).

The R3 vignetting gain multiplies signal and noise in the corners. This weight ignores it (§7.2).

### 2.4 White level and clip penalty

In Yellowmoon's normalised data, white is 1.0 before the gains, so on the working scale it is
`W = S = 65535` in unbalanced units. The clip threshold is `T = 0.8·W = 52428` DN, applied to the
unbalanced value `x_c / g0_c`. In normalised terms: a channel counts as near clipping when
`x_c / g0_c > 0.8`. The penalty (§3.4) raises the gate threshold, so clipped highlights and the
values R5 reconstructed are not sharpened into artefacts.

### 2.5 Strength, the Sharpening Amount, and Richardson–Lucy

- **Strength.** The algorithm has no amount control of its own. `strength = 1` corrects by the fitted
  amount, which is 100 %. `L_s = Y + strength·(L − Y)`, where `L` is the corrected luma (§3.11).
  `strength = 0` returns the input exactly. Values above 1 extrapolate linearly, and overshoot
  grows in proportion. Yellowmoon runs `strength = 1`.
- **Sharpening Amount.** The Amount and Radius controls of Richardson–Lucy do not act on this method:
  the PSF comes from the optics, and the noise gate already adapts to the noise. There is no mapping
  from Amount to `strength`.
- **Composition with Richardson–Lucy: either/or.** One R8 method runs per frame; the two are not
  stacked (§7.1 item 8).

### 2.6 Fixed constants

| Value | Where | Role |
|---|---|---|
| 65535 | everywhere | working scale `S` |
| 461…619 nm, step 1 nm | §3.2 | wavelengths (159 of them) |
| `w(λ) = 1` for λ ≥ 540; `(620 − λ)/80` for λ ≤ 539 | §3.2 | spectral weight (1.9875 at 461 nm, 1.0125 at 539 nm) |
| 0.42 µm, 1/4 | §3.1 | radial step: a quarter of `min(p_eff, 0.42·N/π)` |
| 500 | §3.2 | radial samples, index 0…499 |
| 1/6 | §3.2 | weight of sample 0 in the encircled energy |
| 0.8, 0.01 | §3.2 | encircled-energy fraction; floor of the interpolation step (as a fraction of the total) |
| 4, 6 | §3.2 | support = min(4·r80, 6 pixels) |
| 6, 0.999 | §3.2 | angular samples per quadrant of a ring of radius `ρ_k` px: `floor(6·ρ_k + 0.999)` |
| 8 | §3.2 | quadrant grid size (radius up to 7) |
| 0.99, 0.002 | §3.2 | radius rule: energy fraction, ring peak relative to the kernel peak |
| π/16 × 8 angles, 3 bins per pixel, `floor((17R + 16)/4)` | §3.3 | line-spread sampling and centre bin |
| 0.85, 0.8 | §3.3 | fit range and target of the line-spread energy |
| 0.75, [0.2, 3.0] | §3.3 | σ_G factor and clamp |
| 3.5, 1, 8 | §3.3 | Gaussian radius `min(floor(3.5·σ_G + 1), 8)` |
| 1 DN | §2.3, §3.12 | noise floor; transfer floors |
| 0.8 | §3.4 | clip threshold, fraction of white |
| 25, 5, 1 | §3.4 | clip penalty weights of the smallest, middle and largest excess |
| 5×5 | §3.5, §3.6, §3.10 | envelope window, envelope means, coefficient smoothing |
| 1e-20 | §3.5 | divide guard |
| 12.5, 0.04 | §3.5 | envelope pull-in |
| 9×9 | §3.8 | moment window |
| 1e-15 | §3.9 | floor of `S11`, `S22`, `S33` |
| 0.0005 | §3.9 | ridge, as a fraction of `S11 + S22 + S33` |
| [0, 1) | §3.9 | range of `cH` |
| −0.5 | §3.9 | lower bound of `cD` (and of `cB` when `cD` is clamped) |
| 0.5 | §3.9 | residual gate |
| 2 (= 1/0.5) | §3.9 | residual level `kE` for the fade-out and the confidence weight |
| 1/4, 1/2 | §3.9 | fade-out band of the weighted residual |
| 0.01 | §3.9 | minimum confidence weight |
| 1.5 | §3.12 | ratio cap of the transfer |

#### Derivation of every constant

Every constant of this spec has a row, with the ID of `docs/29-raw-constants.md` §2.1 and one status: **Maths** (a mathematical fact or an error budget gives the value, and a test proves it), **Fit** (only image data can choose it; the task that fits it), **Design** (the owner's look, `docs/29` §6) or **Guard** (a numeric safeguard; what it protects). No row names an origin: each stands on the method written in it. The tests are in `crates/ym-gpu/src/cpu/sharpen/diffraction.rs`, module `derivations`. The convergence runs of D2 to D4 and D6 and the fits of D1 and D5 to D12 are RC-11's (2026-11-18, `docs/29` §5); how they were decided is under "What RC-11 decided" below.

**Where each constant lives (`docs/29` RC-03).** In the code every constant of this table that is a number is a field of `diffraction::Tuning` (`crates/ym-gpu/src/cpu/sharpen/diffraction.rs`), documented with its row, kind, section, status and task. `Tuning::ORACLE` holds the values the oracle vectors were made with and `Tuning::DEFAULT` the shipped ones; they are equal: RC-11 kept every value. The planner reads D1 to D6 once per `(pitch, f-number)`; the device reads D7 to D13 from a WGSL `const` block generated from the `Tuning` and put in front of `shaders/diffraction.wgsl`, so there is one copy of each. The windows of D14, the 8 × 8 grid and the 500 radial samples are structure and stay in the code. `diffraction::REGISTRY` lists id, name, value, kind and status, and `crates/ym-gpu/tests/constants_registry.rs` fails for a copy outside the `Tuning`, for a generated block that differs from the registry, and for a row missing from `docs/29` §2.1 or the other way round.

**Where the constants come from (`docs/29` RC-14).** The values were taken over from research (the owner's answer to question 12 of `docs/STATUS.md`, 2026-11-13: "from research, but you need to readjust"). The project has re-derived every one of them: each is proved by mathematics, fitted on public test images and our own synthetic scenes (no measurement of our own, no real-sensor data), kept with the numbers that show it sits at its optimum or on a trade-off curve, or marked as the owner's look. The derivation of each row is in the table and in `docs/29`; the value that ships is `Tuning::DEFAULT`. No row cites a source for its number, and none needs one. A constant that is not in this table is not added; it is raised as a question first.

RC-11 ran every fit of this spec and **changed no value**: the default stays equal to the oracle. The one fit that is better on the means and worse on a minority of cells (the σ_G factor `0.78`) is not taken.

| ID | Constant | Status | Derivation | Proof or task |
|---|---|---|---|---|
| D1 | 461 to 619 nm, weight 1 from 540 nm, `(620 − λ)/80` below (§3.2) | Fit | The spectral weight of the polychromatic PSF. Compared with the luma response of the eleven published Sony curves (D55, balanced channels, the stage's luma): the PSF moves by 0.043 in L1, and the gate is worse in 153 of 168 numbers with them, so the ramp stays. | RC-11, `tools/fit/d1_spectral_weight.py`; `the_weight_table_replaces_the_ramp_and_a_table_of_the_ramp_changes_nothing` |
| D2 | `0.42` µm, 1/4, 500 samples (§3.1, §3.2) | Maths | Radial step `min(p_eff, 0.42·N/π)/4`, an error budget against a step of one eighth: the kernel is 5.2e-3 away in L1 (budget 1e-3), half the step 1.3e-3, twice the step 2.2e-2. Over the budget, and the finer steps are mixed on the gate, so it stays. | `d2_the_radial_step_misses_the_budget_by_a_factor_of_five_and_an_eighth_step_is_not_better_on_the_gate` |
| D3 | `1/6` (§3.2) | Maths | Weight of sample 0 in the encircled energy. The integral of `r·p(r)` over a profile that is linear between its samples weighs sample `k > 0` by exactly `k` and sample 0 by `1/2 − 1/3 = 1/6`; the disc-area value 1/8 is not exact. | `one_sixth_is_the_exact_weight_of_sample_zero_in_the_encircled_energy`, `the_planner_uses_that_weight_to_find_the_80_percent_radius` |
| D3 | `0.8`, `0.01`, `4·r80`, `6` px, `0.99`, `0.002`, 6 angular samples per px (`+ 0.999`), the 8 × 8 grid, the cos² taper (§3.2) | Maths | Where the kernel is cut. The radius rule (`0.99`, `0.002`) loses 4.7e-4 of the kernel against no cut (R = 7) and the floor `0.01` never binds: both inside the budget. The splat density 6 is 1.9e-2 from 24 (over); the support (`0.8`, `4·r80`, `6` px, the taper) is a window, not a quantity that converges (8·r80 and 8 px move the kernel by 0.22). None of the converged values is better on every gate number, so all stay. | `d3_the_radius_rule_and_the_floor_meet_the_budget_and_the_support_and_the_splat_do_not` |
| D4 | `π/16` × 8 angles, 3 bins per px, `0.85`, `0.8` (§3.3) | Maths | How the line spread of the kernel is measured, tested on Gaussians of known width: the estimate is 0.921 of the true `x80 = 1.2816σ` for σ from 1.25 to 2 (0.89 to 0.97 below), so the factor that returns σ is 0.847, not 0.780. Over the budget; D5 absorbs it, and the cutoff and level are mixed on the gate (a level of 0.82 or a cutoff of 0.9 raise the mean numbers and worsen 77 and 71 of 168). | `d4_the_line_spread_estimate_reads_8_percent_below_the_true_half_width_of_a_gaussian` |
| D5 | `0.75`, `[0.2, 3.0]` (§3.3) | Fit | `σ_G = f·mean(x80)`. For a Gaussian line spread `x80 = 1.2816σ`, so the exact factor of the true half-width is `1/1.2816 = 0.7803`; the estimator of D4 reads 8 % low, so for it the factor that returns the kernel's own σ is 0.847. **RC-11 kept 0.75**: 0.76 to 0.9 raise the mean PSNR, MTF50 and the Kodak PSNR and lower the mean overshoot, but worsen 77 of 168 numbers (§6.4 and "What RC-11 decided"). The clamp is never reached: σ_G is at most 1.87 for every pitch from 0.5 to 20 µm and f/1 to f/50. | `the_exact_sigma_g_factor_for_a_gaussian_is_0_780_and_the_shipped_one_is_0_75`; RC-11 |
| D6 | `3.5`, `1`, `8` (§3.3) | Maths | Radius `min(floor(3.5σ_G + 1), 8)` over pixel-integrated taps cuts at `radius + 1/2 > 3.5σ_G`, so the mass beyond is under `erfc(3.5/√2) = 4.65e-4` for every `σ_G` up to 2.0, and under the budget of 1e-3 up to 2.58 (with the cap of 8). σ_G is at most 1.87 for every valid pitch and f-number, so the budget holds everywhere the planner goes. Radii from `3.25σ` to `3.75σ` change nothing; 3.0 and 4.0 are mixed by under 0.0005 dB. | `the_gaussian_radius_cuts_less_than_the_tail_budget_up_to_sigma_two`, `d6_the_gaussian_radius_meets_the_one_per_mille_budget_up_to_sigma_2_58` |
| D7 | `0.8` of white, `25`, `5`, `1` (§3.4) | Fit | Clip penalty: where the highlight starts and the weights of the sorted channel excesses. **Kept.** The threshold is mixed (0.82 and 0.9 raise the Kodak PSNR by 0.013 and 0.056 dB and the lamps by 0.005 and 0.53 dB, and worsen 17 and 25 gate numbers; 0.5 to 0.78 lose up to 0.99 dB of gate text PSNR and 0.56 dB of Kodak); the weights 5 to 60 and 1 to 15 move no number by more than 0.0005 dB, even on the lamps scene where three channels clip. | RC-11 |
| D8 | `12.5`, `0.04` (§3.5) | Fit | Pull-in of the local envelope estimates. **Kept.** The gain is mixed on every value (3 to 40: 46 to 119 gate numbers worse, 13 and 14 each 50); the band from 0.045 up worsens nothing and moves two numbers by under 1e-4 (rule 3). | RC-11 |
| D9 | `0.0005` (§3.9) | Fit | Ridge of the 3 × 3 local solve. **Kept.** Mixed on every value from 1e-5 to 5e-3 (0.0004 and below lower the mean overshoot and noise and worsen 87 to 93 numbers; 0.0006 and above raise them and worsen 79 to 93). | RC-11 |
| D10 | `0.5`, `2`, `1/4`, `1/2` (§3.9) | Fit | The noise gate and the fade-out band of the residual. **Kept.** The gate (0.2 to 1.0) trades overshoot and flat noise against the PSNR of the natural scene at every value (0.4: overshoot −0.11, noise −0.08, natural PSNR −0.04 dB, 96 numbers worse); the band edges are mixed (the `1/4` and `1/2` from 0.28 and 0.55 up each leave 19 numbers worse for 19 better). | RC-11 |
| D11 | `0.01`, `−0.5`, `[0, 1)` (§3.9) | Fit | Smallest confidence weight; lower bound of `cD`; range of `cH`. **Kept.** The weight floor 0.001 to 0.0125 moves no number by more than 0.003 dB, 0.02 to 0.1 is mixed (0.1: text PSNR +0.043 dB, overshoot −0.08 points, 28 numbers worse); `cD` from −3 to −0.6 moves 34 gate numbers by under 0.001 (mixed), −0.4 to 0 is mixed (0: Kodak −1.1 dB). `cH` is a range, not a value. | RC-11 |
| D12 | `1.5` (§3.12) | Fit | Cap of the luma-to-RGB ratio. **Kept.** From 1.6 to 5 no number moves; 1.4 improves four numbers by under 1e-6; below it mixed. | RC-11 |
| D13 | `1e-20`, `1e-15`, `1 DN` (§2.3, §3.5, §3.9, §3.12) | Guard | Divide guards and floors: they keep the divisions and the solve finite on black and flat input and never bind on a photograph. | n/a |
| D14 | windows 5 × 5 and 9 × 9 (§3.5, §3.8) | Fit | The envelope, moment and smoothing windows; structure, not a number to tune. Kept. | RC-11 |

Counts: Maths 4 (D2, D3, D4, D6), Fit 9 (D1, D5, D7 to D12, D14), Design 0, Guard 1 (D13).

#### What RC-11 decided (2026-11-18)

**Nothing in `Tuning::DEFAULT` changed.** R8 does not use this stage by default, so no picture moves. The rules are those of `docs/29` §2.3 (rule 3: no change for under 1 % of the objective; rule 4 and the owner's answer 1: a value that is better on one number and worse on another is not taken), with this spec's §6.4 table as the floor.

*The evaluation.* Four sets, each with the shipped constants as the base: (1) the §6.4 gate itself (`capture_sharpen_ab --from oracle`: five f-numbers, three ISO, the plain stage and the noise-gate variant, 168 numbers); (2) the eight cases of `fit_diffraction --held` (f/8 and f/16 at ISO 100 and 800, a second noise seed held out; the scenes are blurred with the reference optics); (3) 24 centre crops of 160 × 112 of the Kodak images in linear light, blurred with the same optics, PSNR against the sharp crop (evaluation only, not committed, `YM_TESTDATA`); (4) a lamps scene of ours (two discs of 6 and 10 px at 1.6 times white, three channels past the clip threshold in the larger one, clipped after the blur). `tools/fit/rc11_scan.py NAME v1,v2,...` runs one constant over a grid and prints, per value, the number of gate cells better and worse, the same for the cases, the Kodak and lamps change in dB and the mean change of each gate number. The objectives are taken in the order of §6.3: MTF50 against Richardson–Lucy Auto at f/8 and above, then overshoot, then flat-patch noise (at most ×1.10), then PSNR. Coordinate descent over D5 and D7 to D12 in the order of the sensitivity table of `docs/29` §2.2 found no move: no value of any grid (about 150 evaluations, 4 s each) is better on every number and by more than rule 3; the first sweep accepts nothing, so it is also the last.

*D1.* `tools/fit/d1_spectral_weight.py` takes the eleven curves of `data/cameras/sony`, multiplies each channel by CIE D55 (D50 and equal energy give the same result), balances the channels to the same total (the stage works on balanced RGB), combines them with 0.299, 0.587, 0.114 and normalises to a mean of 1 from 540 nm. The weights are not a ramp: 0.69 at 461 nm, 1.18 at 520, 1.13 at 540, 0.86 at 560, 1.07 at 580, 1.14 at 600, 0.71 at 619 (today's ramp: 1.99, 1.25, 1.0, 1.0, 1.0, 1.0, 1.0); the bodies differ from their mean by 0.006 in PSF L1 at most. The two PSFs differ by **0.043** in L1 (0.027 without balancing), which is above the evaluation's resolution (a change of ±10 % of the ramp width moves the PSF by 0.0075 and no gate number; the stated budget is 1e-3). So the first condition of the rule is met, the second is not: with the derived weights (mean, each body, three illuminants, balanced and raw, smoothed over 21 nm: 19 sets) the §6.4 gate is worse in 146 to 153 of 168 numbers (mean text PSNR −0.04 to −0.13 dB, MTF50 −0.001 to −0.003). They put more weight on the middle of the band than the ramp and less on 461 nm, which makes the PSF wider, and the stage's other constants were shaped on the ramp's PSF. **The ramp stays; the published curves do not agree with it to within the budget, and the derivation is that they were tried and made the gate worse.** The field `weight_table` of `Tuning` (off in both tunings) holds a measured table; the evaluation is repeatable with `--weights`.

*D2 to D4 and D6 (the budget).* **1e-3 of the energy of the kernel the stage uses**, as an L1 distance between the normalised kernel and that of a converged reference (for the Gaussian, between the Gaussian of the estimated width and of the true width). Thirteen lenses (3.76 µm at f/5.6 to f/45, 5.94, 2.4, 7.52 and 1 µm pitches); results in the table above. The radius rule, the floor of the interpolation step and the Gaussian radius meet it (D6 to σ_G = 2.58; σ_G never exceeds 1.87, so it always holds, and RC-02's 4.65e-4 holds to 2.0). The radial step (5.2e-3), the splat density (1.9e-2), the support (a window, 0.22) and the line-spread estimate (8 % low) do not. For each, the converged value was run through the gate (`step_fraction` 0.125 and 0.0625: 27 and 33 numbers worse, the mean PSNR +0.008 and +0.027 dB; `angular_per_px` 12 and 24: 152 and 151 numbers worse; `support_pitch_factor` 8, `support_r80_factor` 6 and 8, `r80_energy` 0.7 to 0.9: 56 to 74 numbers worse or 14 worse for 68 better) and none is better on every number, so none is taken (answer 1).

*D5, the σ_G factor: 0.75 or 0.78.* Both, on the §6.4 cells (text PSNR dB / natural PSNR dB / MTF50 / overshoot % / undershoot % / flat noise at 0.75, then at 0.78): f/8 ISO 100 30.68 / 35.80 / 0.340 / 1.65 / 1.63 / ×1.005, then 30.79 / 35.95 / 0.346 / 1.84 / 1.73 / ×1.005; f/8 ISO 800 29.84 / 34.14 / 0.354 / 2.38 / 1.99 / ×1.007, then 29.90 / 34.17 / 0.360 / 2.27 / 2.10 / ×1.007; f/8 ISO 6400 24.58 / 27.71 / 0.314 / 4.75 / 2.30 / ×1.011, then 24.53 / 27.69 / 0.371 / 4.73 / 2.40 / ×1.012; f/16 ISO 100 26.08 / 31.89 / 0.245 / 3.00 / 2.11 / ×1.577, then 26.14 / 31.92 / 0.246 / 3.11 / 2.28 / ×1.577. On the means over the 168 numbers (165 move): text PSNR +0.038 dB, natural +0.024 dB, MTF50 +0.0047, overshoot −0.041 points, flat noise −0.0008, and on the other sets Kodak +0.089 dB, lamps +0.0003 dB (the clip does not meet this constant), held-out cases 18 better and 6 worse. But 77 of the 165 are worse (overshoot 15, undershoot 23, flat noise 22, ISO 6400 PSNR 13, MTF50 and rise 2 each). 0.78 is therefore **mixed and not taken**; 0.75 stays. The §6.3 bars (MTF50 at least that of Richardson–Lucy Auto, overshoot no higher, flat noise at most ×1.10, over the 12 cells from f/8) are met by 9, 5 and 3 cells and all three by 1 cell with either factor; the noise-gate variant meets them in 0 of 9 cells from f/11 with either. The exact factor for a Gaussian line spread (0.780) is not the factor of this estimator (0.847), which is why the maths does not decide it.

*D7 to D12.* See the rows above. Every value of every grid either worsens a gate number or moves nothing by more than rule 3's 1 %. (Where the grid shows a value that worsens no number, it moves two to nine numbers by 1e-4 or less.)


### 2.7 Derived values (the plan)

| Symbol | Formula | Section |
|---|---|---|
| `p_eff` | `max(pitch_x, pitch_y) / scale` | §3.1 |
| `Δ` | `min(p_eff/4, 0.42·N/(4π))` µm | §3.1 |
| `r80` | 80 % encircled-energy radius, samples | §3.2 |
| `ρ` | `min(4·r80·Δ, 6·p_eff)` µm, support | §3.2 |
| `K`, `R` | kernel and radius, `1 ≤ R ≤ 7` | §3.2 |
| `σ_G`, `r_G` | high-pass width and radius | §3.3 |
| `halo` | `10 + R + r_G` | §1.5 |
| `d`, `σ0`, `σ1` | noise constants | §2.3 |
| `T` | `0.8·S` | §2.4 |

Typical plans (3.76 µm pitch unless stated):

| f-number | R | σ_G | r_G | halo | r80 (µm) | centre tap |
|---|---|---|---|---|---|---|
| 2.8 | 2 | 0.229 | 1 | 13 | 1.30 | 0.764 |
| 5.6 | 3 | 0.427 | 2 | 15 | 2.60 | 0.555 |
| 8 | 3 | 0.500 | 2 | 15 | 3.71 | 0.412 |
| 11 | 4 | 0.661 | 3 | 17 | 5.11 | 0.280 |
| 16 | 5 | 0.791 | 3 | 18 | 7.43 | 0.166 |
| 22 | 5 | 0.994 | 4 | 19 | 10.22 | 0.104 |
| 50 | 6 | 1.610 | 6 | 22 | 23.31 | 0.041 |
| 11 (5.93 µm) | 3 | 0.469 | 2 | 15 | 5.11 | 0.470 |

## 3. Algorithm

Notation: `box_r(X)` is the mean of `X` over the `(2r+1)²` window, with mirrored borders, divided by
`(2r+1)²`. `box⁺_r(X) = max(box_r(X), 0)`. `G(X)` is the separable Gaussian with taps
`exp(−i²/(2σ_G²))` for `|i| ≤ r_G`, normalised to sum 1. `K⊗X` is the 2-D convolution with the
kernel; `K` is symmetric, so convolution and correlation coincide. All windows use Yellowmoon's
mirror rule (§4).

### 3.1 Gate and planning inputs

The stage is **active** only when all of these hold:

- `pixel_pitch_x_um > 0` and `pixel_pitch_y_um > 0`, both finite;
- `1 ≤ f_number ≤ 50` (a NaN fails);
- every `g0_c > 0` and finite.

Otherwise it is inactive: the output equals the input exactly.

    scale' = scale if 0 < scale ≤ 1, else 1
    p_eff  = max(pitch_x, pitch_y) / scale'                      µm
    Δ      = min(p_eff / 4, (0.42 · N / π) / 4)                  µm, radial step

### 3.2 PSF kernel

**Radial profile.** For `k = 1 … 499`:

    P[k] = Σ_{λ=461}^{619} w(λ) · 4 · (J1(x)/x)²,    x = π · k · Δ / (λ_µm · N),   λ_µm = λ / 1000
    P[0] = Σ_{λ=461}^{619} w(λ)

with `w(λ)` from §2.6 and `J1` the Bessel function of the first kind of order 1. `(2·J1(x)/x)²` is
the Airy pattern normalised to 1 at the centre. The largest argument is about 114.

**80 % encircled-energy radius.**

    C_0 = P[0] / 6,   C_j = C_{j−1} + j · P[j]   (j = 1 … 499),   E_tot = C_499
    i*  = smallest i in 0 … 498 with C_{i+1} ≥ 0.8 · E_tot
    r80 = i* + (0.8 · E_tot − C_{i*}) / max(C_{i*+1} − C_{i*}, 0.01 · E_tot)         (samples)

**Support and taper.**

    ρ = min(4 · r80 · Δ, 6 · p_eff)                                    µm
    n = min(floor(ρ / Δ), 500);   if n < 1 the stage is inactive (cannot happen for valid inputs)
    P[k] ← P[k] · cos²(k · (Δ/ρ) · π/2)                                for k = 0 … n−1

**Splat onto the pixel grid.** `G` is an 8×8 grid of f64 zeros, indexed `[row][col]` from 0 to 7.
For `k = 1 … n−1`:

    ρ_k = k · Δ / p_eff                         (pixels)
    m   = floor(6 · ρ_k + 0.999);  skip k if m < 1
    v   = k · P[k] / m
    for j = 0 … m−1:
        θ = j · (π/2) / m;   X = ρ_k · cos θ;   Yp = ρ_k · sin θ
        splat v at (row, col) = (Yp, X), (−X, Yp), (−Yp, −X), (X, −Yp)

To splat `v` at `(pr, pc)`: `r0 = floor(pr)`, `c0 = floor(pc)`, `fr = pr − r0`, `fc = pc − c0`. Add
`v(1−fr)(1−fc)` to `[r0][c0]`, `v(1−fr)fc` to `[r0][c0+1]`, `v·fr(1−fc)` to `[r0+1][c0]` and
`v·fr·fc` to `[r0+1][c0+1]`. Drop any contribution whose row or column is negative (or above 7,
which cannot happen because `ρ_k < 6`).

The four points are the ring sample and its rotations by −90°, 180° and +90°: each ring is sampled
at `4m` points around the full circle. `G` is the full-plane bilinear splat restricted to the
quadrant `row, col ≥ 0`. Row 0 and column 0 receive the spill from the neighbouring quadrants. The
`k = 0` sample contributes nothing (its weight is `0 · P[0]`), so the centre cell gets its mass
only through the bilinear spill of the inner rings.

Symmetrise across the diagonal:

    G[r][c] = G[c][r] = max(0, (G[r][c] + G[c][r]) / 2)        for 0 ≤ r ≤ c ≤ 7

**Kernel radius.** Energy inside the `(2r+1)²` square, and the peak of each ring:

    E[0] = G[0][0]
    E[r] = E[r−1] + 4·G[r][0] + 8·Σ_{c=1}^{r−1} G[r][c] + 4·G[r][r]      r = 1 … 7
    M[r] = max(G[r][0], G[r][1], …, G[r][r]),  M[0] = G[0][0]
    M*[r] = max(M[r], M[r+1], …, M[7])
    R = smallest r in 1 … 6 with E[r] ≥ 0.99 · E[7] and M*[r] < 0.002 · M*[0];  R = 7 if none

The rule picks the first radius whose own ring, and every ring beyond, peaks below 0.2 % of the
kernel peak (normally the centre cell). Unless the fallback `R = 7` applies, the kernel's outermost
ring is therefore negligible (see §7.1, item 5).

**Kernel.** For `|dx|, |dy| ≤ R`:

    K[dy][dx] = f32( G[|dy|][|dx|] / E[R] )

`K` sums to 1 up to f32 rounding, is non-negative, and has the eight symmetries of the square.

### 3.3 High-pass width, halo, noise constants

**Line-spread fit.** `n0 = floor((17R + 16) / 4)`. For each angle `θ_a = a·π/16`, `a = 0 … 7`:

1. Zero the bins `b[0 … 2·n0 + 1]`. For every tap `(dx, dy)`, `|dx|, |dy| ≤ R`, project onto the
   direction in thirds of a pixel: `u = 3·(dy·sin θ_a + dx·cos θ_a) + n0`, `j = floor(u)`,
   `f = u − j`. Add `(1−f)·K[dy][dx]` to `b[j]` and `f·K[dy][dx]` to `b[j+1]`. The indices stay
   inside `1 … 2·n0`.
2. Smooth: `b'[m] = (b[m−1] + b[m] + b[m+1]) / 3` for `m = 1 … 2·n0 − 1`, computed from the
   unsmoothed bins. The other bins keep their values.
3. Cumulative energy: `C_0 = b'[n0]`, `C_m = C_{m−1} + b'[n0 − m] + b'[n0 + m]`.
4. Fit a least-squares line `C ≈ α + β·x` through the points `(x_m, C_m)`, `x_m = m/3`. The points
   are `m = 0` and `m = 1` always, then `m = 2, 3, …, n0 − 1` while `C_m ≤ 0.85`. Stop at the
   first `C_m > 0.85` and do not include it.
5. `x80_a = (0.8 − α) / β` (pixels).

Then:

    σ_G = f32( clamp(0.75 · mean_a(x80_a), 0.2, 3.0) )
    r_G = min(floor(3.5 · σ_G + 1), 8)                 (σ_G as stored, product evaluated exactly)
    halo = 10 + R + r_G

If the mean is not finite the stage is inactive (cannot happen for valid inputs).

**Noise constants:** `d`, `σ0`, `σ1` as in §2.3. **Clip threshold:** `T = 0.8·S`.

### 3.4 Luma, inverse noise variance, clip penalty (per pixel)

    x_c = S · in_c                                   (f64, c = R, G, B)
    Y   = f32( a_R·x_R + a_G·x_G + a_B·x_B )         (stored f32; this value is used everywhere)
    w   = f32( 1 / (σ0² + σ1 · max(Y, 0)) )
    e_c = a_c · max(x_c / g0_c − T, 0)
    sort e ascending: e(1) ≤ e(2) ≤ e(3)
    Pc  = f32( 1 + w · (25·e(1) + 5·e(2) + e(3))² )  (clip penalty, 1 when no channel is near clipping)

The penalty grows fastest when all three channels are near clipping (the smallest excess has the
largest weight).

### 3.5 Envelope estimates (5×5)

For each pixel with centre value `c = Y` and the 25 values `y_n` of its 5×5 window (centre
included), with `δ_n = y_n − c`:

    Sp = Σ_{δ_n ≥ 0} δ_n,   Np = #{δ_n ≥ 0}
    Sn = Σ_{δ_n ≤ 0} δ_n,   Nn = #{δ_n ≤ 0}                  (δ = 0 counts in both; Np, Nn ≥ 1)
    gp = 2·Sp / (Np·(Np − 1) + 1e-20)
    gn = −2·Sn / (Nn·(Nn − 1) + 1e-20)
    top = c + gp·(Np − 1);   bottom = c − gn·(Nn − 1);   mid = (top + bottom) / 2
    pull = 12.5 · min(gp, gn, 0.04·(top − bottom))
    lower = min(bottom + pull, mid);   upper = max(top − pull, mid)

`gp` is the mean spacing that `Np` evenly spaced order statistics above `c` would have, so `top`
estimates the local maximum (likewise `bottom` the minimum). `pull` moves both estimates inwards by
up to half their distance.

### 3.6 Dark and bright layers

    Lm = box⁺_2(lower),   Hm = box⁺_2(upper)
    if Y ≤ Lm:       D = Y,  B = 0
    else if Y ≥ Hm:  D = 0,  B = Y
    else:            t = (Y − Lm)/(Hm − Lm);  B = f32(Hm·t);  D = f32(Y − B)

`D + B = Y` (up to rounding in the middle case). `Lm ≤ Hm` always holds, so the middle case never
divides by zero.

### 3.7 High-pass, reblur, residual

    Hp = Y − G(Y)              high-pass of luma
    F1 = K⊗D,   F2 = K⊗B
    Z  = F1 + F2               (= K⊗Y)
    F3 = Z − G(Z)              high-pass of the reblurred luma
    ε  = Y − Z                 residual

### 3.8 Weighted moments (9×9)

    S11 = box⁺_4(w·F1²)    S22 = box⁺_4(w·F2²)    S33 = box⁺_4(w·F3²)
    S12 = box_4(w·F1·F2)   S13 = box_4(w·F1·F3)   S23 = box_4(w·F2·F3)
    T1  = box_4(w·F1·ε)    T2  = box_4(w·F2·ε)    T3  = box_4(w·F3·ε)
    Q   = box⁺_4(w·ε²)     P̄   = box⁺_4(Pc)

`Q` is the noise-normalised residual energy. For white noise of exactly the modelled variance,
`Q ≈ 1 − 2·K₀ + Σ K²` (`K₀` the centre tap): 0.07 at f/2.8, 0.40 at f/8, 0.75 at f/16, 0.94 at
f/50 (3.76 µm). §7.1, item 1, discusses what this means for the gate.

### 3.9 Per-pixel solve

All in f64, from the stored f32 moments.

    t = 0.5 · P̄
    if Q ≤ t:  (cD, cB, cH) = 0,  ω = 1 + (t − Q)/t;  done      (residual at the noise level)

    S11, S22, S33 ← max(·, 1e-15)
    λ = 0.0005 · (S11 + S22 + S33)
    solve [[S11+λ, S12, S13], [S12, S22+λ, S23], [S13, S23, S33+λ]] · (cD, cB, cH) = (T1, T2, T3)
        (any exact method; the implementation uses cofactors)

    if 0 ≤ cH < 1:
        keep cD, cB from the 3×3 solve;  tie = (cD > cB)
    else:
        cH = clamp(cH, 0, 1)
        T1' = T1 − S13·cH;   T2' = T2 − S23·cH
        solve [[S11, S12], [S12, S22]] · (cD, cB) = (T1', T2')       (no ridge)
        if cD > cB:  cD = cB = (T1' + T2') / (S11 + 2·S12 + S22)
        tie = false

    if tie:                                     (re-solve with cD = cB = c; unknowns c, cH; no ridge)
        A = S11 + 2·S12 + S22;   Bs = S13 + S23;   Ts = T1 + T2;   Dt = A·S33 − Bs²
        cH = (A·T3 − Bs·Ts) / Dt
        if 0 ≤ cH < 1:  c = (S33·Ts − Bs·T3) / Dt
        else:           cH = clamp(cH, 0, 1);  c = (Ts − Bs·cH) / A
        cD = cB = c

    if cD < −0.5:  cD = −0.5;  cB = max(cB, −0.5)
    if cD, cB or cH is not finite:  (cD, cB, cH) = 0,  ω = 1;  done

    E = Q − 2·(cD·T1 + cB·T2 + cH·T3)
          + S11·cD² + S22·cB² + S33·cH² + 2·(S12·cD·cB + S13·cD·cH + S23·cB·cH)
        (weighted residual after the fit; clamped diagonal, no ridge)

    gain = 1                         if Q ≥ P̄
           max((Q − t)/(P̄ − t), 0)   otherwise          (0 at the gate, 1 at P̄)
    kE = 2                                                       (= 1 / 0.5)
    if E > Q/4 + kE:  gain ·= max(((Q/2 + kE) − E) / (Q/4), 0)  (the fit explains too little: fade out)
    ω = 1;  if E > kE:  ω = max((kE/E)², 0.01)                  (confidence weight)

    outputs: uD = cD·gain·ω,  uB = cB·gain·ω,  uH = cH·gain·ω,  and ω

Ordering matters: the tie test uses the regularised 3×3 solution, and the tie re-solve runs only
from that branch.

### 3.10 Coefficient smoothing and correction

    ūD = box_2(uD),  ūB = box_2(uB),  ūH = box_2(uH),  ω̄ = box_2(ω)   (no clamp; ω ≥ 0.01, so ω̄ > 0)
    L  = Y + (D·ūD + B·ūB + Hp·ūH) / ω̄

This is a 5×5 confidence-weighted average of the coefficients. Gated pixels (zero coefficients,
weight 1 to 2) pull the average towards zero.

### 3.11 Strength

    L_s = Y + strength · (L − Y)

### 3.12 Luma to RGB

    k   = min( max(L_s, 1) / max(Y, 1), 1.5 )        (DN; floors at 1 DN)
    off = (L_s − Y·k) / S
    out_c = f32( in_c · k + off )                     (in normalised units, c = R, G, B)

Wherever `L_s/Y ≤ 1.5` and both are at least 1 DN, this is a pure gain: hue and saturation ratios
are kept. Past the cap, or near black, the remainder is added equally to R, G and B. Because the
luma weights sum to 1, the output luma equals `L_s` exactly in exact arithmetic.

### 3.13 Numeric precision

- **Planner:** f64 throughout. The kernel taps and `σ_G` are stored as f32, and the stored values
  are used afterwards (the line-spread fit reads the f32 taps).
- **Planes:** f32 storage. Every window sum, convolution and product accumulates in f64 and is
  rounded once to f32 when stored. The stored f32 `Y` is the one used everywhere, including §3.12.
- **Solve and transfer:** f64 from the f32 inputs; outputs rounded to f32.
- **Sensitivity:** an implementation that does everything in f32 (f32 accumulation, f32 Cramer
  solve) stays within 1e-5 of the oracle on the vectors and within 2e-6 on 72 further synthetic
  scenes (pitch 2.4–5.93 µm, f/1.4 to f/32, three noise levels). The vector tolerance is 2e-5.
  The GPU kernel nevertheless forms the luma exactly (§10).
- **Bessel J1:** any implementation with absolute error below 1e-10 on [0, 120] works. Yellowmoon
  uses the power series below 12 and the Hankel asymptotic expansion above (error below
  1e-12). A platform `j1` gives the same vectors within tolerance.

## 4. Edge cases

- **Borders.** Every window (envelopes, box means, Gaussians, PSF) reads with Yellowmoon's mirror
  rule, without repeating the edge sample (`ym_gpu::cpu::mirror`, period `2(n − 1)`). Windows wider
  than the image are mirrored repeatedly. A dimension of 1 reads its single sample.
- **Tiny images.** Any size from 1×1 works (vector `tiny_7x5`; the tests also use sizes 1×1, 2×3, 1×17).
- **Flat fields come back bit-exact.** The envelope spacings are 0, so `D = Y` and `B = 0`. The
  residual is at most rounding, so `Q ≪ 0.5` and every pixel is gated with zero coefficients. Then
  `L = Y` exactly, `k = 1`, `off = 0`, and `out = in`. This holds for values above white and for
  negatives too.
- **Zero and negative luma.** The variance uses `max(Y, 0)`. The envelope means are clamped at 0,
  so negative luma always falls into the dark layer (`D = Y`, `B = 0`). The transfer floors at 1 DN
  make the transfer additive near black. The output may be negative; nothing clamps it.
- **Values above 1.** No clamp anywhere. The clip penalty raises the gate threshold where any
  unbalanced channel exceeds 0.8 of white (vectors `highlight_clip`, `flat_above_white`).
- **Ratio cap.** Where `L_s/Y > 1.5` (dark pixels next to a brightened edge), `k = 1.5` and the
  rest is added equally to the three channels, which moves those pixels towards grey.
- **Degenerate solve.** A non-finite coefficient leaves the pixel alone (all three coefficients 0,
  `ω = 1`). With finite input this needs pathological moments; the tests sweep extreme inputs
  without hitting it.
- **Inactive stage.** The output is a bit-exact copy (vector `disabled_no_f_number`).
- **Strength.** Clamped to [0, 3]; NaN reads as 1; 0 is a bit-exact identity.
- **CFA phases.** Not applicable: the input is demosaiced RGB. The result does not depend on the
  frame position or the Bayer phase.
- **Non-finite input.** Outside the contract (R7 never produces it). It is not guarded and
  propagates.

## 5. Determinism, tiling and performance

- **Determinism.** No frame-level reductions. Every output pixel depends only on its
  `(2·halo + 1)²` neighbourhood and on per-frame constants, so the result does not depend on the
  thread split. Work is split by output rows (128-row bands in `DiffractionPlan::apply`), as the rest of `ym_gpu::cpu` does; one thread and four give the same bits.
- **Tiling.** With `halo = 10 + R + r_G`, a tile run with that much context, keeping the core,
  equals the whole-frame run bit for bit (tests: three f-numbers, five tile
  positions including frame edges). Running window sums that restart per tile row would not be bit-exact
  but stay far inside the 1e-4 contract. With less context the outermost taps are tiny: 1 px less
  changed nothing on the test scenes, 4 px less up to 4e-5. The full halo is declared anyway.
- **Cost per pixel.**
  - Envelopes: 25 reads.
  - 17 box means, O(1) each with running sums: 2 for the envelopes, 11 for the moments, 4 for the
    coefficients.
  - 2 separable Gaussians, `2·(2r_G + 1)` taps each.
  - 2 PSF convolutions, `(2R+1)²` taps each (81 at R = 4, at most 225). The 8-fold symmetry folds
    them to at most 36 distinct taps. Alternatively `Z = K⊗Y` is computed once and `F2 = Z − F1`; this
    agrees within rounding and saves one convolution.
  - About 100 flops for the solve.

  Roughly 1 kflop per pixel at R = 4 with direct separable box sums. Measured times are in §6.4.
- **Memory.** About 25 f32 planes of the tile are live at the peak; many can be dropped early. A
  512×512 core with a 25 px halo needs about 562² × 25 × 4 B ≈ 32 MB per tile in flight.
- **Planner.** 159 × 499 Bessel evaluations plus a few thousand trig calls, once per frame (under
  10 ms). The kernel, R and σ_G depend only on `(p_eff, N)`; the planner caches them by that key (eight entries), so the scheduler can ask for the halo per tile.
- **GPU.** All passes are per-pixel gathers; there are no atomics or reductions. The kernel (§10) runs eleven dispatches over nine pipelines:
  1. luma, `w`, penalty;
  2. envelopes (5×5 gather);
  3. layer split (the 5×5 means of the envelopes);
  4. Gaussian (separable), for `Y` and then for `Z`;
  5. PSF (2-D gather; kernel in a buffer, R ≤ 7) and the residual;
  6. 9×9 moments (a row pass; the ten products are formed at each tap and never stored);
  7. a column pass fused with the solve (the branches written as selects);
  8. coefficient box, correction and transfer over the tile's core.

## 6. Test plan

### 6.1 Vectors (`crates/ym-gpu/tests/cpu_data/diffraction_deconvolution/*.json`)

JSON fields: `name`, `description`, `params`, `input`, `expected`, `tolerance`.

- `params`: `pixel_pitch_x_um`, `pixel_pitch_y_um`, `f_number`, `scale`, `g0`, `luma`, `noise`
  `{a, b, sigma_row, q}` (normalised, unbalanced), `strength`, `working_scale`.
- `input`: `null` for planner vectors, otherwise `{width, height, layout: "planar_rgb", data}`
  (normalised balanced RGB).
- `expected`: `active` (1 or 0); planner fields `radius`, `gauss_sigma`, `gauss_radius`, `halo_px`,
  `sigma0_dn`, `sigma1_dn` (plus `r80_um`, `support_um` and the row-major `kernel` for planner
  vectors); `width`, `height` and `data` for image vectors.
- `tolerance`: one absolute number for every value in `expected`. Integers must match exactly.
  Image data is in normalised units: 2e-5. Planner vectors: 1e-6. Exact vectors: 0, and their
  `expected` holds only `active` and the image, which must be bit-equal to the input.

| Vector | Size | Covers | Tol. |
|---|---|---|---|
| `plan_p3.76_f2.8` | – | small kernel (R 2) | 1e-6 |
| `plan_p3.76_f8` | – | typical (R 3, σ_G 0.500); noise constants | 1e-6 |
| `plan_p3.76_f16` | – | support cut at 6 px | 1e-6 |
| `plan_p3.76_f50` | – | upper gate limit (R 6, σ_G 1.61) | 1e-6 |
| `plan_p5.93_f11` | – | larger pitch | 1e-6 |
| `plan_p2.4_f4` | – | 1-inch-class pitch | 1e-6 |
| `plan_p8.4_f1` | – | lower gate limit; σ_G clamped to 0.2 | 1e-6 |
| `plan_p3.76x4.0_f11` | – | unequal pitches (larger used) | 1e-6 |
| `plan_p3.76_f8_scale0.25` | – | `scale` (effective pitch ×4, noise ×0.5) | 1e-6 |
| `plan_f_out_of_range` | – | f/64: inactive | 1e-6 |
| `flat_field` | 32×32 | flat colour field: exact identity | 0 |
| `flat_above_white` | 24×24 | flat above white, penalty active: exact identity | 0 |
| `edge_vertical` | 48×40 | grey step blurred by the f/11 PSF, ISO 100 noise | 2e-5 |
| `edge_vertical_strength_half` | 48×40 | same with strength 0.5 | 2e-5 |
| `edge_horizontal_colour` | 40×48 | orange/blue step at f/16, other gains: ratio transfer | 2e-5 |
| `edge_diagonal` | 48×48 | 30° step at 2.4 µm and f/22 (R 6) | 2e-5 |
| `zone_plate` | 48×48 | frequencies up to Nyquist, f/8 | 2e-5 |
| `noisy_high_iso` | 40×40 | high-ISO noise: flat parts gated, square edges corrected | 2e-5 |
| `highlight_clip` | 48×48 | disc at and above white (up to 2.0 after gains): clip penalty | 2e-5 |
| `dark_negatives` | 32×32 | dark checkerboard with negatives: variance clamp, floors | 2e-5 |
| `tiny_7x5` | 7×5 | image smaller than the halo | 2e-5 |
| `preview_scale_quarter` | 32×32 | `scale` 0.25 | 2e-5 |
| `disabled_no_f_number` | 16×16 | f-number 0: inactive, exact copy | 0 |

The blurred inputs are the sharp scene convolved with the plan's own kernel (mirror borders), plus
Poisson–Gaussian noise from the model, applied independently per channel in unbalanced units. The
generator's PRNG is splitmix64 with Irwin–Hall normals (arithmetic only).

**Two classes of vectors (`docs/29` F-06, RC-04).** *Oracle:* the files above, unedited, run under `Tuning::ORACLE`; they prove the structure (borders, phases, ties, decision logic). *Self:* `crates/ym-gpu/tests/cpu_data/diffraction_deconvolution_self/`, the same files, fields and tolerances written by `crates/ym-gpu/examples/gen_self_vectors.rs` from our own Rust under `Tuning::DEFAULT`; they are the regression of the shipped constants and are never called an oracle. The CPU and GPU tests read both folders. After a fit changes a shipped constant, regenerate with `cargo run -p ym-gpu --example gen_self_vectors` and commit the diff; `tests/self_vectors.rs` fails while the committed files differ from a regeneration.

### 6.2 Property tests (`crates/ym-gpu/tests/cpu_diffraction.rs`)

1. **Flat fields exact** (bit-equal): levels 0, 0.18 grey, a colour, above white (1.9, 1.3, 1.6
   and 3.5, 2.0, 2.9), slight negatives.
2. **Identities exact:** `strength = 0`; inactive stage (f/64, f/0.99, NaN, pitch ≤ 0).
3. **Tiling parity:** tiles with `halo` context equal the whole frame (measured difference 0;
   contract: ≤ 1e-4) at f/2.8, f/22 and f/45 (2.4 µm), for tiles in the interior and on the frame
   edges.
4. **Square symmetries:** transpose, flip x and flip y commute with the stage (≤ 1e-5; measured 0).
5. **Scale behaviour:** scaling the input by α together with `a → α·a`, `b → α²·b` (and `σ_row`,
   `q` by α) scales the output by α. Exact for α = 1/2 when the 1 DN floors and the clip threshold
   are not involved; ≤ 1e-5 for α = 0.7.
6. **Kernel:** sums to 1 (±1e-5), non-negative, 8-fold symmetric; `1 ≤ R ≤ 7`; `0.2 ≤ σ_G ≤ 3`;
   `halo = 10 + R + r_G ≤ 25`. Sweep pitch 1–12 µm × f/1 to f/50 × scale 1, 1/2, 1/4. The plans
   tabulated in §2.7 must come out as listed (R exact, σ_G ±0.006); the planner vectors pin them
   more tightly.
7. **No NaN or Inf:** sizes 1×1 to 48×40; values from −0.05 to 4; zero noise model (σ0 at the
   1 DN floor); strength 3; f/1 to f/50.
8. **Transfer:** the output luma (same weights) equals `L_s` to 2e-6 relative.
9. **Noise:** a flat patch whose luma noise matches the model (one normal deviate shared by the
   three channels) grows by at most 5 % at f/8 (measured ×1.019, the worse of ISO 100 and 6400).
   Beyond about f/11 at 3.76 µm it grows (§7.1, item 1). Report the numbers; do not gate on them.
10. **Sharpening works:** a step blurred by the PSF (f/16, 3.76 µm, ISO 100 noise) gets closer to
    the sharp step (summed squared error 5.30 → 3.31) and steeper (10–90 % rise
    2.62 → 2.04 px).
11. **Bessel J1** against tabulated values (≤ 1e-12 measured).

### 6.3 Quality gate in Yellowmoon

`cargo run -p ym-gpu --example capture_sharpen_ab` runs the synthetic comparison, and the ignored test `capture_sharpen_real_files_ab` of `ym-core` runs the real files (crops and `index.html` go to the git-ignored `testdata/capture_sharpen_ab/`).

- **Synthetic.** Sharp scenes (`synth::fine_text`, `synth::natural`, a slanted edge at 5°) are blurred with the diffraction PSF of a given pitch and f-number (f/5.6, f/8, f/11, f/16, f/22). Model noise at ISO 100, 800 and 6400 is added, with channel-correlated luma noise. R8 runs as Richardson–Lucy Auto, then as Diffraction (and, from f/11, as the noise-gate variant of §7.1 item 1). The measures are:
  - MSE/PSNR against the sharp scene;
  - slanted-edge MTF50 and 10–90 % rise distance;
  - overshoot and undershoot as a percentage of the step;
  - luma noise SD in flat patches, relative to the input.
- **Real files.** The local ARWs, and any at f/2.8, f/8 and f/16: edge profiles (10–90 % rise), overshoot and flat-patch noise.
- **Acceptance criterion for making Diffraction selectable:** at f/8 and above, MTF50 at least that of Richardson–Lucy Auto, with no more overshoot and no more flat-patch noise. Below f/5.6 the two methods should be close, because the diffraction PSF is then smaller than a pixel. Neither variant meets it (§6.4).

### 6.4 Measured

Synthetic scenes at a 3.76 µm pitch, regenerated by RC-11 on 2026-11-18 with `cargo run --release -p ym-gpu --example capture_sharpen_ab` at the shipped `Tuning::DEFAULT` (the numbers are those of the first measurement: RC-11 changed no constant). The scenes are blurred with the stage's own PSF; PSNR is against the sharp scene (fine text and natural), MTF50 in cycles per pixel, the 10 to 90 % rise of the edge in px, overshoot and undershoot in % of the step, flat-patch luma noise relative to the input:

| f | ISO | method | text PSNR (dB) | natural PSNR (dB) | MTF50 (c/px) | rise (px) | overshoot (%) | undershoot (%) | flat noise (x) |
|---|---|---|---|---|---|---|---|---|---|
| 5.6 | ISO 100 | input | 32.40 | 37.18 | 0.323 | 1.46 | 0.3 | 0.1 | 1.000 |
| 5.6 | ISO 100 | Richardson-Lucy Auto | 28.65 | 35.97 | 0.419 | 0.91 | 13.9 | 4.3 | 1.216 |
| 5.6 | ISO 100 | diffraction | 33.32 | 38.02 | 0.389 | 1.15 | 1.1 | 1.0 | 1.000 |
| 5.6 | ISO 800 | input | 31.17 | 35.12 | 0.334 | 1.46 | 0.7 | 0.3 | 1.000 |
| 5.6 | ISO 800 | Richardson-Lucy Auto | 28.89 | 34.41 | 0.408 | 0.98 | 10.8 | 4.1 | 1.168 |
| 5.6 | ISO 800 | diffraction | 31.86 | 35.66 | 0.400 | 1.12 | 1.9 | 1.5 | 1.000 |
| 5.6 | ISO 6400 | input | 25.75 | 28.24 | 0.301 | 1.59 | 3.1 | 0.4 | 1.000 |
| 5.6 | ISO 6400 | Richardson-Lucy Auto | 24.99 | 27.51 | 0.387 | 1.11 | 8.9 | 2.9 | 1.118 |
| 5.6 | ISO 6400 | diffraction | 25.68 | 28.25 | 0.375 | 1.24 | 3.1 | 0.8 | 1.000 |
| 8 | ISO 100 | input | 29.30 | 34.56 | 0.273 | 1.94 | 0.3 | 0.1 | 1.000 |
| 8 | ISO 100 | Richardson-Lucy Auto | 33.19 | 38.04 | 0.380 | 1.11 | 7.0 | 2.7 | 1.255 |
| 8 | ISO 100 | diffraction | 30.68 | 35.80 | 0.340 | 1.26 | 1.7 | 1.6 | 1.005 |
| 8 | ISO 800 | input | 28.65 | 33.31 | 0.275 | 1.92 | 0.7 | 0.3 | 1.000 |
| 8 | ISO 800 | Richardson-Lucy Auto | 31.28 | 34.68 | 0.363 | 1.18 | 6.1 | 2.7 | 1.200 |
| 8 | ISO 800 | diffraction | 29.84 | 34.14 | 0.354 | 1.26 | 2.4 | 2.0 | 1.007 |
| 8 | ISO 6400 | input | 24.87 | 27.80 | 0.269 | 2.06 | 3.1 | 0.4 | 1.000 |
| 8 | ISO 6400 | Richardson-Lucy Auto | 24.92 | 27.21 | 0.299 | 1.48 | 5.6 | 1.7 | 1.141 |
| 8 | ISO 6400 | diffraction | 24.58 | 27.71 | 0.314 | 1.27 | 4.7 | 2.3 | 1.011 |
| 11 | ISO 100 | input | 26.50 | 32.30 | 0.227 | 2.29 | 0.3 | 0.1 | 1.000 |
| 11 | ISO 100 | Richardson-Lucy Auto | 31.24 | 35.60 | 0.312 | 1.55 | 4.6 | 1.8 | 1.255 |
| 11 | ISO 100 | diffraction | 28.49 | 34.05 | 0.303 | 1.59 | 2.7 | 2.7 | 1.200 |
| 11 | ISO 100 | diffraction, noise gate | 28.40 | 33.95 | 0.303 | 1.59 | 2.7 | 2.7 | 1.081 |
| 11 | ISO 800 | input | 26.15 | 31.52 | 0.223 | 2.31 | 0.7 | 0.3 | 1.000 |
| 11 | ISO 800 | Richardson-Lucy Auto | 29.42 | 33.11 | 0.306 | 1.66 | 3.7 | 1.7 | 1.200 |
| 11 | ISO 800 | diffraction | 27.88 | 32.56 | 0.319 | 1.55 | 3.3 | 3.1 | 1.242 |
| 11 | ISO 800 | diffraction, noise gate | 27.86 | 32.58 | 0.319 | 1.55 | 3.3 | 3.1 | 1.103 |
| 11 | ISO 6400 | input | 23.65 | 27.21 | 0.211 | 2.36 | 3.1 | 0.4 | 1.000 |
| 11 | ISO 6400 | Richardson-Lucy Auto | 24.16 | 26.81 | 0.264 | 2.03 | 3.7 | 0.7 | 1.141 |
| 11 | ISO 6400 | diffraction | 22.79 | 26.13 | 0.289 | 1.60 | 6.7 | 3.6 | 1.275 |
| 11 | ISO 6400 | diffraction, noise gate | 23.04 | 26.67 | 0.289 | 1.60 | 6.3 | 3.7 | 1.114 |
| 16 | ISO 100 | input | 23.75 | 30.11 | 0.184 | 2.63 | 0.3 | 0.1 | 1.000 |
| 16 | ISO 100 | Richardson-Lucy Auto | 26.96 | 32.25 | 0.238 | 2.17 | 1.0 | 0.3 | 1.255 |
| 16 | ISO 100 | diffraction | 26.08 | 31.89 | 0.245 | 1.97 | 3.0 | 2.1 | 1.577 |
| 16 | ISO 100 | diffraction, noise gate | 25.87 | 31.67 | 0.244 | 1.97 | 3.0 | 2.1 | 1.085 |
| 16 | ISO 800 | input | 23.56 | 29.62 | 0.183 | 2.67 | 0.7 | 0.3 | 1.000 |
| 16 | ISO 800 | Richardson-Lucy Auto | 25.96 | 30.85 | 0.223 | 2.27 | 0.9 | 0.5 | 1.200 |
| 16 | ISO 800 | diffraction | 25.70 | 30.75 | 0.235 | 1.93 | 4.2 | 2.5 | 1.617 |
| 16 | ISO 800 | diffraction, noise gate | 25.66 | 30.83 | 0.236 | 1.93 | 4.2 | 2.5 | 1.101 |
| 16 | ISO 6400 | input | 22.01 | 26.40 | 0.183 | 2.67 | 3.1 | 0.4 | 1.000 |
| 16 | ISO 6400 | Richardson-Lucy Auto | 22.66 | 26.15 | 0.201 | 2.44 | 3.5 | 0.6 | 1.141 |
| 16 | ISO 6400 | diffraction | 21.30 | 24.37 | 0.227 | 1.97 | 7.4 | 3.2 | 1.714 |
| 16 | ISO 6400 | diffraction, noise gate | 21.84 | 25.99 | 0.232 | 1.99 | 5.9 | 3.4 | 1.118 |
| 22 | ISO 100 | input | 21.45 | 28.38 | 0.152 | 3.15 | 0.3 | 0.1 | 1.000 |
| 22 | ISO 100 | Richardson-Lucy Auto | 23.37 | 29.69 | 0.180 | 2.68 | 0.6 | 0.2 | 1.255 |
| 22 | ISO 100 | diffraction | 24.19 | 30.29 | 0.196 | 2.30 | 4.6 | 3.1 | 1.759 |
| 22 | ISO 100 | diffraction, noise gate | 23.90 | 30.02 | 0.196 | 2.30 | 4.6 | 3.1 | 1.080 |
| 22 | ISO 800 | input | 21.34 | 28.05 | 0.150 | 3.14 | 0.7 | 0.3 | 1.000 |
| 22 | ISO 800 | Richardson-Lucy Auto | 22.84 | 28.87 | 0.173 | 2.78 | 0.9 | 0.4 | 1.200 |
| 22 | ISO 800 | diffraction | 23.85 | 29.37 | 0.195 | 2.28 | 5.5 | 3.3 | 1.815 |
| 22 | ISO 800 | diffraction, noise gate | 23.80 | 29.45 | 0.195 | 2.28 | 5.4 | 3.3 | 1.098 |
| 22 | ISO 6400 | input | 20.35 | 25.58 | 0.153 | 3.08 | 3.1 | 0.4 | 1.000 |
| 22 | ISO 6400 | Richardson-Lucy Auto | 20.88 | 25.39 | 0.169 | 2.85 | 3.5 | 0.5 | 1.141 |
| 22 | ISO 6400 | diffraction | 20.09 | 23.52 | 0.202 | 2.25 | 10.1 | 4.3 | 1.900 |
| 22 | ISO 6400 | diffraction, noise gate | 20.65 | 25.33 | 0.203 | 2.28 | 6.8 | 4.6 | 1.116 |

The ISO 100 rows, as the first measurement printed them (MTF50, rise, overshoot, flat noise): f/8 diffraction 0.340, 1.26, 1.7, ×1.005 against Richardson–Lucy 0.380, 1.11, 7.0, ×1.255; f/16 0.245, 1.97, 3.0, ×1.577 (with the noise gate ×1.085) against 0.238, 2.17, 1.0, ×1.255; f/22 0.196, 2.30, 4.6, ×1.759 (gate ×1.080) against 0.180, 2.68, 0.6, ×1.255.

- At ISO 100 Richardson–Lucy has the higher MTF50 at f/5.6 to f/11 (f/5.6 0.419 against 0.389, f/11 0.312 against 0.303) and Diffraction from f/16. Diffraction has less overshoot at f/5.6 and f/8 (1.1 and 1.7 % against 13.9 and 7.0 %) and more at f/16 and f/22.
- The ISO moves the comparison, because the Auto amount of Richardson–Lucy falls as the ISO rises: at ISO 800 Diffraction has the higher MTF50 from f/11 (0.319 against 0.306), and at ISO 6400 from f/8 (0.314 against 0.299 at f/8, 0.289 against 0.264 at f/11, 0.227 against 0.201 at f/16), where at f/8 it also has less overshoot (4.7 against 5.6 %) and less flat-patch noise (×1.011 against ×1.141).
- Flat-patch noise is ×1.00 to ×1.01 up to f/8 and ×1.20 to ×1.90 at f/11 to f/22 (above the ×1.10 bar of §7.1 item 1). The noise-gate variant brings it to ×1.08 to ×1.12 without changing the edge; five of the ISO 800 and 6400 cases at f/11 to f/22 stay marginally over (1.101 to 1.118). The overshoot criterion is met by neither variant at f/16 and f/22.
- On the nine local real files the stage changes little, because the gate is closed on real flat areas (the R6 denoise and R7 leave less noise than the raw model says) and real edges are wider than the Airy blur alone: only four files gave a clean edge (MTF50 0.06 to 0.14). ILCE-6300 cRAW at f/8 goes from 0.141 to 0.148 with Diffraction, against 0.205 with Richardson–Lucy. Flat-patch noise on real files is ×1.000 from f/2.8 to f/8 and, for the one file at f/11 or more (ILCE-7M2 at f/22), ×1.019 on average ungated and ×1.010 gated. No real file at f/16 exists.
- Speed: the CPU stage takes 0.5 to 1.0 s per megapixel on four loaded cores, so 30 to 60 s for a 60 MP frame. §10 has the device parity numbers; speed on a real GPU is a Mac check (M0-H4 in `docs/15-milestones.md`).

## 7. Variants and fixed choices

### 7.1 Variants and known limits

**Item 1 · The gate opens on noise at small apertures.** The gate threshold is fixed at `0.5·P̄`, but the white-noise value of `Q` (§3.8) rises with the PSF width: 0.40 at f/8, 0.58 at f/11, 0.75 at f/16 and 0.85 at f/22 (3.76 µm). When the luma noise matches the model, flat-patch noise therefore grows ×1.21 at f/11, ×1.56 at f/16 and ×1.79 at f/22. With uncorrelated channel noise (luma variance well below the model) there is no growth, so how much R7 leaves decides the real effect.

The **noise-gate variant** `DiffractionPlan::with_noise_gate(true)` scales the gate by the white-noise level. With `q_w = 1 − 2·K₀ + Σ K²` from the plan (`white_noise_residual()`) and `f = max(1, 2·q_w)`, it uses `t = 0.5·f·P̄` and full gain at `Q = f·P̄`. It is identical to the plain stage up to f/8 at 3.76 µm (`f = 1`). A flat patch with luma noise matching the model grows by (plain against variant): f/11 ×1.208 against ×1.081, f/16 ×1.559 against ×1.075, f/22 ×1.786 against ×1.071, f/32 ×1.995 against ×1.074. On the synthetic step the edge error improves slightly (f/16 3.37 → 3.32, f/22 4.86 → 4.78). The product does not use the variant; the A/B tools do.

**Item 3 · Luma weights.** The stage uses `SPEC_LUMA` = 0.299 / 0.587 / 0.114 on linear camera RGB, not Yellowmoon's `LUMA` (0.2126, 0.7152, 0.0722) of the other stages. The weights enter `Y`, `d` (noise), the clip penalty and the transfer, and `DiffractionPlan::new` takes them as one parameter so that all four change together.

**Item 8 · Stacking with Richardson–Lucy.** One R8 method runs per frame. Defocus and lens aberrations add blur that the Airy model does not contain, but the two methods are not stacked.

### 7.2 Fixed choices

| Question | Choice |
|---|---|
| Where do the constants come from? | Taken over from research and re-derived by the project: each row of the constants table (§2.6) has its derivation or its fit, and `docs/29` has the numbers. Nothing is added to a table without a question first |
| Fallback when inactive (no f-number or no pitch) | Richardson–Lucy with the resolved Auto parameters |
| Mapping of the Sharpening Amount | None; `strength = 1` |
| Vignetting gain in the noise weight | Ignored: corners carry `G(r)²` more variance than the model |
| Pixel pitch source order | Registry, then EXIF crop factor, else inactive |
| Clip test | R1's clip mask is not available in RGB at R8, so the stage uses the 0.8 threshold on unbalanced values. In vignetting-corrected corners this can flag unclipped pixels, which only means less sharpening there |
| Lenses with a missing or wrong f-number (manual lenses, adapters, teleconverters) | EXIF is trusted; a missing or zero f-number makes the stage inactive |
| Default R8 method | Richardson–Lucy; Diffraction is reachable only through the internal switch (§1.2) |

## 8. Conventions of the implementation

Choices that an implementer might otherwise make differently:

- **Borders:** Yellowmoon's mirror rule (no repeated edge sample) for every window.
- **Precision:** f64 accumulation for every window sum, convolution and product; an f64 solve; an
  f64 planner (the line-spread positions are averaged in f64, and `r_G` is computed exactly from the
  stored f32 `σ_G`).
- **Variance:** the signal term uses `max(Y, 0)`, as `NoiseModel::pixel_variance` does, so the
  weight stays positive on negative luma.
- **Degenerate solve:** a non-finite coefficient leaves the pixel alone (all three coefficients 0,
  `ω = 1`).
- **Single luma:** the base and data luma are the same plane `Y`.
- **Output in normalised units:** `out = in·k + off/S`, so flat fields, `strength = 0` and inactive
  plans are bit-exact identities.
- **Strength:** `strength = 1` is the base behaviour.
- **Gate inputs:** `scale` outside (0, 1] reads as 1; `g0` must be positive and finite.
- **Splat bounds:** contributions outside the 8×8 quadrant are dropped on every side (the upper
  bound never triggers).
- **Bessel J1:** the implementation's own series and asymptotic expansion (§9); a library
  `j1` gives the same vectors within tolerance.

## 9. The CPU implementation

Code: `crates/ym-gpu/src/cpu/sharpen/diffraction.rs` (planner, stage, `DiffractionParams`, `DiffractionPlan`), `diffraction/eval.rs` (edge, MTF and noise measures of the A/B, not used by the render path), `CaptureSharpen` in `cpu/sharpen.rs`, `Stage::Sharpen(CaptureSharpen)` and the per-method halo in `cpu/tiling.rs`, and the Diffraction arm of `Chain::run_tile`. In `ym-core`: `pipeline::raw::{pixel_pitch, diffraction_params, CaptureMethod, DEFAULT_CAPTURE_METHOD, RawParams::{diffraction, capture_sharpen}}` and `PhotoSource::{sensor, capture_method}`.

- **Fidelity.** The Rust follows the oracle operation by operation (planner f64, planes f32, f64 accumulation in the oracle's order, no `mul_add`). All 13 image vectors agree **bit for bit** (maximum difference 0, tolerance 2e-5); the 10 planner vectors agree to 1.3e-8 (tolerance 1e-6); the numbers of §6.2 come out as listed (squared error 5.30 to 3.31, rise 2.62 to 2.04 px, noise ×1.019 at f/8, ×1.21, ×1.56, ×1.79 at f/11, f/16, f/22, luma of the output against `L_s` 8.1e-8, tiling parity 0, scale 1.1e-7). The loops are organised by rows so that they vectorise; each output sample sums the same terms in the same order.
- **Bessel J1.** Written from Abramowitz and Stegun, the power series 9.1.10 below 12 and the Hankel expansion 9.2.5 above, terms summed until they grow. No library, so no dependency or notice. Error against Bessel's integral on `[0, 120]`: 8.6e-13.
- **Noise.** `NoiseDn::r` that is NaN or infinite (a negative or infinite variance in `NoiseDn::from_model`) reads as 0, so `sigma0` sits at its 1 DN floor; a negative `g` reads as 0.
- **Halo.** `DiffractionParams::halo_px()` is the plan's `10 + R + r_G` (0 when the stage is inactive), independent of the gains and the noise. The 24 px cap belongs to Richardson–Lucy only. A tile with the halo equals the whole frame bit for bit, and so do the 128-row bands of `DiffractionPlan::apply`, with one thread or four.
- **Selection.** `RawParams::diffraction` is `Some` only when `PhotoSource::capture_method` asks for the method (`DEFAULT_CAPTURE_METHOD` is Richardson–Lucy) and the photo has an f-number from 1 to 50 and a pitch. Otherwise, and when the plan is inactive for the gains and the noise model, R8 is Richardson–Lucy with the resolved Auto values. No settings, IPC or export change.
- **Pixel pitch.** `pixel_pitch` takes, in order: the registry body's `SensorGeometry` (`pitch = 1000 · sensor_width_mm / active_width_px`; for a Lossless M or S file `scale = frame_width / active_width_px`); else the crop factor `43266.6 / (crop · √(W² + H²))` with `W × H` the frame R8 receives (`scale = 1`, also for a reduced file); else none. A pitch outside 0.5 to 20 µm counts as not available (a registry value there is ignored and the crop factor is tried). The crop factor is about 2 % high on APS-C (nominal 1.5 against 1.53). The registry values are Sony's published figures (`docs/03-camera-profiles.md`); 27 of 28 bodies have them, and the entries for the A7 III, A7 IV, A7C II, A7S III and A1 are marked `restated` (Sony's figures as restated by a retailer or reference site).

## 10. The GPU kernel

Files: `crates/ym-gpu/src/shaders/diffraction.wgsl` (the per-pixel stages as plain functions), `crates/ym-gpu/src/kernels/diffraction.rs` (`PreparedDiffraction`: upload, nine pipelines, tiling, `run_traced`; `encode_sharpen` for the device chain), `crates/ym-gpu/tests/gpu_diffraction.rs` (parity). The device chain runs the kernel as R8 when the stage is selected (`GpuChain`, `docs/17-gpu.md` section 11).

- **Split.** The planner stays on the CPU (it runs once per `(pitch, f-number)` and is cached there). `PreparedDiffraction::new` takes the `DiffractionPlan` and uploads its f32 kernel and the Gaussian taps (computed in f64 with the CPU's expression, rounded to f32). `sigma0²`, `sigma1`, the strength, the gate factor, the gains `g0` and the luma weights all come from the plan (`DiffractionPlan::g0()` and `luma()` give back the f32 values it was made with), so the kernel cannot be run with others than the CPU's. Eleven dispatches over nine pipelines, as §5 lists them: luma, `w` and `Pc`; envelopes; layers; Gaussian rows and columns (for `Y`, then for `Z = F1 + F2`); PSF convolutions and the residual; the 9×9 moments as a row pass (the ten products `(w u) v` are formed at each tap, never stored) and a column pass fused with the solve; the 5×5 coefficient averages, the correction and the transfer over the tile's core. The kernel needs 124 bytes per pixel of compute rectangle; planes are f32 in every storage mode (the envelope counts need exact f32 order).
- **Precision.** Window sums, convolutions, products and the Cramer solve are f32, in the CPU's order (x first, then y; PSF taps row by row; Gaussian taps from the left). Measured on lavapipe (maximum absolute difference, normalised units): the 13 image vectors 8.9e-7 (tolerance 2e-5); random frames of 11 optics × 6 scene kinds 5.1e-6; other strengths, scales, gates, luma weights and gains 3.6e-6; values from −0.05 to 4 with strength 3 4.3e-6; sizes 1×1 to 7×7 9.2e-7; 2048×1365 through 512 px tiles 4.8e-7 (nothing above 1e-6 in 8.4 million samples). A replica of the CPU port with the 9×9 moments summed in f32 and the solve done in f32 moved the corrected luma by at most 4e-4 DN on the worst pixels of the large frame and 0.011 DN (1.6e-7 normalised) on a hard noise-free edge scene; the ridge keeps the 3×3 system within a condition number of about 2000 (`(tr + λ)/λ` with `λ = 0.0005·tr`).
- **The luma has to be exact.** The all-f32 implementation of §3.13 stays within 1e-5 on small scenes, but on a 2048×1365 natural frame an f32 luma (1 to 2 ulp off the CPU's in a third of the pixels) gave 9.1e-5 at a handful of pixels: a neighbour within an ulp of the centre flips the `≥ 0` / `≤ 0` counts of §3.5, and one count moves the pixel's envelope by up to 8 %. The shader therefore forms `Y` as an exact expansion (Dekker products, Knuth sums, f32 only; `exact_luma`) and gets the CPU's bits (0 differences in 240,000 pixels, and in 17.3 million further random, 14-bit, balanced and unbalanced lumas); the large frame falls to 4.8e-7. It needs every rounded intermediate and every error term hidden from the shader compiler (`opaque`: an OR with a uniform that is always 0; an OR because an XOR applied twice cancels, `docs/17-gpu.md` K8). `fma` is not used, because lavapipe lowers it to a multiply and an add.
- **Fast-math (Metal).** Metal compiles with fast-math (`docs/17-gpu.md` section 13), which re-associates the open error sum of the Knuth sum `(a − ad) + (b − bb)` into `(a + b) − (bb + ad)` = 0 unless every error term goes through the barrier too (the same sequence compiled with clang `-O2 -ffast-math`: 54 % of 2 million lumas off the f64 luma without it, none with it, also under `-ffp-contract=fast`). The Dekker product needs no more than that: its partial products are exact in f32 (12-bit halves), so a contraction into fused multiply-adds cannot change a bit. A finite input sample is **bounded** to `±1e6 DN / (S·max(Σ|a_c|, 1))` (about 15 times white) in the luma pass, so that no product of the solve can overflow f32 whatever order the compiler evaluates it in (with `w ≤ 1` every moment stays below 1e12 and the determinant below 4e36). A frame within the bound is untouched; the CPU reference solves a sample beyond it exactly in f64, so that is the one place where the two differ by design. NaN and infinities are not bounded: they propagate, no further than the halo (tested). The test `the_luma_plane_has_the_cpu_bits` fails on a driver that fuses through the barrier, and the stage then degrades to the 1e-4 level of the plain f32 sum.
- **Tiling.** A tile with the plan's exact halo (`10 + R + r_G`) gives exactly the bits of the whole-frame GPU run (tile sizes 32 to 100 over 203×157 at halos 13, 18 and 24); two runs are bit-identical in every storage mode. The perturbation reach found on the GPU is 13 px at a 15 px halo and 17 at 19 (the outermost taps of the Gaussian and the PSF are tiny).
- **Ill-conditioned input.** On a noise-free smooth ramp every window is evenly spread, so §3.5 gives `lower = upper = mid` up to rounding, `Y` sits within a rounding of both thresholds `Lm` and `Hm` of §3.6, and the dark/bright split is decided by rounding. The CPU port itself moves by up to 3.5e-3 when such an input changes by 0.6 ulp, so no port can match it there. §4's exact flat fields hold; smooth noise-free gradients (a synthetic wedge without noise) are the case it does not cover. On a noise-free scene quantised to 14 bits (a smooth gradient with hard-edged shapes) the GPU differs from the CPU by up to 4.3e-3 in a handful of samples; changing the last mantissa bit of every input sample moves the C++ oracle itself by up to 4.4e-3 on that scene (599 of 36,864 samples above 1e-4, 15 above 1e-3). With one quantisation step of noise the GPU–CPU difference is 1.1e-6. Real frames and the vectors carry sensor noise (`sigma0 ≥ 1` DN), as do the parity scenes. A CPU render and a device render can therefore differ in the last digits of noise-free smooth regions, though not in the noisy frames measured.
- **Against the oracle on new scenes.** Six scenes that are not in the vectors (random discs, bars and gradients blurred by the plan's PSF, with model noise; 131×97 at 4.3 µm f/13, 64×73 at scale 0.5 with Rec. 709 luma and strength 1.7, 120×89 at 1.9 µm f/45, 5×3, a 1×61 strip, 160×110 at f/9 and low ISO): the CPU port equals the oracle bit for bit, and the GPU is within 1.2e-6 of both.
- **Tests added by the review.** `hostile_values_that_the_stage_acts_on_match_the_cpu` (values −0.05 to 0.7 with and without 1 % spikes up to 4, f/1 to f/8, the CPU must change the frame, |gpu − cpu| at most 1.3e-6) and `a_non_finite_sample_stays_local_and_does_not_panic` (a NaN, an infinity or 1e30 changes nothing beyond the halo); `a_finite_sample_beyond_the_bound_acts_like_the_bound` covers the bound.
- **Not measured yet.** Speed on a real GPU is a Mac check (M0-H4 in `docs/15-milestones.md`). The Metal parity, tile and determinism suites run on the Apple-silicon CI runners (`docs/17-gpu.md` sections 1 and 13).

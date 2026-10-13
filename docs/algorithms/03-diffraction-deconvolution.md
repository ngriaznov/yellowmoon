# 03 · Diffraction deconvolution (capture sharpening)

A second R8 capture-sharpening method for Yellowmoon, beside Richardson–Lucy. It computes the
diffraction blur of the lens from the f-number and the pixel pitch, and removes it from luma in one
noise-weighted local least-squares step.

- Reference: `reference/diffraction_deconvolution.cpp` (C++17, `-DSELF_TEST`, `-DGEN_VECTORS`)
- Vectors: `vectors/diffraction_deconvolution/*.json`
- Suggested Rust home: `ym-gpu/src/cpu/sharpen/diffraction.rs`

## 1. Purpose and placement

### 1.1 What the stage does

Stopping a lens down blurs the image by diffraction. That blur depends only on the f-number, the
wavelength and the pixel pitch, so it can be computed rather than guessed. The stage:

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
  resample). It runs wherever Richardson–Lucy runs today: the full-resolution chain, and
  camera-processed Lossless M/S files after R7. Like RL, it does not run on the binned fit view or
  thumbnails.
- **Selection.** R8 becomes `CaptureSharpen::{RichardsonLucy(SharpenParams),
  Diffraction(DiffractionParams)}` in `Chain` and `Stage::Sharpen`. Exactly one method runs
  (either/or, §2.5). Richardson–Lucy stays the default until Mykyta decides (RA-H2), and the
  selection is internal (no settings-schema change).
- **Fallback.** If Diffraction is selected but the stage is inactive (§3.1: no f-number, an
  f-number outside [1, 50], no usable pixel pitch), R8 runs Richardson–Lucy with its resolved
  `SharpenParams`.
- **Module layout.** Turn `sharpen.rs` into `sharpen/mod.rs` (RL code unchanged) and add
  `sharpen/diffraction.rs`.

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
reference works in DN.

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
- Declare the per-frame value from the plan, or the constant 25. Do **not** apply Richardson–Lucy's
  24 px cap (`SHARPEN_HALO_CAP_PX`): at the exact halo the reference's tiles equal the whole frame
  bit for bit.

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

Suggested Rust types:

```rust
/// R8 diffraction deconvolution controls (docs/algorithms/03).
pub struct DiffractionParams {
    /// Sensor pixel pitch (x, y), micrometres.
    pub pixel_pitch_um: [f64; 2],
    /// Lens f-number.
    pub f_number: f64,
    /// Image pixels per sensor pixel, (0, 1].
    pub scale: f64,
    /// Blend between input and corrected luma, 0..=3. 1 is the specified behaviour.
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

`NoiseDn { g, r }` is the shared working-scale noise type of RA-02 (§2.3).

### 2.2 Pixel pitch

Yellowmoon's data had no pixel pitch when this spec was written; since RA-07 the registry has it for 27 of its 28 bodies (§9). Take it from, in this order:

1. **Camera registry (preferred).** Add `sensor_width_mm` and `active_width_px` to each entry of
   `bodies` in `data/cameras/registry.json`. Both describe the full sensor, from the maker's
   published specification (sensor size and effective pixel count), with the source recorded in the
   entry. Then `pitch_x = pitch_y = 1000·sensor_width_mm / active_width_px` µm. An optional
   `pixel_pitch_um` field may override this when the maker states the pitch. A camera crop mode
   (APS-C mode on a full-frame body) does not change the pitch. Example: a 35.7 mm wide sensor with
   9504 active columns has a 3.76 µm pitch.
2. **EXIF crop factor (fallback).** With `crop = focal_length_35mm / focal_length_mm` from
   `RawMetadata`, and `W × H` the frame as R8 receives it (active area in pixels):
   - for 3:2 frames, `pitch ≈ 36000 / (crop · W)` µm;
   - for any aspect ratio, `pitch ≈ 43266.6 / (crop · √(W² + H²))` µm (the 35 mm diagonal is
     43.2666 mm; this equals the 3:2 form for 3:2 frames).

   Use both pitches equal. `focal_length_35mm` is an integer, so the pitch carries an error of up
   to about 2 % at short focal lengths. That is small next to the kernel's radius steps.
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

This is RA-02's `NoiseDn { g, r }`. The per-plane offsets `δ_c` are not used.

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

The R3 vignetting gain multiplies signal and noise in the corners. This weight ignores it (open
question 3).

### 2.4 White level and clip penalty

In Yellowmoon's normalised data, white is 1.0 before the gains, so on the working scale it is
`W = S = 65535` in unbalanced units. The clip threshold is `T = 0.8·W = 52428` DN, applied to the
unbalanced value `x_c / g0_c`. In normalised terms: a channel counts as near clipping when
`x_c / g0_c > 0.8`. The penalty (§3.4) raises the gate threshold, so clipped highlights and the
values R5 reconstructed are not sharpened into artefacts.

### 2.5 Strength, the Sharpening Amount, and Richardson–Lucy

- **Specified behaviour: `strength = 1`.** The algorithm has no amount control of its own. It
  corrects by the fitted amount, which is 100 %.
- **Strength.** `L_s = Y + strength·(L − Y)`, where `L` is the corrected luma (§3.11).
  `strength = 0` returns the input exactly. Values above 1 extrapolate linearly, and overshoot
  grows in proportion.
- **Optional mapping to the Sharpening Amount** (if Mykyta wants one control for both methods):
  `strength = A / 100`, with `A` from 0 to 300. The Auto amount for this method should then resolve
  to `A = 100` at every ISO, because the noise gate already adapts to the noise. RL's Auto curve
  (120 falling to 60) does not apply. `radius_px` has no meaning here: the PSF comes from the
  optics. Default until decided: not mapped, `strength = 1`.
- **Composition with Richardson–Lucy: either/or.** One R8 method runs per frame. Stacking the two
  is an evaluation option (§7.1, item 8).

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
        (any exact method; the reference uses cofactors)

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
  solve) stays within 1e-5 of the reference on the vectors and within 2e-6 on 72 further synthetic
  scenes (pitch 2.4–5.93 µm, f/1.4 to f/32, three noise levels). The vector tolerance is 2e-5.
- **Bessel J1:** any implementation with absolute error below 1e-10 on [0, 120] works. The
  reference uses the power series below 12 and the Hankel asymptotic expansion above (error below
  1e-12). A platform `j1` is fine.

## 4. Edge cases

- **Borders.** Every window (envelopes, box means, Gaussians, PSF) reads with Yellowmoon's mirror
  rule, without repeating the edge sample (`ym_gpu::cpu::mirror`, period `2(n − 1)`). Windows wider
  than the image are mirrored repeatedly. A dimension of 1 reads its single sample.
- **Tiny images.** Any size from 1×1 works (vector `tiny_7x5`; self-test sizes 1×1, 2×3, 1×17).
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
  `ω = 1`). With finite input this needs pathological moments; the self-test sweeps extreme inputs
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
  thread split. Split work by output rows, as the rest of `ym_gpu::cpu` does.
- **Tiling.** With `halo = 10 + R + r_G`, a tile run with that much context, keeping the core,
  equals the whole-frame run bit for bit in the reference (self-test, three f-numbers, five tile
  positions including frame edges). Running window sums that restart per tile row are not bit-exact
  but stay far inside the 1e-4 contract. With less context the outermost taps are tiny: 1 px less
  changed nothing on the test scenes, 4 px less up to 4e-5. Declare the full halo anyway.
- **Cost per pixel.**
  - Envelopes: 25 reads.
  - 17 box means, O(1) each with running sums: 2 for the envelopes, 11 for the moments, 4 for the
    coefficients.
  - 2 separable Gaussians, `2·(2r_G + 1)` taps each.
  - 2 PSF convolutions, `(2R+1)²` taps each (81 at R = 4, at most 225). The 8-fold symmetry folds
    them to at most 36 distinct taps. Alternatively compute `Z = K⊗Y` once and `F2 = Z − F1`; this
    agrees within rounding and saves one convolution.
  - About 100 flops for the solve.

  Roughly 1 kflop per pixel at R = 4 with direct separable box sums.
- **Memory.** About 25 f32 planes of the tile are live at the peak; many can be dropped early. A
  512×512 core with a 25 px halo needs about 562² × 25 × 4 B ≈ 32 MB per tile in flight.
- **Planner.** 159 × 499 Bessel evaluations plus a few thousand trig calls, once per frame (under
  10 ms). The kernel, R and σ_G depend only on `(p_eff, N)` and can be cached by them.
- **GPU port.** All passes are per-pixel gathers; there are no atomics or reductions:
  1. luma, `w`, penalty;
  2. envelopes (5×5 gather);
  3. box means, separable with workgroup shared memory. Batch the 10 products of §3.8 in one pass
     and box all 11 planes together;
  4. layer split;
  5. Gaussian (separable);
  6. PSF (2-D gather; kernel in a uniform buffer, R ≤ 7);
  7. solve (per pixel; write the branches as selects);
  8. coefficient box, correction and transfer.

  Metal has no f64. Expect up to about 1e-5 against the CPU reference (the all-f32 measurement in
  §3.13); a WGSL parity gate of 5e-5 absolute leaves margin.

## 6. Test plan

### 6.1 Vectors

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

### 6.2 Property tests (the reference's self-test; port them)

1. **Flat fields exact** (bit-equal): levels 0, 0.18 grey, a colour, above white (1.9, 1.3, 1.6
   and 3.5, 2.0, 2.9), slight negatives.
2. **Identities exact:** `strength = 0`; inactive stage (f/64, f/0.99, NaN, pitch ≤ 0).
3. **Tiling parity:** tiles with `halo` context equal the whole frame (reference: difference 0;
   contract: ≤ 1e-4) at f/2.8, f/22 and f/45 (2.4 µm), for tiles in the interior and on the frame
   edges.
4. **Square symmetries:** transpose, flip x and flip y commute with the stage (≤ 1e-5; reference: 0).
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
   three channels) grows by at most 5 % at f/8 (reference: ×1.019 worst of ISO 100 and 6400).
   Beyond about f/11 at 3.76 µm it grows (§7.1, item 1). Report the numbers; do not gate on them.
10. **Sharpening works:** a step blurred by the PSF (f/16, 3.76 µm, ISO 100 noise) gets closer to
    the sharp step (reference: summed squared error 5.30 → 3.31) and steeper (10–90 % rise
    2.62 → 2.04 px).
11. **Bessel J1** against tabulated values (≤ 1e-12 in the reference).

### 6.3 Quality gate in Yellowmoon (RA-07)

- **Synthetic.** Take sharp scenes (`synth::fine_text`, `synth::natural`, a slanted edge at 5°).
  Blur them with the diffraction PSF of a given pitch and f-number (f/5.6, f/8, f/11, f/16, f/22).
  Add model noise at ISO 100, 800 and 6400 with channel-correlated luma noise. Run R8 as
  Richardson–Lucy Auto, then as Diffraction. Report:
  - MSE/PSNR against the sharp scene;
  - slanted-edge MTF50 and 10–90 % rise distance;
  - overshoot and undershoot as a percentage of the step;
  - luma noise SD in flat patches, relative to the input.
- **Real files.** The local ARWs, and any at f/2.8, f/8 and f/16: edge profiles (10–90 % rise),
  overshoot, and flat-patch noise. Put crops under "Waiting on Mykyta" (RA-H2).
- **Suggested acceptance as a selectable method:** at f/8 and above, MTF50 at least that of RL
  Auto, with no more overshoot and no more flat-patch noise. Below f/5.6 the two methods should be
  close, because the diffraction PSF is then smaller than a pixel.

## 7. Evaluation notes and open questions

### 7.1 Evaluation notes (A/B candidates; none is the specified behaviour)

1. **The gate opens on noise at small apertures.** The gate threshold is fixed at `0.5·P̄`, but the
   white-noise value of `Q` (§3.8) rises with the PSF width: 0.40 at f/8, 0.58 at f/11, 0.75 at
   f/16 and 0.85 at f/22 (3.76 µm). When the luma noise matches the model, flat-patch noise
   therefore grows ×1.21 at f/11, ×1.56 at f/16 and ×1.79 at f/22. With uncorrelated channel noise
   (luma variance well below the model) there is no growth. How much R7 leaves decides the real
   effect, so measure it (§6.3). A candidate fix: scale the gate by the white-noise level. With
   `q_w = 1 − 2·K₀ + Σ K²` from the plan and `f = max(1, 2·q_w)`, use `t = 0.5·f·P̄` and full gain
   at `Q = f·P̄`. On the synthetic step at f/11 to f/32 this kept flat-noise growth at ×1.07 to
   ×1.08, and the edge error improved slightly (f/16: 3.37 → 3.32; f/22: 4.86 → 4.78).
2. **Spectral weight.** The specified weight rises towards blue (1.99 at 461 nm, flat from
   540 nm), which makes the PSF slightly smaller than a luminance-weighted one. A/B a photopic-like
   weight, `w(λ) = exp(−285.4·(λ/1000 − 0.559)²)` (a Gaussian fit to the photopic curve; the
   reference implements it as `SpectralWeight::kPhotopic`). At 3.76 µm it moves r80 by about 3 %
   (f/8: 3.71 → 3.84 µm; f/16: 7.43 → 7.67 µm), the centre tap by up to 6 % (f/8: 0.412 → 0.396;
   f/16: 0.166 → 0.156) and σ_G by up to 5 %. R is unchanged except in a few cases (5.93 µm at
   f/8: 2 → 3). A further option is the luma-weighted camera spectral sensitivity times a daylight
   spectrum, from the profiled bodies' data.
3. **Luma weights.** The specified weights are 0.299/0.587/0.114 on linear camera RGB. Yellowmoon's
   `LUMA` (0.2126, 0.7152, 0.0722) puts more weight on green, the densest and least noisy CFA
   channel. The weights enter `Y`, `d` (noise), the clip penalty and the transfer, so switch them
   all together. Expect small differences on neutral edges and larger ones on saturated red and
   blue edges.
4. **Noise correlation.** The specified weight assumes fully correlated channels
   (`d = Σ a_c·g0_c`). The uncorrelated form is `σ0² = r²·Σ (a_c·g0_c)²` and
   `σ1 = g·Σ a_c²·g0_c` (neutral pixel), close to Yellowmoon's `luma_sigma_rgb`. A third option
   is Yellowmoon's per-pixel σ map, which includes the vignetting gain: `w = 1/(S·σ_map)²`.
   Compare at high ISO, where the gate matters.
5. **Kernel radius.** The rule (§3.2) keeps one outer ring below 0.2 % of the peak. Using `R − 1`
   would cut the halo by 1 px and the PSF convolutions by about a quarter. Check the effect on edge
   profiles.
6. **Two-luma form.** The stage can take a separate base luma `Y_b` (for example a smoother luma
   from R7) beside the data luma `Y_d` (the luma of the RGB). Then `D`, `B`, `Hp` and `Z` come from
   `Y_b`; `ε = Y_d − Z`, `w` and `Pc` come from `Y_d`; `L = Y_b + …`; and the transfer uses `Y_d`.
   The layers and the high-pass are then taken from the smooth base, so the noise in `Y_d` reaches
   the output only through the 9×9-fitted coefficients, and the output luma carries the base's
   noise level. That is a possible answer to item 1 if R7 can provide such a luma. Default:
   `Y_b = Y_d = Y`.
7. **Ratio or additive transfer.** RL adds `Y' − Y` to all channels; this stage multiplies
   (capped at 1.5). The ratio keeps hue and saturation on brightened edges but scales chroma noise
   with the gain. Compare saturated edges (red and blue on dark) for fringes.
8. **Stacking with RL.** Defocus and lens aberrations add blur that the Airy model does not
   contain. Try Diffraction followed by RL at half its Auto amount, against either alone. Measure
   MTF50, overshoot and noise.
9. **Preview noise factor.** `s = min(2·scale, 1)` only matters for reduced images. If R8 ever runs
   below full resolution, compare it with `binning_noise_factor(k)²`.

### 7.2 Open questions (default in force meanwhile)

1. **Fallback when inactive** (no f-number or no pitch): Richardson–Lucy with the resolved Auto
   parameters, or no capture sharpening? *Default: RL.*
2. **Amount mapping** (§2.5)? *Default: none, strength 1.*
3. **Vignetting gain in the noise weight.** Corners carry `G(r)²` more variance than the model.
   *Default: ignored, as specified.*
4. **Pixel pitch source order.** *Default: registry, then EXIF crop factor, else inactive.*
5. **Clip test.** R1's clip mask is not available in RGB at R8, so the stage uses the 0.8 threshold
   on unbalanced values. In vignetting-corrected corners this can flag unclipped pixels, which only
   means less sharpening there. *Default: threshold as specified.*
6. **Lenses with a missing or wrong f-number** (manual lenses, adapters, teleconverters).
   *Default: trust EXIF; a missing or zero f-number makes the stage inactive.*
7. **Default R8 method.** *Default: Richardson–Lucy until RA-H2.*

## 8. Reference notes

Choices the reference makes that an implementer might otherwise make differently:

- **Borders:** Yellowmoon's mirror rule (no repeated edge sample) for every window.
- **Precision:** f64 accumulation for every window sum, convolution and product; an f64 solve; an
  f64 planner (the line-spread positions are averaged in f64, and `r_G` is computed exactly from the
  stored f32 `σ_G`).
- **Variance:** the signal term uses `max(Y, 0)`, as `NoiseModel::pixel_variance` does, so the
  weight stays positive on negative luma.
- **Degenerate solve:** a non-finite coefficient leaves the pixel alone (all three coefficients 0,
  `ω = 1`).
- **Single luma:** the base and data luma are the same plane `Y` (§7.1, item 6 describes the
  two-luma form).
- **Output in normalised units:** `out = in·k + off/S`, so flat fields, `strength = 0` and inactive
  plans are bit-exact identities.
- **Strength:** an addition; `strength = 1` is the specified behaviour.
- **Gate inputs:** `scale` outside (0, 1] reads as 1; `g0` must be positive and finite.
- **Splat bounds:** contributions outside the 8×8 quadrant are dropped on every side (the upper
  bound never triggers).
- **Bessel J1:** the reference's own series and asymptotic expansion (error below 1e-12); a library
  `j1` gives the same vectors within tolerance.

## 9. Implementation notes (RA-07, 2026-10-13)

Where it lives: `ym-gpu/src/cpu/sharpen/diffraction.rs` (planner, stage, `DiffractionParams`, `DiffractionPlan`),
`diffraction/eval.rs` (edge, MTF and noise measures of the A/B, not used by the render path), `CaptureSharpen` in
`sharpen.rs`, `Stage::Sharpen(CaptureSharpen)` and the per-method halo in `tiling.rs`, the Diffraction arm of
`Chain::run_tile`. In `ym-core`: `pipeline::raw::{pixel_pitch, diffraction_params, CaptureMethod,
DEFAULT_CAPTURE_METHOD, RawParams::{diffraction, capture_sharpen}}` and `PhotoSource::{sensor, capture_method}`.

- **Fidelity.** The Rust follows the reference operation by operation (planner `f64`, planes `f32`, `f64` accumulation
  in the reference's order, no `mul_add`). All 13 image vectors agree **bit for bit** (maximum difference 0, tolerance
  2e-5); the 10 planner vectors agree to 1.3e-8 (tolerance 1e-6); the numbers the reference's self-test prints
  (squared error 5.30 to 3.31, rise 2.62 to 2.04 px, noise x1.019 at f/8, x1.21, x1.56, x1.79 at f/11, f/16, f/22,
  luma of the output against `L_s` 8.1e-8, tiling parity 0, scale 1.1e-7) come out the same. The loops are organised by
  rows so that they vectorise; each output sample sums the same terms in the same order.
- **Bessel J1** (the one piece the spec leaves open): written here from Abramowitz and Stegun, the power series 9.1.10
  below 12 and the Hankel expansion 9.2.5 above, terms summed until they grow. No library, so no new dependency
  or notice. Error against Bessel's integral on `[0, 120]`: 8.6e-13.
- **Noise.** `NoiseDn::r` that is `NaN` or infinite (a negative or infinite variance in `NoiseDn::from_model`) reads as 0,
  so `sigma0` sits at its 1 DN floor, as the reference reads a bad variance; a negative `g` reads as 0. (For an infinite
  variance the reference itself declares the stage inactive; the integration brief asks for 0, which is what is done.)
- **Halo.** `DiffractionParams::halo_px()` is the plan's `10 + R + r_G` (0 when the stage is inactive), independent of the
  gains and the noise; the planner caches kernel, `R`, `sigma_G` and halo by `(p_eff, N)` (eight entries), so the scheduler
  can ask per tile. The 24 px cap belongs to Richardson-Lucy only. A tile with the halo equals the whole frame bit for bit,
  and so do the 128-row bands of `DiffractionPlan::apply`, with one thread or four.
- **Selection.** Internal: `RawParams::diffraction` is `Some` only when `PhotoSource::capture_method` asks for it
  (`DEFAULT_CAPTURE_METHOD` is Richardson-Lucy) and the photo has an f-number from 1 to 50 and a pitch; otherwise, and
  when the plan is inactive for the gains and the noise model, R8 is Richardson-Lucy with the resolved Auto values
  (open question 1 stays as decided). `Chain` given an inactive Diffraction stage directly leaves the image unchanged.
  No settings, IPC or export change; default renders are byte-identical.
- **Pixel pitch.** `pixel_pitch` order: the registry body's `SensorGeometry` (`pitch = 1000 * sensor_width_mm /
  active_width_px`; for a Lossless M or S file `scale = frame_width / active_width_px`), else the crop factor
  `43266.6 / (crop * sqrt(W^2 + H^2))` with `W x H` the frame R8 receives (`scale = 1`, also for a reduced file), else
  none; a pitch outside 0.5 to 20 um counts as not available (a registry value there is ignored, the crop factor then
  tries). The crop factor is 2 % high on APS-C (nominal 1.5 against 1.53), as the spec says to expect. The registry
  values are Sony's published figures, read through search excerpts of the pages (`docs/03`); 27 of 28 bodies have them.
- **Noise-gate variant** (§7.1 item 1), `DiffractionPlan::with_noise_gate(true)`: `f = max(1, 2 q_w)` with `q_w` from
  `white_noise_residual()`; the gate threshold and the full-gain point of `P` scale by `f`. Identical to the specified stage
  up to f/8 at 3.76 um. Flat patch with luma noise matching the model, specified against variant: f/11 x1.208 against
  x1.081, f/16 x1.559 against x1.075, f/22 x1.786 against x1.071, f/32 x1.995 against x1.074 (the spec quotes x1.07 to
  x1.08). Not used by the product; for the A/B.
- **A/B** (`cargo run -p ym-gpu --example ra07_ab`, and the ignored `ra07_real_files_ab` of `ym-core`; crops and
  `index.html` in `testdata/ra07/`, git-ignored). On synthetic scenes at 3.76 um the suggested acceptance of §6.3 is met
  for MTF50 from f/16 up and for overshoot and noise only by the gate variant: at f/16 and f/22 the stage has more
  overshoot (3 to 10 % against 0.6 to 3.5 %) and, with the specified gate, more flat noise (x1.58 to x1.90 against
  x1.14 to x1.26) than Richardson-Lucy Auto, and the gate variant fixes the noise (x1.07 to x1.12). At f/8 it has less
  overshoot (1.7 against 7.0 %) and no flat noise growth (x1.005 against x1.255) but a lower MTF50 (0.340 against 0.380
  at ISO 100): Richardson-Lucy sharpens more at and below f/8, where the Airy blur is small. Details in STATUS.
  On the real files the stage changes little, because the gate is closed on real flat areas (the R6 denoise and R7 leave
  less noise than the raw model says) and real edges are wider than the Airy PSF alone.

Open questions raised by the work (for the owner): (1) which of the two noise gates for the Diffraction method if it ever
becomes selectable; (2) whether to stack Diffraction and Richardson-Lucy (§7.1 item 8): on the real files Richardson-Lucy
alone sharpens the clean f/8 edges more than the Airy model does, which suggests the real blur is wider than diffraction
alone; (3) the pixel-width figures marked `restated` in the registry (A7 III, A7 IV, A7C II, A7S III, A1) should be read
against Sony's own pages when a connected machine is available.


## 10. As built on the GPU (RA-09, 2026-10-13)

Where it lives: `ym-gpu/src/shaders/diffraction.wgsl` (the per-pixel stages as plain functions), `ym-gpu/src/kernels/diffraction.rs`
(`PreparedDiffraction`: upload, nine pipelines, tiling, `run_traced`), `ym-gpu/tests/gpu_diffraction.rs` (parity). It is not
wired into the render path: the raw stages still run the CPU reference.

- **Split.** The planner stays on the CPU (it runs once per `(pitch, f-number)` and is cached there). `PreparedDiffraction::new`
  takes the `DiffractionPlan` and uploads its `f32` kernel and the Gaussian taps (computed in `f64` with the CPU's expression,
  rounded to `f32`); `sigma0^2`, `sigma1`, the strength, the gate factor, the gains `g0` and the luma weights all come from the
  plan (`DiffractionPlan::g0()` and `luma()` give back the `f32` values it was made with), so the kernel cannot be run with others than the CPU's. Eleven dispatches over nine pipelines, as §5 lists them: luma, `w` and `Pc`;
  envelopes; layers; Gaussian rows and columns (for `Y`, then for `Z = F1 + F2`); PSF convolutions and the residual; the 9 x 9 moments
  as a row pass (the ten products `(w u) v` are formed at each tap, never stored) and a column pass fused with the solve; the 5 x 5
  coefficient averages, the correction and the transfer over the tile's core. 124 bytes per pixel of compute rectangle, planes `f32`
  in every storage mode (the envelope counts need exact `f32` order).
- **Precision.** Window sums, convolutions, products and the Cramer solve are `f32`, in the CPU's order (x first, then y; PSF taps row
  by row; Gaussian taps from the left). Measured on lavapipe: the 13 image vectors 8.9e-7 (tolerance 2e-5), random frames of 11 optics
  x 6 scene kinds 5.1e-6, other strengths/scales/gates/luma weights/gains 3.6e-6, values from -0.05 to 4 with strength 3 4.3e-6,
  sizes 1 x 1 to 7 x 7 9.2e-7, 2048 x 1365 through 512 px tiles 4.8e-7 (nothing above 1e-6 in 8.4 million samples). A replica of the CPU
  port with the 9 x 9 moments summed in `f32` and the solve done in `f32` (same inputs otherwise) moved the corrected luma by at most
  4e-4 DN on the worst pixels of the large frame and 0.011 DN (1.6e-7 normalised) on a hard noise-free edge scene; the ridge keeps the
  3 x 3 system within a condition number of about 2000 (`(tr + lambda) / lambda` with `lambda = 0.0005 tr`).
- **The luma has to be exact.** §3.13's "all `f32` stays within 1e-5" holds on small scenes, but on a 2048 x 1365 natural frame an
  `f32` luma (1 to 2 ulp off the CPU's in a third of the pixels) gave 9.1e-5 at a handful of pixels, 7 of them above 5e-5: a neighbour
  within an ulp of the centre flips the `>= 0` / `<= 0` counts of §3.5, and one count moves the pixel's envelope by up to 8 %.
  The CPU port with only its luma replaced by the `f32` formula reproduces the GPU's worst pixels to 0.5 %. The shader forms `Y` as an
  exact expansion (Dekker products, Knuth sums, `f32` only) and gets the CPU's bits (0 differences in 240000 pixels); the large frame
  falls to 4.8e-7. It needs every rounded intermediate hidden from the shader compiler (an xor with a uniform that is always 0): the
  plain transformations returned error terms of exactly 0 on Mesa's lavapipe, and `fma` is not an alternative there (lowered to a
  multiply and an add). Other drivers must be checked (M0-H4): `the_luma_plane_has_the_cpu_bits` fails if they fuse through the
  barrier, and the stage then degrades to the 1e-4 level of the plain `f32` sum.
- **Tiling.** A tile with the plan's exact halo (`10 + R + r_G`) gives exactly the bits of the whole-frame GPU run (tile sizes 32 to
  100 over 203 x 157 at halos 13, 18 and 24); two runs are bit-identical in every storage mode. The perturbation reach found on the GPU
  is 13 px at a 15 px halo and 17 at 19 (the outermost taps of the Gaussian and the PSF are tiny).
- **Finding for the spec (not changed here).** On a noise-free smooth ramp every window is evenly spread, so §3.5 gives
  `lower = upper = mid` up to rounding, `Y` sits within a rounding of both thresholds `Lm` and `Hm` of §3.6, and the dark/bright
  split is decided by rounding. The CPU port itself moves by up to 3.5e-3 when such an input changes by 0.6 ulp, so no port can match
  it there. §4 says flat fields come back exact, which holds; smooth noise-free gradients (a synthetic wedge without noise) are the
  case it does not cover. Real frames and the vectors carry sensor noise (`sigma0 >= 1` DN), and the parity scenes here do too.

### Independent review of RA-09 (2026-10-13)

- **Against the C++ oracle on new scenes.** Six scenes that are not in the vectors (random discs, bars and gradients blurred by the
  plan's PSF, with model noise; 131 x 97 at 4.3 um f/13, 64 x 73 at scale 0.5 with Rec. 709 luma and strength 1.7, 120 x 89 at 1.9 um
  f/45, 5 x 3, a 1 x 61 strip, 160 x 110 at f/9 and low ISO): the CPU port equals the oracle bit for bit (difference 0), the GPU is
  within 1.2e-6 of both.
- **Luma bits.** 0 differences from the CPU's `f64` luma in 17.3 million further samples (uniform -0.05 to 4, 0 to 1, 0 to 0.05, and
  14-bit grey, balanced and unbalanced values), in addition to the 240000 of `the_luma_plane_has_the_cpu_bits`.
- **The ill-conditioned case is real.** On a noise-free scene quantised to 14 bits (a smooth gradient with hard-edged shapes) the GPU
  differs from the CPU by up to 4.3e-3 in a handful of samples. Changing the last mantissa bit of every input sample moves the C++
  oracle itself by up to 4.4e-3 on that scene (599 of 36864 samples above 1e-4, 15 above 1e-3), so no port can do better. With one
  quantisation step of noise the GPU-CPU difference is 1.1e-6, and the oracle's own sensitivity to a one-ulp change of every sample is
  5e-4 (which also breaks the exact ties that the GPU keeps, because its luma has the CPU's bits). Whoever wires the GPU path into
  the viewport should keep the export on the CPU path (AGENTS.md rule 6): a preview and an export can differ in the last digits of such
  regions, though not in the noisy frames measured here.
- **Metal.** `wgpu-hal` 30.0.1 builds the library with `MTLCompileOptions` that set only the language version and `preserveInvariance`;
  no math mode is chosen, so Metal's default (fast math) applies. The xor with the runtime zero keeps the expansion intact under
  re-association and contraction, but this stays unverified until the Mac run (M0-H4: `the_luma_plane_has_the_cpu_bits` and the large
  frame).
- **Tests added by the review.** `hostile_values_that_the_stage_acts_on_match_the_cpu` (the existing hostile test is an identity at f/1
  and f/50, because the clip penalty gates the frame or the fit fades out, so it proved little about the solve: values -0.05 to 0.7 with
  and without 1 % spikes up to 4, f/1 to f/8, the CPU must change the frame, |gpu - cpu| at most 1.3e-6) and
  `a_non_finite_sample_stays_local_and_does_not_panic` (a NaN, an infinity or 1e30 changes nothing beyond the halo).

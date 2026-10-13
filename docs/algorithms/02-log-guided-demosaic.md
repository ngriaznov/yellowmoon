# 02 · Log-guided fast demosaic

A fast Bayer demosaic for previews and thumbnails. It builds a luminance guide in the log domain, then fills each missing channel from the mean of its neighbours scaled by the local guide ratio. Green at red and blue sites blends a horizontal and a vertical estimate with inverse-cube structure weights. The fixed-point form needs only 16-bit integers, two cubic polynomials, five small neighbourhood passes and one division per interpolated sample.

| | |
|---|---|
| Package name | Log-guided fast demosaic |
| Yellowmoon home | `Method::LogGuided` in `ym-gpu/src/cpu/demosaic.rs` (R6, classical family) |
| Reference | `reference/log_guided_demosaic.cpp` (`demosaic_fixed`, `demosaic_float`) |
| Vectors | `vectors/log_guided_demosaic/*.json` (16 files) |
| Halo | 5 px (receptive-field radius) |

---

## 1. Purpose, place in the chain, contract

### 1.1 What it is for

`docs/01` (Preview strategy) renders the fit view and the thumbnails with the classical path, and the full-resolution tiles are slow to arrive. This method stands in for Menon wherever those renders demosaic at full resolution. It costs a few times a bilinear demosaic, and in the reference's tests it gains 1.5 to 4.6 dB PSNR over bilinear (section 6.4). It is not meant for export.

Where it runs in Yellowmoon:

- **R6, classical family**, selectable as `Method::LogGuided` next to `Bilinear`, `Menon` and `Lmmse`. The bilateral denoise that follows classical R6 (`denoise::bilateral_opponent`) is unchanged.
- **Renders**: the fit view, thumbnails and look tiles wherever they demosaic at full resolution. That covers thumbnails of small photos or small crops (`k = 1`) and the coarse frame shown while full-resolution tiles are computed. The as-built raw group bins the mosaic at `k ≥ 2` (`bin_mosaic`) and does not run R6 there. Whether to replace binning at `k = 2` with "this method, then a 2× downsample" is an evaluation question (section 7.1, note 6), not part of this spec.
- **Input**: the **balanced** mosaic that R6 already receives, after R4 (gains `g0`) and R5 (highlight reconstruction). Do not divide by `g0` (section 1.3).

### 1.2 Input and output

| | |
|---|---|
| Input | `ym_gpu::cpu::Mosaic`: one f32 per photosite, row-major, black-subtracted, white = 1.0 before gains, `Cfa` phase of the buffer's (0, 0), `Placement`. |
| Input range | Balanced values. Values above 1.0 occur: R3 divides by the vignetting gain, R4 multiplies red and blue by `g0` (typically 1.5 to 2.5), and R5 writes reconstructed highlights above the clip level. Negatives occur down to about −0.05·g_c (R1 clamps at −0.05 before the gains; R3 can lift that further in the corners). |
| Output | `ym_gpu::cpu::RgbImage`: planar f32 (R plane, G plane, B plane), **balanced** camera RGB, same width, height and `Placement` as the input. |
| Native samples | The sample at a photosite's own colour is copied unchanged from the input, including values above 1.0 and negatives. A non-finite input sample is written as 0. |
| Interpolated samples | Always ≥ 0. In the fixed-point form they are also ≤ H, the headroom (section 2). In the all-float form there is no upper limit. |
| Clip mask | Not used. |
| Noise model | Only the read-noise part (`b`, `sigma_row`, `q`) is used, to set the offset `p` (section 2.2). |

### 1.3 Balanced input, and why it is fine

The method expects the balanced mosaic. There is no `g0` round trip, which the adaptive ratio demosaic (`spec/01`) needs.

The guide is a geometric mean of the three channels, so white balance enters it only as a constant factor. In the log domain the gains add `log g_c` at the photosites of colour `c`. That pattern is a constant, (log g_R + log g_B)/4, plus components at the CFA carrier frequencies (π, 0), (0, π) and (π, π). For every value of the detail parameter, the guide kernel's response is exactly 0 at π on each axis (section 2.3), so only the constant survives. Without the offset `p`, the balanced guide is the unbalanced guide times (g_R·g_B)^(1/4) everywhere.

Every later use of the guide is scale-free. Each estimate multiplies a ratio of guide values (`C / HC`, `C / VC`), and the green blend is homogeneous of degree 0 in the guide and in the structure weights. So the gains cancel, and running on the balanced mosaic gives the same result as running on the unbalanced one and applying `g0` afterwards. Three absolute quantities break the cancellation slightly: the offset `p` (added before the log and to the structure measures), the integer rounding of the fixed-point form, and its clamp at the headroom.

Balanced input is the better of the two for these reasons:

- On balanced data a neutral surface has equal R, G and B, so `p` damps the three channels equally. On unbalanced data, the red and blue samples of a grey are smaller by 1/g_R and 1/g_B and sit closer to the offset.
- `p` comes from the read noise of green, and green's gain is 1, so `p` is the same in balanced and unbalanced units.
- R6 must output balanced RGB, and this way nothing has to be divided out and multiplied back.

Measured with the reference's float form (a 64×64 texture with noise, `g0` = (2.1, 1, 1.7)): balanced input and "unbalanced input, then times `g0`" differ by at most 3.5e-4 (mean relative 2e-5 at full scale and 0.4 % at 1/100 scale, where `p` matters). The noiseless PSNR is identical at full scale and 0.03 dB higher for balanced input at 1/100 scale.

### 1.4 Halo

The halo (receptive-field radius) is **5 px** (Chebyshev). Derivation: the reconstruction (3.7) reads the mosaic and the guide at ±1 px and the smoothed structure measures at the pixel itself. The 5×5 box (3.6) reads the structure measures at ±2. These read the guide at ±1 (3.5), so the guide is read at up to ±3. Each guide value uses the log image at ±2 on each axis (3.3), which gives ±5 in total. Everything else is pointwise.

An implementation that shrinks its valid area by 2 px after each of the four neighbourhood stages (horizontal filter, vertical filter, structure, box) needs an 8-px border. That is a safe bound, but 5 is exact: the self-test perturbs single photosites and finds a reach of exactly 5 in both forms. Declare `Method::LogGuided.halo() = 5`. The scheduler aligns mosaic rectangles to even coordinates itself. In the `docs/05` halo table, the R6 classical row becomes 5 + 2 (bilateral) = 7 px for this method.

---

## 2. Parameters

### 2.1 Parameters

| Name | Type | Default | Range | Meaning |
|---|---|---|---|---|
| `guide_detail` | i32 | 0 | −50 … 100 (clamped) | Sharpness of the luminance guide. Sets the outer tap `k` of the guide kernel (2.3). |
| `headroom_log2` | u8 | per frame (2.4) | 0 … 4 (clamped) | Fixed-point form only. Headroom H = 2^headroom_log2: the normalised value that maps to the top of the 16-bit working scale. |
| `noise` | `NoiseModel` | the photo's model | — | Normalised, unbalanced units. Only `b`, `sigma_row` and `q` are used. |

### 2.2 Derived values

| Symbol | Formula | Notes |
|---|---|---|
| S | 65535 | Working scale: `x_dn = S·x / H` in the fixed form, `x_dn = S·x` in the float form. |
| k | −trunc(2.048·P) = −((2048·P) div 1000), with integer division that truncates toward zero | P = `guide_detail` after clamping. P = 100 → −204, 50 → −102, 25 → −51, 0 → 0, −25 → 51, −50 → 102. |
| K | (k, 256, 512 − 2k, 256, k) / 1024 | The guide kernel, used horizontally and vertically. Taps sum to 1024. |
| H | 2^headroom_log2 | 1, 2, 4, 8 or 16. |
| r_dn | S·sqrt(max(0, b + sigma_row² + q²/12)) | Read-noise sigma in DN at S = 65535. A negative or NaN variance counts as 0. Green's gain is 1, so this is green's read noise in balanced and unbalanced units alike. |
| p (fixed form) | clamp(floor(r_dn / (2H)), 1, 32767) | Integer. Log offset and structure regulariser in the fixed form's units. |
| p (float form) | clamp(floor(r_dn / 2), 1, 32767) | The same with H = 1. |

The log input is `p + x/2` (3.2), so at the input's full scale the offset is `2p ≈ r_dn`: the guide is the log of "signal plus one read-noise sigma". In the shadows this pulls the guide ratios toward 1, so the interpolation there tends toward a plain average of the neighbours and does not transfer noise. `p` is also the floor of the structure measures, which keeps the inverse-cube weights finite and stops them from chasing noise in flat areas. The shot-noise term `a` and `delta` are not used.

### 2.3 `guide_detail`

The kernel K has frequency response

  K(ω) = [(512 − 2k) + 512·cos ω + 2k·cos 2ω] / 1024,

with K(0) = 1 and **K(π) = 0 for every k**. The Bayer carriers are removed exactly, so the guide never contains the mosaic pattern. At the default k = 0 the kernel is [1, 2, 1]/4 on taps −1…1. The separable 3×3 weights give red, green and blue the totals 1/4, 1/2 and 1/4 at every site, so the guide is (log R + 2·log G + log B)/4 of the local neighbours: a geometric-mean luminance (R·G²·B)^(1/4).

| P | k | Kernel (/1024) | K(π/4) | K(π/2) | Peak gain |
|---|---|---|---|---|---|
| 100 | −204 | −204, 256, 920, 256, −204 | 1.252 | 1.297 | 1.375 at 0.40π |
| 50 | −102 | −102, 256, 716, 256, −102 | 1.053 | 0.898 | 1.055 at 0.28π |
| 25 | −51 | −51, 256, 614, 256, −51 | 0.953 | 0.699 | 1 (at 0) |
| 0 | 0 | 0, 256, 512, 256, 0 | 0.854 | 0.500 | 1 (at 0) |
| −50 | 102 | 102, 256, 308, 256, 102 | 0.654 | 0.102 | 1 (at 0) |

Effect:

- **Positive values** make the outer taps (same colour as the centre, two photosites away) negative. This lifts mid frequencies: more fine detail is transferred into the interpolated channels, and the result looks sharper. It also adds noise, and overshoot and zipper risk at hard edges.
- **Negative values** widen the guide and soften the result. They help very noisy files.
- The range stops at −50 because K(π/2) reaches 0 at k = 128 (P ≈ −62) and turns negative beyond it. It stops at 100 because the mid-frequency gain is already 1.38 there.
- **Default 0.** It gives the plain geometric-mean guide, and sharpening belongs to R8 and to output sharpening. Do not expose the parameter in the UI. Keep it as an internal parameter for evaluation (7.1, note 7).

### 2.4 Headroom (fixed form only)

The fixed form maps normalised values to 16-bit integers: `x_dn = round(S·x / H)`, clamped to [0, 65535]. H must be at least the largest value that should survive the interpolation. Balanced mosaics routinely exceed 1.0 (section 1.2), and H = 1 would clip every bright red and blue. That is shown by vector `highlights_headroom1`.

**Default rule.** H is the smallest power of two with H ≥ the maximum sample of the frame's balanced mosaic after R5. It is capped at 16: values above 16 then saturate in the interpolated channels, while their native samples still pass through. Compute H **once per frame**, before tiling, and pass the same H to every tile, otherwise tiles disagree. If the frame maximum is not available, use H = 4.

The cost of a larger H is quantisation: one code is H/65535 (6.1e-5 at H = 4, about the step of a 14-bit raw file). Scaling the input by 2 and doubling H and the read noise gives exactly twice the output (a self-test property). The **float form needs no headroom** and is the recommended form (3.9).

### 2.5 Scale-dependent quantities

Everything is defined at the working scale S = 65535, divided by H in the fixed form. Quantities that depend on that scale:

- `p` (log offset and structure floor): `p_dn = r_dn / (2H)`, with `r_dn = 65535·sqrt(b + sigma_row² + q²/12)`. If Yellowmoon ever uses another working scale S', multiply `r_dn` by S'/65535.
- The integer rounding steps of the fixed form (one code = H/S in normalised units).
- The clamp of the fixed form at 65535 codes (= H normalised).
- The guard `max(·, 1)` on the structure measures before cubing (one code). It never binds, because the measures are ≥ p ≥ 1.

Scale-free quantities: the kernel, the box size, the cube exponent, every ratio, and the log2/exp2 constants. These act on the mantissa and are exact under ×2 (L(2y) = L(y) + 4096).

---

## 3. Algorithm

### 3.0 Notation

- The core maths is written in **padded coordinates**. Mirror the input once by B = 5 px on every side, with Yellowmoon's `mirror` (period 2(n−1), the edge sample is not repeated, parity is kept). Every stage below is symmetric under reflection (symmetric kernels, symmetric rounding), so mirroring each intermediate plane at the buffer edge instead gives identical results (an independent per-stage-mirroring implementation reproduces every vector bit for bit).
- `rhalf(a, b) = floor((a + b + 1) / 2)` for non-negative integers (the rounded halving add).
- `>> n` is floor division by 2^n of a non-negative integer. All products in steps 3.2 and 3.4 fit in 32-bit unsigned integers (the largest are 65534² and 65520², both below 2^32).
- **Site classes.** Read them from the buffer's `Cfa` at the output pixel: red site, blue site, green site in a red row (its left and right neighbours are red), green site in a blue row. The formulas use these classes, not parities, so all four phases work without shifting. Equivalently, shift the buffer so red sits at even/even. The reference uses the classes.

### 3.1 Quantise (fixed form)

For every padded photosite, with v the input sample:

  A = 0 if v is not finite or v·S/H ≤ 0; 65535 if v·S/H ≥ 65535; otherwise round_half_even(v·S/H).

Evaluate v·S/H in binary64. H is a power of two, so the product is exact.

### 3.2 Log transform

  y = p + floor((A + 1)/2)      (1 ≤ y ≤ 65535)
  L = log2_fixed(y)            (L ≈ 4096·log2 y; 4096 codes per stop)

**log2_fixed(y)** for 1 ≤ y ≤ 65535 (define log2_fixed(0) = 0; it is never reached):

```
e  = floor(log2 y)                      // index of the highest set bit, 0..15
n  = y << (15 − e)                      // 32768 ≤ n ≤ 65535
m  = 2n − 65536                         // mantissa fraction in Q16, 0..65534
m2 = (m·m) >> 16
m3 = (m2·m) >> 16
P  = ((46597·m) >> 16) − ((19037·m2) >> 16) + ((5208·m3) >> 16)   // 0..32767
L  = 4096·e + (P >> 3)
```

P/32768 approximates log2(1 + f) with f = m/65536, using the cubic 1.422027587890625·f − 0.580963134765625·f² + 0.158935546875·f³ (the constants are 46597/32768, 19037/32768 and 5208/32768). Error bounds, checked exhaustively in the self-test:

- The cubic alone: −0.00096 … +0.00070 stop.
- log2_fixed including truncation: L − 4096·log2(y) ∈ [−4.98, +2.98] codes, i.e. −0.00122 … +0.00073 stop, for every y in 1 … 65535. Anchors: L(1) = 0, L(2) = 4096, L(65535) = 65535.
- L is not strictly monotone. In the top octave it steps back by 1 code at 23 values of y. This is harmless and part of the specified function.
- Doubling y adds exactly 4096.

### 3.3 Guide filter

Horizontal pass, for padded columns 2 … W'−3 (W' = width + 10) and all padded rows:

  Lh(x, y) = clamp( floor( (Σ_{i=0..4} K_i·L(x − 2 + i, y) + 512) / 1024 ), 0, 65535 )

Vertical pass on Lh, for padded rows and columns 2 … size − 3:

  Lv(x, y) = clamp( floor( (Σ_{j=0..4} K_j·Lh(x, y − 2 + j) + 512) / 1024 ), 0, 65535 )

K_i are the integers (k, 256, 512 − 2k, 256, k). The sum needs a signed 32-bit accumulator (|Σ| ≤ 1840·65535 < 2^31). Floor and truncation toward zero agree after the clamp, because a negative quotient clamps to 0. The order is horizontal first, then vertical. Swapping it can move L by a code or two, so the order is part of the definition.

### 3.4 Back to linear: the guide C

  C = exp2_fixed(Lv)     (C ≈ 2^(Lv/4096); 1 ≤ C ≤ 65522)

**exp2_fixed(L)** for 0 ≤ L ≤ 65535:

```
e  = L >> 12                            // integer part, 0..15
f  = (L mod 4096) << 4                  // fraction in Q16, 0..65520
f2 = (f·f) >> 16
f3 = (f2·f) >> 16
Q  = ((45607·f) >> 16) + ((14741·f2) >> 16) + ((5185·f3) >> 16)   // ≤ 65509
s  = 2^e
C  = s + ((s·Q) >> 16)
```

Q/65536 approximates 2^f − 1 using 0.6959075927734375·f + 0.2249298095703125·f² + 0.0791168212890625·f³ (45607/65536, 14741/65536, 5185/65536). Error bounds, checked exhaustively:

- The cubic alone: relative error −9.7e-5 … +1.38e-4.
- exp2_fixed with t = 2^(L/4096): −(1 + 1.18e-4·t) ≤ C − t ≤ +1.24e-4·t. The "1" is the final truncation, which matters only for small t.
- exp2_fixed(4096·e) = 2^e exactly. The function is monotone non-decreasing, and its maximum is 65522, so no saturation is needed.

C lives in the domain of `p + x/2` (half scale plus offset). Only ratios of C are used later, so the factor ½ does not matter.

### 3.5 Structure measures

For padded positions 3 … size − 4 on both axes:

  Dh = min( |rhalf(C(x−1, y), C(x+1, y)) − C(x, y)| + p, 65535 )      (horizontal second difference)
  Dv = min( |rhalf(C(x, y−1), C(x, y+1)) − C(x, y)| + p, 65535 )      (vertical second difference)

### 3.6 Box smoothing

5×5 box mean of each measure, for padded positions 5 … size − 6 (the output core):

  D̄ = floor( Σ_{5×5} Dh / 25 ),   Ē = floor( Σ_{5×5} Dv / 25 )

The sums fit in 32 bits (≤ 25·65535). The box is separable: a horizontal and a vertical 5-sum, then one division.

### 3.7 Reconstruction

For each output pixel (padded position X, Y = pixel + 5), let A be the quantised mosaic and C the guide:

  HA = rhalf(A(X−1, Y), A(X+1, Y)),   VA = rhalf(A(X, Y−1), A(X, Y+1))
  HC = rhalf(C(X−1, Y), C(X+1, Y)),   VC = rhalf(C(X, Y−1), C(X, Y+1))

HA(X±1) and the others are the same quantities evaluated at the left and right neighbours. The weights are

  w_h = max(Ē, 1)³   (weight of the horizontal estimate: the cube of the *vertical* measure)
  w_v = max(D̄, 1)³   (weight of the vertical estimate: the cube of the *horizontal* measure)

Both are read **at the pixel itself**, at red and at blue sites alike. Dividing the numerator and denominator by w_h·w_v turns them into the inverse-cube weights 1/D̄³ and 1/Ē³: the direction with the smaller second difference wins.

| Site | R | G | B |
|---|---|---|---|
| red | native | **green** | **diag** |
| green, red row | **ratio**(HA, HC) | native | **ratio**(VA, VC) |
| green, blue row | **ratio**(VA, VC) | native | **ratio**(HA, HC) |
| blue | **diag** | **green** | native |

with

```
ratio(a, c) :  q = C / c;                          v = q · a
green       :  num = w_h·HA + w_v·VA
               den = w_h·HC + w_v·VC;   q = C / den; v = num · q
diag        :  sa  = VA(X−1) + VA(X+1)              (the four diagonal neighbours)
               sc  = VC(X−1) + VC(X+1); q = C / sc;  v = sa · q
```

Evaluate in **binary64** in the order shown: the cube as an exact product (≤ 2.9e14 < 2^53), then `q`, then the product. Then round: v ≤ 0 or NaN → 0, v ≥ 65535 → 65535, otherwise round_half_even(v). All denominators are ≥ 1, because C ≥ 1 and the weights are ≥ 1, so NaN cannot occur.

In plain terms: a missing channel is the mean of that channel over the neighbours that have it, times the ratio of the guide at the centre to the mean guide over the same neighbours. This is colour-ratio (constant-hue) interpolation from a log-domain luminance guide.

### 3.8 Output (fixed form)

- Native channel: the input sample itself (0 if it is not finite).
- Interpolated channels: `out = (n·H) / 65535`, evaluated in binary64 and rounded to f32. n is the integer from 3.7.

### 3.9 All-float form (recommended for the GPU and acceptable on the CPU)

The maths is the same without fixed point. The differences:

| Step | Float form |
|---|---|
| 3.1 | `A = 65535·max(v, 0)`, non-finite → 0. **No upper clamp and no headroom.** |
| 3.2 | `y = p + A/2` (p with H = 1); `L = log2(y)` (natural floating log2). |
| 3.3 | `Lh = Σ (K_i/1024)·L`, `Lv = Σ (K_j/1024)·Lh`, no rounding. |
| 3.4 | `C = 2^max(Lv, 0)`. The lower clamp mirrors the fixed form (C ≥ 1). It only acts when a sharpened guide undershoots below one code. |
| 3.5 | `Dh = |(C(x−1) + C(x+1))/2 − C| + p`, same for `Dv`. |
| 3.6 | `D̄ = Σ Dh / 25`, `Ē = Σ Dv / 25`. |
| 3.7 | Exact halves `(a + b)/2` instead of `rhalf`. The same formulas, no rounding, no clamp. |
| 3.8 | Interpolated `out = v / 65535`. Native samples as in the fixed form. |

Precision: f32 storage is enough for every plane. On the CPU, accumulate the filters and the box in f64 (Yellowmoon convention, and what the reference does). On the GPU use f32 throughout. To keep the green blend's magnitudes small in f32, compute it as

  t = (D̄ / Ē)³,   green = C · (HA + t·VA) / (HC + t·VC),

which equals the formula above up to rounding.

**Expected difference between the two forms** (reference, 40×40 scenes, H = 2, normalised units): mean |Δ| 7e-6 on straight edges and 2e-5 to 4e-5 on texture and zone plates. Max |Δ| 4.5e-5 on straight edges, 1.7e-4 on a diagonal edge, 5e-4 on the texture scene, and 1.6e-3 on a zone plate near Nyquist, where small guide errors change large ratios. PSNR differs by less than 0.01 dB. The sources are the log2/exp2 approximations (≤ 0.0012 stop, mostly cancelling in the ratios), the integer L (1/4096 stop), `rhalf` and output rounding (½ code each), the floor in the box (−½ code on average), and in the fixed form the clamp at H. The vectors carry both `expected` (fixed) and `expected_float`. The self-test bounds the difference at max 2e-3 and mean 5e-5.

---

## 4. Edge cases

- **Borders.** The input is mirrored with parity kept (3.0), so the CFA pattern continues across the edge. Results at the border are well defined, and tiles that touch the frame edge mirror the frame itself.
- **Tiny buffers.** Any width and height ≥ 1 work. The mirror wraps several times when the buffer is narrower than the halo (vector `tiny_6x5`). With width or height 1 a colour is missing from the buffer and the output is defined but meaningless. Keep Yellowmoon's dispatch rule that sends buffers under 2×2 to bilinear.
- **Zero input.** A field of zeros has `y = p` everywhere, a constant guide and output 0.
- **Negatives.** Interpolation and the guide see `max(v, 0)`. In the fixed form this is forced by the unsigned scale, and the float form does the same so the two agree. Native samples keep their negative value. The interpolated channels of a deep-shadow noise floor are therefore biased upward, by about 0.4 σ for zero-mean noise. For a preview this is usually acceptable; 7.1 note 3 gives the alternative.
- **Values above 1.0.** The float form keeps them. The fixed form keeps them up to H (2.4) and saturates interpolated samples at H. Native samples always pass through.
- **Non-finite samples.** They count as 0 everywhere.
- **Clipped photosites.** There is no special handling. R5 has already replaced clipped values by estimates, and the clip mask is not read.
- **CFA phases.** All four (RGGB, GRBG, GBRG, BGGR) are handled by site class (3.0). Cropping a buffer by one row or column changes its phase, and the interior result is bit-identical (self-test).
- **Overshoot.** The guide ratio can push an estimate above the largest neighbour (or below the smallest) at strong edges. That is the intended detail transfer. The only clamps are at 0 and, in the fixed form, at H.

---

## 5. Determinism, tiling, performance, GPU

**Determinism.** There are no reductions and no data-dependent iteration order. Every output sample is a fixed function of the mosaic within 5 px, so threading cannot change results. The fixed form is bit-reproducible across platforms when step 3.7 is done in binary64. An f32 evaluation of 3.7 changes about 0.05 % of samples, by one code at most (measured on 122 000 samples).

**Tiling.** A tile with ≥ 5 px of context, mirrored at the tile buffer's edge, gives the whole-frame result bit for bit in both forms. The self-test checks 9×8 cores at odd origins, where every tile has its own phase. In the fixed form, H must be the frame's value (2.4). With the banded runner, `filters::run_mosaic_bands(m, 5, BAND_ROWS, …)`.

**Cost per output pixel.**

| Step | Work |
|---|---|
| 3.1–3.2 | One quantise and one log2 (about 10 integer ops). |
| 3.3 | 2 × 5 multiply-adds. |
| 3.4 | One exp2 (about 8 integer ops). |
| 3.5 | Two 3-tap second differences. |
| 3.6 | About 4 adds with running sums, plus one division per measure. |
| 3.7 | About 15 flops and one or two divisions. |

Planes: A, L, Lh, C, Dh, Dv, D̄, Ē, as u16 in the fixed form. They can stream through 5-row ring buffers. It should cost a few times bilinear and much less than Menon. Measure it in RA-06.

**GPU (WGSL).** Every step is per pixel over a small neighbourhood, with no global reductions. The headroom H is a frame constant passed as a uniform, and only the fixed form uses it. A natural split:

1. log and horizontal filter (log on the fly at ±2);
2. vertical filter and exp2 → C;
3. structure measures and the horizontal 5-sum of the box;
4. vertical 5-sum, division and reconstruction.

Alternatively, run one dispatch over a shared-memory tile of 16×16 plus 5 px of halo. Use the float form. WGSL `log2` and `exp2` accuracy (a few ULP) is far below the fixed form's approximation error. The fixed form can also be ported exactly with `u32`/`i32` arithmetic if bit parity with the vectors is wanted, with 3.7 in f32 (±1 code).

---

## 6. Test plan

### 6.1 Vectors (`vectors/log_guided_demosaic/`)

Each file holds `name`, `description` and `params`. `params` has `guide_detail`, `headroom_log2`, `noise` and `derived` (H, r_dn, p for both forms, k, the kernel). The other fields:

- `input`: width, height, cfa, row-major data.
- `expected`: fixed form, planar RGB.
- `expected_float`: float form, planar RGB.
- `tolerance`: for `expected`, one code of the working scale, 1.01·H/65535. A binary64 implementation of 3.7 is bit-exact; an f32 one is within one code.
- `tolerance_float`: 1e-4 absolute, for `expected_float`.

| File | Size | Covers |
|---|---|---|
| `flat_grey` | 16×16 RGGB | Flat 0.25 on the code grid: every output equals the input exactly. |
| `flat_colour` | 16×16 BGGR | Flat balanced colour (0.42, 0.30, 0.18): each plane exact. |
| `edge_horizontal`, `edge_vertical`, `edge_diagonal` | 32×32 RGGB | Anti-aliased two-colour edges: directional green and ratio transfer. |
| `zone_plate` | 40×40 RGGB | Tinted circular chirp up to Nyquist: aliasing behaviour. |
| `phase_rggb`, `phase_grbg`, `phase_gbrg`, `phase_bggr` | 32×32 | The same texture scene (gradient, diagonal edge, 4-px stripes, coloured disc, maximum 1.23) in all four phases; headroom_log2 = 1. |
| `detail_plus100`, `detail_minus50` | 32×32 RGGB | The guide kernel at both ends of its range. |
| `noisy_dark` | 32×32 GRBG | Dark texture with Poisson–Gaussian noise (a 2e-4, b 4e-8, sigma_row 1e-4, q 2e-4, gains 2.1/1/1.7), R1 clamp. p > 1, negatives pass through at native sites. |
| `highlights_headroom4` | 32×32 RGGB | A source up to 3.4 after reconstruction plus slight negatives, H = 4: nothing saturates. |
| `highlights_headroom1` | 32×32 RGGB | The same with H = 1: interpolated samples saturate at 1.0, native ones keep their value. |
| `tiny_6x5` | 6×5 BGGR | The mirror wraps several times inside the halo; guide_detail 40. |

### 6.2 Property tests (port these to Rust)

The reference's `-DSELF_TEST` runs all of them.

1. **Flat field exact.** Constant grey and constant colour on the code grid, all four phases, H = 1 and 4, guide_detail −50/0/100: the fixed form is bit-exact, the float form within 1e-6 relative.
2. **Tiling parity.** Halo 5, cores at odd origins: bit-exact in both forms (Yellowmoon's limit is 1e-4).
3. **CFA-phase equivariance.** Crop the frame by (1, 0), (0, 1) and (1, 1). The interior (≥ 5 px from the cut) is bit-identical to the uncropped result.
4. **Mirror equivariance.** Horizontal, vertical and both flips of the mosaic flip the output: bit-exact in the fixed form (the symmetric weight read of 3.7 is what makes this hold), 1e-6 relative in the float form.
5. **Receptive field.** Perturbing one photosite changes outputs up to exactly 5 px away and no further.
6. **Scale.** In the float form, signal and read noise ×2 give output ×2 (≤ 1e-6 relative). In the fixed form, signal, read noise and H ×2 give output ×2 exactly, and at the same H the result stays within 32 codes.
7. **Native passthrough and no NaN.** Random inputs in [−0.05, 20] with NaN and infinity samples, every headroom: all outputs finite and native samples unchanged.
8. **Headroom.** Values above H saturate only in interpolated channels.
9. **Fixed vs float.** The bounds of 3.9.
10. **log2_fixed and exp2_fixed.** The bounds and anchors of 3.2 and 3.4, checked exhaustively.

### 6.3 Determinism

`--jobs 1` and `--jobs 4` must give byte-identical output.

### 6.4 Quality gate in Yellowmoon

- **PSNR**: 10 px border excluded, `metrics::psnr`, mosaicked as RGGB. Compare against Bilinear, Menon and LMMSE on the synthetic set of `tests/cpu_demosaic.rs`, and on Kodak, McMaster and PixelShift200 where they can be fetched. Run in linear light and at signal scales 1.0, 0.1 and 0.01, as `docs/05` did for LMMSE. Use the float form and p from a zero-noise model (p = 1).
- **What the reference shows** (40×40 scenes, float form): edges +4.6 dB over bilinear (36.4 vs 31.9 dB), diagonal edge +1.8, zone plate +3.5, texture +1.5. Its position against Menon was not measured here. The RA-06 gate (within 1 dB of Menon on the synthetic set, < 300 ms for the 1/4-resolution fit preview on M1) decides whether it replaces the current preview path. Report the numbers either way.
- **Visual**: zipper and false-colour metrics (`cpu/metrics.rs`) against Menon, and the fit view of the nine local real files at `guide_detail` 0.

---

## 7. Evaluation notes and open questions

### 7.1 Evaluation notes (worth A/B testing)

1. **Offset `p`.** It comes from green's read noise. Alternatives: the guide's effective noise, `r·(g_R·g_B)^(1/4)`; per-colour offsets `g_c·p`, which make the method exactly white-balance-equivariant; or a continuous (non-integer) `p` in the float form. Measure shadow noise and shadow colour on the noisy synthetic scenes.
2. **Headroom.** A per-frame H against the float form, for the fixed form's integer-only use. The float form needs none.
3. **Negatives.** In the float form, let the neighbour means (HA, VA and the diagonal sums) use unclamped values and keep the guide on `max(v, 0)`. This removes the upward bias of interpolated channels in the noise floor. Check black-level neutrality of the fit view on dark frames.
4. **Weights.** Inverse cube against inverse square, and a 5×5 against a 3×3 box. These trade zipper suppression on edges against noise robustness.
5. **Chroma at red and blue sites** is the plain average of the four diagonal neighbours times the guide ratio, with no direction. A directional or green-referenced estimate may reduce false colour on diagonals.
6. **Demosaic then downsample vs binning** for the `k = 2` fit view: sharpness, chroma-edge energy (`docs/01` uses it for binning) and time.
7. **`guide_detail`** at 25 to 50 for thumbnails that are downsampled afterwards, where part of the sharpening is lost.
8. **Overshoot limiting.** Clamp each interpolated sample to [min, max] of the neighbours it was built from, widened by a margin. This trades detail for fewer halos at saturated colour edges.

### 7.2 Open questions (with the default to use meanwhile)

| Question | Default meanwhile |
|---|---|
| Which renders switch to this method? | `k = 1` preview renders (thumbnails of small photos or crops, the coarse frame before full-resolution tiles), after the RA-06 gate passes. Binning stays for `k ≥ 2` until note 6 is measured. |
| Fixed or float form in the product? | Float form on CPU and GPU. Fixed form for the vectors and as an integer-only fallback. |
| How is H chosen when the frame maximum is unknown? | H = 4 (headroom_log2 = 2). |
| Does `p` follow the pooled noise model or the photo's own? | Whatever `RawParams::noise` carries (the pooled model when it is used). |
| Should `guide_detail` follow the sharpening setting? | No. Fixed at 0. |

---

## 8. Reference notes

The reference implements this specification as written. Where the behaviour could have been defined in more than one way, it uses the following:

- **Symmetric weights at blue sites.** Green at a blue site uses the structure weights of the blue site itself, exactly as at a red site. A variant that reads them from the left-hand neighbour is not part of this specification: it widens the receptive field to 6 px horizontally and breaks mirror equivariance, and it changes most blue-site green values (by up to 13 000 codes at sharp noisy edges in random tests).
- **Box mean as an exact integer quotient**, `floor(Σ / 25)`. Multiplying by the single-precision value of 1/25 and truncating gives the same result for every reachable sum (all 0 … 25·65535 checked), so a GPU may use that.
- **Reconstruction in binary64.** Single-precision evaluation in the order of 3.7 differs by at most one code, in about 0.05 % of samples.
- **Borders.** The input is mirrored once (Yellowmoon's `mirror`). The core stages contain no border logic of their own.
- **CFA phase** by site class rather than by shifting the buffer (the two are equivalent).
- **Native samples** are copied from the f32 input, so they are not quantised and keep negatives and values above H.
- **Parameter mapping.** `p` from the noise model per 2.2. `k` by exact integer arithmetic, which equals the single-precision product `trunc(P·(−0.002)·1024)` for every P in −200 … 200.

---

## 9. As built (RA-06)

Decisions and measurements recorded when the method was implemented (`crates/ym-gpu/src/cpu/demosaic/log_guided.rs`). The specification above is unchanged.

### 9.1 API and choices

- **`Method::LogGuided`** runs the **all-float form** (3.9) with `guide_detail = 0`. The noise model comes from `DemosaicInputs::noise`; `demosaic(m, method)` therefore uses the zero-noise model (`p = 1`). The gains and the CA curves of `DemosaicInputs` are ignored. The input is the balanced mosaic: nothing is divided out.
- **Entry points** in `log_guided`: `demosaic(m, noise)` (float, detail 0), `demosaic_float(m, noise, guide_detail)`, `demosaic_fixed(m, noise, guide_detail, headroom_log2)`, `derive`, `log2_fixed`, `exp2_fixed`, and `LogGuidedParams` with the default headroom rule of 2.4 (`LogGuidedParams::for_mosaic` reads the frame maximum; the fixed form is not used by any render path, so no caller computes it yet).
- **Halo** `Method::LogGuided.halo() = LOG_GUIDED_HALO = 5`; the perturbation test finds a reach of exactly 5 in both forms.
- **Invalid noise** (`NoiseDn::r` NaN, infinite or negative, which `NoiseDn::from_model` gives for a negative or infinite variance) reads as 0, so `p = 1`, as 2.2 says. `derive` does this, not `NoiseDn`.
- **Where the Rust differs from the oracle's text** (none of it changes a vector): the 5x5 box is a vertical 5-sum then a horizontal one (every partial sum is exact in `f64`, so the order cannot matter); the structure weights are computed only at red and blue sites, the only place 3.7 reads them; `log2` and `exp2` are the platform's `f64` functions rounded to `f32`. Nothing is fused (`mul_add` is not used).
- **Output rule** for a mosaic whose data length does not match its size: a black image of the stated size (the oracle returns an empty one). Buffers under 2x2 still go to bilinear in `demosaic_with`; `log_guided::*` itself accepts any size from 1x1.

### 9.2 Results

- **Vectors**: all 16, both forms, difference 0.0 against `expected` and `expected_float` (tolerances: one code and 1e-4). Property tests of 6.2 (flat fields, tiling, phase and mirror equivariance, receptive field, scale, passthrough, headroom, fixed against float, the exhaustive log2 and exp2 bounds) and the determinism checks (banded against whole, 1 against 4 threads, byte for byte) pass: `crates/ym-gpu/tests/cpu_log_guided.rs`.
- **Fixed against float** on the 40x40 scenes: max 1.6e-3 (zone plate), mean at most 3.9e-5, inside the bounds of 3.9.
- **PSNR** (`cargo run --release -p ym-gpu --example log_guided_eval -- psnr`; 10 px border, peak scaled with the signal, `p = 1`). Mean dB, log-guided against Menon:

| Set | Scale | Bilinear | Menon | LMMSE | Log-guided | Log-guided minus Menon |
|---|---|---|---|---|---|---|
| Synthetic (7 images) | 1 | 29.28 | 31.60 | 32.00 | 30.76 | -0.84 |
| Synthetic | 0.1 | 29.28 | 31.60 | 31.37 | 30.80 | -0.80 |
| Synthetic | 0.01 | 29.28 | 31.60 | 30.16 | 30.87 | -0.73 |
| Kodak (24) | 1 | 29.82 | 38.88 | 39.16 | 31.94 | -6.95 |
| Kodak | 0.1 | 29.82 | 38.89 | 38.34 | 32.19 | -6.69 |
| Kodak | 0.01 | 29.82 | 38.88 | 38.13 | 32.43 | -6.46 |
| McMaster (18) | 1 | 32.31 | 34.21 | 35.09 | 33.48 | -0.74 |
| McMaster | 0.1 | 32.31 | 34.21 | 34.61 | 33.54 | -0.68 |
| McMaster | 0.01 | 32.31 | 34.21 | 33.60 | 33.66 | -0.55 |

  The gate (within 1 dB of Menon on the synthetic set) holds at every scale. On Kodak, whose images are saturated and gamma-encoded photographs, the method is 6.5 to 7 dB below Menon, and only 2.1 to 2.6 dB above bilinear; at scale 1 the per-image gap on the synthetic set ranges from +1.9 (zone plate) to -3.1 dB (natural 3). PixelShift200 is not available here.
- **CPU time** (`... -- timing`, release build, mosaic of 1/4, 1/2 and full resolution of a 24 MP (6000x4000) and a 61 MP (9504x6336) frame, 4 shared x86-64 cores at load average 8 to 12, so wall times are inflated and CPU seconds are the dependable number). CPU time relative to bilinear and Menon, float form of this method:

| Frame | Resolution | CPU vs bilinear | CPU vs Menon |
|---|---|---|---|
| 24 MP | 1/4 (1500x1000) | 1.7 to 1.8 | 0.49 to 0.51 |
| 61 MP | 1/4 (2376x1584) | 1.7 to 1.8 | 0.40 to 0.54 |
| 24 MP | 1/2 | 1.6 | 0.40 |
| 61 MP | 1/2 | 1.7 | 0.53 |
| 24 MP | full | 1.4 | 0.48 |

  On one thread the 1/4-resolution frame of a 24 MP photo takes about 75 ms wall (90 CPU ms) against 165 to 200 ms for Menon and 42 to 58 ms for bilinear. The fixed form costs 1.0 to 1.6 times the float form. (LMMSE costs 1.1 to 1.8 times Menon.)

### 9.3 Use for the fit view and thumbnails

The numbers meet the targets (within 1 dB on the synthetic set, about twice as fast as Menon), but **no preview render demosaics today**, so there is no preview method to switch: the fit view and the thumbnails at `k >= 2` bin the mosaic (`bin_mosaic`) and never run R6. Only `k = 1` renders do (thumbnails of small photos or small crops), and those read the tiles of the full-resolution chain: the tiles are cached under the same key as the 100 % view, and exports run the same chain. Using this method there needs a second chain with its own cache key, so that its pixels cannot reach the 100 % view or an export, and it would only speed up a rare case. The comparison of 7.1 note 6 (demosaic then downsample against binning at `k = 2`) was not evaluated. `ym-core` is therefore unchanged. The gap on Kodak (-6.95 dB) is a further reason to keep Menon wherever a real photograph is shown at full resolution. The GPU port (RA-09) is the place where this method pays off.

---

## 10. As built (RA-09)

The GPU port of the all-float form (3.9), recorded when it was implemented. The specification above is unchanged. Files: `crates/ym-gpu/src/shaders/log_guided.wgsl`, `crates/ym-gpu/src/kernels/log_guided.rs` (wrapper, `KernelId::LogGuided`), `crates/ym-gpu/tests/gpu_log_guided.rs` (parity tests).

### 10.1 Layout

- **One shader, one pipeline, four dispatches** in the split of section 5, selected by a `pass` field of the uniform: (0) the log taken on the fly at ±2 and the horizontal guide filter, `Lh`; (1) the vertical filter and `exp2`, `C = 2^max(Lv, 0)`; (2) the structure measures `Dh`, `Dv` from `C` and their horizontal 5-sums `SH`, `SV`; (3) the vertical 5-sums, the box means, the weights and the reconstruction. Each dispatch is per pixel over a small neighbourhood; there are no reductions and no barriers. The receptive field found by perturbing single photosites is exactly 5 px, so `Method::LogGuided.halo() = 5` holds for the kernel too.
- **Storage.** `Lh`, `C`, `SH`, `SV` are four `f32` planes of one scratch buffer (16 bytes per pixel) in **every** storage mode: the guide is a logarithm and an `f16` plane of it has steps of 0.008 to 0.016 stop. The mosaic (input) and the three output planes use the device's storage type, as the other raw kernels do, so the kernel can sit between the front end and the colour chain later. With `f16` storage the input and output are rounded to `f16` (the open question on `f16` against `f32` storage, `docs/STATUS.md` question 2, therefore applies to the planes, not to the guide). The `f32` bounds below are for `f32` storage.
- **Entry points** in `ym_gpu::kernels::log_guided`: `demosaic(ctx, &Mosaic, NoiseDn, guide_detail)` (one rectangle, mirrored at its edges), `demosaic_tiled(ctx, &Mosaic, NoiseDn, guide_detail, tile_px, &CancelToken)` and `demosaic_planned(.., &TilePlan, ..)` (tiles with 5 px of context, even left and top edges). They upload, run and read back. `p` and the kernel taps come from `cpu::demosaic::log_guided::derive`, so the numbers are not typed twice. The scratch must fit one storage binding (128 MiB by default, 8 million pixels): a 2048 px tile with its halo needs 68 MB.
- **Not wired into the render path.** The raw stages still run the CPU reference; this task adds the kernel, its wrapper and the registry entry (`KernelId::ALL`, `tests/shaders_naga.rs` validates the module in all storage modes).

### 10.2 Where it is not bit-exact

`f32` against the CPU's `f64`, with the same order of operations except for two exact rewrites.

- `log2` and `exp2` are the GPU's `f32` instructions (a few ulp on lavapipe), not the `f64` library functions rounded to `f32`.
- The guide filters, the box sums and the reconstruction accumulate in `f32`. The filters are evaluated about the centre value (`Lh = Lc + Σ t_i (L_i − Lc)`; the taps sum to exactly 1) and the second differences as `|((C_a − C) + (C_b − C))/2|`. Both are the same function in real arithmetic. They keep the `f32` error relative to the differences of neighbouring values instead of to the values (the log is about 14, `C` is up to 10^5, an ulp of 0.008 against `p = 1`). Without them the weights of a perfectly flat guide differed by rounding noise: on a 1×2 mosaic the green sample was off by 1.1e-3. With them the same case agrees to the last bit.
- The green blend is the `f32` form of 3.9 divided through by the **larger** of the two weights `max(D̄,1)³`, `max(Ē,1)³` instead of always by `w_h`: with `r = min/max` of `max(D̄,1)` and `max(Ē,1)` and `k = r³ ≤ 1`, the weights are `(w_h, w_v) = (1, k)` when `D̄ ≤ Ē` (exactly the spec's `t = (D̄/Ē)³` form) and `(k, 1)` when `D̄ > Ē`, and `green = C (w_h·HA + w_v·VA)/(w_h·HC + w_v·VC)`. It is the same blend. The single `t` form reached `t ≈ 10¹⁵` on a strong horizontal edge (`D̄ ≫ Ē ≈ p`) and `t·VC` overflowed `f32` for mosaic values above about 10⁵, giving NaN where the CPU's `f64` is finite (the review found it with vertical stripes; `huge_values_in_strongly_directional_frames_stay_finite_and_match` guards it).
- The box mean is `Σ/25` (the float form has no floor, so the `floor(Σ·(1/25))` of section 8 does not arise).
- The mosaic is not padded once by 5 px: every read of the mosaic or of a plane is mirrored into the rectangle (period 2(n−1), the edge sample not repeated). Section 3.0 shows this is the same function. Tiles with 5 px of context and the whole frame give the same pixels bit for bit.
- `f32` range: the kernel stays finite up to mosaic values of about 3e32 and overflows near 1e33, where the 25-sample box sum of the structure measures leaves the `f32` range (the CPU's `f32` planes: between 1e33 and 3e33; `A = 65535·v` itself overflows at 5e33). A balanced mosaic is a few times white at most.

### 10.3 Results (lavapipe, `WGPU_BACKEND=vulkan`, Mesa 25.2.8, LLVM 20.1.2; `cargo test -p ym-gpu --test gpu_log_guided`)

Max |GPU − CPU reference|, `f32` storage, in normalised units:

| Case | Asserted | Measured |
|---|---|---|
| 16 vectors against `expected_float` | 2e-5 (the vectors' own tolerance, 1e-4, holds too) | 3.1e-6 (`highlights_*`, values up to 3.4), 1.8e-6 (`zone_plate`), ≤ 8.4e-7 elsewhere |
| random frames in 0..1.3, all four phases, odd and even sizes, `guide_detail` −50, 0, 100, several noise models | 5e-5 | 4.8e-6 |
| frames from −0.05 to 20 with NaN and infinity samples, four phases | 1e-5 of the peak | 1.3e-6 of the peak (native samples bit-exact, interpolated ≥ 0) |
| 1×1 to 7×7, two phases (the mirror wraps inside the halo) | 5e-5 | 1.5e-5 (1×3) |
| 203×157, tiles of 32, 50, 64, 100, two phases | bit-identical to the whole GPU frame; 5e-5 against the CPU | 0; 2.4e-6 |
| 2048×1365 in four tiles of 1024 | 5e-5 | 3.2e-6 |
| crops by one row or column (the phase flips), flips of the mosaic | 5e-5 | 0; 1.2e-7 |
| `f16` storage (packed and native), random frames in 0..1.3 | 5e-5 plus half an `f16` ulp of the value | 4.9e-4 (the half ulp at 1.0) |
| `f16` storage, the 16 vectors (values up to 3.4) | the same | 9.6e-4 (the half ulp at 3.4 is 9.8e-4) |
| stripes of one direction (one structure measure at `p`, the other as large as the signal), values from 4e2 to 4e30, both directions | 1e-5 of the peak, every sample finite | 1.1e-6 of the peak |
| 14 extra inputs of the review against the C++ oracle's `demosaic_float` (sizes 1x1 to 128x96, all four phases, `guide_detail` -50 to 100, several noise models, values from 1e-5 to 136; run outside the repository) | the CPU bound | the Rust CPU port: 0.0 on all; the GPU: at most 2.1e-6 of the peak (9.9e-5 absolute at a peak of 136; 7e-6 on a 97x63 texture with values up to 12) |

Also asserted: flat fields come back within 1e-6 relative in all four phases; a zero field and an all-negative field give zero interpolated samples and unchanged natives; out-of-range `guide_detail` is clamped like the CPU; a plan without context, a plan for another size, an empty mosaic and a cancelled token are refused; two runs (whole and tiled) are bit-identical in every storage mode. Metal and real-GPU runs, and timings on a real GPU, are not done (no Mac or GPU here; `M0-H4`). The 2048×1365 frame takes 1 to 3 s on lavapipe (a CPU rasteriser on shared cores), which says nothing about a GPU.

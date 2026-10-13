# 04 · Output sharpening (linear-light luma unsharp mask with halo control)

| | |
|---|---|
| Reference | `reference/output_sharpening.cpp` (namespace `output_sharpening`) |
| Vectors | `vectors/output_sharpening/*.json` (16 files) |
| Rust home | new `ym-gpu/src/cpu/output_sharpen.rs`, called from the export path right after the resize |
| Ships | off (`ExportOptions::output_sharpen = None`) until the presets are signed off (RA-H3) |

## 1. Purpose and placement

### 1.1 What the stage does

Output sharpening restores the crispness an export loses to resizing, to the display or to print. It is an unsharp mask on one luma plane in linear light:

1. Compute a luma `L = sqrt(wR·R² + wG·G² + wB·B²)` per pixel.
2. For each sharpening stage, take the high-pass `d = L − G_σ * L` (separable, pixel-integrated Gaussian), turn it into a delta with one of three threshold methods, and add `amount × delta` to an accumulator.
3. Optionally route a fraction of every stage's delta through a halo limiter. The limiter clamps that fraction between robust local minimum and maximum envelopes of the luma, so edges do not overshoot past the local extremes.
4. Apply the sharpened luma to RGB as a ratio, `RGB_out = RGB_in · clamp(L'/L, 0, 16)`. Hue and saturation ratios stay the same.

There is no gamma, log or perceptual transform anywhere in the stage. All thresholds are absolute linear levels.

### 1.2 Where it runs in Yellowmoon

Yellowmoon has no output sharpening today. A raster export (JPEG, TIFF, PNG) runs like this:

1. `ym-core/src/export.rs::export_photo` renders C1 to C12 at full resolution in the primaries of the export colour space (`FullTarget::DisplayLinear`). The output is display-linear, interleaved `f32`, with white at 1.0.
2. `ym_export::request::encode_to_writer` calls `resize_lanczos3` (`ym-export/src/resize.rs`) when `Size::LongEdge` asks for a smaller image. The resize is Lanczos-3, separable, in linear light, accumulated in f64, and never upscales.
3. The writers do C13 (the transfer function, and the dither for 8-bit output) and encode.

The new stage goes between 2 and 3, in the same linear space and at output resolution:

```
render C1..C12 (display-linear) -> [Lanczos-3 resize if Long edge] -> OUTPUT SHARPENING -> C13 + dither -> writer
```

- **Long-edge exports:** run the stage on the resized `LinearImage`, right after `resize_lanczos3` returns.
- **Full-size exports:** run it at the same point. The resize is skipped, so the stage takes `req.pixels` and produces a new `LinearImage` for the writers.
- **DNG:** the stage does not run. The raw data stays unsharpened, and the DNG preview is an open question (§7.2).
- **Encoded input:** an `ImageRef` with `PixelEncoding::DisplayEncoded` only comes from library callers. The stage decodes it to linear light first, as the resize does (`space.decode_channel`), and hands the writers display-linear pixels.

Wiring:

- `ExportOptions::output_sharpen: Option<OutputSharpen>`, default `None`. `ExportRequest` carries it into `encode_to_writer`.
- With `None`, the code path and the bytes are exactly as today. RA-08 includes a test for this.
- `ym-export` does not depend on `ym-gpu` today. Adding the dependency creates no cycle, because `ym-gpu` depends only on `ym-color`.
- Progress and cancel follow the resize: check `hooks.is_cancelled()` every 64 rows, and report a share of the encode stage (for example 0.3 to 0.4 when there is a resize).
- The stage is deterministic, so the app and the CLI still write byte-identical files.

Layout: `ym_gpu::cpu::RgbImage` is planar, while the export's `ImageRef` and `LinearImage` are interleaved. Use `RgbImage::from_interleaved` and `to_interleaved`, or run the stage directly on interleaved data. Only the luma (step 1) and the transfer (step 4) touch RGB, and the result must be the same either way.

### 1.3 Input and output contract

| | |
|---|---|
| Input | Display-linear RGB in the export colour space (sRGB/Rec.709, Display P3 or Rec.2020 primaries), white = 1.0, `f32`. It is fully colour-processed (white-balanced, through the display transform). The stage uses no gains and no noise model. |
| Range | Not clamped. Values above 1.0 occur (bright content, about 7 % Lanczos overshoot at steps), and so do slight negatives (Lanczos undershoot, out-of-gamut colours). Both are handled (§4). |
| Output | Same layout, units and size. `RGB_out = RGB_in · ratio` with `ratio` in [0, 16] per pixel. Each channel keeps its sign. The only clipping is a pixel whose sharpened luma goes below 0, which becomes black. |
| Units | Normalised, white = 1.0. |

**Scale.** The package's 16-bit working scale (`S = 65535`, `x_dn = S·x`) does not apply here: every constant of this stage is defined in normalised units. To run the stage at DN, use these conversions:

| Quantity | Normalised | At DN |
|---|---|---|
| Threshold `T` | `levels / 255` | `S · T` |
| Gain-map epsilon | `1e-10` (units of `d²`) | `S² · 1e-10 ≈ 0.4295` |
| Luma floor | `1e-20` | `S · 1e-20` |

Everything else is scale-free. With `T = 0` the stage is homogeneous of degree 1: `f(k·x) = k·f(x)`. For a power-of-two `k` this holds bit-exactly, and a self-test checks it.

### 1.4 Halo (receptive field)

Write `n(σ)` for the tap count (§2.3) and `r(σ) = floor(n(σ)/2)` for the kernel radius. One stage reads:

| Step | Adds |
|---|---|
| High-pass `d` (blur of `L`) | `r(σ)` |
| Gain map, methods 1 and 2 with `T > 0` (blur of a per-pixel map of `d`) | `r(max(2σ, 1))` |
| Method 2 only: 3 × 3 gradient of the blurred luma, which feeds the gain map | `1` |
| Halo limiter, if any stage has halo > 0: 5 × 5 window on `L` and `Hacc` | `2` |
| Halo limiter: blur of the four envelope maps | `r(2s)` |

`halo_px = max over stages(stage chain) + (halo limiter ? 2 + r(2s) : 0)`. The reference computes this exactly (`halo_px(plan)`).

The tap cap bounds it for each mode:

- **Screen** (cap 13 taps, `r ≤ 6`): method 0 needs 6 px, method 1 needs 12, and method 2 needs 13. With halo suppression, add `2 + r(2s) ≤ 2 + 6`: 14 px for method 0, **20 px** for method 1, **21 px** for method 2.
- **Print** (cap 33 taps, `r ≤ 16`): 32 px for method 1 and 33 for method 2. The halo-map kernel has `σ = 2s ≤ 2`, so it is never longer than 13 taps. With halo suppression that gives 40 and 41 px.

For the tiling contract (`tiling::Stage::halo_px`), declare the exact `halo_px(plan)` or the mode bound: **21 px (screen)** or **41 px (print)**. The screen figures 12 px, and 20 px with halo suppression, hold for methods 0 and 1. Method 2 needs one pixel more, because its gradient reads the blurred luma one pixel beyond the blur.

## 2. Parameters

### 2.1 User parameters

| Name | Type | Default (screen preset) | Range | Meaning |
|---|---|---|---|---|
| `mode` | enum | `Screen` | `Screen` = 0, `Print` = 1 | Where the radius comes from, and the tap cap (13 or 33) |
| `method` | enum | `SmoothGain` (1) | 0 `HardThreshold`, 1 `SmoothGain`, 2 `GradientRamp` | How the high-pass is thresholded (§3.3). The output stage uses it as given, with no automatic selection. |
| `amount_pct` | f32 | 100 | 0 to 1000 | Strength in percent. `A = amount_pct / 100`. Below 0.1 % no stage is created. |
| `radius_px` | f32 | 1.0 | 0 to 2.5 (Screen) | Unsharp-mask radius in rendered pixels. Values above 2.5 are capped. Values ≤ 0.09 create no stage. Ignored in Print mode. |
| `threshold_levels` | f32 | 1.0 | 0 to 100 | Threshold in 8-bit levels of the normalised signal. `T0 = threshold_levels / 255` (absolute, linear). |
| `halo_pct` | f32 | 60 | 0 to 100 | Halo suppression: the share of the delta that goes through the limiter, `H = sqrt(halo_pct / 100)` |
| `print_distance` | f32 | 5000 | ≥ 0 (useful to 37 500) | Print mode only: `radius = min(print_distance / 2500, 15)` px |
| `scale` | f32 | 1.0 | (0, 1] | Output long edge / rendered long edge (§2.4) |
| `luma_weights` | [f32; 3] | `[0.299, 0.587, 0.114]` | each finite and > 0 | Weights of the luma (§2.6). A zero weight would make a pure-colour pixel invisible to the luma. |
| `extra_stages` | list | empty | — | More whole-image stages (§2.7). Empty in Yellowmoon v1. |

Validation rejects NaN and out-of-range values, and method values other than 0 to 2.

### 2.2 Derived values of the output stage

```
A      = amount_pct / 100                                          (f32)
R      = Screen: min(radius_px, 2.5)        Print: min(print_distance / 2500, 15)
σ'     = R ≥ 0.5  ?  0.7·R + 0.3
       : R > 0.09 ?  0.97·sqrt(R − 0.09)
       :             0                                             (f32)
stage exists iff σ' ≥ 0.01 and A ≥ 0.001
σ      = scale_sigma(σ', s)                                        (§2.4)
T      = normalise_threshold(threshold_levels / 255, σ', σ)        (§2.4)
H      = h ≤ 0.001 ? 0 : h,  h = sqrt(clamp(halo_pct · 0.01, 0, 1))  (f32)
maxTaps = Screen ? 13 : 33
```

Radius to σ (`s = 1`):

| R (px) | 0.1 | 0.2 | 0.3 | 0.5 | 0.6 | 0.8 | 1.0 | 1.5 | 2.0 | 2.5 |
|---|---|---|---|---|---|---|---|---|---|---|
| σ' | 0.097 | 0.322 | 0.445 | 0.650 | 0.720 | 0.860 | 1.000 | 1.350 | 1.700 | 2.050 |
| taps (cap 13) | 3 | 5 | 5 | 5 | 7 | 7 | 7 | 9 | 11 | 13 |

The map has a small step at R = 0.5: just below it σ' = 0.621, at 0.5 it is 0.650. This is part of the reference behaviour (§7.1).

Halo mapping: 10 → 0.316, 20 → 0.447, 40 → 0.632, 60 → 0.775, 80 → 0.894, 100 → 1.

All derived scalars are computed in f32 exactly as written, so the tap count, which truncates `5σ + 2.5`, is the same in every port. The `erf` terms are computed in f64.

### 2.3 Kernel

```
n0 = (int)(σ·5.0 + 2.5) | 1                (f32 product and sum, truncation toward zero, then set bit 0)
n  = n0 < 3 ? 3 : min(n0, maxTaps)         (odd, 3 ≤ n ≤ maxTaps)
x_i = i − (n − 1)/2,  i = 0 … n−1
k_i = erf((x_i + 0.5)/(σ·√2)) − erf((x_i − 0.5)/(σ·√2))      (f64)
k_i ← k_i / Σ_j k_j                         (f64, sum left to right)
```

Each tap is the Gaussian integrated over its pixel, normalised to unit sum after truncation. Use a full-precision `erf` (any libm `erf`, or the `libm` crate). A low-order approximation shifts the taps by more than the vectors allow. Every vector lists its kernels with `%.17g` so a port can check them directly.

Example, `σ = 1.0`: `[0.005979818, 0.060625743, 0.241842857, 0.383103164, 0.241842857, 0.060625743, 0.005979818]`.

The stage uses three kernels:

| Kernel | σ | Used for |
|---|---|---|
| `G_σ` | the stage σ | the high-pass |
| `G_2` | `max(2σ, 1)` (f32: `σ + σ`, then `≤ 1 → 1`) | smoothing the gain maps of methods 1 and 2 |
| `G_h` | `2s` (f32: `s + s`) | smoothing the four halo-limiter maps |

All three are capped at `maxTaps`. In screen mode, `G_2` is therefore cut to 13 taps (±6 px) from `2σ ≈ 2.3` (σ ≈ 1.15) upward, where the tap rule would ask for more.

### 2.4 Resolution scaling

The radius is in rendered (full-size) pixels. When the export is smaller, σ and the threshold are rescaled to the output grid. In Yellowmoon:

```
s = out_long_edge / rendered_long_edge        (f64 division of the two integers, then f32; 1 for full size)
```

`(out_w, out_h)` come from `Size::output_dimensions`. Use the long edge, because the resize's x and y factors differ slightly after rounding. Yellowmoon never upscales, so `s ≤ 1`. For the output stage:

```
if s < 1 and σ' ≥ 0.15:  σ = s·(σ' − 0.15) + 0.15        (f32)
else:                     σ = σ'
T = (σ == σ') ? T0 : T0 · (1 − c(σ)²) / (1 − c(σ')²),   c(x) = erf(1 / (2·√2·x))     (f64, then f32)
```

`c(x)` is the centre tap of an untruncated pixel-integrated Gaussian, and `1 − c²` is the gain of the 2-D high-pass at its centre. The rescale keeps the threshold meaning the same in terms of edge contrast: a smaller σ has a weaker high-pass, so the threshold shrinks with it. The halo-map kernel uses `σ_h = 2s`.

Example, R = 1.0, threshold 1 level:

| s | 1 | 0.75 | 0.5 | 0.4 | 1/3 | 0.25 | 0.1 |
|---|---|---|---|---|---|---|---|
| σ | 1.000 | 0.788 | 0.575 | 0.490 | 0.433 | 0.363 | 0.235 |
| taps | 7 | 7 | 5 | 5 | 5 | 5 | 3 |
| T (levels) | 1.000 | 0.908 | 0.728 | 0.610 | 0.510 | 0.360 | 0.077 |
| σ_h (taps) | 2.0 (13) | 1.5 (11) | 1.0 (7) | 0.8 (7) | 0.67 (5) | 0.5 (5) | 0.2 (3) |

A 6000 px frame exported at a 2048 px long edge (`s = 0.341`) with the screen preset gets σ = 0.440 and T = 0.523 levels.

### 2.5 Presets (need Mykyta's sign-off, RA-H3)

The presets, their numbers and whether the Export screen offers them are UI decisions and Mykyta's call. Until he decides, the stage ships off.

| | Export for screen | Print |
|---|---|---|
| `mode` | Screen (radius cap 2.5, 13 taps) | Print (radius from distance, 33 taps) |
| `method` | 1 `SmoothGain` | 1 `SmoothGain` |
| `amount_pct` | 100 | 150 |
| `radius_px` / `print_distance` | 1.0 px | 5000 (R = 2 px, σ' = 1.7) |
| `threshold_levels` | 1 | 1 |
| `halo_pct` | 60 | 0 |
| `luma_weights` | reference `[0.299, 0.587, 0.114]` | same |

Rationale (judgement calls):

- **Screen.** Halo 60 and the 2.5 px cap are the screen-mode values of the reference behaviour. Radius 1.0 and amount 100 % sharpen visibly at full size and gently after a downscale. Method 1 with a 1-level threshold leaves low-contrast noise mostly alone.
- **Print.** Halo suppression is 0 in print mode in the reference behaviour, because print diffusion softens overshoot. A larger radius and amount make up for that softening.

Measured on a soft slanted edge (Gaussian σ 0.8 px, 0.1 → 0.5, 10–90 % rise 2.51 px) and on a flat patch with noise σ 0.004:

| | 10–90 % rise | overshoot / undershoot | noise σ gain |
|---|---|---|---|
| Screen, full size | 1.94 px | 3.0 % / 3.0 % | ×1.03 (×1.38 with threshold 0) |
| Screen, `s = 0.4` | 2.30 px | 0.3 % / 0.3 % | ×1.02 |
| Print, full size | 1.13 px | 20.7 % / 20.5 % | ×1.11 |

### 2.6 Luma weights

The reference weights are `[0.299, 0.587, 0.114]`. That is the default, and the vectors use it. These weights are defined on linear camera RGB. In Yellowmoon the stage sees display-linear RGB in a known colour space, so the natural choice there is that space's luminance row, the Y row of `ym_color::consts::Primaries::rgb_to_xyz()`:

| Export space | `[wR, wG, wB]` |
|---|---|
| sRGB / Rec.709 | `[0.2126390, 0.7151687, 0.0721923]` (= `ym_gpu::cpu::LUMA` rounded) |
| Display P3 | `[0.2289746, 0.6917385, 0.0792869]` |
| Rec.2020 | `[0.2627002, 0.6779981, 0.0593017]` |

Keep the weights a parameter, default to the reference values, and test the per-space rows as an evaluation item (§7.1). The weights only decide how much each channel's edges count towards the luma. Because the transfer is a ratio, colour is preserved with any positive weights.

### 2.7 Several stages (machinery kept for later)

Yellowmoon has no local adjustments, so the per-layer masked stages are not part of this spec. The multi-stage machinery stays, so a second whole-image stage (for example a creative "Sharpening" control on the Detail tab) can be added later without changing the stage. The output stage is then stage 0, and each extra stage is built like this:

```
σ'  = radius map (§2.2) of its radius (0 to 15 px)
σ   = scale_sigma(σ', s)          T = normalise_threshold(levels/255, σ', σ)
A   = amount_pct / 100            stage exists iff σ ≥ 0.01 and A ≥ 0.001
H   = halo mapping of halo_pct (§2.2)
method: as given; with automatic selection on and method 0:
    method = (max(σ − 0.65, 0.1) · A · max(T − 0.004, 0.1) ≥ 0.025) ? 1 : 0     (f32, multiplied left to right)
```

For realistic thresholds (T ≤ 0.1) the rule reduces to `A · max(σ − 0.65, 0.1) ≥ 0.25`. For example, R 0.8 (σ 0.86) switches to method 1 from 120 %.

When an output stage exists, every extra stage that is finer than it gets a boost:

```
for each extra stage i with σ_i < σ_out:
    b    = (σ_out − σ_i) · A_out / σ_out
    A_i' = A_i · (1 + b)
    H_i' = A_i · H_i / A_i'          (the halo-limited part H·A is unchanged; the boost is all free)
```

The method choice happens before the boost. The stages are then sorted by σ ascending, with a stable sort so ties keep creation order. Within one image, every stage adds into the same two accumulators, and the limiter runs once after the last stage.

## 3. Algorithm

Notation: all planes are `f32`, `W × H`, row-major. Border reads mirror (§4.1). "f64 sum" means a sum accumulated in f64, in the stated order, and rounded to f32 once at the end.

### 3.1 Luma

```
L = sqrt(((wR·R)·R + (wG·G)·G) + (wB·B)·B)          f32, in exactly this order, no fused multiply-add
```

The halo limiter compares luma values for equality and order, so the f32 luma must be the same bits in every implementation. Yellowmoon's existing `luma()` is f32 too.

### 3.2 High-pass, difference form

For each stage, with the kernel `k` of §2.3 (`n` taps, `r = n/2`):

```
t(x, y) = Σ_{j=0}^{n−1} k_j · (L(x+j−r, y) − L(x, y))                         f64 sum, stored f32
h(x, y) = Σ_{j=0}^{n−1} k_j · ((L(x, y+j−r) − L(x, y)) + t(x, y+j−r))          f64 sum, stored f32
d = −h            (the high-pass L − G_σ * L)
B = L + h         (the blurred luma G_σ * L, f32 sum; needed by method 2 only)
```

The differences are taken in f64 from the f32 samples. Because `Σ k = 1`, this equals `L − G_σ * L` for the separable kernel (horizontal pass, then vertical). The difference form is required, not optional: a flat region gives `d = 0` exactly at any precision, and the limiter in §3.5 depends on exact zeros there (§5.4).

### 3.3 Threshold methods

Each method turns `d` into a delta `s`.

**Method 0 `HardThreshold`.** The amount is compensated so that a full-scale edge keeps its gain:

```
A' = A / (1 − T)                         f32
s  = d − min(max(d, −T), T)              = sign(d) · max(|d| − T, 0)
```

**Method 1 `SmoothGain`.** `A' = A`. If `T ≤ 0`, then `s = d`. Otherwise:

```
P  = |d| · max(|d| − T, 0)               f32
Q  = d·d + 1e-10                         f32
W1 = (G_2 * P) / (G_2 * Q)               separable blurs as in §3.6, then an f32 division
s  = d · min(max(W1, 0), 1)
```

The division is the smoothed numerator over the smoothed denominator, not a smoothing of a per-pixel ratio. At an isolated pixel, `W1 = max(|d| − T, 0)/|d|`, the same shrinkage as method 0 used as a gain. Over a `2σ` neighbourhood it is energy-weighted, so grain and edges get smooth, consistent gains.

**Method 2 `GradientRamp`.** `A' = A`. If `T ≤ 0`, then `s = d`. Otherwise, with `c = n/2` the centre index of `k`:

```
left = Σ_{i<c} k_i                                       f64
if left == 0: w = 0 everywhere                           (the kernel has collapsed to one tap; d = 0 anyway)
g0  = T · ((k_c + k_{c+1}) · 0.5) / left                 f64, then f32
inv = 0.25 / g0                                          f64, then f32
per pixel, on B with mirrored reads (um = B(x−1, y−1), u0 = B(x, y−1), up = B(x+1, y−1),
                                     cm = B(x−1, y),                   cp = B(x+1, y),
                                     dm = B(x−1, y+1), d0 = B(x, y+1), dp = B(x+1, y+1)):
gx = (cp − cm)·0.3125 + (((up + dp) − um) − dm)·0.09375        f32
gy = (d0 − u0)·0.3125 + (((dm + dp) − um) − up)·0.09375        f32 (the transpose of gx)
w  = min(max(inv · (sqrt(gx² + gy²) − g0), 0), 1)
P  = d·d·w,   Q = d·d + 1e-10,   W2 = (G_2 * P) / (G_2 * Q)
s  = d · min(max(W2, 0), 1)
```

`gx`, `gy` is a Scharr-type gradient (weights 10/32 and 3/32), so it measures change per pixel. `g0` is the blurred-luma gradient of an ideal step whose high-pass response equals `T`. The step height is `T/left` and its blurred slope is `(k_c + k_{c+1})/2` per unit height. The gain ramps from 0 to 1 as the gradient goes from `g0` to `5·g0`, so only real edges are sharpened, and noise and texture below the threshold are not.

### 3.4 Accumulation

```
if H > 0:  a_halo = H · A',  a_free = A' − a_halo
else:      a_halo = 0,       a_free = A'
D    = D    + a_free · s      (f32; D starts at 0)
Hacc = Hacc + a_halo · s      (f32; Hacc starts at 0, only touched when H > 0)
```

The stages run in ascending σ (§2.7). With one stage, that is just the output stage.

### 3.5 Halo limiter (only if any stage has H > 0)

**Envelope maps.** On a 5 × 5 window (`r = 2`, `N = 25`, `k_min = (N + 1) >> 2 = 6`) with mirrored reads, the centre included, per pixel `p`:

```
c = L(p),  S(p) = L(p) + Hacc(p)                        f32 (S(q) for each window sample likewise)
over the 25 samples q, with differences taken in f64:
  L(q) ≤ c    → n_lo  += 1, e_lo  += L(q) − c
  L(q) ≥ c    → n_hi  += 1, e_hi  += L(q) − c
  S(q) ≤ S(p) → n_slo += 1, e_slo += S(q) − S(p)
  S(q) ≥ S(p) → n_shi += 1, e_shi += S(q) − S(p)
(a sample equal to the centre counts on both sides; the centre always does, so every count ≥ 1)

m_hi = 2·e_hi/n_hi  (≥ 0)        m_lo = 2·e_lo/n_lo  (≤ 0)
maxL = c + m_hi                  minL = c + m_lo           robust local max / min: 2·mean(values ≥ c) − c
w_hi = max(1 − |n_hi − n_shi|/2, 1/25)      v_hi = (S(p) + 2·e_shi/n_shi) · w_hi
w_lo = max(1 − |n_lo − n_slo|/2, 1/25)      v_lo = (S(p) + 2·e_slo/n_slo) · w_lo
d_hi = m_hi/n_hi                 d_lo = −m_lo/n_lo
if n_hi > 6 and d_hi < d_lo:  t = d_lo/(d_hi + d_lo/25);  v_hi += maxL·t;  w_hi += t
if n_lo > 6 and d_lo < d_hi:  t = d_hi/(d_lo + d_hi/25);  v_lo += minL·t;  w_lo += t
store v_hi, w_hi, v_lo, w_lo as f32                       (all of the above in f64)
```

What the maps mean:

- `2·mean(values ≥ centre) − centre` estimates the local maximum, assuming the values are spread evenly between the centre and the maximum. The minimum is estimated the same way.
- The sharpened signal's envelope gets less weight where sharpening changed how many neighbours lie above or below the centre (`|n − n_s|`).
- On the flat side of an edge, where the spread per sample is smaller than on the other side, the unsharpened extreme is mixed in with weight up to 25. That pins the overshoot to the plateau.

**Smoothing and clamp.** Blur all four maps with `G_h` (§3.6), then per pixel (f32):

```
up = max(Vhi / Whi − L, 0)
lo = min(Vlo / Wlo − L, 0)
D  = D + max(lo, min(Hacc, up))          = clamp(Hacc, lo, up), since lo ≤ 0 ≤ up
```

The limiter does not reduce the gain. It lets the halo-limited share of the delta reach the robust local extremes and stops it there. The free share `(1 − H)·A'` is never limited. `w_hi, w_lo ≥ 1/25` and the kernel is positive, so the divisions are safe.

### 3.6 Separable blur (gain maps and envelope maps)

```
tmp(x, y) = Σ_j k_j · X(x+j−r, y)        f64 sum (taps left to right), stored f32
out(x, y) = Σ_j k_j · tmp(x, y+j−r)      f64 sum, stored f32
```

The horizontal pass runs over every row of the buffer, then the vertical pass.

### 3.7 Transfer back to RGB

```
l     = max(L, 1e-20)
ratio = (L + D) / l                       f32 (L + D first, then the division)
ratio = ratio ≥ 0 ? min(ratio, 16) : 0    (NaN → 0)
R', G', B' = R·ratio, G·ratio, B·ratio
```

### 3.8 Constants

| Constant | Value | Use |
|---|---|---|
| Radius map | 0.5, 0.7, 0.3, 0.09, 0.97 | §2.2 |
| σ knee | 0.15 | resolution scaling |
| Minimum σ, amount | 0.01, 0.001 | stage creation |
| Screen radius cap | 2.5 px | §2.2 |
| Print distance scale and radius cap | 2500 per px, 15 px | §2.2 |
| Tap cap | 13 (screen), 33 (print) | §2.3 |
| Tap rule | 5.0, 2.5, `| 1`, minimum 3 | §2.3 |
| Percent, 8-bit levels | 100, 255 | `A`, `T0` |
| Halo mapping | 0.01, sqrt, dead zone 0.001 | §2.2 |
| Gain kernel | `max(2σ, 1)` | §2.3 |
| Gain epsilon | 1e-10 | §3.3 |
| Gradient | 0.3125, 0.09375, ramp 0.25 (to `5·g0`) | §3.3 |
| Automatic method | 0.65, 0.1, 0.004, 0.1, 0.025 | §2.7 |
| Limiter window | r = 2 (5 × 5), N = 25, k_min = 6, 1/r = 0.5, 1/N = 0.04 | §3.5 |
| Halo-map kernel | `2s` | §2.3 |
| Luma floor, ratio cap | 1e-20, 16 | §3.7 |

### 3.9 Precision summary

| Quantity | Precision |
|---|---|
| Planes (L, d, B, D, Hacc, P, Q, W, maps) | f32 storage |
| Derived scalars σ, T0, A, A', H, s, boost | f32 as written |
| `erf` terms (kernel, `c(σ)`, threshold ratio), `g0`, `left` | f64 |
| Convolutions, high-pass, window sums and envelope formula | f64 sums in the stated order |
| Per-pixel formulas (luma, methods, accumulation, clamp, transfer) | f32, no fused multiply-add |

## 4. Edge cases

### 4.1 Borders

Every read outside the buffer mirrors without repeating the edge sample, as `ym_gpu::cpu::mirror` does: `−1 → 1`, `n → n − 2`, period `2(n − 1)`, and `n = 1` always gives 0. This covers the high-pass, the gain-map and envelope blurs, the 3 × 3 gradient and the 5 × 5 window. The mirror is separable, so the two-pass blur equals a 2-D convolution of the mirrored image. Kernels longer than the image wrap around through the mirror period. That is well defined, and `tiny_image_borders` (5 × 3, 13 taps) pins it down.

### 4.2 Small images

Images as small as 1 × 1 work. A 1 × 1 image, or any flat image, comes back unchanged.

### 4.3 Black, zero and tiny values

- `L = 0` gives `l = 1e-20`. Because every weight is positive, `L = 0` means all three channels are 0. `ratio` is then either 0, or 16 for a positive `D`, and the output is `0 · ratio = 0`. Black stays black.
- A pixel with `0 < L < 1e-20` gets `ratio ≈ L/1e-20` and comes out near 0. This is harmless at 1e-20.

### 4.4 Negative channels

The luma uses squares, so it does not see signs. The transfer multiplies, so each channel keeps its sign. A pixel whose sharpened luma `L + D` goes below 0 becomes black. That is the only clip.

### 4.5 Values above 1.0

No clamp. Thresholds are absolute, so in bright areas they are small relative to the signal. The ratio cap of 16 only matters for very dark pixels next to bright ones.

### 4.6 Collapsed kernel

For σ below about 0.06 (R just above 0.09) the outer taps are 0 in f64. `G_σ` is then the identity, `d = 0`, and method 2's `left` is 0. The ramp is defined as 0, and the stage is a no-op.

### 4.7 NaN and infinity

Inputs must be finite. The export's pixels are. The stage does not sanitise: a NaN spreads through the blurs, and only its own pixel's ratio is forced to 0.

### 4.8 CFA phases

Not applicable. This is an RGB stage after the demosaic. The symmetry checks are flip and transpose equivariance instead (§6.2).

### 4.9 Threshold extremes

With the range capped at 100 levels, `T ≤ 0.392`, so `1/(1 − T) ≤ 1.65`.

## 5. Determinism, tiling, performance, GPU

### 5.1 Determinism

Every output pixel depends only on input pixels within `halo_px`. All sums have a fixed order (taps left to right, horizontal pass first, window row by row). There are no image-wide reductions. Rows can run in parallel in any order and the bytes stay the same, which keeps app and CLI exports byte-identical.

### 5.2 Tiling contract

Running the stage on `core.grow(halo_px)`, clipped to the frame, and keeping the core gives the whole-frame result bit-exactly. The self-test checks this for seven parameter sets on a 71 × 66 image with 17 × 23 tiles: the maximum difference is 0. Tiles that touch the frame edge mirror at the frame edge, exactly like the whole frame.

### 5.3 Performance

The export holds the whole image, so the simplest port runs on the whole frame. Peak memory is about 12 to 16 f32 planes (L, h, t, d, B, P, Q, two blurred maps, D, Hacc, four envelope maps and their blurs), roughly 64 B/px or 1.5 GB at 24 MP. Run in bands of rows instead, for example 256 rows plus `halo_px` above and below, using the tiling contract. The output is the same and memory stays small.

Cost per stage per pixel: a `2n`-tap high-pass, two `2n_2`-tap blurs for the gain map, and for the limiter 25 window samples with four compares plus four `2n_h`-tap blurs. The unoptimised scalar reference takes about 135 ms/MP for the screen preset, 34 ms/MP with halo suppression off, and 50 ms/MP for the print preset (one core). A Rust port with rows in parallel should take well under a second for a 24 MP full-size export on an 8-core machine.

### 5.4 GPU notes

- **Per pixel:** luma, the method formulas, accumulation, the envelope-map window pass (25 reads), the clamp and the transfer.
- **Separable passes:** the high-pass (difference form), the `G_2` gain-map blurs (two per stage), and the `G_h` envelope blurs (four).
- There are no reductions. Pack the four envelope maps into one RGBA texture so the four blurs are one pass.
- **The limiter is discontinuous.** It counts window samples by exact comparisons, and its `t` weights divide small differences. Two rules keep it robust:
  1. Compute the high-pass and the window sums in difference form (§3.2, §3.5). Then flat regions give exact zeros in f32, as they do in f64. With plain sums in f32 (`Σ k·L` minus the centre afterwards, `Σ l` minus `n·c` afterwards), pixels near clean synthetic edges differ from the f64 result by up to 1e-2.
  2. Compute the luma exactly as in §3.1.

  Even then, values that are equal in exact arithmetic but computed along different paths can sit one ulp apart and flip a count. This happens on clean, symmetric synthetic content (diagonal edges, checkerboards) with the limiter on, and changes isolated pixels by up to about 1e-2.
- **GPU parity metric:**
  - On noisy or photographic content: max abs error ≤ 1e-5. The reference's f32-accumulation mode reproduces a GPU's arithmetic closely, and gives 4.8e-7 on the self-test image and at most 7.2e-7 on every vector.
  - On clean synthetic content with the limiter on: compare the 99.9th percentile of the error, and report the maximum.
- `run(..., Accumulation::F32)` in the reference simulates single-precision accumulation, so a GPU port can be checked against a CPU run that should match it closely.

## 6. Test plan

### 6.1 Vectors (`vectors/output_sharpening/`)

Each file is JSON with these fields:

- `name`, `description`.
- `params`: every user parameter, plus `derived`: `max_taps`, `halo_enabled`, `halo_px`, `halo_kernel`, and per stage `output`, `method_id`, `amount` (after the boost, before the method-0 compensation), `sigma`, `threshold`, `halo` (fraction), `kernel` and `gain_kernel`. The kernels are f64 with `%.17g`. `gain_kernel` is listed for every stage but only used by methods 1 and 2 with `T > 0`.
- `input`, `expected`: `{width, height, layout: "planar_rgb", data}`, f32 printed with `%.9g`.
- `tolerance`: absolute, per sample.

| File | Size | Covers | Tol. |
|---|---|---|---|
| `flat_field_screen` | 24² | Screen preset (method 1, halo 60) on a coloured flat field: exact identity | 0 |
| `flat_field_hdr_negative_print` | 24² | Print preset on (1.7, 1.25, −0.02): exact identity with values > 1 and a negative channel | 0 |
| `vertical_edge_hard_threshold` | 32² | Method 0, threshold 2 levels, `1/(1−T)` compensation, no limiter | 1e-5 |
| `horizontal_edge_smooth_gain_halo` | 32² | Method 1 at R 1.5, halo 60, off-centre clean edge (exact zeros on plateaus) | 1e-5 |
| `diagonal_edge_gradient_ramp` | 32² | Method 2 ramp and Scharr gradient on a clean diagonal, R 2 | 1e-5 |
| `zone_plate_plain_usm` | 40² | Plain linear USM (method 0, T 0) up to Nyquist | 1e-5 |
| `fine_pattern_full_halo` | 32² | 1 px checkerboard and 2 px stripes in colour, light noise, halo 100 (all delta limited) | 1e-5 |
| `noisy_patch_smooth_gain` | 32² | Method 1 threshold on noise σ 0.004 (noise changes by rms 0.00036 only) | 1e-5 |
| `noisy_edge_gradient_ramp` | 32² | Method 2 on a noisy edge, halo 60 | 1e-5 |
| `hdr_values_and_negatives` | 32² | Method 0 with halo 60 on values up to 8.0, slight negatives, black pixels | 1e-5 |
| `downscaled_export_screen` | 32² | Screen preset at `s = 0.4`: σ and T rescaled, `σ_h = 0.8` | 1e-5 |
| `print_mode_distance` | 40² | Print preset, R = 2 px from distance 5000, 33-tap cap | 1e-5 |
| `print_mode_long_kernel` | 48² | Print distance 20000: σ 5.9, 33 taps | 1e-5 |
| `two_stage_machinery` | 32² | Output stage plus an automatic-method extra stage: sorting, boost, shared limiter | 1e-5 |
| `amount_zero_identity` | 16² | Amount 0: no stage, copy | 0 |
| `tiny_image_borders` | 5 × 3 | 13-tap kernels on a 5 × 3 image, method 2, halo 60: mirror semantics | 1e-5 |

Every vector is also checked by the generator against the reference's f32-accumulation mode, and all stay within 7.2e-7. So the tolerance only fails a port that deviates from the spec, not one that rounds differently. Vectors with the limiter on clean symmetric content were given light noise so they do not hinge on near-ties (§5.4).

### 6.2 Property tests (the reference's `-DSELF_TEST`, to mirror in Rust)

- **Flat field exact.** Seven parameter sets × four colours (including values > 1, a negative channel, black and 3e-5): output equals input bit for bit.
- **Identity.** Amount 0 gives a copy.
- **Tiling parity.** Tiles of `core + halo_px` match the whole frame bit-exactly (maximum difference 0).
- **Flip and transpose equivariance.** Within 1e-5. The reference gives 2.4e-7 and 1.2e-7.
- **Scale behaviour.** With `T = 0`, `f(2x) = 2·f(x)` and `f(x/4) = f(x)/4` bit-exactly.
- **Robustness.**
  - No NaN or Inf on random images in [−0.05, 3] or on 1 × 1 to 7 × 5 images, in every mode.
  - A collapsed kernel with method 2 is a finite no-op.
  - Validation rejects `s > 1` and NaN.
- **Precision.** f32 accumulation stays within 1e-5 of f64 on noisy content, and flat fields stay exact with f32 accumulation.
- **Hue.** Channel ratios are preserved per pixel (relative error 1e-6).
- **Halo suppression.** On a hard 0.1 → 0.5 step with amount 300 and R 1.5, halo 100 cuts the overshoot from 0.43 to 0.007 and the undershoot from 0.10 (clipped at black) to 0.007.
- **Plan derivation.**
  - σ for R 0.8 / 1.0 / 2.5.
  - Tap counts.
  - Kernel symmetry and unit sum.
  - The halo mapping and the automatic method rule.
  - Threshold rescaling.
  - `halo_px` values 6 / 12 / 13 / 20 / 21 / 32 / 40.
  - Boost and sorting.

### 6.3 Yellowmoon tests (RA-08)

- `output_sharpen = None` leaves export bytes unchanged (golden hashes unchanged).
- Unit tests for the two presets: derived σ, T, H and `halo_px` at `s = 1` and at a 2048 px long edge from 6000 px (σ 0.440, T 0.523 levels).
- **Edge-profile test** through `encode_to_writer`: a slanted soft edge (5°, Gaussian σ 0.8 px, 0.1 → 0.5 display-linear) at full size and at a 1000 px long edge. Measure the 10–90 % rise distance, overshoot and undershoot in % of the step, and the noise σ gain on a flat patch with noise σ 0.004. Expected values for the screen preset at full size: rise 2.51 → 1.94 px, overshoot ≤ 3.5 %, noise gain ≤ ×1.05. Record the numbers in STATUS.
- Determinism: `--jobs 1` and `--jobs 4` give identical bytes.

### 6.4 Quality gate (visual and edge-profile)

There is no PSNR target, because sharpening is a look.

- **Edge profiles.** As in §6.3, at full size and at long edges of 3000, 2048 and 1000 px. Compare halo 60 with halo 0, and the three methods.
- **Visual.** 100 % crops of the nine local real ARWs exported for screen (2048 px) and print (full size), with and without output sharpening, laid out under "Waiting on Mykyta" for RA-H3. Look for halos on high-contrast edges (branches against sky), sharpened noise in skies and shadows, and stair-stepping on fine diagonals.

## 7. Evaluation notes and open questions

### 7.1 Evaluation notes (design choices worth an A/B)

1. **Luma weights.** The reference weights `[0.299, 0.587, 0.114]` against the export space's own Y row (§2.6). Test saturated red-on-blue and green-on-magenta edges, where the weights change which edges are found.
2. **Radius follows the export size.** σ shrinks toward 0.15 px as `s` falls. With the screen preset a 2048 px export gets σ 0.44 and only an 8 % faster rise on the soft test edge. A/B against keeping σ in output pixels (`s = 1` for the sharpening only) for small exports.
3. **Absolute threshold in linear light.** A fixed `T` is a larger fraction of the signal in shadows, so shadows are protected more than highlights. Alternatives:
   - a threshold proportional to the local noise, `sqrt(g·L + r²)` with Yellowmoon's noise model carried through the display transform;
   - applying the threshold to an encoded luma.
4. **Display-linear placement.** The stage runs after the display transform, where highlights are already compressed by the tone curve. That is the natural place in Yellowmoon's chain, but compare it with running the stage on scene-linear camera RGB, before the output colour conversion, on a full-size export.
5. **Limiter robustness.** The rank counts and the `t` weights make the limiter discontinuous (§5.4). A tolerance in the comparisons (for example counting `|l − c| ≤ 1e-6` as equal), or smooth ranks, would make it continuous and GPU parity tight. Check that it does not change the look.
6. **Step in the radius map at R = 0.5** (σ' 0.621 → 0.650). A UI with fine radius steps shows a small jump. A continuous map is `0.97·sqrt(R − 0.09)` up to the crossing point with `0.7R + 0.3`.
7. **Method-0 compensation `1/(1 − T)`** sharpens strong edges slightly more than amount alone suggests. Method 1 has no compensation. Check that the methods look equally strong at the same amount.
8. **Kernel truncation.** In screen mode the gain kernel is capped at 13 taps (±6 px) for σ ≥ 1.15, where the tap rule asks for more. The high-pass kernel is never capped in screen mode (σ ≤ 2.05 gives at most 13 taps). Compare 33-tap gain kernels for σ ≥ 1.15.
9. **Interaction with the Lanczos-3 ringing.** The resize overshoots about 7 % at steps, and sharpening after it amplifies that. Compare halo 60 with halo 0 on downsized exports of high-contrast edges.
10. **Automatic method for the output stage.** The output stage uses its method as given. Try the §2.7 rule for the output stage too.

### 7.2 Open questions (default to use meanwhile)

| Question | Default meanwhile |
|---|---|
| Ship output sharpening on by default, which presets the Export screen offers, and their numbers (RA-H3) | Off (`None`); presets as in §2.5, not exposed in UI or CLI |
| How the UI expresses print distance | Keep `print_distance` as the parameter. A natural mapping from viewing distance `D` (mm) and print resolution `ppi` makes the radius one arcminute of visual angle: `R = D · ppi / 25.4 · tan(1/60°)`, `print_distance = 2500 · R`. The default 5000 (R = 2 px) corresponds to about 580 mm at 300 ppi. |
| Luma weights | Reference weights; per-space Y row under evaluation (§7.1 item 1) |
| Sharpen the DNG's embedded preview? | No |
| Record the setting in the export's XMP (`ym:Settings` or a new `ym:` property, `docs/09`) | Not recorded until the UI exists. When it ships, add the preset name and the five numbers to `ym:Settings`. |
| Expose `method` and `halo_pct` in the UI | Not exposed; preset values only |

## 8. Reference notes

Where an implementation could reasonably differ, the reference makes the following choices. A port should follow them. Except for items 3 and 4, they change results only near frame borders or by less than 1e-5.

1. **Borders.** Mirror without repeating the edge sample (§4.1), not edge replication. This affects only pixels within `halo_px` of the frame edge.
2. **Accumulation.** Convolutions, window sums and the envelope formula accumulate in f64. Kernel taps are f64 and normalised in f64.
3. **Difference forms.** The high-pass and the window sums are taken as differences to the centre sample (§3.2, §3.5). This is mathematically the same as the plain sums, but flat regions give exact zeros. With plain f32 sums, outputs near clean synthetic edges with the limiter on differ by up to about 1e-2.
4. **Luma.** The luma is f32 in a fixed operation order (§3.1). Computing it at higher precision moves some values by one ulp and can flip the limiter's rank counts, which changes isolated pixels by up to about 2e-3 on noisy content.
5. **Thresholds.** Thresholds below 0.0001 are applied as given. No shortcut ignores them.
6. **Per-stage blur.** Each stage blurs with its own σ. A blur or gain map is never reused for a stage whose σ differs by a small amount.
7. **Stage order.** Stages are sorted with a stable sort.
8. **Threshold rescale.** The threshold rescale is skipped (ratio 1) when scaling leaves σ unchanged (σ' < 0.15 or `s = 1`). This avoids 0/0 for σ' below about 0.1.
9. **No upscaling.** `s` is limited to (0, 1], because the export never upscales. The `s > 1` branch (σ = s·σ' for extra stages) is not part of this spec.
10. **Boost.** The boost (§2.7) is skipped for stages with σ ≥ σ_out (b = 0), so their amount and halo values are untouched.
11. **Method 2 guard.** Method 2 defines the ramp as 0 when the kernel has collapsed (`left = 0`), instead of evaluating `0 · ∞`.
12. **Gradient symmetry.** The gradient's y component is the exact transpose of the x component (§3.3).
13. **Halo parameter.** The output stage's halo suppression is a parameter with preset values (60 screen, 0 print), not tied to the mode.
14. **Validation.** Out-of-range and NaN parameters and unknown method values are rejected. An unknown method never silently skips stages.

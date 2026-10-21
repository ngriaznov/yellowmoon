# Single-frame denoise (R6)

This document describes the denoise that follows the demosaic in every full-resolution render (the 100 % view and every export): method A, block matching with 3D DCT hard thresholding, run in a display-like working encoding, followed by a false-colour step. It covers the method, its parameters and thresholds, how the Detail tab's controls set them, the false-colour step, halos, the device kernels and their parity, and the measured quality. The methods that exist in the code but do not run by default are described at the end.

Labels such as G1, A-a, BC-c and D-b name implementation details that the code comments cite.

| Method | Idea | Working space | Strength inputs | Runs |
|---|---|---|---|---|
| A | block matching + 3D DCT hard thresholding | Y / Cb / Cr | L, C, R | **default**, CPU and device |
| B | multiscale NL-Bayes, 2×2 patches, 8 per group | R / G / B | L, R | internal switch, CPU |
| C | multiscale NL-Bayes, 2×2 patches, 24 per group | Y / Cb / Cr | L, C, R | internal switch, CPU |
| D | global noise level for a learned denoiser | Y / Cb / Cr | – | estimate only, measurement tool |

## 0. Conventions

- Images are float, black = 0 and white = 1, in the non-linear state the viewer sees (the working encoding, *Integration* below). The fixed constants (0.2–0.8 clamps, 30/255, 220/255) assume that scale.
- Strength inputs: luma strength `L ∈ [0, 100]`, chroma strength `C ∈ [0, 100]`, radius `R` (a small non-negative integer, 0 to 3), blend `B ∈ [0, 100]`.
- `box_k(x)` is a k×k mean filter.
- **G1 Data.** The reference functions take and give `RgbImage` (three `f32` planes) in the encoded state. Samples outside [0, 1] are kept, never clamped, except where a rule below clamps. Arithmetic inside is `f64` (`AGENTS.md`: colour maths on the CPU reference path).
- **G2 Y/Cb/Cr of methods A and C.** BT.709: `Y = 0.2126 R + 0.7152 G + 0.0722 B`, `Cb = (B − Y) / 1.8556`, `Cr = (R − Y) / 1.5748`, and the exact inverse. Y is in [0, 1], Cb and Cr are centred on 0. Conversion happens before the method and back after. Method D keeps its own coefficients (BT.709 scaled by 224/219), used for its estimate only.
- **G3 Determinism.** The output is the same bits for any number of threads: every sum that more than one reference adds to is accumulated in a fixed order (the passes of A.5 in the order `(i, j)` raster, then the references of a pass in raster order, then the slots of a group in order; methods B and C the same with their own passes). Ties of a distance are broken by the candidate's raster index (smaller first).
- **G4 Frame coordinates.** Every grid (A's reference blocks, B and C's reference patches, the detector's cells) is anchored to the frame's origin (`RgbImage::place`), never to the buffer's, so that a tile with the stated halo gives the whole frame's pixels.
- **G5 Blend.** The final blend of every method is `out = lerp(denoised, input, B/100)`, applied only when `B > 0`; it uses the input as given to the method (in RGB).

## A. Method A: block matching and 3D DCT hard thresholding

Code: `ym_gpu::cpu::bm_dct` (reference), `ym_gpu::kernels::bm_dct` and `shaders/bm_dct*.wgsl` (device).

### A.1 Parameters

```
τY = 0.1  · L                  hard threshold, luma
τC = 0.06 · C                  hard threshold, chroma
θ  = 0.2  · max(L, C)          block-distance threshold
s  = 8 − 2·R                   reference-block step (pixels), R ∈ {0, 1, 2, 3}
```

The luma channel is skipped when `τY ≤ 1e-6`, and both chroma channels when `τC ≤ 1e-6`; a plane that is not filtered is the input plane. The render uses `R = 1` (`s = 6`).

### A.2 Window

`K(i, j) = k(i)·k(j)`, i, j ∈ 0..7, where `k(n) = I0(2·sqrt(1 − (2n/7 − 1)²)) / I0(2)`: a Kaiser window with N = 8 and β = 2,

`k = [0.4387, 0.6813, 0.8768, 0.9858, 0.9858, 0.8768, 0.6813, 0.4387]`.

The sum of K is about 35.59.

### A.3 Per reference block

Reference blocks are 8×8 with top-left corners on a regular grid of step `s` in x and y.

1. **Candidates.** 25 blocks with top-left at `ref + (dx, dy)`, `dx, dy ∈ {−8, −4, 0, 4, 8}`, indexed 0..24 in row-major order; index 12 is the reference.
2. **Distance**, on Y only: `D = Σ_{i,j} K(i,j) · |Y_ref(i,j) − Y_cand(i,j)|`.
3. **Grouping.** Slot 0 is always the reference. Slots 1..7 take the 7 non-reference candidates of smallest D. Any slot whose D > θ is replaced by the reference block, so duplicates are allowed. The result is a group of 8 blocks. Chroma reuses the luma grouping.
4. **3D transform**, per filtered channel (Y with τY; Cb and Cr with τC): an 8-point DCT-II along rows, then columns, then across the 8 blocks. The transform is unnormalised: each 1D pass is the orthonormal DCT-II times √8, so the DC term is the plain sum. Relative to an orthonormal transform every coefficient is 8^1.5 ≈ 22.6 times larger, and τ is compared on that scale.
5. **Hard threshold.** Every coefficient with `|c| < τ` is set to 0, the DC term included.
6. **Inverse 3D transform** (same scaling, so a forward and inverse round trip gains 8³ = 512).
7. **Aggregation.** For every pixel of every block in the group, `acc_ch += K(i,j) · v / 512` and `w += K(i,j)`. The weight is accumulated once per pixel, not per channel.

### A.4 Output

`out_ch = acc_ch / w`. With `s ≤ 8` every pixel receives at least one contribution. The result goes back to RGB and is blended with the input (G5). When no channel is filtered, or the frame is narrower or lower than 8 px, the input comes back bit for bit.

### A.5 Parallel scheduling

A reference block touches at most a 24×24 footprint (8 px block, ±8 px search). References are processed in passes: pass `(i, j)` handles the references at `(24·m + i·s, 24·n + j·s)`. Within a pass, footprints never overlap, so accumulation needs no atomics. Inside a pass the rows of references run in parallel, each into its own band of the accumulators.

### Details of method A (Yellowmoon's reading)

- **A-a Edges.** A candidate block that does not lie wholly inside the frame is not a candidate (distance +∞); a slot left without a candidate takes the reference, like a slot over θ. Reference corners are the multiples of `s` from 0 up to `W − 8`, plus `W − 8` when it is not one of them (the same for y), so every pixel is covered.
- **A-b Highlight tweak.** None. The optional lowering of the threshold for bright reference blocks is not applied; its effect is minor.
- **A-c Passes.** The passes over the regular grid come first, `(i, j)` in raster order (the y phase `j` outer, the x phase `i` inner; `i, j < 24 / s`; every `s` of `R ≤ 3` divides 24). The references that A-a adds follow in passes of their own, so that two references of one pass are always 24 px apart: the extra column (`x = W − 8`) in passes `j` by `y = 24 n + j s`, then the extra row (`y = H − 8`) in passes `i` by `x = 24 m + i s`, then the corner `(W − 8, H − 8)`.
- **A-d Halo.** A pixel depends on the references whose group can cover it (corners within 15 px to the left and above, 8 px to the right and below) and those read within 8 px of their own block: 23 px on each side (`bm_dct::HALO_PX`). A buffer that is a window of the frame computes only the references whose read span `[r − 8, r + 16)`, clipped to the frame, lies inside it; the others reach only the margin the caller drops. A pixel no reference reached (only in that margin) keeps its input value. 22 px fails the tile test.
- **A-e Build details.** `R` above 3 is read as 3, so the step stays positive. Inputs must be finite (a distance that is not finite is not a candidate). The cosine table comes from `libm`, not the platform's library, so the bits are the same on macOS and Linux; `2n/7 − 1` is computed as `(2n − 7)/7`, so the window is exactly symmetric. Working memory is the three Y/Cb/Cr planes and four accumulator planes of the buffer, all `f64`.

## Integration

The code: `ym_gpu::cpu::frame_denoise` (the stage), `ym_gpu::cpu::denoise_space` (the encoding), `Chain::frame_denoise` (`cpu/chain.rs`).

### Where it runs

R6, after the demosaic: `Stage::FrameDenoise`, followed by `Stage::FalseColor` when Colour is above 0. The stage replaces the light bilateral denoise and R7's detail re-injection; of R7 only the colour guided filter stays, where the luma alternates (*False colour* below). The fit view and the thumbnails bin the mosaic and never demosaic; they keep the bilateral denoise of the binned image (`docs/05-raw-processing.md` R6).

### Working encoding

The raw chain holds balanced linear camera RGB, white 1.0 at the clip before the gains. The method is made for the encoded image the viewer sees, so the stage encodes with

`W(x) = srgb(2^CAMERA_GAIN_EV · x)`, `CAMERA_GAIN_EV = 1.11`,

runs the method and decodes. `srgb` is the sRGB transfer function on the whole line: `s·v` below the knee `v = 0.0031308`, with the slope `s = 12.91999…` made exact at the knee so the curve and its inverse are continuous and strictly increasing, and `1.055 v^(1/2.4) − 0.055` above (also beyond 1, where highlights and white-balance gains take a sample; negative values keep their sign). Middle grey of a photo at the camera's exposure lands at 0.46, as in an sRGB picture, and the 1/2.4 power nearly evens out shot noise, so one threshold suits the shadows and the lights. The encoding depends on nothing that changes after R6 (exposure, look and curves are colour stages), so a slider there never re-runs the raw chain.

### Controls

The Detail tab's controls keep their Auto values and their meaning relative to Auto (`docs/05-raw-processing.md` R7). With `r = L / L_auto(ISO)` and `c = C / 50`:

| Input | Method A |
|---|---|
| Luma strength | `L_A = r · kL · σY · 22.627 / 0.1`, so that `τY = r · kL · σY · 22.627` |
| Chroma strength | `C_A = c · kC · σC · 22.627 / 0.06`, so that `τC = c · kC · σC · 22.627` |
| Blend `B` | `100 · 0.15 (D / 50)²`: 15 % at Auto, 60 % at Detail 100, 0 at Detail 0 |
| Radius `R` | 1 |

- `22.627 = 8^1.5` is the scale of the unnormalised transform (A.3 step 4). `kL = 3.5` and `kC = 8` put the thresholds at that many standard deviations of the photo's noise.
- `σY` and `σC` are the photo's luma and chroma noise in the working encoding at middle grey (`frame_denoise::working_noise`). A balanced channel `c` at raw grey `x = 0.18 / 2^1.11` has variance `g0_c · a · x + g0_c² · b` from the noise model (`docs/06-noise.md`) and the R4 gains `g0`; the encoding's slope at grey scales it; Y, Cb and Cr weigh the channels with their G2 coefficients, taken as independent; `σC = sqrt((σCb² + σCr²) / 2)`. The demosaic correlates the noise, which `kL` and `kC` absorb.
- `L_auto(ISO) = clamp(10 · (1 + log2(ISO / 100)), 10, 90)`; the Auto colour and detail values are 50.
- Strengths are clamped to [0, 100]. When Luminance and Colour are 0, or the photo has no noise, the method is an identity and the chain has no stage (the encoding's round trip never touches the pixels).

### Selection

`DEFAULT_DENOISE = DenoiseMethod::BmDct` is what every full-resolution render uses. `DenoiseMethod` (`bilateral`, `bm_dct`, `nl_bayes_rgb`, `nl_bayes_ycc`; `name` and `from_name`) is an internal switch, `RenderOptions::denoise` in `ym-core`, like the demosaic method of R6: no settings field, no UI control, no CLI flag, not part of the cache keys. Tests and evaluation tools set it.

### False colour

Code: `cpu/false_color.rs`, `Stage::FalseColor`.

The demosaic leaves false colour on stripes near the sensor's limit (blinds, shutters, fabric; the golden chart's 2 px grating). Method A keeps it: block matching groups the repeating pattern and the hard threshold keeps what the group agrees on. R7's colour guided filter removes it, but run everywhere it smears colour across every edge the luma does not share (it costs 1.8 dB on Kodak at ISO 100). So the guided filter runs only where the luma alternates:

1. `Y`: the luma of the working encoding (BT.709 weights), so one threshold holds in the shadows and the lights.
2. Alternation: `s = max(0, −(Y(x) − Y(x−1)) · (Y(x+1) − Y(x)))` along x plus the same along y. It is zero on a ramp, a step or a blurred edge; stripes of amplitude `A` and a period of 2 px give `(2A)²` per direction.
3. Energy: `E = sqrt(box₅ₓ₅(s))`, then `M = max₅ₓ₅(E)`, so the ends of a striped area are covered.
4. Weight: `w = smoothstep((M − T0) / (T1 − T0))` with `T0 = 0.05`, `T1 = 0.15` (working-encoding units).
5. Output: `out = x + w · (G − x)` per channel, with `G` the output of R7's colour guided filter at the photo's Colour and noise (it changes only the chroma).

Colour 0 turns the step off with the rest of the colour noise reduction. Buffer edges are clamped. Halo: `max(2r, 5)` px, with `r = round(2 + 6·C/100)` the guided filter's radius (10 px at Auto Colour); the gate itself reads 1 + 2 + 2 px.

### Tiles and halos

Method A asks for its 23 px halo, the false-colour step for its own after it, and a tile equals the whole frame bit for bit (tested through the chain). In the halo table of `docs/05-raw-processing.md` the R6 classical row is the demosaic's halo plus 23, then the false-colour step's. Methods B and C read the whole frame, so a tile of them asks for the whole raw frame: correct and exact, and slow.

### On the device

Method A runs on the graphics card like every other raw stage (`docs/17-gpu.md`): the device chain records it as R6's step after the demosaic, with the CPU stepper's rectangles, in one job with the stages around it.

- **Method A** (`kernels::bm_dct`, `bm_dct_prep.wgsl`, `bm_dct_pass.wgsl`, `bm_dct_output.wgsl`). A prep pass makes the encoding, Y/Cb/Cr and four zeroed accumulators. One dispatch per pass of A.5, in the order of A-c, has one workgroup of 256 threads per reference whose read span lies inside the buffer (A-d). An output pass divides, converts back, blends and decodes. With `R = 1` that is 16 regular passes and up to 9 extra ones, so 18 to 27 dispatches per tile; strengths that filter nothing, or a frame under 8 px, take one dispatch (the output pass alone). Memory: at most 13 `f32` per pixel of the buffer; the largest binding is the accumulators', 16 bytes per pixel.
- **False colour** (`kernels::false_color`, `false_color.wgsl`). R7's guided filter kernels (the CPU's bits, four passes) and a gate of four passes: the luma of the working encoding; the alternation with its row sums; the column sums and the root; the 5×5 maximum, the weight and the blend. Eight dispatches.
- **Order and exactness.** Every pixel's sums are added in the CPU's order (pass by pass, at most one reference per pixel and pass, slot by slot with a storage barrier between slots), so a tile with the 23 px halo equals the whole frame bit for bit on the device too.
- **Precision.** The device works in `f32` against the reference's `f64`. Away from a decision the output is within a few `f32` ulp of the CPU's. The method's decisions (`|c| < τ` on 512 coefficients per channel and reference, the order of the 24 candidate distances, `D > θ`) are taken on values that differ by those ulp, so a coefficient within rounding of `τ`, or two candidates within rounding of each other, can go the other way; such a flip moves the reference's 8 blocks by up to about `τ / 512` in the encoding. The false-colour gate has no decision that jumps (`w` is continuous in the energy and the blend in `w`), so it differs by rounding alone.
- **Fast-math (Metal).** Every product, sum and difference that feeds a decision or the output is formed behind the runtime-zero OR barrier (`bm_op`), divisions are correctly rounded (`bm_div`), the power, `log2` and `exp2` of the encoding are basic operations behind the barrier (`bm_pow`, `shaders/working_encoding.wgsl`), and the false-colour root is correctly rounded in integer arithmetic (`fc_sqrt`). The arithmetic is one IEEE `f32` sequence on every backend, so Metal takes the decisions lavapipe takes (`docs/17-gpu.md` K8).
- **Parity on lavapipe** (`tests/gpu_bm_dct.rs`, `tests/gpu_false_color.rs`; each bound is twice the first value measured):

  | Case | Result |
  |---|---|
  | method A, 30 frames from 8×8 to 301×203, R 0 to 3, luma only, chroma only and both, blend 0 and 15, values from below 0 to above 1 | max abs difference 2.09e-6; 0 of 1.9 million samples over 1e-4 |
  | method A, whole frames of the tile test | 1.55e-6 in three of four; in the fourth one decision of the luma channel went the other way: 1.68e-3, 204 of 99 381 over 1e-4 |
  | method A, noise-free synthetic content (ramps, flat patches, clipped plateaus), where exact ties of mirror-image candidates are broken by each side's rounding | 7.3e-3 largest, 5.7e-2 of the samples over 1e-4 |
  | method A, windows of an 8658×5772 frame at its origin, middle and far corner | 6.6e-7 |
  | false-colour weights, noisy natural frames, 2 and 3 px gratings, stripes on half a frame | 5.07e-7 (0 in the gratings) |
  | false-colour stage, Colour 30 to 100, scalar and plane σ | 1.19e-7 over 957 372 samples |
  | tiles (two sizes, windows at odd origins with the halo, one at the far edges), a poisoned device, `LP_NUM_THREADS=1` | bit-identical to the whole frame; finite |

Methods B and C have no kernel: the device chain refuses a whole-frame method, and the photo's raw chain then runs on the CPU.

## Measurement

`crates/ym-gpu/examples/denoise_gate.rs` (`cargo run -p ym-gpu --release --example denoise_gate -- all`; the subcommands are `calibrate`, `compare` and `render`). The numbers below are for pipeline 0.5.0 on Linux x86_64, 3 threads of a shared 4-core machine.

Setup: Kodak (24 images) and McMaster (18) from `testdata/demosaic` (`tools/fetch/demosaic-sets.sh`) in linear light, placed where a photo at the camera's exposure sits on the raw scale (middle grey 0.083), mosaicked as RGGB, with Gaussian noise of variance `a x + b` per photosite at five ISO-like levels (the demosaic gate's stand-ins for 100, 800, 1600 and 6400, and 12800 by the same scaling; not measurements, no row noise), and demosaiced with the adaptive ratio demosaic and the matching model. PSNR against the clean image in the working encoding (peak 1), mean of per-image dB, 10 px border excluded. "Bilateral + R7" is the bilateral denoise with a σ map and R7 at Auto; the render is method A at the render's settings (Colour at Auto, blend 15 %) followed by the false-colour step. Full tables: `testdata/denoise_gate/numbers.md` after a run.

| Set | ISO-like | none | bilateral + R7 | A | A + false colour (the render) |
|---|---|---|---|---|---|
| Kodak | 100 | 36.89 | 35.27 | 37.14 | **37.00** |
| Kodak | 800 | 34.54 | 34.19 | 35.84 | **35.60** |
| Kodak | 1600 | 32.78 | 33.35 | 34.72 | **34.51** |
| Kodak | 6400 | 27.61 | 30.61 | 31.31 | **31.23** |
| Kodak | 12800 | 25.05 | 29.10 | 29.44 | **29.37** |
| McMaster | 100 | 34.07 | 31.62 | 34.30 | **34.12** |
| McMaster | 800 | 32.01 | 30.91 | 33.13 | **32.98** |
| McMaster | 1600 | 30.69 | 30.35 | 32.27 | **32.15** |
| McMaster | 6400 | 26.82 | 28.42 | 29.90 | **29.82** |
| McMaster | 12800 | 24.73 | 27.19 | 28.49 | **28.40** |

- The render is above the bilateral path at every level (+1.7, +1.4, +1.2, +0.6 and +0.3 dB on Kodak, +2.5, +2.1, +1.8, +1.4 and +1.2 on McMaster from ISO 100 to 12800) and above no denoise at every level, so it does not soften a clean photo. The bilateral path is below no denoise at ISO 100 and 800 on both sets, and at ISO 1600 on McMaster.
- The false-colour step costs 0.07 to 0.24 dB against A alone. What it buys is not in these numbers: on the Kodak blinds (kodim01) and shutters (kodim08) A alone keeps the demosaic's blue and orange bands, and the render removes them.
- Time on the CPU, seconds per megapixel (wall, 3 threads): bilateral + R7 0.2 to 0.3, A 0.5, A with the false-colour step 0.5 to 0.7 (2 threads).
- `R = 2` adds 0.02 to 0.07 dB to A (without the blend) at 1.5 times its time; `R = 0` loses 0.05 to 0.4 dB.

**Calibration of `kL` and `kC`** (first 8 Kodak images, `kL` 2.5 to 5 against `kC` 6 to 16): the best `kL` per level is 3.5, 3, 3.5, 4, 4 and the best `kC` 16, 8, 8, 6, 6 (ISO 100 to 12800). `kL = 3.5`, `kC = 8` is within 0.12 dB of the best luma setting and 0.2 dB of the best chroma setting at every level. The noise measured after the demosaic is 1.5 to 3.9 times the model's (the demosaic's error dominates at low ISO and the demosaic correlates the noise); the constants are fitted against the model's noise, which is what a render knows. Method D's estimate reads 0.69 to 0.96 of that measured σY (lower as the noise grows) and 0.53 to 0.63 of its σC.

## Other methods in the code

All are selected through `RenderOptions::denoise` (*Selection* above) and are used by tests and evaluation tools.

### Bilateral (`bilateral`)

The light bilateral denoise followed by R7 (colour guided filter, detail re-injection). It is what the reduced renders (fit view, thumbnails) always run, on the binned image, and `docs/05-raw-processing.md` R6 and R7 describe it. It has a device kernel.

### B and C: multiscale NL-Bayes on 2×2 patches

Code: `cpu/nl_bayes.rs`, `cpu/flat_regions.rs`. Selected as `nl_bayes_rgb` (B) and `nl_bayes_ycc` (C). In the render's controls B gets `L_B = 30 r` and C gets `L_C = 80 r`, `C_C = 90 c` (the values measured best), with the same blend and `R = 1`. B ignores the chroma strength. On the measurement sets C is 0.1 to 0.2 dB above A at ISO 100 and 800 and below it from 1600 up, at about twice A's time (1.0 s/MP) and with the whole frame to read; B (0.9 s/MP) is below both. B and C read the whole frame (pyramid and measurement), so they run on the CPU and a tile of them asks for the whole raw frame.

**Parameters.**

```
                       B (RGB)           C (Y/Cb/Cr)
patch                  2×2 (4-vector, order (0,0),(1,0),(0,1),(1,1))
group size K           8                 24
search window S        R + 6             R + 6      half-width h = floor(S/2)
reference stride       2                 2          (reference patches every 2 px in x and y)
σ1 (channel 0)         0.02 · L          0.0125 · L
σ2 (channels 1, 2)     0.02 · L          0.02 · C
pyramid base           128               256
intensity noise model  yes               no
final-step noise gain  2                 1
```

σ is a gain on the measured noise, not an absolute level: in method B, `L = 50` means "remove exactly the measured noise".

**Pipeline.**

1. (C only) Convert RGB to Y/Cb/Cr.
2. **Pyramid**: `n = max(0, trunc(log2(min(W, H) / base)))`; levels 0..n, each half the size of the previous (2×2 box mean).
3. **Flat regions** are detected once, on the coarsest level, and stored in normalised coordinates.
4. **Coarse to fine**: for k = n down to 1, denoise level k to get `D_k`, then replace the low frequencies of the next finer level, `level_{k−1} ← level_{k−1} + up(D_k) − up(level_k)` (bilinear upsampling); then denoise level 0.
5. (C only) Convert back to RGB. Apply the blend.

**Flat-region detector.**

- `g = 0.299R + 0.587G + 0.114B` and `m = max(R, G, B)`.
- Variance box size `b` and smoothing size `Wv` from the image width w:

  | w | b | Wv |
  |---|---|---|
  | < 256 | 3 | 9 |
  | 256–511 | 5 | 9 |
  | 512–1023 | 7 | 15 |
  | ≥ 1024 | 9 | 25 |

- Local variance `v = box_b(g²) − box_b(g)²`, smoothed `V = box_Wv(v)`.
- Tile the image with cells of side `c = floor(1.5·Wv)`. In each cell take the minimum of V over the cell inset by `c/4` on each side; its value is the score and its location the candidate. Add 1000 to the score when, at the cell centre, `m > 220/255` (near clipping) or `g < 30/255` (too dark).
- Sort scores ascending. Keep candidates with `score ≤ 2·best`, at least 2 and at most 10. Each region is a c×c square centred on its candidate's minimum.

**Noise profile** (per channel, measured again at every pyramid level).

1. Map the flat regions to the level's pixel grid. For each region r (side `a = min(width, height)`), `μ_r` is the mean of the channel over the region.
2. Sample 2×2 patches `p` (4-vectors) on a G×G grid inside the region, spacing `d = max(1, floor(a/5))`, `G = ceil(a/d)` (about 5).
3. `C_r = 1/(G²−1) · Σ (p − μ_r)(p − μ_r)ᵀ` (4×4). Average over regions: `P = mean_r C_r`, `m_flat = mean_r μ_r`.
4. Regularise P (n = 4): `d̄ = mean(diag P)`; replace each interior diagonal entry `P[k][k]` (k = 1..n−2) by the mean of its four matrix neighbours; set `P[0][0] = P[0][1]` and `P[n−1][n−1] = P[n−1][n−2]`; median-filter P as an image twice (5-point cross inside, the available neighbours at edges and corners); Toeplitz-average the diagonal offsets 0..n−3 (both triangles); set the main diagonal to `d̄`.
5. Noise covariance `N = σ² · P` (σ = σ1 for channel 0, σ2 otherwise); keep `m_flat`.

With no flat region the input is returned unchanged.

**Denoising one level** (two passes). The guide image for the patch search is a luma computed from RGB (B) or Y (C). For every reference patch (top-left at stride-2 positions), compare the guide's 2×2 patch with every 2×2 patch whose top-left lies in the S×S window `[ref − h, ref − h + S)` by the L1 sum over the 4 pixels; keep the reference plus the K−1 smallest distances.

- *Pass 1 (basic estimate)*, per channel and group: stack the group into Y (4×K), mean `ȳ`, centred `Yc = Y − ȳ`, covariance `Cy = Yc·Ycᵀ / (K−1)`. Group noise covariance `Cn[i][j] = N[i][j] · 0.6^|i−j| · s`, with `s = (clamp(mean(ȳ), 0.2, 0.8) / m_flat)^(−7/6)` for B (more noise in darker groups) and `s = 1` for C. Clamp `Cy[i][i] = max(Cy[i][i], Cn[i][i])`. Each patch becomes `x̂ = ȳ + (yc − Cn·Cy⁻¹·yc)`, with an LDLᵀ 4×4 inverse; if any pivot is ≤ 2⁻²³ the group stays unfiltered. Patch estimates are added to their pixels and divided by the count.
- *Pass 2 (final estimate)*: repeat the search on the basic image's guide. For each group, gather the noisy patches Y and the basic patches Yb at the same positions; `ȳb` and `Cb = Ybc·Ybcᵀ / (K−1)` from the basic patches; `Cn` as in pass 1 (s from the noisy mean) times the final-step gain (2 for B, 1 for C); `x̂ = ȳ + Cb·(Cb + Cn)⁻¹·(y − ȳ)`. Aggregate as in pass 1.

**Details (BC-a to BC-i).**

- **BC-a Pyramid.** A 2×2 box mean; a level of odd size is `ceil(w / 2)` by `ceil(h / 2)` with its last column or row averaging the replicated edge. `up` is bilinear with pixel centres at half-integers, clamped at the edges, to the exact size of the finer level.
- **BC-b Detector input.** The detector reads RGB (method C runs it before its conversion), on the coarsest level. Cells are whole cells tiled from the frame's origin; a partial cell at the right or bottom is not used. The region square is clipped to the image.
- **BC-c Profiles.** Measured on the levels as built in step 2, before any level's low frequencies are replaced, so all of them are known before the coarse-to-fine walk. A region of `G = 1` adds nothing; with no usable region at a level, that level is passed through unchanged.
- **BC-d Patches at the edges.** Reference patch corners are the even positions from 0 up to `W − 2`, plus `W − 2` when it is odd (the same for y). The search window is clipped to the corners `[0, W − 2] × [0, H − 2]`, not moved. A group has the reference and the `K − 1` nearest of the others, or all the others when the window has fewer; a group of one is left unfiltered.
- **BC-e Pass 2.** A failed inversion (a pivot ≤ 2⁻²³) leaves the group unfiltered (`x̂ = y`), as in pass 1.
- **BC-f Aggregation order.** A reference's patches lie in `S + 1` px from `ref − h`, so passes of references `P` px apart (`P` the smallest even number ≥ `S + 1`) never overlap: the regular passes `(i, j)`, `i, j < P / 2`, in raster order, then the extra column, row and corner of BC-d in passes of their own as in A-c, and the patches of a group in slot order (the reference first).
- **BC-g Build details.** The depth is exact: the largest `n` with `base · 2^n ≤ min(W, H)`. `up` reads `u = (x + 0.5) / 2 − 0.5` (weights 3/4 and 1/4, indices clamped), and the replacement is `up(D_k − level_k)`, which equals `up(D_k) − up(level_k)` up to rounding. The detector's `b` and `Wv` come from the width of the level it reads (the coarsest); `v = max(0, box(g²) − box(g)²)`; its box filters mirror without repeating the edge sample; the inset is `floor(c / 4)`; the candidate is the first minimum in raster order; the penalty is read at `origin + floor(c / 2)`; ties of a score go to the smaller cell index; the region is `[m − floor(c / 2), m − floor(c / 2) + c)`, clipped. Regions map onto a level by `round(edge · size)`; `μ_r` is the mean over the whole region and the patch grid uses `a = min(width, height)` on both axes; a region with `a < 2` is skipped. The two median passes read the matrix before each pass; the median of an even count is the mean of the two middle values; Toeplitz averaging takes both triangles of a diagonal together, so the matrix stays symmetric. A pivot that is not a number counts as failed; a failure leaves that group unfiltered for that channel alone. Pass 2's `Cb` is about the basic patches' own mean, its `x̂` about the noisy mean. A channel of σ 0, a level under 2 px, and in method B a level whose `m_flat` is not positive pass through; with no region or no σ above 0 the input comes back bit for bit. `0.6^|i−j|` is a table and the powers come from `libm`, so the bits are the same on every platform.
- **BC-h Tiles.** Level `k` of a window starts at `x0 / 2^k`; windows are clipped to the buffer, so a tile does not equal the whole frame (a tile would have to start on a multiple of `2^n` and the measurement would have to come from the whole frame). In the chain the methods read the whole frame.
- **BC-i What the measurement does on detail.** Method B's detector reads a level where noise dominates the local variance (base 128), so on detailed photos its 52 px regions catch soft edges and its σ reads about 1.5 times the truth (15 against 10 in 1/255 units on a natural synthetic scene; within 6 % on a smooth one). Method C's 26 px regions match the analytic noise within 2 %.

### Appendix D: global noise level for a learned denoiser (method D)

Code: `cpu/noise_level.rs`. A pretrained network conditioned on two noise levels needs `levelY` and `levelC`; no such network is trained (`docs/07-ml-models.md`), so method D is the estimate alone, used by `denoise_gate` to compare with the model's noise.

1. RGB to `Y = 0.2126R + 0.7152G + 0.0722B`, `Cb = −0.117188R − 0.394228G + 0.511415B`, `Cr = 0.511416R − 0.464522G − 0.046894B`.
2. Flat rectangle: the best (lowest score) region of the B/C detector, run as for method C (base 256), mapped to full resolution (D-a).
3. High-pass every plane: `hp = x − GaussianBlur(x, 21×21, σ = 18.9)`, normalised, mirrored at the edges (D-b).
4. Inside the rectangle: `σY = std(hp_Y)` and `σC = std(hp_Cb ∪ hp_Cr)`; standard deviations are of the population (divided by N) and σC is one standard deviation of the Cb and Cr samples together (D-c).
5. `levelY = min(5·σY, 1)^0.7` and `levelC = min(5·σC, 1)^0.7`.

The detector picks the flattest cell, so on a frame whose short side is under 512 px (read at full resolution) σY reads about 8 % low; with one pyramid step it is within 3 %. For inference such a network would take the input clamped to [−0.2, 1.5], run in overlapping tiles and blend each tile with a linear ramp across the overlap.

# Single-frame image denoising: algorithm specification

The owner's specification (2026-10-20), as given. Sections 0 to D are his text; *Yellowmoon's reading* at the end
records how this repository reads the points the text leaves open, and *Integration* how the methods enter the raw
chain (`docs/05-raw-processing.md` R6 and R7).

This document specifies three denoisers that each work on one frame, plus a noise-level estimator used to
condition a learned denoiser:

| method | idea | working space | strength inputs |
|---|---|---|---|
| A | block matching + 3D DCT hard thresholding | Y / Cb / Cr | L, C, R |
| B | multiscale NL-Bayes, 2×2 patches, 8 per group, noise model measured from the image | R / G / B | L, R |
| C | multiscale NL-Bayes, 2×2 patches, 24 per group, measured noise model | Y / Cb / Cr | L, C, R |
| D | global noise level for a learned denoiser (appendix) | Y / Cb / Cr | – |

## 0. Conventions

- Images are float, black = 0 and white = 1, in the non-linear (display- or log-encoded) state the user sees.
  The fixed constants below (0.2–0.8 clamps, 30/255, 220/255) assume that scale.
- Strength inputs: luma strength `L ∈ [0,100]`, chroma strength `C ∈ [0,100]`, radius `R` (a small
  non-negative integer, typically 0–2), blend `B ∈ [0,100]`.
- Y/Cb/Cr: Y in [0,1], Cb and Cr centred on 0. Any standard matrix works (BT.709 suggested). Convert in
  before denoising and back after.
- Final blend for every method: `out = lerp(denoised, input, B/100)`, applied only when `B > 0`.
- `box_k(x)` is a k×k mean filter. `clamp`, `lerp`, `max` have their usual meaning.

---

## A. Block matching + 3D DCT hard thresholding

### A.1 Parameters
```
τY = 0.1  · L                  hard threshold, luma
τC = 0.06 · C                  hard threshold, chroma
θ  = 0.2  · max(L, C)          block-distance threshold
s  = 8 − 2·R                   reference-block step (pixels), R ∈ {0,1,2,3}
```
Skip the luma channel if `τY ≤ 1e-6`, and both chroma channels if `τC ≤ 1e-6`.

### A.2 Window
`K(i,j) = k(i)·k(j)`, i,j ∈ 0..7, where `k(n) = I0(2·sqrt(1 − (2n/7 − 1)²)) / I0(2)`, a Kaiser window with
N = 8 and β = 2:
`k = [0.4387, 0.6813, 0.8768, 0.9858, 0.9858, 0.8768, 0.6813, 0.4387]`. The sum of K is ≈ 35.59.

### A.3 Per reference block
Reference blocks are 8×8 with top-left corners on a regular grid of step `s` in x and y.

1. **Candidates**: 25 blocks with top-left at `ref + (dx, dy)`, where `dx, dy ∈ {−8, −4, 0, 4, 8}`.
   Index them 0..24 in row-major order; index 12 is the reference.
2. **Distance**, on Y only: `D = Σ_{i,j} K(i,j) · |Y_ref(i,j) − Y_cand(i,j)|`.
3. **Grouping**: slot 0 is always the reference. Fill slots 1..7 with the 7 non-reference candidates of
   smallest D. Any slot whose D > θ is replaced by the reference block, so duplicates are allowed.
   The result is a group of 8 blocks.
4. **3D transform**, per channel (Y with τY; Cb and Cr with τC; chroma reuses the luma grouping): an
   8-point DCT-II along rows, then columns, then across the 8 blocks. The transform is unnormalised: each
   1D pass is the orthonormal DCT-II multiplied by √8, so the DC term is the plain sum. Relative to an
   orthonormal transform every coefficient is 8^1.5 ≈ 22.6× larger, and τ is compared on that scale.
5. **Hard threshold**: set every coefficient with `|c| < τ` to 0, including the DC term.
   Optional highlight tweak: when the reference block's top-left pixel has `Y > 0.4`, use the threshold
   `τ·(1.4 − Y)` for one 8-coefficient line of the spectrum. Its effect is minor; it can be omitted or
   applied to the whole spectrum.
6. **Inverse 3D transform** (same scaling, so a forward and inverse round trip gains 512).
7. **Aggregation**: for every pixel of every block in the group, `acc_ch += K(i,j) · v / 512` and
   `w += K(i,j)`. The weight is accumulated once per pixel, not per channel.

### A.4 Output
`out_ch = acc_ch / w`. With `s ≤ 8` every pixel receives at least one contribution.

### A.5 Parallel scheduling (optional)
A reference block touches at most a 24×24 footprint (8 px block, ±8 px search). Process references in
passes: pass (i, j) handles references at `(24·m + i·s, 24·n + j·s)`. Within a pass, footprints never
overlap, so accumulation needs no atomics.

---

## B and C. Multiscale NL-Bayes on 2×2 patches

### B/C.1 Parameters
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
σ is a gain on the measured noise, not an absolute level. In method B, `L = 50` means "remove exactly the
measured noise". Method B ignores the chroma strength C.

### B/C.2 Pipeline
1. (C only) Convert RGB → Y/Cb/Cr.
2. **Pyramid**: `n = max(0, trunc(log2(min(W, H) / base)))`. Build levels 0..n, each half the size of the
   previous (factor 0.5; any standard area or box 2× downsampler).
3. **Flat regions**: detected once, on the coarsest level (§B/C.3), and stored in normalised coordinates.
4. **Coarse to fine**: for k = n down to 1:
   - denoise level k (§B/C.5) to get `D_k`;
   - replace the low frequencies of the next finer level:
     `level_{k−1} ← level_{k−1} + up(D_k) − up(level_k)` (bilinear upsampling);

   then denoise level 0.
5. (C only) Convert back to RGB. Apply the blend.

### B/C.3 Flat-region detector
- `g = 0.299R + 0.587G + 0.114B` and `m = max(R, G, B)`.
- Pick a variance box size `b` and smoothing size `Wv` from the image width w:

  | w | b | Wv |
  |---|---|---|
  | < 256 | 3 | 9 |
  | 256–511 | 5 | 9 |
  | 512–1023 | 7 | 15 |
  | ≥ 1024 | 9 | 25 |

- Local variance `v = box_b(g²) − box_b(g)²`, smoothed `V = box_Wv(v)`.
- Tile the image with cells of side `c = floor(1.5·Wv)`. In each cell, take the minimum of V over the
  cell inset by `c/4` on each side. Its value is the score and its location the candidate. Add 1000 to the
  score when, at the cell centre, `m > 220/255` (near clipping) or `g < 30/255` (too dark).
- Sort scores ascending. Keep candidates with `score ≤ 2·best`, but always at least 2 and at most 10.
- Each region is a c×c square centred on its candidate's minimum.

### B/C.4 Noise profile (per channel, re-measured at every pyramid level)
1. Map the flat regions to the level's pixel grid. For each region r (side `a = min(width, height)`),
   let `μ_r` be the mean of the channel over the region.
2. Sample 2×2 patches `p` (4-vectors) on a G×G grid inside the region, with spacing `d = max(1, floor(a/5))`
   and `G = ceil(a/d)` (≈ 5).
3. `C_r = 1/(G²−1) · Σ (p − μ_r)(p − μ_r)ᵀ`, a 4×4 matrix.
   Average over regions: `P = mean_r C_r`, `m_flat = mean_r μ_r`.
4. **Regularise P** (n = 4):
   - `d̄ = mean(diag P)`.
   - Replace each interior diagonal entry `P[k][k]` (k = 1..n−2) by the mean of its four matrix neighbours.
     Set `P[0][0] = P[0][1]` and `P[n−1][n−1] = P[n−1][n−2]`.
   - Median-filter P as an image, twice: 5-point cross (self + 4 neighbours) inside, the available
     neighbours at edges (4 values) and corners (3 values).
   - Toeplitz averaging: for diagonal offsets 0..n−3, set every entry of that diagonal (both triangles)
     to the diagonal's mean.
   - Set the main diagonal to `d̄`.
5. **Noise covariance** for the channel: `N = σ² · P` (σ = σ1 for channel 0, σ2 otherwise). Keep `m_flat`.

If no flat region is found, skip denoising and output the input.

### B/C.5 Denoising one level (two passes)
**Guide image** for patch search: method B uses a luma image computed from RGB (any standard luma
weights); method C uses Y.

**Patch search**: for every reference patch (top-left at stride-2 positions), compare the guide's 2×2 patch
with every 2×2 patch whose top-left lies in the S×S window `[ref − h, ref − h + S)`. Distance is the L1 sum
over the 4 pixels. Keep the reference plus the K−1 smallest distances.

**Pass 1 (basic estimate)**, per channel, per group:
- Stack the group into Y (4 × K). Mean per dimension: `ȳ` (4-vector). Centred: `Yc = Y − ȳ`.
- Empirical covariance: `Cy = Yc·Ycᵀ / (K−1)`.
- Group noise covariance: `Cn[i][j] = N[i][j] · 0.6^|i−j| · s`, where for method B
  `s = (clamp(mean(ȳ), 0.2, 0.8) / m_flat)^(−7/6)` (more noise in darker groups) and s = 1 for method C.
- Clamp: `Cy[i][i] = max(Cy[i][i], Cn[i][i])`.
- Estimate each patch: `x̂ = ȳ + (yc − Cn·Cy⁻¹·yc)`. Invert with an LDLᵀ/Cholesky 4×4 SPD inverse. If any
  pivot is ≤ 2⁻²³ (≈ 1.19e−7), leave the group unfiltered (x̂ = y).
- **Aggregate**: add every patch estimate to its pixels, count contributions, then divide by the count.
  This gives the basic image.

**Pass 2 (final estimate)**: redo the patch search on the basic image's guide. For each group, gather the
noisy patches Y and the basic patches Yb at the same positions.
- `ȳ`, `ȳb` are their means. `Cb = Ybc·Ybcᵀ / (K−1)`.
- `Cn` as in pass 1 (s computed from the noisy mean ȳ), multiplied by the final-step noise gain
  (2 for B, 1 for C).
- `x̂ = ȳ + Cb·(Cb + Cn)⁻¹·(y − ȳ)`.
- Aggregate as in pass 1.

---

## Appendix D. Global noise level for a learned denoiser

The method-D denoiser is a pretrained network conditioned on two noise levels. The network itself is not
specified here. The conditioning is:

1. RGB → `Y = 0.2126R + 0.7152G + 0.0722B`,
   `Cb = −0.117188R − 0.394228G + 0.511415B`, `Cr = 0.511416R − 0.464522G − 0.046894B`.
2. Flat rectangle: the B/C.3 detector (or a user-supplied rectangle).
3. High-pass every plane: `hp = x − GaussianBlur(x, 21×21, σ = 18.9)`.
4. Inside the rectangle: `σY = std(hp_Y)` and `σC = std(hp_Cb ∪ hp_Cr)`.
5. `levelY = min(5·σY, 1)^0.7` and `levelC = min(5·σC, 1)^0.7`.

Inference notes: clamp the input to [−0.2, 1.5]; run in overlapping tiles; blend each tile into the
output with a linear ramp across the overlap (weight `(x/ox)·(y/oy)` inside the leading overlap, else 1).

---

## Yellowmoon's reading

The points the text leaves open, read the simplest way that keeps its numbers (`AGENTS.md`, source of truth). Each
item is pinned by a test in `crates/ym-gpu/src/cpu/` (named in brackets).

**General.**

- *G1 Data.* The reference functions take and give `RgbImage` (three `f32` planes) in the encoded state; samples
  outside [0, 1] are kept, never clamped, except where the text clamps. Arithmetic inside is `f64` (`AGENTS.md`,
  colour maths on the CPU reference path).
- *G2 Y/Cb/Cr of methods A and C.* BT.709: `Y = 0.2126 R + 0.7152 G + 0.0722 B`, `Cb = (B − Y) / 1.8556`,
  `Cr = (R − Y) / 1.5748`, and the exact inverse. Appendix D keeps its own coefficients (they are BT.709's
  scaled by 224/219), used for the estimate only.
- *G3 Determinism.* The output is the same bits for any number of threads: every sum that more than one
  reference adds to is accumulated in a fixed order (method A by the passes of A.5 in the order `(i, j)` raster,
  then the references of a pass in raster order, then the slots of a group in order; methods B and C the same
  with their own passes, below). Ties of a distance are broken by the candidate's raster index (smaller first).
- *G4 Frame coordinates.* Every grid (A's reference blocks, B and C's reference patches, the detector's cells) is
  anchored to the frame's origin (`RgbImage::place`), never to the buffer's, so that a tile with the stated halo
  gives the whole frame's pixels (method A now; methods B and C once their tiling is built, *Integration*).
- *G5 Blend.* `B = 0` or no blend input means no blend; the blend uses the input as given to the method (in RGB).

**Method A.**

- *A-a Edges.* A candidate block that does not lie wholly inside the frame is not a candidate (distance +∞); a slot
  left without a candidate takes the reference, like a slot over θ. Reference corners are the multiples of `s`
  from 0 up to `W − 8`, plus `W − 8` when it is not one of them (the same for y), so every pixel is covered. A
  frame narrower or lower than 8 px is returned unchanged.
- *A-b Highlight tweak.* Omitted (the text allows it).
- *A-c Passes.* A.5's passes over the regular grid come first, `(i, j)` in raster order (`i, j < 24 / s`; every
  `s` of `R ≤ 3` divides 24). The references that A-a adds follow in passes of their own, so that two references
  of one pass are always 24 px apart: the extra column (`x = W − 8`) in passes `j` by `y = 24 n + j s`, then the
  extra row (`y = H − 8`) in passes `i` by `x = 24 m + i s`, then the corner `(W − 8, H − 8)`.
- *A-d Halo.* A pixel depends on the references whose group can cover it (corners within 15 px to the left and
  above, 8 px to the right and below) and those read within 8 px of their own block: 23 px on each side. A
  buffer that is a window of the frame computes only the references whose read span `[r − 8, r + 16)`, clipped
  to the frame, lies inside it; the others reach only the margin the caller drops (22 px fails the tile test).
- *A-e Details of the build.* "Raster order" of the passes is the y phase `j` outer, the x phase `i` inner. `R`
  above 3 is read as 3, so the step stays positive. A channel is filtered when its `τ > 1e-6`. Inputs must be
  finite (a distance that is not finite is not a candidate). The cosine table comes from `libm`, not the
  platform's library, so the bits are the same on macOS and Linux; `2n/7 − 1` is computed as `(2n − 7)/7`, so
  the window is exactly symmetric.

**Methods B and C.**

- *BC-a Pyramid.* A 2 × 2 box mean; a level of odd size is `ceil(w / 2)` by `ceil(h / 2)` with its last column or
  row averaging the replicated edge. `up` is bilinear with pixel centres at half-integers, clamped at the edges,
  to the exact size of the finer level.
- *BC-b Detector input.* The detector reads RGB (method C runs it before its conversion), on the coarsest level.
  Cells are whole cells tiled from the frame's origin; a partial cell at the right or bottom is not used. The
  region square is clipped to the image.
- *BC-c Profiles.* Measured on the levels as built in step 2, before any level's low frequencies are replaced, so
  all of them are known before the coarse-to-fine walk. A region of `G = 1` adds nothing; with no usable region
  at a level, that level is passed through unchanged.
- *BC-d Patches at the edges.* Reference patch corners are the even positions from 0 up to `W − 2`, plus `W − 2`
  when it is odd (the same for y). The search window is clipped to the corners `[0, W − 2] × [0, H − 2]`, not
  moved. A group has the reference and the `K − 1` nearest of the others, or all the others when the window has
  fewer; a group of one is left unfiltered.
- *BC-e Pass 2.* A failed inversion (a pivot ≤ 2⁻²³) leaves the group unfiltered (`x̂ = y`), as in pass 1.
- *BC-f Aggregation order.* As method A: a reference's patches lie in `S + 1` px from `ref − h`, so passes of
  references `P` px apart (`P` the smallest even number ≥ `S + 1`) never overlap; the regular passes `(i, j)`,
  `i, j < P / 2`, in raster order, then the extra column, row and corner of BC-d in passes of their own as in A-c,
  and the patches of a group in slot order (the reference first).

- *BC-g Details of the build.* The depth is exact: the largest `n` with `base · 2^n ≤ min(W, H)`. `up` reads
  `u = (x + 0.5) / 2 − 0.5` (weights 3/4 and 1/4, indices clamped), and the replacement is `up(D_k − level_k)`,
  which equals `up(D_k) − up(level_k)` up to rounding (`up` is linear). The detector's `b` and `Wv` come from the
  width of the level it reads (the coarsest); `v = max(0, box(g²) − box(g)²)`; its box filters mirror without
  repeating the edge sample; the inset is `floor(c / 4)`; the candidate is the first minimum in raster order;
  the penalty is read at `origin + floor(c / 2)`; ties of a score go to the smaller cell index; the region is
  `[m − floor(c / 2), m − floor(c / 2) + c)`, clipped. Regions map onto a level by `round(edge · size)`; `μ_r` is
  the mean over the whole region and the patch grid uses `a = min(width, height)` on both axes; a region with
  `a < 2` is skipped. The two median passes read the matrix before each pass; the median of an even count is the
  mean of the two middle values; Toeplitz averaging takes both triangles of a diagonal together, so the matrix
  stays symmetric. A pivot that is not a number counts as failed; a failure leaves that group unfiltered for that
  channel alone. Pass 2's `Cb` is about the basic patches' own mean, its `x̂` about the noisy mean, as written. A
  channel of σ 0, a level under 2 px, and in method B a level whose `m_flat` is not positive, pass through; with
  no region or no σ above 0 the input comes back bit for bit. `0.6^|i−j|` is a table and the powers come from
  `libm`, so the bits are the same on every platform.
- *BC-h Tiles, not yet.* Level `k` of a window starts at `x0 / 2^k`; the windows are clipped to the buffer, so a
  tile does not equal the whole frame yet (a tile would also have to start on a multiple of `2^n`, and the
  measurement would have to come from the whole frame). In the chain the methods read the whole frame
  (*Integration*).
- *BC-i What the measurement does on detail.* Method B's detector reads a level where noise dominates the local
  variance (base 128), so on detailed photos its 52 px regions catch soft edges and its σ reads about 1.5 times
  the truth (15 against 10 in 1/255 units on a natural synthetic scene; within 6 % on a smooth one). Method C's
  26 px regions match the analytic noise within 2 %. This follows from the text, not from the build.

**Appendix D.**

- *D-a Rectangle.* The best region of the detector (lowest score), run as for method C (base 256), mapped to full
  resolution.
- *D-b Blur.* 21 × 21 taps, σ 18.9, normalised, mirrored at the edges. The network itself is not part of this
  repository: no trained network exists (`docs/07`), so method D is the estimate alone.
- *D-c Statistics.* Standard deviations are of the population (divided by N); σC is one standard deviation of
  the Cb and Cr samples together. The detector picks the flattest cell, so on a frame whose short side is under
  512 px (read at full resolution) σY reads about 8 % low; with one pyramid step it is within 3 %.

## Integration

How the methods enter the raw chain (`docs/05-raw-processing.md` R6 and R7). The code: `ym_gpu::cpu::frame_denoise`
(the stage), `ym_gpu::cpu::denoise_space` (the encoding), `Chain::frame_denoise` (`cpu/chain.rs`).

- **Where.** R6, after the demosaic, in place of the light bilateral denoise and R7 (the colour guided filter and
  the detail re-injection): the stage `Stage::FrameDenoise`. The fit view and the thumbnails, which bin the mosaic
  and never demosaic, keep the bilateral denoise of the binned image.
- **Working encoding.** The raw chain holds balanced linear camera RGB, white 1.0 at the clip before the gains. The
  methods are made for the encoded image the viewer sees, so the stage encodes with `W(x) = srgb(2^CAMERA_GAIN_EV x)`
  (the sRGB curve on the whole line, its linear slope made exact at the knee), runs the method, and decodes. Middle
  grey of a photo at the camera's exposure lands at 0.46, as in an sRGB picture, and the 1/2.4 power nearly evens
  out shot noise, so one threshold suits the shadows and the lights. The encoding depends on nothing that changes
  after R6 (exposure, look and curves are colour stages), so a slider there never re-runs the raw chain.
- **Controls.** The Detail tab's controls keep their Auto values and their meaning relative to Auto (`docs/05` R7):
  with `r = L / L_auto(ISO)` and `c = C / 50`, method A gets `L_A = r · kL · σY · 22.627 / 0.1` and
  `C_A = c · kC · σC · 22.627 / 0.06`, where `σY` and `σC` are the photo's luma and chroma noise in the working
  encoding at middle grey from its noise model and R4 gains (`working_noise`), and `kL = 3.5`, `kC = 8`
  (*Measurement*); method B `L_B = 30 r`; method C `L_C = 80 r`, `C_C = 90 c`. Detail sets the blend to today's
  re-injection weight, `B = 100 · 0.15 (D / 50)²` (15 % at Auto, 60 % at 100). `R` is 1 for every method. The
  strengths are clamped to [0, 100].
- **The default is method A** (the owner, 2026-10-20: "I don't need A/B test, I think this is good"; the choice by
  the measurement below). B and C stay behind the renderer's internal switch (`DenoiseMethod`), like the demosaic
  methods of R6.
- **Tiles.** Method A asks for its 23 px halo and a tile equals the whole frame bit for bit (tested through the
  chain). Methods B and C read the whole frame (pyramid and measurement), so a tile of them asks for the whole
  raw frame: correct and exact, and slow; they are for evaluation.

## Measurement

`crates/ym-gpu/examples/denoise_gate.rs` (`cargo run -p ym-gpu --release --example denoise_gate -- all`), 2026-10-20,
Linux x86_64, 3 threads of a shared 4-core machine. Kodak (24 images) and McMaster (18) from `testdata/demosaic`
in linear light, placed where a photo at the camera's exposure sits on the raw scale (middle grey 0.083), mosaicked
as RGGB, Gaussian noise of variance `a x + b` per photosite at five ISO-like levels (the demosaic gate's stand-ins
for 100, 800, 1600 and 6400, and 12800 by the same scaling; not measurements, no row noise), the adaptive ratio
demosaic with the matching model. PSNR against the clean image in the working encoding (peak 1), mean of per-image
dB, 10 px border excluded; "today" is the bilateral denoise with a σ map and R7 at Auto.

*Calibration of method A* (first 8 Kodak images, `kL` 2.5 to 5 against `kC` 6 to 16): the best `kL` per level is
3.5, 3, 3.5, 4, 4 and the best `kC` 16, 8, 8, 6, 6 (ISO 100 to 12800); `kL = 3.5`, `kC = 8` is within 0.12 dB of the
best luma setting and 0.2 dB of the best chroma setting at every level. The noise measured after the demosaic is 1.5
to 3.9 times the model's (the demosaic's error dominates at low ISO and the demosaic correlates the noise); the
constants are fitted against the model's noise, which is what a render knows. Appendix D's estimate reads 0.69 to
0.96 of that measured σY (lower as the noise grows) and 0.53 to 0.63 of its σC.

*Comparison* (R 1, blend 15, the render's settings; working encoding, dB; in brackets C without the blend):

| set | ISO-like | none | today | A | B (L 30) | C (L 80, C 90) |
|---|---|---|---|---|---|---|
| Kodak | 100 | 36.89 | 35.27 | **37.14** | 36.79 | 37.20 (37.17) |
| Kodak | 800 | 34.54 | 34.19 | **35.84** | 35.24 | 35.74 (35.77) |
| Kodak | 1600 | 32.78 | 33.35 | **34.72** | 33.93 | 34.51 (34.61) |
| Kodak | 6400 | 27.61 | 30.61 | **31.31** | 29.75 | 30.69 (31.00) |
| Kodak | 12800 | 25.05 | 29.10 | **29.44** | 27.45 | 28.50 (28.91) |
| McMaster | 100 | 34.07 | 31.62 | **34.30** | 34.22 | 34.47 (34.43) |
| McMaster | 800 | 32.01 | 30.91 | **33.13** | 32.84 | 33.24 (33.31) |
| McMaster | 1600 | 30.69 | 30.35 | **32.27** | 31.82 | 32.32 (32.45) |
| McMaster | 6400 | 26.82 | 28.42 | **29.90** | 28.80 | 29.53 (29.84) |
| McMaster | 12800 | 24.73 | 27.19 | **28.49** | 26.88 | 27.87 (28.27) |

Method A is above today's path at every level of both sets: +1.9, +1.7, +1.4, +0.7 and +0.3 dB on Kodak and +2.7,
+2.2, +1.9, +1.5 and +1.3 on McMaster from ISO 100 to 12800. **Today's path is below no denoise at all at ISO 100 and
800** (it smooths detail that the noise did not hide); A is above no denoise everywhere, so it does not soften a
clean photo. C is 0.1 to 0.2 dB above A at ISO 100 and 800 and below it from 1600 up (0.05 to 0.6 dB with the blend,
up to 0.5 dB without), at twice A's time and with the whole frame to read; B is below both. `R = 2` adds 0.02 to
0.07 dB to A (without the blend) at 1.5 times its time; `R = 0` loses 0.05 to 0.4 dB. Seconds per megapixel on the
CPU (wall, 3 threads): today 0.2 to 0.3, A 0.5, B 0.9, C 1.0. The full
tables: `testdata/denoise_gate/numbers.md` after a run.

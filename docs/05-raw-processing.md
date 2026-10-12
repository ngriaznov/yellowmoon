# 05 · Raw processing (mosaic to scene-linear camera RGB)

Crates: `ym-core` (orchestration), `ym-gpu` (kernels), `ym-ml` (network), `ym-noise` (σ map). All stages run in scene-linear float, in tiles on the GPU, with a CPU reference for tests and for machines without a usable GPU.

**CPU reference (M1-12).** `ym-gpu::cpu` holds the reference implementation of R1 and R3 to R9 (one module per stage, `cpu::chain` for the whole chain, whole frame or one tile). The WGSL kernels are parity-tested against it. Conventions: `f32` storage; `f64` for positions, spline evaluation and anything that accumulates; images are *planar* (all R, then all G, then all B) because the GPU reads a plane as one coalesced range and R7 to R9 treat planes separately; every buffer carries its position in the full frame (`Placement`) so tiles give the whole-frame result; borders mirror (parity-preserving) except R9, which clamps. Inputs are plain slices and a 2×2 `Cfa`; the mapping from `ym_raw::RawImage` is in the module documentation of `ym-gpu::cpu`.

## Order

| # | Stage | Notes |
|---|---|---|
| R1 | Black subtract, white normalise, clip mask | per CFA position, from SR2 levels |
| R2 | Noise estimate and illuminant estimate | on binned planes / the 1/8 preview (`docs/06`, `docs/02`) |
| R3 | Vignetting correction | Sony 0x7032 |
| R4 | Initial white-balance gains `g0` | from the illuminant estimate |
| R5 | Highlight reconstruction on the mosaic | opposed-channel method |
| R6 | Demosaic + denoise | neural (export, 1:1) or classical (preview, fallback) |
| R7 | Detail controls | colour noise, detail re-injection |
| R8 | Capture sharpening | Richardson–Lucy, Gaussian PSF |
| R9 | Distortion + lateral CA in one resample | Sony 0x7037, 0x7035 |

Output: camera RGB balanced with `g0`, white = 1.0 before gains, consumed by the colour pipeline (`docs/02`, stage C1).

## R1 · Normalise

`x = (raw − black[c]) / (white[c] − black[c])` per CFA position. Negative values stay (they carry noise statistics); clamp at −0.05. Clip mask: `raw ≥ white[c] − 4·qstep` (or `≥ 0.995` normalised where qstep is absent). The mask travels with the image to R5, R6 and C7. Black and white levels are indexed by CFA *position* `2·(y&1) + (x&1)` of the active area (for the all-RGGB Sony bodies that is R, G1, G2, B); a tile uses the frame parity of its photosites. `Levels::from_rggb` maps values in the R, G1, G2, B order of the SR2 tags onto the positions of any pattern phase (a frame cropped to an odd origin starts with GRBG, say). `qstep` present with value 0 (lossless files fed to the network as a zero plane) means "clipped only at exactly the white level"; the 0.995 rule applies only when no qstep plane exists.

## R3 · Vignetting (0x7032)

Knots `v_i`, i = 0…n−1 (n = 16 on current bodies). Knot `i` sits at the normalised radius `r_i = (i + 0.5)/(n−1)`, where r = 1 is the half-diagonal of the active area (the last knot lies half a step beyond the corner). A knot gives the relative illumination the lens leaves at its radius: `G_i = 2^(0.5 − 2^(v_i·2^−13 − 1))`. Interpolate `G(r)` **linearly** between the knots (end values held outside them) and **divide** each photosite by `G(r)`, clamping `G` at 1e-4.

*Status (M1-12, verified):* the first reading of this section (squared gain, knots at `i/(n−1)`, a monotone cubic, multiplying by the gain) was wrong on four counts, found by running the lens oracle (§Verifying lens corrections) on synthetic Sony files. `darktable-cli` 4.6.1 (embedded metadata, algorithm 2, the module's default) applies exactly the rule above: the radial gain it produces agrees with `VignetteProfile::gain_at` to 8e-5 (gate: 0.5%; `tests/cpu_lens.rs::vignetting_matches_darktable`). Readings rejected by the same renders (maximum gain deviation): this section's first reading, multiplying by `G²` with a cubic through knots at `i/(n−1)` (66%); the same dividing by `G²` (17%); darktable's older algorithm 1, dividing by `G²` with the right knots (15%); the right gain with knots at `i/(n−1)` (1.4%). So positive `v` means a darker corner (`G < 1`, a gain up to 181 at the largest encodable knot) and negative `v` a gain below one; a typical lens has `v` from 0 at the centre to a few thousand at the corner. Radius: the centre of a pixel measured from the centre of the active area, over half its diagonal, `hypot(width, height) / 2` (darktable counts pixel indices instead of pixel centres, a half-pixel shift worth about 1e-5 in the gain). `ym_raw::LensTable::knots()` drops the stored count; note that the count of tag 0x7035 is 2n (darktable rejects the data unless it is), not n.

## R4 · Initial gains

`g0 = 1 / w_scene_initial` (green = 1), from the illuminant estimate at open. If the estimate fails, use the as-shot gains from SR2 0x7313. Gains are applied to the mosaic before R5 and R6 because both work better on balanced data. Later white-balance changes do not re-run R5–R9 (`docs/02` C1 applies a residual). The clip mask is not changed by the gains; a clipped photosite then sits at `g0_c` (times the vignetting gain), and R5 finds it through the mask, not by a threshold. SR2 0x7313 levels are multipliers (green about 1024), so `g0 = [R/G, 1, B/G]` with `G` the mean of the two green levels (`gains::g0_from_wb_levels`); from a scene white `w`, `g0 = [w_G/w_R, 1, w_G/w_B]` (`gains::g0_from_scene_white`).

## R5 · Highlight reconstruction (opposed channels)

Clean-room from darktable's published description ("inpaint opposed"); don't read or port its code.

1. Build a half-resolution superpixel image from each 2×2 quad: R, G = mean of the two greens, B, with a per-channel clipped flag.
2. For each clipped channel `c` of a superpixel, the **opposed estimate** is the cube-root mean of the unclipped other channels over the 3×3 superpixel neighbourhood: `ŷ = (mean(x^(1/3)))³`.
3. **Global chroma correction:** over all unclipped superpixels that border clipped ones, average `x_c − ŷ_c`; add that offset to every estimate for channel `c`.
4. Reconstructed value: `max(original, ŷ_c + offset_c)`. Write it back into the mosaic photosites of that channel, keeping the clip mask so later stages know these values are estimates.

*Implementation notes (M1-12):*

- A superpixel's green is flagged clipped when either green photosite is; the estimate is written to the clipped green photosites only. With no usable neighbour (all other channels clipped over the whole 3×3) the photosite keeps its value.
- Step 3 is read per channel: for channel `c`, the reference superpixels are those with no clipped channel that have channel `c` clipped somewhere in their 3×3 neighbourhood, so a red lamp does not bias the blue offset. `ŷ_c` there is computed as if `c` were clipped, from the unclipped other channels of the neighbourhood.
- The offsets are a global reduction. For tiles, accumulate `OffsetAccumulator`s over the tile cores in a fixed order, merge, and pass the same `ChromaOffsets` to each tile (`highlights::{accumulate_offsets, apply_offsets}`); applying needs only one superpixel (2 px) of context, the table below keeps the documented 4 px. Tiles start on even coordinates.
- Because the offset is one additive constant per channel, calibrated where a channel has just clipped, it under-reconstructs cores that are many stops over the clip level: a source at 2.6× the clip level comes back at roughly 1.3× with a paler hue (synthetic scenes in `tests/cpu_highlights.rs`). Region chroma (chroma-only ΔE00 of the mean colour of the blown region against the unclipped surround, through the stand-in camera matrix in `cpu::metrics`) is within 4.7 of the surround in all ten synthetic scenes (gate: 5; up to 10.4 without recovery), and the hue at the clip boundary is within 13° of the truth for gentle falloffs (18° and 24° without recovery on the two scenes where it matters). For hard-edged sources (all channels clip within a few pixels) recovery changes the boundary hue only a little (37° and 67° against 39° and 69° without). Whether this passes the blind review on the 20 real photos is open.
5. `decode.highlight_recovery = false` skips R5; clipped values stay at 1.0 (they then render as white or a hue cast, which is the point of the switch).

Gate: on 20 test photos with blown skies and lamps, no magenta or cyan fringes at the clip boundary in blind review, and the reconstructed region's chroma within ΔE00 5 of the surrounding unclipped area.

## R6 · Demosaic and denoise

### Neural (default for export and 1:1)

One U-Net does both (`docs/07-ml-models.md`). Per tile, inputs:

- packed Bayer planes (4 × H/2 × W/2), balanced with `g0`;
- σ map (1 plane, `docs/06-noise.md`), scaled by the Luminance control;
- quantisation-step map (1 plane; zeros for lossless files);
- clip mask (1 plane);

output: linear camera RGB at full resolution. Tiles overlap by half the network's receptive field, computed from the architecture (64 full-res px for the starting U-Net in `docs/07`), and are cropped, not blended. A test renders a frame tiled and whole and requires a max difference ≤ 1e-4.

### Classical (preview and fallback)

Menon 2007 (DDFAPD) or Zhang–Wu LMMSE, ported from permissive references: colour-demosaicing (BSD-3) for Menon, Getreuer's IPOL code (BSD-2) for LMMSE. Pick the one with better PSNR on the test sets in M1. Followed by a light classical denoise (bilateral in an opponent space, strength from σ) so previews aren't noisier than the export.

*Decision (M1-12, revised 2026-10-12): the default classical method is **Menon**; LMMSE and bilinear stay selectable (`demosaic::Method`). LMMSE wins on gamma-encoded sRGB test images (below), but the pipeline demosaics scene-linear data, where our LMMSE's fixed regularisation constants make it scale-dependent: on Kodak in linear light it gives 36.67 / 36.17 / 35.61 dB at signal scales 1.0 / 0.1 / 0.01 against Menon's constant 36.43 dB. Shadows matter more than a 0.2 dB lead at full scale. Revisit if LMMSE is made scale-invariant.* All methods handle the four Bayer phases and mirror at the borders.

- **Menon** is a port of `colour_demosaicing.demosaicing_CFA_Bayer_Menon2007` (BSD-3-Clause, Copyright 2015 Colour Developers; licence text in the source and to be listed in `NOTICE`), checked against the package's output on fixtures (max difference < 3e-5, all phases, with and without the refining step).
- **LMMSE**: Getreuer's IPOL code could not be fetched (ipol.im is unreachable from the build environment), so this is our own implementation of the Zhang–Wu idea, not a port: horizontal and vertical colour-difference estimates (from the `[−1 2 2 2 −1]/4` filter) are Wiener-smoothed over a 9-tap Gaussian window with local statistics (noise variance from the second-difference energy, `eps` 1e-5), fused with weights inverse to their error variances; red and blue come from colour-difference interpolation along the diagonal of smaller variation.
- **PSNR** (sRGB 8-bit images mosaicked as RGGB, 10 px border excluded, mean of per-image dB, local evaluation only: the sets are never committed): Kodak (24 images, the 500×500 version distributed with cszn/FFDNet) bilinear 29.82, Menon 38.88, **LMMSE 39.16**; McMaster (18 images, the copy in JingyunLiang/SwinIR) bilinear 32.31, Menon 34.21, **LMMSE 35.09**. On the procedural set in `tests/cpu_demosaic.rs` (edges, fine text, correlated-channel scenes) LMMSE also leads; on the zone plate and flat patches all methods are within 0.7 dB.
- **Denoise** (`denoise::bilateral_opponent`): 5×5 bilateral on `(Y, R−Y, B−Y)`, spatial σ 1.5 px, range bandwidth `h = strength · √6 · σ_noise` on the Euclidean opponent-space distance (`√6 σ` is the RMS distance of two pure-noise pixels, so strength 1 gives a typical noise pair weight about 0.6); σ is the scalar or σ-map value at the centre pixel, scaled by `(L/L_auto)^1.5` like the network's σ. σ = 0 leaves a pixel untouched.

`decode.neural_demosaic = false` forces the classical path everywhere, including export.

## R7 · Detail controls

`noise_reduction` resolves to `{ luminance L, color C, detail D }`, each 0–100. Auto values: `L_auto = clamp(10 · (1 + log2(ISO/100)), 10, 90)` (ISO 100 → 10, ISO 400 → 30, ISO 3200 → 60, ISO 25600 → 90; the number grows with ISO so the slider reads naturally), `C_auto = 50`, `D_auto = 50`. The Automatic card shows the resolved values ("Luminance 30").

| Control | Operation |
|---|---|
| Luminance `L` | σ fed to the network = `σ_est · (L / L_auto)^1.5`: the estimated noise at the auto value, 0 at L = 0 (the network demosaics without denoising), up to (100/L_auto)^1.5 × at 100. The network is trained across σ including 0, so this is a true strength control. |
| Colour `C` | After the network: guided filter on the two chroma channels of an opponent space (Y, R−Y, B−Y), guided by Y. Radius `2 + 6·C/100` px at full resolution, ε from `σ_est² · (C/50)²`. |
| Detail `D` | Re-inject fine luma texture: `Y += β · (Y_classical − Y_net)` with `β = 0.15 · (D/50)²` (0 at D = 0, 0.6 at 100). `Y_classical` is the classical demosaic of the same tile, computed only when β > 0. |

*Implementation notes (M1-12):* `C = 0` returns the input untouched (the literal formula gives `ε = 0`, and a guided filter with `ε = 0` still smooths chroma noise that is uncorrelated with the guide). The radius is `round(2 + 6·C/100)` px (2 to 8). The guided filter (He et al.) runs two box filters in series, so its receptive field is **twice** the radius, up to 16 px: the halo table below says so. `ε` is per pixel when σ is a map. Both operations leave the other half of the opponent space exactly as it was: the chroma filter touches only `R−Y` and `B−Y`, the detail term adds the same delta to all three channels. In the classical path the "network output" is the denoised image and `Y_classical` the plain demosaic before the denoise, so Detail puts back a fraction of what the denoise removed.

## R8 · Capture sharpening

Richardson–Lucy deconvolution on linear luminance with a Gaussian PSF, clean-room from RawTherapee's published description.

- `sharpen` resolves to `{ amount A 0–300, radius ρ 0.3–2.5 px }`. Iterations `n = round(A / 30)` (0–10), damping in flat areas proportional to σ to avoid amplifying noise, PSF σ = ρ at the centre and `1.25ρ` in the corners (linear in r²).
- Auto: ρ = 0.8 (0.7 at f/8 and above, 0.9 below f/2); A = 120 at base ISO falling linearly in log ISO to 60 at ISO 12800.
- Applied to luminance only; chroma is untouched.

*Implementation notes (M1-12):* the PSF is truncated at `ceil(3σ_max)` and varies slowly, so the forward blur and its transpose are both gathers with the kernel of the pixel being written, separable (horizontal then vertical), kernels interpolated from a 33-level bank between `ρ` and `1.25ρ`. Damping (Lucy-style): the correction ratio `d/e` becomes `1 + w·(d − e)/e` with `w = r²/(r² + (σ_n)²)`, `r = d − e`, `σ_n` the luminance noise σ; residuals well below the noise (flat areas) get `w ≈ 0`, structure gets plain RL. The result adds `Y' − Y` to all three channels. Iterations: `round(A/30)` clamped to 0 to 10; the Auto amount interpolates linearly in `log2(ISO/base)` from 120 to 60 at ISO 12800. Each iteration blurs twice, so the exact dependency radius is `2·n·ceil(3σ_max)` (24 px for the Auto values n = 4, ρ = 0.8): twice the "3σ·n" of the table below. The scheduler caps it at 24 px; beyond the cap (A near 300) tiles differ from the whole frame by about 2e-7 on photographic content.

## R9 · Lens geometry (0x7037 distortion, 0x7035 CA)

One resample, bicubic (Catmull–Rom), from the output pixel position to the source position:

- Distortion knots `d_i` → factor `F_i = d_i·2^−14 + 1`, knot `i` at the radius `r_i = (i + 0.5)/(n−1)` (half-diagonal = 1), linear interpolation between knots. Source radius `r_s = r · F(r)`.
- CA knots: the tag holds 2n values, red then blue. `K_c,i = ca_i·2^−21 + 1`, same knot radii and interpolation. Red and blue sample at `r_s · K_c(r)`; green at `r_s`.
- `decode.lens_corrections = false` skips distortion and vignetting; `decode.chromatic_aberration = false` skips CA only.
- After distortion correction, crop to the largest rectangle of valid pixels with the original aspect ratio (the "auto crop"), before the user's crop.

The factors above come from public reverse-engineering, not a Sony document.

*Implementation notes (M1-12):* the knot positions and the linear reading are those of darktable 4.6.1 and were measured against it, not assumed: the positions of 373 dark-line centres in its distortion-only render agree with `LensGeometry::source_position` (after undoing its zoom) to 0.002 px rms and 0.009 px at most, the three colour planes of its CA-only render to 0.003 px rms (gate: 0.2 px; `tests/cpu_lens.rs::{distortion,lateral_ca}_matches_darktable`). Knots at `i/(n−1)` would be 1.5 to 2 px off, and a monotone cubic through the right knots 0.15 px off at the corner for a quadratic profile, which is most of the gate. darktable does not crop: it zooms the corrected image by the largest value of `F(r)·K_c(r)` between the half-short-side radius and the corner (200 samples, three planes) so that no black corner is left; our crop and its zoom show the same mapping (`tools/lens/lens_model.py: darktable_zoom`). Coordinates are continuous (pixel `i` covers `[i, i+1)`, centre `i + 0.5`); the frame centre is `(W/2, H/2)` and `r` is the distance over half the source frame's diagonal. `K_c(r)` is evaluated at the *output* radius `r` and multiplies the same direction as `F(r)`: red and blue sample at `centre + (p − centre)·F(r)·K_c(r)`, green at `centre + (p − centre)·F(r)`. The CA tag needs an even count of at least 4. A source position is valid when it lies in the hull of the pixel centres, `0.5 ≤ s ≤ size − 0.5`; taps beyond the buffer edge replicate the edge pixel. The auto crop is the largest centred rectangle whose pixels sample valid positions in all three channels, with the frame's aspect ratio and even width and height (a 10% barrel correction on 600×400 crops to 552×368); the whole frame when nothing needs cropping. The resample works on any output rectangle of the cropped image; `LensGeometry::source_rect` gives the source rectangle it needs (bounding box of all channels' positions plus the Catmull–Rom support and one pixel). With `lens_corrections = false` the distortion tag is ignored, with `chromatic_aberration = false` the CA tag is.

### Verifying lens corrections (no test shoots)

- **darktable-cli as an oracle** (run as a separate process in CI, never linked or copied; it's GPL): `tools/lens/run_oracle_suite.sh` writes synthetic Sony files with known lens tables (`tools/lens/lensgen`: a flat field, vertical and horizontal lines; the raw data itself is undistorted), runs `darktable-cli` with only the lens module added (embedded metadata, one correction per render) and saves the module's own float input and output (`--dump-pipe lens`; a TIFF render mixes the planes through the camera matrix and smears lateral CA). The vignetting gain and the line positions are then compared with ours (radial functions within 0.5% gain and 0.2 px displacement at the corners): measured 8e-5 and 0.009 px (§R3, §R9). `record_oracle.py` stores the measured numbers in `crates/ym-gpu/tests/cpu_data/darktable_oracle.json`, so `cargo test -p ym-gpu --test cpu_lens` pins the conventions without darktable. darktable 4.6.1 (Ubuntu 24.04 `apt-get install darktable`) needs the module history row that `make_lens_params.py` builds, a camera in rawspeed's list (ILCE-7RM4 works) and a CA tag that starts with 2n. What remains for real files is the same run on them (the three Sony tags in the raw sub-IFD, `Exif.SubImage1.*CorrParams` in exiv2).
- **Adobe DNG Converter** (on a Mac, if available to the agent): converting an ARW writes Sony's corrections as DNG opcodes (`WarpRectilinear`, `FixVignetteRadial`); compare their radial polynomials with ours. A one-off check, recorded in `STATUS.md`.
- **Image checks** on public files: straight architectural lines stay straight after correction (line-fit residual drops), and sky or wall regions near the corners get flatter (corner-to-centre ratio closer to 1).
- **Unit tests** for the knot interpolation and resample maths.

## Tiling and halos

| Stage | Halo needed |
|---|---|
| R5 | 2 superpixels (4 px); 2 px when the chroma offsets are precomputed |
| R6 neural | half the receptive field (64 px for the starting U-Net) |
| R6 classical | demosaic 8 px (bilinear 1; Menon and LMMSE measured at 8) + bilateral 2 px = 10 px |
| R7 guided filter | 2 × radius, up to 16 px |
| R8 RL | 2·n·⌈3σ_max⌉, capped at 24 px |
| R9 | bounding box of the tile's source positions, ±2 px Catmull–Rom support, +1 px (`LensGeometry::source_rect`) |

The scheduler computes a tile's required input rectangle by walking the stages backwards (`ym_gpu::cpu::tiling::required_input_rect`); mosaic-domain rectangles start on even coordinates. The receptive fields of the demosaic methods are measured by perturbing single photosites (`tests/cpu_demosaic.rs`). Tiled-versus-whole tests (`tests/cpu_tiling.rs`, `cpu::chain`) give a maximum difference of 0 (pointwise stages bit-exact; the others also 0 in practice, 2e-7 for RL beyond its cap), against a limit of 1e-4.

**GPU tiling, as built (M0-07).** `ym_gpu::tiles::TilePlan::new(width, height, tile_px, stages)` cuts the frame into cores of 1024 or 2048 px (`tiles::TILE_SIZES`; `kernels::raw::recommended_tile_px` picks the largest that fits half the memory budget with two tiles in flight and one storage binding) and gives each tile its `compute` rectangle from `required_input_rect` over the stages (the table above, so the halos are the measured ones, mosaic origins even). The kernels run over the whole compute rectangle and mirror at its edge like the CPU reference; only the last kernel is restricted to the core, so the outer ring (one halo wide) is never read back: tiles are cropped, not blended, and the tiled result equals the whole-frame result bit for bit for the pointwise and bilinear stages (`tests/gpu_spike.rs`, `tests/gpu_pipeline.rs`: 0.0 difference, gate 1e-4). Tile rows complete in raster order and `kernels::raw::RowStreamer` hands the encoder a full-width band as soon as the last tile of a row is in. The device thread never pre-empts a running job (`ym_gpu::queue`), so an export that must not hold up viewport frames is cut into jobs at Export priority, one per tile row or per tile (`TilePlan::tile_rows`, `TilePlan::tile_range`: same frame and tile numbers, fewer tiles; one `RowCollector` or `RowStreamer` on the full plan collects the pieces); `tests/gpu_pipeline.rs` shows a Viewport job running between the rows of such an export and the pixels equal to the single-job run. `GpuContext::warm_up` compiles every kernel at start-up, so the colour chain's large shader does not land on the first viewport frame. A plan whose tiles lack the demosaic's 1 px of halo, or a raw frame whose pattern differs from the parameters, is refused with `GpuError::InvalidInput` instead of rendering seams or a wrong Bayer phase. Row strides are even, so two f16 samples share a 32-bit word and one thread writes both; image intermediates are f16 in memory (`SHADER_F16` native, else two halves per `u32` with `pack2x16float`, else f32) and f32 in registers.

*GPU stages implemented so far:* R1 normalise with R4 gains (one kernel, `shaders/spike_normalize.wgsl`) and the bilinear demosaic (`shaders/spike_demosaic.wgsl`); the other raw stages (R3, R5 to R9) still run on the CPU reference only. Measured on Mesa lavapipe (software Vulkan, no Metal here): the spike chain (normalise, gains, bilinear, fixed matrix, sRGB) on a 50 MP frame in 2048 px tiles differs from the CPU reference by at most 7.9e-4 with f16 storage (5.4e-4 on the small test frames; 2.4e-7 with f32 storage), and the peak GPU memory of the run is 0.281 GiB by the pool ledger (0.282 GiB by the driver's allocator report), against a gate of 1.5 GiB. Lavapipe timings mean nothing for Metal (HUMAN M0-H4).

*f16 storage and the 1e-3 gate (review note).* The gate is met by the 50 MP scene and by the smooth test scenes, but it is data dependent: the balanced mosaic and the demosaiced planes are rounded to f16 (relative step 4.9e-4, absolute 4.9e-4 to 9.8e-4 between 1 and 4), and the gains push saturated or clipped samples to 2 to 4. A frame of full-range, uncorrelated noise on every photosite (an unrealistic worst case, `white_noise_scene` in `tests/gpu_spike.rs`: values up to 1.4 × white, gains 2 and 1.6, a matrix with negative terms, so a dark output is a difference of large f16-rounded terms that the sRGB toe then multiplies by 12.9) reaches 4.8e-3 to 6.1e-3 against the plain f32 reference, five times the gate. Against a reference that rounds to f16 at the same two places (`f16_aware_reference`) the GPU agrees to 1.4e-6 in every storage mode and every scene, so the difference is storage precision, not kernel error; `spike_matches_an_f16_aware_reference_to_rounding_noise` pins it at 2e-5 for at least 99.9 % of the samples. A tight colour gate should be judged against the f16-aware reference or on the f32 device.

## Performance targets (Apple M1, 50 MP)

| Path | Target |
|---|---|
| Fit preview after open (classical, 1/4 res) | < 300 ms |
| Colour-only edit, viewport refresh | < 16 ms |
| 1:1 neural tiles visible on a 1440p viewport | < 1.5 s |
| Full neural export, JPEG | < 6 s per photo |
| CPU-only fallback export | < 90 s per photo |

These are targets; report measured values at each gate.

**CPU reference, measured (M1-12, 4-core x86-64, `cargo bench -p ym-gpu --bench cpu_stages`).** The machine was shared with other jobs (load average 13 to 17 on 4 cores), so wall times are inflated several-fold; CPU seconds are the dependable number. Whole classical chain R1 to R9 (LMMSE, bilateral, guided filter, RL with the Auto values, distortion + CA resample), 50 MP: 69 CPU-seconds, i.e. about 17 s on four idle cores if it scales (measured wall 103 s under that load). Per stage at 50 MP, CPU seconds: normalise 0.35, vignetting 1.3, gains 0.05, highlights 2.2, demosaic (LMMSE) 9.2, bilateral 19.4, colour guided filter 9.7, detail 0.9, RL (n = 3) 12.1, lens resample 9.9. The CPU-only fallback target (< 90 s per photo) therefore looks met with a wide margin on four cores; confirm on an idle machine and on the M1.

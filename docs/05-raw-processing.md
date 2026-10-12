# 05 · Raw processing (mosaic to scene-linear camera RGB)

Crates: `ym-core` (orchestration), `ym-gpu` (kernels), `ym-ml` (network), `ym-noise` (σ map). All stages run in scene-linear float, in tiles on the GPU, with a CPU reference for tests and for machines without a usable GPU.

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

`x = (raw − black[c]) / (white[c] − black[c])` per CFA position. Negative values stay (they carry noise statistics); clamp at −0.05. Clip mask: `raw ≥ white[c] − 4·qstep` (or `≥ 0.995` normalised where qstep is absent). The mask travels with the image to R5, R6 and C7.

## R3 · Vignetting (0x7032)

Knots `v_i`, i = 0…n−1, at evenly spaced normalised radii `r_i = i/(n−1)`, where r = 1 is the half-diagonal of the active area. Gain per knot: `G_i = (2^(0.5 − 2^(v_i·2^−13 − 1)))²`. Interpolate `G(r)` with a monotone cubic and multiply each photosite by `G(r)`. Verify the radius normalisation and the direction (gain vs attenuation) against independent implementations (below, §Verifying lens corrections); adjust the spec if it differs.

## R4 · Initial gains

`g0 = 1 / w_scene_initial` (green = 1), from the illuminant estimate at open. If the estimate fails, use the as-shot gains from SR2 0x7313. Gains are applied to the mosaic before R5 and R6 because both work better on balanced data. Later white-balance changes do not re-run R5–R9 (`docs/02` C1 applies a residual).

## R5 · Highlight reconstruction (opposed channels)

Clean-room from darktable's published description ("inpaint opposed"); don't read or port its code.

1. Build a half-resolution superpixel image from each 2×2 quad: R, G = mean of the two greens, B, with a per-channel clipped flag.
2. For each clipped channel `c` of a superpixel, the **opposed estimate** is the cube-root mean of the unclipped other channels over the 3×3 superpixel neighbourhood: `ŷ = (mean(x^(1/3)))³`.
3. **Global chroma correction:** over all unclipped superpixels that border clipped ones, average `x_c − ŷ_c`; add that offset to every estimate for channel `c`.
4. Reconstructed value: `max(original, ŷ_c + offset_c)`. Write it back into the mosaic photosites of that channel, keeping the clip mask so later stages know these values are estimates.
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

`decode.neural_demosaic = false` forces the classical path everywhere, including export.

## R7 · Detail controls

`noise_reduction` resolves to `{ luminance L, color C, detail D }`, each 0–100. Auto values: `L_auto = clamp(10 · (1 + log2(ISO/100)), 10, 90)` (ISO 100 → 10, ISO 400 → 30, ISO 3200 → 60, ISO 25600 → 90; the number grows with ISO so the slider reads naturally), `C_auto = 50`, `D_auto = 50`. The Automatic card shows the resolved values ("Luminance 30").

| Control | Operation |
|---|---|
| Luminance `L` | σ fed to the network = `σ_est · (L / L_auto)^1.5`: the estimated noise at the auto value, 0 at L = 0 (the network demosaics without denoising), up to (100/L_auto)^1.5 × at 100. The network is trained across σ including 0, so this is a true strength control. |
| Colour `C` | After the network: guided filter on the two chroma channels of an opponent space (Y, R−Y, B−Y), guided by Y. Radius `2 + 6·C/100` px at full resolution, ε from `σ_est² · (C/50)²`. |
| Detail `D` | Re-inject fine luma texture: `Y += β · (Y_classical − Y_net)` with `β = 0.15 · (D/50)²` (0 at D = 0, 0.6 at 100). `Y_classical` is the classical demosaic of the same tile, computed only when β > 0. |

## R8 · Capture sharpening

Richardson–Lucy deconvolution on linear luminance with a Gaussian PSF, clean-room from RawTherapee's published description.

- `sharpen` resolves to `{ amount A 0–300, radius ρ 0.3–2.5 px }`. Iterations `n = round(A / 30)` (0–10), damping in flat areas proportional to σ to avoid amplifying noise, PSF σ = ρ at the centre and `1.25ρ` in the corners (linear in r²).
- Auto: ρ = 0.8 (0.7 at f/8 and above, 0.9 below f/2); A = 120 at base ISO falling linearly in log ISO to 60 at ISO 12800.
- Applied to luminance only; chroma is untouched.

## R9 · Lens geometry (0x7037 distortion, 0x7035 CA)

One resample, bicubic (Catmull–Rom), from the output pixel position to the source position:

- Distortion knots `d_i` → factor `F_i = d_i·2^−14 + 1` at radii `r_i = i/(n−1)` (half-diagonal = 1). Source radius `r_s = r · F(r)` with `F` a monotone cubic through the knots.
- CA knots: the tag holds 2n values, red then blue. `K_c,i = ca_i·2^−21 + 1`. Red and blue sample at `r_s · K_c(r)`; green at `r_s`.
- `decode.lens_corrections = false` skips distortion and vignetting; `decode.chromatic_aberration = false` skips CA only.
- After distortion correction, crop to the largest rectangle of valid pixels with the original aspect ratio (the "auto crop"), before the user's crop.

The factors above come from public reverse-engineering, not a Sony document.

### Verifying lens corrections (no test shoots)

- **darktable-cli as an oracle** (run as a separate process in CI, never linked or copied; it's GPL): render corpus files with only the lens module enabled in "embedded metadata" mode, and with it disabled; derive its vignetting gain and distortion mapping per file and compare with ours (radial functions within 0.5% gain and 0.2 px displacement at the corners).
- **Adobe DNG Converter** (on a Mac, if available to the agent): converting an ARW writes Sony's corrections as DNG opcodes (`WarpRectilinear`, `FixVignetteRadial`); compare their radial polynomials with ours. A one-off check, recorded in `STATUS.md`.
- **Image checks** on public files: straight architectural lines stay straight after correction (line-fit residual drops), and sky or wall regions near the corners get flatter (corner-to-centre ratio closer to 1).
- **Unit tests** for the knot interpolation and resample maths.

## Tiling and halos

| Stage | Halo needed |
|---|---|
| R5 | 2 superpixels (4 px) |
| R6 neural | half the receptive field (64 px for the starting U-Net) |
| R6 classical | 8 px |
| R7 guided filter | 8 px |
| R8 RL | 3σ·n, capped at 24 px |
| R9 | max displacement in the tile + 4 px |

The scheduler computes a tile's required input rectangle by walking the stages backwards.

## Performance targets (Apple M1, 50 MP)

| Path | Target |
|---|---|
| Fit preview after open (classical, 1/4 res) | < 300 ms |
| Colour-only edit, viewport refresh | < 16 ms |
| 1:1 neural tiles visible on a 1440p viewport | < 1.5 s |
| Full neural export, JPEG | < 6 s per photo |
| CPU-only fallback export | < 90 s per photo |

These are targets; report measured values at each gate.

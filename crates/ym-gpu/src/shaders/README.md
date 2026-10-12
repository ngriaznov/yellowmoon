WGSL compute shaders live in this folder. Each kernel has a CPU reference (`src/cpu/` for the raw stages, `ym_color::display::ColorChain` for the colour chain) and a parity test with a stated maximum absolute error.

| File | Kernel | CPU reference |
|---|---|---|
| `common.wgsl` | helpers shared by every kernel: LogC3/LogC4, sRGB and BT.1886 transfer, mirror index, NaN test by bits | `ym_color::{logc, encode}` |
| `spike_normalize.wgsl` | R1 normalise + R4 gains | `cpu::normalize`, `cpu::gains` |
| `spike_demosaic.wgsl` | R6 bilinear demosaic | `cpu::demosaic::bilinear` |
| `spike_matrix.wgsl` | fixed 3x3 and sRGB (the M0-07 spike's last pass) | `kernels::spike::cpu_reference` |
| `color.wgsl` | C1 to C13 in one pass: fused matrix, C7, C8, C9, C10 (CDL, tetrahedral LUT, LogC3 path, blend), C11 curves, C12 dispatch, C13 encode and dither | `ym_color::display::ColorChain` |
| `aces2.wgsl` | the ACES 2 output transform, with tables uploaded from `ym_color::aces2` | `ym_color::aces2` |

The files are templates: `ST_ELEM` and the `//@ACCESSORS name [ro]` lines are expanded by `kernels/shader.rs` for the device's storage mode (native f16, packed f16, f32), and the constants (LogC, gamut matrices, grade constants, table offsets, the geometry of the ACES 2 hue tables) are written in front of the module from `ym_color`, so those numbers are not typed twice. The ACES 2 algorithm's own constants (the 27.13 of the cone response, the smoothing and blend factors, the chroma-compression polynomial) are private to `ym_color::aces2` and stay in `aces2.wgsl` beside the code that uses them; the OpenColorIO vectors and the CPU parity tests pin them. `kernels::shader_source(mode, id)` returns the final text; `tests/shaders_naga.rs` validates it offline.

Conventions: one thread handles two horizontally adjacent pixels (a packed f16 word is written by one thread), `@workgroup_size(16, 16)`, row strides are even, planes are planar. Look cubes, shapers and curve tables are f32 storage buffers read with explicit tetrahedral and linear interpolation, never hardware filtering (`docs/02`, C10 and C11).

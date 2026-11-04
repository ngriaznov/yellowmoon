# Spectral upsampling table

```
Computed by Yellowmoon from CIE 1931 2° colour-matching functions (CIE, CC BY-SA 4.0).
Modified by: Yellowmoon (computed, not copied).
Licensed CC BY-SA 4.0: https://creativecommons.org/licenses/by-sa/4.0/ (the text is in LICENSE, in this folder).
```

This table is **Adapted Material** in the sense of the licence: every number in it is computed from the CIE 1931 2 degree
colour-matching functions (`data/film/cie/CIE_xyz_1931_2deg.csv`, CIE 2019, DOI: 10.25039/CIE.DS.xvudnb9b, CC BY-SA 4.0,
published by the International Commission on Illumination (CIE), Vienna). The table is a new work in a different form (three
coefficients and a luminance per chromaticity cell, not a copy of the observer), so it is shared under the same licence, as the
ShareAlike condition asks. It contains nothing from any film profile and nothing from anyone else's table; the method is the
sigmoid-of-a-quadratic spectrum of Jakob and Hanika (Eurographics 2019), written from the paper.

| File | What it is |
|---|---|
| `coeffs.f32` | little-endian float32 `[256][256][4]` = `(c0, c1, c2, Y_s)`, `y` the outer index |
| `flags.u8` | `[256][256]` bytes: 0 solved, 1 inpainted |
| `error.f32` | `[256][256]` float32: distance in xy between the cell and the chromaticity of its stored spectrum |
| `table.json` | the format, the generator's version and the SHA-256 of the inputs and outputs |

The format and the lookup are in `docs/24-film.md` section 3.3. Written by `tools/film/upsampling/make_table.py`; do not edit by
hand. `SHA256SUMS` lists every file of this folder except itself and this file.

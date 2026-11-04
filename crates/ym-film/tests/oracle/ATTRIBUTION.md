# Oracle fixtures

```
Numbers computed by running spektrafilm (Andrea Volpato, https://github.com/andreavolpato/spektrafilm), commit
3bb2c2d2801ff68b92019cf1dbcbb133d60832bc, as a separate process, on the film and paper profiles of Andrea Volpato.
Modified by: Yellowmoon (computed, not copied). The numbers are Adapted Material of the profiles.
Licensed CC BY-SA 4.0: https://creativecommons.org/licenses/by-sa/4.0/ (the text is in LICENSE, in this folder).
```

These files are the expected values of `crates/ym-film/tests/oracle.rs` (`docs/24` section 8, *Oracle*): the taps of spektrafilm's
per-pixel mode (couplers, halation, grain, glare and blurs off) on 24 chart patches, a 125-point RGB lattice and a 21-step grey ramp,
for seven stocks. They are numbers only. No code of spektrafilm is in this repository; it ran in a scratch environment.
The generator is `tools/oracle/spektrafilm/`. `SHA256SUMS` lists every file of this folder except itself and this file.

| File | What it is |
|---|---|
| `inputs.json` | the inputs in linear ARRI Wide Gamut 4: 24 ColorChecker patches (Ohta reflectances under D50, adapted to D65), a 125-point RGB lattice (levels 0.04, 0.09, 0.18, 0.36, 0.72) and a 21-step grey ramp (1/3-stop steps, index 12 is 18 %). Written by our own `write_inputs` test; not derived from spektrafilm |
| `<film id>.json` | for one film, the taps `log_e_film`, `cmy_film`, `log_e_print`, `cmy_print`, `rgb_out` and `log_e_film_nowindow` (the adaptation window off) at every input, rounded to 1e-6; `paper_white_rgb`, the colour of a print with no density; the SHA-256 of `inputs.json` |

Run with Python 3.13, numpy 2.5.4, scipy, colour-science 0.4.7, numba 0.68.1. spektrafilm's per-pixel mode: grain, couplers, halation,
glare, blurs, auto exposure and the input and output gamut compression off; the adaptation window on; output linear sRGB.

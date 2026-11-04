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

State at the time of writing: the generator has not yet been run, so this folder holds no fixtures.

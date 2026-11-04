# spektrafilm oracle (F-06)

`run_oracle.py` runs spektrafilm (Andrea Volpato, GPL-3.0, https://github.com/andreavolpato/spektrafilm) as a **separate process and a
black box** on the inputs of `crates/ym-film/tests/oracle/inputs.json`, and writes the numbers the Rust test
`crates/ym-film/tests/oracle.rs` compares with (`docs/24` section 8, *Oracle*). It is a test oracle only, like rawler and LibRaw in
`tools/oracle/`: never a dependency of a shipped crate, and none of its code, defaults or formulas is in this repository. The driver
uses its public interface only (`init_params`, `simulate`, the parameter names and the tap names).

The fixtures it writes are numbers computed from the CC BY-SA 4.0 profiles, so they are Adapted Material and live in
`crates/ym-film/tests/oracle/` with `LICENSE`, `ATTRIBUTION.md` and `SHA256SUMS` (`tools/licences/check_bundle.py` checks the folder).
CI does not run this driver; the fixtures are committed.

## Set up (once, in a scratch directory outside the repository)

```
S=/path/to/scratch
git clone https://github.com/andreavolpato/spektrafilm.git $S/spektrafilm
git -C $S/spektrafilm checkout 3bb2c2d2801ff68b92019cf1dbcbb133d60832bc
python3.13 -m venv $S/venv
$S/venv/bin/pip install --no-deps -e $S/spektrafilm
$S/venv/bin/pip install "numpy~=2.4" "scipy~=1.17" "colour-science~=0.4.6" "scikit-image~=0.26" "matplotlib~=3.10" \
    "opt-einsum~=3.4.0" "numba~=0.64" "pyfftw~=0.15.0" exiv2 rawpy lensfunpy OpenImageIO
```

The package declares GUI dependencies (Qt, napari) that the runtime does not need; `--no-deps` plus the list above leaves them out
(about 700 MB). It needs Python 3.13.

## Run

```
$S/venv/bin/python tools/oracle/spektrafilm/run_oracle.py \
    --inputs crates/ym-film/tests/oracle/inputs.json --out crates/ym-film/tests/oracle
(cd crates/ym-film/tests/oracle && sha256sum $(ls | grep -v -e SHA256SUMS -e ATTRIBUTION.md) > SHA256SUMS)
cargo test -p ym-film --release --test oracle -- --nocapture
```

`--only kodak_portra_400,...` runs some stocks. The inputs are written by `YM_ORACLE_WRITE_INPUTS=1 cargo test -p ym-film --release
--test oracle -- --ignored write_inputs` (our own code; the ColorChecker patches need `testdata/spectral/`).

## What is switched on and off

Per-pixel mode: grain, DIR couplers, halation, glare, lens and scanner blurs, auto exposure, the input gamut compression and the output
gamut and lightness compression are off; the adaptation window is on (a second run with it off gives `log_e_film_nowindow`). Input
colour space ARRI Wide Gamut 4, linear; output sRGB primaries, linear, no encoding. A negative is printed on its matching paper (Vision3
500T on Portra Endura, as the engine does) and the print is scanned; a slide is scanned directly (`io.scan_film`).
`paper_white_rgb` is the scan of a print with no density, got by injecting zero density at the `cmy_print` tap.

## How the Rust test compares (the oracle's enlarger, lamp and paper exposure differ from ours)

* Stage 2 and stage 5 are compared as they are and relative to the 18 % grey. Stage 5 uses our per-layer curves of the profile
  (stage 4), because that is what the oracle's "couplers off" is; our own coupler amount 0 means the curves with the inhibition taken
  out, which the oracle has no counterpart for.
* The print side is **normalised to mid grey**: the yellow and magenta filters and the exposure shift of our enlarger are solved so that
  the grey's three paper dye densities equal the oracle's `cmy_print` of the grey; stages 7 and 8 and the final colour are compared after
  that. Stage 7 (`log_e_print`) is compared relative to the grey.
* The oracle's `rgb_out` is divided by `paper_white_rgb` (our scan sets paper white to 1). For a slide the one knob is the exposure, until
  the grey's luminance equals the oracle's.
* Points the soft chromaticity compression of our stage 1 changes are skipped (the oracle's input compression is off).

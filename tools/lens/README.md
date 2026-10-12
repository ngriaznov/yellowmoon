# Lens-correction oracle (M1-12, `docs/05` section Verifying lens corrections)

Compares our vignetting and distortion/CA maths (`crates/ym-gpu/src/cpu/{vignette,lens,resample}.rs`)
with `darktable-cli`. darktable is GPL: it only ever runs as a separate process here and nothing
of it is linked, copied or shipped. The numbers it produces are measured, the maths in this
repository is ours.

| File | Purpose |
|---|---|
| `run_oracle_suite.sh` | The whole thing in one command: writes the synthetic files, runs darktable on them, runs the three gates, with `--record` refreshes the snapshot. |
| `lensgen/` | A standalone cargo project (its own workspace root, uses `ym_raw::testgen`) that writes the synthetic ARWs: `flat.ARW`, `vlines.ARW`, `hlines.ARW` and `tags.json`. Optional second argument: a multiplier of the CA knots. |
| `darktable_oracle.sh` | Runs ARW files through `darktable-cli` with the lens module added to the history (embedded metadata) and saves the module's float input and output (`lens_in.pfm.gz`, `lens_out.pfm.gz`). |
| `make_lens_params.py` | Builds the module's parameter blob (module version 10, 356 bytes) that `darktable_oracle.sh` adds to the history; `--flags` picks vignetting / distortion / CA, `--algorithm` the embedded-metadata algorithm. |
| `lens_model.py` | Independent Python implementation of the R3 / R9 maths (NumPy). `--write-vectors` regenerates `crates/ym-gpu/tests/cpu_data/lens_vectors.json`. Also holds darktable's zoom rule. |
| `record_oracle.py` | Measures darktable's output (vignetting gain, line positions per colour plane) and writes `crates/ym-gpu/tests/cpu_data/darktable_oracle.json`, the snapshot the Rust tests compare with. Exits 1 if the model does not reproduce the measurement. |
| `compare_with_darktable.py` | The gates: `vignette` (gain within 0.5%), `distortion` (line positions within 0.2 px, `--horizontal` for horizontal lines), `ca` (red, green, blue within 0.2 px), `selftest` (checks the script on synthetic data, needs no darktable). |
| `lens_io.py` | Reads the gzipped PFM files. |

## Run

```
sudo apt-get install -y darktable          # 4.6.1 on Ubuntu 24.04; needs >= 4.4 (embedded Sony metadata, --dump-pipe)
pip3 install numpy scipy
python3 -I tools/lens/compare_with_darktable.py selftest          # works without darktable or files
tools/lens/run_oracle_suite.sh build/lens-oracle                  # a few minutes; add --record to refresh the snapshot
```

The suite needs about 60 MB in its directory and about 220 MB of temporary files per render while it
runs. Exit status 0 means all gates pass.

## What was measured (darktable-cli 4.6.1, Ubuntu 24.04, headless, as root, no GPU)

* darktable does not switch the lens module on by itself for Sony files, and `darktable-cli` has no
  switch for it, but it accepts a history row. `make_lens_params.py` writes the module's parameter
  struct (read off darktable's public module source, then proven by use), so no GUI session is needed.
* darktable only uses the tags when the distortion tag starts with `nc <= 16`, the CA tag with `2 * nc`
  and the vignetting tag with `nc` (its own consistency check). A file whose CA tag starts with `nc`
  is silently left uncorrected: the earlier version of `lensgen` wrote it that way, which is why the first
  oracle run saw "no lens module".
* A TIFF render cannot be used for the CA measurement: the camera matrix mixes the planes (an sRGB
  green line is part red and part blue) and the lines come out 0.15 to 0.4 px off. The float image going
  into and out of the lens module, dumped with `--dump-pipe lens`, is clean: **vignetting gain 8e-5,
  distortion 0.002 px rms (0.009 px max), CA 0.003 px rms (0.009 px max)** against our model.
* The model that fits (see `lens_model.py` and `docs/05` sections R3 and R9): knot `i` of `n` at the
  normalised radius `(i + 0.5)/(n - 1)`; linear interpolation, end values held; vignetting
  `G = 2^(0.5 - 2^(v * 2^-13 - 1))` and the photosite is *divided* by `G` (algorithm 2; algorithm 1 uses
  `G^2`); distortion `F = d * 2^-14 + 1`, source radius `r * F(r)`; CA `K = ca * 2^-21 + 1` for red and
  blue, `K` and `F` taken at the output radius; radius over half the frame diagonal.
* darktable zooms the corrected image by the largest `F(r) * K_c(r)` between the half-short-side radius
  and the corner so that no black corner is left (`lens_model.darktable_zoom`); our R9 crops instead.
  The comparison undoes the zoom.
* The oracle's own TIFF transfer curve was checked against the sRGB curve with flat fields of 10 to 90%
  exposure (within 0.1%) before it was dropped in favour of the float dump.

## Input files

`lensgen` writes ILCE-7RM4 files of 3000 x 2000 (LibRaw decodes them; darktable identifies camera and lens and
renders 2968 x 2000, because rawspeed's camera entry crops 32 columns, so the comparisons take the frame
size from the images):

* `flat.ARW`: a flat field at 40% of the range with tag 0x7032 (`v = 8192 t^2`: a corner 0.5 EV darker).
* `vlines.ARW`, `hlines.ARW`: dark Gaussian lines every 100 px (vertical, horizontal), undistorted in the
  raw data, with tags 0x7037 (barrel, 7% at the corner) and 0x7035 (lateral CA, red out / blue in).
  The suite also writes `vlines.ARW` with eight times the CA for the CA gate (a large CA against the
  sub-pixel noise of the line tracker).

For real files the requirement is the same: a camera model that exists in rawspeed's `cameras.xml`
and the three lens tags in the raw sub-IFD (exiv2: `Exif.SubImage1.VignettingCorrParams`,
`.ChromaticAberrationCorrParams`, `.DistortionCorrParams`). Pass them to `darktable_oracle.sh` and then to
`compare_with_darktable.py` with a `tags.json` holding the raw knots without the count:
`{"vignetting": [...], "distortion": [...], "ca": [...]}`.

## CI job sketch (`lens-oracle`, M1-12)

```yaml
lens-oracle:
  runs-on: ubuntu-24.04
  steps:
    - uses: actions/checkout@v4
    - run: sudo apt-get install -y darktable && pip3 install numpy scipy
    - run: tools/lens/run_oracle_suite.sh build/lens-oracle
```

`cargo test -p ym-gpu --test cpu_lens` needs none of this: it checks the Rust code against the recorded
snapshot `crates/ym-gpu/tests/cpu_data/darktable_oracle.json`.

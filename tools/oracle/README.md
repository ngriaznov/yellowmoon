# Oracles

Test-only helpers that compare Yellowmoon with independent tools. Nothing here is linked into a
shipped crate or bundled in the app (`docs/14-licensing.md`): LibRaw (LGPL/CDDL), rawler (LGPL)
and ExifTool (Perl) run as separate programs, or, for rawler, from the excluded `oracle` crate.

| Tool | What it does |
|---|---|
| `rawpy_meta.py FILE [--pixels] [--thumb-tail N]` | Prints LibRaw's view of a file as one JSON object: make, model, ISO, shutter, aperture, raw size, CFA pattern, black and white levels, white balance, an SHA-256 of `raw_image`, and the tail of the extracted thumbnail. Needs `rawpy` (`pip install rawpy`). |
| `compare_metadata.py FILE_OR_DIR...` | Runs `yellowmoon info --json`, `rawpy_meta.py` and (optionally) ExifTool on each file and reports every field that differs. This is the M0-04 corpus gate. Exit 0 = no differences. |
| `rawpy_compare.py FILE --out RAW.bin` | Writes LibRaw's `raw_image` (undithered, the whole sensor, little-endian `u16`) and prints its levels as JSON. For Lossless M and S it writes LibRaw's pseudo-raw RGB (see below). |
| `libjpeg_dump.py FILE --out RAW.bin` | Decodes the Lossless L tiles with libjpeg (`pip install pylibjpeg-libjpeg numpy`), a third implementation that has nothing in common with LibRaw or rawler. Covers every predictor and restart interval. |
| `src/main.rs` (crate `oracle`) | `oracle --corpus DIR --report FILE` compares ym-raw's decode of every `*.ARW` with LibRaw, libjpeg (Lossless L) and rawler (in process) and writes a JSON report plus a table; `oracle dump FILE --out RAW.bin` writes our decode and a JSON sidecar (never committed). |
| `ci.sh` | What the CI `oracle` job runs (see below). |

## The decoder oracle (M0-05)

```
cargo run -p ym-raw --features testgen --release --example gen_corpus -- testdata/synthetic
cargo run -p oracle --release -- --corpus testdata/synthetic --report oracle-synthetic.json
cargo run -p oracle --release -- --corpus testdata/real/files --report oracle-real.json   # real files, if any
```

Exit code 0: every comparison passed; 1: a pixel, level or white balance mismatch or a file we could not read; 2: LibRaw could not run (rawpy is missing) or a usage error. Options: `--no-libraw`, `--no-libjpeg`, `--no-rawler`, `--keep DIR` (keep LibRaw's dumps), `--python`, `--script`.

What counts as a pass, per variant:

| Variant | LibRaw 0.22.1 | rawler 0.8 |
|---|---|---|
| uncompressed 14-bit, packed 12-bit, ARW 1, Lossless L | identical samples, identical black, white and white balance | identical samples (ARW 1: its 8 extra rows are dropped; its white level is not compared) |
| cRAW | identical samples and levels (cRAW is exact in LibRaw) | within rawler's dither window: `base + ((delta * r + 1024) >> 12)`, `r` in 0..2048, around the undithered value; `delta` is the curve step per code |
| Lossless M and S | identical to its pseudo-raw RGB (`sony_ycbcr_load_raw`) when decoded with replicated chroma and the white balance left in; black 1024 and maximum 17536 equal | information only (its matrix and chroma interpolation differ) |
| Lossless L | also libjpeg: identical samples | |

Differences that are the oracle's, not ours, are reported as `oracle_error` or `informational`, never as mismatches: LibRaw does not read 4:4:4 M/S, restart intervals of more than one row with a two-dimensional predictor, or a Lossless L file of a model that is not `ILCE-1`, `ILCE-7*` or `ILCE-9*`; it takes a file with exactly four tile offsets for a Sinar 4-shot file; it keeps 14-bit-scale levels for 12-bit words. rawler assumes 512 x 512 tiles that cover the image and predictor 1 without restarts, and panics otherwise. LibRaw 0.22.1 reaches its YCbCr loader only for Compression 6, so `rawpy_compare.py` decodes a temporary copy of an M/S file with that one value changed (`"input_patched": true`).

### Conventions that differ between the tools

- **cRAW and the 14-bit scale.** LibRaw 0.22.1 (`sony_arw2_load_raw`) and rawler return the tone-curve output unshifted: `curve[code << 1]`, on the 14-bit scale of the other ARW modes (black 512; the white level is the SR2 value, 15360 on the real files; values up to 17204 are legal and not clamped). The 12-bit scale of dcraw and old LibRaw releases (`>> 2`, black 128, white 4095) is not used by either. ym-raw returns the same numbers, so for cRAW the samples, black and white are compared for equality with LibRaw (`raw_image`; LibRaw does not dither). rawler adds a dither of up to about a quarter of the local curve step, so against rawler cRAW is compared inside that window (`base + ((delta * r + 1024) >> 12)`, `r` in 0..2048). Six real cRAW files confirm both statements.
- **Lossless M and S against LibRaw.** The file holds white-balanced, demosaiced, chroma-subsampled YCbCr. LibRaw's `raw_image` for it is already RGB (`sony_ycbcr_load_raw`: replicated chroma, BT.601 `ycc2rgb`, black 1024, maximum 17536, white balance still in the data). ym-raw's shipped default differs on purpose (linear chroma, white balance divided out, camera RGB). The oracle therefore decodes ours with `ChromaUpsampling::Replicate` and `undo_white_balance: false` for the comparison, which must be bit-identical; `oracle dump --replicate-chroma --keep-white-balance` writes that variant, and a plain `oracle dump` writes the shipped default. Whether LibRaw's scale and ceiling match a real camera is not verified: no real M or S file was available (HUMAN, `docs/04`).

The CI job (`oracle`, in the nightly and pull-request workflows of `docs/13-testing.md`) runs
`tools/oracle/ci.sh`; it needs Rust and `python3` with `rawpy`, `numpy` and `pylibjpeg-libjpeg` (the script refuses to run without any of them; the `oracle` binary alone skips the libjpeg column when the module is missing). The job is still commented out in `.github/workflows/ci.yml`, which the decoder task does not own; this is the job to put there (it also serves the nightly run, the schedule trigger only adds the real corpus once `tools/fetch/samples.sh` can fill it):

```yaml
  oracle:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install rawpy numpy pylibjpeg-libjpeg
      - uses: actions/cache@v4
        with: { path: testdata, key: "testdata-${{ hashFiles('tools/fetch/manifest.json') }}" }
      - run: ORACLE_OUT="$RUNNER_TEMP/oracle" tools/oracle/ci.sh
      - uses: actions/upload-artifact@v4
        if: always()
        with: { name: oracle-reports, path: "${{ runner.temp }}/oracle" }
```


```
pip install rawpy numpy pylibjpeg-libjpeg
tools/oracle/ci.sh                      # synthetic corpus, the two Rust oracle tests, then testdata/arw if it holds *.ARW files
ORACLE_OUT=out tools/oracle/ci.sh --synthetic    # synthetic corpus only; reports go to out/ (default: the current folder)
```

The script sets `YM_REQUIRE_ORACLE=1`, regenerates `testdata/synthetic`, writes `oracle-synthetic.json` and (with a real corpus) `oracle-report.json`, and exits with the code of `oracle`.
The synthetic corpus is 48 files: ARW 1, packed 12-bit, 12-bit and 14-bit uncompressed, cRAW, Lossless L (every predictor, restart intervals, three tile sizes, partial tiles, skewed Huffman tables) and Lossless M/S (4:2:0, 4:2:2, 4:4:4), on several bodies.

Rust integration tests that use the same oracles on synthetic files:

- `crates/ym-raw/tests/oracle_decode.rs` (pixels of every decoder variant against LibRaw; skips without `python3` + `rawpy`; its ignored test `real_files_match_libraw` runs on `testdata/real/files` and `testdata/arw`: `cargo test -p ym-raw --test oracle_decode --release -- --ignored`)
- `tools/oracle/tests/report.rs` (the `oracle` binary: reports, `dump`, the rawler and LibRaw comparisons)
- `crates/ym-raw/tests/oracle_libraw.rs` (skips without `python3` + `rawpy`)
- `crates/ym-raw/tests/oracle_exiftool.rs` (skips without ExifTool; set `YM_EXIFTOOL` to the path of
  an `exiftool` script from a clone of github.com/exiftool/exiftool to run it with `perl`)

`YM_REQUIRE_ORACLE=1` turns a skip into a failure (use it in the oracle CI job).

```
# synthetic files, then compare
cargo run -p ym-raw --features testgen --example gen_arw -- ILCE-7RM4 128 96 /tmp/a.ARW iso=800
cargo build -p ym-cli
python3 tools/oracle/compare_metadata.py --report /tmp/report.json /tmp/a.ARW
```

## Real files without the corpus

While `testdata/arw` is empty (the corpus hosts are unreachable from the build machine), nine real ARW
files are available from public GitHub repositories (`git clone` works; the proxy blocks the corpus hosts and
GitHub's code search). Their licences are not checked: use them for local evaluation only, never commit them
and never copy them elsewhere (`testdata/` is git-ignored).

| File | Repository and path | What it is |
|---|---|---|
| DSLR-A500, ILCE-5000, ILCE-7M3, ILCE-7M4, NEX-5N, NEX-7 | `drewnoakes/metadata-extractor-images`, `arw/` | five cRAW files and one Lossless L file (A7 IV) |
| ILCE-6300 (cRAW), ILCE-7M2 (14-bit uncompressed) | `immich-app/test-assets`, `formats/raw/Sony/` | |
| DSLR-A300 (ARW 1, Huffman) | `AcademySoftwareFoundation/OpenImageIO-images`, `raw/RAW_SONY_A300.ARW` | |

```
mkdir -p testdata/real/files /tmp/dl && cd /tmp/dl
git clone --depth 1 --filter=blob:none --no-checkout https://github.com/drewnoakes/metadata-extractor-images.git mei
git -C mei sparse-checkout init --no-cone && git -C mei sparse-checkout set 'arw/*' && git -C mei checkout HEAD
git clone --depth 1 --filter=blob:none --no-checkout https://github.com/immich-app/test-assets.git immich
git -C immich sparse-checkout init --no-cone && git -C immich sparse-checkout set 'formats/raw/Sony/*' && git -C immich checkout HEAD
git clone --depth 1 --filter=blob:none --no-checkout https://github.com/AcademySoftwareFoundation/OpenImageIO-images.git oiio
git -C oiio sparse-checkout init --no-cone && git -C oiio sparse-checkout set 'raw/RAW_SONY_A300.ARW' && git -C oiio checkout HEAD
# copy the nine files into testdata/real/files/, then
cargo test -p ym-raw --test oracle_decode --release -- --ignored
cargo run -p oracle --release -- --corpus testdata/real/files --report /tmp/oracle-real.json
python3 tools/oracle/compare_metadata.py testdata/real/files    # metadata against LibRaw and ExifTool
```

Result (M0-05): all nine decode; eight are bit-identical to LibRaw 0.22.1 (six cRAW, one Lossless L, one
14-bit uncompressed), the ninth (ARW 1) too after its extra rows are dropped; the Lossless L file is also
bit-identical to libjpeg and rawler; the six cRAW files are inside rawler's dither window. They caught
three defects in the container reader that the synthetic files could not (docs/04): the maker note has
no `SONY DSC` header, cRAW says BitsPerSample 12 and the first-thumbnail choice. ISO differs by design: ours is
the camera's measured `SonyISO`, LibRaw's is the nominal EXIF value (the ILCE-9 dump in lclevy/sony_raw: 2619 against 2500), so the
script allows a third of a stop.

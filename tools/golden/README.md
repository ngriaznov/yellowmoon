# tools/golden

Golden-image tooling (`docs/13-testing.md`, section Golden images). Test tooling: never shipped, depends on `ym-raw` with its
`testgen` feature.

| Part | State |
|---|---|
| `golden synth` | done: writes the synthetic ARW set from the scenes of `training/synth` and checks every file |
| render and compare runner (`yellowmoon render --json`, CIEDE2000 against stored references) | later task (M1-15): `golden` without a subcommand prints a notice and exits with code 2 |
| `manifest.json` | the golden entries; the synthetic ones are generated, the others come with the runner |
| `oracle_check.py` | LibRaw (rawpy) and ExifTool against the written files |

## `golden synth`

```bash
tools/fetch/synthetic.sh                     # the whole pipeline: scenes, then this
cargo run -p golden -- synth --scenes testdata/synthetic/scenes --out testdata/synthetic/arw \
    [--variants uncompressed14,packed12,craw,lossless-l,lossless-ms] [--only TEXT]... [--no-verify] [--jobs N]
```

For every scene sidecar (`<stem>.json` next to `<stem>.bin`, the little-endian `u16` mosaic, and `<stem>.jpg`, the embedded JPEG;
format in `training/synth/README.md`) and every variant, it writes `<stem>__<variant>.ARW` with `ym_raw::testgen`: Make SONY, model
and model ID, ISO, exposure time, f-number, focal length, lens, capture time, SR2 black, white and as-shot white balance (as Sony
writes them, on the 14-bit scale), neutral lens tables, the scene's JPEG as the maker-note preview. Without `--variants` each scene
gets the variants its sidecar recommends; with it, every listed variant that fits (a 14-bit mosaic for all but `packed12`, a width
that is a multiple of 32 for `craw`); the rest is reported as skipped with the reason.

Then each file is decoded again with `ym_raw::decode` and compared:

| Variant | Check |
|---|---|
| `uncompressed14`, `packed12`, `lossless-l` | every sample equals the mosaic |
| `craw` | `2 * abs(decoded - true) <= qstep` for every sample (the decoder's quantisation map) |
| `lossless-ms` | the YCbCr planes decode exactly as the writer's planes say, and with the gains divided out and replicated chroma the block means of the RGB match the bilinear RGB the file was made from (median over 16 x 16 blocks below 2 DN; 0.2 DN on the real scenes) |

and the metadata (make, model, model ID, ISO, exposure, f-number, focal length, lens, capture time, white balance, CFA, size,
levels, the preview bytes) must read back as written. A failure prints the first difference and the exit code is 1; a variant name in a sidecar that this crate does not know is a failure too (it would otherwise vanish from the set while the manifest still lists it). `index.json` in
the output folder lists every file with its size, SHA-256, the round-trip result and the hash of what our decoder returns (for Lossless
M/S in LibRaw's view: replicated chroma, white balance left in), which `oracle_check.py` compares with LibRaw. A run with `--no-verify` decodes nothing,
so it keeps the hash and result of an earlier verified entry with the same file bytes (and has none for new bytes: `oracle_check.py` then compares
with the mosaic only and says so).

The embedded JPEG: `ArwBuilder` can only write its own tiny gradient JPEG into the maker note, so `src/preview.rs` writes the
file with a placeholder of the right size, finds it through `ym_raw::metadata::probe_bytes`, and replaces it byte for byte with the
scene's JPEG padded with `COM` segments to exactly the same length; no offset in the file moves. (A `preview_jpeg` setter on the
builder would make this unnecessary.) The `Software` tag says the file is synthetic.

## `oracle_check.py`

```bash
python3 -I tools/golden/oracle_check.py [--arw testdata/synthetic/arw] [--scenes testdata/synthetic/scenes] [--only TEXT] [--json OUT]
```

LibRaw (through `rawpy`; a test oracle, never linked) must reproduce the mosaic exactly for the uncompressed, packed 12-bit and
lossless L files, and agree with ym-raw on every file it reads, cRAW and the Lossless M/S pseudo-raw included. LibRaw 0.22 reads
Lossless L only for the ILCE-1, ILCE-7x and ILCE-9x models; for the other bodies of the set (A6400, RX100 IV) it leaves the image
empty, and those files are reported as a model gate, not as a disagreement. ExifTool must read back every metadata field and extract the
embedded preview. Exit code 0 when everything agrees.

## Tests

`cargo test -p golden` (22 tests, a few seconds after the build): every variant on every CFA phase from scenes the tests write
themselves, preview embedding byte for byte, impossible requests skipped with a reason, idempotence, a corrupted or mismatched
sidecar refused, negative controls (the checks fail on a changed sample, a changed ISO or white balance), the padding of the preview at
every gap size, the demosaic, the CLI, and the typed ground truth (`Sidecar::truth()`): every sidecar in `testdata/synthetic/scenes`
parses and obeys its own rules (neutral patches neutral, one grey reference at LogC4 0.2784, probes inside the image, the metered grey
in the mosaic; skipped with a message when the scenes are not generated, a failure with `YM_REQUIRE_SYNTHETIC=1`).
The Python side: `python3 -m unittest training.synth.test_synth` (53 tests) and `python3 -m unittest discover -s tools/golden -p 'test_*.py'` (4 tests of the
verdict logic of `oracle_check.py`, no files needed).

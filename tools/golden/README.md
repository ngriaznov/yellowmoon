# tools/golden

Golden-image tooling (`docs/13-testing.md`, section Golden images). Test tooling: never shipped. It depends on `ym-raw` with its
`testgen` feature (the synthetic ARW writer) and on `ym-color` (CIEDE2000); the renders come from the real `ym` binary.

| Command | What it does |
|---|---|
| `golden check` | render the golden files with `ym render --json --settings ...`, compare with the stored pictures (ΔE00 per pixel on a 1/4 downsample: median, p99, max); fails on an unexplained change above ΔE00 0.5 |
| `golden check --device gpu` | the same files rendered strictly on the graphics device (`ym render --require-gpu`) and held against the CPU pictures by the device gates (below); a render that fell back to the CPU fails |
| `golden device-check FILE...` | real photos (the owner photos of CI): each rendered on the CPU and, strictly, on the device, and compared by the same gates; no manifest, no `expected.json` |
| `golden update --note "why"` | store new pictures and hashes; a note is required when an existing picture changes |
| `golden truth` | the end-to-end colour of the product against the ground truth of the synthetic scenes (a report, no gate) |
| `golden synth` | write the synthetic ARW set from the scenes of `training/synth` and check every file |
| `manifest.json` | the golden entries: the synthetic ones are generated, the others come with the fetch scripts |
| `expected.json` | the SHA-256 of every golden picture's pixels and the log of explained updates (committed) |
| `oracle_check.py` | LibRaw (rawpy) and ExifTool against the written files |
| `manifest_table.py` | prints the golden-set table of `docs/15` M0-09 (file, body, light, ISO, variant, licence tier; `--summary` for one row per body) as Markdown, from `manifest.json`; tests: `test_manifest_table.py`, `test_oracle_check.py` (stdlib only, `python3 -I -m unittest discover -s tools/golden`; the `fetch-tools` CI job runs them) |

## `golden check` and `golden update`

```bash
cargo build -p ym-cli                                  # the binary under test (target/debug/ym or target/release/ym)
tools/fetch/synthetic.sh                               # the input files (testdata/synthetic), a few minutes, no download
cargo run -p golden --release -- check                 # the default synthetic set (61 files), about 2 minutes on 4 cores
cargo run -p golden --release -- check --set cc0       # the CC0 tier, once tools/fetch/golden.sh has fetched it
cargo run -p golden --release -- check --filter D6500 --filter ILCE-1
cargo run -p golden --release -- update --note "why the pictures changed"
```

`--set` takes `synthetic` (the default: tier `synthetic`, set `default`), `synthetic-full`, `synthetic-eval` (local only),
`cc0`, `tier2` (tier `free`, Play Raw files), `tier3` (copyrighted, local only, never in CI) and `all`; `--filter TEXT`
(repeatable) keeps the entries whose id contains every text. Other options: `--ym PATH` (else `$YM_BIN`, else the newer of
`target/release/ym` and `target/debug/ym`), `--data DIR` (`YELLOWMOON_DATA` for a release build outside the workspace),
`--testdata DIR` (else `$YM_TESTDATA`, else `testdata/`), `--refs DIR`, `--expected FILE`, `--jobs N`, `--report FILE`
(everything as JSON), `--quiet`.

**What a render is.** One `ym render --cpu --apple-auto off --json --quiet --settings <the entry's settings> --format png --bits 16 --colour-space srgb` (with `YELLOWMOON_GPU=off`: golden hashes pin the CPU reference, `docs/17-gpu.md` section 12; Apple Auto off: the golden pictures are Yellowmoon's own Auto on every machine, also on a Mac where Apple Intelligence could answer, `docs/19` section 7)
process per file, in an empty `YELLOWMOON_HOME` (so the user's default look, display rendering and imported looks cannot
change it). One process per file because a photo's automatic exposure and pooled noise model depend on the other photos
of the same run (`docs/01`, Session): the same file rendered in other company would give other pixels. A render that takes
longer than 900 s (`GOLDEN_RENDER_TIMEOUT_SECS` changes it) is stopped and reported as `RENDER FAILED`, so one hung process
cannot hold a CI job until its own time-out.

**What a check is.**

```text
hash of the 16-bit pixels == expected.json          identical        pass
hash differs, reference picture present
     ΔE00 max <= tolerance (0.5)                    within           pass, with the numbers
     ΔE00 max >  tolerance                          EXCEEDED         fail
hash differs, no reference picture                  NO REFERENCE     fail (cannot say by how much)
reference is not the one expected.json describes    STALE REFERENCE  fail (an old cache)
no entry in expected.json                           NOT EXPECTED     fail
the render fails (ym exit 1, 3 ...)                 RENDER FAILED    fail
the input file is not on disk                       missing          skipped (--require-all: fail)
```

The hash is over the decoded samples and the size, not the file: the PNG's XMP carries the application version. The ΔE00
is taken on a 4 x 4 box filter in linear light, CIE L*a*b* (D65), CIEDE2000 (`ym_color::deltae`); `--metric p99` holds the
99th percentile against the tolerance instead of the maximum, `--tolerance` changes the 0.5. Exit codes: 0 pass, 1 a
file failed, 2 setup or usage (no `ym`, a bad manifest), 3 nothing could be checked (`--allow-empty` makes it 0).

**Changing a picture on purpose.** `golden update --note "why"` renders the selection again, writes the new pictures to
`testdata/golden/refs/<id>.png` and their hashes to `expected.json`, and appends the note, the pipeline version and the
largest ΔE00 it moved to `expected.json`'s `log`. Without `--note` an update that would change an existing picture
prints the pictures and their ΔE00 and writes nothing (exit 1). The note and the new hashes are the explanation
`docs/13` asks for; commit `expected.json` with the code change, and bump `pipeline_version` when pixels changed
(`golden check` prints a reminder when the pixels moved and the version did not). Added files need no note.
`--prune` drops the entries the manifest no longer lists.

**Where the pictures are.** `testdata/golden/refs/` is git-ignored (about 280 MB for the default synthetic set). A check
whose hashes are all identical needs none of them; they are read only to measure a change. CI keeps them in a cache
(see the `golden` job in `.github/workflows/ci.yml`): `golden check --require-all --write-missing-refs` fills the cache on a
green run of the main branch, and `--require-all` makes a file that is not on disk (a half generated set) a failure instead
of a skip.

**The manifest.** The entries of tier `synthetic` are written by `python3 -m training.synth.make_scenes --write-manifest
tools/golden/manifest.json` and leave `expected_hash` and `reference` empty on purpose (a test pins that): this runner
keeps the hashes in `expected.json` and the pictures under `testdata/golden/refs`. An entry's `settings` is the
`ym:Settings` document passed to `--settings` (`{}` is everything Auto; a field that is not a setting is an error in
`ym`, so a typo fails loudly).

## GPU mode: `golden check --device gpu` and `golden device-check`

The hashes of `expected.json` pin the CPU reference (`golden check` renders with `ym render --cpu` and `YELLOWMOON_GPU=off`). The device is held
to the CPU's pictures instead, within gates (`docs/17-gpu.md` section 12, `docs/13` "GPU golden"):

```bash
cargo build --release -p ym-cli && cargo build --release -p golden
golden check --device gpu --subset tools/golden/gpu-subset.txt          # the ten files a push checks (WGPU_BACKEND=vulkan on Linux)
golden check --device gpu                                               # all 61, the nightly run
golden check --device gpu --reference render --subset ...               # a Mac: against a CPU render of the same run
golden device-check --long-edge 1500 testdata/readme-photos/src/A1M01749.ARW   # one real photo, CPU against device
```

**What a GPU render is.** `ym render --require-gpu --apple-auto off --json` with `YELLOWMOON_GPU=require`, one process per file as for the CPU. Strict means no silent
fallback: no adapter is exit 4 (and `golden` stops with exit 2 before it renders anything, after asking `ym doctor --json` for the adapter), a device
error fails the photo (`error.kind` `gpu`, status `RENDER FAILED`). On top of that the JSON must say `"device": "gpu"` (it is `ExportReport.device`, the
path the photo took) and carry no note that mentions the graphics card ("The graphics card could not render this photo, so the CPU did: ..."): a
file that says otherwise is `FELL BACK` and fails whatever its pixels are. The notes about the classical demosaic and the like are not a fallback and
are kept in the report.

**What it is compared with.** `--reference stored` (the default): the pictures of `golden update` (`testdata/golden/refs`, the CI cache; a picture whose
pixels hash to `expected.json`'s own is `identical` without reading one; a missing or stale reference fails like in the CPU mode).
`--reference render`: a CPU render (`--cpu`) of the same run on the same machine, for a machine whose CPU path is not the one `expected.json` was
written on (a Mac: another libm moves the CPU hashes by a code) and where the question is how far the device is from the CPU of the same machine.
The Metal CI job uses it; the lavapipe job uses the stored pictures.

**The gates** (`GpuGate`, replaced one by one by `--gpu-median`, `--gpu-p99`, `--gpu-max`, `--gpu-codes-share`; `--gpu-gates plan` starts from the first
proposal): the ΔE00 of the 1/4 downsample (a box filter in linear light, CIEDE2000 in Lab, `ym_color::deltae`: the statistic of the CPU mode) must have a median, a
99th percentile and a maximum under the gates, and the share of the samples of the full-resolution picture that differ by more than 8 codes of 16 bits
must stay under its gate. The report (`--report FILE`) has, per file, the device `ym` reported, its notes, the ΔE00 statistics and the count.

| Gate | Proposed (`plan`) | Calibrated (default) | First run on lavapipe, 61 files | Ten-file subset |
|---|---|---|---|---|
| median ΔE00 | 0.02 | 2.1e-4 | worst 1.003e-4 | 9.8e-5 |
| 99th percentile | 0.05 | 2.2e-2 | worst 1.066e-2 | 9.2e-3 |
| maximum | 0.5 | 3.8 | worst 1.855 (`D6500__ILCE-7SM3__iso400__rggb__12b__packed12`), then 1.176 (`D6500__DSC-RX100M4 craw`) and four files between 0.55 and 0.57; the other 55 under 0.5 | 0.548 |
| samples more than 8 codes off | 1e-3 | 7.4e-3 | worst 3.66e-3 (largest difference 40741 codes) | 3.34e-3 |

The first proposal calls its gates "proposed ... calibrated on the first lavapipe run", and the project's rule for a calibrated number is twice the first measured value, so the default is twice
the worst of the 61 files (Mesa 25.2, LLVM 20.1, commit 4c5939f plus the GPU checks, 2026-10-15). It is **tighter than the proposal for the median and the 99th percentile
and looser for the maximum and the count**: with the proposed gates 50 of the 61 files fail, all of them on the count of samples (6 also on the maximum). What moves is not a stage but the output sharpening's halo
limiter, which decides on a rank of luma values: the device's f32 colour chain and the CPU's f64 one differ in the last bits, and a pixel at a hard edge of the chart (the charts' patch
borders) takes the other side of a rank (`docs/algorithms/04` section 10; the sharpening kernel alone matches the CPU to 3e-7). A median of 1e-4 and a 99th percentile of 1e-2 say that
the rest of the picture is the CPU's to the last code or two. Metal will have its own numbers (its f32 chain differs in other last bits); the Metal CI job prints them, and the gates are
raised there only with Mykyta's agreement, as for every gate (`AGENTS.md`). `PLAN` stays in the code so that the question is one flag away. Mykyta agreed to the calibrated defaults on 2026-10-15 (`docs/STATUS.md`, question 26).

**Files.** `tools/golden/gpu-subset.txt` lists the ten files of a push (one id per line, `#` comments; an id that is not in the manifest, or that `--set` leaves out, is an error: a subset never shrinks without a word): the four CFA phases, cRAW,
lossless L, 12 and 14 bit, camera-processed (Lossless M), the highest ISO, a tungsten light and a second body. Every chart scene carries blown highlights, so each file covers clipped highlights.
`--subset FILE` applies to every runner command, after `--set` and `--filter`.

**Real photos.** `golden device-check FILE...` needs no manifest: it renders each file twice (`--cpu`, then strictly on the device), at a long edge of `--long-edge` pixels (default 1500; 0 compares the
full-size pictures; the raw chain still runs at full resolution on both paths, so the work compared is the same), and applies the gates with `--reference render`. CI feeds it one of the owner's photos
(`testdata/readme-photos/src/A1M01749.ARW`, the `readme-photos` branch in Git LFS, cached by `tools/fetch/owner-photos.sha256`; `tools/fetch/owner-photos.sh --verify` checks the files). First measurement on lavapipe, long edge 1500: ΔE00 median 0 (mean 6.5e-5), p99 3.8e-4, max 0.0283, 94 of 4 500 000 samples (2.09e-5) more than 8 codes off (largest 102), 82 s for both renders on four shared cores; well inside the gates, so CI passes no overrides.

## `golden truth`

```bash
cargo run -p golden --release -- truth [--set synthetic] [--filter TEXT] [--patches] [--report truth.json]
```

For every scene of the set (one ARW per sidecar: the plain mosaic is preferred over cRAW and Lossless M/S) the real `ym`
writes a DNG with default settings, except that the look is Neutral, the highlight range is 0, "as recorded"
(`rolloff.highlights`), and the white lift is 0 (`white_offset_ev` -3). The default look is Signature since pipeline 0.3.0,
the highlights get an automatic lift since pipeline 0.3.0 (`docs/02` C12 step 1b), and since pipeline 0.4.0 the default
range of 1 places the camera's clip at white (`docs/02` C7); the truth is the colour of the scene, not a look, and the DNG
carries all three. At 0 the range is the identity below the ALEXA's clip, so the `rolloff` probes (the scenes' six-stop
highlight ramps) measure the colour of bright, unclipped tones with no tone stage in between. An entry's own `look`,
`rolloff` or `white_offset_ev` is kept.
The DNG holds the picture after the colour chain up to the look and curve (C1 to C11),
as linear AWG4: the last place before the display transform where Yellowmoon's colour can be compared with the ALEXA
estimate in the sidecar without going through a tone curve. Per patch, the interior is averaged in linear light, the picture is scaled by 0.18
over the luminance of its 18 % grey patch (the truth is reflectance-referred; a render has its own automatic exposure:
one number, so a white-balance error stays in the result) and CIEDE2000 is taken in CIE L*a*b* of the AWG4 colours
against `truth.probes[].alexa.awg4`. Three numbers per scene:

* `med`, `p95`, `max`: the 190 chart patches, render against the ALEXA-estimate truth;
* `b.*`: the baseline, Sony camera RGB through the same ARRI matrix with no sensor map (what a profile-less pipeline would give);
* `d.*`: the render against the truth after the ACES 2 display transform on both sides (the colour as it is seen);
* `neutr`: the median of the neutral ramp patches.

Patches whose noise-free value reaches the clip point are left out of the statistics. The scenes, the noise and the light
are synthetic assumptions (`training/synth/README.md`); the number says how well the chain recovers the colour of
spectral reflectances seen through the published Sony curves, not how it does on real cameras. No gate is defined for it.

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

The embedded JPEG: `ArwBuilder::preview_jpeg` writes the scene's own JPEG (`<stem>.jpg`) into the maker note as it is, byte for byte, with nothing added and nothing swapped afterwards; `src/preview.rs` only checks that its frame header has the size the sidecar says. The `Software` tag says the file is synthetic.

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

`cargo test -p golden`: the synthetic writer (22 tests, a few seconds after the build: every variant on every CFA phase from scenes the
tests write themselves, preview embedding byte for byte, impossible requests skipped with a reason, idempotence, a corrupted or
mismatched sidecar refused, negative controls, the demosaic, the typed ground truth: every sidecar in `testdata/synthetic/scenes`
parses and obeys its own rules, skipped with a message when the scenes are not generated, a failure with `YM_REQUIRE_SYNTHETIC=1`);
the runner against a stand-in `ym` (`tests/runner.rs`: identical, within, exceeded, no reference, stale reference, size change,
missing input, a failing render, the note an update needs, pruning, the exit codes); the image maths (ΔE00, the box filter in
linear light, the hash); the DNG reader against files `ym-export` writes; and `tests/real_ym.rs`, which renders two files with the
real `ym`, stores them, checks them identical and reads one scene's colour truth back (it needs the `ym` binary and
`testdata/synthetic`, and says so and passes without them, unless `YM_REQUIRE_SYNTHETIC=1`).
The Python side: `python3 -m unittest training.synth.test_synth` (53 tests) and `python3 -m unittest discover -s tools/golden -p 'test_*.py'` (4 tests of the
verdict logic of `oracle_check.py`, no files needed).

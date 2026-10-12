# 13 · Testing and gates

Every stage gets a measurable gate. The numbers below are starting targets; tighten them once we have real data. Loosening one needs Mykyta's agreement, recorded in `docs/STATUS.md`.

## Test layers

| Layer | Tooling | Runs |
|---|---|---|
| Unit tests (maths, parsers, components) | `cargo test`, Vitest | every push |
| GPU kernel parity (WGSL vs CPU reference) | `cargo test -p ym-gpu` on Linux CI with Mesa lavapipe (software Vulkan); on a Mac with Metal locally or on a self-hosted runner | every push (lavapipe); before each gate (Metal) |
| Model parity (PyTorch vs ONNX Runtime vs chosen runtime) | golden tiles | every model change |
| Decoder oracle | `tools/oracle` (rawler), rawpy (LibRaw) | every push touching `ym-raw`; nightly full corpus (`schedule:` trigger in `ci.yml`) |
| Golden images | `tools/golden` via the CLI | every push (small set), nightly (full set) |
| App/CLI parity | export the same job through both | every push |
| DNG | `dng_validate` from the DNG SDK | every push touching `ym-export` |
| End-to-end UI | Playwright against `api.mock.ts` | every push |
| Fuzzing | `cargo fuzz` (TIFF walker, SR2, cRAW, LJPEG, M/S, `.cube` parser) | nightly, 10 min per target |
| Licences | `cargo deny`, npm licence check; the `deny-selftest` job (`.github/scripts/deny-selftest.sh`) proves with local fixture crates that `deny.toml` rejects GPL, LGPL and banned crates and still lets the excluded `oracle` use `rawler` | every push |
| Test-data scripts | `fetch-tools` job: shellcheck and the unit tests in `tools/fetch/test_ymfetch.py` (local mock server, no network) | every push |
| Human review | blind A/B, visual checks | at gates [HUMAN] |

## GPU tests without a GPU

GitHub-hosted macOS runners may not expose a Metal device. GPU tests look for an adapter and skip with a clear message when none exists, unless `YM_REQUIRE_GPU=1` is set. The `gpu-lavapipe` CI job sets it and runs the kernels on Mesa's software Vulkan driver, which checks correctness (not speed). Metal behaviour and performance are checked on a real Mac before each gate [HUMAN M0-H4] or on a self-hosted Apple silicon runner if Mykyta registers one.

**As built (M0-07, M1-09 to M1-11).** `crates/ym-gpu/tests/common` opens one device per f16 storage mode the adapter supports (native `f16`, packed `f16`, plain f32; each is a separate device, so every mode is exercised) and skips when `Gpu::start` finds no adapter; with `YM_REQUIRE_GPU=1` that is a panic. A skipped test passes without testing anything (its loops are empty), so the skip is announced loudly: a banner written straight to the process's stderr, which the harness's output capture does not swallow (`eprintln!` from a passing test is captured and would have been invisible), plus a `::warning::` annotation on GitHub Actions; `a_missing_adapter_skips_loudly_or_fails_when_required` runs one GPU test in a child process on a backend that cannot exist (DirectX 12 on Linux and macOS) to prove both halves of the contract. Run them with `WGPU_BACKEND=vulkan YM_REQUIRE_GPU=1 cargo test -p ym-gpu` (about 3 minutes in debug on a loaded 4-core machine, almost all of it compiling, and 20 seconds of test time in release; the 50 MP test is `#[ignore]`, `cargo test -p ym-gpu --release --test gpu_spike -- --ignored --nocapture gpu_spike_50mp`, 43 seconds on lavapipe in release including the CPU reference).

| Test file | What it proves | Gate |
|---|---|---|
| `src/queue.rs` unit tests (no GPU) | priority order, FIFO within a priority, cancellation (pending, running, export vs navigation), shutdown and drop without deadlock, panics, context built on the worker thread, four threads submitting, cancelling and shutting down at once, a job waiting for another job from the worker thread gets `WouldDeadlock` instead of hanging | n/a |
| `tests/shaders_naga.rs` (no GPU) | every generated WGSL module, in all three storage modes, parses, validates and translates to MSL and SPIR-V with naga | n/a |
| `tests/gpu_spike.rs` | spike chain vs the CPU chain (four CFA phases, odd sizes, tiles of 32 to 128, all storage modes), the same against a reference that rounds to f16 where the GPU does (smooth scene and full-range white noise), tiled vs whole frame, u8 and u16 outputs, cancellation, plans without halo or with the wrong pattern refused, 50 MP memory (ignored) | 1e-3 (f16), 2e-5 against the f16-aware reference, tiled 1e-4, peak memory < 1.5 GiB |
| `tests/gpu_color.rs` | every stage C6 to C13 vs `ColorChain::trace` on random f16-exact inputs for 28 chains (Auto and everything-on for the four outputs, LogC3 looks, display-baked looks at 0, 25 and 100 %, ARRI display LUT variants, CDL-only and LUT-only looks), dither and u16 outputs vs the CPU, tiles vs whole image bit for bit, the DNG stop and the Before view, NaN and infinite inputs, 60 seeded random parameter sets (random subsets of the C9 controls, curves, CDL looks) | C7 to C11 2e-4 in LogC4, C13 one 10-bit code |
| `tests/gpu_color_vectors.rs` | ACES 2 for all four outputs and tetrahedral LUTs against the committed OpenColorIO vectors (`crates/ym-color/tests/data`); a negative control showing trilinear would fail; the f16-table precision argument | one 10-bit code |
| `tests/gpu_color_monotone.rs` | C7 non-decreasing over 192 parameter sets on a ramp that runs from 1e-5 through every shoulder to 60, every C9 control monotone in its value, the neutral ramp stays increasing with one control at its extremes | n/a |
| `tests/gpu_pipeline.rs` | raw `u16` to display pixels with the planes kept on the device vs an f16-aware CPU reference, row streaming, an export cut into one job per tile row (a Viewport job runs between the rows, same pixels as one job), tile size from the budget | one 10-bit code; 99.9 % of samples within a twentieth of a code |
| `tests/gpu_device.rs` | adapter report for `ym doctor`, forced storage modes, pool ledger and budget (a request just below the adapter maximum is not refused by the size-class rounding), queue ordering and navigation cancellation on the real device, shutdown, the pipeline-cache hook, kernel warm-up in every storage mode, a nested `Gpu::run` from the device thread is an error not a freeze, the skip contract above | n/a |

Inputs of the colour parity tests are f16-representable, so the CPU reference sees exactly what the GPU reads; the OpenColorIO DRT vectors are the exception, because their AP0 inputs are f32: the f32-storage device takes them exactly (0.05 code values at worst), the f16 devices are compared to the CPU chain on the rounded inputs (0.015 code values) and report their distance to OpenColorIO (up to 2.4 codes on four extremely saturated, bright wheel colours, the price of rounding the input to f16).

## Test data

Never commit raw files or large data. `tools/fetch/` downloads into `testdata/` (git-ignored) and checks SHA-256 against `tools/fetch/manifest.json`. CI caches `testdata/` keyed on the manifest hash. Usage, manifest format, licence rules and exit codes: `tools/fetch/README.md`. Which data backs which gate, where it may be stored and what is reachable from the agent environment: `docs/validation-data.md`.

**Status of the corpus (M0-09).** rawdb.dnglab.org, raw.pixls.us and discuss.pixls.us were blocked where the scripts were written, so `manifest.json` ships with the body list and an empty file list; `samples.sh --refresh` on a machine with open internet fills it. Until then decoder, detection and pipeline tests run on synthetic ARW files from the test-only writer (`ym-raw`, feature `testgen`) with LibRaw (rawpy) and rawler as independent oracles. Results obtained that way are reported as "synthetic only" and do not close a corpus gate.

**The synthetic ARW set (stop-gap).** `tools/fetch/synthetic.sh` generates a deterministic, physically based synthetic set into `testdata/synthetic/` (git-ignored; scenes in `scenes/`, ARW files in `arw/`; files directly in `testdata/synthetic/` are the decoder-variant corpus of `ym-raw`'s `gen_corpus` example and this script leaves them alone; about 37 mosaics and 61 ARW files, under 300 MB, a few minutes; `--full` adds every light under every body, a 6240 x 4160 scene for timing and an evaluation scene with the CIE CES: 88 mosaics, 114 files, about 740 MB; measured on the shared 4-core build machine, the default set takes 2.5 minutes cold). `training/synth` renders scenes (the 190 rawtoaces reflectances in a chart with gradients, a neutral ramp, a zone plate, slanted edges, a blown highlight and a deep shadow) under blackbody 2850 and 3200 K, CIE D 5500, 6500 and 7500 K, FL11, LED-B3, LED-V2 and one mixed light, through the published curves of six bodies (A7R IV, A7 III, A7S III, A6400, RX100 IV, and the A1 with model ID 384 on the A7R IV curves), with a known Poisson-Gaussian noise model and four CFA phases, as raw mosaics with a JSON sidecar of ground truth (the true light and white, the noise model, per patch the ALEXA-estimate reference in AWG4 and LogC4 and the "Sony through ARRI matrix" baseline, a made-up embedded JPEG). `golden synth` writes each mosaic as uncompressed 14-bit, packed 12-bit, cRAW, lossless L and lossless M/S ARW files through `ym_raw::testgen` and checks that decoding gives the mosaic back (exactly, or within `qstep / 2` for cRAW). Details: `training/synth/README.md`. The ground truth was recomputed by code that shares nothing with the generator except the data files (the published curves, the rawtoaces reflectances, ARRI's runtime matrix table) and agrees to the nine digits the sidecars keep on every patch (checked on all 37 default scenes in the review, on four in `training/synth/test_synth.py`, `IndependentRecomputation`); the mosaics hold the stated camera DN and noise (mean within sampling error, variance `a x + b + 1/12`). The set is **synthetic only**: it exercises the decoder, the colour chain and the comparison machinery, and says nothing about real cameras, real tone curves or real noise. The noise parameters and the JPEG rendering are our own assumptions, labelled as such in every sidecar.

Independent readers check the set: LibRaw (rawpy) reproduces the mosaic exactly for every uncompressed, packed 12-bit and lossless L file it reads (LibRaw reads lossless L only for the ILCE-1, 7x and 9x models; the A6400 and RX100 IV files are reported as a model gate), agrees with ym-raw on every cRAW file and on the Lossless M/S pseudo-raw RGB, and ExifTool reads back every metadata field and the embedded preview (`tools/golden/oracle_check.py`, `synthetic.sh --oracle`).

| Set | Source | Licence | Use |
|---|---|---|---|
| ARW corpus | RawDB (rawdb.dnglab.org, JSON API) and raw.pixls.us | CC0 (check each file; some older raw.pixls.us files are CC BY-NC-SA: exclude them) | decoder, detection, golden |
| A1 golden set | public sources in three licence tiers: CC0 (RawDB, raw.pixls.us), freely licensed Play Raw files, copyrighted evaluation-only files (`docs/test-photo-sources.md`) | per file, recorded in the manifest | golden images (tiers 1–2), visual review and blind A/B (all tiers, tier 3 locally only) |
| Illuminant ground truth | synthetic scenes (rawtoaces reflectances × reference lights); Cube++ (CC BY 4.0) converted per profile; NUS-8 Sony A57 subset (non-commercial, evaluation only) | as stated | illuminant estimator |
| Lens oracle | darktable-cli renders (GPL tool, run as a separate process); Adobe DNG Converter opcodes on a Mac | tool output only | lens corrections |
| Demosaic ground truth | PixelShift200 held-out scenes (CC BY 4.0) | CC BY 4.0 | demosaic gate |
| Colour sets | rawtoaces 190 patches (Apache-2.0), CIE 99 CES (CC BY-SA 4.0, fetched, test only) | as stated | sensor map |
| Demosaic sets | Kodak, McMaster, Gharbi hard and moiré | research use; fetched, evaluation only | demosaic gate |
| Noise truth | ELD Sony A7S II parameters | evaluation only | noise gate |
| ACES 2 vectors | generated offline with OpenColorIO ≥ 2.4 | ours (generated) | DRT parity |
| LUT vectors | generated with OCIO (tetrahedral) on our synthetic `.cube` files | ours | LUT parity |

## Gates

| Area | Test | Gate | Milestone |
|---|---|---|---|
| ALEXA curve | leave-one-camera-out on 52 cameras, reproduced in CI from `research/alexa-ssf/scripts/` | median RFSE ≤ 3%; emulation ΔE00 median ≤ 0.3 | M1 |
| ALEXA curve | compare with HdM SXT curve and CIC 2015 figure, if obtainable | report only | M1 |
| Sensor map | 99 CES samples × 9 held-out lights | per-light ΔE00 median ≤ 0.5, p95 ≤ 1.2 | M1 |
| Illuminant estimate | held-out synthetic scenes; Cube++ test split; NUS-8 Sony (report only) | median angular error ≤ 2° synthetic, ≤ 3° Cube++; sensor-map gate holds with estimated lights | M1 |
| Lens corrections | darktable-cli oracle; DNG Converter opcodes; line and flatness checks (`docs/05`) | gain within 0.5%, corner displacement within 0.2 px | M1 |
| Auto exposure | camera's embedded JPEG as reference | within ±0.5 EV of the JPEG's median luminance for ≥ 80% of the golden set | M1 |
| ARRI colour step | LogC3/4 and matrices vs colour-science and ARRI reference values | ≤ 1e-5 | M1 |
| Display transform | our ACES 2 vs OCIO vectors | ≤ 1 code value at 10 bits | M1 |
| LUT path | our tetrahedral vs OCIO vectors (and Resolve renders [HUMAN] optional) | ≤ 1 code value at 10 bits | M1 |
| Decoder | every corpus file vs LibRaw (and rawler) | identical raw values (rawler cRAW within its dither) | M0 (formats), M2 (ARW 6.0) |
| Camera detection | every Sony corpus sample | right body and profile; fallback reported | M1 |
| Camera profiles | own vs stand-in (port of `sony_profiles.py`) | matches the Python numbers ±0.02 | M1 |
| Demosaic + denoise | Kodak, McMaster, Gharbi sets; PixelShift200 held-out scenes | ≥ 2 dB PSNR over the classical fallback; no zipper or false colour in blind review | M2 |
| Noise estimate | noise-on-noise on the corpus; A1 studio ISO series (evaluation tier); ELD A7S II | increment within 15% median; `a` within 15% median | M2 |
| Noise appearance | A1 ISO ladder 100–25600 (studio ISO series, evaluation tier, local) | blind A/B parity with Capture One and DxO | M2 [HUMAN] |
| Highlights | 20 blown-highlight photos | no fringes in review; chroma within ΔE00 5 of surroundings | M1 |
| Golden images | fixed files through the CLI | no unexplained ΔE00 > 0.5 change | from M1 |
| App and CLI | same job through both | byte-identical | M3 |
| DNG | `dng_validate` render vs our JPEG | passes validation; ΔE00 median ≤ 1.5 | M3 |
| Performance | `docs/05` targets on M1 and M3 machines | report; regressions > 20% fail | every gate |
| Whole app | blind A/B of 50 A1 photos vs Capture One, Lightroom, DxO exports | ours preferred or tied in ≥ 50% of pairs | M4 [HUMAN] |

## Golden images

Four tiers, so CI works for everyone:

- **Synthetic tier (stop-gap):** the generated set above (`tier: "synthetic"` in the manifest; the evaluation scene is `synthetic-eval`, local only). It needs no download, so it is the only golden tier that can run in the environment that wrote this plan, and it runs wherever `tools/fetch/synthetic.sh` does (a few minutes). It is a **stop-gap until the CC0 tier is fetchable**: it tests regressions of our own pipeline on known, reproducible inputs and gives colour and exposure checks against ground truth, but it is not a stand-in for real files, and its references are our own earlier outputs like every golden reference. When the CC0 tier works, the synthetic entries stay as the always-available baseline (and the only source of ground truth for colour); they do not replace the real files. Generation is a pure function of the recipes, the code version and the library versions, so the manifest lists the files and their parameters and **no content hash of the ARW files** (another numpy version may draw other noise): the expected output hash and the reference image belong to the golden runner and its stored references.
- **CC0 tier:** RawDB and raw.pixls.us files with fixed settings. Runs on every push and pull request, including forks.
- **Free-licence tier:** the Play Raw files (CC0, CC BY, CC BY-SA) with attribution in the manifest. Same triggers; fetched from their original URLs or from a public test bucket that keeps the attributions.
- **Evaluation tier:** copyrighted files (Photography Life, Imaging Resource), downloaded by hand to a development machine for visual review and the blind A/B only. Never in CI, never uploaded.


- `tools/golden/manifest.json` lists files, settings (JSON, same schema as `ym:Settings`) and the expected output hash and a reference image (16-bit PNG, stored outside git with the test data). Each entry also has `tier` (`synthetic`, `synthetic-eval`, and for the real files `cc0`, `free`, `eval`), `id`, `file` (relative to `testdata/`) and, for generated files, `sidecar` (the scene's ground truth), `scene`, `light`, `body`, `model_id`, `iso`, `cfa`, `bits`, `variant`, `group` and `set` (`default`: a plain `synthetic.sh` run writes the file, 61 of them; `full`: only with `--full`, 35; `eval`: only with `--full`, the CIE scene, 18). A runner that takes `set == "default"` therefore finds exactly the files of the default run. The synthetic entries are written by `python3 -m training.synth.make_scenes --write-manifest tools/golden/manifest.json` (one per ARW file, with `settings: {}`, i.e. everything Auto, and empty `expected_hash` and `reference` until the runner exists); a test checks the committed file against the recipes. Entries of other tiers are left alone by that command.
- The runner renders through `yellowmoon render --json --settings <manifest entry>`, compares with the reference: ΔE00 per pixel on a 1/4-resolution downsample; reports median, p99 and max.
- When a change is intended, the PR updates the references with a note and (if pixels changed) bumps `pipeline_version`.

## Blind A/B protocol [HUMAN]

- 50 A1 photos covering daylight, shade, tungsten, LED, mixed light, night, skin, foliage, high ISO.
- Each photo exported by Yellowmoon (Neutral look) and by Capture One, Lightroom and DxO with default settings and a matched exposure. Pairs shown in random order and side, on a calibrated P3 display, through a small web page that records choices (`tools/abtest`, written by the agent).
- Raters: Mykyta plus at least two others. Report preference with a binomial confidence interval.

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
| Licences | `cargo deny`, npm licence check | every push |
| Human review | blind A/B, visual checks | at gates [HUMAN] |

## GPU tests without a GPU

GitHub-hosted macOS runners may not expose a Metal device. GPU tests look for an adapter and skip with a clear message when none exists, unless `YM_REQUIRE_GPU=1` is set. The `gpu-lavapipe` CI job sets it and runs the kernels on Mesa's software Vulkan driver, which checks correctness (not speed). Metal behaviour and performance are checked on a real Mac before each gate [HUMAN M0-H4] or on a self-hosted Apple silicon runner if Mykyta registers one.

## Test data

Never commit raw files or large data. `tools/fetch/` downloads into `testdata/` (git-ignored) and checks SHA-256 against `tools/fetch/manifest.json`. CI caches `testdata/` keyed on the manifest hash.

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

Two tiers, so CI works for everyone:

- **CC0 tier:** RawDB and raw.pixls.us files with fixed settings. Runs on every push and pull request, including forks.
- **Free-licence tier:** the Play Raw files (CC0, CC BY, CC BY-SA) with attribution in the manifest. Same triggers; fetched from their original URLs or from a public test bucket that keeps the attributions.
- **Evaluation tier:** copyrighted files (Photography Life, Imaging Resource), downloaded by hand to a development machine for visual review and the blind A/B only. Never in CI, never uploaded.


- `tools/golden/manifest.json` lists files, settings (JSON, same schema as `ym:Settings`) and the expected output hash and a reference image (16-bit PNG, stored outside git with the test data).
- The runner renders through `yellowmoon render --json --settings <manifest entry>`, compares with the reference: ΔE00 per pixel on a 1/4-resolution downsample; reports median, p99 and max.
- When a change is intended, the PR updates the references with a note and (if pixels changed) bumps `pipeline_version`.

## Blind A/B protocol [HUMAN]

- 50 A1 photos covering daylight, shade, tungsten, LED, mixed light, night, skin, foliage, high ISO.
- Each photo exported by Yellowmoon (Neutral look) and by Capture One, Lightroom and DxO with default settings and a matched exposure. Pairs shown in random order and side, on a calibrated P3 display, through a small web page that records choices (`tools/abtest`, written by the agent).
- Raters: Mykyta plus at least two others. Report preference with a binomial confidence interval.

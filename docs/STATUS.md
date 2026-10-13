# Status

Keep this file current (`AGENTS.md`). Newest first within each section.

## Project

Repository: github.com/ngriaznov/yellowmoon · Domain: not given yet (the handover message didn't include one; `docs/12` and the app have no domain-dependent setting yet).

Work so far is on branch `claude/yellowmoon-v1` (one commit per task or task group, no PR yet). Commits are authored as ngriaznov; links below are to commits.

## Current milestone

M0–M3 were built in parallel by task lanes in one session (2026-10-12), each lane implemented and then independently reviewed. Integration is done: ym-core (session, render pipeline, export queue), the `ym` CLI, the Tauri app, the frontend wired to it, the golden runner and dngfit. A polish and review pass is running. M4 hasn't started.

Two facts shape every number below:

- **No Mac.** The agent ran on Linux x86_64 (4 cores). GPU code was checked on Mesa lavapipe (software Vulkan); Metal, the native viewport, timings on M1/M3 and the release/updater flow need a Mac (M0-H4).
- **No CC0 corpus.** rawdb.dnglab.org and raw.pixls.us were unreachable, so the decoder, golden and noise gates run on (a) a physically based synthetic ARW set with ALEXA ground truth and (b) nine real ARW files from a public GitHub test-image repository, used for local evaluation only (never committed). The CC0 gates are blocked, not passed.

## Done

| Task | Commit | Notes |
|---|---|---|
| M3-12 product side: `ym` writes the fitted DNG profile | [7197d04](https://github.com/ngriaznov/yellowmoon/commit/7197d04) | white level floored at scene 16 (Question 9); Lightroom and other apps are M3-H6 |
| CI golden, dng and serde-order jobs; local-eval fetch tier; M1-08 auto-exposure study | [5ac59a5](https://github.com/ngriaznov/yellowmoon/commit/5ac59a5) | study only, no algorithm change (see Gate results and Questions) |
| Review follow-ups: American identifiers, noise pooling fix, early-export parity, thumbnails via `ym-frame://`, guarded update restart, size estimate refit | [425485f](https://github.com/ngriaznov/yellowmoon/commit/425485f) | |
| M3-02…M3-04 frontend wired to the real backend | [fd62641](https://github.com/ngriaznov/yellowmoon/commit/fd62641) | generated bindings with a compile-time contract; real-shell smoke test on 9 real + 48 synthetic files; native viewport needs a Mac |
| M1-15 golden runner, M3-12 dngfit and DNG gates | [4de91f6](https://github.com/ngriaznov/yellowmoon/commit/4de91f6) | 61 synthetic files hashed |
| M3-02…M3-04 Tauri app: IPC, bindings, fallback viewport, menus, updater | [8244118](https://github.com/ngriaznov/yellowmoon/commit/8244118) | native macOS viewport type-checked only (M0-H4); app and `ym` give identical JPEG bytes on real files |
| M1-14 `ym` CLI (render, info, looks, doctor) and the `ym` rename | [49e8cee](https://github.com/ngriaznov/yellowmoon/commit/49e8cee) | 109 synthetic files render; `--jobs 1` vs `--jobs 4` byte-identical |
| ym-core part 3: export queue, preferences, looks service, Core API | [dc9d8c3](https://github.com/ngriaznov/yellowmoon/commit/dc9d8c3) | M3-11, M3-13, M3-14 core half |
| ym-core part 2: render graph, scheduler, fit/100 %/thumbnail renders | [b960c3e](https://github.com/ngriaznov/yellowmoon/commit/b960c3e) | performance targets need a Mac (see Gate results) |
| M1-06 illuminant estimator | [2b50516](https://github.com/ngriaznov/yellowmoon/commit/2b50516) | FFCC-style, 11 per-profile models; gate passes on synthetic scenes; Cube++/NUS-8 blocked |
| Synthetic golden set | [283038a](https://github.com/ngriaznov/yellowmoon/commit/283038a) | 61 ARWs × 5 variants with ALEXA ground truth; stand-in for the CC0 tier |
| ym-core part 1 (settings, auto estimates, session) | [cd92c88](https://github.com/ngriaznov/yellowmoon/commit/cd92c88) | |
| M0-04, M0-05, M2-03, M2-09 ARW container and decoders | [55bc009](https://github.com/ngriaznov/yellowmoon/commit/55bc009) | every V1 variant except ARW 6.0 (gated); 9/9 real files identical to LibRaw |
| M0-07, GPU halves of M1-09/10/11, C11, C13 | [defd910](https://github.com/ngriaznov/yellowmoon/commit/defd910) | lavapipe only; Metal is M0-H4 |
| M3-01, M3-05…M3-08 UI, M3-09/11/13 UI, M3-15 | [deebc1c](https://github.com/ngriaznov/yellowmoon/commit/deebc1c) | frontend against the mock; wiring to the real backend in progress |
| Profile locus in camera raw scale (defect fix) | [e432ddd](https://github.com/ngriaznov/yellowmoon/commit/e432ddd) | real as-shot whites now Duv −0.0065 (A7 III), −0.0008 (A7 IV) |
| M1-07, M1-08 WB, fused C1–C6 chain, auto exposure | [132ba6b](https://github.com/ngriaznov/yellowmoon/commit/132ba6b) | M1-08 gate needs the golden set |
| M2-04, M2-05, M2-07 runtime, M0-08 harness | [47b2685](https://github.com/ngriaznov/yellowmoon/commit/47b2685) | no trained weights (M2-H4); classical path ships |
| Display transform + CPU colour chain (C7–C13) | [b7d6732](https://github.com/ngriaznov/yellowmoon/commit/b7d6732) | |
| M1-02, M1-04, M1-05 Rust runtime (ym-profiles, spectral, sensor map) | [7eeaf61](https://github.com/ngriaznov/yellowmoon/commit/7eeaf61) | |
| M1-10 library, M3-09 core (ym-looks) | [b91cc83](https://github.com/ngriaznov/yellowmoon/commit/b91cc83) | fingerprints wait for M3-H2 |
| M2-02 noise estimator | [e71a8c0](https://github.com/ngriaznov/yellowmoon/commit/e71a8c0) | synthetic validation; per-body priors need the corpus (M2-01) |
| M1-13, M3-12 writer (export) | [59a25da](https://github.com/ngriaznov/yellowmoon/commit/59a25da) | dng_validate passes locally; dngfit in progress |
| M1-12 CPU raw processing R1–R9 | [d232c3c](https://github.com/ngriaznov/yellowmoon/commit/d232c3c) | lens maths corrected against darktable-cli |
| M1-11 CPU ACES 2 | [f0f40a3](https://github.com/ngriaznov/yellowmoon/commit/f0f40a3) | |
| M1-01, M1-09/M1-10 CPU, C11, C13 (ym-color core) | [36515b2](https://github.com/ngriaznov/yellowmoon/commit/36515b2) | |
| M1-05 sensor map in CIE Lab (your decision) | [f402f7d](https://github.com/ngriaznov/yellowmoon/commit/f402f7d) | LED-V2 stays a reported gate failure |
| M0-01 Rust workspace, Tauri shell, CLI stub, data lookup | [fe74202](https://github.com/ngriaznov/yellowmoon/commit/fe74202) | |
| M1-02, M1-03, M1-04 fits, profiles, registry, ALEXA validation | [c10f0e7](https://github.com/ngriaznov/yellowmoon/commit/c10f0e7) | |
| M0-02, M0-03 (repo side), M0-09 (scripts; corpus blocked), M0-10 | [93e91e2](https://github.com/ngriaznov/yellowmoon/commit/93e91e2) | signing key is yours (M0-03); coverage and golden-set tables below |
| M0-01 frontend skeleton | [131a619](https://github.com/ngriaznov/yellowmoon/commit/131a619) | |

CI jobs enabled: frontend, e2e, rust (macOS), deny, deny-selftest, fetch-tools, gpu-lavapipe, oracle, lens-oracle, data-checks, synthetic-set, golden, dng, serde-order, eval-tools, parity, training-dry-run, alexa-curve. CI runs on pushes to `main` and on pull requests, so none of them has run on GitHub yet (no PR exists).

## Corpus and golden set (M0-09)

**Corpus coverage.** `tools/fetch/manifest.json` has no files yet (rawdb.dnglab.org and raw.pixls.us are unreachable from the build machine; M0-09 HUMAN), so every cell of the body × variant table is a gap: 0 files for all 38 bodies (19 required, 8 same-sensor siblings, 2 recent, 9 legacy). `tools/fetch/samples.sh --coverage` prints the table once the manifest is filled.

**Golden set, synthetic tier** (`tools/golden/manifest.json`, default set: 61 files, ours, CI-safe; 14-bit unless noted, packed12 is 12-bit):

| Body | cRAW | Lossless L | Lossless M/S | 12-bit packed | Uncompressed 14 | Lights | ISO |
|---|---|---|---|---|---|---|---|
| ILCE-7RM4 | 4 | 16 | 1 | 1 | 4 | D5500, D6500, D7500, F11, LED-B3, LED-V2, 2850 K, 3200 K, mixed | 100, 400, 800, 3200, 12800 |
| ILCE-7M3 | 1 | 1 | 1 | 1 | 3 | D6500, LED-B3, 2850 K | 400 |
| ILCE-7SM3 | 1 | 3 | 1 | 1 | 1 | D6500, LED-B3, 2850 K | 400 |
| ILCE-6400 | 3 | 1 | 1 | 1 | 1 | D6500, LED-B3, 2850 K | 400 |
| DSC-RX100M4 | 3 | 1 | 1 | 1 | 1 | D6500, LED-B3, 2850 K | 400 |
| ILCE-1 | 1 | 3 | 1 | 1 | 1 | D6500, LED-B3, 2850 K | 400 |

The `synthetic-full` and `synthetic-eval` sets (53 more, the latter with CIE reflectances, local only) and the CC0, free and Play Raw tiers have no entries yet.

**Local evaluation tier** (`tools/fetch/real-eval.sh`, never committed or used in CI): DSLR-A300 (ARW 1), DSLR-A500, ILCE-5000, ILCE-7M3, NEX-5N, NEX-7 (cRAW), ILCE-7M4 (Lossless L), ILCE-6300 (cRAW), ILCE-7M2 (14-bit uncompressed). Sources and licence notes in `docs/validation-data.md`.

## Gate results

| Gate | Date | Numbers | Pass |
|---|---|---|---|
| ALEXA curve, leave-one-camera-out (M1-03) | 2026-10-12 | median RFSE 2.17 % (≤ 3 %), emulation ΔE00 median 0.233 (≤ 0.3); reproduces the published 2.2 % / 0.23 | yes |
| Camera profiles vs Python (M1-04) | 2026-10-12 | Rust reproduces `sony_profiles.csv` within 0.0005 (±0.02) | yes |
| Sensor map, A7R IV, 99 CES × 9 held-out lights (M1-05) | 2026-10-12 | 8 of 9 lights pass; **LED-V2 median 0.759 / p95 1.515** (≤ 0.5 / ≤ 1.2), with the Lab objective you chose | no (LED-V2, reported) |
| Illuminant, held-out synthetic (M1-06) | 2026-10-12 | median 1.22–1.30° on all 11 profiles (≤ 2°) | yes |
| Illuminant, Cube++ (M1-06) | — | host unreachable | blocked |
| Sensor map with estimated lights (M1-06) | 2026-10-12 | end to end worst median 3.04 (white error dominates: 1° ≈ 1.2–1.9 ΔE00); map-only increase +0.15 median | no — see Questions |
| ARRI colour step, LogC, matrices (M1-01) | 2026-10-12 | every docs/02 constant to 5e-11; colour-science 1e-14 | yes |
| Display transform, ACES 2 vs OCIO 2.6 (M1-11) | 2026-10-12 | CPU max 0.03 code values at 10 bits (≤ 1); WGSL 0.05 (f32 storage) | yes |
| LUT path vs OCIO tetrahedral (M1-10) | 2026-10-12 | 0.0002 code values at 10 bits (≤ 1) | yes |
| GPU parity on lavapipe (M0-07, M1-09) | 2026-10-12 | C7–C11 ≤ 1.2e-5 LogC4 (≤ 2e-4); spike 50 MP 7.9e-4 (≤ 1e-3); peak GPU memory 0.28 GiB (≤ 1.5 GiB) | yes (lavapipe) — f16 caveat in Questions |
| Lens corrections vs darktable-cli (M1-12) | 2026-10-12 | distortion and CA ≤ 0.009 px (≤ 0.2 px), gain ≤ 0.008 % (≤ 0.5 %), synthetic Sony files | yes (synthetic) |
| Highlights (M1-12) | 2026-10-12 | chroma ΔE00 ≤ 4.66 on 10 synthetic scenes (≤ 5); blind review is yours | partial |
| Demosaic PSNR (M1-12) | 2026-10-12 | Kodak 29.82 / 38.88 / 39.16 dB, McMaster 32.31 / 34.21 / 35.09 dB (bilinear / Menon / LMMSE) | reported |
| Decoder vs LibRaw (M0-05) | 2026-10-12 | 9/9 real files and 46/46 readable synthetic files identical; cRAW within rawler's dither | synthetic + 9 real; corpus blocked |
| Camera detection on the corpus (M1-04) | — | corpus unreachable; all 28 registry bodies tested synthetically | blocked |
| Noise, noise-on-noise (M2-02) | 2026-10-12 | increment median 1.8 %, p90 5.6 % (≤ 15 % / 30 %), synthetic | yes (synthetic) |
| Noise, absolute `a` on A1 files within 15 % (docs/06) | — | needs the CC0 corpus | blocked |
| Noise, ELD A7S II `a` within 15 % (docs/06) | — | ELD dataset not fetched (host unreachable) | blocked |
| Noise, filtered-file detector ≥ 90 % flagged, ≤ 5 % false positives (docs/06) | 2026-10-12 | synthetic: smooth scenes 8/8 flagged, 0/8 false positives on every scene kind; mixed scenes 40 % (median 3×3) / 85 % (box), busy scenes 17.5 % / 57.5 % | no (only on smooth scenes) |
| Noise batch pooling (docs/06) | 2026-10-13 | 18 synthetic scenes of 2–4 frames: pooled `a` within 2.2 % of the single-file fits (was 65 % low on 3 frames) | yes (synthetic) |
| Auto exposure vs embedded JPEG (M1-08) | 2026-10-13 | 9 real files (local only, not the golden set): gate as built 5 of 9 within ±0.5 EV; DRT-neutral metric G3 (our exposure vs the camera's own gain) 6 of 9 within ±0.5, 9 of 9 within ±0.75, median −0.17 EV; misses are key-extreme scenes (two with DRO). G3 is the gate metric from 2026-10-13 (your decision) | no (6 of 9 < 80 %; outliers for M1-H5) |
| Export size estimate (docs/09) | 2026-10-13 | 9 real renders × 4 sizes, estimate/actual median 0.98–1.06, p90 error ≤ 0.30 in-sample (JPEG q80–q100, TIFF, PNG), leave-one-out up to 0.46 at q50–q89; low (≈ 0.66) with noise reduction off | median yes; p90 over the ±35 % target in leave-one-out |
| DNG passes dng_validate (M3-12) | 2026-10-12 | Adobe dng_validate 1.7.1 accepts every variant, all 61 synthetic and 9 real exports (local build) | yes (local) |
| DNG render vs our JPEG ΔE00 ≤ 1.5 (M3-12) | 2026-10-13 | DNGs as `ym` writes them (fitted profile), every file's median: synthetic 0.57–1.04 (61 files, median 0.69), real 0.45–1.06 (9 files, median 0.73, portrait A7 III 1.02); against the 16-bit PNG 0.03–0.10 | yes |
| End-to-end colour vs ALEXA-estimate truth (M1-15 report, no gate) | 2026-10-12 | scene-linear AWG4, 190 chart patches: median 0.73 ΔE00, p95 2.05 over 37 scenes (Sony-through-ARRI-matrix baseline 1.72); A7R IV ISO 400 single lights 0.68 / 1.61; ISO 12800 2.24 | reported |
| Golden images (M1-15) | 2026-10-12 | 61 synthetic files identical on re-run, thread-count independent; a saturation +3 change fails at ΔE00 3.6 | yes (synthetic tier) |
| Demosaic + denoise network (M2) | — | no trained weights (M2-H4) | blocked |
| 500 files, commands < 100 ms (M3-03) | 2026-10-13 | `open_paths` 31 ms, slowest of 5112 commands while developing 62.5 ms (ignored test, local, busy machine); webview keypress on screen in 5–72 ms (median 16), one manual run | yes (local; not run in CI) |
| Performance (docs/05) | 2026-10-12 | Linux x86 only: Lossless L 50 MP decode 333 ms on 4 cores; classical chain ~64–69 CPU-s per 50 MP; Mac numbers are M0-H4 | reported |

## Waiting on Mykyta

| ID | What | Needed for |
|---|---|---|
| M0-03 | Generate the updater signing key and set the two GitHub secrets; put the public key in `src-tauri/tauri.conf.json` (exact commands in `docs/release-signing.md`). Also decide whether the release bot may push the cask update to `main`. | first release |
| M0-H4 | Mac checks: native viewport (M0-06 checklist), Metal parity and timings (`cargo test -p ym-gpu`, `crates/ym-ml/examples/bench.rs` for the ML runtime choice), decode/render timings, the 0.0.1 → 0.0.2 self-update, `brew install --cask`. | M0 gate |
| M0-09 | On a machine with open internet: `tools/fetch/samples.sh --refresh`, review and commit `tools/fetch/manifest.json`; promote raw.pixls.us files after a licence check; collect Play Raw tier-2 files. | decoder, detection, golden, noise gates |
| M1-H3 | Email HdM about the SXT curve. | ALEXA curve check |
| M1-H4 | Confirm that shipping coefficients fitted with CIE SPDs (CC BY-SA 4.0) is acceptable (profiles, illuminant models). | M1 |
| M1-H5 | Visual review of renders once the CLI renders the golden set. | M1 gate |
| M2-H3 | Email Sony about RDD 34 / ARW 6.0. | ARW 6.0 |
| M2-H4 | GPU time and PixelShift200 to train `jdd` (`make -C training/jdd train-jdd`). | M2 |
| M2-H5, M4-H1 | Blind A/B tests. | M2, M4 |
| M3-H1 | Design and name the Signature look (`docs/08` parametric look format is ready). | M3 |
| M3-H2 | Download the five ARRI packages and run `tools/looks/fingerprint.py`. | wizard recognition by hash |
| M3-H4, M3-H5, M3-H6 | App review on a Mac; a real shoot; DNGs in Lightroom, RawTherapee, darktable, Capture One, Apple Photos. | M3 gate |
| — | A real Lossless M or S file (A7R V, A9 III, A1 II or RX1R III) to confirm the M/S scale. | M2-09 |
| M3-H3 | Email ARRI about an in-app download of the free looks (optional). | M3 |
| M4-H2…M4-H5 | Developer ID signing and notarisation decision; trademark clearance (EUIPO, INPI, WIPO, UK IPO); freedom-to-operate check on US 11948279; announcement. Not started. | M4 |

## Questions for Mykyta

1. **Sensor-map gate with estimated lights (docs/02, M1-06).** As worded it can't pass: any white-balance error dominates (1° ≈ 1.2–1.9 ΔE00, while the gate needs the white right to about 0.3°). Proposal in docs/02: gate the estimator by angular error only, and state the estimated-light check as the map-only increase (≤ +0.2 median, ≤ +0.7 p95 — measured +0.15 / +0.69, so these thresholds are post-hoc). Accept, or another wording?
2. **GPU storage precision.** The spec says f16 tiles. f16 storage alone can exceed the 1e-3 spike gate on white-noise data (up to 6e-3) and the 1-code DRT gate on 3–4 of 898 extreme vectors; f32 storage meets both and doubles GPU memory (50 MP peak ~0.28 → ~0.56 GiB, gate 1.5 GiB). Options: f32 by default; keep f16 and accept a data-dependent gate; f16 only when memory-bound.
3. **16-bit TIFF default.** On a real 24 MP frame 16-bit LZW was 26 % larger than uncompressed and 38 % larger than ZIP. Switch the default to ZIP?
4. **Spec error in M1-07.** docs/15 says "3200 K under daylight renders warmer than Auto"; physically it renders bluer. The test checks the ALEXA-maths amount with the correct sign. OK to fix the wording?
5. **M1-02 tolerance.** "Runtime values match to 1e-6 on a test grid" holds on grid; off grid the 10 K tables reach 5e-6 (blackbody) and 4e-4 (daylight, because colour-science rounds the D-series M1/M2 coefficients). Accept these tolerances, or use a 1 K blackbody grid (1.4e-7) and unrounded daylight?
6. **Robust as-shot prior for the illuminant.** A two-component prior that discounts a wrong white-balance preset improves p90 4.93 → 4.16° and wrong-preset scenes 5.9 → 3.1°, but the worst case rises 22 → 33°. Adopt (needs a retrain)?
7. **Tint readout near 4000 K.** The blackbody and daylight locus branches meet with a 0.04 log-chroma gap, so small magenta tints on ~4100 K lights read with the wrong sign. Fix with one Planckian readout locus or a blend over 3500–4500 K (changes `profile.json`)?
8. **IJG licence.** JPEG export uses `jpeg-encoder`, whose forward DCT is IJG-licensed (permissive, needs "This software is based in part on the work of the Independent JPEG Group." in docs and About). It's allowed in `deny.toml` now; please confirm.
9. **DNG brightness outside Adobe apps.** Adobe's renderer applies a highlight shoulder rather than a scale for a negative `BaselineExposure`, so the fitted profile stores scene values up to 16 with the white level there and `BaselineExposure` about 0. Adobe apps render it to match our JPEG; RawTherapee and darktable show the stored linear image about 4 EV darker until their own exposure handling. The alternative (per-photo white level) risks Adobe's shoulder. Recommended: keep the floor and say so in the export help. OK?

## Known issues

- C9 grade controls can fold the neutral ramp when two are combined at opposite extremes (e.g. Highlights +100 with White −100); each alone is monotone. Needs a UI limit or a constants refit in M3.
- Day/Shade/Tungsten presets carry the 3.5–6 % difference between the Weta curves' white and the camera ColorMatrix white (A7 III/A7 IV); a per-body raw-white calibration would need data.
- Noise estimator over-reads `a` on busy low-ISO scenes (median 14 %, p90 38 %); borderline against the 15 % gate.
- The illuminant's synthetic brightness range stops at Bv 13; real sunlit files read 14.3–14.6. LED and fluorescent class recognition is weak on independent chart scenes.
- Kelvin readout rounded to 50 K leaves up to 1.6 % red/blue imbalance below 2300 K.
- The GPU memory budget isn't scaled to system RAM; device loss isn't handled.
- M1-12's WGSL half is open: only R1 + R4 (normalise, gains) and the colour chain have GPU kernels; R3, R5, R6 (Menon/LMMSE), R8 and R9 run on the CPU reference. The fit-preview and 100 % performance targets depend on them (and on a Mac, M0-H4). The noise σ map (M2-02 "on GPU") is CPU only too.

## Environment

Linux x86_64 container, 4 vCPUs, Rust 1.97.0, Node 22, Python 3.13; Mesa lavapipe for wgpu; darktable-cli 4.6.1 (lens oracle, separate process); LibRaw 0.22.1 via rawpy, rawler 0.8 and libjpeg as decoder oracles; OpenColorIO 2.6 for test vectors; Adobe DNG SDK 1.7.1 `dng_validate` built from the GrapheneOS AOSP mirror. No Mac: every Metal, native-viewport, timing and release check is listed under M0-H4.

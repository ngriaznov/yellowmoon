# Status

Keep this file current (`AGENTS.md`). Newest first within each section.

## Project

Repository: github.com/ngriaznov/yellowmoon · Domain: not given yet (the handover message didn't include one; `docs/12` and the app have no domain-dependent setting yet).

Work so far is on branch `claude/yellowmoon-v1` (one commit per task or task group, no PR yet). Commits are authored as ngriaznov; links below are to commits.

## Current milestone

M0–M3 were built in parallel by task lanes in one session (2026-10-12), each lane implemented and then independently reviewed. Integration is done: ym-core (session, render pipeline, export queue), the `ym` CLI, the Tauri app, the frontend wired to it, the golden runner and dngfit. A review pass followed, and on 2026-10-13 the raw-algorithms wave (below) changed the R6 and sharpening defaults. M4 hasn't started.

Two facts shape every number below:

- **No Mac.** The agent ran on Linux x86_64 (4 cores). GPU code was checked on Mesa lavapipe (software Vulkan); Metal, the native viewport, timings on M1/M3 and the release/updater flow need a Mac (M0-H4).
- **No CC0 corpus.** rawdb.dnglab.org and raw.pixls.us were unreachable, so the decoder, golden and noise gates run on (a) a physically based synthetic ARW set with ALEXA ground truth and (b) nine real ARW files from a public GitHub test-image repository, used for local evaluation only (never committed). The CC0 gates are blocked, not passed.

## Raw algorithms (RA wave, 2026-10-13)

Mykyta's handover package (`docs/algorithms/`; tasks in `docs/algorithms/TASKS.md`). Decisions in force (`docs/16`): the adaptive ratio demosaic is the R6 default; R8 stays Richardson–Lucy; output sharpening is a per-photo setting, default 100 %; no C++ in the repository. The choice of R6 or R8 method is internal: no setting, no UI control, no CLI flag.

| Task | State | Commit | In one line |
|---|---|---|---|
| RA-01 | done | [09144fe](https://github.com/ngriaznov/yellowmoon/commit/09144fe) | Specs, vectors and the prior-art record are in. The C++ references stay outside the repository. |
| RA-02 | done | [ddca8d3](https://github.com/ngriaznov/yellowmoon/commit/ddca8d3) | Shared plumbing: `NoiseDn`, `DemosaicInputs`, `Chain::noise`. No render changed. |
| RA-03 | done | [ddca8d3](https://github.com/ngriaznov/yellowmoon/commit/ddca8d3) | Adaptive ratio demosaic: 13 of 17 vectors bit-identical, the other four 1.2e-7 off; equal to the C++ oracle on thousands of random cases. |
| RA-03b | done | [01f025f](https://github.com/ngriaznov/yellowmoon/commit/01f025f) | CA-aware refinement, vector-exact. The chain no longer reaches it: lateral CA is corrected on the mosaic instead (R5b, question 11). |
| RA-04 | done | [ddca8d3](https://github.com/ngriaznov/yellowmoon/commit/ddca8d3) | `demosaic_with` takes the balanced mosaic and divides by `g0` itself; test-only method switch; the app and `ym` give identical bytes. |
| RA-05 | done, two sets blocked | [ddca8d3](https://github.com/ngriaznov/yellowmoon/commit/ddca8d3) | Adaptive ratio wins the PSNR rule; the owner made it the default after the blind look (RA-H4). Gharbi and PixelShift200 could not be fetched. |
| RA-06 | done | [ddca8d3](https://github.com/ngriaznov/yellowmoon/commit/ddca8d3) | Log-guided demosaic, within 1 dB of Menon on the synthetic set. No render path uses it. |
| RA-07 | done | [ddca8d3](https://github.com/ngriaznov/yellowmoon/commit/ddca8d3) | Diffraction deconvolution and the pixel pitch in the registry (27 of 28 bodies). Richardson–Lucy stays the default; RA-H2 is open. |
| RA-08 | done | [ddca8d3](https://github.com/ngriaznov/yellowmoon/commit/ddca8d3) | Output sharpening: Output slider 0–200 %, default 100, `ym render --output-sharpening`; every picture, never a DNG. |
| RA-09 | done on lavapipe | [ddca8d3](https://github.com/ngriaznov/yellowmoon/commit/ddca8d3), [1bcaef7](https://github.com/ngriaznov/yellowmoon/commit/1bcaef7), [fdab64d](https://github.com/ngriaznov/yellowmoon/commit/fdab64d) | WGSL kernels for all four algorithms, parity tests pass. None is wired into a render path; Metal is unchecked. |
| RA-10 | done | [e7dd7fd](https://github.com/ngriaznov/yellowmoon/commit/e7dd7fd) | `docs/05`, `docs/09`, `docs/01`, `docs/16`, `docs/algorithms/*`, `CHANGELOG.md` and this file match the code. |
| R6 default | done | [e0ca394](https://github.com/ngriaznov/yellowmoon/commit/e0ca394) | `DEFAULT_METHOD` is `AdaptiveRatio`; a missing or unusable noise model renders as zero noise instead of failing. |
| Tone card | done | [ddca8d3](https://github.com/ngriaznov/yellowmoon/commit/ddca8d3) | Look › Tone: Temperature, Contrast and Saturation, mirrored from the Colour and Light tabs; saturation is subtractive (`docs/02` C9). |
| Lateral CA (R5b) | built, pending your OK | | CA corrected on the mosaic before the demosaic at full resolution; real-file renders with a CA table change (question 11). |

**Golden set.** RA-02 to RA-07 only added code behind switches: the 61 synthetic files stayed identical where it was checked (all 61 after RA-02; all 61 after RA-04 with Output at 0; RA-06 and the lens work reported 61 identical; a 7-file subset after RA-07 with Output at 0), and so did all 61 with RA-08's amount at 0. At the new defaults (adaptive ratio, Output 100 %) all 61 move: per-file ΔE00 median 0.007 to 1.04, p99 3.6 to 9.3, maximum 6.6 to 26.3, mostly ringing at the charts' hard patch edges. `tools/golden/expected.json` was re-baselined once, at the end of the wave, with one `golden update --note` and `PIPELINE_VERSION` 0.1.1 → 0.1.2 ([1c25c2c](https://github.com/ngriaznov/yellowmoon/commit/1c25c2c)). The note reads "pipeline 0.1.2: R6 default adaptive ratio demosaic (owner decision RA-H4; linear-light PSNR 36.05 dB vs Menon 34.43) and output sharpening 100 % screen preset (owner decision RA-H3)". `golden check` just before the update (release `ym`, `--jobs` default): 0 identical, 61 exceeded, ΔE00 against the old pictures per file: median 0.007 to 1.04 (median of the 61 medians 0.354), p99 3.6 to 9.3 (median 8.1), maximum 6.6 to 26.3 (median 21.3; the largest is the D6500 ILCE-7SM3 uncompressed14 chart). After the update: 61 changed, 0 added, and `golden check` gives 61 identical, 0 failed with `--jobs 1` and with `--jobs 4`.

**Final verification of the wave (Linux x86_64, lavapipe; `WGPU_BACKEND=vulkan YM_REQUIRE_GPU=1`).** `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps` and `cargo deny check` (advisories, bans, licences, sources ok) are clean. `cargo test --workspace --no-fail-fast`: 2439 passed, 0 failed, 30 ignored (dngfit 40/0/1, golden 66/0/0, oracle 11/0/0, yellowmoon-app 95/0/3, ym-cli 149/0/0, ym-color 463/0/6, ym-core 391/0/8, ym-export 164/0/4, ym-gpu 501/0/4, ym-looks 158/0/1, ym-ml 29/0/0, ym-noise 101/0/0, ym-profiles 39/0/0, ym-raw 232/0/3; passed/failed/ignored, doc-tests included). `app_cli_parity` against a separately built `ym`: 2 passed with the plain binary (the forced-method test skips itself, as designed) and 2 passed with the `force-demosaic` binary and `YM_REQUIRE_FORCE_DEMOSAIC=1`. `npm test` 1168 passed in 66 files, `npx tsc -b` and `npm run build` clean, `npm run e2e` 212 passed. The vectors against the C++ oracle (kept outside the repository): all four algorithms self-test and match (18, 23, 16 and 16 vectors); the committed vectors are unchanged. Python tool tests: licences 22, eval 77, golden 9. The Mac-only checks stay under M0-H4.

**Costs, CPU only.** The new default takes the demosaic stage 1.5 to 1.7 times Menon's time: +3 to +13 % CPU time for a whole export, +15 to +40 % wall time for the 100 % view and a full render. The Output stage costs 825 ms per 2 MP fit frame and 178 ms per 512 px tile at 100 %, and re-runs after every edit. The GPU wave (`docs/16`, the owner's direction) is meant to remove both costs.

**Auto exposure, `ym info` against `ym render` (checked at the end of the wave).** The auto-exposure study once stopped on A1M01749 ("the display render used +0.7229 EV, the study needs +0.6784 EV"). On the current tree the two agree to the last digit on all 12 local ARWs (the 3 readme photos and the 9 real files): `ym info` `auto.exposure_ev`, the render's `auto.exposure_ev` and its `settings.exposure.ev` are the same number (A1M01749: +0.6783832054405404). They cannot differ in one binary: both read `AutoEstimates::compute` through `Session::settings`, the render's exposure is `chain.exposure_ev()` of the all-Auto chain, and the estimate is the same with 1, 2 or 8 threads, on 1 CPU and under load. Pooling and bursts do not move it either (three A1 files opened together read the same exposures as alone). `auto.rs` and `session.rs` are unchanged since the study's commit, and its `info` value (+0.678383) is today's, so the +0.7229 render must have come from another build than the one that answered `ym info`. The cause that fits is the shared `target/debug/ym`, which other lanes rebuilt while the study was running, between its `ym info` and its renders of that photo (the old binary is gone, so this is inferred, not replayed). `tools/eval/autoexp.py` now refuses to go on when the `ym` file changes (inode, size or modification time) during a run and says so, instead of reporting a mismatch of unknown cause (test `test_a_binary_that_is_rebuilt_during_the_run_is_refused`). The auto-exposure target is unchanged.

## GPU wave (2026-10-14, the owner's direction in `docs/16`)

Everything after decoding moves onto the wgpu device (Metal on the Mac), in the app and in `ym`, exports included: `docs/gpu-plan.md` (ten tasks G-01 to G-10 in two lanes and five waves, contracts, tests). Golden hashes stay pinned to the CPU reference. Metal correctness runs in CI on GitHub's Apple-silicon runners (`.github/workflows/mac-gpu.yml`); speed and the screen stay M0-H4.

| Wave | Lane A | Lane B | State |
|---|---|---|---|
| 1 | G-01 device policy, flags, goldens on the CPU | G-02 device foundation | done, reviewed (2026-10-14): `GpuPolicy` and `YELLOWMOON_GPU`, `ym render --cpu`, one `open_gpu` for app and `ym`, golden runner on `--cpu` (61 identical); `Auto` storage is f32, poison mode, device-loss handling, cancel tokens, `GpuImage`/`GpuRaw`, HLSL and capability checks |
| 2 | G-03 adaptive ratio on a device chain | G-04 view kernels | done, reviewed (2026-10-14): the default demosaic runs on the device inside the full-resolution tile path (`GpuChain`, `StageRange`, split above 2.5 Mpixel; app and `ym` byte-identical on the device; a GPU error re-renders the whole photo on the CPU, never a mix of tiles; `--require-gpu` fails it with kind `gpu`); view kernels: persistent colour tables, device colour stage, geometry sampler, C13 bands, sharpening chained to C13. Metal fixes for the 14 baseline failures (exact luma and output-sharpening luma hidden from the fast-math optimiser, checked with a clang fast-math model; overflow bounds; a Metal column for the adaptive ratio tie vector), to be confirmed by Metal CI. Golden 61 identical on the CPU path |
| 3 | G-05 view wiring | G-06 denoise, R7, R8 kernels | running |
| 4 | G-08 export tail | G-07 raw front, lens, fit view | — |
| 5 | G-09 CI, GPU golden, determinism, ML seam | G-10 device residency | — |

First Metal CI run (2026-10-14): the Metal adapter opens on macos-14 and macos-15; the app builds and its tests pass on macOS (the native Metal viewport compiled for the first time); the frontend build failed on macOS because `wheel.ts` shadowed `Wheel.tsx` on a case-insensitive file system (fixed, guarded by `src/casing.test.ts`); the adaptive ratio vector test fails on Metal (`edge_knight` 40 tie flips against 10 on lavapipe, worst 6.6e-4) and is handed to G-03.

Metal baseline on the verified commit d629dbe (`docs/gpu-plan.md` section 12; identical on macos-14 and macos-15, adapter "Apple Paravirtual device"): 23 of 27 `ym-gpu` test binaries pass on Metal, including every colour-pass test; 14 tests fail in four binaries — adaptive ratio tie flips (1), diffraction (6, among them the exact luma: 30304 of 60000 values differ), log-guided overflow to inf (1), output sharpening (6, up to 8.8e-4) — consistent with Metal's default fast-math (reassociation, FMA contraction, no-NaN/Inf assumptions). They go to G-03 and G-04 as acceptance items. `ym-core` showed five macOS portability failures unrelated to the GPU, now fixed: the default export folder keeps the path as given (`ym_export::naming::default_destination`, one rule for the app, `ym` and the preview; a symlink in the path is no longer resolved, so `/var/...` stays `/var/...`), the non-UTF-8 folder test is ignored on macOS with its reason and a cross-platform test replaces it, and the thumbnail-look failure was a test race (deterministic hooks now; 1000 stress runs, no failure).

Metal after wave 2 (commit b6f2e4b, run 38519087612; identical on macos-14 and macos-15): `ym-core` (every test, the GPU ones against the CPU, with fault injection), the device-fault and ML tests, app ↔ `ym` export parity on the device and the app job all pass. `ym-gpu`: 618 pass, 6 fail (14 at the baseline): the six diffraction failures are fixed with unchanged limits, two output-sharpening ones and the adaptive ratio vector test pass; still failing are four output-sharpening tests (up to 1.96e-3 against 1.2e-3), the log-guided 1e30 case (relative 1.06e-5 against 1e-5) and one new test, the adaptive ratio on a strong-lens corner tile in the chain (921 samples over 1e-4 against about 630). A Metal fix lane (kernels only, lavapipe limits unchanged) is running beside wave 3; it also assesses turning Metal fast-math off for every kernel instead of fixing kernels one by one.

## Done

| Task | Commit | Notes |
|---|---|---|
| Adversarial review (42 findings, 40 confirmed) and fixes: neutral clipped highlights, licence notices and THIRD-PARTY.md in the bundle, same-name exports, terminal-safe `ym info`, dialog focus, folder drops with dots, nightly fuzzing, CI coverage | [09144fe](https://github.com/ngriaznov/yellowmoon/commit/09144fe) | goldens re-baselined (55 of 61, clipped patches only), pipeline 0.1.1 |
| RA-01 raw-algorithms package in (specs, integration notes, test vectors; C++ kept outside the repository) | [09144fe](https://github.com/ngriaznov/yellowmoon/commit/09144fe) | RA-02 to RA-10: see Raw algorithms above |
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
| M3-01, M3-05…M3-08 UI, M3-09/11/13 UI, M3-15 | [deebc1c](https://github.com/ngriaznov/yellowmoon/commit/deebc1c) | frontend against the mock; wired to the real backend in the M3-02…M3-04 row above |
| Profile locus in camera raw scale (defect fix) | [e432ddd](https://github.com/ngriaznov/yellowmoon/commit/e432ddd) | real as-shot whites now Duv −0.0065 (A7 III), −0.0008 (A7 IV) |
| M1-07, M1-08 WB, fused C1–C6 chain, auto exposure | [132ba6b](https://github.com/ngriaznov/yellowmoon/commit/132ba6b) | M1-08 gate needs the golden set |
| M2-04, M2-05, M2-07 runtime, M0-08 harness | [47b2685](https://github.com/ngriaznov/yellowmoon/commit/47b2685) | no trained weights (M2-H4); classical path ships |
| Display transform + CPU colour chain (C7–C13) | [b7d6732](https://github.com/ngriaznov/yellowmoon/commit/b7d6732) | |
| M1-02, M1-04, M1-05 Rust runtime (ym-profiles, spectral, sensor map) | [7eeaf61](https://github.com/ngriaznov/yellowmoon/commit/7eeaf61) | |
| M1-10 library, M3-09 core (ym-looks) | [b91cc83](https://github.com/ngriaznov/yellowmoon/commit/b91cc83) | fingerprints wait for M3-H2 |
| M2-02 noise estimator (partial) | [e71a8c0](https://github.com/ngriaznov/yellowmoon/commit/e71a8c0) | synthetic validation; per-body priors need the corpus (M2-01); filtered-file detector passes on smooth scenes only; σ map CPU only |
| M1-13, M3-12 writer (export) | [59a25da](https://github.com/ngriaznov/yellowmoon/commit/59a25da) | dng_validate passes locally; dngfit and the DNG gates are in the M1-15 and M3-12 rows above |
| M1-12 CPU raw processing R1–R9 | [d232c3c](https://github.com/ngriaznov/yellowmoon/commit/d232c3c) | lens maths corrected against darktable-cli |
| M1-11 CPU ACES 2 | [f0f40a3](https://github.com/ngriaznov/yellowmoon/commit/f0f40a3) | |
| M1-01, M1-09/M1-10 CPU, C11, C13 (ym-color core) | [36515b2](https://github.com/ngriaznov/yellowmoon/commit/36515b2) | |
| M1-05 sensor map in CIE Lab (your decision) | [f402f7d](https://github.com/ngriaznov/yellowmoon/commit/f402f7d) | LED-V2 stays a reported gate failure |
| M0-01 Rust workspace, Tauri shell, CLI stub, data lookup | [fe74202](https://github.com/ngriaznov/yellowmoon/commit/fe74202) | |
| M1-02, M1-03, M1-04 fits, profiles, registry, ALEXA validation | [c10f0e7](https://github.com/ngriaznov/yellowmoon/commit/c10f0e7) | |
| M0-02, M0-03 (repo side), M0-09 (scripts; corpus blocked), M0-10 | [93e91e2](https://github.com/ngriaznov/yellowmoon/commit/93e91e2) | signing key is yours (M0-03); coverage and golden-set tables below |
| M0-01 frontend skeleton | [131a619](https://github.com/ngriaznov/yellowmoon/commit/131a619) | |

CI jobs enabled: frontend, e2e, rust (macOS), deny, deny-selftest, fetch-tools, gpu-lavapipe, oracle, lens-oracle, data-checks, synthetic-set, golden, dng, serde-order, eval-tools, parity (also builds `ym` with the `force-demosaic` feature for the forced-method test), licences, spectral-fixtures, ml-backends, training-dry-run, alexa-curve; nightly: fuzz; weekly: the 500-photo latency check. CI runs on pushes to `main` and on pull requests, so none of them has run on GitHub yet (no PR exists).

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
| Demosaic PSNR, gamma-encoded (M1-12) | 2026-10-12 | Kodak 29.82 / 38.88 / 39.16 dB, McMaster 32.31 / 34.21 / 35.09 dB (bilinear / Menon / LMMSE); reproduced by RA-05 | reported; the R6 rule now uses linear light (next row) |
| Demosaic PSNR, linear light, 9 cells (RA-05, `docs/05` R6) | 2026-10-13 | adaptive ratio 36.05 dB, Menon 34.43, LMMSE 34.28; best in 7 of 9 cells (5 of 9 with the noise model scaled, mean 35.45; the six photographic cells level with Menon and LMMSE); zipper 20.3 against 39.1 % of edge pixels (procedural set); CPU 1.5–1.7× Menon | yes by the R6 rule; Gharbi and PixelShift200 blocked |
| Log-guided within 1 dB of Menon, synthetic set (RA-06) | 2026-10-13 | −0.84 / −0.80 / −0.73 dB at signal scales 1 / 0.1 / 0.01; about half of Menon's CPU time; 3.4–3.6 dB below Menon on Kodak | yes; no render uses it; the 300 ms fit-preview target needs the Mac |
| Diffraction deconvolution against Richardson–Lucy (RA-07, spec 03 §6.3) | 2026-10-13 | ISO 100: MTF50 at f/8 0.340 against 0.380, at f/22 0.196 against 0.180; flat-patch noise ×1.005 (f/8), ×1.577 (f/16), ×1.759 (f/22) against ×1.255, with the noise gate ×1.085 and ×1.080; overshoot 3.0 and 4.6 % against 1.0 and 0.6 % at f/16 and f/22; real files change little | no, as a selectable method; Richardson–Lucy stays the default (RA-H2 open) |
| Output sharpening (RA-08) | 2026-10-13 | edge rise 2.49 → 1.91 px, overshoot 3.2 %, flat noise ×1.030; JPEG q92 +1.27 / +4.29 / +1.70 % at full size; CPU slider move 825 ms (2 MP fit frame), 178 ms (512 px tile at 100 %) | yes; the cost waits for the GPU kernel |
| RA-09 kernels against the CPU reference, lavapipe | 2026-10-13 | adaptive ratio 1.4e-6 on the vectors, 7.4e-6 on random frames, isolated tie flips up to 1.55e-2 of white on real mosaics; log-guided 3.1e-6; diffraction 8.9e-7; output sharpening 7.2e-7 on the vectors, 4.3e-5 on 2048 × 1365 (`docs/05` GPU stages) | yes (lavapipe); Metal is M0-H4 |
| Lateral CA residual after R5b, real files (`docs/05` R9) | 2026-10-13 | rms radial red/blue shift over r ≥ 0.25. `ca_measure.py`, 5 files: 0.03–0.28 px after R5b against 0.06–0.77 for R9 after the demosaic. `ca_xcorr.py` (independent estimator, review), 6 files: 0.11–0.38 px against 0.17–1.22; its noise floor is 0.10–0.15 px. R5b is the lowest or level on every file with either estimator (the A7 III and the A7 IV blue do not gain over no correction at all) | reported; question 11 |
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
| Golden images (M1-15) | 2026-10-13 | pipeline 0.1.2, 61 synthetic files re-baselined at the RA defaults (adaptive ratio demosaic, Output 100 %; the move is ΔE00 median 0.007 to 1.04, maximum 6.6 to 26.3 per file, see Raw algorithms above); afterwards 61 identical with `--jobs 1` and `--jobs 4`. Before the re-baseline (2026-10-12, pipeline 0.1.1): 61 identical on re-run, thread-count independent; a saturation +3 change failed at ΔE00 3.6 | yes (synthetic tier) |
| Demosaic + denoise network (M2) | — | no trained weights (M2-H4) | blocked |
| 500 files, commands < 100 ms (M3-03) | 2026-10-13 | `open_paths` 31 ms, slowest of 5112 commands while developing 62.5 ms (ignored test, local, busy machine); webview keypress on screen in 5–72 ms (median 16), one manual run | yes (local; not run in CI) |
| Performance (docs/05) | 2026-10-12 | Linux x86 only: Lossless L 50 MP decode 333 ms on 4 cores; classical chain ~64–69 CPU-s per 50 MP (with LMMSE; the adaptive ratio default adds about 2); Mac numbers are M0-H4 | reported |

## Waiting on Mykyta

| ID | What | Needed for |
|---|---|---|
| M0-03 | Generate the updater signing key and set the two GitHub secrets; put the public key in `src-tauri/tauri.conf.json` (exact commands in `docs/release-signing.md`). Also decide whether the release bot may push the cask update to `main`. | first release |
| M0-H4 | Mac checks: native viewport (M0-06 checklist), Metal parity and timings (`cargo test -p ym-gpu`, `crates/ym-ml/examples/bench.rs` for the ML runtime choice), decode/render timings, the 0.0.1 → 0.0.2 self-update, `brew install --cask`. Added by the RA wave: run `gpu_adaptive_ratio`, `gpu_log_guided`, `gpu_diffraction` and `gpu_output_sharpen` on Metal (fast-math `log2` and `exp2`, fused multiply-add, the exact-luma tests `the_luma_plane_has_the_cpu_bits` and `luma_has_the_cpu_bits`; the tie-flip bounds were calibrated on lavapipe), and time the Output slider on a retina fit frame. | M0 gate |
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
| RA-H2 | Richardson–Lucy or diffraction deconvolution as the R8 default. Look at the 25 crops in `testdata/ra07` (open `testdata/ra07/index.html`; the folder is local and git-ignored: `cargo run -p ym-gpu --example ra07_ab` rebuilds the synthetic crops and `testdata/ra07/make_index.py` the page). Evidence so far favours Richardson–Lucy at ISO 100: MTF50 at f/8 0.380 against 0.340 on synthetic scenes, and on the real files it sharpens clean edges more (ILCE-6300 f/8: 0.205 against 0.148); at ISO 100 diffraction is better only from f/16, with more overshoot and, ungated, ×1.58 to ×1.90 flat noise. At ISO 6400 the synthetic numbers lean towards diffraction from f/8 (MTF50 0.314 against 0.299, flat noise ×1.011 against ×1.141 at f/8). Also open: which noise gate, and whether to stack the two (`docs/05` R8). | R8 default |
| RA-05 | The Gharbi hard and moiré sets and PixelShift200 for the demosaic gate: on a machine with open internet put them in `testdata/demosaic/{gharbi_hard,gharbi_moire,pixelshift200}` and run `tools/eval/demosaic/gate.sh`. The build environment's proxy refuses their hosts. | demosaic gate |
| RA-07 | Check the sensor widths and pixel counts in `data/cameras/registry.json` against Sony's pages. All came from search excerpts, because the Sony hosts could not be fetched; the five bodies marked `restated` (A7 III, A7 IV, A7C II, A7S III, A1) are the least certain. The ILME-FX3 has none. The A7R V and A7CR fail the LibRaw sibling check and use the A7R IV profile (`docs/03`). | pixel pitch for R8 diffraction |
| M0-H4 (added) | On a Mac: "Move to Trash" for a downloaded look zip goes through Finder (the `trash` crate's default), which may ask for Automation permission the first time (`src-tauri/Info.plist` now carries the usage string). Check that the prompt appears once and the move works. | M3 |
| M4-H2…M4-H5 | Developer ID signing and notarisation decision; trademark clearance (EUIPO, INPI, WIPO, UK IPO); freedom-to-operate check on US 11948279 and on the patent leads in `research/raw-algorithms-prior-art.md` §2.2 (Kodak, Epson, Sony, Apple, IBM families; diffraction restoration filters); announcement. Not started. | M4 |

## Auto exposure and the DNG (your decisions of 2026-10-14)

Research: `research/auto-exposure-2026-10-14.md`. Of A1M01749's −1.16 EV against its camera JPEG, −0.85 EV is where ACES 2 SDR puts mid grey and −0.30 EV is our exposure. Your decisions:

- **Exposure: on hold.** You first chose to keep all of the camera's exposure decision (a fixed camera gain of +1.11 EV, exposure compensation in full), then held it: Yellowmoon has to be clearly more dynamic than a raw processor plus an ARRI-like LUT, and a research workflow on dynamic stages (how much our per-photo colour already differs from the best static LUT, and which per-photo stages to add) runs first. Auto exposure stays as it is until then.
- **Display brightness: on hold.** You first chose +0.57 EV before ACES 2 (ARRI's own renderings keeping ARRI's brightness). Your commercial comparison LUT (Resolve-generated, S-Gamut3.Cine/S-Log3 in, 65³) puts mid grey at 39.5 % video, about where ACES 2 is today (38 %) and where ARRI's Rec.709 rendering is quoted (38 to 40 %); +0.57 would put ours near 46 %. You will choose between +0, +0.35 and +0.57 after side-by-sides of your photos through that LUT and through our chain.
- **The DNG carries everything Yellowmoon does except sharpening.** It opens as bright as the app; its pixels carry no capture sharpening (R8) and no output sharpening, while the demosaic's built-in shadow step stays. It becomes available, approximately, for LogC3-to-Rec.709 looks. The DNG gate keeps comparing with our (sharpened) JPEG.

State: the DNG changes are queued; they start when GPU wave 3 ends, before wave 4. The commercial LUT stays local (`testdata/luts/`, git-ignored) and is never committed; its comparison pictures go only to you.

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
10. **Raw-algorithms constants.** The prior-art record (`research/raw-algorithms-prior-art.md`) traces 55 of 74 pieces to known work, but public sources don't explain some constants (spec 01: 1.6, 2.5, 18 and 102, 121/123, 0.03, 0.75, 5, 3; spec 02: the log2/exp2 cubics; specs 03/04: the constants paragraphs of §3.3/§3.4), and spec 01 §8 reads as if an integer variant exists elsewhere. Where do they come from (tuned by you, fitted, taken from another program)? The answer goes into `docs/14`. Patent leads from the record are added to M4-H4.

11. **Lateral CA on the mosaic (R5b, `docs/05` R9, `docs/16`).** On real Sony files, correcting CA in R9 after the demosaic made the red/blue shift larger (0.04–0.2 px before, 0.3–0.7 px after) because a green-guided demosaic keeps only about half of the CA and R9 then applies the full table. The tables are right; the cause is the order of the stages. The correction now runs on the mosaic before the demosaic, at full resolution only (the binned fit view and thumbnails keep it in R9), and leaves less red/blue misregistration than the old order on every real file measured, with two independent estimators (`docs/05` R9 has the numbers; the amplitude of a table is known to about a factor of two). It changes every real-file render that has a CA table; the synthetic goldens have neutral tables and do not move. Recommended: keep it on by default, as built. Alternative: `decode.chromatic_aberration` off by default. A blind look at a few photos is the natural check.
12. **NEX-5N and NEX-7 lens data.** Their vignetting, distortion and CA tables are only in the encrypted SR2 sub-IFD, which `ym-raw` does not read, so they show "Lens data: none". Reading them is a few lines and switches all three corrections on for NEX files (the NEX-5N's CA table is the best fit of the files tested, rms 0.16 px). Read them?

13. **GPU wave, question 1 (`docs/gpu-plan.md` section 9).** f32 as the default device storage (STATUS question 2, option 1). Cost: twice the plane memory (50 MP spike peak 0.28 to 0.56 GiB, gate 1.5 GiB); nothing in the RA kernels. The wave proceeds with f32 because it is the accurate choice and the RA kernels already require it.
14. **GPU wave, question 2 (`docs/gpu-plan.md` section 9).** Gate `Auto` per stage: a stage is on in `Auto` once its suites pass on the Metal CI runners (`mac-gpu.yml`, GitHub's Apple-silicon VMs), so nothing unverified on Metal reaches users, and the physical-Mac session (M0-H4) judges speed and the screen? Alternatives: physical-Mac sign-off only (stricter, the wave stays invisible in the app until M0-H4), or every stage on in `Auto` from the first merge.
15. **GPU wave, question 3 (`docs/gpu-plan.md` section 9).** A GPU render may differ from the CPU reference within the stated tolerances; `PIPELINE_VERSION` does not change and the device is recorded nowhere in a file (AGENTS rule 8 is read as "same machine and device"; the 2026-10-13 direction already says so). Confirm. The alternative (a version or XMP tag per device) would break app = CLI bytes across machines.
16. **GPU wave, question 4 (`docs/gpu-plan.md` section 9).** Unified-memory budget on Apple silicon: GPU = clamp(RAM/4, 512 MiB, 4 GiB), CPU budget lowered by the same, sum at most 0.6 x RAM. Acceptable on 8 GB Macs? *As built (G-03, pending your answer):* the GPU gets `clamp(RAM/4, 512 MiB, 4 GiB)` and the CPU `min(its budget, 0.6 x RAM - GPU)` (8 GB: 2 GiB and 2.8 GiB; 16 GB: 4 and 5.6 GiB; 64 GB: 4 and 6 GiB). The first rule took the whole GPU amount off the CPU and cut every Apple-silicon Mac to 2 GiB of CPU while only the colour pass used the device.
17. **GPU wave, question 5 (`docs/gpu-plan.md` section 9).** `Auto` uses the CPU when the only adapter is a software one (lavapipe, WARP). Agreed?
18. **GPU wave, question 6 (`docs/gpu-plan.md` section 9).** Additive `device` and `notes` in `ym render --json`; no "Use graphics card" preference (the persistence list in `docs/01` is exhaustive; development uses `YELLOWMOON_GPU`).
19. **GPU wave, question 7 (`docs/gpu-plan.md` section 9).** The CPU-only list of section 8 (histogram counts, the R6 test-switch methods, Lossless M/S, the DNG white, the full-chain R5 offsets) is acceptable for this wave.
20. **GPU wave, question 8 (`docs/gpu-plan.md` section 9).** Is the 6 s export target a JPEG-only target (`docs/05` says "Full neural export, JPEG")? TIFF and PNG 16-bit encode alone are about twice that on the dev box. A GPU JPEG front end is out of scope; it is revisited only if the Mac shows DCT + Huffman above about 2 s per 50 MP.
21. **One C++ file in the repository.** `tools/dng/dng_jxl_stub.cpp` (40 lines) is a stand-in that CI compiles with Adobe's DNG SDK to build `dng_validate`; it is unrelated to the raw algorithms. Keep it, or have the CI script write it at build time so the repository holds no C++?

## Known issues

- C9 grade controls can fold the neutral ramp when two are combined at opposite extremes (e.g. Highlights +100 with White −100); each alone is monotone. Needs a UI limit or a constants refit in M3.
- Day/Shade/Tungsten presets carry the 3.5–6 % difference between the Weta curves' white and the camera ColorMatrix white (A7 III/A7 IV); a per-body raw-white calibration would need data.
- Noise estimator over-reads `a` on busy low-ISO scenes (median 14 %, p90 38 %); borderline against the 15 % gate.
- The illuminant's synthetic brightness range stops at Bv 13; real sunlit files read 14.3–14.6. LED and fluorescent class recognition is weak on independent chart scenes.
- Kelvin readout rounded to 50 K leaves up to 1.6 % red/blue imbalance below 2300 K.
- The GPU memory budget isn't scaled to system RAM; device loss isn't handled.
- The raw chain runs on the CPU. GPU kernels exist for R1 + R4, the colour chain, and (RA-09, not wired) the adaptive ratio demosaic, the log-guided demosaic, diffraction deconvolution and output sharpening; R3, R5, R5b, the bilateral denoise, R7, Richardson–Lucy and R9 have none, and nothing in the raw chain calls a kernel. The fit-preview and 100 % performance targets depend on them (and on a Mac, M0-H4). The noise σ map (M2-02 "on GPU") is CPU only too. The owner's GPU direction of 2026-10-13 (`docs/16`) puts every stage on the GPU next.
- The GPU adaptive ratio demosaic is not bit-identical to the CPU port (f32 against f64 where a direction test ties): on real mosaics 1.65e-4 of the samples differ by more than 1e-4, worst 1.55e-2 of white. A GPU export will differ from the CPU fallback by such isolated samples.
- Output sharpening runs on the CPU in the viewport: it cannot be cancelled once started (0.5 to 2.6 s per frame on a retina fit frame) and re-runs after every edit above it. Filmstrip thumbnails do not follow the Output slider, and a 100 % view differs from the export within about 21 px of a crop edge (the view reads photo pixels beyond the crop, the export mirrors at it).
- The colour plan (`pipeline/color.rs`) still reads the photo's raw noise model, so a finite but absurd read-noise variance (`b` of 1e300) blackens the frame there. The estimator never produces one; the raw group already reads such a model as zero noise.
- A cropped photo in a cell larger than the crop shows uncropped photo content in the letterbox of the viewport (also with Output at 0). Check against `docs/11`.

## Environment

Linux x86_64 container, 4 vCPUs, Rust 1.97.0, Node 22, Python 3.13; Mesa lavapipe for wgpu; darktable-cli 4.6.1 (lens oracle, separate process); LibRaw 0.22.1 via rawpy, rawler 0.8 and libjpeg as decoder oracles; OpenColorIO 2.6 for test vectors; g++ (C++17) for the owner's C++ references of the raw algorithms, used outside the repository only (none of that C++ is committed; the one C++ file in the repository is `tools/dng/dng_jxl_stub.cpp`, a 40-line stand-in that CI uses to build Adobe's `dng_validate`, unrelated to the algorithms and older than the decision); Adobe DNG SDK 1.7.1 `dng_validate` built from the GrapheneOS AOSP mirror. No Mac: every Metal, native-viewport, timing and release check is listed under M0-H4.

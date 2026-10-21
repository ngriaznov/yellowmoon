# 15 · Milestones and tasks

Five milestones run in sequence; each ends at a gate that can be measured. Work in task order within a milestone unless dependencies allow parallel work. The UI design is in `design/`; M0 and M1 don't wait for UI work.

Task format: **ID · title** (depends on) — what to build. *Done when:* acceptance criteria. [HUMAN] tasks are the owner's (`AGENTS.md` rule 9): prepare around them and list them in `docs/STATUS.md`.

**Environment note.** Most crates build and test on Linux and macOS. GPU kernels are tested on Linux CI with Mesa lavapipe (software Vulkan) for correctness. The app shell, the native viewport, Metal-specific behaviour and performance need a Mac; if you aren't running on one, say so in `STATUS.md` and leave those checks to [HUMAN M0-H4].

**V1 needs no test shoots.** Every gate uses public data, generated data or independent tools (`docs/test-photo-sources.md`, `docs/13`). Controlled captures (grey cards, flat fields, calibration frames, pixel shift) are V2 work, listed at the end.

---

## M0 · Foundations

Goal: a skeleton that builds, tests and releases; ARW files decode identically to the oracles; the three big technical unknowns (native viewport, GPU tiling, ML runtime) are answered.

**M0-01 · Workspace skeleton** ()
Cargo workspace with the crates and package names in `docs/01` (empty APIs with doc comments; explicit member list), `rust-toolchain.toml`, `src-tauri` (Tauri 2, plugins updater/process/dialog, config from `templates/tauri.conf.json` with placeholders), `tools/dev/prepare-sidecar.sh` (builds `ym-cli`, copies it to `src-tauri/binaries/yellowmoon-cli-<host triple>`; wired into `beforeDevCommand` and CI), React 19 + TS 5.8 + Vite 7 frontend with Vitest and Playwright as in plugout, `src/styles/tokens.css` from `design/`, Figtree fonts, `api.ts` + `api.mock.ts`, the `data/` lookup order (`docs/01`), `docs/STATUS.md`.
*Done when:* on a clean checkout, `tools/dev/prepare-sidecar.sh && cargo test --workspace`, `npm test` and `npm run build` pass; `npm run tauri dev` on a Mac opens a window showing a placeholder Start screen in Figtree with the moon mark.

**M0-02 · CI and licence policy** (M0-01)
Install `templates/github/workflows/ci.yml` and `templates/deny.toml`; add `CHANGELOG.md`, issue templates (bug report asks for `ym doctor` output).
*Done when:* CI is green on a PR; a test PR adding a GPL crate fails `cargo deny`.

**M0-03 · Release pipeline and updater** (M0-02)
Generate the updater signing key and set the secrets (`docs/12` §Signing); install `release.yml` and `Casks/yellowmoon.rb` for the project repository; wire `updater.ts` (from plugout) and the update cell placeholder; sidecar build step with a stub `ym-cli`.
*Done when:* merging a version bump publishes `v0.0.1` with dmg, tarball, signature and `latest.json`, updates the cask; an installed 0.0.1 updates itself to 0.0.2 [checked on a Mac]; `brew install --cask` from the tap installs app and `ym --version`.

**M0-04 · ARW container and metadata** (M0-01, M0-09)
`ym-raw`: TIFF/IFD walker, EXIF, Sony maker note (model ID 0xB001, ISO), SR2 decryption and tags (0x7310/0x7300, 0x787F, 0x7313/0x7303), lens tags (0x7032/0x7035/0x7037), tone curve (0x7010), embedded preview location, compression and bit depth detection, `RawMetadata`. `ym info` (text and `--json`, metadata only).
*Done when:* on the corpus, metadata fields (make, model, model ID, ISO, exposure, dimensions, compression, black, white, WB) equal LibRaw/ExifTool values; fuzz targets run 10 min without panics.

**M0-05 · Decoders and oracle CI** (M0-04, M0-09)
Uncompressed 12- and 14-bit, cRAW (undithered, with `qstep`), lossless L (own LJPEG-92, parallel tiles). Lossless M/S and ARW 6.0 detected and reported as unsupported for now. `tools/oracle` (rawler, test-only) and a rawpy (LibRaw) script; enable the `oracle` CI job.
*Done when:* every corpus file of these variants decodes to values identical to LibRaw (rawler: within its dither for cRAW); decode timings recorded against `docs/04` targets.

**M0-06 · Native viewport spike** (M0-01)
Metal layer behind a transparent webview in the Tauri window; rect sync from a `<div data-viewport>`; pan/zoom/click forwarding; Display P3 surface; frame timing overlay; resize, fullscreen, Spaces, moving between P3 and sRGB displays.
*Done when:* the manual checklist in `docs/11` passes on a Mac and input latency ≤ 1 frame; or the fallback (native child view) is chosen with reasons. Decision recorded in `docs/16`.

**M0-07 · GPU tiling spike** (M0-05)
`ym-gpu`: device ownership on one thread, priority queue, tile scheduler with halos, f16 storage, buffer pool. A trivial chain (normalise → bilinear demosaic → fixed matrix → sRGB) through `ym-cli render --format png` on a 50 MP lossless file. CPU reference of the same chain. Lavapipe CI job for kernel parity.
*Done when:* GPU output equals the CPU reference within 1e-3; peak GPU memory under 1.5 GiB for 50 MP; timings on M1/M3 recorded.

**M0-08 · ML runtime benchmark** (M0-07)
Random-weight `jdd` of the starting architecture (`docs/07`) exported to ONNX; run on 512×512 half-res tiles through burn (wgpu, shared device) and ort (CoreML and CPU).
*Done when:* time per tile, full-frame time, memory and parity for both are in `docs/16`, and the runtime is chosen by the rule in `docs/07`.

**M0-09 · Test data fetching** (M0-01)
`tools/fetch/samples.sh` + `manifest.json`: CC0 files from RawDB and raw.pixls.us covering every Sony variant and as many bodies as possible (at least: A7R II, A7R III, A7R IV, A7R V, A7 III, A7 IV, A7 V, A7R VI, A7S II, A7S III, A9, A9 III, A1, A1 II, A6400, A6700, RX100 IV, ZV-E1, FX3), with per-file licence check. Build the golden set from public sources per `docs/test-photo-sources.md`: the CC0 tier, then the Play Raw tier (search, licence check, attribution), and a list of the evaluation-only files for local use.
*Done when:* the script downloads and verifies hashes; CI caches the result; the coverage table (variant × body) and the golden-set table (file, body, light, ISO, licence tier) are in `docs/STATUS.md` with gaps listed.

**M0-10 · Validation data plan** (M0-09)
`docs/validation-data.md`: for every gate in `docs/13`, the public dataset or oracle that replaces a test shoot (V1 has none), with licence, download script and what may be stored where.
*Done when:* every gate has a data source; nothing in the plan needs a test shoot.

**[HUMAN] M0-H4** Run the Mac-only checks (M0-06, M0-07 timings) if the agent isn't on a Mac. The agent generates the updater signing key, and the golden set comes from public sources.

**Gate M0:** decoders match oracles for all V1 variants except M/S and ARW 6.0; viewport approach decided; 50 MP tiles render within memory; ML runtime chosen; a release has updated itself.

---

## M1 · Colour MVP

Goal: correct ALEXA-style colour end to end through the CLI, with the classical demosaic. JPEG/TIFF/PNG export.

**M1-01 · Colour constants and ARRI step** (M0-01)
`ym-color`: LogC3/LogC4, AWG3/AWG4, NPM from primaries, CAT02, AWG→AP0, ARRI 17-CCT table loading and bracketing-pair mired interpolation, native→AWG3.
*Done when:* every check in `docs/02` §Constants and §Tests passes (to 1e-9 / 1e-5).

**M1-02 · Spectral engine** (M1-01)
Load rawtoaces-format curves; fit scripts precompute `w_A` and camera-locus tables (blackbody, CIE daylight, and the class reference lights) into `profile.json`; runtime interpolation and RGB ↔ CCT/Duv in `ym-color`. The `colorimetry` crate is a dev-dependency only, for cross-checks.
*Done when:* runtime values match `training/alexa/ssf_core.py` to 1e-6 on a test grid; `cargo tree -e normal` shows no `colorimetry` in shipped crates.

**M1-03 · ALEXA curve validation in CI** (M0-02)
Run `training/alexa/validate_loo.py` in a Python CI job with pinned colour-science 0.4.7; fetch sources by script.
*Done when:* the job reproduces the published LOO numbers (median RFSE 2.2%, emulation ΔE00 median 0.23) within ±0.1.

**M1-04 · Profiles, registry and detection** (M1-02, M0-04)
`training/sensor_map/fit.py` (port of `fit_map`) for the 11 bodies and 4 light classes; `profile.json` schema and files; `data/cameras/registry.json` with the model-ID table in `docs/03`; sibling ColorMatrix checks; `ym-profiles` lookup; `CameraMatch` in `ym-raw`.
*Done when:* detection gate passes on the corpus; the Rust evaluation reproduces `sony_profiles.csv` "own" numbers ±0.02.

**M1-05 · Sensor map runtime and gate** (M1-04)
Class-blended, mired-interpolated map; optional MLP experiment.
*Done when:* sensor-map gate (`docs/13`) passes for the A7R IV profile and is reported for all 11.

**M1-06 · Illuminant estimator** (M1-02)
Synthetic training data, Cube++ converted per profile, FFCC-style filter, class head, as-shot prior; training in `training/illum/`; Rust inference in `ym-color`.
*Done when:* illuminant gate passes; the sensor-map gate still passes with estimated lights.

**M1-07 · Fused colour chain** (M1-05, M1-06)
C1–C6 as one matrix per photo; Auto WB, Kelvin/tint, light presets, Pick white maths; per-photo `AutoEstimates`.
*Done when:* unit tests for each mode; "3200 K under daylight" renders warmer than Auto by the amount the ALEXA maths predicts (test against a CPU spectral simulation).

**M1-08 · Auto exposure** (M1-07)
Algorithm in `docs/02`; burst consistency.
*Done when:* on the golden set, auto exposure lands within ±0.5 EV of the camera's own exposure for ≥ 80% of photos, measured with G3: our `EV_auto` minus the camera engine's gain, where the gain is `log2(0.18 / x)` and `x` is the scene-linear level the embedded JPEG renders as mid grey (quantile match of the JPEG against our scene-linear render at the JPEG's size; `tools/eval/autoexp.py`). This keeps the display transform and the camera's tone curve out of the exposure check. Outliers are listed for the owner's review (M1-H5).

**M1-09 · Tone stages** (M1-07, M0-07)
C7 roll-off and toe, C8 LogC4, C9 primary grade; CPU + WGSL.
*Done when:* GPU/CPU parity ≤ 2e-4; monotonicity tests for C7 and every C9 control.

**M1-10 · Looks engine** (M1-09)
CDL, `.cube` parser (fuzzed), tetrahedral interpolation on CPU and GPU, intensity blend, LogC3 path, Rec.709-baked path; built-in Neutral; `ym looks import <.cube> --input …` and `looks list`; library folder and manifest with locking.
*Done when:* LUT parity gate vs OCIO vectors; importer tests incl. concurrent CLI imports.

**M1-11 · ACES 2 display transform** (M1-01)
Port the output transform (CPU reference, then WGSL) for the four outputs in `docs/02`; ARRI display-LUT path.
*Done when:* DRT gate vs OCIO vectors (≤ 1 code value at 10 bits).

**M1-12 · Classical raw processing** (M0-07)
R1 normalise and clip mask, R3 vignetting, R4 gains, R5 highlight reconstruction, R6 classical demosaic (Menon or LMMSE, pick by PSNR) + light denoise, R8 RL sharpening, R9 lens resample + auto crop; CPU + WGSL; a `lens-oracle` CI job that installs darktable on Ubuntu and compares with `darktable-cli`. *Current defaults (`docs/05`, `docs/algorithms/`): R6 is the adaptive ratio demosaic (Menon, LMMSE, bilinear and log-guided stay selectable through a test-only switch) followed by the single-frame denoise; R8 is Richardson–Lucy (diffraction deconvolution is available); output sharpening is a per-photo setting, default 100 %.*
*Done when:* highlight gate; lens-correction gate against darktable-cli and the image checks (`docs/05`); demosaic PSNR reported on Kodak/McMaster.

**M1-13 · Export writers** (M1-11)
Display encode with dither; JPEG/TIFF/PNG; ICC; EXIF copy; XMP `ym:` namespace; resize; naming templates; conflict handling.
*Done when:* `docs/09` tests for these formats pass.

**M1-14 · CLI v0** (M1-13)
`render` and `info` with the core flags, `--json`, exit codes, stdin input, headless GPU with CPU fallback, `doctor`.
*Done when:* `docs/10` tests pass for implemented flags; renders the golden set.

**M1-15 · Golden images** (M1-14, M0-09)
`tools/golden` runner, `tools/fetch/golden.sh` (CC0 and free-licence tiers), references, CI job.
*Done when:* CI fails on an unexplained ΔE00 > 0.5 change.

**[HUMAN] M1-H3** Email HdM about a licence for the SXT curve (validation). **M1-H4** Decide on CIE-derived coefficient tables (`docs/14`). **M1-H5** Visual review of the golden set: colour, exposure, white balance. (V1 has no test shoots; the captures are V2 work, below.)

**Gate M1:** ALEXA curve, sensor map, illuminant, ARRI step, DRT, LUT, highlights and detection gates pass; the CLI renders the golden set; the owner's review accepts the colour direction.

---

## M2 · Raw quality

Goal: Capture One–level demosaic and noise; every V1 format decodes.

**M2-01 · Noise priors from public files** (M0-09, M1-12)
`training/noise/public_priors.py` per `docs/06`: run the prior-free estimator over the CC0 and free-licence corpus, fit per-body `log a` vs `log ISO` and `b` per regime where enough ISOs exist → `data/cameras/noise/<model-id>-<model>.json` and `gain-regimes.json`.
*Done when:* fits reported with residual plots for each qualifying body; bodies without enough data listed.

**M2-02 · Noise estimator** (M2-01, M1-12)
`ym-noise` per `docs/06`: smooth-region bins, clip-aware MAP fit, texture fallback, refinement, row noise, filtered-file check, priors, batch pooling, σ map on GPU.
*Done when:* noise-estimate gates in `docs/06` pass.

**M2-03 · cRAW test encoder and quantisation map** (M0-05)
Test-only encoder; property tests for `qstep`.
*Done when:* every true value lies within ±qstep/2 of the decode on synthetic images.

**M2-04 · Training data pipeline** (M2-03)
PixelShift200 loader, synthetic patterns, the sample generator in `docs/07` (virtual-camera noise, quantisation, clipping, σ jitter).
*Done when:* generator unit tests; a visual sample sheet in the PR.

**M2-05 · `jdd` training code** (M2-04)
Model, loss, training loop with seeds and config, evaluation scripts for the gate sets, ONNX and safetensors export, receptive-field calculator (sets the tile halo), model card generator; enable the `training-dry-run` CI job.
*Done when:* dry run passes in CI; a short CPU run overfits a tiny set.

**M2-06 · Training runs and model selection** (M2-05, M2-H4)
Run on a GPU machine; pick the smallest model that meets the gate; write the model card.
*Done when:* demosaic gate (`docs/07`, `docs/13`) passes; weights and card committed.

**M2-07 · Runtime integration** (M2-06, M0-08)
`ym-ml` backend, tiles with halos, σ/qstep/clip inputs, Detail controls (R7), parity tests.
*Done when:* parity ≤ 2e-3; export of 50 MP within `docs/05` targets on M1.

**M2-08 · Preview strategy** (M2-07)
Classical fit view, background neural tiles at 1:1, neural export; status text shows the path.
*Done when:* colour-only edits stay < 16 ms; 1:1 tiles appear within targets.

**M2-09 · Lossless M/S path** (M0-05)
YCbCr decode, chroma upsampling, inverse transform, undo WB, reduced pipeline, REDUCED tag.
*Done when:* oracle comparison on RawDB M/S sets; renders without artefacts in review.

**M2-10 · ARW 6.0 decoder (gated)** (M0-05, M2-H3)
Behind cargo feature `arw6`; lightcraft as reference/dependency; LibRaw master as oracle; quantisation map.
*Done when:* oracle match on all public ARW 6.0 samples; the feature is enabled in release builds only after the Sony gate clears.

**M2-11 · Performance pass** (M2-08)
Profile and fix against `docs/05` targets on M1 (8 GB) and an Intel Mac.
*Done when:* targets met or the gaps documented with a plan.

**[HUMAN] M2-H3** Email Sony about RDD 34-based decoding. **M2-H4** Provide GPU compute and run training. **M2-H5** Blind A/B of the A1 ISO ladder (evaluation-tier studio files) vs Capture One and DxO. (Calibration frames and pixel-shift captures are V2 work, below.)

**Gate M2:** demosaic, noise and ISO-ladder gates pass; all formats decode (ARW 6.0 pending Sony if needed).

---

## M3 · App, looks and UI

Goal: the app as mocked, with the wizard, DNG export and full CLI parity.

**M3-01 · Component library** (M0-01)
All components in `docs/11` from tokens, with Vitest tests.
*Done when:* a Storybook-free `components.html` dev page shows every component state; tests pass.

**M3-02 · IPC layer** (M3-01, M1-14)
Commands and events from `docs/01` via specta/tauri-specta; `api.mock.ts` with fixture photos (pre-rendered PNGs).
*Done when:* type generation runs in CI and fails on drift.

**M3-03 · Session in the app** (M3-02)
Open files/folders, drag and drop, progress events, thumbnails, errors per file, ⌘W.
*Done when:* opening 500 files keeps the UI responsive (< 100 ms input latency).

**M3-04 · Viewport integration** (M0-06, M2-08)
Production version of the spike: fit/zoom/pan, 3:2 cell, before mode, pixel readout, white pick clicks, crop overlay drawing.
*Done when:* `docs/11` viewport checklist passes on a Mac.

**M3-05 · Start screen** (M3-03)
As mocked, all states.
*Done when:* Playwright flow and screenshot comparison pass.

**M3-06 · Photos screen and Look tab** (M3-03, M1-10)
Header, filmstrip with picking, tab bar, Automatic card, look grid with lazy thumbnails, search, keys 1–9, Apply to, DEFAULT.
*Done when:* Playwright flow and screenshots pass.

**M3-07 · Other tabs** (M3-06)
Light, Colour (wheels), Detail, Curve (plot and histogram), RAW (camera card, switches, roll-off); undo/redo.
*Done when:* every control round-trips through `update_settings`; screenshots pass.

**M3-08 · Crop and Pick white** (M3-04)
*Done when:* both work by mouse and keyboard; crop is written into exports and DNG crop tags.

**M3-09 · ARRI import wizard** (M1-10, M3-H2)
Recogniser from `data/looks/arri-packages.json`, `tools/looks/fingerprint.py`, Downloads watcher, five-step UI, generic `.cube` question, `looks import/verify/default` CLI parity.
*Done when:* wizard happy path and failure paths (wrong package, damaged look file, missing file) tested with synthetic packages; the owner imports the real packages successfully [M3-H4]. ARRI Look Files are read too (`docs/08-looks.md`).

**M3-10 · Signature look** (M1-10, M3-H1)
`looks bake` tool and contact sheets; ship `signature.json` once designed.
*Done when:* the owner signs off the look.

**M3-11 · Export panel and queue** (M1-13, M3-06)
As mocked: options, estimate, CLI preview with Copy, queue states, cancel, Show in Finder, export-aware updater.
*Done when:* Playwright flow; exports keep running while browsing.

**M3-12 · DNG export** (M3-11)
Writer on `tiff`, `tools/dng/build_dng_validate.sh` (builds `dng_validate` from the DNG SDK at a pinned commit of the AOSP mirror), profile fit tool (`tools/dngfit` with `dng_validate`), BaselineExposure per photo, preview JPEG, tags; enable the `dng` CI job.
*Done when:* DNG gates (`docs/09`) pass.

**M3-13 · Preferences, menus, About, CLI install** (M3-06)
*Done when:* all menu items work; preferences persist; "Install Command-Line Tool…" creates a working link.

**M3-14 · Full CLI and parity** (M3-11, M3-12)
Every flag in `docs/10`; app/CLI byte-identical test.
*Done when:* parity test passes for every format.

**M3-15 · End-to-end tests** (M3-05…M3-14)
Playwright flows and screenshot comparisons for every mocked state.
*Done when:* CI green.

**M3-16 · Desktop feel** (M3-04, M3-06)
A two-finger pan or a pinch over the photo moves or zooms only the photo; nothing else on the page scrolls, bounces or zooms. Text is not selectable except fields, the command line, paths, errors, file names, version and legal text. Buttons have the arrow cursor and links the hand; WebKit's page menu (Reload, Back) never opens. `docs/11` §Photos input and desktop feel.
*Done when:* `src/styles/desktop.node.test.ts`, the Vitest specs (`usePhotoGestures`, `desktopFeel`, `PhotoView`) and `e2e/desktop-feel.spec.ts` pass; on a Mac a two-finger pan at Fit and zoomed, over the header, the filmstrip, the panel's end stops and a dialog moves only the intended content, a pinch zooms the photo about the fingers, and nothing selects by drag, ⌘A, double click on the photo or triple click in the header [M0-H4].

**M3-17 · Viewport progress** (M3-04)
The core's viewport finishes and shows the frame it is drawing before it starts on newer input, so a gesture that outruns one render still shows the photo moving: only the speculative sharp-tile pass gives way to a request (`docs/01` §Native viewport). Shell side.
*Done when:* the `viewport.rs` tests (a frame in flight is not cancelled by a request, the tile pass is; a new photo cancels both) and the smoke test (200 `invalidate()` calls 2 ms apart present more than one frame) pass; on a Mac a scripted slider drag on every tool tab shows the photo updating continuously, with `cancelled/presented` in the log [M0-H4].

**M3-18 · Render fan-out** (M3-06, M3-17)
A frame drawn by the viewport, a wheel zoom or a tile of an export redraws only what shows it, not the Photos screen and the hundreds of look tiles under it. Wheel, pinch and drag input goes to the core once per animation frame, in order, one command at a time. The histogram and the middle readout are asked for at most five times a second while the Light or Curve tab shows them. `docs/11` §Photos input and desktop feel.
*Done when:* `Photos.render.test.tsx` (50 `render_stats`, 20 zoom steps and an export run leave the screen, the Look tab and the tiles alone), the `ViewportInputQueue` and store specs (100 wheel events and one frame are one `viewport_input`; the histogram throttle) and the Playwright spec (60 wheel events are at most two pans with the summed distance) pass.

**M3-19 · Look tiles** (M3-06, M3-18)
The Look tab with hundreds of looks stays quick: one `IntersectionObserver` for the grid, only the tiles in view refresh when the settings change (and 400 ms after they stop changing), the others keep their picture and ask again when they come back; at most two `look_thumbnail` requests are out at once, the newest first, and the ones not started when the settings change are dropped as superseded, not failed. Folded sections draw no tiles.
*Done when:* `LookTab.tiles.test.tsx`, the limiter specs in `store.viewport.test.ts` and `Photos.render.test.tsx` pass; on a Mac a slider drag with the ARRI library imported keeps the photo updating continuously.

**M3-20 · Shell main thread and perf logging** (M3-02, M3-17)
`ym-frame` answers off the main thread; `render_stats` and `photo_progress` events are coalesced in the emitter; a main-thread watchdog and the viewport counters (`requests`, `presented`, `cancelled`, input to present) go to the local log with `YM_PERF=1` and in debug builds. The page measures itself with `?perf`, `localStorage.ymPerf` or, started by the shell with `YM_PERF`, `window.__YM_PERF__`: late animation frames, renders of Photos, the Look tab and its tiles, store notifications per second, and the calls and round trips of the api in `window.__ymPerf.snapshot()` (`src/perf.ts`). Nothing is sent anywhere (AGENTS rule 7). Shell side: `docs/01`, `docs/13`.
*Done when:* the emitter and protocol tests pass, `src/perf.test.ts` and the `?perf` Playwright test pass, and the Mac job prints the log lines in its summary [M0-H4]. The numbers are not a gate yet.

**[HUMAN] M3-H1** Design the Signature look and name it. **M3-H2** Download the five ARRI packages by hand and run the fingerprint script. **M3-H3** Email ARRI about in-app download (optional). **M3-H4** Review the app on a Mac against the mockups; import the real ARRI packages. **M3-H5** Use the app for one real shoot and report friction. **M3-H6** Check DNGs in Lightroom, RawTherapee, darktable, Capture One and Apple Photos.

**Gate M3:** every mocked state implemented; app/CLI parity; DNG gates; the owner's review.

---

## M4 · Beta to 1.0

**M4-01 · Betas** 0.9.x releases to a small group; local panic log; triage. *Done when:* no open crash or data-loss bugs.
**M4-02 · Hardening** 8 GB Macs, Intel Macs, 2000-file folders, corrupt files, disk-full during export. *Done when:* each case handled with a clear message.
**M4-03 · Documentation** README (public), user guide in `docs/user/`, CLI man page (`clap_mangen`), DNG notice. *Done when:* reviewed by the owner.
**M4-04 · Accessibility** VoiceOver labels, focus order, keyboard-only use, contrast check of tab colours. *Done when:* checklist passes.
**M4-05 · Fuzzing campaign** 8 hours per target. *Done when:* no open findings.
**M4-06 · Release 1.0.0** *Done when:* gate passes and the release pipeline publishes it.

**[HUMAN] M4-H1** Blind A/B of 50 A1 photos vs Capture One, Lightroom and DxO. **M4-H2** Decide on Apple Developer ID signing and notarisation; provide certificates if yes. (Decided: no; the app ships unsigned.) **M4-H3** Trademark clearance (EUIPO, INPI, WIPO, UK IPO). **M4-H4** Freedom-to-operate check on US 11948279. **M4-H5** Announce.

**Gate M4 (1.0):** whole-app blind A/B ≥ 50% preferred or tied; all earlier gates still pass; legal items closed or explicitly accepted by the owner.

---

## V2 backlog: controlled captures

Not part of V1. When the owner is ready to shoot, these improve accuracy for the A1:

- Grey-card shots under daylight, shade, tungsten, LED and fluorescent (illuminant-estimator truth on the target body).
- Flat fields and lens test shots per lens (vignetting and distortion truth).
- Dark and flat calibration frames per ISO (A1 noise truth; replaces the public-data prior).
- 4-shot pixel-shift captures of 100+ detailed scenes (demosaic training and gate on the target body).
- Spectral measurement of the A1 (its own colour profile).

The agent writes a capture plan with exact shot lists when V2 starts.

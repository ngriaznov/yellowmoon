# 15 · Milestones and tasks

Five milestones run in sequence; each ends at a gate that can be measured. Milestone N (nodes and masks, released as 0.2.0) was added on 2026-11-01 and sits between M3 and M4; its full spec is `docs/23-nodes.md`. Milestone F (the Film node, `docs/24-film.md`) is planned and not scheduled; it would follow N. Milestone D (the Depth mask, `docs/23` §4.8, `docs/algorithms/06` §5.8 and §8) was added on 2026-11-07 on the owner's decision (`docs/STATUS.md` question 41) and sits after F on its own branch, `depth-mask`; it needs the masks of N and no part of F. Milestone X (the mixers: Parallel, Layer, and Splitter and combiner, `docs/25-mixers.md`) was added on 2026-11-10 on the owner's request (`docs/STATUS.md` question 43), on its own branch, `mixers`; it needs the graph, plan and panel of N and no part of F or D. Work in task order within a milestone unless dependencies allow parallel work. The UI design is in `design/`; M0 and M1 don't wait for UI work.

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
*Later:* the ACES 2 transform was replaced by the cine rendering (`docs/02` C12, pipeline 0.8.0); the DRT gate is now the cine fixture and the kernel parity test (`docs/13`).

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

## N · Nodes and masks (0.2.0)

Goal: a node view behind a Nodes button, with a real graph of corrector nodes between a fixed Input and Output, masks in every corrector (shapes, ranges, Brush, and AI masks: Object from SAM 2.1 small, Subject and People from Apple Vision, Background from the blur map), and Pick white moved into the Colour tab. The spec is `docs/23-nodes.md` and the maths `docs/algorithms/06-masks.md`; the owner approved the plan on 2026-11-01. The work happens on one branch, `nodes`, from `main`: one reviewed commit per task with its checks green; at each checkpoint the owner gets a test DMG (checkpoints are not releases); when the whole plan is built and the last test build is approved, the branch is merged and released once as **0.2.0**. `PIPELINE_VERSION` goes to 0.9.0 (the curve moves in front of the look). Phases: B core (N-02, N-02b, N-03 to N-05; the settings become node-first first, with no visible change), C interface (N-06 to N-09), D masks (N-10 to N-12), E raster masks (N-13 to N-15), F SAM 2 (N-16 to N-18), G Vision (N-19), H finish (N-20). B before C before D; E and F alongside once D is done; G any time after E.

**N-01 · Spec** () — `docs/23-nodes.md`, `docs/algorithms/06-masks.md`, the changes to other specs, `design/nodes/README.md`, `research/nodes-masks-2026-11-01.md`. **Done 2026-11-01 (the owner approved the spec, N-H1).** *Done when:* the owner approves it (N-H1).
**N-02 · Node-first settings in the core, the CLI and the shell's Rust side** (N-01) — `DevelopSettings` becomes the graph (Input, Main, Output and any other nodes, links): the model, validation with plain errors, Auto only in Main, node patches, reset, copy rules, resolved settings and XMP, the schema (node-first only), the refusal of the flat form of 0.1; **the CLI follows**: flags as node patches, node-first `--settings` and JSON, the help and its snapshots, `docs/10`, the docs test (`crates/ym-cli/tests/docs.rs`); the golden manifest, test fixtures, CLI tests and schema test rewritten to node-first. **The shell's Rust side follows** (wire types, `bindings.ts`, the commands and their Rust tests), so `cargo test --workspace` is green at the end; `npm test` and `tsc -b` fail on the contract until N-02b. *Done when:* every existing render and export is byte-identical, the app and `ym` agree, the refusal and the node-patch tests pass, and the CLI's help, its snapshots and `docs/10` match.
**N-02b · Node-first settings in the frontend** (N-02) — `types.ts`, `contract.ts` and the mock (the wire types and `bindings.ts` are N-02's); the store with node patches and per-node undo; every tab through the selectors (`inputOf`, `mainOf`, `outputOf`). *Done when:* nothing visible changes: every unit test, e2e test and reference screenshot is as before.
**N-03 · The plan and the CPU reference** (N-02) — `ym_color::plan` (live and dead nodes, order, registers, passes), nodes in plan order, trims, bypass, the curve before the look, identity skipping, `PIPELINE_VERSION` 0.9.0 (also in `ym --version`, `ym doctor` and their snapshots). *Done when:* the default graph is byte-identical to 0.8.2 (the curve-and-look case has its own test) and the plan and node tests pass.
**N-04 · The GPU plan loop** (N-03) — the node storage buffer and plan loop in the fused colour kernel, per-node curves and pins, the multi-pass split, DNG bakes nodes. *Done when:* parity (2e-4, one 10-bit code), tiles, determinism, app/`ym` parity and the goldens pass.
**N-05 · Node thumbnails** (N-04, N-02b) — core, IPC, cache. *Done when:* thumbnails match truncated renders.
**N-06 · Pick white into the Colour tab** (N-02b) — header button gone; the card row, the hint pill, W, off when the Colour tab is left. *Done when:* e2e passes and the screenshots are retaken.
**N-07 · Node view shell** (N-02b, N-06) — Nodes button and N, layout A, divider and Expand, the viewport follows, the switcher, the simple-view notice and glyph. *Done when:* e2e passes; the native viewport check is listed for the Mac.
**N-08 · The graph** (N-05, N-07) — `@xyflow/react`: markers, cards with thumbnails, ports and links, select, add, delete, bypass, rename, move and reorder, links drawn by hand (connect, replace, pick up, remove, the refusals, the keyboard path), zoom and Fit, keys, a11y. *Done when:* e2e, a11y and the glass audit pass.
**N-09 · The panel** (N-08) — compact head, tabs by node kind, trim cards, per-node histogram, undo, copy and paste. *Done when:* e2e passes and the screenshots are reviewed.
**[HUMAN] N-H2** Checkpoint 1: a test DMG of nodes without masks.
**N-10 · Shapes and ranges** (N-04, N-09) — Linear, Radial, Colour range, Brightness range, key input and output, invert, opacity, feather; CPU and GPU. `ym`: masks in `--settings`, the `doctor` Masks line; `docs/10` and the help change in the same commit. *Done when:* parity, preview against export and the smooth-edge rule pass.
**N-11 · Show mask** (N-10) — Overlay and Mask only rendered by the core, M, the toolbar toggle, the View card. *Done when:* e2e passes; the Mac check is listed.
**N-12 · The Mask tab** (N-10, N-11) — Add mask menu, rows, cards, handles, range picking, copy rules. *Done when:* e2e passes and the screenshots are reviewed.
**[HUMAN] N-H3** Checkpoint 2: a test DMG with shapes and ranges.
**N-13 · Raster masks** (N-12) — photo-frame raster, resample, grow, feather, guided-filter Refine edges; CPU and GPU; tile halos; the memory cache. `docs/10` and the help change in the same commit if `ym` changes. *Done when:* parity, tiles and preview against export pass.
**N-14 · Brush** (N-13) — strokes, erase, size, hardness, drawing on the photo. *Done when:* e2e and parity pass.
**N-15 · Background from blur** (N-13) — the blur map as a mask. *Done when:* the tests on the blur-map fixtures pass.
**N-16 · SAM 2 weights** (N-01) — conversion script, safetensors f16, our release asset, the fetch with SHA-256, bundled in the app and in `data/` for `ym`, model card, notice, `docs/14`. *Done when:* the licence check passes and the build bundles the weights.
**N-17 · SAM 2 runtime** (N-16) — encoder, prompt encoder, decoder; deterministic; oracle fixtures from Meta's code. *Done when:* the IoU (≥ 0.99) and logit gates pass and the timing is measured.
**N-18 · Object in the app** (N-13, N-17) — `mask_find`, encoder in the background, feature cache, points and box, the "Object found" pill and the nothing-found error. `ym`: the `mask` error kind and the Masks line gain Object; `docs/10` and the help in the same commit. *Done when:* e2e passes and the app and `ym` agree on an Object mask.
**[HUMAN] N-H4** Checkpoint 3: a test DMG with SAM 2, timed on the owner's Mac.
**N-19 · Subject and People** (N-13) — Vision through `ym-apple` behind the macOS gate, a cached run-time probe, the items hidden where unsupported, the row message and the `ym` error for a document that has one anyway. `ym`: the `mask` error for Subject and People; `docs/10` and the help in the same commit. *Done when:* the Mac CI and the parity on a Mac pass; Linux and older-Mac builds show no Vision items.
**N-20 · Finish** (N-12 to N-19) — speed checks on a Mac, docs, `STATUS`, `CHANGELOG` (the JSON of `ym` changed shape; a 0.1 settings file is refused), README (its CLI section), `docs/10` read once more against the help and every output, screenshots; then the 0.2.0 release after N-H5. *Done when:* the gates are in `docs/STATUS.md`.
**[HUMAN] N-H5** Checkpoint 4: the final test of the whole plan; the owner's OK releases 0.2.0. **N-H1** (approve the spec) closes N-01.

**Gate N (0.2.0):** the gates of `docs/23` §11 pass (old documents byte-identical, parity, determinism and tiles, app and CLI, SAM 2, masks, speed on a Mac); the owner's approval of the last test build.

---

## F · Film node (planned, not scheduled)

A **Film node** that simulates a colour negative, a print and a scan with our own engine (`ym-film`, Apache-2.0) and Andrea Volpato's spektrafilm profiles (CC BY-SA 4.0), shipped as separate credited data. **Scheduled 2026-11-04 (F-H0), on branch `film`;** the owner's decisions of that day are in `docs/24` §1.3 (no letter to the author, variant A, real stock names with the disclaimer, the CIE tables shipped as credited data). F-01, the mockup round, is first. The spec is `docs/24-film.md`; the facts are in `research/film-spektrafilm-2026-11-01.md`. It needs the graph, plan and multi-pass loop of N (N-02 to N-04), the panel and masks (N-08 to N-12) and the tile-halo machinery (N-13).

Tasks (details, dependencies and *Done when* in `docs/24` §10): **F-01** mockups and the revised spec · **F-02** data and licences · **F-03** the engine's colour reference · **F-04** baking and the return through the cine rendering · **F-05** the spatial passes on the CPU · **F-06** the oracle and the provenance check · **F-07** the node in the core and `ym` · **F-08** the GPU passes · **F-09** the interface · **F-10** finish. **[HUMAN]** F-H0 schedule (done 2026-11-04) · F-H1 approve the mockups (variant A and the stock names decided 2026-11-04) · F-H2 write to the author and to hanatos (closed 2026-11-04: "It's fine just to mention Volpato") · F-H3 test build on a Mac · F-H4 trademark and data-rights check.

**Gate F:** the gates of `docs/24` §8 pass; the credit lines are in `NOTICE`, About and the node; the owner's approval of the last test build.

---

## D · Depth mask (done, on `main`; the owner's approval 2026-11-09)

Goal: a **Depth** mask in the Find group, cut from a relative depth map that **Depth Anything V2-Small** makes from the photo (24.8 M parameters, Apache-2.0, about 50 MB at f16, bundled like SAM 2.1's 77 MB), as a gradient to a far limit (first built as Near, Far and Softness bounds; the owner replaced them on 2026-11-08, `docs/23` §4.8), **Pick depth** and **Focus plane** (from the Sony focus box), a greyscale **Show depth** view, and `ym` support. The model is for masks only: lens character keeps its blur map. The spec is `docs/23-nodes.md` §4.8, the maths `docs/algorithms/06-masks.md` §5.8 and §8, the model `docs/07-ml-models.md`, the licence row `docs/14-licensing.md`; the facts and sources are in `research/depth-mask-2026-11-07.md`. **The owner's decisions of 2026-11-07** (`docs/STATUS.md` question 41): a depth model for masks only; bundled, never downloaded by the app; the teacher chain accepted (Apache-2.0 weights whose unreleased teacher trained on Hypersim, CC BY-SA 3.0, and Virtual KITTI 2, non-commercial), stated in the model card as SAM's card states SA-1B; the model is Depth Anything V2-Small; the working choices (band bounds with rank normalisation, Focus plane from the focus box, exports from the CPU map, evaluation on DIODE plus public photos) are settled on the evaluation sheets. V1's data rule holds: no measurements of our own, and no photos are shot for it.

**D-01 · Spec** () — `docs/23` §3.3, §3.10a and §4.8, `docs/algorithms/06` §5.8 and §8, `docs/07`, `docs/14` (the licence row, the teacher chain), `docs/10`, `docs/11` (short, pending the mockup), `docs/12`, this milestone, the model card draft `data/models/depth/MODEL_CARD.md` and `NOTICE` item 13. *Done when:* the specs agree with each other, `python3 tools/licences/check_bundle.py` passes, and the lead has read them. Everything not yet measured is marked as such.
**D-02 · Weights and oracle** (D-01) — `training/depth/` (`export.py`: the checkpoint to f16 safetensors with a self-describing header; `oracle.py`: fixtures from the upstream code; `oracle_photos.json`: 20 public photos with licences and hashes; `requirements.txt` with the pinned commit and versions; the tests of the scripts), a `depth` job in `.github/workflows/models.yml` that publishes the prerelease `models-depth-anything-v2-small-1`, `tools/fetch/depth.sh` (with `--fixtures`) with its SHA-256 pins, `data/models/depth/` (`.gitignore`, `SHA256SUMS`), and `tools/licences/check_bundle.py` taught the depth weights and `NOTICE` item 13. Conversion runs in CI, not on a developer's disk. The first fixtures settle the sign of the output (larger for nearer), how the sky is treated, the class-token and batch-norm settings of the checkpoint, and the exact tensor count and size. *Done when:* the release and the fixtures are published, the checkpoint's SHA-256 and the licence sentence are re-read at the pinned commit and written into `docs/14` and the card, `check_bundle.py --require-weights` passes with the fetched file, and the cost of f16 against the f32 checkpoint is reported.
**D-03 · Runtime** (D-02) — `ym_ml::depth`: the network in Rust on the `sam2` kernels, the new operations (patch embedding, LayerScale, DINOv2's positional resize, the transposed convolutions, the dense 3 × 3 convolutions, the bilinear resizes) each with a test against its definition, the picture's resize and normalisation, the mid-rank (section 5.8 step 1), `DepthModel::probe`. *Done when:* the gates of `docs/algorithms/06` §8.6 pass against the oracle (the resize, the four maps, the map, **the cut's IoU at least 0.99** on 20 photos at five bands, a second run and 1, 2 and 8 threads bit-identical), the timing on the Linux machine is measured and written into the spec and `docs/STATUS.md`, and the Mac timing is listed under M0-H4.
**D-04 · Core and `ym`** (D-01; real weights from D-03, a small fake model before) — `MaskKind::Depth` with its fields, validation and clamping (`docs/23` §3.3, §3.5), the map and its worker (150 ms debounce, cancellable, a viewer never waits), the key and its cache, `RasterSource::Depth` and the plane cache, the scheduler footprints (`StepKind` and `OpKind` have no wildcard arm), the grade-plan key, `sample_depth`, `focus_plane`, the Show depth view, `mask_kinds`, `DepthModel` in `MaskSupport`, `ym doctor`'s Masks line and the `mask` error, the schema, the copy rule, the mock's data, and a fake model so that the random-graph test draws Depth. `docs/10` and the help change in the same commit. *Done when:* `cargo test --workspace` is green; the rank, band, Pick depth and Focus plane tests of `docs/algorithms/06` §7 pass; **the app and `ym` write byte-identical files for a Depth row**; **the golden images and every document without a Depth row are unchanged**; the map's key follows the Input and not a node; the docs test passes.
**D-05 · Interface** (D-04) — first the mockup of the Depth card (`docs/11` *The Depth card*) for the owner's approval, then the card, Pick depth, Focus plane, the Show depth view, the row, the menu item and the states (`docs/23` §7.6), then the e2e tests and the new reference screenshots. *Done when:* the mockup is approved (D-H1), e2e passes against the mock, the glass audit and a11y pass with the new states, the screenshots are looked at by the owner before they are committed, and the native-viewport check on a Mac is listed.
**[HUMAN] D-H1 · Approve the mockup** of the Depth card (D-05 builds nothing before it).
**D-06 · Evaluation** (D-03, D-04) — public data only (V1 data rule): the **ordering** of near and far on DIODE (MIT) against the oracle's own ordering and the DA-2K figure for ViT-S that the authors report (95.3 % of pairs; the exact pair metric is written first); contact sheets of public photos for hair, foliage, sky, glass, mirrors and water, macro, shallow-focus backgrounds and flat copy work, with the band at several settings, with and without Refine edges and with a Subject row; the rank against a percentile scaling on the same sheets; the defaults (`far` 0.5, `softness` 0.5, `refine` 0.5, the pick width 0.2, the ramp minimum 0.02) and the Focus plane's box and width; whether a width from `FocusPosition2` helps (on public files only); Linux timing, and the Mac timing listed under M0-H4; the model card's known limits filled in from the sheets. *Done when:* the numbers and the sheets are in `docs/STATUS.md`, the card is finished, and every changed default is in the spec.
**[HUMAN] D-H2 · Blind review** of the contact sheets by the owner: keep, change the defaults, or drop the kind.

**Gate D:** the gates of `docs/algorithms/06` §8.6 (resize, maps, the cut's IoU at least 0.99 on 20 photos, 1, 2 and 8 threads equal) and of `docs/23` §11 (the Depth row: app and `ym` byte-identical, documents without a Depth row byte-identical, the golden images unchanged); `cargo deny check` and `check_bundle.py --require-weights` pass with the depth weights and `NOTICE` item 13; the DIODE ordering of our runtime within half a point of the oracle's (starting); the timings written down (Mac: M0-H4); the owner's verdict at D-H2. A failed gate is reported, never loosened.

---

## X · Mixers (in progress, branch `mixers`)

The node kinds the Add node menu shows as "Later": **Parallel** (adds each branch's change to the picture the branches start from), **Layer** (stacks its inputs and blends each over the ones below, in linear light, through the mask of the node that feeds it) and **Splitter and combiner** (one channel per branch, then one picture again). The spec is `docs/25-mixers.md`; the facts are in `research/research_notes/Node mixers/resolve_mixers.md`. **The owner's decisions of 2026-11-10** (`docs/STATUS.md` question 43): "All three now"; Parallel is "Input + each branch's change" on LogC4 code values; the Layer's modes are the "Scene-safe short list" (Normal, Add, Subtract, Multiply, Lighten, Darken, Difference) in linear light; a layer's alpha is the "Feeding node's mask, as Resolve". The other choices are working choices that wait on X-H1 (`docs/25` §9). `PIPELINE_VERSION` takes a minor step to 0.11.0; graphs without the new kinds change by no bit. It needs the graph, plan and panel of N (N-02 to N-12).

Tasks (details, dependencies and *Done when* in `docs/25` §10): **X-01** spec (this milestone, `docs/25`, the changes to `docs/23`, `docs/algorithms/06`, `docs/10`, `docs/11` and `docs/STATUS.md`) · **X-02** mockups in `design/mixers/` · **X-03** format, validation, patches, wire and TS types, schema and CLI (no change in the app's interface) · **X-04** the plan with steps of several inputs and the CPU reference, with random graphs that have joins · **X-05** the device (the plan record within the K3 bindings, the WGSL, parity on lavapipe; Metal through the Mac GPU CI) · **X-06** the interface (the menu, the cards with numbered ports and "+", the link rules, the graph operations, the panels, the mock, e2e, screenshots the owner reviews) · **X-07** finish (the gates, `CHANGELOG`, pipeline 0.11.0, a test build). **[HUMAN]** X-H1 approve the mockups and the working choices (no interface work before it) · X-H2 test build and review on a Mac.

**Gate X:** the gates of `docs/25` §7 pass (formulas and identities on the CPU, the plan, random graphs with joins, tiles and determinism, app and `ym` byte-identical, device parity on lavapipe and Metal, the interface); `golden check` 61 of 61 unchanged; the owner's approval of the last test build.

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

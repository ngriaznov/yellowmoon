# 01 · Architecture

This document describes how Yellowmoon is built: the workspace and its crates, the core's session, settings and render graph, the job scheduler, the export queue, the native viewport, the IPC between the webview and the core, what persists, and how the app finds its data.

## Shape of the system

```
┌──────────────── Yellowmoon.app ─────────────────┐      ┌──────── ym (CLI) ────────┐
│  Webview (React/TS): panels, filmstrip, wizard  │      │  clap, JSON output       │
│        │ Tauri commands / events (typed)        │      │                          │
│  src-tauri: app shell, window, updater, menus   │      │                          │
│        │                                        │      │                          │
│  Native viewport (wgpu surface on Metal) ◄──────┼──┐   │                          │
└────────┼────────────────────────────────────────┘  │   └────────────┼─────────────┘
         ▼                                           │                ▼
   ┌───────────────────────── ym-core ──────────────────────────────────────┐
   │ Session (in memory) · DevelopSettings · auto estimates · render graph   │
   │ job scheduler (viewport > thumbnails > export) · export queue          │
   └──┬─────────┬─────────┬──────────┬──────────┬──────────┬──────────┬─────┘
      ▼         ▼         ▼          ▼          ▼          ▼          ▼
   ym-raw   ym-profiles ym-color   ym-noise    ym-ml     ym-looks   ym-export
   decode   registry    LogC/AWG   estimate    U-Net     library    JPEG/TIFF/
   detect   + data      ACES2,LUT  σ map       runtime   import     PNG/DNG
      └────────────── ym-gpu (wgpu device, tiling, WGSL kernels) ──────────────┘
```

All image processing happens in `ym-*` crates. `src-tauri` and `ym-cli` are thin front ends: they translate user actions into `ym-core` calls and show results. `ym-apple` is the bridge to Apple's on-device model for Apple Auto (`docs/19-apple-auto.md`); both front ends link it and hand its advisor to `ym-core`.

Data flow:

1. **Opening files.** `ym-core` probes each file with `ym-raw`, then a background job per photo decodes it, computes the automatic estimates (light, exposure, white balance, noise, sharpening) and renders a thumbnail.
2. **Editing.** A control change is a settings patch for one photo. The render graph re-runs only the stages the patch affects, and the viewport draws the new frame.
3. **Export.** The export queue renders each photo at full resolution in tiles, and the writers in `ym-export` encode the file. The CLI and the app call the same function, so the bytes are the same.

## Repository layout

```
Cargo.toml                 workspace (members listed explicitly: crates/*, src-tauri, tools/oracle, tools/golden, tools/dngfit)
rust-toolchain.toml
deny.toml                  cargo-deny licence and advisory policy
crates/
  ym-core/                 session, settings, auto development, render graph, scheduler, export queue,
                           preferences, look service
  ym-raw/                  ARW container, SR2, decoders, camera detection, metadata
  ym-profiles/             camera profile registry; loads data/cameras/**
  ym-color/                LogC3/4, AWG3/4, ARRI matrices, WB from curves, sensor map, illuminant estimator,
                           CDL, cube LUTs, ACES 2 output transform (CPU reference)
  ym-gpu/                  wgpu device/queue ownership, tiling, buffer pools, WGSL kernels + parity tests,
                           the CPU reference implementations of the raw and output stages
  ym-noise/                per-image noise estimation, priors, σ maps
  ym-ml/                   model loading, inference on tiles (burn or ort behind one trait), the crop model and the bokeh map model (CPU)
  ym-looks/                look library, ARRI package recognition, importer, manifest
  ym-export/               JPEG, TIFF, PNG, DNG writers, ICC, EXIF/XMP; runs output sharpening (CPU code of ym-gpu) after the resize
  ym-compose/              Auto crop: the straightener and the frame proposer (docs/20-auto-crop.md; pure Rust, CPU; the crop model is ym-ml's)
  ym-apple/                Apple Auto's bridge to Apple's on-device model (Swift, compiled only for Apple silicon
                           with the macOS 27 SDK or later; a stub elsewhere)
  ym-cli/                  the `ym` binary
src/                       React + TypeScript frontend (Vite, Vitest, Playwright)
src-tauri/                 Tauri app crate (window, commands, events, updater, native viewport)
data/                      shipped data: alexa/, arri/, cameras/, dng/, looks/, models/ (small weights, committed with model cards)
training/                  PyTorch training code, dataset scripts, model cards (not shipped)
tools/
  oracle/                  test-only crate linking rawler (LGPL) to compare decodes; never shipped
  fetch/                   scripts that download test corpora and weights with SHA-256 checks
  golden/                  golden-image runner (uses ym-cli)
  dngfit/                  fits the DNG profile and checks the DNG gates
  eval/ lens/ looks/ dng/ ci/ dev/ release/ licences/ vectors/ readme/ social-card/
                           script-only folders (evaluation studies, lens oracle, look tooling, DNG validation,
                           CI guards, sidecar and Mac GPU helpers, version bump, licence texts, reference vectors,
                           README screenshots)
testdata/                  git-ignored; filled by tools/fetch
design/                    mockups, screenshots, tokens (reference for the UI)
research/                  reports and experiments (background, read-only)
docs/                      these specs and STATUS.md
templates/                 CI, release and cask templates
Casks/yellowmoon.rb        Homebrew cask
```

### Package names

| Path | Cargo package | Binary |
|---|---|---|
| `crates/ym-*` | `ym-core`, `ym-raw`, … (same as the folder) | — |
| `crates/ym-cli` | `ym-cli` | `ym` |
| `src-tauri` | `yellowmoon-app` | `Yellowmoon` (Tauri `mainBinaryName`) |
| `tools/oracle` | `oracle` (test-only, links rawler) | `oracle` |
| `tools/golden` | `golden` | `golden` |
| `tools/dngfit` | `dngfit` | `dngfit` |

Script-only folders under `tools/` are not workspace members.

### Building the app crate

`tauri-build` checks at compile time that every `bundle.externalBin` and `bundle.resources` path exists, on every `cargo build/test/clippy` of `src-tauri`. So:

- `tools/dev/prepare-sidecar.sh` builds `ym-cli` and copies it to `src-tauri/binaries/ym-<host triple>`. CI and `npm run tauri dev` (via `beforeDevCommand`) run it first. Release builds copy the per-architecture binaries and the universal one (`release.yml`).
- `data/models/` and `data/looks/builtin/` always exist (each has a README), so the resource paths resolve.

### Dependency rules

- `ym-core` depends on all other `ym-*` library crates except `ym-apple`. Nothing depends on `ym-core` except `src-tauri`, `ym-cli` and `ym-apple` (which implements the advisor trait of `ym-core`).
- `ym-color` is pure maths with no GPU dependency. `ym-gpu` depends on `ym-color` for constants and CPU reference functions used in parity tests, and on `ym-noise` for the noise model the raw algorithms take (`docs/algorithms/INTEGRATION.md` §3).
- `ym-raw` has no GPU dependency; it returns CPU buffers.
- `ym-ml` reaches the shared wgpu device through an optional `ml` feature of `ym-gpu`.
- No crate under `crates/` depends on Tauri, the webview, or `tools/oracle`.
- `ym-cli` does not depend on Tauri.

## Core concepts (ym-core)

### Session

A `Session` holds the photos opened in this run, in memory only.

```rust
pub struct Session { photos: Vec<Photo>, current: usize, picked: BitSet, defaults: SessionDefaults }
pub struct Photo {
    id: PhotoId,                 // stable for the session only
    path: PathBuf,               // original, never written
    meta: RawMetadata,           // camera, lens, exposure, dimensions, compression, timestamps
    camera: CameraMatch,         // detected body + profile used + why (exact / sensor family / default)
    auto: AutoEstimates,         // illuminant, exposure, WB, noise model, sharpening — computed once
    settings: DevelopSettings,   // user offsets from auto; defaults = "Auto" everywhere
    caches: PhotoCaches,         // decoded mosaic, previews; evictable
}
```

The structs are the shape, not the code (`session.rs`, `auto.rs`, `camera.rs`). `Session` holds its state behind one mutex and a decode pool, takes `&self` everywhere (`Arc<Session>` is how the app shares it) and keeps each photo's own estimates next to the ones it serves.

#### Environment

`SessionEnv::load(data_root)` loads the profile registry, the noise priors (`cameras/noise`) and the illuminant models (`models/illum/<profile>.json`). `with_events`, `with_thumbnails`, `with_default_look`, `with_decode_threads` and `with_forced_profile` set the rest (`with_forced_profile` is the CLI's `--profile`, refused with the valid ids when the id is unknown). `EventSink::new(callback)` or `EventSink::channel()` receives the events. The sink runs while an internal lock is held, which is what makes "no event after `close()` returns" true, so it must not call back into the session.

#### Opening files

- `Session::open(paths)` probes each file (`ym_raw::metadata::probe`: a fast metadata read, no full decode) in parallel and registers it before `open` returns, so the filmstrip has its photos at once. Then one background job per photo decodes, estimates and renders a thumbnail. Progress events drive the Start screen's "Developing N photos" list.
- Folders are walked for `*.ARW` (extension compared case-insensitively), one level deep unless `OpenOptions { recursive: true }` (the CLI's `--recursive`). Other files and `.`-files (macOS's `._DSC0001.ARW` resource forks on exFAT cards) are ignored and counted in `OpenOutcome::ignored`. Paths that do not exist are listed in `missing`. A file already in the session (same canonical path) is not added twice (`already_open`).
- A photo's path is made absolute against the working directory without resolving symbolic links, so `ym render .` and `ym render DSC0001.ARW` work like their absolute forms. The folder name, and the `<folder name> export` name of a `.` or `..` folder, come from the folder it resolves to.
- Photos are sorted by path, case-insensitively. The first photo of an empty session is current.
- Photo ids count up for the life of the `Session`, also across `close()`. A command that names a closed session's photo gets `UnknownPhoto` and can never reach a photo of the next session.
- `OpenOutcome::no_arw()` / `message()` is the Start screen's "No ARW files in what you dropped." It applies when nothing ARW was found at all, not when everything was already open.
- A probe that fails makes a photo that is Skipped with the `RawError` text of `docs/04-raw-decoding.md`; the UI shows CAN'T OPEN.
- `SessionState::folder_name` is the one folder given (or the one folder all photos share, else `None`). `default_export_folder` is `<parent>/<folder name> export`; the export queue still checks that it is writable and falls back to `~/Pictures/Yellowmoon export`.

#### Development jobs

One job per photo runs on a pool of `min(available_parallelism - 2, 6)` threads (at least one; `default_decode_threads()`), CPU only: decode, `AutoEstimates::compute`, then the thumbnail hook (`ThumbnailRenderer`, implemented by the render graph; without one the photo is Ready after its estimates). The pool takes the current photo's job first.

- A failing job ends in Skipped for that photo only. A panic inside a development job is caught, logged with its message and reported the same way; a panic while prefetching a neighbour's mosaic is only logged, and the thread survives either way. A decode error is `RawError`'s text (ARW 6.0 included, so the `arw6` feature needs no change here).
- `close()` bumps an epoch: queued jobs are dropped, running ones stop at their next stage, and their results and events are discarded. The session is empty and reusable at once; it does not wait for the pool.

#### Events

Per photo, in this order: `probe(done)`, `decode(start)`, `decode(done)`, `estimate(start)`, `estimate(done)`, `thumbnail(start)`, `ThumbnailReady`, `thumbnail(done)` (= Ready). A failure after the probe ends the sequence with one event whose `stage` is the failing one, `done = true` and `error` set (Skipped). A file that fails the probe has just `probe(done, error)`.

`SessionEvent::EstimatesChanged { id }` is sent when other photos finishing moved a photo's shared exposure (below), and for the current photo when its pooled noise model changed.

#### Mosaic cache

At most `MOSAIC_CACHE_LEN` (3) decoded mosaics are kept: the current photo's and its neighbours' (by filmstrip index). A mosaic outside that window is dropped when its job is done. `Session::raw_image(id)` decodes again when something needs one (the render, Pick white, Re-run) and keeps the result only if the photo is in the window. `set_current` re-fits the window and queues the missing neighbours (prefetch jobs run after development jobs).

- **One decode per photo at a time.** `raw_image` is single-flight: callers that ask for a photo that is being decoded wait for that decode and share its image. After a jump to an uncached photo the viewport frame, the histogram, a tile for each look of the Look tab and the prefetch all ask at once, and a mosaic is 122 MB at 61 MP against a budget of three. A leader that fails or panics wakes the others, which look again and fail with their own error. Tests: `tests/session.rs` `callers_that_ask_for_one_uncached_mosaic_at_once_share_one_decode`, the `SingleFlight` unit tests.
- **A prefetch is checked when it runs.** The job is dropped unless its photo is still in the window and not cached yet, so stepping through photos faster than they decode leaves no decodes for photos already passed (`a_prefetch_queued_for_a_photo_the_person_has_left_decodes_nothing`).
- Previews are not kept: the 1/8 preview is 11 MB for 61 MP and exists only while the estimates are computed.

#### Estimates the photos share

Two estimates depend on the other photos: the burst median exposure (`docs/02-colour-pipeline.md` Auto exposure step 6) and the pooled noise model (`docs/06-noise.md` Batch pooling). The session keeps each photo's own `AutoEstimates` and serves `Session::estimates(id)` and `view(id)` with the burst median in `exposure.ev` (`own_ev` keeps the photo's own value) and the pooled model in `noise`.

- Photos that do not state their ISO are not pooled: the ISO is part of the pool key, and an absent one cannot be told apart. Each keeps its own estimate.
- Both are deterministic functions of the set of photos (checked for 1, 3 and 4 decode threads). The burst groups are recomputed over every photo with estimates. The pool of a group (`NoisePool`) is built from its first 16 photos in id order, whichever finished first.
- A pooled model is used only if its variance at three signal levels is within 1.6 of the median of its members' own (`POOL_AGREEMENT_FACTOR`). The check protects against a pool that contradicts its members' own models, such as a group that mixes filtered and plain frames or files with different in-camera processing at one ISO; each file then keeps its own estimate. `ym_noise::BatchPool` treats a merged bin as precise as its best member and no more (`docs/06-noise.md` Implementation notes), so three frames of one scene pool to their own fit with the check passing (`tests/auto.rs`).
- Pooling frames that cover different intensity ranges works as intended: three such frames give `a` within 2 % of the truth and `b` within 20 %, which no single frame resolves.

**App/CLI parity for early exports.** The CLI waits for the whole session (`wait_idle`) before it exports, so its files carry the final shared estimates. An export queued earlier from the app must write the same bytes. `Session::estimates_final(id)` says whether the estimates of a photo that depend on other photos are final: every photo of its burst (the groups are made over every photo with a capture record, by metadata alone) and every photo of its noise pool (model, ISO, bit depth, compression) has its own estimates or has failed. Thumbnails, prefetching and viewport renders are not part of it; an export must not wait for them.

`ExportQueue::start` records, before it snapshots each photo, whether `estimates_final` was already true. For a photo for which it was not, the job's turn for that photo waits (a poll of 25 ms that a cancel interrupts) until it is, then takes the estimates (burst median exposure and pooled noise) from the session again and keeps everything else of the snapshot (settings, Pick white sample) as queued. A photo with no estimates of its own yet cannot be snapshotted (`QueueError::Developing`: "Some photos are still being developed. Wait until they are ready, then export again."). If the session is closed while a job waits, development has stopped for good and the job goes on with the estimates it was queued with. Tests: `tests/export_queue.rs` (a job queued while its burst is developing waits, then writes the bytes the wait-for-idle path writes, and a photo outside the burst and the pool does not hold it up), `tests/session.rs` (`estimates_final`).

#### Apple Auto worker

Apple Auto (`docs/19-apple-auto.md`) is an optional second opinion on the automatic settings. With an advisor (`SessionEnv::with_apple_auto`; the app and `ym` give one through `CoreOptions::advisor`, nothing else does) the session has one more thread.

- **When it acts.** Apple Auto acts when an advisor exists, the `apple_auto` preference or the run's `AppleAutoMode` is on, and the advisor is available. This is read again when a session opens, when the status is asked, when the preferences change and before each export.
- **Order.** The worker asks the advisor about one photo at a time: the photo on screen, then the photos an export waits for (`ExportQueue::start` moves them up, and the session moves each one's burst frames without an answer up with it), then the others in filmstrip order.
- **The picture.** It is rendered by the `PictureRenderer` hook (`pipeline::AdvisorPictures`, a renderer of its own without the GPU) from the photo's own estimates, so it does not depend on the order photos develop in. The development job renders it after the thumbnail and keeps at most 64 at a time; the worker renders the others from the file when their turn comes.
- **Calls.** The call blocks the worker only, never a caller's thread, and never under the session's lock. `rateLimited` and `concurrentRequests` are tried twice more, 2 s apart; every other error fails the photo, which then uses the rule. A panic anywhere in a request fails that photo (kind `other`) and the worker goes on. A worker that cannot start or stops leaves the session as one without an advisor, so nothing waits for it.
- **Answers.** The answer (or the failure) is kept in memory with the photo, also while Apple Auto is switched off. It is dropped by Re-run (which asks again), by new instructions for the model (`Session::set_apple_auto_instructions`: every photo is asked again, the photo on screen first, and an answer to the older instructions that arrives later is dropped) and by `close()`.
- **Effect on the estimates.** The served estimates take the answer's values on top of the rule's (burst frames share the median of their steps, `docs/19-apple-auto.md` section 5). Photos whose values or state moved get `EstimatesChanged`, and those whose picture changed get a new thumbnail (`ThumbnailReady`; a photo still developing gets it after its first). While Apple Auto acts, `estimates_final` also waits for the photo's and its burst's answers, so an export waits for them exactly as for the burst's exposure. `wait_idle` waits for the worker too (`ym` waits for the answers of the photos it develops). The viewer never waits: a photo is served with the rule until its answer arrives.

Tests: `crates/ym-core/tests/apple_auto.rs`.

#### Photo view

`Session::view(id)` returns a `PhotoView` (path, probe metadata, `PhotoCamera`, profile, a snapshot of the settings, the served estimates, the Pick white sample, `estimates_version`) for the render graph and the export queue. `estimates_version` changes whenever the served estimates or the Pick white sample change (the sample is taken when the pick is set, so a new pick bumps it); it belongs in the render cache key of the stages after C1.

#### Camera

`PhotoCamera` composes `ym_profiles::detect` with `ym_raw`'s `DecodeKind`: `reduced` (Lossless M or S), `stand_in` (anything but `Exact`), the `CameraCopy` of `docs/03-camera-profiles.md` and, for the RAW tab, `card()`: Detected, Profile, STAND-IN, the explanation, the REDUCED tag with the note of `docs/04-raw-decoding.md` and, for ARW 6.0 and other layouts without a decoder, the unsupported message.

#### Estimates in detail

`AutoEstimates::compute` (`auto.rs`) makes the estimates when a photo is opened.

- **Illuminant.** `ym_color::illum`'s model for the camera's profile, with the as-shot white balance as its prior. `RawImage::wb_as_shot` is stored per CFA position and `auto::as_shot_rggb` reads it by colour, so a BGGR or GRBG mosaic does not swap red and blue (every Sony body is RGGB, where the two agree).
- **Illuminant fallback** (a profile without a model file, or a model that fails): the as-shot white balance as a camera white `[g_G/g_R, 1, g_G/g_B]` (green is the mean of the two greens of SR2 0x7313), confidence 0.5, class probabilities from its CCT alone (tungsten up to 3500 K, daylight from 4500 K, a linear blend between, nothing for fluorescent and LED). A file with no white balance at all gets the profile's 5500 K daylight white with confidence 0.
- **The 1/8 preview** bins the photo's area (the camera's crop, else the active area, else the readout) 8 × 8 per colour. Its clip mask marks a block when any photosite in it is at or above `white − 4·qstep` (or 0.995 of the range without a qstep plane).
- **Lossless M and S** have no mosaic. Their preview comes from the RGB data, normalised with **one range for the three channels, the green one**, not channel by channel. `docs/04-raw-decoding.md` gives red and blue lower white levels because the camera's white balance was divided out again; dividing each channel by its own range would put the camera's balance back, and the initial gains `g0` would then balance the data a second time. The render's normalise stage R1 scales this layout the same way, so channel `c` clips at `1 / g0_c` as `docs/02-colour-pipeline.md` C1 says; clipping is still judged against each channel's own white level. Their noise model is the centre of the body's prior (`NoiseSource::PriorOnly`; ISO 100 is assumed when the file has no ISO). A frame the estimator refuses (too small) is treated the same.
- **Sharpening and noise reduction** Auto come from `ym_gpu::cpu` (`auto_sharpen`, `NoiseReduction::auto`) so the card and the render agree. The base ISO of the sharpening ramp is the gain-regime table's (100 when the body is not in it).
- **Kelvin and tint** are the Auto readout of the all-Auto, EV 0 chain.

#### Auto exposure meters the picture the render makes

`docs/02-colour-pipeline.md` Auto exposure step 1 meters C1–C6 with EV = 0 on the 1/8 preview. The render divides every photosite by the lens's relative illumination (R3, `docs/05-raw-processing.md`), so the metering applies the same gain:

- `auto::build_preview` computes the R3 gain `1 / G(r)` at the centre of every 8 × 8 block (`Preview::vignette_gain`, from the same 0x7032 table and the same radius as `VignetteProfile`). A file without a usable table has none, and neither do Lossless M and S, which the render does not correct either.
- `Preview::metered(g0)` applies it before the initial gains, in the order of the raw stages.
- Only the metering reads it. `Preview::as_raw_preview` (the illuminant estimator's input, whose contract is the sensor's own picture) and the clip mask do not.
- The estimates are made when the photo is opened, before any setting changes, so the gain does not follow the RAW tab's `decode.lens_corrections` switch. The exposure is metered as the default render shows the picture, and a user who turns the corrections off keeps the exposure of the corrected picture, as for any other setting that only Re-run picks up.
- A file whose table corrects nothing (every knot 0, as the synthetic golden set's files have: `G = 1` exactly) meters as it would without the gain.

On the nine real files the gain moves `EV_auto` by −0.06 (ILCE-5000), −0.05 (ILCE-6300), −0.10 (A7 II), −0.07 (A7 III) and −0.10 EV (A7 IV), the five with tables (corner gain 1.2 to 1.8), and not at all for the other four, which have none. The test is a flat field behind a vignetting lens, which must meter like the same field without the lens (`tests/auto.rs`: −0.585 EV both, −0.504 EV without the gain).

#### Measured (estimates)

- Synthetic scenes (`crates/ym-core/tests/common`: a known light, brightness and Poisson-Gaussian noise): the as-shot fallback reads 2800 to 7500 K back within 50 K. Auto exposure of a flat neutral scene lands on `log2(0.18 / Y)` within 0.08 EV (and follows the camera's exposure compensation at half weight). A ramp's noise is recovered within 1.1 % in sigma and 15 % in `a` (three seeds, ISO 800).
- The shipped illuminant model for the A7R IV profile finds the light of a patchwork scene within 0.55° from the pixels alone and within 0.49° with the right as-shot prior. When the as-shot white balance is badly wrong (daylight gains under 2800 to 6500 K light) the error is 2.9° at worst, under half the camera's own error.
- On the nine real files of `testdata/real/files` (local, not committed) every file develops (all nine in 2.2 s on two threads), and auto exposure is +0.6 to +1.9 EV (median +1.0). `tests/real_files.rs` prints, for information only, our render at `EV_auto` against the embedded JPEG (median luminance in linear light): 6 of 9 within 0.5 EV, with errors in stops A300 −0.94, A500 +1.19, ILCE-5000 −0.42, ILCE-6300 −0.49, A7 II +0.37, A7 III −0.22, A7 IV −0.41, NEX-5N −0.28, NEX-7 +0.80. The auto-exposure gate is G3, `EV_auto` against the camera engine's own gain (`tools/eval/autoexp.py`, `docs/02-colour-pipeline.md` Evaluation), which removes the display transform and the shape of the camera's curve that the embedded-JPEG metric mixes in.
- All 48 files of `testdata/synthetic` (every raw variant, with gradient, noise, clipped and dark content) develop; the dark ones meter to the +3 EV limit.

#### Removing photos

`Session::remove(ids) -> RemoveOutcome { removed, current, estimates_changed }` takes photos out of the filmstrip. `Session::restore_removed() -> RestoreOutcome { restored, already_open, current, estimates_changed }` puts the last removal back (`restorable()` lists what it would). The UI side is `docs/11-ui.md` §*Removing photos*. Code: `session.rs`, `app.rs`, `pipeline/graph.rs`, `scheduler.rs`.

**The files on disk are not touched** (AGENTS rule 5; `tests/session_remove.rs` compares bytes and modification times). Nothing is saved per photo (rule 4), so a photo's settings, picks, estimates and place go into one `Removal` kept in memory until the next removal that takes something out, `close()`, or files opened into the emptied session (a new session begins; ids that are not in the session are ignored and do not replace it).

`remove` works in one call under the emit lock and the state lock:

1. The photos leave `photos`, `index`, `picked` and `known` (so the same file can be added again) and their mosaics leave the cache. The current photo hands over to the next photo that stays, else the previous one, else none. An emptied session reports no folder; the next `open` sets it.
2. Queued development and prefetch jobs for them are dropped, and a running job stops at its next stage. `progress` and the thumbnail event check that the photo is still in the session, and each photo carries an `incarnation` that a removal bumps and a job copies when it is queued, so the job of a photo that was removed and put back (its development restarts as `Waiting`) cannot report as the new one. **No event about a removed photo is sent once `remove` returns.**
3. The estimates the others share are made again from the photos that are left: the burst median exposure of the frames of its burst and the pooled noise of its noise group (`State::refresh_effective`, `rebuild_pool`; both stay functions of the set of photos). Photos whose served estimates moved get `SessionEvent::EstimatesChanged` and are in the outcome. The shell's event sink tells the webview with `settings_changed`, as it does for every `EstimatesChanged`, and `remove_photos` and `restore_photos` queue them a new thumbnail.
4. The new current photo's neighbours are queued for decoding.

`restore_removed` inserts each photo at its old place (a file added again meanwhile stays out and is counted in `already_open`) and brings back the picks, the folder of an emptied session, and the photo on screen: the removal's current photo is current again when it was taken away and the person has not moved since (`current == successor`).

`Core::remove_photos` adds the renderer's side. `Renderer::retire_photos(ids)` marks the ids (a request that was on its way is refused as cancelled, as after a close), cancels their queued and running viewport, tile, thumbnail, histogram, readout and look-tile jobs on the CPU scheduler (`JobScheduler::cancel_photos`) and on the device (`Scheduler::cancel_owners`), forgets their graphs and the thumbnail statistic, and lets go of the photo being looked at when it is one of them. `Core::restore_photos` lifts the mark first.

**Export jobs are the one thing not cancelled.** `ExportQueue::start` snapshots every photo of its scope, so a running or queued export keeps the photos it was started with and finishes them (`load_raw` decodes from the file when the session no longer has the photo, `wait_for_final_estimates` goes on with the queued estimates), and an export render of a retired photo drops the graph it made when it is done. One accepted caveat: a job queued while the photos of its burst or noise pool were still developing re-reads the shared estimates when its turn comes, and that is from the photos that are left then. `ym` has no removal (it is a front end of one run).

**Apple Auto** (`docs/19-apple-auto.md` section 6). A removed photo keeps its answer or its failure in its `Removal`, with the rest of what the session keeps of it, so a photo that comes back with one is not asked again; nothing else about it stays live.

- `State::take_out` drops its picture (`apple_picture`, in memory only, hard rule 4), takes it out of the order of the photos an export waits for (`apple.urgent`) and bumps its incarnation. The worker only looks at `State::photos`, so a removed photo is never chosen.
- `Inner::apple_store` and `apple_recover` drop an answer or a failure whose photo is not in the session or has another incarnation (`AppleTask::incarnation`). That is the request that was in flight when the photo left, also when the photo was put back since (it comes back without an answer and is asked again), and the retry loop stops asking about a photo that left.
- The steps a burst shares are made again by the same `State::refresh_effective` that makes the burst median exposure again (`share_burst_steps` over the frames that are left). The photos whose values moved are in `estimates_changed` (they are sent `EstimatesChanged`) and their thumbnails are drawn again (`Inner::repaint`, after the event lock is let go). `remove` and `restore_removed` wake the Apple Auto worker and whoever waits in `wait_idle`.
- An export that waits for the answer of a frame or a photo that left stops waiting: `estimates_final` of the frames that are left no longer counts it, and for the photo itself it is `UnknownPhoto`, which the queue reads as "go on with the snapshot it was queued with".
- `put_back` drops the answers of a removal whose `instructions_gen` is not the session's (new instructions dropped every answer of the session, and these photos were not in it). It puts a restored photo without an answer into `apple.urgent` when a photo an export waits for has it in its burst (`State::prioritise`, which `Session::prioritise_apple_auto` also uses), and leaves the restored photos out of `estimates_changed` (the restore hands them over) but not out of the thumbnails drawn again.

Tests: `crates/ym-core/tests/apple_auto.rs` (section *Removing photos from the filmstrip, and Undo*) and `session_remove.rs` (*Apple Auto*).

**The shell.** `AppApi::remove_photos` / `restore_photos` (commands `remove_photos(photo_ids)` and `restore_photos()`, both answering `SessionChange { photo_ids, already_open, session }`) forget the pending thumbnail requests of the photos, set their thumbnails aside in the `FrameStore` (`set_aside_photos`: not served, kept for Undo; the next removal drops the earlier ones; Undo announces them again at the addresses they had), and follow the current photo with the renderer and the viewport (`sync_current`).

### DevelopSettings

Every control in the UI and every CLI flag maps to one field. Defaults reproduce the automatic development.

| Field | Type / range | Default | UI |
|---|---|---|---|
| `look` | `LookId` | user's default look (Signature out of the box) | Look tab |
| `look_intensity` | 0–100 % | 100 | Look tab |
| `exposure` | `Auto` or `Manual(ev)` −4…+4, step 0.01 | Auto | Light › Exposure |
| `contrast` | `Auto` or `Manual(value)` −100…100 | Auto (0, or Apple Auto's value) | Light › Exposure; Contrast also on Look › Tone |
| `brightness` | −100…100 | 0 | Light › Exposure |
| `highlights`, `shadows`, `black` | `Auto` or `Manual(value)` −100…100 | Auto (0, or Apple Auto's value) | Light › High dynamic range |
| `white` | −100…100 | 0 | Light › High dynamic range |
| `light_source` | Auto, Day, Shade, Tungsten, LED, Fluorescent | Auto | Colour › White balance (segmented) |
| `white_balance` | `Auto` or `Manual { kelvin 2000–11000 step 50, tint −50…50 }` | Auto | Colour › White balance; Kelvin also on Look › Tone (Temperature) |
| `color_balance` | shadows and highlights wheels: `Auto` or a point `(a, b)` in a unit disc; midtones wheel: a point | Auto, (0, 0), Auto | Colour › Colour balance |
| `saturation` | `Auto` or `Manual(value)` −100…100, keeps the luminance (`docs/02-colour-pipeline.md` C9) | Auto (0, or Apple Auto's value) | Colour › Colour balance; also on Look › Tone |
| `color_warp` | `{ pins: [...] }`, at most 16 pins; a pin is `{ source: [a, b], target: [a, b], locked, range 0.01–0.25, shadows 0–1, highlights 0–1, pivot_ev −6…6, exposure_ev −2…2 }`, points scaled back to a radius of at most 0.5 (`docs/02-colour-pipeline.md` C9b); every pin field but `source` has a default (`target` = `source`, `locked` false, `range` 0.04, `shadows` and `highlights` 1, `pivot_ev` and `exposure_ev` 0). A patch replaces the whole list. | no pins | Colour › Colour warp |
| `lens` | `{ kind, flares 0–100, edges 0–100, anamorphic 0–100 }`; `kind` is `default`, `classic`, `vintage`, `petzval`, `triplet`, `sonnar` or `detuned` (`docs/21-lens-character.md` §2). All three amounts at 0 is no lens, whatever `kind` says. A patch may carry only the parts it changes; choosing a lens sends its starting Flares and Edges with it (the card and `--lens`), the core moves nothing by itself. | `{ default, 0, 0, 0 }` | Look › Lens (rendering: docs/21, in progress) |
| `sharpen` | `Auto` or `{ amount 0–300, radius_px 0.3–2.5 }` | Auto | Detail › Sharpening (Amount, Radius): capture sharpening, R8 |
| `output_sharpening` | percent, 0–200, step 5; a plain number (not Auto) | 100 | Detail › Sharpening › Output: sharpening of the picture at the size it is shown and written (`docs/algorithms/04-output-sharpening.md`); 0 is off |
| `noise_reduction` | `Auto` or `{ luminance, color, detail }` 0–100 | Auto | Detail › Noise reduction |
| `curve` | luma + R, G, B: up to 16 control points each in [0,1]² | identity | Curve |
| `decode` | `{ neural_demosaic, highlight_recovery, lens_corrections, chromatic_aberration }` bools | all true | RAW › Decode |
| `white_offset_ev` | stops, −3…+3, step 0.05; a plain number (not Auto) | 0 | Light › White › Offset: added to the photo's **automatic white lift** (`docs/02-colour-pipeline.md` C12 step 1b); 0 is the automatic amount, and a negative offset of the amount or more gives no lift. A document without the field reads as 0. It belongs to the Grade group (the lift sits in front of ACES 2). Only the built-in ACES 2 display rendering has the lift. `ResolvedValues` also carries `white_ev`, the lift in effect (`auto.white_ev + offset`, never below 0), and `AutoEstimates.white_ev` the automatic amount |
| `rolloff` | `{ highlights 0–1, low_light 0–1 }` | 1, 0.5 | RAW › Range: `highlights` is the highlight range of C7 (`docs/02-colour-pipeline.md`: where the camera's clip lands, 0 as recorded, 1 at the ALEXA's EI 800 clip), `low_light` the toe. It works alongside the white lift (`white_offset_ev`) |
| `crop` | `Option<{ rect in [0,1]², angle_deg −45…45, aspect: Free/Original/1:1/4:5/16:9 }>` | None | Crop (header) |
| `orientation` | 0, 90, 180, 270 degrees, applied after the camera's EXIF orientation | 0 | Crop › Rotate 90° |
| `white_pick` | `Option<(x, y)>` sampled point | None | Pick white (header) |

"Auto" values resolve against `AutoEstimates` at render time. The UI shows the resolved value with an AUTO tag; moving a slider turns it to `Manual`, and a slider dropped on the Auto value sends Auto. "Reset" restores `Auto`. "Re-run" on the Automatic card recomputes `AutoEstimates` and resets exposure, the white lift's offset, white balance and noise reduction to Auto, and while Apple Auto acts on the photo also contrast, saturation, highlights, shadows, black and the shadows and highlights wheels (`docs/19-apple-auto.md` section 5).

#### Applying to several photos

The Look card's "Apply to" (This photo / Picked / All) copies `look` and `look_intensity` to those photos. Other controls apply to the current photo only. Every `update_settings` names its photo, and the core applies it to exactly that photo. `apply_look` names no photo (the core's own "current photo" is only what This photo means to it, and the webview never sends This photo: it sends the photo's `update_settings` alone). The webview therefore sends the copies in the order of the picks, one at a time, and reads again the settings of every photo whose look a copy changed (`docs/11-ui.md` *Apply to*).

**Copy settings and Paste settings** (`docs/11-ui.md` §Copy and paste settings) are the one way to move the other settings between photos.

- `copy_settings(photo_id)` makes a `SettingsPatch` from a photo's settings (`SettingsPatch::copied_from`). It takes every field but `crop`, `orientation` and `white_pick`, which belong to the photo's own geometry and pixels. Auto stays Auto. A white balance that came from a picked white (a white picked with the light preset and the white balance both Auto) becomes the manual Kelvin and tint it resolved to; a manual white balance next to a pick, and a preset chosen after a pick, are copied as they are.
- `paste_settings(photo_ids, patch)` applies a patch to many photos in one call (`Session::paste_settings`).
- The patch lives in the webview store for the running app only: never written to disk (hard rule 4), never put on the system clipboard.

Export can target the current photo, picked photos or all.

#### Settings in the core

Code: `ym-core`'s `settings.rs`.

- **Wire format.** `DevelopSettings`, its patch and `ResolvedSettings` have the serde shape of `src/api/types.ts` (snake_case, `kind`-tagged unions). A test compares the field names of every wire type with its interface in that file, so a drift fails in CI. The same JSON is the `ym:Settings` XMP property, the CLI's `--settings` and the golden manifest. `crates/ym-core/schema/settings.schema.json` is its JSON Schema (tested against what the core writes, against documents it refuses, and against the ranges the core clamps to).
- **Reading.** Missing fields take their defaults (a settings file may hold only what it changes). Unknown fields are an error (a typo is not a silent no-op). `look` is an `ym_looks::library::LookId`. `orientation` is a number of degrees; any multiple of 90 is read and normalised (−90 is 270). A plain number for `contrast`, `highlights`, `shadows`, `black` or `saturation` reads as Manual with that value, and 0 as Auto. Writing always writes every field. `DevelopSettings::from_json` / `from_value` read a document over the defaults of a photo opened with the user's default look (a document without `look` gets `default_look`, where `DevelopSettings::default()` alone says Neutral), clamp it like a patch and report the fields that were changed. A misspelt field or a value of the wrong type is `SettingsError::Parse` with serde's message. `ym render --settings` reads its file with `from_value`.
- **Output sharpening (`output_sharpening`).** An `f64` in percent, 0 to 200 (`OUTPUT_SHARPENING_RANGE`), default 100 (`OUTPUT_SHARPENING_DEFAULT`), a plain number and not an `AutoOr`: Reset (`SettingsPatch::reset`) and the UI's row Reset send 100. `SettingsField::OutputSharpening` belongs to `StageGroup::Output`, the last group, so a patch that changes it alone re-runs only the sharpening of the frames already rendered (*Render graph*, below). A document without the field reads as 100 (serde's struct default), `from_json` clamps like any field, and a value that is not a number is refused (`SettingsError::NotFinite("output_sharpening")`). It is written with every other field into the `ym:Settings` of an export (`ResolvedSettings::pinned_settings` keeps it as it is), into the CLI's `settings` object, `settings.schema.json` and `ResolvedValues`. Re-run (`reset_automatic`) leaves it alone. The UI row is Detail › Sharpening › Output (`docs/11-ui.md`); the CLI option is `--output-sharpening` (`docs/10-cli.md`).
- **Colour warp (`color_warp`).** `ColorWarp { pins }` of `ColorWarpPin`, which is `ym_color::warp::WarpPin` with serde (`From` both ways; `ColorWarp::to_color` gives the `WarpParams` of stage C9b).
  - Reading: every pin field but `source` may be left out (`target` is then `source`, `locked` false, `range` 0.04, `shadows` and `highlights` 1, `pivot_ev` and `exposure_ev` 0). `target: null`, a pin without `source` and unknown fields are refused. Writing writes every field.
  - Checking (`validated`, on every patch) keeps the first `WARP_MAX_PINS` (16) pins and clamps each with `WarpPin::clamped`: points scaled back to `|c| <= 0.5`, the range to 0.01…0.25, the weights to 0…1, the pivot to ±6 and the exposure to ±2 stops; a locked pin's target is its source and its weights, pivot and exposure the defaults. It reports `SettingsField::ColorWarp` in `clamped` when anything changed. A point that the scaling leaves one ulp outside the disc is moved inside, so a second check changes nothing and a later patch never reports the warp as changed. A number that is not finite in any pin is `SettingsError::NotFinite("color_warp.<field>")`.
  - `SettingsField::ColorWarp` (`"color_warp"`) belongs to `StageGroup::Grade` and is in `SettingsField::ALL`, so Copy settings takes it (it is not in `COPY_LEAVES_OUT`), Reset gives no pins, and a patch replaces the whole list (`{ "pins": [] }` removes them). `ResolvedValues::color_warp` and `pinned_settings` carry the pins as stored.
  - Rendering: `pipeline::color::warp_params` sets `ChainParams::warp` in `grade_plan` (the device path takes the pins from the chain's parameters). The grade key holds the warp as the chain prepares it (`Warp::new`), so a pin that moves re-renders, while locked pins alone, or a pin whose target is its source, leave the key and the frame as they are without them. No pins (and locked pins alone) skip the stage, so a render without the warp has the pixels it would have without the field. The `ym:Settings` an export records includes `"color_warp":{"pins":[]}`. The JSON Schema has `color_warp` (`$defs/warp_pin`, `$defs/chroma_point`).
  - Tests: `tests/settings.rs` (reading with and without the optional fields, 17 pins to 16, clamping, the locked target, non-finite values, idempotence on random points, patch and Reset, Copy settings), `tests/render_api.rs` (a warp moves the picture, locked pins alone and the Before view are byte for byte unchanged, the grade key, `chroma_scatter`, `sample_chroma`), `tests/render_graph.rs` (a warp rebuilds the grade group only), `tests/session.rs` (paste), `crates/ym-cli/tests/render.rs` and `parity.rs` (`--settings` with pins, app and `ym` byte for byte).
- **Lens (`lens`).** `LensSettings { kind, flares, edges, anamorphic }` (`ym_core::settings`; `kind` is `ym_lens::LensKind`, re-exported, with the seven ids as lower-case JSON strings), `LensPatch` (every part optional, like `RolloffPatch`) and the range `LENS_AMOUNT_RANGE` (0 to 100). Rendering is `docs/21-lens-character.md`; the setting is wired through the core, the schema, the shell, the command line and the mock, and a lens stage that does not exist yet renders nothing.
  - A document without `lens` reads as the default (`{ "kind": "default", "flares": 0, "edges": 0, "anamorphic": 0 }`); writing always writes every field, so the `ym:Settings` of every export, and the CLI's `settings` object, gain `"lens":{…}` (no pixel changes). An unknown `kind` or field is `SettingsError::Parse`; an amount outside 0 to 100 is clamped and reported in `PatchReport::clamped`; one that is not finite is `SettingsError::NotFinite("lens.flares")` (`lens.edges`, `lens.anamorphic`) and nothing is applied.
  - `SettingsField::Lens` (`"lens"`) belongs to the new `StageGroup::Lens` between `Raw` and `Color` and is in `SettingsField::ALL`, so Copy settings takes it (it is not in `COPY_LEAVES_OUT`) and "Apply to" does not (it copies the look only), Reset gives the default lens, and Re-run (`reset_automatic`) leaves it alone. `ResolvedValues::lens` and `pinned_settings` carry it as stored. `PhotoSource::for_before_view` sets it to the default: the Before view has no lens.
  - The lens data (`data/lenses`) are loaded once with the data folder (`SessionEnv::lens_library`, `ym_lens::LensLibrary::load`) and handed to the renderer in `RenderOptions::lens`; `Core::lens_list()` answers the IPC command `lens_list` and `ym lenses list` (`LensInfo`: kind, name, line, start Flares and Edges, version, design source). A missing or damaged `data/lenses` is no crash: the list is empty, `Core::lens_library()` says why, and a lens setting renders as no lens with the note "The lens data could not be loaded, so this photo has no lens effect." Choosing a lens moves Flares and Edges to its starting amounts in the card and in `--lens`; the core does not.
  - Tests: `tests/settings.rs` (reading with and without the field, partial patches, clamping, `NotFinite`, an unknown kind or field, Reset, Copy settings, the TypeScript field test with `LensSettings`, the schema), `tests/lens_setting.rs` (all three amounts at 0 are the bytes of a render without the field, for every kind, with the library and without; the Before view; a missing or damaged lens folder; `Core::lens_list`), `src-tauri/src/api/tests.rs` (`core_types_convert_without_loss`), `crates/ym-cli/tests/lenses.rs`.
- **Clamping.** A number outside its range is clamped, not refused; `PatchReport::clamped` names the fields. What is refused (`SettingsError`, nothing applied: patches are atomic) is a value that is not a number, a curve that is not a curve (2 to 16 points; they are clamped into the unit square and sorted by x; two points at one x are refused), a crop without area, and a Pick white spot the chain refuses. Colour-balance points longer than 1 are scaled back to the circle. A crop rectangle is moved and shrunk into the photo, and its angle clamped to ±45°. Steps (0.01 EV, 50 K) are the UI's granularity: the core keeps the value it is given, so `--wb 5025` means 5025 K.
- **Patch.** `SettingsPatch` is `Partial<DevelopSettings>`, with `crop: null` and `white_pick: null` clearing the field (absent leaves it).
  - Two generosities beyond the TypeScript type: `decode`, `rolloff`, `lens`, `color_balance` and `curve` may carry only the parts to change, and a manual `white_balance`, `sharpen` or `noise_reduction` may leave out a part, which then keeps its current resolved value (a patch with only `kelvin` on an Auto white balance is Manual at that Kelvin and the Auto tint: that is what moving one slider means).
  - `apply_patch` reports the fields that changed and the first render-stage group they affect (`StageGroup::{Raw, Lens, Color, Grade, Geometry, Output}`, the table in *Render graph*; `SettingsField::group()`), which is what the render graph invalidates by.
  - **Linked fields.** A patch with a `white_pick` also sets `light_source` and `white_balance` to Auto unless it says otherwise, so the picked spot renders neutral and the Kelvin and tint rows show the picked light (`docs/11-ui.md` Pick white; the core does it through the scene light, `docs/02-colour-pipeline.md` C1). The frontend records undo from the settings before and after a change, so the extra fields are restored with it. A preset (`light_source`) does not clear a pick: the chain applies the pick first and the preset after it, so Day, Shade and Tungsten override it, and LED and Fluorescent keep the picked chromaticity.
- **Pick white.** `white_pick` is `[x, y]` as fractions of the photo **as the camera oriented it** (EXIF orientation applied; before the user's crop, straighten and `orientation`, so the point keeps pointing at the same spot when those change). `auto::display_to_sensor` maps it back to the sensor. The session samples the 5 × 5 median of g0-balanced camera RGB at that point when the pick is set (`auto::pick_white_sample`: a cheap 3 × 3 same-colour demosaic on the raw data without the vignetting gain, which scales all three channels alike) and keeps the sample with the photo. `PhotoView::picked_white()` is what goes to the chain. The viewport converts clicks in the photo cell to this space before it sets the pick (*Native viewport*, below).
- **Resolving.** `ResolvedSettings::resolve(photo_id, settings, estimates, profile, picked_white)` builds the fused chain (`AutoEstimates::fused_chain`) and takes the Kelvin and tint from it. With `white_balance` Auto they therefore follow a light preset (Day 5500 K, Shade 7500 K, Tungsten 3200 K, tint 0; LED and Fluorescent keep the estimate's CCT) and a picked white. `AutoFlags` says which of exposure, contrast, highlights, shadows, black, the two colour-balance wheels, saturation, light source, white balance, sharpening and noise reduction are still Auto. The UI's rule that "white balance counts as automatic only when `white_balance` is Auto, `light_source` is Auto and no white was picked" sits on top of these flags.
- **`pinned_settings()`.** `ResolvedSettings::pinned_settings()` is the document an export records as `ym:Settings` and the CLI prints as `settings`. Exposure, sharpening and noise reduction that are Auto are written as the manual values they resolved to (a manual value renders exactly like the Auto it came from, and the file then says which numbers were used). White balance stays Auto when it was: a manual white balance at the shown Kelvin and tint does not render like Auto (the tint scales the green gain and Kelvin is rounded to 50 K), so the numbers it resolved to go in the `auto` object of the CLI output instead.
- **Session commands** (all on `Session`, same names as the IPC commands): `state()`, `set_current`, `set_picked`, `pick_all`, `clear_picks`, `settings(id)` (`get_settings`), `update_settings(id, patch)`, `replace_settings(id, settings)` (the CLI's `--settings`: clamped like a patch), `apply_look(scope, look, intensity)` (This photo is the current photo; skipped photos are left out; intensity clamped), `rerun_auto(id)`, `copy_settings(id)`, `paste_settings(ids, patch)`, `set_default_look`.
  - `rerun_auto(id)` recomputes the estimates on the calling thread and resets what Re-run resets (above); sharpening and everything else stay.
  - Settings live in memory with the photo and nowhere else (hard rule 4).
  - Commands may arrive from several threads (one per slider move; a Pick white decodes the photo again, which takes a while). `update_settings` applies its patch to the settings the photo has when it commits and starts again from the current ones if another command changed them meanwhile, so two commands on different fields both take effect in either order. `replace_settings` is the last word and never retries. A command on a session that was closed meanwhile returns `UnknownPhoto`.
- **Copy and paste** (`Session::copy_settings`, `Session::paste_settings`, `SettingsPatch::copied_from`, `COPY_LEAVES_OUT`).
  - The patch `copied_from` makes holds every part of every manual value (a manual white balance carries both Kelvin and tint, sharpening both amount and radius, and so on), so it applies to a photo that has no automatic estimates yet and needs no seeds.
  - `paste_settings` takes the state lock once and strips `crop`, `orientation` and `white_pick` from the patch whatever it says (a paste never touches the photo's own geometry, pick or sample). It skips a photo that is not in the session, failed or has no estimates yet (the same test as `Session::view`). It applies the patch to a copy of every other photo's settings first (`DevelopSettings::apply_patch`, the same code as `update_settings`, so the result is what `ym render --settings` would make of those settings: rule 6) and stores them only if every one applied, so a patch that cannot be applied changes nothing.
  - It answers `PasteReport { pasted: [{ id, before, after }], skipped }`. `before` and `after` are patches of just the fields that changed (`SettingsPatch::of_fields` of the old and the new settings), the pair the webview keeps as that photo's one undo step; both are empty when the photo already had the settings. Nothing is rendered or resolved in the call.
  - Cost: about 5 µs a photo in the core (59 photos took 98 µs; the same 59 with one `update_settings` each took 311 µs). The reason for the one call is the bridge, not the core: five hundred photos are one command, not a `get_settings` and an `update_settings` each.
  - `Core::paste_settings` refuses a look in the patch that is not in the library or whose file is gone, as `apply_look` does (`LooksError`), and changes nothing.
- **In the shell.** `AppApi::paste_settings` draws the viewport again if the current photo changed, queues a thumbnail for every photo that changed (`Thumbnails::refresh`, 300 ms settle, thumbnail priority behind the viewport: a paste onto five hundred photos does not render them on the foreground worker), and emits `settings_changed` once for each (the webview asks again only for a photo whose settings it holds that the paste did not report). `open_paths` and `open_dialog` answer with `SessionState.opened: { added, already_open }` (the counts of that call, `OpenOutcome`; null in every other answer), which is how Photos says "Added 3 photos. 2 were already open." when files are added to a session.

### Render graph

Each photo's render is a fixed chain of stages (`docs/02-colour-pipeline.md` and `docs/05-raw-processing.md`). Each stage has a cache key made from its inputs. Changing a setting invalidates only the stages after the first one it affects:

| Stage group | Invalidated by |
|---|---|
| Decode, black/white, clip mask | never within a session |
| Noise estimate, illuminant estimate, auto exposure, white lift estimate | Re-run |
| Vignetting, highlights, demosaic+denoise, detail controls, sharpening, lens resample (R3–R9) | `decode.*`, `noise_reduction`, `sharpen` |
| Lens (L1–L3; the bend L4 is geometry) | `lens`, and everything that invalidates the raw group; the quarter turns, which set the anamorphic axis (the canvas's horizontal) that the streaks, the starbursts and the bokeh ovals follow |
| WB, sensor map, ARRI matrix, AWG4, exposure, highlight range and toe, LogC4 (C1–C8) | `light_source`, `white_balance`, `white_pick`, `exposure`, `rolloff.*` |
| Primary grade, look, curve, white lift and ACES 2 (DRT), display encode (C9–C13) | everything else (except `output_sharpening`); the white lift's amount is the photo's automatic one plus `white_offset_ev` |
| Geometry (crop, straighten, orientation) | `crop`, `orientation`: applied when sampling for display and export; no pixel stage re-runs |
| Output sharpening (the last stage: the display-linear picture at the size it is shown or written, then C13) | `output_sharpening`; it also re-runs after every change above, because it sharpens what they made. Moving its slider re-runs only this stage on the display-linear frame already rendered. |

Most edits only re-run the last group, which is cheap (one pass of per-pixel maths and two LUT lookups) and must stay interactive (< 16 ms for the viewport at screen resolution).

#### The renderer and its keys

Code: `ym-core`'s `pipeline.rs` and `pipeline/*.rs`. A `Renderer` (cheap to clone, one per app or CLI run; `Renderer::new` returns `RenderError::Thread` only when the operating system cannot start the render thread) owns one `PhotoGraph` per photo. Every call takes a `PhotoSource`: a snapshot of one photo's decoded image, profile, estimates, settings and Pick white sample (`PhotoSource::from_session`, `from_view`, or the thumbnail hook's input).

A settings change is therefore just a new source. Each group compares the key of its inputs with the key of what it holds and recomputes only on a mismatch, so "a change invalidates only the groups after the first one it affects" is a property of the keys, not of notifications. `tests/render_graph.rs` pins it by counting what ran per request (`RenderReport::runs`, and `RenderReport::timings` for how long). Keys are the numbers themselves (an f64's `Debug` text is its shortest exact form), so equal keys mean equal pixels.

| Group | What it holds | Key |
|---|---|---|
| Decode | The photo's area cut out of the readout (`FrameData`: the camera's crop, else the active area, else the readout; copied only when it is not the whole readout; quantisation steps as f32). It lives for one request and is not cached: a copy per photo would be a second decoded mosaic beyond the session's three. | |
| Raw (R1, R3 to R9) | An `RgbImage` (planar f32, camera RGB balanced with `g0`) per reduction `k`. The noise sigma plane is cached on the noise model and `g0` alone. | `RawParams` (the gains `g0`, all four `decode.*` switches, the resolved noise reduction and sharpening, the ISO and the noise model) and `k`. Not the estimates' version: a burst median that moves only the exposure leaves this group alone. |
| Lens planes (`docs/21` §3) | The lens's planes and lists for the photo: the lens plane at `k_L`, the tail plane's FFT, the source list, the blur map and the ball list (`ym_lens::LensPlanes`), and the lens plan they were made for. One set per photo, kept in the graph **outside the evictable caches**, so that a pan or a slider drag can never push them out; counted in `PhotoGraph::bytes`. Made from the raw group at `k_L` (the largest power of two, at least 2, that keeps the long edge at or above 1024 px; 1 for the tiny frames of the tests) and the camera's focus point mapped through R9's crop. | The raw group's key, the lens and its data version, the camera's curves, the auto-cropped frame, the anamorphic axis, `g0` and the focus point. **Not the three amounts**: a slider move keeps the planes. |
| Lens (L1 to L3) | Caches after the lens, beside the raw group's, of host images on the CPU route and of device images on the device route: the field stage `Z = L2(L1(X))` on its own (`lens_field_groups`, `lens_field_tiles`) and the flare terms of L3 (`lens_flare_groups`, `lens_flare_tiles`, host `f64` planes), so a Flares move runs L3 alone and an Edges move runs L1 and L2; and three of the picture after the lens: `lens_groups` (the raw group at 1/k after `ym_lens::apply` over the whole frame: the fit view, thumbnails, look tiles and the histogram), and `lens_tiles` (the 512 px tiles of the 100 % view, each made from the raw region grown by the lens halo, `ym_lens::halo_px`, which the raw tiles' own cache serves). An export caches nothing: each tile is the raw region grown by the halo, one call, then `apply`. | The raw group's key, the lens text (`LensPlan::key`: lens, data versions, the three amounts, camera, frame, axis, f-number, `g0`) with the focus point, and `k` or the tile; the field caches' key is `LensPlan::field_key` (Edges and Anamorphic, not Flares) and the flare caches' `LensPlan::flares_key` (Flares and Anamorphic, not Edges). They are in the viewport frame key and the display-frame key too, so a slider move makes new pixels. |
| Colour (C1 to C8) | The fused matrix and the C7 parameters. | The scene light, `light_source`, `white_balance`, `white_pick` and its sample, `exposure`, `rolloff` and the noise model. |
| Grade (C9 to C13) | The prepared `ym_color::display::ColorChain`. | The colour key, the grade controls, the curves, the look (id, intensity and the LUT file's SHA-256), the display choice and the output. |
| Geometry | Not a group: orientation, straighten and crop are applied while sampling (*Geometry*, below), so changing them re-runs only the sampling and the per-pixel colour pass. | |

- **The lens and the counters.** With all three lens amounts at 0 (`LensSettings::is_identity`) no lens object exists for the request: no plan, no planes, no counter, no key text, the bytes of a render without the field. Otherwise `RenderReport::runs` counts `lens_planes` (the planes were made), `lens_groups` (a reduced group went through the lens), `lens_tiles` (a 100 % tile or an export tile did), `lens_fields` and `lens_flares` (the field stage and the flare terms ran, over a group or a tile: a Flares move gives `lens_flares` 1 and `lens_fields` 0), `timings.lens_ms` says how long, and `RenderReport::lens` (and `render_stats.lens`) carries `ym_lens::ReshapeReport`: `{ reshaped, reason }` with the reason `not_asked` or `little_blur`, for the Lens card's note; a frame served from the cache says it too. `tests/render_lens.rs` pins it: a slider move gives `raw_work() == 0`, `lens_planes == 0`, one lens group.
- **The lens on the device.** With `GpuStages::LENS` on and a view whose results stay on the device, the lens runs as kernels per tile (`pipeline/graph/lens_device.rs`; L1, L2 and L3 in `ym-gpu`, `docs/21` section 3): the tile's raw window is gathered from the raw group's pieces where they live, `Z` is cached as device images, and the result is a device image the sampler and the colour pass read without a readback. L1 is a kernel too, where it acts: twelve dispatches over L2's input rectangle (`docs/21` section 4, *As built (L1 on the device)*); the blur detector, the ball list and the map stay on the CPU, once per photo. Any other request reads the raw group back and runs the lens on the CPU; a device error in a lens tile falls back to the CPU lens with a note, and an export starts again wholly on the CPU.
- **Exports and determinism.** An export makes the lens's planes for itself, from the CPU reference raw group (`group_on_host`, or the CPU tile for the tiny frames), whatever device the renderer has and whatever the viewer cached, so the bytes of an export do not depend on what was looked at first and the app and `ym` write the same file (`tests/render_lens.rs`, `crates/ym-cli/tests/parity.rs`). The working-set reservation of an export grows by the lens's halo (`Job::lens_halo`).
- **Which graphs are kept.** The renderer keeps the graphs of the `MemoryBudget::max_photos` (3) most recently used photos and the graph of the photo being looked at (`Renderer::navigate`). The thumbnails of a folder that is still developing touch every photo in turn, and without the exception they would push the current photo's reduced images, tiles and finished frames out (`tests/render_review.rs`).
- **Image identity.** A graph remembers a fingerprint of the decoded image (size, levels, white balance and a few thousand samples) and drops everything when the image behind a photo id changes (a second session, a file replaced). `tests/render_scheduling.rs` checks it.
- **Why the colour and grade groups cache a plan, not an image.** The per-pixel stages are one fused pass (on the device one compute dispatch from camera RGB to RGBA8, on the CPU `ColorChain::render_u8`), so there is nothing between C8 and the display worth keeping: a cached C8 image would cost 4 floats a pixel, need `ym-color` to expose C9 from LogC4, and save about a fifth of a pass. What a colour or grade edit reuses is the raw group's image; what it re-runs is one pass over the pixels shown. The viewport frame cache (`FrameOptions::no_cache` to bypass) serves a repeated request without even that. With output sharpening on (the default) one more image is kept: the display-linear frame under the last stage, which is what makes the Output slider cheap (*Output sharpening*, below).

#### Raw group below full resolution

Code: `pipeline/raw.rs`. It serves the fit view, the thumbnails and the look tiles.

1. R1 and R4 run on the full mosaic.
2. R5 runs only on the 128-row bands that hold a clipped photosite (exactly `highlights::reconstruct`'s result; nothing at all when no photosite is clipped, which `RawPlan::has_clipped` answers by the same rule as R1's mask).
3. The photosites are binned `k × k` (`bin_mosaic`; a box filter is the demosaic at 1/k scale). Green is the mean of the block's greens. Red and blue are the mean over the block of their lattice interpolated linearly (a few partly weighted sites beyond the block), because the plain mean of a block's reds sits half a photosite up-left of its middle and that of its blues half a photosite down-right, which puts a red and blue fringe on every sharp edge. A linear ramp bins to three equal planes in every CFA phase.
4. The vignetting gain is applied per output pixel, after R5 (the gain is smooth, so `max(original, estimate)` commutes with it).
5. The bilateral denoise runs with the sigma the binning left: `ym_noise::sigma_map::sigma_quad` on the unvignetted data, then times the gain and `binning_noise_factor(k)`, which is `sqrt(sum_c w_c^2 v_c)` with the variance factor `v_c` of each colour's mean (`2 / k^2` for green; for red and blue the squared sum of the lattice weights per axis, squared; 0.52 at 1/2). The factor is measured against flat noisy mosaics in `pipeline/raw.rs`.
6. R9 resamples with the lens maps built for the smaller frame and the full-resolution auto crop scaled by 1/k.

**R7 and R8 are skipped below full resolution.** Radii and PSFs are in full-resolution pixels (`docs/05-raw-processing.md`) and binning has averaged the noise they cure; the fit view is therefore not sharpened, and the 100 % view is.

Measured: on a lens-blurred step the red and blue channels end up 15 and 9 display codes apart at 1/2 and 1/4 (85 and 22 with the plain block mean), and the chroma-edge energy of the nine real files' Fit views is 10 to 42 % lower than with the plain mean (`tests/render_review.rs`, `raw.rs`).

#### Full resolution

`ym_gpu::cpu::chain::Chain` runs, with the photo's levels at the area's parity:

1. The vignetting profile and `g0`.
2. The lateral CA correction on the mosaic (R5b, when the lens has a CA table).
3. The adaptive ratio demosaic (`DEFAULT_METHOD`). It gets `g0` and the noise model on the 16-bit working scale, read as zero noise when the model is unusable. Menon, LMMSE, bilinear and log-guided exist only through the test-only `RenderOptions::demosaic`.
4. The bilateral denoise (skipped when Luminance is 0) and the Detail controls.
5. R8: Richardson-Lucy, skipped for 0 iterations. Diffraction deconvolution runs only when the internal `PhotoSource::capture_method` asks and the photo has an f-number and a pixel pitch (`docs/05-raw-processing.md` R8).
6. The lens maps with their auto crop, and a full-frame sigma map from `sigma_quad`.

`highlight_recovery` is switched off inside the chain when the frame has no clipped photosite.

With a lens (`docs/21`) the full-resolution raw region of a tile is grown by the lens's halo (`ym_lens::halo_px`: the amounts and the photo's blur map decide it; 0 to a few hundred pixels at 50 MP) and goes through `ym_lens::apply`, which returns the rectangle asked for and gives any tiling the bits of the whole frame; the 100 % view caches the result as 512 px lens tiles, an export does not.

The 100 % view is made of 512 px tiles (`Chain::run_tile`) cached per `RawParams`. An export renders in tiles of 1024 px (`Renderer::render_full`), each tile's source rectangle found by mapping the tile through the geometry, so only one tile of intermediates is alive. A tile or a render that is cancelled leaves nothing half-made in a cache. The tile sizes (`RenderOptions::tile_px` 512, `full_tile_px` 1024) belong to the bit-identity contract of `AGENTS.md` rule 6: the stages are tile-exact to 0 or about 2e-7 (R8 beyond its halo cap), which can move a 16-bit code, so the app and the CLI must both keep the defaults.

#### Camera-processed files (Lossless M and S)

R1 uses one range (green's) for the three channels, then `g0`, the light bilateral denoise with a prior-only noise model, and R7 and R8 at full resolution. There is no R3, R5, R6 or R9: the camera applied its own shading and lens corrections to the pixels it wrote. Reduced scales bin the RGB image; the full-resolution image is the group at `k = 1`, and there are no tiles.

#### What a full render makes

`FullTarget` selects the output of a full render.

- **`DisplayLinear(output)`**: C1 to C12 for JPEG, TIFF and PNG, with the orientation (the camera's and the user's), straighten and crop **baked into the pixels**.
- **The lens (L1 to L3) is in all three targets**: each tile of a render is the raw region after the lens (`Job::lens_region`), so a DNG's raw pixels and its preview carry the lens, and the export tail's host tiles are the lens's output. A photo with no lens has none of it.
- **The two DNG targets** follow `docs/09-export.md` ("Pixels": the raw pixels keep the lens auto-crop only and the user's crop goes into the default crop tags; "90° steps go into the TIFF Orientation tag without rotating pixels") and `ym_export::dng::DngParams` (the raw and the preview "not rotated").
  - `DngLinearAwg4` is C1 to C11 decoded to linear AWG4 in the **buffer's own orientation**, uncropped, with the user's crop as `FullImage::window` (whole pixels, on the buffer, for the default crop tags) when there is no straighten angle. With an angle, the angle and the crop are baked in (a DNG has no rotated crop), still unturned, and `window` is `None`.
  - `DngPreview` is the sRGB display-linear picture in the same unturned orientation, cropped and straightened as the user sees it, so it has the size of the window.
  - The writer takes the `Orientation` tag from `ym_export::dng::compose_orientation` (the file's EXIF orientation and the user's quarter turns).
  - `Geometry::upright` makes the unturned geometry from the baked one (the lens's bend travels with it, its axis swapped when the buffer's sides are): the crop window is rounded on the canvas first and then mapped through the turns on the pixel lattice, so the three images hold exactly the same pixels, and a mirrored EXIF orientation (2, 4, 5, 7) reverses the sense of the angle. `tests/render_review.rs` and the unit test in `geometry.rs` check it against an independent forward map for all 8 orientations, 4 turns, three angles and fractional crops.

#### The Before view's raw front

The Before view (`docs/02-colour-pipeline.md` *The Before view*) leaves out R3. `PhotoSource::for_before_view` marks the source, `RawParams::vignetting` is off for a photo that has a vignetting table to apply, and the flag is part of the raw group's key, so the Before view's groups, tiles and display and frame entries are its own. The graph keeps two plans and two prepared chains per photo (with and without R3). The sigma plane, which is made before the gain, is shared. A photo with no table to apply shares the final picture's raw front.

#### Geometry

Code: `pipeline/geometry.rs` (the frontend's model is `src/fixtures/grade.ts`).

- The buffer is turned by the EXIF orientation, then by the user's `orientation` (clockwise quarter turns), into the *canvas*. `crop.angle_deg` rotates the photo about the canvas centre (clockwise positive; the canvas keeps its size, and its corners may be empty). `crop.rect` is a rectangle of the canvas, normalised.
- Every output pixel maps to a buffer position through one composed affine map and the dihedral turn. Resampling is a pixel copy where the grid lines up (orientation only), Catmull-Rom while rotating or magnifying, and a box-filtered bilinear when shrinking by more than 1.25.
- **The lens's bend (L4, `docs/21` section 4)** sits between the canvas position and the dihedral turn: when the photo's lens has a bend (Anamorphic above 0, or Edges above 0 on a lens with distortion), `Geometry::warp` (`SamplerWarp`, plain `Copy` data made once per photo by `Job::geometry()` from the lens plan, on the full-resolution canvas) replaces the canvas position `c` by the photo position it reads, `F^-1(c / z)`. The inside test stays on the canvas position, before the bend; straightening and the crop act on the bent, zoomed canvas. A bent grid is never a pixel copy, and its kernel is chosen by the largest shrink over the canvas (Catmull-Rom up to 1.25 as before, the box beyond). A source rectangle follows the bent boundary of the output rectangle instead of its four corners. Without a bend nothing in this section changes, down to the bytes.
- Pixels that fall outside the photo are the paper colour in a viewport frame and zero in a full render.
- **At full resolution the grid is on the pixel grid** (`Geometry::at_full_resolution`, used by the 100 % tiles and exports). The crop rectangle is a fraction of the canvas, so its edges fall between pixels, and sampling it as it is would resample every cropped export, and every 100 % view of a crop, half a pixel off and visibly softer than the uncropped export. The crop origin is therefore rounded to a whole pixel (kept inside the canvas), an output of exactly the crop's size maps one pixel to one pixel, and without a straighten angle sampling copies pixels. Viewport frames at exactly 100 % do the same for the pan (*Native viewport*, below). A reduced image keeps the fractional crop: it is resampled anyway.
- `white_pick` is in the oriented photo over the image area before the lens crop. `Renderer::pick_point_at` converts a click (through the straightening, the lens's bend, the turn and the lens maps) to it.

#### Output sharpening

Every display-referred picture ends with `docs/algorithms/04-output-sharpening.md` at the size it is shown or written: the screen preset (radius 1 px, threshold 1 level, method smooth gain, halo suppression 60 %, reference luma weights) at the amount `DevelopSettings::output_sharpening`. The slider "Output" is the spec's `amount_pct`, and nothing else is exposed.

- **Where.**
  - In a viewport frame, a thumbnail or a look tile (`pipeline/frame.rs`, module documentation) the colour chain stops at C12 and gives display-linear f32 for the view **grown by the stage's receptive field** on every side (`ViewSharpening::halo`, `SharpenPlan::halo_px`: 17 px at full size, less when shrunk, 21 at most). The stage sharpens the photo's own pixels in it, the border is cut off, and C13 (the transfer function and the dither at the pixel's place in the view) makes the 8-bit frame.
  - For an export the stage sits in the writer right after the Lanczos-3 resize (`docs/09-export.md`), and never for a DNG.
- **Scale.** `s = long edge shown / long edge rendered at full resolution` (`scale_for_edges`): for a viewport frame the zoom, for a thumbnail its size over the photo's, and 1 for the 100 % view and for a full-size export.
- **Above 100 %** the stage runs on the 100 % picture the view magnifies (`s = 1`: the pixel grid an export has, over the part of the final image the view covers plus the halo and the 2 px of the magnifying filter, clipped to the final image). The sharpened picture is then magnified to the view with the Catmull-Rom filter every magnified view uses, so the view at 200 % is the sharpened export at 200 % (`tests/render_sharpen.rs`, within one code). The colour chain then also runs at 100 %, and the magnifying is done in display-linear light.
- **The picture's own edge.** The stage works on the bounding box of the pixels inside the photo and mirrors at its edge, as it does at the edge of an export. A letterbox (paper around a fitted photo) therefore never leaks into the photo, and a straightened photo's empty corners are sharpened against black, as an export sharpens them.
- **Caching.** The display-linear frame of the grown view is cached per photo (`PhotoGraph::cached_display`, `DisplayFrame`, within the byte budget of the finished frames), keyed by everything the frame depends on **except the amount**: the raw group's key, the grade plan's key, geometry, the view and the halo. Moving the slider samples and colours nothing again and runs only the stage and C13 (`StageRuns { output_passes: 1, display_hits: 1 }`, `tests/render_sharpen.rs`). The finished 8-bit frame is cached too, with the amount in its key, so going back to a value that was shown is a frame-cache hit. An edit of any group above re-runs the stage on its new display-linear frame.
- **Amount 0** (and anything below 0.1 %) creates no stage: the plain road runs (sampling and the fused colour pass; `output_passes` stays 0).
- **Not sharpened:** the Before view, the histogram and the pixel readout (the Before view is not the finished picture; the others read colour at a point).
- **100 % view.** The tiles the grown view reads are prepared with it (`Renderer::prepare_tiles` reads the same frame geometry: the view grown by the halo, or the 100 % picture a magnified view covers). A view at 100 % shows exactly the part of the sharpened export it covers: a pan is a window of the whole picture within one code (the dither is the only difference), and a view the size of the photo equals the exported picture sharpened at `s = 1` and encoded (`tests/render_sharpen.rs`).
- **Where it runs.** The stage and C13 run on the device with the colour pass when the renderer's stages allow it (*View stages on the device*, below). Otherwise the colour pass (on the device or not) gives display-linear f32 to the host, and the stage and C13 run on the CPU.
- **The CPU stage is cancellable.** It runs in strips of 256 rows with the stage's receptive field of context above and below (the tiling contract of spec 04 section 5.2, so the result equals the whole-frame result bit for bit: `sharpening_tests::the_cpu_stage_in_strips_is_the_stage_on_the_whole_frame_bit_for_bit`), side by side on the rayon pool. Each strip looks at the request's cancel token before it starts, so a frame nobody wants any more stops after the strips that are running (about 0.45 Mpixel each on a 2 MP fit frame) instead of after the 0.5 to 2.6 s of the whole stage. Side by side the frame takes 1.02 to 1.17 times as long as in one piece (1764 × 1189 and 2948 × 1988 at the fit scale, four shared cores); strips one after the other take 1.8 times as long.
- **Cost.** See *Measured* under *Preview strategy*: on the CPU the stage is the largest cost of a finished frame, and every edit above it pays it again. The app cannot tell a drag from a click on `update_settings`, so there is no cheaper frame during a drag.

#### View stages on the device

Code: `pipeline/frame.rs` and `pipeline/frame/device.rs`. With a `ym_gpu::device::Gpu` in `RenderOptions::gpu`, a viewport frame, a thumbnail and a look tile are **one device job each**:

1. The raw group's pixels (the reduced image, or the window of 100 % tiles the view reads, cut to the rectangle the view samples from) are uploaded as camera RGB.
2. The sampler maps them through orientation, turns, straighten and crop into four planes (RGB and the letterbox mask).
3. The colour chain makes C1 to C12 from them with the persistent tables of `ym-gpu` (an unchanged chain uploads nothing; a look cube goes up once).
4. Output sharpening fused with C13 and the dither makes the RGBA8 frame, which is the one readback. A frame without the stage (amount 0, or the Before view as the colour kernel's RGBA8 stop) is the colour pass to C13 in one dispatch.

The job carries the request's priority (Viewport, TileNeural, Thumbnail; a thumbnail's job never passes a viewport's) and owner, and the render's own cancel token: a cancelled render or a navigation to another photo drops its queued job at once, and a running job ends within itself. Nothing in the job waits for the rayon pool (the upload planes, the sampler's grid and the photo's bounding box are made on the render thread).

- **Which stage runs where** follows `GpuStages` (`RenderOptions::gpu_stages`, set by the GPU policy). `Auto` runs `GpuStages::SIGNED_OFF`: the stages whose parity, tile and determinism suites pass on the Metal CI runners. That is the raw chain, the sampler, the colour pass, output sharpening and C13, the export tail and the raw front, so on a hardware adapter `Auto` runs every stage. `On` and `Require` run all.
  - `SAMPLER` off means the CPU samples (`sample_to_camera`, a call of `ym_gpu::cpu::sampler`, the kernel's CPU reference, kept bit for bit equal to the pipeline's own sampling by `sampler_tests`) and the sampled pixels are uploaded.
  - `COLOR` off means the CPU colours.
  - The stage and C13 are one fused device pass, which runs when `OUTPUT_SHARPEN`, `ENCODE` and `COLOR` are all on; otherwise both stay on the CPU, which then reads the display-linear frame back.
  - Every one of the 16 subsets renders the CPU's picture within the golden gates (`tests/render_gpu.rs` `every_subset_of_the_view_stages_renders_the_picture_of_the_cpu`).
- **A view magnified above 100 %** is one device job as well when the tail is on. The colour pass makes the 100 % picture under the view, the stage sharpens it at `s = 1` into a copy of it, and the magnification kernel (`ym_gpu::kernels::magnify`, `magnify.wgsl`) samples that copy with the Catmull-Rom filter for every device pixel of the view, paints the paper and encodes (C13 fused) into the RGBA8 frame.
  - The kernel reads taps the host makes (`ym_gpu::cpu::magnify::MagnifyMap`: per column and per row of the view, the four source indices with the edge replicated, the four `f32` weights, the mask sample and the inside test, in `f64` by the CPU reference's own code, `frame::magnify_map`) and does only the sums, in the CPU's order behind the fast-math barrier. The magnified values are the CPU reference's bit for bit, and the codes differ only by C13's rounding (`docs/17-gpu.md` section 16).
  - A magnified view whose 100 % picture (16 bytes a pixel) or RGBA8 frame does not fit `view_band_bytes` is cut into bands of rows like any other view (`band_split`). Each band reads the 100 % picture under its own rows. The stage's window is the whole view's box moved to where that picture starts (`magnified_band_window`, so the stage mirrors only at the photo's edge, as in one piece), the dither counts from the band's row, and a row's taps come from the band's map (the view's with the band's offset folded in, one `f64` rounding, as for the sampler's bands). The bands are the view in one piece (`tests/device_frames.rs`, straightened photos with the empty corner across the cuts included).
  - With the tail off, or a map that is not a scale and a translation (no viewport makes one), the view takes the host road: the device colours the 100 % picture and the CPU sharpens and magnifies it (`magnify_and_encode`, which runs the same reference, `magnify_with`).
- **The Output slider.** With the tail on the device, the display-linear frame (`DisplayFrame`, four f32 planes over the view grown by the stage's receptive field) **stays resident on the device** in the display cache, its bytes counted in the cache's budget and in the pool's ledger. A move of the slider is one small job that sharpens and encodes it again. Measured on lavapipe for a 576 × 384 cell: one submission, 2 KB of uniforms uploaded, the 884,736-byte frame read back, `StageRuns { output_passes: 1, display_hits: 1 }`, no sampler and no colour pass (`a_move_of_the_slider_is_one_small_job_with_no_sampler_and_no_colour_pass`). A resident frame is used only for a tail on the same device, a host frame only for a tail on the CPU.
- **Views too big for one binding.** A view whose frame (16 bytes a pixel over the grown view) or camera RGB does not fit `RenderOptions::view_band_bytes` (128 MiB, the design binding of `docs/17-gpu.md` section 4; a constant, so that the cut does not depend on the machine) is cut into horizontal bands. Each band is a view of its own with its own grown frame and the dither counted from the band's place in the view. The bands give the pixels of the view in one piece bit for bit (`a_view_cut_into_bands_has_the_pixels_of_the_view_in_one_piece`, also for the reduced images of a large photo). A 5K cell or the retina fit view of a 50 MP photo (its 1/2 image is 150 MB) is cut. Three rules keep the cut exact and on the device (`tests/render_gpu.rs`):
  1. **The stage's window is the whole view's.** The stage works on the bounding box of the pixels inside the photo, so every band is given the box of the frame of the view in one piece, cut to its rows (`band_window`). A band's own box is narrower where a straightened photo's edge is slanted, and the stage would mirror there where the view in one piece sees the black of the empty corner (4 codes at the slanted edge: `bands_of_a_tilted_photo_with_empty_corners_have_the_pixels_of_the_view_in_one_piece`). The box is found from the geometry alone (`prepare_view_with(.., pixels: false)` assembles no tile) and is part of a band's display-cache key.
  2. **Whichever stages are on.** The cut is made whenever the colour pass is on the device. With the sampler off the CPU samples each band and uploads it. With the tail off (`Auto` with the colour pass alone) the device makes the display-linear frame in bands of rows (`build_display_in_bands`: the chain is pointwise) and the CPU stage then takes the whole frame.
  3. **A band of paper** (a panorama in a tall cell) reads nothing of the photo, uploads nothing and is the device's like any other band (`Done::paper`).
- **Not on the device yet.** A rotated strip is read through its bounding box, `width * tan(angle)` rows tall even for one row. A view whose single row would still read more than one binding (about 8000 pixels wide at a tilt of 10 degrees, at 100 %) cannot be cut by rows and is made on the CPU, `Require` too. Cutting such a view into tiles would put it on the device.
- **Faults.** A device error makes the whole frame again on the CPU with the note "The graphics card could not render this frame, so the CPU did: ..." (never a frame of both, also when one band of a cut view fails). A shader that does not compile switches the view stages off for the process. Any other error but a buffer that does not fit (tried once more after the pool and the colour tables were let go, then the CPU for that frame) ends the route for the photo. `Require` fails the request with `RenderError::Gpu` instead.
- **Histogram.** Its sample comes from the device sampler when the sampler is on, and the counting stays on the CPU (`docs/17-gpu.md` section 11, the histogram row).
- **Measured.** Against the CPU reference on Mesa lavapipe (`tests/render_gpu.rs`, the raw chain on the CPU so that only the view stages differ): fit, retina, 100 %, a straightened crop, a quarter turn, amounts 0 and 200, and a magnified view all have ΔE00 on the golden runner's 1/4 downsample of median 0, p99 0, max 0.082, at most one code of 255 and a mean of at most 7.1e-5 codes.

The raw stages are on the device per `docs/05-raw-processing.md`.

#### Resident caches

Code: `pipeline/graph.rs`, `pipeline/frame.rs`, `pipeline/frame/device.rs`. With a device that runs the sampler (`GpuStages::SAMPLER`) and `RenderOptions::resident_caches` on (the default), the caches of the raw group hold `Backing { Cpu(Arc<RgbImage>), Gpu(Arc<ResidentImage>) }`:

- the reduced group `Job::group_backing(k)` as the bands of rows the device made it in (`GpuReduced`, one job per slab of work, so that a viewport frame queued behind it waits for one step and not for the group);
- the tiles of the 100 % view as the images `GpuChain::run_tile_device` left on the device.

A view reads them in place. The sampler's input is the window of the group the view samples from, **gathered with the copy kernel** (`ym_gpu::kernels::copy::encode_gather`: a window that is one piece is that piece, else a copy made in the sampling job), which moves samples and changes none, so the frame is the one the host round trip (read the group back, upload the window) makes bit for bit. The fit frame of a photo reads back its RGBA8 frame and nothing else, and a 100 % frame over cached tiles uploads nothing.

- **One side only.** An entry is on one side only. A consumer that needs host pixels (the CPU colour pass, a device that failed under the frame, a device other than the one the pieces are on) reads the pieces back and **replaces the entry with the host image** (`Job::demote`: the cache's byte count is then the memory the entry takes where it is). If the device cannot hand them over (lost, out of memory), the CPU makes the group or the tile again, with a note, and the cache keeps that.
- **The readout under the pointer** (`Renderer::sample_pixel`, asked for on every move of the mouse) needs host pixels only when the pixels are on the host. Pixels on the device are sampled there (`sample_view_in_place`) and one pixel comes back, so moving the mouse over the photo does not bring the caches back to the host and the next frame uploads nothing.
- **Budgets.** Device bytes are counted in the same `ByteLru` budgets as host bytes, with the budget of a cache that keeps images on the device clamped to an eighth of the device budget (`min(cpu_bytes / 6, gpu_bytes / 8)`), so the groups and tiles of a photo cannot fill the pool the working buffers of a render need. Leaving the photo drops its tiles (`release_full_resolution`) and forgetting it drops everything; the pool's ledger is then back at the resident set (the raw words of at most two photos).
- **Warm-up.** The work ahead of a view (`Job::warm_tiles`) leaves its tiles on the device and brings no pixel to the host.
- **Neighbours.** `Renderer::prefetch_raw` (queued by `spawn_raw_prefetch` at `Priority::Thumbnail`) uploads the raw words of a photo the person has not reached yet, only while fewer than `RESIDENT_PHOTOS` (2) photos have theirs on the device, so that the photo being looked at and the one before it keep theirs. A thumbnail's own upload follows the same rule of the two-photo LRU.

The numbers (readbacks, submissions, the ledger) are in `docs/05-raw-processing.md` "Device residency".

#### Look and display sources

`LookProvider` (implemented for `ym_looks::library::Library`, `RwLock<Library>` and `Mutex<Library>`; `NeutralOnly` for tests) returns a look with a fingerprint. A look that cannot be loaded renders as Neutral in the viewport, with the library's words ("File missing or changed. Import again.") in `RenderReport::notes` of **every** request that uses the plan (a cached plan or frame carries its notes too, as does a cached lens plan with its "table ignored" lines). For a look tile it is an error (`RenderError::Look`). `Renderer::set_display(DisplayChoice::{Aces2, ArriLut(id)})` is Preferences › Display rendering.

#### Noise floor of C7

The read noise of the photo's model (`sqrt(b + sigma_row^2 + q^2 / 12)`), balanced by `g0`, is carried through the fused matrix (exposure included) channel by channel as independent noise, and averaged over the three AWG4 channels.

### Preview strategy

- **Viewport at fit.** Classical demosaic at reduced resolution, full colour pipeline. With `z` the zoom in device pixels per full-resolution image pixel, `z <= 25 %` uses the 1/4 image, `z <= 50 %` the 1/2 image, and anything larger the full-resolution tiles (`frame::scale_for_zoom`). The image therefore has at least as many pixels as the viewport shows and is never upscaled: choosing the scale the other way round would shrink a 6000 px photo to 1500 px for a 1440 px cell and blur it.
- **Viewport at 100 % and above.** Full-resolution 512 px tiles of the visible area plus a margin, computed in the background. A frame asked for with `coarse_only` shows the 1/2 image until the tiles are ready (below).
- **Thumbnails.** Rendered once after open and again after a look, crop or turn change, at most 240 px on the long edge. A thumbnail uses the largest of 1/8, 1/4 and 1/2 that leaves twice the thumbnail size **of the picture it shows** (after the crop: a crop to a quarter of the width is made from a finer image than the whole photo; full resolution for small photos or small crops). It is rendered in **sRGB** (the webview shows a PNG without a profile as sRGB; the viewport is Display P3) with the photo's crop, straighten and turn applied, and is never enlarged.
- **Export.** Full resolution, in tiles (*Render graph*).

**The demosaic path.** The learned demosaic and denoise network (`docs/07-ml-models.md`) is not installed, so every render uses the classical path. `render_stats.path` says `classical`, and `RenderReport::notes` records "No trained demosaic model is installed yet, so the classical demosaic was used." (nothing when the user switched `decode.neural_demosaic` off). The header's status text ("Classical · 42 ms · Display P3") shows the active demosaic path, the last render time and the output space. When the network is installed, the visible tiles of the 100 % view are computed with it in the background, and the classical result shows until each tile is ready.

- **Coarse frames.** `Renderer::viewport_frame` can be asked for a `coarse_only` frame. When the zoom needs full-resolution tiles that are not cached, it serves the 1/2 image at once and marks the frame `sharp: false`; the app then queues `Renderer::spawn_tiles` at tile priority and asks again.
- **Render statistics.** `Renderer::last_stats` is the `render_stats` of the last **viewport frame** of a photo (a cached frame included). Thumbnails, look tiles, the histogram, pixel readouts and exports of the same photo do not replace it (an export's 17 s must not become the header's status text).
- **Leaving a photo.** `Renderer::navigate` frees the full-resolution intermediates (tiles, the prepared chain and its sigma map) of the photo left. Its reduced images stay, so going back is a colour pass.
- **Preview buffers** are kept for the three most recent photos and, for every photo, as the thumbnail PNG the app holds: a 24 MP quarter-resolution image is 18 MB, so five hundred photos would be 9 GB.

#### Measured

`cargo run -p ym-core --example perf`, 24 MP. The photo is a synthetic 6000 × 4000 uncompressed 14-bit ARW (a gradient with noise, lens tables, ISO 400, the Auto settings, no look). The machine has 4 x86-64 cores shared with other jobs, and the workspace crates are built at opt-level 3 without debug assertions (`CARGO_PROFILE_DEV_OPT_LEVEL=3 CARGO_PROFILE_DEV_DEBUG_ASSERTIONS=false CARGO_PROFILE_DEV_OVERFLOW_CHECKS=false`; not the LTO release profile). Two runs agree within about 10 %; the first is given. Output sharpening is off (amount 0) in this table; the next table has it on.

| Step | Time on 4 CPU cores | Where it goes |
|---|---|---|
| Open and develop (decode, estimates) | 171 ms | |
| Fit preview, 1440 × 960 cell (1/4 image), cold | 1111 ms | noise sigma plane 177, raw group R1 to R9 418, sampling 45, colour pass 447 |
| the same, warm (colour pass again) | 496 ms | sampling 39, colour pass 457 |
| Colour-only edit (a grade control) | 465 to 581 ms | no raw work (the harness asserts it), the colour pass |
| White-balance edit | 520 ms | colour and grade plans, the colour pass |
| Fit preview, 2880 × 1920 cell (retina, 1/2 image), cold / warm | 3361 / 1957 ms | raw group 1171, sampling 160, colour pass 1818 |
| 100 % view, 1440 × 960 cell, cold (12 tiles) / warm | 2.7 s / 449 ms | adaptive ratio demosaic and bilateral per 512 px tile (2.3 s cold with the Menon demosaic) |
| Thumbnail, cold / raw group cached | 382 / 27 ms | |
| Full-resolution render (display-linear f32), 24 MP | 22.2 s | adaptive ratio demosaic; 17.4 s with Menon, whose peak ledger is 559 MiB |

Reading them against the targets of `docs/05-raw-processing.md`:

- The full-resolution CPU render extrapolates linearly to about 46 s for 50 MP (an extrapolation, not a measurement), inside the CPU-only target of 90 s.
- On the CPU path the colour-only edit (16 ms target) and the cold fit preview are not met: the per-pixel colour stages run in f64 on the CPU at roughly 3 Mpixel/s. The colour-only target depends on the device colour pass.
- With `--gpu` on Mesa llvmpipe (software Vulkan, which says nothing about Metal): warm fit 174 ms, colour-only edits 166 to 197 ms, and 3278 ms for the first pass because it builds the pipelines.
- Timings on Apple hardware are Mac-only checks (`docs/13-testing.md`).

**Output sharpening on the CPU.** The same synthetic frame, 4 x86-64 cores shared with other jobs at a load average of 5 to 7 (against 1.7 to 2.5 for the table above, so wall times are two to three times what an idle machine gives and the two tables are not comparable; `cargo run -p ym-core --example perf`, the rows under "output sharpening"). The stage ran on the CPU in this measurement, also when the colour pass ran on a device.

| Step | Time |
|---|---|
| Output slider move on a 2 MP fit frame (1730 × 1155, the 1/2 raw group, `s = 0.29`; display-linear picture cached): the stage plus C13 | 825 ms (840 ms the whole request; 802 to 861 over five moves) |
| Output slider move on a 512 px tile of the 100 % view (`s = 1`, 546 px with the halo) | 178 ms (179 ms the whole request; 148 to 209) |
| The same frames without the stage (the plain road): the CPU colour pass alone | 1192 ms and 157 ms |
| Warm fit frame of the 1440 × 960 cell, stage on | 849 ms, 306 of them in the stage (`StageTimings::output_ms`) |
| Retina cell, stage on | 5.9 s, 2.6 s of them in the stage |
| Colour-only edit, stage on | 848 to 1157 ms |
| Cold frame, stage on | 2165 ms |

Every edit above the stage re-runs it (it is the last stage), so the colour-only target of `docs/05-raw-processing.md` (< 16 ms) needs the stage and the colour pass on the device (*View stages on the device*). Exports: the stage costs about the render's own order of magnitude per megapixel on the CPU (`docs/algorithms/04-output-sharpening.md` section 9: 220 to 450 ms per megapixel at the full size on four shared cores, 83 ms with halo suppression off).

### Scheduling and threads

- **Decode pool:** `min(available_parallelism − 2, 6)` threads, CPU only. Decoding and metadata parsing never touch the GPU.
- **GPU thread:** one thread owns the wgpu `Device` and `Queue`. All GPU work is submitted through a priority queue: viewport frames > visible-tile neural passes > thumbnails > export.
- **Cancellation:** every job carries a token. Moving to another photo cancels its pending viewport and tile jobs, and the requests of the panels for it (the Look tab's tiles and the histogram). Filmstrip thumbnails and export jobs are never cancelled by navigation. Taking a photo out of the filmstrip cancels all of its jobs but its exports (*Removing photos*, above).
- **Memory budget:** at most 3 decoded mosaics (current, previous, next) and the preview buffers of *Preview strategy*. Full-resolution intermediates live only while a tile job needs them. Defaults: 4 GiB GPU, 6 GiB CPU; the actual limits are read from wgpu and the OS and scaled down (the CPU budget is at most half of the installed RAM, 512 MiB at least).
- **Shared memory** (`docs/17-gpu.md` section 8). On a machine whose GPU is the CPU's memory (Apple silicon; wgpu's `DeviceType::IntegratedGpu`) the two budgets come out of one pool (`MemoryBudget::for_unified_memory`). The GPU gets `clamp(RAM / 4, 512 MiB, 4 GiB)` and the CPU budget becomes `min(its own, 0.6 × RAM − GPU)`, so together they stay under 0.6 of the RAM and the CPU loses only what that bound takes.

  | RAM | GPU budget | CPU budget |
  |---|---|---|
  | 8 GB | 2 GiB | 2.8 GiB |
  | 16 GB | 4 GiB | 5.6 GiB |
  | 24 GB and up | 4 GiB | 6 GiB |

  A GPU with its own memory keeps 4 GiB and the CPU's 6 GiB. The device is opened with the GPU figure (`MemoryBudget::gpu_bytes` goes into `GpuConfig::memory_budget_bytes`). `ym_core::app::gpu::open_gpu`, which both front ends call, opens it again with the shared-memory figure when the first budget turns out to be wrong for the adapter, and the renderer's CPU budget follows (`Renderer::set_budget`). The rule and its constants are in `scheduler.rs`.

#### The job scheduler

Code: `ym-core`'s `scheduler.rs`.

- **Foreground worker.** Viewport, tile and thumbnail jobs run on one worker thread through `ym_gpu::queue::Scheduler` (the queue that fronts the GPU thread: highest priority first, equal priorities in submission order, no pre-emption, tokens, panics caught), so the ordering and cancellation rules are the ones its tests pin. Each job is data-parallel inside (rayon, scoped threads), so a second *foreground* worker would only compete for the same cores.
- **The export worker gives way to the foreground** (`JobScheduler`). An export job starts only when no foreground job is queued or running, and `Renderer::spawn_full` looks again before every tile (`JobScheduler::wait_for_foreground`). An export is one long job that nothing else could pass (4.5 s for a 6 MP test photo, about 36 s for 50 MP on four cores), so without this a viewport frame asked for during an export would wait for all of it. While an export tile runs, a foreground job shares the cores with it; the overlap is bounded by one tile.
- **Tokens and errors.** The two workers keep the same `JobError` and token rules. Navigation cancels only foreground jobs, `cancel_all` and `shutdown` reach both, and a cancelled export that is still waiting is called with its token set and returns `Cancelled`.
- **Priorities and spawning.** `Priority::{Viewport, TileNeural, Thumbnail, Export}`. `Renderer::spawn_viewport_frame`, `spawn_tiles`, `spawn_thumbnail`, `spawn_look_thumbnail` and `spawn_full` submit at those priorities and return a `JobHandle`. `Renderer::thumbnail_renderer()` is the session's thumbnail hook: it queues the render at thumbnail priority and waits, polling the session's cancel flag, so a viewport frame asked for meanwhile runs first.
- **Navigation.** `Renderer::navigate(photo)` cancels the pending and running viewport and tile jobs of the photo being left (their tokens are set; the render checks them between stages and tiles) and releases its full-resolution intermediates. Filmstrip thumbnail and export jobs are untouched, and an export stops only through its own handle.
- **Panel requests are superseded** (`JobScheduler::submit_superseded`, used by `Renderer::spawn_look_thumbnail` and `spawn_histogram`). The Look tab asks for a tile per look and the Light tab for a histogram for every photo the person steps to, and a queued job holds its photo's decoded mosaic. Stepping faster than about 120 ms a tile (24 MP) would pile up jobs and mosaics for photos nobody is looking at, and the current photo's own tiles would wait behind them. Navigating away from a photo therefore cancels its queued panel requests, **drops their closures and with them the mosaic at once**, and sets the token of the one that runs. The handle resolves as `JobError::Cancelled`, and the app answers "The render was cancelled.", which the webview ignores. A request for the photo being moved to is never cancelled, whatever the order in which the move and the request arrive (`tests/render_scheduling.rs`).
- **Memory.** `MemoryBudget { max_mosaics: 3, max_photos: 3, cpu_bytes, gpu_bytes }`. `detect()` gives 6 GiB CPU scaled down to half the RAM and 4 GiB GPU. `max_mosaics` is the session's figure (`MOSAIC_CACHE_LEN`) and `gpu_bytes` is the budget the application opens the GPU context with; the renderer reads neither. The decoded mosaics are the session's. The render caches are byte-bounded LRU maps (`ByteLru`: reduced images one sixth of the CPU budget per photo, tiles one sixth, finished frames one thirty-second). A full-resolution render first reserves the output image plus one tile's working set in the `MemoryLedger`, and refuses with `RenderError::Memory` ("There is not enough memory to render this photo at full size. Close other apps or export a smaller size.") instead of starting.

### GPU tiling

- A 50 MP frame in RGBA f32 is about 0.8 GiB and exceeds wgpu's default 256 MiB buffer and 8192 px texture limits. Frames are processed in tiles of 1024 or 2048 px (configurable) with halos: half the network's receptive field (computed from the architecture; 64 full-resolution px for the starting U-Net), plus the resample's maximum displacement for the lens stage. A tiled-versus-whole-image test guards against seams.
- **Storage format f32** (`docs/17-gpu.md` section 2). Every image plane that crosses a kernel boundary is f32 in memory, in every mode, and `StoragePreference::Auto` resolves to `StorageMode::F32` on every device. The two f16 modes (`array<f16>` with `SHADER_F16`, or two halves per `u32` with `pack2x16float`) stay behind `StoragePreference::Force` for the legacy kernels and their tests.
  - Why: f16 storage differed from the CPU reference by 7.9e-4 to 6e-3 (`docs/05-raw-processing.md`, "f16 storage and the 1e-3 gate"), and the raw-algorithm kernels need f32.
  - The price is twice the plane memory: the 50 MP spike chain peaks at 0.355 GiB of pool ledger with f32 (0.28 GiB with f16 storage), measured on lavapipe by `gpu_spike_50mp`, against the 1.5 GiB gate. The full device chain has its own gate (`docs/17-gpu.md` section 8).
  - Raw words are `u16` pairs in a `u32`; look cubes and tables are f32. Per-pixel colour maths runs in f32 registers.
- Tiles are independent after decode, so export runs tile by tile and writes rows to the encoder as soon as a tile row is complete.

### Export, preferences and looks

Code: `ym-core`'s `export.rs`, `export_queue.rs`, `cli_preview.rs`, `preferences.rs`, `looks.rs` and `app.rs`. The options, the queue rules and the file formats are specified in `docs/09-export.md`. This section describes how the core is put together and what the front ends call.

#### One export function

`export::export_photo(&ExportContext, &ExportPhoto, &ExportParams, &ExportControl) -> Result<ExportReport, ExportError>` is the only thing that turns a photo into a file. The app's queue and the CLI both call it, so the same photo, settings and options are the same bytes whichever front end made them. `tests/export.rs` runs three contexts on three threads at once, and a second process, and they give identical JPEG and DNG bytes; two exports of a real file, in all four formats, repeat exactly.

- `ExportPhoto` is the render's `PhotoSource` (decoded image, profile, estimates, the settings that are exported) plus the path, probe metadata, camera match and `ResolvedSettings` the metadata needs. `ExportPhoto::from_session` builds one; `PhotoSnapshot::take` plus `with_raw` is the same in two steps (the queue snapshots at queue time and reads the image when the photo's turn comes).
- `ExportParams` is the job's options, the `{seq}` of the photo and the preference "Remove location from exports" **as it was when the job was queued**.
- `ExportControl` is a `CancelToken` and a progress callback (0 to 1, any thread, not always in order).
- The CLI starts after `Session::wait_idle` and builds its core without a thumbnail renderer (`CoreOptions::thumbnails = false`): an export's render gives way to foreground work, thumbnails included, so thumbnails would gain it nothing. The app keeps its thumbnails and may start an export earlier, because the queue waits per photo for the shared estimates to be final (*Session*, "App/CLI parity for early exports"). An early export waits only for estimates, which come before a photo's thumbnail, and its render then yields to foreground work between tiles.

#### What `export_photo` does

1. **Look and display LUT.** The look and the chosen display LUT must load. **An export never falls back to Neutral or ACES 2 the way the viewport does**, because the file's XMP says which look and transform made it. A look that is gone is `ExportError::Look("File missing or changed. Import again.")` (CLI exit 3).
   - A DNG of a look with its own display rendering is made, approximately (`docs/09-export.md` "What a DNG carries"): its pixels are the scene values that ACES 2 turns into the look's picture. `ExportEstimate::dng_approximate` tells the panel when a job has such a DNG.
   - The look's and the display LUT's fingerprints are taken again once the pixels are rendered (a render takes seconds to minutes, and a DNG's two renders even longer). A look replaced, removed or switched, or a Display rendering preference changed in between, fails that photo with "The look or the display rendering changed while this photo was exporting. Export it again." (`CHANGED_DURING_EXPORT_MESSAGE`, kind `look`, CLI exit 3), instead of writing a file whose XMP names a look or transform it was not made with.
   - A panic below `export_photo` (a writer, a look provider) is that photo's `ExportError::Render` ("This photo can't be exported because of an unexpected problem…", logged with the panic text), never an unwinding queue thread.
2. **Name and folder.** `{name}`, `{look}` (the slug of the photo's look id, also when the look is gone), `{date}` (capture date), `{seq}`, `{camera}` (slug of the body); the folder from `ym_export::naming`. With "If the file exists: Skip" and a file in the way, **nothing is rendered**.
3. **Render**, on the render scheduler's export worker (`Renderer::spawn_full_reporting`: gives way to the viewport between tiles, reports the fraction of tiles done).
   - JPEG, TIFF and PNG take `FullTarget::DisplayLinear(output)` for the chosen colour space and hand it to `ym_export::write_export`, which resizes in linear light and does C13.
   - A DNG takes `DngLinearAwg4` and `DngPreview` (both rendered with R8 capture sharpening off, `FullTarget::is_dng`): the raw pixels unturned with the user's crop in `FullImage::window` (`CropRect` of the default crop tags; none when a straighten angle baked the crop in), and the `Orientation` tag from `compose_orientation(camera, user turns)`. A DNG therefore costs two renders; one pass that makes both is not implemented.
   - The DNG profile is the fitted one, `FittedProfile::embedded()` (`data/dng/profile.json`, made by `tools/dngfit` and compiled in; `ExportContext::new` holds it, `with_dng_profile` replaces it), written with `BaselineExposure::Auto { fit_offset_ev }`, so the app and the CLI write the same DNG bytes (`docs/09-export.md`). `DngProfile::identity()` is for structure tests only.
4. **Metadata.** EXIF from the probe (`exif_of`: never copied from the file; GPS left out when the preference is on, both here and in the writer). XMP with the `ym:` properties, all dates the capture time:
   - `PipelineVersion`; `Profile` and `ProfileVersion` of the profile the render used; `ProfileMatch` (`exact`, `sensor family`, `default`, `forced`).
   - `Look`; `LookSHA256` (the hash of the look's file; for a built-in look the hash of `data/looks/builtin/<slug>.json`, absent for a look with no file); `LookIntensity`.
   - `DisplayTransform`: `aces2-sdr100-rec709lim` for sRGB, `aces2-sdr100-p3lim` for Display P3, `aces2-sdr100-p3lim-rec2020` for Rec.2020, or the id of the imported display LUT (`arri-display:logc4:rec709-g24`). A DNG records the sRGB one, which is what its preview is.
   - `Models` (absent: the classical demosaic has none; `docs/07-ml-models.md` fills it when the network runs).
   - `Auto` (which Auto made the photo's automatic values, `rule` or `apple`) and `AppleAuto` (what Apple Auto said about the photo, only when it was on for it; `docs/19-apple-auto.md` section 9).
   - `Settings` (`ResolvedSettings::pinned_settings()`, *DevelopSettings* above).
   - A DNG's scaling (`x_white`, clipped fractions) goes to the log at info level.
5. **Progress.** Render 0 to 75 % of the photo (a DNG: raw 0 to 35 %, preview 35 to 70 %), then the writer. Cancel is polled between render tiles and between writer bands (about a million pixels), and the temporary file is removed.

#### Estimates

`export::estimate` is the `export_estimate` size and time.

- **Size.** `ym_export::estimate_bytes_with_detail` works out the output size from the probe alone (`estimated_output_size`: the camera's crop, orientation, the user's turn and crop; not the lens auto-crop or straightening) and the detail statistic of the photo's newest thumbnail (`DetailStat::measure_rgb8` of the 240 px sRGB render, never the embedded camera JPEG, which is the camera's own rendering). The renderer keeps the statistic per photo when it makes a thumbnail (`Renderer::thumbnail_png`, so the session's first thumbnail and every refresh after a look, crop or turn update it; a look tile does not; it goes with the session), and `ExportQueue::estimate` reads it with `Renderer::thumbnail_detail`. A photo without a thumbnail yet is estimated as a picture of typical detail (`None`, what `estimate_bytes` assumes).
- **Calibration.** The app's thumbnails come from a reduced raw group, not from a box average of the full render the estimate was calibrated on. Measured on the nine real files (`tests/export_real.rs`, ignored; `cargo test -p ym-core --test export_real the_detail_statistic -- --ignored --nocapture`), the thumbnail's statistic over the full-size render's is 0.98 (A300), 0.96 (A500), 0.85 (ILCE-5000), 0.99 (ILCE-6300), 0.94 (A7 II), 1.02 (A7 III), 1.10 (A7 IV), 0.89 (NEX-5N), 0.97 (NEX-7), geometric mean 0.965. A quality 92 estimate (size grows as the statistic to the power 0.46) therefore moves by −7 % at worst, +4 % at best and −1.6 % on average.
- **Known limit.** On the nine real files (183 MP in all) the size estimate said 77 MB and the files came to 30 MB. A denoised camera frame at quality 92 came to 0.7 to 2.7 bits a pixel, 1.3 on average over the nine, where the table of `docs/09-export.md` (from film scans) says 3.3. That table is not yet re-measured on camera frames.
- **Time.** `ExportSpeed` gives the seconds. It starts from a render of 17.44 s for 24 MP (`RENDER_S_PER_MP_PRIOR`, the CPU render measured with the Menon demosaic; the adaptive ratio default measures 22.2 s, so the first estimate runs low) and the encode timings of `docs/09-export.md`, and is replaced by the speed of this machine after the first exported photo (the first sample outright, later ones as a running mean). A DNG counts two renders. `ExportSpeed` keeps one set of numbers per path, the CPU reference and the export tail on the device (a photo that fell back to the CPU teaches only the CPU set), and the estimate reads the device's when the renderer runs the tail. The device set starts as the CPU's until measured on a Mac (`docs/09-export.md`, "As built (export tail on the device)"). On the nine real files the starting numbers said 183 s for the job and the job took 200 s.

#### The queue

`ExportQueue::start(session, job, remove_location)` snapshots every photo of the scope **now** (settings, estimates, Pick white, camera, metadata: editing a photo or closing the session does not change a queued job) and returns the job id. One worker thread runs jobs in order and the photos of a job in order, decoding the next photo (from the session's mosaic cache, else from the file) while the current one renders.

- **Refused at `start`**, with a plain message and nothing queued: a scope with no photo (for This, a current photo that cannot be developed is refused with the reason it was skipped with, not with "open a photo"), options the writers refuse, and a photo that is still developing (`QueueError::Developing`).
- **Events** (`ExportEvent`, `#[non_exhaustive]`: a consumer matches the variants it knows and ignores the rest):
  - `Progress(ExportProgress)` is `export_progress`, in the order the panel's `applyProgress` expects. When a photo starts: `{ file, done: i, total, file_percent: 0 }` (`file` is the name with the "Keep both" number the folder will need). Then whole percents, each at most once. A photo that failed or was skipped is reported once as it finishes, with the new count (`error`, or `skipped: true`). A last event has `done == total`. A cancelled job sends nothing more.
  - `File(FileDone)` is sent for every photo that was exported, skipped or failed, just before the `Progress` event that counts it: the photo, its position, the `ExportReport` (path, bytes, size, timings, the render's notes) or the failure (kind, message, CLI exit code), and the milliseconds its decode took. The CLI can run its jobs through the queue and print one JSON object per file from these.
  - `Busy(bool)` is sent by the worker when a busy period starts (before its first progress event) and ends.
- **A name the job has already written is numbered, whatever "If the file exists" says.** Replace and Skip look at the disk, and a file this job wrote a moment ago is on the disk too. Two photos of one job that map to one name (two cards that both hold `DSC00001.ARW`, or a template with `{date}` and no `{seq}`) would otherwise replace each other with both reported as saved (Skip: the second reported as "already existed"). The queue makes one `WrittenFiles` set per job (the device and inode of every file written, so that a name differing only in case on a case-insensitive volume is recognised too) and passes it in `ExportParams::written`. `export_photo` numbers a photo whose name is in it as "Keep both" numbers it (`DSC00001_neutral 2.jpg`), the panel's `file` shows that name, and the report carries the note `SAME_NAME_IN_JOB_NOTE`. A file from before the job is still replaced or skipped as asked. Tests: `tests/export_queue.rs` (`two_photos_of_one_job_with_one_name_never_overwrite_each_other_whatever_if_exists_says`, `replace_still_replaces_a_file_from_before_the_job_and_skip_still_skips_it`), `crates/ym-cli/tests/render.rs`.
- **Failures.** One photo failing (a look that is gone, a full disk, a panic) does not stop the job. An event listener that panics loses that event only. A job that panics outside `export_photo` ends with one last `export_progress` (`done == total`, `error`) so that the panel closes instead of waiting. Percent events are sent in the order they are counted, also when two writer threads report together.
- **Control.** `cancel(job_id)` stops after the current tile or band, deletes the partial file and skips the rest; a job that has not started is dropped. `is_busy()` is true from `start` until the last job has finished or been cancelled: **"Restart to update" (`docs/12-distribution.md`) stays disabled while it is**.
- **Summary.** `summary(job_id)` keeps the last sixteen jobs (counts, files written, destination folders, the error of each failed file). `files` lists what was **written**: a photo skipped because its file exists is counted in `skipped` and not listed, as the CLI prints no path for it, while its folder is still in `folders`. `reveal_path(job_id)` is what "Show in Finder" opens: the file when the job wrote exactly one, else the first folder (a job that wrote nothing opens its destination). Several folders happen when no destination is chosen and the photos come from several (each goes next to its own source folder, as the CLI does).

#### `cli_preview`

`cli_preview::cli_preview` builds `ym render <inputs> --look … --format … --out …` (one option to a line, each line but the last ending in a backslash, as the mockup shows) with the options and none of the defaults (`--compression lzw`, `{name}_{look}`, Keep both).

- The inputs are the folder when the job is exactly every ARW file in one folder, else each file by absolute path.
- `--out` is the chosen folder, else `<source folder> export` when the photos share a folder (none with several: the CLI then does what the app does). Words are shell-quoted only when they need it.
- A command has one look: the look (and intensity, `--intensity` when not 100) most photos of the job have, the first on a tie, as its slug when the slug names exactly that look.
- `adjusted_photos` (also `ExportEstimate::adjusted_photos`) counts the photos whose settings are not *Auto with that look at that intensity*: adjustments beyond the look, and also a different look, which the command would silently export with the wrong one. The sentence is `cli_preview::adjustments_note`.

#### Preferences

`PreferencesStore` reads and writes `preferences.json`, which is `Preferences` of `src/api/types.ts` and nothing else. The folder is `ym_looks::library::default_root()`: the Mac's `~/Library/Application Support/app.yellowmoon`, `$XDG_DATA_HOME/app.yellowmoon` on Linux, and `YELLOWMOON_HOME` replaces it.

- **Reading never fails.** A missing file is the defaults and creates nothing. A missing or unknown field is ignored. A value that does not fit takes its default while the rest is kept (field by field, also inside `export_defaults`). A file that is not JSON is all defaults. Each case logs a warning and is in `warnings()`.
- **Writing.** A change re-reads the file first (the CLI may have changed another field), validates the export defaults, and writes atomically (temp file, fsync, rename, fsync of the folder). A patch that changes nothing writes nothing, and a failed write changes nothing.
- **Defaults:** the Signature look, ACES 2, the export defaults of `docs/09-export.md`, location kept in exports, update check on, Downloads watch off, Apple Auto on.
- A full open, edit and export cycle writes **no file** unless a preference changes (`tests/core.rs` compares the trees). Choosing an export folder remembers it in `last_used_folders` and writes `preferences.json`, which is on the list.

#### Looks

`LookService` keeps one `ym_looks::Library` behind a lock; the renderer reads through `LookService::provider()` (a `LookProvider` that also gives `look_sha256`). An import or a removal works on a second `Library` over the same folder, under `looks/.lock`, and takes the write lock only to re-read the manifest, so a slow import (or a CLI import holding the lock) never stalls a frame (`tests/looks_service.rs`).

- **Operations.** `list()` is `looks_list` (flags `is_default` and `missing`); `display_luts()`; `import(path, ImportRequest { input, cancel }, progress)` is the wizard's importer (same recognition, hashes and library folder as `ym looks import`; the result has the looks added or replaced, display LUTs added and files unchanged); `remove(id)`; `set_default(id)`; `set_display_rendering`; `apply_look(session, scope, id, intensity)`; `resolve(query)` (`--look`, exit 2 or 3); `verify()`; `reload()`.
- **Missing looks.** A look whose file is gone or changed stays in the list, flagged `missing`, and cannot be made the default or applied ("File missing or changed. Import again."). A default look that is gone is not used (new photos get Neutral, `effective_default()`), but the preference is kept until the look is removed or the user chooses another. Removing the default look or the chosen display LUT resets the preference to Neutral or ACES 2.
- **Display rendering.** The one in effect is `display_choice()`: the chosen display LUT while it is imported and its file is there, else ACES 2 (the viewport falls back; **an export with a chosen display LUT that is gone is refused**, `check_display_rendering`).
- **`on_change(f)`** is `looks_changed`. It runs after an import, a removal, a new default, or a `reload`/`verify` that found a different list, outside every lock. "Different" is judged from before `reload` re-reads the preferences and the manifest: the ids, whether each can be used, the SHA-256 of each file (an ARRI look replaced in place by a newer file) and the default look (the CLI chooses it through the preferences file).
- `Core::looks_list` reloads first, so an import made by the CLI shows up. So does `Core::export_start`, so that a look or display LUT whose file has gone since the last look at the library is noticed when the job is queued (a display LUT refuses the job; a look fails the photos that use it).

#### `Core`

`Core::open(CoreOptions)` (`app.rs`) builds the preferences, the look service, the renderer (reading looks through the library and showing the preferred display rendering), the session (new photos get the effective default look), the export context and the queue, and keeps the session and the renderer in step with `looks_changed`. It writes nothing. All of it is `Send + Sync`. `CoreError::scope()` is the `AppError` scope.

Its methods are the IPC commands of the table below: `preferences_get/set` (validated against the library, then the session and renderer follow), `looks_list/import/remove/set_default`, `apply_look`, `export_estimate/start/cancel`, `cli_preview`, plus `export_summary`, `export_reveal_path`, `export_busy`, `close_session` and `shutdown`.

- `close_session` cancels the viewport, tile, thumbnail, histogram and look-tile renders, queued and running, forgets what the renderer cached, and **retires the closed session's photo ids** with `Renderer::retire_through`. That refuses any request for those photos that was still on its way, such as a command that read its photo before the close and queued its job after the scheduler's cancel, which would refill the caches of a photo that was gone (`closing_the_session_cancels_its_queued_renders...`). Export renders are not refused and leave exports running.
- `shutdown` cancels the exports and the renders, then **waits up to `SHUTDOWN_EXPORT_WAIT` (5 s) for the export queue to go idle**. The process ends the moment it returns, so a worker still writing would never delete its hidden `.yellowmoon-*.tmp`, tens or hundreds of megabytes in the person's export folder for good. A cancel is noticed between render tiles and writer bands, about 0.4 s at 24 MP. Tests: `tests/core.rs` `quitting_while_an_export_is_writing_leaves_no_partial_file`, and `AppApi::shutdown` in `src-tauri/src/api/tests.rs`.

## Native viewport

The webview covers the whole window and is transparent where the photo is. A wgpu surface (CAMetalLayer) sits behind it inside the same NSWindow (`src-tauri/src/native.rs`). Where the photo is drawn, and how the cell, the page and the window behind it fit together:

**Transparent where the photo is** means every box under the photo cell, not only the cell. While the Photos screen shows with the native viewport, the page's root, body, `#root` and the Photos screen draw no background. `PhotoView` sets `data-native-viewport` on the root element for this (the rule is in `src/styles/screens.css`), and also sets `background-color: transparent !important` inline on the same four boxes, putting back what was there when the view goes. The stylesheet alone is not enough: `index.html` paints the root paper in an inline style (so nothing shows through before the style sheet loads), which only an `!important` rule overrides, and a running transition sits above `!important` in the cascade (WebKit then leaves the backgrounds at their old paper). With Reduce motion on, the reduced-motion rule of `global.css` therefore sets `transition-property: none` (no transitions at all), and the four boxes have `transition: none` while the native viewport shows.

The window behind the page is paper (`PAPER_SRGB` in `src-tauri/src/native.rs`, which a test keeps equal to `--paper`; paper is the dark `#0f0f10`, `docs/11-ui.md`), so the page looks the same wherever it draws nothing, and the cell shows paper before the first frame and after the picture is taken away. The letterbox of the frames the core draws for the app is the same colour (`PAPER_RGB8` in `src-tauri/src/api.rs`), set by the app alone: `RenderOptions::background` stays white by default and exports have no letterbox, so the command line and every export are unchanged.

The viewport job of `.github/workflows/build.yml` runs the app on a Mac runner with photos and fails when the page paints anything under the middle of the cell or the layer never shows a picture; its screenshot is the artefact. Its page line also prints the root's inline style, whether the rule's selector matches, which loaded sheet carries the rule, whether Reduce motion is on and the page's running animations.

1. The frontend measures the photo cell (`<div data-viewport>`) and sends `viewport_set_rect { x, y, width, height, scale }` on mount, resize and layout change (ResizeObserver). The shell draws nothing, and applies no zoom, pan or Before, until the first rectangle with an area (both sides at least 2 CSS pixels) has arrived: the controller starts with a 1 × 1 cell, and drawing into it would make a frame of one pixel for the webview or the native layer to throw away.
2. `src-tauri` positions the Metal layer to that rect and asks `ym-core` for frames.
3. Pointer input over the photo arrives in the webview first. The frontend sends `viewport_input` events (zoom to point, pan delta, double-click to toggle fit/100 %, white-pick click) to the core. What the shell does with an input matters: see *Which request stops which render* below, and `docs/13-testing.md` §Performance logging for the counters that show where the time goes.
4. Surface colour space: Display P3 (`Bgra8Unorm` with sRGB-encoded P3 values; the DRT outputs Display P3 for the viewport). When the surface does not offer Display P3 the native sink refuses to start and the fallback viewport is used, so colour is never shown wrong silently. HDR/EDR is V2.
5. The photo cell is always exactly 3:2 (`docs/11-ui.md`); the image is letterboxed inside it with the paper colour, never with grey.

**The fallback viewport** (`FallbackSink`, `src-tauri/src/viewport.rs`) draws everywhere the native sink does not: development builds on other systems, and a Mac where the surface cannot be made. Each frame is encoded as a PNG with the Display P3 profile, kept in a `FrameStore` and served to the webview at `ym-frame://localhost/<launch>/frame-<seq>.png`; the `viewport_frame` event says when a new one is there. This is the one path on which a full image crosses the bridge. `viewport_info` tells the webview which backend draws, `native` or `fallback`.

### Viewport state and conventions

Code: `ym-core`'s `pipeline/viewport.rs` and `pipeline/frame.rs`. The core keeps one `ViewportState` for the current photo: the cell (`ViewportRect`, the payload of `viewport_set_rect`), `ViewportMode` (`fit` or `percent { value }`), the pan centre and the Before switch. The wire types are the ones of `src/api/types.ts`. Conventions the frontend relies on:

- **Zoom** is device pixels per pixel of the final image (after lens crop, orientation, straighten and crop, at full resolution), so 100 % is one image pixel on one device pixel and the header's "Fit 24 %" is `fit_zoom`.
  - At exactly 100 % the view's origin is rounded to a whole image pixel (`ViewportState::view_to_final`). A pan, or a cell of odd width, would otherwise leave the origin between pixels, and every frame would be a half-pixel bicubic resample that softens every edge at the zoom people use to judge sharpness.
  - The cell is capped at `MAX_CELL_PX` (8192) device pixels a side, and a size that is not a number is one pixel: a hostile or glitched rectangle must not ask for a frame of gigabytes.
  - A fixed zoom below the fit zoom is the fit zoom. The upper limit is 800 %.
- **Pointer coordinates** in `viewport_input` and `sample_pixel(x, y)` are CSS pixels inside the cell (origin at the cell's top left, as the webview reports them); the cell's `scale` turns them into device pixels. `sample_pixel` returns nothing for a point in the letterbox.
- **Inputs.** `zoom` multiplies the zoom about the pointer (the point under the pointer stays put). `pan` moves the image with the pointer and is clamped so the cell never shows beyond the image. `toggle_zoom` flips between Fit and 100 % about the pointer. `pick_white` does not change the view: the core converts the click to a point of the photo as the camera oriented it (`Renderer::pick_point_at`, the space of `white_pick`, through the straightening, the turns and the lens maps) and the application sends that point as `white_pick` through `update_settings`, where the session takes the sample the chain uses (`auto::pick_white_sample`, from the raw data). The sample is the session's alone; the "no vignetting gain" contract is tested on it (`tests/render_review.rs`).
- **The core reports its zoom.** `RenderStats::{zoom_percent, fit_percent}` come from the same `ViewportState::zoom` and `fit_zoom` that choose the frame (`Renderer::viewport_frame` sets `RenderReport::zoom`, also on a frame served from the cache; a thumbnail, a look tile or an export has none) and ride on `render_stats`, which both backends send after every viewport frame. The webview takes the Fit zoom from the stats (`fitZoomOf`) and keeps its own estimate only until the first frame has reported and in the mock, because an estimate from the photo's size and the cell cannot see the lens crop, the straighten angle or the crop. A wheel zoom is mirrored at once, before the core has drawn a frame to report, and the header's number for a fixed zoom is the requested one, which is what the core draws (above Fit, up to 800 %).
- **The frame** is RGBA8 in Display P3 (`Renderer::viewport_frame`), the size of the cell in device pixels, with the letterbox painted in the paper colour and the 8-bit output dithered (the device colour pass and the CPU pass dither the same way). The Before view is the same frame with the colour chain replaced by the camera default rendering: the camera's as-shot white balance, the body's D65 colour matrix, the camera gain and the camera tone curve, then the output primaries and C13 (`docs/02-colour-pipeline.md` *The Before view*). No setting reaches it except the decode switches and the lens corrections, which change the raw.
- **`coarse_only`.** A frame asked for with `coarse_only` is served from the reduced image when the full-resolution tiles are not ready yet and says so (`RenderedFrame::sharp` is false). The tile job (`Renderer::spawn_tiles`) then runs at the lower priority and the next frame is sharp.

### Which request stops which render

The shell (`src-tauri/src/viewport.rs`) records what the webview asks for (inputs, settings patches, look changes, a cell that moved, Before, the mode) and wakes its worker. It does not cancel the render in flight for each request. A frame is only finished at the end of a render (the core checks for cancellation between stage groups and before the readback, so a cancelled render yields no frame). If every request cancelled the render in flight, then once a render took longer than the gap between two requests (a wheel's 8 ms, a slider's 16 ms) no frame would reach the screen until the hand stopped, and the picture would jump. The rules:

1. A request (`Shared::request`) stops only the pass that makes sharp tiles (`Flight::Tiles`, which nobody is waiting for: the round that follows draws a coarse frame first).
2. A request that makes the frame being drawn *wrong* rather than old (`Shared::request_now`: another photo or none, which includes the session closing, and a cell of another size or scale) stops the frame as well, and a round taken before such a request does not start drawing. A cell that only moved is the first kind.
3. The frame that finishes is presented, and the operations (pan, zoom) that arrived meanwhile are applied at the start of the next round, from the newest state, so the picture follows the hand at the rate the renderer can keep.

`snapshot_synced` (Pick white, the Option-key readout) waits for the next round to apply what is pending, so up to one frame, and still gives up after 1.5 s. Exports never go through the viewport, so the byte-for-byte parity of `ym` and the app is untouched.

Not implemented: applying pan and zoom eagerly (once the final image size is cached in the state), or showing the last frame offset by the input delta at once. Either would help if the counters (`docs/13-testing.md`) show that a frame is still too slow to follow the hand.

### One device, zero-copy

The native sink draws with the core's own graphics device.

1. `AppApi::start` opens the device first (`open_gpu`) and makes the sink with it (`make_sink(Some(&gpu))`).
2. `NativeSink::create` makes the surface of its `NSView` from that device's `wgpu::Instance` (the instance the adapter came from, `Gpu::wgpu_setup`), checks that the adapter can draw to it, and configures it with the core's `Device` (`Bgra8Unorm`, Display P3; refused otherwise, so the fallback viewport is used, without Display P3).
3. The worker asks for frames that stay on the device (`FrameSink::takes_device_frames`, `Renderer::spawn_viewport_frame_to(…, FrameOutput::Device)`). The view job runs the same kernels and waits for them, but reads nothing back.
4. The frame (`RenderedFrame::on_device`, a `DeviceFrame`: the pool buffer of RGBA8 the kernels wrote, its size and its device) is bound as a read-only storage buffer of the sink's full-screen triangle (`src-tauri/src/present.rs`), which writes it texel for texel into the surface texture on the core's queue.
5. The pass is submitted after the job that wrote the buffer, so the queue orders them. The frame is borrowed until the pass is submitted, then dropped, and its buffer goes back to the pool (a later job that gets it is submitted later still).

A frame that never crosses to the processor avoids a readback (24 MB for a 3000 × 2000 cell) and an upload to a second device. A view magnified above 100 % is a device frame too (the stage and the magnification run on the device, *View stages on the device* above), so the viewport draws zero-copy at every zoom.

- **Host frames.** A frame the CPU made (a device error, a stage that is off) is uploaded and drawn from host bytes.
- **The sink's own device.** It remains for a core without one (policy `off`, no adapter, a software adapter under `auto`, a device lost at start), and replaces the core's device if that one is lost later: the next frame rebuilds the surface on a device of the sink's own, and the worker asks for host frames from then on.
- **Error scopes.** The sink's work on the core's device runs inside error scopes, so a mistake of the viewport never taints the device the renderer uses.
- **Log and counters.** The log says at start which way it draws: "native viewport: zero-copy on the core's device" or "native viewport: own device, host frames". `YM_PERF` counts the frames drawn each way and times the wait for a drawable and the pass (`docs/13-testing.md`).
- The view is our own `NSView` behind the transparent webview (Tauri's multi-webview API is not used). `docs/17-gpu.md` §16 has the device side.

## IPC between webview and core

Tauri commands carry requests and Tauri events carry progress. The TypeScript types are generated from Rust with `specta` + `tauri-specta` (MIT), so the two sides cannot drift: `src-tauri/src/bindings.rs` checks that the committed bindings are what the code generates, and `src/api/contract.ts` checks `src/api/types.ts` against them. The frontend also has an `api.mock.ts` so it runs in a browser for Vitest and Playwright without Tauri.

**Commands** (request → response)

| Command | Purpose |
|---|---|
| `open_paths(paths)` / `open_dialog(kind: files or folder)` | Start a session or add photos. The answer's `opened: { added, already_open }` says what this call did |
| `session_state()` | Photos, current index, picked set, per-photo summary |
| `close_session()` | Forget the session (⌘W, "Start over"). Cancels development that is still running; running exports keep going |
| `set_current(photo_id)`, `set_picked(photo_id, bool)` | Navigation and picking |
| `remove_photos(photo_ids)` → `{ photo_ids, already_open, session }` | Remove from filmstrip. Takes the photos that are in the session out (ids that are not are ignored; the answer lists the ones that were), hands the current photo over to the next one (the previous one when the last went), drops their queued work, caches and thumbnails, and gives the burst frames that stay their new median (`settings_changed` for those whose estimates moved). The files are not touched; exports in flight keep their photos. Keeps the removal in memory for `restore_photos` |
| `restore_photos()` → `{ photo_ids, already_open, session }` | Undo of the last `remove_photos`: the photos come back where they were with their settings, picks and thumbnails (`thumbnail_ready` again). `already_open` counts photos whose file was added again meanwhile and which stay out. An empty change when there is nothing to put back |
| `get_settings(photo_id)` → resolved settings with Auto flags | Panel state |
| `update_settings(photo_id, patch)` | Any control change; returns resolved settings |
| `apply_look(scope, look_id, intensity)` | Apply to This photo / Picked / All. The webview sends these one at a time, in the order the looks were picked |
| `rerun_auto(photo_id)` | Recompute automatic estimates |
| `copy_settings(photo_id)` → patch | The photo's settings as a patch to paste onto other photos: all but the crop, the orientation and the Pick white point, Auto kept, a pick's white balance as manual Kelvin and tint. Stores nothing. Refused for a photo that failed or is not developed yet |
| `paste_settings(photo_ids, patch)` → `{ pasted: [{ id, before, after }], skipped }` | Applies a copied patch to those photos in one call under one lock; skips photos that are still developing or failed; `before` and `after` are each photo's change as one undo step; refuses a look that is no longer in the library; emits `settings_changed` and queues a thumbnail per changed photo |
| `set_menu_state(state)` | `{ copy_settings, paste_settings, paste_settings_to_picked }`: which of the Edit menu's three settings items are available (the webview knows; the native menu starts with all three off). A synchronous command, so calls run on the main thread in the order sent |
| `viewport_info()` | Which backend draws the photo: `native` or `fallback` (`null` in the browser mock, which draws its own picture) |
| `viewport_set_rect(rect)`, `viewport_input(event)`, `viewport_set_mode(fit or 100)`, `set_before(bool)` | Native viewport |
| `sample_pixel(x, y)` → RGB display values | Readout and white pick |
| `histogram(photo_id)` → `{ red, green, blue }` | Light tab and Curve plot: 64 bins of the LogC4 signal at the curve's input, the tallest bin of any channel is 1; ask again after every `render_stats` |
| `chroma_scatter(photo_id)` → `{ points }` | Colour › Colour warp's plane: the photo's colours at the warp's input (the C9 output, `ym_color::warp::chroma_of`), read from the histogram's view (the whole photo at most 192 px on the long edge, the letterbox left out) by keeping every n-th pixel in reading order, so at most 2048 points and the same points for the same photo and settings. Flat `[a0, b0, a1, b1, ...]`, each coordinate rounded to 1e-4. A thumbnail-priority job that a move to another photo cancels, like the histogram. An empty list while the photo is still being developed (the histogram rejects then). Ask again after every `render_stats` (`Renderer::chroma_scatter`) |
| `suggest_crop(photo_id, aspect, orientation)` → `CropSuggestion` | The Crop panel's Auto (`docs/20-auto-crop.md` §3): the straightener's answer (`level`, `rotate` or `unsure`), the angle, up to three frames on the canvas best first, whether the whole photo is kept, and where the subject came from. The first call for a photo renders Apple Auto's picture on the CPU and analyses it (kept in memory until Re-run changes the photo's estimates); later calls only make the frames. Refused while the photo develops |
| `sample_chroma(x, y)` → `{ a, b }` or `null` | Pick colour on the Colour warp card: the chroma at the warp's input of the pixel `sample_pixel` reads, with `x`, `y` in CSS pixels from the cell's corner as there. Always the final picture's colour, also while the Before view is shown (the pins act on the final picture). `null` on the letterbox, for a point that is not a number and with no photo shown (`Renderer::sample_chroma`) |
| `look_thumbnail(photo_id, look_id)` → address (string) | The Look tab's tiles: the photo as it is developed now through one look at 100 %, at most 240 px on the long edge, sRGB, rendered lazily as tiles scroll into view. The answer is the address of the PNG on the `ym-frame` scheme (below). Rejects with scope `looks` and "File missing or changed. Import again." when the look's file is gone |
| `looks_list()`, `looks_import(path)`, `looks_set_default(id)`, `looks_remove(id)` | Look library |
| `looks_recognise(path)` | What a zip, a folder or a `.cube` file is, before anything is copied (`docs/08-looks.md` §Recognition rules). Never rejects for a bad file: it answers `unusable` or `missing` |
| `looks_display_luts()` | The display LUTs imported so far (Preferences › Display rendering, the wizard's Finish step) |
| `looks_watch_downloads(enabled)` | Start or stop the Downloads watcher (FSEvents, `docs/08-looks.md` §Wizard step 2). Asks macOS for access the first time; `problem` says so when it is refused. Found zips arrive as `download_found` |
| `export_estimate(job)` → bytes and seconds | Size and time estimate in Export |
| `export_start(job)`, `export_cancel(job_id)` | Export queue |
| `export_busy()` | An export is queued or running in the core, also one this window did not start (after a reload). "Update ready" waits while it is |
| `cli_preview(job)` → string | The equivalent `ym render …` command |
| `preferences_get()`, `preferences_set(patch)` | Preferences |
| `apple_auto_status()` | Whether Apple Auto is available on this computer (read again for the answer) and whether it acts (`docs/19-apple-auto.md` sections 1 and 2). Preferences asks when it opens and shows "Auto with Apple Intelligence" only while it is available |
| `install_cli_link()`, `cli_link_status()` | The "Install Command-Line Tool…" menu item, and whether the link exists and where (Preferences › Command-line tool) |
| `app_info()` | Version numbers for About |
| `thumbnails()` → `[{ id, url }]` | The address of the thumbnail of every photo that has one, to get them again after the webview reloaded (the core announces each thumbnail once). Only addresses cross the bridge |
| `choose_folder(start)`, `choose_look_files()` | The system's folder chooser (Save to › Choose…) and file chooser for zips and `.cube` files (wizard › Choose file…). `null` when cancelled |
| `open_url(url)`, `reveal_in_finder(path)`, `trash_path(path)` | Open a page in the system browser (ARRI's page, GitHub); show a file or folder in Finder (Export › Show in Finder); move a file to the Trash, recoverable (the wizard's "Move to Trash"; the `trash` crate) |
| `restart_app()` | Restarts into the update that was installed ("Update ready"). Refused with scope `export` and "An export is still running. Restart after it finishes." while an export is queued or running; once it has begun, no export is queued. The webview has no other way to restart the app: the process plugin's `restart` permission is not granted, because Tauri's own restart cannot be vetoed by the exit handler (`ExitRequestApi::prevent_exit` is ignored for it) |

**Events** (core → webview)

| Event | Payload and use |
|---|---|
| `photo_progress` | `{ id, stage, done }`. Stages are `probe`, `decode`, `estimate`, `thumbnail`, and with Apple Auto acting `apple` (Apple's answer, `docs/19-apple-auto.md` section 6). A photo is ready when `thumbnail` is done |
| `thumbnail_ready` | `{ id, url }`: the photo's thumbnail is at `url` (below) |
| `render_stats` | `{ ms, path, space, zoom_percent, fit_percent }`: the header's status text. `path` is `neural` or `classical`, `space` is `srgb`, `display_p3` or `rec2020`; `RenderStats::status_text` makes "Classical · 42 ms · Display P3". `zoom_percent` is the zoom the last viewport frame was drawn at and `fit_percent` the zoom at which the whole final image fits the cell, both in percent and not rounded; `null` for anything that is not a viewport frame and from the mock |
| `viewport_frame` | `{ seq, width, height, sharp, url }`: the fallback viewport has a new frame of the photo cell, or none any more (empty `url`, 0 × 0) |
| `viewport_busy` | `{ busy, cpu }`: the viewport is making a frame of the photo on screen or the full-resolution tiles a frame needs (from just before the frame is asked for until the round ends, whichever way), and `cpu` says the photo's raw stages run on the CPU (`Renderer::raw_on_cpu`: no device, `RAW_CHAIN` off, a device error ended the route, the device lost, a method the device does not run). Sent on change only, latest-wins through the `Throttle` below with a repeat of the last one dropped, and false when the cell shows no photo. The photo cell says "Rendering on the CPU" once both have held for 250 ms (`docs/11-ui.md` §Photos) |
| `settings_changed` | `{ photo_id }`: the core changed a photo's settings by itself (after a Pick white click, once for each photo a `paste_settings` changed, and when a photo's estimates moved); ask `get_settings` again |
| `export_progress` | `{ job_id, file, done, total, error? }` |
| `export_busy` | `{ busy }`: an export is queued or running, or the queue has just gone quiet. "Update ready" waits while it is true |
| `looks_changed` | No data: the look library changed |
| `looks_import_progress` | Files of one source copied so far during `looks_import` (the wizard's Import step) |
| `download_found` | The Downloads watcher found a package zip (`looks_watch_downloads`) |
| `menu_action` | `{ action }`: a native menu item that needs the webview was chosen. Items the shell handles itself (Show Looks Folder, the Help links) are not sent |
| `update_state` | `{ phase, version?, percent?, message? }` |
| `error` | `{ scope, message }` |

### Thumbnails and tiles by address

Thumbnails do not cross the bridge as bytes. `thumbnail_ready` carries the photo's id and the **address** of the picture, which the shell keeps and serves on the `ym-frame` scheme that also serves the fallback viewport's frames (`ym-frame://localhost/<launch>/thumb-<id>-<version>.png`, `http://ym-frame.localhost/…` on Windows; `img-src` allows it). The `<img>` loads it. A thumbnail PNG (at most 240 px on the long edge) is about 250 KB, and as a JSON array of byte values it would be four times that per photo, 125 MB of bridge traffic for 500 photos. The pictures live in the shell instead of the webview: the same bytes, held once, not copied through JSON, `Blob`s and object URLs.

- **Versions.** Every new thumbnail of a photo (a look, a crop or a turn was applied) has a new version and so a new address. The shell keeps the last two versions of a photo's thumbnail, so a picture the webview is still showing can be loaded again a moment after a newer one was announced. Closing the session forgets them all (the old addresses answer 404).
- **After a reload of the webview** the store asks `thumbnails()` for the addresses of the pictures the shell holds (the core announces each thumbnail once).
- **The Look tab's tiles** travel the same way: `look_thumbnail` answers with the address (`ym-frame://localhost/<launch>/look-<photo>-<n>.png`; a new address for every request). The shell keeps the last two tiles of each photo and look, at most `LOOK_TILE_BUDGET_BYTES` (256 MiB, about 4,500 tiles) in all. When that is passed, the photos used least recently (a tile put or served counts as a use) lose all theirs; the photo just given a tile never does. Closing the session forgets them all. A photo whose tiles were let go answers 404 for the addresses it was given, until the webview asks again, which it does when the photo's settings change. The store caches the address per photo, look and settings.
- The mock announces and answers with `blob:` addresses of its own. Full images never cross the bridge (the fallback viewport excepted).

### The `ym-frame` handler and its cache

The scheme is registered as an *asynchronous* protocol (`register_asynchronous_uri_scheme_protocol`). wry calls a synchronous handler on the main thread, and a request copies the PNG under a mutex (the filmstrip asks for hundreds of pictures when a folder opens, the Look tab for as many whenever it is shown). The handler therefore only hands the path to a blocking thread, which builds the response and answers; the main thread gets the response and nothing else (`app.rs` `serve_frame`; a test holds the tiles' mutex and checks that the handler returns at once and the answer arrives after the release).

- **Launch id.** Every address starts with a launch id (`ym-frame://localhost/<launch>/…`: the start time in hexadecimal milliseconds, a hyphen and a count of the stores the process made; not a secret). The version counters of thumbnails and tiles start at 1 in every launch, so the first thumbnail of photo 3 would otherwise have the same address in two launches. A request with another launch id is a 404.
- **Caching.** The launch id is what makes caching safe: a 200 for a thumbnail or a look tile is sent as `Cache-Control: private, max-age=31536000, immutable` (the address names the same picture for as long as the app runs and is never used for another), so the `<img>`s that the Look tab remakes when it is shown again do not ask the shell again. Frames of the fallback viewport and every 404 stay `no-store` (a frame is replaced by the next, and a 404 for a tile that was not made yet may be a 200 later). The webview is not known to keep custom-scheme answers between launches, and the launch id makes that irrelevant. Whether WKWebView honours `Cache-Control` for a custom scheme at all is a Mac check (`docs/13-testing.md`).

### Event coalescing

Each event the shell sends costs the main thread a message that runs a script in the webview (`evaluateJavaScript`). `render_stats` comes once per presented frame and `photo_progress` four or five times per photo (thousands when a folder of hundreds opens), so `TauriEmitter` sends them through `Throttle` (`events.rs`): one thread, at most one `render_stats` and one batch of `photo_progress` per 33 ms (30 Hz), the first after a quiet period at once.

- `render_stats` is latest-wins.
- `photo_progress` is latest per photo while the waiting event of that photo only says a stage started. An event with `done` or an `error` is never dropped and never moved past a later event of its photo (a photo that is Ready and then developed again arrives as Ready, then Developing), so what the webview knows about every photo is what it was told last.
- What waits is sent when the emitter is dropped. Every other event is sent as it comes.

### The core's side of the IPC

Code: `ym-core`'s `app.rs`. `Core` has one method per command above (`export_estimate`, `export_start`, `export_cancel`, `cli_preview`, `preferences_get`, `preferences_set`, `looks_list`, `looks_import`, `looks_set_default`, `looks_remove`, `apply_look`; the session commands are `Core::session`'s) and takes the events as callbacks: `CoreOptions::session_events` (`photo_progress`, `thumbnail_ready`, `estimates_changed`), `CoreOptions::export_events` (`ExportEvent::Progress` is `export_progress`, `ExportEvent::Busy` tells the updater when the queue starts and stops) and `Core::on_looks_changed` (`looks_changed`).

The wire types the export, preferences and looks commands use serialise to the interfaces of `src/api/types.ts`. A test compares the field names of `ExportOptions`, `ExportJob`, `ExportEstimate`, `ExportProgress`, `Preferences`, `LastUsedFolders`, `Look`, `DisplayLut` and `LooksImportResult` with the interface text, so a drift fails in CI. `cli_preview` answers with the command as the CLI is named (`ym`); `ExportEstimate::adjusted_photos` and `CliPreview::note()` carry "N photos have adjustments this command doesn't include."

## Persistence

Only these persist, under `~/Library/Application Support/app.yellowmoon/`:

```
preferences.json                  default look, display rendering (ACES 2 or an imported ARRI display LUT),
                                  export defaults (format, quality, bits, compression, colour space, size,
                                  destination, name template, if-exists), remove location from exports,
                                  check for updates automatically, watch Downloads during import,
                                  last-used folders, Auto with Apple Intelligence and its instructions
                                  (docs/19)
looks/
  manifest.json                   imported looks and display LUTs: id, name, group, input encoding,
                                  source URL, import date, SHA-256 per file (no preferences)
  arri/<group_no>-<group_slug>/   imported ARRI look files (copied, never modified)
  arri-display/                   imported ARRI display LUTs
  user/                           other imported .cube files
logs/yellowmoon.log               local only, rotated at 5 MB × 3
```

This list is exhaustive (hard rule 4 of `AGENTS.md`): nothing else persists. Plus `~/Library/Caches/app.yellowmoon/` for the wgpu pipeline cache only (`$XDG_CACHE_HOME/app.yellowmoon` or `~/.cache/app.yellowmoon` on Linux). No image data is cached on disk. The CLI uses the same folders and takes a file lock (`looks/.lock`) when importing. `YELLOWMOON_HOME` replaces the application-support folder.

Code: `ym-core`'s `preferences.rs` and `looks.rs`. `PreferencesStore` reads and writes `preferences.json` (*Export, preferences and looks* above); `LookService` owns `looks/` and its lock. The file has exactly the nine fields of the list: `default_look`, `display_rendering`, `export_defaults`, `remove_location_from_exports`, `check_for_updates_automatically`, `watch_downloads_during_import`, `last_used_folders`, `apple_auto` and `apple_auto_instructions`. A field of a newer version is ignored when reading and dropped when the file is next written.

- `apple_auto` ("Auto with Apple Intelligence", `docs/19-apple-auto.md` section 2) is an app setting: default on, a file without it reads as on, read on every platform and acting only where Apple Auto is available.
- `apple_auto_instructions` is the instructions Apple's model reads, edited in Preferences under that switch (`docs/19-apple-auto.md` section 2): a string of at most 2000 characters. `""` (also absent) means the default instructions, and the default text itself is saved as `""`; longer text in a file edited by hand gives the default, with a warning.
- Both are app settings like the others, not data about a photo.
- `logs/` is the application's (`ym-core` writes no log file; it logs through `tracing`).

## Finding `data/`

The app and the CLI look for the data folder in this order and use the first candidate that exists as a directory:

1. `YELLOWMOON_DATA` environment variable (tests, development). A value that names nothing falls through to the next candidate.
2. Next to the canonicalised executable inside the bundle: `Contents/MacOS/<exe>` → `Contents/Resources/data/`. Canonicalise first, because the CLI on PATH is a symlink into the bundle.
3. Debug builds only: `<workspace root>/data/`, from `CARGO_MANIFEST_DIR` at compile time.

If none exists, fail with "Yellowmoon's data folder is missing. Reinstall the app." and exit code 2 in the CLI. The lookup is `ym_core::data_dir::find_data_dir`.

## Errors and logging

- `tracing` with a local file subscriber (`logs/yellowmoon.log`, three files of at most 5 MB). Log levels: info for user actions and job summaries, debug for stage timings (`YELLOWMOON_LOG=debug` asks for debug).
- **Nothing of the person's own is named at info level or above** (warn and error included; the default filter is info), so a log in a bug report does not leak paths. A photo, an export folder or file, the preferences file and the looks list are not named: a failed export logs its job and the kind of failure (`export failed job=3 kind="write"`), a failed preferences write the kind (`PreferencesError::kind`), a damaged preferences file the sentence about it, a damaged looks list the kind of problem (`kind="manifest"`). The file name, the message (which names the folder) and the paths are logged at debug level, where a person looking for the cause can ask for them; `ym doctor` prints what it includes. Errors that name the application's own files (a model or the shipped `data/` folder) are logged whole. Tests: `tests/export_queue.rs` `a_failed_export_logs_its_kind_and_job_but_no_folder_or_file_name`, `tests/looks_service.rs` `a_damaged_looks_list_is_logged_without_the_path_of_the_persons_folder`, and the test in `preferences.rs`.
- Every error that reaches the UI carries a plain message and an action ("This file is from a camera mode we can't read yet: ARW 6.0 compressed. Export it as uncompressed or lossless in the camera, or wait for an update.").
- A file that fails to decode stays in the filmstrip with an error tag; it never blocks other photos.

## Versions written into outputs

- `pipeline_version`: semver of the processing chain (`PIPELINE_VERSION` in `ym-core`'s `version.rs`), bumped when output pixels change for the same settings.
- Camera profile id and version, model ids and versions, look id and SHA-256, DRT id.

These go into XMP (`docs/09-export.md`) so any exported file can be traced.

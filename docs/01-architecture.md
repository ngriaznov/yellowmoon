# 01 · Architecture

## Shape of the system

```
┌──────────────── Yellowmoon.app ─────────────────┐      ┌──── yellowmoon (CLI) ────┐
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

All image processing happens in `ym-*` crates. `src-tauri` and `ym-cli` are thin front ends: they translate user actions into `ym-core` calls and show results.

## Repository layout

```
Cargo.toml                 workspace (members listed explicitly: crates/*, src-tauri, tools/oracle, tools/golden, tools/dngfit)
rust-toolchain.toml
deny.toml                  cargo-deny licence and advisory policy
crates/
  ym-core/                 session, settings, auto development, render graph, scheduler, export queue
  ym-raw/                  ARW container, SR2, decoders, camera detection, metadata
  ym-profiles/             camera profile registry; loads data/cameras/**
  ym-color/                LogC3/4, AWG3/4, ARRI matrices, WB from curves, sensor map, illuminant estimator,
                           CDL, cube LUTs, ACES 2 output transform (CPU reference)
  ym-gpu/                  wgpu device/queue ownership, tiling, buffer pools, WGSL kernels + parity tests
  ym-noise/                per-image noise estimation, priors, σ maps
  ym-ml/                   model loading, inference on tiles (burn or ort behind one trait)
  ym-looks/                look library, ARRI package recognition, importer, manifest
  ym-export/               JPEG, TIFF, PNG, DNG writers, ICC, EXIF/XMP
  ym-cli/                  the `yellowmoon` binary
src/                       React + TypeScript frontend (as plugout: Vite, Vitest, Playwright)
src-tauri/                 Tauri app crate (window, commands, events, updater, native viewport)
data/                      shipped data: alexa/, arri/, cameras/, looks/, models/ (small weights, committed with model cards)
training/                  PyTorch training code, dataset scripts, model cards (not shipped)
tools/
  oracle/                  test-only crate linking rawler (LGPL) to compare decodes; never shipped
  fetch/                   scripts that download test corpora and weights with SHA-256 checks
  golden/                  golden-image runner (uses ym-cli)
testdata/                  git-ignored; filled by tools/fetch
design/                    mockups, screenshots, tokens (reference for the UI)
research/                  reports and experiments (background, read-only)
docs/                      these specs and STATUS.md
templates/                 CI, release and cask templates to install in M0
Casks/yellowmoon.rb        Homebrew cask (installed from templates in M0)
```

### Package names

| Path | Cargo package | Binary |
|---|---|---|
| `crates/ym-*` | `ym-core`, `ym-raw`, … (same as the folder) | — |
| `crates/ym-cli` | `ym-cli` | `yellowmoon` |
| `src-tauri` | `yellowmoon-app` | `Yellowmoon` (Tauri `mainBinaryName`) |
| `tools/oracle` | `oracle` (test-only, links rawler) | `oracle` |
| `tools/golden` | `golden` | `golden` |
| `tools/dngfit` | `dngfit` | `dngfit` |

Script-only folders under `tools/` (`fetch/`, `looks/`, `dng/`, `abtest/`, `dev/`) are not workspace members.

### Building the app crate

`tauri-build` checks at compile time that every `bundle.externalBin` and `bundle.resources` path exists, on every `cargo build/test/clippy` of `src-tauri`. So:

- `tools/dev/prepare-sidecar.sh` (written in M0-01) builds `ym-cli` and copies it to `src-tauri/binaries/yellowmoon-cli-<host triple>`; CI and `npm run tauri dev` (via `beforeDevCommand`) run it first. Release builds copy the per-architecture binaries and the universal one (`release.yml`).
- `data/models/` and `data/looks/builtin/` exist from the start with a README, so the resource paths resolve before M1/M2 fill them.

### Dependency rules

- `ym-core` depends on all other `ym-*` library crates. Nothing depends on `ym-core` except `src-tauri` and `ym-cli`.
- `ym-color` is pure maths with no GPU dependency. `ym-gpu` depends on `ym-color` only for constants and CPU reference functions used in parity tests.
- `ym-raw` has no GPU dependency; it returns CPU buffers.
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

Opening files: `Session::open(paths)` probes each file (fast metadata read, no full decode), then a background job decodes, estimates and renders a thumbnail for each. Progress events drive the start screen's "Developing N photos" list.

**As built (ym-core: `session.rs`, `auto.rs`, `camera.rs`).** The structs above are the shape, not the code: `Session` holds its state behind one mutex and a decode pool, takes `&self` everywhere (`Arc<Session>` is how the app shares it) and keeps the photo's own estimates next to the ones it serves.

- *Environment.* `SessionEnv::load(data_root)` loads the profile registry, the noise priors (`cameras/noise`) and the illuminant models (`models/illum/<profile>.json`); `with_events`, `with_thumbnails`, `with_default_look`, `with_decode_threads`, `with_forced_profile` (the CLI's `--profile`, refused with the valid ids when unknown) set the rest. `EventSink::new(callback)` or `EventSink::channel()` receives the events. The sink runs while an internal lock is held, which is what makes "no event after `close()` returns" true, so it must not call back into the session.
- *Opening.* Folders are walked for `*.ARW` (extension compared case-insensitively), one level unless `OpenOptions { recursive: true }` (the CLI's `--recursive`); other files and `.`-files (macOS's `._DSC0001.ARW` resource forks on exFAT cards) are ignored and counted in `OpenOutcome::ignored`; paths that do not exist are listed in `missing`; a file already in the session (same canonical path) is not added twice (`already_open`). A photo's path is made absolute against the working directory without resolving symbolic links (`ym render .` and `ym render DSC0001.ARW` work like their absolute forms, and the folder name and `<folder name> export` of a `.` or `..` folder come from the folder it resolves to). Photo ids count up for the life of the `Session`, also across `close()`, so a command that names a closed session's photo gets `UnknownPhoto` and can never reach a photo of the next session. What is found is sorted by path, case-insensitively, probed in parallel (`ym_raw::metadata::probe`) and registered before `open` returns, so the filmstrip has its photos at once. `OpenOutcome::no_arw()` / `message()` is the Start screen's "No ARW files in what you dropped." (nothing ARW at all; not when everything was already open). A probe that fails makes a photo that is Skipped with the `RawError` text of `docs/04`: the UI's CAN'T OPEN. The first photo of an empty session is current; `SessionState::folder_name` is the one folder given (or the one folder all photos share, else `None`), `default_export_folder` is `<parent>/<folder name> export` (the export queue still checks it is writable and falls back to `~/Pictures/Yellowmoon export`).
- *Jobs.* One job per photo on a pool of `min(available_parallelism - 2, 6)` threads (at least one; `default_decode_threads()`), CPU only: decode, `AutoEstimates::compute`, then the thumbnail hook (`ThumbnailRenderer`, implemented by the render graph; without one the photo is Ready after its estimates). The pool takes the current photo's job first. A failing job ends in Skipped for that photo only, a panic inside a development job is caught, logged with its message and reported the same way (a panic while prefetching a neighbour's mosaic is only logged, and the thread survives either way), and a decode error is `RawError`'s text (ARW 6.0 included, so the `arw6` feature needs no change here). `close()` bumps an epoch: queued jobs are dropped, running ones stop at their next stage and their results and events are discarded; the session is empty and reusable at once (it does not wait for the pool).
- *Events.* Per photo, in this order: `probe(done)`, `decode(start)`, `decode(done)`, `estimate(start)`, `estimate(done)`, `thumbnail(start)`, `ThumbnailReady`, `thumbnail(done)` (= Ready). A failure after the probe ends the sequence with one event whose `stage` is the failing one, `done = true` and `error` set (Skipped); a file that fails the probe has just `probe(done, error)`. `SessionEvent::EstimatesChanged { id }` is added to the list in this document: it is sent when other photos finishing moved a photo's shared exposure (below), and for the current photo when its pooled noise model changed.
- *Mosaic cache.* At most `MOSAIC_CACHE_LEN` (3) decoded mosaics: the current photo's and its neighbours' (by filmstrip index). A mosaic outside that window is dropped when its job is done; `Session::raw_image(id)` decodes again when something needs one (the render, Pick white, Re-run) and keeps it only if the photo is in the window. `set_current` re-fits the window and queues the missing neighbours (prefetch jobs run after development jobs). Previews are not kept: the 1/8 preview is 11 MB for 61 MP and exists only while the estimates are computed.
- *Estimates the photos share.* Burst median exposure (`docs/02` Auto exposure step 6) and pooled noise (`docs/06` Batch pooling) depend on the other photos, so the session keeps each photo's own `AutoEstimates` and serves `Session::estimates(id)` / `view(id)` with the burst median in `exposure.ev` (`own_ev` stays) and the pooled model in `noise`. Photos that do not state their ISO are not pooled (each keeps its own estimate: the ISO is part of the pool key and an absent one cannot be told apart). Both are deterministic functions of the set of photos (checked for 1, 3 and 4 decode threads): the burst groups are recomputed over every photo with estimates, the pool of a group (`NoisePool`) is built from its first 16 photos in id order, whichever finished first. An export or the CLI must therefore start after `wait_idle` (or after the photos it exports are Ready *and* no other photo is still developing); exporting early would use fewer pool members. A pooled model is used only if its variance at three signal levels is within 1.6 of the median of its members' own: `ym_noise::BatchPool` treats merged bins as independent samples, and on three frames of one scene its one-sided robust refit slid from the members' 5.1e-4 to 1.77e-4 (true 5e-4); each file then keeps its own estimate. Pooling frames that cover different intensity ranges works as intended (three such frames: a within 2% of the truth and b within 20%, which no single frame resolves).
- *Photo view.* `Session::view(id)` returns a `PhotoView` (path, probe metadata, `PhotoCamera`, profile, a snapshot of the settings, the served estimates, the Pick white sample, `estimates_version`) for the render graph and the export queue. `estimates_version` changes whenever the served estimates or the Pick white sample change (the sample is taken when the pick is set, so a new pick bumps it); it belongs in the render cache key of the stages after C1.
- *Camera.* `PhotoCamera` composes `ym_profiles::detect` with `ym_raw`'s `DecodeKind`: `reduced` (Lossless M or S), `stand_in` (anything but `Exact`), the `CameraCopy` of docs/03 and, for the RAW tab, `card()`: Detected, Profile, STAND-IN, the explanation, the REDUCED tag with the note of docs/04 and, for ARW 6.0 and other layouts without a decoder, the unsupported message.
- *Estimates in detail.* See `auto.rs`. The illuminant is `ym_color::illum`'s model for the camera's profile with the as-shot white balance as its prior (`RawImage::wb_as_shot` is stored per CFA position; `auto::as_shot_rggb` reads it by colour, so a BGGR or GRBG mosaic does not swap red and blue; every Sony body is RGGB, where the two agree). **Fallback** (a profile without a model file, or a model that fails): the as-shot white balance as a camera white `[g_G/g_R, 1, g_G/g_B]` (green is the mean of the two greens of SR2 0x7313), confidence 0.5, class probabilities from its CCT alone (tungsten up to 3500 K, daylight from 4500 K, a linear blend between, nothing for fluorescent and LED); a file with no white balance at all gets the profile's 5500 K daylight white with confidence 0. The 1/8 preview bins the photo's area (the camera's crop, else the active area, else the readout) 8 × 8 per colour; its clip mask marks a block when any photosite in it is at or above `white − 4·qstep` (or 0.995 of the range without a qstep plane). A Lossless M or S file has no mosaic: its preview comes from the RGB data, normalised with **one range for the three channels, the green one**, not channel by channel (`docs/04` gives red and blue lower white levels because the camera's white balance was divided out again: dividing each channel by its own range would put the camera's balance back, and the initial gains `g0` would then balance the data a second time; found in review, a neutral grey metered 0.4 EV too bright with a strong colour cast in the estimator's view; the render's normalise stage R1 must scale this layout the same way, so that channel `c` clips at `1 / g0_c` as `docs/02` C1 says; clipping is still judged against each channel's own white level), its noise model is the centre of the body's prior (`NoiseSource::PriorOnly`; ISO 100 is assumed when the file has no ISO), and a frame the estimator refuses (too small) is treated the same. Sharpening and noise-reduction Auto come from `ym_gpu::cpu` (`auto_sharpen`, `NoiseReduction::auto`) so the card and the render agree; the base ISO of the sharpening ramp is the gain-regime table's (100 when the body is not in it). Kelvin and tint are the Auto readout of the all-Auto, EV 0 chain.
- *Measured here.* Synthetic scenes (`crates/ym-core/tests/common`: a known light, brightness and Poisson-Gaussian noise): the as-shot fallback reads 2800 to 7500 K back within 50 K; auto exposure of a flat neutral scene lands on `log2(0.18 / Y)` within 0.08 EV (and follows the camera's exposure compensation at half weight); a ramp's noise is recovered within 1.1% in sigma and 15% in `a` (three seeds, ISO 800); the shipped illuminant model for the A7R IV profile finds the light of a patchwork scene within 0.55° from the pixels alone and within 0.49° with the right as-shot prior (2.9° at worst, and under half the camera's own error, when the as-shot white balance is badly wrong, daylight gains under 2800 to 6500 K light). On the nine real files of `testdata/real/files` (local, not committed) every file develops (all nine in 2.2 s on two threads); auto exposure is +0.6 to +1.9 EV (median +1.0). The M1-08 comparison on those nine (our render at `EV_auto` against the embedded JPEG, median luminance in linear light, `tests/real_files.rs`): 5 of 9 within 0.5 EV (the gate wants 80 % of the golden set; nine files decide nothing), errors in stops A300 −0.94, A500 +1.19, ILCE-5000 −0.46, ILCE-6300 −0.55, A7 II +0.30, A7 III −0.25, A7 IV −0.46, NEX-5N −0.28, NEX-7 +0.80: the spread is wider than the tolerance and runs both ways, so the starting constants of M1-08 need the golden-set tuning the spec expects. All 48 files of `testdata/synthetic` (every raw variant, with gradient, noise, clipped and dark content) develop; the dark ones meter to the +3 EV limit.

### DevelopSettings

Every control in the UI and every CLI flag maps to one field. Defaults reproduce the automatic development.

| Field | Type / range | Default | UI |
|---|---|---|---|
| `look` | `LookId` | user's default look (Neutral out of the box) | Look tab |
| `look_intensity` | 0–100 % | 100 | Look tab |
| `exposure` | `Auto` or `Manual(ev)` −4…+4, step 0.01 | Auto | Light › Exposure |
| `contrast`, `brightness` | −100…100 | 0 | Light › Exposure |
| `highlights`, `shadows`, `white`, `black` | −100…100 | 0 | Light › High dynamic range |
| `light_source` | Auto, Day, Shade, Tungsten, LED, Fluorescent | Auto | Colour › White balance (segmented) |
| `white_balance` | `Auto` or `Manual { kelvin 2000–11000 step 50, tint −50…50 }` | Auto | Colour › White balance |
| `color_balance` | shadows/midtones/highlights, each `(a, b)` in a unit disc | (0, 0) ×3 | Colour › Colour balance |
| `saturation` | −100…100 | 0 | Colour › Colour balance |
| `sharpen` | `Auto` or `{ amount 0–300, radius_px 0.3–2.5 }` | Auto | Detail › Sharpening |
| `noise_reduction` | `Auto` or `{ luminance, color, detail }` 0–100 | Auto | Detail › Noise reduction |
| `curve` | luma + R, G, B: up to 16 control points each in [0,1]² | identity | Curve |
| `decode` | `{ neural_demosaic, highlight_recovery, lens_corrections, chromatic_aberration }` bools | all true | RAW › Decode |
| `rolloff` | `{ highlights 0–1, low_light 0–1 }` | 0.5, 0.5 | RAW › Roll-off |
| `crop` | `Option<{ rect in [0,1]², angle_deg −45…45, aspect: Free/Original/1:1/4:5/16:9 }>` | None | Crop (header) |
| `orientation` | 0, 90, 180, 270 degrees, applied after the camera's EXIF orientation | 0 | Crop › Rotate 90° |
| `white_pick` | `Option<(x, y)>` sampled point | None | Pick white (header) |

"Auto" values resolve against `AutoEstimates` at render time. The UI shows the resolved value with an AUTO tag; moving a slider turns it to `Manual`. "Reset" restores `Auto`. "Re-run" on the Automatic card recomputes `AutoEstimates` and resets exposure, WB and noise to Auto.

**Applying to several photos.** The Look card's "Apply to" (This photo / Picked / All) copies `look` and `look_intensity` to those photos. Other controls apply to the current photo only. Export can target the current photo, picked photos or all.

**As built (ym-core: `settings.rs`).** `DevelopSettings`, its patch and `ResolvedSettings` have the serde shape of `src/api/types.ts` (snake_case, `kind`-tagged unions); a test compares the field names of every wire type with its interface in that file, so a drift fails in CI. The same JSON is the `ym:Settings` XMP property, the CLI's `--settings` and the golden manifest; `crates/ym-core/schema/settings.schema.json` is its JSON Schema (tested against what the core writes, against documents it refuses, and against the ranges the core clamps to).

- *Reading.* Missing fields take their defaults (a settings file may hold only what it changes), unknown fields are an error (a typo is not a silent no-op). `look` is an `ym_looks::library::LookId`. `orientation` is a number of degrees; any multiple of 90 is read and normalised (-90 is 270). Writing always writes every field. `DevelopSettings::from_json` / `from_value` read a document over the defaults of a photo opened with the user's default look (a document without `look` gets `default_look`, where `DevelopSettings::default()` alone says Neutral), clamp it like a patch and report the fields that were changed; a misspelt field or a value of the wrong type is `SettingsError::Parse` with serde's message. `ym render --settings` should read its file with it.
- *Clamping.* A number outside its range is clamped, not refused; `PatchReport::clamped` names the fields. What is refused (`SettingsError`, nothing applied: patches are atomic) is a value that is not a number, a curve that is not a curve (2 to 16 points; they are clamped into the unit square and sorted by x; two points at one x are refused), a crop without area, and a Pick white spot the chain refuses. Colour-balance points longer than 1 are scaled back to the circle; a crop rectangle is moved and shrunk into the photo, its angle clamped to ±45°. Steps (0.01 EV, 50 K) are the UI's granularity: the core keeps the value it is given, so `--wb 5025` means 5025 K.
- *Patch.* `SettingsPatch` is `Partial<DevelopSettings>` with `crop: null` and `white_pick: null` clearing the field (absent leaves it). Two generosities beyond the TypeScript type: `decode`, `rolloff`, `color_balance` and `curve` may carry only the parts to change, and a manual `white_balance`, `sharpen` or `noise_reduction` may leave out a part, which then keeps its current resolved value (a patch with only `kelvin` on an Auto white balance is Manual at that Kelvin and the Auto tint: that is what moving one slider means). `apply_patch` reports the fields that changed and the first render-stage group they affect (`StageGroup::{Raw, Colour, Grade, Geometry}`, the table below; `SettingsField::group()`), which is what the render graph invalidates by. **Linked fields:** a patch with a `white_pick` also sets `light_source` and `white_balance` to Auto unless it says otherwise, so the picked spot renders neutral and the Kelvin and tint rows show the picked light (`docs/11` Pick white: "the core sets the white balance from the pick"; it does it through the scene light, `docs/02` C1). The frontend records undo from the settings before and after a change, so the extra fields are restored with it. A preset (`light_source`) does not clear a pick: the chain applies the pick first and the preset after it, so Day, Shade and Tungsten override it and LED and Fluorescent keep the picked chromaticity.
- *Pick white.* `white_pick` is `[x, y]` as fractions of the photo **as the camera oriented it** (EXIF orientation applied; before the user's crop, straighten and `orientation`, so the point keeps pointing at the same spot when those change). `auto::display_to_sensor` maps it back to the sensor. The session samples the 5 × 5 median of g0-balanced camera RGB at that point when the pick is set (`auto::pick_white_sample`: a cheap 3 × 3 same-colour demosaic on the raw data without the vignetting gain, which scales all three channels alike) and keeps the sample with the photo; `PhotoView::picked_white()` is what goes to the chain. The viewport (part 2) converts clicks in the photo cell to this space before it sets the pick.
- *Resolving.* `ResolvedSettings::resolve(photo_id, settings, estimates, profile, picked_white)` builds the fused chain (`AutoEstimates::fused_chain`) and takes the Kelvin and tint from it, so with `white_balance` Auto they follow a light preset (Day 5500 K, Shade 7500 K, Tungsten 3200 K, tint 0; LED and Fluorescent keep the estimate's CCT) and a picked white. `AutoFlags` says which of exposure, light source, white balance, sharpening and noise reduction are still Auto; the UI's "white balance counts as automatic only when `white_balance` is Auto, `light_source` is Auto and no white was picked" is the UI's rule on top of these flags. `ResolvedSettings::pinned_settings()` is the document an export records as `ym:Settings` and the CLI prints as `settings`: exposure, sharpening and noise reduction that are Auto are written as the manual values they resolved to (a manual value renders exactly like the Auto it came from, and the file then says which numbers were used), white balance stays Auto when it was (a manual white balance at the shown Kelvin and tint does not render like Auto: the tint scales the green gain and Kelvin is rounded to 50 K, so the numbers it resolved to go in the `auto` object of the CLI output instead).
- *Session commands* (all on `Session`, same names as the IPC commands): `state()`, `set_current`, `set_picked`, `pick_all`, `clear_picks`, `settings(id)` (`get_settings`), `update_settings(id, patch)`, `replace_settings(id, settings)` (the CLI's `--settings`: clamped like a patch), `apply_look(scope, look, intensity)` (This photo is the current photo; skipped photos are left out; intensity clamped), `rerun_auto(id)` (recomputes the estimates on the calling thread, sets exposure, `light_source`, `white_balance`, `white_pick` and `noise_reduction` back to Auto and keeps sharpening and everything else), `set_default_look`. Settings live in memory with the photo and nowhere else (hard rule 4). Commands may arrive from several threads (one per slider move; a Pick white decodes the photo again, which takes a while): `update_settings` applies its patch to the settings the photo has when it commits and starts again from the current ones if another command changed them meanwhile, so two commands on different fields both take effect in either order; `replace_settings` is the last word and never retries. A command on a session that was closed meanwhile returns `UnknownPhoto`.

### Render graph

Each photo's render is a fixed chain of stages (`docs/02` and `docs/05`). Each stage has a cache key made from its inputs. Changing a setting invalidates only the stages after the first one it affects:

| Stage group | Invalidated by |
|---|---|
| Decode, black/white, clip mask | never within a session |
| Noise estimate, illuminant estimate, auto exposure | Re-run |
| Vignetting, highlights, demosaic+denoise, detail controls, sharpening, lens resample (R3–R9) | `decode.*`, `noise_reduction`, `sharpen` |
| WB, sensor map, ARRI matrix, AWG4, exposure, roll-off and toe, LogC4 (C1–C8) | `light_source`, `white_balance`, `white_pick`, `exposure`, `rolloff.*` |
| Primary grade, look, curve, DRT, display encode (C9–C13) | everything else |
| Geometry (crop, straighten, orientation) | `crop`, `orientation`: applied when sampling for display and export; no pixel stage re-runs |

Most edits only re-run the last group, which is cheap (one pass of per-pixel maths and two LUT lookups) and must stay interactive (< 16 ms for the viewport at screen resolution).

### Preview strategy

- **Viewport at fit:** classical demosaic at reduced resolution (half or quarter, chosen so the image covers the viewport at ≥ 1 device pixel per image pixel), full colour pipeline.
- **Viewport at 100% and above:** neural demosaic+denoise on the visible tiles plus a margin, computed in the background; the classical result shows until each tile is ready.
- **Thumbnails:** quarter-resolution classical path, rendered once after open and on look changes.
- **Export:** full resolution, neural path (unless `decode.neural_demosaic` is off or no usable GPU).

The status text in the header ("Neural · 42 ms · Display P3") shows the active demosaic path, last render time and output space.

### Scheduling and threads

- **Decode pool:** `min(available_parallelism − 2, 6)` threads, CPU only. Decoding and metadata parsing never touch the GPU.
- **GPU thread:** one thread owns the wgpu `Device` and `Queue`. All GPU work is submitted through a priority queue: viewport frames > visible-tile neural passes > thumbnails > export.
- **Cancellation:** every job carries a token. Moving to another photo cancels its pending viewport and tile jobs. Export jobs are never cancelled by navigation.
- **Memory budget:** keep at most 3 decoded mosaics (current, previous, next) and preview buffers for all photos in the session. Full-resolution intermediates live only while a tile job needs them. Budget defaults: 4 GiB GPU, 6 GiB CPU; read the actual limits from wgpu and the OS and scale down.

### GPU tiling

- A 50 MP frame in RGBA f32 is ~0.8 GiB and exceeds wgpu's default 256 MiB buffer and 8192 px texture limits. Process in tiles of 1024 or 2048 px (configurable) with halos: half the network's receptive field (computed from the architecture; 64 full-res px for the starting U-Net), plus the resample's maximum displacement for the lens stage. A tiled-versus-whole-image test guards against seams.
- Storage format f16 (`SHADER_F16` when available, else f32 storage with f16 packing). Per-pixel colour maths in f32 registers.
- Tiles are independent after decode, so export runs tile by tile and writes rows to the encoder as soon as a tile row is complete.

## Native viewport

The webview covers the whole window and is transparent where the photo is. A wgpu surface (CAMetalLayer) sits behind it inside the same NSWindow.

1. The frontend measures the photo cell (`<div data-viewport>`) and sends `viewport_set_rect { x, y, width, height, scale }` on mount, resize and layout change (ResizeObserver).
2. `src-tauri` positions the Metal layer to that rect and asks `ym-core` for frames.
3. Pointer input over the photo arrives in the webview first. The frontend sends `viewport_input` events (zoom to point, pan delta, double-click to toggle fit/100%, white-pick click) to the core. M0's spike measures latency; if it is above one frame, move pan/zoom handling to native event monitoring and keep only clicks in the webview.
4. Surface colour space: Display P3 with an sRGB-encoded 8-bit or 10-bit format; the DRT outputs Display P3 for the viewport. HDR/EDR is V2.
5. The photo cell is always exactly 3:2 (see `docs/11-ui.md`); the image is letterboxed inside it with the paper colour, never with grey.

M0 task M0-06 proves this design or switches to the fallback (a native child view placed by the same rect messages).

## IPC between webview and core

Use Tauri commands for requests and Tauri events for progress. Generate TypeScript types from Rust with `specta` + `tauri-specta` (MIT) so the two sides can't drift. The frontend also has an `api.mock.ts` (as in plugout) so it runs in a browser for Vitest and Playwright without Tauri.

**Commands** (request → response)

| Command | Purpose |
|---|---|
| `open_paths(paths)` / `open_dialog(kind: files or folder)` | Start a session or add photos |
| `session_state()` | Photos, current index, picked set, per-photo summary |
| `set_current(photo_id)`, `set_picked(photo_id, bool)` | Navigation and picking |
| `get_settings(photo_id)` → resolved settings with Auto flags | Panel state |
| `update_settings(photo_id, patch)` | Any control change; returns resolved settings |
| `apply_look(scope, look_id, intensity)` | Apply to This photo / Picked / All |
| `rerun_auto(photo_id)` | Recompute automatic estimates |
| `viewport_set_rect(rect)`, `viewport_input(event)`, `viewport_set_mode(fit or 100)`, `set_before(bool)` | Native viewport |
| `sample_pixel(x, y)` → RGB display values | Readout and white pick |
| `looks_list()`, `looks_import(path)`, `looks_set_default(id)`, `looks_remove(id)` | Look library |
| `export_estimate(job)` → bytes and seconds | Size and time estimate in Export |
| `export_start(job)`, `export_cancel(job_id)` | Export queue |
| `cli_preview(job)` → string | The equivalent `yellowmoon render …` command |
| `preferences_get()`, `preferences_set(patch)` | Preferences |
| `install_cli_link()` | "Install command-line tool" menu item |

**Events** (core → webview)

`photo_progress { id, stage, done }`, `thumbnail_ready { id, png_bytes }`, `render_stats { ms, path, space }`, `export_progress { job_id, file, done, total, error? }`, `looks_changed`, `update_state { phase, version?, percent?, message? }`, `error { scope, message }`.

Thumbnails cross the bridge as small PNGs (≤ 240 px long edge). Full images never cross it.

## Persistence

Only these persist, under `~/Library/Application Support/app.yellowmoon/`:

```
preferences.json                  default look, display rendering (ACES 2 or an imported ARRI display LUT),
                                  export defaults (format, quality, bits, compression, colour space, size,
                                  destination, name template, if-exists), remove location from exports,
                                  check for updates automatically, watch Downloads during import,
                                  last-used folders
looks/
  manifest.json                   imported looks and display LUTs: id, name, group, input encoding,
                                  source URL, import date, SHA-256 per file (no preferences)
  arri/<group_no>-<group_slug>/   imported ARRI look files (copied, never modified)
  arri-display/                   imported ARRI display LUTs
  user/                           other imported .cube files
logs/yellowmoon.log               local only, rotated at 5 MB × 3
```

This list is exhaustive (hard rule 4). Anything else that persists needs Mykyta's agreement.

Plus `~/Library/Caches/app.yellowmoon/` for the wgpu pipeline cache only. No image data is cached on disk. The CLI uses the same folders and takes a file lock (`looks/.lock`) when importing.

## Finding `data/`

The app and the CLI look for the data folder in this order:

1. `YELLOWMOON_DATA` environment variable (tests, development).
2. Next to the canonicalised executable inside the bundle: `Contents/MacOS/<exe>` → `Contents/Resources/data/`. Canonicalise first, because the CLI on PATH is a symlink into the bundle.
3. Debug builds only: `<workspace root>/data/`, from `CARGO_MANIFEST_DIR` at compile time.

If none exists, fail with "Yellowmoon's data folder is missing. Reinstall the app." and exit code 2 in the CLI.

## Errors and logging

- `tracing` with a local file subscriber. Log levels: info for user actions and job summaries, debug for stage timings. Never log file contents or full paths at info level in a way that would leak into bug reports without consent; `yellowmoon doctor` prints what it includes.
- Every error that reaches the UI carries a plain message and an action ("This file is from a camera mode we can't read yet: ARW 6.0 compressed. Export it as uncompressed or lossless in the camera, or wait for an update.").
- A file that fails to decode stays in the filmstrip with an error tag; it never blocks other photos.

## Versions written into outputs

- `pipeline_version`: semver of the processing chain, bumped when output pixels change for the same settings.
- Camera profile id and version, model ids and versions, look id and SHA-256, DRT id.

These go into XMP (`docs/09-export.md`) so any exported file can be traced.

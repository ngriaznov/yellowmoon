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

# Yellowmoon development plan

2026-10-11 · @Mykyta Hriaznov

## Summary and scope

We are building Yellowmoon, an open-source desktop raw processor that reads only Sony ARW files and renders them the way an ARRI ALEXA would. It has few controls and aims for Capture One–level raw quality. It is a Tauri app with a Rust core and auto-update, licensed Apache-2.0. The name is written as one word and stays clear of "ARRI" and "ALEXA".

- **V1 (MVP to 1.0):**
  - macOS only, universal build like plugout.
  - ARW files from any Sony camera: uncompressed, lossless (L) and lossy cRAW. Eleven Sony cameras get their own measured colour profile, cameras that share one of those sensors use it, and the rest use the A7R IV.
  - ALEXA ALEV III colour (Classic through Mini LF), from Weta's A7R IV curves and our ALEXA estimate.
  - Learned demosaic and denoise.
  - Looks: shipped defaults plus an ARRI import wizard.
  - SDR export: JPEG, TIFF, PNG, and DNG with the look baked in.
  - A command-line tool on the same core, so the app fits into scripts and pipelines.

- **V2:**
  - More camera profiles where measured data exists, starting with the A1.
  - HDR and gain-map export.
  - Windows and Linux, if there's demand.
  - Better far-red data for the ALEXA curve, if HdM or ARRI release measured curves.

- **Out of scope:** ALEXA 35 and REVEAL colour, video, catalog and library management, saved edits and sidecar files, tethering, local masks and retouching.

## Decisions settled

The research reports and the curve experiment settle these; we revisit one only if new evidence appears.

| Decision | Evidence |
|---|---|
| Platform for V1: macOS only, universal build | Your decision |
| Viewport: native wgpu surface on Metal; the webview carries only panels | Your decision; wgpu's Metal backend supports all of its HDR surface colour spaces |
| Desktop shell: Tauri 2 with auto-update in plugout's style | Your decision; RapidRAW shows Tauri works for a raw editor |
| Sony curves: Weta ILCE-7RM4 (Apache-2.0, rawtoaces-data), applied to files from every Sony body | The A1's Adobe matrix is closest to the 7RM4/7RM5; recent Sony bodies differ by 0.5–1.5% of peak |
| ALEXA curves: our matrix-fitted ALEV III estimate, shipped as an estimate | The only measured ALEXA curve (HdM SXT) is unlicensed and flagged wrong. Leave-one-out test: 2.2% median curve error, 3–5% expected for the ALEXA |
| The D-21 curve is not used as "the ARRI sensor" | Ranks 53 of 54 under ARRI's matrices (3.95 mean ΔE00) |
| Sensor map: white balance first, then a 3×3 per illuminant whose coefficients come from the illuminant estimate; root-polynomial optional | Per-light 3×3 gives 0.16–0.45 ΔE00 median across 9 lights; one global 3×3 gives 1.19 under tungsten; root-polynomial adds ≤ 0.13 |
| No static sensor LUT; no per-pixel spectral reconstruction | LUTs ignore the illuminant; learned spectral reconstruction drifts with exposure |
| ARRI colour step from ARRI's 17-CCT matrices (aces-dev, AMPAS licence), mired-interpolated | ARRI-supplied, permissively licensed |
| Working encoding LogC4/AWG4; LogC3 looks run in LogC3 | LogC4 is ARRI's current look ecosystem; the conversion between them is exact maths |
| Shipped display transform: ACES 2 or our own; ARRI's LUTs only after the user imports them | ARRI's LUTs and looks are all rights reserved; ACES 2 output is Apache-2.0 |
| ARRI files: user imports them through a wizard; no bundling and, for now, no in-app download | All six packages are free, carry no licence text, and arri.canto.de's robots.txt disallows automated access |
| Learned joint demosaic and denoise, 0.2–1 M parameters, classical fallback | Adobe, DxO and Capture One ship learned denoise; NTIRE 2025 efficient-track results |
| Noise calibration per ISO and channel on our own A1; no Bulb frames | ELD-style noise models; DxO calibrates every body |
| Lens corrections from Sony's embedded tags, reimplemented | Lensfun's database is CC BY-SA |
| No GPL/AGPL code copied; shipped weights trained only on CC0, CC BY, Apache or our own captures | Licence survey |
| Command-line tool in V1, on the same Rust core, with output identical to the app's export | Your decision; it also serves as the golden-image test harness |
| Nothing is saved per photo: open, adjust, export, done. Only imported looks and a few preferences, such as the default look, persist | Your decision |
| UI: bold grid on white. Thick black rules, Figtree, a flat colour block per tool tab, square boxed controls; the photo fills its frame edge to edge. Yellow is the brand colour: the moon mark, the start screen and the Look tab | Your decision, from the design rounds (Precision glass and a dark variant were the alternatives) |
| The app opens ARW from every Sony camera; colour mapping uses the A7R IV curves for all of them | Your decision |
| The camera is detected from every file, and colour, noise and decode details come from a per-camera profile; V1 ships 11 measured Sony profiles (Weta, Apache-2.0), same-sensor cameras share them, and the A7R IV covers the rest | Your decision; adding a camera later is a data file and tests |
| Name: Yellowmoon, written as one word; the command is yellowmoon | Your decision; no software or Rust crate uses the name |

## Architecture

All image processing runs in the Rust core on the GPU. Two front ends call it: the Tauri app, whose webview only shows results and sends commands, and a command-line tool with no webview. The frontend follows plugout: React and TypeScript.

[diagram: app architecture · app and CLI, Rust core, stores]

The photo is drawn natively: the Rust core renders into a wgpu surface on Metal, and a transparent webview on top carries the panels. M0's first spike proves this compositing, Display P3 output and frame time.

| Layer | Pick | Notes |
|---|---|---|
| Shell | Tauri 2, `tauri-plugin-updater`, `tauri-plugin-process`, `tauri-plugin-dialog` | Same set as plugout |
| Decoder | In-house ARW decoder, Apache-2.0 | rawler and LibRaw only as CI oracles |
| GPU | wgpu 30 on the Metal backend, WGSL compute, tiles of 1–2k px in f16 | A 50 MP RGBA f32 frame (~0.8 GiB) exceeds default buffer limits |
| ML runtime | burn 0.22 with burn-onnx on the shared wgpu device | ort (ONNX Runtime) as benchmark and fallback |
| Colour | `colorimetry` crate plus our own LogC3/4, AWG3/4, ARRI matrices, ACES 2 port | References: colour-science, OCIO, ACES repos |
| Colour management | moxcms | Pure Rust ICC |
| Export | tiff (also under our DNG writer), exr, jpeg-encoder, ravif | Avoid `dng` (AGPL), rawler's DNG writer (LGPL) and jpegxl-rs (GPL) |
| Training | PyTorch and colour-science | Export safetensors plus static-shape ONNX |
| Command line | clap, headless wgpu on Metal, serde_json output | Separate binary in the same workspace; no Tauri dependency |
| Camera profiles | Registry keyed by Sony model ID; one versioned data file per profile | Default profile A7R IV; per-camera decoder hook for quirks |

## Colour science pipeline

Colour passes through one small learned step, the sensor map. After it, everything follows ARRI's published processing or the look and display transform the user picks.

[diagram: colour pipeline · 11 stages, 3 inputs]

- **Scene light and your CCT stay separate.** The illuminant estimate drives the sensor map. Your CCT drives ALEXA's white balance and ARRI's matrix. So "3200 K under daylight" looks the way it would on an ALEXA.
- **Sensor map coefficients** come from a tiny network that reads the illuminant estimate (chromaticity plus light-type probabilities). When the type is uncertain, it blends per-light maps. The map is exposure-invariant and white-preserving by construction.
- **ALEXA white-balance gains** come from our estimated curve, since ARRI's own gains are unpublished.
- **Out-of-gamut LED colours** pass through unclipped and are compressed only at the display transform, as ACES's reference gamut compression does.
- **Highlight roll-off** is a display-transform decision. The A1 clips earlier than an ALEXA at matched mid-grey.
- **Data files:** our `alexa_alev3_estimate_380_780_5` curve, Weta's `Sony_ILCE-7RM4` curve, and ARRI's `Alexa_aces_matrix.txt` from aces-dev.

## Camera detection and profiles

V1 ships a measured profile for every Sony camera whose curves are public and permissively licensed: 11 bodies from Weta's rawtoaces-data set. Cameras that share a sensor with one of them use its profile, and the A7R IV covers the rest. The code treats the camera as an input from day one, so adding a camera later means adding a profile; the pipeline stays as it is.

**Detection**

- Read Make and Model from the main IFD and Sony's model ID (tag 0xB001) from the maker note. The model ID wins when they disagree.
- The result names the body, its sensor family and the compression used in the file.
- The app shows the detected camera and the profile in use with each photo, and the CLI's `info` prints both.

**What a profile holds**

| Part | What it holds | V1 |
|---|---|---|
| Spectral curves | The camera's sensitivity curves, input to the sensor map | Weta measurements for 11 bodies |
| Sensor map | Per-light 3×3 coefficients or the trained network weights | Fitted for each profile |
| Decode | Black and white levels, crop, compression quirks | Read from each file |
| Noise | Per-ISO noise model | Estimated per file, A1 profiles as reference |
| Highlights | Clip point and recovery settings | From each file's white level |
| Lens | How to read the embedded correction tags | Sony tags 0x7032, 0x7035, 0x7037 |

**Which profile each camera gets**

| Cameras | Profile |
|---|---|
| A7 III, A7 IV, A7C II, A7R II, A7R III, A7R IV, A7S II, A7S III, A9, A6400, RX100 IV | Their own measured curves |
| A7R III A | A7R III (same sensor) |
| A7R IV A, A7R V, A7CR | A7R IV (same 61 MP sensor) |
| A7C | A7 III (same sensor) |
| FX3, ZV-E1 | A7S III (same sensor) |
| A9 II | A9 (same sensor) |
| A6100, A6600 | A6400 (same sensor) |
| A1 and every other body | A7R IV, until measured |

Shared sensors come from published specifications, and each pairing is checked against the camera's colour matrices before it ships.

**What the check showed** (scripts/sony_profiles.py, 11 October 2026)

We mapped each measured body's colours to the ALEXA rendering with its own profile and with the A7R IV profile. Numbers are ΔE00 median / 95th percentile over 138 test colours under 9 lights; the map itself costs about 0.3 even with the right profile.

| Camera | Own profile | A7R IV profile |
|---|---|---|
| A7 IV | 0.34 / 0.84 | 0.37 / 0.82 |
| A7C II | 0.36 / 0.92 | 0.40 / 0.92 |
| A7R III | 0.25 / 0.65 | 0.33 / 0.74 |
| A7 III | 0.27 / 0.72 | 0.44 / 1.05 |
| A7R II | 0.27 / 0.65 | 0.58 / 1.13 |
| A7S III | 0.28 / 0.76 | 0.59 / 1.45 |
| A9 | 0.37 / 0.99 | 0.63 / 1.48 |
| A6400 | 0.35 / 0.98 | 1.20 / 3.13 |
| RX100 IV | 0.31 / 1.05 | 1.60 / 3.63 |
| A7S II | 0.47 / 1.55 | 2.69 / 4.43 |

- Recent full-frame bodies lose little on the A7R IV profile; the A6400, RX100 IV and A7S II lose a lot, so measured profiles pay off from day one.
- Curves estimated from two published colour matrices (Solomatov and Akkaynak, 2023) were worse than simply using the A7R IV profile for 8 of 10 bodies, and too noisy to pick a stand-in camera. We don't use estimated Sony curves; stand-ins come from shared sensors.

**Lookup and extension**

- Lookup goes exact model, then sensor family (bodies that share a sensor), then the default A7R IV profile. A Sony file is never refused for lack of a profile; the app falls back and says which profile it used.
- Profiles are versioned data files in the app bundle (curves, coefficients, noise JSON), found through a small registry keyed by model ID. Adding a camera is a data file plus its tests. Code changes only when a body has new decoding quirks, through a per-camera hook in the decoder.
- The profile name and version go into each exported file's metadata, so a later profile change can be traced.
- The CLI's `--profile <name>` forces a profile, for comparing a new one against the default.
- A new profile needs measured spectral curves, noise frames and a test set, and must pass the same gates as the others. Curves fitted from two published colour matrices proved too rough for Sony bodies; the ALEXA fit worked because ARRI publishes matrices at 17 colour temperatures. The A1 comes first, in V2, measured on your own camera.
- The ALEXA target uses the same kind of profile, so a second target such as the ALEXA 35 would slot in the same way.

## Raw processing

Every stage runs in scene-linear float on the GPU, in tiles. The learned demosaic-and-denoise network is where most of the Capture One–level quality comes from.

1. **Decode.** Our own Apache-licensed ARW decoder: TIFF walk and SR2 decryption for black, white and white-balance tags (0x7300/0x7310, 0x787f, 0x7313). It reads the Sony raw variants listed under Sony raw formats below. Black and white levels and bit depth come from each file (A1: 512 and 16383, measured on your files; compressed modes can be 12-bit). rawler (LGPL) and LibRaw serve only as CI test oracles. A variant we can't read yet gets a clear message naming the camera and format.
2. **Calibrate.** Per-ISO noise profiles (shot, read and row noise) from lens-capped dark frames and flat fields shot on our A1, fitted separately for each gain regime (the A1 switches gain near ISO 500). Every other body gets its noise estimated from each file; see Noise for every camera below.
3. **Lens corrections.** Read Sony's embedded tags: 0x7032 vignetting, 0x7035 chromatic aberration, 0x7037 distortion, up to 16 knots each. Apply vignetting before demosaic; apply distortion and CA together as one resample afterwards.
4. **Highlights.** Reconstruct clipped channels on the mosaic from the opposing channels, written clean-room from darktable's published description. The clip mask travels with the image.
5. **Demosaic and denoise.** One small U-Net, 0.2–1 M parameters, taking the mosaic, a noise-σ map and the clip mask. Training data:
  - synthetic mosaics with calibrated A1 noise;
  - PixelShift200 (Sony, CC BY 4.0);
  - our own A1 16-shot pixel-shift captures as ground truth.

6. **Fallback.** A classical demosaic (Menon or LMMSE, from permissive references) for previews and machines without a usable GPU.
7. **Capture sharpening.** Richardson–Lucy deconvolution with a Gaussian PSF on linear data, clean-room. It may later fold into the network.
8. **Preview strategy.** Fit-to-screen uses the classical path; 1:1 view and export use the network. One 50 MP frame is about 10.6 TFLOP: sub-second on a modern GPU, tens of seconds on CPU.

**Sony raw formats** (from the LibRaw, rawler, rawspeed and ExifTool sources, 11 October 2026)

| Variant | Bodies | Lossy | Bayer | Open-source decoders | V1 |
|---|---|---|---|---|---|
| 12-bit uncompressed (ARW 1 and 2) | A100 to the A900 era | No | Yes | LibRaw, rawler, rawspeed | Yes |
| cRAW (11-bit base plus 7-bit deltas) | A700 (2007) to current bodies | Yes | Yes | All three | Yes |
| 14-bit uncompressed | A7R II (2015) onward; not the A7 V | No | Yes | All three | Yes |
| Lossless L (tiled lossless JPEG) | A1 (2021) onward | No | Yes | All three | Yes |
| Lossless M and S | A7R V, A9 III, A1 II, RX1R III | No, but camera-processed | No: white-balanced, chroma-subsampled YCbCr | LibRaw, rawler | Yes, reduced path |
| Compressed RAW 2 (ARW 6.0, wavelet) | A7 V, A7R VI | Yes | Yes, rebuilt from planes | LibRaw master only; lightcraft (Rust, MIT/Apache, 14-bit only) | Yes, last (M2) |

- ARW 5.0 (A9 III, A1 II, A1 firmware 2.0) changed metadata only. Firmware updates can still add modes, so new files go through the CI sample run before we claim support.
- M and S files are already white-balanced and demosaiced, so the sensor map runs on reconstructed camera RGB. Quality is lower and the app says so.
- ARW 6.0's compressed modes use a wavelet codec based on SMPTE RDD 34, with constants that aren't stored in the files. RDD 34 sends IP enquiries to Sony, so we ask before shipping our decoder. Until then those files get the clear "not supported yet" message.
- Out of scope: ARQ pixel-shift files and the older SRF and SR2 formats.
- Samples: RawDB (rawdb.dnglab.org, CC0) and raw.pixls.us (CC0) cover most variants, ARW 6.0 included. We shoot or borrow the gaps: FX2, RX10 V, 12-bit ARW 6.0, A1 M and S.
- Any camera without its own colour profile uses the A7R IV profile.

**Noise for every camera**

- Model per image: variance = a·signal + b, plus row noise and quantisation. One shot-noise gain a is shared by the four colour planes, with a small black-level offset per plane; that offset was the biggest single gain in the ELD study.
- Estimator, written from the papers because the reference code licences don't allow reuse: a Foi-style fit on smooth regions of each half-resolution colour plane, with a clipping-aware fit near white; a PCA-based fallback for heavily textured photos; one refinement pass after a first denoise.
- Priors: the A1 calibration (log a linear in log ISO, per gain regime). Other bodies get a wide prior around a ∝ ISO and a table of gain-switch ISOs we build ourselves. Files from the same camera and ISO in one batch share their estimate; nothing is saved.
- Sony quirks: modern ARW files keep no usable masked pixels; the 16-bit ISO tag reads 65535 above that value; extended and top "Hi" ISOs don't follow the trend; some bodies filter long exposures spatially ("star eater"). A check of 2×2-binned residuals spots filtered files, and we then raise σ, since over-estimating is the safer error.
- Training: the network learns on "virtual cameras" sampled the ELD way, with the σ map jittered by about ±25%, so it tolerates estimate error.
- Validation: the A1 calibration is the truth; ELD's released Sony A7S II parameters are a second body; injecting known noise into raw.pixls.us files tests the estimator across many bodies. darktable's noise profiles (GPL) are a reference only and never shipped.

## Looks, defaults and the ARRI import wizard

The app ships with two looks of its own and lets users import ARRI's free Look Library and display LUTs through a wizard; we never host or bundle ARRI files.

**Shipped defaults**

- **Neutral** (the default): emulated ALEXA colour, our display transform, no creative look.
- **Signature look**: designed by you as parameters in LogC4 (contrast, white-balance shifts, hue and saturation curves), so it ships under Apache-2.0 and adapts to each image. Its name describes its character and avoids ARRI and film-stock names.
- Every look has a 0–100% intensity slider. The user's chosen default persists.

**What users can import** (verified on arri.com, 11 Oct 2026)

| Package | Size | Contents | Role in the app |
|---|---|---|---|
| Look Library, LogC4 log-to-log | 181 MB | 87 .cube, 65-point | Recommended looks |
| Display LUTs, LogC4 | 32 MB | 22 .cube (Rec.709, P3, Rec.2020, HLG, PQ) | ARRI's exact display rendering |
| Look Library, LogC3 log-to-log | 30 MB | 87 .cube, 33-point | Looks run in LogC3 |
| Look Library, LogC3 to Rec.709 | 32 MB | 87 .cube | Complete rendering; replaces our display transform |
| Display LUTs, LogC3 | 46 MB | 48 .cube, including the classic Rec.709 | Classic ALEXA rendering |

All packages are direct public downloads with no login or click-through, and contain no licence text. ALF4 and ALF2 versions are binary containers we skip.

**Wizard**

1. **Choose.** Look Library (LogC4) is checked; display LUTs (LogC4) is optional. One line explains that ARRI publishes these free and the app can't include them.
2. **Get the files.** "Open ARRI's page" launches the system browser on ARRI's Look Files page. The wizard then waits: it watches Downloads if the user allows, and accepts "Choose file…" or a dropped zip.
3. **Recognise.** Identify the package from folder and file names and LUT sizes: look group numbers, log-to-log or Rec.709-baked, LogC3 or LogC4. A wrong package gets a plain explanation.
4. **Import.** Copy only the .cube files into the app's library folder, grouped as ARRI groups them. Record source URL, date and SHA-256 per file.
5. **Finish.** Thumbnails of the looks on the current photo; ask whether to use ARRI's display rendering by default and which look to make default; offer to move the zip to the Trash.

**Rules**

- Any other .cube uses the same importer plus one question: the input it expects (LogC4, LogC3, or Rec.709 baked in).
- If ARRI agrees by email, we add one-click download using a manifest of URLs and checksums kept outside the app, with import as the fallback.

## DNG export

DNG export carries the look. A Yellowmoon DNG opens in other raw editors looking like our render, and keeps room to adjust exposure, highlights and white balance.

- Pixels: demosaiced, denoised, lens-corrected linear data (LinearRaw, 3 × 16-bit) in ARRI Wide Gamut 4, with the look applied in LogC4 and converted back to linear. The look's colours survive in every app that opens linear DNGs.
- Profile: matrices that describe linear AWG4, plus a tone curve and a look table fitted to our default display transform (ACES 2), so Adobe's pipeline shows nearly what we show. We fit to ACES 2, never to ARRI's display LUTs.
- A unique camera model name ("Yellowmoon AWG4") keeps Lightroom from swapping in Adobe's Sony profiles. A full-size JPEG preview holds the exact render.
- Tags mark noise reduction as done and black as already subtracted. Adobe's dng_validate checks every exported file in CI.
- White-balance changes made later in another app act after the baked look, so they're approximate.
- ARRI's looks are baked into pixels only, never written as profile tables: a table derived from ARRI's LUT would put a copy of their data into every file.

| App | What it shows |
|---|---|
| Lightroom, Camera Raw | Close to Yellowmoon's render |
| RawTherapee, darktable | The look's colours under their own tone curve |
| Capture One | The look's colours; exposure may be off, as with other linear DNGs |
| DxO PhotoLab | Doesn't open third-party DNGs |
| Apple Photos, Preview, Quick Look | The embedded preview or Apple's own rendering (unverified) |

- Writer: our own, on the `tiff` crate (MIT), with uncompressed strips first. The `dng` crate (AGPL) and rawler's writer (LGPL) are out.
- Adobe's DNG patent licence is royalty-free. It requires the notice "This product includes DNG technology under license by Adobe." in our source and documentation, and ends if we bring a patent action against Adobe.

## Command line

The app ships a command-line tool so people can put it into their own workflows: ingest scripts, batch jobs, Shortcuts, render machines. It is a second front end on the same Rust core, so a file rendered from the terminal matches the app's export exactly: same decoder, same automatic exposure and white balance, same looks, same export code.

**Commands**

| Command | What it does |
|---|---|
| `render <files or folders>` | Develops and exports. Defaults match the app: automatic exposure and white balance, and the user's default look. Flags change them for that run only. |
| `info <file>` | Prints metadata and the automatic estimates (exposure, illuminant, white balance, noise level) without rendering. |
| `looks list`, `looks import <zip or folder>`, `looks verify` | The wizard's importer without the UI: same library folder, same recognition rules, same checksums. |
| `doctor` | Reports GPU, model weights, noise profiles, imported looks and version, for bug reports. |

Options for `render` mirror the app's few controls: `--look`, `--intensity`, `--exposure auto|<EV>`, `--wb auto|<K>[,<tint>]`, `--format jpeg|tiff|png|dng`, `--quality`, `--bits`, `--colour-space`, `--long-edge`, `--out`, `--name` (a filename template), `--jobs`, and `--skip-existing` or `--overwrite`.

**Setting****s**

- Nothing is saved per photo, in the app or the CLI: open, adjust, export, done. Each run starts from the automatic development and the default look, and flags such as `--look` or `--exposure` apply to that run only.
- Looks are referenced by name. A missing look stops that file with a message that points to `looks import`.

**Built for scripts**

- `--json` writes one JSON object per file on stdout: input, output path, look, automatic estimates, timings and errors. Progress and logs go to stderr.
- Exit codes: 0 when every file rendered, 1 when some failed, 2 for a usage error, 3 for a missing look, 4 when `--require-gpu` finds no GPU.
- `render -` reads file paths from stdin, so it composes with find, fd and xargs. `--jobs` sets parallel decodes; the GPU keeps a single queue.
- The same inputs and version give the same output. The version, look checksum and settings go into the exported file's XMP metadata.
- It runs headless through wgpu's Metal backend, with no window or surface, and falls back to the classical CPU path when no GPU is available.
- It is safe to run while the app is open: both use the same library folder, and imports take a lock.

```
yellowmoon render ~/Pictures/2026-10-04 --look neutral --format jpeg --long-edge 3000 --out ~/Exports
fd -e ARW . /Volumes/A1 | yellowmoon render - --json --format tiff --bits 16 > results.jsonl
yellowmoon looks import ~/Downloads/<ARRI look package>.zip
```

**Install and update**

- The binary sits inside the app bundle as a Tauri external binary, so it is signed, shipped and auto-updated together with the app.
- The Homebrew cask links it onto PATH. A .dmg install gets an "Install command-line tool" menu item that creates the link, as VS Code does for `code`.
- The command is `yellowmoon`.
- V2 options: a CLI-only Homebrew formula for machines without the app, and native Shortcuts actions through App Intents.

## Distribution and auto-update

We copy [plugout](https://github.com/ngriaznov/plugout)'s setup: every version bump on main becomes a signed GitHub release, and the app updates itself from it.

**Release pipeline** (GitHub Actions on macOS, triggered by a push to main)

1. Read the version from `src-tauri/Cargo.toml`. If tag `v<version>` already exists, stop.
2. Build the command-line tool as a universal binary, then a universal macOS bundle (aarch64 + x86_64) that carries it, with `createUpdaterArtifacts`, signed with `TAURI_SIGNING_PRIVATE_KEY`.
3. Write `latest.json`: version, publication date, and the `.app.tar.gz` URL and signature for both macOS architectures.
4. Publish release `v<version>` with the `.dmg`, the `.app.tar.gz` and `latest.json`, plus generated release notes.
5. Update the Homebrew cask in the repo (version and SHA-256) and push it.

**In the app**

- `tauri-plugin-updater` with the minisign public key embedded and one endpoint: `releases/latest/download/latest.json`. Restart goes through `tauri-plugin-process`.
- A quiet check shortly after launch. Failures stay silent, so an unreachable endpoint never bothers the user.
- An update pill in the toolbar, hidden while idle: "Update available" → downloading → "Restart to update" → "Update failed" with the reason on hover.
- An update never interrupts an export: "Restart to update" waits until the export queue is empty.

**Install paths**

- Homebrew tap in the same repo, with `auto_updates true` and a `zap` stanza that removes app data, caches and imported looks.
- The `.dmg` from Releases.
- The command-line tool comes with both: the cask links it onto PATH, and the app offers to install the link.

**Differences from plugout**

- The bundle carries the ML weights and noise profiles, a few MB. If they grow, weights move to a separate versioned download.
- Windows and Linux, if we add them, need a build matrix and per-platform entries in `latest.json`.
- plugout ships unsigned and documents `xattr -cr`. That's acceptable for betas; for 1.0 we should decide on Apple Developer ID signing and notarisation.

**Name and availability** (checked 11 October 2026; not legal advice)

- The US register has no YELLOWMOON or YELLOW MOON mark in classes 9 or 42, and we found no software product with the name.
- Nearest in our field: Yellow Moon Post Production Ltd, a film and TV post house in Northern Ireland since 1996 (yellowmoon.net). It sells services, not software, but may hold unregistered UK rights.
- Still to check by hand: EUIPO/TMview, INPI, WIPO and the UK IPO, in classes 9 and 42.
- Taken: yellowmoon.com, yellowmoon.io and the GitHub user name. Probably free: yellowmoon.app, .dev and .photo. Free: the crates.io, npm and Homebrew names.
- So the repository lives under an organisation, for example zerosize/yellowmoon, and the site at yellowmoon.app once a registrar confirms it.

## Milestones

Five milestones run in sequence, and each ends at a gate we can measure. Dates come after M0, once the spikes show the real effort.

[diagram: roadmap · 5 milestones, 5 gates]

The UI design (bold grid on white) feeds M3; M0 and M1 don't wait for it.

## Validation and testing

Each stage gets a measurable gate. The gates below are proposed starting targets, to tighten once we have real numbers. Colour is tested synthetically first; raw quality against real files and Capture One.

| Area | Test | Gate |
|---|---|---|
| ALEXA curve | Leave-one-camera-out on 52 cameras, reproduced in CI | Median curve error ≤ 3%; emulation ΔE00 median ≤ 0.3 |
| ALEXA curve | Compare with HdM SXT curve and digitised CIC 2015 Fig. 1, when available | Report only; no shipping dependency |
| Sensor map | 99 CIE CES samples under 9 lights, held-out light families | Per-light ΔE00 median ≤ 0.5, p95 ≤ 1.2 |
| Illuminant estimate | Synthetic scenes plus our A1 grey-card shots | Angular error median ≤ 2°; sensor-map ΔE stays within gate with real estimates |
| ARRI colour step | Our LogC3/LogC4 and matrices against colour-science and ARRI's published reference values | Bit-exact to 1e-5 |
| Display transform | Our ACES 2 port against the reference implementation | ≤ 1 code value (10-bit) |
| ARRI LUT path | Our .cube interpolation against Resolve output on test frames | ≤ 1 code value (10-bit) |
| Decoder | Every Sony ARW sample on RawDB and raw.pixls.us (both CC0) and our own files, decoded and compared with rawler and LibRaw | Identical raw values |
| Demosaic and denoise | Kodak, McMaster, Gharbi hard and moiré sets; our A1 pixel-shift pairs | PSNR above the classical fallback by ≥ 2 dB; no zipper or false colour in blind review |
| Noise | A1 ISO ladder 100–25600 | Visual parity with Capture One and DxO in blind A/B |
| Whole app | Blind A/B of 50 A1 photos against Capture One, Lightroom and DxO exports | Ours preferred or tied in ≥ 50% of pairs |
| Golden images | Fixed A1 files rendered through the CLI on every commit | No unexplained ΔE00 > 0.5 change |
| App and CLI | Same files and settings exported from both | Bit-identical output |
| Camera detection | Every raw.pixls.us Sony sample | Right body and profile for every file; fallback reported |
| Camera profiles | Own profile against stand-in, ΔE00 to the ALEXA rendering (sony_profiles.py) | Measured bodies ship on their own curves; stand-ins only for shared sensors |
| Noise estimate | Per-image estimate against the A1 calibration on textured, clipped and high-ISO shots; injected-noise test on other bodies | a and b within 10% median on the A1 |
| DNG export | dng_validate render against our JPEG of the same photo | ΔE00 median ≤ 1.5 |

Model parity tests run the same tiles through PyTorch, ONNX Runtime and burn on every model change.

## Licensing and legal checklist

The project is Apache-2.0; everything it ships must be compatible, and everything ARRI-owned stays on the user's machine.

| Input | Licence | Handling |
|---|---|---|
| ARRI 17-CCT ALEXA matrices (aces-dev v1.3.1) | AMPAS, notice retained | Ship, with NOTICE |
| Our ALEXA curve estimate | Ours, Apache-2.0; derived from AMPAS and Apache data | Ship, labelled as an estimate, with NOTICE |
| Weta Sony curves, 190 training patches (rawtoaces-data) | Apache-2.0 | Ship |
| LogC3/LogC4, AWG3/AWG4 formulas | Published specs, no licence text | Implement from spec |
| ACES 2 output transform | Apache-2.0 | Port as default display transform |
| ARRI Look Library and display LUTs | All rights reserved, no grant | User imports; never bundled or redistributed |
| HdM ALEXA SXT curve, CIC 2015 figure | No licence / © IS&T | Validation only, never shipped |
| PixelShift200 | CC BY 4.0 | Training, with attribution in the model card |
| CIE 99 CES, CIE illuminants | CC BY-SA 4.0 | Test only, fetched by script |
| RawTherapee, darktable, librtprocess, RapidRAW | GPL-3 / AGPL-3 | Read for ideas only; clean-room from papers |
| rawler, LibRaw | LGPL-2.1 / LGPL or CDDL | CI test oracles only |
| Tauri, wgpu, burn, ort, colour-science | MIT/Apache, BSD-3 | Dependencies |
| raw.pixls.us Sony sample files | CC0 | Decoder tests only, fetched by script |
| Solomatov and Akkaynak 2023 estimated curves (1,000+ cameras) | No licence in the repository | Evaluation only, never shipped |
| lightcraft ARW 6.0 decoder | MIT / Apache-2.0 | Reference or dependency, after the Sony IP question |
| SMPTE RDD 34 (the codec behind ARW 6.0) | IP enquiries go to Sony | Ask Sony before shipping an ARW 6.0 decoder |
| Adobe DNG specification | Royalty-free patent licence with a notice | Notice in source and documentation |
| ELD noise parameters (Sony A7S II) | Code MIT; data for evaluation only | Validation only |
| darktable noise profiles | GPL-3.0 | Reference only, never shipped |
| Foi noise-estimation code | Non-commercial only | Not used; reimplemented from the paper |
| RawDB Sony sample sets | CC0 | Decoder tests |

- [ ] Run `cargo-deny` and an npm licence check in CI on every release.
- [ ] Ship a model card per network: training data, licences, provenance.
- [ ] Email ARRI (arriraw-dev@arri.de) about in-app download of the Look Library.
- [ ] Ask HdM for a licence on the SXT curve, for validation and possible future use.
- [ ] Freedom-to-operate check on US 11948279 (joint demosaic and denoise with a lightweight U-Net) before 1.0.
- [ ] App name and UI without "ARRI" or "ALEXA"; only factual compatibility lines, such as "imports ARRI Look Library files".

## Risks and open decisions

The biggest risk is the ALEXA target itself; the open decisions are mostly settled by M0 and M1 experiments.

**Risks**

| Risk | Effect | Mitigation |
|---|---|---|
| ALEXA curve estimate is off by more than 5% | Emulated colour drifts, mostly in saturated reds and greens | Ensemble spread published; validate against HdM and CIC 2015 curves; swap in measured data if released |
| Far-red tail above 660 nm is unconstrained | Skin and red fabrics under tungsten may render wrong | Ensemble variants for the tail; targeted review of skin under tungsten |
| Estimate pulled toward the stills-camera prior | Real Sony-to-ALEXA differences understated | Prior variants without Sony bodies; compare with ARRI-rendered sample footage |
| Illuminant class hard to estimate from RGB alone | Wrong per-light matrix under LEDs and fluorescents | Blend per-light maps by estimated probability; manual light picker in UI |
| Metal surface under a transparent Tauri webview | Flicker, input routing or resize glitches | M0 spike; fallback is a separate native child view for the viewport |
| wgpu's Apple wide-gamut fixes were unreleased in Oct 2026 | Colour shifts in Display P3 output | Pin the wgpu version; test P3 output in M0 against exported files |
| GPU limits on 50 MP frames | Crashes or slow previews on smaller GPUs | Tiled f16 pipeline; classical fallback |
| burn API churn | Rework on upgrades | ONNX models as the stable artefact; ort as fallback runtime |
| ARRI changes download URLs or objects to automated access | Wizard breaks or must change | Import from file is the primary path; wizard opens ARRI's page rather than deep links |
| No GPU in some headless contexts, such as virtualised CI runners | CLI renders slowly or fails | Classical CPU fallback; --require-gpu fails fast; doctor reports the device |
| Some Sony variants decode wrongly (older bodies, cRAW quirks, M/S files) | Wrong colour or artefacts on those files | Every raw.pixls.us Sony sample checked against LibRaw in CI; a clear message for anything not yet verified |
| ARW 6.0 compressed codec: undocumented, hard-coded constants, possible Sony patents | A7 V and A7R VI compressed files don't open, or decode wrongly | Ship it last; check against LibRaw master and lightcraft; ask Sony about RDD 34; clear message meanwhile |
| M and S files aren't Bayer | Sensor emulation is approximate on them | Reduced path, flagged in the app |
| Noise estimate off on heavily textured or filtered files | Over- or under-smoothing | Texture fallback, filtered-file check, bias toward over-estimating, σ jitter in training |
| Other apps render our DNGs differently | The look's tone differs outside Lightroom | Look baked into pixels; full-size preview; behaviour documented per app |
| Yellow Moon Post Production (Northern Ireland) works in a neighbouring field | Naming dispute in the UK | Clearance search before launch |

**Open decisions**

| Decision | Options | Decides it |
|---|---|---|
| Code signing for 1.0 | Unsigned like plugout, or Apple Developer ID + notarisation | Your call; matters more for an app that opens user files |
| ML runtime | burn on wgpu, or ort with CoreML | M0 benchmark at our tile size |
| Illuminant estimator | FFCC, C5 or DMCC style | M1 synthetic plus A1 tests |
| Default display transform | ACES 2 as is, or our own tuned DRT | M1 visual comparison against ARRI's on test frames |
| Signature look | Its character and name | You design it in M3 |
| Tint control | Our own green-gain mapping | ARRI's is unpublished; decide in M1 |
| Trademark clearance | Check EUIPO, INPI, WIPO and the UK IPO by hand, or ask a trademark attorney | You, before announcing |
| Repository and domain | For example zerosize/yellowmoon and yellowmoon.app | You |

## Sources

- [ARRI ALEXA matrices in aces-dev v1.3.1](https://github.com/ampas/aces-dev/tree/v1.3.1/transforms/ctl/idt/vendorSupplied/arri/alexa) (AMPAS licence)
- [rawtoaces-data](https://github.com/AcademySoftwareFoundation/rawtoaces-data) (Weta Sony curves, Apache-2.0)
- [HdM Open Film Tools camera responsivities](https://www.hdm-stuttgart.de/open-film-tools/en/camera_responsivities)
- [ARRI Look Files and Look Library downloads](https://www.arri.com/en/learn/camera-systems/image-science/look-files)
- [ARRI LogC4 specification](https://www.arri.com/resource/blob/278790/bea879ac0d041a925bed27a096ab3ec2/2022-05-arri-logc4-specification-data.pdf)
- [ACES output transforms](https://docs.acescentral.com/system-components/output-transforms/)
- [plugout](https://github.com/ngriaznov/plugout) (release pipeline and updater pattern)
- [RapidRAW](https://github.com/CyberTimon/RapidRAW) (Tauri raw editor, reference only)
- [PixelShift200](https://huggingface.co/datasets/guochengqian/PixelShift200) (CC BY 4.0)
- Our research reports "Sony ARW to ARRI look research" and "ARRI sensor spectral data sources", and the ALEXA curve estimate bundle.
- [Solomatov and Akkaynak, Spectral sensitivity estimation without a camera (2023)](https://arxiv.org/abs/2304.11549) and [its code and estimates](https://github.com/COLOR-Lab-Eilat/Spectral-sensitivity-estimation), evaluation only
- [RawDB, Sony sample sets (CC0)](https://rawdb.dnglab.org/)
- [LibRaw source, including the ARW 6.0 decoder](https://github.com/LibRaw/LibRaw)
- [lightcraft ARW 6.0 decoder pull request](https://github.com/storytold/lightcraft/pull/625)
- [SMPTE RDD 34 (LLVC)](https://pub.smpte.org/latest/rdd34/rdd34-2015.pdf)
- [Sony A7 V help guide: RAW file types](https://helpguide.sony.net/ilc/2540/v1/en/contents/251h_raw_file_type.html)
- [ExifTool Sony tags](https://github.com/exiftool/exiftool/blob/master/lib/Image/ExifTool/Sony.pm)
- [Foi et al. 2008, Poissonian-Gaussian noise fitting](https://webpages.tuni.fi/foi/papers/Foi-PoissonianGaussianClippedRaw-2007-IEEE_TIP.pdf)
- [ELD: physics-based noise formation](https://arxiv.org/abs/2108.02158) and [its code](https://github.com/Vandermode/ELD)
- [Chen et al. 2015, noise level estimation](https://openaccess.thecvf.com/content_iccv_2015/papers/Chen_An_Efficient_Statistical_ICCV_2015_paper.pdf)
- [YOND, 2025](https://arxiv.org/abs/2506.03645)
- [Adobe DNG specification and patent licence](https://helpx.adobe.com/camera-raw/desktop/dng-and-file-formats/digital-negative.html)
- [DNG SDK source (AOSP mirror)](https://android.googlesource.com/platform/external/dng_sdk/)
- [US trademark search for "yellow moon" (Justia)](https://trademarks.justia.com/search?q=yellow+moon)
- [Yellow Moon Post Production (thinkbusiness.ie)](https://www.thinkbusiness.ie/articles/lights-camera-and-action-for-ni-tv-and-film-firm-yellow-moon/)

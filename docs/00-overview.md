# 00 · Overview

## What Yellowmoon is

A desktop raw processor for photos from Sony cameras. It reads ARW files, develops them automatically, and renders them with the colour of an ARRI ALEXA cinema camera (ALEV III sensor: ALEXA Classic, XT, SXT, Mini, LF and Mini LF). The user can pick a look, nudge a few controls, and export. Nothing is kept afterwards except the exported files.

- Platform: macOS 14 Sonoma or later (owner's decision of 2026-10-17, `docs/16`), universal binary (Apple silicon and Intel). The photo is drawn by a native GPU view (wgpu on Metal).
- Licence: Apache-2.0. A personal art project on Mykyta's GitHub account (`github.com/ngriaznov/yellowmoon`), set up like github.com/ngriaznov/plugout: GitHub Releases, a Homebrew cask in the same repo, and in-app auto-update.
- Two front ends on one Rust core: the Tauri app, and a command-line tool `ym` that gives bit-identical results.
- Name: Yellowmoon, one word. The command is `ym` (decision 2026-10-12). Brand colour #FFC20E, the yellow moon mark.

**Who it's for.** People who shoot Sony and want cinema-like colour without grading every photo: family, travel, everyday shooting. They expect it to "just work" and to look better than the camera JPEG and a generic LUT.

**Quality bar.** Raw quality (demosaic, noise, detail, highlights) on par with Capture One at the same ISO, judged in blind A/B tests. Colour that matches what an ALEXA would have recorded of the same scene, judged against the ARRI reference pipeline and synthetic tests.

**The point of the app (Mykyta, 2026-10-14).** Perfect ARRI-like colour on default settings, without fiddling with parameters, and it has to beat a camera raw processor followed by a static ARRI-like LUT (for example Resolve with a commercial emulation LUT). What a static LUT cannot do is what Yellowmoon must do well: follow each photo's light and content (today the per-photo light estimate driving the sensor map; later intelligent automatic settings). It is a photo app, not video: it may be brighter than a cinema LUT's placement of mid grey. How "beats a static LUT" is measured is being defined (STATUS, the dynamic-stage research).

## Principles

1. **Automatic first.** Every photo opens already developed: exposure, white balance, noise reduction, lens corrections and the default look. Controls are offsets from the automatic result.
2. **Few controls.** Six tabs (Look, Light, Colour, Detail, Curve, RAW) with a handful of sliders each. No masks, layers or retouching.
3. **Open, adjust, export, done.** No catalogue, no sidecars, no edit history. Closing the app forgets the session.
4. **Light-aware colour.** The scene's light is estimated per photo and drives how the Sony sensor's response is mapped to the ALEXA's. That is what a static LUT can't do.
5. **Honest about estimates.** Where we use an estimate (the ALEXA curves, a stand-in camera profile, a reduced decode path), the app says so in plain words.
6. **Clean licensing.** Everything shipped is Apache-compatible. ARRI's own looks and LUTs stay on the user's machine.

## Scope

**V1 relies on available measurements for Sony cameras.** Camera colour comes from published spectral measurements: Weta's curves for 11 Sony bodies (rawtoaces-data, Apache-2.0). Everything else (noise priors, test photos, validation) comes from public files, public datasets and independent tools. We take no measurements of our own in V1: no spectral measurements, no calibration frames, no test shoots. Bodies without published curves use a same-sensor sibling's profile or the A7R IV profile. Our own measurements are V2 work.

**V1 (this plan, milestones M0–M4)**

- Sony ARW from every Sony camera that writes ARW:
  - 12-bit uncompressed (ARW 1 and 2), 14-bit uncompressed, lossy cRAW, lossless L (tiled LJPEG-92).
  - Lossless M and S (camera-processed YCbCr) through a reduced path, flagged in the UI.
  - ARW 6.0 "Compressed RAW 2" (A7 V, A7R VI), gated on a question to Sony (`docs/04-raw-decoding.md`). Until it ships, those files get a clear message.
- Camera detection on every file. Eleven measured Sony profiles; same-sensor bodies share them; every other body uses the A7R IV profile.
- ALEXA ALEV III colour: illuminant-conditioned sensor map, ARRI's CCT matrices, LogC4/AWG4 working space.
- Learned joint demosaic and denoise, with a classical fallback. Per-image noise estimation for every body.
- Looks: Signature (default) and Neutral (the identity) built in; ARRI Look Library and display LUTs imported by the user through a wizard; any other `.cube` with one question.
- Export: JPEG, TIFF, PNG, and DNG with the look baked in.
- Command-line tool with the same features.
- Auto-update, Homebrew cask, `.dmg`.

**V2 (not in this plan)**

- Controlled captures on the A1: grey cards, flat fields, noise calibration, pixel shift (V1 validates on public data only).
- More measured camera profiles, starting with the A1.
- HDR display and HDR/gain-map export.
- Windows and Linux, if there's demand.
- Better far-red data for the ALEXA curve if HdM or ARRI publish measured curves.
- One-click download of ARRI looks if ARRI agrees.

**Out of scope**

- ALEXA 35 and REVEAL colour science (unpublished; a different sensor).
- Video, tethering, catalogue or library management, saved edits, sidecar files, local masks, retouching, panoramas, HDR merge.
- ARQ pixel-shift files, SRF and SR2 (pre-ARW formats).
- Non-Sony raw files.

## Decisions already settled

Don't reopen these without new evidence. The reasoning is in `research/`.

| Area | Decision |
|---|---|
| Platform | macOS only for V1, universal build. |
| Shell | Tauri 2 (React + TypeScript webview), `tauri-plugin-updater` and `-dialog`, as in plugout. The restart into an installed update is the app's own guarded `restart_app` command, not `tauri-plugin-process` (Tauri's restart cannot be vetoed while an export runs; `docs/12`, `docs/01` IPC). |
| Viewport | The photo renders in a native wgpu surface on Metal behind a transparent webview. Fallback if compositing fails in M0: a native child view. |
| Core | Rust workspace; image processing on the GPU (wgpu 30, WGSL compute) in f16 tiles of 1–2k px. CPU reference path for tests and machines without a usable GPU. |
| Decoder | Our own ARW decoder (Apache-2.0). rawler and LibRaw are CI oracles only. |
| Camera profiles | V1 uses only available measurements: 11 bodies measured by Weta (rawtoaces-data, Apache-2.0). Registry keyed by Sony model ID; same-sensor siblings share; A7R IV is the default for everything else. A Sony file is never refused for lack of a profile. No measurements of our own in V1. |
| ALEXA target | Our ALEV III spectral curve estimate (`data/alexa/`), shipped and labelled as an estimate. |
| Sensor map | White balance first, then a per-illuminant 3×3 from the illuminant estimate, white-preserving and exposure-invariant. No static LUT; no per-pixel spectral reconstruction. |
| ARRI colour step | ARRI's 17-CCT matrices (aces-dev v1.3.1, AMPAS licence) interpolated in mired, to AWG3, then exactly to AWG4. |
| Working space | Linear AWG4 and LogC4. LogC3 looks run in LogC3 (EI 800) through an exact conversion. |
| Display transform | ACES 2 output transform ported to WGSL as the default. ARRI's display LUTs only after the user imports them. |
| Looks | ASC CDL then 3D LUT in LogC4, intensity 0–100%. Built-in Signature (default; a luma curve and saturation, parametric, ours) and Neutral (the identity). |
| ML | Joint demosaic+denoise U-Net, 0.2–1 M parameters. Runtime: burn on the shared wgpu device or ort, decided by the M0 benchmark. Training in PyTorch; weights only from CC0/CC BY/Apache data (no own captures in V1). |
| Noise | Per-image Poisson–Gaussian estimate for every body, with priors fitted from public files. No calibration shoots in V1. Nothing stored. |
| Lens | Sony's embedded correction tags (vignetting, distortion, CA). No Lensfun. |
| Export | JPEG, TIFF, PNG; DNG as LinearRaw AWG4 with the look baked into pixels and a profile fitted to ACES 2. |
| CLI | `ym` binary in the same workspace, shipped inside the app bundle, bit-identical to app exports. |
| Persistence | Nothing per photo. Only imported looks and preferences (default look, display rendering choice, export defaults, watch-Downloads permission). |
| UI | Dark glass (owner's decision, 2026-10-16): glasscn components on Tailwind 4 and Radix, Figtree, yellow `#FFC20E` only as a button accent, a still cool aurora, dark only, photo edge to edge. The visual reference is `design/screenshots-c2/`; see `docs/11` and `docs/16`. The earlier "Bold grid, white" design (`design/mockups/`, `design/screenshots/`) is kept for history. |
| Distribution | plugout's release pipeline: version bump on main → signed updater artefacts → GitHub release → cask update. |

## Glossary

| Term | Meaning |
|---|---|
| ARW | Sony's raw format (TIFF-based). Versions 1.0–6.0. |
| cRAW | Sony's lossy "compressed RAW": blocks of 16 same-colour pixels, 11-bit max/min plus 7-bit deltas. |
| SR2 / SR2SubIFD | An encrypted sub-IFD in ARW holding black level, white level and white-balance gains. |
| ALEV III | The CMOS sensor of the ALEXA family from 2010 to 2022. Our emulation target. |
| AWG3, AWG4 | ARRI Wide Gamut 3 and 4: RGB primaries with a D65 white. AWG4 is the current one. |
| LogC3, LogC4 | ARRI's log encodings. LogC4 is a single curve (0.18 → 0.2784); LogC3 has one curve per EI (we use EI 800). |
| CCT, mired | Correlated colour temperature in kelvin; mired = 10⁶ / CCT. Matrices interpolate linearly in mired. |
| Illuminant estimate | Our per-photo estimate of the scene light: chromaticity plus class probabilities (daylight, tungsten, LED, fluorescent…). |
| Sensor map | The 3×3 that turns white-balanced Sony RGB into emulated ALEV III RGB for the estimated light. |
| DRT | Display rendering transform: scene-linear to display (tone mapping and gamut mapping). Default: ACES 2. |
| CDL | ASC Color Decision List: slope, offset, power, saturation. |
| Look / LUT | A creative colour transform. ARRI's looks are 3D `.cube` LUTs, log-to-log. |
| σ map | Per-pixel noise standard deviation, an input to the denoise network. |
| ΔE00 | CIEDE2000 colour difference. Below ~1 is hard to see side by side. |
| RFSE | Relative RMS error of a spectral curve, as a percentage of its peak. |
| Stand-in profile | A camera profile used for a body that has none of its own (A7R IV by default). |
| [HUMAN] | A task only Mykyta can do. |

## Where to find things

| Need | Look in |
|---|---|
| How the code is organised | `docs/01-architecture.md` |
| Colour maths and constants | `docs/02-colour-pipeline.md`, `data/` |
| Camera detection and profiles | `docs/03-camera-profiles.md` |
| ARW decoding | `docs/04-raw-decoding.md`, `research/notes/sony-arw-formats.md` |
| Demosaic, highlights, lens, sharpening | `docs/05-raw-processing.md` |
| Noise | `docs/06-noise.md`, `research/notes/noise-estimation.md` |
| Models and training | `docs/07-ml-models.md` |
| Looks and the ARRI import wizard | `docs/08-looks.md` |
| Export and DNG | `docs/09-export.md`, `research/notes/dng-export.md` |
| Command line | `docs/10-cli.md` |
| UI | `docs/11-ui.md`, `design/` |
| Release and updates | `docs/12-distribution.md`, `templates/` |
| Tests and gates | `docs/13-testing.md`, `docs/test-photo-sources.md` |
| Licences | `docs/14-licensing.md`, `NOTICE` |
| The GPU path | `docs/17-gpu.md` |
| Task plan | `docs/15-milestones.md` |
| Risks and open decisions | `docs/16-risks-and-decisions.md` |
| Progress | `docs/STATUS.md` |

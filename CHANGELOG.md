# Changelog

All notable changes to Yellowmoon are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) (0.x during M0 to M3, 1.0.0 at the end of M4).

A version bump on `main` publishes a release (`docs/12-distribution.md`), so every "Release x.y.z" pull request
moves the entries below from Unreleased into a dated section for that version.

## [Unreleased]

Nothing has been released yet. These entries cover what is in the repository so far: the M0 skeleton, the `ym` command-line tool and the raw-algorithms wave (`docs/algorithms/`) with its owner decisions (`docs/16`).

### Changed

- The Signature look is the default (the owner's decisions of 2026-10-14; pipeline version 0.3.0, was 0.2.0, golden images re-baselined). Every photo opens with more contrast and more colour than before: a soft curve on the brightness of the scene (slope 1.25 through the middle, a soft toe and shoulder) and a colour boost of 1.20, with middle grey exactly where it was. Neutral stays in the list as the look that changes nothing. The name may change later. An ARRI look you imported still replaces it, as before.
- Cameras without their own colour data now read white balance correctly. The A1 (and any body that borrows another camera's colour profile) used to read a daylight photo as about 4400 K, which made it too warm; it now reads about 5200 K, and its Daylight preset sits at its nominal temperature. Each such body carries a small correction to its red and blue white, taken from the white-balance presets the camera writes into its own files. Cameras with their own data are unchanged.
- Highlights reach a real white. Each photo gets a small automatic lift of its brightest tones (at most about one stop on the 12 test photos, nothing for a photo whose highlights are not bright; middle grey and everything below one stop over it stay put), sized from the photo's own highlights. The Automatic card shows it as a White row, the Light tab has a White offset slider (0 is the automatic amount; bring the lift to nothing by moving it down), and `ym render --white-offset EV` does the same. The DNG carries the lift. An imported ARRI display LUT and the LogC3-to-Rec.709 looks keep ARRI's own highlights.
- Exposure, brightness and the DNG (the owner's decisions of 2026-10-14; pipeline version 0.2.0, was 0.1.2, golden images re-baselined). Auto uses the camera's own exposure and your exposure compensation in full: a fixed +1.11 EV, the median gain of the camera JPEG engines on twelve Sony photos, with a highlight guard that can only lower it, and the compensation you dialled on the camera reaches the picture as it does in the camera's JPEG. The brightness of the photo no longer moves Auto, so a shot taken too dark stays dark until you move the Exposure slider; this is an interim rule until the intelligent automatic settings arrive. Pictures are 0.85 EV brighter on screen, about as bright as a Sony JPEG, with the ARRI colours unchanged (the lift comes after the look, so a look still sees middle grey where it expects it). ARRI's own display renderings, an imported ARRI display LUT and the LogC3-to-Rec.709 looks, keep ARRI's brightness, so the Intensity slider of such a look also changes brightness. The DNG opens in Adobe's apps as bright as the app and carries every change except sharpening (no capture or output sharpening; the demosaic's shadow step stays). It now works with the LogC3-to-Rec.709 looks and an imported display LUT, approximately: the DNG is close to, not the same as, the picture in the app.
- The classical demosaic (R6) is now the adaptive ratio demosaic (`docs/05`, RA-05; the owner's decision of 2026-10-13 after a blind look). In linear light its mean PSNR over nine cells is 36.05 dB against 34.43 for Menon, the previous default, and 20 % of edge pixels show zipper artefacts against 39 %. The demosaic stage takes 1.5 to 1.7 times Menon's CPU time, which adds 3 to 13 % to an export. Menon, LMMSE, bilinear and the log-guided demosaic stay available to tests through a switch; the method is not a setting. A photo whose noise model is missing or unusable renders as if it had no noise, instead of failing. Every default render changes, so the pipeline version is 0.1.2 (was 0.1.1) and the golden images are re-baselined.
- Lateral chromatic aberration is corrected on the mosaic before the demosaic (R5b), not in the lens resample after it. On the real files measured, with two independent estimators, it leaves less red/blue misregistration than correcting after the demosaic (`docs/05`, R9, has the numbers). The binned fit view and the thumbnails still correct it in R9 (`docs/05`).
- Saturation is subtractive: more saturation makes colours denser and darker, less makes them paler, and neutrals and the brightest channel stay (`docs/02`, C9). Renders at 0 are unchanged.
- The command-line tool is called `ym` everywhere (decision 2026-10-12): the binary, `--version`
  (`ym <version> (pipeline <v>, <sha>)`), the help text, the sidecar (`Contents/MacOS/ym`, so it cannot
  collide with the app's `Yellowmoon` binary), the Homebrew cask's `binary` stanza, the issue forms
  (`ym doctor`), the README and the docs. The sidecar files are `src-tauri/binaries/ym-<triple>`
  (`tools/dev/prepare-sidecar.sh`, `release.yml`).

### Added

- Output sharpening: the develop setting `output_sharpening` (0 to 200 %, default 100), the Output slider in Detail › Sharpening and `ym render --output-sharpening`. A linear-light luma unsharp mask with halo control, applied to every picture at the size it is shown or written, in JPEG, TIFF and PNG exports after the resize, never in a DNG; 0 switches it off. It makes a quality 92 JPEG 0.4 to 4.3 % larger (three real files, full size and 2048 px; RA-08).
- The Tone card on the Look tab: Temperature, Contrast and Saturation, the same settings as on the Colour and Light tabs.
- Diffraction deconvolution for capture sharpening (R8), an Airy-pattern PSF from the f-number and the pixel pitch, with sensor widths in the camera registry for 27 of 28 bodies. It is not the default and has no setting; Richardson–Lucy stays until the owner decides (RA-07).
- A log-guided fast demosaic, internal and not used by any render (RA-06).
- WGSL kernels with parity tests on software Vulkan for the adaptive ratio demosaic, the log-guided demosaic, diffraction deconvolution and output sharpening. None is wired into a render yet (RA-09).
- `docs/algorithms/`: the four algorithm specifications of the raw-algorithms package, the integration guide, the task list and the test vectors (RA-01).
- `ym render` with every option of `docs/10-cli.md`: files, folders (`--recursive`) and `-` for paths on
  stdin; look, intensity, exposure (`--exposure`, `--exposure-offset`), white balance and light presets, the
  grade, noise reduction, sharpening, the RAW switches, roll-off, a forced profile and the display rendering;
  JPEG, TIFF, PNG and DNG with their options, `--out`, `--name`, `--skip-existing`/`--overwrite`; `--settings`
  (a file or stdin, one document or one per file name); `--json` (one object per file, as each finishes);
  `--require-gpu` (exit 4); a progress line on a terminal; Ctrl-C stops after the current tile, deletes the
  partial file and exits 130. Exit codes 0 to 5 as documented (M1-14, M3-14).
- `ym info` shows the camera profile and why it was chosen and the automatic estimates (exposure, white
  balance, light, noise model, sharpening, noise reduction), with `--json` in the shape of `render --json`.
- `ym looks list`, `import`, `verify`, `default` and `bake` (a 33-point `.cube` and a contact sheet), and
  `ym doctor` (GPU, render path, model weights with hashes, profiles, looks, folders; `--json` too).
- App and CLI parity test: the same job through the app's export path and through the binary is
  byte-identical for JPEG, TIFF (8 and 16 bit, none, LZW, ZIP), PNG (8 and 16 bit) and DNG, with options and
  settings (M3-14).
- Cargo workspace with the `ym-*` library crates, the `ym-cli` binary (`ym`), the `src-tauri` app crate
  and the test-only tools, with empty documented APIs and an explicit member list (M0-01).
- React, TypeScript and Vite frontend skeleton with Vitest and Playwright, the design tokens and a placeholder
  Start screen (M0-01).
- Continuous integration (`.github/workflows/ci.yml`): frontend tests and build with an npm licence check,
  Playwright end-to-end tests, Rust format, clippy and tests, `cargo deny`, a self-test that proves `deny.toml`
  rejects GPL, LGPL and banned crates, and checks for the fetch scripts.
  Later jobs (GPU on software Vulkan, decoder oracle, golden images, DNG validation, training dry run) are
  present as comments and are switched on as their tasks land (M0-02).
- Licence policy in `deny.toml`: Apache-2.0-compatible licences only; GPL, LGPL and AGPL crates and the crates
  `dng`, `rawler`, `rawloader`, `ljpeg` and `jpegxl-rs` are rejected in shipped code (M0-02).
- Issue forms for bug reports (asks for `ym doctor` output, macOS version, camera and steps) and feature
  requests, and a pull request template (M0-02).
- Release pipeline (`.github/workflows/release.yml`): a version bump on `main` builds the universal macOS
  bundle with a universal command-line sidecar, signs the updater artefacts, publishes the release with
  `latest.json` and updates the Homebrew cask in `Casks/yellowmoon.rb` (M0-03).
- `docs/release-signing.md`: the steps to create the updater signing key and set the repository secrets (M0-03).
- Test data fetching (`tools/fetch/`): `samples.sh` for CC0 Sony ARW files from RawDB and raw.pixls.us and
  `golden.sh` for the golden photo set, both checking SHA-256 against `tools/fetch/manifest.json` (M0-09).
- `docs/validation-data.md`: the public dataset or independent tool behind every gate in `docs/13`, so no
  gate needs a test shoot (M0-10).

[Unreleased]: https://github.com/ngriaznov/yellowmoon/commits/main

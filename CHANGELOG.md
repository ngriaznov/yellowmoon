# Changelog

All notable changes to Yellowmoon are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) (0.x during M0 to M3, 1.0.0 at the end of M4).

A version bump on `main` publishes a release (`docs/12-distribution.md`), so every "Release x.y.z" pull request
moves the entries below from Unreleased into a dated section for that version.

## [Unreleased]

The M0 skeleton (`docs/15-milestones.md`, milestone M0). Nothing here is a user-facing feature yet.

### Changed

- The command-line tool is called `ym` everywhere (decision 2026-10-12): the binary, `--version`
  (`ym <version> (pipeline <v>, <sha>)`), the help text, the sidecar (`Contents/MacOS/ym`, so it cannot
  collide with the app's `Yellowmoon` binary), the Homebrew cask's `binary` stanza, the issue forms
  (`ym doctor`), the README and the docs. The sidecar files are `src-tauri/binaries/ym-<triple>`
  (`tools/dev/prepare-sidecar.sh`, `release.yml`).

### Added

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

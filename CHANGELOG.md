# Changelog

All notable changes to Yellowmoon are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) (0.x during M0 to M3, 1.0.0 at the end of M4).

A version bump on `main` publishes a release (`docs/12-distribution.md`), so every "Release x.y.z" pull request
moves the entries below from Unreleased into a dated section for that version.

## [Unreleased]

The M0 skeleton (`docs/15-milestones.md`, milestone M0). Nothing here is a user-facing feature yet.

### Added

- Cargo workspace with the `ym-*` library crates, the `ym-cli` binary (`yellowmoon`), the `src-tauri` app crate
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
- Issue forms for bug reports (asks for `yellowmoon doctor` output, macOS version, camera and steps) and feature
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

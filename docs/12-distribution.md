# 12 · Distribution and auto-update

How a version of Yellowmoon is numbered, built, signed, published and installed, and how the app updates itself. Every version bump on `main` becomes a signed GitHub release; the app and the Homebrew cask read from it. The setup follows the maintainer's plugout project. Templates: `templates/github/workflows/`, `templates/Casks/yellowmoon.rb`, `templates/tauri.conf.json`.

## Versioning

- One version for the app, the CLI and the workspace: `[workspace.package] version` in the root `Cargo.toml`. `package.json` and `package-lock.json` mirror it (`src/repo.node.test.ts` checks that they match).
- Semver. The app is in the 0.1.x series while it is tested: each release is the next patch version. **0.2.0 is the node view and masks** (`docs/23-nodes.md`, milestone N): built on the branch `nodes` over several test builds that are not releases, then released once. **0.2.1 is the Film node** (`docs/24-film.md`, milestone F): built on the branch `film` over several test builds that are not releases, then released once. 1.0.0 is the end of M4.
- `tools/release/bump-version.sh patch` (or an explicit `x.y.z`) sets the version everywhere it lives: the root `Cargo.toml` including the workspace's own crates in `[workspace.dependencies]`, `package.json`, `package-lock.json`, and every `Cargo.lock` (the workspace's and those of the folders outside it that depend on its crates by path). It edits files only. A `## [x.y.z] - date` section in `CHANGELOG.md` then takes the Unreleased entries. Pushing the bump to `main` releases it.
- `pipeline_version` (`docs/01-architecture.md`) is separate and changes only when output pixels change.

## Release pipeline (`release.yml`)

GitHub Actions on a macOS runner (the `xcode-27` image, whose SDK the Apple Auto Swift bridge needs, `docs/19`), on every push to `main`. `workflow_dispatch` retries a release whose run failed. Runs queue (`concurrency: release`) and never cancel each other.

1. Read the version from the root `Cargo.toml`. If tag `v<version>` exists, stop: the push was not a release.
2. Stop with an explicit error if the secret `TAURI_SIGNING_PRIVATE_KEY` is missing or `tauri.conf.json` still has the placeholder public key. Every release carries everything the in-app updater needs, and a version built without the public key could never update itself afterwards. `TAURI_SIGNING_PRIVATE_KEY_PASSWORD` is optional (the current key has no password).
3. **Fetch the SAM 2.1 small weights** (from 0.2.0) from our own release asset into `data/models/sam2/` and check their SHA-256 against `data/models/sam2/SHA256SUMS` (`tools/fetch/sam2.sh`; the file is cached between runs and checked again after the cache); a release without them stops here, as it does while `SHA256SUMS` still holds the placeholder `TBD-after-first-run`. Then generate `THIRD-PARTY.md` (`tools/licences/generate.py`) and check the bundle (`tools/licences/check_bundle.py --require-weights`); see Bundle contents.
4. Build `ym-cli` for `aarch64-apple-darwin` and `x86_64-apple-darwin` and place three sidecars in `src-tauri/binaries/`: `ym-aarch64-apple-darwin`, `ym-x86_64-apple-darwin` (each per-architecture cargo build of `src-tauri` checks for its own triple) and the `lipo`-merged `ym-universal-apple-darwin`.
5. `npm run tauri build -- --target universal-apple-darwin`, signed ad hoc (`signingIdentity "-"`, see Signing), with `createUpdaterArtifacts` and the minisign key from the secrets: the `.app.tar.gz` and its `.sig`. Tauri's disk-image step (`bundle_dmg.sh`) sometimes fails on hosted runners, so the step makes up to three tries, detaching leftover images between them.
6. Check the bundle: `codesign --verify --deep --strict`, both architectures in the app binary, `ym --version` starts with `ym <version> `, the disk image is `Yellowmoon_<version>_universal.dmg`, and the tarball and its signature exist and are not empty.
7. Write `latest.json` with `jq`: `version`, `pub_date`, and for `darwin-aarch64` and `darwin-x86_64` the same universal tarball URL and signature. Check that both platforms carry a signature and a URL that ends in `.app.tar.gz`.
8. Publish release `v<version>` ("Yellowmoon <version>", `softprops/action-gh-release@v2`) with the `.dmg`, the `.app.tar.gz`, its `.sig` and `latest.json`; a missing file fails the step. The notes are the version's section of `CHANGELOG.md` plus how to install and allow the app to open.
9. Update `Casks/yellowmoon.rb` (version and the dmg's SHA-256) and push it to `main`. That commit exists on `main` only: a branch that goes on must merge `main` before its next fast-forward. If `main` is protected, `github-actions[bot]` must be allowed to push to it.

Build environment: `MACOSX_DEPLOYMENT_TARGET` is `14.0` (the app's `minimumSystemVersion`) for every binary, including the cargo-built sidecar. Binaries are not stripped (`CARGO_PROFILE_RELEASE_STRIP=none`) because the runner's dyld refuses the stripped proc-macro dylibs.

**Sidecar name.** The app's main binary is `Yellowmoon`, and macOS file systems are usually case-insensitive, so a sidecar called `yellowmoon` would collide with it inside `Contents/MacOS`. The command is `ym`, so the sidecar is `Contents/MacOS/ym`; the cask's `binary` stanza and the "Install Command-Line Tool…" menu link it as `ym`.

## Test builds (`build.yml`)

By hand only (Actions > Build > Run workflow; universal, or Apple silicon only for about half the build time). It builds the app from any branch or commit on a macOS runner and keeps the disk image as the run's artefact `Yellowmoon-<commit>-<architecture>` for 14 days, to try something before it is released. It is signed ad hoc and has no updater artefacts. Downloads for people are GitHub Releases.

`update-check.yml` (by hand only) installs a published release on a Mac runner, launches it and captures the screen, to see what its updater does against the latest release.

## In the app

- `tauri-plugin-updater` with the minisign public key embedded and one endpoint: `https://github.com/ngriaznov/yellowmoon/releases/latest/download/latest.json`.
- The restart into an installed update is the app's own `restart_app` command (`docs/01-architecture.md`, IPC), not `tauri-plugin-process`. Tauri's own restart cannot be vetoed by the exit handler, so an export in progress could not stop it, while `restart_app` refuses with "An export is still running. Restart after it finishes." The process plugin is not a dependency (neither the crate nor `@tauri-apps/plugin-process`) and its `restart` permission is not granted.
- A quiet check runs 10 s after launch and then every hour while the app runs, if "Check for updates automatically" is on. An update it finds is downloaded and installed at once, in the background. The header then says "Update ready"; a click restarts into it, and the next launch runs it anyway.
- Failures stay silent: an unreachable endpoint never bothers the user, and a background download that fails leaves the update offered as "Update available" (a click tries again).
- "Check for Updates…" in the app menu checks now and reports the result. The update it finds downloads when "Update available" is pressed.
- The update cell in the header: hidden while idle → "Update available" (click downloads) → "Downloading n%" → "Update ready" (click restarts) → "Update failed" (reason on hover). The state machine is `src/updater.ts`.
- An update never interrupts an export: "Update ready" stays disabled ("Restarts after the export finishes") until the queue is empty, and the shell refuses a restart that gets through.

## Install paths

- **Homebrew:** a tap in the same repository.
  ```
  brew tap ngriaznov/yellowmoon https://github.com/ngriaznov/yellowmoon
  brew install --cask yellowmoon
  ```
  Recent Homebrew may ask for `brew trust ngriaznov/yellowmoon` once. The cask (`Casks/yellowmoon.rb`) has `auto_updates true`, `depends_on macos: ">= :sonoma"`, a `binary` stanza for the CLI, a `zap` stanza for app data, caches and imported looks, and a caveat that says how to allow the app to open.
- **Disk image** from GitHub Releases.
- **CLI:** linked by the cask, or by the app's "Install Command-Line Tool…" menu item (`src-tauri/src/cli_link.rs`: `/usr/local/bin/ym`, or `~/.local/bin/ym` when that needs rights the person does not give; an existing `ym` that is not a link to a working file is never replaced).

## Bundle contents

The app, the sidecar CLI, `data/` (the camera-curve data in `data/alexa`, the colour matrices with their notice in `data/arri`, camera profiles in `data/cameras`, the lens data in `data/lenses`, built-in looks in `data/looks/builtin`, `data/looks/arri-packages.json`, model weights in `data/models`: a few MB in total, and from 0.2.0 the 77 MB of SAM 2.1 small, and from milestone D about 50 MB of Depth Anything V2-Small), the Figtree fonts, `NOTICE`, `LICENSE` and `THIRD-PARTY.md`.

`THIRD-PARTY.md` holds the licence texts of the Rust crates (the app's and the `ym` command's, for macOS), the npm packages and the Figtree OFL text. It is generated by `tools/licences/generate.py` (cargo-about and license-checker; `release.yml` runs it before `tauri build`, and the `licences` CI job runs it on every change) and listed in `bundle.resources` next to `NOTICE` and `LICENSE`. `tools/licences/check_bundle.py` fails when the resource is missing. During development `tools/dev/prepare-sidecar.sh` writes a stub of the file (tauri-build wants every resource to exist) that the generator never accepts as the real one.

The weights stay in the bundle. If they ever grow beyond about 30 MB, they move to a versioned release asset downloaded on first run with a hash check, with a classical-only fallback (`AGENTS.md`, rule 7).

**The exception, SAM 2.1 small (owner, 2026-11-01).** The Object mask's weights are 77.1 MB at f16 (image path only, 38.5 M parameters; `docs/07`, `docs/23` §4.4), far over that line, and the owner chose to **bundle** them: the app grows by about 80 MB (the disk image and the update tarball too), stays offline and needs no first-run download. They are too big for git, so the **build fetches them**: they are converted once from Meta's checkpoint by `training/sam2/export.py` in CI (`.github/workflows/models.yml`, which runs on a push to a branch `models/**` or by hand), published as assets of the Yellowmoon GitHub **prerelease `models-sam2.1-small-1`** (created with `--prerelease --latest=false`, never overwritten; it must never be the "latest" release, because the in-app updater reads `releases/latest/download/latest.json`) pinned by the SHA-256 in `data/models/sam2/SHA256SUMS`, and `tools/fetch/sam2.sh` downloads them before `tauri build` (build tooling may download; the app and `ym` never do, hard rule 7). The same release carries the oracle fixtures of N-17's gate (`sam2-oracle-fixtures-1.tar.zst`, about 0.6 GB, fetched only by tests) and a `SHA256SUMS` asset. `tauri.conf.json` already bundles `data/models` whole, so the weights land in `Contents/Resources/data/models/sam2/`, where `ym` finds them like the other models. The committed part of the folder is `SHA256SUMS`, `MODEL_CARD.md` and a `.gitignore`; the Apache-2.0 notice is in `NOTICE` and the model card. The `.safetensors` file is git-ignored; `tools/licences/check_bundle.py --require-weights` (release.yml, build.yml) fails when the file is missing or its hash differs from `SHA256SUMS`, and the same two workflows check the weights inside the built `.app` as well. Updates through the in-app updater carry them (the tarball is the whole app), so 0.2.0's first update is large. No other weights move.

**The second exception, Depth Anything V2-Small (owner, 2026-11-07, milestone D).** The Depth mask's weights are about 50 MB at f16 (24.8 M parameters, 49.6 MB of tensors plus a header; `docs/07`, `docs/23` §4.8; the exact size is measured by D-02), over the same line, and the owner chose to **bundle** them as SAM's were: the app grows by **about 50 MB** (the disk image and the update tarball too, so the first update that carries it is large), stays offline and needs no first-run download. They follow SAM's path exactly. They are converted once from the authors' checkpoint by `training/depth/export.py` in CI (a `depth` job of `.github/workflows/models.yml`), published as assets of the Yellowmoon GitHub **prerelease `models-depth-v2-small-1`** (proposed name; created with `--prerelease --latest=false`, never overwritten, never the "latest" release that the updater reads), pinned by the SHA-256 in `data/models/depth/SHA256SUMS`, and `tools/fetch/depth.sh` downloads them before `tauri build` (build tooling may download; the app and `ym` never do, hard rule 7). The same release carries the oracle fixtures (`depth-oracle-fixtures-1.tar.zst`, fetched only by tests). `tauri.conf.json` already bundles `data/models` whole, so the file lands in `Contents/Resources/data/models/depth/`, where `ym` finds it. The committed part of the folder is `SHA256SUMS`, `MODEL_CARD.md` and a `.gitignore`; the Apache-2.0 notice is `NOTICE` item 13 and the model card. `tools/licences/check_bundle.py --require-weights` gains the depth weights (D-02), so that `release.yml` and `build.yml` fail when the file is missing or its hash differs, and check the `.app` as well. Together the two models grow the bundle by about 130 MB over 0.1's (about 80 and about 50).

## Signing

**Updater artefacts** are always signed with a Tauri minisign key. The private key is a repository secret (`TAURI_SIGNING_PRIVATE_KEY`, plus `TAURI_SIGNING_PRIVATE_KEY_PASSWORD` for a key with a password); the public key is in `src-tauri/tauri.conf.json` under `plugins.updater.pubkey`. Neither the private key nor a password appears in the repository, logs, pull-request text or `docs/STATUS.md`. Losing the key means installed copies can never auto-update again; leaking it lets anyone ship updates. `docs/release-signing.md` describes how the key is created, stored and replaced.

**Apple code signing and notarisation are not used.** The app ships without a Developer ID. The bundle is signed ad hoc only (Apple silicon needs a signature to run at all), so macOS asks once before it first opens a downloaded copy. The README section "Allow Yellowmoon to open" says how: System Settings › Privacy & Security › Open Anyway on macOS 15 and later, Control-click › Open before that, or `xattr -dr com.apple.quarantine`. The cask prints the same as a caveat. Updates the app installs itself are not quarantined and open without asking. The `APPLE_*` variables in `release.yml` stay commented.

## Repository and name

- Repository: `github.com/ngriaznov/yellowmoon`, under the maintainer's personal account.
- Before announcing, trademark clearance in the EU, UK, France and WIPO in classes 9 and 42 is needed [HUMAN M4-H3]. Details: `research/notes/name-and-trademark.md`.

## What the release adds to a plain Tauri project

| Topic | Yellowmoon |
|---|---|
| Workspace | Cargo workspace at the root; `src-tauri` is one member |
| Version source | `[workspace.package] version` in the root `Cargo.toml` |
| Sidecar | `ym`, a universal binary |
| Bundle data | `data/` resources |
| Window | transparent webview over a native Metal layer (`macOSPrivateApi`) |
| CI | fmt, clippy, test, vitest, playwright, plus cargo-deny, npm licence check, decoder oracle, golden images, `dng_validate`, training dry run |

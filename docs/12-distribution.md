# 12 · Distribution and auto-update

We copy github.com/ngriaznov/plugout's setup: every version bump on main becomes a signed GitHub release, and the app updates itself from it. Templates: `templates/github/workflows/`, `templates/Casks/yellowmoon.rb`, `templates/tauri.conf.json`.

## Versioning

- One version for the app, the CLI and the workspace: `[workspace.package] version` in the root `Cargo.toml`; `package.json` mirrors it (a test checks they match).
- Semver. 0.x during M0–M3 betas; 1.0.0 at the end of M4. **While the app is tested, every build that goes to Mykyta is the next 0.1.x** (Mykyta, 2026-10-15; 0.1.0 was the first): `tools/release/bump-version.sh patch` sets the version everywhere it lives (root `Cargo.toml` with the workspace's own crates, `package.json`, `package-lock.json`, every `Cargo.lock`), then a `## [0.1.x] - date` section in `CHANGELOG.md` moves the Unreleased entries. Pushing the bump to `main` releases it.
- `pipeline_version` (`docs/01`) is separate and changes only when output pixels change.

## Release pipeline (`release.yml`, GitHub Actions on macOS, on push to main)

1. Read the version from the root `Cargo.toml` (`[workspace.package] version`). If tag `v<version>` exists, stop: the push was not a release (`workflow_dispatch` retries a release whose run failed).
2. Stop with an explicit error if either signing secret is missing or `tauri.conf.json` still has the placeholder public key: every release carries everything the in-app updater needs, and a version built without the public key could never update itself afterwards.
3. Build `ym-cli` for aarch64 and x86_64 and place three sidecars: `ym-aarch64-apple-darwin`, `ym-x86_64-apple-darwin` (each per-arch cargo build of `src-tauri` checks for its own) and the `lipo`-merged `ym-universal-apple-darwin`.
4. `npm run tauri build -- --target universal-apple-darwin`, signed ad hoc (`signingIdentity "-"`: no Developer ID, `docs/16`), with `createUpdaterArtifacts` and the minisign key from the secrets: the `.app.tar.gz` and its `.sig`. Check the bundle: `codesign --verify`, both architectures, `ym --version` says the version, the disk image is `Yellowmoon_<version>_universal.dmg`, the tarball and its signature are there and not empty.
5. Write `latest.json` with `jq` (version, publication date, URL and signature for `darwin-aarch64` and `darwin-x86_64`, the same universal tarball) and check that both platforms carry a signature.
6. Publish release `v<version>` ("Yellowmoon <version>") with the `.dmg`, the `.app.tar.gz`, its `.sig` and `latest.json` (a missing file fails the step); the notes are the version's section of `CHANGELOG.md` plus how to install and allow the app to open (`softprops/action-gh-release@v2`).
7. Update `Casks/yellowmoon.rb` (version, SHA-256 of the dmg) and push to main. That commit is on `main` only: a branch that goes on must merge `main` before its next fast-forward.

**Bundle naming detail:** the app's main binary is `Yellowmoon` and macOS file systems are usually case-insensitive, so a sidecar called `yellowmoon` would collide with it inside `Contents/MacOS`. The command is `ym` (decision 2026-10-12), so the sidecar is `Contents/MacOS/ym` and there is nothing to rename: the cask's `binary` stanza and the "Install Command-Line Tool…" menu link it as `ym`.

## Test builds (`build.yml`)

By hand only (Actions > Build > Run workflow, universal or Apple silicon only): builds the app from any branch or commit on a macOS runner and keeps the disk image as the run's artefact `Yellowmoon-<commit>-<architecture>` for 14 days, to try something before it is released. Downloads for people are GitHub Releases. Signed ad hoc and without updater artefacts, like a release before the updater key exists.

## In the app

- `tauri-plugin-updater` with the minisign public key embedded and one endpoint: `https://github.com/ngriaznov/yellowmoon/releases/latest/download/latest.json`. The restart into the installed update is the app's own `restart_app` command (`docs/01` IPC), not `tauri-plugin-process`: Tauri's restart cannot be vetoed by the exit handler, so an export in progress could not stop it, while `restart_app` refuses with "An export is still running. Restart after it finishes." The process plugin is not a dependency (neither the crate nor `@tauri-apps/plugin-process`) and its `restart` permission is not granted.
- A quiet check 10 s after launch and then every 6 hours while running (if "Check for updates automatically" is on). Failures stay silent; an unreachable endpoint never bothers the user. "Check for Updates…" in the app menu checks now and reports the result.
- The update cell in the header: hidden while idle → "Update available" (click downloads) → "Downloading n%" → "Update ready" (click restarts) → "Update failed" (reason on hover). `src/updater.ts` comes from plugout unchanged except for naming and the restart (`restart_app`, above).
- An update never interrupts an export: "Update ready" stays disabled until the queue is empty.

## Install paths

- **Homebrew:** a tap in the same repository, as plugout does:
  ```
  brew tap ngriaznov/yellowmoon https://github.com/ngriaznov/yellowmoon
  brew install --cask yellowmoon
  ```
  (Recent Homebrew may ask for `brew trust ngriaznov/yellowmoon` once.) The cask has `auto_updates true`, a `binary` stanza for the CLI, and a `zap` stanza for app data, caches and imported looks.
- **DMG** from GitHub Releases.
- **CLI:** linked by the cask, or by the app's "Install Command-Line Tool…" menu item.

## Bundle contents

App, sidecar CLI, `data/` (ALEXA curve, ARRI matrices with notice, camera profiles, built-in looks, model weights: a few MB in total), Figtree fonts, `NOTICE`, `LICENSE` and `THIRD-PARTY.md`. `THIRD-PARTY.md` holds the licence texts of the Rust crates (the app's and the `ym` command's, for macOS), the npm packages and the Figtree OFL text; it is generated at release time by `tools/licences/generate.py` (cargo-about and license-checker; `release.yml` runs it before `tauri build`, the `licences` CI job on every change) and listed in `bundle.resources` next to `NOTICE` and `LICENSE`. `tools/licences/check_bundle.py` fails when the resource is missing. During development `tools/dev/prepare-sidecar.sh` writes a stub of the file (tauri-build wants every resource to exist) that the generator never accepts as the real one. If weights grow beyond ~30 MB, move them to a versioned release asset downloaded on first run with a hash check, and keep a classical-only fallback.

## Signing

- Updater artefacts are always signed with the Tauri minisign key. The agent creates it in M0-03:
  1. Generate a strong random password and the key pair: `npm run tauri signer generate -- -w ~/.tauri/yellowmoon.key -p <password>` (the private key file stays outside the repository).
  2. Store both as repository secrets with the GitHub CLI: `gh secret set TAURI_SIGNING_PRIVATE_KEY < ~/.tauri/yellowmoon.key` and `gh secret set TAURI_SIGNING_PRIVATE_KEY_PASSWORD` (password on stdin). This needs admin rights on the repository; without them, prepare the exact commands and list them under "Waiting on Mykyta".
  3. Put only the public key (`~/.tauri/yellowmoon.key.pub`) into `tauri.conf.json` and commit it.
  4. Write the password next to the key (`~/.tauri/yellowmoon.key.password`, mode 600) and tell Mykyta in `STATUS.md` where both files are, so he can move them to his password manager. Losing the key means installed copies can never auto-update again; leaking it lets anyone ship updates.
  The private key and password never appear in the repository, logs, PR text or `STATUS.md`.
  **Decision (M0-03): do not generate the key in a throwaway environment.** If the agent has no authenticated `gh` with admin rights, or runs in a container that is discarded at the end of the session, it must not create the key: a key it cannot set as a secret and cannot hand over safely is lost on exit, and handing it over through a transcript leaks it. Instead the agent keeps `release.yml` and `tauri.conf.json` ready, writes the exact commands to `docs/release-signing.md` (no secrets in it) and lists the step under "Waiting on Mykyta". The M0 build did this. In that case Mykyta, not the agent, holds the only copy of the key (steps 1 to 4 above, run by him), and the password file in step 4 is replaced by his password manager.
- Apple code signing and notarisation: **not used** (Mykyta, 2026-10-15; `docs/16`, M4-H2 answered). The app ships without a Developer ID, like plugout; the bundle is signed ad hoc only (Apple silicon needs a signature to run at all), so macOS asks once before it first opens a downloaded copy. The README section "Allow Yellowmoon to open" says how (System Settings › Privacy & Security › Open Anyway on macOS 15 and later, Control-click › Open before that, or `xattr -dr com.apple.quarantine`), and the cask prints the same as a caveat. Updates the app installs itself are not quarantined and open without asking. The `APPLE_*` variables in `release.yml` stay commented.

## Repository and name

- Repository: `github.com/ngriaznov/yellowmoon`, under Mykyta's personal account, like plugout. Yellowmoon is a personal art project; the zerosize organisation is for libraries such as Rightsize and Sundog. The domain comes with the handover message (`AGENTS.md`).
- Before announcing: trademark clearance in the EU, UK, France and WIPO in classes 9 and 42 [HUMAN M4-H3]. Nearest known neighbour: Yellow Moon Post Production Ltd (Northern Ireland, film and TV post, since 1996). Details: `research/notes/name-and-trademark.md`.

## Differences from plugout

| Topic | plugout | Yellowmoon |
|---|---|---|
| Workspace | single crate in `src-tauri` | Cargo workspace at the root; `src-tauri` is one member |
| Version source | `src-tauri/Cargo.toml` via grep | `cargo metadata` (workspace version) |
| Sidecar | none | `ym` universal binary |
| Bundle data | none | `data/` resources |
| Window | normal webview | transparent webview over a native Metal layer (`macOSPrivateApi`) |
| CI | fmt, clippy, test, vitest, playwright | plus cargo-deny, npm licence check, decoder oracle, golden images, dng_validate, training dry run |

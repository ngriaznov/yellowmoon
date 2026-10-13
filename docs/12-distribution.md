# 12 · Distribution and auto-update

We copy github.com/ngriaznov/plugout's setup: every version bump on main becomes a signed GitHub release, and the app updates itself from it. Templates: `templates/github/workflows/`, `templates/Casks/yellowmoon.rb`, `templates/tauri.conf.json`.

## Versioning

- One version for the app, the CLI and the workspace: `[workspace.package] version` in the root `Cargo.toml`; `package.json` mirrors it (a test checks they match).
- Semver. 0.x during M0–M3 betas; 1.0.0 at the end of M4.
- Bump the version in a PR titled "Release x.y.z" with a CHANGELOG entry. Merging it releases.
- `pipeline_version` (`docs/01`) is separate and changes only when output pixels change.

## Release pipeline (`release.yml`, GitHub Actions on macOS, on push to main)

1. Read the version with `cargo metadata` (package `yellowmoon-app`). If tag `v<version>` exists, stop: the push was not a release. Otherwise, before any build, stop with an explicit error if either signing secret is empty or `tauri.conf.json` still holds the `REPLACE_WITH_MINISIGN_PUBLIC_KEY` placeholder (setup: `docs/release-signing.md`). A release that cannot be signed fails in seconds, not after the full macOS build. `rust-toolchain.toml` pins the toolchain, so the job runs `rustup target add` for both Apple targets on the pinned one.
2. Build `ym-cli` for aarch64 and x86_64 and place three sidecars: `ym-aarch64-apple-darwin`, `ym-x86_64-apple-darwin` (each per-arch cargo build of `src-tauri` checks for its own) and the `lipo`-merged `ym-universal-apple-darwin` (bundled).
3. `npm run tauri build -- --target universal-apple-darwin` with `TAURI_SIGNING_PRIVATE_KEY` (and its password) from repository secrets; `createUpdaterArtifacts` produces `.app.tar.gz` and `.sig`.
4. Write `latest.json` with `jq`: version, publication date, URL and signature for `darwin-aarch64` and `darwin-x86_64` (the same universal tarball).
5. Publish release `v<version>` with the `.dmg`, the `.app.tar.gz` and `latest.json`, with generated release notes (`softprops/action-gh-release@v2`).
6. Update `Casks/yellowmoon.rb` (version, SHA-256 of the dmg) and push to main.

**Bundle naming detail:** the app's main binary is `Yellowmoon` and macOS file systems are usually case-insensitive, so a sidecar called `yellowmoon` would collide with it inside `Contents/MacOS`. The command is `ym` (decision 2026-10-12), so the sidecar is `Contents/MacOS/ym` and there is nothing to rename: the cask's `binary` stanza and the "Install Command-Line Tool…" menu link it as `ym`.

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

App, sidecar CLI, `data/` (ALEXA curve, ARRI matrices with notice, camera profiles, built-in looks, model weights: a few MB in total), Figtree fonts with OFL text, `NOTICE`, `LICENSE`. If weights grow beyond ~30 MB, move them to a versioned release asset downloaded on first run with a hash check, and keep a classical-only fallback.

## Signing

- Updater artefacts are always signed with the Tauri minisign key. The agent creates it in M0-03:
  1. Generate a strong random password and the key pair: `npm run tauri signer generate -- -w ~/.tauri/yellowmoon.key -p <password>` (the private key file stays outside the repository).
  2. Store both as repository secrets with the GitHub CLI: `gh secret set TAURI_SIGNING_PRIVATE_KEY < ~/.tauri/yellowmoon.key` and `gh secret set TAURI_SIGNING_PRIVATE_KEY_PASSWORD` (password on stdin). This needs admin rights on the repository; without them, prepare the exact commands and list them under "Waiting on Mykyta".
  3. Put only the public key (`~/.tauri/yellowmoon.key.pub`) into `tauri.conf.json` and commit it.
  4. Write the password next to the key (`~/.tauri/yellowmoon.key.password`, mode 600) and tell Mykyta in `STATUS.md` where both files are, so he can move them to his password manager. Losing the key means installed copies can never auto-update again; leaking it lets anyone ship updates.
  The private key and password never appear in the repository, logs, PR text or `STATUS.md`.
  **Decision (M0-03): do not generate the key in a throwaway environment.** If the agent has no authenticated `gh` with admin rights, or runs in a container that is discarded at the end of the session, it must not create the key: a key it cannot set as a secret and cannot hand over safely is lost on exit, and handing it over through a transcript leaks it. Instead the agent keeps `release.yml` and `tauri.conf.json` ready, writes the exact commands to `docs/release-signing.md` (no secrets in it) and lists the step under "Waiting on Mykyta". The M0 build did this. In that case Mykyta, not the agent, holds the only copy of the key (steps 1 to 4 above, run by him), and the password file in step 4 is replaced by his password manager.
- Apple code signing and notarisation: open decision for 1.0 (`docs/16`). plugout ships unsigned with an `xattr -cr` caveat; that's acceptable for betas. If adopted [HUMAN M4-H2]: Apple Developer ID certificate and an app-specific password as secrets; uncomment the `APPLE_*` variables in `release.yml`; remove the cask caveat.

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

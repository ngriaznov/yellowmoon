# Release signing: one-time setup [HUMAN]

Status (2026-10-15): **the key exists; one step is Mykyta's.** At his request the agent generated the updater key pair (minisign, no password, like plugout's; public key ID `27EF00347147056C`), put the public key into `src-tauri/tauri.conf.json`, checked that the key signs and that the signature verifies against the public key, and handed the private key file to Mykyta as an attachment in the session (it is in no repository file, log or status text). The container that made it is discarded, so his copy is the only one besides the secret. Left for Mykyta: store the file in a password manager, then `gh secret set TAURI_SIGNING_PRIVATE_KEY --repo ngriaznov/yellowmoon < yellowmoon-updater.key` (the session's GitHub proxy does not allow the secrets API), and delete the download. No password secret: the key has none. Steps 1 to 4 below are the record of how such a key is made. Until you do, a version bump on `main` stops in the "Require the updater signing setup" step with an error that points here, and nothing is released: a version built without the public key could never update itself afterwards.

## Why the agent did not do this

`AGENTS.md` and `docs/12` say the agent creates the key and sets the secrets. In the environment that built the M0 skeleton that was not possible to do safely:

- It had no authenticated `gh` with admin rights on `ngriaznov/yellowmoon`, so it could not set repository secrets. `docs/12` says to prepare the exact commands in that case. These are they.
- The container is thrown away when the session ends. A key generated there would be lost, and the only way to hand it over would be to print it into a transcript. A leaked private key lets anyone ship an update to every installed copy, so we did not generate one.

No key, password or secret is in this repository, and none is in this file.

## What you need

- Admin rights on the repository, and the GitHub CLI signed in: `gh auth login`, then `gh auth status`.
- A checkout of the repository with `npm ci` done (the Tauri CLI is a dev dependency).
- A password manager.

## Steps

### 1. Create the key pair

```sh
mkdir -p ~/.tauri && chmod 700 ~/.tauri
npm run tauri signer generate -- -w ~/.tauri/yellowmoon.key
```

The CLI asks for a password twice. Use a long random one from your password manager. Do not pass it with `-p` on the command line: it would end up in your shell history and in the process list.

This writes two files:

| File | What | Secret? |
|---|---|---|
| `~/.tauri/yellowmoon.key` | the private key (encrypted with your password) | **yes** |
| `~/.tauri/yellowmoon.key.pub` | the public key | no, it goes into the repository |

### 2. Back them up now

Put the private key file's contents and its password in your password manager, as two entries or one entry with an attachment. **Losing the key means that installed copies of Yellowmoon can never auto-update again** (they would have to be reinstalled by hand), and leaking it lets anyone ship an update to them. Do this before step 3.

### 3. Set the repository secrets

```sh
gh secret set TAURI_SIGNING_PRIVATE_KEY --repo ngriaznov/yellowmoon < ~/.tauri/yellowmoon.key
gh secret set TAURI_SIGNING_PRIVATE_KEY_PASSWORD --repo ngriaznov/yellowmoon
gh secret list --repo ngriaznov/yellowmoon
```

The second command prompts for the value (the input is hidden); paste the password. The last command should list both names. A key without a password (plugout's has none) needs only the first secret: skip the second, since GitHub cannot store an empty one, and the release workflow then uses the key without a password. If the password is wrong the release fails at the signing step with an error about decoding the secret key.

### 4. Put the public key into the Tauri config

```sh
jq --arg k "$(cat ~/.tauri/yellowmoon.key.pub)" '.plugins.updater.pubkey = $k' \
  src-tauri/tauri.conf.json > src-tauri/tauri.conf.json.new \
  && mv src-tauri/tauri.conf.json.new src-tauri/tauri.conf.json
git diff src-tauri/tauri.conf.json
```

The value replaces the `REPLACE_WITH_MINISIGN_PUBLIC_KEY …` placeholder in `plugins.updater.pubkey`. `jq` may re-indent the file; the diff should show the one changed value and nothing else that matters. (Editing the value by hand works just as well.) Commit it in its own pull request. The public key is safe to publish. The release workflow refuses to build while the placeholder is still there.

### 5. Optional local check

This proves that the key file and the password belong together, before CI spends macOS minutes on it:

```sh
echo test > /tmp/yellowmoon-signing-check.txt
npm run tauri signer sign -- -f ~/.tauri/yellowmoon.key /tmp/yellowmoon-signing-check.txt
ls /tmp/yellowmoon-signing-check.txt.sig && rm /tmp/yellowmoon-signing-check.txt*
```

The CLI asks for the password if it is not in the environment. Do not put the password in a file.

### 6. First release

1. The first release is 0.1.0: the version and the `## [0.1.0]` section of `CHANGELOG.md` are already in place (later ones: `tools/release/bump-version.sh patch` and a new section, `docs/12`).
2. Merge it into `main`. The Release workflow runs on macOS and:
   - builds the two command-line binaries and merges them with `lipo`,
   - builds the universal bundle and signs the updater tarball,
   - publishes the GitHub release `v0.1.0` with `Yellowmoon_0.1.0_universal.dmg`, `Yellowmoon.app.tar.gz`, its `.sig` and `latest.json`,
   - updates `Casks/yellowmoon.rb` (version and the dmg's SHA-256) and pushes it to `main`.
3. Check the release page for the four files, and open `latest.json`: both `darwin-aarch64` and `darwin-x86_64` need a non-empty `signature`.

If `main` is protected, the last step (the cask commit) needs the `github-actions[bot]` to be allowed to push to it, or the step fails after the release was already published. Allow it in the branch protection rule, or tell the agent so it can switch the cask update to a pull request.

### 7. Checks that need a Mac [M0-H4]

These are the M0-03 acceptance checks that no Linux container can run:

1. Install 0.1.0 from the dmg, publish 0.1.1 the same way (`tools/release/bump-version.sh patch`), and confirm that the installed app finds the update, downloads it and restarts into 0.1.1. A "signature verification failed" message means the public key in the installed app does not match the secret key used in CI: redo steps 3 and 4 with one key pair.
2. `brew tap ngriaznov/yellowmoon https://github.com/ngriaznov/yellowmoon && brew install --cask yellowmoon`, then `ym --version`. (Recent Homebrew may ask for `brew trust ngriaznov/yellowmoon` once.)

## If something goes wrong later

- **Lost the key or the password:** installed copies can no longer verify new updates. Generate a new pair (steps 1 to 4, with `-f` to overwrite) and publish a release that people install by hand. Say so in the release notes.
- **Key leaked:** same as above, and delete the old secrets first (`gh secret delete TAURI_SIGNING_PRIVATE_KEY --repo ngriaznov/yellowmoon`).
- **Apple code signing and notarisation** are not used: Yellowmoon ships without a Developer ID (Mykyta, 2026-10-15; `docs/12`, `docs/16`). The `APPLE_*` variables in `release.yml` stay commented; the README tells people how to allow the app to open.
- **Newer Tauri CLIs** can bind a signature to the app version (`tauri signer sign --app-version`, set automatically by `tauri build`). Whether to turn on the updater's matching check is a later hardening decision; nothing here depends on it.

# Release signing [HUMAN]

How the updater signing key is created, stored, used by the release workflow and replaced. The in-app updater accepts an update only if its signature verifies against the public key built into the app, so the key is what makes every installed copy trust a release. Creating or replacing the key and setting repository secrets need admin rights on the repository, so they are the maintainer's steps. The release pipeline itself is in `docs/12-distribution.md`.

## Current state

- The updater key pair is a minisign pair. The private key has no password; its public key ID is `27EF00347147056C`.
- The public key is in `src-tauri/tauri.conf.json` under `plugins.updater.pubkey`. It is safe to publish.
- The private key is the repository secret `TAURI_SIGNING_PRIVATE_KEY`; its backup belongs in the maintainer's password manager (step 2 below). There is no `TAURI_SIGNING_PRIVATE_KEY_PASSWORD` secret because the key has none (GitHub cannot store an empty secret, so the release workflow uses the key without a password when the secret is absent).
- The release workflow stops before building, with an error that points here, when the private-key secret is missing or `tauri.conf.json` still holds the `REPLACE_WITH_MINISIGN_PUBLIC_KEY …` placeholder. A version built without the public key could never update itself afterwards.

No key, password or secret is in this repository, and none belongs in this file, in a log, in a pull request or in `docs/STATUS.md`.

## What the maintainer needs

- Admin rights on the repository, and the GitHub CLI signed in: `gh auth login`, then `gh auth status`.
- A checkout of the repository with `npm ci` done (the Tauri CLI is a dev dependency).
- A password manager.

## Creating a key pair

Used for the first key and for a replacement. Never create the key in an environment that is discarded afterwards, and never print it into a transcript: a key that cannot be stored safely is lost, and a leaked one lets anyone ship an update to every installed copy.

### 1. Create the key pair

```sh
mkdir -p ~/.tauri && chmod 700 ~/.tauri
npm run tauri signer generate -- -w ~/.tauri/yellowmoon.key
```

The CLI asks for a password twice. Use a long random one from the password manager, or an empty one for a key without a password. Do not pass the password with `-p` on the command line: it would end up in the shell history and in the process list.

This writes two files:

| File | What | Secret? |
|---|---|---|
| `~/.tauri/yellowmoon.key` | the private key (encrypted with the password, if there is one) | **yes** |
| `~/.tauri/yellowmoon.key.pub` | the public key | no, it goes into the repository |

### 2. Back them up

Put the private key file's contents and its password in the password manager, as two entries or one entry with an attachment. Losing the key means that installed copies of Yellowmoon can never auto-update again (they have to be reinstalled by hand), and leaking it lets anyone ship an update to them. Do this before step 3.

### 3. Set the repository secrets

```sh
gh secret set TAURI_SIGNING_PRIVATE_KEY --repo ngriaznov/yellowmoon < ~/.tauri/yellowmoon.key
gh secret set TAURI_SIGNING_PRIVATE_KEY_PASSWORD --repo ngriaznov/yellowmoon
gh secret list --repo ngriaznov/yellowmoon
```

The second command prompts for the value (the input is hidden); paste the password. The last command lists the names that are set. A key without a password needs only the first secret: skip the second. If the password is wrong, the release fails at the signing step with an error about decoding the secret key.

### 4. Put the public key into the Tauri config

```sh
jq --arg k "$(cat ~/.tauri/yellowmoon.key.pub)" '.plugins.updater.pubkey = $k' \
  src-tauri/tauri.conf.json > src-tauri/tauri.conf.json.new \
  && mv src-tauri/tauri.conf.json.new src-tauri/tauri.conf.json
git diff src-tauri/tauri.conf.json
```

The value replaces `plugins.updater.pubkey`. `jq` may re-indent the file; the diff should show the one changed value and nothing else that matters (editing the value by hand works just as well). Commit it in its own pull request.

### 5. Optional local check

This proves that the key file and the password belong together, before CI spends macOS minutes on it:

```sh
echo test > /tmp/yellowmoon-signing-check.txt
npm run tauri signer sign -- -f ~/.tauri/yellowmoon.key /tmp/yellowmoon-signing-check.txt
ls /tmp/yellowmoon-signing-check.txt.sig && rm /tmp/yellowmoon-signing-check.txt*
```

The CLI asks for the password if it is not in the environment. Do not put the password in a file.

## What a release needs from the key

A push to `main` that raises the version runs `release.yml` (`docs/12-distribution.md`, Release pipeline). With the secrets and the public key in place it builds the universal bundle, signs the updater tarball, and publishes the disk image, `Yellowmoon.app.tar.gz`, its `.sig` and `latest.json`. After a release, check the release page for the four files and open `latest.json`: both `darwin-aarch64` and `darwin-x86_64` need a non-empty `signature`.

If `main` is protected, the last step of the workflow (the cask commit) needs `github-actions[bot]` to be allowed to push to it, or the step fails after the release was published.

## Checks that need a Mac [M0-H4]

These are the M0-03 acceptance checks that no Linux container can run:

1. Install one release from the disk image, publish the next one (`tools/release/bump-version.sh patch`), and confirm that the installed app finds the update, downloads it and restarts into it. A "signature verification failed" message means the public key in the installed app does not match the secret key used in CI: redo steps 3 and 4 with one key pair.
2. `brew tap ngriaznov/yellowmoon https://github.com/ngriaznov/yellowmoon && brew install --cask yellowmoon`, then `ym --version`. (Recent Homebrew may ask for `brew trust ngriaznov/yellowmoon` once.)

## If something goes wrong

- **Lost the key or the password:** installed copies can no longer verify new updates. Generate a new pair (steps 1 to 4, with `-f` to overwrite) and publish a release that people install by hand. Say so in the release notes.
- **Key leaked:** same as above, and delete the old secrets first (`gh secret delete TAURI_SIGNING_PRIVATE_KEY --repo ngriaznov/yellowmoon`).
- **Apple code signing and notarisation** are not used: Yellowmoon ships without a Developer ID (`docs/12-distribution.md`, Signing). The `APPLE_*` variables in `release.yml` stay commented; the README tells people how to allow the app to open.
- **Newer Tauri CLIs** can bind a signature to the app version (`tauri signer sign --app-version`, set automatically by `tauri build`). The updater's matching check is not turned on; nothing here depends on it.

# Licence texts of the dependencies

NOTICE item 8 and the About window promise the licence texts of everything we build on, and a binary release has to carry
them (BSD, MIT, ISC, Apache-2.0 and the OFL all make that a condition). This folder produces them.

| File | What it is |
|---|---|
| `generate.py` | Writes `THIRD-PARTY.md` at the repository root and verifies it. Rust: `cargo about generate --format json` for the two programs that ship (`src-tauri` and `crates/ym-cli`), macOS targets, normal dependencies only. npm: `license-checker --production`, plus Tailwind CSS (a build tool whose output is in the app's stylesheet: `BUILD_TOOL_OUTPUT`). Fonts: the Figtree OFL text. Our own crates are not listed (`LICENSE` covers them). |
| `about.toml` | The cargo-about configuration: the accepted licences (the allow list of `deny.toml`; a unit test keeps the two equal), the two macOS targets, no build or dev dependencies. |
| `check_bundle.py` | Fails when `src-tauri/tauri.conf.json` does not put `NOTICE`, `LICENSE` and `THIRD-PARTY.md` into `bundle.resources`, or when `NOTICE` lacks the BSD 3-Clause conditions of the ported Menon demosaic or the MIT notice of glasscn. |
| `texts/` | Licence texts of npm packages that ship no LICENSE file (`HAND_ADDED_TEXTS` in `generate.py`): today `react-remove-scroll-bar`, a dependency of Radix. Each is the LICENSE file of the package's repository. |
| `THIRD-PARTY.stub.md` | What a development build gets in place of the real file, so that tauri-build finds the resource. `generate.py --verify` never accepts it. |
| `test_licences.py` | Unit tests (stdlib only): `python3 -I -m unittest discover -s tools/licences`. |

```
# one time: cargo-about 0.9.2 (a prebuilt binary is in the GitHub release; taiki-e/install-action installs it in CI)
cargo install cargo-about --version 0.9.2 --locked

npm ci
python3 tools/licences/generate.py          # THIRD-PARTY.md, about 0.5 MB; exits 1 on a package without a licence text
python3 tools/licences/check_bundle.py      # the bundle and NOTICE
```

`THIRD-PARTY.md` is generated, git-ignored and never edited by hand. The output depends only on `Cargo.lock`,
`package-lock.json` and the tool versions (no date, sorted). Crates are fetched first (`cargo fetch --locked`) and
cargo-about runs offline, so the texts are the LICENSE files inside the crates, copyright lines included, and do not depend
on a web service.

**Where it runs.** The CI job `licences` runs the unit tests, the generator and the bundle check on every push and pull
request, so a dependency without a licence text, a licence outside the accepted list, or a bundle that stops carrying the
file fails there. `release.yml` runs the generator and the bundle check before `tauri build`. `tools/dev/prepare-sidecar.sh`
writes the stub when `tauri.conf.json` lists the file and no real one exists.

**A package without a licence file.** The generator stops. The Tauri plugins' npm packages ship only an SPDX tag (`LICENSE.spdx`);
for them (and only for `@tauri-apps/plugin-*`) the texts of the Rust crate of the same repository are used, and the file says so
(`rust_counterpart` in `generate.py`). Another package needs the same kind of explicit rule, or its text added by hand: save the repository's LICENSE in `texts/<package>.LICENSE` and name the package in `HAND_ADDED_TEXTS`.

**What is not covered here.** Data and ported code are attributed in `NOTICE` (AMPAS terms, rawtoaces-data,
the Menon demosaic). Fonts other than Figtree, if one is ever added, need a line in `collect_fonts`.

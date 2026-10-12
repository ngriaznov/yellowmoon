# 10 · Command line

Crate: `ym-cli` (binary `yellowmoon`). Built with `clap` (derive), headless wgpu on Metal, `serde_json` output. No Tauri dependency.

The CLI is a second front end on the same core. A file rendered from the terminal matches the app's export exactly: same decoder, same automatic exposure and white balance, same looks, same export code. It is also our golden-image test harness.

## Commands

| Command | What it does |
|---|---|
| `yellowmoon render <files or folders…>` | Develop and export. Defaults match the app: automatic development and the user's default look. Flags change them for this run only. |
| `yellowmoon render -` | Read file paths from stdin, one per line (composes with `find`, `fd`, `xargs`). |
| `yellowmoon info <file…>` | Metadata and automatic estimates without rendering. |
| `yellowmoon looks list` | Built-in and imported looks, with ids, names, groups, inputs, and which is default. |
| `yellowmoon looks import <zip or folder or .cube…> [--input logc4 or logc3 or logc3-rec709]` | The wizard's importer without the UI: same library folder, recognition rules and checksums. `--input` answers the generic question for non-ARRI `.cube` files. |
| `yellowmoon looks verify` | Re-hash every imported look; report missing or changed files. |
| `yellowmoon looks default <id>` | Set the default look. |
| `yellowmoon looks bake <look.json> [--preview <ARW>…] [--out <dir>]` | Developer tool for the Signature look: bake to a 33³ LUT and render a contact sheet. |
| `yellowmoon doctor` | GPU (name, backend, limits), runtime chosen, model weights and hashes, profiles and versions, imported looks count, preferences path, version. For bug reports. |
| `yellowmoon --version` | Version, pipeline version, git SHA. |

Folders are walked for `*.ARW` (case-insensitive). `--recursive` walks subfolders.

## `render` options

They mirror the app's controls. Anything not given uses the automatic value or the user's preference.

| Option | Values | Notes |
|---|---|---|
| `--look <look>` | full id (`builtin:neutral`), its slug after the colon (`neutral`), or the display name, case-insensitive | ambiguous → exit 2 listing matches; missing → exit 3 |
| `--settings <file or ->` | JSON with the `ym:Settings` schema (resolved `DevelopSettings`, `docs/01`) | covers everything the flags can't (colour balance, curve, crop, orientation, white pick); flags given alongside override it; per-file settings via an object keyed by file name |
| `--intensity <0–100>` | percent | default 100 |
| `--exposure auto or <EV>` | absolute EV −4…+4, or `auto` | `--exposure-offset <EV>` adds to auto: resolved per file to `Manual(EV_auto + offset)`, so there's no extra settings field |
| `--wb auto or <K>[,<tint>]` | 2000–11000, tint −50…50 | `--light day or shade or tungsten or led or fluorescent` sets the light preset |
| `--contrast`, `--brightness`, `--highlights`, `--shadows`, `--white`, `--black`, `--saturation` | −100…100 | |
| `--nr auto or <L>,<C>,<D>` | 0–100 each | |
| `--sharpen auto or <amount>,<radius>` | 0–300, 0.3–2.5 | |
| `--no-neural`, `--no-highlight-recovery`, `--no-lens`, `--no-ca` | flags | RAW › Decode switches |
| `--rolloff <highlights>,<low-light>` | 0–1 each | |
| `--profile <id>` | camera profile id | forces a profile, for comparisons |
| `--display aces2 or arri` | | display transform (arri needs imported display LUTs) |
| `--format jpeg or tiff or png or dng` | | default from preferences, else jpeg |
| `--quality <50–100>` | JPEG | default 92 |
| `--bits 8 or 16` | TIFF, PNG | |
| `--compression none or lzw or zip` | TIFF | |
| `--colour-space srgb or p3 or rec2020` | alias `--color-space` | not for DNG |
| `--long-edge <px>` | | not for DNG |
| `--out <dir>` | | default: `<source folder> export` |
| `--name <template>` | `{name}`, `{look}`, `{date}`, `{seq}`, `{camera}` | default `{name}_{look}` |
| `--skip-existing` or `--overwrite` | | default keeps both (" 2" suffix) |
| `--jobs <n>` | parallel decodes | the GPU keeps one queue |
| `--json` | | one JSON object per file on stdout |
| `--require-gpu` | | exit 4 if no usable GPU instead of falling back to CPU |
| `--quiet`, `--verbose` | | logs go to stderr |

## JSON output (`--json`)

One object per input file, written when that file finishes, in completion order:

```json
{
  "input": "/Volumes/A1/DSC04204.ARW",
  "output": "/Users/me/Pictures/Lisbon export/DSC04204_neutral.jpg",
  "status": "ok",
  "camera": { "body": "Sony A1", "model_id": 384, "profile": "sony-ilce-7rm4", "profile_version": "1.0.0", "match": "default" },
  "look": { "id": "builtin:neutral", "intensity": 100 },
  "auto": { "exposure_ev": 0.30, "kelvin": 5450, "tint": 0, "light_class": "daylight", "light_confidence": 0.82,
            "noise": { "a": 1.9e-5, "b": 3.1e-7 } },
  "settings": { "...": "resolved settings, same as ym:Settings" },
  "timings_ms": { "decode": 310, "develop": 2410, "encode": 180 },
  "pipeline_version": "1.0.0",
  "error": null
}
```

On failure: `"status": "error"`, `"output": null`, `"error": { "kind": "unsupported", "message": "…" }`. `info --json` prints the same object without `output` and `timings_ms`.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | every file rendered |
| 1 | some files failed (others succeeded) |
| 2 | usage error |
| 3 | a requested look is missing |
| 4 | `--require-gpu` and no usable GPU |
| 5 | no input files found |

## Behaviour

- Nothing is saved per photo. Each run starts from the automatic development and the default look.
- Same inputs, settings and version give the same output bytes.
- Runs headless through wgpu's Metal backend with no window or surface. Falls back to the CPU path when no GPU is available (with a warning on stderr), unless `--require-gpu`.
- Safe to run while the app is open: both use the same library folder; imports take a lock.
- Progress bar on stderr when stderr is a TTY and `--json`/`--quiet` aren't set.
- Ctrl-C finishes the current tile, deletes the partial file, prints a summary and exits 130.

## Examples

```
yellowmoon render ~/Pictures/2026-10-04 --look neutral --format jpeg --long-edge 3000 --out ~/Exports
fd -e ARW . /Volumes/A1 | yellowmoon render - --json --format tiff --bits 16 > results.jsonl
yellowmoon looks import ~/Downloads/<ARRI look package>.zip
yellowmoon info DSC04204.ARW --json | jq .auto
```

## Install

- The binary ships inside the app bundle as a Tauri sidecar (`bundle.externalBin`), so it is signed, shipped and auto-updated with the app.
- The Homebrew cask links it onto PATH (`binary` stanza).
- A `.dmg` install gets the menu item "Install Command-Line Tool…", which creates `/usr/local/bin/yellowmoon` (asking for admin rights through macOS) or `~/.local/bin/yellowmoon` if the user prefers, as VS Code does for `code`.
- V2 options: a CLI-only Homebrew formula; Shortcuts actions through App Intents.

## Tests

- `assert_cmd` + `insta` snapshot tests for `--help`, `info --json` on fixtures, exit codes for each failure.
- App/CLI parity test: export through `ym-core`'s app path and through the binary; byte-identical.
- Golden runner (`tools/golden`) calls `yellowmoon render --json` on the golden set on every commit and compares ΔE00 to stored references.

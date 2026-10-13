# 10 · Command line

Crate: `ym-cli` (binary `ym`). Built with `clap` (derive), `serde_json` output. No Tauri dependency.

> **Decision 2026-10-12 (Mykyta):** the command is `ym`, not `yellowmoon`. It is `ym` everywhere: the binary, the cask, the link `Install Command-Line Tool…` creates, the help text, the docs and `ym doctor`. The crate stays `ym-cli`. See docs/16 decision log.

The CLI is a second front end on the same core. A file rendered from the terminal matches the app's export exactly: same decoder, same automatic exposure and white balance, same looks, same export code (`ym_core::export::export_photo`, the one function that makes a file; the CLI drives the app's export queue). It is also our golden-image test harness.

## Commands

| Command | What it does |
|---|---|
| `ym render <files or folders…>` | Develop and export. Defaults match the app: automatic development and the user's default look. Flags change them for this run only. |
| `ym render -` | Read file paths from stdin, one per line (composes with `find`, `fd`, `xargs`). |
| `ym info <file…>` | Metadata and automatic estimates without rendering. |
| `ym looks list` | Built-in and imported looks, with ids, names, groups, inputs, and which is default. |
| `ym looks import <zip or folder or .cube…> [--input logc4 or logc3 or logc3-rec709]` | The wizard's importer without the UI: same library folder, recognition rules and checksums. `--input` answers the generic question for non-ARRI `.cube` files. |
| `ym looks verify` | Re-hash every imported look; report missing or changed files. |
| `ym looks default <look>` | Set the default look. |
| `ym looks bake <look.json> [--preview <ARW>…] [--out <dir>]` | Developer tool for the Signature look: bake to a 33³ LUT and render a contact sheet. |
| `ym doctor` | GPU (name, backend, limits), runtime chosen, model weights and hashes, profiles and versions, imported looks count, preferences path, version. For bug reports. |
| `ym --version` | Version, pipeline version, git SHA. |

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
| `--contrast`, `--brightness`, `--highlights`, `--shadows`, `--white`, `--black`, `--saturation` | −100…100 | `--saturation` is subtractive (`docs/02` C9): more makes colours denser and darker, less makes them paler, and the brightest channel and neutrals stay. |
| `--nr auto or <L>,<C>,<D>` | 0–100 each | |
| `--sharpen auto or <amount>,<radius>` | 0–300, 0.3–2.5 | capture sharpening (R8) |
| `--output-sharpening <0–200>` | percent, default 100 | Detail › Sharpening › Output (`docs/algorithms/04`): sharpens JPEG, TIFF and PNG files at the size they are written, right after the resize; 0 is off and leaves the bytes of an export without the stage; a value outside 0–200 is a usage error (exit 2); a DNG is never sharpened (`--output-sharpening` with `--format dng` is warned about and ignored). It is the field `output_sharpening` of the settings object. |
| `--no-neural`, `--no-highlight-recovery`, `--no-lens`, `--no-ca` | flags | RAW › Decode switches |
| `--rolloff <highlights>,<low-light>` | 0–1 each | |
| `--profile <id>` | camera profile id | forces a profile, for comparisons |
| `--display aces2 or arri` | | display transform (arri needs imported display LUTs) |
| `--format jpeg or tiff or png or dng` | | default from preferences, else jpeg. The help of `dng` is "DNG with the look baked in. Apps other than Adobe's show the stored linear image darker until their own exposure handling is applied." The second sentence is `ym_export::dng::OTHER_APPS_NOTE`, used word for word by the Export panel's DNG paragraph; tests compare both with the constant (`docs/09`). |
| `--quality <50–100>` | JPEG | default 92 |
| `--bits 8 or 16` | TIFF, PNG | |
| `--compression none or lzw or zip` | TIFF | |
| `--colour-space srgb or p3 or rec2020` | alias `--color-space` | not for DNG |
| `--long-edge <px>` | | not for DNG |
| `--out <dir>` | | default: `<source folder> export` |
| `--name <template>` | `{name}`, `{look}`, `{date}`, `{seq}`, `{camera}` | default `{name}_{look}` |
| `--skip-existing` or `--overwrite` | | default keeps both (" 2" suffix). Both look at the files that were there **before the run**: a name that an earlier photo of the same run has already written is always numbered, whatever is chosen (two folders that both hold `DSC00001.ARW`, or `--name {date}` with no `{seq}`), and the run says so once in a note. |
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
| 2 | usage error (also a data folder or home folder that can't be found: set `YELLOWMOON_DATA` or `YELLOWMOON_HOME`) |
| 3 | a requested look is missing |
| 4 | `--require-gpu` and no usable GPU |
| 5 | no input files found |
| 130 | interrupted with Ctrl-C |

## Behaviour

- Nothing is saved per photo. Each run starts from the automatic development and the default look.
- Same inputs, settings and version give the same output bytes.
- JPEG, TIFF and PNG files are sharpened at the size they are written, after the resize (Detail › Sharpening › Output, `docs/algorithms/04`; `--output-sharpening`, default 100 %); `--output-sharpening 0` writes the picture without that stage. A DNG is never sharpened. The amount is part of the `settings` the JSON reports and the file records, so feeding that back with `--settings` writes the same file, and a settings file from before the option reads as 100.
- Runs headless through wgpu's Metal backend with no window or surface. Falls back to the CPU path when no GPU is available (with a warning on stderr), unless `--require-gpu`.
- Safe to run while the app is open: both use the same library folder; imports take a lock.
- Progress bar on stderr when stderr is a TTY and `--json`/`--quiet` aren't set.
- Ctrl-C finishes the current tile, deletes the partial file, prints a summary and exits 130.

## Examples

```
ym render ~/Pictures/2026-10-04 --look neutral --format jpeg --long-edge 3000 --out ~/Exports
fd -e ARW . /Volumes/A1 | ym render - --json --format tiff --bits 16 > results.jsonl
ym looks import ~/Downloads/<ARRI look package>.zip
ym info DSC04204.ARW --json | jq .auto
```

## Install

- The binary ships inside the app bundle as a Tauri sidecar (`bundle.externalBin` is `binaries/ym`, so it is `Contents/MacOS/ym`), so it is signed, shipped and auto-updated with the app. It cannot collide with the app's own binary `Yellowmoon` on a case-insensitive volume.
- The Homebrew cask links it onto PATH (`binary` stanza).
- A `.dmg` install gets the menu item "Install Command-Line Tool…", which creates `/usr/local/bin/ym` (asking for admin rights through macOS) or `~/.local/bin/ym` if the user prefers, as VS Code does for `code`.
- V2 options: a CLI-only Homebrew formula; Shortcuts actions through App Intents.

## Tests

- `assert_cmd` + `insta` snapshot tests for `--help` of every command, `info --json` on fixtures, exit codes for each failure.
- App/CLI parity test: export through `ym-core`'s app path (`Core::export_start`, what the Tauri commands call) and through the binary; byte-identical. It covers JPEG, TIFF (8 and 16 bit, none, LZW and ZIP), PNG (8 and 16 bit), DNG, the Display P3 and Rec.2020 spaces, a long edge, a folder with a name template, settings given as options, settings given as a file, and a photo of several export tiles exported with different decode-thread counts.
- Golden runner (`tools/golden`) calls `ym render --json` on the golden set on every commit and compares ΔE00 to stored references.

## As built (M1-14 CLI v0, M3-14 full CLI and parity)

The CLI is thin: it parses, calls `ym-core` and prints. What it decides itself is recorded here.

**Inputs.** `-` among the inputs is replaced by the paths on stdin (blank lines and a trailing carriage return are dropped); `--settings -` also reads stdin, and giving both is a usage error. Folders are walked by the rules of `Session::open`: `*.ARW` in any case, names starting with a dot ignored (macOS leaves `._DSC0001.ARW` on exFAT cards), other files ignored, one level unless `--recursive`. Photos are developed and exported in the order a session lists them (by absolute path, ignoring case), each file once. A path the user named that cannot be a photo (it does not exist, or is not an ARW file) is reported as an error (`kind` `not_found` or `not_arw`) and counts as a failure, but when no ARW file is left to render at all the exit code is 5.

**Settings.** The photo starts as the session opens it (automatic, the effective default look). Then `--settings` replaces it wholesale (read over the defaults like `DevelopSettings::from_json`, so a file may hold only what it changes; unknown fields are an error; out-of-range numbers are limited and a warning names the fields), then the options are applied as a patch. A settings file whose keys are all settings fields (or that is empty) is one document for every photo; one whose keys are all names of ARW files (`DSC00001.ARW`, in any case) is an object keyed by file name (matched exactly, else ignoring case, else by absolute path; a photo without an entry is developed automatically, an entry that names no photo is a warning); a file that mixes the two is a usage error, and so is a key that is neither (a misspelt field such as `color_ballance` with an object as its value would otherwise pass for a photo's file name and change nothing). All of it is checked before any photo is opened, so a typo costs nothing. Options are checked against the ranges of the table (`--contrast 500` is a usage error, exit 2) and `--exposure` conflicts with `--exposure-offset`. `--exposure-offset` is relative to the photo's *served* automatic exposure (`auto.exposure_ev` of the JSON: the burst median when the photo has neighbours, `docs/02`); when automatic exposure plus offset passes the limit of 4 EV the exposure is limited to it and a warning names the photo. `--wb 5200` without a tint keeps the automatic tint (the semantics of moving the Kelvin slider, `docs/01`). `--light auto` clears a preset a settings file chose. An option that does not apply to the format (`--quality` with PNG, `--long-edge` with DNG) is a warning and is ignored.

**Export options.** Only `--format` takes its default from the preferences (as the table says); the others start from the table of `docs/09` (`--quality` 92, `--bits` 16, `--compression` lzw, `--colour-space` srgb, `{name}_{look}`, keep both), so the command `cli_preview` prints, which leaves those defaults out, does what the panel does whatever the preferences say. The preference "Remove location from exports" applies to the CLI too. The CLI never writes the preferences: `--out` is not remembered as the app's last-used folder, and `--look` does not change the default look (`ym looks default` does).

**`--look` and `--display`.** `--look` resolves through the look library (`LookService::resolve`) before anything is opened: a name that matches nothing exits 3, one that matches several exits 2 and lists them, a look whose file is gone exits 3. A look named in a settings file that turns out to be missing fails that photo only (`error.kind` `look`); when every failure of a run is a missing look the exit code is 3. `--display aces2` forces ACES 2; `--display arri` uses the ARRI display LUT chosen in the app's preferences, else the only one imported, and otherwise says which exist (exit 2) or that none is imported (exit 3); a full display LUT id is accepted too. Without `--display` the preference applies, and an export with a chosen display LUT that is gone is refused (exit 3), not rendered with ACES 2.

**Running.** The core is opened without thumbnails (an export waits behind thumbnail work). Photos are developed in the session's decode pool (`--jobs` sets its size), the CLI waits until every photo is developed (burst medians and noise pools depend on the whole set), then queues one export job over exactly the photos that were developed and given their settings, and prints each file as the queue reports it. A photo that fails does not stop the others. A closed stdout (`| head -n 1`) stops the output, not the export or the library command: the files and the library are the product (`render`, `info` and every `looks` command write their data lines through a writer that ignores a reader that has gone).

**GPU.** Export rendering is on the CPU in this version (the colour pass of viewport frames and thumbnails is the only GPU work in `ym-core`). `--require-gpu` opens the device first and exits 4 if none opens; a software adapter (Mesa lavapipe) counts, as in CI. Without it the device is looked at while the photos develop, and "No usable GPU was found. Rendering on the CPU, which is slower." is a warning on stderr. Opening a device makes the graphics driver write its own shader cache (Mesa on Linux does, under `~/.cache/mesa_shader_cache`); that file is the driver's, and Yellowmoon itself writes nothing but the exports, the preferences and the look library.

**Output streams.** stdout carries data only: with `--json` one object per line, otherwise the path of every file written (one per line; a skipped file prints nothing). Everything for a person goes to stderr: warnings, per-file errors (not with `--json`, where the object has them), the summary ("Exported 3 photos to … in 4.2 s. 1 photo already existed and was skipped. 2 photos failed."), the progress line (TTY only; it is redrawn in place, so a file name longer than 32 characters is shortened with an ellipsis to keep the line on one terminal row) and, once, the note that the classical demosaic was used. `--quiet` leaves errors; `--verbose` adds per-file timings and the `tracing` messages of our crates (`RUST_LOG` overrides the filter).

**JSON.** Keys come in the order of the example above (`input`, `output`, `status`, `camera`, `look`, `auto`, `settings`, `timings_ms`, `notes`, `pipeline_version`, `error`); the order is kept by the CLI itself, not by a `serde_json` feature (`preserve_order` would reorder the `ym:Settings` inside every file the binary writes, and the CLI must write the app's bytes). `input` is an absolute path. `status` is `ok`, `skipped` (`--skip-existing` and the file was there: `output` is the existing file, nothing was rendered) or `error`. `camera` has `make`, `model`, `model_id`, `body`, `firmware`, `arw_version`, `profile` (the id), `profile_name` (`A7R IV`), `profile_version`, `match` (`exact`, `sensor_family`, `default` or `forced`), `stand_in` and `reduced`. `auto` is the `AutoEstimates` of `src/api/types.ts` (the exposure, Kelvin and tint, light class and confidence, the noise model, and what Auto sharpening and noise reduction resolve to). `settings` is the document an export records as `ym:Settings`: exposure, sharpening and noise reduction that were Auto are written as the numbers they resolved to, white balance stays Auto when it was; feeding it back with `--settings` reproduces the file byte for byte. `timings_ms` is `decode` (reading the mosaic, which overlaps the render of the photo before it), `develop` (the render) and `encode`. `notes` are the plain-words notes of the render (today: the classical demosaic). `info --json` adds the metadata blocks (`exposure`, `dimensions`, `raw`, `black`, `white`, `wb_as_shot`, `levels_source`, `lens_corrections`, `tone_curve`, `preview`, `gps`); for a file that has metadata but cannot be developed yet (ARW 6.0) `look`, `auto` and `settings` are null and `notes` says why. A failure that happens after the photo was developed (a look that is gone) still carries `camera`, `look`, `auto` and `settings`; one before (not an ARW file) has `input`, `output`, `status`, `pipeline_version` and `error` only.

**Ctrl-C, SIGTERM and SIGHUP.** Once photos are being developed (`render`, `info`), imported (`looks import`) or rendered into a contact sheet (`looks bake --preview`) the first signal only asks the command to stop: during an export it asks the export queue to finish the current tile or band, delete the partial file and skip the rest; the CLI waits for that, prints "Interrupted: N of M photos exported. Nothing half-written is left behind." and exits 130. Before that point (reading the list on stdin, opening the library, `doctor`, `looks list`) nothing would look at a flag and nothing is half-written, so the first signal ends the process at once with 130. A second signal always ends the process at once, which can leave the temporary `.yellowmoon-<pid>-<n>.tmp` file of the photo that was being written in the destination folder. A folder created for the export stays (empty) when the first photo was cancelled.

**`looks`.** `list` prints a table (`--json` an object with `looks` and `display_luts`, the shapes of `looks_list` and `looks_display_luts`). `import` prints the id of every look it added (stdout) and a summary (stderr); a `.cube` that no known package claims needs `--input` (exit 2 otherwise, the message names the three words); Ctrl-C stops between files. `verify` prints one line per missing or changed file and exits 1 when there is one. `default <look>` takes what `--look` takes. `bake` writes `<name>.cube` (33 points, `TITLE` the look's name) into `--out` (default: the folder of the look file) and, with `--preview`, `<name>_contact.png`: a row per photo, Neutral on the left and the baked look on the right, each 480 px on the long edge, in sRGB; the paths are printed on stdout.

**`doctor`.** A text report whose first sentence says what it includes (the version, the GPU, the render path, the model weights and their hashes, the camera profiles, how many looks are imported, the folders) and that it leaves out photos, file names, locations and look names. `--json` prints the same facts as one object. The GPU section is "none, CPU fallback (reason)" when no device opens. Model weights are everything under `data/models/` except the illuminant models of `models/illum/` (listed separately, each with its SHA-256) and the metrics.

**Not done.** The progress line and the TTY-only behaviour are not covered by a test (a test would need a pseudo-terminal); the unit tests cover the bar. A missing data folder is covered at the unit level (`core_failure`) because debug builds fall back to the workspace's `data/` (`docs/01` Finding `data/`), so the binary of a test build cannot be made to miss it.

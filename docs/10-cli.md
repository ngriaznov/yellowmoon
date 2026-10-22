# 10 · Command line

The `ym` command: its commands, options, output, exit codes and the rules it follows. Crate `ym-cli` (binary `ym`), built with `clap` (derive) and `serde_json` output, with no Tauri dependency. The command is `ym` everywhere: the binary, the Homebrew cask, the link "Install Command-Line Tool…" creates, the help text, the docs and `ym doctor`.

The CLI is a second front end on the same core. A file rendered from the terminal matches the app's export byte for byte: same decoder, same automatic exposure and white balance, same looks, same export code (`ym_core::export::export_photo`, the one function that makes a file; the CLI drives the app's export queue). It is also the golden-image test harness. The CLI is thin: it parses, calls `ym-core` and prints.

## Commands

| Command | What it does |
|---|---|
| `ym render <files or folders…>` | Develop and export. Defaults match the app: automatic development and the user's default look. Flags change them for this run only. |
| `ym render -` | Read file paths from stdin, one per line (composes with `find`, `fd`, `xargs`). |
| `ym info <file…>` | Metadata and automatic estimates without rendering, with what Apple Auto did (`docs/19-apple-auto.md`). `--suggest-crop[=ASPECT]` adds Auto crop's proposal (below). |
| `ym looks list` | Built-in and imported looks, with ids, names, groups, libraries, inputs, and which is default. |
| `ym looks import <zip or folder or .cube or look file…> [--input logc4 or logc3 or logc3-rec709]` | The wizard's importer without the UI: same library folder, recognition rules and checksums. Sources are zips, folders, `.cube` files and ARRI Look Files (`.alf4`, `.alf4c`, `.aml`). `--input` answers the generic question for non-ARRI `.cube` files; look files say what they expect and ignore it. |
| `ym looks verify` | Re-hash every imported look; report missing or changed files. |
| `ym looks default <look>` | Set the default look. |
| `ym looks bake <look.json> [--preview <ARW>…] [--out <dir>]` | Developer tool for the Signature look: bake to a 33-point LUT and render a contact sheet. |
| `ym doctor` | GPU (name, backend, limits), the graphics policy and the stages it runs, the memory budgets, runtime chosen, whether Apple Auto is available and which model answers, model weights and hashes, profiles and versions, imported looks count, preferences path, version. For bug reports. `ym doctor --bench FILE.ARW` times the render path on one photo instead. |
| `ym --version` | Version, pipeline version, git SHA: `ym <version> (pipeline <pipeline version>, <sha>)`. |

Global options: `--quiet` (`-q`, only errors on stderr) and `--verbose` (`-v`, stages, timings and notes on stderr); the two exclude each other.

Folders are walked for `*.ARW` (case-insensitive). `--recursive` walks subfolders.

**Removing photos is app-only.** Removing a photo from the filmstrip, and its Undo (`docs/11-ui.md`, Removing photos), belong to the app's session, which lasts while the person works. `ym render` opens the photos it is given, develops them and exits, so it has no removal and no flag for it: to leave a photo out, leave it out of the command.

## `render` options

The options mirror the app's controls and are grouped as in `ym render --help`. Anything not given uses the automatic value or the user's preference. Numeric options are checked against the ranges below: a value outside a range is a usage error (exit 2), never a quietly clamped value (a settings *file* is different, see Settings).

### Photos

| Option | Values | Notes |
|---|---|---|
| `<files or folders>` | paths, or `-` | `-` is replaced by the paths on stdin |
| `--recursive` | flag | walk the subfolders of folders too |

### Look and display

| Option | Values | Notes |
|---|---|---|
| `--look <look>` | full id (`builtin:neutral`), its slug after the colon (`neutral`), or the display name, case-insensitive | ambiguous: exit 2 listing the matches; missing: exit 3 |
| `--intensity <0–100>` | percent | default 100 |
| `--display aces2 or arri` | | display rendering; `arri` needs an imported ARRI display LUT: the one chosen in the app's preferences, else the only one imported, or a full display LUT id |
| `--settings <file or ->` | JSON with the `ym:Settings` schema (the resolved `DevelopSettings`, `docs/01-architecture.md`) | covers everything the flags can't: colour balance (`{ "kind": "auto" }` puts a shadows or highlights wheel back to Auto), colour warp, curve, crop, orientation, white pick. Flags given alongside it override it. A file with several photos' settings is an object keyed by file name |

### Light

| Option | Values | Notes |
|---|---|---|
| `--exposure auto or <EV>` | absolute EV −4…+4, or `auto` | conflicts with `--exposure-offset` |
| `--exposure-offset <EV>` | stops | added to each photo's automatic exposure: resolved per file to `Manual(EV_auto + offset)`, so there is no extra settings field. It is relative to the photo's *served* automatic exposure (`auto.exposure_ev` in the JSON: the burst median when the photo has neighbours, `docs/02-colour-pipeline.md`). When automatic exposure plus offset passes the limit of 4 EV, the exposure is limited to it and a warning names the photo |
| `--contrast auto or <N>` | −100…100, or `auto` | `auto` is the photo's Auto value: 0, or the value Apple Auto chose (`docs/19-apple-auto.md` section 5); it is the default, and it puts back a value a `--settings` file set by hand. A number is a manual value. Bends the tones around mid grey and keeps black and the top code value (`docs/02-colour-pipeline.md` C9) |
| `--highlights auto or <N>`, `--shadows auto or <N>`, `--black auto or <N>` | −100…100, or `auto` | as `--contrast`: `auto` is the photo's Auto value, 0 or the value Apple Auto chose, and the default. The shadows and highlights colour-balance wheels take Auto in a `--settings` file (`{ "kind": "auto" }`) |
| `--brightness <N>`, `--white <N>` | −100…100 | |
| `--white-offset <EV>` | −3…3 stops, default 0 | Light › White › Offset (`docs/02-colour-pipeline.md` C12 step 1b): stops added to each photo's **automatic white lift**, the per-photo gain on the tones above +1 stop over middle grey that gives the brightest tones of a high-key photo a real white. The automatic amount is sized from the photo's own highlights when it is opened and reported as `auto.white_ev` in the JSON (0 for a low-key photo). 0 is that amount; a negative offset lowers it; an offset that takes it to 0 or below switches the lift off (`--white-offset -3` always does); a positive one adds to it (up to 3 stops in all). The lift acts only with the built-in ACES 2 display rendering (an imported ARRI display LUT and a LogC3-to-Rec.709 look keep ARRI's own highlights), and **a DNG carries it** (it is a scene-linear gain, baked into the DNG's pixels like the look, `docs/09-export.md`). It is the field `white_offset_ev` of the settings object, and the lift in effect is `values.white_ev` of the panel's state |

### Colour

| Option | Values | Notes |
|---|---|---|
| `--wb auto or <K>[,<tint>]` | 2000–11000 K, tint −50…50 | `--wb 5200` without a tint keeps the automatic tint (the semantics of moving the Kelvin slider, `docs/01-architecture.md`) |
| `--light auto or day or shade or tungsten or led or fluorescent` | | light preset instead of the estimated light: day 5500 K, shade 7500 K, tungsten 3200 K; LED and fluorescent at the estimated colour. `auto` clears a preset a settings file chose |
| `--saturation auto or <N>` | −100…100, or `auto` | as `--contrast`. Scales colour about each pixel's luminance (`docs/02-colour-pipeline.md` C9): more makes colours stronger, less makes them greyer, and brightness and neutrals stay |

### Detail

| Option | Values | Notes |
|---|---|---|
| `--nr auto or <L>,<C>,<D>` | 0–100 each | luminance, colour and detail strengths |
| `--sharpen auto or <amount>,<radius>` | amount 0–300, radius 0.3–2.5 px | capture sharpening (R8) |
| `--output-sharpening <0–200>` | percent, default 100 | Detail › Sharpening › Output (`docs/algorithms/04-output-sharpening.md`): sharpens JPEG, TIFF and PNG files at the size they are written, right after the resize. 0 is off and leaves the bytes of an export without the stage. A DNG is never sharpened: `--output-sharpening` and `--sharpen` with `--format dng` are warned about and ignored, because a DNG carries no capture sharpening either (`docs/09-export.md`). It is the field `output_sharpening` of the settings object |

### RAW

| Option | Values | Notes |
|---|---|---|
| `--no-neural` | flag | the classical demosaic instead of the learned one |
| `--no-highlight-recovery` | flag | switch highlight reconstruction off |
| `--no-lens` | flag | switch the lens corrections (vignetting, distortion) off |
| `--no-ca` | flag | switch the chromatic-aberration correction off |
| `--rolloff <highlights>,<low-light>` | 0–1 each, default 1,0.5 | RAW › Range (`docs/02-colour-pipeline.md` C7). Highlights is the **highlight range**: where the camera's clipping point lands, from where it was recorded (0) to white, the clip of an ALEXA at EI 800 (1); mid grey and everything under the knee one stop below the clip never move. Low light is the toe near black. It works alongside the white lift (`--white-offset`) |
| `--profile <id>` | camera profile id, e.g. `sony-ilce-7rm4` | forces a profile, for comparisons |

### Output

| Option | Values | Notes |
|---|---|---|
| `--format jpeg or tiff or png or dng` | `jpg` and `tif` are accepted | default from the preferences, else jpeg. The help text of `dng` is "DNG with the look baked in. Apps other than Adobe's show the stored linear image darker until their own exposure handling is applied. With a look that includes its own display rendering, the DNG is close to, not the same as, the picture you see." The second and third sentences are `ym_export::dng::OTHER_APPS_NOTE` and `ym_export::dng::OWN_DISPLAY_NOTE`, used word for word by the Export panel's DNG paragraph; tests compare both with the constants (`docs/09-export.md`) |
| `--quality <50–100>` | JPEG | default 92 |
| `--bits 8 or 16` | TIFF, PNG | default 16 |
| `--compression none or lzw or zip` | TIFF | default lzw |
| `--colour-space srgb or p3 or rec2020` | alias `--color-space` | default srgb; not for DNG |
| `--long-edge <px>` | 16–65535 | scales so the long edge has this many pixels, never enlarges; not for DNG |
| `--out <dir>` | | default: `<source folder> export` |
| `--name <template>` | `{name}`, `{look}`, `{date}`, `{seq}`, `{camera}` | default `{name}_{look}` |
| `--skip-existing` or `--overwrite` | | default keeps both (" 2" suffix). Both look at the files that were there **before the run**: a name that an earlier photo of the same run has already written is always numbered, whatever is chosen (two folders that both hold `DSC00001.ARW`, or `--name {date}` with no `{seq}`), and the run says so once in a note. The two options exclude each other |

An option that does not apply to the format (`--quality` with PNG, `--long-edge` with DNG) is a warning and is ignored.

### Run

| Option | Values | Notes |
|---|---|---|
| `--jobs <n>` | 1–64 | photos read and analysed at the same time; the export itself renders one photo at a time |
| `--json` | | one JSON object per file on stdout, as each file finishes |
| `--cpu` | | render on the CPU only and open no graphics device. The golden images are made this way (`tools/golden` passes it). Beats `YELLOWMOON_GPU`; with `--require-gpu` it is a usage error (exit 2) |
| `--require-gpu` | | exit 4 if no usable GPU opens at the start, instead of falling back to the CPU; with it, a GPU error on a photo fails that photo (kind `gpu`) instead of re-rendering it on the CPU. Same as `YELLOWMOON_GPU=require` |
| `--apple-auto on or off` | | Apple Auto for this run (`docs/19-apple-auto.md`, below): on a Mac with Apple Intelligence, Apple's on-device model grades each photo, setting its Auto exposure, highlights, shadows, black, contrast, warmth, tint and saturation and the colours of its shadows and highlights. Default: the app's setting (`apple_auto` in the preferences) where Apple Auto is available, off elsewhere. Also on `ym info` |
| `--apple-auto-instructions <file>` | UTF-8 text, at most 2000 characters | the instructions Apple's model reads for this run, instead of those saved in Preferences (`apple_auto_instructions`, `docs/19-apple-auto.md` section 2). The file's final line break is not part of them; an empty file is the default instructions. A file that cannot be read or is too long is a usage error (exit 2). Also on `ym info` |

## Inputs

`-` among the inputs is replaced by the paths on stdin (blank lines and a trailing carriage return are dropped). `--settings -` also reads stdin, and giving both is a usage error.

Folders are walked by the rules of `Session::open`: `*.ARW` in any case, names starting with a dot ignored (macOS leaves `._DSC0001.ARW` on exFAT cards), other files ignored, one level unless `--recursive`. Photos are developed and exported in the order a session lists them (by absolute path, ignoring case), each file once.

A path the user named that cannot be a photo (it does not exist, or is not an ARW file) is reported as an error (`kind` `not_found` or `not_arw`) and counts as a failure, but when no ARW file is left to render at all the exit code is 5.

## Settings

1. The photo starts as the session opens it: automatic development and the effective default look.
2. `--settings` replaces that wholesale. The file is read over the defaults like `DevelopSettings::from_json`, so it may hold only what it changes; unknown fields are an error; out-of-range numbers are limited and a warning names the fields.
3. The options are applied as a patch.

A settings file whose keys are all settings fields (or that is empty) is one document for every photo. One whose keys are all names of ARW files (`DSC00001.ARW`, in any case) is an object keyed by file name: matched exactly, else ignoring case, else by absolute path; a photo without an entry is developed automatically, and an entry that names no photo is a warning. A file that mixes the two is a usage error, and so is a key that is neither (a misspelt field such as `color_ballance` with an object as its value would otherwise pass for a photo's file name and change nothing). All of it is checked before any photo is opened, so a typo costs nothing.

Nothing is saved per photo: the settings live with the photo for this run, and each run starts from the automatic development and the default look.

## Export options and preferences

Only `--format` takes its default from the preferences. The others start from the defaults of `docs/09-export.md` (`--quality` 92, `--bits` 16, `--compression` lzw, `--colour-space` srgb, `{name}_{look}`, keep both), so the command `cli_preview` prints, which leaves those defaults out, does what the panel does whatever the preferences say. The preference "Remove location from exports" applies to the CLI too. The CLI never writes the preferences: `--out` is not remembered as the app's last-used folder, and `--look` does not change the default look (`ym looks default` does).

## `--look` and `--display`

`--look` resolves through the look library (`LookService::resolve`) before anything is opened: a name that matches nothing exits 3, one that matches several exits 2 and lists them, a look whose file is gone exits 3. A look named in a settings file that turns out to be missing fails that photo only (`error.kind` `look`); when every failure of a run is a missing look, the exit code is 3.

`--display aces2` forces ACES 2. `--display arri` uses the ARRI display LUT chosen in the app's preferences, else the only one imported; otherwise it says which exist (exit 2) or that none is imported (exit 3). A full display LUT id is accepted too. Without `--display` the preference applies, and an export with a chosen display LUT that is gone is refused (exit 3), not rendered with ACES 2.

## Running

The core is opened without thumbnails (an export waits behind thumbnail work). Photos are developed in the session's decode pool (`--jobs` sets its size). The CLI waits until every photo is developed (burst medians and noise pools depend on the whole set), then queues one export job over exactly the photos that were developed and given their settings, and prints each file as the queue reports it. A photo that fails does not stop the others.

A closed stdout (`| head -n 1`) stops the output, not the export or the library command: the files and the library are the product (`render`, `info` and every `looks` command write their data lines through a writer that ignores a reader that has gone).

The CLI runs headless through wgpu (Metal on a Mac) with no window or surface. It is safe to run while the app is open: both use the same library folder, and imports take a lock.

## Graphics device

One policy for the app and `ym` (`ym_core::app::gpu`, `docs/17-gpu.md` section 1):

| Policy | Meaning |
|---|---|
| `auto` (default) | a hardware adapter only, never a software one such as Mesa lavapipe or WARP, running the stages in `GpuStages::SIGNED_OFF`, which is every stage |
| `on` | any adapter, software ones too, every stage; for CI and development |
| `off` | never: the CPU reference |
| `require` | like `on`, strict: no silent fallback |

**Precedence: the flag, then `YELLOWMOON_GPU`, then `auto`.** `--cpu` is `off`, `--require-gpu` is `require`, and the two together are a usage error (exit 2). A flag beats the variable whatever it says. The variable takes `auto`, `on`, `off` or `require` in any case (and `1` and `0` for `on` and `off`); anything else is a warning on stderr ("YELLOWMOON_GPU=maybe is not a graphics setting. Use auto, on, off or require.") and `auto`. The app reads the same variable (it has no flags) through the same function.

**Opening the device.** A policy that needs a device (`require`) opens it first: exit 4 and nothing rendered when none opens ("No usable GPU was found (…). --require-gpu was given, so nothing was rendered."; for the variable: "YELLOWMOON_GPU=require is set, so nothing was rendered."). Every other policy opens the device while the photos develop, on its own thread, and gives it to the core (`Core::attach_gpu`) before the export is queued, so that one export is wholly on the device or wholly on the CPU. "No usable GPU was found. Rendering on the CPU, which is slower." is a warning on stderr when `auto` or `on` found no adapter; a software adapter under `auto` is a line of detail (`--verbose`), not a warning.

**What runs on the device.** An export runs the raw front, the raw chain of every tile and the export tail on the device (`GpuStages::USED_BY_EXPORTS` is `RAW_FRONT`, `RAW_CHAIN` and `EXPORT_TAIL`); decoding, the estimates and the encoders stay on the CPU (`docs/17-gpu.md` section 11). Where `auto` finds only a software adapter, an export is the CPU's, byte for byte, and `device` in the JSON is `cpu`. When the stages run on the device, `device` is `gpu` and the file is not the CPU's bytes: the kernels agree with the CPU within the tolerances of `docs/05-raw-processing.md`, not bit for bit, so a photo's pixels differ in the last bits and at the few places where a direction test is a tie. The app and `ym` give the same bytes for the same job on the same machine (tested for JPEG, 16-bit TIFF and PNG, with one photo and several in flight). The golden hashes pin the CPU path (`ym render --cpu`).

**Device errors.** A device error during a photo (a lost device, an out-of-memory that a second try after trimming the buffer pool did not cure, a shader that does not compile) re-renders the whole photo on the CPU and adds a note to the photo ("The graphics card could not render this photo, so the CPU did: …"); a shader error also switches the raw chain off on the device for the rest of the process. Under `require` the same error fails the photo (`error.kind` `gpu`) instead. A cancelled or superseded render never reaches the device twice: its queued device jobs are dropped.

A device that does open leaves the graphics driver to write its own shader cache (Mesa on Linux does, under `~/.cache/mesa_shader_cache`); that file is the driver's. Yellowmoon itself writes nothing but the exports, the preferences and the look library (`ym` passes no pipeline-cache folder).

## JSON output (`--json`)

One object per input file, written when that file finishes, in completion order:

```json
{
  "input": "/Volumes/A1/DSC04204.ARW",
  "output": "/Users/me/Pictures/Lisbon export/DSC04204_neutral.jpg",
  "status": "ok",
  "camera": { "body": "Sony A1", "model_id": 384, "profile": "sony-ilce-7rm4", "profile_version": "1.0.0", "match": "default" },
  "look": { "id": "builtin:neutral", "intensity": 100 },
  "auto": { "exposure_ev": 0.81, "white_ev": 0.77, "kelvin": 5450, "tint": 0, "contrast": 15, "saturation": 0,
            "highlights": -20, "shadows": 0, "black": 0, "balance_shadows": [-0.476, -0.275], "balance_highlights": [0, 0],
            "source": "apple", "apple": { "state": "answered", "...": "…" }, "apple_steps": { "...": "…" },
            "light_class": "daylight", "light_confidence": 0.82, "noise": { "a": 1.9e-5, "b": 3.1e-7 } },
  "auto_source": "apple",
  "apple_auto": { "state": "answered", "reason": null, "model": "AFM 3 Core", "os": "27.0.1 (27A404)",
                  "instructions": "default", "instructions_sha256": "932b96907b63", "prompt": 3, "table": 2,
                  "steps": { "exposure": -1, "highlights": -1, "shadows": null, "blacks": null, "contrast": 1, "white": 0,
                             "tint": null, "saturation": 0, "shadow_color": "teal", "shadow_color_strength": 2,
                             "highlight_color": "none", "highlight_color_strength": null },
                  "scene": "Harbour in morning light", "plan": "Cool, crisp morning with teal shadows" },
  "settings": { "...": "resolved settings, same as ym:Settings" },
  "device": "cpu",
  "timings_ms": { "decode": 310, "develop": 2410, "encode": 180 },
  "pipeline_version": "0.5.1",
  "error": null
}
```

On failure: `"status": "error"`, `"output": null`, `"error": { "kind": "unsupported", "message": "…" }`. `info --json` prints the same object without `output` and `timings_ms`.

**Key order.** Keys come in the order of the example (`input`, `output`, `status`, `camera`, `look`, `auto`, `auto_source`, `apple_auto`, `settings`, `device`, `timings_ms`, `notes`, `pipeline_version`, `error`; `render` writes `apple_auto` only for a photo Apple Auto acted on). The order is kept by the CLI itself, not by a `serde_json` feature: `preserve_order` would reorder the `ym:Settings` inside every file the binary writes, and the CLI must write the app's bytes (`tools/ci/check_serde_order.sh`). Inside the `apple_auto` object and below the top level, keys are sorted.

**Fields.**

- `input` is an absolute path. `status` is `ok`, `skipped` (`--skip-existing` and the file was there: `output` is the existing file, nothing was rendered) or `error`.
- `camera` has `make`, `model`, `model_id`, `body`, `firmware`, `arw_version`, `profile` (the id), `profile_name` (`A7R IV`), `profile_version`, `match` (`exact`, `sensor_family`, `default` or `forced`), `stand_in` and `reduced`.
- `auto` is the `AutoEstimates` of `src/api/types.ts`: the exposure, Kelvin and tint, the Auto contrast and saturation, which Auto made them (`source`) and where the photo stands with Apple Auto (`apple`), light class and confidence, the noise model, and what Auto sharpening and noise reduction resolve to. The exposure, Kelvin, tint, contrast and saturation include Apple Auto's answer.
- `settings` is the document an export records as `ym:Settings`: exposure, contrast, saturation, sharpening and noise reduction that were Auto are written as the numbers they resolved to, white balance stays Auto when it was. Feeding it back with `--settings` reproduces the file byte for byte. The output-sharpening amount is part of it, and a settings file without the field reads as 100.
- `device` is where the pictures of the export were made, `cpu` or `gpu` (`ExportReport.device`, the path the export actually took): `gpu` when any stage of it ran on the device (the raw chain of a tile, the export tail), else `cpu`, also for a photo that a device error sent back to the CPU, whose `notes` then carry the reason. A strict job reads both, so a silent fallback cannot pass. `null` for a file that was skipped; a failure object has none.
- `timings_ms` is `decode` (reading the mosaic, which overlaps the render of the photo before it), `develop` (the render) and `encode`.
- `notes` are the plain-words notes of the render (today: the classical demosaic).
- `info --json` adds the metadata blocks (`exposure`, `dimensions`, `raw`, `black`, `white`, `wb_as_shot`, `levels_source`, `lens_corrections`, `tone_curve`, `preview`, `gps`). For a file that has metadata but cannot be developed yet (ARW 6.0), `look`, `auto`, `auto_source`, `apple_auto` and `settings` are null and `notes` says why.
- A failure that happens after the photo was developed (a look that is gone) still carries `camera`, `look`, `auto` and `settings`; one before (not an ARW file) has `input`, `output`, `status`, `pipeline_version` and `error` only.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | every file rendered |
| 1 | some files failed (others succeeded); also an import in which some files were refused |
| 2 | usage error (also a data folder or home folder that can't be found: set `YELLOWMOON_DATA` or `YELLOWMOON_HOME`) |
| 3 | a requested look is missing |
| 4 | `--require-gpu` (or `YELLOWMOON_GPU=require`) and no usable GPU |
| 5 | no input files found |
| 130 | interrupted with Ctrl-C |

## Output streams and behaviour

- **stdout carries data only:** with `--json` one object per line, otherwise the path of every file written (one per line; a skipped file prints nothing). The lines of data on stdout (the path of every file written, `looks bake`) are the exact path when stdout is a pipe or a file, because the next program needs the real name, and are escaped like the rest when stdout is a terminal (see Text from outside).
- **stderr carries everything for a person:** warnings, per-file errors (not with `--json`, where the object has them), the summary ("Exported 3 photos to … in 4.2 s. 1 photo already existed and was skipped. 2 photos failed."), the progress line and, once, the note that the classical demosaic was used. The progress bar shows when stderr is a TTY and `--json` and `--quiet` are not set; it is redrawn in place, so a file name longer than 32 characters is shortened with an ellipsis to keep the line on one terminal row. `--quiet` leaves errors; `--verbose` adds per-file timings and the `tracing` messages of our crates (`RUST_LOG` overrides the filter).
- **Determinism.** The same inputs, settings and version give the same output bytes. With Apple Auto the version includes the macOS build, the model, the prompt and table versions and the instructions (`docs/19-apple-auto.md` section 7); `--apple-auto off` gives the same bytes on every machine.
- **Ctrl-C, SIGTERM and SIGHUP.** Once photos are being developed (`render`, `info`), imported (`looks import`) or rendered into a contact sheet (`looks bake --preview`), the first signal only asks the command to stop. During an export it asks the export queue to finish the current tile or band, delete the partial file and skip the rest; the CLI waits for that, prints "Interrupted: N of M photos exported. Nothing half-written is left behind." and exits 130. Before that point (reading the list on stdin, opening the library, `doctor`, `looks list`) nothing would look at a flag and nothing is half-written, so the first signal ends the process at once with 130. A second signal always ends the process at once, which can leave the temporary `.yellowmoon-<pid>-<n>.tmp` file of the photo that was being written in the destination folder. A folder created for the export stays (empty) when the first photo was cancelled.

## Text from outside

A file name (a folder unpacked from a downloaded archive) or a tag in a file can hold characters that a terminal obeys: ESC and the other C0 controls, DEL, and the C1 controls U+0080 to U+009F. Every line meant for a person goes through one function (`crates/ym-cli/src/text.rs`) that shows them as escapes: `\x1b` for a C0 control or DEL, `\u{9b}` for a C1 control, `\xNN` for a byte of a path that is not UTF-8. That covers the paths in the progress line, the error and warning lines, `ym info` (stdout), `ym doctor`, `ym looks list` (the name of an imported `.cube` is its file name), `ym looks import` and `bake`; as a last net, `Console` escapes every line it prints to stderr.

In `--json` the values keep the exact characters and are escaped the way JSON does it, with DEL and the C1 controls written as `\u007f`, `\u0085` … `\u009f` (the line stays valid JSON and parses to the same strings).

## `looks`

- **`list`** prints a table with the columns ID, NAME, GROUP, LIBRARY, INPUT and a last one for `default` and `missing: import again`. `--json` prints an object with `looks` and `display_luts`, the shapes of `looks_list` and `looks_display_luts`. LIBRARY is the library the Look tab lists the look under, in its short word: `Built in`, `LogC4`, `LogC3`, `LogC3 to Rec.709`, `ALF2` or `Your LUTs` (`docs/08-looks.md`, Look model, Library). A look in `--json` has `library` (`built_in`, `arri_logc4`, `arri_logc3`, `arri_logc3_to_rec709`, `arri_alf2` or `user`), `format` (`cube`, `alf4`, `alf4c` or `aml`; `null` for a built-in look) and `lut_size` (points per axis; `null` for a built-in look) next to `group` and `input`. The rows are in the core's order (built-in looks, ARRI's by group number and name, then the person's own by name), so ARRI's libraries are mixed there; the Look tab groups them by library.
- **`import`** prints the id of every look it added (stdout) and a summary (stderr).
  - A `.cube` that no known package claims needs `--input` (exit 2 otherwise; the message names the three words). An ARRI Look File does not need it (it says what it expects).
  - A file that cannot be used (damaged, an `.alf4c` with no creative look, an `.aml` made for another display) is refused in plain words, exit 1.
  - In a folder or zip the other files are still imported. Each refused file is named on stderr with its reason ("`<source>`: Torn.alf4 cannot be used. Its checksum does not match, so the file is damaged. Download it again.", also with `--quiet`) and the exit code is 1, so a script can tell that not everything arrived.
  - Notes about what an import did print on stderr too: an `.alf4c`'s display transforms are not used; looks that stayed or became the 3D-LUT version of a library; own LUTs that turned out to be ARRI's files, with their old and new ids, and the default look moved along with its id; files that are the same as a LUT already imported under another name.
  - Importing again never duplicates a look, so a repeat import prints no ids and leaves the looks list untouched. Ctrl-C stops between files.
- **`verify`** prints one line per missing or changed file and exits 1 when there is one.
- **`default <look>`** takes what `--look` takes.
- **`bake`** writes `<name>.cube` (33 points, `TITLE` the look's name) into `--out` (default: the folder of the look file) and, with `--preview`, `<name>_contact.png`: a row per photo, Neutral on the left and the baked look on the right, each 480 px on the long edge, in sRGB. The paths are printed on stdout.

## `doctor`

A text report whose first sentence says what it includes (the version, the GPU, the render path, whether Apple Auto is available, the model weights and their hashes, the camera profiles, how many looks are imported, the folders) and that it leaves out photos, file names, locations and look names. `--json` prints the same facts as one object. Model weights are everything under `data/models/` except the illuminant models of `models/illum/` (listed separately, each with its SHA-256) and the metrics.

- **GPU.** The section reads "none, CPU fallback (reason)" when no device opens. It reports the adapter whatever the policy does with it (a software adapter that `auto` leaves alone, and the one on a machine where `YELLOWMOON_GPU=off`, are listed), then the policy in force and where it came from (`GPU policy`, for example `auto (default): uses the CPU, only a software adapter was found; YELLOWMOON_GPU=on uses it`), the stages `auto` runs and the ones `on` and `require` run (`GPU stages`) and the memory budgets (`Memory budget`: CPU and GPU, lowered for a GPU that shares the memory of the CPU, `docs/01-architecture.md`). `--json` adds `gpu_policy` (`setting`, `source`, `warning`, `uses_gpu`, `reason`, `stages`, `auto_stages`, `all_stages`) and `memory_budget` (`cpu_bytes`, `gpu_bytes`, `shared_memory`).
- **Apple Auto** (`docs/19-apple-auto.md` section 8). The row `Apple Auto` after `Render path` reads `available · Apple Intelligence · AFM 3 Core · macOS 27.0.1 (27A404) · prompt 3 · table 2` (the version string of `docs/19-apple-auto.md` section 7, with ` · edited instructions (<digest>)` when the preference holds the person's own instructions and ` · turned off in Preferences` when the preference is off) or `not available (this is not a Mac)`. `--json` has `apple_auto: { "available", "reason", "model", "os", "preference", "instructions", "instructions_sha256", "prompt", "table" }` (`reason` the wire name, `not_a_mac`, …, or null; `model` and `os` null where it is not available). It is `Core::apple_auto_status` without a session: the advisor `ym` uses (`ym_apple::system_advisor`, read now) and the preference. The report writes nothing.

### `doctor --bench FILE [--repeat N]`

Instead of the diagnostics, times the render path on one photo, on the path the graphics policy picks (`YELLOWMOON_GPU`, else `auto`; `off` is the CPU reference), and prints the result as short lines or, with `--json`, as one object (`"bench": 1`). It reports:

- the compile time of each kernel set on the device (`view`, `raw`, `tail`);
- the decode;
- the views: fit in a 1440 x 960 and a 2880 x 1920 cell, cold and with the frame cache off, the first sharp 100 % frame, a colour-only edit and the thumbnail, each with the wall time of its stages as the renderer reports them and the device that made it;
- the exports through the queue the app and `ym render` share (JPEG, 16-bit TIFF, 16-bit PNG): wall time, render and encode time, size, device;
- the repeat check (`--repeat N`, 1–1000, default 10): the same JPEG at a long edge of 2000 px N times, whose bytes must be equal (`repeat.identical`);
- the peaks (the host ledger, the device pool in use and allocated) and the device counters;
- where the adapter has `TIMESTAMP_QUERY` (Metal, Vulkan on hardware; not a software adapter), the GPU time of each pass label in each phase (`gpu_passes`: label, count, total and longest in ms, from timestamp queries).

It names no file, no folder and no look: the photo is its camera model and size. Apple Auto is off for the bench (the core gets no advisor): it times the render path, and a call to Apple's model is not part of it. A policy that needs a device and finds none is exit 4. `tools/dev/mac-gpu-check.sh` runs it for the Mac session and puts it in one JSON file with the suites.

## Apple Auto on the command line

`docs/19-apple-auto.md` describes Apple Auto itself; this section is what the CLI does with it.

**Wiring.** `render` and `info` hand the core the app's advisor (`ym_apple::system_advisor()`: Apple's on-device model where Apple Auto is available, the reason it is not everywhere else, the test advisor with `YELLOWMOON_TEST_APPLE_AUTO=fake`) and the mode of `--apple-auto` (`AppleAutoMode::FromPreference` without it, so by default `ym` makes the app's picture on the same Mac).

**Not available.** `--apple-auto on` where Apple Auto is not available prints "Apple Auto is not available on this computer (<reason in words>). Auto uses Yellowmoon's own settings." once on stderr (a warning, so `--quiet` hides it) after the core opens, and the run goes on with the same exit code and Yellowmoon's own rule. `render` reads the availability again just before it queues the export, as the app does before each export (`docs/19-apple-auto.md` section 1): when the run wanted Apple Auto (`on`, or the preference on without the option), it was available and has gone since, it prints the same notice then (still once per run) and the export uses the rule.

**Auto crop.** `ym info FILE --suggest-crop[=ASPECT]` (ASPECT `original`, the default, `free`, `1:1` or `square`, `4:5`, `16:9`; any other is a usage error, exit 2) adds the proposal of the Crop panel's Auto (`docs/20-auto-crop.md` §8) for the photo as the camera turned it: a text row **Auto crop**, for example `Levelled by 1.4° · frame 0.0412 0.0310 0.9176 0.9380 (from the camera's focus point)` or `Already level · whole photo kept (from the picture)`, and in `--json` an `auto_crop` object with the app's `suggest_crop` answer field for field (`{"error": "…"}` when the photo was not developed). Without the flag `info` renders nothing. `ym render` has no Auto crop flag: an Auto crop exported by the app is in its settings (`crop.rect`, `crop.angle_deg`), so `--settings` reproduces it. Tests: `tests/info.rs` (the row, the JSON equal to the session's answer, a bad aspect).

**Waiting.** When Apple Auto acts, `ym render` and `ym info` wait for the answers of the photos they develop (`Session::wait_idle` waits for the session's Apple Auto worker too), so the settings, the JSON and the files carry Apple's values. The export of a photo waits for its burst's answers, as the app's does. While every photo is developed and answers are still coming, the progress line (a terminal only) reads `Asking Apple Intelligence [bar] 3/12` instead of `Developing` (`Reading` for `info`).

**JSON.**

- `auto_source` is `rule` or `apple`: which Auto made the photo's automatic values, as `ym:Auto` records it (`apple` also for a photo that failed but takes its burst's steps).
- `apple_auto` is `{ "state", "reason", "model", "os", "instructions", "instructions_sha256", "prompt", "table" }` with `steps`, `scene` and `plan` when the photo was answered.
  - `state` is `answered`, `failed`, `off`, `unavailable` (or `pending`, only for a run that was interrupted).
  - `reason` is the wire name of why Apple Auto is not available (`not_a_mac`, `apple_intelligence_off`, …) or why the request failed (`timeout`, `rate_limited`, …), else null.
  - `model` and `os` name the model that answered or was asked; `instructions` (`default` or `edited`) and `instructions_sha256` (the first 12 hex digits of their SHA-256) name the instructions it read, null when none was.
  - `steps` are the photo's own answer, as `ym:AppleAuto` records them (the steps in effect can be its burst's, `docs/19-apple-auto.md` section 5; Apple's message of a failure is `auto.apple.message`): `exposure` −3…3; `highlights`, `shadows`, `blacks`, `contrast`, `white` (the warmth), `tint`, `saturation` −2…2; `shadow_color` `none`, `teal`, `blue`, `green`, `amber` or `magenta` and `highlight_color` `none`, `warm`, `gold`, `cool` or `pink`, each with a `…_strength` 1…3; `null` for a value the model left empty, which sets nothing.
- `info --json` has `apple_auto` for every developed photo; `render --json` only for a photo Apple Auto acted on (answered, failed or pending).
- In every run, with Apple Auto on, off or unavailable: `auto` has `apple` (`{ "state": "off" }`, or `{ "state": "unavailable", "reason": "not_a_mac" }` and the like), `apple_steps` (null without an answer), `contrast`, `saturation`, `highlights`, `shadows`, `black` (0 without an answer), `balance_shadows`, `balance_highlights` (`[0, 0]`) and `source`; the object has `auto_source`; `settings.contrast`, `.saturation`, `.highlights`, `.shadows` and `.black` are `{ "kind": "manual", "value": 0 }` and `settings.color_balance.shadows` and `.highlights` are `{ "kind": "manual", "point": [0, 0] }`: Auto pinned to the number or point it resolved to, as for exposure (`docs/19-apple-auto.md` section 5). A script must read those as objects, not as numbers or points.

**The text of `ym info`** has a row **Apple Auto** after Auto white (where the Automatic card puts Apple's rows). It says what the steps in effect do, in the order of the table (exposure, highlights, shadows, black, contrast, warmth, tint, saturation, the shadows' colour, the highlights' colour) and only those that were set and change something, then the scene and the plan:

- `Darker -0.3 EV · highlights -20 · contrast +15 · teal shadows (medium) · scene "Harbour in morning light" · plan "Cool, crisp morning with teal shadows"`
- `Brighter +0.3 EV · warmer 10 mired · tint +5 · saturation -15 · scene "…" · plan "…"` (the table's numbers; the white in mired of a warmer or cooler picture)
- `No change · scene "…"` when every step that was set is keep or none
- ` (shared with its burst)` after the changes when the burst's steps are not the photo's own
- `Failed (macOS asked Yellowmoon to wait); Yellowmoon's own Auto` (the failure in words, `docs/19-apple-auto.md` section 8; for a failure of no known kind: `Failed (an error: <message>)`), or `Failed (…); its burst's answer: <changes>` when the burst has steps
- `Off`
- `Not available (Apple Intelligence is off in System Settings)`
- `Waiting for Apple Intelligence` (an interrupted run)

The scene, the plan and Apple's message go through the escaping of text from outside.

**`doctor`** reports the instructions of the preference: `instructions` and `instructions_sha256` in its `apple_auto` object, and ` · edited instructions (<digest>)` in its row when they are the person's own.

**What does not use Apple Auto.** `doctor --bench` and `looks bake --preview` (a contact sheet of the Signature look's development against Neutral, from its own session) give the core no advisor. The golden tool (`tools/golden`: `ym render --cpu --apple-auto off`, and its stand-in binary refuses a render without it), the parity tests (`tests/parity.rs`, `tests/parity_gpu.rs`), the snapshots of `render --json` and `info --json`, the auto-exposure study (`tools/eval/autoexp.py`), the blind demosaic test (`tools/eval/demosaic/blind_look.py`) and the metadata oracle (`tools/oracle/compare_metadata.py`) pass `--apple-auto off`: they are about Yellowmoon's own Auto or the metadata, the same on every machine, and Apple's answers never tune anything of ours (`docs/19-apple-auto.md` section 7). `tools/readme` and `tools/social-card` do not run `ym`.

The tests of this crate run the binary with `YELLOWMOON_TEST_APPLE_AUTO=none` (Apple Auto not available, for the reason a build without the bridge gives on that platform, so they get the rule even on a Mac with Apple Intelligence): `Sandbox::ym`, the `ym()` helpers of `tests/cli.rs` and `tests/info.rs` and every direct spawn of the binary set it (`common::NO_APPLE_AUTO`); they ask for the test advisor where they want it (`Sandbox::ym_fake`). `src-tauri/tests/app_cli_parity.rs` and the golden tool pass `--apple-auto off`.

## Examples

```
ym render ~/Pictures/Lisbon --look neutral --format jpeg --long-edge 3000 --out ~/Exports
fd -e ARW . /Volumes/A1 | ym render - --json --format tiff --bits 16 > results.jsonl
ym looks import ~/Downloads/<ARRI look package>.zip
ym info DSC04204.ARW --json | jq .auto
ym info DSC04204.ARW --json | jq .apple_auto
ym render ~/Pictures/Lisbon --apple-auto off --contrast 10
```

## Install

- The binary ships inside the app bundle as a Tauri sidecar (`bundle.externalBin` is `binaries/ym`, so it is `Contents/MacOS/ym`), so it is signed, shipped and auto-updated with the app. It cannot collide with the app's own binary `Yellowmoon` on a case-insensitive volume.
- The Homebrew cask links it onto PATH (`binary` stanza).
- A `.dmg` install gets the menu item "Install Command-Line Tool…", which creates `/usr/local/bin/ym` (asking for admin rights through macOS) or `~/.local/bin/ym` if the user prefers, as VS Code does for `code` (`docs/12-distribution.md`, Install paths).
- Scope for V2: a CLI-only Homebrew formula; Shortcuts actions through App Intents.

## Tests

- `assert_cmd` and `insta` snapshot tests for `--help` of every command, `info --json` on fixtures, and the exit code of each failure.
- App/CLI parity (`tests/parity.rs`): export through `ym-core`'s app path (`Core::export_start`, what the Tauri commands call) and through the binary; the files must be byte-identical. It covers JPEG, TIFF (8 and 16 bit, none, LZW and ZIP), PNG (8 and 16 bit), DNG, the Display P3 and Rec.2020 spaces, a long edge, a folder with a name template, settings given as options, settings given as a file, and a photo of several export tiles exported with different decode-thread counts. It runs with Apple Auto off on both sides (no advisor in the app's core, `--apple-auto off` for the binary). Its Apple Auto leg gives both sides the test advisor (`FakeAdvisor` in the core, `YELLOWMOON_TEST_APPLE_AUTO=fake` for the binary) and compares a JPEG, a 16-bit TIFF and a DNG of a folder of two photos.
- Apple Auto (`tests/apple_auto.rs`, with the test advisor on every machine): `render --json` (`auto_source`, `apple_auto`, the values in the settings), `--apple-auto off` and `on` (the notice where Apple Auto is not available), the preference, `info --json` and its text row, `doctor` JSON and text, `--contrast`, `--saturation`, `--highlights`, `--shadows` and `--black auto`, the instructions (the preference, `--apple-auto-instructions` on `render` and `info`, an empty file, a final line break, a file too long or missing), `ym:Auto` and `ym:AppleAuto` in the XMP, two renders byte-identical and their settings reproducing the file.
- Hostile text (`tests/hostile_text.rs`) and `doctor --bench` (`tests/doctor_bench.rs`).
- Golden runner (`tools/golden`) calls `ym render --json` on the golden set on every commit and compares ΔE00 to stored references.

Not covered by a test: the progress line and the TTY-only behaviour (a test would need a pseudo-terminal; the unit tests cover the bar), and a missing data folder at the level of the binary (it is covered at the unit level, `core_failure`, because debug builds fall back to the workspace's `data/`, `docs/01-architecture.md` Finding `data/`). The notice before the export, when Apple Auto goes away during a run, is covered by a unit test over a core with a switchable advisor (`runtime::tests`), since the test advisors of the binary do not change while it runs.

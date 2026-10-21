# 08 · Looks, defaults and the ARRI import wizard

This file covers the look model and library, the built-in looks, how ARRI's looks and any other LUT or look file are recognised and imported, the import wizard, the Look tab and the library's storage and import rules. Crate: `ym-looks` (the ARRI Look File reader is `ym_color::alf`). UI: Look tab, the wizard (`design/mockups/import-looks.html`). CLI: `ym looks …`.

## Look model

```rust
pub struct Look {
    pub id: LookId,               // "builtin:neutral", "builtin:signature", "arri:<group>:<slug>", "user:<slug>"
    pub name: String,             // display name
    pub group: LookGroup,         // BuiltIn | Arri { group_no, group_name } | User
    pub input: LookInput,         // LogC4 | LogC3Ei800 | LogC3ToRec709 (includes display rendering)
    pub cdl: Option<Cdl>,         // built-in parametric looks only; a look file's own CDL is read from the file when the look is built
    pub lut: Option<LutRef>,      // path + SHA-256 + format (cube, alf4, alf4c, aml); loaded lazily, cached in memory
    pub source: Option<Source>,   // URL, import date, original file name
}

impl Look {
    pub fn library(&self) -> LookLibrary; // the section it is listed under; worked out, never stored
    pub fn format(&self) -> Option<LookFormat>; // Cube | Alf4 | Alf4c | Aml; None for a built-in look
    pub fn lut_size(&self) -> Option<usize>;    // points per axis (65, 33, ...); None for a built-in look
}
```

**Library.** `LookLibrary` (`crates/ym-looks/src/library/model.rs`) says which of six lists a look belongs to: the Look tab shows one section per library, in this order, and `ym looks list` has a LIBRARY column. It is worked out from the group, the input and the file format by `LookLibrary::classify`, so the app and the CLI cannot disagree (the app gets it as `library` next to `format` and `lut_size`, the CLI in `--json` as well):

| `LookLibrary` | JSON | Section heading (`label()`) | Short word (`qualifier()`) | Which looks |
|---|---|---|---|---|
| `BuiltIn` | `built_in` | Built in | none | Neutral, Signature (group built in) |
| `ArriLogC4` | `arri_logc4` | ARRI looks · LogC4 | LogC4 | ARRI group, input LogC4: the 65-point `.cube` library, or its `.alf4` look files (also an `.alf4c`) |
| `ArriLogC3` | `arri_logc3` | ARRI looks · LogC3 | LogC3 | ARRI group, input LogC3 EI 800: the log-to-log `.cube` library (ids end `-logc3`) |
| `ArriLogC3ToRec709` | `arri_logc3_to_rec709` | ARRI looks · LogC3 to Rec.709 | LogC3 to Rec.709 | ARRI group, input LogC3 to Rec.709, not an `.aml`: the `.cube` library (ids end `-logc3-rec709`) |
| `ArriAlf2` | `arri_alf2` | ARRI looks · LogC3 to Rec.709 · ALF2 | ALF2 | ARRI group, input LogC3 to Rec.709, format `aml`: the 261 look files, three strengths of each look |
| `User` | `user` | Your LUTs | Your LUTs | user group, any input, any format (`.cube`, `.alf4`, `.alf4c`, `.aml`) |

The short word is what tells two looks of one title apart where the section is not visible (the look chip, the default look choices: `docs/11-ui.md` §Look tab sections). `format` and `lut_size` come with the look too (the tooltip of a tile says "3D LUT, 65 points" or "ALF4 look file, 33 points"). The person's own `.aml` is in Your LUTs, not in ARRI's ALF2 section: that one is for ARRI's own library, recognised by the package. Tests: `crates/ym-looks/tests/libraries.rs` (the five downloads and the person's own files, imported), the unit tests of every group, input and format combination in `model.rs`.

Application maths is in `docs/02-colour-pipeline.md` (C10). Intensity is 0–100 % per photo, default 100.

## Built-in looks (shipped, Apache-2.0)

- **Signature** is the default look out of the box, shipped as `data/looks/builtin/signature.json`. It is a LogC4 luma tone curve through 15 points, then ASC CDL saturation 1.45 with Rec.709 luma weights on the LogC4 codes:
  - The curve is applied as `v += T(L) - L` on all three channels, `L` the LogC4 code of the AWG4 luminance of the pixel (the way the Curve tab's luma curve works, `docs/02` C11), through the monotone cubic of `ym_color::curve::Curve` (Fritsch-Carlson tangents). The slope at mid grey is about 1.40.
  - **Mid grey is unchanged** (LogC4 0.2784 for scene 0.18, moved by less than 1e-6 in code value), neutrals stay neutral, and the tone curve rises everywhere.
  - Against Neutral it darkens the shadows (about -0.75 stop from -3 to -6 stops, -0.62 at -2), lifts the lights (+0.36 stop at +1 stop, +0.66 at +2, +0.83 at +3) and eases the highest lights back (+0.78 at +6).
  - The constants were chosen by eye and never fitted to any LUT, ARRI's or a commercial one. On 49 photos the default render's median chroma (CIELAB C*, 512 px) is 10.5 against 8.9 for Neutral and 9.8 for the camera JPEGs, and the median standard deviation of L* is 25.9 against 24.5 for Neutral (camera 24.3), without more clipping.

  The loader bakes it into a 65³ cube like every parametric look. The cube follows the exact look to under one 8-bit display code (worst 0.47 code over 20 000 random colours, 0.18 on a neutral ramp, 0.64 on a hue sweep; `crates/ym-color/tests/look_signature.rs`). The tuning tool is `ym looks bake signature.json --preview <ARW>…`, which writes a 33-point `.cube` and renders a contact sheet against Neutral (`docs/10-cli.md`). The same cube runs on the GPU through the existing look path with no kernel of its own: `crates/ym-gpu/tests/gpu_signature_look.rs` compares it with the CPU chain at 0, 60 and 100 % on three outputs (worst |GPU − CPU| 1.8e-7 in LogC4 after the look, gate 2e-4, and 5.3e-6 in display code, gate one 10-bit code, on lavapipe), and intensity 0 is the chain without a look bit for bit on both.
- **Neutral** is the identity: emulated ALEXA colour through our display transform, no creative look. It stays in the list as the "no look" choice and as the look a missing default falls back to.

**Parametric look files (M1-10).** `data/looks/builtin/<name>.json`, loaded by `ym_color::look::ParametricLook::from_json` and baked into a 65³ cube (`bake`, `to_transform`), so the CPU and the GPU read every parametric look as an ordinary cube and need no kernel of their own for it. Schema `yellowmoon.parametric-look/1`; every field except `schema` and `name` is optional, unknown fields are rejected, and `{"schema": "yellowmoon.parametric-look/1", "name": "Neutral"}` is the identity:

```json
{
  "schema": "yellowmoon.parametric-look/1",
  "name": "Example",
  "input": "logc4",
  "luma_curve": [ { "x": 0, "y": 0 }, { "x": 0.2784, "y": 0.2784 }, { "x": 0.6, "y": 0.63 }, { "x": 1, "y": 1 } ],
  "cdl": { "slope": [1, 1, 1], "offset": [0, 0, 0], "power": [1, 1, 1], "saturation": 1 },
  "hue_vs_sat":  [ { "hue_deg": 30, "value": 1.15 }, { "hue_deg": 210, "value": 0.9 } ],
  "hue_vs_hue":  [ { "hue_deg": 60, "value": -6 } ],
  "luma_vs_sat": [ { "luma": 0.15, "value": 0.8 }, { "luma": 0.6, "value": 1.1 } ]
}
```

`luma_curve` is a tone curve on the luminance of the pixel: 2 to 16 points `{ "x", "y" }` in [0, 1]² with `x` strictly increasing, where `x` is the LogC4 code value of the AWG4 luminance (`LogC4::luma`, the `L` of the Curve tab's luma curve) and `y` what it becomes. It is applied as `v += T(L) - L` on all three channels, so it moves tone and leaves the differences between the channels in LogC4 (hue, chroma) alone. `T` is the monotone cubic of `ym_color::curve::Curve`; it holds its end values between 0 and the first point and between the last point and 1, and continues with slope 1 below 0 and above 1. Points on `y = x` from 0 to 1 are the identity. Its `x` is not the `luma` of `luma_vs_sat`, which is the Rec.709-weighted luma of the codes; the two are never mixed.

`hue_vs_sat` and `luma_vs_sat` values multiply the chroma (0 to 8); `hue_vs_hue` values are degrees (±180, positive turns red towards yellow). Hue is in degrees on the LogC4 hue circle (0 red, 60 yellow, 120 green, 180 cyan, 240 blue, 300 magenta), `luma` is the Rec.709-weighted LogC4 luma in [0, 1]. At most 64 points per curve; the curves are monotone (Fritsch–Carlson) splines, periodic for hue, so they never overshoot the values given. The order is `luma_curve`, then the CDL, then the hue and luma-vs-saturation curves, which see the CDL output and keep its luma. Full details: the module documentation of `crates/ym-color/src/look.rs`.

## Library folder

```
~/Library/Application Support/app.yellowmoon/looks/
  manifest.json
  arri/<group_no>-<group_slug>/<file>.cube       (or .alf4, .aml: the look's own file, as it came)
  arri-display/<file>.cube
  user/<file>.cube                               (or .alf4, .alf4c, .aml)
```

`manifest.json`:

```json
{
  "version": 1,
  "looks": [
    { "id": "arri:3000:<slug>", "name": "…", "group": { "kind": "arri", "no": 3000, "name": "Film" },
      "input": "logc4", "file": "arri/3000-film/<file>.cube", "sha256": "…", "lut_size": 65,
      "source": { "package": "look-library-logc4", "url": "https://www.arri.com/…", "imported": "<UTC time, RFC 3339>", "original_name": "…" } }
  ],
  "display_luts": [ { "id": "arri-display:logc4:rec709-g24", "input": "logc4", "target": "rec709-g24", "file": "arri-display/…", "sha256": "…" } ]
}
```

A look file's entry has `"format": "alf4"` (or `"alf4c"`, `"aml"`) and its `file` ends in that extension. An entry with no `format` is a `.cube`, and a `.cube` entry is written without the key, so a library of cubes has none. The manifest version stays 1. A user LUT's group is `{ "kind": "user" }`, and `source.package` and `source.url` are left out for files that did not come from a known package.

The default look and the display rendering choice are preferences (`preferences.json`, `docs/01`), not part of the manifest.

Writes are atomic (write temp, fsync, rename) and take `looks/.lock` so the app and CLI can import at the same time safely. `looks verify` re-hashes every file and reports changes. A damaged `manifest.json` does not stop the app: built-in looks keep working, the problem is reported, and the next import moves the file aside as `manifest.json.damaged-<n>` and starts a new list. A manifest written by a newer version (`"version"` above 1) is never overwritten.

## What users can import

ARRI's Look Files page (https://www.arri.com/en/learn/camera-systems/image-science/look-files) offers five downloads. The app imports all five: the three **3D-LUT** packages as `.cube` files and the two camera **look-file** packages as ARRI Look Files (section *ARRI Look Files* below). The two look-file packages hold the same looks as two of the 3D-LUT packages and are the same wizard package (the Choose step lists five packages). Two display LUT packages come from ARRI's Technical Downloads page.

| ARRI's title for the download (release) | Size | Contents | Role in the app |
|---|---|---|---|
| Look Library for LogC4 (3D-LUTs), log-to-log (2025-07-30) | 190 MB | 87 `.cube`, 65-point | recommended looks |
| ARRI Look Library LogC3 (3D-LUTs), log-to-log (2024-11-28) | 31 MB | 87 `.cube`, 33-point (the page states no LUT size; the files have 33) | looks run in LogC3 |
| ARRI Look Library LogC3 (to Rec709) 3D-LUTs (2024-03-12) | 34 MB | 87 `.cube`, 33 mesh points (stated on the page) | complete rendering; replaces our display transform |
| Look Library for LogC4 (ALF4 Look Files) (2025) | 16.8 MB (16,823,854 bytes) | 87 `.alf4`, 33-point, in nine group folders | the same looks as the LogC4 3D-LUT package, 11 times smaller and less exact (33 points against 65) |
| ARRI Look Library LogC3 (Look Files) (2023-02-10) | 38.7 MB (38,709,051 bytes) | 261 `.aml`, 33-point: each look in three strengths (Low, Medium, High) | complete rendering like the LogC3 to Rec709 package; the High strength is the one closest to that package's cubes |
| LogC4 LUT Package, display LUTs (Technical Downloads page) | 32 MB | 22 `.cube` (Rec.709, P3, Rec.2020, HLG, PQ); names from ARRI's LUT naming guideline, `ARRI_LogC4-to-Gamma24_Rec709-D65_v1_65.cube`; the count can change | ARRI's exact display rendering |
| LogC3 LUT Package, display LUTs (Technical Downloads page) | 46 MB | 48 `.cube`, including the classic Rec.709 | classic ALEXA rendering |

The dates are the releases ARRI states for the packages the fingerprints were taken from. The two display LUT packages are recognised by names alone. The page states no download sizes, so sizes are hints (`download_mb`), never a rule.

**The 3D-LUT zips.** Each holds ONE top folder named for the package (`ARRI Look Library for LogC4 (3D-LUTs)/`, `ARRI Look Library LogC3 (log-to-log)/`, `ARRI Look Library LogC3 to Rec709 3D-LUTs/`) with 87 `.cube` files directly in it: **no group folders** and **no encoding word in the file names** (`LogC4` is only in the zip and folder names). A file is called `<look code> <title>.cube` (`1110 Black and White.cube`, `1120 Black and White - Night sky.cube`; with hyphens in the Rec709 package, `3250-Cross.cube`, `8120-20s.cube`). The code has four digits and its first digit is the group, so the 87 looks are in nine groups: 1000 Black and White, 2000 Application, 3000 Film, 4000 Tinted, 5000 Environment, 6000 Season, 7000 Contrast, 8000 Period, 9000 Special (names from the folders of the camera look-file zips). Several looks share a title (`1110` and `1111` are both "Black and White"). The zips are named `ARRI Look Library LogC4 (log-to-log) 3D-LUTs_data.zip`, `ARRI Look Library LogC3 (log-to-log) 3D LUTs_data.zip` and `ARRI Look Library LogC3 (to Rec709) 3D-LUTs_data.zip`; the Rec709 one also holds a `__MACOSX` tree of `._` files called `.cube`, which are not LUTs. Header comments are generic (`# 3 columns rgb, 274625 rows`); none names the encoding.

**The look-file zips.** `arri-look-library-logc4-log-to-log-alf4-data.zip` holds one folder, `ARRI Look Library LogC4 (log-to-log) ALF4/`, with nine group folders (`1000 Black-and-White` … `9000 Special`) and 87 files called `LL-1120-Black-and-White-Night-sky.alf4`. `ARRI Look Library LogC3 (log-to-Rec709) ALF2_data.zip` holds `ARRI Look Library LogC3 to Rec709 ALF2/` with the same groups and 261 files called `LL-1110-Black-and-White-Low.aml` (`-Low`, `-Medium`, `-High`; one file is misnamed `LL-2130-Beauty-How.aml`). Both carry a `__MACOSX` tree and `.DS_Store` files. They are binary camera files, not LUTs; Yellowmoon reads them (section *ARRI Look Files*).

All are free public downloads with no login and no licence text. The Look Library has 87 looks in nine groups (group 3000 is "Film").

ARRI files are never bundled, hosted, downloaded automatically or committed. arri.canto.de's robots.txt disallows automated access, so the app opens ARRI's page in the browser and the user downloads.

`data/looks/arri-packages.json` ships in the bundle (patterns, group names and fingerprints: names, sizes and hashes only, never LUT data).

## Recognition rules

Recognition is data-driven: `data/looks/arri-packages.json` lists, per package, the expected folder and file-name patterns, LUT sizes, counts, the names of ARRI's look groups and the SHA-256 of every file of the five fingerprinted packages (the three 3D-LUT packages and the two look-file packages; the two display LUT packages have none). It contains no LUT data.

**Fingerprints (M3-H2).** They were taken with `tools/looks/fingerprint.py` from ARRI's public downloads (the real zips, under the names ARRI's site gives them): names, sizes, LUT sizes and SHA-256 per file in `arri-packages.json`. The two display LUT packages have none; they are recognised by names alone.

Rules, in order:

1. Unzip in memory (or stream) and list the look files: `.cube` files and ARRI Look Files (`.alf4`, `.alf4c`, `.aml`); ignore `__MACOSX` (any case), `._` files, `.DS_Store`, `.xml` (ARRI's older look files, which are not read), PDFs. (There is no `.alf2` file: ALF-2 files are `.aml`.)
2. Match file hashes against `arri-packages.json`: an exact match identifies the file, with its group and name taken from the fingerprinted path (whatever the zip, folder or file is called now).
3. Otherwise use names and headers, per file: `LogC4` vs `LogC3` in the names of the zip or folder, the folders the file sits in, the file name or its `TITLE`; "to" followed by a display name as a whole word (`Gamma24`, `Rec709`, `P3`, `Rec2020`, `HLG`, `PQ`, `sRGB`, on the target side of `LogC4-to-Gamma24_Rec709-D65_v1_65`) means a display LUT or, with a group, a Rec.709-baked look (`log-to-log` and `LogC4-to-LogC4` name no display; `Pqrs`, `P3ach` or a look called "Warm P3 Glow" are not display names); `LUT_3D_SIZE` 65 vs 33 when the package lists a size; a group is a four-digit group folder **or a four-digit look code at the start of the file name** (`1110 Black and White`, `8120-20s`; the first digit is the group, the table in *What users can import*).
4. **The package as a whole.** When the name of the zip or folder, or the folder every file sits in, is a package's (`zip_name` patterns, with the encoding words), and at least half of the files that package should have are `.cube` files with an allowed LUT size, those files are that package whatever their layout (flat, one top folder, one folder per look, named groups) and whatever words their names hold; the name wins over what the file names suggested, except for files a hash identified. Layout (group folders, display names) only breaks a tie between two packages the name fits equally; the file count closest to what the package should have comes first. Counts are expected values: more files than expected are imported with a note, fewer (down to half) are an incomplete download with the numbers. Download sizes decide nothing. Looks in a flat list are grouped by their look code, else by the folder that holds several of them, else "Ungrouped"; two looks that share a title keep their code in the name (`Black and White 1110`). A file that a hash identified keeps it whether the whole package or one loose file of it is imported: its title is counted over every file the package lists, not over the files that are in the source, so one look has one id (`arri:1000:black-and-white-1111`) however it arrives, and the same look in its other form (the `.cube` and the look file) is found, not added a second time. Files recognised by their names alone have only the source to go by.
5. A `.cube` file that matches nothing gets the generic question (below). A look file that matches nothing needs no question: it is the person's own look (section *ARRI Look Files*).
6. A wrong package gets a plain explanation: the NOT CHOSEN row says what it is and which downloads to take, by ARRI's own titles (`arri_title` in `arri-packages.json`; a library that comes both ways names both: "Download 'Look Library for LogC4 (3D-LUTs)' (or 'Look Library for LogC4 (ALF4 Look Files)') from ARRI's page …"). The small "LogC4 WVO LUT Package" is not a look or display package and is not recognised as one.

**Per format.** A package holds files of one kind (`format` in `arri-packages.json`: `cube`, the default, `alf4` or `aml`). A look file is claimed by a package only by its hash (rule 2) or by the package rule (rule 4): the name of the zip or folder (or the folder every file sits in) must be the package's, and at least half of its files must be look files of its kind with its LUT size. The zip-name patterns keep the four libraries apart (`alf4`, `alf2` and `lookfiles` are in the 3D-LUT patterns' `none_of`, and the look-file patterns ask for them).

**Details.** The evidence for an encoding is the source's own name (the zip or folder), the file's path and its `TITLE`, with separators and case ignored; `LogC4` and `LogC3` together are a conflict and count as no evidence. A file in a four-digit group folder, or whose name starts with a four-digit look code (`1110 Black and White`), is a look; a file with `-to-` plus a display name and no group folder is a display LUT; both together (LogC3 only) is a Rec.709-baked look. The display name may stand in the file name, in a folder name, in the `TITLE` or in the name of the zip (a Rec.709-baked look is display-referred, so it must not be taken for a log-to-log look just because its file names do not repeat what the zip says). Paths are relative to the chosen zip or folder; the chosen folder's own name is evidence but never a group. A group folder starts with exactly four digits and the number does not run on into more digits (a folder named like a date or a version, such as a year with a month and a day, `2024_05` or `2024.1`, is not a group). Identical LUTs published under two names count as two files of the package. The LUT size must agree with the package when the package lists sizes. Whatever fits exactly one package is that package's (and rule 4 reads a whole zip or folder as one package when its name says so); everything else gets the generic question, with a suggestion when the names or the size hint at one. Files that name ARRI or ALEXA but fit none of the five packages are explained as "from a package Yellowmoon does not know" and may be imported as the user's own LUTs. A package with fewer files than it should have is reported `Incomplete` with the missing file names; the wizard blocks "Next" on it, the library only refuses it when asked to (`require_complete`).

**`arri-packages.json`** carries, per package: id, wizard title, `arri_title` (ARRI's name for the download, for the words that say what to download), role (`looks` or `display_luts`), `format` (`cube`, the default, `alf4` or `aml`; `expected.cube_files` counts the files of that kind), input, download size, zip-name patterns (for the Downloads watcher), expected counts and LUT sizes, name markers (encoding words, whether a display rendering and group folders are required, forbidden or either) and the `fingerprints` that `tools/looks/fingerprint.py` fills. The patterns work on all five real downloads (rules 3 and 4), so the five look packages are identified by hash and, by names alone, in every layout. `crates/ym-looks/tests/real_downloads.rs` (ignored; `YM_ARRI_DOWNLOADS=<folder with logc4.zip, logc3.zip, logc3-rec709.zip, alf4.zip, alf2.zip>`; `YM_ALF_SAMPLES` for the files from ARRI's look builder) checks the zips, the unzipped folders, the look-file zips and a real import of each, by hash and by names, and the gates of *ARRI Look Files*. The CI tests rebuild the real layouts from made-up looks (`tests/package_layouts.rs`).

## Wizard (five steps, `design/mockups/import-looks.html`)

1. **Choose.** Look Library (LogC4) checked; Display LUTs (LogC4) optional; "Show 3 older LogC3 packages" reveals the rest. One line: ARRI publishes these looks for free; Yellowmoon can't include them, so you download them from ARRI and the app does the rest. Shows the download total.
2. **Get the files.** "Open ARRI's page" opens https://www.arri.com/en/learn/camera-systems/image-science/look-files in the system browser. Then the wizard waits for the zip:
   - "Watch Downloads" switch (asks macOS for access to ~/Downloads the first time; remembered as a preference). The watcher looks for new `.zip` files whose names match the patterns and waits until the file size is stable for 2 s (see *Downloads watcher*).
   - "Choose file…" and drag-and-drop of a zip or folder work at any time.
   - Next unlocks when every chosen package has arrived.
3. **Recognise.** Show each package found, with a tick or an explanation. Mismatches are listed in plain words. **A chosen package that nothing matches never locks Import**, so a download the rules do not recognise cannot leave a person stuck. Import is unavailable only (a) while a file is still being checked, (b) while a set of unknown `.cube` files has no answer to "what does this LUT expect as input?" (the answer starts from what the names suggest, so a person who agrees only presses Import), and (c) when there is nothing to import; the footer then says what is missing ("Look Library · LogC4 is missing. Go back and add it."). When something can be imported but a chosen package is not here, the footer says so and Import stays on: "Look Library · LogC4 isn't here. You can import the rest now and add it later."
   - An answered set of unknown files (a zip or folder, not one loose file) counts for the chosen looks package that has the same input (answered LogC4: Look Library · LogC4), so no note appears for it; its files are imported as the person's own LUTs ("Your LUTs") with that input. Display LUT packages can't be answered for.
   - ARRI Look Files ask nothing: a library as look files is a MATCH for the package it is the looks of (the row says "87 .alf4 · 33-point"), and the person's own look files are a READY row, "ARRI Look Files · LogC4", counted for the chosen looks package that has the same input when they are a zip or folder (not one loose file) whose files all expect it.
4. **Import.** Copy only the look files (`.cube`, `.alf4`, `.alf4c`, `.aml`) into the library folder, grouped as ARRI groups them; hash each; write the manifest. Progress bar per file count. Takes seconds. The screen says "Only the .cube files" when only `.cube` files are copied, "Only the look files" when only look files are, and "Only the .cube files and the look files" when both are: a folder of ARRI's cubes can hold look files of the person's own, which are imported too, and two sources can be of different kinds. The recognition says how many look files a package or a question row also holds, `look_file_count`.
5. **Finish.** Thumbnails of the imported looks on the current photo; choose the default look; choose the display rendering ("Ours · ACES 2" or "ARRI's" when display LUTs were imported); offer to move the zip to the Trash (`trash` crate, recoverable) or keep it. The default choices list Neutral, Signature and the first imported look, and select the one the library marks default.

Re-importing the same package is idempotent: same hashes, nothing duplicated, and the list is not even rewritten. **Importing again never duplicates a look**: the same bytes are never imported twice, whatever the file is called and whichever group it sits in (*Import protocol*).

**Downloads watcher.** It polls the folder every 500 ms, reports a matching `.zip` once its size and modification time have stayed the same for 2 s, ignores zips that were already there when watching started, and reports a changed file again. A failed listing (access withdrawn, volume unmounted) keeps what the watcher knew, so old downloads are not reported as new when the folder comes back. The macOS permission prompt for `~/Downloads` appears at the first listing, so `start` returns that error.

## Any other `.cube`

Same importer plus one question: what does this LUT expect as input?

- LogC4 (ARRI Wide Gamut 4),
- LogC3 (ARRI Wide Gamut 3, EI 800),
- LogC3 with Rec.709 output baked in (replaces the display rendering).

Validation: the LUT must parse, be 3D (1D shaper allowed), size 2–129, all values finite. Rejected with a reason otherwise.

## ARRI Look Files (`.alf4`, `.alf4c`, `.aml`)

Yellowmoon reads ARRI's camera look files. The reader is `ym_color::alf` (`crates/ym-color/src/alf.rs`: the byte layouts are in its module documentation); the importer, the manifest and the wizard are `ym-looks`. No ARRI file is in the repository; the tests use the free Look Library downloads and files made with ARRI's look builder from our own LUTs (`docs/14`).

| File | Container | What it holds | What Yellowmoon makes of it |
|---|---|---|---|
| `.alf4` (ALEXA 35) | `3FLA`, version 3 | an optional ASC CDL, then one LogC4 to LogC4 3D LUT (the CMT), at any size from 2 to 129 (ARRI's library: 33; ARRI's tool keeps 17, 33 and 65) | a `LogC4` look: CDL on the LogC4 codes, then the LUT |
| `.alf4c` (custom colour management) | `4FLA`, version 5 | a CDL and up to three LUTs: the CMT and two display transforms (SDR, HDR), perhaps design data | the CMT and the CDL as a `LogC4` look; the display transforms and the design data are not used |
| `.aml` (ALF-2, LogC3 cameras) | `KOLA`, version 3 | an optional CDL, then one 33-point LUT from LogC3 to Rec.709 gamma 2.4, video rendering included; plus ARRI's video look parameters | a `LogC3ToRec709` look (it replaces our display transform), the CDL in front of the LUT |

ALF-1 (`.xml`, ALEXA Classic and XT: printer lights and an EI-specific tone map) is not read; a source that holds only `.xml` files says so ("Yellowmoon cannot read ARRI's older look files (ALF-1 …)").

**What is checked.** The marker and the version (3 for `.alf4` and `.aml`, 5 for `.alf4c`), the size field, the CRC-32 (zlib/IEEE, over everything after it), that every offset and length of every table, string and LUT lies inside the file, that the LUT is 2 to 129 points per side with a code scale between 1 and 65535, and caps on the counts and string lengths (at most 4096 parameters, 64 MiB per file). Parameters are looked up **by id**: their order differs between files, and unknown ids are ignored. Nothing in the file is trusted for a size or an offset before it has been checked, and a damaged file is never half-read: the reader never panics (a mutation test in `crates/ym-color/tests/alf.rs`, 3,000 random damages per container (9,000 in all) plus every cut-off length of each container, and the `alf_parse` fuzz target of `crates/ym-color/fuzz`).

**Refused, in plain words** (`AlfError`, shown as the file's reason; the rest of a folder or zip still imports, and the refused file is named, with its reason, on the source's row of the Recognise step and, in `ym looks import`, on stderr with exit 1, so a folder never looks as if all of it arrived):

| Case | The sentence |
|---|---|
| not an ARRI Look File | "It is not an ARRI Look File. A .alf4, .alf4c or .aml file starts with a marker this file does not have." |
| another version | "It is version 9 of the .alf4 format. Yellowmoon reads version 3. Update Yellowmoon, or export the look again." |
| cut short, changed or damaged | "It is cut short or damaged: its size does not match. Download it again." / "Its checksum does not match, so the file is damaged. Download it again." / "It is damaged: a table points outside the file. Download it again." |
| LUT size, scale | "Its LUT has 300 points per side. Yellowmoon reads 2 to 129." / "Its LUT uses a value scale of 0, which Yellowmoon does not read." (a scale of 0 or above 65535) |
| colour correction not a number, or out of range | "Its colour correction (CDL) holds a value that is not a number, so the file is damaged." / "… is outside the allowed ranges. Slope and saturation cannot be negative and power must be above zero." |
| `.alf4c` with no CMT | "It holds only display transforms and no creative look (CMT), so there is nothing to import." |
| `.alf4` or `.alf4c` look for another colour space | "Its look is made for another colour space (…). Yellowmoon can use ARRI Look Files made for LogC4." |
| `.aml` made for another display | "It is made for a Rec.2020 PQ display. Yellowmoon can use ARRI Look Files made for Rec.709." (Rec.2020, Rec.2020 HLG, P3 and unknown targets likewise) |
| `.aml` with no LUT (only ARRI's look parameters) | "It has no LUT, only ARRI's look parameters, so there is nothing to import." |

A file over the size limit (64 MiB) is refused before it is read, with the importer's own sentence ("It is larger than the 64 MB limit for a .alf4 file."); `AlfError::TooLarge` is the parser's guard for the same case and is not the sentence a person reads.

The colour space of an `.alf4` or `.alf4c` look must be `com.arri.colorspace.awg4.d65.logcv4` (LogC4 / AWG4 in and out) and that of an `.aml` `com.apple.proapps.colorspace.rec1886` (Rec.709 with BT.1886, gamma 2.4); ids are compared without regard to case. ARRI's guide says an `.alf4` is always LogC4 to LogC4; the colour-space rule only protects against a file that says otherwise.

**The CDL.** Slope is parameter ids 3011, 3012, 3013 (R, G, B), offset 3001, 3002, 3003, power 3021, 3022, 3023 and saturation 3040. The CDL is applied by `ym_color::cdl::Cdl` on the look's own codes (LogC4 or LogC3) **before** the LUT, as ARRI says ("LogC4 → ASC CDL → CMT"), so a look file is `LookTransform::new(input, cdl, lut)` like every look. A missing id is the identity value; a CDL that is the identity is dropped. The ids are the structure's reading (the table order slope, offset, power; defaults 1, 0, 1): every ARRI file seen carries the identity CDL, so the order within a triple and slope against power are not confirmed by a file with distinct values.

**Look Intensity** (parameter 10006, 1.0 in every file seen) is **not read**: a look file opens at 100 % like every look, and the Intensity slider is per photo. ARRI's camera blends its looks with the plain LogC4 image the same way (`v' = in + k (out - in)`), so the slider means what ARRI's does, except that a CDL is not scaled by it. **ARRI's video look parameters** in an `.aml` are ignored: a LUT is always present and the parameters sit at their defaults then.

**Stored as they came.** Import keeps the file byte for byte (the SHA-256 in the manifest is the file's, `looks verify` and loading check it like a `.cube`) and the library reads it again when the look is built (`LutRef::load_look`): the CDL and the LUT are the file's, never a converted copy. A file is kept under the kind its bytes are (an `.aml` that was renamed `.alf4` is stored as `.aml`). The LUT is read at its own size and transposed from the file's blue-fastest order to ours (`Lut3d` runs red fastest). The GPU takes the same CDL and cube as for any look, and `crates/ym-gpu/tests/gpu_alf_look.rs` checks it against the CPU on Mesa lavapipe in every storage mode: a 33-point `.alf4` with a CDL, an `.alf4c` and an `.aml`, at 25, 60 and 100 % and three outputs, worst |GPU − CPU| 2.4e-7 in LogC4 after the look (gate 2e-4) and 4.8e-6 in display code (gate one 10-bit code, 9.8e-4).

**Names, ids and groups.** The file's name field is a *note*, not a look name (ARRI's library notes read "1120 Black and White - Night sky from ARRI_S35_4k_LookLibrary_v02_211215", files from ARRI's look builder have none), so the look's name is the file's name: `LL-1120-Black-and-White-Night-sky.alf4` is look code 1120, title "Black and White Night sky" (the `LL-` and the code are not part of the title), in the group of the first digit of the code, "Black and White", named as `group_names` in `arri-packages.json` names it. **The ids are the 3D-LUT library's:** `arri:1000:black-and-white-night-sky` is the same look whether it came as `.cube` or `.alf4` (87 of 87 ids agree on the real downloads), so a script's `--look` works with either, and a look is one entry. An `.aml` looks like `LL-1110-Black-and-White-Low.aml` and the strength is part of the look: "Black and White · Low", id `arri:1000:black-and-white-low-logc3-rec709` (`-low`, `-medium`, `-high`, then the LogC3 to Rec.709 suffix), so the three strengths and the 3D-LUT package (`arri:1000:black-and-white-logc3-rec709`) never collide; two looks that share a title keep their code ("Black and White 1110 · Low"). The strength word is read only when it is one of the three, as a whole word at the end of the name; ARRI's one misnamed file `LL-2130-Beauty-How.aml` is simply "Beauty How".

**One look, one entry.** The LogC4 library comes as 3D LUTs (65 points) and as `.alf4` files (33 points, ARRI's own input to its camera) with the same ids. Importing both would double the list, so **the `.cube` wins, whichever came first.**

- A look file never replaces a `.cube` that an earlier import put there: the look stays as it is, and the import reports it as already imported, with the note "87 looks were already imported from the 3D-LUT version, which is kept: it has more points than the look file."
- A `.cube` replaces the look file of an earlier import in place: the entry takes the `.cube`, the look file is deleted once the list is saved, and the note reads "87 looks were replaced by the 3D-LUT version, which has more points than the look file."
- The rule is for ARRI's library looks only: the person's own files of the same name stay separate (`user:mine`, `user:mine-2`, …), and the three strengths of an `.aml` are not the LogC3 to Rec.709 cubes (different ids).
- It holds **inside one import too**: when one source holds both (a folder with the two unzipped libraries), the importer plans ARRI's `.cube` files before its look files, so the look file meets the `.cube` and is left out the same way, reported as already imported with "87 looks came both as 3D LUTs and as look files. The 3D LUTs are kept: they have more points." (The wizard imports each chosen source in turn, so for a zip and a zip it is the earlier-import case above.) A folder with both libraries is read file by file by the hashes of the shipped list; with names alone no folder holds all the files, and the look files would be taken for the person's own. Numbers are for the person's own files and for two different ARRI files that collide in one import.

**Except when the `.cube` is not there any more.** An entry that is a `.cube` only keeps a look file out while the library's copy of that `.cube` is intact. When an earlier import's `.cube` is missing or changed (the looks show "File missing or changed. Import again."), a look file of the same look takes the entry over instead (its file, format, checksum and LUT size; the damaged `.cube` is deleted once the list is saved), so the look works again, with the note "The 3D LUTs of 2 looks were missing or had changed, so those looks are now the look files. Import the 3D-LUT version again to get the 3D LUTs back." Importing the `.cube` version again replaces the look file as above. (`Slot::Repair`: the planner decides from the list alone and the importer looks at the disk.)

**The person's own look files.** A look file that no package claims (a loose file, or a zip or folder of them) is the person's own look and asks nothing: the file says what it expects. It is named after its file (`Warm Pull.alf4` is "Warm Pull", `user:warm-pull`), listed under "Your LUTs", and imported with the input the file says (an `.alf4` and an `.alf4c` as LogC4, an `.aml` as LogC3 to Rec.709); an answer to the question for `.cube` files (`--input`) does not apply to it. The wizard row is READY, "ARRI Look Files · LogC4", "3 .alf4 · 33-point", with "Imported as your own looks, with the input read from each file." For an `.alf4c` the row adds "Custom.alf4c: its own display transforms are not used; Yellowmoon's display rendering is." (the SDR and HDR transforms of a custom colour management are ARRI's display rendering for ARRI's cameras; ours replaces them).

**CLI.** `ym looks import` takes the same sources (`docs/10`); the notes above print on stderr, and `--input` is for `.cube` files only.

**Not confirmed or not used.**

- *Look Intensity* (id 10006) is ignored, so a look saved at 50 % opens at 100 %. How the camera blends a look with a CDL (whether the intensity scales the CDL too) is not documented.
- *CDL clamping.* ASC's CDL clamps to [0, 1]; `Cdl::apply` clamps at 0 only, because it runs on LogC codes and on LUT output that may leave [0, 1]. Where a file's CDL pushes a code beyond 1 the two can differ.
- *`.alf4c` details.* Where ARRI's Reference Tool stores the mandatory "Custom Color Management Name" is not known (ARRI's look builder sets none), so the look is named after the file; a CMT whose destination is not LogC4 (ARRI's second CMT option) is refused by its colour-space id; the display transforms and design data are not used.
- *`.aml` details.* An `.aml` with no LUT (a look made only from ARRI's video look parameters) is refused by its structure (no LUT size or no payload). The LUTs are taken as made for EI 800 (the mid-grey of the library's High strength lands where ARRI's Rec.709 rendering of EI 800 puts it); a look made at another EI would be off by the difference.
- *Interpolation.* ARRI's own resampling of a 17-point cube to 33 points is tetrahedral (to within one 16-bit code); what the camera does is not known. Yellowmoon samples every LUT tetrahedrally, as for a `.cube`.
- *UI text.* The Choose and Get-the-files screens describe the 3D-LUT downloads only (their text is in the design screenshots): the look-file versions are found by the Recognise step, and the NOT CHOSEN sentence names both titles. The menu item and the Look tab button read "Import .cube…" although they take look files too.

**Verified against the real downloads and files from ARRI's look builder** (Linux x86-64, ignored tests with `YM_ARRI_DOWNLOADS` and `YM_ALF_SAMPLES`): every `.alf4` LUT of the library equals the even nodes of the matching 65-point `.cube` within 7.63e-6 (87 of 87, worst 7.6295e-6, which is half a 16-bit code); all 261 `.aml` files parse, import and render; all 348 files are rebuilt byte for byte by `AlfFile::to_bytes` from what was read; the nine files that ARRI's look builder wrote from our own cubes (17-, 33- and 65-point) match their cubes within 7.63e-6 (the `.aml` made from 17 points, which ARRI resamples to 33 tetrahedrally, within 1.5e-5), parse, rebuild byte for byte, and import or are refused as described. A single loose file of each package, in either order with the whole package or with its other form, has the id it has in the whole download (`1111 Black and White.cube` and `LL-1111-Black-and-White.alf4` are `arri:1000:black-and-white-1111`, `9111-Dream.cube` is `arri:9000:dream-9111-logc3-rec709`), and importing `9111-Dream.cube` before the whole LogC3 to Rec.709 package still ends with its 87 looks, `5121-City` included (`single_files_of_the_real_downloads_are_the_looks_their_names_say`).

## Look tab behaviour

- Sections, one per library (`docs/11-ui.md` §Look tab sections): "Built in" (Neutral, Signature), "ARRI looks · LogC4", "ARRI looks · LogC3", "ARRI looks · LogC3 to Rec.709", "ARRI looks · LogC3 to Rec.709 · ALF2" and "Your LUTs", in that order, each shown only when it has looks (see *Library* above for which look is in which). Counts on the right. Each header folds its section away (remembered while the app runs, never saved); search filters by name across sections, opens every section with matches and counts them ("2 of 7"). A tile's tooltip names its section and the kind of file. Where a look's name stands alone, a name that another library has too gets that library's short word ("Film · LogC3").
- Tiles render the current photo through each look at thumbnail size; render lazily as tiles scroll into view; cache in memory for the session.
- Keys 1–9 pick the first nine tiles in display order.
- The default look carries a DEFAULT tag; "Make default" appears on the selected tile when it is not the default.
- If no ARRI looks are imported (in none of the four libraries), a call to action shows in their place: "The free ARRI Look Library is not imported yet." and "Import ARRI looks →" opening the wizard.

## Missing looks

If a look in the manifest has lost its file, or its hash changed, it's shown disabled with "File missing or changed. Import again." The CLI exits with code 3 when `--look` names a missing look (`docs/10-cli.md`).

## Implementation notes (`ym-looks`)

The code and tests are in `crates/ym-looks` (tasks M1-10 and M3-09).

**Where things live.** `Library::open(root, builtin_dir)` takes the *app folder* (`default_root()`: `~/Library/Application Support/app.yellowmoon` on a Mac, `$XDG_DATA_HOME/app.yellowmoon` or `~/.local/share/app.yellowmoon` on Linux; `YELLOWMOON_HOME` replaces it, for tests); the library is its `looks/` folder. `builtin_dir` is `data/looks/builtin` (the crate does not look for `data/` itself; `ym-core` does). Opening creates nothing on disk.

**Default look.** The default look is the preference `default_look`, `builtin:signature` out of the box (`ym_core::preferences::DEFAULT_LOOK`; `LookId::factory_default()`; Neutral is `LookId::neutral()`). `LookService::effective_default` returns the preference when that look exists and its file is there, else Signature, else Neutral (which exists in code, so a data folder without `signature.json` still renders). Removing the imported look that was the default sets the preference back to Signature. `Core::open` hands the effective default to the session, so new photos, `ym render` and `ym looks list` (the `default` tag) agree. `DevelopSettings::default()` is the identity development with Neutral, because it needs no library (tests and a core without one); a settings document that names no look, a `--settings` file for example, reads as the user's default look.

**Ids.** `arri:<group_no>:<slug>` where the slug is the look's name; the two older Look Library packages add `-logc3` and `-logc3-rec709` to it (and to the copied file's name), so the three libraries, which share names, never collide. The look-file libraries get the ids of their 3D-LUT siblings (`.alf4`) or of the LogC3 to Rec.709 package with a strength (`.aml`), see *ARRI Look Files*. A display LUT is `arri-display:<logc4|logc3>:<target>`; the target is read from the part of the file name after `-to-` (`Gamma24_Rec709` and `Rec709_Gamma24` both give `rec709-g24`). Two different files that would get the same id get `-2`, `-3`, … so nothing the user imported is ever overwritten.

**`--look` resolution** tries three levels in order and the first level that matches anything decides: the full id; the slug after the last colon, or everything after the first colon (`3000:film-warm`); the display name. All comparisons ignore case and extra spaces. A level that matches several looks is ambiguous (exit 2, the message lists the ids); nothing matching is exit 3. A look whose file is missing or changed still resolves; loading it fails with exit 3 and the "File missing or changed. Import again." text.

**Parametric looks.** `ParametricLook::from_spec` (`ym-color`) refuses a look whose CDL slope, offset and power make an input of 1.0 reach `CDL_OUTPUT_LIMIT` (1000) or NaN in any channel (`LookError::CdlOutOfRange`): such values leave nothing usable, and the hue curves after the CDL would turn them into NaN.

**Import protocol.** Under `looks/.lock`: re-read the manifest, read each source file once and check its hash against the recognition, write the copies to hidden temporary names, rename them into place, then replace `manifest.json` atomically. A failure or a cancel before the last step removes what was staged. An import that changes nothing does not touch the manifest. Identity is id plus SHA-256 (and, for the one case of a look in two forms, the `.cube` outranks the look file: *ARRI Look Files*). The same id and hash is "unchanged" (a copy that went missing or was altered in the library is written again). For ARRI files, the same id and file path with a new hash replaces the entry in place (ARRI re-released it). For user LUTs a new hash under an old name becomes a new numbered entry. Importing the same user LUT again with a different answer to the input question updates its input. Two files of one import never replace each other: when their ids and file names collide (names that differ only in punctuation), the later one gets `-2`, `-3`, … and the replace-in-place rule applies only to entries from an earlier import.

**The same bytes are already imported.** A file that would become a new entry is compared by its SHA-256, with the same input, with what the library holds, whatever the file is called now and whichever group the entry sits in (`crates/ym-looks/src/import/plan.rs`, `same_bytes`):

- *The person's own file* (a LUT or look file that no package claims) whose bytes are already in the library, as the person's own entry or as one of ARRI's looks or display LUTs, is "unchanged": the entry stays as it is under its own name (a renamed copy of `Mine.cube` is not `user:mine-2`), the report lists it as unchanged and says "1 file is the same as a LUT that is already imported under another name, so it was not added again." Two copies inside one import are one entry too. The same bytes with another input are another LUT (the person's to say), and a new answer for the same name still updates the input as above.
- *An ARRI package that claims a file the person imported earlier* (same bytes, same input, an entry under "Your LUTs"): the person's entry becomes the ARRI look in place, with its ARRI id, name, group and place in the library (its file moves to the ARRI folder; the old file is deleted once the list is saved), and no second entry is made. The report lists it as updated and says "1 of your own LUTs is a file from ARRI's library, so it moved to the ARRI looks. It is not listed twice." The old id is gone from the library, so what named it follows: the report lists the pair (`ImportReport.renamed`: `user:warm` to `arri:1000:gray-test-night`), says it in a second note ("user:warm is now arri:1000:gray-test-night. A script or a setting that names the old id needs the new one."; five pairs at most, then "and N more"), and when the old id was the **default look** the preference moves to the new id ("Your default look user:warm is now arri:1000:gray-test-night, so new photos still start with it."), so new photos and `ym render` do not quietly fall back to Signature. Photos that are open with the old look lose it, as they do when a look is removed. A look file claimed by an `.alf4` package and a display LUT claimed by a display package follow the same rule (a look entry becomes a display LUT when the package says it is one and the inputs agree: LogC4 for LogC4, LogC3 for LogC3). When ARRI's look is already in the library in another form (the `.cube` beside a person's own look file of it), the `.cube` is kept and the person's entry stays theirs: the bytes differ, so nothing says that they are the same look.
- *ARRI files are not merged with each other by their bytes inside one import*: ARRI's names say which looks a package has, and two of them may hold the same LUT. Against entries of an earlier import they are, when the file was recognised by its name alone (an ARRI file under another name whose bytes are an ARRI look already imported is that look: unchanged). **A file that a hash identified is never merged**: its id is ARRI's own and says which look it is (the LogC3 to Rec.709 package has two files, `5121-City.cube` and `9111-Dream.cube`, with the same bytes; a loose `9111-Dream.cube` imported first must not swallow `5121-City.cube` when the whole zip comes later). Only the person's own entries can be claimed by it (`Spec::by_hash`).
- *Order inside one import.* ARRI's `.cube` files are planned first, then its look files, then the person's own files, each in the order of the list, so the `.cube` wins, an ARRI file never meets a person's file of the same import, and the numbering of the person's own files is unchanged.
- *The note "N looks were replaced by the 3D-LUT version" counts only look files that a `.cube` replaced.* An own LUT that an ARRI package claimed has its own note and is not counted there; a look file replaced by its own look file after a lost `.cube` has the one above.
- A step that adds nothing (everything is "unchanged" or already imported) leaves `manifest.json` byte for byte as it was. The checks are in `crates/ym-looks/tests/reimport.rs`: a sequence of imports (the LogC4 cubes, again, the ALF4 zip, the ALF4 zip with the LogC4 files in one folder, the unzipped folder, LogC3, LogC3 to Rec.709, the ALF2 zip, again, one own cube, the same cube renamed, and the look files into an empty library followed by the cubes) on made-up looks laid out like the real downloads, counting the library after each step (no id with a collision number, a file for every entry and an entry for every file, a list that verifies clean), and the same sequence on ARRI's real zips in `real_downloads.rs` (ignored; every step by hash, and by names alone except the folder that holds both libraries). File names in the library are cut to 120 bytes (255 is the file-system limit, and the encoding suffix, a collision number and the hidden temporary name come on top).

**Status and verify.** Opening the library only checks that each file exists (`Missing`). `verify` re-hashes every file (`Changed`), and loading a look re-hashes the file it reads, so a changed file is never applied even before `verify` has run. A look that `verify` or a reload finds missing or changed is also dropped from the in-memory cache, so a disabled look is never applied from memory. `verify` on a looks list that is damaged or from a newer version returns that error; it never reports a list it could not read as clean.

## One-click download (V2, only with ARRI's consent)

[HUMAN M3-H3] Email arriraw-dev@arri.de asking whether the app may download the Look Library directly. If ARRI agrees, add a manifest of URLs and checksums hosted outside the app bundle (so it can be withdrawn), and keep import as the fallback.

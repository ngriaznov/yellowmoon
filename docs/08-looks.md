# 08 · Looks, defaults and the ARRI import wizard

Crate: `ym-looks`. UI: Look tab, the wizard (`design/mockups/import-looks.html`). CLI: `ym looks …`.

## Look model

```rust
pub struct Look {
    pub id: LookId,               // "builtin:neutral", "builtin:signature", "arri:<group>:<slug>", "user:<slug>"
    pub name: String,             // display name
    pub group: LookGroup,         // BuiltIn | Arri { group_no, group_name } | User
    pub input: LookInput,         // LogC4 | LogC3Ei800 | LogC3ToRec709 (includes display rendering)
    pub cdl: Option<Cdl>,
    pub lut: Option<LutRef>,      // path + SHA-256; loaded lazily, cached in memory
    pub source: Option<Source>,   // URL, import date, original file name
}
```

Application maths is in `docs/02-colour-pipeline.md` (C10). Intensity 0–100% per photo, default 100.

## Built-in looks (shipped, Apache-2.0)

- **Signature** (the default out of the box): the owner's decision of 2026-10-14 (`docs/16`), shipped as `data/looks/builtin/signature.json`. Its name may change later; only the file's `name` then changes. It is a LogC4 luma tone curve through 15 points, then ASC CDL saturation 1.20 with Rec.709 luma weights on the LogC4 codes:
  - the curve is applied as `v += T(L) - L` on all three channels, `L` the LogC4 code of the AWG4 luminance of the pixel (the way the Curve tab's luma curve works, `docs/02` C11), through the monotone cubic of `ym_color::curve::Curve` (Fritsch-Carlson tangents); the 15 points are candidate B of the look preview;
  - **mid grey is unchanged** (LogC4 0.2784 for scene 0.18, moved by less than 1e-6 in code value), neutrals stay neutral, and the tone curve rises everywhere;
  - against Neutral it darkens the shadows (about -0.4 stop from -3 to -6 stops), lifts the lights (+0.22 stop at +1 stop, +0.42 at +2, +0.52 at +3) and eases the highest lights back (+0.49 at +6). On a neutral ramp at the app's +0.85 EV display placement the 8-bit codes at -2 stops, mid grey and +2 stops are 42, 120 and 204, against 52, 120 and 192 for Neutral;
  - the constants were chosen by eye on twelve photos. They were never fitted to any LUT, ARRI's or a commercial one.
  The loader bakes it into a 33³ cube like every parametric look. The cube follows the exact look to under one 8-bit display code (worst 0.96 code over 20 000 random colours, 0.41 on a neutral ramp, 0.69 on a hue sweep; `crates/ym-color/tests/look_signature.rs`). The agent's tuning tool is `ym looks bake signature.json --preview <ARW>…`, which renders a contact sheet against Neutral.
- **Neutral**: identity. Emulated ALEXA colour through our display transform, no creative look. It stays in the list as the "no look" choice and as the look a missing default falls back to; it is not the default any more.

**Parametric look files (M1-10, `luma_curve` added 2026-10-14).** `data/looks/builtin/<name>.json`, loaded by `ym_color::look::ParametricLook::from_json` and baked into a 33³ cube (`bake`, `to_transform`), so the CPU and the GPU read every parametric look as an ordinary cube and need no kernel of their own for it. Schema `yellowmoon.parametric-look/1`; every field except `schema` and `name` is optional, unknown fields are rejected, and `{"schema": "yellowmoon.parametric-look/1", "name": "Neutral"}` is the identity:

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

`luma_curve` is a tone curve on the luminance of the pixel: 2 to 16 points `{ "x", "y" }` in [0, 1]² with `x` strictly increasing, where `x` is the LogC4 code value of the AWG4 luminance (`LogC4::luma`, the `L` of the Curve tab's luma curve) and `y` what it becomes. It is applied as `v += T(L) - L` on all three channels, so it moves tone and leaves the differences between the channels in LogC4 (hue, chroma) alone. `T` is the monotone cubic of `ym_color::curve::Curve`; it holds its end values between 0 and the first point and between the last point and 1, and continues with slope 1 below 0 and above 1. Points on `y = x` from 0 to 1 are the identity. Its `x` is not the `luma` of `luma_vs_sat`, which is the Rec.709-weighted luma of the codes; the two are never mixed. `hue_vs_sat` and `luma_vs_sat` values multiply the chroma (0 to 8); `hue_vs_hue` values are degrees (±180, positive turns red towards yellow). Hue is in degrees on the LogC4 hue circle (0 red, 60 yellow, 120 green, 180 cyan, 240 blue, 300 magenta), `luma` is the Rec.709-weighted LogC4 luma in [0, 1]. At most 64 points per curve; the curves are monotone (Fritsch–Carlson) splines, periodic for hue, so they never overshoot the values given. The order is `luma_curve`, then the CDL, then the hue and luma-vs-saturation curves, which see the CDL output and keep its luma. Full details: the module documentation of `crates/ym-color/src/look.rs`.

## Library folder

```
~/Library/Application Support/app.yellowmoon/looks/
  manifest.json
  arri/<group_no>-<group_slug>/<file>.cube
  arri-display/<file>.cube
  user/<file>.cube
```

`manifest.json`:

```json
{
  "version": 1,
  "looks": [
    { "id": "arri:3000:<slug>", "name": "…", "group": { "kind": "arri", "no": 3000, "name": "Film" },
      "input": "logc4", "file": "arri/3000-film/<file>.cube", "sha256": "…", "lut_size": 65,
      "source": { "package": "look-library-logc4", "url": "https://www.arri.com/…", "imported": "2026-11-02T10:14:00Z", "original_name": "…" } }
  ],
  "display_luts": [ { "id": "arri-display:logc4:rec709-g24", "input": "logc4", "target": "rec709-g24", "file": "arri-display/…", "sha256": "…" } ]
}
```

The default look and the display rendering choice are preferences (`preferences.json`, `docs/01`), not part of the manifest.

Writes are atomic (write temp, fsync, rename) and take `looks/.lock` so the app and CLI can import at the same time safely. `looks verify` re-hashes every file and reports changes.

## What users can import (verified on arri.com, 11 Oct 2026)

| Package | Size | Contents | Role in the app |
|---|---|---|---|
| Look Library, LogC4 log-to-log | 181 MB | 87 `.cube`, 65-point | recommended looks |
| Display LUTs, LogC4 | 32 MB | 22 `.cube` (Rec.709, P3, Rec.2020, HLG, PQ) | ARRI's exact display rendering |
| Look Library, LogC3 log-to-log | 30 MB | 87 `.cube`, 33-point | looks run in LogC3 |
| Look Library, LogC3 to Rec.709 | 32 MB | 87 `.cube` | complete rendering; replaces our display transform |
| Display LUTs, LogC3 | 46 MB | 48 `.cube`, including the classic Rec.709 | classic ALEXA rendering |

All are free public downloads with no login and no licence text. ALF4/ALF2 binary containers in the same packages are skipped. The Look Library has 87 looks in nine groups (group 3000 is "Film").

ARRI files are never bundled, hosted, downloaded automatically or committed. arri.canto.de's robots.txt disallows automated access, so the app opens ARRI's page in the browser and the user downloads.

`data/looks/arri-packages.json` ships in the bundle (it holds patterns and fingerprints only). A placeholder with an empty `packages` list exists from the start.

## Recognition rules

Recognition is data-driven: `data/looks/arri-packages.json` lists, per package, the expected folder and file-name patterns, LUT sizes, counts and (once fingerprinted) the SHA-256 of every file. It contains no LUT data.

[HUMAN M3-H2] Mykyta downloads the five packages by hand and runs `tools/looks/fingerprint.py <zip>…`, which writes names, sizes, LUT sizes and hashes into `arri-packages.json`. The agent writes the script and the recogniser; until the fingerprints exist, the recogniser uses the patterns alone.

Rules, in order:

1. Unzip in memory (or stream) and list `.cube` files; ignore `__MACOSX`, `.alf4`, `.alf2`, PDFs.
2. Match file hashes against `arri-packages.json`: an exact package match identifies everything.
3. Otherwise use names and headers: `LogC4` vs `LogC3` in names or `TITLE`; `-to-` plus a display name (`Gamma24_Rec709`, `P3`, `Rec2020`, `HLG`, `PQ`) means a display LUT or a Rec.709-baked look; `LUT_3D_SIZE` 65 vs 33; look group numbers (four digits) in folder names.
4. A file that matches nothing gets the generic question (below).
5. A wrong package (for example ARRI camera LUTs for another sensor, or a zip of ALF4 files only) gets a plain explanation: "This zip has ARRI Look Files (.alf4) but no .cube files. Download the 'LUT (.cube)' version from ARRI's page."

## Wizard (five steps, `design/mockups/import-looks.html`)

1. **Choose.** Look Library (LogC4) checked; Display LUTs (LogC4) optional; "Show 3 older LogC3 packages" reveals the rest. One line: ARRI publishes these looks for free; Yellowmoon can't include them, so you download them from ARRI and the app does the rest. Shows the download total.
2. **Get the files.** "Open ARRI's page" opens https://www.arri.com/en/learn/camera-systems/image-science/look-files in the system browser. Then the wizard waits for the zip:
   - "Watch Downloads" switch (asks macOS for access to ~/Downloads the first time; remembered as a preference). Watch with FSEvents for new `.zip` files whose names match the patterns, and wait until the file size is stable for 2 s.
   - "Choose file…" and drag-and-drop of a zip or folder work at any time.
   - Next unlocks when every chosen package has arrived.
3. **Recognise.** Show each package found, with a tick or an explanation. Mismatches are listed in plain words.
4. **Import.** Copy only the `.cube` files into the library folder, grouped as ARRI groups them; hash each; write the manifest. Progress bar per file count. Takes seconds.
5. **Finish.** Thumbnails of the imported looks on the current photo; choose the default look; choose the display rendering ("Ours · ACES 2" or "ARRI's" when display LUTs were imported); offer to move the zip to the Trash (`trash` crate, recoverable) or keep it.

Re-importing the same package is idempotent: same hashes, nothing duplicated.

## Any other `.cube`

Same importer plus one question: what does this LUT expect as input?

- LogC4 (ARRI Wide Gamut 4),
- LogC3 (ARRI Wide Gamut 3, EI 800),
- LogC3 with Rec.709 output baked in (replaces the display rendering).

Validation: the LUT must parse, be 3D (1D shaper allowed), size 2–129, all values finite. Rejected with a reason otherwise.

## Look tab behaviour

- Groups: "Built in" (Neutral, Signature) and "ARRI looks · imported" (then "Your LUTs" if any). Counts on the right. Search filters by name across groups ("2 of 7").
- Tiles render the current photo through each look at thumbnail size; render lazily as tiles scroll into view; cache in memory for the session.
- Keys 1–9 pick the first nine tiles in display order.
- The default look carries a DEFAULT tag; "Make default" on the selected tile.
- If no ARRI looks are imported, the group shows a call to action: "Import ARRI looks →" opening the wizard.

## Missing looks

If a look in the manifest has lost its file, or its hash changed, it's shown disabled with "File missing or changed. Import again." The CLI exits with code 3 when `--look` names a missing look (`docs/10-cli.md`).

## Implementation notes (`ym-looks`, M1-10 and M3-09)

Decisions taken where this spec was silent. The code and tests are in `crates/ym-looks`.

**Where things live.** `Library::open(root, builtin_dir)` takes the *app folder* (`default_root()`: `~/Library/Application Support/app.yellowmoon` on a Mac, `$XDG_DATA_HOME/app.yellowmoon` or `~/.local/share/app.yellowmoon` on Linux; `YELLOWMOON_HOME` replaces it, for tests); the library is its `looks/` folder. `builtin_dir` is `data/looks/builtin` (the crate does not look for `data/` itself; `ym-core` does). Opening creates nothing on disk.

**Default look (2026-10-14).** The default look is the preference `default_look`, `builtin:signature` out of the box (`ym_core::preferences::DEFAULT_LOOK`; `LookId::factory_default()`; Neutral is `LookId::neutral()`). `LookService::effective_default` returns the preference when that look exists and its file is there, else Signature, else Neutral (which exists in code, so a data folder without `signature.json` still renders). Removing the imported look that was the default sets the preference back to Signature. `Core::open` hands the effective default to the session, so new photos, `ym render` and `ym looks list` (the `default` tag) agree. `DevelopSettings::default()` stays the identity development with Neutral, because it needs no library (tests and a core without one); a settings document that names no look, a `--settings` file for example, reads as the user's default look. The wizard's Finish step and Preferences list Neutral, Signature and (in the wizard) the first imported look, and select the one the library marks default. A `preferences.json` written before this change keeps what it saved (V1 is unreleased, so nothing is migrated).

**Ids.** `arri:<group_no>:<slug>` where the slug is the look's name; the two older Look Library packages add `-logc3` and `-logc3-rec709` to it (and to the copied file's name), so the three libraries, which share names, never collide. A display LUT is `arri-display:<logc4|logc3>:<target>`; the target is read from the part of the file name after `-to-` (`Gamma24_Rec709` and `Rec709_Gamma24` both give `rec709-g24`). Two different files that would get the same id get `-2`, `-3`, … so nothing the user imported is ever overwritten.

**`--look` resolution** tries three levels in order and the first level that matches anything decides: the full id; the slug after the last colon, or everything after the first colon (`3000:film-warm`); the display name. All comparisons ignore case and extra spaces. A level that matches several looks is ambiguous (exit 2, the message lists the ids); nothing matching is exit 3. A look whose file is missing or changed still resolves; loading it fails with exit 3 and the "File missing or changed. Import again." text.

**Manifest.** Exactly the layout above, with two additions: a user LUT's group is `{ "kind": "user" }`, and `source.package` and `source.url` are left out for files that did not come from a known package. A damaged `manifest.json` does not stop the app: built-in looks keep working, the problem is reported, and the next import moves the file aside as `manifest.json.damaged-<n>` and starts a new list. A manifest written by a newer version (`"version"` above 1) is never overwritten.

**Import protocol.** Under `looks/.lock`: re-read the manifest, read each source file once and check its hash against the recognition, write the copies to hidden temporary names, rename them into place, then replace `manifest.json` atomically. A failure or a cancel before the last step removes what was staged. An import that changes nothing does not touch the manifest. Identity is id plus SHA-256: the same id and hash is "unchanged" (a copy that went missing or was altered in the library is written again); for ARRI files, the same id and file path with a new hash replaces the entry in place (ARRI re-released it); for user LUTs a new hash under an old name becomes a new numbered entry. Importing the same user LUT again with a different answer to the input question updates its input. Two files of one import never replace each other: when their ids and file names collide (names that differ only in punctuation), the later one gets `-2`, `-3`, … and the replace-in-place rule applies only to entries from an earlier import. File names in the library are cut to 120 bytes (255 is the file-system limit, and the encoding suffix, a collision number and the hidden temporary name come on top).

**Status and verify.** Opening the library only checks that each file exists (`Missing`). `verify` re-hashes every file (`Changed`), and loading a look re-hashes the file it reads, so a changed file is never applied even before `verify` has run. A look that `verify` or a reload finds missing or changed is also dropped from the in-memory cache, so a disabled look is never applied from memory. `verify` on a looks list that is damaged or from a newer version returns that error; it never reports a list it could not read as clean.

**Recognition details.** The evidence for an encoding is the source's own name (the zip or folder), the file's path and its `TITLE`, with separators and case ignored; `LogC4` and `LogC3` together are a conflict and count as no evidence. A file in a four-digit group folder is a look; a file with `-to-` plus a display name and no group folder is a display LUT; both together (LogC3 only) is a Rec.709-baked look. The display name may stand in the file name, in a folder name, in the `TITLE` or in the name of the zip (a Rec.709-baked look is display-referred, so it must not be taken for a log-to-log look just because its file names do not repeat what the zip says). Paths are relative to the chosen zip or folder; the chosen folder's own name is evidence but never a group. A group folder starts with exactly four digits and the number does not run on into more digits (`2026-10-12` and `2024_05` are dates, not groups). Identical LUTs published under two names count as two files of the package. The LUT size must agree with the package when the package lists sizes. Whatever fits exactly one package is that package's; everything else gets the generic question, with a suggestion when the names or the size hint at one. Files that name ARRI or ALEXA but fit none of the five packages are explained as "from a package Yellowmoon does not know" and may be imported as the user's own LUTs. A package with fewer files than it should have is reported `Incomplete` with the missing file names once fingerprints exist; the wizard blocks "Next" on it, the library only refuses it when asked to (`require_complete`).

**`arri-packages.json`** carries, per package: id, wizard title, role (`looks` or `display_luts`), input, download size, zip-name patterns (for the Downloads watcher), expected counts and LUT sizes, name markers (encoding words, whether a display rendering and group folders are required, forbidden or either) and the `fingerprints` that `tools/looks/fingerprint.py` fills. The patterns shipped today are best guesses from ARRI's package descriptions and have not been checked against the real downloads; fingerprints make recognition exact (M3-H2).

**Downloads watcher.** It polls the folder every 500 ms instead of using FSEvents, reports a matching `.zip` once its size and modification time have stayed the same for 2 s, ignores zips that were already there when watching started, and reports a changed file again. A failed listing (access withdrawn, volume unmounted) keeps what the watcher knew, so old downloads are not reported as new when the folder comes back. The macOS permission prompt for `~/Downloads` appears at the first listing, so `start` returns that error.

## One-click download (V2, only with ARRI's consent)

[HUMAN M3-H3] Email arriraw-dev@arri.de asking whether the app may download the Look Library directly. If ARRI agrees, add a manifest of URLs and checksums hosted outside the app bundle (so it can be withdrawn), and keep import as the fallback.

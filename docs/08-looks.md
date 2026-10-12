# 08 · Looks, defaults and the ARRI import wizard

Crate: `ym-looks`. UI: Look tab, the wizard (`design/mockups/import-looks.html`). CLI: `yellowmoon looks …`.

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

- **Neutral** (default out of the box): identity. Emulated ALEXA colour through our display transform, no creative look.
- **Signature**: parameters designed by Mykyta in M3 [HUMAN M3-H1] as JSON in `data/looks/builtin/signature.json`: CDL (slope, offset, power, saturation), plus hue-vs-saturation and hue-vs-hue curves and a luma-vs-saturation curve, all evaluated in LogC4. The loader bakes it into a 33³ LUT. Its final name describes its character and avoids ARRI and film-stock names. The agent provides a tuning tool: `yellowmoon looks bake signature.json --preview <ARW>…` that renders a contact sheet.

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

## One-click download (V2, only with ARRI's consent)

[HUMAN M3-H3] Email arriraw-dev@arri.de asking whether the app may download the Look Library directly. If ARRI agrees, add a manifest of URLs and checksums hosted outside the app bundle (so it can be withdrawn), and keep import as the fallback.

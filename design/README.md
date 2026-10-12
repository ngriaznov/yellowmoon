# Yellowmoon UI mockups

Interactive mockups of the Yellowmoon V1 screens, designed at 1440 × 900 (desktop app window).

## Pages

| Page | Source board | What it shows |
|---|---|---|
| `mockups/start.html` | `BoldStart.dc.html` | Start screen: drop ARW files or a folder, every photo develops automatically. |
| `mockups/photos.html` | `Bold.dc.html` | Editor: look first, then Light, Colour, Detail, Curve and RAW tabs; filmstrip; before/after. |
| `mockups/export.html` | `BoldExport.dc.html` | Export panel: photos, format (JPEG, TIFF, PNG, DNG), colour space, size, destination, CLI equivalent, progress. |
| `mockups/import-looks.html` | `BoldWizard.dc.html` | Five-step wizard to import ARRI's free look packages. |

`mockups/index.html` links all four.

## Opening them

Double-click `mockups/index.html` (or any page). They work offline from disk: the small rendering runtime is inlined in each page and the Figtree font is bundled in `mockups/fonts/` (SIL Open Font License, `fonts/OFL.txt`). No server or network needed. Checked in Chromium.

## What is simulated

These are static mockups with fake data: 12 photos from "Lisbon" drawn as flat shapes, fixed camera metadata, nothing reads or writes files.

- Buttons, tabs, segmented controls, switches, sliders, look search and links between pages work.
- Timers fake progress: developing on Start (about 4 s after "Open files…"), export (about 4 s for the 3 picked photos), import (about 3 s).
- Some controls have no behaviour in the mock: Crop, Pick white, Fit 34%, Choose…, Show in Finder, Open ARRI's page, Choose file….
- Each page exposes its component as `window.__inst`; `window.__inst.setState({...})` jumps to any state. `window.__errors` collects unresolved template bindings (debug aid, empty in all captured states).

## Screenshots

`screenshots/`, 1440 × 900 at 2× (2880 × 1800 PNG), taken from the pages above. `overview.png` is a 1440 × 900 contact sheet of all of them.

| File | State |
|---|---|
| `start-idle.png` | Start, nothing opened |
| `start-busy.png` | Developing, 5 of 12 ready (`busy:true, prog:5.4`) |
| `start-done.png` | All 12 ready (`prog:12`) |
| `photos-look.png`, `photos-light.png`, `photos-color.png`, `photos-detail.png`, `photos-curve.png`, `photos-raw.png` | Each tool tab (`tab`) |
| `photos-look-teal.png` | Teal and orange picked, side panel scrolled to the look library |
| `photos-look-search.png` | Look search "film" |
| `photos-before.png` | Before/after showing "before" |
| `export-jpeg.png`, `export-tiff.png`, `export-png.png`, `export-dng.png` | Each format (`fmt`) |
| `export-running.png` | Exporting, 1 of 3 done, second at 45% |
| `export-finished.png` | Exported |
| `import-1-choose.png` … `import-5-finish.png` | Wizard steps 1–5 as numbered in the UI (`step` 0–4) |
| `import-4-importing.png` / `import-4-imported.png` | Import step mid-copy (40 of 87) and done |

## Source

`source/` holds the four original design boards (`BoldStart.dc.html`, `Bold.dc.html`, `BoldExport.dc.html`, `BoldWizard.dc.html`), unmodified. They need the design tool's runtime to open. The mockups were generated from them; the only changes are the bundled font instead of Google Fonts, links rewritten to the mockup file names, and page titles.

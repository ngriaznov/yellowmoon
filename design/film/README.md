# Film node mockups (F-01)

Mockups for the Film node of `docs/24-film.md`, drawn for the owner's review (F-H1). **`docs/24` wins wherever the two differ**; §4 of the spec was revised to match these pictures. Where a mockup and `docs/24` are both silent, `docs/23` §7 and `docs/11` decide.

**These are mockups, not screenshots of the app.** The interface around the pictures is the app's real DOM and CSS (glasscn components, `design/tokens.css`, as in `design/nodes/`), at 1440 × 900 and shot at 2× (2880 × 1800 JPEGs). `index.html` is the walk-through with a caption under each picture.

## What is hand-made

**Every film picture is a hand-made illustration** of what a print might look like, drawn with a small image script (`make-pictures.py`: a gentle S-curve, warmer highlights, a slightly lifted black, a little halation glow around bright edges, fine grain). It is not the engine, not spektrafilm and not made from any film profile. Each stock's tile has its own hand-set numbers (Portra warmer and soft, Ektar saturated, Velvia and the slides vivid with deep blacks, the cine negatives flatter), so the tiles differ plausibly, but they say nothing about how the real stocks look. The photo is the owner's own (`design/nodes/html/assets/photo-graded.jpg`, the picture the viewer showed). The node card's thumbnail is the same hand-made picture, small. The radial overlay on screen 09 is hand-drawn too.

Stock and paper names are the profiles' marketing names as `research/film-spektrafilm-2026-11-01.md` §3.1 lists them. Where the ids do not give the name outright ("Kodak Portra 800 push 1", "Kodak Professional Portra Endura") the wording is a guess to be checked against `info.name` when the profiles land (F-02). Names are the owner's choice: real names with the line (docs/24 §1.3). No logo, box colour or typeface of a maker is used.

## The pictures

| File | Screen |
|---|---|
| `01-add-node-menu.jpg` | Add node menu with **Film** enabled under Corrector; Later items follow, greyed |
| `02-film-tab.jpg` | Graph Input → 01 Main → 02 Film → Output, Film selected, Film tab: Stock, Exposure, Strength, the credit line |
| `03-stock-picker-a-list.jpg` | **Option A**: grouped list with ISO |
| `04-stock-picker-b-tiles.jpg` | **Option B**: tiles with a small render of the photo |
| `05-print-tab-lights-a.jpg` | Print tab: Paper, Print exposure, Lights as two sliders, Cyan under More |
| `05b-paper-list.jpg` | The paper list: Matching paper, then the six papers; cine print films hidden |
| `06-print-lights-b-pad.jpg` | **Option B**: lights as a colour pad |
| `07-slide-stock.jpg` | A slide stock: the Print tab dimmed, "Slide film is scanned as it is. There is no print." |
| `08a-grain-two-tabs-a.jpg` | **Option A**: Halation and Grain as two tabs (Grain tab, photo at 100 %) |
| `08c-halation-tab-a.jpg` | Option A, the Halation tab |
| `08b-grain-one-tab-b.jpg` | **Option B**: one Grain tab with both cards (photo at 100 %) |
| `09-mask-tab.jpg` | Mask tab with one Radial row |
| `10-output-look-note.jpg` | Output on its Look tab with the note about the look acting after the Film node |
| `11-about.jpg` | About: credit line, CC BY-SA 4.0 link, trademark line |
| `12-brightness-option.jpg` | **Option**: "As the app" or "As printed", under More on the Print tab |

`html/pages/*.html` are the same screens as pages (`01.html` … `12.html`, plus `05b`, `08a`–`08c`). They load `../app.css` (the app's CSS, extracted once from the captured pages), `../nv.css` (the node view) and `../fm.css` (the few new pieces: the Option tag, pickers, colour pad, More row, About). Keep the folders together.

## Options the owner chooses (F-H1)

1. **Stock picker:** A, grouped list (03), or B, tiles (04).
2. **Lights:** A, two sliders with Cyan under More (05), or B, a colour pad (06).
3. **Halation and Grain:** A, two tabs (08a, 08c), or B, one tab named Grain with both cards (08b). B is four tabs in all, A five.
4. **Brightness:** "as the app" only (recommended), or the two-segment control under More (12).

The "Option" tags on the pictures are annotations, not part of the interface.

## What the mockups draw that the spec does not yet settle

- A **slide stock keeps its Print tab openable**, dimmed, so the sentence can be read. docs/24 §4.1 says "disables the Print tab"; §4 now says this.
- **Cine negatives** are in the picker as the brief asks, but their matching print film (Kodak 2383) is hidden (§6.4). Nothing is drawn for what "Matching paper" means for them; see the questions in `docs/24` §9.
- The **Graph · List switch** the node mockups still carry was dropped by the owner (2026-11-01) and is not drawn here.
- The header's **Before, Crop and Nodes** buttons are drawn without the icons they have since 0.2.0 (the pages were built from the node mockups); the app's header is the reference.
- Stock names on the pages were typed before `data/film/params/names.json` existed; the app uses that file ("Kodak Portra 800 (Push 1)", not "push 1").
- The node's head line reads "Film · whole photo" (and "Film · Radial" with a mask), after the corrector's "Corrector · whole photo".

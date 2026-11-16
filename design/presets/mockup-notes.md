# Presets mock-ups: notes

Pictures only. Nothing here is product code. The spec is `docs/30-presets.md`; where it and a picture differ, `docs/30` wins for words and behaviour.

## Decided (the owner, 2026-11-16: "Go with your recommendations on presets")
- Presets persist in `presets/`; they sit on a **Presets card on the Look tab, between Automatic and Look**; a preset holds the **eight groups** the person ticks, the changed ones ticked; phase 1 first.

## Choices where the proposal was silent
- **Six tiles, then "Show all 14".** The proposal said "at most 12 tiles until Show all 14" and asked for six tiles with "Show all 14" in picture 1. With three tiles to a row, twelve are four rows and push the Look card far below the fold; six are two rows. The spec takes six (`docs/30` §17 decision 1). **Question for the owner (P-H1): six or twelve?**
- **Header:** "Presets", the count in the quiet type of a look section's count, and **Save…** in the card's link style at the right (like Reset on the Look card). The empty state has no count and no Save… (its button is in the body).
- **The ring** is the look tiles' own selected ring, and means "applying this would change nothing" (`docs/30` §4.3). In 01 and 05 and 07 the photo wears Warm evening (the chip reads "Film warm · 100%", a warm tint on the photo).
- **No number badges** on the tiles (keys 1 to 9 stay with the looks); the tooltip names the groups.
- **The dialog is 480 px wide**, one row per group with a checkbox, the label and one quiet line. The checkbox is drawn in `presets.css` because glasscn's registry component is added in P-04 (`docs/30` §9.3); size 18 px, the accent when ticked, black tick. At 900 px high the dialog nearly fills the window, which is the reason for one line per group and no group heading.
- **The Replace question** swaps the dialog's body (same title bar, same buttons): the focus goes to Cancel, Replace is primary.
- **The tile menu** has two rules (after Apply to Picked Photos and Update from This Photo, and before Delete); Delete is in the destructive colour. The menu opens at the pointer; in the picture it sits beside the second tile.
- **The older-export line** is a quiet glass note under the boxes, not a warning: nothing is wrong, the numbers are just fixed.
- **The macOS menu** (07) is drawn in dark appearance with the system font: after the nine Apply Look items, one rule, then Save Preset… ⇧⌘S, Import Presets…, Apply Preset ▸ (hovered, its submenu with a check on the presets that match the photo), New Preset from Exported Photo…, a rule, Show Presets Folder, Show Looks Folder.
- **Where the panel is scrolled:** pictures 01 to 08 show the Look tab scrolled to the top, so the Presets card is under the Automatic card and its third row of tiles is behind the foot row; the panel scrolls.

## Not drawn
- The damaged-file rows under the tiles (`docs/30` §9.1), the Rename dialog, the Update question, the import's Replace question (the same dialog as 04), the notice texts other than 06, and the card while the photo is developing. Their copy is in `docs/30` §9.
- A hover on a tile, and the focus ring (it is the look tiles' `--focus`).

# Presets mock-ups

Mock-ups of the Presets card, the Save preset dialog, a tile's menu, the notice for a missing look and the new Looks menu items, for milestone P (task P-01, `docs/30-presets.md`). They wait for the owner's review (P-H1); no interface work (P-04) starts before it. **The spec `docs/30` wins wherever the two differ** (copy and behaviour); these pictures win on layout and size; `docs/11` and `design/screenshots-c2/` win where both are silent. They are in the dark glass look, at 1440 × 900 (2× in the pictures), with the app's own sheets and tokens.

**These are mock-ups, not screenshots of the finished app.** The pages are the DOM of the built app on its mock API (flat-shape placeholder photos, no photo of the owner's), with the Presets card, the dialogs, the menu and the notice written in by `capture.mjs` from the app's own classes. The tile pictures are the mock's look pictures with a colour filter, to tell the tiles apart: the real ones come from the renderer (`docs/30` §9.6). The Looks menu is drawn by macOS, not by the app; the mock-up draws its look.

| File | What it shows |
|---|---|
| `01-look-tab-presets.jpg` | The Look tab with the **Presets card** between Automatic and Look: header ("Presets", the count 14, **Save…**), Apply to on **This photo**, six tiles with the names under them, the first one **ringed** (applying it would change nothing), **Show all 14** |
| `02-empty-state.jpg` | The card with no preset: "Set a photo the way you like it, then save it as a preset." and **Save preset…** |
| `03-save-dialog.jpg` | The **Save preset** dialog: Name "Warm evening", eight boxes with Look, Light and Colour ticked (the groups the photo changed), "Auto stays Auto, so each photo keeps its own exposure and white balance.", Cancel and Save |
| `04-replace-question.jpg` | The same dialog asking **"Replace Warm evening?"** when the name exists |
| `05-tile-menu.jpg` | A tile's right-click menu: Apply to Picked Photos, Update from This Photo, Rename…, Export…, Show in Finder, Delete |
| `06-missing-look-notice.jpg` | The notice bar after applying a preset whose look is not imported: "Warm evening uses the look Film warm, which isn't imported. Everything else was applied." |
| `07-looks-menu.jpg` | The **Looks** menu with its new items (Save Preset… ⇧⌘S, Import Presets…, Apply Preset ▸ with its submenu, New Preset from Exported Photo…, Show Presets Folder) |
| `08-older-export-line.jpg` | The Save dialog opened from New Preset from Exported Photo for an export made by an older Yellowmoon: the name empty, only Look ticked, and the one line about fixed numbers |

## Files

- `html/pages/01.html` … `08.html` are the screens as pages (they open from disk in a browser at 1440 × 900). They load `html/app.css` (the compiled stylesheet of the app, from `dist/` of 2026-11-13), `html/graph.css`, `html/presets.css` (the few rules the app does not have: the checkbox, the macOS menu bar) and `html/assets/fonts/` (Figtree, SIL OFL, as in `design/mockups/fonts/`).
- `capture.mjs` writes the pages from a built app (`node capture.mjs <dist>`); `shoot.mjs` renders them to the JPEGs (`node shoot.mjs`). Neither is product code.
- `index.html` is the walk-through; `mockup-notes.md` has what was chosen where the spec was silent and the question for the owner.

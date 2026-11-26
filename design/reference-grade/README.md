# Reference Grade node mockups

Mockups for the Reference Grade node (node kind `reference_grade`; the spec is `docs/34-reference-grade.md`, §9 and §17; task RG-07, 2026-11-25, waiting on the owner's review RG-H2). Same method as `design/blur/`, `design/colour-uniform/`, `design/mixers/` and `design/skin/`: pages at 1440 × 900 built from the app's captured DOM and CSS, shot at 2× to JPEGs, the graph at 100 % zoom. `index.html` walks through the pictures. `docs/34` wins wherever the two differ; `mockup-notes.md` lists what the drawings decided and what they showed to be awkward.

**The pictures are drawings, and the pictures of the result are computed.** The scene is **synthetic**: a drawn lakeside village (sky, foliage, ground, buildings, a small pond, and a flat drawing of a figure with a skin-toned face), made in linear light by `html/rg_demo.py`. The **reference** is another drawing of different content, a harbour at dusk with a teal-and-amber grade painted in. No film still, no photograph, no third-party image, nobody's likeness, none of the owner's photographs. Every "after" picture is **the maths of `docs/34`** (§3 the global curves and saturation, §4.2 the hue step with Rec.709 luma restored as the owner answered in STATUS question 48, §4a.2 the element grades through the §4a.3 soft masks, §3.2 Strength) applied to that scene by `html/rg_ref.py` (numpy), through soft masks that are the guided filter of §4a.3 taken on the drawing's own element layers. **The values are illustrative; no fit was run.** The display transform is a simple filmic curve and the sRGB encoding, not the product's. A real photograph and a real fit would replace them in the final screenshots.

| File | What it shows |
|---|---|
| `01-empty.jpg` | Main → Match → Output; the node selected on its Reference tab, no reference yet: the chooser, a disabled Fit with its tooltip, Strength, Skin protection, "Choose a picture to match." |
| `02-choose-file.jpg` | Choose reference… open: Image file… (the system open dialog) and Photo in the strip |
| `02b-choose-from-strip.jpg` | The strip as a chooser: "Click a photo in the strip. Esc cancels."; the photo being graded and a photo still developing are off |
| `03-fitting.jpg` | Fitting: a progress bar and Cancel, "Fitting… 2 of 3" |
| `04-fitted.jpg` | Fitted: "Fitted to the reference.", the reference's thumbnail and name, Fit again |
| `05-hue-tab.jpg` | The Hue tab: Hue vs hue and Hue vs saturation, periodic, on the rainbow track |
| `06-elements-tab.jpg` | The Elements tab: Sky, Vegetation, Water (not in the photo, greyed), Ground …; each card has Exposure, Hue, Chroma, its share and an eye for its mask |
| `06b-elements-sheet.jpg` | Sheet: all seven cards (Sky, Vegetation, Water, Ground, Built, Skin, Subject), one with its mask shown, one while the scene is made |
| `06c-element-mask.jpg` | The eye in the Sky card on: the sky's soft mask over the photo |
| `07-look-changed.jpg` | "The look changed since the fit. Fit again to match it with the new look." with the accent on Fit again |
| `08-edited-by-hand.jpg` | "Edited by hand since the fit. Fit again to start over." |
| `09-add-node-menu.jpg` | The Add node menu with Reference Grade |
| `10-simple-view.jpg` | The simple view of a photo with the node: no controls, the existing notice |
| `11-status-and-errors.jpg` | Sheet: every status line, note and error of `docs/34` §9.3 with its exact copy |
| `12-before-after.jpg` | Sheet: the photo, the reference, the fitted result, the result at Strength 50 %, the result without the element grades, and the values used |
| `states.jpg` | Sheet: card states (normal, selected, bypassed, dead), the glyph proposal, panel heads, the copy of the tooltips |

`01` to `10` except `06b` are 2880 × 1800; `06b`, `11`, `12` and `states` are full-page sheets at 1.5×.

## Files

- `html/pages/*.html` are the screens as pages. They load the app's compiled stylesheet and the other mock-ups' sheets from `design/mixers/html/`, `design/nodes/html/`, `design/skin/html/` and `design/colour-uniform/html/cc.css`, then `html/rg.css` (the few new pieces: the Reference card, the strip chooser, the plots, the Elements cards, the sheets). The pictures come from `html/assets/`; keep the folders together.
- `html/rg_ref.py` is a numpy sketch of the node's maths. **It is not product code**: the product's reference is the Rust code of RG-02, in f64. It stands in for the cosine-eased periodic curve where the product has a Fritsch–Carlson table, and for the product's LogC4 with ARRI's published formula only to place the codes (it does not touch ARRI's files).
- `html/rg_demo.py` draws the scene, the reference and the strip photos, computes the element masks and the pictures, and writes `assets/`. `html/build.py` writes the pages (Python 3, `bs4`; it imports the Colour Uniform, skin and mixers builders); `shoot.mjs` renders them, prints console errors and any clipped text.
- To rebuild, in this folder (Playwright is found through `NODE_PATH`; if it wants another Chromium build, set `PW_CHROMIUM` or `PLAYWRIGHT_BROWSERS_PATH` to a folder with the installed build and do not run `playwright install`):

```
python3 -I html/rg_demo.py     # numpy and PIL; about 30 s
python3 html/build.py
node shoot.mjs                 # or: node shoot.mjs 01 06
```

None of it is product code.

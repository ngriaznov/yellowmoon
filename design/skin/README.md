# Skin node mockups

Mockups for the Skin node of `docs/27-skin-retouch.md` (task S-11, design step). They wait for the owner's review (S-H1); no interface code (S-11) starts before it. **`docs/27` and `docs/23` win wherever they differ**; `mockup-notes.md` lists the choices made where the specs were silent, and the questions for the owner. Same method as `design/mixers/` and `design/film/`: pages at 1440 × 900 built from the app's captured DOM and CSS, shot at 2× to JPEGs, graph at 100 % zoom.

**The pictures are drawings.** The repository holds no licence-clean portrait photograph, so the faces are neutral illustrations made for this folder by `html/faces.py` (shapes and gradients; nobody's likeness, no third-party image, nothing to record in a licence file). They carry a few small red spots, one mole and soft under-eye shadows so the overlays have something to point at. A real photograph would replace them in the final screenshots.

| File | What it shows |
|---|---|
| `01-add-node-menu.jpg` | The Add node menu in the node view with **Skin** (icon, name, line "Retouches the skin of each face.") after Film; a note that Skin is not in the simple view |
| `02-simple-view.jpg` | The simple main view of a photo that has a Skin node: no Skin controls; the existing note "This photo has 2 nodes…" and the Open nodes link |
| `03-skin-per-face.jpg` | Main → Skin → Output over a group of four people and a small one behind; Skin selected; Faces card (Face 2 highlighted on the photo and in the list); Face 2's controls with Blemish changed on that face; Copy to other faces; Protect card |
| `04-one-face-masks.jpg` | One portrait; **Show masks** on: skin (pink), under the eyes (blue), the even-skin core (green outline) and the legend |
| `05-protect-brush.jpg` | The Protect brush active: brush cursor and a stroke round the mole; "2 strokes", Clear on |
| `06-twelve-faces.jpg` | Twelve faces, **Same for all faces** on: every face ringed, the list scrolls (two and a half rows) |
| `states.jpg` | Sheet: node card normal, selected, bypassed, dead; the panel with 1, 4 (per face and Same for all), 12 and 24 faces, no faces, finding faces, a face too small, masks shown with the neck on; the copy for the tooltips and notes (2880 × 7688) |

Everything but `states.jpg` is 2880 × 1800. `index.html` is the walk-through.

## Files

- `html/pages/01.html` … `06.html` and `states.html` are the screens as pages. They load `../../../mixers/html/app.css` (the app's compiled stylesheet), `../../../nodes/html/nv.css`, `../../../mixers/html/mixers.css` and `html/skin.css` (the few new pieces: face crops, switch, notes, sheet). The pictures and the font come from `html/assets/` and `design/nodes/html/assets/`; keep the folders together. `html/assets/scene-*.svg` are the drawings.
- `html/build.py` writes the pages (Python 3 and `bs4`; it imports `design/mixers/html/build.py` for the shared pieces), `html/faces.py` draws the pictures and overlays, `shoot.mjs` renders them (`node shoot.mjs`, or `node shoot.mjs 03 04`). If Playwright wants another chromium build, point `PLAYWRIGHT_BROWSERS_PATH` at a scratch folder of symlinks to the installed build; do not run `playwright install`. None of it is product code.
- The switch is drawn by hand in `skin.css` in the look of glasscn's `Switch` (the mockup captures hold none); the face crops are plain buttons that the interface task builds from glasscn's toggle group.

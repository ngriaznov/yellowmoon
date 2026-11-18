# Blur node mockups

Mockups for the Blur node (node kind `blur`; the spec is `docs/33-blur.md`; task BL-01, 2026-11-18, waiting on the owner's review BL-H1). Same method as `design/colour-uniform/`, `design/mixers/` and `design/skin/`: pages at 1440 × 900 built from the app's captured DOM and CSS, shot at 2× to JPEGs, the graph at 100 % zoom. `index.html` walks through the pictures. `docs/33` wins wherever the two differ; `mockup-notes.md` lists what the drawings decided.

**The pictures are drawings, and the pictures of the result are computed.** The repository holds no licence-clean photograph that suits a background blur, so the scene is **synthetic**: a drawn street at dusk with string lights and a figure in a red jacket (nobody's likeness, no third-party image, none of the owner's photographs), made in linear light by `html/blur_demo.py` so that the lights are far brighter than white, as a raw file's are. Every "after" picture is **the maths of `docs/33` §3 applied to that scene** by `html/blur_ref.py` (numpy), through the inverted Subject mask (or a Depth mask for one picture). The display transform is a simple filmic curve and the sRGB encoding, not the product's. A real photograph would replace them in the final screenshots.

| File | What it shows |
|---|---|
| `01-soft.jpg` | Main → Blur → Output; the node selected, its Blur tab with the Type control (Soft), Radius 50 and the Mask edges card; the background blurred, the figure sharp |
| `02-lens.jpg` | Lens: Radius, Shape (6 blades), Rotation, Highlights; the lights are hexagons |
| `03-motion.jpg` | Motion: Length and Angle |
| `04-zoom.jpg` | Zoom: Amount, Pick centre, and the centre handle with its rays on the photo |
| `05-mask-tab.jpg` | The Mask tab: one Subject row, inverted, and its card (a corrector's Mask tab) |
| `06-show-mask.jpg` | Show mask on the node: the ordinary mask `m`, here the background, in the mask accent |
| `07-add-node-menu.jpg` | The Add node menu with Blur after Colour Uniform |
| `08-simple-view.jpg` | The simple view of a photo with the node: no controls, the existing notice |
| `09-respect-mask-edges.jpg` | **Respect mask edges off and on**, side by side, on the same picture and mask: the edge of the jacket three times larger, and the whole picture with the crop marked. Off: a red glow in the blurred street. On: none |
| `10-types.jpg` | Sheet: the scene, Soft, Lens, Motion and Zoom through one mask, and Soft with a Depth mask |
| `states.jpg` | Sheet: card states (normal, selected, bypassed, dead), five panel states and the copy of every tooltip and note (drafts) |

`01` to `08` are 2880 × 1800; `09`, `10` and `states` are full-page sheets at 1.5×.

## Files

- `html/pages/01.html` … `10.html` and `states.html` are the screens as pages. They load the app's compiled stylesheet and the other mock-ups' sheets from `design/mixers/html/`, `design/nodes/html/`, `design/skin/html/` and `design/colour-uniform/html/cc.css`, then `html/bl.css` (the few new pieces: the Type control's width, the Zoom centre row, the sheets). The pictures come from `html/assets/`; keep the folders together.
- `html/blur_ref.py` is a numpy sketch of the node's maths. **It is not product code**: the product's reference is `ym_color::blur` (BL-03), in f64 in Rust. `python3 -I blur_ref.py` runs the property checks of `docs/33` §8 on the sketch and prints their numbers (37 lines: identity at 0, mask 0, black picture, flat field, **nothing from outside the mask reaches inside**, and that the same test fails with the switch off so it bites, repeat, mirror symmetry, energy of a point, tile invariance).
- `html/blur_demo.py` draws the scene and writes the pictures and `assets/numbers.txt` (the share of the blurred colour that comes from the figure just outside its silhouette: 27 % with the switch off, 0 % with it on); `html/build.py` writes the pages (Python 3, `bs4`; it imports the Colour Uniform, skin and mixers builders); `shoot.mjs` renders them.
- To rebuild, in this folder (Playwright is found through `NODE_PATH`; if it wants another Chromium build, point `PLAYWRIGHT_BROWSERS_PATH` at a folder with the installed build and do not run `playwright install`):

```
python3 -I html/blur_demo.py     # numpy and PIL; about 50 s
python3 html/build.py
node shoot.mjs                   # or: node shoot.mjs 01 09
```

None of it is product code.

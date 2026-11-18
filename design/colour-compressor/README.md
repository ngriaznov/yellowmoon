# Colour Compressor node mockups

Mockups for the Colour Compressor node (node kind `color_compressor`; the spec is `docs/32-color-compressor.md`; task CC-01). They wait for the owner's review (CC-H1); no interface work (CC-09, CC-10) starts before it. **`docs/32` wins wherever the two differ**; `mockup-notes.md` lists what the drawings decided and the questions for the owner. Same method as `design/mixers/` and `design/skin/`: pages at 1440 × 900 built from the app's captured DOM and CSS, shot at 2× to JPEGs, the graph at 100 % zoom. `index.html` walks through the pictures.

**The pictures are drawings, and the compressed one is computed.** The repository holds no licence-clean portrait photograph, so the face is the neutral illustration of `design/skin/html/faces.py` (nobody's likeness, no third-party image). It has a green-olive cast on one side, so the node has something to even out. The "after" picture and the "affected area" picture are **the maths of `docs/32` §3 applied to the drawing** by `html/compress_ref.py` (numpy; the drawing's sRGB is read as scene light in place of AWG4, which is near enough for a mock-up). A real photograph would replace them in the final screenshots.

| File | What it shows |
|---|---|
| `01-node.jpg` | Main → Colour Compressor → Output; the node selected, its Compress tab with Target (swatch, Pick colour, Hue, Saturation, Lightness), Compress (60 / 60 / 0 %) and Range; the face after the pull |
| `02-pick-target.jpg` | A new node: Pick colour on, the empty swatch, the off fields and sliders with their line, the crosshair and the 7 × 7 square on the photo, the hint |
| `03-show-affected-area.jpg` | Show mask on the node: the weight (mask times range) in the mask accent; the eye in the Range card pressed |
| `04-add-node-menu.jpg` | The Add node menu with Colour Compressor after Face Refinement |
| `05-simple-view.jpg` | The simple view of a photo with the node: no controls, the existing notice |
| `06-how-the-pull-works.jpg` | A diagram drawn from the maths: the chroma plane with the range and nine colours before and after, and the law on the hue and lightness axes |
| `states.jpg` | Sheet: card states (normal, selected, bypassed, dead), six panel states, and the copy of every tooltip and note (drafts) |

Everything but `states.jpg` (2880 × 6798) is 2880 × 1800.

## Files

- `html/pages/01.html` … `06.html` and `states.html` are the screens as pages. They load `../../../mixers/html/app.css` (the app's compiled stylesheet), `../../../nodes/html/nv.css`, `../../../mixers/html/mixers.css`, `../../../skin/html/skin.css` and `html/cc.css` (the few new pieces: swatch, hint pill, diagram). The pictures come from `html/assets/`, the font from `design/nodes/html/assets/`; keep the folders together.
- `html/compress_ref.py` is a numpy sketch of the node's maths. **It is not product code**: the product's reference is `ym_color::compress` (CC-02), in f64 in Rust. `python3 compress_ref.py` runs the property checks of `docs/32` §8 on the sketch (identity, fixed point, no NaN, grey never moves, continuity across the hue wrap, monotone along each axis).
- `html/compress_demo.py` writes `portrait-before.jpg`, `portrait-after.jpg`, `portrait-affected.jpg` and `target.json` from `html/assets/portrait-before.png`; `render-scene.mjs` writes that PNG from the drawing in `design/skin/`; `html/build.py` writes the pages (Python 3, `bs4`, `numpy`, `PIL`; it imports the skin and mixers builders); `shoot.mjs` renders them.
- To rebuild, in this folder (Playwright is found through `NODE_PATH`; if it wants another Chromium build, point `PLAYWRIGHT_BROWSERS_PATH` at a folder with the installed build and do not run `playwright install`):

```
node render-scene.mjs
python3 html/compress_demo.py
python3 html/build.py
node shoot.mjs            # or: node shoot.mjs 01 03
```

None of it is product code.

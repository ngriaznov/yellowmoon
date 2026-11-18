# Colour Uniform node mockups

Mockups for the Colour Uniform node (node kind `color_uniform`; the spec is `docs/32-color-uniform.md`; task CU-01, revised 2026-11-18 after the owner's answers, CU-H1). `docs/32` wins wherever the two differ; `mockup-notes.md` lists what the drawings decided. Same method as `design/mixers/` and `design/skin/`: pages at 1440 × 900 built from the app's captured DOM and CSS, shot at 2× to JPEGs, the graph at 100 % zoom. `index.html` walks through the pictures.

**The pictures are drawings, and the picture of the result is computed.** The repository holds no licence-clean portrait photograph, so the face is the neutral illustration of `design/skin/html/faces.py` (nobody's likeness, no third-party image). It has a green-olive cast on one side, so the node has something to even out. The "after" picture and the "mask" picture are **the maths of `docs/32` §3 applied to the drawing** by `html/compress_ref.py` (numpy), blended through the **Colour range mask row that the first pick adds** (`docs/32` §2.3): `in + m (node(in) − in)`. The drawing's sRGB is read as scene light in place of AWG4, which is near enough for a mock-up. A real photograph would replace them in the final screenshots.

| File | What it shows |
|---|---|
| `01-node.jpg` | Main → Colour Uniform → Output; the node selected, its Compress tab with Target (swatch, Pick colour, Hue, Saturation, Lightness) and Compress (60 / 60 / 0 %); the face after the pull |
| `02-pick-target.jpg` | A new node: Pick colour on, the empty swatch, the target's fields on, the Compress sliders off with their line, the crosshair and the 7 × 7 square on the photo, the hint |
| `03-show-mask.jpg` | Show mask on the node: the node's mask, the seeded Colour range row, in the mask accent |
| `03b-mask-tab.jpg` | The node's Mask tab with the COLOUR RANGE row the first pick added, its card (Pick colour, Width, Softness, Opacity) and the View card |
| `04-add-node-menu.jpg` | The Add node menu with Colour Uniform after Face Refinement |
| `05-simple-view.jpg` | The simple view of a photo with the node: no controls, the existing notice |
| `06-how-the-pull-works.jpg` | A diagram drawn from the maths: the chroma plane with the reach of the hue pull and fourteen colours before and after, and the law on the hue and lightness axes |
| `states.jpg` | Sheet: card states (normal, selected, bypassed, dead), five panel states, and the copy of every tooltip and note (drafts) |

Everything but `states.jpg` and `06-how-the-pull-works.jpg` is 2880 × 1800.

## Files

- `html/pages/01.html` … `06.html`, `03b.html` and `states.html` are the screens as pages. They load `../../../mixers/html/app.css` (the app's compiled stylesheet), `../../../nodes/html/nv.css`, `../../../mixers/html/mixers.css`, `../../../skin/html/skin.css` and `html/cc.css` (the few new pieces: swatch, hint pill, diagram). The pictures come from `html/assets/`, the font from `design/nodes/html/assets/`; keep the folders together.
- `html/compress_ref.py` is a numpy sketch of the node's maths. **It is not product code**: the product's reference is `ym_color::uniform` (CU-02), in f64 in Rust. `python3 compress_ref.py` runs the property checks of `docs/32` §8 on the sketch and prints their numbers (identity, fixed point, 100 % and 50 %, no NaN, grey, near-neutral gain, continuity across the hue wrap, monotone along each axis, the Jacobian, the mask blend). The file keeps the name it had when the plan was written about Resolve's Color Compressor.
- `html/compress_demo.py` writes `portrait-before.jpg`, `portrait-after.jpg`, `portrait-affected.jpg` and `target.json` from `html/assets/portrait-before.png`; `render-scene.mjs` writes that PNG from the drawing in `design/skin/`; `html/build.py` writes the pages (Python 3, `bs4`, `numpy`, `PIL`; it imports the skin and mixers builders); `shoot.mjs` renders them.
- To rebuild, in this folder (Playwright is found through `NODE_PATH`; if it wants another Chromium build, point `PLAYWRIGHT_BROWSERS_PATH` at a folder with the installed build and do not run `playwright install`):

```
node render-scene.mjs
python3 html/compress_demo.py
python3 html/build.py
node shoot.mjs            # or: node shoot.mjs 01 03 03b
```

None of it is product code.

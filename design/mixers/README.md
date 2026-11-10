# Mixer node mockups

Mockups for the four mixer node kinds of `docs/25-mixers.md` (task X-02): Parallel, Layer, Splitter and Combiner. They wait for the owner's approval (X-H1); no interface work (X-06) starts before it. **The spec `docs/25` wins wherever the two differ**; `mockup-notes.md` lists what the spec left open and what drawing it showed up. Same method as `design/nodes/`: pages at 1440 × 900 built from the app's DOM and CSS, shot at 2× to JPEGs. Pictures are the owner's Provence photo from `design/nodes/html/assets/`; the thumbnails of the Red, Green and Blue correctors are the channels of Main's thumbnail, made for this folder. The graph is at 100 % zoom throughout.

| File | What it shows |
|---|---|
| `01-add-node-menu.jpg` | The Add node menu with Corrector, Film, a rule, then Parallel, Layer and Splitter and combiner enabled, each with its line, over a photo with four correctors |
| `02-parallel.jpg` | Main → Skin and Sky (masked) → Parallel → Output; the Parallel selected, its Inputs tab (what it does, where the inputs start, two inputs, no drag handles); the "+" port under its inputs |
| `03-layer-blend.jpg` | Warmth, Glow and Sky into a Layer; Blend tab: Mode, Opacity, the Inputs card with "On top", "Below", drag handles and the hint; no Mask tab |
| `04-layer-mode-menu.jpg` | The Mode menu open, the seven modes |
| `05-splitter-and-combiner.jpg` | The trio just added after Main (Splitter, Red, Green, Blue, Combiner); the Splitter selected with its Channels tab; channel ports with a tint and a letter |
| `06-combiner.jpg` | The same graph with the Combiner selected and its Channels tab |
| `07-drag-to-plus.jpg` | A picture link dragged from a corrector's output onto a Parallel's "+": valid inputs lit, key ports and upstream inputs dim |
| `states.jpg` | Sheet: Parallel and Layer normal, selected, bypassed, dead, 1 input, 8 inputs; Splitter and Combiner normal, selected, dead; three options for the channel ports; the tooltips of the refusals (2880 × 3396) |

Everything but `states.jpg` is 2880 × 1800 (1440 × 900 at 2×).

## Files

- `html/pages/01.html` … `07.html` and `states.html` are the screens as pages. They load `html/app.css` (the compiled stylesheet of the app as `design/nodes/html/pages/12.html` carries it), `design/nodes/html/nv.css`, and `html/mixers.css` (the app's own rules for cards, ports, links, the Add node menu and the node head, copied from `src/styles/glass-c2.css`, then the mockup-only additions). The pictures and the Figtree font come from `design/nodes/html/assets/`; keep the folders together.
- `html/build.py` writes the pages (needs only Python 3 and `bs4`; reads `design/nodes/html/pages/06.html` and `12.html`). `shoot.mjs` renders them (`node shoot.mjs`, or `node shoot.mjs 03 05` for some). Neither is product code.
- `index.html` is the walk-through; `mockup-notes.md` has the choices and the questions for the owner at X-H1.

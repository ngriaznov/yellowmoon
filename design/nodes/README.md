# Node view references

The approved mockups of the node view and of Pick white's new place (the owner, 2026-11-01). They are the layout, size and copy reference for milestone N; the spec is `docs/23-nodes.md`, and **`docs/23` wins wherever the two differ**. Where a mockup and `docs/23` are both silent, `docs/11` (the dark glass interface, `design/screenshots-c2/`) decides. `docs/algorithms/06-masks.md` has the maths; `research/nodes-masks-2026-11-01.md` the reasons.

**These are mockups, not screenshots of the app.** The pictures are the owner's own photos (folder "Provence", 7 photos, in capture order). **The masks, the overlays and the node thumbnails are hand-made pictures**, drawn to show where things go and what they look like, not what the core makes: a real Object mask, its edge and the thumbnails come from the renderer (`docs/23` §4 and §5). The interface around them was built from the app's real DOM and CSS (glasscn components, `design/tokens.css`), at 1440 × 900.

## The pictures

Each is a 2880 × 1800 JPEG (1440 × 900 at 2×); `index.html` shows them in order with a caption under each.

| File | What it shows |
|---|---|
| `01-simple-light.jpg` | The simple view on the Light tab with the new header: Before · Crop · **Nodes** · Add photos · Fit. Pick white has left the header |
| `02-simple-colour-pick-white.jpg` | The Colour tab: the White balance card with the new **Pick white** row (a toggle with an eyedropper, "or press W") between the light presets and the Kelvin slider |
| `03-simple-picking-white.jpg` | While picking: the toggle is pressed (yellow), the hint is a glass pill at the top centre of the photo ("Click something neutral: grey or white. Esc cancels."), the cursor is a crosshair |
| `04-simple-photo-has-nodes.jpg` | A photo with four nodes in the simple view: the notice under the tab row ("This photo has 4 nodes. These controls change the input, the output and node 1, Main." with **Open nodes**) and the nodes glyph on the filmstrip thumbnail |
| `05-nodes-default.jpg` | The node view of an untouched photo, layout A: the photo on top (666 × 444), the divider, the graph (Input, 01 Main, Output), the panel on Main's Light tab with the compact head |
| `06-nodes-add-node-menu.jpg` | The graph with four correctors (Main, Mountain, Sky, Warmth) and the **Add node** menu open: Corrector, then Parallel, Layer, Splitter and combiner and Outside greyed with "Later" |
| `07-nodes-object-mask.jpg` | 02 Mountain on the Mask tab: an Object row (Add, 2 points, selected) and a Radial row (Subtract, Feather 45); the Object card, the View card; the tinted overlay with the Radial's hole and handles on the photo |
| `07b-mask-rows-invert.jpg` | A crop of the Masks card with the Radial row's Invert pressed (in the mask accent): this mask alone is inverted; the Invert in the card header inverts the node's whole mask |
| `08-nodes-add-mask-menu.jpg` | The **Add mask** menu: Find (Object, Subject, People, Background), Shapes (Linear, Radial, Brush), Keys (Colour range, Brightness range) |
| `09-nodes-linear-mask.jpg` | 03 Sky on the Colour tab with its Linear mask on the photo (a solid centre line, dashed start and end lines, a rotate handle); the Temperature and tint trim card, Colour balance and Colour warp |
| `10-nodes-input-colour.jpg` | The Input selected, on its Colour tab: only the White balance card, with Pick white. Tabs RAW · Lens · Light · Colour · Detail |
| `11-nodes-output-look.jpg` | The Output selected, on its Look tab: today's Look card (Intensity, Apply to, search, the libraries). Tabs Look · Detail |
| `12-later-graph.jpg` | **Marked Later, not built.** A non-linear graph (branches, a Parallel and a Layer mixer, an Outside node fed by a key output) and the Layer mixer's inspector. It shows what the format and the engine are built for; the mixers and Outside come in a later version |

## The pages

The same screens as pages that open in a browser at 1440 × 900, so an agent can read the DOM, the sizes and the copy:

- `html/pages/01.html` … `12.html` are the screens above (`05b.html` and `07c.html` are variants kept from the drawing; use `05` and `07`; `states.html` is a sheet of the node card's states: normal, selected, AI mask, shape or key mask, bypassed, key output linked, a mixer).
- `html/nv.css` is their stylesheet and `html/assets/` their pictures and the Figtree font. The pages load `../nv.css` and `../assets/`, so keep the folders together.
- `index.html` is the walk-through of the JPEGs with a caption under each.
- `mockup-notes.md` has the drawing notes with the exact geometry (layout A: the photo 666 × 444, the divider 10, the editor 371 with a 44 px toolbar, Expand 545; the head 52 px, the tab row 46). The notes were written before several decisions: **where they differ from `docs/23`, `docs/23` wins** (they mention the list view and the "Graph · List" switch, describe the graph as a chain, and list open questions that `docs/23` has answered).

## What the mockups draw that is not built

- **The "Graph · List" switch.** The toolbar of screens 05 to 12 and of the pages has a **Graph · List** switch after Delete. It was **dropped with the list view** (the owner, 2026-11-01). The node toolbar has no such switch: Add node ▾ · Bypass · Delete on the left; the photo switcher, Show mask and Expand on the right (`docs/23` §7.3).
- **The list view itself** (screen 13, "Option: list view of the graph") was dropped by the owner. Its picture is not in this folder; nothing refers to it.
- **A third segment "Off"** in the View card (and the tooltip "M steps through the three"). The View card has two segments, Overlay and Mask only; the toolbar's Show mask and the key M turn the view on and off (`docs/23` §7.6).
- **The graph as a chain.** The screens draw a chain with straight links. The data is a real graph (nodes, typed ports and links, `docs/23` §3); screen 12 shows what that allows later. In this version the links cannot be drawn by hand.
- **Screen 12** as a whole, and the "Later" tag in its toolbar.

## Copy

Every word on the screens is the copy of `docs/23` and `docs/11` unless `docs/23` §14 (decision 17) lists a line as new. The look names in screen 11 are invented ("Your LUTs": Film warm, Film cool, …) so that no real ARRI look name appears; the section "ARRI looks · LogC3 to Rec.709" is the app's own name for an imported library, filled from the person's own import.

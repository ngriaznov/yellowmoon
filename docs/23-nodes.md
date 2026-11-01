# 23 · Nodes and masks

Nodes give a photo a grade that acts on part of it. A **Nodes** button in the header opens the node view: the photo on top, a graph of nodes below it, today's panel on the right. The photo's light is read by a fixed **Input**, graded by **corrector nodes** (each with its own light, colour and curve controls and with **masks**: shapes, ranges, painted strokes and masks that find an object, the people or the background), and made into a picture by a fixed **Output** (the look and the display rendering). The simple view stays the default and works as it does today. **The settings themselves are the graph** (the owner, 2026-11-01): a photo's settings are nodes and links, the Input, Main and the Output are nodes like the others, and the simple view reads and writes the simple graph. This file covers what the person sees, the settings (a node-first document of nodes and links), where every control lives, how masks combine, the rendering and its plan, the node view, Pick white's move into the Colour tab, the IPC, the tests and the tasks.

The maths is in `docs/algorithms/06-masks.md` (the node's order, the two trims, the mask combination, every shape, the raster pipeline, the SAM 2.1 runtime). The reasons and sources are in `research/nodes-masks-2026-11-01.md`. The owner's decisions of 2026-11-01 are in `docs/16` (*Nodes and masks*).

**Status: Spec, from the plan approved by the owner on 2026-11-01. Built on branch `nodes`, released as 0.2.0.** The whole plan is built on the branch and released once; the checkpoints (N-H2 to N-H4) are test builds, not releases. The work is milestone N of `docs/15`. `PIPELINE_VERSION` goes from 0.8.2 to **0.9.0** (section 1.4).

**The mockups.** `design/nodes/` holds the approved pictures (`01` to `12`, `07b`) and the same screens as pages that open in a browser at 1440 × 900 (`design/nodes/html/pages/*.html`, built from the app's DOM and CSS, with `nv.css` and `assets/`), with `design/nodes/README.md` and `design/nodes/mockup-notes.md`. Use them for layout, sizes and copy. **This file wins where they differ**: the "Graph · List" switch drawn in the toolbar was dropped with the list view and is not built; the mockup's three-way "Overlay · Mask only · Off" is two (section 7.6); the settings are a real graph (section 3), where the mockups only draw a chain; and the notes' open questions are answered here. Where a mockup and this file are both silent, `docs/11` (the dark glass interface, `design/screenshots-c2/`) decides.

## 1. What it does

### 1.1 What is built, and what is not

- **A node view behind a Nodes button** (header, shortcut N). Today's interface is the default and is unchanged.
- **Corrector nodes** in a graph between a fixed Input and a fixed Output, up to 64 nodes per photo, Main included.
- **Masks in every corrector.** Shapes, ranges and AI masks add up; each row can be subtracted or inverted, and so can the node's whole mask.
- **AI masks.** Object (SAM 2.1 small, from a click or a box), Subject and People (Apple Vision, macOS 14 or later, only where this Mac supports them), Background (our blur map).
- **A real graph in the data and the engine.** Nodes, typed ports and links are the format from version 1, and the core evaluates a compiled plan (section 5.2). The interface edits the graph by inserting, deleting, reordering and moving nodes, and by **drawing links by hand**: drag from an output port to an input port, as in Resolve and other node editors (the owner, 2026-11-01; section 7.2).
- **Pick white moves** from the header into the Colour tab (section 8).

**Not in this version.** Parallel, Layer, Splitter and combiner, and Outside nodes (the Add node menu shows them as "Later"; the format, the validator and the plan are built so they can be added as new node kinds without changing the format); DCTL and OpenFX; presets and nodes shared across photos; the list view (dropped by the owner); a "pick neutral" in a corrector.

### 1.2 The three zones

| Zone | What it does | Today's stages | Space out |
|---|---|---|---|
| **Input** (fixed) | Turns the camera's data into the scene | R1–R9 raw, L1–L4 lens, C1–C6 white balance, sensor map and camera matrix with exposure, C7 highlight range, C8 LogC4 encode | LogC4 / AWG4 |
| **Nodes** | The grade | per node: trims, C9 primary grade, C9b colour warp, C11 curve, then the mask blend | LogC4 / AWG4 |
| **Output** (fixed) | Look and display | C10 look, C12 cine rendering, output sharpening, C13 encode, export | Display or file |

Nodes work in LogC4 / AWG4 between a fixed Input and a fixed Output; there is no per-node colour space. The look is set once per photo, on the Output, and acts after every node. The crop, the straightening and the quarter turn (`crop`, `orientation`) belong to none of the three zones: they act on the picture's frame while it is sampled.

### 1.3 Inside a corrector node

The steps run in this order (`docs/algorithms/06` section 1):

1. **Trims**, in linear AWG4: an exposure trim in stops (decode, multiply by 2^ev, encode) and a temperature and tint trim (a von Kries gain along the daylight locus, section 2.2 of the maths file).
2. **C9** primary grade: contrast, brightness, the four tonal zones, colour balance and saturation (`docs/02` C9).
3. **C9b** colour warp (`docs/02` C9b).
4. **C11** curve (`docs/02` C11).
5. **The mask blend:** `out = in + m · (node(in) − in)`, in LogC4, where `in` is the node's input and `m` its mask (section 4.2). The look's intensity blends the same way.

A step at its identity is skipped, so a node at defaults changes no bit.

### 1.4 What changes for existing photos

Putting the look on the Output moves the curve in front of the look. Until now the order was C9 → C9b → **C10 look → C11 curve**; it is now C9 → C9b → **C11 curve** (inside each node) → **C10 look** (in the Output).

- A photo with **both** a curve that is not the identity **and** a look other than Neutral renders a little differently. Every other photo renders exactly as before.
- No golden entry has a curve, so the goldens do not move.
- Looks that include a display rendering (ARRI's LogC3-to-Rec.709 looks) already ran the curve first, so the two kinds of look now follow one rule.
- The owner accepted this on 2026-11-01. `PIPELINE_VERSION` goes to **0.9.0**, written to `ym:PipelineVersion` as before. `docs/02` C10 and C11 say so.
- **Nothing else changes for a photo's pixels.** Making the settings node-first moves fields between objects and moves no pixel: every golden entry and settings fixture, rewritten to the node-first form, renders the bytes of 0.8.2 (section 11), apart from the curve-and-look case above. The settings of Yellowmoon 0.1 were never saved, so there is nothing to carry over; its flat form is refused (section 3.12).

### 1.5 The owner's decisions

The decisions of 2026-11-01 (way in, layout, graph, colour space, tabs, look on the Output, masks, overlap maths, SAM 2.1 small bundled, Vision only where supported, what a copy does, Main, the limit of 64 nodes, the release as 0.2.0, the real graph from the start) are the table of `docs/16` *Nodes and masks*. They are final; this file turns them into a build.

## 2. Where every control lives

The simple view is the **simple graph**: its tabs are views of **three nodes**, the Input, **node 1, Main** and the Output, which every photo has. A tab reads a node's fields through the store's selectors (`inputOf`, `mainOf`, `outputOf`; `src/state/settingsGraph.ts`) and writes them with a patch of that one node (`patchOf`, section 3.7), with the fields it has always had and the types it has always had; nothing visible changes. A document without `nodes` is the default graph: Input → Main → Output. Auto (Apple Auto, Claude Code, Antigravity) keeps writing the Input and Main only.

| Control in the simple view (tab › card › row) | Node | Field |
|---|---|---|
| Look › Automatic (the rows, Re-run) | Input and Main | reads and resets the Auto values of the fields below |
| Look › Look (the tiles, Intensity, search, Reset) | Output | `look`, `look_intensity` |
| Look › Look › Apply to | Output | copies `look` and `look_intensity` to other photos |
| Look › Tone › Temperature | Input | `white_balance` (Kelvin) |
| Look › Tone › Contrast | Main | `grade.contrast` |
| Look › Tone › Saturation | Main | `grade.saturation` |
| Lens › Lens, Amounts | Input | `lens` |
| Light › Exposure › Exposure | Input | `exposure` |
| Light › Exposure › Contrast, Brightness | Main | `grade.contrast`, `grade.brightness` |
| Light › High dynamic range › Highlights, Shadows, White, Black | Main | `grade.highlights`, `grade.shadows`, `grade.white`, `grade.black` |
| Colour › White balance (light presets, Pick white, Kelvin, Tint) | Input | `light_source`, `white_pick`, `white_balance` |
| Colour › Colour balance (three wheels, Saturation) | Main | `grade.color_balance`, `grade.saturation` |
| Colour › Colour warp | Main | `grade.color_warp` |
| Detail › Sharpening › Amount, Radius | Input | `sharpen` |
| Detail › Sharpening › Output | Output | `output_sharpening` |
| Detail › Noise reduction | Input | `noise_reduction` |
| Curve | Main | `grade.curve` |
| RAW › Camera (read only) | Input | none |
| RAW › Decode | Input | `decode` |
| RAW › Range | Input | `rolloff` |
| Header › Crop, Crop panel | Input (its framing) | `crop`, `orientation` |
| Header › Before | none (a view) | none |

What no simple-view control reaches is in the same document: Main's name, bypass, trim, masks and card position, and every other node, link and mask (section 3).

## 3. The settings

### 3.1 Shape

**The settings of a photo are the graph and nothing else** (the owner, 2026-11-01: "We need to make sure that nodes will be part of settings that can be copied and pasted. Actually current settings should become node-first and basic UI just reads simple node graph."). `DevelopSettings` has one field:

```json
{ "nodes": { "items": [ … ], "links": [ … ] } }
```

- `items`: the nodes, in no particular order (written by id). Every graph has **exactly one Input** (id `"input"`, kind `input`), **exactly one Output** (id `"output"`, kind `output`) and **Main** (a corrector, id `1`). They are items like the others; the Input and the Output cannot be added to or taken out.
- `links`: the links, from an output port to an input port (section 3.4).
- **Nothing is top-level.** What used to be a settings field is now a field of one node, **with the same name and the same type**: the Input holds the raw, light and framing settings, the Output the look, and a corrector (Main included) holds its grade (section 3.2). The simple view reads and writes them through the Input, Main and the Output of the default graph (section 2).
- **A document is read over the default graph.** A document may hold only what it changes: the three fixed nodes need not be listed, a listed node's missing fields take their defaults, and `links` left out are the default links (Input → Main → Output); `links` given **replace** them whole. `{}` is the default graph. A node with an id the default graph does not have must give its `kind`.
- **Order when written.** The Input first, then the correctors by id, then the Output; the links in the order of their ends (from-node in that order, then from-port, to-node, to-port); object keys sorted. The same graph is the same bytes (hard rule 8).
- **The default graph** is Input → Main → Output with every field at its default: Auto where there is Auto, the user's default look on the Output, no trim, no masks.
- **There is no format marker.** The key `nodes` is the marker: a document that has it is node-first, and a document that has none of the keys of section 3.12 is the default graph. Anything else at the top level is an unknown field and is refused. A marker (`"format": 2`) will be added the first time the node-first form itself has to change; adding one now would only be one more field to refuse.
- **The form of Yellowmoon 0.1 is not read** (section 3.12).

A full example, the graph of screen 07 of the mockups: the Input with an exposure of +0.3 EV, Main at its defaults, three correctors in a line (node 2, Mountain, has an Object mask minus a Radial; node 3, Sky, a Linear mask; node 4, Warmth, a trim) and the Output with the Signature look. A corrector other than Main has plain values (section 3.2, *Auto belongs to Main*):

```json
{ "nodes": {
  "items": [
    { "id": "input", "kind": "input", "pos": [-1, 0],
      "exposure": { "kind": "manual", "ev": 0.3 },
      "light_source": "auto",
      "white_balance": { "kind": "auto" },
      "white_pick": null,
      "rolloff": { "highlights": 0.6, "low_light": 0.5 },
      "decode": { "neural_demosaic": true, "highlight_recovery": true,
                  "lens_corrections": true, "chromatic_aberration": true },
      "noise_reduction": { "kind": "auto" },
      "sharpen": { "kind": "auto" },
      "lens": { "kind": "default", "flares": 0, "edges": 0, "anamorphic": 0 },
      "crop": null,
      "orientation": 0 },
    { "id": 1, "kind": "corrector", "name": "Main", "bypass": false, "pos": [0, 0],
      "trim": { "exposure_ev": 0, "temperature": 0, "tint": 0 },
      "grade": {
        "contrast": { "kind": "auto" }, "brightness": 0,
        "highlights": { "kind": "auto" }, "shadows": { "kind": "auto" },
        "white": 0, "black": { "kind": "auto" },
        "color_balance": { "shadows": { "kind": "auto" }, "midtones": [0, 0],
                           "highlights": { "kind": "auto" } },
        "saturation": { "kind": "auto" },
        "color_warp": { "pins": [] },
        "curve": { "luma": [[0, 0], [1, 1]], "red": [[0, 0], [1, 1]],
                   "green": [[0, 0], [1, 1]], "blue": [[0, 0], [1, 1]] } },
      "masks": { "invert": false, "items": [] } },
    { "id": 2, "kind": "corrector", "name": "Mountain", "bypass": false, "pos": [1, 0],
      "trim": { "exposure_ev": 0.35, "temperature": 0, "tint": 0 },
      "grade": {
        "contrast": { "kind": "manual", "value": 0 }, "brightness": 0,
        "highlights": { "kind": "manual", "value": -12 },
        "shadows": { "kind": "manual", "value": 0 },
        "white": 0, "black": { "kind": "manual", "value": 0 },
        "color_balance": { "shadows": { "kind": "manual", "point": [0, 0] },
                           "midtones": [0, 0],
                           "highlights": { "kind": "manual", "point": [0, 0] } },
        "saturation": { "kind": "manual", "value": 0 },
        "color_warp": { "pins": [] },
        "curve": { "luma": [[0, 0], [1, 1]], "red": [[0, 0], [1, 1]],
                   "green": [[0, 0], [1, 1]], "blue": [[0, 0], [1, 1]] } },
      "masks": { "invert": false, "items": [
        { "kind": "object", "mode": "add", "invert": false, "enabled": true, "opacity": 1,
          "feather": 0, "grow": 0, "refine": 0.5,
          "points": [[0.41, 0.33, true], [0.52, 0.31, true]], "box": null },
        { "kind": "radial", "mode": "subtract", "invert": false, "enabled": true, "opacity": 1,
          "center": [0.55, 0.32], "radii": [0.06, 0.03], "angle_deg": 0, "feather": 0.45 } ] } },
    { "id": 3, "kind": "corrector", "name": "Sky", "bypass": false, "pos": [2, 0],
      "trim": { "exposure_ev": 0, "temperature": -18, "tint": 4 },
      "masks": { "invert": false, "items": [
        { "kind": "linear", "mode": "add", "invert": false, "enabled": true, "opacity": 1,
          "center": [0.5, 0.2], "angle_deg": 4, "feather": 0.25 } ] } },
    { "id": 4, "kind": "corrector", "name": "Warmth", "bypass": false, "pos": [3, 0],
      "trim": { "exposure_ev": 0, "temperature": 30, "tint": 0 },
      "masks": { "invert": false, "items": [] } },
    { "id": "output", "kind": "output", "pos": [4, 0],
      "look": "builtin:signature", "look_intensity": 100, "output_sharpening": 100 }
  ],
  "links": [
    { "from": { "node": "input", "port": "picture" }, "to": { "node": 1,        "port": "picture" } },
    { "from": { "node": 1,       "port": "picture" }, "to": { "node": 2,        "port": "picture" } },
    { "from": { "node": 2,       "port": "picture" }, "to": { "node": 3,        "port": "picture" } },
    { "from": { "node": 3,       "port": "picture" }, "to": { "node": 4,        "port": "picture" } },
    { "from": { "node": 4,       "port": "picture" }, "to": { "node": "output", "port": "picture" } }
  ] } }
```

(The core writes every field; here `grade` is left out of Sky and Warmth to keep the example short, which reads as a grade at its defaults.) What a person writes by hand may be much shorter, because a document is read over the default graph: this one sets the Input's exposure, Main's contrast and the Output's look, and nothing else.

```json
{ "nodes": { "items": [
  { "id": "input",  "exposure": { "kind": "manual", "ev": 0.3 } },
  { "id": 1,        "grade": { "contrast": { "kind": "manual", "value": 20 } } },
  { "id": "output", "look": "builtin:neutral" } ] } }
```

A graph with a **key link**, a node that only makes a mask for another (a dead picture branch with a live key): node 5 reads Main's picture, selects skin colours with a Colour range, and its key output feeds node 3's key input, which gives node 3 that mask as an extra region (section 4.2):

```json
{ "id": 5, "kind": "corrector", "name": "Skin", "bypass": false, "pos": [1, 1],
  "masks": { "invert": false, "items": [
    { "kind": "color_range", "mode": "add", "invert": false, "enabled": true, "opacity": 1,
      "center": [0.06, 0.04], "width": 0.05, "softness": 0.5 } ] } }
```
```json
{ "from": { "node": 1, "port": "picture" }, "to": { "node": 5, "port": "picture" } },
{ "from": { "node": 5, "port": "key" },     "to": { "node": 3, "port": "key" } }
```

### 3.2 The nodes

Every item has `id`, `kind` and `pos`. **`id`** is `"input"`, `"output"` or an integer 1 to 2³¹ − 1 (Main is 1); it is stable within the photo and is used by links, selection, thumbnails and undo. A number on a card is not the id: it is the node's place in the flow (section 7.2). New ids are one more than the largest the page has used in the session, so an id is not reused after a delete. **`kind`** is `input`, `output` or `corrector` (the id `"input"` is always kind `input`, and so on); any other kind is refused (section 3.5), and Parallel, Layer, Splitter and combiner and Outside will be new values. **`pos`** is `[x, y]`, finite numbers in −1000 to 1000, in grid units: the card's place on the canvas (one unit is 214 px across and 150 px down at 100 % zoom). A document may leave it out: the core then places the nodes in a row by their number, the Input first and the Output last, at y = 0 (the default graph is −1, 0 and 1). It is always written, it travels with copy and paste and with undo like the rest, and it lives in memory only.

A node has only the fields of its kind below; another field is an unknown field and is refused.

**Input** (`kind` `input`; fixed name, no bypass). Today's names and types, moved:

| Field | Type / range | Default | Simple view |
|---|---|---|---|
| `exposure` | `Auto` or `Manual { ev }` −4…+4, step 0.01 | Auto | Light › Exposure |
| `light_source` | `auto`, `day`, `shade`, `tungsten`, `led`, `fluorescent` | `auto` | Colour › White balance (segmented) |
| `white_balance` | `Auto` or `Manual { kelvin 2000–11000, tint −50…50 }` | Auto | Colour › White balance; Look › Tone |
| `white_pick` | `null`, or `[x, y]` in [0, 1]²: the Pick white point (`docs/01`) | `null` | Colour › White balance › Pick white |
| `rolloff` | `{ highlights 0–1, low_light 0–1 }` | `{ 0.6, 0.5 }` | RAW › Range |
| `decode` | `{ neural_demosaic, highlight_recovery, lens_corrections, chromatic_aberration }` bools | all true | RAW › Decode |
| `noise_reduction` | `Auto` or `{ luminance, color, detail }` 0–100 | Auto | Detail › Noise reduction |
| `sharpen` | `Auto` or `{ amount 0–300, radius_px 0.3–2.5 }` | Auto | Detail › Sharpening (Amount, Radius) |
| `lens` | `{ kind, flares, edges, anamorphic }` (`docs/21`) | no lens | Lens |
| `crop` | `null`, or the crop (`docs/01`: rect, angle, aspect) | `null` | Crop |
| `orientation` | 0, 90, 180 or 270 | 0 | Crop › Rotate 90° |

The crop and the turn are the Input's framing: they are data of the Input, and they still act when the picture is sampled (after the Output, `docs/01` *Geometry*).

**Output** (`kind` `output`; fixed name, no bypass):

| Field | Type / range | Default | Simple view |
|---|---|---|---|
| `look` | a look id | the user's default look (Signature out of the box) | Look › Look |
| `look_intensity` | 0–100 % | 100 | Look › Look › Intensity |
| `output_sharpening` | 0–200 %, a plain number | 100 | Detail › Sharpening › Output |

**Corrector** (`kind` `corrector`; Main is the corrector with id 1):

| Field | Type / range | Default | Notes |
|---|---|---|---|
| `name` | text, 1 to 32 characters after trimming | `"Corrector"` (Main: `"Main"`) | A longer name is cut; an empty one reads as the default. Long names are cut at about ten letters on a card, with the whole name as its tooltip |
| `bypass` | bool | false | A bypassed node passes its picture through. Its masks still exist (its key output still carries `m`). Main can be bypassed |
| `trim` | `{ exposure_ev −4…4, temperature −100…100, tint −100…100 }` | all 0 | `docs/algorithms/06` section 2. A partial object is read over the defaults |
| `grade` | the grade below | all defaults | May be left out |
| `masks` | `{ invert: bool, items: [mask] }` | `{ false, [] }` | Section 3.3 |

**The grade** has the fields and the types the settings have always had for them: `contrast`, `highlights`, `shadows`, `black` and `saturation` are `Auto` or `Manual { value }` in −100…100; `brightness` and `white` are plain numbers in −100…100; `color_balance` is `{ shadows, midtones, highlights }` with the shadows and highlights wheels `Auto` or `Manual { point }` and the midtones wheel a point `[a, b]` in the unit disc (longer vectors are scaled back to it); `color_warp` is the warp (`{ pins }`, at most 16 pins, each clamped as `docs/01` says); `curve` is the curve object (luma, red, green, blue, 2 to 16 points each, the identity by default). They mean what the same-named fields have always meant.

**Auto belongs to the photo's Main.** Auto (the Automatic card, Apple Auto, Claude Code, Antigravity, `docs/19`, `docs/22`) grades the photo's Input and Main and writes nothing else. So:

- In **Main** (id 1) a grade field may be Auto, and a missing one is Auto; this is what the default graph has, and what Re-run puts back.
- In **any other corrector** Auto **reads as the neutral value and is written as a manual 0**: `{ "kind": "auto" }` for a contrast, a highlights, a shadows, a black or a saturation is read as `{ "kind": "manual", "value": 0 }`, and an Auto wheel as `{ "kind": "manual", "point": [0, 0] }`. A field left out of such a node is a manual 0 too, and **a new node is manual 0 everywhere** (`graphOps.addNode`). The reading is not an error and is not reported as clamping; it is done by every path that makes a node, a document, a patch, a paste.
- The tests of section 11 pin it: a document with Auto in node 2 reads, writes and renders as a manual 0; Main keeps its Auto through a round trip; a patch that sets `{ "kind": "auto" }` on node 3 stores a manual 0; a pasted graph keeps Main's Auto in the target's Main and gives node 3 its manual 0.

### 3.3 Masks

`masks.items` is the list of rows, in the order they are drawn and combined (section 4.2), at most 64. `masks.invert` is the node's Invert. Every row has:

| Field | Type / range | Default |
|---|---|---|
| `kind` | `object`, `subject`, `people`, `background`, `linear`, `radial`, `brush`, `color_range`, `brightness_range` | none |
| `mode` | `"add"` or `"subtract"` | `"add"` |
| `invert` | bool, this mask alone | false |
| `enabled` | bool; off (the eye) counts as absent | true |
| `opacity` | 0 to 1 | 1 |

and, by kind (sizes are fractions of the photo's long edge; points are in the photo frame, section 3.6):

| Kind | Fields (range, default) |
|---|---|
| `object` | `points` `[[x, y, positive]]` (at most 256; `[]`), `box` `[x0, y0, x1, y1]` or `null` (`null`), `feather` 0…1 (0), `grow` −1…1 (0), `refine` 0…1 (0.5) |
| `subject`, `people` | `feather` (0), `grow` (0), `refine` (0.5) |
| `background` | `level` 0…1 (0.5), `softness` 0…1 (0.5), `feather` (0), `grow` (0), `refine` (0.5) |
| `linear` | `center` `[x, y]` ([0.5, 0.5]), `angle_deg` −180…180 (0), `feather` 0…1 (0.2) |
| `radial` | `center` ([0.5, 0.5]), `radii` `[rx, ry]` 0.002…1 ([0.25, 0.18]), `angle_deg` (0), `feather` 0…1 (0.3) |
| `brush` | `strokes` `[{ size 0.001…0.25 (0.03), hardness 0…1 (0.5), erase (false), points [[x, y]] }]` (at most 100,000 points in all; `[]`), `feather` (0), `grow` (0), `refine` (0) |
| `color_range` | `center` `[a, b]` in the C9 chroma plane, length at most 0.5 ([0, 0]), `width` 0.005…0.3 (0.05), `softness` 0…1 (0.5) |
| `brightness_range` | `low` 0…1 (0.35), `high` 0…1 (1), `softness` 0…1 (0.5); `low` above `high` is read as both at their mean |

The numbers are what the sliders show divided or scaled as `docs/algorithms/06` section 4.5 says. An Object row with no `points` and no `box` **selects nothing**.

### 3.4 Ports and links

| Node | Input ports | Output ports |
|---|---|---|
| Input (`"input"`) | none | `picture` |
| a corrector (`"corrector"`) | `picture`, `key` | `picture`, `key` |
| Output (`"output"`) | `picture` | none |

A `picture` carries the LogC4 picture as the node before made it. A `key` carries a mask. **A node's key output carries its final mask `m`** (after its Invert; it does not depend on its Bypass). **A node's key input feeds its mask as one more region term**: an implicit first row, added like a mask, with the linked node's `m` as its value (`docs/algorithms/06` section 3). A link names its two ends: `{ "from": { "node": <id | "input">, "port": … }, "to": { "node": <id | "output">, "port": … } }`.

A corrector's picture output may have several links (a card feeding its next node and a branch); an input has at most one.

### 3.5 Validation, clamping and limits

A document or a patch that breaks the **graph rules** is refused with a plain error, nothing is applied (patches are atomic), and the photo keeps its settings. The core checks them on every read and every patch (`DevelopSettings::validated`, after the patch is applied); the page's own edits cannot break them.

| Rule | Error text |
|---|---|
| the graph has one Input (`"input"`), one Output (`"output"`) and Main (id 1, a corrector) | "A photo needs its input, its output and node 1, Main." |
| a node has only the fields of its kind | "A corrector has no field called “exposure”." (the unknown-field error of every document) |
| a link goes from an output port to an input port | "A link goes from the output of one node to the input of another." |
| a link joins ports of the same type, picture to picture or key to key | "A link joins a picture to a picture, or a key to a key." |
| both ends exist: a node of `items` | "A link names node 7, which this photo does not have." |
| the ports exist on those kinds (section 3.4) | "Node 7 has no port called “mask”." |
| an input port has at most one link | "An input can have one link. Node 3 has two on its picture input." |
| the Output's picture input has exactly one link | "The output needs one picture link." |
| Main takes its picture from the Input | "Node 1, Main, must take its picture from the input." |
| no cycles, counting picture and key links together | "The nodes of this photo are linked in a circle. Remove a link." |
| ids are unique | "Two nodes have the id 3." |
| node kinds are known | "This settings file has a node of a kind (“parallel”) that this version of Yellowmoon does not have. Update Yellowmoon." |
| at most 64 corrector nodes, Main included (the Input and the Output do not count) | "A photo can have up to 64 nodes." |
| at most 64 masks in a node | "A node can have up to 64 masks." |
| at most 100,000 brush points in a mask, 256 points in an Object mask | "A brush mask can have up to 100,000 points." / "An Object mask can have up to 256 points." |
| a patch does not remove the Input, the Output or Main | "The input, the output and node 1, Main, can't be removed." |
| a patch names a node it neither has nor adds (no `kind`), or gives a `kind` other than the node's | "The patch names node 7, which this photo does not have." / "Node 7 is a corrector, not an input." |

Nodes that are not on a path to the Output are **allowed and kept**; they do nothing, and their cards are dimmed, as in Resolve (a node whose key output feeds a live node is live). Links are bounded by the ports: an input has at most one link, so a graph has at most 2 × 64 + 1 = 129 of them. The 64-node limit is the working limit; the other numbers are **sanity bounds** on what a settings file may hold (a broken or hostile file given to `ym --settings`, or a patch, cannot hang the app or use all its memory), a person never reaches them, and a document over one is refused like the rest. The colour warp keeps its own limit of 16 pins per node.

**Clamping, as for every field.** Numbers outside their range are clamped and reported in `PatchReport::clamped` (as `<node>.<field>`, for example `1.grade.contrast`); a number that is not finite is `SettingsError::NotFinite("nodes.items[2].trim.tint")` and nothing is applied; an unknown field is `SettingsError::Parse`. Points of a mask are clamped to the photo frame (0 to 1), a colour-balance vector longer than 1 is scaled back, and a curve is checked as `curve` always was (2 to 16 points, sorted by x; two points at one x are refused). Names are trimmed and cut. Auto in a corrector other than Main reads as a manual 0 (section 3.2) and is not clamping.

### 3.6 Coordinates and sizes

Points and shapes are stored in the **photo frame**, normalised: the photo as the camera oriented it (EXIF orientation applied), after the camera's own lens corrections (R9: distortion and the crop that hides the empty corners) and **before** the lens character's bend and zoom (L4), the user's crop, the straightening and the quarter turn. A mask stays on its subject when the photo is cropped or turned, or the lens sliders move, and the core converts between the frame and the screen (`viewport_map_points`, section 9).

(The plan said "the frame `white_pick` uses". `white_pick` is in a slightly different frame, the oriented image area *before* the lens crop, `Renderer::pick_point_at`; Pick white keeps its own, and masks use the frame above, which is the Input's own picture, the one the AI masks read: decision 2 of section 14.)

**Sizes** (feather, grow, brush, radii) are fractions of the photo frame's long edge, so a preview and an export at any size agree. A render's `u_px` is the number of output pixels along that long edge at the render's scale.

### 3.7 Patches, Reset and undo

**A patch is node-level.** `SettingsPatch` is

```json
{ "nodes": { "<id>": NodePatch, … }, "remove": [<id>, …], "links": [ <link>, … ] }
```

each part optional:

- **`nodes`** maps node ids (as strings: `"input"`, `"output"`, `"1"`, `"2"`) to a `NodePatch`: **a partial of that node's fields, nested as today's patches are** (`decode`, `rolloff`, `lens`, `trim`, `grade`, `color_balance` and `curve` may carry only the parts to change; a manual `white_balance`, `sharpen` or `noise_reduction` may leave out a part, which keeps its current resolved value). So `{ "nodes": { "input": { "exposure": { "kind": "manual", "ev": 0.3 } } } }` moves one slider, and `{ "nodes": { "1": { "grade": { "contrast": { "kind": "manual", "value": 12 } } } } }` another, on Main. A node whose id the graph does not have is **added** when the patch gives its `kind` (and any of its fields; the rest are defaults); a node that exists is changed.
- **`remove`** lists the nodes to delete (corrector nodes other than Main). The links that touch a removed node go with it.
- **`links`**, when present, **replaces the links whole** (at most 129 small objects). Adding, deleting, reordering and linking nodes are patches that carry `nodes`, `remove` and `links` together.
- **A patch is atomic**: it is applied to a copy of the settings, the result is validated (section 3.5), and nothing changes if anything is refused. `apply_patch` reports the fields that changed, as `FieldRef { node, field }`, and the first render-stage group they affect (`SettingsField::group()`): the Input's `decode`, `noise_reduction` and `sharpen` re-run the raw group, `lens` the lens group, `exposure`, `light_source`, `white_balance`, `white_pick` and `rolloff` the colour group, `crop` and `orientation` only the geometry; a corrector's `trim`, `grade`, `bypass` and `masks`, and the links, the nodes' group; the Output's `look` and `look_intensity` the look group and `output_sharpening` the last stage; **`name` and `pos` re-run nothing**.
- **A slider drag sends one node's partial, never the graph.** The whole-graph patch is for copy and paste and for undo of a change that touched many nodes.

**Masks.** `masks.items` is a list, replaced whole by a patch that gives it (add, remove or reorder rows). Two smaller operations keep a drag small:

- `masks.edit` maps a row's index (a string) to a **row patch**: a partial of that row's fields (`{ "nodes": { "2": { "masks": { "edit": { "0": { "feather": 0.3 } } } } } }`). A row patch cannot change the row's `kind`.
- A **Brush** row also takes `strokes_append` (strokes added after the existing ones) and `strokes_keep` (an integer: keep only the first n strokes). **A stroke is sent as an append when the pointer is released**, so a mask's other strokes are never resent: `{ "masks": { "edit": { "0": { "strokes_append": [ { "size": 0.03, "hardness": 0.5, "erase": false, "points": [[0.4, 0.5], …] } ] } } } }`. Its inverse for undo is `strokes_keep` with the old count. An Object row's `mask_find` answer is a row patch of `points` and `box` (at most 256 points).
- `masks.invert` is the node's Invert.

**Reset** (`SettingsPatch::reset`) names a node and a group of its fields and gives the patch that puts them at their defaults (Auto where there is Auto, for the Input and Main). The node head's Reset (section 7.5) is a reset of the node's trim and grade (the Input's and the Output's: all their fields).

**Undo and redo** are the webview's (`PhotoHistory`, 100 steps). **An entry diffs per node and for the links**: it holds the node patches of the nodes a change touched (added nodes whole with their `kind`, removed ones in `remove`) and `links` when they changed, as the two patches `before` and `after`, so a step holds only what it touched. A slider drag is one entry with one node's fields; a node added, deleted, reordered or linked is one entry; a brush stroke is one entry, `strokes_append` and `strokes_keep`. `pos` is part of it: moving a card is one step. Unchanged nodes are not copied into the entries.

**Re-run** (`reset_automatic`) resets the Auto fields of the Input and Main and leaves the rest of the graph alone. **Apply to** copies the Output's `look` and `look_intensity` only (`{ "nodes": { "output": { "look": …, "look_intensity": … } } }`), as it always copied the look and its intensity.

The webview builds patches with pure functions (`src/state/graphOps.ts`, section 7.2; `src/state/settingsGraph.ts`: `inputOf`, `mainOf`, `outputOf` and `patchOf(node, partial)`), and the core takes any valid patch.

### 3.8 Copy settings and Paste

**Copy takes the whole graph: every node and every link**, names, positions, trims, grades and masks included, as a patch that holds every node whole, `links`, and nothing to remove. It leaves out three fields of the Input, which belong to that photo's own pixels and frame (`COPY_LEAVES_OUT`: the Input's **`crop`, `orientation` and `white_pick`**), and then:

- Auto stays Auto in Main and in the Input (each photo keeps its own automatic exposure, white balance, sharpening and noise). A white balance that came from a picked white is copied as the manual Kelvin and tint it resolved to, as before (`docs/11`).
- Linear, Radial, Brush, Colour range and Brightness range rows are copied as they are.
- **Subject, People and Background rows are worked out again on the new photo**: the row is the same, its raster is made from the new photo's pixels.
- **Object clicks are not copied**: the pasted Object row arrives **empty** (no `points`, no `box`), so it selects nothing, and its row reads "Click the object on this photo". The node then changes nothing until you click, instead of grading the whole photo.

**A paste replaces the target's graph** in one undo step in that photo's history: the patch gives the source's nodes whole (the Input's and Output's fields except the three above, which the target keeps), `remove` lists the target's corrector nodes that the source does not have, and `links` are the source's. The result is validated like any patch. The copy is kept in the webview's memory for the running app, never on disk or the system clipboard, and it carries the whole graph. **Apply to** copies the Output's look and intensity only.

The rules are `SettingsPatch::copied_from` and `Session::paste_settings` in the core, not the webview's; the mock repeats them (`copiedPatch`).

### 3.9 Resolved settings, XMP, DNG and `ym`

- **Resolved settings are node-first too.** `ResolvedSettings.settings` is the stored document with Auto kept; `values` is the same graph with every Auto replaced by the number it resolves to (the Input's `exposure_ev`, `kelvin`, `tint`, `sharpen` and `noise_reduction`; Main's `contrast`, `highlights`, `shadows`, `black`, `saturation` and the wheels' points; the other correctors' values as they are); **`auto`** says which Auto fields are still automatic, grouped by node (`auto.input`: `exposure`, `light_source`, `white_balance`, `sharpen`, `noise_reduction`; `auto.main`: `contrast`, `highlights`, `shadows`, `black`, `saturation`, `balance_shadows`, `balance_highlights`); **`estimates`** keeps what belongs to the photo (`source`, `provider`, `apple`, `apple_steps`, `light_class`, `light_confidence`, `noise`) and groups the numbers Auto resolves to by the node they are for (`estimates.input`: `exposure_ev`, `kelvin`, `tint`, `sharpen`, `noise_reduction`; `estimates.main`: `contrast`, `saturation`, `highlights`, `shadows`, `black`, `balance_shadows`, `balance_highlights`).
- **XMP.** `ym:Settings` is the node-first document, pinned like the rest: every Auto of the Input and Main that has a number is written as that number (exposure, contrast, saturation, highlights, shadows, black, the wheels, sharpening, noise reduction; a white balance stays Auto when it was), with every default written out, so a file's settings reproduce its pixels through `--settings`. A JPEG's XMP packet holds 65,502 bytes: when the pinned document would not fit (a graph with a lot of brush work), the JPEG writer leaves `ym:Settings` out and adds the note "The settings of this photo are too long for a JPEG's XMP, so they are not recorded." to the export report (`notes` in `ym render --json`); TIFF, PNG and DNG have no such limit. (The page keeps a stroke's points few: section 7.6.)
- **DNG.** The nodes and their masks are baked into the pixels, like today's grade: the DNG is C1 to C11, nodes included, then the look (`docs/09`). A raster mask is evaluated on the DNG's own grid (the photo frame, uncropped).
- **`ym`.** `--settings` reads the node-first form, links included: a hand-written file may hold key links, dead branches and any valid shape, and `ym` renders it as the app does. **There are no new flags: every existing settings flag now sets a field of a node** (the Input, Main or the Output; the table is in `docs/10`), and `ym render --json` prints the photo's settings in the node-first form. A mask this machine cannot make stops that photo with a plain error (section 3.10). `docs/10` is the CLI's own document and is kept in step with this one.
- **Versions.** A document with a node kind this version does not know is refused with the text of section 3.5, so an older Yellowmoon never renders a newer graph wrongly; a document in the form of Yellowmoon 0.1 is refused (section 3.12).

### 3.10 Subject and People only where this Mac supports them

Subject and People are built for macOS only: the Vision code sits behind a macOS gate in `ym-apple`, and builds for other systems contain none of it (as for the Apple Auto bridge). At run time the core asks this Mac whether it can run the two Vision requests, once, and caches the answer (as Apple Auto's availability probe does; the probe makes each request on a 64 × 64 picture). Where the answer is no (an older macOS, or a Mac that cannot run them), **Subject and People are not in the Add mask menu at all**: hidden, not greyed out. `ym` follows the same answer.

A settings document that has one anyway (copied from another Mac, or made on Linux) cannot be rendered faithfully:

- **In the app** the mask row says "Subject masks need a Mac that supports them" (or "People masks need a Mac that supports them") and selects nothing; the photo renders, with that row selecting nothing, and an export of it is **refused** with the same sentence, never written silently without the mask.
- **In `ym`** the photo stops with a plain error that names the mask and why: `error.kind` `mask`, "Node 2 has a Subject mask, and this computer cannot make one. Use a Mac that supports Subject masks, or remove the mask." The other photos of the run go on, and the exit code is 1 (`docs/10`).

### 3.11 Nothing persists

Nothing about nodes is saved per photo (hard rule 4 of `AGENTS.md`). The graph lives in the session's memory like every setting, and so do the SAM features, the Vision masks, the blur maps and the masks found: all kept in small in-memory caches, never on disk. The only thing that persists is what `docs/01` *Persistence* already lists; nothing new is added to it.

### 3.12 Documents of Yellowmoon 0.1

Yellowmoon 0.1 kept no settings per photo (hard rule 4): they lived in the session, and the only copies that ever left it were the `ym:Settings` in the XMP of an exported file and what `ym render --json` printed (Copy settings kept its copy in the session's memory only). In those the settings were **flat**: every field at the top level (`look`, `exposure`, `contrast`, `crop`, …). **The core reads only the node-first form.**

- **A flat document is refused with a plain error.** A document that has no `nodes` and whose keys are all fields of the 0.1 form (`look`, `look_intensity`, `exposure`, `contrast`, `brightness`, `highlights`, `shadows`, `white`, `black`, `light_source`, `white_balance`, `color_balance`, `saturation`, `color_warp`, `lens`, `sharpen`, `output_sharpening`, `noise_reduction`, `curve`, `decode`, `rolloff`, `crop`, `orientation`, `white_pick`, and the retired `white_offset_ev`) is refused with: **"This settings file is in the old form of Yellowmoon 0.1. Settings are now a graph of nodes: `ym render --help` shows the new form, and `ym render --json` prints a photo's settings in it."** (`SettingsError::OldForm`; the text is checked word for word by the test of section 11, which also compares it with `docs/10`.) It is a usage error in `ym` (exit 2, before any photo is opened) and an error in the app's `update_settings`.
- **Nothing is carried over.** Nothing was saved, so there is nothing to read in the old form. A document that has `nodes` and any other top-level key is refused as having an unknown field.
- **The repository's own documents are node-first.** N-02 rewrites the golden manifest's `settings` (`tools/golden/manifest.json`; `{}` is the default graph and stays valid), `truth_settings` in `tools/golden/src/truth.rs`, the test fixtures, the CLI's tests and the schema test; no fixture keeps the old form except the one test of the refusal.
- **The schema** (`crates/ym-core/schema/settings.schema.json`) describes the node-first form only.

## 4. Masks

### 4.1 Kinds

| Group | Kind | How it is made | Copies to another photo |
|---|---|---|---|
| Find | Object | SAM 2.1 small, from points (Add point, Remove point) or a box (section 4.4) | No: arrives empty |
| Find | Subject | Apple Vision foreground mask; macOS only, and only where this Mac supports it (sections 3.10 and 4.5) | Worked out again |
| Find | People | Apple Vision person mask; the same conditions | Worked out again |
| Find | Background | Our blur map (`bokeh_net`), with a level and a softness (section 4.6) | Worked out again |
| Shapes | Linear | Start and end lines with a feather; analytic | Yes |
| Shapes | Radial | An ellipse with a feather and an angle; analytic | Yes |
| Shapes | Brush | Strokes with size, hardness and erase, drawn into a mask | Yes |
| Keys | Colour range | A chroma centre, width and softness, read from the node's input; picked by clicking the photo | Yes |
| Keys | Brightness range | Low, high and softness on the node's input | Yes |

The menu calls the last group **Keys**; this file and the maths call its rows **ranges**, because *key* also names the ports and links of section 3.4. A Find item is listed only once its phase is built and only where the core can make it (`mask_kinds`, section 9).

### 4.2 How they combine

The formulas are in `docs/algorithms/06` section 3. In words:

1. Each mask gives `a`. It is inverted if its own Invert is on, then scaled by its Opacity.
2. A linked **key input** is an implicit first region row, added with the linked node's mask as its value.
3. **Regions** (the key input, Find and Shapes) combine in list order. Add: `R = 1 − (1 − R)(1 − a)`. Subtract: `R = R · (1 − a)`. The screen rule is the owner's choice for overlapping soft edges.
4. **Ranges** (Keys) combine among themselves the same way into `K`.
5. **The node's mask** is `m = R · K` (a range keeps only its part of what the regions gave; a node with only ranges uses the range alone). With no regions `R = 1`; with no ranges `K = 1`; with nothing at all `m = 1`, the whole photo.
6. The node's Invert gives `m = 1 − m`.
7. A switched-off row (the eye) counts as absent.

A row that selects nothing (an Object row with no point, a Subject row on a Mac that cannot make one) is present with `a = 0`, so a node whose only mask is empty changes nothing.

### 4.3 Edges that never depend on a rounding, and raster masks

**Smooth edges.** Every edge, shape or range, is a smooth ramp at least one output pixel wide, even at Feather 0, so the GPU and the CPU agree within the gate and Metal's fast maths cannot flip a pixel (`docs/17` K8). Shapes use `max(feather, one pixel)`; ranges use `RANGE_RAMP_MIN = 0.02` of a LogC4 code value (maths, section 4).

**Raster masks** (Object, Subject, People, Background, Brush) go through the steps of `docs/algorithms/06` section 5:

1. The mask is made once in the photo frame at a working size (SAM 1024 px on the long edge; Vision's own output size, from a 1024 px picture; Background the blur map's 512 px; Brush 2048 px) and kept in memory.
2. At render time it is resampled to the output grid with the view's geometry (crop, straightening, turn, the lens's bend).
3. **Grow** and **Feather** are applied in photo-relative sizes.
4. **Refine edges** runs an edge-aware guided filter, guided by the Input's picture at that size, so the edge follows the hair or the ridge rather than SAM's coarse outline.
5. This is the only spatial step in the grade, so tiled exports give it a halo, and tiles still equal the whole frame bit for bit.

### 4.4 Object, and SAM 2.1 small

- **Model.** SAM 2.1 *small* (46 million parameters; the owner's choice of 2026-11-01), for images only. The image encoder runs once per photo; the prompt decoder runs per click. The video parts (memory attention and the rest) are not needed and are not built (`docs/07`, maths section 6).
- **Runtime.** Our own pure-Rust code, f32 on the CPU, as for our three shipped models. The weights are safetensors with SHA-256 checks. It is deterministic: fixed order, no fused multiply-add, `libm`, independent of the thread count; so the app and `ym` give the same mask. Python is used only to convert the weights and to make test fixtures from Meta's own code (the oracle); nothing in the app runs Python.
- **Speed** (estimates, to be measured on a Mac by N-17 and N-20): the encoder about 1.5 to 5 s a photo on Apple silicon (the small model has more encoder blocks than the tiny one), started in the background as soon as an Object row exists on the photo (the core sees it in the patch that makes the row); each click then about 50 to 150 ms. If the encoder is too slow, the fallback is to run it on the GPU, still deterministic within a stated bound, before N-18.
- **Weights.** Bundled in the app, about 92 MB at f16, too big for git. They are converted once, published as an asset of a Yellowmoon GitHub release pinned by SHA-256, fetched by the build and bundled; the app stays offline and needs no first-run download (hard rule 7 of `AGENTS.md`). The model card, `SHA256SUMS` and the Apache-2.0 notice are in `data/models/sam2/`; the weights are not (`docs/12`, `docs/14`).
- **A click** is sent with `mask_find` (section 9). The core adds the point to the row's points and runs the decoder; if a mask is found the answer carries the new `points` (or `box`) and the page applies them as one row patch (`masks.edit`, section 3.7), **one undo step**. The pill at the top of the photo reads "Object found · 0.4 s" (the time of the decoder, one decimal) for 3 s. The click is **refused** when the best mask is not an object (the object score is not above 0) or covers under 0.05 % of the photo.
- **Nothing found:** the click adds no point, no empty mask is kept, nothing changes, and the error bar says "Nothing found there. Click another part of the object, or draw a box."
- **Alt-click** takes a part away: the click is a negative point. The Object card's tool (Add point, Remove point, Box) sets what a plain click does (section 7.6).
- **Memory.** SAM's features take about 16 MB a photo and the photo-frame rasters 4 to 8 MB each; both live in a small in-memory cache that the renderer keeps for the photos it keeps (three), never on disk.

### 4.5 Subject and People

On a Mac that supports them (section 3.10): Subject is the union of the instances of `VNGenerateForegroundInstanceMaskRequest` and People that of `VNGeneratePersonInstanceMaskRequest`, both macOS 14.0 and later, on the Input's picture at 1024 px (`docs/algorithms/06` section 5.3). The code is Swift behind the macOS gate in `crates/ym-apple` with a C ABI like the Apple Auto bridge (`@_cdecl("ym_apple_mask_*")`), built for both architectures of the universal binary where the SDK has the requests; the Foundation Models bridge keeps its own gate. Vision is Apple's on-device framework and sends nothing anywhere (hard rule 7). Vision may answer differently on another macOS version; within one Mac the app and `ym` agree, and the Linux goldens cannot cover it.

### 4.6 Background

From the blur map that lens character makes (`ym_ml::bokeh_net`, the classical detector as its fallback), thresholded with a **Level** (default 50: where the map's own decision falls) and a **Softness**, then through the raster steps (`docs/algorithms/06` section 5.4). The photo's blur map is made when a Background row exists, once per photo, and is the one lens character uses when both are on. A photo with no out-of-focus area gives an empty mask.

### 4.7 What AI masks look at

SAM, Vision and the blur map read the photo **as the Input renders it**: the Input's picture (C1 to C8), the Neutral look, the cine rendering and no output sharpening. No node grade, no look the person chose and no output sharpening reach them. **Grading a node never moves its own mask or another node's.**

## 5. Rendering

### 5.1 Order

Per node: trims, C9, C9b, C11, mask blend (section 1.3). After the nodes, in the plan's order: C10 look, C12 cine rendering, output sharpening, C13 (`docs/02` *Stage order*). The curve is before the look, with `PIPELINE_VERSION` 0.9.0 (section 1.4). A bypassed node, a node at its identity and a node with `m = 0` everywhere are skipped as a whole.

### 5.2 The plan

The core **compiles the graph into a plan** (`ym_color::plan`; `docs/algorithms/06` section 1.3). The CPU reference evaluates that order; the kernels read the same plan.

- **Live and dead.** A node is live when the Output needs its picture (it is on a chain of picture links from the Input to the Output), or when a live node's key input needs its mask. Dead nodes are not in the plan (they do nothing, and their cards are dimmed).
- **Order.** A topological order over the picture and key links, with ties broken by Main first, then the card's `pos` (x then y), then the id. A card's number is its place in the same order over *all* nodes, dead ones included.
- **Registers.** Each step reads a picture register and writes one, and reads and writes key slots. All of it is per pixel.
- **One pass.** A run of per-pixel nodes on one picture path **fuses into the one colour pass** with the node loop, as planned; keys fed through key links are kept per pixel in that pass. This version has no joins, so **every v1 graph compiles into one pass** unless it needs more than 8 pictures or 16 keys alive at once (only a hand-written graph with many branches can).
- **Several passes.** Joins (the mixers of a later version: Parallel, Layer, Splitter and combiner, Outside) will split the plan into several passes with f32 buffers between them. **The compiler is written for that from the start**: it knows passes, the values that cross from one to the next and the buffers that carry them, and v1 only ever asks it for one, except for the over-wide case above, which has its own test.
- **Stops.** A thumbnail is the plan compiled to end at a node's picture and then run through the Output; the mask view is the plan compiled to end at a node's `m`.

### 5.3 CPU reference

`ym_color::display::ColorChain` gains the plan: its steps (each a node's `NodeParams` with the trim gain, the grade, the warp, the baked curves and the mask rows), then the Output. Each node runs section 1.3 in f64. The raster masks have CPU references for the resample, the grow, the feather and the guided filter (`ym_gpu::cpu::mask_raster`). The settings become parameters in `ym-core/src/pipeline/color.rs` (`color_plan`, `grade_plan`, `grade_params`); every node's grade, Main's included, comes from its item in `nodes`, and the Input's and the Output's fields from theirs.

### 5.4 GPU

**One fused pass, as today.** The node parameters, trims, zones, warp pins and mask shapes move from the uniform block into a **storage buffer** (the node records and the plan's steps); the kernel runs the plan's steps in f32 registers, so there are no lower-precision buffers between nodes.

- **Bindings** (`docs/17` K3 allows 8 storage buffers and 1 uniform): today's five storage buffers (the dither tables, the curves, the look LUTs, the picture in and out) and one uniform; the node records (+1 storage, binding 6); the raster masks for the frame (+1 storage, binding 7), u16 planes packed two to a word. The curve tables become **one table set per node** in the existing curves buffer (4 × 1024 f32 a node). That is **7 of the 8** storage buffers, plus the one uniform, whose node fields (the grade and the warp tail) have left it. If a later node type needs more, the plan is split across two passes with f32 buffers between them (section 5.2).
- **Registers and keys.** `PLAN_REGISTERS = 8` picture registers and `PLAN_KEYS = 16` key slots as private arrays; a plan that needs more is split.
- **A step** is `{ node record, picture register in, picture register out, key slot in, key slot out }`. A record holds the trim gain (three f32), the grade parameters, the warp pins, the base of its curve tables, and the node's mask rows (`MaskOp` records: kind, mode, invert, opacity, eight parameters, the raster plane's index).
- **Stops.** `CU.mode.x` keeps its values; value 11 stops after the node `CU.mode.y` and runs the Output (thumbnails); value 12 ends at that node's `m` (the mask view, section 5.8).
- **Identity.** A node at its identity, a bypassed node and a graph whose only node is Main at defaults run no node code beyond Main's own C9, C9b and C11, on the expressions of 0.8.2, so a default graph gives lavapipe the same bytes as before wherever the kernel's expressions are unchanged.
- **Parity gates** keep today's numbers: 2e-4 in LogC4 for the grade, now with up to 16 nodes and 8 masks each; one 10-bit code after C12 and C13; for the raster masks a stated bound set in N-13 (starting number: 4e-4 absolute on the mask plane).

### 5.5 The raster mask pass

Resample, grow, feather and guided filter run **before the colour pass**, as kernels with CPU twins (`mask_raster.wgsl`): first the Input's luma plane for the tile and its halo (the guide), then each raster row's plane (steps 2 to 5 of section 4.3), written as u16 planes that the colour pass reads. The CPU reference quantises to u16 in the same way, so the planes are bytes, not approximations. A tile's planes are made over the tile grown by the halo `H` (maths section 5.2) and cut. At most 48 planes are alive in one tile job (96 MiB at 1024²); a frame with more raster rows than that is split by the plan compiler into passes of fewer nodes.

### 5.6 The render graph

`docs/01` *Render graph* has the groups and keys. In short:

- A change to a corrector's fields, to the links, or to the nodes that exist (any patch of `nodes`, `remove` or `links`), rebuilds the **grade group** (the compiled plan and the per-node parameters; its key holds the graph, every node's values and the look); it never re-runs the raw group. Only the colour pass runs again.
- A raster row's **photo-frame raster** has its own key (the row's source: the points and box for Object, the macOS build for Vision, the blur map's key for Background, the strokes for Brush) and is kept per photo. Feather, Grow and Refine edges re-run only steps 2 to 5; Opacity, Mode, Invert and Bypass re-run nothing but the colour pass.
- Changing the look, the crop or a slider of another node never makes a raster again.

### 5.7 Node thumbnails

Each card shows the picture after its node: the plan stopped there, then the Output (so a thumbnail looks like the photo the person knows), rendered at about 160 px on the long edge on the thumbnail path (`pipeline/thumbs.rs`), one thumbnail for every node of the photo, dead ones included. They are made **after a change lands, not during a drag** (the page asks when a final patch lands and when a node is added or removed) and are **kept in memory**, keyed by the photo, the node and the plan up to it, so a change late in the graph makes no new thumbnail for the nodes before it. `node_thumbnails(photo)` is answered by a `node_thumbnails_ready` event (section 9); they are PNGs on the `ym-frame` scheme like the look tiles.

### 5.8 Show mask

The core renders it: the photo is a native layer under a transparent page, and the page cannot tint it. The toolbar's **Show mask** (key M) turns it on and off; the View card chooses **Overlay** or **Mask only**. With the mask view set, the viewport frame is the plan compiled to the selected node's `m`:

- **Overlay** tints the selected node's mask in the mask accent (orchid pink, `#FF6FD2`) over the photo as it is now: `out = display + 0.55 · m · (accent − display)` in display-linear, after output sharpening and before C13.
- **Mask only** shows `m` as grey: the code value after C13 is `m`.

The mask view is a view: it is not in an export, the Before view, a thumbnail or the pixel readout. It is made on the device with the colour pass when the renderer's stages allow, and on the CPU otherwise, with the same gates as a view (one 10-bit code). **Shape handles and SAM points are drawn by the page over the photo,** like the crop frame, in the cell's CSS pixels (`viewport_map_points`).

### 5.9 Speed targets (on a Mac; checked in N-20)

- A slider move in the fit view with **8 nodes**: within 100 ms, the target the lens spec already uses.
- A click on an Object mask: the mask shows within **300 ms** once the encoder has run.
- The fused kernel is already large; a 16-node plan with masks may cost time on older Macs (N-04 and N-10 measure it, and the plan's identity skipping keeps untouched nodes free).

### 5.10 Determinism and tiles

The same input, settings and version give the same bytes (hard rule 8). Tiles equal the whole frame bit for bit, the raster masks' halos included; the plan, the registers and the raster passes depend on a pixel's own input and its receptive field and on nothing else (`docs/17` D4 to D8). The app and `ym` give byte-identical exports for the same graph, whatever the masks, Object included (SAM is deterministic) and Subject and People on the same Mac.

## 6. The simple view

The simple view is today's Photos screen with these changes and no others. **It reads the simple graph:** every tab is a view of the Input, Main or the Output of the photo's graph through the store's selectors (section 2), and nothing visible changes from them.

- **Header.** Before · Crop · **Nodes** · Add photos · Fit. **Pick white has left the header** (section 8). Nodes is a toggle like Crop; its tooltip reads "Nodes (N)"; it is pressed in the node view.
- **A photo that has anything the simple view cannot show:** the panel shows, under the tab row, a notice in a glass pill with the nodes glyph: "This photo has 4 nodes. These controls change the input, the output and node 1, Main." and a link, **Open nodes**, at the end of the text (it takes two lines, so the pill does not grow to three). The number is the number of corrector nodes in the graph. When the photo has one node, Main, and Main has a trim, masks or a bypass, the notice reads "Node 1, Main, has a trim, masks or a bypass that these controls don't show. Open nodes". It is shown on every tab. **Open nodes** switches to the node view with Main selected.
- **The filmstrip thumbnail** of such a photo carries a small glass glyph (the nodes icon, bottom left of the thumbnail, a tooltip with "4 nodes" or "Node 1 has masks or a trim").
- **Main.** The controls of the simple view write Main's grade, so a bypassed Main ignores them; the notice says so ("a bypass").
- The photo shows the result of every node.
- Everything else is unchanged.

## 7. The node view

**The references:** `design/nodes/` (the paragraph *The mockups* at the top says how to use them). The copy below is word for word as in the mockups unless a line says it is new; copy that the mockups do not have is listed in section 14 (decision 17). User-facing text is British, short and plain, with no exclamation marks.

### 7.1 Layout A

The photo is on top with a draggable divider under it; the graph sits under the divider; the panel stays on the right at 400 px and full height; the filmstrip is hidden.

| Part | At 1440 × 900 | Rule |
|---|---|---|
| Header | 75 px, as today | unchanged: Before · Crop · **Nodes** (pressed) · Add photos · Fit |
| Photo cell | 666 × 444, 3:2, centred in the left column | the height left over under the divider; always 3:2 |
| Divider | 10 px, a grab handle in the middle | `role="separator"`, accessible name "Resize the graph", ↑/↓ move it 10 px (⇧ 50), double click returns to 45 % |
| Graph editor | 371 px: a 44 px toolbar and a 327 px canvas | **45 %** of the height under the header, `round(0.45 · (H − 75))` |
| Panel | 400 px, full height | today's, editing the selected node |

- **Expand** makes the editor **66 %** of the height under the header (545 px; the photo shrinks to 405 × 270) and back. The divider may be dragged between 25 % and 70 %, and the photo cell keeps at least 200 px of height. At 1728 × 1117 the rules give a photo of about 844 × 563 and an editor of 1328 × 469. The divider and Expand are session state, not saved (nothing persists).
- **The viewport follows.** The photo cell reports its rectangle to the native viewport as it does today (`viewport_set_rect`, on mount, resize, DPI change and a divider drag; the shell throttles). Fit is recalculated for the smaller cell. The page paints nothing over the cell except the overlays inside it (the look or mask chip, the camera line, the pill, the mask handles).
- **The filmstrip is replaced by a switcher** in the toolbar, "‹ 2 of 7 ›". ← and → still switch photos when no control that uses them has focus. Picks and Remove from filmstrip are out of sight: P still picks, and Delete never removes a photo in the node view.
- **Export, Copy settings and Paste** act on the whole photo in the node view too; the foot row stays under the panel.
- **Crop and Export** replace the panel as now. While one shows, the graph area is dimmed and read only.
- **Photo-cell chips.** The look chip (bottom left, "SIGNATURE · 100%") is unchanged. While the selected node has a selected mask row it reads "MASK · OBJECT" instead (the kind's name in capitals: OBJECT, SUBJECT, PEOPLE, BACKGROUND, LINEAR, RADIAL, BRUSH, COLOUR RANGE, BRIGHTNESS RANGE). The camera line (bottom right) is unchanged. The pill at the top centre is the "Object found" pill of section 4.4 and the hint of section 8.

### 7.2 The graph

The graph component is **`@xyflow/react`** (MIT, on our licence allow list), with our own node cards in glass (`src/components/glass/`). It is chosen now because the mixers of a later version will need its links, panning and zoom. It goes in `docs/14` and `THIRD-PARTY.md`.

**Markers.** The Input and the Output are fixed round markers of about 32 px with a lock glyph, the name under them ("Input", "Output"), a square output port at the Input's right and an arrowhead port at the Output's left. They are selectable. They sit one grid unit left of the leftmost card and one right of the rightmost (section 3.2).

**Cards.** About 150 × 112 px at 100 % zoom:

- the node's name above the card (cut at about ten letters, the whole name as its tooltip; double click to rename, tooltip "Double-click to rename");
- a thumbnail band 81 px high (the picture after the node, section 5.7);
- a footer row 30 px high with the number chip ("01") at the left and the mask glyphs at the right: a sparkle with the tooltip "AI mask" when the node has an enabled Object, Subject, People or Background row, and a half-filled disc with the tooltip "Mask" when it has any other enabled mask row;
- selected: a 2 px yellow ring; bypassed: the thumbnail at half strength and a slash glyph in the footer; **dead** (not on a path to the Output and not feeding a live key): the whole card at 45 % with the tooltip "Not connected to the output.".

A card's **number** is its place in the flow: the order of section 5.2 over all nodes. **Main is always 01.** Numbers update on insert and delete. The names are not numbers: "Corrector" is the default name and several nodes may have it.

**Ports and links.** Drawn on every node, as decided ("fine with clutter for now"). A picture input is a light grey triangle at the card's left, a picture output a light grey square at its right (at the thumbnail's mid height); a **key** input is a triangle and a key output a square in the mask accent (orchid pink), at the left and right of the footer row. Picture links are grey lines with a double arrowhead into the next input; key links are dashed in the mask accent. **Links are drawn by hand** (the owner, 2026-11-01: "Links needs to be drawn by hand, drag from output to input, as it is done in Resolve and many other node editing-based applications"); the rules are under *Links by hand* below, and the node operations that follow also rewire.

**What you can do in the graph.**

- **Select** (click, or the arrow keys: the graph is one Tab stop, and the arrows move to the neighbouring card in that direction).
- **Add after the selected node** (Add node ▾, Corrector): the new node goes **into the selected node's outgoing picture link**. With S selected and a link S → T, the result is S → N → T; if S has several outgoing picture links, N goes into the one that leads to the Output, else the first; if S has none, N is linked after S alone. With the Input selected, N goes after Main (Main stays first); with the Output selected, into the Output's incoming link. N's `pos` is S's plus one unit to the right, and the nodes downstream of N move one unit right. It is selected, with its Light tab open. It is off at 64 nodes, with the tooltip "A photo can have up to 64 nodes."
- **Delete** (the toolbar, the key Delete, More › Delete node): any corrector but Main. **The neighbours are reconnected**: with U → D → T₁ … Tₙ, the result is U → Tᵢ for each; the links of D's key ports are removed (a node that had D's key as its key input loses that input). The previous node is selected. For Main the toolbar's Delete is off with the tooltip "Main can't be deleted. Bypass it instead."
- **Bypass** (toolbar, the head's icon, the key B): toggles the selected corrector's bypass; Main too. Off for the Input and the Output, with the tooltip "The input and the output can't be bypassed."
- **Rename** (double-click the name on the card or in the head, or More › Rename): Return accepts, Esc cancels.
- **Move a card on the canvas** by dragging it (changes `pos`; nothing is rewired).
- **Reorder:** drag a card and release it with its centre over another picture link, which lights up (or More › Move earlier / Move later, or ⌥← and ⌥→ with the card focused): **the node is taken out of its place and the neighbours reconnected, then put into that link**, the same two operations as Delete and Add; its key links stay. Main cannot be reordered, and nothing can be put before Main (the link Input → Main is fixed). A move that would make a circle (the node's key output feeds a node that would then be upstream of it) is refused, and the error bar says "That would link the nodes in a circle."
- **Zoom, pan and Fit.** The canvas has a zoom control at the bottom right ("−", "100%", "+", "Fit"; accessible names "Zoom out", "Zoom in", "Fit the graph"), from 25 % to 200 %; drag on empty canvas to pan; a two-finger scroll pans and a pinch or ⌘-scroll zooms (the page never scrolls, `docs/11` desktop feel). At 100 % four cards fit in the editor with 63 px between them; more need zoom or Expand.
- With only Main in the graph the canvas says, under the Main → Output link, "Add a node to change only part of the photo."

**Links by hand.** As in Resolve:

- **Connect.** Press on an output port (a picture square or a key square) and drag: a link follows the pointer, the input ports that can take it light up (the same type, and no circle would result) and the others dim. Release over a lit input to link them. Release anywhere else and nothing changes. Picture goes to picture and key to key; a port of the other type never lights.
- **An input has one link.** Dropping onto an input that already has one **replaces** it (the old link goes, in the same step). An output may feed any number of inputs: a node's picture can go on to its next node and to a branch.
- **Pick up a link's end.** Press on an input port that has a link and drag: its end comes away from the port. Drop it on another input of the same type to move it there; drop it on empty canvas to **remove the link**.
- **Select and delete.** A click on a link selects it (it thickens, in yellow); Delete or Backspace removes it; a right click on a link offers Delete link.
- **What stays fixed, and what is refused** (the link is not made, nothing changes, and the error bar says why):
  - the link **Input → Main** cannot be moved or removed: "Main always takes its picture from the input.";
  - **the Output's picture link** can be replaced (drop another node's picture output on the Output) but not removed: "The output needs a picture. Link another node to it instead.";
  - a link that would make a circle, counting picture and key links together: "That would link the nodes in a circle.".
- **Dead branches.** A node whose picture no longer reaches the Output (and whose key feeds no live node) is kept and dimmed, with the tooltip "Not connected to the output."; linking it again brings it back.
- **The keyboard path.** With a card focused, More › **Link picture to ▸** and **Link key to ▸** list the inputs it can be linked to (by node number and name; the Output for a picture), and More › **Remove link ▸** lists its links. The same rules apply.
- Each connect, move or removal of a link is **one patch (`links`) and one undo step**.

Each operation is one patch and one undo step (`nodes`, `remove` and `links` as section 3.7 says), built by pure functions of the graph (`src/state/graphOps.ts`: `addNode`, `insertAfter`, `remove`, `reorder`, `moveCard`, `toggleBypass`, `rename`, `connect`, `disconnect`), tested without a screen; the same functions say which inputs light up during a drag and why a link is refused; the core validates what they send.

### 7.3 The toolbar

44 px, glass. Left to right: **Add node ▾** (a menu button) · **Bypass** · **Delete**; at the right the photo switcher **‹ 2 of 7 ›** (accessible names "Previous photo", "Next photo"), **Show mask** (a toggle, yellow when on; tooltip "Show mask (M)") and **Expand** (a toggle with an expand icon, yellow when on). The mockups' "Graph · List" switch, after Delete, is **not built**: the list view was dropped by the owner.

### 7.4 The Add node menu

A glass menu that opens upward from the button over the photo:

| Item | Line | State |
|---|---|---|
| **Corrector** | "Grades all or part of the photo." | enabled |
| *(separator)* | | |
| Parallel | "Grade branches side by side." | greyed, tag "Later" |
| Layer | "Blend one branch over another." | greyed, tag "Later" |
| Splitter and combiner | "Split a picture, then join it again." | greyed, tag "Later" |
| Outside | "Everything outside a mask." | greyed, tag "Later" |

The greyed items are `aria-disabled`, with the tooltip "Not in the first version".

### 7.5 The panel: head and tabs

The panel edits the selected node. A compact head (52 px) has no 40 px title: a **number chip** (a lock for the Input and the Output), the **name at 20 px bold**, a quiet line, and icon buttons.

| Node | Head line | Icon buttons (tooltip) |
|---|---|---|
| Corrector | "Corrector · whole photo" (no masks), "Corrector · 1 mask" and "Corrector · 2 masks" (the row text says which), "Corrector · Linear" (one mask: its kind), "Corrector · bypassed", "Corrector · not connected" | Bypass ("Bypass this node (B)"), Reset ("Reset this node"), More ("More") |
| Input | "Turns the camera's data into the scene." | Reset ("Reset input") |
| Output | "Cine rendering, look and export" | Reset ("Reset output") |

The tabs follow the head (46 px), each underlined in its tab colour (`docs/11`); the mask tab's colour is the mask accent.

| Node | Tabs |
|---|---|
| Input | **RAW · Lens · Light · Colour · Detail** |
| Corrector | **Light · Colour · Curve · Mask** |
| Output | **Look · Detail** |

What each tab holds. Where a tab matches today's, its cards are today's.

| Node › tab | Cards |
|---|---|
| Input › RAW | Camera · Decode · Range |
| Input › Lens | Lens · Amounts |
| Input › Light | the histogram · **Exposure** (AUTO tag, Reset; the row Exposure only) |
| Input › Colour | **White balance** (AUTO tag, Reset; the light presets, Pick white, Kelvin, Tint). Only this card belongs to the Input; the tab is short |
| Input › Detail | Sharpening (Amount, Radius; no Output row) · Noise reduction |
| Corrector › Light | the histogram · **Exposure** (Reset; Exposure trim, Contrast, Brightness) · **High dynamic range** (Reset; Highlights, Shadows, White, Black). No AUTO tag: Auto belongs to the Input |
| Corrector › Colour | **Temperature and tint** (Reset; Temperature, Tint, the "Temperature and tint trim"; no Pick white in this version) · **Colour balance** (Reset; three wheels, Saturation) · **Colour warp** (Pick colour acts on this node) |
| Corrector › Curve | **Curve** (Luma · Red · Green · Blue; the plot with the node's histogram) |
| Corrector › Mask | the Masks card, the selected mask's card, the View card (section 7.6) |
| Output › Look | **Look**: Intensity, Apply to (This photo · Picked · N · All · N), search, the libraries (today's Look card, one look per photo) |
| Output › Detail | **Sharpening** with the **Output** row only; its head reads "Output · cine rendering, look and export" |

- **On Main** the Contrast, Highlights, Shadows, Black, Saturation and wheel controls keep their Auto semantics (a value equal to Auto *is* Auto; the detent and Reset are at the Auto value, `docs/11` *Auto semantics*). On every other node they are plain numbers with the detent and Reset at 0.
- **The Automatic card, the Tone card and Re-run** belong to the simple view. The Input's Reset puts every Input field to its default, Auto where there is Auto (the Output's: the look, its intensity and the output sharpening); Re-run itself is not in the node view.
- **Reset this node** (the head) puts the node's trim and grade back at their defaults (Main's at Auto where there is Auto, the others' at a manual 0); **masks stay** (the Masks card has its own Reset). One undo step.
- **More** (correctors): Rename · Move earlier · Move later · Link picture to ▸ · Link key to ▸ · Remove link ▸ · Delete node (Delete node is off for Main, which also has no Move items; Main's picture input cannot be relinked). The Input's More has Link picture to ▸ (an extra branch from the Input).
- **Tab memory.** Entering the node view with the simple view on a tab selects the matching node and tab: Look → Output › Look; Lens → Input › Lens; Light → Main › Light; Colour → Main › Colour; Detail → Input › Detail; Curve → Main › Curve; RAW → Input › RAW. Nodes of one kind share a remembered tab. ⌘1 to ⌘5 switch tabs in the order of the row (⌘1 to ⌘4 on a corrector, ⌘1 and ⌘2 on the Output).
- **Pick white** is on the Input's Colour tab (section 8). In the node view the key W selects the Input and opens that tab.
- **The histogram and the Curve plot** read the signal at the selected node: its curve's input for a corrector, the Input's output for the Input (`histogram`, `chroma_scatter` and `sample_chroma` take the node, section 9). The pixel readout (⌥) is the final picture.

### 7.6 The Mask tab

Three cards fit without scrolling at 1440 × 900 with **two mask rows** (the head is 52 px, the tab row 46, and the cards are 244 + 316 + 114 = 674 px in a 678 px scroll area); from a third row the panel scrolls a little.

**The Masks card.** Header: "MASKS", **Add mask ▾**, **Invert** (a toggle with an invert icon: the node's whole combined mask, `masks.invert`) and **Reset** (in the mask accent: removes every row and un-inverts). Then the rows, then one line:

"Masks add up. Subtract cuts a mask out. Colour and brightness ranges keep only their part of the rest."

**The Add mask menu** (520 px tall; it opens to the left of the panel):

| Group | Item | Line |
|---|---|---|
| FIND | Object | "Click the photo, or draw a box" |
| | Subject | "The main thing in the photo" |
| | People | "Each person. macOS 14 or later" |
| | Background | "What is out of focus, from blur" |
| SHAPES | Linear | "A gradient across the photo" |
| | Radial | "An oval, soft at the edge" |
| | Brush | "Paint where the node acts" |
| KEYS | Colour range | "Pick colours to select" |
| | Brightness range | "Select light or dark tones" |

Only the kinds that `mask_kinds` lists are shown. Subject and People are **hidden**, not greyed out, where this Mac does not support them (section 3.10). Adding a row selects it and starts its tool.

**A row.** An icon for the kind, the name in bold (Object, Radial, …) with its subtitle under it, a segmented **Add | Subtract**, an **Invert** icon toggle (this mask alone; accessible name "Invert this mask"; pressed in the mask accent), and an **eye** toggle (the row is used or absent; "Show this mask"). The selected row has a 2 px ring in the mask accent. The row's accessible name is the kind, "mask" and the subtitle ("Object mask, 2 points", "Radial mask, Feather 45"). Removing a row: select it and press Delete or Backspace, or right-click for **Move up · Move down · Delete mask**.

| Kind | Subtitle |
|---|---|
| Object | "2 points" ("1 point"; "Box"; "Box and 2 points"); with none: "Click the object on this photo" |
| Subject, People | "Foreground", "People"; on a Mac that cannot: "Subject masks need a Mac that supports them" / "People masks need a Mac that supports them"; nothing found: "Nothing found" |
| Background | "Level 50" |
| Linear, Radial | "Feather 20", "Feather 45" (feather × 100) |
| Brush | "3 strokes" ("1 stroke"); none: "Paint on the photo" |
| Colour range | "Width 40" |
| Brightness range | "35 to 100" |

While a raster mask is being made the subtitle shows a spinner and "Finding…"; while the SAM encoder runs, "Getting ready…".

**The selected mask's card** (its title is the kind in capitals; its **Reset** puts the row back as a new row of the kind, same mode):

| Kind | Content |
|---|---|
| OBJECT | A segmented tool, **Add point · Remove point · Box**, with the line "Click the photo to add to the object. ⌥-click to take a part away." (for Remove point: "Click the photo to take a part away from the object."; for Box: "Drag a box around the object."), then sliders **Refine edges** (50%), **Feather** (0), **Grow** (0, bipolar) and **Opacity** (100%) |
| SUBJECT, PEOPLE | Refine edges, Feather, Grow, Opacity |
| BACKGROUND | **Level** (50), **Softness** (50), Refine edges, Feather, Grow, Opacity |
| LINEAR | **Angle** (degrees), Feather, Opacity (the position, angle and feather also by the handles) |
| RADIAL | **Width**, **Height**, **Angle**, Feather, Opacity (and the handles) |
| BRUSH | **Paint · Erase**, **Size**, **Hardness**, then Refine edges (0), Feather, Grow, Opacity |
| COLOUR RANGE | **Pick colour** (a toggle with an eyedropper, like the Colour warp's), **Width**, **Softness**, Opacity |
| BRIGHTNESS RANGE | **Low**, **High**, **Softness**, Opacity |

Sliders are the standard slider; Refine edges and Opacity show percent, the others plain numbers (`docs/algorithms/06` section 4.5). The sliders for Angle, Width, Height, Low and High exist so that every number has a control that is not a pointer drag.

**The View card.** "VIEW", with the key chip "M" at the right; one row, **Show mask**, a segmented control **Overlay · Mask only** (the mockups draw a third segment, "Off"; it is not built, because the toolbar's toggle and M turn the view on and off). Choosing a segment sets the style **and turns the view on**. The toolbar toggle and M keep the last style.

**On the photo** (drawn by the page over the cell, mapped with `viewport_map_points`; the handles are focusable and the arrows nudge them by 1 px, ⇧ by 10 px):

- *Object:* the points as circles with a "+" (add) or "−" (remove), and the box; a click adds a point (the tool Add point), ⌥-click or the tool Remove point a negative one, a drag with the tool Box a box. Each click is a `mask_find`.
- *Linear:* a solid centre line, dashed start and end lines (the feather) and a rotate handle on the centre line (an orchid circle with a rotate icon).
- *Radial:* the outline, the inner (feather) ring, a centre handle, handles for the two radii and a rotate handle.
- *Brush:* a cursor circle at the brush size; a stroke is drawn on the photo while the pointer is down and sent when it is released (one undo step). **A stroke is stored simplified:** the page drops the points nearer than 1/8 of the brush size to the line between their neighbours, so a document stays small (the sanity bound of 100,000 is for files).
- *Colour range:* with Pick colour on, a click samples the chroma of the node's input (`sample_chroma`) and sets the centre; Esc ends the pick.
- Handles hide while Crop or a pick shows. **Esc** ends a pick or a mask tool first, as now.

### 7.7 Keys

| Key | Action |
|---|---|
| N | Switch between the simple view and the node view (not in a text box); the tooltip of the Nodes button says "Nodes (N)" |
| M | Show or hide the selected node's mask (node view) |
| Delete or Backspace | In the node view: removes the selected link if a link is selected, else the selected node (never Main, the Input or the Output), and **never a photo**; with a mask row selected and focus in the Masks card, removes that row. Not in a text box, on a slider, in a menu or a dialog |
| B | Bypass the selected node (node view) |
| W | Pick white: opens the Colour tab and starts picking (section 8); in the node view it selects the Input first |
| Esc | Ends a pick or a mask tool first, as now |
| ⌥← and ⌥→ | Move the focused node earlier or later |
| ← → ↑ ↓ | With the graph focused, move the selection; otherwise ← and → switch photos, as now |
| ⌘1 to ⌘5 | Tabs of the selected node |
| ⌘Z, ⇧⌘Z | Undo and redo the graph like any setting |

(⌘M is Minimise, so M is bare.) All of it goes in `shortcutFor` (`src/state/shortcuts.ts`) and its test.

### 7.8 Accessibility

- The graph is **one Tab stop**; the arrows move between nodes; Return or Space selects (selection follows focus).
- **Every card has a name:** "Node 2, Mountain, masked by Object and Radial" (the number as a plain number, the name, the kinds of its masks; "Node 1, Main" with none), with ", bypassed", ", not connected" and "selected" (`aria-pressed`) where they apply. The Input and the Output are "Input" and "Output". The ports and links are drawn for the pointer and are not Tab stops; every link can be made, moved and removed from the keyboard through the card's More menu (Link picture to, Link key to, Remove link), and a card's name says what feeds it (", picture from node 1, key from node 5").
- The divider, the zoom buttons, the toolbar toggles (`aria-pressed`) and the menu items have names; the Later items are `aria-disabled`.
- The mask handles are focusable and nudged by the arrows (section 7.6); the sliders of section 7.6 give every number a keyboard control.
- The existing a11y and glass audits gain the new page states (section 11): the glass audit's `STATES` grow from 41 to 50 and `HAS_COUNTERPART` and `NO_COUNTERPART` are updated; `a11y.spec.ts` walks the node view (graph, menus, each node kind's tabs, the Mask tab) and fails on any control without a name.

## 8. Pick white's move

Pick white **moves into the Colour tab**, as part of this feature (task N-06, before the node view).

- **The header** loses its Pick white button and its hint. The header cells are Before, Crop, Nodes, Add photos, Fit.
- **The card row.** A row of its own in the White balance card, between the light presets and the Kelvin slider: a small toggle with an eyedropper and "Pick white", and at the right "or press" and a key chip "W". Pressed, the toggle is yellow, like the header toggles. The card header keeps AUTO and Reset.
- **While picking** the cursor over the photo is a crosshair, drawn white with a dark outline so that it reads on light and dark parts. The hint is the glass pill at the top centre of the photo (the "Object found" pill's style, with an eyedropper): **"Click something neutral: grey or white. Esc cancels."** A click samples the colour exactly as before (`viewport_input { kind: "pick_white" }`; the core converts the point; `docs/11` *Pick white*, `docs/02` C1); a spot the core refuses shows its words in the error bar and changes nothing. After a pick the mode ends and the card shows EDITED.
- **W** opens the Colour tab and starts picking; W again, or Esc, ends it. In the node view W selects the Input and opens its Colour tab, then starts picking.
- **It turns off when the Colour tab is left**, as Pick colour does: a tab change, a node change in the node view, the Nodes toggle, Crop, a photo change. Pick white and Pick colour are exclusive.
- **A corrector's Colour tab has no Pick white** in this version (the Input is where the photo's white is set).
- The keyboard path of the layer is unchanged (focus, the arrows move a crosshair, Return picks).

## 9. IPC, wire types and the mock

**Settings are node-first on the wire** (the Rust wire types and `bindings.ts` are N-02; `types.ts`, `contract.ts` and the mock are N-02b). `DevelopSettings` is `{ nodes: NodesSettings }` with `NodesSettings { items: NodeItem[], links: NodeLink[] }`. `NodeItem` is a union tagged by `kind`: `InputNode`, `OutputNode` and `CorrectorNode`, each with the fields of section 3.2 **under today's names and types** (`AutoOr`, `PercentSetting`, `BalanceSetting`, `LensSettings`, `Crop`, …), plus `NodeTrim`, `NodeGrade` and, for masks, `MaskSettings { invert, items }` and `MaskItem` (a union tagged by `kind`, section 3.3). `NodeId` is `"input"`, `"output"` or a number (Main is 1); `NodeLink { from, to }` with `NodeRef { node: NodeId, port: "picture" | "key" }`. **`SettingsPatch`** is `{ nodes?: { [id]: NodePatch }, remove?: NodeId[], links?: NodeLink[] }` (section 3.7), with `NodePatch` the partial of a node's fields, `MaskPatch` the row patch (`masks.edit`, `strokes_append`, `strokes_keep`) and `FieldRef { node, field }` in `PatchReport`. **`ResolvedSettings`** keeps `{ photo_id, settings, values, auto, estimates }` with the node-first shapes of section 3.9, and `PastedPhoto.before` and `.after` are patches in the same form. All of it is in `src/api/types.ts` (the hand contract), in the generated `src/api/bindings.ts` and in the mock, and the field-name test of `docs/01` compares them.

**Commands that change shape** (`docs/01` *IPC*): `get_settings`, `update_settings(photo_id, patch)` (a node patch) and `rerun_auto` answer the node-first `ResolvedSettings`; `copy_settings(photo_id)` answers a patch that holds the whole graph and `paste_settings(photo_ids, patch)` applies it (section 3.8); `apply_look(scope, look, intensity)` writes the Output; a `viewport_input` Pick white click sets the Input's `white_pick`; `settings_changed { photo_id }` is unchanged.

**Commands** (additions to `docs/01` *IPC*):

| Command | Purpose |
|---|---|
| `node_thumbnails(photo_id)` | Asks for the thumbnails of every node of the photo, dead ones included (section 5.7). Answers `null` at once; the pictures come as a `node_thumbnails_ready` event |
| `viewport_map_points(points, from)` | Maps points between the photo frame and the cell, for handles and outlines. `from` is `"photo"` (points `[x, y]` in the photo frame, answered as CSS pixels from the cell's corner, which may lie outside the cell) or `"cell"` (CSS pixels, answered as photo-frame fractions, `null` for a point on the paper). The webview never converts itself: the straightening, the turn, the crop and the lens's bend are the core's |
| `mask_find(photo_id, node, mask, prompt)` | Runs SAM for an Object row (`mask` is the row's index in the node's `masks.items`). `prompt` is `{ kind: "point", x, y, positive }` or `{ kind: "box", x0, y0, x1, y1 }` in the photo frame. Answers **found** with the row's new `points` and `box` (and the decoder's `took_ms`) or **nothing found**. **Only a found answer changes settings**: the page sends it as one row patch, one undo step. It waits for the encoder if it is still running. The core caches the mask it found, so the render that follows does not run the decoder again |
| `mask_kinds()` | The kinds of mask this machine can make now (`MaskKind[]`): the shapes and ranges always; Object when the SAM weights are there; Subject and People where the Vision probe said yes; Background when its phase is built. The Add mask menu lists exactly these |
| `viewport_set_mode(mode, mask_view)` | Gains a second argument, `mask_view`: `{ node, style }` with `style` `"overlay"` or `"mask_only"`, or `null` for no mask view (section 5.8) |
| `histogram(photo_id, at)`, `chroma_scatter(photo_id, at)`, `sample_chroma(x, y, at)` | Gain `at`: a node's id, `"input"` or a number (left out: Main, 1). The Light and Curve tabs and the Colour warp and Colour range read the signal at the selected node (section 7.5) |

**Events:** `node_thumbnails_ready { photo_id, thumbnails: [{ node, url }] }` (only the thumbnails that changed; PNGs of about 160 px on the `ym-frame` scheme, the address of a picture, never its bytes). The error scope `mask` carries the plain sentences of this file ("This copy of Yellowmoon has no Object model. Reinstall the app.", the Subject sentence of section 3.10).

**The browser mock** (`src/api.mock.ts`) draws flat scenes, so masks act on whole scene areas: enough for the interface tests, not for pictures. It validates the graph with the core's rules (`src/fixtures/graph.ts`, checked against the core's error texts by `src/repo.node.test.ts`), applies the copy rules (`copiedPatch`), makes thumbnails from the flat scenes after final patches, answers `mask_kinds` from `?mockMasks=` (default: every kind; `?mockMasks=noai` leaves out Object, Subject, People and Background; `=nomac` leaves out Subject and People), and `mask_find` finds the scene's mountain when the point is on it and answers nothing found anywhere else (`?mockMaskFind=none` always answers nothing found). The e2e suite uses it as it does now.

## 10. Where things live

- **Core:** `ym-core` (`settings.rs`: the node-first document, its validation, node patches, copy rules and `ResolvedSettings`; `pipeline/color.rs`: the plan's parameters; `pipeline/graph/masks.rs`: the raster caches; `pipeline/thumbs.rs`: node thumbnails), `ym-color` (`plan`, `mask`, the node loop in `display`), `ym-gpu` (the fused kernel and `mask_raster`), `ym-ml` (`sam2`), `ym-apple` (Vision), `ym-cli` (`cli.rs` and `develop.rs`: the flags as node patches, `--settings`, the `mask` error, `ym doctor`'s masks line; `docs/10`), `tools/golden` (the manifest and `truth_settings`).
- **Shell:** `src-tauri/src/commands.rs`, `wire.rs`, `viewport.rs` (the mask view), `bindings.rs`.
- **Frontend:** `src/screens/nodes/` (`NodeView.tsx`, `NodeGraph.tsx`, `NodeCard.tsx`, `NodeToolbar.tsx`, `NodeHead.tsx`, `MaskTab.tsx`, the mask cards and handles), `src/state/settingsGraph.ts` (the selectors `inputOf`, `mainOf`, `outputOf`, `patchOf`, the per-node diff of undo) and `src/state/graphOps.ts` with their tests, `src/state/nodes.ts` (selection, the tab memory, the view state), `src/screens/PickLayer.tsx` (Pick white's new place).

## 11. Tests and gates

Every gate below must pass in CI before a phase is done. **A failed gate is reported, never loosened** (`AGENTS.md`); the numbers marked *starting* are the first bounds, which the named task measures and writes into `docs/STATUS.md`.

| Area | Test | Gate | Task |
|---|---|---|---|
| Node-first settings | every golden entry and every settings fixture, **rewritten to the node-first form** (the repository keeps no flat document), renders **the bytes of 0.8.2** on the CPU reference, **except** the case of a curve with a look other than Neutral (section 1.4), which has its own test; the app and `ym` agree; `{}` is the default graph; a node has only its kind's fields | byte-identical | N-02 |
| The form of 0.1 | a flat document (today's top-level fields, no `nodes`) is refused with the sentence of section 3.12, in `from_json`, `--settings` (exit 2) and `update_settings`. **This is the only test that holds a flat document**, and the sentence is compared word for word with `docs/10` | the refusal | N-02 |
| Auto belongs to Main | Auto in any corrector but Main reads, writes and renders as a manual 0 (every field, the wheels too); Main keeps its Auto through read, write and copy; a patch with Auto on node 3 stores a manual 0; a new node is manual 0; Re-run and the providers write the Input and Main only | tested | N-02 |
| Node patches | a patch of one node's partial changes only that node's fields; a new id with a `kind` adds a node; `remove` drops the node and its links; `links` replaces the links whole; the Input, the Output and Main cannot be removed; a bad patch changes nothing; `masks.edit`, `strokes_append` and `strokes_keep` (an append and its inverse restore the strokes exactly); the patch report names `<node>.<field>` and the stage group (`name` and `pos` re-run nothing) | tested | N-02 |
| Resolved settings | `values`, `auto` and `estimates` are node-first and agree with the flat numbers they replace; `pinned_settings` pins the Input's and Main's Auto and nothing else; the XMP `ym:Settings` and the CLI's `settings` are the pinned document and feeding them back through `--settings` reproduces the file | byte-identical | N-02 |
| The graph's rules | one test per rule of section 3.5 with its error text: link direction, port types, unknown node, unknown port, a second link on an input, the Output's link, Main's input, **cycles** (through picture links, through key links, through both), duplicate ids, unknown kind, Main with a grade, 64 nodes, 64 masks, 100,000 brush points, 256 Object points; `links` and node ids survive a round trip; clamping and `NotFinite` paths | every refusal is atomic and has its plain text | N-02 |
| Dead branches | **a graph with a dead branch** (a node whose picture goes nowhere) renders the bytes of the same graph without it; the same node with its key output linked to a live node's key input is live and its mask acts | byte-identical | N-03 |
| The plan | the default graph compiles to `[Main]` in one pass; the order is topological with Main first and the ties of section 5.2; two files that list the same graph in different orders give one plan; **a graph that needs 9 registers compiles to two passes** with an f32 buffer between them and renders the CPU reference's picture within the grade gate; thumbnail and mask-view stops are prefixes | exact order; 2e-4 | N-03, N-04 |
| Parity | CPU against GPU on lavapipe at today's gates: **2e-4 in LogC4** for the grade, now with up to **16 nodes and 8 masks each**; **one 10-bit code** after C12 and C13; cases with 1, 4 and 16 nodes, masks of every kind, inverted and subtracted, a key link, a dead branch | 2e-4 · 1 code; raster masks: *starting* 4e-4 on the plane (N-13) | N-04, N-10, N-13 |
| Determinism and tiles | a repeat render is identical; tiles equal the whole frame **including the raster masks' halos**; `PIPELINE_VERSION` is 0.9.0 | bit for bit | N-04, N-13 |
| App against `ym` | byte-identical exports with a graph, masks of every kind including Object, a hand-written settings file with a key link, and a settings file | byte-identical | N-02, N-10, N-13, N-18 |
| Thumbnails | each node's thumbnail equals a truncated render of the plan stopped there, then the Output | within one 8-bit code | N-05 |
| SAM 2 | our runtime against fixtures from Meta's own code on **20 public photos**, 3 prompts each; **mask IoU at least 0.99**; the logits within a stated bound; the encoder's time measured | IoU ≥ 0.99; logits *starting* 0.05 max, 0.005 mean | N-17 |
| Masks | preview against export: the same mask at k = 4 and k = 1 within a stated bound; a photo cropped or turned keeps its mask on the subject | *starting* largest 0.1, mean 0.005 after reducing the export | N-10, N-13 |
| Copy and paste | the copy holds **every node and every link** and leaves out the Input's `crop`, `orientation` and `white_pick` and the Object points (a picked white copies as the manual Kelvin and tint); a paste replaces the target's graph in one undo step and keeps the target's own three fields; Apply to copies the look and its intensity only; Subject, People and Background are worked out again | tested | N-02, N-12 |
| Pick white | the header has no Pick white; the card row, the hint pill, W and "turns off when the Colour tab is left"; e2e and the screenshots retaken | e2e | N-06 |
| Node-first store | the store sends one node's partial for a slider drag; undo and redo diff per node and for the links (an entry holds only the nodes a change touched); every tab of the simple view reads and writes through the selectors and **nothing visible changes**: every unit test, every e2e test and every reference screenshot is as before (the tests that name a patch's shape are rewritten to the node form, and nothing else in them) | green, screenshots unchanged | N-02b |
| The CLI documentation and the help | `docs/10`'s flag table, its node-first example file and its refusal sentence agree with `ym render --help` and with the code: every settings flag of `cli.rs` is in the table with the node and field it sets, the example file is parsed and validated by the core, the 0.1 sentence is the one the code prints, and the help's mapping line equals the table | the test fails on any difference | N-02 (and every later task that changes `ym`) |
| Interface | unit tests for the store (graph operations: insert, delete with reconnection, reorder, key links on delete, **connect, replace an input's link, pick up and move a link's end, remove a link, the refusals (Main's input, the Output's link, circles, port types)**, undo of each, the copy rules); Playwright drags a link from an output to an input, onto an occupied input, and off an input onto the canvas; **Playwright** for the node view, the menus, the keys, Pick white and the mask rows; the **glass audit and a11y** gain the new states; new reference screenshots, which the owner looks at before they are committed | green | N-06 to N-12 |
| Vision | on a Mac: Subject and People make a mask twice with equal bytes; **Linux and older-Mac builds show no Vision items** | Mac CI | N-19 |
| Licences | `cargo deny check`; the npm licence check with `@xyflow/react`; `check_bundle.py` knows the SAM weights and their notice | pass | N-08, N-16 |
| Speed (on a Mac) | a slider move in the fit view with 8 nodes; a click on an Object mask once the encoder has run | ≤ 100 ms; ≤ 300 ms | N-20 |

**Unit and integration tests by area** (the names are for the code): `crates/ym-core/tests/settings.rs` (the node-first document, every rule, Auto in a corrector, node patches, clamping, the schema, the refusal of the 0.1 form, copy rules, `reset`, resolved settings), `crates/ym-core/tests/session.rs` (paste and apply-to on graphs), `tests/render_nodes.rs` (the plan on the CPU, trims, bypass, masks, identity skipping, dead branches, key links), `tests/render_gpu.rs` and `crates/ym-gpu/tests/gpu_nodes.rs` (parity and the multi-pass plan), `crates/ym-color/tests/nodes.rs` and `trims.rs` (the maths of section 2 of `docs/algorithms/06`), `tests/mask_raster.rs`, `crates/ym-ml/tests/sam2.rs`, `crates/ym-cli/tests/parity.rs`, `render.rs`, `info.rs`, `apple_auto.rs`, `auto_providers.rs` and `lenses.rs` (the flags as node patches, graph files, the `mask` error, `ym doctor`; the tests that read `settings` and `auto` in the JSON are rewritten to the node-first paths), `crates/ym-cli/tests/docs.rs` (**`docs/10` and the help agree**, below), `src/state/settingsGraph.test.ts` and `src/state/history.test.ts` (selectors, node patches and the per-node undo), `src/state/graphOps.test.ts`, `src/state/nodes.test.ts`, `src/state/shortcuts.test.ts` (N, M, B, Delete in the node view, W), `src/screens/nodes/*.test.tsx`, `e2e/nodes.spec.ts`, `e2e/masks.spec.ts`, `e2e/pick-white.spec.ts`, `src/repo.node.test.ts` (the copy that this file and the component share).

**`crates/ym-cli/tests/docs.rs`** (`the_cli_documentation_and_the_help_agree`) reads `docs/10-cli.md` and `ym render --help` and fails on any difference between them: every long option of `RenderArgs` that sets a field of a node (`cli.rs`) has a row in the *Where the flags land* table of `docs/10` with the node and field it sets, and every row names a real option; the help's closing paragraph (the one that says which flags set the Input, Main and the Output) lists the same flags as the table; the node-first example file of `docs/10` is extracted from the document, parsed with `DevelopSettings::from_value`, validated and written back unchanged; the per-file example parses as `--settings` does; and the sentence for the form of Yellowmoon 0.1 is the same in `docs/10`, in `docs/23` section 3.12 and in the error the code returns.

**New reference screenshots** (`design/screenshots-c2/`, taken on the mock API like the others; the owner looks at them before they are committed): `photos-has-nodes.png` (screen 04), `photos-pick-white.png` (03), `photos-nodes.png` (05), `photos-nodes-add-node.png` (06), `photos-nodes-mask-object.png` (07), `photos-nodes-add-mask.png` (08), `photos-nodes-mask-linear.png` (09), `photos-nodes-input-colour.png` (10), `photos-nodes-output-look.png` (11). The eleven Photos references that show the header are retaken (the header changes). **The glass audit** gains the states 42 to 50 with the same nine screens, so `STATES` goes from 41 to 50.

## 12. Risks

- **SAM 2 speed on the CPU** is the biggest unknown. N-17 measures it early. If it is too slow, the encoder moves to the GPU, still deterministic within a stated bound, before N-18.
- **App size.** About 95 MB more (the weights at f16, plus the model card).
- **Apple Vision** may answer differently on another macOS version; within one Mac the app and `ym` agree, and the Linux goldens cannot cover it. The Intel slice of the universal binary may fail the probe (then the kinds are hidden there).
- **The fused kernel** is already large. A plan of 16 nodes with masks may cost time on older Macs; N-04 and N-10 measure it.
- **Bit-identity** of the default graph is guaranteed on the CPU reference. On the GPU it is within today's gates and identical on lavapipe where possible.
- **Memory.** SAM features about 16 MB a photo and raster masks 4 to 8 MB each, in a small in-memory cache, never on disk. Undo entries hold only the nodes a change touched (section 3.7); a brush stroke is an append and its inverse, and N-02b checks that 100 steps over a graph with a 100,000-point brush stay within a stated size.
- **Hand-written graphs.** A settings file, and now the person drawing links, may make fan-outs, dead branches and key links. The validator and the compiler treat them as first-class; a graph that needs more than 8 pictures at once takes the several-pass path, which is tested.
- **`@xyflow/react`** is a new dependency (MIT, with `@xyflow/system`, `classcat` and `zustand`, all MIT on their repositories); the graph could be hard to make accessible. N-08 checks the licences with the npm gate and the keyboard path with the a11y audit.
- **The refactor touches every setting.** Making the settings node-first moves every field; the gate is byte-identity (N-02) and unchanged screens (N-02b), and the two tasks land before any feature. **The JSON of `ym` changes shape** (`settings` and `auto` are node-first, a 0.1 settings file is refused): scripts that read `settings.contrast` break at 0.2.0, which the CHANGELOG says and `docs/10` documents.
- **JPEG XMP** holds 65,502 bytes: a large brush document does not fit (section 3.9).

## 13. Tasks

Task format of `docs/15`. Each task updates the specs it touches and `docs/STATUS.md`. The work happens on the branch `nodes` from `main`, one reviewed commit per task with its checks green (`cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`, `npm test`, `npm run build`). At each checkpoint the owner gets a **test DMG** (`build.yml`); checkpoints are not releases. When the whole plan is built and the owner approves the last test build, the branch is merged to `main` and released once as **0.2.0**.

**N-01 · Spec** (): turn the plan into this file, `docs/algorithms/06-masks.md`, the changes to the other specs, `design/nodes/README.md` and the research note. *Done when:* the owner approves the spec (N-H1).
**N-02 · Node-first settings in the core, the CLI and the shell's Rust side** (N-01): `DevelopSettings` becomes the graph (section 3): the model (the Input, the Output and correctors with their fields under today's names and types, links, mask rows), the read over the default graph, **validation** with the plain texts of section 3.5, **Auto belongs to Main** (section 3.2), **node patches** (`nodes`, `remove`, `links`, `masks.edit`, `strokes_append`, `strokes_keep`; section 3.7) with the patch report, reset, **copy and paste** (section 3.8), the resolved and pinned settings and the XMP `ym:Settings` (section 3.9), the **schema** (the node-first form only) and its test, and the refusal of the form of Yellowmoon 0.1 (section 3.12). **The CLI follows** (`docs/10` is the spec): `cli.rs` and `develop.rs` turn every settings flag into a patch of the Input, Main or the Output, `--settings` reads the node-first form (one document, or an object keyed by ARW name) and refuses the 0.1 form with its sentence, `ym render --json` and `ym info --json` print `settings` and `auto` node-first, and **the help matches**: the closing paragraph of `ym render --help` (which names the node each flag sets and shows the node-first form), the `--settings` text and the new `ym:Settings` wording. **The repository's own documents are rewritten to node-first**: the golden manifest and `truth_settings` (`tools/golden`, with its README), the test fixtures, the CLI tests and the schema test; no fixture keeps the old form but the one test of the refusal. **The CLI's snapshots are updated**: `cli__help_render` (the help text), `render__render_json_shape`, `apple_auto__render_json_shape_apple_auto` and `info__json_snapshot_of_a_synthetic_a7r_iv` (the `settings` and `auto` objects), and the tests that read `settings` and `auto` paths in the JSON (`render.rs`, `parity.rs`, `info.rs`, `apple_auto.rs`, `auto_providers.rs`, `lenses.rs`). **`crates/ym-cli/tests/docs.rs`** is added (section 11), and `README.md`, `docs/09` and `docs/10` say the same. **The shell's Rust side follows** (the lead's change of 2026-11-01): `src-tauri/src/wire.rs` (the wire types are node-first, section 9), the commands that touch settings (`src-tauri/src/api.rs`), `src/api/bindings.ts` regenerated (`YM_UPDATE_BINDINGS=1 cargo test -p yellowmoon-app --bin yellowmoon-app bindings`) and the Rust tests of `src-tauri`, so that `cargo test --workspace` is green at the end of N-02. Between N-02 and N-02b `npm test` and `tsc -b` fail on the contract (`src/api/types.ts` and `contract.ts` still describe the flat settings); that is allowed only there, and the one Rust test that compares field names with `types.ts` (`field_names_match_the_typescript_contract`, with the two such tests of `crates/ym-core/tests/settings.rs`) is `#[ignore = "N-02b"]` until N-02b rewrites `types.ts`. No pixel and no `PIPELINE_VERSION` changes. *Done when:* every existing render and export is byte-identical (every golden entry and settings fixture, rewritten to node-first, renders the bytes of 0.8.2), the app and `ym` agree (`tests/parity.rs`), the node patch, Auto-in-a-corrector, copy and 0.1-refusal tests pass, and the CLI's help, its snapshots and `docs/10` match the node-first settings.
**N-02b · Node-first settings in the frontend** (N-02): `src/api/types.ts`, `contract.ts` and the mock (section 9; the wire types and `bindings.ts` are N-02's), and the removal of the three `#[ignore = "N-02b"]`; the store with node patches (a slider drag sends one node's partial) and **per-node undo** (section 3.7); `src/state/settingsGraph.ts` with the selectors `inputOf`, `mainOf`, `outputOf` and `patchOf`; every tab of the simple view reads and writes through them; Copy settings and Paste carry the graph. *Done when:* **nothing visible changes**: every unit test, every e2e test and every reference screenshot is as before (the tests that name the shape of a patch are rewritten to the node form, and nothing else in them), and `ym` and the app still agree.
**N-03 · The plan and the CPU reference** (N-02): `ym_color::plan` (live and dead, the order, registers and passes, stops), nodes in plan order, trims, bypass, **the curve before the look**, identity skipping, `PIPELINE_VERSION` 0.9.0, `docs/02`; `ym --version` and `ym doctor` print 0.9.0 (the snapshots and `tests/cli.rs` that hold 0.8.2 change, and `docs/10`'s examples). *Done when:* the default graph is byte-identical to 0.8.2 (section 11), the curve-and-look case has its own test, and the plan and node tests pass.
**N-04 · The GPU plan loop** (N-03): the node storage buffer and the plan loop in the fused colour kernel, per-node curves and pins, registers and key slots, the multi-pass split (tested on a graph that needs 9 registers), the stop modes, DNG bakes nodes. *Done when:* parity, tiles, determinism, app/`ym` parity and the goldens pass.
**N-05 · Node thumbnails** (N-04, N-02b): core, IPC, cache. *Done when:* thumbnails match truncated renders.
**N-06 · Pick white into the Colour tab** (N-02b): the header button and hint gone, the card row, the hint pill, W, off when the Colour tab is left (section 8). *Done when:* e2e passes and the screenshots are retaken.
**N-07 · Node view shell** (N-02b, N-06): the Nodes button and N, layout A, the divider and Expand, the viewport follows, the switcher, the simple-view notice and glyph, the mask accent token. *Done when:* e2e passes and the native viewport check on a Mac is listed.
**N-08 · The graph** (N-05, N-07): `@xyflow/react`, markers, cards with thumbnails, ports and links, select, add, delete, bypass, rename, move and reorder, **links drawn by hand** (connect, replace, pick up, remove, the refusals and the More menu's keyboard path) (`graphOps`), zoom and Fit, keys, a11y. *Done when:* e2e, a11y and the glass audit pass and the licence check includes the new package.
**N-09 · The panel** (N-08): the compact head, the tabs of each node kind, the trim cards, the histogram per node, undo, copy and paste. *Done when:* e2e passes and the screenshots are reviewed.
**[HUMAN] N-H2 · Checkpoint 1:** a test DMG of nodes without masks. *You test it.*
**N-10 · Shapes and ranges** (N-04, N-09): Linear, Radial, Colour range, Brightness range, the key input and output, invert, opacity, feather; CPU and GPU in the fused pass. **`ym`:** masks in `--settings` (an example file with a masked node in `docs/10`, the help's paragraph, the `docs.rs` test), `ym doctor`'s **Masks** line (the kinds this machine can make, `mask_kinds`; the doctor snapshot and `docs/10`). *Done when:* parity, preview against export and the smooth-edge rule pass, and `docs/10` and the help are updated in the same commit.
**N-11 · Show mask** (N-10): Overlay and Mask only rendered by the core, M, the toolbar toggle, the View card. *Done when:* e2e passes and the Mac check is listed.
**N-12 · The Mask tab** (N-10, N-11): the Add mask menu, the rows (Add or Subtract, Invert, eye), the cards, the handles, range picking, the copy rules. *Done when:* e2e passes and the screenshots are reviewed.
**[HUMAN] N-H3 · Checkpoint 2:** a test DMG with shapes and ranges.
**N-13 · Raster masks** (N-12): the photo-frame raster, resample, grow, feather and the guided-filter Refine edges; CPU and GPU; tile halos; the memory cache. *Done when:* parity, tiles and preview against export pass. **`ym`:** raster masks render from `--settings` (the app and `ym` give byte-identical files), and `docs/10` and the help are updated in the same commit if anything a person sees changes.
**N-14 · Brush** (N-13): strokes, erase, size, hardness; drawing on the photo. *Done when:* e2e and parity pass.
**N-15 · Background from blur** (N-13): the blur map as a mask. *Done when:* the tests on the blur-map fixtures pass.
**N-16 · SAM 2 weights** (N-01): the conversion script, safetensors f16, the release asset, the fetch with SHA-256, bundled in the app and in `data/` for `ym`, the model card, the notice, `docs/14`. *Done when:* the licence check passes and the build bundles the weights.
**N-17 · SAM 2 runtime** (N-16): the encoder, the prompt encoder and the decoder, deterministic, with oracle fixtures from Meta's code. *Done when:* the IoU and logit gates pass and the timing is measured and written down.
**N-18 · Object in the app** (N-13, N-17): `mask_find`, the encoder in the background, the feature cache, points and box, the "Object found" pill and the nothing-found error. **`ym`:** the `mask` error kind for an Object mask whose weights are missing, the Masks line gains Object, `docs/10` and the help. *Done when:* e2e passes, the app and `ym` agree on an Object mask, and `docs/10` and the help are updated in the same commit.
**[HUMAN] N-H4 · Checkpoint 3:** a test DMG with SAM 2, timed on the owner's Mac.
**N-19 · Subject and People** (N-13): Vision through `ym-apple` behind the macOS gate, the cached run-time probe, the menu items hidden where unsupported, the row message and the `ym` error for a document that has one anyway, determinism checked on the Mac CI. **`ym`:** the `mask` error for Subject and People where the Mac cannot make them (the sentence of section 3.10), the Masks line, `docs/10` and the help. *Done when:* the Mac CI and the parity on a Mac pass, Linux and older-Mac builds show no Vision items, and `docs/10` and the help are updated in the same commit.
**N-20 · Finish** (N-12 to N-19): speed checks on a Mac, docs, `docs/STATUS.md`, `CHANGELOG.md` (it says that the JSON of `ym render --json` and `ym info --json` changed shape and that settings files of 0.1 are refused), the README (its CLI section), `docs/10` read once more against `ym render --help` and every `ym` output, the screenshots; then the 0.2.0 release after N-H5. *Done when:* the gates are in `docs/STATUS.md`.
**[HUMAN] N-H5 · Checkpoint 4:** the final test of the whole plan; the owner's OK releases 0.2.0.

**Order.** B (N-02, N-02b, N-03 to N-05) before C (N-06 to N-09) before D (N-10 to N-12); E (N-13 to N-15) and F (N-16 to N-18) can run alongside each other once D is done; G (N-19) any time after E. **The refactor lands first:** N-02 and N-02b change no pixel and nothing visible before any node feature starts. N-16 depends only on this spec, and N-06 on N-02b. **Every task that changes what `ym` does or prints updates `docs/10` and the help in the same commit**; the docs test (`crates/ym-cli/tests/docs.rs`) fails when they differ.

## 14. Decisions made while writing the spec

The plan and the owner's later answers fixed the product; these are the readings made where they were silent or where they and the code differed. The owner answered the first (section 7.2). Decisions 23 to 34 are those of the node-first revision (the owner, 2026-11-01: the settings are the graph, the form of 0.1 is not read).

1. **Links are drawn by hand** (the owner's answer of 2026-11-01 to the question this item first asked): drag from an output port to an input port, as in Resolve; an input has one link and a drop replaces it; a link's end is picked up from its input and moved or dropped on the canvas to remove it; Input → Main is fixed and the Output's link can be replaced but not removed (section 7.2).
2. **The photo frame.** The plan says masks use "the frame `white_pick` uses", after the lens geometry. `white_pick` is in the oriented image area *before* the lens crop (`Renderer::pick_point_at`). Masks use the frame the plan describes in words: oriented, after R9, before L4, crop, straightening and turn (section 3.6). It is the Input's own picture, which the AI masks read, so a raster mask needs no un-distortion.
3. **`trim` on every node.** The plan's example put the exposure, temperature and tint inside a node's `grade`. They are a node's `trim`, so that every corrector, Main included, has one shape (`trim`, `grade`, `masks`); a node without `grade` is all defaults.
4. **The trims' maths.** The photo's resolved white after the Input is AWG4's own white (D65), so the temperature and tint trim starts there and depends on nothing else: ±100 temperature is ±50 mired on the CIE daylight locus, ±100 tint is ±0.01 in `(u, v)` across it, and the gain is a von Kries gain in AWG4 with green 1. Using `ym_color::wb`'s ALEXA-native helpers would make a node's gain follow the Input's Kelvin, the scene's class mix and the camera, which breaks copy and paste between photos (`docs/algorithms/06` section 2.2).
5. **The key input.** An implicit first region row (added like a mask); the key output is the node's final `m` and does not depend on its Bypass.
6. **A lone Subtract** starts from the whole photo ("the photo, minus this"); the plan's rules left it undefined.
7. **Live and dead.** A node is live when the Output needs its picture or a live node needs its mask. Dead nodes are kept, numbered and dimmed with the tooltip "Not connected to the output.". Numbers follow the plan's order (Main first, then `pos`, then id).
8. **Graph operations.** Add after the selected node goes into its outgoing picture link and shifts the nodes after it one unit right; Delete reconnects the neighbours and removes the deleted node's key links; Reorder takes the node out the same way and puts it into the link it is dropped on, keeps its key links, and is refused when it would make a circle; Main cannot be reordered and the link Input → Main is fixed.
9. **Plan limits.** `PLAN_REGISTERS` 8 and `PLAN_KEYS` 16; a graph that needs more splits into passes (a path that v1 only reaches from hand-written files, and that the mixers will use).
10. **Commands the plan did not list:** `mask_kinds` (the menu must know what to hide), an `at` argument on `histogram`, `chroma_scatter` and `sample_chroma` (a node's Light, Curve and Colour tabs read their own node), and `viewport_set_mode`'s second argument `mask_view`. The `at` default is Main.
11. **The View card has two segments**, Overlay and Mask only. The mockups draw a third, "Off" (and a tooltip "M steps through the three"); the plan says the toolbar toggle and M turn the view on and off and the card chooses the style. Choosing a segment turns the view on.
12. **Keys and menus the plan did not list:** B (the head's tooltip says "Bypass this node (B)"), the More menu (Rename, Move earlier, Move later, Link picture to, Link key to, Remove link, Delete node), ⌥← and ⌥→ for reorder (a keyboard path), Delete on a selected mask row, a row's context menu (Move up, Move down, Delete mask: the mockup's row has no remove control). The head's Reset leaves the masks and the Masks card's Reset clears them.
13. **Thumbnails** are the picture after the node *through the Output* (look and rendering), so they look like the photo; one for every node, dead ones included; about 160 px.
14. **Mask rows.** The subtitles of section 7.6; the sliders for Angle, Width, Height, Low and High; the Object-found pill stays 3 s; a stroke is sent when the pointer is released and stored simplified.
15. **The simple view's notice** also shows for a Main that has a trim, masks or a bypass (with its own sentence), and the filmstrip glyph follows the same rule, so the simple view never hides what changes the picture.
16. **Raster details:** the planes are u16, at most 48 alive in a tile job; Grow is a square structuring element; Feather 100 is 2 % of the long edge, Grow ±100 is 1 %, Refine edges 100 is a window of 0.8 % with ε = 1e-5 (the maths gives the formulas); the ramps are `RANGE_RAMP_MIN` 0.02 and `BLUR_RAMP_MIN` 0.05; a click that finds under 0.05 % of the photo, or whose object score is not above 0, is "nothing found".
17. **Copy that the mockups do not have:** "Main can't be deleted. Bypass it instead."; "The input and the output can't be bypassed."; "Not connected to the output."; "That would link the nodes in a circle."; the Colour range, Brightness range and Brush row subtitles; "Finding…" and "Getting ready…"; "Click the object on this photo" (the plan's); the Remove point and Box lines of the Object card; the notice for a hidden Main; the zoom, divider and switcher names; the graph rule texts of section 3.5; "Corrector · bypassed" and "Corrector · not connected".
18. **Sanity bounds the plan did not name:** 256 points in an Object mask and 32 characters in a name; the JPEG XMP rule of section 3.9 when the settings do not fit.
19. **Vision.** Subject and People are the union of all the request's instances; the probe runs each request on a 64 × 64 picture and caches the answer for the process; the bridge is built for both architectures of the universal binary where the SDK has the requests; `ym doctor` prints a "Masks" line with what the machine can make.
20. **"No node grades, default look"** (the plan, for what AI masks look at) is read as the Neutral look, which `DevelopSettings::default()` already is, with no output sharpening, so a mask never depends on the look the person chose or on the preference for the default look.
21. **The node view's panel** has no Automatic card, no Tone card and no Re-run (the mockups' Input and Output tabs have none); the tab on which the simple view stands decides which node and tab opens (section 7.5).
22. **Starting bounds.** The SAM logit bound (0.05 max, 0.005 mean), the raster-plane parity (4e-4) and the preview-against-export bound (0.1 largest, 0.005 mean) are the first numbers, which N-17, N-13 and N-10 measure; they are reported, never loosened (`AGENTS.md`).

23. **Node-first, with no format marker.** The key `nodes` is the marker; a document without any of the 0.1 field names is the default graph; anything else at the top level is an unknown field. A `format` field will be added the first time the node-first form itself changes. The Input and the Output are items with the ids `"input"` and `"output"`, and **they have a `pos`** like every item (the default graph places them at x = −1 and 1), so the whole canvas is one list.
24. **A document is read over the default graph.** Nodes may be listed partially, the fixed three may be left out, `links` left out are the default links, and a new id needs its `kind`. It keeps a hand-written `--settings` file short, as the flat form was.
25. **The limit of 64 counts correctors**, Main included; the Input and the Output do not count. The number a person sees (64 nodes) is the number of cards they can add.
26. **Auto belongs to Main** (the owner's rule): in any other corrector Auto reads as the neutral value and is written as a manual 0, silently (it is not "clamped"); a new node is manual 0. The default graph's Main keeps today's Auto defaults.
27. **Patches are node-level** (`nodes`, `remove`, `links`). `links` replace whole; `remove` takes a node's links with it; adding a node is giving its `kind`. For masks, `masks.items` replaces the list, `masks.edit` patches a row by index, and a Brush row has `strokes_append` and `strokes_keep`, so a drag or a stroke never resends the rest. Undo diffs per node and for the links.
28. **The Input's framing.** `crop` and `orientation` are fields of the Input and are still applied when the picture is sampled; copy leaves out the Input's `crop`, `orientation` and `white_pick` (`COPY_LEAVES_OUT`), and a paste keeps the target's own.
29. **Resolved settings, grouped by node.** `values` is the graph with Auto resolved; `auto` and `estimates` are grouped as `input` and `main` (with the photo-level facts, such as the provider and the noise model, kept beside them). The CLI's `settings` and `auto` objects follow, so **the JSON of `ym` changes shape at 0.2.0** (a script that reads `settings.contrast` reads `settings.nodes.items[…].grade.contrast`); the CHANGELOG and `docs/10` say so.
30. **The form of 0.1 is refused** (the owner: "There are no per-photo settings from 0.1.x - they were never saved, just copied to clipboard."). The sentence is "This settings file is in the old form of Yellowmoon 0.1. Settings are now a graph of nodes: `ym render --help` shows the new form, and `ym render --json` prints a photo's settings in it." It applies to a document that has no `nodes` and only 0.1 field names; any other unknown key is an unknown field. No fixture keeps the old form but the one test of the refusal.
31. **The golden manifest keeps `{}`** for an entry without settings (the default graph), and `truth_settings` builds a node-first patch (the Output's Neutral look and the Input's highlight range 0).
32. **`ym` flags are patches of nodes** and **there are no new flags**: `--contrast`, `--brightness`, `--highlights`, `--shadows`, `--white`, `--black` and `--saturation` set Main; `--exposure`, `--exposure-offset`, `--wb`, `--light`, `--lens`, `--flares`, `--edges`, `--anamorphic`, `--nr`, `--sharpen`, `--rolloff`, `--no-neural`, `--no-highlight-recovery`, `--no-lens`, `--no-ca` and `--auto-crop` set the Input; `--look`, `--intensity` and `--output-sharpening` set the Output. The full table is in `docs/10`.
33. **The simple view reads the simple graph** through `inputOf`, `mainOf` and `outputOf`. If Main is bypassed its controls still edit its grade (the notice of section 6 says so).
34. **The CLI documentation is tested** (`crates/ym-cli/tests/docs.rs`), like the copy that the app and its specs share, so `docs/10`, `ym render --help` and the code cannot drift.

## 15. As built

*(Filled in by the tasks as they land: what each task decided, what it measured, and what differs from this file.)*

### N-02 · Node-first settings in the core, the CLI and the shell's Rust side

**The split.** The lead moved the shell's Rust side into N-02 (section 13): `src-tauri/src/wire.rs`, the commands that touch settings, `src/api/bindings.ts` and the Rust tests of `src-tauri`. N-02b keeps `src/api/types.ts`, `contract.ts`, the store, the tabs, the mock and their tests. Between the two, `npm test` and `tsc -b` fail on the contract; three Rust tests that read `src/api/types.ts` are `#[ignore = "N-02b"]` (`bindings::tests::field_names_match_the_typescript_contract` and, in `crates/ym-core/tests/settings.rs`, `the_rust_types_have_the_fields_of_the_typescript_interfaces` and `resolved_settings_have_the_shape_of_the_typescript_type`). Their bodies are already node-first: they name the interfaces N-02b writes (`NodesSettings`, `InputNode`, `OutputNode`, `CorrectorNode`, `NodeTrim`, `NodeGrade`, `MaskSettings`, `NodeLink`, `NodeRef`, `ResolvedValues`, `ResolvedInput`, `ResolvedOutput`, `ResolvedCorrector`, `GradeValues`, `AutoFlags`, `InputAutoFlags`, `MainAutoFlags`, `AutoEstimates`, `InputEstimates`, `MainEstimates`), so N-02b removes the three ignores and the tests say what is missing.

**The Rust model** (`crates/ym-core/src/settings.rs` and `settings/{graph,masks,patch}.rs`; every name below is public in `ym_core::settings`):

```rust
pub struct DevelopSettings { pub nodes: NodeGraph }
pub struct NodeGraph {            // the page's NodesSettings { items, links }
    pub input: InputNode, pub main: CorrectorNode,
    pub extra: BTreeMap<u32, CorrectorNode>,   // every corrector but Main
    pub output: OutputNode, pub links: Vec<NodeLink>,
}
pub struct InputNode { pos, exposure, light_source, white_balance, white_pick, rolloff, decode,
                       noise_reduction, sharpen, lens, crop, orientation }   // today's names and types
pub struct OutputNode { pos, look, look_intensity, output_sharpening }
pub struct CorrectorNode { name, bypass, pos, trim: NodeTrim, grade: NodeGrade, masks: MaskSettings }
pub struct NodeGrade { contrast, brightness, highlights, shadows, white, black, color_balance,
                       saturation, color_warp, curve }                       // today's names and types
pub struct MaskSettings { invert: bool, items: Vec<MaskItem> }
pub struct MaskItem { mode, invert, enabled, opacity, shape: MaskShape }     // one flat object on the wire
pub enum MaskShape { Object(ObjectMask), Subject(FindMask), People(FindMask), Background(..), Linear(..),
                     Radial(..), Brush(..), ColorRange(..), BrightnessRange(..) }
pub enum NodeId { Input, Corrector(u32), Output }  // "input" | number | "output"; ordered Input < correctors < Output
pub struct NodeLink { from: NodeRef, to: NodeRef }  pub struct NodeRef { node: NodeId, port: Port }
pub struct SettingsPatch { nodes: BTreeMap<NodeId, NodePatch>, remove: Vec<NodeId>, links: Option<Vec<NodeLink>> }
pub struct NodePatch { kind, pos, <the Input's fields>, <the Output's>, name, bypass,
                       trim: TrimPatch, grade: GradePatch, masks: MasksPatch }   // one flat object, whatever the kind
pub struct MasksPatch { invert, items, edit: BTreeMap<String, MaskRowPatch> }
pub struct FieldRef { node: NodeId, field: SettingsField }   // written "1.grade.contrast"
pub struct PatchReport { changed, clamped: Vec<FieldRef>, added, removed: Vec<NodeId>, links_changed: bool }
```

The fixed three are fields, not items, so a graph without them cannot be built; the page's list of items is written from them. Access is `settings.input()`, `main()`, `main_grade()`, `output()` and the `_mut` twins (the render and every caller outside `settings` use only these). `NodeGraph` is the Rust name of what the wire calls `NodesSettings` (the lead's design).

**What the render does.** Nothing new: `pipeline::color` reads the Input, Main's grade and the Output through the accessors, and Main's `trim`, `bypass` and `masks`, the other correctors and the links are read, checked, patched, copied and written but never rendered (N-03). `PIPELINE_VERSION` is 0.8.2.

**Readings of the spec, where it was silent or two places differed.**

- *Errors.* Section 3.5 lists the texts; `GraphError` has one variant for each, carried by `SettingsError::Graph`. The unknown-field rule is `GraphError::UnknownField` ("A corrector has no field called “exposure”.", "An input …", "An output …") for a known field on the wrong kind, in a document and in a patch; a name no node has (a typo) is serde's `SettingsError::Parse`. `NotFinite` now holds a `String`, the path in the document (`nodes.items[2].trim.tint`, items counted in written order). Three more texts the spec did not give: "A mask can't change its kind. Add a new mask instead.", "A Linear mask has no field called “points”." and "Node 2 has no mask number 12."; and "Node 5 is not in the default graph, so it needs its “kind”." for a new id without a kind.
- *Which rule first.* A link is checked in this order: both nodes exist; it does not leave the Output or enter the Input; the ports exist on those nodes (the text names the node that lacks the port: "Node 1 has no port called “mask”.", "The output has no port called “key”."); the two ports have one type. Then an input with two links; the Output's picture linked; Main's picture from the Input; a circle (Kahn's algorithm over picture and key links together; a node whose key feeds itself is a circle). The Output "needs one picture link" when it has none; two links on it are the generic two-links text.
- *Fixed nodes in a document.* `id` and `kind` must agree (`"input"` is an input, 1 is a corrector); a document that gives another kind for a fixed id, or `input`/`output` kind for a number, is "A photo needs its input, its output and node 1, Main." An id other than `"input"`, `"output"` and 1 needs its `kind`.
- *Positions.* A node without `pos` is placed in a row by its number: the Input at −1, the correctors from 0 in id order, the Output after the last; a node added by a patch without `pos` goes at x = the number of correctors it joins. A `pos` outside ±1000 is clamped and reported as the node's `pos`.
- *Names.* Trimmed, cut at 32 characters, an empty one reads as "Main" or "Corrector"; none of this is reported.
- *Auto belongs to Main* is done in `sanitise` (every path that makes or changes a node ends there): Auto in any other corrector becomes a manual 0 (a wheel at the centre) and is not reported. A document that lists Main without a `grade` gets Main's Auto grade; a new corrector's default is the manual-0 grade (`NodeGrade::neutral()`).
- *Patches.* A patch is applied to a copy, the result is checked, and nothing changes on an error. Removing a node drops the links that touch it; removing a node that was on the chain leaves a graph that must still be valid (the patch gives `links` with it when the Output would lose its link). `SettingsPatch::is_empty` is true for a patch that says nothing (an empty `grade` is nothing). `links` equal to the current links is no change. A node patch's fields are checked against the node's kind before they are applied; the first field the kind does not have is named.
- *Undo and paste.* `SettingsPatch::between(from, to)` is the patch that turns one settings into the other: per node the fields that differ (whole top-level fields), a node `to` has and `from` has not whole and with its `kind`, `remove` for the correctors only `from` has, `links` when they differ. `PastedPhoto.before` is `between(new, old)` and `after` `between(old, new)`. `SettingsPatch::pasted_onto(target)` strips the Input's `crop`, `orientation` and `white_pick` and, when the patch carries `links` (a whole-graph copy), adds the target's correctors the patch lacks to `remove`; a patch without `links` replaces nothing. `Session::paste_settings` does it for every target.
- *Reset.* `SettingsPatch::reset(node, field, default_look)` and `reset_node`; the defaults of a corrector other than Main are the manual 0, the Output's the default look.
- *Resolved settings.* `ResolvedSettings.values` is `{ input, main, nodes, output }`: `ResolvedInput` (`exposure_ev`, `kelvin`, `tint`, `sharpen`, `noise_reduction`, and the Input's other fields as they are), `ResolvedCorrector` for Main and for each other corrector (`id`, `name`, `bypass`, `pos`, `trim`, `grade: GradeValues`, `masks`) and `ResolvedOutput`; `auto` is `{ input: InputAutoFlags, main: MainAutoFlags }`; `estimates` (`AutoSummary`) is `{ input, main, source, provider, apple, apple_steps, light_class, light_confidence, noise }`. `pinned_settings` pins the Input's exposure, sharpening and noise reduction and Main's contrast, saturation, highlights, shadows, black and wheels, and nothing else.
- *Schema.* `crates/ym-core/schema/settings.schema.json` describes `{ nodes: { items, links } }` only: the three node kinds (told by `id`), the grade, trim, the nine mask kinds, the link; its test checks what the core writes (a graph with every mask kind), what it refuses, and the ranges. The plain-number forms of `contrast` and the like (a number reads as Manual, 0 as Auto) are kept, as the spec says to move the types and not rewrite them.
- *JPEG XMP.* When a JPEG's `ym:Settings` would not fit the segment (the packet over 65,502 bytes), `export_photo` leaves it out and adds `SETTINGS_TOO_LONG_NOTE` to the report's `notes`.
- *The form of 0.1.* `DevelopSettings::from_value` refuses a document with no `nodes` whose keys are all in `OLD_FORM_KEYS` (the 24 fields and `white_offset_ev`) with `SettingsError::OldForm` (`OLD_FORM_SENTENCE`); the core's `SettingsPatch` refuses a flat patch with the same sentence when deserialised. The app's wire patch has `deny_unknown_fields` and refuses a flat patch as an unknown field. `RetiredField` and the schema's `white_offset_ev` are gone.
- *The CLI.* `Overrides::patch` builds one patch of up to three nodes. A `--settings` file is one document when its only key is `nodes` (or it is empty) and a map when every key names an ARW file; a mixed file, or a key that is neither (a misspelt `nodes` included), is "… not `nodes` or the name of an ARW file …" (exit 2). Clamping warnings name the field as `<node>.<field path>` (`1.grade.contrast`, `input.lens`). The old form is a usage error with the sentence alone.
- *docs/10 and docs/01.* `docs/10` had a table and a `--settings` row twice after N-01 (a merge of two revisions); the duplicate is removed. `docs/01` kept a bullet for `SettingsField::Nodes` and `NodesSettings` from the earlier plan; it now describes the graph as built.
- *The wire* (`src-tauri/src/wire.rs`): the wire types are node-first mirrors with `NodeItem` and `MaskItem` as `kind`-tagged unions; `SettingsPatch.nodes` is keyed by `String` (`"input"`, `"1"`), `NodeId` is the TypeScript type `"input" | "output" | number`; `NodePatch`, `MaskPatch` and `SettingsPatch` have a manual `Serialize` that leaves out what is absent (so `crop: null` is told from no `crop`). `AppApi::touches_thumbnail` looks at the five fields it looked at (look, intensity, crop, orientation, lens) on any node.

**What N-02 measured.** Byte identity holds: `cargo run -p golden --release -- check` renders **61 of 61 files identical** to `expected.json` (largest ΔE00 0.000), with the default graph for every entry (`{}`); the app and `ym` write the same bytes in one build (`crates/ym-cli/tests/parity.rs` 19, `parity_gpu.rs` 7 on lavapipe) and across two builds (`src-tauri/tests/app_cli_parity.rs` 3, with `YM_BIN`). `PIPELINE_VERSION` is 0.8.2. The tests: `crates/ym-core/tests/settings.rs` 60 run and 2 ignored (`N-02b`), `session.rs` 33, `apple_auto.rs` 38, `export.rs` 35, `render_*` and `gpu_*` as before, `ym-core` unit tests 206; `ym-cli` unit tests 69 and `tests/docs.rs` 3 (new); `src-tauri` 170 run and 4 ignored (one of them is `N-02b`'s); `cargo fmt --all --check` and `cargo clippy --workspace --all-targets -- -D warnings` are clean. **What fails until N-02b:** `tsc -b` has seven errors, `src/api.ts` lines 327 to 332 (the settings commands, whose bindings are now node-first) and `src/api/contract.ts` lines 36 and 69 (`resolvedSettings`, `developSettings`, `autoEstimates` and `settingsPatch` no longer match), and `npm run build` with it; `npm test` (1966 tests, the mock is still flat) and the Playwright suite do not touch the generated types and are unaffected. `tools/readme/driver.js` reads `resolved.settings.color_warp` and belongs to N-02b. `cargo deny check` was not run: no dependency changed.


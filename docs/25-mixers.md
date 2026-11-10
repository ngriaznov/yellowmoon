# 25 · Mixers

Mixer nodes join pictures that a graph has split. A **Parallel** node adds the change each of its inputs makes to the picture they started from. A **Layer** node stacks its inputs and blends each one over the ones below it, through the mask of the node that feeds it. A **Splitter** sends the red, green and blue of a picture to three outputs, and a **Combiner** joins three pictures back into one, a channel from each. This file covers what the four node kinds are, their settings and ports, what each one computes, how the plan and the kernels run them, the interface, the command line, the tests, the working choices and the tasks.

**Status: Spec (X-01), written 2026-11-10 on branch `mixers` from the owner's answers of the same day. Nothing is built.** Until X-06 the Add node menu of `docs/23` §7.4 shows the three items greyed with "Later". The mockup round (X-02) and the owner's approval (X-H1) come before any interface work. The work is milestone X of `docs/15`. The facts behind it (how Resolve's mixers work, the blend maths of other tools) are in `research/research_notes/Node mixers/resolve_mixers.md`; this file wins where they differ.

**In short.**

- Four new node kinds in the graph of `docs/23`: `parallel`, `layer`, `splitter` and `combiner` (§2). They are numbered nodes like correctors and count toward the 64. They have no masks, no key ports, no card number and no thumbnail.
- **Parallel** (§3.2): `out = base + Σᵢ (inᵢ − base)` on LogC4 code values. The base is the last picture that every branch shares. The node has no controls.
- **Layer** (§3.3): input 1 is at the bottom and a higher number sits on top. Seven blend modes, computed in linear light. A layer's alpha is the mask of the node that feeds it, times the Layer's Opacity.
- **Splitter and Combiner** (§3.4): `(r, r, r)`, `(g, g, g)` and `(b, b, b)` out, and `(red.r, green.g, blue.b)` back in.
- **Rendering** (§4): a mixer is a per-pixel step that reads several registers. It fuses into the colour pass like any node, and the device needs no new binding.
- `PIPELINE_VERSION` takes a minor step to **0.11.0**. Graphs without the new kinds change by no bit (§2.6).

## 1. What they are

### 1.1 The four node kinds

| Kind | Input ports | Output ports | Fields | Add node item and line |
|---|---|---|---|---|
| `parallel` | `picture.1` … `picture.N` | `picture` | `name`, `bypass`, `pos` | **Parallel**, "Grade branches side by side." |
| `layer` | `picture.1` … `picture.N` | `picture` | `name`, `bypass`, `pos`, `mode`, `opacity` | **Layer**, "Blend one branch over another." |
| `splitter` | `picture` | `red`, `green`, `blue` | `name`, `pos` | **Splitter and combiner**, "Split a picture, then join it again." (one item adds both, §5.4) |
| `combiner` | `red`, `green`, `blue` | `picture` | `name`, `pos` | |

Parallel and Layer are the **mixers**. N is the number of linked inputs, at most `MIXER_INPUTS` = 8.

Three typical graphs:

- **Parallel.** Main feeds two correctors, Skin and Sky, each with its own mask, and both feed a Parallel that feeds the Output. Skin's change and Sky's change are both applied to Main's picture. Where their masks overlap, the two changes add.
- **Layer.** Warmth, a whole-photo corrector, is on input 1 and Glow, a masked corrector, on input 2 (screen 12 of `design/nodes/`). Inside Glow's mask Glow's picture covers Warmth's; outside it Warmth shows.
- **Splitter and combiner.** Each channel of Main's picture goes through its own corrector (Red, Green and Blue) and comes back as one picture.

### 1.2 The owner's decisions (2026-11-10)

The owner asked: **"Let's implement greyed-out node types in nodes view"** (the "Later" items of the Add node menu). The answers:

| Question | The owner's answer | What it means |
|---|---|---|
| Scope | **"All three now"** | Parallel, Layer, and Splitter and combiner, all in milestone X |
| How Parallel combines | **"Input + each branch's change"** | `out = base + Σᵢ (inᵢ − base)`, per channel, on LogC4 code values (the nodes' working space). Each branch keeps its full strength; outside a branch's mask its change is zero; overlapping changes add. The node has no controls of its own |
| Layer blend modes | **"Scene-safe short list"** | Normal, Add, Subtract, Multiply, Lighten, Darken and Difference, computed on linear light: decode LogC4 to linear AWG4, blend, encode back |
| Layer alpha | **"Feeding node's mask, as Resolve"** | each input's layer shows where the mask of the node that feeds it is on, times the Layer's Opacity |

Everything else in this file is a working choice made while writing it, listed in §9. The owner may change any of them at X-H1.

### 1.3 What is not in this milestone

- **Other blend modes.** Screen, Overlay, Soft light, Hard light, Colour dodge and burn, Exclusion and the hue, saturation and luminosity modes assume a display picture with white at 1 and a pivot at 0.5. On LogC4 that pivot sits about 3.6 stops over grey (research note §4). The owner chose the short list.
- **A mode per input.** One mode per Layer, as in Resolve. A Layer can feed another Layer.
- **A weight per input** (Resolve's Key Output Gain). A branch is limited by its own masks and their Opacity.
- **Masks and key ports** on the four kinds (§2.1, §2.2).
- A Key Mixer, an Outside node, and changing a Parallel into a Layer in place (Resolve's "Morph").
- A Splitter into channels other than red, green and blue (YUV, HSL, Lab).
- DCTL (parked, `docs/STATUS.md` question 42).

## 2. Settings and the graph

### 2.1 Kinds and fields

Four new values of `kind` (`docs/23` §3.2). The new nodes are **numbered nodes**: their id is an integer from the same space as correctors and Film nodes, and **they count toward the 64 nodes** of a photo. Each has these fields and no others; another field is an unknown field and is refused.

| Field | Kinds | Type / range | Default |
|---|---|---|---|
| `name` | all four | text, 1 to 32 characters after trimming, as a corrector's | `"Parallel"`, `"Layer"`, `"Splitter"`, `"Combiner"` |
| `pos` | all four | `[x, y]`, as every item's | placed in a row like any node |
| `bypass` | `parallel`, `layer` | bool | false |
| `mode` | `layer` | `"normal"`, `"add"`, `"subtract"`, `"multiply"`, `"lighten"`, `"darken"`, `"difference"` | `"normal"` |
| `opacity` | `layer` | 0 to 1 | 1 |

- An `opacity` out of range is clamped and reported (`7.opacity`); one that is not finite is `SettingsError::NotFinite` and nothing is applied. A `mode` that is not in the list is refused as an unknown mask kind is (`SettingsError::Parse`).
- **No Auto.** Auto belongs to Main (`docs/23` §3.2). Re-run and the Auto providers never touch these nodes.
- **No masks.** None of the four has `masks`. Resolve's mixers take no windows or qualifiers either; the nodes that feed them carry the masks.
- **Names of the kinds in sentences** (`NodeKind::with_article` and `sentence`): "a parallel mixer", "a layer mixer", "a splitter", "a combiner". So: "A parallel mixer has no field called “masks”." and "Node 7 is a layer mixer, not a corrector.".

### 2.2 Ports

| Node | Input ports | Output ports |
|---|---|---|
| a mixer (`parallel`, `layer`) | `picture.1` … `picture.N`, N at most 8 | `picture` |
| a Splitter (`splitter`) | `picture` | `red`, `green`, `blue` |
| a Combiner (`combiner`) | `red`, `green`, `blue` | `picture` |

- **Every one of these ports carries a picture.** A port's type is picture or key: `picture`, `picture.1` to `picture.8`, `red`, `green` and `blue` are pictures, and `key` is a key. A link joins two ports of the same type (`docs/23` §3.5). So a Splitter's `red` may feed a corrector's `picture`, a mixer's `picture.3` or a Combiner's `green` (a channel swap), and a corrector's `picture` may feed a Combiner's `red`. No key link starts or ends on one of the four.
- **Mixers use only the numbered form.** A mixer has no input called `picture`: a link that ends on `{ "node": 7, "port": "picture" }` of a mixer is refused with "Node 7 has no port called “picture”.". The first input has one spelling, so a graph has one written form.
- **N is the number of linked inputs.** The linked inputs of a mixer are `picture.1` to `picture.N` with no gap. The page adds `picture.N+1` and renumbers after a removal (§5.3); the core refuses a gap (§2.3).
- **A picture input with no link reads the Input's picture**, as a corrector's does (`docs/algorithms/06` §1.3, *Readings built in N-03*). For the new kinds: a Splitter with no link on `picture` splits the Input's picture; a Combiner input with no link takes its channel from the Input's picture; a mixer with no linked input has one input, the Input's picture.
- **Written order.** Ports sort as `picture`, `key`, `picture.1` to `picture.8` by number, then `red`, `green`, `blue`. Links are written in the order of `docs/23` §3.1 with that port order, so the same graph is the same bytes.

### 2.3 Validation and limits

The rules of `docs/23` §3.5 hold for the new kinds: the ports exist on their kinds, an input port has one link, the Output has one picture link, Main takes its picture from the Input, no cycles (picture and key links together), ids are unique and kinds are known. New and changed:

| Rule | Error text |
|---|---|
| a mixer's input is `picture.k` with k from 1 to 8; `picture.9` and above are refused | "A mixer can have up to 8 inputs." |
| a mixer's linked inputs are numbered from 1 with no gap | "Node 7 has a link on input 3 and none on input 2. A mixer's inputs are numbered from 1 with no gap." |
| a port the kind does not have: `picture` on a mixer's input side, `picture.0`, `picture.01`, `key` on any of the four, `red` on a mixer | "Node 7 has no port called “picture”." (the existing text) |
| at most 64 numbered nodes, Main included: correctors, Film nodes, mixers, Splitters and Combiners | "A photo can have up to 64 nodes." (unchanged) |
| node kinds are known | the existing text; its example in `docs/23` §3.5 and `docs/10` changes from “parallel” to “outside” |

- **Links.** An input port has at most one link, so a graph holds fewer than 64 × 8 + 1 = 513 links (it was 2 × 64 + 1 = 129). That is the sanity bound of `docs/23` §3.5 and the most a patch's `links` holds. The code has no separate count: the one-link rule is the check.
- A patch that would leave a gap or a ninth input is refused whole, like any patch. The page never sends one.

### 2.4 Examples

A **Parallel of two correctors**. Main feeds Skin (more saturation in a Colour range) and Sky (a cooler trim under a Linear mask); both feed the Parallel, which feeds the Output. The base is Main's picture (§3.2).

```json
{ "nodes": {
  "items": [
    { "id": 2, "kind": "corrector", "name": "Skin", "pos": [1, -0.5],
      "grade": { "saturation": { "kind": "manual", "value": 15 } },
      "masks": { "invert": false, "items": [
        { "kind": "color_range", "center": [0.06, 0.04], "width": 0.05, "softness": 0.5 } ] } },
    { "id": 3, "kind": "corrector", "name": "Sky", "pos": [1, 0.5],
      "trim": { "temperature": -18 },
      "masks": { "invert": false, "items": [
        { "kind": "linear", "center": [0.5, 0.2], "angle_deg": 4, "feather": 0.25 } ] } },
    { "id": 4, "kind": "parallel", "pos": [2, 0] }
  ],
  "links": [
    { "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } },
    { "from": { "node": 1, "port": "picture" }, "to": { "node": 2, "port": "picture" } },
    { "from": { "node": 1, "port": "picture" }, "to": { "node": 3, "port": "picture" } },
    { "from": { "node": 2, "port": "picture" }, "to": { "node": 4, "port": "picture.1" } },
    { "from": { "node": 3, "port": "picture" }, "to": { "node": 4, "port": "picture.2" } },
    { "from": { "node": 4, "port": "picture" }, "to": { "node": "output", "port": "picture" } }
  ] } }
```

A **Splitter and a Combiner** around three correctors, as the Add node item makes them (§5.4): Red lifted by 0.2 EV and Blue lowered by 0.2 EV, so the photo's red channel is lifted and its blue lowered.

```json
{ "nodes": {
  "items": [
    { "id": 2, "kind": "splitter", "pos": [1, 0] },
    { "id": 3, "kind": "corrector", "name": "Red", "pos": [2, -1], "trim": { "exposure_ev": 0.2 } },
    { "id": 4, "kind": "corrector", "name": "Green", "pos": [2, 0] },
    { "id": 5, "kind": "corrector", "name": "Blue", "pos": [2, 1], "trim": { "exposure_ev": -0.2 } },
    { "id": 6, "kind": "combiner", "pos": [3, 0] }
  ],
  "links": [
    { "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } },
    { "from": { "node": 1, "port": "picture" }, "to": { "node": 2, "port": "picture" } },
    { "from": { "node": 2, "port": "red" },     "to": { "node": 3, "port": "picture" } },
    { "from": { "node": 2, "port": "green" },   "to": { "node": 4, "port": "picture" } },
    { "from": { "node": 2, "port": "blue" },    "to": { "node": 5, "port": "picture" } },
    { "from": { "node": 3, "port": "picture" }, "to": { "node": 6, "port": "red" } },
    { "from": { "node": 4, "port": "picture" }, "to": { "node": 6, "port": "green" } },
    { "from": { "node": 5, "port": "picture" }, "to": { "node": 6, "port": "blue" } },
    { "from": { "node": 6, "port": "picture" }, "to": { "node": "output", "port": "picture" } }
  ] } }
```

A **Layer** item as the core writes it:

```json
{ "id": 7, "kind": "layer", "name": "Layer", "bypass": false, "pos": [4, 0],
  "mode": "normal", "opacity": 1 }
```

### 2.5 Patches, Reset, copy, resolved settings, XMP and DNG

- **Patches** (`docs/23` §3.7). A node of a new kind is added by giving its `kind` in `nodes`, and removed with `remove` (any numbered node but Main); its links go with it. `SettingsField` gains `mode` and `opacity`, in the nodes' group (a change runs the colour pass again and nothing else). `name` and `pos` re-run nothing; `bypass` and the links are in the nodes' group, as for a corrector.
- **Reset** (`SettingsPatch::reset_node_in`). A Layer's `mode` and `opacity` go back to `"normal"` and 1. A Parallel, a Splitter and a Combiner have nothing to reset: the patch is empty. Reset never touches a name, a bypass or a link.
- **Copy and paste** carry the new nodes like any node (the whole graph, `docs/23` §3.8). Undo diffs per node and for the links.
- **Resolved settings.** The new nodes have no Auto, so they are in `ResolvedSettings.settings` and `values` as they are.
- **XMP and DNG.** `ym:Settings` holds them; the DNG bakes them into the pixels like every node (`docs/09`).
- **The JSON Schema** gains `parallel_node`, `layer_node`, `splitter_node` and `combiner_node`, and the port names (a pattern for `picture.1` to `picture.8`, and `red`, `green`, `blue`).

### 2.6 Pipeline version

No existing graph changes by a bit. An older build refuses a graph with a kind it does not know, so `PIPELINE_VERSION` takes a minor step to **0.11.0** with the mixers, as it took one to 0.10.0 with the Film node (`docs/01` *pipeline_version*; the owner's "Minor step to 0.10.0" of 2026-11-04). Only the written version string changes: `ym --version`, `ym doctor`, `ym info` and `ym:PipelineVersion`.

## 3. What each node computes

### 3.1 Notation

`v` is a LogC4 triple. `lin(v)` is its linear AWG4 value (`ym_color::logc::logc4_decode_rgb`) and `enc` the LogC4 encode (`logc4_encode_rgb`), the exact formulas of `docs/02`. `inᵢ` is the picture linked into input i (the Input's picture when none is, §2.2). All arithmetic is per channel and per pixel. The CPU reference runs it in f64; the device runs the same expressions in f32 (§4.3).

### 3.2 Parallel

**The base.** A *picture value* is the Input's picture or one picture output of a node (a Splitter has three). A value *leads to* the outputs of every node that reads it on a picture input; a picture input with no link reads the Input's picture. The **base** of a Parallel is the last value that every path from the Input's picture to the Parallel goes through: the immediate dominator of the Parallel in this graph of values, rooted at the Input's picture. The plan compiler computes it from the links alone (`ym_color::plan`), and the page computes it the same way for the panel (`graphOps`).

- **Branches that leave one node.** The base is that node's picture. Main → Skin → P and Main → Sky → P: the base is Main's picture.
- **Branches that leave different places.** The base is the last place they share. Input → Main → A → P.1 and Input → C → P.2: the base is the Input's picture, so `out = A + (C − Input)`. This is the use of a branch fed from the start of the graph that Cullen Kelly describes (research note §1, example 5).
- **Branches through a Splitter.** A Splitter's outputs are three values, so two branches that leave two outputs of one Splitter share only the value that feeds the Splitter, which is then the base.
- **An input that is the base.** Main → P.1 and Main → Sky → P.2: input 1's change is 0, so `out = Sky` up to rounding.

**The sum.** `out = base + Σᵢ (inᵢ − base)`, per channel, on LogC4 code values. In port order: `acc = base`; for i = 1 … N, unless input i is the base value itself, `acc = acc + (inᵢ − base)`; `out = acc`. Whether an input is the base is a fact of the plan (an integer), never a comparison of floats. Nothing is clamped: a sum may leave LogC4's usual range, as a corrector's gain may, and the Output renders any value.

What follows from it:

- A branch whose node has a mask changes nothing outside the mask (the node's output is its input there), so its change is zero there. Where two branches' masks overlap, both changes apply and add.
- Exposure changes add in stops, nearly: two branches that each lift grey by one stop (code 0.2784 to 0.3365) give 0.3947, 1.95 stops over grey, because LogC4 is not a pure logarithm near grey.
- A Parallel with one input gives that input bit for bit (its change is skipped). A Parallel whose branches are all at their identity gives its base bit for bit (each change is exactly 0, and `x + 0 = x`).
- The order of the inputs changes nothing beyond rounding, so the panel lists them without a drag handle (§5.5).

### 3.3 Layer

**The stack.** Input 1 is the bottom layer; a higher number sits on top. Screen 12 draws it as "2 · 06 Glow · On top" over "1 · 05 Warmth · Below". Dragging a row in the panel renumbers the ports (§5.5). Resolve stacks the same way: the first input is the bottom, the last one wins.

**The layer and the alpha of an input.** For each input i from 2 to N:

- If it is linked from the `picture` output of a **corrector or a Film node that is not bypassed**, its layer `Lᵢ` is that node's picture **before its own mask blend**: a corrector's `v_c` (`docs/algorithms/06` §1.1, step 4), a Film node's film picture (`docs/24` §2.6). Its alpha `αᵢ` is that node's `m` times the Layer's `opacity`, with `m` as the node's key output carries it (after the node's Invert; for a Film node, times its Strength). A soft mask edge is therefore applied once, by the Layer, and not once by the node and again by the Layer.
- Otherwise (the Input's picture, a mixer, a Combiner, a Splitter's output, a bypassed node), `Lᵢ` is the picture that is linked in, and `αᵢ` is the `opacity` alone.

A corrector with no mask rows, no Invert and no key input has `m = 1` and `v_c` equal to its output, so the two readings give the same layer. A node's `m` is made for a Layer that reads it even when the node's own grade is at its identity, as it is made for a key output.

**Input 1 is opaque.** It is taken as the picture linked in, whatever feeds it: nothing lies below it, so its node's own blend stands, and outside that node's mask the bottom layer is the node's input.

**Compositing**, from the bottom up, in linear light:

```
acc = lin(in₁)
for i = 2 … N:
    if αᵢ = 0:  next i                 (exactly 0: the layer is not read)
    c = B(acc, lin(Lᵢ))                (the mode, per channel)
    acc = αᵢ = 1 ? c : acc + αᵢ (c − acc)
out = enc(acc)
```

- **No round trip when nothing is on top.** When no layer above the first has `αᵢ ≠ 0` at a pixel, `out = in₁` itself, with no decode and encode. So a Layer at Opacity 0, a Layer with one input, and a pixel outside the masks of every upper layer give input 1 bit for bit.
- **The Input's own picture** is taken in linear as C7 (no decode of C8), as Main reads it (`docs/algorithms/06` §1.1, *The luma*).
- `acc + α (c − acc)` is the W3C rule for a source over an opaque backdrop, `(1 − α)·Cb + α·B(Cb, Cs)` (research note §4).

**The modes**, with `a` the stack below (`acc`) and `b` the layer (`lin(Lᵢ)`), per channel. The examples are computed in f64 with `a` = 0.18 (code 0.2784) and `b` = 0.09 (code 0.2256):

| Mode | Value | Example (linear, code) |
|---|---|---|
| Normal | `b` | 0.09, 0.2256 |
| Add | `a + b` | 0.27, 0.3119 |
| Subtract | `a − b` | 0.09, 0.2256 |
| Multiply | `a · b` | 0.0162, 0.1347 |
| Lighten | `max(a, b)` | 0.18, 0.2784 |
| Darken | `min(a, b)` | 0.09, 0.2256 |
| Difference | `|a − b|` | 0.09, 0.2256 |

At α = 0.5, Normal gives 0.135 (code 0.2557).

- **Subtract takes the layer from what is below it**, `a − b`, as Photoshop and Krita do. Resolve's manual describes the opposite, the bottom taken from the top (research note §2). In a stack the layer is what is applied, so `a − b` is the reading used here.
- **Negative linear values.** LogC4 carries values below 0 linear: out-of-gamut colours, and pictures pushed below black, on the encode's linear segment (`docs/02`). Nodes never clip them (a trim multiplies them like any value, `docs/algorithms/06` §2.1). The modes are kept as written, with no clamp: Normal at α = 1 must give the top layer for every colour, and the Output's cine rendering (C12) already decides what a negative value looks like, so a clamp here would add a second rule. What it means: Subtract and Difference can give values far below black (0.18 − 0.27 = −0.09, code −0.6333), which render as black; Lighten and Darken compare signed values; Multiply of two negative values gives a small positive one (−0.01 × −0.05 = 0.0005, just above black). Every result is finite for finite inputs.

### 3.4 Splitter and Combiner

- **Splitter.** For an input `(r, g, b)` in LogC4, `red` is `(r, r, r)`, `green` is `(g, g, g)` and `blue` is `(b, b, b)`. A node after one output grades a grey picture of one channel, as in Resolve ("red/red/red"). Its masks read that grey picture too, so a Colour range finds no colour in it.
- **Combiner.** `(R.r, G.g, B.b)`: the red channel of the picture on its `red` input, the green of the one on `green`, the blue of the one on `blue`.
- Both move values and compute nothing. A Splitter, three correctors at their identity and a Combiner give the input bit for bit, on the CPU and on the device.

### 3.5 Bypass

- A bypassed **Parallel** outputs its base, and a bypassed **Layer** outputs input 1. Both are bit for bit.
- A Splitter and a Combiner have no Bypass.
- A bypassed node that feeds a Layer is read as its picture, with α the Opacity (§3.3).

## 4. Rendering

### 4.1 The plan

The plan of `docs/23` §5.2 and `docs/algorithms/06` §1.3, with steps that have several inputs and outputs.

- **Steps.** A step reads its picture inputs in port order (one for a corrector, a Film node and a Splitter; N for a mixer; three for a Combiner), its key input (correctors and Film nodes), a Parallel's base, and, for a Layer, the alpha of each input from 2 to N (a key value, or none). It writes the values that something reads: its picture (a Splitter: `red`, `green` and `blue`), its key `m`, and, for a corrector or Film node that feeds input 2 or above of a Layer, its **graded value**, the picture before its mask blend (not a port; only a Layer reads it).
- **Registers.** As now: a value takes a register (pictures) or a key slot (keys and alphas) from the step that makes it to the last step of the pass that reads it, and a step may write into a register its own inputs leave. A Parallel's base is one of its reads, so it stays alive until the Parallel runs.
- **Joins do not split the plan.** A mixer and a Combiner are per-pixel steps that read several registers, so they fuse into the colour pass like any node. The compiler starts a new pass only when the next step would need more than `PLAN_REGISTERS` (8) pictures or `PLAN_KEYS` (16) keys alive at once, as it does today. A step may read its inputs from buffers: a Parallel of eight branches and its base needs nine pictures alive, so it always reads some of them from f32 buffers that an earlier pass wrote. That graph is the test of a split with a join (§7).
- **Live and dead.** The Output needs its picture source and everything that source needs: every picture input of a mixer, a Combiner and a Splitter, and every key input. A mixer is live when the Output needs it, and then the chains of all its inputs are live. A Parallel's base is an ancestor of its inputs, so it is live already. A dead node of the new kinds is dimmed like a dead card.
- **Order and numbers.** The order is today's topological order over picture and key links, with today's ties (Main first, then `pos`, then the id), over every node, the new kinds included. **Card numbers count correctors and Film nodes only.** A mixer, a Splitter and a Combiner take their place in the order and get no number (`design/nodes/mockup-notes.md`, question 8: "Mixers have none"). Main is still 01.
- **Ancestors.** `Plan::picture_ancestors` follows every picture input, the base, the alphas and the graded values back. It gives a Film node the pictures it reads (`docs/23` §5.12, *A Film node reads only its ancestors' pictures*), so a Film node after a Layer reads the pictures of both branches. This is needed for correct pixels, not for speed.
- **Stops.** A thumbnail still stops at a node's picture; the four kinds have no thumbnail of their own (§5.2). A mask view that names one of them shows the photo, as one that names the Input does.
- **The scheduler** (`docs/23` §5.12). Each new kind gets a step kind whose footprint is per pixel and reads the pictures of its inputs (`StepKind::declared` in `ym-color`, `pipeline::schedule::step_kind` in `ym-core`; their exhaustive matches do not compile without it).

### 4.2 CPU reference

`ColorChain::run_nodes` (`ym-color`, `display/chain.rs`) runs the new steps in f64 with the formulas of §3, reading their inputs from registers, key slots and buffers as a corrector's step does. `pipeline::color::plan_graph` and `node_set` (`ym-core`) give the compiler every picture input of every node, and give the chain the Layer's mode and opacity. Threads split rows, and the result is the same for any number of threads (`docs/17` D5). No `fma`. The trace behind the histogram, the scatter and the pixel readout (`ColorChain::trace`) sees the mixers, because they are per pixel, except where a Layer reads a Film node's picture, which the trace does not have (`docs/24` §4.7).

### 4.3 GPU

- **The record.** Today the plan loop of `color.wgsl` reads one picture and one key per step (`R_WIRE` and `R_OUTS` of the 1200-byte `NodeRecU`). A step of the new kinds needs up to 8 picture references (a register, a buffer, the Input's picture, or a Film picture's planes), a Parallel's base, up to 7 alpha references (a key slot, a key buffer, or none), the mode and the opacity, and up to three picture outputs (a Splitter's; a corrector's graded value beside its picture). They go into rows of the record: rows a mixer does not use for a grade, or rows added at its end. X-05 chose rows added at its end (eight, `R_JOIN`; §11), and the `offset_of!` test pins the layout.
- **No new binding.** The records stay in binding 6 and the values that cross passes in binding 8, so the colour kernel keeps 8 storage buffers and 1 uniform (`docs/17` K3).
- **What the kernel needs.** Additions, subtractions, `abs`, `max`, `min`, and the LogC4 decode and encode that the trims use (`logc4_decode3` and `logc4_encode3` of `common.wgsl`). No other `pow`, `log` or `exp`.
- **Metal (K8).** Which values a step reads and writes, whether an input is the base, the mode, and which stop is at which record are integers that the host writes. The float tests that remain (α exactly 0 or exactly 1, the toe of the LogC4 decode and encode, `max` and `min`) choose between results that agree within rounding: each function is continuous across its test. So no result depends on a rounding, and Metal's fast-math (re-association, contraction, its approximate `exp2` and `log2`; `docs/17` §13) moves results only within the parity bound. The bound is measured on lavapipe and on the Mac GPU CI (`.github/workflows/mac-gpu.yml`). If a Layer misses the export's gate there (Multiply of two bright layers doubles the relative error of the decode), the export side uses `precise.wgsl`'s compensated LogC4 pair, as C8 and C10 do (F-08e); a view keeps the plain one.
- **Parity bounds** (X-05 measured them, §11): Parallel and Layer within the grade's 2e-4 in LogC4 and one 10-bit code after C12 and C13, for the pixels of a photo's range; a Layer that subtracts bright layers has its own bounds (§11). The Splitter and the Combiner are exact: they move f32 values, so on the device a trio with identity correctors also returns its input bit for bit.
- **Identity.** A graph without the new kinds runs the expressions it ran before, so lavapipe gives the bytes it gave (`docs/23` §5.4, *Identity*).

### 4.4 Determinism, tiles and the rest

- The same input, settings and version give the same bytes (hard rule 8). The new steps are per pixel, so tiles equal the whole frame bit for bit, a Film node after a mixer included (its reach comes from §4.1's ancestors).
- The app and `ym` write byte-identical files for a graph with the new kinds.
- The DNG bakes them like any node. The thumbnails of the nodes after a mixer include it. The Before view runs no node.

## 5. Interface

The mockups come first (X-02, in `design/mixers/`). This section is the brief for them; the owner's approval (X-H1) settles it. Screen 12 of `design/nodes/` (`12-later-graph.jpg`, `html/pages/12.html`) is the starting point: its glyph cards, its Layer head "Layer · Mixer · 2 inputs" with Bypass, Reset and More, its Blend card with Mode and Opacity, and its Inputs list. Two things in screen 12 change: the Layer has no Mask tab (§2.1), and the cards get a "+" port (§5.2). Copy is British, plain and short, with no exclamation marks.

### 5.1 The Add node menu

The order of `src/screens/nodes/AddNodeMenu.tsx` stays: Corrector, Film, a rule, then the three items, now enabled.

| Item | Line | State |
|---|---|---|
| **Corrector** | "Grades all or part of the photo." | enabled |
| **Film** | "Models a colour negative, a print and a scan." | enabled |
| *(separator)* | | |
| **Parallel** | "Grade branches side by side." | enabled |
| **Layer** | "Blend one branch over another." | enabled |
| **Splitter and combiner** | "Split a picture, then join it again." | enabled |

The tag "Later" and the tooltip "Not in the first version" go. Every item is off at 64 nodes with the tooltip "A photo can have up to 64 nodes."; Splitter and combiner is off from 60 nodes, because it adds five.

### 5.2 Cards and ports

- **Glyph cards.** A mixer, a Splitter and a Combiner are glyph cards, as screen 12 draws the Parallel and the Layer: 56 px wide; 64 px high with two input ports and 24 px more for each further port; a rounded box with the kind's glyph (the menu's icons for Parallel and Layer; X-02 draws the Splitter's and the Combiner's). The name is above the card as on a corrector (cut at ten letters; double click to rename). **No thumbnail and no number chip**; no footer, no mask glyphs and no key ports.
- **States.** Selected: the 2 px yellow ring. Bypassed (a mixer): the glyph at half strength with the slash. Dead: the whole card at 45 % with the tooltip "Not connected to the output.".
- **Ports.** The picture inputs are stacked on the card's left, 24 px apart and centred, in port order from the top; the picture output is at the right, at mid height. A Splitter's three outputs and a Combiner's three inputs are stacked top to bottom as red, green and blue (Resolve's order). X-02 decides whether they carry a colour mark.
- **The "+" port.** A Parallel and a Layer with fewer than 8 inputs draw one more input port under the last: an empty triangle with a "+". Dropping a picture link on it adds the next numbered input (§5.3). With 8 inputs it is not drawn. A mixer with no linked input draws only the "+".
- **Size.** Screen 12 is drawn at 75 % zoom (its correctors are 112.5 × 84 px) while its glyph cards are 56 px wide, so X-02 confirms the glyph card's size at 100 %. A card with many ports is taller than a grid unit's 150 px (nine ports: 232 px); X-02 checks how the cards around it sit.
- **Code.** The cards are xyflow nodes of their own type, with one `Handle` per port (the port's name is its id; the "+" has an id of its own that never reaches the settings). A card asks xyflow to measure it again when its ports change (`useUpdateNodeInternals`).

### 5.3 Links by hand

The rules of `docs/23` §7.2 hold. Added:

- **What lights up.** A link dragged from a picture output lights every picture input that may take it, mixer inputs, "+" ports and Combiner inputs included, and never a key port. A key link lights nothing on the four kinds.
- **Drops.** A drop on "+" adds `picture.N+1`. A drop on an input that has a link replaces it, as on any input.
- **Renumbering.** After any change to a mixer's links (a link removed, an end picked up and dropped elsewhere, a node deleted upstream), its inputs are numbered 1 to N again in their old order. Removing the link into input 2 of a Layer with three inputs makes input 3 the new input 2, and the stack keeps its order. A drop makes the new link first and then removes the lifted one, so a Layer's input 1 picked up and dropped on its own "+" goes to the top.
- Each connect, move or removal is **one patch (`links`) and one undo step**.
- **The keyboard path.** More › Link picture to ▸ lists a mixer's inputs as "Layer, input 1", "Layer, input 2" and "Layer, a new input", and a Combiner's as "Combiner, red", "Combiner, green" and "Combiner, blue". More › Remove link ▸ names them the same way.

### 5.4 Graph operations

Each is one patch and one undo step, built by pure functions in `src/state/graphOps.ts`.

- **Add node › Parallel or Layer** (the owner at X-H1, 2026-11-10: **"Add a corrector too, as Resolve"**). With S selected, fed from U (S's picture input's source; the Input when it has none), and S → T: a new corrector N (manual 0, the next default name) is made beside S, fed from U too; S goes to the mixer's input 1 and N to its input 2, so on a Layer N is on top; the mixer M feeds what S fed (M → T, every outgoing picture link of S moves to M). So U → S → M (`picture.1`), U → N → M (`picture.2`), M → T, and a Parallel's base is U. With the Input selected, S is Main; with the Output selected, S is the node that feeds the Output; S may be a mixer, a Splitter output's node or a Combiner (any node with a picture output). Places: N one row under S at S's x (the nodes in its way move down one row), M one unit right of S, and the nodes after S move one unit right. N is selected with its tab open, so there is something to grade at once. It is one patch and one undo step; the 64-node limit counts both new nodes (the items are off from 63 nodes).
- **Add node › Splitter and combiner.** With S selected and S → T, the result is S → Splitter; Splitter `red` → Red, `green` → Green, `blue` → Blue (three new correctors, manual 0 everywhere, named "Red", "Green" and "Blue"); Red → Combiner `red`, Green → `green`, Blue → `blue`; Combiner → T. Places: the Splitter one unit right of S, the correctors two units right on the rows y − 1, y and y + 1, the Combiner three units right; T and the nodes after it move three units right. The Splitter is selected. One patch adds all five.
- **Delete** (any of the four). Every node that it fed, through any of its picture outputs, is linked instead from the source of its **first picture input** (`picture.1` of a mixer, `picture` of a Splitter, `red` of a Combiner; the Input when that input has no link). Its other links go. This is the corrector's rule (U → Tᵢ) with "the first input" for "the input", so the Output never loses its link: a deleted Layer leaves its bottom picture, a deleted Parallel its first branch. The source of the first input is selected (Main when it is the Input).
- **Reorder** is refused for the four kinds, with "This node can't be moved along a line. Change its links instead." (Move earlier, Move later, ⌥← and ⌥→, and a card dropped on a link; a drag still moves the card on the canvas). A corrector or Film node may be dropped on a link into a mixer input or out of a Splitter output, and goes into that link. Move earlier from a node whose source is one of the four goes into the link that feeds that node's first picture input; Move later works as it does now.
- **Bypass** (the toolbar, B, the head). A Parallel and a Layer toggle `bypass`. A Splitter and a Combiner refuse it: "A splitter or a combiner can't be bypassed."
- **Rename**: all four, as a corrector.

### 5.5 The panel

The head is the corrector's compact head (`docs/23` §7.5) with the kind's glyph in a chip where the number chip is, as screen 12 draws it.

| Node | Head line | Icon buttons | Tabs |
|---|---|---|---|
| Parallel | "Mixer · 3 inputs" ("Mixer · 1 input"), "Mixer · bypassed", "Mixer · not connected" | Bypass, Reset (off), More | **Inputs** |
| Layer | as the Parallel's | Bypass, Reset, More | **Blend** |
| Splitter | "Channels · 3 outputs", "Channels · not connected" | Bypass (off), Reset (off), More | **Channels** |
| Combiner | "Channels · 3 inputs", "Channels · not connected" | Bypass (off), Reset (off), More | **Channels** |

- **Off buttons** are shown and `aria-disabled`. Reset's tooltip is "This node has nothing to reset."; Bypass's is "A splitter or a combiner can't be bypassed.".
- **More**: Rename · Link picture to ▸ · Remove link ▸ · Delete node. No Move items and no Link key to.
- **Layer › Blend.** The **Blend** card (its Reset puts Mode to Normal and Opacity to 100 %): **Mode**, glasscn's `Select` with the seven modes in the order of §3.3, and **Opacity**, a slider from 0 to 100 % in steps of 1 with its value box ("100%"). The **Inputs** card: one row per input, from the top of the stack (input N) to the bottom (input 1). A row has a drag handle, the feeding node's thumbnail (its glyph for a mixer, a Splitter output or the Input), the input's number chip, the feeding node's number and name ("06 Glow"), and "On top" on the first row and "Below" on the last. Under the rows: "Drag a row to change which input sits on top." A drag renumbers the ports: one patch, one undo step. The keyboard path is a row's context menu: Move up · Move down · Remove input.
- **Parallel › Inputs.** The **Inputs** card: "Adds the change each input makes to the picture they start from." and the base: "They start from node 1, Main." ("They start from the input." when the base is the Input's picture, "They start from Splitter, red." for a Splitter's output). Then the rows, as the Layer's without the drag handle and the words, because the order changes nothing. A row's context menu: Remove input.
- **Splitter › Channels.** "Sends the red, green and blue of the picture to three outputs. Each output is a grey picture of one channel, so the nodes after it grade that channel." Three rows, Red, Green and Blue, each with where it goes ("To node 3, Red", "Not linked").
- **Combiner › Channels.** "Makes one picture: the red of its red input, the green of its green input and the blue of its blue input." Three rows with where each comes from ("From node 3, Red", "From the input" when not linked).
- **Tab colour.** Proposed: the Look tab's yellow, as the Film node's tabs have (`docs/11`). X-02 shows it.
- **No histogram** on these panels. **Show mask** (the toolbar's toggle and M) is off on them, `aria-disabled` with the tooltip "This node has no mask.".
- **Keys.** ⌘1 selects the one tab. ⌥← and ⌥→ are refused as Reorder is.

### 5.6 Accessibility

- **Card names.** "Layer, mixer with 2 inputs, from node 5 and node 6" (the inputs in port order), with ", bypassed" and ", not connected" where they apply; "Splitter, picture from node 1"; "Combiner, red from node 3, green from node 4, blue from node 5". A source with no number is named by its name ("from Parallel", "from Splitter, red"), and the Input as "the input". A corrector's name names such a source the same way (", picture from Layer").
- The "+" ports are not Tab stops, as no port is; the keyboard path is the More menu.
- The Layer's input rows are focusable and have the context menu of §5.5.
- The a11y and glass audits gain the new page states (X-06 counts them).

### 5.7 Elsewhere in the app

- **The simple view's notice** counts every node but the Input and the Output: "This photo has 6 nodes." (`docs/23` §6).
- **Thumbnails.** `node_thumbnails` makes none for the four kinds.
- **No new command.** `histogram`, `chroma_scatter` and `sample_chroma` are not asked at these nodes; an `at` that names one reads the picture after it (for a Splitter, its input).
- **Wire types** (`docs/23` §9). `NodeKind` gains the four kinds; `Port` gains `picture.1` to `picture.8`, `red`, `green` and `blue`; `NodeItem` gains `ParallelNode`, `LayerNode`, `SplitterNode` and `CombinerNode`, with `BlendMode`. `bindings.ts` is regenerated and the field-name test compares them.
- **The mock** (`src/api.mock.ts`). Its validator (`src/fixtures/graph.ts`) has the rules and texts of §2.3, checked against the core's by `src/repo.node.test.ts`. Its flat scenes go through the formulas of §3 (`src/fixtures/grade.ts`); the mock draws no masks, so a layer's alpha is the Opacity.

### 5.8 Copy

New lines, drafts for X-H1: the menu's three lines (unchanged, now live); the head lines of §5.5; "Drag a row to change which input sits on top." (screen 12); "On top", "Below"; "Adds the change each input makes to the picture they start from."; "They start from node 1, Main." and its variants; the Splitter's and the Combiner's sentences; "To node 3, Red", "From node 3, Red", "From the input", "Not linked"; "This node has nothing to reset."; "A splitter or a combiner can't be bypassed."; "This node has no mask."; "This node can't be moved along a line. Change its links instead."; the rule texts of §2.3; the card names of §5.6; "Layer, input 2", "Layer, a new input", "Combiner, red"; "Move up", "Move down", "Remove input".

## 6. Command line

- **No new flags.** The flags set fields of the Input, Main and the Output (`docs/10`), and the new kinds are not in the default graph. A graph with them comes through `--settings`; the two files of §2.4 are `docs/10`'s examples, and the docs test reads and validates them (`crates/ym-cli/tests/docs.rs`).
- **The help.** The closing paragraph of `ym render --help` names the new kinds: "other nodes (correctors, mixers, splitters and combiners, and Film nodes whose stocks `ym film list` shows)". `docs/10` quotes it word for word and the snapshot `cli__help_render` changes with it, in X-03.
- **Errors.** A file that breaks a rule of §2.3 is a usage error (exit 2) with the rule's sentence after "The settings file can't be used:", as for every graph rule.
- **JSON.** The hidden `ym render --json` and `ym info --json` print the new kinds in `settings` as the document holds them, with their `kind`.
- **The unknown kind.** `docs/10`'s example becomes “outside”, and so do the tests that use "parallel" as the unknown kind (`crates/ym-core/tests/settings.rs`, `crates/ym-cli/tests/render.rs`).
- **Version.** `ym --version`, `ym doctor` and `ym info` print pipeline 0.11.0 from X-07.

## 7. Tests and gates

Every gate must pass in CI (or the named job) before its task is done. **A failed gate is reported, never loosened** (`AGENTS.md`). The numbers marked *starting* are first bounds, which the named task measures and writes into `docs/STATUS.md`.

| Area | Test | Gate | Task |
|---|---|---|---|
| Format and rules | each rule of §2.3 with its text; the ports of every kind; port types (a Splitter's `red` into a Combiner's `green` is allowed, a key into a mixer is refused); a gap, a ninth input and a bare `picture` on a mixer; 64 numbered nodes counting the new kinds; the unknown kind “outside”; the fields, their defaults, `opacity` clamped and reported, an unknown `mode`, a value that is not finite; the written order of ports and links (a round trip is the same bytes); the schema | every refusal is atomic and has its plain text | X-03 |
| Patches | adding and removing each kind; `mode` and `opacity` patches and their report (`7.opacity`, the nodes' group); Reset of each kind (a Layer to Normal and 1, the others an empty patch); copy and paste of a graph with all four; undo per node and for the links | tested | X-03 |
| Existing graphs | every golden entry and settings fixture, the CLI's and the app's parity tests, the device suites | **byte-identical**; `golden check` 61 of 61 (no default graph uses a mixer); only the written version moves, in X-07 | X-04, X-05, X-07 |
| Formulas (CPU) | each mode on hand-computed values (the table of §3.3, a negative `a`, a negative `b`, two negative values for Multiply), α = 0.5; Parallel sums on hand-computed values (two one-stop branches give 0.3947); the Splitter's and the Combiner's channels; a channel swap | 1e-12 in LogC4 | X-04 |
| Identities (CPU) | a Parallel with one input equals the input; a Parallel whose branches are at their identity equals its base; a bypassed Parallel equals its base; a Layer at Opacity 0, with one input, or bypassed equals input 1; a Layer at Normal and α = 1 equals its top input; a Splitter, three identity correctors and a Combiner equal the input | bit for bit; the Normal case within 1e-12 (one decode and encode), and bit for bit when the top is the Input's picture (C7) | X-04 |
| The mask once | a Layer whose top input is a corrector with a feathered Linear mask: where `m` = 0.5 the result is `enc(lin(bottom) + 0.5 (lin(v_c) − lin(bottom)))`, not a blend of the node's blended picture | 1e-12 | X-04 |
| The plan | the base in each case of §3.2 (one branching node, different places, through a Splitter, an input that is the base, nested mixers); a Layer of two branches compiles into one pass; a Parallel of eight branches compiles into two passes with buffers and renders the CPU reference's picture; live and dead with the new kinds; card numbers skip them; `picture_ancestors` of a Film node after a Layer holds both branches; two files that list one graph in different orders give one plan | exact order; 2e-4 for the split | X-04 |
| Random graphs | `crates/ym-core/tests/any_graph.rs` draws the four kinds from `NodeKind::ALL`: mixers with one to three inputs from earlier pictures, a Layer's mode and opacity, Splitters whose outputs feed later inputs, Combiners with linked and unlinked inputs; the run asserts that every kind appeared | CPU: tiles and threads bit for bit (X-04). Device: on the device, not refused, within the CPU bounds, exports byte for byte across tile sizes (X-05) | X-04, X-05 |
| Determinism and tiles | a repeat render; tiles of 97, 160 and 256 px on 1, 3 and 4 threads equal the whole frame with a Layer and a Parallel; a Film node after a Layer | bit for bit | X-04, X-05 |
| App against `ym` | the two example files of `docs/10` and a Layer with masked inputs: JPEG, TIFF 16, PNG 16 and DNG | byte-identical | X-04, X-05 |
| Device parity | lavapipe: every mode, Parallel up to 8 inputs, a split with a join, negative values, a Layer that reads a Film node; the Mac GPU CI (Metal, fast-math), the same cases | 2e-4 in LogC4 and one 10-bit code after C12 and C13 (*starting*); the Splitter and the Combiner exact | X-05 |
| CLI | `docs/10`'s example files read and validated; the help's closing paragraph; the unknown-kind sentence (`crates/ym-cli/tests/docs.rs`) | the test fails on any difference | X-03 |
| Interface | `graphOps` unit tests: adding each kind and the trio, Delete with the link from the first input, the "+", renumbering after a removal and after a pick-up, the Layer's row drag, the refusals (Reorder, Bypass on a Splitter, a key into a mixer, a ninth input, circles, 64 nodes); `cardStates` gives the four no number; the base line of the Parallel's panel. Playwright: the menu (five items, all enabled), the cards and ports, a link dropped on "+", a row dragged, each panel, the keyboard path. The glass audit and a11y with the new states. Reference screenshots | green; the owner sees the screenshots before they are committed | X-06 |
| Speed (on a Mac) | a slider move in the fit view with 8 nodes, two of them mixers | ≤ 100 ms (*starting*, the node view's target) | X-07 |
| Licences | `cargo deny check` and the npm licence check | pass; no new dependency is expected | X-03 to X-06 |

## 8. Risks

1. **The fused kernel grows.** The plan loop gains a step kind that reads up to nine values. Register pressure on older Macs is unknown; the Mac's speed check (X-07, M0-H4) measures it.
2. **The Layer's precision on the device.** The decode and encode in f32, and Multiply, which doubles the relative error. The export's fallback is the compensated pair (§4.3).
3. **The base can surprise.** When branches start at different places, the base is the last place they share. The Parallel's panel says where they start.
4. **A sum in LogC4 is not light added.** Tints added in code values are not the sum of two lights; that is the owner's choice (§1.2). A Layer's Add is the physical one.
5. **Negative values** in Subtract, Difference and Multiply (§3.3). Documented, finite and rendered by C12.
6. **Tall glyph cards** with many ports (§5.2).
7. **The unknown-kind example** “outside” names a kind that a later version may add (screen 12 draws an Outside node); the example would then change again.
8. **Hand-written graphs** with many joins take the several-pass path. It is tested (§7).

## 9. Decisions made while writing the spec

The owner's four answers are §1.2. These are the working choices: the simplest readings where the answers were silent. **Each is a working choice that waits on X-H1**, and the owner may change any of them. Items 1 to 25 are the lead's brief of 2026-11-10; items 26 to 36 were made while writing this file.

| # | Choice | Why |
|---|---|---|
| 1 | The kinds are `parallel`, `layer`, `splitter` and `combiner` | the menu's names, one word each |
| 2 | `PIPELINE_VERSION` 0.11.0, a minor step; graphs without the new kinds change by no bit | the rule the Film node set (`docs/01`) |
| 3 | Mixers have inputs `picture.1` … `picture.N`, N from 1 to `MIXER_INPUTS` = 8, and one `picture` output; no key ports | numbered inputs as Resolve's; eight keeps the record and the registers small |
| 4 | A mixer has no bare `picture` input; the numbered form is the only one | one spelling for the first input |
| 5 | The Splitter has a `picture` input and `red`, `green`, `blue` outputs; the Combiner has `red`, `green`, `blue` inputs and a `picture` output | Resolve's three outputs and inputs, named |
| 6 | Validator: one link per input; a mixer's inputs numbered from 1 with no gap, at most 8; the Splitter's and Combiner's ports exact; no cycles; links bounded by 64 × 8 + 1 = 513 | the existing rules, extended; the old 129 was 2 × 64 + 1 |
| 7 | The unknown-kind example becomes “outside” | "parallel" is now a kind |
| 8 | The Parallel's base is the immediate dominator of the mixer in the graph of picture values from the Input; an input that is the base adds nothing | the last picture every branch shares; Resolve does not say |
| 9 | A bypassed Parallel outputs its base | nothing of the branches is applied |
| 10 | Layer: input 1 at the bottom, higher numbers on top; a row drag renumbers the ports | Resolve's order and screen 12's words |
| 11 | Layer fields `mode` (default `normal`) and `opacity` (0 to 1, default 1) | screen 12's Blend card |
| 12 | Compositing bottom-up in linear: `acc = lin(in₁)`, then `acc + αᵢ (B(acc, lin(Lᵢ)) − acc)`, then encode | the W3C blend over an opaque backdrop |
| 13 | The formulas of the seven modes (§3.3), per channel | the standard ones |
| 14 | Negative linear values are kept, no clamp | LogC4 carries them and C12 renders them (§3.3) |
| 15 | A Layer input from a corrector or Film node that is not bypassed takes that node's picture before its mask blend, with α its `m` (a Film node's times Strength) times Opacity; anything else is its picture with α the Opacity | the owner's "as Resolve", with a soft edge applied once |
| 16 | A bypassed Layer outputs input 1 | the bottom of the stack |
| 17 | Splitter `(r, r, r)`, `(g, g, g)`, `(b, b, b)`; Combiner `(red.r, green.g, blue.b)`, in LogC4 | Resolve's "red/red/red" |
| 18 | The Splitter and combiner item adds a Splitter, three correctors named Red, Green and Blue, and a Combiner, linked | Resolve's ready-made trio |
| 19 | The Splitter and the Combiner have no Bypass and no fields to reset; Reset is shown off with a tooltip | nothing to bypass or reset |
| 20 | No masks on the four kinds; the Layer's Mask tab of screen 12 is dropped | Resolve's mixers take none; the feeding nodes carry them |
| 21 | Panels: the Layer has one tab, Blend, with the Inputs card; the Parallel's panel says what it does and lists its inputs; the Splitter's and the Combiner's say what they do | there is nothing else to set |
| 22 | Glyph cards of 56 px with no thumbnail and no number; picture inputs stacked on the left; a "+" port below them while a mixer has fewer than 8 inputs; removing an input renumbers the later ones | screen 12, and a way to add an input by hand |
| 23 | Add node › Parallel or Layer goes into the selected node's outgoing picture link, which becomes input 1; Delete links the nodes it fed from its first input's source; the four kinds cannot be reordered along a line | the corrector's operations, so the Output never loses its link |
| 24 | The plan does not split at a join; a mixer reads several registers and fuses into the colour pass; the base stays alive until the Parallel runs; `picture_ancestors` follows every input; the live set is the Output's ancestors | splitting only when registers run out keeps one pass |
| 25 | CPU in f64, device in f32 with stated bounds; the Metal rule of `docs/17` K8 | as every node |
| 26 | **A `name` on all four kinds**, default the kind's name, renamed as a corrector is | screen 12 draws names above the mixer cards; two Layers must be told apart in the More menu and in accessible names |
| 27 | **The new kinds count toward the 64 nodes** | one limit; the sentence "A photo can have up to 64 nodes." stays true |
| 28 | **An input with no link reads the Input's picture**, for every input of the new kinds: a mixer with no linked input has one input, the Input's picture | the corrector's existing rule; no new refusal |
| 29 | **The Parallel sums in port order**, skipping an input that is the base; nothing is clamped | a fixed order makes the bits fixed |
| 30 | **A Layer at a pixel where no upper α is nonzero outputs input 1 with no decode and encode**; the Input's own picture is read in linear as C7 | identities bit for bit, as nodes at their identity are |
| 31 | **Input 1 of a Layer is opaque and taken as its node's output** (its node's blend stands) | nothing lies below it |
| 32 | **Subtract is `a − b`**, the layer taken from what is below; Resolve's manual says the opposite | a layer is what is applied to the stack |
| 33 | **Any picture output may feed any picture input**, channel swaps between a Splitter and a Combiner included; a Splitter's output may feed the Output | port types, not port names, decide |
| 34 | **Places and selection** of the Splitter-and-combiner trio (one, two and three units right; the correctors on three rows; the Splitter selected), and its limit of 60 nodes | room for five nodes |
| 35 | **The words of the interface** (§5.5, §5.6, §5.8): the tabs Inputs, Blend and Channels; the head lines; "On top" and "Below" (screen 12's words, also with more than two inputs); the off buttons' tooltips; the card names; the simple view's notice counting every node but the Input and the Output | drafts for X-02 |
| 36 | **Answered at X-H1 (2026-11-10): "Add a corrector too, as Resolve".** Add node › Parallel or Layer makes a corrector beside the selected node, on the mixer's input 2 (§5.4) | the owner's choice; one step to a working branch |

## 10. Tasks

Task format of `docs/15`. The work is on branch `mixers`, one reviewed commit per task with its checks green (`cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`, `npm test`, `npm run build`). Every task that changes what `ym` does or prints updates `docs/10` and the help in the same commit.

**X-01 · Spec** (): this file; `docs/23` §1.1, §3.2, §3.4, §3.5, §3.7, §5.2, §7.4, §7.5 and §7.8; `docs/algorithms/06` §1.3; milestone X in `docs/15`; `docs/10` and `docs/11` where they would otherwise be wrong; `docs/STATUS.md` (question 43). *Done when:* the specs agree with each other and the lead has read them.

**X-02 · Mockups** (X-01): `design/mixers/`, made like `design/nodes/`: pages at 1440 × 900 built from the app's DOM and CSS (`html/pages/*.html` with their stylesheet and assets), the same screens as 2× JPEGs, `index.html` and `README.md`. The screens: the Add node menu with the items live; a graph with a Parallel of three branches and its Inputs tab; a Layer with its Blend tab and the Mode menu open; the Splitter-and-combiner trio just added, with the Splitter's tab; the Combiner's tab; a sheet of card states (a mixer normal, selected, bypassed, dead, with "+", with 8 inputs; a Splitter; a Combiner); a link dropped on "+"; a Layer row being dragged. The visual working choices of §9 are drawn, with options where the owner should choose. *Done when:* the owner approves them (X-H1).

**[HUMAN] X-H1 · Approve the mockups and the working choices of §9.** No interface work (X-06) starts before it.

**X-03 · Format, validation, patches, types, schema and CLI** (X-01): `NodeKind` with the four kinds (`ALL`, names, sentences); `Port` with the numbered and channel ports, the type of each and the written order; the fields, defaults and clamping; validation with the texts of §2.3; patches (add, remove, `mode`, `opacity`), Reset, copy and paste, resolved settings and XMP; the JSON Schema; the wire types, `bindings.ts`, `types.ts`, `contract.ts`, the mock's validator and its text test (`src/repo.node.test.ts`); `ym --settings` reads and validates such files; `docs/10` (the node-first form names the kinds, the two example files of §2.4 as JSON blocks of its *The node-first form*, the unknown kind “outside”) and the help's closing paragraph, with `crates/ym-cli/tests/docs.rs` (it counts that section's JSON blocks, four today, and reads and validates each) and the help snapshot; the tests that use "parallel" as the unknown kind use "outside". No change in the app's interface. Until X-04 the plan compiler refuses the new kinds with a plain error, never a wrong picture (on the branch only). *Done when:* the format, rule, patch, schema, copy and CLI gates of §7 pass, every rule of §2.3 has a test with its text, and every existing render is byte-identical.

**X-04 · The plan and the CPU reference** (X-03): `ym_color::plan` with steps of several inputs and outputs (the base, the graded values, the alphas, the Splitter's outputs, live and dead, numbers that skip the four kinds, `picture_ancestors` through every input); the scheduler's step kinds; `plan_graph` and `node_set`; `ColorChain::run_nodes` with §3 in f64. Until X-05 the device refuses a plan with a new kind, and the frame is the CPU's with its reason (`RenderReport::cpu_reason`). The random-graph generator learns the four kinds and runs them on the CPU (tiles, threads, determinism); its device assertions for them wait for X-05. *Done when:* the formula, identity, mask-once, plan and CPU gates of §7 pass, the app and `ym` agree on the two example files, and `golden check` is 61 of 61 unchanged.

**X-05 · The device** (X-04): the record of §4.3, the WGSL of the new steps, the graded values and the Splitter's outputs; parity on lavapipe; the random-graph test's device assertions for the four kinds; `docs/17` (the record, K3 unchanged, the K8 note of §4.3); Metal through the Mac GPU CI. *Done when:* the parity, tiles, determinism and random-graph gates pass on lavapipe and on the Mac GPU CI.

**X-06 · The interface** (X-H1, X-03; X-05 for pictures from the device): the menu; the glyph cards with numbered ports and "+"; the link rules and the renumbering; the graph operations in `graphOps` (add a mixer, add the trio, Delete, the refusals); the panels and their heads; Show mask and Reset off; the accessible names; the mock's grades; e2e and unit tests (the menu's items are live: `e2e/nodes.spec.ts`, `e2e/a11y.spec.ts` and `src/screens/nodes/NodeView.test.tsx` stop expecting them `aria-disabled` with "Not in the first version"); the glass audit and a11y with the new states; reference screenshots, which the owner looks at before they are committed; `docs/11` (the panel and the cards as built) and `docs/23` §7 in step. *Done when:* the unit tests, e2e, a11y and the glass audit pass, and the owner has seen the screenshots.

**X-07 · Finish** (X-05, X-06): the gates of §7 run and reported in `docs/STATUS.md`; `PIPELINE_VERSION` 0.11.0 (`ym --version`, `ym doctor`, `ym info` and their snapshots; `docs/01` *pipeline_version*); `CHANGELOG.md`; the README's node paragraph; `docs/16` (a Mixers entry with the owner's decisions); the speed check on a Mac listed under *Waiting on the owner* (M0-H4); a test build. *Done when:* the gates are in `docs/STATUS.md` and the test build is with the owner.

**[HUMAN] X-H2 · Test build and review on a Mac.** The owner's OK merges `mixers` to `main` and releases it; the version number is the owner's.

**Order.** X-01, then X-02 and X-H1. X-03 may start after X-01, since it changes nothing the person sees in the app; X-04 after X-03; X-05 after X-04; X-06 after X-H1 and X-03; X-07 last.

**Gate X:** the gates of §7 pass; `golden check` 61 of 61 unchanged and no existing graph changed by a bit; the owner's approval of the last test build (X-H2).

## 11. As built

*(Filled in by the tasks as they land: what each task decided, what it measured, and what differs from this file.)*

**X-03 · Format, validation, patches, types, schema and CLI (built on branch `mixers`).** The four kinds are in `ym-core` (`settings/mixers.rs`: `ParallelNode`, `LayerNode`, `SplitterNode`, `CombinerNode`, `BlendMode`, the enum `MixerNode`; `NodeGraph::mixers` holds them by id beside `extra` and `films`). `NodeKind::ALL` has eight kinds, `Port` has the variants `PictureN(1..=8)`, `Red`, `Green` and `Blue` (its derived order is the written order of §2.2), `ports_in` and `ports_out` give the ports of each kind, and `SettingsField` gains `Mode` and `Opacity` (the nodes' group). The rules of §2.3 are in `NodeGraph::check`, with two new `GraphError` variants for the two new texts; a mixer's gap names the first linked input after the first missing one ("a link on input 3 and none on input 2"), and a port `picture.9` and above on a mixer's input side is "A mixer can have up to 8 inputs." (on any other node it is the plain unknown-port sentence). The patches, Reset (a Layer's `mode` and `opacity`; empty for the other three), copy and paste, undo, the JSON Schema, the wire types, `bindings.ts`, `types.ts`, the mock's validator (`src/fixtures/graph.ts`) and `ym --settings` follow §2. `ResolvedSettings.values` lists the extra correctors only, as it does not list Film nodes either: the new nodes are in `ResolvedSettings.settings`. Until X-04 `pipeline::color::node_plan` (and so every render, thumbnail and export) refuses a graph that has any of the four kinds, even a dead one, with "This photo has a mixer, a splitter or a combiner node, which this version of Yellowmoon can't render yet. Remove the node to render the photo."; `plan_graph` and `node_set` are exhaustive `match`es over the kinds, `schedule::step_kind` gives the four a per-pixel stand-in, and `tests/any_graph.rs` lists them in `NOT_GENERATED_NODES` until X-04. The interface is unchanged: the mock keeps the nodes' types in line and a node of the four kinds reads as a corrector's panel, with no card name or head line, until X-06.

**X-06 · The interface (built on branch `mixers`).** The menu, the glyph cards, the links, the graph operations and the panels of §5 are built as `docs/11` *The nodes of docs/25 as built* and `docs/23` §7.4 describe; the owner's three answers at X-H1 are in (the corrector with Add › Parallel or Layer, tinted channel ports with letters, a mode menu of names). What the build decided or changed:

- **Card numbers and liveness.** `cardStates` keys every numbered node and gives the four kinds `number: 0`: they are in the plan's order (so a corrector's number does not change when a mixer is added) and in the liveness walk (a Parallel's, a Layer's, a Combiner's every input; a Splitter's one), but they take no number. The simple view's notice counts every node but the Input and the Output (§5.7), Film nodes included.
- **Add node › Splitter and combiner.** The nodes after the Combiner move right as far as it takes to leave it its unit (three units when they were next to S, none where a gap is), as `insertAfter` does, not a fixed three. A Parallel or Layer added with a Splitter selected takes the picture of the output that leads to the Output (else the first), and moves only the links of that output.
- **Add node › Parallel or Layer.** The nodes in the column of S from one row under it move down one row (a node at S's x within half a unit); the nodes after S shift as `slotFor` does.
- **Delete** selects the node the first input came from (Main when it is the Input) and, as every link change, numbers the mixers' inputs from 1 again. **Reorder** refuses the four kinds with "This node can't be moved along a line. Change its links instead."; a corrector dropped on a link into a mixer input or out of a Splitter output goes into it (the port keeps its number); Move earlier from a node whose source is one of the four goes into the link that feeds that node's first picture input.
- **The "+" and the drop target.** The "+" is the handle `plus` and stands for `picture.N+1` in `inputStates`; the hit targets of ports 24 px apart overlap by 4 px and the nearer centre wins. Inputs that would close a circle dim, as in the mockup.
- **Card names** put ", bypassed" and ", not connected" before the list of sources, as a corrector's name does (§5.6 gave the sources first).
- **The mock** (`src/fixtures/mixers.ts`): a CPU approximation of §3 on the four colours of a flat scene, enough for the page and the screenshots; it is not the core's maths and draws no masks. A Parallel's inputs that are the base add nothing; a Layer's alpha is the Opacity.
- **Not built here.** The real app refuses to render a graph with any of the four kinds until X-04 lands (`MIXERS_NOT_RENDERED`); the pictures from the device wait for X-05. The Layer-row drag has no auto-scroll (the list is short).

**X-04 · The plan and the CPU reference (built on branch `mixers`).** `ym_color::plan` compiles the four kinds: `compile_with(&PlanGraph, &[PlanJoin], Stop)` (`compile` is the same without joins). A mixer, Splitter or Combiner is a `PlanJoin` (id, `pos`, `JoinKind`, its picture inputs in port order, each with a `graded` flag that `ym-core`'s `plan_joins` sets for a Layer input above the first that is linked from a corrector or Film node that is not bypassed); `PictureSource` gains `Split(node, channel)` for a Splitter's outputs. A `Step` keeps its four old fields and gains `graded_out` (a corrector's or Film node's picture before its mask blend, made only when a Layer reads it) and `join: Option<JoinStep>` (the kind, up to `PLAN_INPUTS` = 8 input refs, the Parallel's base ref and an `is_base` bit set, a Layer's per-input alpha key refs, a Splitter's three outputs). Values are registers, key slots and buffers as before; the compiler works on values (`Pic(node, channel)`, `Graded(node)`, `Key(node)`), so a join reads some of its inputs from buffers when a pass splits and nothing else about the layout changed (a graph without joins gets the same passes, registers and buffers it had). The base of a Parallel is the immediate dominator of its output in the graph of picture values, computed by dominator sets over the topological order and given in `Plan::bases`. `Card::number` is `Option<u32>` (`None` for the four kinds, which do not count in the numbering). `Plan::picture_ancestors` follows every input, the base, the alphas and the graded values; `Plan::has_joins()` says whether a plan has a join. `Stop::Mask` on one of the four is `PlanError::NoMask` (the core maps a mask view of one to the photo, as it does for the Input); a mixer with more than 8 inputs is `PlanError::TooManyInputs`. `StepKind` gains `Parallel`, `Layer`, `Splitter` and `Combiner` (per pixel, read pictures, no planes). The CPU reference is `ym_color::display::mix` (`BlendMode`, `JoinParams`, `parallel`, `layer`, `split`, `combine`) called from `ColorChain::run_nodes`; `ChainParams::joins` carries each mixer's bypass, mode and opacity. `ym-core`: `plan_graph` and the new `plan_joins` give the compiler every node; `node_plan` no longer refuses (`MIXERS_NOT_RENDERED` is gone); `node_set` returns the mixers' `JoinParams` beside the Film steps; `nodes_key` includes a mixer's bypass, mode and opacity; `step_kind` gives the four their own kinds; node thumbnails are made for numbered cards only. **The device refuses**: `NodePlan::new` in `ym-gpu` returns `GpuError::InvalidInput("the colour kernel cannot run a mixer, a splitter or a combiner node yet")` for a plan with a join, the route of an input check (the CPU makes the frame or export, `RenderReport::cpu_reason` carries that sentence); no shader changed. Tests: `ym-color/tests/plan.rs` (29, ten new: the base in every case of section 3.2, nested mixers, one-pass Layer with its graded value and key, a Parallel of eight branches in two passes with buffer reads, dead joins, ancestors through a Layer, a Parallel's base, a Splitter and a Combiner, order-independence, bad joins, and 600 random graphs with all four kinds checked against direct evaluation), `ym-color/tests/mixers.rs` (17: the mode table, negative values, W3C compositing, every identity bit for bit, the mask applied once at a soft edge, a channel swap, a Parallel of eight through buffers), `ym-core/tests/render_nodes.rs` (identities through the core, the grade key), `ym-core/tests/any_graph.rs` (`generate_joined`: random graphs of correctors, Film nodes and the four kinds, run on the CPU on tiles of 97, 160, 256 and 1024 px and 1, 3 and 4 threads with bit-equal results, Show mask on a mixer, and the device refusal with its reason), `ym-cli/tests/parity.rs` (the two example files and a Layer with masked inputs: JPEG, TIFF 16, PNG 16 and DNG, byte-identical through the app and `ym`). Differences from this file: `Card::number` is an `Option`, joins are passed beside the `PlanGraph` and not inside it (so every existing caller compiles unchanged), a Splitter stop (`Stop::Picture`) is its red output, and the spec's example code of Subtract at 0.18 - 0.27 was printed as -0.7213, which is the code of -0.10; the code of -0.09 is -0.6333, now printed in section 3.3. Not done here (X-06): `signal_source` (histogram, scatter and pixel readout `at` a mixer) still answers "This photo has no node k." for the four kinds.
- **Tests.** `src/state/graphOps.mixers.test.ts` (39: the operations, refusals, links, renumbering, rows, base, words), `GlyphCard.test.tsx` (7), `MixerPanel.test.tsx` (11), `AddNodeMenu.test.ts`, `src/fixtures/mixers.test.ts` (8), `e2e/mixers.spec.ts` (17), `e2e/a11y.spec.ts` (the nodes of docs/25), eight glass-audit states (77 to 84) and seven reference screenshots (`photos-mixers-*.png`; `photos-nodes-add-node.png` is retaken because the menu's three items are no longer greyed).

**X-05 · The device (built on branch `mixers`).** The four kinds run in the fused colour kernel; `NodePlan::new` no longer refuses a plan with a join, and the CPU makes a frame of these kinds only for a device error. What was decided:

- **The record.** `NodeRecU` grows from 1200 to 1328 bytes (83 rows): eight `vec4<u32>` rows at its end (`R_JOIN`, `JOIN_ROWS`), not the warp's rows overlaid, so that a corrector's graded value has its own lanes and a join's wiring reads as it is written (`docs/17` section 11a lists the rows: `J_HEAD`, `J_IN`, `J_AL`, `J_BASE`, `J_SPLIT`). A plan of 64 steps is 83 KiB of records. The bindings are unchanged (K3: 8 storage buffers, 1 uniform); `ctl.x` gains `N_JOIN` (a mixer, a Splitter or a Combiner) and uses `N_SKIP` for Bypass. A graph without the four kinds runs the same expressions as before (the kernel's other code is unchanged; the extra rows hold `NODE_NONE`).
- **The kernel** (`color.wgsl`: `join_parallel`, `join_layer`, `blend_mode`, `rd_pic`, `rd_key`; the Splitter and the Combiner inline in `run_nodes`). Every input and alpha is read before anything is written. A corrector or Film node that feeds a Layer's input 2 or above writes its **graded value** (the picture before its mask blend; for a Film node its picture when `m` is not 0, else its input) to the register or buffer the plan names, beside its picture and key. A stop on a join (kernel stop 11) ends at its picture, a Splitter's at its red; a mask stop on a join cannot happen (`PlanError::NoMask`).
- **Precision.** The Layer decodes and encodes with the compensated LogC4 pair of `precise.wgsl` on an export and with the plain functions of the trims in a view (`px_fast()`, F-08e), as the final stage does; this is §4.3's fallback, taken from the start because the device's `exp2` is 45 ulp on Mesa and a Layer subtracts what it decodes. The Parallel's differences and sums are behind the OR barrier (`px_op`) so that branches at their identity give the base's bits under Metal's fast-math as well; the Layer's alpha tests and the LogC4 toes are continuous (K8, `docs/17` section 11a).
- **Measured on lavapipe** (Mesa 25.2.8, LLVM 20.1.2; `crates/ym-gpu/tests/gpu_mixers.rs`, 11 tests, 41 x 29 to 97 x 61 px scenes of `camera_image`, the 17-stop range with negative samples and 0.1 % outliers to 80 times grey; GPU against the CPU reference, C10 in LogC4, C13 in codes of 1; the three storage modes F32, F16Packed and F16Native give the same numbers): **Normal, Add, Multiply, Lighten and Darken** 3e-7 to 1e-6 in C10 and 1.0e-6 to 1.9e-6 in C13 (export), 1.3e-6 to 2.5e-6 (view); **a Parallel of 1, 3 and 8 inputs** (8 is two passes with 9 buffers) 3e-7, 5e-7 and 9e-7 in C10, 2.6e-6 in C13 at most; **a Parallel from different places** and **with the base as an input** 3e-7; **Splitter, three correctors with a mask on one of them, Combiner** 1.9e-7, and a channel swap exact (`assert_bit_equal`); **a Layer of eight over two passes** (graded values and alphas read from the buffers) 6e-7, bright pixels 1.1e-6; **a Layer with masked inputs and a key** 1.3e-5 (bright pixels 1.95e-4, below black 9.1e-5 relative). **Subtract and Difference** cancel two layers, and the last digit of the f32 codes of the nodes before the Layer (1.2e-7 of a code, 1.2e-6 of a linear value near code 1) shows as a few 1e-4 of a code where the layers are bright and the result sits on LogC4's steep linear segment (8.8 codes per linear unit): for pixels of the photo's range (camera value up to 1, 2.5 stops over grey) the error is 2.2e-6 (Subtract) and 2.2e-6 (Difference); for brighter pixels it is 2.5e-4 (Subtract, export; 2.6e-4 in a view) and 1.4e-5 (Difference), and for results far below black (code under -0.5, rendered black) 1.5e-4 of the code (2.7e-4 in a view). The gates the tests hold are 2e-4 (range of a photo), 1e-3 (bright pixels), 5e-4 relative (far below black), 1e-4 relative (code beyond 20) and one 10-bit code in C13 (largest 4.8e-5); **this is looser than §4.3's "2e-4 for everything"** for bright pixels under Subtract and Difference only, and it is the f32 transport of the codes between nodes, not the Layer's own arithmetic. A Layer that cancels two layers brighter than 2.5 stops over grey to within 1e-6 is not a case a photo makes.
- **Identities on the device, bit for bit** (`gpu_mixers.rs`): a Layer at Opacity 0, with one input, bypassed, or over the Input with nothing on top (C10 equals C8); a Parallel of identity branches; a bypassed Parallel; a Splitter, three identity correctors and a Combiner (C10 equals C8). Tiles of 24, 33 and 64 px equal the whole frame (8-bit with dither and f32 planes), a repeat render is identical.
- **Random graphs.** `generate_joined`'s graphs are part of `graphs()` in `tests/any_graph.rs`, so every device assertion holds for the four kinds: the frame is the device's and not refused, within the Film bounds of the CPU's, at Fit and 100 %, the exports (PNG 16 and JPEG) are the device's, within one 10-bit code of the CPU's and the same bytes at tile 97 as at 256, at any binding size of the sweep; `graphs_with_joins_are_exported_on_the_device` exports all eight (worst PNG 61 of 65535 from the CPU's, the gate 64). The test that asserted the refusal is gone.
- **App against `ym` on the device** (`crates/ym-cli/tests/parity.rs`, `mixers_are_the_same_files_through_the_app_path_and_the_binary_on_the_device`): the two examples of §2.4 and a Layer with masked inputs, JPEG, TIFF 16, PNG 16 and DNG, byte-identical with the device on both sides (`YELLOWMOON_GPU=on`).
- **For the Mac GPU run.** The new things Metal could move: the barrier on the Parallel's sum (identity branches must still give the base's bits: `a_parallel_of_identity_branches_and_a_bypassed_one_are_the_base_bit_for_bit`); the Layer's Subtract and Difference numbers above (the "bright" and "below black" lines of `every_mode_of_a_layer_matches_the_cpu`); pointer parameters to the registers and key slots (`ptr<function, array<..>>`) in the MSL; the Layer's two float tests. If the Mac numbers differ from lavapipe's by more than a last digit, look at the compensated decode first.

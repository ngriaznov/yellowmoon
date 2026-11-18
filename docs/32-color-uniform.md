# 32 · Colour Uniform

Colour Uniform is a node that makes colours more alike. The person picks one colour on the photo, the **target**, and the node pulls the colours its masks let through toward it: their hue, their saturation and their lightness, each by an amount from 0 to 100 %. It is for the things a photographer wants to be one colour and that light has made several: skin with a cast on one cheek, a dress or a car in uneven light, a lawn, a patch of sky. This file covers what it does, its controls, the maths, where it sits in the graph, the settings, the pipeline version, the tests, the speed budget, the owner's answers, the mock-ups and the tasks.

**Status: planned (CU-01, 2026-11-18), revised the same day after the owner's answers (CU-H1). Nothing here is built.** The owner's request of 2026-11-18: "plan 'color compressor' node, similar to Resolve color compressor, that will also go into 0.2.4". The owner named it **Colour Uniform** and decided that it has **no range of its own: masks alone, as Resolve does** (§10). This is the plan, as presets (`docs/30`) and mixers (`docs/25`) were planned: a spec, mock-ups (`design/colour-uniform/`) and a task list (milestone U of `docs/15`). The research behind it is `research/color-compressor-2026-11-18.md` (it is about Resolve's tool, so it keeps that name); **this file wins where they differ**. The maths is ours: we copy no code and no formula from any tool (`docs/14`).

**In short.**

- A new node kind, `color_uniform`, in the graph of `docs/23`: a numbered node like a corrector, with masks, a key output, a thumbnail and a number on its card. Add node › **Colour Uniform**, "Pulls colours toward one colour." (§4). It counts toward the 64 nodes. In a sentence it is "a Colour Uniform node".
- **Target** (§2.1): chosen on the photo with Pick colour, or by dragging or typing its Hue, Saturation and Lightness (a rainbow track for Hue), and shown as a swatch. **Compress** (§2.2): Hue, Saturation, Lightness, each 0 to 100 %.
- **Selection** (§2.3): **the node's masks, and nothing else** (`W = m`, `docs/23` §4). **The first pick also adds a Colour range mask row** centred on the picked colour, so the first result is local; this is our decision and is flagged in §10.
- **Maths** (§3): on the LogC4 / AWG4 values the node reads, in the plane and the mean that the colour warp and the Colour range mask already use. Each axis moves `x → t + (x − t)(1 − κ)`, with a neutral guard on the hue and saturation pulls and a fade on the far side of the wheel for hue. Exact identity at 0 %, the target a fixed point, continuous across the hue wrap, no NaN on grey or black, and monotone along each axis; the whole map is proved to have a positive Jacobian below 100 % (§3.8).
- **A new node changes no pixel** until it has a target and an amount above 0 (§2.5).
- **Rendering** (§6): one per-pixel step, no new binding, CPU in f64 and device in f32 within the grade's 2e-4. `PIPELINE_VERSION` takes the next minor step; no existing graph changes by a bit (§7).

## 1. What it does

A photographer looks at a face where one side has a green cast, or a red dress that is orange in the shade. A corrector can lift or lower a colour for the whole photo, or for a mask; the colour warp can move a colour, with its neighbours following. Neither makes the spread of colours **smaller**. Colour Uniform does:

1. Click a colour that is right (a good patch of skin). The node adds a Colour range mask on that colour, so it acts on the colours near it.
2. The node pulls each colour its masks select toward the picked colour. At 50 % a colour is halfway between where it was and the picked colour; at 100 % it is the picked colour on that axis. Hue, saturation and lightness are set apart, so the person can even out the cast (hue and saturation) and leave the light and shade (lightness) alone, which is what keeps a face round.

It is the tool Resolve calls Color Compressor and Capture One's Skin Tone *Uniformity*. Resolve's tool has no selection of its own: it relies on a qualifier or a window before it (`research/color-compressor-2026-11-18.md` §1). So does ours: the selection is the node's masks, which every corrector already has (`docs/23` §3.3). The one difference is the first pick, which starts the node with a Colour range row (§2.3).

### 1.1 How it differs from the colour warp, and why both exist

Both read the same chroma plane and both have Pick colour. They do opposite things.

| | Colour warp (`docs/02` C9b) | Colour Uniform |
|---|---|---|
| What a pin or target does | Moves a colour to somewhere else; the colours near it follow by the same move | Pulls the colours the masks select **toward** it, each by a share of its own distance |
| Effect on the spread of colours | Keeps it (a translation, folded never: it is a flow) | Shrinks it: two colours end nearer each other than they began |
| Axes | The chroma plane, with an exposure and a saturation per pin | Hue, saturation and lightness set apart, each with its own amount |
| Selection | The pin's Gaussian range, tonal weights for shadows and highlights | The node's masks |
| Use | "Make these reds more orange", a look | "Make this skin one colour", a correction |
| Lives | A card in a corrector's Colour tab | Its own node |

A warp cannot do what Colour Uniform does: moving every colour in a range by the same amount keeps the cast, and shrinking the spread would take a pin for each colour. Colour Uniform cannot do what the warp does: it has no destination but one colour. Neither replaces the other, so both stay. Colour Uniform reuses the warp's plane, its neutral guard and its Pick colour, so a colour picked for one reads the same in the other (§3.1).

### 1.2 What is not in this version

- **A range of its own.** The owner's answer (§10, 3): masks alone. A colour range, a brightness range or any other mask goes in the node's Mask tab.
- **A separate shift of the target.** Dragging or typing the target's Hue, Saturation and Lightness does it (§2.1).
- **A target that is a colour range, or several targets.** One target per node; add a node for another colour.
- **Pick by dragging a rectangle** (the mean of a region). Pick takes the mean of a 7 × 7 square at the click (§3.11). A rectangle is a later idea.
- **A Blend or Strength slider.** The node's masks (with their Opacity) and the three amounts do the job.
- **A shipped Skin preset.** A click on skin is the starting point; an example for `ym` users goes into `docs/10`.
- **Compressing toward a grey.** Neutrals are kept (§3.7) and the target is held off grey: a tool that makes colours alike does not colour or uncolour a neutral. Saturation does that.
- **Auto.** No Auto, as for every node but Main (`docs/23` §3.2).
- **Skin detection.** The node does not know what skin is; the person's click does. Face Refinement (`docs/27`) is the node that knows faces.

## 2. Controls

All copy is British, short and plain. Controls are glasscn's (`docs/16`); a control glasscn has is never hand-written. The mock-ups are `design/colour-uniform/` (§12).

### 2.1 Target

A card named **Target**, first in the node's **Compress** tab.

| Control | Range | Default | Notes |
|---|---|---|---|
| **Swatch** | | empty (a checked square) | The target colour as a patch, as the picture shows it, with the colour's name beside it ("Orange", "Orange, a little pale"). Not a Tab stop; its name is read as text. **It opens no picker:** `src/components/glass/` has **no colour picker component** (checked 2026-11-18: accordion, aurora, badge, button, button-group, card, checkbox, collapsible, context-menu, dialog, dropdown-menu, glass, glass-group, input, item, kbd, label, popover, progress, radio-group, scroll-area, scroll-edge, search-field, segmented-control, select, separator, slider, spinner, switch, tabs, textarea, toggle, toggle-group, toolbar, tooltip), and we do not design one. **The three sliders are the picker** |
| **Pick colour** | a toggle with an eyedropper | **on when the node is added**, off after the first pick | Like the warp's and the Colour range's. A click on the photo sets the target to the colour there. It stays on for the next click and **Esc** ends it. Picking the target is the heart of the node |
| **Hue** | 0 to 359° (wraps) | – | A number field with the slider's look (`Slider` with its value box), **the track drawn as a rainbow** (a hue ramp). Dragging it or typing in it is the other way to choose the target colour |
| **Saturation** | 2 to 50 | – | The chroma of the colour times 100, the number the warp's range uses (a skin tone is about 5 to 12; a strong red 13) |
| **Lightness** | −4 to +8 EV | – | Stops from middle grey (code 0.2784), `+0.5 EV` |
| **Reset** (card header) | | | Clears the target (back to none). The amounts and the masks stay |

- **The three fields are on with no target.** They rest at Hue 0°, Saturation 2 and Lightness 0 and read "–"; the first change to any of them sets a target from the three resting values (so dragging Hue alone gives a colour). A target made this way adds **no mask row** (§2.3): the person asked for no click.
- **What a pick reads.** The mean colour of a 7 × 7 pixel square at the click, in the picture the node **reads** (its input, before the node), as a Colour range's pick does (`docs/11` *Pick colour of a Colour range*, §3.11). A click on the letterbox does nothing.
- **A pick too grey to have a hue.** Below Saturation 2 the target is held at 2 on the direction of the pick (a pick of exactly neutral takes hue 0) and the card says "That colour is too grey for a hue. It was moved to Saturation 2." once (a notice line under the swatch, gone at the next change), as a clamp is reported elsewhere.
- **The first pick** (a pick made while the node has no target) also sets Compress to Hue 50 %, Saturation 50 % and Lightness 0 % when all three amounts are 0, and adds the Colour range row of §2.3 when the node has no mask. All in the same patch: **one undo step**. Later picks change the target only.
- **Keyboard.** The pick is a pointer action, so the three number fields are the keyboard way to a target (`docs/28`). The swatch's text is the colour's name and its three numbers: "Target colour: orange, hue 26 degrees, saturation 9.7, lightness minus 0.03 stops."

### 2.2 Compress

A card named **Compress**, second.

| Slider | Range | Default | Step | Reads as |
|---|---|---|---|---|
| **Hue** | 0 to 100 % | 0 | 1 | "50%" |
| **Saturation** | 0 to 100 % | 0 | 1 | "50%" |
| **Lightness** | 0 to 100 % | 0 | 1 | "0%" |

- 0 % moves nothing. At 100 % every colour the masks let through fully has the target's value on that axis, with the two exceptions of §3.2 (neutrals are kept, and a hue far on the other side of the wheel is not dragged across it). The line under the sliders: "At 100 % every colour in the mask takes the target's value." (in the card's quiet text).
- The sliders are `aria-disabled` while the node has no target, with the line "Pick or set a target colour first." in their place.
- **Reset** (card header) puts the three at 0.
- Hue and Saturation are the axes that even out a cast; Lightness flattens shading, so it is a separate amount and the first pick leaves it at 0.

### 2.3 Selection: masks alone, and the first pick's Colour range row

The node has **no Range card and no range fields**. What it acts on is its masks (the Mask tab, the key input, Invert; `docs/23` §3.3, §4.2): the node's mask `m` is the selection `W`, as for any corrector, and the node's result is blended as `out = in + m (node(in) − in)` (`docs/algorithms/06` §1.1). A node with no mask acts on the **whole photo**.

**Decided by us, flagged in §10: the first pick adds a Colour range row.** After a pick of skin, Hue 50 % with no mask would pull every non-grey colour in the photo halfway to orange: a bad first result. So:

- The **first pick** (a pick made while the node has no target) **adds one `color_range` row to the node's masks**, when the node has **no mask rows and no key link** (`docs/23` §3.4). The row has `center` = the picked colour's chroma `[a, b]` in the C9 plane (as read, before the target is held off grey), and the Colour range's own defaults for the rest: `width` 0.05 (the Width slider reads 39) and `softness` 0.5, `mode` add, `opacity` 1, enabled (`docs/23` §3.3, §4.1 and §7.6; maths in `docs/algorithms/06` §4.3).
- It happens on the **first** pick only. A later pick moves the target and leaves the masks alone. A target set by the sliders (no pick) adds no row.
- **The pick and the row are one undo step.** The Colour range row is an ordinary mask: the person can widen it (Width), add other masks, delete the row, or invert; the node then acts on whatever the masks say. The row's own Pick colour moves its centre and does not move the target.
- The row is not selected and starts no tool; the panel stays on the Compress tab, where the first pick was made.
- The skin seed is the default because the Colour range reads chroma only: a disc of radius 0.05, with an edge of 0.025, about the picked chroma. For a skin pick (chroma 0.097) it reaches from about 0.02 to 0.17, so near-neutral colours (chroma below 0.02) are outside it, which is also why the near-neutral gain of §3.7 does not show in the first result.

The alternative (no automatic mask; a hint line in the empty Mask tab) is in §10.

### 2.4 Show mask

The toolbar's **Show mask** (M) and the Mask tab's View card (`docs/23` §5.8, §7.6) are **the ordinary Show mask of the node's mask `m`**: Overlay tints it in the mask accent at 0.55, Mask only shows it as grey. The node has no second view and no eye of its own. Neither style reads the amounts: the mask is shown at 0 % too, so it can be tuned before the pull is raised. A node with no mask shows nothing tinted (`m = 1`, the whole photo). The mask view is a view: not in an export, the Before view, a thumbnail or the pixel readout (`docs/23` §5.8).

### 2.5 Defaults and the identity

- A new node is `target: null`, Compress 0, 0, 0, no masks, not bypassed. **It changes no pixel.**
- With **no target** the node is the identity whatever the amounts are. With a target and **all three amounts 0** it is the identity too. Either is skipped: it runs no code and changes no bit (§3.5).
- So "a colour is picked or an amount is raised" reads: picking sets the target **and** raises Hue and Saturation to 50 % (§2.1), and raising an amount in the panel needs a target. A document or `ym --settings` file can hold an amount with no target; it does nothing.

### 2.6 The panel

| Part | Content |
|---|---|
| Head | The corrector's compact head (`docs/23` §7.5): the number chip, the name ("Uniform"), the line, Bypass, Reset, More. Line: "Colour Uniform", "Bypassed", "Not connected" (short, so the line is not cut at 400 px; the corrector's mask text is not repeated) |
| Tabs | **Compress · Mask**, in the Colour tab's green (`--tab-colour`; `docs/11`) and the mask accent. The Mask tab is a corrector's (`docs/23` §7.6) |
| Compress › cards | **Target · Compress** (about 250 + 190 px: both fit the 678 px scroll area at 1440 × 900) |
| Mask › cards | the Masks card, the selected mask's card, the View card, as a corrector's; after the first pick: one **Colour range** row (Width 39) and its **COLOUR RANGE** card (Pick colour, Width, Softness, Opacity) |
| Histogram | none (the node has no light to show) |
| Keys | ⌘1 and ⌘2 select the tabs. No new single key: Pick colour is the toggle, and it is on when the node is added |

**Reset this node** (the head) puts target and Compress back to their defaults; masks stay (`docs/23` §7.5). The More menu is a corrector's.

## 3. The maths

The CPU reference is `ym_color::uniform` (CU-02), in f64. The kernel (CU-05) is the same expressions in f32 (§3.10). A sketch of it in Python (`design/colour-uniform/html/compress_ref.py`) was used to try every statement below and to draw the mock-ups; it is not product code and the property checks of §3.8 run on it.

### 3.1 The space, and why

The node reads a LogC4 / AWG4 triple `v`, like every node (`docs/23` §1.2), and works in the **orthonormal basis of LogC4 code space** that the colour warp uses:

```
e1 = (1, −½, −½)/√1.5      red against cyan        (docs/02 C9)
e2 = (0, 1, −1)/√2          green against blue
e3 = (1, 1, 1)/√3           brightness along the neutral axis

c = (e1·v, e2·v)      the chroma, a point of the warp's plane;   r = |c|
m = (v_R + v_G + v_B)/3   the mean of the three codes: the tone
```

`v = m·(1,1,1) + c₁ e1 + c₂ e2` exactly. The three axes are:

- **Hue** is the angle of `c`, counter-clockwise from `+e1` (red 0°, yellow 60°, green 120°, cyan 180°, blue 240°, magenta 300°), the angles of `docs/02` C9b and the warp's accessible names.
- **Saturation** is `r`, the chroma, in the plane's own units (`WARP_PLANE_EXTENT` 0.30). The UI shows `100 r`.
- **Lightness** is `m`, shown as stops from middle grey: `ev = (m − 0.2784)/S` with `S = LOGC4_STOP` = 0.0648 (`ym_color::consts`); middle grey has `m = 0.2784` and `ev = 0`.

**Why this space.**

1. **It is the plane everything else reads.** The colour warp's pins, the Colour range mask's centre, the chroma scatter and Pick colour (`sample_chroma`) all use `c`. A colour picked for the warp or the range reads the same here, and the seeded Colour range row of §2.3 is centred on the same `c`. No new colour space or conversion is added.
2. **Hue and saturation are nearly independent of exposure.** In LogC4 a change of exposure adds nearly the same amount to the three channels, so it moves `m` and leaves `c` almost alone. That is why a ramp across a face's shading keeps its hue and saturation, so the node can even out hue and saturation without touching the light.
3. **The moves are exact and need no transfer function.** The basis is orthonormal, so a move of `c` and a move of `m` are independent, and the node needs no decode and no encode: no `exp2` or `log2` on the device, which is what keeps it cheap and its parity tight (§3.10). The tone is the mean of the three codes rather than the luma code **on purpose**: a move of `m` is an exact addition to all three channels, so the law of §3.3 is exact on `m` and the lightness axis can be proved monotone. For the colours of one hue the two differ by a nearly constant offset, so a person sees no difference; the swatch and the Lightness field use `m`, so they agree with what the node does.
4. **Not HSL or HSV.** Resolve's tool speaks of H, S and L of an unstated space. HSL of a display colour is not exposure-independent and its hue is badly behaved in a log space (the Frame.io note, `research/color-compressor-2026-11-18.md` §1). Not OKLab: it needs a transfer function and a matrix per pixel and gives no better hue on scene values than the plane the product already has.

**Where in the graph.** The node runs on the picture it reads, in LogC4 / AWG4, between its neighbours, like any node. Its mask `m` is a function of the node's input pixel and the other nodes' keys, as every node's is (`docs/algorithms/06` §1), and the result is `in + m (node(in) − in)`.

### 3.2 The pulls and their two guards

The stored settings are converted once (the core, f64): the target `t = (c_t, m_t)` with `r_t = Saturation/100`, `c_t = r_t (cos h, sin h)`, `θ_t = h`, `m_t = 0.2784 + ev·S`, and the amounts `κ_h`, `κ_s`, `κ_l` in [0, 1] (Compress ÷ 100). **There is no selection weight in the node's maths:** the node's mask is the selection, and the node computes the pull for every pixel it is given.

For a pixel with chroma `c`, `r = |c|` and tone `m`:

```
S(x)   = u²(3 − 2u),  u = clamp(x, 0, 1)                         the smoothstep of docs/algorithms/06

δ      = atan2(c_t × c, c_t · c)   in (−π, π]: the signed angle from the target's hue to the pixel's,
                                    taken the short way round (the cross and dot products; 0 when r = 0)
g(r)   = S(r / 0.02)               the neutral guard of the warp (WARP_NEUTRAL_GUARD)
a(δ)   = 1 − S((|δ| − π/2) / (π/3))   the antipode fade: 1 up to 90° from the target's hue, 0 from 150° on
```

- **The neutral guard `g`** is 0 at `r = 0` and 1 from `r = 0.02`. It multiplies the hue and the saturation pulls, so a grey, or a black, never gets a hue or a tint, and a colour with a faint cast is pulled by a share that grows with its chroma (§3.7). The lightness pull has no guard: it acts on everything in the mask.
- **The antipode fade `a`** multiplies the hue pull only. It is a constant of the node, not a setting. Without it the hue pull would be **discontinuous at the colour opposite the target**: a pixel at 179° and one at 181° would be sent to opposite sides, 358 (1 − κ)° apart. No continuous map of the hue circle can go from the identity at 0 % to "every hue is the target's" at 100 % (the identity has winding number 1, a constant has 0), so something has to give: here it is the colours far from the target's hue, which the pull leaves alone, and the pull on them fades in smoothly (§3.6).
- Both guards are smoothsteps with zero slope at their ends. No result depends on a rounding: every `if` in the kernel chooses between results that agree to within rounding (§3.10).

### 3.3 The compression law

```
hue         δ'  = δ · (1 − κ_h · g(r) · a(δ))      the angle to the target's hue shrinks (not for neutrals, not on the far side)
saturation  r'  = r_t + (r − r_t) · (1 − κ_s · g(r))   the distance to the target's chroma shrinks (not for neutrals)
lightness   m'  = m_t + (m − m_t) · (1 − κ_l)      the distance to the target's tone shrinks, for every pixel

c'  = r' · (cos(θ_t + δ'), sin(θ_t + δ'))              (c' = 0 when r = 0)
v'  = v + (c'₁ − c₁) e1 + (c'₂ − c₂) e2 + (m' − m)·(1,1,1)
```

Each line has the form `x → t + (x − t)(1 − κ·w)` with a weight `w` in [0, 1] that is 1 for a colour of ordinary chroma and ordinary hue distance: the colour keeps the share `1 − κ w` of its distance to the target on that axis. So:

- **0 %** (`κ = 0`) is the identity on that axis. **100 %** puts a colour on the target's value on that axis, as the Resolve manual defines its 1.000 ("identical to the Target Color"), for every colour with `r ≥ 0.02` (and, for hue, within 90° of the target's hue); **50 %** puts it halfway.
- **The colour of the target is a fixed point**: at `δ = 0`, `r = r_t`, `m = m_t` every line returns its input, whatever `κ` (the target has `r_t ≥ 0.02`, so `g = 1`).
- **Hue is pulled by angle**, not by the straight line between two colours. A straight pull in the plane would also pull the saturation toward zero on the way, and the axes would not be independent. Because `δ` is the short way round, a pixel 30° from the target is pulled 30° · `(1 − κ_h)`, never the long way.
- The three axes use **the original coordinates** of the pixel, so they do not depend on one another's results and their order does not matter.
- **Saturation first, then rotation.** `r'` is the chroma the pixel ends with; the rotation by `δ' − δ` is applied to the direction. A pixel's hue changes without its chroma changing (when `κ_s = 0`, `r' = r`).

A **worked example** (computed with the sketch in f64). Target: Hue 26°, Saturation 9.7, Lightness −0.03 EV. A pixel at Hue 46°, Saturation 12.0, Lightness +0.5 EV is 20° from the target's hue, with `g = 1` and `a = 1`: at 50 % on all three it becomes Hue 36°, Saturation 10.9, Lightness +0.22 EV; at 100 % it becomes Hue 26°, Saturation 9.7, Lightness −0.03 EV, the target. The tests (§8) hold these to 1e-9 with the target's exact stored values.

### 3.4 Skipping

- **No target, or all three amounts 0:** the node is skipped, and `v' = v` exactly (§3.5).
- **An amount of 0 skips its axis.** With `κ_h = 0` the direction of `c` is kept and only its length changes: `c' = c · r'/r` (and `c' = c` when `r = 0`), with no trig, so a Saturation-only pull moves the length and nothing else. With `κ_s = 0, κ_h > 0` `r' = r`. With `κ_l = 0`, `m' = m` and the `(m' − m)` term is not added.
- **A pixel with `r = 0`** gets `c' = 0` and no `atan2`: the kernel evaluates `atan2` only where `r > 0` (a `select`).

### 3.5 The identity, bit for bit

A node with no target, with all amounts 0, with its mask 0 at a pixel, or bypassed gives its input's bits. The plan skips the first two (a step at its identity is skipped, `docs/23` §1.3); the mask gives the third: `out = in + m (node(in) − in)` with `m = 0` is the input exactly, and with `m = 1` the node's output exactly (`docs/algorithms/06` §1.1; a first row is taken exactly). A grey with Hue and Saturation on and Lightness 0 moves by rounding only (the sketch: 1e-16 in f64), because `g(0) = 0` makes its chroma pull exactly zero and its `c` is zero to rounding.

### 3.6 Hue wrap-around

- `δ` is the signed angle **the short way round**, `atan2(cross, dot)` of the target's and the pixel's chroma vectors, in (−π, π]. The 0°/360° seam does not exist for it: a target at 355° and a pixel at 5° are 10° apart.
- `a(δ)` is exactly 0 for `|δ| ≥ 150°`. Across the jump of `atan2` at `δ = ±π` (the colour exactly opposite the target) `a = 0`, so `δ' = δ`, and both sides give the pixel's own hue: **the map is continuous on the whole plane**, with no seam at the target's opposite or at 0°/360°. Measured on the sketch: a sweep of the whole hue circle in steps of 0.01° for targets at 0°, 90°, 180° and 359.99°, with amounts up to 100 %, has no step above 3.7e-5 of a code value (the gate of §8 is 5e-3).
- The rotation uses `cos` and `sin` of `θ_t + δ'`, which are periodic, so no angle is ever wrapped by hand and none is compared.
- **The price of the continuity** is that at 100 % a colour more than 90° from the target's hue is only partly pulled, and one beyond 150° not at all. For a skin target that is the blues and cyans: they keep their hue, which is what a person wants. (`HUE_FULL` = π/2 and `HUE_FADE` = π/3 are constants of the core, not settings; §11.)

### 3.7 Neutrals and low chroma

Hue is undefined at grey and unstable near it, so **the hue and saturation pulls are multiplied by `g(r) = S(r/0.02)`**, 0 at `r = 0` and 1 from `r = 0.02` (the warp's neutral guard, so the two agree on what grey is).

- At `r = 0` the chroma stays 0: a grey, or a black, gets no hue and no tint, whatever the target. The lightness pull still moves its tone (it acts on everything in the mask) and it stays grey. `δ` is then not used, so no NaN reaches the result (§3.10).
- Near grey `g` is small, so a pixel with a faint cast moves by a small share, and its displacement is at most `κ g (r_t − r)` in chroma and at most `2r` by rotation: the map is continuous at the origin. **How much tint a near-neutral gains** (measured on the sketch, Hue and Saturation at 100 %, target Saturation 9.7): a pixel of chroma 0.001 gains at most 0.0007, one of 0.005 gains 0.014, one of 0.01 gains 0.044 and one of 0.02 gains 0.078 (Saturation 7.8 on a pixel of 2). That is the honest cost of pulling saturation toward a target without a range: the guard protects exact neutrals, not faint colours. **The seeded Colour range row keeps them out of the first result** (§2.3); a node with no mask, or a mask that includes faint colours, does pull them.
- **The target is held at Saturation 2 or more** (`r_t ≥ 0.02`, §2.1, §4). This is what keeps the saturation axis monotone near grey (§3.8): `g` rises only below `0.02`, and the target is above it.

### 3.8 Monotonicity: no colour order is inverted

**What is guaranteed (proved):** the node, with its mask at 1, is **continuous** and, along each axis with the other two coordinates fixed, **non-decreasing**, and **strictly increasing for any amount below 100 %**. So an ordered ramp of colours (a gradient across a face, a graduated sky) stays an ordered ramp: no inversion, no band where two different colours swap places, and no posterisation below 100 %. At 100 % colours are given the target's value on the axis: that is the point of the tool (a plateau in the axis: the colours are made one), and the map stays non-decreasing.

*Proof.* On the hue axis at fixed `r` and `m`, `δ' = δ(1 − κ G a(δ))` with `G = g(r)` constant and `a` non-increasing in `|δ|`. For `δ > 0`: `dδ'/dδ = 1 − κ G a − κ G δ a'(δ) ≥ 1 − κ G a ≥ 1 − κ ≥ 0`, because `a' ≤ 0` there; `δ < 0` is the mirror. On the saturation axis, for `r ≥ 0.02`, `g = 1` and `r' = (1 − κ) r + κ r_t` has slope `1 − κ`; for `r < 0.02 ≤ r_t`, `r − r_t < 0` and `g' ≥ 0`, so `dr'/dr = 1 − κ g − κ (r − r_t) g' ≥ 1 − κ g ≥ 1 − κ`. On the lightness axis `m'` has slope `1 − κ_l`. ∎

**The whole map is also a diffeomorphism below 100 %, by a triangular Jacobian.** In the coordinates `(r, δ, m)` the map is triangular: `r'` depends on `r` only, `m'` on `m` only, and `δ'` on `δ` and `r`, never the other way round. So its Jacobian determinant in `(c₁, c₂, m)` is the product of the diagonal terms and the polar factor, `(r'/r) · (dr'/dr) · (∂δ'/∂δ) · (1 − κ_l)`, and each is at least its `1 − κ` (`r'/r = 1 − κ_s g + κ_s g r_t/r ≥ 1 − κ_s`). **The determinant is at least `(1 − κ_s)² (1 − κ_h)(1 − κ_l)`, positive for amounts below 100 %**: the node never folds the colour space. (This replaces the measured statement of the plan's first version, which had a range window that depended on all three axes.) Measured on the sketch (200,000 random points of the cube and of the neighbourhood of grey, central differences of step 1e-6, all three amounts equal): the smallest determinant is 0.075 at 50 %, 2.8e-4 at 90 % (the bound is 1e-4), 2.1e-7 at 99 %, and −2.8e-8 at 100 %, which is rounding noise around the exact 0 of a collapse. CU-02 turns this into a test (§8).

**With masks.** A spatial mask (Linear, Radial, Brush, Object, …) mixes the node's picture with its input pixel by pixel, so colours at different places have different amounts and no colour order is defined across them: the guarantee is for the node's own law. A Colour range or Brightness range row makes the mix depend on colour, with a smooth edge no narrower than `RANGE_RAMP_MIN`; the result is continuous and the guarantee is the row's.

### 3.9 Gamut

- **Nothing is clipped.** Like a corrector, the node does not clamp: a result may leave LogC4's usual range, and the Output's cine rendering (C12) decides what a colour outside the display's gamut looks like (`docs/02`, `docs/25` §3.3).
- **The pull stays between the pixel and the target.** The saturation axis is a convex combination of `r` and `r_t`; `r' ≤ max(r, r_t)`, and the hue rotation does not change the length. The lightness axis is a convex combination of `m` and `m_t`. So the node never takes a colour further out than the farther of the pixel and the target on any axis.
- **Where it can leave the gamut.** Moving a dark colour toward a more saturated target can give a channel below the LogC4 code of zero (a negative linear value), as any corrector's gain can. Such values are finite and are rendered like any other. The tests assert finiteness, not a range (§8).
- No gamut mapping step is added; a colour that the target would carry out of the display's gamut is the person's choice of target.

### 3.10 f64 on the CPU, f32 on the device, and the parity bound

- **CPU reference** (`ym_color::uniform`): f64, `f64::atan2`, `f64::cos` and `f64::sin`, no `fma`, the order of §3.2 and §3.3. Rows are split across threads and the result is the same for any number of threads (`docs/17` D5). Across platforms a last-digit difference of the three trig functions is below the golden rule's ΔE00 0.001 (`docs/13` *The x86_64 slice*: 21 identical, 4 within the rule).
- **Device**: the same expressions in f32 in the colour kernel's plan loop (`color.wgsl`), with `atan2`, `cos` and `sin` of WGSL. It needs no `exp2`, `log2` or `pow`. The arithmetic is additions, multiplications, two dot products, one square root, two smoothsteps, one `atan2` and one `sincos`.
- **NaN safety.** `atan2(0, 0)` is not defined in WGSL: the kernel computes it only where `r > 0` and takes 0 elsewhere. A non-finite code value in the input (never made by the pipeline) would give a non-finite result and is not guarded.
- **Parity bound.** The device is held to the CPU reference at **2e-4 in LogC4** (the grade's, `docs/13`, `docs/17` C10) and **one 10-bit code after C12 and C13**. The expected error is a few 1e-6; CU-05 measures it and writes the number into `docs/STATUS.md`, as X-05 did, and the gate stays 2e-4. The steepest part is the neutral guard: `g` has a slope of at most 75 (`1.5/0.02`), so an input that differs by 1e-7 moves `g` by 8e-6 and a pixel by less than 3e-6.
- **Metal (K8).** The amounts, the target and the identity are constants the host writes; the float tests that remain (`r > 0`, the smoothsteps' clamps) choose between results that agree within rounding, each function being continuous across its test, so fast-math re-association moves a result by rounding only. The Mac GPU CI (`.github/workflows/mac-gpu.yml`) runs the parity cases on Metal.
- **Determinism.** The node is per pixel, so tiles of any size equal the whole frame bit for bit and a repeat render is identical (hard rule 8). The app and `ym` write byte-identical files (hard rule 6).

### 3.11 The pick

`sample_chroma(x, y, at, tap)` (`docs/01`, the IPC table) already returns the chroma `{ a, b }` of the pixel in the plane of `e1`, `e2`. For this node it gains two things, **both additive** (a caller that does not read them is unchanged):

- the answer carries **`tone`**, the mean `m` of the three codes of the same pixel, so the page can set a target without converting a colour;
- an optional **`radius_px`** (default 0, so every existing caller reads one pixel): the colour is the **mean of the LogC4 triples** in the square of `2·radius_px + 1` pixels about the point, in the picture the tap reads, then reduced to `c` and `m`. The node's pick asks for `radius_px: 3` (a 7 × 7 square): a single pixel of skin is full of noise and a target taken from one is a noisy target. The square is in output pixels at the viewport's current scale.

The tap is `"node_input"`, the picture the node reads (as a Colour range's, `docs/11`): the target is a colour of the picture the node acts on, so the picked colour is a fixed point of the node at once (§3.3). The page then converts to the stored form (Hue in degrees from `atan2(b, a)`, Saturation `100 √(a² + b²)`, Lightness `(tone − 0.2784)/S`), the one place it converts, with the formula of `ym_color::uniform` ported to `uniformMath.ts` and checked against the Rust constants by a node test, as `warpMath.ts` is. **The same answer centres the seeded Colour range row** (§2.3): `center = [a, b]`, so the row and the target agree on what was clicked.

## 4. Where it sits

**A corrector-like numbered node** (the Film node's precedent, `docs/24` §4.5; the mixers' kinds, `docs/25` §2). `kind` is `color_uniform`.

| Property | Value |
|---|---|
| Ports | in `picture`, `key`; out `picture`, `key` (as a corrector's, `docs/23` §3.4) |
| Number, thumbnail | numbered in the plan's order; a card with a thumbnail and a number chip; the node view and the thumbnails path make one thumbnail for it (`docs/23` §5.7) |
| Masks | `masks` as a corrector's (`docs/23` §3.3): the Mask tab, Invert, the key input, Show mask. They are the node's whole selection (§2.3) |
| Bypass | a bypassed node passes its picture through; its key output still carries `m` (as a corrector's) |
| Add node | **Colour Uniform**, after Face Refinement and before the separator: "Pulls colours toward one colour." **Adds one node after the selected one, like Film** (`docs/23` §7.4), selects it, opens its Compress tab and turns Pick colour on. Off at 64 nodes with "A photo can have up to 64 nodes." |
| Auto | none |
| More menu | a corrector's: Rename · Move earlier · Move later · Link picture to ▸ · Link key to ▸ · Remove link ▸ · Delete node. Delete links what it fed from its source, as a corrector's |
| Card | a corrector's card (112 × 84 px at 100 %: thumbnail, number chip, name, footer glyphs); the node's glyph, a pair of arrows pointing at a dot, sits in the footer beside the mask glyph (the Film node's and Face Refinement's way) |
| Simple view | not shown there. The notice counts it with the others ("This photo has 3 nodes.", `docs/23` §6), and **Open nodes** goes to the node view |
| Name | the field `name`, default `"Uniform"` (seven letters, so it is not cut on the card), 1 to 32 characters as a corrector's. The menu and the panel's head line say "Colour Uniform" |
| Sentence name | "a Colour Uniform node": "Node 4 is a Colour Uniform node, not a corrector." (`NodeKind::with_article`, `sentence`) |

The node does not belong to Main: it is a separate node after Main (or anywhere in the graph), so Main's Auto, the simple view's controls and Re-run never touch it.

## 5. Settings

### 5.1 The format

A new value of `kind` (`docs/23` §3.2): `color_uniform`. The node has these fields and no others; another field is an unknown field and is refused.

| Field | Type / range | Default | Notes |
|---|---|---|---|
| `name`, `bypass`, `pos`, `masks` | as a corrector's | `"Uniform"`, false, – , none | |
| `target` | `null`, or `{ hue_deg, saturation, lightness_ev }` | `null` | `hue_deg` any finite number, read modulo 360 into [0, 360) (**not clamped, not reported**); `saturation` 2 to 50 (clamped, reported; a value below 2 is held at 2); `lightness_ev` −4 to +8 (clamped, reported) |
| `compress` | `{ hue, saturation, lightness }` | `{ 0, 0, 0 }` | each 0 to 100 (percent). A partial object is read over the defaults |

There is no `range`. A file that has one is refused as an unknown field (§5.2).

```json
{ "id": 4, "kind": "color_uniform", "name": "Uniform", "bypass": false, "pos": [2, 0],
  "target": { "hue_deg": 26, "saturation": 9.7, "lightness_ev": -0.03 },
  "compress": { "hue": 60, "saturation": 60, "lightness": 0 },
  "masks": { "invert": false, "items": [
    { "kind": "color_range", "mode": "add", "invert": false, "enabled": true, "opacity": 1,
      "center": [0.0875, 0.0423], "width": 0.05, "softness": 0.5 } ] } }
```

The numbers are in the units the panel shows, so a file written by hand reads like the panel (the core converts to the plane's units once, §3.2). The mask row is the one the first pick adds (§2.3). A graph that has the node, as the core writes it, is `docs/23` §3.1's with one more item and two more links.

### 5.2 Validation and clamping

- A value out of range is **clamped and reported** by its path (`4.target.saturation`, `4.compress.lightness`), as `docs/23` §3.5; a value that is **not finite** is `SettingsError::NotFinite` and nothing is applied. A `target` that is not `null` or an object, or has a field of another type, is `SettingsError::Parse`.
- The fields `kind`, `target` and `compress` follow the unknown-field rule: **another field is refused** ("A Colour Uniform node has no field called “range”.").
- The rules of `docs/23` §3.5 hold: ports exist on their kind, an input port has one link, no cycles, ids are unique, **at most 64 numbered nodes** counting this one ("A photo can have up to 64 nodes."), the kind is known. The unknown-kind example of `docs/10` and the tests stays “outside” (`docs/25`).
- A `target` with `saturation` below 2 reads as 2 with the clamp reported, so the target is always held off grey (§3.7).

### 5.3 Patches, Reset, Copy and Paste, undo

- **Patches** (`docs/23` §3.7): a node is added by giving its `kind` in `nodes` and removed with `remove`; its links go with it. `SettingsField` gains `target` and `compress`, in the **nodes' group** (a change runs the colour pass again and nothing else; names and `pos` re-run nothing). A patch carries a partial `compress` over what the node has; `"target": null` clears. **The first pick of §2.1 and §2.3 is one patch** (`target`, `compress` and `masks.items` with the one row), one undo step. A drag on a slider is a patch per frame that folds into one undo step, as every slider.
- **Reset** (`SettingsPatch::reset_node_in`): target to `null`, `compress` to its defaults; name, bypass, masks and links stay.
- **Copy settings and Paste** carry the node like any node, in the whole graph (`docs/23` §3.8). Undo diffs per node.
- **Resolved settings, XMP and DNG.** The node has no Auto, so it is in `ResolvedSettings.settings` as it is. `ym:Settings` holds it and the DNG bakes it into the pixels like every node (`docs/09`).
- **The JSON Schema** (`crates/ym-core/schema/settings.schema.json`) gains `color_uniform_node` (with `target_value` and `uniform_amounts` definitions) and the kind in the `kind` enum. The wire types gain `NodeKind::ColorUniform` and `NodeItem::ColorUniformNode`; `bindings.ts` is regenerated and the field-name test compares them.

### 5.4 Presets and albums

- **Presets** (`docs/30` §2). A preset's **`nodes` group** holds every node other than the Input, Main and the Output, so Colour Uniform nodes are in it with no change to the groups: a preset saved from a photo that has the node holds the node, its target, amounts and masks (the seeded Colour range row among them), and applying it adds or replaces them like the other nodes of the group. The preset's table text of §2 gains "Colour Uniform nodes" in the list of kinds (CU-07). **A target is a colour of one photo**, so a preset applied to another photo carries that colour: it is a recipe the person can open and re-pick, the same as a Colour range's centre and a warp's pins, which presets also keep. Nothing about the node is a stroke or a click, so §3 of `docs/30` (what a preset never holds) is unchanged. The test that a field in no group fails (`docs/30` §14) covers the new fields: they are in `nodes`.
- **Albums** (`docs/31`). An album holds the photo's whole settings graph as stored, so the node travels in `settings` with no change to the file (version 1 stays). An album written with the node is refused by an older build with "This album was made by a newer version of Yellowmoon. Update to open it." (the unknown-kind rule). `written_by.pipeline` records the version (§7).
- **Older documents.** Every document of an older version has no such node and reads as it did. A document that holds the node is refused by an older Yellowmoon with the plain unknown-kind sentence (`docs/23` §3.5).

### 5.5 `ym`

- **No new flag.** The flags set fields of the Input, Main and the Output (`docs/10`); the new kind is not in the default graph. A graph with it goes through `--settings`; **the example of §5.1 in a graph is `docs/10`'s example file**, and the docs test reads and validates it (`crates/ym-cli/tests/docs.rs`, which counts the JSON blocks of *The node-first form*).
- **The help.** The closing paragraph of `ym render --help` names the kind: "other nodes (correctors, Colour Uniform nodes, mixers, splitters and combiners, and Film nodes whose stocks `ym film list` shows)" (CU-03; `docs/10` quotes it word for word and the snapshot `cli__help_render` changes with it).
- **Getting a target by hand.** A person writing a file sets `target` from a colour they know; there is no `ym` command that reads a pixel. The app's Pick colour is the way to read a pixel's colour, and a command that prints one is a later idea. A `ym` user who wants the node local adds a `color_range` row, as the example does.
- **Errors.** A file that breaks a rule is a usage error (exit 2) with the rule's sentence after "The settings file can't be used:", as for every graph rule. `ym render --json` and `ym info --json` print the node in `settings` as the document holds it.
- **Version.** `ym --version`, `ym doctor` and `ym info` print the new pipeline version from CU-12 (§7).

## 6. Rendering

- **The plan** (`docs/23` §5.2, `docs/algorithms/06` §1.3). The node is a step with a picture input, a key input and a key output, exactly as a corrector: one **per-pixel step** that reads its input register and the node's mask rows and writes its picture and its key `m`. It needs no planes of its own (the raster mask planes are a mask's). It fuses into the colour pass like a corrector's grade. `StepKind` gains `ColorUniform` (`ym_color::plan`, `pipeline::schedule::step_kind`; their exhaustive matches fail to compile without it), per pixel, reads the picture, no planes. A step at its identity (§3.5) is skipped: a node with no target or no amount costs nothing.
- **The CPU reference** is `ColorChain::run_nodes` calling `ym_color::uniform::apply` in f64 (CU-04), then the node's mask blend.
- **The record.** A node needs about 10 numbers: `c_t`, `m_t` and its unit vector, the three amounts. CU-05 uses the grade rows of `NodeRecU` that a corrector's grade would use (this kind has no grade) if they are free, and otherwise adds rows at the end as X-05 did (`docs/17` §11a); the `offset_of!` test pins the layout either way, and **no new binding** is added (K3: 8 storage buffers and 1 uniform).
- **Show mask** (`docs/23` §5.8). The node's mask stop is a corrector's: it shows `m`, with nothing new for this kind.
- **Thumbnails, histogram, scatter, pixel readout.** The thumbnail of the node is the picture after it (`docs/23` §5.7). The histogram and scatter at the node read the picture at the node's input (the panel has none, §2.6). `signal_source` (`pipeline::thumbs`) answers an `at` that names the node as it does a corrector's.
- **Ancestors.** `Plan::picture_ancestors` is a corrector's (one picture input). A Film node after Colour Uniform reads its picture like any other node's.
- **Dead nodes** are dimmed like a dead corrector. Not run when the Output does not need them.

## 7. `PIPELINE_VERSION`

No existing graph changes by a bit (§3.5: a node that is not there runs nothing), so **no golden entry moves and `golden check` stays 61 of 61 on the current entries.** An older build refuses a graph with a kind it does not know, so `PIPELINE_VERSION` takes **the next minor step** with the node, the rule `docs/13` and `docs/23` set when a node kind lands: 0.10.0 with the Film node, 0.11.0 with the mixers, 0.12.0 with Face Refinement (`docs/01` *pipeline_version*). **It is 0.13.0 today**, written to `ym:PipelineVersion`, `ym --version`, `ym doctor`, `ym info` and an album's `written_by` by CU-12 (the snapshot and the tests that assert the number follow). **RC-04 does not bump it.** If an accepted raw-constants fit (RC-13, `docs/29`) lands first, CU-12 takes the next free minor number above it; the rule does not depend on the value. Only the written version string changes for an existing file. The new golden entries (node on, §8) are new, so they pin the new maths from this version and are not a change of an existing one.

## 8. Tests

Every gate must pass in CI (or the named job) before its task is done. **A failed gate is reported, never loosened** (`AGENTS.md`). Numbers marked *starting* are first bounds that the named task measures and writes into `docs/STATUS.md`. The sketch's measured numbers (§3.7, §3.8, §3.6) are the ones to beat.

| Area | Test | Gate | Task |
|---|---|---|---|
| Format and rules | the fields, defaults and clamping (`target.saturation` 2 to 50, `compress` limits, `hue_deg` read modulo 360 and not reported), a value that is not finite, an unknown field (a `range` among them) and its text, a partial `compress`, `"target": null`, the 64-node limit counting the node, the schema, the written order | every refusal is atomic and has its plain text | CU-03 |
| Patches | add and remove; `target` and `compress` patches and their reports; **the first pick's single patch** (target, amounts and the Colour range row, and that a second pick, a node with a mask, a node with a key link and a target set by the sliders add no row); Reset; copy and paste of a graph with the node; undo (one step for the pick and the row) | tested | CU-03 |
| Identity | no target; all amounts 0; a bypassed node; a pixel with mask 0; a grey ramp and black (a grey with Hue and Saturation on moves by rounding only, 1e-15; with Lightness it stays grey); one amount at 0 with the others on (Saturation only changes the length of `c` and not its direction) | bit for bit where stated | CU-02 |
| Property tests | **fixed point** (the target colour at any amount); **identity at 0 %** on each axis; **100 %** puts a colour with `r ≥ 0.02` and `|δ| ≤ 90°` on the target's value on each axis to 1e-9; **50 %** is halfway to 1e-12; **monotone** along each axis (hue, saturation, lightness) at 100 samples of the other two for amounts 0.1 to 1 (non-decreasing within 1e-12, strictly increasing below 100 %); **Jacobian** of the whole map ≥ `(1 − κ_s)²(1 − κ_h)(1 − κ_l)` up to numerical noise for equal amounts up to 90 % on a grid, and ≥ −1e-6 at 100 % (the sketch: 2.8e-4 at 90 %, −2.8e-8 at 100 %); **continuity across the hue wrap**: a sweep of the whole hue circle in steps of 0.01° for 8 targets including 0°, 90°, 180° and 359.99° has no step above 5e-3 of a code in the output (the sketch: 3.7e-5); the swap of the 0°/360° seam gives the same result; **determinism** (the same input twice; threads 1, 3, 4); **no NaN or infinity** on grey, black, negative, huge and subnormal codes; **neutrals** (`r = 0` has chroma 0 after); the near-neutral gain of §3.7 within the sketch's numbers; the guards `g` and `a` monotone and in [0, 1]; the worked example of §3.3 to 1e-9 | as stated | CU-02 |
| Formulas | the table of §3.3 on hand-computed values (the worked example, a pixel at 60° and one at 120° on the fade), the hue's short way round (355° and 5°), the target's opposite (`a = 0`, no jump), `a` exactly 0 at 150° | 1e-12 in LogC4 | CU-02 |
| The mask | the node blended through a feathered mask, `in + m (node(in) − in)`; the seeded row's mask is 1 at the picked chroma; Show mask on the node is the ordinary mask `m` | 1e-12 | CU-04 |
| Existing graphs | every golden entry and settings fixture, the CLI's and the app's parity tests, the device suites | **byte-identical**; `golden check` 61 of 61 on the current entries; only the written version moves, in CU-12 | CU-04, CU-05, CU-12 |
| Plan and random graphs | the node in the plan with correctors and Film nodes, live and dead, numbering (it takes a number), `picture_ancestors`; `crates/ym-core/tests/any_graph.rs` draws `color_uniform` from `NodeKind::ALL`, with targets, amounts and masks; the run asserts that the kind appeared | CPU: tiles of 97, 160 and 256 px on 1, 3 and 4 threads bit for bit; device: not refused, within the CPU bounds, exports byte for byte across tile sizes | CU-04, CU-05 |
| Device parity | lavapipe: every axis alone and together, a target at 0°, at 180° and near grey (Saturation 2), pixels on the fade (60° to 150° from the target), negative codes, a node after a node; the Mac GPU CI (Metal, fast-math) the same cases | **2e-4 in LogC4** and one 10-bit code after C12 and C13; the identity cases exact; expected error a few 1e-6 | CU-05 |
| Golden images, node on | three new entries of the synthetic set with the node on, each rendered by `ym` (CPU) and pinned in `expected.json`: **skin** (a target on a skin ramp of the chart, a Colour range row, 60 / 60 / 0), **one hue** (a saturated patch, Hue only at 100 %) and **tone** (a grey-brown ramp, Lightness 100 %); the device checked against them on the GPU subset; the entries are new so no hash moves | `golden check` 64 of 64; device within the golden gates | CU-06 |
| App against `ym` | the example file of §5.1 and one with another mask on the node: JPEG, TIFF 16, PNG 16 and DNG | byte-identical, on the CPU and on the device (`parity.rs`, `parity_gpu.rs`) | CU-06 |
| Presets and albums | a preset saved from a photo with the node holds it in `nodes`, applies it to another photo (target, amounts, masks arrive; a Brush row without strokes); an album written with the node round-trips and an older build's refusal text is the album's; the field test (`docs/30` §14) | tested | CU-07 |
| CLI | the example file read and validated (`docs/10`, `crates/ym-cli/tests/docs.rs`), the help's closing paragraph, the unknown-field sentence | the test fails on any difference | CU-03 |
| Interface | unit tests: `graphOps` (add after the selected node with Pick colour on, delete, reorder like a corrector, the 64-node limit), `uniformMath.ts` against the Rust constants, the target's conversions (a pick's `{a, b, tone}` to Hue, Saturation and Lightness and back), **the first pick's patch (target, amounts and the seeded row; no row on a later pick, with a mask, or from the sliders)**, the disabled states, the swatch's text, the colour names; Playwright against the mock: the Add node menu, picking (the crosshair, Esc, the row appearing in the Mask tab, one undo for both), dragging and typing a target, the sliders, Show mask, Reset of each card and of the node, the keyboard path (the three number fields), Copy and Paste; the glass audit and a11y with the new page states. Reference screenshots (`design/screenshots-c2/photos-uniform-*.png`) | green; the owner sees the screenshots before they are committed | CU-08, CU-09, CU-11 |
| Speed (on a Mac) | a slider move in the fit view with 8 nodes, one of them a Colour Uniform node with all three amounts on; the node's own cost per megapixel | the node view's 100 ms; the per-megapixel budget of §9 (*starting*) | CU-12, M0-H4 |
| Licences | `cargo deny check` and the npm licence check | pass; no new dependency | CU-03 to CU-09 |

## 9. Speed

`docs/23` §5.9 asks for a slider move in the fit view with 8 nodes within 100 ms. The node is cheap: per pixel it does two dot products, one square root, two smoothsteps, one `atan2`, one `sincos` and a few multiplications, and **no `exp2` or `log2`**, which a trim or a Layer needs. With no range windows there is less to do than in the first plan.

| Where | Budget (*starting*; measured by the task named) |
|---|---|
| GPU, Apple M-series, the colour kernel | **≤ 0.12 ms per megapixel** for a node with three amounts on; **≤ 0.03 ms per megapixel** when the node's mask is 0 everywhere (the step returns the input without computing the pull). At the fit view (about 2 megapixels) a slider move re-runs the pass: the node adds ≤ 0.25 ms, a small share of the 100 ms. Measured in CU-12 on a Mac (M0-H4) and on Mesa lavapipe for the parity run; a lavapipe figure is not a Mac figure and is reported as such |
| CPU reference (the fallback and `ym --cpu`) | ≤ 150 ms per megapixel on one core, split across rows and threads; a 61 MP export on 8 cores about 1.2 s for the node. Measured in CU-04 on the Linux machine |
| Memory | no new buffer; the node's record is about 10 numbers (§6) |
| Thumbnail | the node's thumbnail (about 160 px) is a few milliseconds |

## 10. Questions for the owner: answered, and one of ours

### Decided by us, say if you'd rather not

**The first pick also adds a Colour range mask row** (§2.3). With masks alone, a node with no mask acts on the whole photo: after a pick of skin, Hue 50 % would pull every non-grey colour in the photo halfway to orange. So the node's **first pick** adds one Colour range row centred on the picked colour, with the Colour range's own default width (0.05, the slider reads 39) and softness (0.5), when the node has no mask rows and no key link. A later pick moves the target and leaves the masks alone. The pick and the row are one undo step. The person can widen the row, add other masks, or delete it in the Mask tab.

**The alternative:** no automatic mask. The first pick sets only the target and the amounts, and the empty Mask tab shows a hint line: "Colour Uniform acts on the whole photo until a mask limits it. Add mask › Colour range picks a colour to limit it." The first result would be global; the person adds the mask. **Recommended default: the automatic row** (it is what the mock-ups show). If the owner prefers the alternative, §2.1 loses its row, §2.3 and §5.3 lose the seed, the first pick's patch has no `masks.items`, and CU-08 and CU-09 gain the hint line.

### The owner's answers of 2026-11-18 (CU-H1)

| # | Question | Answer (2026-11-18) | Where it is in the spec |
|---|---|---|---|
| 1 | The name in the UI | **"Colour Uniform"**; the identifier is `color_uniform`; "a Colour Uniform node" in sentences | everywhere; §4 |
| 2 | A shift of the target (an Offset row) | **No separate shift.** The target's Hue, Saturation and Lightness can be typed and dragged | §2.1 |
| 3 | Which amounts are on after the first pick | **Hue 50 %, Saturation 50 %, Lightness 0 %**, as planned | §2.1 |
| 4 | A built-in Skin starting point | **No** | §1.2 |
| 5 | Its own Range, or masks alone as Resolve | **Masks alone, as Resolve does.** No Range card, no range fields, no eye. The node's selection is its masks, `W = m`. **A neutral guard stays on the hue and saturation pulls**; the lightness pull acts on everything in the mask | §2.3, §3.2, §3.7 |
| 6 | Show mask shows the mask times the range | **Show mask is the ordinary Show mask of the node's mask**, now that there is no range | §2.4 |
| 7 | Pick colour on when the node is added, and Add node selects the node | **Yes** | §2.1, §4 |
| 8 | The pick is the mean of a 7 × 7 square | **Yes** | §3.11 |
| 9 | A Strength or Blend slider | **No** | §1.2 |
| 10 | Tab colour and place | **The Compress tab in the Colour tab's green; the node after Face Refinement in the Add node menu** | §2.6, §4 |
| 11 | The pipeline version | **The next minor step** (0.13.0 today); RC-04 does not bump it; if RC-13 lands first, the next free minor number | §7 |
| 12 | The owner's extra line | **"Colour Uniform node needs to allow picking target color."** The Target card keeps Pick colour and the swatch, and Hue, Saturation and Lightness with a rainbow track for Hue; dragging them is the other way to choose the colour. glasscn has no colour picker component, so the sliders are the picker | §2.1 |

## 11. Decisions made while writing the spec

The simplest readings where the request was silent. The owner's answers are in §10; these are ours, and the first two follow from them.

| # | Choice | Why |
|---|---|---|
| 1 | The kind is `color_uniform`; the default name is `"Uniform"` | a card cuts a name at ten letters; the menu says "Colour Uniform" |
| 2 | The first pick adds a Colour range row (flagged in §10) | masks alone leave the first result global |
| 3 | A numbered node with masks, a key output, a thumbnail and a number, as a corrector and a Film node; no new port type | the corrector's operations (add, delete, reorder, links) and every mask work unchanged |
| 4 | The space is LogC4 / AWG4 in the warp's plane `(c, m)`: hue is the angle of `c`, saturation is `|c|`, lightness is the mean of the three codes | §3.1: the plane Pick colour, the scatter, the Colour range and the warp share; exposure-independent hue and saturation; exact orthonormal moves; no transfer function, so a light, tight kernel |
| 5 | The lightness axis is the mean of the codes `m`, not the luma code | an exact, provable law (§3.1, §3.8) |
| 6 | The law is `x → t + (x − t)(1 − κ w)` on each axis, with the angle taken the short way round | Resolve's definition of 0, 0.5 and 1; convex; fixed point at the target |
| 7 | The neutral guard `g(r) = S(r/0.02)` on the hue and saturation pulls, none on lightness | the owner's answer 5; the warp's guard, so the two agree on what grey is |
| 8 | **The antipode fade on the hue pull** (full to 90° from the target's hue, zero from 150°), constants and not settings | with no range window the hue pull would be discontinuous at the colour opposite the target; the owner's guarantee of continuity across the wrap needs it (§3.2, §3.6). A change would be the wheel's reach, e.g. 120° to 180°, a one-line change in the core |
| 9 | The target is held at Saturation 2 or more | the saturation axis is then monotone near grey (§3.8); the node never makes a neutral colour |
| 10 | A zero amount skips its axis | identities exact; no trig for a Saturation-only pull |
| 11 | `target` and `compress` in human units (degrees, ×100 chroma, stops, percent) | a file written by hand reads like the panel; the core converts once |
| 12 | Show mask is the ordinary mask; nothing new in the IPC | the owner's answer 6; reuse `docs/23` §5.8 |
| 13 | `sample_chroma` gains `tone` and `radius_px`, both additive | one pick command for the warp, the range and this node |
| 14 | The first pick raises Hue and Saturation to 50 % when all amounts are 0 | a click shows something; the document default stays the identity |
| 15 | The target's three fields are on with no target and rest at 0°, 2, 0; a change makes a target from them, without a mask row | the owner's extra line: dragging them is a way to choose the colour |
| 16 | Presets: the node is in the `nodes` group; albums need no change | the groups were made to hold every node but the three fixed ones |
| 17 | Golden: three new entries with the node on (skin, one hue, tone); none moves | `docs/13`'s rule for a new node kind |
| 18 | `PIPELINE_VERSION` takes the next minor step | the rule of `docs/01` *pipeline_version* |
| 20 | `target` needs all three of `hue_deg`, `saturation` and `lightness_ev` (in a file, in a patch, on the wire); a part left out is a parse error. A patch replaces the target whole, or clears it with `null`; only `compress` is partial (CU-03) | a half target has no meaning; the panel's three fields rest at 0°, 2 and 0 and the first change sends all three |
| 21 | A `target` or `compress` that is a list (`[26, 9.7, 0]`) is a parse error, though serde would read a list as a struct (CU-03) | §5.2 says "not `null` or an object" |
| 22 | The help's closing paragraph keeps "Face Refinement nodes" and adds the new kind: "other nodes (correctors, Colour Uniform nodes, mixers, splitters and combiners, Face Refinement nodes, and Film nodes whose stocks `ym film list` shows)" (CU-03). §5.5's quoted wording was written before milestone S's words were in the paragraph | the paragraph must keep naming every kind |
| 23 | Until CU-04 a Colour Uniform node renders as the identity (the plan and the frame treat it as a bypassed corrector: it has a number, a key output and a place in the plan) and the random-graph test leaves the kind out with that reason; milestone U's CU-03 line says "the plan compiler refuses the kind", the coordinator's brief says identity, and the brief is followed (CU-03). `nodes_key` already holds the node's target, amounts, bypass and masks | a node that does nothing is honest while it has no step; CU-04 replaces both arms (`TODO(CU-04)` in `pipeline/color.rs`, `pipeline/schedule.rs` and `tests/any_graph.rs`) |
| 24 | The first pick's rules for the row (no row on a later pick, with a mask, with a key link, or from the sliders) belong to the interface (CU-08, `graphOps`). The core's patch carries `target`, `compress` and `masks.items` as one atomic patch, and the undo step is the patch between the two documents (CU-03 tests that) | the core has no notion of a pick |
| 25 | The colour names (CU-08, `colourName` in `uniformMath.ts`): the hue's name from the angle (the warp's names), "pale" under Saturation 4, "a little pale" under 12, "strong" over 30, "dark" below -1.5 EV and "light" above +1.5 EV; two qualifiers are joined with a comma | it gives the mock-up's "Orange, a little pale" for hue 26, Saturation 9.7 |
| 26 | The Target card's sub-line is "Pick again on the photo, or set the numbers." whenever there is a target; with none it is "Click a colour in the photo, or set Hue, Saturation and Lightness." (CU-09) | the target does not record whether it was picked or typed, so the mock-up's "Picked from the photo" would be wrong half the time |
| 27 | The Target and Compress cards both have a Hue and a Saturation slider; the cards (regions "Target" and "Compress") tell them apart for a screen reader, as in the mock-up's labels (CU-09) | the labels are the mock-up's; a distinct name would read worse |
| 28 | CU-04: `NodeParams` gains `uniform: Option<UniformStep>` (target and amounts; `Some` for every node of the kind, so the step kind is `ColorUniform` even when the node is skipped); `PreparedNode` skips a bypassed node or one at its identity. Until CU-05 `PreparedChain::prepare` (`ym-gpu`) returns `GpuError::InvalidInput("a Colour Uniform node is not on the device yet")` for a plan with a live node, which the render takes as it takes any refusal: the CPU makes the frame and `RenderReport::cpu_reason` says why. `signal_source` reads the picture at the node's input for every tap (the rewrite with the node bypassed). `radius_px` is held to 15 (a 31 × 31 square); the square is `1/zoom` final pixels per output pixel (one for the single-pixel reading as before), and the point decides `null` (a square that reaches past the edge reads what the view makes there). The wire's `tone` and optional `radius_px` are in the app (`sample_chroma`, `bindings.ts`) | the spec was silent on the limit, the unit of the square and the refusal text |
| 19 | Other documents (`docs/23`, `docs/10`, `docs/11`, `docs/17`, `docs/30`, `docs/algorithms/06`) are updated by the tasks that change what they describe, not by this plan | as X-01 listed them; this plan only adds `docs/32`, the research, the mock-ups, milestone U and the STATUS lines |

## 12. Mock-ups

`design/colour-uniform/` (`README.md`, `index.html`, `mockup-notes.md`, the JPEGs, `shoot.mjs`, `html/`): pages at 1440 × 900 built from the app's DOM and CSS, as `design/mixers/` and `design/skin/`, shot at 2× to JPEGs. The pictures are drawings (a face from `design/skin/`) with the **maths of §3 applied by the sketch** and blended through the seeded Colour range mask, so the before and after are what the node would do. They show: the node with its two cards (the face after the pull, `01`); picking the target (`02`); Show mask on the seeded row's mask (`03`); the node's Mask tab with the seeded COLOUR RANGE row (`03b`); the Add node menu (`04`); the simple view (`05`); a diagram of how the pull works on the plane and the tone (`06`); and a sheet of states and copy. The owner's answers (CU-H1) are in; the pictures follow them.

## 13. Tasks

Task format of `docs/15`, milestone U. The work goes on its own branch, one reviewed commit per task with its checks green (`cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`, `npm test`, `npm run build`). Every task that changes what `ym` does or prints updates `docs/10` and the help in the same commit. The tasks are listed with their *Done when* in `docs/15` (milestone U: **CU-01** to **CU-12**, with **CU-H1** answered and **[HUMAN] CU-H2**); the order is CU-01 (done), then CU-02 and CU-03; CU-04 after CU-02 and CU-03; CU-05 after CU-04; CU-06 and CU-07 after CU-05 (CU-07 needs only CU-03); CU-08 after CU-03; CU-09 and CU-10 after CU-08 and CU-04; CU-11 after CU-09 and CU-10; CU-12 last, then CU-H2.

## 14. As built

*(Filled in by the tasks as they land.)*

**CU-03 (format, validation, patches, types, schema and CLI).** `crates/ym-core/src/settings/uniform.rs` (`ColorUniformNode`, `TargetValue`, `UniformAmounts`, the ranges, `wrap_hue_deg`); `NodeKind::ColorUniform`, `NodeGraph.uniforms`, `SettingsField::Target` and `Compress` (the nodes' group), `NodePatch.target` (`Option<Option<_>>`) and `compress` (`UniformAmountsPatch`); the report paths are `nodes.items[N].target.saturation` and the like, collected as the `Target` and `Compress` fields; `crates/ym-core/schema/settings.schema.json` (`color_uniform_node`, `target_value`, `uniform_amounts`); the wire types in `src-tauri/src/wire.rs`, `src/api/bindings.ts`, `types.ts` and `contract.ts`; the mock's validator and patch application (`src/fixtures/graph.ts`, `src/state/settingsGraph.ts`, `src/api.mock.ts`). In the interface the kind has the Mask tab only until CU-09 (`src/components/tabs.ts`).

**CU-08 and CU-09 (the webview and the panel, branch `cu-08-web`).** `src/components/uniformMath.ts` (the TypeScript port of the conversions, the guards, `applyUniform`, the colour names and the swatch's name, with `uniformMath.node.test.ts` against the Rust constants); `src/state/graphOps.ts` (`insertAfter` with `kind: "color_uniform"`, `resetNode`, the head's line, and `pickUniform`, the first pick's one patch of section 2.3); `store.pickUniform`; the Add node item; the card's glyph; `src/fixtures/uniform.ts` and the mock's `sample_chroma` with `tone`; `src/screens/nodes/UniformTab.tsx` (the Compress tab) and `Slider`'s `trackBackground` for the Hue track. Tests: `graphOps.uniform.test.ts`, `UniformPanel.test.tsx`, `AddNodeMenu.items.test.tsx`, `TabBar.test.tsx`, and the Colour Uniform test of `e2e/a11y.spec.ts`. Decisions: section 11 items 25 to 27. `ChromaSample.tone` is optional in `api/types.ts` and `radiusPx` is not yet passed by the real `tauriApi` (`TODO(CU-10)`).

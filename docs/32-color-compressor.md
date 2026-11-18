# 32 · Colour Compressor

The Colour Compressor is a node that makes a range of colours more alike. The person picks one colour on the photo, the **target**, and the node pulls the colours near it toward it: their hue, their saturation and their lightness, each by an amount from 0 to 100 %. It is for the things a photographer wants to be one colour and that light has made several: skin with a cast on one cheek, a dress or a car in uneven light, a lawn, a patch of sky. This file covers what it does, its controls, the maths, where it sits in the graph, the settings, the pipeline version, the tests, the speed budget, the questions for the owner, the mock-ups and the tasks.

**Status: planned (CC-01, 2026-11-18). Nothing here is built.** The owner's request of 2026-11-18: "plan 'color compressor' node, similar to Resolve color compressor, that will also go into 0.2.4". This is the plan, as presets (`docs/30`) and mixers (`docs/25`) were planned: a spec, mock-ups for the owner to look at (`design/colour-compressor/`, **CC-H1**) and a task list (milestone C of `docs/15`). No interface work starts before the owner has looked. The research behind it is `research/color-compressor-2026-11-18.md`; **this file wins where they differ**. The maths is ours: we copy no code and no formula from any tool (`docs/14`).

**In short.**

- A new node kind, `color_compressor`, in the graph of `docs/23`: a numbered node like a corrector, with masks, a key output, a thumbnail and a number on its card. Add node › **Colour Compressor**, "Pulls a range of colours toward one colour." (§4). It counts toward the 64 nodes.
- **Target** (§2.1): picked on the photo with Pick colour, as the colour warp's is, and shown as a swatch with Hue, Saturation and Lightness as numbers that can be typed. **Compress** (§2.2): Hue, Saturation, Lightness, each 0 to 100 %. **Range** (§2.3): how wide a slice of colours is selected, in hue, saturation and lightness, with Softness. **Show affected area** is the existing Show mask (§2.4).
- **Maths** (§3): on the LogC4 / AWG4 values the node reads, in the plane and the mean that the colour warp and the Colour range mask already use. A selection weight `W` from three smooth windows; each axis moves `x → t + (x − t)(1 − κ·W)`. Exact identity at 0 %, the target is a fixed point, continuous across the hue wrap, never selects greys, and non-decreasing along each axis (§3.8).
- **A new node changes no pixel** until it has a target and an amount above 0 (§2.5).
- **Rendering** (§6): one per-pixel step, no new binding, CPU in f64 and device in f32 within the grade's 2e-4. `PIPELINE_VERSION` takes a minor step; no existing graph changes by a bit (§7).

## 1. What it does

A photographer looks at a face where one side has a green cast, or a red dress that is orange in the shade. A corrector can lift or lower a colour for the whole photo, or for a mask; the colour warp can move a colour, with its neighbours following. Neither makes the spread of colours **smaller**. The Colour Compressor does:

1. Click a colour that is right (a good patch of skin).
2. The node finds the colours near it: the ones within a range of hue, of saturation and of lightness, with a soft edge.
3. It pulls each of them toward the picked colour. At 50 % a colour is halfway between where it was and the picked colour; at 100 % it is the picked colour on that axis. Hue, saturation and lightness are set apart, so the person can even out the cast (hue and saturation) and leave the light and shade (lightness) alone, which is what keeps a face round.

It is the tool Resolve calls Color Compressor and Capture One's Skin Tone *Uniformity*, with a selection built in. Resolve's tool has none: it relies on a qualifier or a window before it (`research/color-compressor-2026-11-18.md` §1). Here the range **is** the qualifier, and the node's masks (`docs/23` §3.3) still work on top, so either way of working is available.

### 1.1 How it differs from the colour warp, and why both exist

Both read the same chroma plane and both have Pick colour. They do opposite things.

| | Colour warp (`docs/02` C9b) | Colour Compressor |
|---|---|---|
| What a pin or target does | Moves a colour to somewhere else; the colours near it follow by the same move | Pulls the colours near the target **toward** it, each by a share of its own distance |
| Effect on the spread of colours | Keeps it (a translation, folded never: it is a flow) | Shrinks it: two colours end nearer each other than they began |
| Axes | The chroma plane, with an exposure and a saturation per pin | Hue, saturation and lightness set apart, each with its own amount |
| Selection | The pin's Gaussian range, tonal weights for shadows and highlights | Three windows (hue, saturation, lightness) with Softness |
| Use | "Make these reds more orange", a look | "Make this skin one colour", a correction |
| Lives | A card in a corrector's Colour tab | Its own node |

A warp cannot do what the compressor does: moving every colour in a range by the same amount keeps the cast, and shrinking the spread would take a pin for each colour. The compressor cannot do what the warp does: it has no destination but one colour. Neither replaces the other, so both stay. The compressor reuses the warp's plane, its neutral guard and its Pick colour, so a colour picked for one reads the same in the other (§3.1).

### 1.2 What is not in this version

- **A shift of the target** (Capture One's Amount sliders; Resolve's tool has none). Typing a new Hue, Saturation or Lightness into the target does it (§2.1); a separate offset row is a later idea (§10, question 2).
- **A target that is a colour range, or several targets.** One target per node; add a node for another colour.
- **Pick by dragging a rectangle** (the mean of a region). Pick takes the mean of a small square at the click (§3.12). A rectangle is a later idea.
- **A Blend or Strength slider.** Resolve has one on every ResolveFX filter; here the node's masks (with their Opacity) and the three amounts do the job (§10, question 9).
- **Compressing toward a grey.** Neutrals are never selected and the target is held off grey (§3.7): a tool that makes colours alike does not colour or uncolour a neutral. Saturation does that.
- **Auto.** No Auto, as for every node but Main (`docs/23` §3.2).
- **Skin detection.** The node does not know what skin is; the person's click does. Face Refinement (`docs/27`) is the node that knows faces.

## 2. Controls

All copy is British, short and plain. Controls are glasscn's (`docs/16`); a control glasscn has is never hand-written. The mock-ups are `design/colour-compressor/` (§8).

### 2.1 Target

A card named **Target**, first in the node's **Compress** tab.

| Control | Range | Default | Notes |
|---|---|---|---|
| **Swatch** | | empty (a checked square) | The target colour as a patch, as the picture shows it (its own hue, saturation and lightness), with the colour's name beside it ("Orange", "Orange, a little pale"). Not a Tab stop; its name is read as text |
| **Pick colour** | a toggle with an eyedropper | **on when the node is added**, off after the first pick | Like the warp's and the Colour range's. A click on the photo sets the target to the colour there. It stays on for the next click and **Esc** ends it |
| **Hue** | 0 to 359° (wraps) | – | A number field with the slider's look (`Slider` with its value box), the track drawn as a hue ramp |
| **Saturation** | 2 to 50 | – | The chroma of the colour times 100, the number the warp's range uses (a skin tone is about 5 to 12; a strong red 13) |
| **Lightness** | −4 to +8 EV | – | Stops from middle grey (code 0.2784), `+0.5 EV` |
| **Reset** (card header) | | | Clears the target (back to none). The amounts stay |

- **Hue, Saturation and Lightness can be typed**, so a target can be set without a pick and the picked one can be moved: that is the target's shift. They are disabled (`aria-disabled`, with "Pick a colour first.") until the node has a target; typing into one of them with no target is not offered.
- **What a pick reads.** The mean colour of a 7 × 7 pixel square at the click, in the picture the node **reads** (its input, before the node), as a Colour range's pick does (`docs/11` *Pick colour of a Colour range*, §3.12). A click on the letterbox does nothing.
- **A pick too grey to have a hue.** Below Saturation 2 the target is held at 2 on the direction of the pick (a pick of exactly neutral takes hue 0) and the card says "That colour is too grey for a hue. It was moved to Saturation 2." once (a notice line under the swatch, gone at the next change), as a clamp is reported elsewhere.
- **The first pick** also sets Compress to Hue 50 % and Saturation 50 % when all three amounts are 0, in the same patch (one undo step), so a click shows something. Later picks change the target only (§10, question 3).
- **Keyboard.** The pick is a pointer action, so the three number fields are the keyboard way to a target (`docs/28`). The swatch's text is the colour's name and its three numbers: "Target colour: orange, hue 26 degrees, saturation 9.7, lightness minus 0.03 stops."

### 2.2 Compress

A card named **Compress**, second.

| Slider | Range | Default | Step | Reads as |
|---|---|---|---|---|
| **Hue** | 0 to 100 % | 0 | 1 | "60%" |
| **Saturation** | 0 to 100 % | 0 | 1 | "60%" |
| **Lightness** | 0 to 100 % | 0 | 1 | "0%" |

- 0 % moves nothing. At 100 % every colour the Range selects fully has the target's value on that axis; colours at the soft edge move by a share (§3.3). The line under the sliders: "At 100 % every colour in the range takes the target's value." (shown in the card's quiet text).
- The sliders are `aria-disabled` while the node has no target, with the line "Pick a colour to compress toward." in their place.
- **Reset** (card header) puts the three at 0.
- Hue and Saturation are the axes that even out a cast; Lightness flattens shading, so it is a separate amount and the first pick leaves it at 0.

### 2.3 Range

A card named **Range**, third, with the eye of §2.4 in its header.

| Slider | Stored | Range | Default | Step | Reads as |
|---|---|---|---|---|---|
| **Hue** | `hue_deg` | 1 to 85° (a half-width) | 20 | 1 | "±20°" |
| **Saturation** | `saturation` | 0.5 to 30 (a half-width, in the units of Target › Saturation) | 5 | 0.5 | "±5" |
| **Lightness** | `lightness_ev` | 0.1 to 8 EV (a half-width) | 1.5 | 0.1 | "±1.5 EV" |
| **Softness** | `softness` | 0 to 100 % | 50 | 1 | "50%" |

- A range is a **half-width** about the target: Hue ±20° selects colours up to 20° either side of the target's hue, fully, and Softness adds a soft edge beyond (§3.2).
- **Softness** is one control for all three edges: the edge is that share of each half-width (and never narrower than a floor, §3.2), so 0 is a short, steep edge and 100 % an edge as wide as the half-width.
- **Hue stops at ±85°**, so that a colour on the other side of the wheel is never selected (§3.6). The Saturation and Lightness ranges reach 30 (the whole of the plane the warp draws is 0 to 30) and ±8 EV, which is every colour and every tone of a photo: the axis is then effectively off.
- **Reset** puts the Range at its defaults. The tooltips (drafts): Hue "How far from the target's hue colours are still pulled."; Saturation "How far from the target's saturation colours are still pulled."; Lightness "How far from the target's lightness colours are still pulled."; Softness "How gradually the pull fades at the edge of the range."
- With no target the Range card is shown and works (the person may set it first); the eye is off (§2.4).

### 2.4 Show affected area

The toolbar's **Show mask** (M) and the Mask tab's View card (`docs/23` §5.8, §7.6) already show where a node acts. **They fit, with one rule: on a Colour Compressor the weight shown is `m · W`**, the node's mask times its selection weight (§3.2), so the picture shows exactly where the node will pull, with its soft edge. Overlay tints it in the mask accent at 0.55; Mask only shows it as grey. Neither reads the amounts: the area is shown at 0 % too, so it can be tuned before the pull is raised.

The **eye in the Range card's header** ("Show affected area") is a second switch for the same view: it turns Show mask on and off. It writes nothing to the settings and makes no undo step. Both stay in step (the toolbar toggle and the eye show the same state), and the eye is off with the tooltip "Pick a colour first." until the node has a target (a node with no target has no `W`; Show mask on it shows `m` alone, as on a corrector). The mask view is a view: not in an export, the Before view, a thumbnail or the pixel readout (`docs/23` §5.8).

### 2.5 Defaults and the identity

- A new node is `target: null`, Compress 0, 0, 0, Range 20 / 5 / 1.5 / 50, no masks, not bypassed. **It changes no pixel.**
- With **no target** the node is the identity whatever the amounts are. With a target and **all three amounts 0** it is the identity too. Either is skipped: it runs no code and changes no bit (§3.5).
- So "a colour is picked or an amount is raised" reads: picking sets the target **and** raises Hue and Saturation to 50 % (§2.1), and raising an amount in the panel needs a target. A document or `ym --settings` file can hold an amount with no target; it does nothing.

### 2.6 The panel

| Part | Content |
|---|---|
| Head | The corrector's compact head (`docs/23` §7.5): the number chip, the name, the line, Bypass, Reset, More. Line: "Colour Compressor", "Bypassed", "Not connected" (short, because the name is long and the line is cut after it) |
| Tabs | **Compress · Mask**, in the Colour tab's green (`--tab-colour`; `docs/11`). The Mask tab is a corrector's (`docs/23` §7.6) |
| Compress › cards | **Target · Compress · Range** |
| Histogram | none (the node has no light to show); the node's Mask tab and Show mask work as on a corrector |
| Keys | ⌘1 and ⌘2 select the tabs. No new single key: Pick colour is the toggle, and it is on when the node is added |

The panel is 400 px wide and three cards are about 250 + 190 + 330 px, so at 1440 × 900 the Range card scrolls a little (the panel's scroll area is 678 px); the first two cards, the ones used most, fit. **Reset this node** (the head) puts target, Compress and Range back to their defaults; masks stay (`docs/23` §7.5). The More menu is a corrector's.

## 3. The maths

The CPU reference is `ym_color::compress` (CC-02), in f64. The kernel (CC-05) is the same expressions in f32 (§3.10). A sketch of it in Python (`design/colour-compressor/html/compress_ref.py`) was used to try every statement below and to draw the mock-ups; it is not product code and the property checks of §3.8 run on it.

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

1. **It is the plane everything else reads.** The colour warp's pins, the Colour range mask's centre, the chroma scatter and Pick colour (`sample_chroma`) all use `c`. A colour picked for the warp or the range reads the same here, and no new colour space or conversion is added.
2. **Hue and saturation are nearly independent of exposure.** In LogC4 a change of exposure adds nearly the same amount to the three channels, so it moves `m` and leaves `c` almost alone. That is why a ramp across a face's shading keeps its hue and saturation, so the compressor can even out hue and saturation without touching the light, and why a lightness range of ± so many EV is a range in stops.
3. **The moves are exact and need no transfer function.** The basis is orthonormal, so a move of `c` and a move of `m` are independent, and the node needs no decode and no encode: no `exp2` or `log2` on the device, which is what keeps it cheap and its parity tight (§3.10). The tone is the mean of the three codes rather than the luma code (`L` of the Brightness range) **on purpose**: a move of `m` is an exact addition to all three channels, so the law of §3.3 is exact on `m` and the lightness axis can be proved monotone. For the colours of one hue the two differ by a nearly constant offset, so a person sees no difference; the swatch and the Lightness field use `m`, so they agree with what the node does.
4. **Not HSL or HSV.** Resolve's tool speaks of H, S and L of an unstated space. HSL of a display colour is not exposure-independent and its hue is badly behaved in a log space (the Frame.io note, `research/color-compressor-2026-11-18.md` §1). Not OKLab: it needs a transfer function and a matrix per pixel and gives no better hue on scene values than the plane the product already has.

**Where in the graph.** The node runs on the picture it reads, in LogC4 / AWG4, between its neighbours, like any node. `W` is a function of the node's input pixel only (never of its own result), as every node's `m` is (`docs/algorithms/06` §1).

### 3.2 The selection weight

The stored settings are converted once (the core, f64): the target `t = (c_t, m_t)` with `r_t = Saturation/100`, `c_t = r_t (cos h, sin h)`, `m_t = 0.2784 + ev·S`; the half-widths `w_h = hue_deg·π/180`, `w_r = saturation_range/100`, `w_m = lightness_ev·S`; the softness `σ = Softness/100`.

For a pixel with chroma `c`, `r = |c|` and tone `m`:

```
S(x)   = u²(3 − 2u),  u = clamp(x, 0, 1)                         the smoothstep of docs/algorithms/06
win(d, w, q) = 1 − S((d − w)/q)           1 for d ≤ w, 0 for d ≥ w + q, smooth between

δ      = atan2(c_t × c, c_t · c)   in (−π, π]: the signed angle from the target's hue to the pixel's,
                                    taken the short way round (the cross and dot products; 0 when r = 0)
g(r)   = S(r / 0.02)                the neutral guard of the warp (WARP_NEUTRAL_GUARD)

W_h = win(|δ|,       w_h, max(σ w_h, 0.05))                     0.05 rad ≈ 2.9°
W_r = win(|r − r_t|, w_r, max(σ w_r, 0.02))
W_m = win(|m − m_t|, w_m, max(σ w_m, 0.02))

W   = g(r) · W_h · W_r · W_m        in [0, 1]
```

- Each window is **1 inside its half-width and falls to 0 smoothly over `q`**, with zero slope at both ends. No edge is hard, however small Softness is: `q` has a floor (0.05 rad for hue, 0.02 of a code value for saturation and for lightness, the Colour range's and Brightness range's `RANGE_RAMP_MIN`, `docs/23` §4.3), so the device and the CPU agree and Metal's fast maths cannot flip a pixel (`docs/17` K8). No result depends on a rounding: every `if` in the kernel chooses between results that agree to within rounding (§3.10).
- **The product of three windows** selects a slice (a wedge of hue, a ring of saturation, a band of lightness), and `g` takes the neutrals out (§3.7). A colour is selected where it is near the target in all three.
- `W` is the weight Show mask draws (§2.4).

### 3.3 The compression law

With the amounts `κ_h`, `κ_s`, `κ_l` in [0, 1] (Compress ÷ 100):

```
hue         δ'  = δ · (1 − κ_h W)                      the angle to the target's hue shrinks
saturation  r'  = r_t + (r − r_t) · (1 − κ_s W)        the distance to the target's chroma shrinks
lightness   m'  = m_t + (m − m_t) · (1 − κ_l W)        the distance to the target's tone shrinks

c'  = r' · (cos(θ_t + δ'), sin(θ_t + δ'))              θ_t = atan2(c_t)  (the target's hue angle)
v'  = v + (c'₁ − c₁) e1 + (c'₂ − c₂) e2 + (m' − m)·(1,1,1)
```

Each line has the form `x → t + (x − t)(1 − κ W)`: the colour keeps the share `1 − κ W` of its distance to the target on that axis. So:

- **0 %** (`κ = 0`) is the identity on that axis. **100 % with `W = 1`** puts the colour on the target's value on that axis, as the Resolve manual defines its 1.000 ("identical to the Target Color"); **50 %** puts it halfway.
- **The colour of the target is a fixed point**: at `δ = 0`, `r = r_t`, `m = m_t` every line returns its input, whatever `κ` and `W`.
- A colour at the soft edge (`0 < W < 1`) moves by `κ W` of its distance: the pull fades out. A colour outside the window (`W = 0`) does not move.
- **Hue is pulled by angle**, not by the straight line between two colours. A straight pull in the plane would also pull the saturation toward zero on the way, and the axes would not be independent. Because `δ` is the short way round, a pixel 30° from the target is pulled 30° · `(1 − κ W)`, never the long way.
- The three axes use the **same `W`**, evaluated on the pixel's original coordinates, so they do not depend on one another's results and their order does not matter.
- **Saturation first, then rotation.** `r'` is the chroma the pixel ends with; the rotation by `δ' − δ` is applied to the direction. A pixel's hue changes without its chroma changing (when `κ_s = 0`, `r' = r`).

A **worked example** (computed with the sketch in f64). Target: Hue 26°, Saturation 9.7, Lightness −0.03 EV. Range: ±20°, ±5, ±1.5 EV, Softness 50 %. A pixel at Hue 46°, Saturation 8.5, Lightness +0.47 EV is 20° from the target's hue (on the edge of the full range), so `W = 1`: at 50 % on all three it becomes Hue 36°, Saturation 9.1, Lightness +0.22 EV; at 100 % it becomes Hue 26°, Saturation 9.7, Lightness −0.03 EV, the target. A pixel at Hue 56° (30° out, where the hue window is half way down its 10° edge, `W = 0.5`) at 100 % on Hue only lands on 41° (`30° · 0.5 = 15°` left).

### 3.4 Skipping

- **No target, or all three amounts 0:** the node is skipped, and `v' = v` exactly (§3.5).
- **An amount of 0 skips its axis.** With `κ_h = 0` the direction of `c` is kept and only its length changes: `c' = c · r'/r` (and `c' = c` when `r = 0`), with no trig, so a Saturation-only pull moves the length and nothing else, to the bit. With `κ_s = 0, κ_h > 0` `r' = r`. With `κ_l = 0`, `m' = m` and the `(m' − m)` term is not added.
- **A pixel with `W = 0` exactly returns its input bit for bit** (the kernel and the reference test `W = 0` first and skip the rest). `W` reaches 0 smoothly (zero slope), so the two results agree within rounding on either side of the test (K8).

### 3.5 The identity, bit for bit

A node with no target, with all amounts 0, with `W = 0` at a pixel, or bypassed gives its input's bits. The plan skips the first two (a step at its identity is skipped, `docs/23` §1.3); the third is §3.4. The blend with the node's mask follows: `out = in + m_node (node(in) − in)` (`docs/algorithms/06` §1.1), so `m_node` of 0 or 1 gives the input or the node's output exactly.

### 3.6 Hue wrap-around

- `δ` is the signed angle **the short way round**, `atan2(cross, dot)` of the target's and the pixel's chroma vectors, in (−π, π]. The 0°/360° seam does not exist for it: a target at 355° and a pixel at 5° are 10° apart.
- The hue half-width is at most 85° and the edge at most as wide again, so **`W_h` is exactly 0 for `|δ| ≥ 170°`** (`w_h + q ≤ 85° + 85°`, `q ≤ σ w_h ≤ w_h`; the floor 0.05 rad adds 2.9° only when `w_h` is below 2.9°). Across the jump of `atan2` at `δ = ±π` (the colour exactly opposite the target) `W = 0`, so `δ' = δ`, `r' = r`, and both sides give the pixel's own colour: **the map is continuous on the whole plane**, with no seam at the target's opposite or at 0°/360°.
- The rotation uses `cos` and `sin` of `θ_t + δ'`, which are periodic, so no angle is ever wrapped by hand and none is compared.

### 3.7 Neutrals and low chroma

Hue is undefined at grey and unstable near it, so the node **never selects a neutral**: `g(r) = S(r/0.02)` is a factor of `W` and is 0 at `r = 0` and 1 from `r = 0.02` (the warp's neutral guard, so the two agree on what grey is).

- At `r = 0` `W = 0`: a grey, or a black, does not move, whatever the target. `δ` is then not used: the kernel evaluates `atan2` only where `r > 0` (a `select`), so no NaN reaches the result (§3.10).
- Near grey `W` is small, so a pixel with a faint cast moves by a small share, and a pixel's displacement is at most `2r` (the rotation of a vector of length `r`) so the map is continuous at the origin.
- **The target is held at Saturation 2 or more** (`r_t ≥ 0.02`, §2.1, §4). This is what keeps the saturation axis monotone near grey (§3.8): `g` rises only below `0.02`, and the target is above it.

### 3.8 Monotonicity: no colour order is inverted

**What is guaranteed (proved):** the node, with its mask at 1, is **continuous** and, along each axis with the other two coordinates fixed, **non-decreasing**, and **strictly increasing for any amount below 100 %**. So an ordered ramp of colours (a gradient across a face, a graduated sky) stays an ordered ramp: no inversion, no band where two different colours swap places, and no posterisation below 100 %. At 100 % colours the window selects fully are all given the target's value on the axis: that is the point of the tool (a plateau in the axis: the colours are made one), and the map stays non-decreasing.

*Proof.* On the hue axis at fixed `r` and `m`, the new angle is `δ' = δ(1 − κ W(δ))` with `W(δ) = A · W_h(|δ|)`, `A ≥ 0` a constant and `W_h` equal to 1 up to `w_h` then non-increasing in `|δ|`. For `δ > 0`: `dδ'/dδ = 1 − κ W − κ δ W'(δ) ≥ 1 − κ W ≥ 1 − κ ≥ 0`, because `W' ≤ 0` there; `δ < 0` is the mirror. On the saturation axis, `W(r) = g(r) W_r(r)·B` with `B` constant: for `r ≥ r_t` (and `r_t ≥ 0.02` so `g = 1`), `W` is non-increasing, and `dr'/dr = 1 − κW − κ (r − r_t) W' ≥ 1 − κ`; for `r < r_t` `W` is non-decreasing (`g` and `W_r` both rise), so `−κ (r − r_t) W' ≥ 0` and the same bound holds. On the lightness axis it is the same as hue's with `m`. In each case the weight is **unimodal about the target** and constant in the other two axes, which is what the windows of §3.2 are made to be. ∎

**What is not guaranteed, and is measured:** the three axes together. A pixel's `W` depends on all three coordinates, so the map of the whole space can in principle fold where a colour is pulled across another that is pulled less. Measured on the sketch (200,000 random points of the plane and the tone range, three range settings from tight to the widest): the Jacobian determinant of the whole map is **positive everywhere for amounts up to 90 %** (smallest 2.7e-4, so no fold) and **zero or numerically zero, never meaningfully negative, at 100 %** (the colours the window selects fully collapse onto the target: a determinant of exactly 0, with rounding noise of 1e-10 on a finite difference of step 1e-6). CC-02 turns this into a test (§7) with the bound stated there, and writes the numbers it measures in `docs/STATUS.md`.

**With masks.** A spatial mask (Linear, Radial, Brush, Object, …) mixes the node's picture with its input pixel by pixel, so colours at different places have different amounts and no colour order is defined across them: the guarantee is for the node's own law. A Colour range or Brightness range row makes the mix depend on colour, with a smooth edge no narrower than `RANGE_RAMP_MIN`; the result is continuous and the guarantee is the row's.

### 3.9 Gamut

- **Nothing is clipped.** Like a corrector, the node does not clamp: a result may leave LogC4's usual range, and the Output's cine rendering (C12) decides what a colour outside the display's gamut looks like (`docs/02`, `docs/25` §3.3).
- **The pull stays between the pixel and the target.** The saturation axis is a convex combination of `r` and `r_t`; `r' ≤ max(r, r_t)`, and the hue rotation does not change the length. The lightness axis is a convex combination of `m` and `m_t`. So the node never takes a colour further out than the farther of the pixel and the target on any axis: it cannot push a pixel to a more extreme chroma than one of the two it was made from.
- **Where it can leave the gamut.** Moving a dark colour toward a more saturated target can give a channel below the LogC4 code of zero (a negative linear value), as any corrector's gain can. Such values are finite and are rendered like any other (the Output's C12 rule). The tests assert finiteness, not a range (§7).
- No gamut mapping step is added; a colour that the target would carry out of the display's gamut is the person's choice of target.

### 3.10 f64 on the CPU, f32 on the device, and the parity bound

- **CPU reference** (`ym_color::compress`): f64, `f64::atan2`, `f64::cos` and `f64::sin`, no `fma`, the order of §3.2 and §3.3. Rows are split across threads and the result is the same for any number of threads (`docs/17` D5). The same libm functions that LogC4's `exp2` and `log2` use give the same results on one machine; across platforms a last-digit difference of the three trig functions is below the golden rule's ΔE00 0.001 (`docs/13` *The x86_64 slice*: 21 identical, 4 within the rule).
- **Device**: the same expressions in f32 in the colour kernel's plan loop (`color.wgsl`), with `atan2`, `cos` and `sin` of WGSL. It needs no `exp2`, `log2` or `pow`. The mask-and-compress arithmetic is additions, multiplications, two dot-product-like sums, three smoothsteps and one `atan2`, one `sincos`.
- **NaN safety.** `atan2(0, 0)` is not defined in WGSL: the kernel computes it only where `r > 0` and takes 0 elsewhere, where `W = 0` anyway. A non-finite code value in the input (never made by the pipeline) would give a non-finite result and is not guarded.
- **Parity bound.** The device is held to the CPU reference at **2e-4 in LogC4** (the grade's, `docs/13`, `docs/17` C10) and **one 10-bit code after C12 and C13**. The expected error is a few 1e-6 (f32 rounding of values below 1 and of `sin`, `cos` and `atan2`); CC-05 measures it and writes the number into `docs/STATUS.md`, as X-05 did, and the gate stays 2e-4. The steepest part is the neutral guard: `g` has a slope of at most 75 (`1.5/0.02`), so an input that differs by 1e-7 (what CPU and device agree on, `docs/23` §4.3) moves `W` by 8e-6 and a pixel by less than 3e-6.
- **Metal (K8).** Which window a pixel is in, the amounts, the target and the identity are integers and constants the host writes; the float tests that remain (`W == 0`, `r > 0`, the windows' clamps) choose between results that agree within rounding, each function being continuous across its test, so fast-math re-association moves a result by rounding only. The Mac GPU CI (`.github/workflows/mac-gpu.yml`) runs the parity cases on Metal.
- **Determinism.** The node is per pixel, so tiles of any size equal the whole frame bit for bit and a repeat render is identical (hard rule 8). The app and `ym` write byte-identical files (hard rule 6).

### 3.11 The pick

`sample_chroma(x, y, at, tap)` (`docs/01`, the IPC table) already returns the chroma `{ a, b }` of the pixel in the plane of `e1`, `e2`. For this node it gains two things, **both additive** (a caller that does not read them is unchanged):

- the answer carries **`tone`**, the mean `m` of the three codes of the same pixel, so the page can set a target without converting a colour;
- an optional **`radius_px`** (default 0, so every existing caller reads one pixel): the colour is the **mean of the LogC4 triples** in the square of `2·radius_px + 1` pixels about the point, in the picture the tap reads, then reduced to `c` and `m`. The node's pick asks for `radius_px: 3` (a 7 × 7 square): a single pixel of skin is full of noise and a target taken from one is a noisy target. The square is in output pixels at the viewport's current scale.

The tap is `"node_input"`, the picture the node reads (as a Colour range's, `docs/11`): the target is a colour of the picture the compression acts on, so the picked colour is a fixed point of the node at once (§3.3). The page then converts to the stored form (Hue in degrees from `atan2(b, a)`, Saturation `100 √(a² + b²)`, Lightness `(tone − 0.2784)/S`), the one place it converts, with the formula of `ym_color::compress` ported to `compressMath.ts` and checked against the Rust constants by a node test, as `warpMath.ts` is.

## 4. Where it sits

**A corrector-like numbered node** (the Film node's precedent, `docs/24` §4.5; the mixers' kinds, `docs/25` §2). `kind` is `color_compressor`.

| Property | Value |
|---|---|
| Ports | in `picture`, `key`; out `picture`, `key` (as a corrector's, `docs/23` §3.4) |
| Number, thumbnail | numbered in the plan's order; a card with a thumbnail and a number chip; the node view and the thumbnails path make one thumbnail for it (`docs/23` §5.7) |
| Masks | `masks` as a corrector's (`docs/23` §3.3): the Mask tab, Invert, the key input, Show mask; the masks multiply on top of `W` (§2.4) |
| Bypass | a bypassed node passes its picture through; its key output still carries `m` (as a corrector's) |
| Add node | **Colour Compressor**, after Face Refinement and before the separator: "Pulls a range of colours toward one colour." **Adds one node after the selected one, like Film** (`docs/23` §7.4), selects it, opens its Compress tab and turns Pick colour on. Off at 64 nodes with "A photo can have up to 64 nodes." |
| Auto | none |
| More menu | a corrector's: Rename · Move earlier · Move later · Link picture to ▸ · Link key to ▸ · Remove link ▸ · Delete node. Delete links what it fed from its source, as a corrector's |
| Card | a corrector's card (112 × 84 px at 100 %: thumbnail, number chip, name, footer glyphs); the node's glyph, a pair of arrows pointing at a dot, sits in the footer where a mask glyph does (the Film node's and Face Refinement's way) |
| Simple view | not shown there. The notice counts it with the others ("This photo has 3 nodes.", `docs/23` §6), and **Open nodes** goes to the node view |
| Name | the field `name`, default `"Compressor"` (ten letters, so it is not cut on the card), 1 to 32 characters as a corrector's. The menu and the panel's head line say "Colour Compressor" |
| Sentence name | "a colour compressor": "Node 4 is a colour compressor, not a corrector." (`NodeKind::with_article`, `sentence`) |

The node does not belong to Main: it is a separate node after Main (or anywhere in the graph), so Main's Auto, the simple view's controls and Re-run never touch it.

## 5. Settings

### 5.1 The format

A new value of `kind` (`docs/23` §3.2): `color_compressor`. The node has these fields and no others; another field is an unknown field and is refused.

| Field | Type / range | Default | Notes |
|---|---|---|---|
| `name`, `bypass`, `pos`, `masks` | as a corrector's | `"Compressor"`, false, – , none | |
| `target` | `null`, or `{ hue_deg, saturation, lightness_ev }` | `null` | `hue_deg` any finite number, read modulo 360 into [0, 360) (**not clamped, not reported**); `saturation` 2 to 50 (clamped, reported; a value below 2 is held at 2); `lightness_ev` −4 to +8 (clamped, reported) |
| `range` | `{ hue_deg, saturation, lightness_ev, softness }` | `{ 20, 5, 1.5, 50 }` | `hue_deg` 1 to 85, `saturation` 0.5 to 30, `lightness_ev` 0.1 to 8, `softness` 0 to 100. A partial object is read over the defaults |
| `compress` | `{ hue, saturation, lightness }` | `{ 0, 0, 0 }` | each 0 to 100 (percent). A partial object is read over the defaults |

```json
{ "id": 4, "kind": "color_compressor", "name": "Compressor", "bypass": false, "pos": [2, 0],
  "target": { "hue_deg": 26, "saturation": 9.7, "lightness_ev": -0.03 },
  "range": { "hue_deg": 30, "saturation": 5, "lightness_ev": 1.5, "softness": 60 },
  "compress": { "hue": 60, "saturation": 60, "lightness": 0 },
  "masks": { "invert": false, "items": [] } }
```

The numbers are in the units the panel shows, so a file written by hand reads like the panel (the core converts to the plane's units once, §3.2). A graph that has the node, as the core writes it, is `docs/23` §3.1's with one more item and two more links.

### 5.2 Validation and clamping

- A value out of range is **clamped and reported** by its path (`4.target.saturation`, `4.range.hue_deg`, `4.compress.lightness`), as `docs/23` §3.5; a value that is **not finite** is `SettingsError::NotFinite` and nothing is applied. A `target` that is not `null` or an object, or has a field of another type, is `SettingsError::Parse`.
- The fields `kind`, `target`, `range` and `compress` follow the unknown-field rule: **another field is refused** ("A colour compressor has no field called “gain”.").
- The rules of `docs/23` §3.5 hold: ports exist on their kind, an input port has one link, no cycles, ids are unique, **at most 64 numbered nodes** counting this one ("A photo can have up to 64 nodes."), the kind is known. The unknown-kind example of `docs/10` and the tests stays “outside” (`docs/25`).
- A `target` with `saturation` below 2 reads as 2 with the clamp reported, so the target is always held off grey (§3.7).

### 5.3 Patches, Reset, Copy and Paste, undo

- **Patches** (`docs/23` §3.7): a node is added by giving its `kind` in `nodes` and removed with `remove`; its links go with it. `SettingsField` gains `target`, `range` and `compress`, in the **nodes' group** (a change runs the colour pass again and nothing else; names and `pos` re-run nothing). A patch carries a partial `range` or `compress` over what the node has; `"target": null` clears. The first pick of §2.1 is one patch (`target` and `compress`), one undo step. A drag on a slider is a patch per frame that folds into one undo step, as every slider.
- **Reset** (`SettingsPatch::reset_node_in`): target to `null`, `range` and `compress` to their defaults; name, bypass, masks and links stay.
- **Copy settings and Paste** carry the node like any node, in the whole graph (`docs/23` §3.8). Undo diffs per node.
- **Resolved settings, XMP and DNG.** The node has no Auto, so it is in `ResolvedSettings.settings` as it is. `ym:Settings` holds it and the DNG bakes it into the pixels like every node (`docs/09`).
- **The JSON Schema** (`crates/ym-core/schema/settings.schema.json`) gains `color_compressor_node` (with `target_value`, `compress_range` and `compress_amounts` definitions) and the kind in the `kind` enum. The wire types gain `NodeKind::ColorCompressor` and `NodeItem::ColorCompressorNode`; `bindings.ts` is regenerated and the field-name test compares them.

### 5.4 Presets and albums

- **Presets** (`docs/30` §2). A preset's **`nodes` group** holds every node other than the Input, Main and the Output, so the Colour Compressor is in it with no change to the groups: a preset saved from a photo that has the node holds the node, its target, range, amounts and masks, and applying it adds or replaces them like the other nodes of the group. The preset's table text of §2 gains "Colour Compressors" in the list of kinds (CC-07). **A target is a colour of one photo**, so a preset applied to another photo carries that colour: it is a recipe the person can open and re-pick, the same as a Colour range's centre and a warp's pins, which presets also keep. Nothing about the node is a stroke or a click, so §3 of `docs/30` (what a preset never holds) is unchanged. The test that a field in no group fails (`docs/30` §14) covers the new fields: they are in `nodes`.
- **Albums** (`docs/31`). An album holds the photo's whole settings graph as stored, so the node travels in `settings` with no change to the file (version 1 stays). An album written with the node is refused by an older build with "This album was made by a newer version of Yellowmoon. Update to open it." (the unknown-kind rule), which is the album's own rule for a field this version does not know. `written_by.pipeline` records the version (§7).
- **Older documents.** Every document of an older version has no such node and reads as it did. A document that holds the node is refused by an older Yellowmoon with the plain unknown-kind sentence (`docs/23` §3.5).

### 5.5 `ym`

- **No new flag.** The flags set fields of the Input, Main and the Output (`docs/10`); the new kind is not in the default graph. A graph with it goes through `--settings`; **the example of §5.1 in a graph is `docs/10`'s example file**, and the docs test reads and validates it (`crates/ym-cli/tests/docs.rs`, which counts the JSON blocks of *The node-first form*).
- **The help.** The closing paragraph of `ym render --help` names the kind: "other nodes (correctors, colour compressors, mixers, splitters and combiners, and Film nodes whose stocks `ym film list` shows)" (CC-03; `docs/10` quotes it word for word and the snapshot `cli__help_render` changes with it).
- **Getting a target by hand.** A person writing a file sets `target` from a colour they know; there is no `ym` command that reads a pixel. the app's Pick colour is the way to read a pixel's colour, and a command that prints one is a later idea.
- **Errors.** A file that breaks a rule is a usage error (exit 2) with the rule's sentence after "The settings file can't be used:", as for every graph rule. `ym render --json` and `ym info --json` print the node in `settings` as the document holds it.
- **Version.** `ym --version`, `ym doctor` and `ym info` print the new pipeline version from CC-12 (§7).

## 6. Rendering

- **The plan** (`docs/23` §5.2, `docs/algorithms/06` §1.3). The node is a step with a picture input, a key input and a key output, exactly as a corrector: one **per-pixel step** that reads its input register and the node's mask rows and writes its picture and its key `m`. It needs no planes of its own (the raster mask planes are a mask's). It fuses into the colour pass like a corrector's grade. `StepKind` gains `ColorCompressor` (`ym_color::plan`, `pipeline::schedule::step_kind`; their exhaustive matches fail to compile without it), per pixel, reads the picture, no planes. A step at its identity (§3.5) is skipped: a node with no target or no amount costs nothing.
- **The CPU reference** is `ColorChain::run_nodes` calling `ym_color::compress::apply` in f64 (CC-04).
- **The record.** A node needs about 16 numbers: `t = (c_t, m_t)` and its unit vector, three half-widths with their `q`, `σ`, and the three amounts. CC-05 uses the grade rows of `NodeRecU` that a corrector's grade would use (this kind has no grade) if they are free, and otherwise adds rows at the end as X-05 did (`docs/17` §11a); the `offset_of!` test pins the layout either way, and **no new binding** is added (K3: 8 storage buffers and 1 uniform).
- **Show mask** (`docs/23` §5.8). The plan compiled to a mask stop on this node computes `m · W` on the node's input (§2.4). It is made in the kernel with the colour pass, or on the CPU when the stage allows only that, with the view's gates (one 10-bit code).
- **Thumbnails, histogram, scatter, pixel readout.** The thumbnail of the node is the picture after it (`docs/23` §5.7). The histogram and scatter at the node read the picture at the node's input (the panel has none, §2.6). `signal_source` (`pipeline::thumbs`) answers an `at` that names the node as it does a corrector's.
- **Ancestors.** `Plan::picture_ancestors` is a corrector's (one picture input). A Film node after the compressor reads its picture like any other node's.
- **Dead nodes** are dimmed like a dead corrector. Not run when the Output does not need them.

## 7. `PIPELINE_VERSION`

No existing graph changes by a bit (§3.5: a node that is not there runs nothing), so **no golden entry moves and `golden check` stays 61 of 61 on the current entries.** An older build refuses a graph with a kind it does not know, so `PIPELINE_VERSION` takes a **minor step** with the node, the rule `docs/13` and `docs/23` set when a node kind lands: 0.10.0 with the Film node, 0.11.0 with the mixers, 0.12.0 with Face Refinement (`docs/01` *pipeline_version*). **It is 0.13.0**, written to `ym:PipelineVersion`, `ym --version`, `ym doctor`, `ym info` and an album's `written_by` by CC-12 (the snapshot and the tests that assert the number follow). If another branch with a pixel change (the raw constants of `docs/29`) has moved the number by then, CC-12 takes the next free minor step above it; the rule does not depend on the value. Only the written version string changes for an existing file. The new golden entries (node on, §8) are new, so they pin the new maths from this version and are not a change of an existing one.

## 8. Tests

Every gate must pass in CI (or the named job) before its task is done. **A failed gate is reported, never loosened** (`AGENTS.md`). Numbers marked *starting* are first bounds that the named task measures and writes into `docs/STATUS.md`.

| Area | Test | Gate | Task |
|---|---|---|---|
| Format and rules | the fields, defaults and clamping (`target.saturation` 2 to 50, `range` and `compress` limits, `hue_deg` read modulo 360 and not reported), a value that is not finite, an unknown field and its text, a partial `range` or `compress`, `"target": null`, the 64-node limit counting the node, the schema, the written order | every refusal is atomic and has its plain text | CC-03 |
| Patches | add and remove; `target`, `range` and `compress` patches and their reports; the first pick's single patch; Reset; copy and paste of a graph with the node; undo | tested | CC-03 |
| Identity | no target; all amounts 0; a bypassed node; a pixel with `W = 0`; a grey ramp and black; one amount at 0 with the others on (the axis is exact: Saturation only changes the length of `c` and not its direction to the bit) | bit for bit | CC-02 |
| Property tests | **fixed point** (the target colour at any amount and any `W`); **identity at 0 %** on each axis; **100 %** puts a colour inside the full range on the target's value on that axis to 1e-9; **50 %** is halfway to 1e-12; **monotone** along each axis (hue, saturation, lightness) at 100 samples of the other two for amounts 0.1 to 1 (non-decreasing within 1e-12, strictly increasing below 100 %); **Jacobian** of the whole map ≥ 0 up to numerical noise at 100 % and > 0 up to 90 % on a grid (*starting*; the sketch's 2.7e-4 is the number to beat); **continuity across the hue wrap**: a sweep of the whole hue circle in steps of 0.01° for 8 targets including 0°, 180° and 359.99° has no step above 5e-3 of a code in the output; the swap of the 0°/360° seam gives the same result; **determinism** (the same input twice; threads 1, 3, 4); **no NaN or infinity** on grey, black, negative, huge and subnormal codes; **grey never moves**; the selected weight `W` in [0, 1] and equal to 1 at the target; the neutral guard (`g`) monotone; a worked example of §3.3 to 1e-9 | as stated | CC-02 |
| Formulas | the table of §3.3 on hand-computed values (the worked example, a pixel on the soft edge), the hue's short way round (355° and 5°), the target's opposite, a hue range of 85° with Softness 100 % (`W_h` exactly 0 at 170°) | 1e-12 in LogC4 | CC-02 |
| The mask | `m · W` for Show mask, with a feathered mask on the node; `W` equals the node's input's; the mask blend `in + m (node(in) − in)` | 1e-12 | CC-04 |
| Existing graphs | every golden entry and settings fixture, the CLI's and the app's parity tests, the device suites | **byte-identical**; `golden check` 61 of 61 on the current entries; only the written version moves, in CC-12 | CC-04, CC-05, CC-12 |
| Plan and random graphs | the node in the plan with correctors and Film nodes, live and dead, numbering (it takes a number), `picture_ancestors`; `crates/ym-core/tests/any_graph.rs` draws `color_compressor` from `NodeKind::ALL`, with targets and ranges that select pixels and amounts; the run asserts that the kind appeared | CPU: tiles of 97, 160 and 256 px on 1, 3 and 4 threads bit for bit; device: not refused, within the CPU bounds, exports byte for byte across tile sizes | CC-04, CC-05 |
| Device parity | lavapipe: every axis alone and together, hue range 85°, a target at 0°, at 180° and near grey (Saturation 2), `W` at the soft edge, negative codes, the mask stop, a node after a node; the Mac GPU CI (Metal, fast-math) the same cases | **2e-4 in LogC4** and one 10-bit code after C12 and C13; the identity cases exact; expected error a few 1e-6 | CC-05 |
| Golden images, node on | three new entries of the synthetic set with the node on, each rendered by `ym` (CPU) and pinned in `expected.json`: **skin** (a target on a skin ramp of the chart, 60 / 60 / 0), **one hue** (a saturated patch, Hue only at 100 %) and **tone** (a grey-brown ramp, Lightness 100 %); the device checked against them on the GPU subset; the entries are new so no hash moves | `golden check` 64 of 64; device within the golden gates | CC-06 |
| App against `ym` | the example file of §5.1 and one with masks on the node: JPEG, TIFF 16, PNG 16 and DNG | byte-identical, on the CPU and on the device (`parity.rs`, `parity_gpu.rs`) | CC-06 |
| Presets and albums | a preset saved from a photo with the node holds it in `nodes`, applies it to another photo (target, range, amounts, masks arrive; a Brush row without strokes); an album written with the node round-trips and an older build's refusal text is the album's; the field test (`docs/30` §14) | tested | CC-07 |
| CLI | the example file read and validated (`docs/10`, `crates/ym-cli/tests/docs.rs`), the help's closing paragraph, the unknown-field sentence | the test fails on any difference | CC-03 |
| Interface | unit tests: `graphOps` (add after the selected node with Pick colour on, delete, reorder like a corrector, the 64-node limit), `compressMath.ts` against the Rust constants, the target's conversions (a pick's `{a, b, tone}` to Hue, Saturation and Lightness and back), the first pick's patch, the disabled states, the swatch's text, the colour names; Playwright against the mock: the Add node menu, picking (the crosshair, Esc), typing a target, the sliders, Show affected area and its eye, Reset of each card and of the node, the keyboard path (the three number fields), Copy and Paste; the glass audit and a11y with the new page states. Reference screenshots (`design/screenshots-c2/photos-compressor-*.png`) | green; the owner sees the screenshots before they are committed | CC-08, CC-09, CC-11 |
| Speed (on a Mac) | a slider move in the fit view with 8 nodes, one of them a Colour Compressor with all three amounts on; the node's own cost per megapixel | the node view's 100 ms; the per-megapixel budget of §9 (*starting*) | CC-12, M0-H4 |
| Licences | `cargo deny check` and the npm licence check | pass; no new dependency | CC-03 to CC-09 |

## 9. Speed

`docs/23` §5.9 asks for a slider move in the fit view with 8 nodes within 100 ms. The node is cheap: per pixel it does two dot products, one `atan2`, one `sincos`, three smoothsteps and a few multiplications, and **no `exp2` or `log2`**, which a trim or a Layer needs. A pixel with `W = 0` (the great majority in a photo whose range is a face) returns early.

| Where | Budget (*starting*; measured by the task named) |
|---|---|
| GPU, Apple M-series, the colour kernel | **≤ 0.15 ms per megapixel** for a node with three amounts on and a range that selects half the frame; **≤ 0.03 ms per megapixel** for a node that selects nothing. At the fit view (about 2 megapixels) a slider move re-runs the pass: the node adds ≤ 0.3 ms, a small share of the 100 ms. Measured in CC-12 on a Mac (M0-H4) and on Mesa lavapipe for the parity run; a lavapipe figure is not a Mac figure and is reported as such |
| CPU reference (the fallback and `ym --cpu`) | ≤ 150 ms per megapixel on one core, split across rows and threads; a 61 MP export on 8 cores about 1.2 s for the node. Measured in CC-04 on the Linux machine |
| Memory | no new buffer; the node's record is about 16 numbers (§6) |
| Thumbnail | the node's thumbnail (about 160 px) is a few milliseconds |

## 10. Questions for the owner

Each has a recommended default, the one the plan and the mock-ups use. The owner answers at CC-H1 (`docs/STATUS.md`, *Waiting on the owner*); a different answer changes the named part only.

| # | Question | Recommended default | What changes if the answer differs |
|---|---|---|---|
| 1 | **The name in the UI.** "Colour Compressor" is the name of Resolve's tool (a ResolveFX filter). Use it, or a name of our own? | **"Colour Compressor"**: it is the owner's own phrase, says what the node does, and the product already uses the words of the tools it follows (Parallel, Layer, Splitter, Combiner). A name of our own: **"Colour Uniformity"** (says the result; Capture One's slider is called Uniformity, so it echoes another product) or **"Colour Gather"** (says the action, nobody's name) | The menu's item and line, the head line, the sentence name "a colour compressor", the kind's article, the schema's definition names (the identifier `color_compressor` could stay) and the mock-ups' text |
| 2 | **A shift of the target** (Capture One's Amount; Resolve's tool has none). Add an Offset row (Hue, Saturation, Lightness) to the Target card? | **No.** Typing a new Hue, Saturation or Lightness into the target is the same thing and is already there (§2.1). One set of numbers, not two | An Offset row in the Target card and three more fields in the format; a later task |
| 3 | **Which axes are on by default.** After the first pick, which amounts does the node start at? | **Hue 50 %, Saturation 50 %, Lightness 0 %.** Hue and saturation even out a cast; lightness flattens shading and looks artificial when pushed (a trainer's warning in the research), so it is a deliberate move | The first pick's patch (§2.1) |
| 4 | **A built-in "Skin" starting point?** A shipped preset, or a skin button? | **No shipped preset.** The defaults (Range ±20° / ±5 / ±1.5 EV; after a pick of skin, 50 / 50 / 0) are already skin-friendly; presets are the person's own files (`docs/30`) and none ships today; a click on skin is the starting point. A skin example goes into `docs/10` for `ym` users | A "Skin" item in the node's More menu or a shipped preset (which would be the first built-in preset, a change to `docs/30`'s store) |
| 5 | **Does the node have its own Range, or rely on masks as Resolve does?** | **Its own Range**, and masks on top (§1). A Range keeps one node doing the whole job, shows what it selects, and is what Capture One has | Without it the node has no Range card, is Compress alone, and the person adds a Colour range mask; the maths of §3 would use `W = 1` |
| 6 | **Show mask on this node shows `m · W`** (the node's mask times its own selection), not `m` alone. | **Yes** (§2.4), with an eye in the Range card as a shortcut | A separate view state and a field in `mask_view` |
| 7 | **Pick colour is on when the node is added, and Add node selects the node.** | **Yes**, as a Colour range row's pick is on when the row is added | The first click on the photo would select nothing; the Pick toggle starts off |
| 8 | **The pick is the mean of a 7 × 7 square**, not one pixel. | **Yes** (§3.11). One pixel of skin is noise | `radius_px` default; a later drag-a-rectangle pick |
| 9 | **A Strength or Blend slider** (Resolve's ResolveFX Blend). | **No.** The three amounts, the node's masks and their Opacity do it | One more slider (0 to 100 %) and a field |
| 10 | **Tab colour and place.** The Compress tab in the Colour tab's green; the node after Face Refinement in the Add menu. | **Yes** | The mock-ups' colour and the menu's order |
| 11 | **Pipeline 0.13.0**, a minor step, as for the earlier node kinds. | **Yes** (§7) | The number only |

## 11. Decisions made while writing the spec

The simplest readings where the request was silent. **Each is a working choice that waits on CC-H1.**

| # | Choice | Why |
|---|---|---|
| 1 | The kind is `color_compressor`; the default name is `"Compressor"` | a card cuts a name at ten letters; the menu says "Colour Compressor" |
| 2 | A numbered node with masks, a key output, a thumbnail and a number, as a corrector and a Film node; no new port type | the corrector's operations (add, delete, reorder, links) and every mask work unchanged |
| 3 | The space is LogC4 / AWG4 in the warp's plane `(c, m)`: hue is the angle of `c`, saturation is `|c|`, lightness is the mean of the three codes | §3.1: the plane Pick colour, the scatter, the Colour range and the warp share; exposure-independent hue and saturation; exact orthonormal moves; no transfer function, so a light, tight kernel |
| 4 | The lightness axis is the mean of the codes `m`, not the luma code | an exact, provable law (§3.1, §3.8) |
| 5 | The law is `x → t + (x − t)(1 − κ W)` on each axis, with the angle taken the short way round | Resolve's definition of 0, 0.5 and 1; convex; fixed point at the target |
| 6 | The selection is the product of three windows with smoothstep edges and a floor on each edge, times a neutral guard | smooth, no hard edge, unimodal about the target (monotone), neutrals never selected |
| 7 | The target is held at Saturation 2 or more | the saturation axis is then monotone near grey (§3.8); the node never makes a neutral colour |
| 8 | The hue half-width stops at 85° | `W` is 0 beyond 170°, so the `atan2` jump at 180° is harmless (§3.6) |
| 9 | One `W` for the three axes, evaluated on the original coordinates | the axes are independent of each other's results and of order |
| 10 | A zero amount skips its axis; a pixel with `W = 0` is returned unchanged | identities bit for bit; early-out on the device |
| 11 | `target`, `range` and `compress` in human units (degrees, ×100 chroma, stops, percent) | a file written by hand reads like the panel; the core converts once |
| 12 | Show mask shows `m · W`; the eye in the Range card is a second switch for it | reuse `docs/23` §5.8; nothing new in the IPC (the view's `node` is the node) |
| 13 | `sample_chroma` gains `tone` and `radius_px`, both additive | one pick command for the warp, the range and this node |
| 14 | The first pick raises Hue and Saturation to 50 % when all amounts are 0 | a click shows something; the document default stays the identity |
| 15 | Presets: the node is in the `nodes` group; albums need no change | the groups were made to hold every node but the three fixed ones |
| 16 | Golden: three new entries with the node on (skin, one hue, tone); none moves | `docs/13`'s rule for a new node kind |
| 17 | `PIPELINE_VERSION` takes a minor step to 0.13.0 | the rule of `docs/01` *pipeline_version* |
| 18 | Other documents (`docs/23`, `docs/10`, `docs/11`, `docs/17`, `docs/30`, `docs/algorithms/06`) are updated by the tasks that change what they describe, not by this plan | as X-01 listed them; this plan only adds `docs/32`, the research, the mock-ups, milestone C and the STATUS lines |

## 12. Mock-ups

`design/colour-compressor/` (`README.md`, `index.html`, `mockup-notes.md`, the JPEGs, `shoot.mjs`, `html/`): pages at 1440 × 900 built from the app's DOM and CSS, as `design/mixers/` and `design/skin/`, shot at 2× to JPEGs. The pictures are drawings (a face from `design/skin/`) with the **maths of §3 applied by the sketch**, so the before and after are what the node would do. They show: the node in the graph with its three cards (the compressed face); picking the target; Show affected area; the Add node menu; the simple view; a sheet of states and copy; and a diagram of how the pull works on the plane and the tone. They wait for the owner (CC-H1).

## 13. Tasks

Task format of `docs/15`, milestone C. The work goes on its own branch, one reviewed commit per task with its checks green (`cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`, `npm test`, `npm run build`). Every task that changes what `ym` does or prints updates `docs/10` and the help in the same commit. The tasks are listed with their *Done when* in `docs/15` (milestone C: **CC-01** to **CC-12**, with **[HUMAN] CC-H1** and **CC-H2**); the order is CC-01, then CC-02, CC-03 and CC-H1; CC-04 after CC-02 and CC-03; CC-05 after CC-04; CC-06 and CC-07 after CC-05; CC-08 after CC-03 (it needs no pictures from the device); CC-09 and CC-10 after CC-H1; CC-11 after CC-09 and CC-10; CC-12 last.

## 14. As built

*(Filled in by the tasks as they land.)*

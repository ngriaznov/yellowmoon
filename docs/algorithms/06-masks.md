# 06 · Nodes and masks (the maths)

This file is the maths of `docs/23-nodes.md`: what a corrector node computes in what order, how the graph of nodes becomes a plan that is evaluated, the two trims, how masks combine (the key input among them), the formula of every mask kind, the raster pipeline that Object, Subject, People, Background and Brush masks go through, the SAM 2.1 small runtime, and the tests that pin them. The numbered sections are cited by the code (`ym-color`, `ym-gpu`, `ym-core`, `ym-ml`). The data, the interface and the tasks are in `docs/23`; the colour stages it builds on are in `docs/02`.

| | |
|---|---|
| Spec | `docs/23-nodes.md` |
| CPU reference | `ym_color::display` (the node loop and the trims), `ym_color::mask` (shapes, keys, combination), `ym_gpu::cpu::mask_raster` (resample, grow, feather, guided filter) |
| GPU kernels | the fused colour kernel `ym-gpu/src/shaders/color.wgsl` (nodes, trims, shapes, keys), `ym-gpu/src/shaders/mask_raster.wgsl` (the raster pass) |
| SAM 2.1 small | `ym_ml::sam2` (CPU, f32), weights in `data/models/sam2/` |
| Ships | on: a photo without nodes beyond Main, with no trim and no mask renders the bytes of pipeline 0.8.2 (section 1.4) |

Notation. `v` is a LogC4 triple, `lin = decode(v)` linear AWG4 (`docs/02` Constants). `Y(lin)` is the AWG4 luminance row of C9, `L = encode(Y(lin))`. `S(t) = t²(3 − 2t)` for `t` in [0, 1] (smoothstep; 0 below, 1 above). `e₁ = (1, −½, −½)/√1.5` and `e₂ = (0, 1, −1)/√2` are the chroma axes of C9 and C9b, and `c(v) = (e₁·v, e₂·v)`. All CPU arithmetic is f64; the GPU runs the same expressions in f32 (`docs/17` K7: constants are written from the Rust constants).

## 1. One corrector node

### 1.1 Order

A node takes the LogC4 value `v_in` that the node before it (or C8, for Main) made, and gives `v_out`:

```
1. trims        lin  = decode(v_in) ⊙ g          g = 2^ev · d(temperature, tint)        (section 2)
                v_t  = encode(lin)
2. C9           v_9  = grade(v_t, L_t)           L_t = encode(Y(lin))                   (docs/02 C9)
3. C9b          v_w  = warp(v_9)                                                         (docs/02 C9b)
4. C11          v_c  = curve(v_w)                                                        (docs/02 C11)
5. mask         m    = mask(v_in, L_in)                                                  (section 3)
                v_out = v_in + m · (v_c − v_in)
```

- **Identity skipping.** A step at its identity is skipped, so a node at defaults changes no bit. Step 1 is skipped when `ev = 0`, `temperature = 0` and `tint = 0` (each exactly 0); step 2 as in `docs/02` C9 (each row at 0 is skipped); step 3 when the node has no pin that moves or acts (`docs/02` C9b); step 4 when all four curves are the identity (two points at (0,0) and (1,1)). A node whose steps 1 to 4 are all skipped, a bypassed node and a node whose `m` is 0 everywhere are skipped as a whole (`v_out = v_in`, and its masks are not evaluated).
- **The luma `L`.** C9 and C11 need the luma of the pixel they read. For Main, whose input is C8's, `L_t = encode(Y(lin))` is computed from the linear values that C7 made, with the trim's gain in them when there is one: no `decode(encode(x))` round trip, so Main with no trim has the bytes C9 had before nodes existed. For every later node `lin = decode(v_in)` is what is known, and `L_t = encode(Y(decode(v_t)))` (with `lin` taken after the trim when the trim runs). `L_in`, the luma the Brightness range reads, is `encode(Y(decode(v_in)))` (for Main the same linear values as above, before the trim).
- **The mask reads the node's input.** `m` is a function of `v_in` (and of the pixel's place, for shapes and raster masks), never of `v_c`: grading a node does not move its own mask or the next node's (`docs/23` section 4.7).
- **The blend is in LogC4.** The look's intensity blends the same way (`docs/02` C10). `m = 1` gives `v_c` exactly and `m = 0` gives `v_in` exactly (the kernel does not compute `v_in + 1·(v_c − v_in)`: it branches on `m` being exactly 1 or exactly 0, so a node without masks has no rounding of the blend).

### 1.2 What changes in the chain

Before nodes the per-pixel chain after C8 was C9 → C9b → C10 (look) → C11 (curve) → C12 → C13. It is now **the nodes in plan order (each: trims, C9, C9b, C11, mask blend) → C10 → C12 → C13**: the curve moved in front of the look (owner, 2026-11-01). A photo with both a curve that is not the identity and a look other than Neutral renders a little differently; every other photo renders as before (section 1.4). LogC3-to-Rec.709 looks already ran the curve first (`docs/02` C10), so every look now follows one rule.

### 1.3 From the graph to the plan

The settings hold a graph (`docs/23` section 3): corrector nodes, the Input and the Output, and links between typed ports (`picture` and `key`). The core **compiles** the graph into a plan; the CPU reference evaluates the plan's steps in order and the kernels read the same plan.

1. **Live nodes.** The Output's picture input has one link. A node is **live** when it is needed: it is on the chain of picture links that leads back from the Output to the Input, or its key output feeds (through any number of key links) the key input of a live node, or it is a picture ancestor of such a node. Other nodes are **dead**: they are kept in the settings and do nothing (their cards are dimmed).
2. **Order.** The plan is a topological order of the live nodes over all links, picture and key (a validated graph has no cycle). Ties are broken by Main first, then by the card's `pos` (x, then y), then by the id, so the order is the same on every machine. A step is one node: `(node, picture source, key source)`.
3. **Registers.** Each step reads the picture value of its picture source and writes its own output picture value; a value is kept until its last reader. Values are LogC4 triples and the plan gives each a *register*; a step that reads a key (section 3, step 2) reads the key value of its key source, a scalar `m`, kept in a *key slot* until its last reader. Both are per pixel: a pixel's whole plan is computed from the pixel's own input and nothing else, so tiles stay exact.
4. **Passes.** A run of per-pixel steps on one pixel is one **pass**. A graph without joins (every graph of this version: a node has one picture input, and the Output has one) compiles into **one pass**, as long as its steps need at most `PLAN_REGISTERS = 8` pictures and `PLAN_KEYS = 16` keys alive at once, which a chain or a chain with a few keyed side branches always does. A graph that needs more, which only a hand-written file with many fan-outs can, and every graph with a join (Parallel, Layer, Splitter and combiner, Outside: a later version) is split by the compiler into several passes with f32 RGB buffers between them (one buffer per value that crosses a split). The compiler is written for splits from the start; this version has the code path and tests it with a graph that needs 9 registers (`docs/23` section 11).
5. **Stops.** A thumbnail of node `k` is the plan compiled to end at `k`'s picture value and then run through the Output; the mask view of `k` is the plan compiled to end at `k`'s `m`. Both name a node, live or dead (a dead node's thumbnail shows what it would make from its own input).

The default graph, Input → Main → Output, is the plan `[Main]`. Nodes numbered on a card are the node's place in this order.

### 1.4 Bit-identity of the default graph

With `nodes` absent (the default graph: Main with no trim and no masks) the plan is `[Main]` and the chain runs Main's C9, C9b and C11 on the top-level settings, in the order of section 1.1 with steps 1 and 5 skipped, and then C10. For a photo whose curve is the identity or whose look is Neutral this is the sequence of 0.8.2 (a Neutral look and an identity curve are identities, and C10 and C11 commute when either is the identity), so the bytes are equal on the CPU reference. The one case that differs, a curve and a look together, has its own test (section 7). On the device the gates of `docs/17` §12 apply and lavapipe gives the same bytes where the kernel's expressions are unchanged.

### 1.5 The Output

The Output holds C10 (the look and its intensity), C12 (the cine rendering), output sharpening and C13, and is not a node. Its settings are `look`, `look_intensity` and `output_sharpening`. It runs once, after the last node.

## 2. The trims

Both trims are multiplications in linear AWG4. They act first, so the grade after them sees the corrected picture.

### 2.1 Exposure trim

`ev` in stops, −4 to +4 (`TRIM_EXPOSURE_RANGE_EV`), default 0, shown with two decimals and step 0.01. `lin' = lin · 2^ev`, then `encode`. It is exact: the same maths as the Input's exposure (`docs/02` C5–C6), applied to this node's input, with C7's range and toe unchanged (the trim is after C7). A negative `lin` (an out-of-gamut colour) is multiplied like any other, so the trim never clips.

### 2.2 Temperature and tint trim

Temperature and tint are `t` and `n`, each −100 to +100 (`TRIM_TEMPERATURE_RANGE`, `TRIM_TINT_RANGE`), default 0, step 1. They move the white that the picture is balanced for, as the Kelvin and Tint rows of the Input do, and give a von Kries gain in linear AWG4 (a diagonal matrix, green 1).

**The photo's white after the Input.** The Input makes the photo's resolved white neutral: whatever the Kelvin, tint, light preset or Pick white of the photo, its white is `[1, 1, 1]` in AWG4 at the Input's output, and `[1, 1, 1]` in AWG4 is the gamut's white, D65 (x 0.3127, y 0.3290, `docs/02` Gamuts). The trim therefore starts from D65 whatever the Input says. That is the reading of "the photo's resolved white" that this spec uses (`docs/23` section 14, decision 4): a node's gain depends on its two numbers and on nothing else, so the same node pasted on another photo, or kept while the Input's Kelvin changes, renders the same change, and a node's cache key needs no Input state.

**The moved white.** On the CIE 1960 `(u, v)` diagram:

1. **Temperature.** `m₀ = 1e6 / 6504` mired (D65's correlated colour temperature; 153.76 mired). The balanced-for white moves to `m₁ = m₀ − 0.5·t` mired: `t = +100` is 50 mired lower (about 9630 K, a bluer light, so the picture comes out **warmer**, as the Kelvin row does), `t = −100` is 50 mired higher (about 4910 K, a cooler picture). The white's place is the CIE daylight locus at `T₁ = 1e6 / m₁`: `x_D(T)` is the CIE polynomial (4000 K to 7000 K: `−4.6070e9/T³ + 2.9678e6/T² + 0.09911e3/T + 0.244063`; 7000 K to 25000 K: `−2.0064e9/T³ + 1.9018e6/T² + 0.24748e3/T + 0.237040`), `y_D = −3·x_D² + 2.870·x_D − 0.275`, and `(u, v) = (4x, 6y) / (−2x + 12y + 3)`. The range stays inside 4900 K to 9700 K, where the daylight locus is the right one (the 4000 K junction of `docs/02` C3–C4 is never crossed).
2. **The step is a displacement.** The moved white is `W₁ = W₀ + (uv_D(T₁) − uv_D(T₀))` with `W₀` = D65's `(u, v)` and `T₀ = 6504 K`, so that `t = 0` gives `W₀` exactly (the locus polynomial reads D65 about 1.2e-4 off in `y`, and that offset is not used).
3. **Tint.** `n` moves the white by `0.0001·n` in `(u, v)` perpendicular to the locus, at `T₁`: `n = ±100` is ±0.01 Duv. The unit normal `ν` is that of the polyline of `uv_D` at `T₁ ± 1 mired`, pointing **towards green** (the side of the daylight locus on which a light is greener, positive CIE Duv). A positive Tint gains **magenta** in the picture, as on the Input, so the white the picture is balanced for is the greener one: `W₂ = W₁ + 0.0001·n·ν`. For `n < 0` the white moves the other way and the picture gains green.

**The gain.** `(u, v)` to `(x, y) = (3u, 2v) / (2u − 8v + 4)`, `XYZ = (x/y, 1, (1 − x − y)/y)`, then `R(W) = M_XYZ→AWG4 · XYZ(W)` with the matrix of `docs/02` Gamuts. The trim's gain is `d = R(W₀) / R(W₂)` per channel (the colour of a light `W₂` seen by a picture balanced for `W₀`, inverted), normalised to green 1: `d ← d / d_g`. It is a von Kries adaptation done in AWG4 itself.

**Properties** (tests, section 7): `t = n = 0` gives `d = [1, 1, 1]` and the step is skipped; `t > 0` raises red and lowers blue (warmer), `t < 0` the reverse; `n > 0` raises red and blue against green (magenta), `n < 0` the reverse; the red-to-blue ratio is monotone in `t` and the red-and-blue-to-green ratio monotone in `n`; `d_g = 1`; `t = +100` moves the blue-to-red ratio by about −0.17 in log2 (the same order as the Kelvin row's 50 mired). The existing white-point functions `ym_color::wb` (`nudged_cct`, `tinted_white`, `AlexaWhite::locus_white`) work on ALEXA native whites and need the photo's scene light, the class mix and the ARRI matrix at the user's CCT; using them would make a node's gain depend on the Input's state (`docs/23` section 14, decision 4). The profile's `locus.xy` table, the same in all 11 shipped profiles, agrees with the polynomial to within 1.7e-4 in `x` and `y` over 4900 K to 9700 K (measured while writing this spec), so the choice does not change what the Kelvin row and the trim mean.

**The combined gain.** `g = 2^ev · d` is one three-vector per node, computed on the CPU in f64 once per node and given to the kernel as three f32 numbers; the kernel does `encode(decode(v) · g)` and no more.

## 3. How masks combine

A node's mask rows are, in list order, `i = 1 … N`. Each has a kind (section 4), a `mode` (add or subtract), `invert`, `enabled` and `opacity`. Rows with `enabled = false` are absent: they are dropped before anything below.

1. **One mask's value.** The kind gives `s_i` in [0, 1] at the pixel (sections 4 and 5). Then `a_i = o_i · (inv_i ? 1 − s_i : s_i)` with `o_i` the opacity.
2. **Regions** are the Object, Subject, People, Background, Linear, Radial and Brush rows; **keys** are the Colour range and Brightness range rows. Each group is folded in list order: `acc ← 1 − (1 − acc)(1 − a)` for an add and `acc ← acc · (1 − a)` for a subtract. The first `acc` is 0, except that when the group's first row is a subtract it is 1 (the group then reads "the whole photo, minus this"). The two accumulators are `R` and `K`.
3. **No group, no mask.** With no region rows `R = 1`; with no key rows `K = 1`; with no rows at all `m = 1`, the whole photo.
4. **The node's mask** is `m = R · K` (the keys keep only their part of what the regions gave), and the node's Invert gives `m ← 1 − m`.

The screen rule `1 − (1 − R)(1 − a)` is the owner's choice for overlapping soft edges (2026-11-01): two half-covered pixels give three quarters, nothing exceeds 1, and the order of adds does not matter. A subtract multiplies by `1 − a`, so it cuts exactly its own coverage. A row that selects nothing (an Object mask with no points, a Subject mask the Mac cannot make) is `s = 0`, an enabled row in the fold: with add it adds nothing, so `m` can be 0 and the node changes nothing.

**Cost and evaluation order.** The rows run in list order in the kernel (`MaskOp` records, `docs/23` section 5.3). Keys read `v_in`; shapes read the pixel's photo-frame position; raster rows read a plane (section 5).

## 4. The shapes and the keys

All positions are fractions of the photo frame (`docs/23` section 3.5): `(px, py)` is a pixel's place in [0, 1]², and the frame is `W × H` photo pixels with long edge `L_photo`. Lengths are fractions of `L_photo`. For shapes the pixel is taken in an isotropic frame, `p = ((px − cx)·W, (py − cy)·H) / L_photo`, so a circle is a circle on any photo.

**The smooth-edge rule.** Every edge, shape or key, is a smooth ramp at least one output pixel wide, also with Feather 0 (`docs/17` K8: no result depends on a rounding that Metal's fast maths may do differently). In the output grid `u_px` is the number of output pixels along the photo frame's long edge (the render's scale; `docs/23` section 3.5), so one output pixel is `1/u_px` in `L_photo` units; `ε_px = 1/u_px`. A shape's ramp is `max(its feather, ε_px)`; a key's is `max(its softness band, KEY_RAMP_MIN)` in code units. Two renders of one photo at different scales therefore differ at an edge by the width of one pixel of the coarser one, which the gate of section 7 states.

### 4.1 Linear

A band across the photo: `center` `c`, `angle_deg` `θ` (clockwise from the horizontal, 0 = a horizontal line), `feather` `f` in [0, 1] (the half-width of the transition is `0.5·f` of `L_photo`). With `n̂ = (−sin θ, cos θ)` (x to the right, y down; for θ = 0 it points down), `s = p · n̂` the signed distance from the centre line and `h = max(0.5·f, ε_px)`:

`s_lin = 1 − S((s + h) / (2h))`

It is 1 above the start line (`s ≤ −h`), 0 below the end line (`s ≥ h`) and 0.5 on the centre line. The mockup's dashed start and end lines are at `s = −h` and `s = +h`; the solid line is `s = 0`. `n = 1 − s_lin` by the row's Invert.

### 4.2 Radial

An ellipse: `center`, `radii` `(rx, ry)` in `L_photo` units (0.002 to 1), `angle_deg` `θ` (the rotation of the `rx` axis, clockwise), `feather` `f` in [0, 1]. With `(u, w)` the pixel `p` rotated by `−θ`, `d = √((u/rx)² + (w/ry)²)` (1 on the outline) and `q = max(f, ε_px / min(rx, ry))` (the ramp, in units of `d`, so it is at least one pixel on the narrower side):

`s_rad = 1 − S((d − (1 − q)) / q)` for `d ≥ 1 − q`; 1 inside `d ≤ 1 − q`; 0 for `d ≥ 1`.

The outline (`d = 1`) is where the mask reaches 0; the feather extends inwards, to the inner ring at `d = 1 − q`. At `f ≥ 1` the mask is a smooth cone from the centre. The outline and ring are what the page draws.

### 4.3 Colour range

A key on chroma, read from the node's input. `center` `c₀ = (a₀, b₀)` in the plane of `e₁, e₂`, `width` `w` in [0.005, 0.3] (the plane the Colour warp draws reaches 0.30), `softness` `σ` in [0, 1]. With `d = |c(v_in) − c₀|` and `q = max(σ·w, KEY_RAMP_MIN)`:

`s_col = 1 − S((d − w) / q)` (1 for `d ≤ w`, 0 for `d ≥ w + q`).

The Colour range reads chroma only: it does not look at tone, so a neutral shadow and a neutral highlight have the same `s`. Picking a colour (`sample_chroma`, `docs/23` section 7.8) sets `c₀` to the picked pixel's `c(v_in)` at the node's input.

### 4.4 Brightness range

A key on tone, read from the node's input. `low` ≤ `high` in [0, 1] (LogC4 code values of the luma `L_in`; mid grey is 0.2784, the top code value 1), `softness` `σ` in [0, 1], `q = max(0.25·σ, KEY_RAMP_MIN)`:

`s_bri = S((L_in − (low − q)) / q) · (1 − S((L_in − high) / q))`

a trapezoid: it rises over `[low − q, low]`, is 1 up to `high` and falls over `[high, high + q]`. A document with `low > high` is read with both set to their mean. Numbers outside [0, 1] are clamped.

`KEY_RAMP_MIN = 0.02` (about a third of a stop in LogC4; a key edge is then at most `1.5/0.02 = 75` steep, so a GPU input that differs from the CPU's by `1e-5` moves `m` by at most `7.5e-4`).

### 4.5 The masks' sizes in the UI

The sliders map to the stored numbers as: Feather and Softness 0 to 100 are 0 to 1; Refine edges and Opacity 0 to 100 % are 0 to 1; Grow −100 to 100 is −1 to 1; Width of a Colour range is `0.005 + 0.295·(x/100)²`; Low and High are the code values times 100. The mapping is in `src/screens/tools/maskScales.ts` and tested against the stored ranges.

## 5. Raster masks

Object, Subject, People, Background and Brush are **raster masks**: a picture made once from the photo's pixels (or the person's strokes), kept in memory, and brought to each render's grid. They need the steps below; the shapes and keys need none.

### 5.1 Step by step

1. **The photo-frame raster `S₀`.** A plane of f32 values in [0, 1] over the photo frame (`docs/23` section 3.5), made at the working size of its kind: Object 1024 px on the long edge (SAM's input, section 6), Subject and People the size of Vision's mask, from a 1024 px picture (section 5.3), Background 512 px (the blur map's cells, section 5.4), Brush 2048 px (section 5.5). The plane is kept per photo in a small in-memory cache (4 to 8 MB each) and never on disk. It does not depend on the node's grade, Feather, Grow, Opacity, Invert or mode, so changing those re-runs steps 2 to 5 only.
2. **Resample to the output grid.** The output pixel `(i, j)` of the render (a viewport frame, a 100 % tile, a thumbnail or an export tile) maps to a photo-frame position through the same geometry the picture uses (the inverse of crop, straightening, the quarter turn and the lens's bend, `docs/01` *Geometry*), and `S₀` is sampled there. Bilinear when `S₀` is as coarse as the grid or coarser; when the grid is coarser than `S₀` by more than 1.25 the sample is the box-filtered bilinear the picture uses when shrinking. Never Catmull-Rom: a mask must not overshoot. The result is clamped to [0, 1]. Where the position lies outside the photo frame (the empty corner of a straightened photo) the value is 0.
3. **Grow.** `g ∈ [−1, 1]`. The radius is `r_g = round(|g| · 0.01 · u_px)` output pixels (`GROW_MAX = 0.01` of the long edge at |Grow| = 100). Positive `g` is a grey-scale dilation, negative an erosion, with a **square** structuring element of half-side `r_g` (a separable maximum or minimum). `r_g = 0` skips the step. The square is a choice for cost and exactness (section 5.6); it is visible only at |Grow| above about 50, and the feather and the guided filter that follow soften it.
4. **Feather.** `f ∈ [0, 1]`. A Gaussian blur of `σ_f = f · 0.02 · u_px` output pixels (`FEATHER_MAX = 0.02` of the long edge at Feather 100), made as three cascaded box filters whose widths are the standard ones for that variance (sum of the three variances `σ_f²`). `σ_f < 0.5` px skips the step.
5. **Refine edges (the guided filter).** `ρ ∈ [0, 1]`, default 0.5 (Brush 0). After He, Sun and Tang (2010), with the Input's luma as guide `G`:
   - `r = round(ρ · 0.008 · u_px)` output pixels (`GF_RADIUS_MAX = 0.008` of the long edge at Refine edges 100) and `ε = 10^(−2 − 3ρ)` (1e-2 at 0, 3.2e-4 at 0.5, 1e-5 at 100 %), on `G` in code values 0 to 1.
   - With `P` the mask from step 4, and the mean over the window of half-side `r` written `⟨·⟩` (window clipped at the frame's edge and divided by the number of samples):

     `μ_G = ⟨G⟩`, `μ_P = ⟨P⟩`, `σ²_G = ⟨G²⟩ − μ_G²`, `cov = ⟨G·P⟩ − μ_G·μ_P`
     `a = cov / (σ²_G + ε)`, `b = μ_P − a·μ_G`
     `q = ⟨a⟩ · G + ⟨b⟩`, clamped to [0, 1].
   - **Refine edges 0** (`r = 0`): the step is skipped and the mask is `P` as step 4 made it. **Refine edges 100**: `r` is 0.8 % of the long edge (48 px on a 6000 px photo at full size) and `ε = 1e-5`, so even edges of a few hundredths of a code value in the guide pull the mask's edge to them: hair, a ridge, a window frame. In between `r` grows linearly and `ε` falls by a decade per third of the range. `r < 1` skips the step. At the sizes of a fit view `r` is a few pixels.
   - The guide `G` is the luma `encode(Y(lin))`, clamped to [0, 1], of **the Input's picture** at the output grid (after C8, before any node: section 5.7), so it does not move when a node is graded. It is made for the tile plus its halo before the colour pass.
6. **Mode, Invert and Opacity** are then applied where the mask is used (section 3, step 1).

The pipeline is run per output grid. A thumbnail of 160 px, a fit view of 1500 px and an export of 6000 px run it at their own `u_px`, with the sizes of steps 3 to 5 scaled to it, so one setting looks the same at every size and the mask's content is the same raster `S₀`.

### 5.2 Receptive field and tiles

Steps 2 to 5 are the only spatial steps of the grade. A tile's output depends on the grid around it by

`H = 2 + r_g + R_f + 2·r + 1` output pixels

(the bilinear taps, the dilation, the feather's three boxes with `R_f` the sum of their half-widths, and the guided filter's two nested windows), and for several masks the largest `H` of the tile's masks. The tile is computed over itself grown by `H`, then cut. All window sums and the box filters use fixed-length direct sums or blocked sums with absolute block origins (`docs/17` D4), and all positions are absolute, so **a tile equals the whole frame bit for bit** (`docs/17` D5). When `H` makes the grown tile larger than the design binding allows (`docs/17` section 4), the pass splits the tile in halves as the other kernels do.

For a very large window the sums are taken on a subsampled grid (the Fast Guided Filter of He and Sun, 2015): above `r = 16` px `a` and `b` are computed at a subsampling `s = ⌈r/16⌉` with a window of at most 16 samples and upsampled bilinearly before they meet `G`. This changes `H` to `2·s·16 + s`, still a function of the settings alone.

### 5.3 Subject and People

On macOS 14 or later, from Apple's Vision framework (`VNGenerateForegroundInstanceMaskRequest` for Subject and `VNGeneratePersonInstanceMaskRequest` for People; `docs/23` section 4.5):

1. The Input's picture, with the Neutral look and no output sharpening (section 5.7), at 1024 px on the long edge, rounded to 8 bits without dither, sRGB, is handed to the request as a `CGImage` in the oriented photo frame.
2. The result is an instance mask. `S₀` is the **union of all its instances** (`allInstances`) at the picture's size, `generateScaledMaskForImage(forInstances:from:)`, converted to f32 in [0, 1]. A photo in which Vision finds no instance gives `S₀ = 0` and the row says "Nothing found."
3. Nothing is trained, tuned or stored from the answers. The same picture on the same Mac gives the same mask (the macOS CI job runs it twice); another macOS version may answer differently (`docs/23` section 12).

### 5.4 Background from blur

From the blur map of lens character (`docs/21` section 7: `ym_ml::bokeh_net` on the lens plane, cached per photo; the classical detector is its fallback). `P(x)` is the probability that the map's cell is out of focus (`P(b ≥ 2·b_min)`), on the cells of the 1024 px plane halved (512 px on the long edge). The row has `level` `λ` in [0, 1] (default 0.5) and `softness` `σ` in [0, 1] (default 0.5):

`S₀ = S((P − (1 − λ) + w/2) / w)`, `w = max(σ, BLUR_RAMP_MIN)`, `BLUR_RAMP_MIN = 0.05`

so `λ = 0.5` puts the edge where the map's own decision falls (`P = 0.5`), a higher level lets more of the photo count as background and a lower one only what is strongly out of focus, and Softness widens the ramp in `P`. The map is the one of the photo's raw group and lens-free picture (it does not move when the lens sliders do: it is made from the raw group, as lens character's, `docs/21` section 3). Nothing is added to the photo where it has no blur: an evenly sharp photo gives `S₀ ≈ 0` (the map says "little out-of-focus background").

### 5.5 Brush

The strokes of the row, in list order, drawn into a plane of 2048 px on the long edge, starting at 0. A stroke has `size` `D` (the brush's diameter, 0.001 to 0.25 of `L_photo`), `hardness` `h` (0 to 1), `erase` and the path `points`. The path is resampled by arc length at a spacing of `0.15·D` (never less than 0.5 plane pixels); a single point is one stamp. A stamp at distance `d` from its centre has weight

`w(d) = 1` for `d ≤ h·D/2`, `1 − S((d − h·D/2) / ((1 − h)·D/2))` up to `D/2`, 0 beyond

(a hard disc at `h = 1`, and at least the plane's one pixel of ramp). A paint stamp makes `S ← 1 − (1 − S)(1 − w)`, an erase stamp `S ← S·(1 − w)`. The plane is made on the CPU in f32 in this order and is the same on every machine. A stroke's `points` are stored simplified (the page drops the points nearer than `D/8` to the line between their neighbours when it ends the stroke, `docs/23` section 7.9).

### 5.6 Choices and their reasons

- **Order of steps 3 to 5.** Grow acts on the mask as the model made it, Feather then softens the outline the way a person expects ("make it softer"), and the guided filter last pulls the result to the picture's edges, so a feathered Object mask with Refine edges up still ends on the object's edge where the picture has one.
- **Box-based filters.** Every filter of the pass is a (cascade of) box filter or a separable maximum, so it costs a few operations a pixel at any radius, and with absolute block origins it is exactly tile-independent (`docs/17` D4). A true Gaussian or disc would cost its radius in taps a pixel at 24 MP.
- **Photo-relative sizes.** Sizes are fractions of the long edge so a thumbnail, a fit view and an export agree (`docs/23` section 3.5).

### 5.7 What the AI masks look at

SAM, Vision and the blur map read **the photo as the Input renders it**: C1 to C8, the Neutral look, the cine rendering (C12) and no output sharpening, at 1024 px (SAM and Vision) or the blur map's own input. No node grade, no look the person chose and no output sharpening reach them, so grading a node, changing the look or moving a slider of another node never moves a mask, and a photo copied to another photo's settings finds its own objects.

## 6. SAM 2.1 small

Object masks come from Meta's SAM 2.1, the **small** size (checkpoint `sam2.1_hiera_small`, 46 million parameters), run by our own pure-Rust CPU runtime `ym_ml::sam2`. The model's code and checkpoints are Apache-2.0 (`docs/14`). Python is used only to convert the weights once and to write fixtures from Meta's own code (the oracle); nothing in the app or `ym` runs Python (`docs/00`).

### 6.1 What we implement

SAM 2 is built for video, and its image mode uses only a part of it. We implement the image path, and nothing of the video path (no memory attention, no memory encoder, no object pointers, no tracking):

- **The image encoder**: Hiera-S (embedding dimension 96, 1 head at the first stage, stages of 1, 2, 11 and 2 blocks, global attention in blocks 7, 10 and 13, windowed attention elsewhere with the windows of Meta's configuration, query pooling by 2 at the start of stages 2 to 4, channel widths 96, 192, 384, 768 at strides 4, 8, 16, 32) and the FPN neck (256 channels, `fpn_top_down_levels` 2 and 3, nearest-neighbour upsampling, `scalp` 1, sine position encoding), as in `sam2.1_hiera_s.yaml`. Its outputs are the image embedding (64 × 64 × 256, stride 16) plus the learned `no_mem_embed` added to it (`directly_add_no_mem_embed`), and the two high-resolution maps for the decoder (256 × 256 × 32 and 128 × 128 × 64, after the 1 × 1 convolutions `conv_s0` and `conv_s1`). The positional embedding of the trunk (7 × 7 background plus the 8 × 8 window embedding, bicubic to the feature size with PyTorch's kernel `a = −0.75`) is computed once when the weights are loaded.
- **The prompt encoder**: positive and negative point embeddings, the two box-corner embeddings, the random-Fourier positional encoding of coordinates (`(x + 0.5)/1024`), and the dense positional encoding of the 64 × 64 grid. No mask prompt is given: **a mask is a function of the points and the box alone**, so settings that hold only the points reproduce it.
- **The mask decoder**: the two-way transformer (depth 2, 256 dimensions, 8 heads, MLP 2048, downsampling rate 2), the output upscaling with the high-resolution features (`use_high_res_features_in_sam`), the hypernetwork MLPs for 3 + 1 mask tokens, the IoU head (sigmoid) and the object-score head (`pred_obj_scores`). A single positive point is decoded with `multimask_output` on and the mask with the highest predicted IoU is taken; two or more points, or a box, with it off (Meta's own advice for unambiguous prompts), with the dynamic fall-back to the best multimask output where the single mask's stability is below 0.98 (`dynamic_multimask_via_stability`, delta 0.05).
- **Not implemented**: memory attention, the memory encoder, the video predictor and its propagation, mask prompts, the mask-input path.

### 6.2 Input and preprocessing

`SAM_INPUT_PX = 1024`. The photo, as in section 5.7 (the Input's picture, Neutral look, 8 bits, sRGB), is resized to 1024 × 1024 **without keeping the aspect ratio** (as Meta's `SAM2Transforms`), with an antialiased bilinear resize, then normalised with mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`. The resize is our own: a separable triangle filter whose support is scaled by the shrink factor, in f32, in a fixed order. Point and box coordinates in the photo frame (`x, y` in [0, 1]) are multiplied by 1024 (`transform_coords` of Meta's code). Meta's `Resize` is torchvision's default on a tensor, which is bilinear with antialiasing in the torchvision versions the fixtures record; the fixtures hold both the 8-bit picture and the tensor Meta's code made from it, so the encoder's parity (section 6.5) is measured from identical tensors and our resize has its own bound.

### 6.3 Output and the raster

The decoder gives low-resolution logits of 256 × 256, clamped to ±32 (Meta's clamp). `S₀` of the Object row is `σ(logit)` upsampled bilinearly to 1024 px on the long edge in the photo frame (the 256 × 256 square is stretched back to the photo's aspect), with `σ(x) = 1/(1 + e^{−x})`. A soft mask goes into section 5.1 as it is. **Nothing found:** the click is refused when the object score is not above 0 or when the mask at logit > 0 covers less than 0.05 % of the photo (`OBJECT_MIN_AREA = 0.0005`); no point is added and no empty mask is kept (`docs/23` section 4.4).

### 6.4 Determinism (as `bokeh_net`)

Everything is f32 in a fixed order, with no fused multiply-add and with `libm` for `exp`, `erf`, `tanh` and `log`, so the same picture and prompt give the same bits on every run, on any number of threads (every output row of every convolution, matrix product and attention is computed by one task whatever the pool), and, as far as IEEE 754 goes, on every platform. Matrix products accumulate the bias first and then the inputs in index order; softmax, layer norm and the mean in a global-context block sum in f64 in row-major order. The weights are read from a safetensors file (f16 stored, widened to f32 on load; `format` and `version` in the header, a SHA-256 checked against `SHA256SUMS`) whose bytes are written deterministically. Window attention and the unfold of the windows follow the order of Meta's code. A model whose header differs from what the runtime implements is refused with an error, never a panic.

### 6.5 The oracle and its gates

- **Fixtures.** `tools/oracle/sam2` (Python, test tooling only, never a dependency of a shipped crate; Meta's `sam2` package at a pinned commit on CPU in f32) writes, for 20 public photos (CC0 and public-domain sets listed in `docs/test-photo-sources.md`; a fixed list with SHA-256), the 8-bit picture, the 1024² tensor, the encoder's three feature maps, and for each of 3 prompts per photo (one click, three clicks of which one negative, one box) the low-resolution logits, the IoU and object scores and the mask. The fixtures are generated data and are not committed when large; `tools/fetch/` makes them.
- **Gates.** Mask IoU of our mask against Meta's at logit > 0, at the 256 × 256 resolution, **at least 0.99** on every photo and prompt; the logits within a stated bound from identical tensors (starting numbers: largest absolute difference 0.05, mean 0.005; the encoder's maps within 2e-3 of their RMS); our resize within 1e-2 of Meta's tensor in the normalised values (mean 1e-3); a second run bit-identical; the runtime's tiles-and-threads independence (1, 2 and 8 threads give equal bits). The encoder's time on the CPU is **measured and recorded** (estimates: 1.5 to 5 s on Apple silicon; the decoder 50 to 150 ms a click) and if the encoder is too slow the fallback is the same encoder on the GPU, still within a stated bound before N-18. A gate that fails is reported, never loosened.

## 7. Tests (`docs/23` section 11 has the gates)

- **Trims.** `ev = 0`, `t = n = 0` are skipped and change no bit; `ev` is `encode(decode(v)·2^ev)` exactly (f64, 1e-13); `d` for the named properties of section 2.2; `t = n = 0` gives `[1, 1, 1]`; the red-to-blue ratio is monotone in `t`, the (red, blue) to green ratio in `n`; `d_g = 1`; `t = +100` and `t = −100` are inverse in the sense `d(t)·d(−t) ≈ [1, 1, 1]` to 5e-3.
- **Node order.** A node with each step on its own equals the same step in the chain it replaced (C9, C9b and C11 of 0.8.2) bit for bit; Main with a curve and Neutral equals 0.8.2; the curve-and-look case differs and is pinned by its own vectors.
- **Combination.** Add and subtract against hand-worked cases (including the lone subtract), screen on overlapping soft edges, the keys' product, Invert per mask and per node, Opacity, a row with the eye off is absent, an empty Object row gives `m = 0`, no rows gives `m = 1`; the fold is independent of the order of two adds.
- **Shapes and keys.** Linear and Radial against an independent f64 evaluation on a grid; the ramp is never narrower than one output pixel at Feather 0, at k = 4, 2 and 1; the value at the centre line is 0.5; keys against the formulas, the plane's colour at the picked pixel gives 1; GPU against CPU within 2e-4 in LogC4 on the blended output for nodes whose effect is at most 0.25 in LogC4 (the starting bound; a larger `KEY_RAMP_MIN` is the remedy if it fails, a decision of the owner).
- **Raster pass.** Resample, grow, feather and the guided filter against independent f64 references on synthetic masks and pictures; Refine edges 0 and Grow 0 and Feather 0 are exact skips; the guided filter on a step edge with a matching guide returns a step to within 1e-3; **tiles equal the whole frame bit for bit** at two tile sizes with halos from `H`; the same mask at k = 4 and k = 1 agrees after the larger grid is reduced by the same box (largest difference 0.1, mean 0.005, starting bounds, set in N-10 and N-13); a photo cropped and turned keeps its mask on the subject (the mask's centroid moves with the subject's within one output pixel).
- **SAM 2.** Section 6.5.
- **Vision.** On a Mac (`mac-gpu.yml`-style job): the two requests give a mask for a synthetic subject and for a rendered person, twice with equal bytes; on an older system or a Mac that cannot run them the probe says no and the kinds are not offered.

## 8. References

- K. He, J. Sun, X. Tang, "Guided Image Filtering", ECCV 2010 (and IEEE TPAMI 35(6), 2013).
- K. He, J. Sun, "Fast Guided Filter", arXiv:1505.00996, 2015.
- N. Ravi et al., "SAM 2: Segment Anything in Images and Videos", arXiv:2408.00714, 2024; code, configurations and checkpoints at `github.com/facebookresearch/sam2` (Apache-2.0).
- Apple, Vision: `VNGenerateForegroundInstanceMaskRequest`, `VNGeneratePersonInstanceMaskRequest`, `VNInstanceMaskObservation` (macOS 14.0 and later), `research/nodes-masks-2026-11-01.md`.
- CIE daylight locus polynomial: CIE 15 (the `x_D(T)`, `y_D(x_D)` of the D series).

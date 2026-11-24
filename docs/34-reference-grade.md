# 34 · Reference Grade

Reference Grade is a node that matches the grade of a **reference picture**. The person gives it a picture, a film still, a favourite photo or another photo of the same shoot, and the node **fits its own grade** so that this photo, seen through the person's real Output (their look and the cine rendering), looks like the reference. It is for the things a photographer does by eye with a picture beside the screen: "make this look like that". This file covers what it does, where it sits, the settings, the new hue curves, the fit, how references are read, the kernel change, `ym`, the panel, the tests and gates, the risks, the questions for the owner, the mock-ups and the tasks.

**Status: specified (RG spec, 2026-11-24); nothing is built. A proof of concept (POC) is being built first so that the owner can see it work (branch `refgrade-poc`, CPU only); its numbers go into §16, and RG-01 adopts them.** The owner's decisions: 2026-11-23, the approach is Phase 1 of the research (fit at run time, no trained model), reference sources are both image files and other photos in the filmstrip, and hue moves use **a new hue-curve field on the node** instead of warp pins (so the colour kernel's node code changes and `diag-alf` must pass before a release). 2026-11-24, "Start the spec. But we want POC of our reference grade approach so I could see how it works before we do full implementation". The research is `research/reference-grade-2026-11-23.md`; **this file wins where they differ** (§15 lists where). The maths is ours: we copy no code and no formula from any tool (`docs/14`).

**In short.**

- A new node kind, `reference_grade`, in the graph of `docs/23`: a numbered node like a corrector, with masks, a key output, a thumbnail and a number. Add node › **Reference Grade**, "Matches the grade of a reference picture." (§9.1). It counts toward the 64 nodes. In a sentence it is "a Reference Grade node".
- **It is a fitted grade, not a model and not a LUT copy.** There is no learned model and no training data. A run-time fit, in f64 on the CPU, chooses about 30 numbers of the node's own fields (a luma curve, three channel curves, saturation and two hue curves) so that the photo, rendered through **the real chain from the node's input to the Output**, matches the reference in display-referred CIELAB (§5). The fitted numbers are ordinary settings. The reference is not saved (§3.7).
- **Global looks only.** It copies the colour and tone of a whole picture. A vignette, a graded sky or a different grade in the shadows of the left half are local looks and are out of scope (§13).
- **A new field: hue curves** (§4). The node has `hue_curves`, two curves, `hue_vs_hue` and `hue_vs_sat`, periodic, on the LogC4 chroma-plane hue axis the parametric look uses. This is a kernel change (§7), made as small as the llvmpipe rule asks.
- **Strength** (0 to 100 %) scales the fitted values from the identity, instantly, with no re-fit (§3.2).
- **Skin protection** (on by default) keeps skin from drifting, using MediaPipe's skin classes, which are deterministic. Never Apple Vision (§5.5).
- **App and `ym` agree to the bit** (§8): `ym render photo.ARW --reference ref.jpg` fits as the app does, and an export from the app pins the fitted numbers in `ym:Settings`.
- **A new node changes no pixel** until it is fitted or edited, and no existing document changes. `PIPELINE_VERSION` takes the next minor step (§3.8).

## 1. What it does

A photographer has a still from a film, or a photo they love, and wants their own photo to feel like it. Resolve has Shot Match, and Photoshop has Match Color (`research/reference-grade-2026-11-23.md`, *Precedents*). Here:

1. Add node › Reference Grade. It does nothing yet.
2. Choose a reference: an image file, or another photo in the filmstrip.
3. Press **Fit**. In a second or two the node's Light, Colour, Curve and Hue values are set. The photo now looks, through the person's own look, like the reference.
4. Lower **Strength** to take part of it. Adjust any tab by hand. Limit it with masks. Export.

### 1.1 What it is, and what it is not

- **It is a node that fits its own grade.** The numbers are the node's corrector fields and its hue curves (§3). After a fit the person can read them, move them and undo them, as for any corrector.
- **It is fitted against the person's real Output.** The look (the Signature look, an imported ARRI look or the person's own LUT) and the cine rendering are part of the chain the fit runs, so the match is what the person sees. A grade matched to a reference on a model of the chain would be wrong on their Mac (§2.2).
- **It is not a learned model.** Nothing is trained, nothing is downloaded, no model runs for the fit but the skin classifier of §5.5 (already shipped). A learned initialiser is Phase 2 of the research and is out of scope (§13).
- **It is not a LUT copy.** The result is a small set of readable controls, not a table. It cannot reproduce what its parameters cannot express (§12).
- **It is not a transfer of pixels.** It never moves a pixel of the reference into the photo. It fits statistics of two pictures that may show different things (§5.2), which is why it copies global looks only.
- **It is not Auto.** Auto resolves the Input's and Main's numbers from one picture and a rule (`docs/19`, `docs/22`); its providers receive one picture and cannot carry a reference (`docs/22` §1 and the research). This node has no Auto and no provider.

### 1.2 How it differs from what exists, and what it reuses

| | What it does | Reference Grade |
|---|---|---|
| **A corrector** (`docs/23`) | The person sets the trims, the grade and the curve by hand | **Reuses the corrector's fields and steps whole** (trims, C9, C9b, C11, the mask blend). The fit writes them. Adds one field, the hue curves (§4) |
| **Colour Uniform** (`docs/32`) | Pulls colours toward one picked colour; has the corrector's tabs after the pull | **Reuses its way of being a corrector-like node with extra fields**, its tabs (Light, Colour and Curve are the same components), its kernel precedent (§7) and its first-release rules. The maths is different: no target, no pull |
| **The parametric look's hue curves** (`docs/08`, `crates/ym-color/src/look.rs`) | `hue_vs_hue` and `hue_vs_sat` baked into a 65³ cube, in the Output slot | **The same axis and the same meaning** (§4.1), now as a node field, evaluated per pixel. The look's curves are not changed and a look still bakes to a cube |
| **The colour warp** (C9b, `docs/02`, `warp.rs`) | Up to 16 Gaussian pins in the chroma plane; radial, not a function of hue angle alone | **Not used by the fit.** The owner chose true hue curves (2026-11-23). The warp stays a tab of the node, as on a corrector |
| **Auto** (`docs/19`, `docs/22`) | Writes numbers into Main and pins them at export | **Reuses the pattern**: fitted numbers are plain settings and `ym:Settings` pins them (§8) |
| **Presets and albums** (`docs/30`, `docs/31`) | A recipe of settings; photos as they are | **Carry the node** like any node (§3.6) |
| **`tools/eval/ab`, `deltae.rs`** | Blind A/B kit; CIEDE2000 | **Reused** for E3 and E1 (§10) |

### 1.3 Photo-to-photo matching

A reference can be another photo in the filmstrip, rendered through that photo's own current settings (§6.4). Matching one photo of a series to another is a strong use. The reference is a **snapshot** of that photo's picture at the moment it was chosen, not a link: if the other photo is graded again later, this node keeps matching the picture it was given (§6.4).

## 2. Where it sits

### 2.1 The node

**A corrector-like numbered node** (Colour Uniform's and Film's precedent). `kind` is `reference_grade`.

| Property | Value |
|---|---|
| Ports | in `picture`, `key`; out `picture`, `key` (as a corrector's, `docs/23` §3.4) |
| Number, thumbnail | numbered in the plan's order; a card with a thumbnail and a number chip (`docs/23` §5.7) |
| Masks | `masks` as a corrector's (`docs/23` §3.3): the Mask tab, Invert, the key input, Show mask. They limit where the grade acts. **The fit ignores them** (§5.1) |
| Order inside the node | trims → C9 → C9b → C11 → **hue curves (new)** → the mask blend (§2.3) |
| Bypass | a bypassed node passes its picture through; its key output still carries `m` |
| Add node | **Reference Grade**, after Colour Uniform and before the separator: "Matches the grade of a reference picture." Adds one node after the selected one, like Colour Uniform, selects it and opens its Reference tab. It does **not** open a file dialog by itself. Off at 64 nodes with "A photo can have up to 64 nodes." |
| Auto | none |
| More menu | a corrector's: Rename · Move earlier · Move later · Link picture to ▸ · Link key to ▸ · Remove link ▸ · Delete node |
| Card | a corrector's card (112 × 84 px at 100 %: thumbnail, number chip, name, footer glyphs); the node's glyph is chosen in RG-07 |
| Simple view | not shown. The notice counts it with the others ("This photo has 3 nodes.", `docs/23` §6), and **Open nodes** goes to the node view |
| Name | the field `name`, default `"Match"` (five letters, never cut on the card), 1 to 32 characters as a corrector's. The menu and the panel's head line say "Reference Grade" |
| Sentence name | "a Reference Grade node": "Node 4 is a Reference Grade node, not a corrector." (`NodeKind::with_article`, `sentence`) |

The node does not belong to Main. Main's Auto, the simple view's controls and Re-run never touch it.

### 2.2 The chain the fit runs, and its dependence on the look

The fit measures the photo **after the real chain**, not the node's own output. The chain, from the node's input, is:

```
node input  →  [the node, with the candidate numbers]  →  every per-pixel step after it  →  the Output: C10 look, C12 cine rendering, C13 encode
```

- **From the node's input.** The part of the plan before the node (the Input, Main, earlier nodes) is rendered **once** per fit, at the analysis size (§5.1), on the CPU reference path as Auto's analysis picture is. Its result is the fixed input of the search. So the fit's input bits are the same in the app and in `ym`.
- **Through the nodes after it and the Output.** Every per-pixel colour step after the node runs for each candidate: correctors, Colour Uniform nodes, mixers, then C10, C12 and C13. An ARRI LogC3-to-Rec.709 look replaces C12 (`docs/02`); the fit follows because it runs the person's actual Output.
- **Spatial nodes after the node are not run** (Film, Blur, Face Refinement): they make their picture before the colour pass and cost far too much to run per candidate. A graph with one after the node shows a note on the card (§9.3). This is a decision of ours, flagged in §14 (question 7).
- **On the CPU reference path, in f64.** The GPU render that follows is held to the parity bounds as for every node (`docs/17` D8).

**The fit depends on the Output look.** A grade matched under one look is wrong under another. So **changing the Output's look, its intensity, or anything that replaces C12 makes the fit stale** (§14, question 5):

- The page keeps, for the session only, a fingerprint of the Output's look settings at the moment of the fit. When it differs, the card's status line says **"The look changed since the fit. Fit again to match it with the new look."** and the Fit button reads "Fit again" with an accent. **Nothing is re-fitted by itself** and no number moves: a re-fit takes a second or two and would replace the person's hand edits (§3.3).
- Other changes (an earlier node's exposure, a node after) do not mark the fit stale. The person can press Fit again at any time.
- After the app restarts there is no fingerprint (nothing about the fit is stored, §3.7), so there is no hint: the values are just settings.

### 2.3 The order inside the node

```
out = v_in + m · ( hue( C11( C9b( C9( trims(v_in) ) ) ) ) − v_in )
```

The corrector's steps keep their order whole (`docs/23` §1.3, `ym_color::display::node`). The hue curves are **step 6, last**, after C11, as the Colour Uniform pull is last in `node_apply` and as the parametric look puts its colour curves after its tone (`docs/08`). Reasons: the fit's tone and channel curves have then done their work and the hue curves refine the colour that is left; and a step that runs last is the shape the kernel's precedent proved safe (§7.2). A fixed order also keeps the fit's search stages simple (§5.4).

## 3. Settings

### 3.1 The format

A new value of `kind` (`docs/23` §3.2): `reference_grade`. The node has these fields and no others; another field is an unknown field and is refused. As a new kind, the node **writes every field** (the convention for a kind, as `docs/23` §3.1 gives for a corrector).

| Field | Type / range | Default | Notes |
|---|---|---|---|
| `name`, `bypass`, `pos`, `masks` | as a corrector's | `"Match"`, false, – , none | |
| `trim` | a corrector's (`docs/23` §3.2) | all 0 | the person's; the fit does not write it |
| `grade` | a corrector's | all manual 0, no pins, the identity curves | **the fit writes `grade.curve` (luma, red, green, blue) and `grade.saturation`**; the other fields are the person's (§3.3). Auto reads as a manual 0, silently, as on Colour Uniform |
| `hue_curves` | `{ hue_vs_hue, hue_vs_sat }` (§4.1) | both empty (the identity) | the fit writes both |
| `strength` | 0 to 100 (percent) | 100 | scales what the fit writes (§3.2). The number is in the units the panel shows, like a look's `look_intensity` (`docs/10`) |
| `skin_protection` | bool | `true` | read by the next fit only; it changes no pixel by itself (§5.5) |

There is no `reference`, no file name, no statistics and no fit status in the file (§3.7).

```json
{ "id": 4, "kind": "reference_grade", "name": "Match", "bypass": false, "pos": [2, 0],
  "trim": { "exposure_ev": 0, "temperature": 0, "tint": 0 },
  "grade": {
    "contrast": { "kind": "manual", "value": 0 }, "brightness": 0,
    "highlights": { "kind": "manual", "value": 0 }, "shadows": { "kind": "manual", "value": 0 },
    "white": 0, "black": { "kind": "manual", "value": 0 },
    "color_balance": { "shadows": { "kind": "manual", "point": [0, 0] }, "midtones": [0, 0],
                       "highlights": { "kind": "manual", "point": [0, 0] } },
    "saturation": { "kind": "manual", "value": -12 },
    "color_warp": { "pins": [] },
    "curve": { "luma": [[0, 0], [0.2136, 0.2050], [0.2784, 0.2784], [0.4080, 0.4310], [1, 1]],
               "red": [[0, 0], [0.2784, 0.2840], [1, 1]],
               "green": [[0, 0], [0.2784, 0.2784], [1, 1]],
               "blue": [[0, 0], [0.2784, 0.2700], [1, 1]] } },
  "hue_curves": {
    "hue_vs_hue": [ { "hue_deg": 0, "value": -4 }, { "hue_deg": 120, "value": 6 }, { "hue_deg": 240, "value": 0 } ],
    "hue_vs_sat": [ { "hue_deg": 0, "value": 1.05 }, { "hue_deg": 120, "value": 0.85 }, { "hue_deg": 240, "value": 1 } ] },
  "strength": 100, "skin_protection": true,
  "masks": { "invert": false, "items": [] } }
```

The example is an illustration with fewer knots than a fit writes; it is not a result. The knots' positions are the stored numbers; the fit's own knot table is in §16.1.

### 3.2 Strength

`strength` scales **what the fit writes**, from the identity, at render time and with no re-fit. For each **fitted field** (`grade.curve.luma`, `.red`, `.green`, `.blue`; `grade.saturation`; `hue_curves.hue_vs_hue`; `hue_curves.hue_vs_sat`) and `k = strength / 100`, the render uses the **effective** values:

| Field | Effective value |
|---|---|
| a curve's knot `(x, y)` | `(x, x + k (y − x))` |
| `saturation`, manual `s` | `k s` (saturation is linear in `s`: `λ = 1 + s/100`, `docs/02` C9) |
| `hue_vs_hue` `δ` (degrees) | `k δ` |
| `hue_vs_sat` `σ` (a chroma multiplier) | `1 + k (σ − 1)` |

- **Monotone stays monotone.** A knotwise mix of two increasing curves is increasing, so a curve at any Strength is a valid curve.
- **Strength 0 is the identity**, bit for bit, when the node's other fields are at their defaults. **Strength 100 is the stored values.**
- The mix is made in f64 in one function, `ReferenceGradeNode::effective`, which the CPU reference, the cache key and the device's host packing all call. **So the kernel never sees Strength** and needs nothing for it.
- Strength does **not** scale `trim` or the grade fields the fit does not write (Contrast, Brightness, the zones, Colour balance, the warp). They are the person's own numbers (§3.3).
- A hand edit of a fitted field is scaled by Strength like the fit's own value. The tabs show the **stored** numbers; at a Strength below 100 the picture shows less than the numbers say. The Reference card says so in a line under the slider (§9.3).

### 3.3 What happens when the person edits a fitted value

**The rule: a fitted value is an ordinary setting. An edit changes that stored value and nothing else.** Strength still scales it. A fit never reads the old numbers: **Fit starts again from the identity** for the fitted fields, so its result depends only on the photo, the reference, the node's other fields and the chain, never on what was there.

- **Fit (and Fit again) overwrites the fitted fields** (§3.2's list) in one patch, one undo step. It leaves everything else alone: `trim`, the other grade fields, the masks, the name, `strength` and `skin_protection`.
- **An edit does not re-fit and does not mark the look stale.** The page keeps a session flag "edited since the fit"; the status line then says "Edited by hand since the fit. Fit again to start over." The flag is cleared by the next fit and is not stored.
- **Undo** restores the earlier numbers, edit or fit.
- **The other fields act as set.** The fit runs the node with its non-fitted fields as they are, so they are part of what the fit sees (a Contrast of +10 the person set earlier is in the chain the fit matches through). A later change of those fields is an ordinary edit.

This is the simplest rule that is consistent: one set of numbers, one dial over the fit's part, no hidden second copy of the "fitted" numbers. It is flagged as question 3 in §14.

### 3.4 Validation and clamping

- Values out of range are **clamped and reported** by path (`4.strength`, `4.hue_curves.hue_vs_sat[1].value`), as `docs/23` §3.5; a value that is **not finite** is `SettingsError::NotFinite` and nothing is applied. A `hue_curves` that is not an object, or a point with a field of another type, is `SettingsError::Parse`.
- Unknown fields are refused: **"A Reference Grade node has no field called “x”."** `trim`, `grade` and the rules of `docs/23` §3.5 hold: ports exist on their kind, an input port has one link, no cycles, unique ids, **at most 64 numbered nodes** counting this one.
- **Hue curve points** (§4.1): at most 16 per curve; `hue_deg` any finite number, read modulo 360 into [0, 360) (**not reported**); two points at the same hue after the reduction are refused ("Two points of a hue curve are at the same hue."); `hue_vs_hue` `value` ±180 and `hue_vs_sat` `value` 0 to 8 (clamped, reported, the parametric look's limits). The list is sorted by hue on read.
- `strength` 0 to 100 is clamped and reported. `skin_protection` must be a bool (a parse error otherwise).

### 3.5 Patches, Reset, Copy and Paste, undo

- **Patches** (`docs/23` §3.7). A node is added by giving its `kind` in `nodes` and removed with `remove`; its links go with it. `SettingsField` gains `HueCurves`, `Strength` and `SkinProtection` (the nodes' group; a change runs the colour pass again and nothing else). The node takes `Trim` and the grade fields as Colour Uniform does (`of_graded_field`, `apply_graded`, `graded_same`, `docs/32` §12a.3): **one place** turns a trim or a grade field into a patch. A patch carries `hue_curves` partially: a curve given replaces that curve, a curve left out stays.
- **The fit is one patch**: `grade.curve`, `grade.saturation` and both hue curves, in one undo step. A slider drag folds into one undo step as every slider.
- **Reset** (`SettingsPatch::reset_node_in`): the node's own fields (`trim`, `grade`, `hue_curves`) to their defaults, `strength` to 100 and `skin_protection` to true; name, bypass, masks and links stay. Each card keeps its own Reset (§9).
- **Copy settings and Paste** carry the node like any node. Undo diffs per node.
- **Resolved settings, XMP and DNG.** The node has no Auto, so it is in `ResolvedSettings.settings` as it is, and `ResolvedSettings.values.nodes` has an entry with its trim and grade (as for Colour Uniform, `resolvedGraded`). `ym:Settings` holds the node with its fitted numbers; the DNG bakes it into the pixels like every node (`docs/09`).
- **The JSON Schema** (`crates/ym-core/schema/settings.schema.json`) gains `reference_grade_node` (with `trim` and `grade` as `$ref`s to the corrector's definitions, and `hue_curves`, `hue_point`) and the kind in the `kind` enum. The wire types gain `NodeKind::ReferenceGrade` and `NodeItem::ReferenceGradeNode`; `bindings.ts` is regenerated and the field-name test compares them.

### 3.6 Presets and albums

- **Presets** (`docs/30`). The node is in the **`nodes` group**, which holds every node other than the Input, Main and the Output; no new group. A preset saved from a photo that has the node holds the fitted numbers, `strength`, `skin_protection` and the masks. **Applying it to another photo applies the same numbers; it does not re-fit.** A preset is therefore a recipe of a fixed grade, as a Colour range's centre and a warp's pins are fixed. The person can open the node and Fit again with a reference. The preset table text of `docs/30` §2 gains "Reference Grade nodes" in the list of kinds, and the test that a field in no group fails (`docs/30` §14) covers the new fields. Nothing in the node is a stroke or a click, so `docs/30` §3 (what a preset never holds) is unchanged.
- **Albums** (`docs/31`). An album holds the photo's whole graph as stored, so the node travels in `settings` with no change to the file (version 1 stays). An older build refuses it with "This album was made by a newer version of Yellowmoon. Update to open it." `written_by.pipeline` records the version (§3.8).
- **Older documents** read as they did. A document that holds the node is refused by an older Yellowmoon with the plain unknown-kind sentence (`docs/23` §3.5).

### 3.7 What is saved, and what is never saved

By rule 4 nothing is kept per photo. For this node:

| Saved, as ordinary settings | Never saved |
|---|---|
| the fitted numbers (curves, saturation, hue curves), `strength`, `skin_protection`, the masks | **the reference picture**, in any form; its analysis picture; **any statistic computed from it**; the fit's objective, seed or status; the Output-look fingerprint of §2.2; the "edited since the fit" flag |
| | **a label of the reference** (file name, path, photo name) |

**Recommended: no label is stored** (§14, question 4). Reasons: (1) `ym:Settings` goes out inside every exported file, and a file name is information about the person's library, and for a film still a hint of what was copied; (2) a path says nothing on another Mac and is private; (3) a label has no effect on any pixel, and a setting that does nothing is noise in documents, presets and albums; (4) it would go stale at the first hand edit. The card shows the reference's name **on screen, for the session only**: the file's name, or "Photo 3 of 7" for a strip photo.

### 3.8 `PIPELINE_VERSION`

No existing graph changes by a bit (a node that is not there runs nothing; §7.3 keeps the kernel's text for every other node identical), so **no golden entry moves**. An older build refuses a graph with a kind it does not know, so `PIPELINE_VERSION` takes **the next minor step** with the node, the rule `docs/01` *pipeline_version* and `docs/23` set when a kind lands. Today's is 0.16.0, so **0.17.0**, set by RG-10 and written to `ym:PipelineVersion`, `ym --version`, `ym doctor`, `ym info` and an album's `written_by` (the snapshots and the tests that assert the number follow). If another change takes 0.17.0 first, the next free minor number is used; the rule does not depend on the value. The golden set gains **one** entry (§10).

## 4. The hue curves

### 4.1 The field

`hue_curves` has two curves, with the parametric look's meaning (`docs/08`, `crates/ym-color/src/look.rs`):

- **`hue_vs_hue`**: a point `{ hue_deg, value }`; `value` is **degrees** (±180) the hue turns; positive turns red towards yellow.
- **`hue_vs_sat`**: a point `{ hue_deg, value }`; `value` multiplies the chroma (0 to 8; 1 is unchanged, 0 is grey).

An empty list is the identity. **One point makes a constant.** At most 16 points. The curve between points is a **monotone cubic (Fritsch–Carlson) spline, periodic with period 360°**: the segment after the last point runs on to the first plus 360°, and every point has two neighbours, so the curve is C1 across 0 and 360 and never overshoots the values given. This is `ym_color::curve`'s `fritsch_carlson_tangents_periodic` and the look's `Spline::new_periodic`, reused.

**The axis** is the LogC4 chroma plane the parametric look, the colour warp, the Colour range and Colour Uniform already use (`docs/32` §3.1):

```
p = v · e1,   q = v · e2         e1 = (1, −½, −½)/√1.5     e2 = (0, 1, −1)/√2     (grade::{E1, E2})
r = hypot(p, q)                  chroma
h = atan2(q, p)                  hue: 0 red, 60 yellow, 120 green, 180 cyan, 240 blue, 300 magenta
m = (v_R + v_G + v_B)/3          the tone, along (1,1,1)
```

### 4.2 The step

```
δ(h)  = hue_vs_hue(h)  in radians          σ(h) = hue_vs_sat(h)
(p', q') = σ(h) · R(δ(h)) · (p, q)         R(δ) the rotation by δ
v' = v + (p' − p) e1 + (q' − q) e2
```

- **The tone `m` is exact.** The basis is orthonormal and `e1`, `e2` are orthogonal to `(1,1,1)`, so `v'` has the same mean as `v`: the step moves hue and chroma only. **It does not restore Rec.709 luma** as the parametric look does (that needs a decode and an encode per pixel; `docs/32` §3.1 point 3 argues the same). A hue move therefore changes luma a little, and the fit sees that, because it runs the real chain (§5); the luma curve covers the rest. Flagged: question 13.
- **No neutral guard is needed.** The step scales and rotates the vector `(p, q)`, so as `r → 0` the move tends to 0: it is continuous at grey and exact at `r = 0`. A pixel with `r = 0` is returned as it is.
- **The identity is exact.** A node with both curves empty (or Strength 0) runs no hue step (`N_HUE` is not set; §7).
- **The curves are evaluated from baked tables**, one of 1024 f64 samples around the circle for each curve, built from the spline in f64 by one function, with linear interpolation between samples and wrap-around. **The table is the definition**: the CPU reference reads the same table in f64 and the device reads it as f32, so the baking error is not a parity term. The error of the table against the exact spline is measured in RG-03 and stated in `docs/STATUS.md` (§10).

### 4.3 Where else

Only the Reference Grade node has `hue_curves` in this version. Correctors and Colour Uniform do not; giving them the field is a later idea (§13), the same work in another kind.

## 5. The fit

**This section's structure is fixed; its numbers come from the POC.** Every constant, weight, count and bound marked *"to be set from the POC (section X)"* is chosen by RG-01 from §16 and written there before RG-02 starts. Nothing in §5 is a measured result.

### 5.1 Pictures

| | Photo | Reference |
|---|---|---|
| Source | the node's input rendered by the plan (§2.2), at the analysis size | the loaded reference (§6) at the analysis size |
| Size | long edge *to be set from the POC (§16.2)*; the research's budget is **about 8 to 16 thousand stratified pixels** for the search | the same family of size |
| Reduction | the deterministic area average of §6.5 | the same |
| Sampling | a fixed stratified grid of pixels, in a fixed order, from the photo's analysis picture; each candidate renders only these | all pixels of the analysis picture, or a fixed stratified sample of the same count (§16.2) |
| Colour | the display codes of the chain's output (Display P3, the viewport's space), converted to CIELAB D50 with known primaries | the reference's own colour space (§6.3) converted to CIELAB D50 |

- **The photo's chain-before-the-node is rendered once**; only the node and the steps after it run per candidate (§2.2).
- **The fit ignores the node's masks.** A reference describes a whole picture, so the grade is fitted as if the mask were 1 everywhere; the masks then limit where it acts. (Fitting inside the masks would need the reference's matching region; out of scope.) Question 8.
- **The statistics are of distributions, not of pixel pairs**: the two pictures may show different things (§12).

### 5.2 The parameters

About 30 numbers, mapped to the node's fields (§3.1). **This list is a proposal; the POC sets the final one (§16.1).**

| Group | Field | Values (starting) |
|---|---|---|
| Tone | `grade.curve.luma`: knots at **fixed scene stops** about middle grey (x from the stop table of §16.1; end points fixed) | 8 |
| Colour balance | `grade.curve.red`, `.green`, `.blue`: 3 knots each (shadows, mid, highlights) | 9 |
| Saturation | `grade.saturation` | 1 |
| Hue | `hue_curves.hue_vs_hue` and `.hue_vs_sat`: 6 knots each at fixed hues 0°, 60°, … 300° | 12 |

Total 30. Each parameter has a **range** *(to be set from the POC, §16.1)*, a scale for the regulariser and an identity value. Luma and channel curve knots are kept strictly increasing in `x` and monotone in `y` by construction (the search moves their increments, not their values).

### 5.3 The objective

The fit minimises a sum of terms over CIELAB D50 values, `L*`, `a*`, `b*`, with `C* = hypot(a*, b*)` and `h*` its angle. **The terms are fixed here; their weights, quantile levels, band edges and sector counts are *to be set from the POC (§16.3)*.**

| # | Term | What it measures |
|---|---|---|
| T1 | **`L*` quantile shape, level weak** | at fixed quantile levels `q_k`, `d_k = Q_photo(q_k) − Q_ref(q_k)`; with `d̄` their mean, the term is `mean_k (d_k − d̄)² + w_level d̄²`. The **shape** of the tone curve is matched strongly and its **level** weakly (`w_level` small), so a night reference does not darken a beach (§12) |
| T2 | **Band means** | the mean `a*` and `b*` in fixed `L*` bands, each band weighted by the smaller of its two shares, so a band one picture lacks counts for little |
| T3 | **Chroma per hue sector** | the mean `C*` in fixed `h*` sectors, weighted the same way |
| T4 | **Identity regulariser** | `Σ_i ((θ_i − θ_i^id) / s_i)²`: the fit prefers the smallest grade that matches |
| T5 | **Clip penalty** | the excess of the photo's out-of-gamut or clipped share over the reference's own share (`max(0, clip_photo − clip_ref)²`), so the fit does not buy a match with clipped colour |
| T6 | **Skin protection** (only when `skin_protection` is on and skin is found) | on the photo's skin pixels (§5.5): the penalty for a hue turn or a chroma ratio beyond limits, **measured against the same pixels at the identity fit**, not against the reference |

An optional **sliced Wasserstein distance** over a seeded fixed set of projections (the research's list) is **not** in the first set of terms; the POC says whether it is needed (§16.3).

### 5.4 The optimiser

- **Staged, seeded, with fixed iteration counts, no restarts that depend on the data.**
  0. **Closed form.** The luma knots from the quantile map of T1; the channel curves from the band means of T2; saturation from the median chroma ratio; the hue curves at the identity.
  1. **Tone.** A derivative-free refinement of the luma knots (and, jointly, the level).
  2. **Colour.** The channel curves and saturation.
  3. **Hue.** The two hue curves.
  4. **Joint.** All parameters with a smaller step schedule.
- The method inside a stage (coordinate search with step halving, a seeded simplex, or another), the step schedules, the iteration counts and the seed are *to be set from the POC (§16.4)*. The chain is run forward, so **nothing needs to be differentiable**.
- **A result that is no better than the identity is the identity**: if the final objective is not below the identity's, the fit writes the identity and the status says "The photo is already close to the reference. Nothing was changed." (§9.3). That is not a failure.
- **A non-finite value, a classifier error that cannot be recovered, or a search that cannot evaluate the chain is a failure** and writes nothing (§9.3).

### 5.5 Skin protection

- **Source: MediaPipe, never Apple Vision.** Apple Vision is not reproducible across Macs (`docs/23` §4.5; the research). The skin comes from **Selfie Multiclass**, the bundled MediaPipe segmenter in `ym_ml::mediapipe` (the same one Face Refinement uses, behind the `mediapipe` feature of `ym-retouch`), whose classes `BodySkin` (2) and `FaceSkin` (3) (`crates/ym-ml/src/mediapipe/selfie.rs`) give the skin plane. It runs in our pure-Rust CPU runtime, deterministic, on the photo's analysis picture, **once per fit**.
- **What it limits.** T6 only: it limits how far the fitted grade may turn the hue of, and scale the chroma of, the pixels in the plane. It does not mask the grade: the node's grade is global, and skin gets the grade the rest gets, up to the limits *(thresholds to be set from the POC, §16.3, from public data only)*.
- **No skin found, or the classifier absent** (a build without the feature): the term is off and the status says "No skin was found, so Skin protection did nothing." (found nothing) or is silent when the switch is off.
- Skin protection is **read at fit time**: changing the switch changes nothing until the next Fit (§9.3).

### 5.6 Determinism and time

- **CPU, f64, a fixed order of operations, no data-dependent threading**: the evaluation of a candidate splits pixels into fixed chunks reduced in a fixed order, so 1 and N threads give the same bits (as the CPU chain's tiles do). The skin plane is deterministic. The same photo, reference, settings and version give the same fitted numbers on every run (rule 8).
- **Time: the target is 1 to 2 s on an M1** (the research's unmeasured target). It is **not measured**; the Mac check is RG-H3 (`M0-H4`). The Linux figure from the POC and from E1 is reported as a Linux figure (§16.7). If the target is missed, the order of remedies is: fewer sample pixels, fewer iterations, cutting a stage; the quality gate is re-run after each (§10).

## 6. References

### 6.1 Image files

- **Formats: JPEG, PNG, TIFF**, 8 or 16 bits per sample, RGB or grey (a grey picture matches as black and white; the status line says so). Decoders: `zune-jpeg`, `png` and `tiff`, already in the workspace.
- **No HEIC.** libheif is LGPL (`docs/14`, rule 1). HEIC and every other format get the "not supported" error (§9.3), which says to save as JPEG or PNG first. No RAW file is accepted as a reference: a raw's picture depends on its settings; use a photo in the strip, or export it first.
- **Orientation is ignored.** The fit uses distributions, which a rotation does not change.
- **A file is only read.** Rule 5 holds: nothing is written, moved or renamed. The file is read once, reduced and let go of (§6.5).

### 6.2 What a reference is read as

An image gives a grid of display-referred code values in its own colour space. §6.3 turns them into CIELAB D50; §6.5 reduces them. **The reference is used at run time and never stored, uploaded or cached** (§6.6).

### 6.3 Colour handling

1. **Untagged files are taken as sRGB** (the sRGB curve and primaries, IEC 61966-2.1). This is a guess and is listed as a risk (§12): a wrongly tagged or untagged picture shifts the whole fit.
2. **Tags that are read**, in this order: an embedded ICC profile (JPEG `APP2`, PNG `iCCP`, TIFF tag 34675); PNG `cICP` (the code points our exports write: sRGB 1/13, Display P3 12/13, Rec.2020; `docs/09`); PNG `sRGB`. The others (`gAMA`, `cHRM`) are not read.
3. **Profiles that are handled: RGB matrix/TRC profiles**: the colorant tags `rXYZ`, `gXYZ`, `bXYZ` (already in the profile connection space, D50) and the curves `rTRC`, `gTRC`, `bTRC` as `curv` (the identity, a gamma or a table) or `para` (types 0 to 4). That covers sRGB, Display P3, Adobe RGB, Rec.2020 and **the "Display" profile macOS puts in screenshots** (the research set had one; RG-05 adds it, and the others, as fixtures). **A profile with a LUT (`A2B0`), a CMYK or a named-colour profile is not handled.**
4. **A profile that cannot be read** (damaged, or not a matrix/TRC kind) is **not an error**: the file is taken as sRGB and the card says so ("The picture's colour profile couldn't be read, so it is taken as sRGB.", §9.3). Stopping would block a picture that is usually fine, and the person sees the note.
5. **The conversion** is in f64, in a fixed order: code values → linear by the curves → XYZ D50 by the matrix → CIELAB D50. No chromatic adaptation is needed for an ICC profile (its colorants are D50); for an untagged or `cICP` picture the primaries' D65 white is adapted to D50 by the Bradford matrix.
6. **A 16-bit picture is read at its full precision**; 8-bit pictures are read as they are (no dithering, no smoothing).

**Decision: our own minimal ICC reader, not a crate** (§14, question 10). The reader is a few hundred lines in `ym-color` (the tag table, `XYZ `, `curv`, `para`), in f64 and a fixed order. Why: (1) **determinism**: a general colour-management crate may choose SIMD or f32 paths by CPU, and the fit's bits must not depend on them (rule 8); (2) **scope**: we need exactly matrix/TRC, and refuse the rest; (3) **no new dependency** for `cargo deny` and `docs/14`; (4) our own export already writes such profiles (`ym-export` `icc.rs`: sRGB, Display P3, Rec.2020), so the layout is known and the reader is tested against them. `moxcms` is listed in `docs/14` among the permitted dependencies (MIT/Apache-2.0/BSD) but is **not in the workspace manifest today**; if the owner prefers a crate, it is the candidate, with the determinism check as an acceptance test. The **oracle** for the reader's tests is an independent tool in CI only (RG-05 picks: `moxcms` as a dev dependency, or littlecms' command-line tool), never a dependency of a shipped crate.

### 6.4 A photo in the filmstrip

- **It is rendered through its own current settings** to a display picture at the analysis size: the same chain, the same Output, the same C13 space the photo of §5.1 uses, so no profile is involved. The result is a snapshot (§1.3).
- **The picture is held in the core, in memory, for the session**; it is the reduced picture, not the file and not a link. The page asks the core to set it and gets back a thumbnail and a short label.
- **Rules.** The photo being graded cannot be its own reference ("That's the photo you're grading. Choose another."). A photo that is still developing, pending or failed cannot be chosen (its tile is disabled with "This photo isn't ready yet."). The chooser is a mode like Pick colour: Esc ends it.

### 6.5 Letterbox and size

- **Letterbox removal: uniform bars only.** Rows and columns are examined **from each edge inward**. A line (row or column) is part of a bar when its **standard deviation is very low and its mean is low**; the bar is the run of such lines from the edge and ends at the first line that fails. Both tests are on display luma, with thresholds *to be set from the POC (§16.5)*.
  - **A dark row is not a bar.** Night frames have many dark rows that are picture. A "dark rows are bars" rule would trim them, so the rule needs **both** a standard deviation below the noise of a real picture **and** a mean near black, with a minimum bar width and a **step** at the bar's inner edge (the first lines inside differ from the bar by more than noise). A dark sky with sensor or compression noise has a standard deviation well above a flat bar.
  - **Ringing.** JPEG ringing at the bar's edge is skipped by taking a margin of a few pixels inside the bar *(constants: §16.5)*.
  - **Limits.** A bar is never more than a fraction of the picture's side *(§16.5)*; a picture in which every line qualifies has no bars removed and then fails the "too small" or "empty" check below.
  - The card says "Black bars were removed from the reference." when it removed any.
- **Size.** After removal, the picture's **short side must be at least a minimum** *(to be set from the POC, §16.6)* or the reference is refused ("too small", §9.3). A picture with no area inside its bars is refused ("empty", §9.3).
- **Reduction to the analysis size (§5.1).** A deterministic area average (a box filter with exact weights) **in linear light**, computed in f64, to the analysis size; a smaller picture is not enlarged. The long edge is *to be set from the POC (§16.2)*.

### 6.6 Copyright and what is never kept

References are supplied by the person, used at run time and never stored: **not in the settings, not in a preset or an album, not in a log, not in the XMP, not in a cache.** The core holds the reduced analysis picture in memory for the session and drops it when the node is deleted, the photo is closed or the app quits. A film still is usually copyrighted; Yellowmoon reads it as the person's own tool does, and keeps no copy. **Evaluation stills are local only** (§10): never committed, never cached by CI (`docs/14`, *Photography Life and Imaging Resource sample raws*).

## 7. Rendering

### 7.1 The plan and the CPU reference

- **The plan** (`docs/23` §5.2). The node is a corrector-like per-pixel step with a picture input, a key input and a key output. It fuses into the colour pass like a corrector's grade. `StepKind` gains `ReferenceGrade` (`ym_color::plan`, `pipeline::schedule::step_kind`; their exhaustive matches fail to compile without it): per pixel, reads the picture, no planes. A step at its identity (every step at its identity after `effective`, §3.2) is skipped.
- **The CPU reference** is `ColorChain::run_nodes` calling `PreparedNode::apply` (`ym_color::display::node`): the corrector's steps in their order from `effective` values, then `hue::apply` in f64 (§4.2), then the node's mask blend. `color_plan` fills `trim`, `grade`, `warp` and `curves` through `grade_params` as Colour Uniform's does, plus the baked hue tables. `nodes_key` (`pipeline/color.rs`) adds the node's stored values and `strength` to the key, or an edit would not invalidate the frame.
- **Show mask, thumbnails, histogram, scatter, pixel readout, ancestors, dead nodes**: as a corrector's. The node's thumbnail is the picture after it.

### 7.2 The device: the least change

The colour kernel binds all 8 storage buffers (K3: 8 storage buffers and 1 uniform; `docs/17`). **No binding is added.** The llvmpipe rule (`docs/17`, *A lavapipe miscompile on AMD Zen 4 and 5*, `docs/STATUS.md`: the 2026-11-20 and 2026-11-21 miscompiles of `run_nodes` after small control-flow changes) applies in full: **change `run_nodes`, `node_apply` and `node_mask` as little as possible, prefer host packing, and pass `diag-alf` before a release.** The design:

1. **The hue tables are packed by the host into binding 2**, the curve-table buffer (K3 is full, so nothing new is bound). Today binding 2 holds, in plan order, four 1024-sample `f32` tables for every node that runs a curve, and a record names its first table. **After all of the curve tables**, the host appends **two tables of 1024 `f32`** (`hue_vs_hue` in radians, `hue_vs_sat` as a multiplier) for each live Reference Grade node whose hue curves are not the identity, in plan order. Because they come last, **no existing node's table offset moves and the buffer's bytes for every existing graph are unchanged.** `ChainTables::prepare` writes them from the f64 tables of §4.2, rounded once to `f32`.
2. **One record flag and one row.** A new bit of `ctl.x`, **`N_HUE`**, set by the host for a live node with a non-identity hue step. The first hue table's index is written to a row the node's kind never uses: **`R_HUE` = `R_JOIN + J_IN`, lane x** (the mixers' rows; a Reference Grade node is neither a mixer nor a Colour Uniform node, and `docs/32` §12a.4 already moved the pull's rows there). The constant is written by the host from `offset_of!(NodeRecU, join)`; **the record stays 1328 bytes**, `REC_ROWS` is unchanged, and a test pins the rows and the size.
3. **The fewest new branches.** The kernel gains **one `if` on `N_HUE`** and, inside the new function, **no branch on the data**: the hue is read as `h = select(0.0, atan2(q, p), r > 0.0)` (a `select`, the shape that fixed the Blur node's miscompile), so the step has no `r == 0` branch. The step is: `p`, `q`, `r`; `h`; two table reads with linear interpolation and wrap-around; one `sincos`; the rotation and scale; the add back along `e1` and `e2`. It resembles the Colour Uniform pull (one `atan2`, one `sincos`, one square root; `docs/32` §9), without `exp2` or `log2`.
4. **The shape, in order of preference** (each tried on the same runners; the first that passes is built):
   - **H1 (recommended): an early return, the shape of W1** (`docs/32` §12a.4, `docs/17`). Below the `N_SKIP` return in `node_apply`: `if ((bits & N_HUE) != 0u) { return rg_apply(ni, v_in, bits); }`, and `rg_apply` is a **copy** of `cu_graded_apply` with the pull replaced by the hue step at the end: trims, C9, C9b and C11, then `hue_node`. **The body of `node_apply` below the early return, `run_nodes`, `node_mask` and the mask code are byte-identical to main**, so every other node runs the code it did. This is the shape that passed on all four runners for the graded Colour Uniform node; it costs a second copy of the four steps in WGSL, kept honest by the parity test.
   - **H2: a block last in `node_apply`**, after C11, in the shape of `uniform_node`'s block, guarded by `(bits & N_HUE) != 0u`. Smaller, but it changes `node_apply` itself.
   - **H3: refuse on the device.** If neither passes, the device refuses a Reference Grade node with non-identity hue curves (`InvalidInput`, which the render takes as it takes any refusal: the CPU makes the frame and says why, `RenderReport::cpu_reason`). The node, the fit and the Hue tab stay; the owner decides whether a slow node is acceptable. The remedy order is that of the 2026-11-21 round.

### 7.3 Parity and the release gate

- **A CPU f64 reference and a parity test** (`crates/ym-gpu/tests/gpu_reference_grade.rs`): the device against `PreparedNode::apply` at **2e-4 in LogC4** (the bound of `docs/02`), and one 10-bit code after C12 and C13; the identity cases (empty curves, Strength 0, a node with no hue step) **exact**. Cases, on lavapipe and on the Mac GPU CI (Metal, fast-math, K8): `hue_vs_hue` alone, `hue_vs_sat` alone, both, with trims, grade and curves, a pixel at `r = 0` and near grey, colours at 0°, 60° and 359.99° (the seam), negative codes, a node after a node, a node with a mask. The measured worst error is written into `docs/STATUS.md`; none is claimed here.
- **K8 (Metal's fast-math).** Nothing is decided by a rounding that changes the result beyond the gate: the `select` on `r > 0` chooses between results that agree to rounding near `r = 0`, because the step scales and rotates a vector of length `r`. The flag, the row and the table index are integers written by the host.
- **`diag-alf` must pass before any release that contains the change.** RG-04 runs the workflow on the four GitHub runner types (AMD EPYC 9V74 and 9V45, EPYC 7763 and the Xeon Platinum 8573C), comparing `main`'s commit with the RG-04 commit, counting the failures of `gpu_alf_look`, `gpu_masks`, `gpu_chain`, the graded Colour Uniform test and the new `gpu_reference_grade` in three runs each. **The gate is 0 failures on every runner**, reported in `docs/STATUS.md` as the earlier rounds were. The Mac GPU CI runs the same suites. A failed gate is reported, never loosened.
- **Speed.** The hue step's cost per megapixel is measured in RG-04 on lavapipe and, as a Mac figure, in RG-H3; the starting budget is the Colour Uniform pull's (`docs/32` §9), which the step resembles. No figure is claimed here.

## 8. `ym`

- **The flags.** `ym render <photo> --reference <file> [--reference-strength <0–100>] [--no-skin-protection] [--reference-node <id>]`. The flags set fields of the node and run the fit; they are not new settings. The strength is in percent, as `--intensity <0–100>` is (`docs/10`); it sets `strength`.
- **What the fit does.** `--reference` fits **exactly as the app does** (§5): the same analysis size, loader, ICC reader, letterbox rule, chain and optimiser. A graph with **one** Reference Grade node is fitted; a graph with **none** gets one added after Main (as Add node does in a default graph); with **several**, `--reference-node <id>` is required (exit 2: "The graph has more than one Reference Grade node. Say which with --reference-node."). The fit overwrites the node's fitted fields as Fit again does (§3.3). With several photos on the command line each is fitted to the same reference.
- **Bit-identity.** For **the same reference picture**, the same photo, settings and version, `ym` and the app write the same fitted numbers and so the same bytes (rule 6). **An export from the app pins the fitted values in `ym:Settings`**, like the Auto answers (`docs/19`, `docs/22`), so `ym render photo --settings exported.xmp` (with no `--reference`) renders the same bytes as the app's export. A reference from the filmstrip is a display picture held in memory, not a file, so `ym` cannot take it; the pinned settings reproduce its fit.
- **The reference is a file only.** Errors have the app's texts (§9.3) after "The reference can't be used:" and exit 2 as an unusable `--settings` file does. `ym render --json` and `ym info --json` print the node in `settings` as the document holds it, with no reference.
- **The help** names the kind in the closing paragraph ("other nodes (correctors, Colour Uniform nodes, Reference Grade nodes, Blur nodes, mixers, …)") and `docs/10` quotes it and gains a *Reference Grade* section and the example of §3.1 as a JSON block that `crates/ym-cli/tests/docs.rs` reads and validates. The snapshot `cli__help_render` changes with it.
- **Version.** `ym --version`, `ym doctor` and `ym info` print the new pipeline version from RG-10 (§3.8).

## 9. The panel

Following `docs/11` and the glass components (`docs/16`): a control that glasscn has is never hand-written; our components wrap them. Copy is British, short and plain; errors say what went wrong and what to do next. **Mock-ups are a separate task (RG-07) and the owner reviews them before the build (RG-H2):** they follow this section; they do not exist yet, and `design/reference-grade/` is where they will go.

### 9.1 Tabs

| Tab | What it holds |
|---|---|
| **Reference** (the node opens on it) | the reference and the fit (§9.2) |
| **Light** | the corrector's Light tab, reused whole (`CorrectorLightTab`) |
| **Colour** | the corrector's Colour tab, reused (Temperature and tint, Colour balance, Colour warp) |
| **Curve** | the corrector's Curve tab, reused |
| **Hue** | **new**: two cards, "Hue vs hue" and "Hue vs saturation" (§9.4) |
| **Mask** | the corrector's Mask tab, as before |

Six tabs: 32 letters in a 400 px panel; the simple view's seven tabs are 33 and fit (`docs/32` §12a.2), so six fit; RG-08 tests that no label is clipped at 1440 × 900, 1180 × 760 and 1800 × 1100, and the fix, if one clipped, is the row's type size for this kind, never a shortened label. ⌘1 to ⌘6 select the tabs in the order of the row; the remembered tab is per kind. The tab components read the node's numbers through `resolvedGraded` (Colour Uniform's widening of `resolvedCorrector`), which learns this kind.

### 9.2 The Reference tab

One card, **Reference**:

1. **The thumbnail** of the reference (session only; empty until chosen), with its name under it (§3.7).
2. **Choose reference…** (a button that opens a menu): **Image file…** (the system open dialog, JPEG, PNG and TIFF) and **Photo in the strip** (the filmstrip becomes a chooser, like Pick colour: "Click a photo in the strip. Esc cancels."; the photo being graded and the unready ones are disabled, §6.4).
3. **Fit** (glasscn button; "Fit again" after a fit). Disabled without a reference, with the tooltip "Choose a reference first." While fitting it shows an indeterminate progress and a **Cancel** (Esc); a cancel changes nothing.
4. **Strength** (glasscn slider, 0 to 100 %, a number field, Reset at 100). A line under it: "Scales the fitted values. The tabs show them before Strength."
5. **Skin protection** (glasscn switch, on) with the line "Keeps skin tones from drifting. Applies at the next fit."
6. **The status line** (a muted `RowNote` in a `role="status"` group, so a screen reader hears it change), and, below it, the notes (§9.3).

The head's line stays "Reference Grade", "Bypassed", "Not connected". **Reset this node** puts every field back at its default; masks stay; one undo step.

### 9.3 Copy

**Status line**, one at a time:

| State | Text |
|---|---|
| no reference | "Choose a picture to match." |
| a reference, not fitted | "Press Fit to match the photo to this picture." |
| fitting | "Fitting…" |
| fitted | "Fitted to the reference." |
| nothing to change | "The photo is already close to the reference. Nothing was changed." |
| edited since the fit | "Edited by hand since the fit. Fit again to start over." |
| **look changed since the fit** | "The look changed since the fit. Fit again to match it with the new look." |
| values, no reference (after a restart, a paste or a preset) | "The values are kept. Choose a reference to fit again." |
| spatial nodes after the node | "Nodes after this one that blur or add film aren't part of the fit." (a note) |

**Notes** (under the status, as `RowNote`s, only when they apply): "Black bars were removed from the reference." · "The picture's colour profile couldn't be read, so it is taken as sRGB." · "This picture has no colour. The photo will be matched in black and white." · "No skin was found, so Skin protection did nothing."

**Errors**, shown in the card as a notice with the text and no code, and the same text after "The reference can't be used:" in `ym`:

| Case | Text |
|---|---|
| unreadable file | "This file can't be read. It may be damaged. Choose another picture." |
| unsupported format | "Yellowmoon reads JPEG, PNG and TIFF pictures. Save this one as one of those, then choose it again." |
| reference too small | "This picture is too small to match. Use one that is at least *N* pixels on its short side." (*N* from §16.6) |
| nothing inside the bars | "There's no picture inside the black bars. Choose another picture." |
| the same photo | "That's the photo you're grading. Choose another." |
| an unready photo | "This photo isn't ready yet." (a tile tooltip) |
| fit failed | "The grade couldn't be fitted. Try another reference." |

Strings are in `src/` with the rest of the copy; no exclamation marks. A test pins each text.

### 9.4 The Hue tab

Two cards on the Curve card's plot widget, **taught a periodic x axis** (0° to 360° with the rainbow track the Hue slider of Colour Uniform uses): **Hue vs hue** (y: −180° to +180°, a detent at 0) and **Hue vs saturation** (y: 0 to 8, a detent at 1, the axis drawn on a log or a linear scale chosen in RG-07). Points are added by a click on the plot, dragged, removed by Delete or a double click, and moved by the arrow keys; at most 16 per curve ("A curve can have up to 16 points."). Each card has its own Reset. The plot's accessible names and the keyboard path follow the Curve card. The fit's knots are at fixed hues but the person's points are free.

### 9.5 The rest

- **Masks work as on any node.** The Mask tab is the corrector's; the node's Show mask is the ordinary one.
- **Thumbnails.** The card's thumbnail is the picture after the node. The reference's thumbnail is the Reference tab's.
- **IPC.** The commands are additive and named in RG-06 (`docs/01`): choose a file, choose a strip photo, `fit_reference(node_id)`, `clear_reference(node_id)`; the core holds the reference per node for the session, the page holds only a thumbnail and a label, and the mock holds a stand-in picture so that the e2e tests run (the mock's fit returns a fixed patch). Nothing the page does saves the reference.
- **Accessibility** (`docs/28`): the glass audit and axe run on the new states; the status line announces; the chooser mode is keyboard-reachable; the Hue plots have the Curve card's keyboard path.

## 10. Tests, evaluation and gates

Every gate must pass in CI (or the named job). **A failed gate is reported, never loosened** (`AGENTS.md`). The thresholds of the evaluation are the owner's to set (§14, question 15); the numbers marked *starting* come from the POC (§16.8).

| Area | Test | Gate | Task |
|---|---|---|---|
| Format and rules | the fields, defaults and clamping; hue points (modulo, duplicates, limits, 16 points); a value not finite; an unknown field and its text; the 64-node limit; the schema; the written order | every refusal atomic with its plain text | RG-03 |
| Patches | add and remove; a partial `hue_curves`; the fit as one patch; Reset; copy and paste; undo (one step for a fit) | tested | RG-03 |
| Identity | empty curves; Strength 0; a bypassed node; a pixel with mask 0; a default node | bit for bit | RG-03 |
| Hue maths | one point is a constant; the periodic seam (0° and 360° give one value, a sweep has no step above 5e-3 of a code); `r = 0` and grey give no NaN and are unchanged; the tone `m` is preserved to 1e-12; rotation by 180°; the table against the exact spline (error stated) | 1e-12 where stated | RG-03 |
| Strength | each fitted field's formula of §3.2; monotone at every Strength; Strength 0 is the identity; the same `effective` on the CPU and in the host packing | 1e-12 | RG-03 |
| Order | `apply` equals `hue(C11(C9b(C9(trims(v)))))` and differs from the hue step first by more than 1e-3 somewhere | 1e-12 | RG-03 |
| Plan and random graphs | the node in the plan, live and dead, numbering, `picture_ancestors`; `any_graph.rs` draws `reference_grade` from `NodeKind::ALL` with fitted-style values and hue curves (a seed list joins the regression seeds; the existing draws do not change) | CPU: tiles and threads bit for bit; device: not refused, within the bounds, exports byte for byte across tile sizes | RG-03, RG-04 |
| Device parity | §7.3 | 2e-4 in LogC4; one 10-bit code after C12 and C13; identity exact | RG-04 |
| `diag-alf` | §7.3 | 0 failures on each of the four runners, 3 runs | RG-04 |
| Reference loading | JPEG, PNG and TIFF, 8 and 16 bits, grey; untagged → sRGB; ICC matrix/TRC fixtures (our own generated sRGB, Display P3 and Rec.2020, a `para` variant, a **screenshot "Display" profile**), checked against the oracle; an unreadable profile → sRGB with the note; HEIC refused; **letterbox**: a synthetic picture with black bars is trimmed, a **night frame with many dark rows** (noise, a dark sky) is not, a flat dark picture is not trimmed into nothing, pillarbox, JPEG ringing | the bars exact; the ICC conversion within the stated bound of the oracle (starting: set in RG-05) | RG-05 |
| The fit: properties | the identity photo (reference = the photo's own display picture) gives the identity and "already close"; no worse than the identity objective; knots monotone; ranges held; **1 and N threads give the same bits; two runs give the same bits** | exact | RG-02 |
| **E1, synthetic recovery** (CI, public data only) | grade a public photo with a **known random grade from the node's own parameter family** (seeded, including hue curves), through the real chain, to make the reference; fit the ungraded photo to it from the identity; measure CIEDE2000 (`crates/ym-color/src/deltae.rs`) between the fitted picture and the reference, median and 95th percentile; once per look (Signature, Neutral and an own synthetic `.cube`); a **negative control**: fit under one look and render under another is worse; also through an 8-bit sRGB JPEG round trip of the reference (the file path) | thresholds: the owner's (question 15); *starting* from §16.8 | RG-09 |
| **E2, cross-content transfer** (local) | photo A graded with G is the reference; photo B graded with G is the truth; fit B to A's picture; ΔE00 of the fit to the truth, against **three baselines**: no change, Reinhard (lαβ) and MKL, run as pixel maps for comparison only (never shipped); a few public photos of different content, including a mismatched pair (night against day) | the numbers are reported in `docs/STATUS.md`; the fit beats no-change on matched content | RG-09 |
| **E3, blind A/B [HUMAN]** | the owner picks references and keeps them local; `tools/eval/ab` shows the reference and two candidates (the fit, and a baseline) in random order; the rater picks the closer one; against no change and against the Reinhard map | the existing provisional gate: win share at least 60 % and the Wilson lower bound above 50 % (`docs/19` §13); **thresholds the owner's** | RG-H4 |
| App against `ym` | the example of §3.1 in a graph: JPEG, TIFF 16, PNG 16 and DNG; a fit through the app's path and through `ym render --reference` on the same file | byte-identical, on the CPU and on the device (`parity.rs`, `parity_gpu.rs`) | RG-06 |
| Golden | **one new entry**, `node-reference-grade`: a fixed set of fitted-style values on a synthetic scene, rendered by `ym` and pinned in `expected.json`; the device checked against it on the GPU subset | `golden check`: the current entries unchanged and the new one (so *N + 1* of *N + 1*, where *N* is the count when RG-06 lands); **no existing hash moves** | RG-06 |
| Presets and albums | a preset saved from a photo with the node holds it in `nodes`, applies the numbers elsewhere (no re-fit); an album round-trips and an older build's refusal text is the album's; the field test of `docs/30` §14 | tested | RG-03 |
| CLI | the example read and validated (`docs/10`, `crates/ym-cli/tests/docs.rs`); the help's paragraph; every error text; `--reference-node`; a graph with none, one and two nodes | the test fails on any difference | RG-06 |
| Interface | unit: the tabs, `NODE_TABS.reference_grade`, `resolvedGraded` for the kind, the shortcuts (⌘1 to ⌘6), `graphOps` (add after the selected node, the 64-node limit, reset), the status-line state machine (including "look changed" and "edited"), each copy string, the Hue plots' keyboard path; Playwright against the mock: the menu, choosing, Fit and Cancel, Strength, Skin protection, each error, the strip chooser, Reset, Copy and Paste, undo; **no tab clipped** at three window sizes; the glass audit and a11y; reference screenshots `design/screenshots-c2/photos-reference-grade-*.png` that **the owner looks at before they are committed** | green | RG-08 |
| Speed (on a Mac) | the fit's time (§5.6); a slider move in the fit view with 8 nodes, one a Reference Grade node with both hue curves; the node's cost per megapixel | the node view's 100 ms (`docs/23` §5.9); the fit target of 1 to 2 s, **unmeasured** | RG-H3 |
| Licences | `cargo deny check` and the npm licence check | pass; no new dependency (if one is chosen, it is recorded in the PR) | RG-05 to RG-08 |

**Data rules for the evaluation.**

- **Public photos only for choosing constants** (§16). The owner's photos are **a held-out check at most**, reported but not used to tune: the owner's rule "I won't let you tune specifically for my photos" (`docs/32` §2.3).
- **Copyrighted stills are local evaluation only**: never committed, never cached by CI, never used for shipped weights (there are none), never uploaded. E3 runs on references the owner keeps local, with `tools/eval/ab`.
- **Sources** are those of `docs/test-photo-sources.md` and `docs/13`. If a gate seems to need a measurement of our own, use public data and note the limitation in `docs/STATUS.md` (`AGENTS.md`, V1 data rule).

## 11. Speed

| Where | Budget |
|---|---|
| **The fit** | **1 to 2 s on an M1** (the research's target; unmeasured; RG-H3). The POC's and E1's Linux times are reported as Linux times (§16.7) |
| The photo's chain before the node | rendered once per fit at the analysis size |
| The node on the GPU | the hue step is one `atan2`, one `sincos`, one square root and two table reads; budget set in RG-04 from the Colour Uniform pull's (`docs/32` §9) |
| The node on the CPU reference | the corrector's steps and the hue step in f64, split across rows and threads; measured in RG-03 on the Linux machine |
| Memory | the reference's analysis picture (a few megabytes, session only); two 1024-sample tables per node on the device |
| Thumbnail | a few milliseconds, as for any node |

## 12. Risks

| Risk | What happens | What the design does |
|---|---|---|
| **Content mismatch** | A night reference could darken a beach; a reference full of one colour pulls the whole photo | T1 matches the tone **shape** and the level only weakly; the identity regulariser T4; the clip penalty T5; **Strength** scales the result; the status "Nothing was changed" when the fit does not help. E2 includes a mismatched pair |
| **Look dependence** | A fit is valid for one Output look | the fit runs the real chain (§2.2); changing the look marks the fit stale and shows a hint; nothing is re-fitted by itself |
| **Colour tagging of references** | An untagged file is a guess (sRGB); a wrong guess shifts the whole fit | tags are read (§6.3); the note when a profile cannot be read; Strength; the person can see the thumbnail |
| **Fit time** | Unmeasured on a Mac | a stated target, a Cancel, the remedies of §5.6, RG-H3 |
| **Skin** | A global grade can push skin toward the reference's cast | T6 limits the hue turn and chroma ratio on MediaPipe skin pixels; thresholds from public data only; Strength and the switch |
| **Local looks** | Vignettes, graded skies, split looks in one half cannot be copied by a global grade | out of scope (§13); the status never promises it; masks and the person's hand cover some |
| **Spatial nodes after the node** | Film, Blur and Face Refinement are not run in the fit | a note on the card; E1 and the golden entry have none after the node; question 7 |
| **The kernel change** | llvmpipe has miscompiled small control-flow changes in `run_nodes` | §7: host packing, one `if`, `select`, H1 first, `diag-alf` before release, H3 as the last resort |
| **Table resolution** | Hue tables of 1024 samples | the error against the exact spline is measured in RG-03 and stated |
| **Hue move changes luma a little** | The step keeps the mean, not luma | the fit sees it (§4.2); the luma curve compensates |
| **Reference copyright** | A film still is copyrighted | used at run time, never stored (§6.6); evaluation stills local only |

## 13. Not in this version

- **A learned initialiser** (Phase 2 of the research: a tiny predictor on synthetic pairs, feeding the fit). Only if Phase 1 misses its bar, and only with the owner's yes and a Modal budget (the research's question 4).
- **Spatial or local adaptation** (Phase 3: region-aware fits, a bilateral grid pass). Only if needed; it costs a binding, the `diag-alf` gate again and a new pipeline step.
- **Any provider's involvement**: Claude, Antigravity and Apple Intelligence receive no reference and write no numbers here (`docs/22`).
- **HEIC and other formats; RAW files as references; a reference from the web or a URL.**
- **Hue curves on correctors and Colour Uniform nodes.**
- **Batch matching** (apply one reference to a selection in one step). Each photo is fitted on its own node; a preset or Paste carries the values.
- **Matching inside masks, or a reference region.**
- **Our own measurements of any kind** (V2).

## 14. Questions for the owner (RG-H1), each with a recommended default

| # | Question | Recommended default | Why, and what changes if the answer differs |
|---|---|---|---|
| 1 | **The name.** | **"Reference Grade"** in the UI; kind `reference_grade`; "a Reference Grade node"; default node name "Match" | never an ARRI product name (`AGENTS.md` rule 3). Another UI name changes the strings and no code |
| 2 | **Strength: which fields, what scale.** | **Strength scales only what the fit writes** (the four curves, saturation, both hue curves), from the identity; stored and typed as 0 to 100 % like a look's intensity | the research and the request say "0 to 1"; the house rule is that a file reads like the panel (`docs/32` §5.1), so the stored number is 100 times larger. The alternative, scaling **every** field of the node, is a change to `effective` and more cases |
| 3 | **What an edit of a fitted value does.** | **It changes that stored value only; Strength still scales it; Fit starts again from the identity and overwrites the fitted fields; a session flag says "Edited by hand"** (§3.3) | the alternatives are a second stored copy ("fitted" and "edited") or an automatic re-fit; both are more state |
| 4 | **Is a label of the reference stored** (a file name)? | **No** (§3.7) | it would travel in every exported file; if wanted, a `reference_label` string, written to `ym:Settings`, presets and albums, with the privacy cost |
| 5 | **When the Output look changes after a fit.** | **A hint; no automatic re-fit** (§2.2); the look's id and intensity (and anything replacing C12) are what count | automatic re-fit would overwrite hand edits and spend 1 to 2 s on each look change; the alternative is "re-fit while the reference is still in memory" |
| 6 | **Does an upstream edit also mark the fit stale?** | **No** | an exposure tweak is a normal step after a fit; the person can press Fit again |
| 7 | **Spatial nodes after the node in the fit.** | **Not run; a note on the card** (§2.2) | running them per candidate is far too slow; the alternative is to refuse Fit when one is after the node |
| 8 | **The fit and the node's masks.** | **Ignored** (§5.1) | the reference describes a whole picture; fitting inside masks needs a matching region |
| 9 | **Skin protection.** | **On by default; MediaPipe Selfie Multiclass skin classes; read at the next fit** (§5.5) | off by default is possible; the term can be removed with its test |
| 10 | **ICC: our own minimal reader or a crate.** | **Our own** (§6.3) | `moxcms` is the candidate crate (licence class allowed in `docs/14`, not in the manifest); it needs a determinism check |
| 11 | **Letterbox removal: a switch?** | **Automatic, no switch**; a note says when it happened (§6.5) | a "Remove black bars" switch is cheap if the rule misfires; the person can also crop the file |
| 12 | **A reference from the strip is a snapshot, not a link.** | **Snapshot** (§1.3, §6.4) | a link would re-fit or go stale when the other photo changes, and would need the other photo's settings kept in memory |
| 13 | **Hue step: keep the mean, or restore luma as the look does?** | **Keep the mean** (§4.2) | restoring luma needs a decode and an encode per pixel in the kernel; the fit and the luma curve absorb the difference |
| 14 | **Hue curves only on this node** | **Yes** (§4.3) | the same field on correctors and Colour Uniform is the same work again, later |
| 15 | **The thresholds** of E1, E2 and E3. | **Set after the POC numbers (§16.8) by the owner.** E3 starts from the existing provisional gate: win share at least 60 % and the Wilson lower bound above 50 % against no change and against the Reinhard map | the research gives no published bar for this shape of problem (`research/reference-grade-2026-11-23.md`, *Methods*) |
| 16 | **The pipeline version.** | **0.17.0**, the next minor step (§3.8) | a patch step is allowed if the lead prefers; the number only has to be above 0.16.0 |
| 17 | **A fit-time limit.** | **Target 1 to 2 s on an M1, a Cancel button, no hard limit** | a hard limit would need a partial-result rule; decide after RG-H3 |
| 18 | **Mock-ups before the build.** | **Yes** (RG-07, RG-H2) | the node has a new tab and a new chooser mode |

## 15. Decisions made while writing the spec

The simplest readings where the request or the research was silent, and **where this file differs from the research**.

| # | Choice | Why |
|---|---|---|
| 1 | **The research says Phase 1 needs "no WGSL change"; this spec has one** (§7). | The owner chose hue curves over warp pins on 2026-11-23, and the research itself said that choice is a kernel change. The research text of *Parameter set* and *Effort* is superseded; the warp-pin parameter set is not used |
| 2 | **The research says the fit runs "through any nodes after it and the Output".** The spec runs the **per-pixel** steps only (§2.2). | Spatial nodes make their picture before the colour pass and cannot be run per candidate |
| 3 | **The research proposes CIELAB or Oklab; the brief chose CIELAB.** | CIELAB D50 matches the ICC connection space and `deltae.rs` |
| 4 | **The research leaves open which MediaPipe output gives skin; it is `BodySkin` (2) and `FaceSkin` (3) of Selfie Multiclass** (`crates/ym-ml/src/mediapipe/selfie.rs`). | Read from the code |
| 5 | **The research asks for ICC handling for "sRGB, Display P3 and Rec.709".** The spec reads any **RGB matrix/TRC** profile and refuses the rest (§6.3). | The macOS "Display" profile of screenshots is a matrix/TRC profile; a fixed list of three would miss it |
| 6 | **`strength` is stored 0 to 100, not 0 to 1.** | The house rule for stored units (§14, question 2) |
| 7 | **The hue step is last in the node** (after C11), not before the curve (§2.3). | The proven kernel shape; the fit's search stages |
| 8 | **The hue curves are baked to 1024-sample tables that are the definition** (§4.2). | One definition for the CPU and the device; the table error is not a parity term |
| 9 | **The node writes every field** (a new kind), unlike Colour Uniform's `trim` and `grade` (§3.1). | The omit-at-default convention existed for a kind that older builds already read |
| 10 | **A result no better than the identity is the identity**, not a failure (§5.4). | A reference that matches already is a normal answer |
| 11 | **The hue knots of the fit are at fixed hues, the person's points are free** (§9.4). | The fit's parameter count is fixed; the field is not |
| 12 | **`moxcms` is named in `docs/14` but is not in the workspace manifest.** | Found by reading `Cargo.toml` and `docs/14`; stated in §6.3 |

## 16. POC findings

**To be filled in by RG-01 from the POC (branch `refgrade-poc`). No number is invented here.** Each heading below is a decision that §5 and §6 point to. A line says what the POC measured, on what data, and whether the number is a Linux figure.

### 16.1 The parameter list and the knots
*To be filled in: the final parameter count and the list; the stop table for the luma knots; the ranges and identity scales; the three-knot positions of the channel curves; the hue knots.*

### 16.2 Analysis sizes and sampling
*To be filled in: the long edge of the analysis pictures; the number of stratified pixels; the reference's sampling; the reduction filter.*

### 16.3 The objective
*To be filled in: the quantile levels of T1 and `w_level`; the `L*` band edges of T2; the hue sectors of T3; the weights of T1 to T6; the skin limits; whether a sliced Wasserstein term was needed.*

### 16.4 The optimiser
*To be filled in: the method of each stage; the step schedules; the iteration counts; the seed.*

### 16.5 Letterbox constants
*To be filled in: the standard-deviation and mean thresholds; the minimum bar width; the step test; the ringing margin; the maximum bar fraction; the pictures it was tested on, including the night frames.*

### 16.6 The minimum reference size
*To be filled in: *N* of the "too small" text.*

### 16.7 Time
*To be filled in: the fit's time on the POC's machine (label it as Linux, not Mac), per stage; where the time goes.*

### 16.8 Quality
*To be filled in: E1 and E2 numbers against the baselines, on public photos; the mismatched pair; what did not work; the starting thresholds for §10.*

### 16.9 What the POC did not cover
*To be filled in: the GPU, the ICC reader, the strip source, the UI, the Mac timing.*

## 17. Mock-ups

None yet. **RG-07** makes them in `design/reference-grade/` (pages at 1440 × 900 built from the app's DOM and CSS, shot to JPEGs, as `design/colour-uniform/` and `design/blur/`): the node card, the Reference tab in each state of §9.3, the chooser mode in the strip, the Hue tab, and the error notices. The pictures are synthetic scenes and references, with no film still and no likeness. The owner reviews them (**RG-H2**) before RG-08 builds the panel.

## 18. Tasks

Task format of `docs/15`, milestone RG. The work goes on its own branch, one reviewed commit per task with its checks green (`cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`, `npm test`, `npm run build`). Every task that changes what `ym` does or prints updates `docs/10` and the help in the same commit. The tasks and their *Done when* are in `docs/15` (milestone RG: **RG-01** to **RG-10**, with **[HUMAN] RG-H1** to **RG-H4**).

The order is RG-01, then RG-02 and RG-03 (parallel); RG-04 after RG-03; RG-05 beside them; RG-06 after RG-02, RG-03 and RG-05; RG-07 after RG-01 (it can start at once); RG-H2 after RG-07; RG-08 after RG-H2 and RG-06; RG-09 after RG-02 and RG-06; RG-10 last. The `diag-alf` gate (RG-04) blocks any release that contains the change.

## 19. As built

*(Filled in by the tasks as they land.)*

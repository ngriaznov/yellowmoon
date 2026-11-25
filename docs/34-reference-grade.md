# 34 · Reference Grade

Reference Grade is a node that matches the grade of a **reference picture**. The person gives it a picture, a film still, a favourite photo or another photo of the same shoot, and the node **fits its own grade** so that this photo, seen through the person's real Output (their look and the cine rendering), looks like the reference. It is for the things a photographer does by eye with a picture beside the screen: "make this look like that". This file covers what it does, where it sits, the settings, the hue curves and the element grades, the scene (what is sky, foliage, water, ground, a building, skin), the fit, how references are read, the kernel change, `ym`, the panel, the tests and gates, the risks, the questions for the owner (answered), the mock-ups and the tasks.

**Status: specified (RG spec 2026-11-24; rewritten to the proof of concept's round 15 by RG-01, 2026-11-25); nothing is built.** The proof of concept (POC) was a Rust example on the branch `refgrade-poc` (CPU only, no UI). It ran fifteen rounds; its last, round 15, was accepted on a held-out set (§16). This file adopts its design. **The owner's decisions (RG-H1, 2026-11-25):**

1. **Grade per element, with a scene segmenter.** Six elements (sky, vegetation, water, ground, built, skin) each get an exposure offset, a hue rotation and a chroma scale; the subject gets an exposure offset only. The segmenter is EoMT small (COCO panoptic): MIT weights and code on a DINOv2 ViT-S backbone (Apache-2.0), 24.0 M parameters, about 48 MB at f16, run by our own Rust runtime and shipped as SAM 2.1 is (§4b.1). Skin comes from MediaPipe Selfie Multiclass, already shipped. The grading network is hand-written, ours and shipped as a text file; it sets targets and gates, and the solver sets the node's settings.
2. **A fit may take several seconds** on a Mac with all cores, with progress and Cancel. Speed work is a task (RG-11); the owner times it (RG-H3).
3. **A quality round on skin and tinted whites before release** (RG-13), judged on a fresh held-out set under the owner's three acceptance rules (§10).
4. **The 18 questions of §14: every recommended default is accepted as written.** Round 15 changed what a few of them mean; §14 says which.

Earlier decisions: 2026-11-23, fit at run time, reference sources are image files and other photos in the filmstrip, hue moves use **a new hue-curve field on the node** (so the colour kernel's node code changes and `diag-alf` must pass before a release). 2026-11-24, "Start the spec. But we want POC of our reference grade approach so I could see how it works before we do full implementation"; then, during the POC, "Grading should be per-element if needed and global" and "We still want a small hand-written model". The research is `research/reference-grade-2026-11-23.md`; **this file wins where they differ** (§15 lists where). The maths is ours: we copy no code and no formula from any tool (`docs/14`).

**In short.**

- A new node kind, `reference_grade`, in the graph of `docs/23`: a numbered node like a corrector, with masks, a key output, a thumbnail and a number. Add node › **Reference Grade**, "Matches the grade of a reference picture." (§9.1). It counts toward the 64 nodes. In a sentence it is "a Reference Grade node".
- **It is a fitted grade.** A run-time fit, in f64 on the CPU, chooses the node's own fields so that the photo, rendered through **the real chain from the node's input to the Output**, reaches the targets a small hand-written network sets from measurements of the photo and the reference (§4b, §5). The fields are a global grade (a luma curve, three channel curves, saturation, two hue curves) and **element grades** (§4a): up to 19 numbers more, each element through a soft mask. The fitted numbers are ordinary settings. The reference is not saved (§3.7).
- **One learned model, for masks only.** EoMT small finds the sky, foliage, water, ground and buildings of the photo and of the reference; MediaPipe finds skin. They make masks and measurements. Every grading decision is in the network file, which is text we wrote. The fit trains nothing.
- **A fit is a search of a few seconds, with progress and Cancel.** The POC measured about 30 s per fit on one Linux thread (§16.7); Mac figures wait for RG-H3.
- **Walls keep the grade from harm.** The objective carries walls against purple turns, skin drift, memory colours (sky, foliage), tinted whites, clipping and colours that turn into other colours (§5.4). Round 15 cut the share of fits with any measured harm from 34.8 % to 15.7 % on held-out photos without losing look (§16.8).
- **Global and element looks only.** A vignette, a graded sky gradient or a different grade in the left half of the frame are not copied (§13). The element grades are the only local part.
- **A new field: hue curves** (§4): `hue_vs_hue` and `hue_vs_sat`, periodic, on the LogC4 chroma-plane hue axis the parametric look uses. A kernel change (§7), made as small as the llvmpipe rule asks.
- **Strength** (0 to 100 %) scales the fitted values from the identity, instantly, with no re-fit (§3.2).
- **Skin protection** (on by default) switches on the walls that keep skin from drifting (§5.6).
- **App and `ym` agree to the bit** (§8): `ym render photo.ARW --reference ref.jpg` fits as the app does, and an export from the app pins the fitted numbers in `ym:Settings`.
- **A new node changes no pixel** until it is fitted or edited, and no existing document changes. `PIPELINE_VERSION` takes the next minor step (§3.8).

## 1. What it does

A photographer has a still from a film, or a photo they love, and wants their own photo to feel like it. Resolve has Shot Match, and Photoshop has Match Color (`research/reference-grade-2026-11-23.md`, *Precedents*). Here:

1. Add node › Reference Grade. It does nothing yet.
2. Choose a reference: an image file, or another photo in the filmstrip.
3. Press **Fit**. After a few seconds, with a progress bar and a Cancel button, the node's Light, Colour, Curve, Hue and Elements values are set. The photo now looks, through the person's own look, like the reference.
4. Lower **Strength** to take part of it. Adjust any tab by hand. Limit it with masks. Export.

### 1.1 What it is, and what it is not

- **It is a node that fits its own grade.** The numbers are the node's corrector fields, its hue curves and its element grades (§3). After a fit the person can read them, move them and undo them, as for any corrector.
- **It is fitted against the person's real Output.** The look (the Signature look, an imported ARRI look or the person's own LUT) and the cine rendering are part of the chain the fit runs, so the match is what the person sees. A grade matched to a reference on a model of the chain would be wrong on their Mac (§2.2).
- **It reads the scene.** It knows where the sky, the foliage, the water, the ground, the buildings and the skin are, in the photo and in the reference, and it grades each on its own as well as the whole (§4a, §4b). That is the part a global grade cannot do: a sky that goes teal while the skin stays natural.
- **It has one learned model and it is only for masks.** Nothing is trained for grading. The segmenter (EoMT small, §4b.1) and MediaPipe make masks and measurements; the rules that turn measurements into targets are a hand-written network of ours (§4b.5), and a solver finds the settings (§5). Nothing is downloaded at run time (`AGENTS.md` rule 7).
- **It is not a LUT copy.** The result is a small set of readable controls, not a table. It cannot reproduce what its parameters cannot express (§12).
- **It is not a transfer of pixels.** It never moves a pixel of the reference into the photo. It compares statistics of two pictures that may show different things (§5.1), which is why it copies global and element looks only.
- **It is not Auto.** Auto resolves the Input's and Main's numbers from one picture and a rule (`docs/19`, `docs/22`); its providers receive one picture and cannot carry a reference (`docs/22` §1 and the research). This node has no Auto and no provider.

### 1.2 How it differs from what exists, and what it reuses

| | What it does | Reference Grade |
|---|---|---|
| **A corrector** (`docs/23`) | The person sets the trims, the grade and the curve by hand | **Reuses the corrector's fields and steps whole** (trims, C9, C9b, C11, the mask blend). The fit writes them. Adds two fields: the hue curves (§4) and the element grades (§4a) |
| **Colour Uniform** (`docs/32`) | Pulls colours toward one picked colour; has the corrector's tabs after the pull | **Reuses its way of being a corrector-like node with extra fields**, its tabs (Light, Colour and Curve are the same components), its kernel precedent (§7) and its first-release rules. The maths is different: no target, no pull |
| **The parametric look's hue curves** (`docs/08`, `crates/ym-color/src/look.rs`) | `hue_vs_hue` and `hue_vs_sat` baked into a 65³ cube, in the Output slot | **The same axis and the same meaning** (§4.1), now as a node field, evaluated per pixel. The look's curves are not changed and a look still bakes to a cube |
| **The colour warp** (C9b, `docs/02`, `warp.rs`) | Up to 16 Gaussian pins in the chroma plane; radial, not a function of hue angle alone | **Not used by the fit.** The owner chose true hue curves (2026-11-23). The warp stays a tab of the node, as on a corrector |
| **Object, Subject and Depth masks** (`docs/23` §4) | Masks a person draws or picks, for a node's `masks` | **The element masks are not mask rows.** They are made by the node from the scene model, one per element, and act as the element grades' own weights (§4a.3). The node's `masks` still limit the whole node |
| **Auto** (`docs/19`, `docs/22`) | Writes numbers into Main and pins them at export | **Reuses the pattern**: fitted numbers are plain settings and `ym:Settings` pins them (§8) |
| **Presets and albums** (`docs/30`, `docs/31`) | A recipe of settings; photos as they are | **Carry the node** like any node (§3.6) |
| **`tools/eval/ab`, `deltae.rs`** | Blind A/B kit; CIEDE2000 | **Reused** for the blind A/B and the key-preserving E2 line (§10) |

### 1.3 Photo-to-photo matching

A reference can be another photo in the filmstrip, rendered through that photo's own current settings (§6.4). Matching one photo of a series to another is a strong use. The reference is a **snapshot** of that photo's picture at the moment it was chosen, not a link: if the other photo is graded again later, this node keeps matching the picture it was given (§6.4).

## 2. Where it sits

### 2.1 The node

**A corrector-like numbered node** (Colour Uniform's and Film's precedent). `kind` is `reference_grade`.

| Property | Value |
|---|---|
| Ports | in `picture`, `key`; out `picture`, `key` (as a corrector's, `docs/23` §3.4) |
| Number, thumbnail | numbered in the plan's order; a card with a thumbnail and a number chip (`docs/23` §5.7) |
| Masks | `masks` as a corrector's (`docs/23` §3.3): the Mask tab, Invert, the key input, Show mask. They limit where the grade acts. **The fit ignores them** (§5.1). The element masks of §4a.3 are the node's own, not mask rows, and the fit uses them |
| Order inside the node | trims → C9 → C9b → C11 → **hue curves (new)** → **element grades (new)** → the mask blend (§2.3) |
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
- **The fit measures the Output as sRGB display codes** (§5.1), whatever space the viewport shows, and the element masks are part of each candidate (§4a.3).

**The fit depends on the Output look.** A grade matched under one look is wrong under another. So **changing the Output's look, its intensity, or anything that replaces C12 makes the fit stale** (§14, question 5):

- The page keeps, for the session only, a fingerprint of the Output's look settings at the moment of the fit. When it differs, the card's status line says **"The look changed since the fit. Fit again to match it with the new look."** and the Fit button reads "Fit again" with an accent. **Nothing is re-fitted by itself** and no number moves: a re-fit takes several seconds and would replace the person's hand edits (§3.3).
- Other changes (an earlier node's exposure, a node after) do not mark the fit stale. The person can press Fit again at any time.
- After the app restarts there is no fingerprint (nothing about the fit is stored, §3.7), so there is no hint: the values are just settings.

### 2.3 The order inside the node

```
w   = hue( C11( C9b( C9( trims(v_in) ) ) ) )                       the corrector's grade, then the hue curves
w'  = element grades of w, in the order sky, vegetation, water, ground, built, skin, subject
out = v_in + m · ( w' − v_in )                                      m: the node's own masks
```

with each element grade, in that order, `w ← w + a_e · (local_e(w) − w)` where `a_e` is the element's soft mask at the pixel (§4a.3) and `local_e` the grade of §4a.2.

- The corrector's steps keep their order whole (`docs/23` §1.3, `ym_color::display::node`). The hue curves are **step 6**, after C11, as the Colour Uniform pull is last in `node_apply` and as the parametric look puts its colour curves after its tone (`docs/08`). Reasons: the fit's tone and channel curves have then done their work and the hue curves refine the colour that is left; and a step that runs last is the shape the kernel's precedent proved safe (§7.2).
- The **element grades are step 7, last, in LogC4**, after the hue step and before the Output (C10 look, C12, C13). This is the order the POC ran (§16): the global grade first, the elements on its result. The seven masks `a_e` do not depend on the node's grade (§4a.3), so a grade never moves its own mask.
- A fixed order keeps the fit's search stages simple (§5.5).

## 3. Settings

### 3.1 The format

A new value of `kind` (`docs/23` §3.2): `reference_grade`. The node has these fields and no others; another field is an unknown field and is refused. As a new kind, the node **writes every field** (the convention for a kind, as `docs/23` §3.1 gives for a corrector).

| Field | Type / range | Default | Notes |
|---|---|---|---|
| `name`, `bypass`, `pos`, `masks` | as a corrector's | `"Match"`, false, – , none | |
| `trim` | a corrector's (`docs/23` §3.2) | all 0 | the person's; the fit does not write it |
| `grade` | a corrector's | all manual 0, no pins, the identity curves | **the fit writes `grade.curve` (luma, red, green, blue) and `grade.saturation`**; the other fields are the person's (§3.3). Auto reads as a manual 0, silently, as on Colour Uniform |
| `hue_curves` | `{ hue_vs_hue, hue_vs_sat }` (§4.1) | both empty (the identity) | the fit writes both |
| `elements` | `{ sky, vegetation, water, ground, built, skin, subject }` (§4a.1): per element `exposure_ev`, `hue_deg`, `chroma`; the subject has `exposure_ev` only | every element at the identity (0, 0, 1; the subject 0) | the fit writes all seven |
| `strength` | 0 to 100 (percent) | 100 | scales what the fit writes (§3.2). The number is in the units the panel shows, like a look's `look_intensity` (`docs/10`) |
| `skin_protection` | bool | `true` | read by the next fit only; it changes no pixel by itself (§5.6) |

There is no `reference`, no file name, no statistics, no element map and no fit status in the file (§3.7).

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
  "elements": {
    "sky":        { "exposure_ev": -0.30, "hue_deg": 8,  "chroma": 1.10 },
    "vegetation": { "exposure_ev": 0,     "hue_deg": -6, "chroma": 0.90 },
    "water":      { "exposure_ev": 0,     "hue_deg": 0,  "chroma": 1 },
    "ground":     { "exposure_ev": 0,     "hue_deg": 0,  "chroma": 1 },
    "built":      { "exposure_ev": 0,     "hue_deg": 0,  "chroma": 1 },
    "skin":       { "exposure_ev": 0,     "hue_deg": 3,  "chroma": 1 },
    "subject":    { "exposure_ev": 0.20 } },
  "strength": 100, "skin_protection": true,
  "masks": { "invert": false, "items": [] } }
```

The example is an illustration with fewer knots than a fit writes; it is not a result. The knots' positions are the stored numbers; the fit's own knot table is in §5.2.

### 3.2 Strength

`strength` scales **what the fit writes**, from the identity, at render time and with no re-fit. For each **fitted field** (`grade.curve.luma`, `.red`, `.green`, `.blue`; `grade.saturation`; `hue_curves.hue_vs_hue`; `hue_curves.hue_vs_sat`; every number of `elements`) and `k = strength / 100`, the render uses the **effective** values:

| Field | Effective value |
|---|---|
| a curve's knot `(x, y)` | `(x, x + k (y − x))` |
| `saturation`, manual `s` | `k s` (saturation is linear in `s`: `λ = 1 + s/100`, `docs/02` C9) |
| `hue_vs_hue` `δ` (degrees) | `k δ` |
| `hue_vs_sat` `σ` (a chroma multiplier) | `1 + k (σ − 1)` |
| an element's `exposure_ev` `e` and `hue_deg` `h` | `k e` and `k h` |
| an element's `chroma` `c` (a multiplier) | `c^k` (a ratio scales as a power, so half of ×1.6 is ×1.26, and Strength 0 is ×1) |

- **Monotone stays monotone.** A knotwise mix of two increasing curves is increasing, so a curve at any Strength is a valid curve.
- **Strength 0 is the identity**, bit for bit, when the node's other fields are at their defaults. **Strength 100 is the stored values.**
- The mix is made in f64 in one function, `ReferenceGradeNode::effective`, which the CPU reference, the cache key and the device's host packing all call. **So the kernel never sees Strength** and needs nothing for it. An element whose effective grade is the identity makes no mask (§4a.3).
- Strength does **not** scale `trim` or the grade fields the fit does not write (Contrast, Brightness, the zones, Colour balance, the warp). They are the person's own numbers (§3.3).
- A hand edit of a fitted field is scaled by Strength like the fit's own value. The tabs show the **stored** numbers; at a Strength below 100 the picture shows less than the numbers say. The Reference card says so in a line under the slider (§9.3).
- The POC scaled its search parameters (a slope in log space, a shift in stops) and not the stored knots; the two agree at Strength 100, where every gate of §10 is measured. Strength is a product control the POC did not exercise (§16.9).

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
- **Element grades** (§4a.1): `elements` must be an object whose members are `sky`, `vegetation`, `water`, `ground`, `built`, `skin` and `subject`; another name is refused ("A Reference Grade node has no element called “x”."). A member or a field left out of a document reads as its default. The subject has `exposure_ev` only; a `hue_deg` or `chroma` on it is an unknown field of that member. `exposure_ev` is held to ±1, `hue_deg` to ± the element's cap (40, 35, 35, 30, 25, 15) and `chroma` to 0.5 to 1.6: clamped and reported by path (`4.elements.sky.hue_deg`); a value that is not finite is `SettingsError::NotFinite`.
- `strength` 0 to 100 is clamped and reported. `skin_protection` must be a bool (a parse error otherwise).

### 3.5 Patches, Reset, Copy and Paste, undo

- **Patches** (`docs/23` §3.7). A node is added by giving its `kind` in `nodes` and removed with `remove`; its links go with it. `SettingsField` gains `HueCurves`, `Elements`, `Strength` and `SkinProtection` (the nodes' group; a change runs the colour pass again and nothing else). The node takes `Trim` and the grade fields as Colour Uniform does (`of_graded_field`, `apply_graded`, `graded_same`, `docs/32` §12a.3): **one place** turns a trim or a grade field into a patch. A patch carries `hue_curves` partially: a curve given replaces that curve, a curve left out stays; it carries `elements` the same way, member by member.
- **The fit is one patch**: `grade.curve`, `grade.saturation`, both hue curves and all seven element members, in one undo step. A slider drag folds into one undo step as every slider.
- **Reset** (`SettingsPatch::reset_node_in`): the node's own fields (`trim`, `grade`, `hue_curves`, `elements`) to their defaults, `strength` to 100 and `skin_protection` to true; name, bypass, masks and links stay. Each card keeps its own Reset (§9).
- **Copy settings and Paste** carry the node like any node. Undo diffs per node.
- **Resolved settings, XMP and DNG.** The node has no Auto, so it is in `ResolvedSettings.settings` as it is, and `ResolvedSettings.values.nodes` has an entry with its trim and grade (as for Colour Uniform, `resolvedGraded`). `ym:Settings` holds the node with its fitted numbers; the DNG bakes it into the pixels like every node (`docs/09`).
- **The JSON Schema** (`crates/ym-core/schema/settings.schema.json`) gains `reference_grade_node` (with `trim` and `grade` as `$ref`s to the corrector's definitions, and `hue_curves`, `hue_point`, `elements`, `element_grade`) and the kind in the `kind` enum. The wire types gain `NodeKind::ReferenceGrade` and `NodeItem::ReferenceGradeNode`; `bindings.ts` is regenerated and the field-name test compares them.

### 3.6 Presets and albums

- **Presets** (`docs/30`). The node is in the **`nodes` group**, which holds every node other than the Input, Main and the Output; no new group. A preset saved from a photo that has the node holds the fitted numbers, `strength`, `skin_protection` and the masks. **Applying it to another photo applies the same numbers; it does not re-fit.** A preset is therefore a recipe of a fixed grade, as a Colour range's centre and a warp's pins are fixed. The person can open the node and Fit again with a reference. The preset table text of `docs/30` §2 gains "Reference Grade nodes" in the list of kinds, and the test that a field in no group fails (`docs/30` §14) covers the new fields. Nothing in the node is a stroke or a click, so `docs/30` §3 (what a preset never holds) is unchanged.
- **Albums** (`docs/31`). An album holds the photo's whole graph as stored, so the node travels in `settings` with no change to the file (version 1 stays). An older build refuses it with "This album was made by a newer version of Yellowmoon. Update to open it." `written_by.pipeline` records the version (§3.8).
- **Older documents** read as they did. A document that holds the node is refused by an older Yellowmoon with the plain unknown-kind sentence (`docs/23` §3.5).

### 3.7 What is saved, and what is never saved

By rule 4 nothing is kept per photo. For this node:

| Saved, as ordinary settings | Never saved |
|---|---|
| the fitted numbers (curves, saturation, hue curves, element grades), `strength`, `skin_protection`, the masks | **the reference picture**, in any form; its pictures; **any statistic computed from it**; **the scene of the photo or of the reference** (label maps, skin plane, soft masks, measures); the fit's objective, seed or status; the Output-look fingerprint of §2.2; the "edited since the fit" flag |
| | **a label of the reference** (file name, path, photo name) |

**Recommended: no label is stored** (§14, question 4). Reasons: (1) `ym:Settings` goes out inside every exported file, and a file name is information about the person's library, and for a film still a hint of what was copied; (2) a path says nothing on another Mac and is private; (3) a label has no effect on any pixel, and a setting that does nothing is noise in documents, presets and albums; (4) it would go stale at the first hand edit. The card shows the reference's name **on screen, for the session only**: the file's name, or "Photo 3 of 7" for a strip photo.

### 3.8 `PIPELINE_VERSION`

No existing graph changes by a bit (a node that is not there runs nothing; §7.3 keeps the kernel's text for every other node identical), so **no golden entry moves**. An older build refuses a graph with a kind it does not know, so `PIPELINE_VERSION` takes **the next minor step** with the node, the rule `docs/01` *pipeline_version* and `docs/23` set when a kind lands. Today's is 0.16.0, so **0.17.0**, set by RG-02 and written to `ym:PipelineVersion`, `ym --version`, `ym doctor`, `ym info` and an album's `written_by` (the snapshots and the tests that assert the number follow). If another change takes 0.17.0 first, the next free minor number is used; the rule does not depend on the value. The golden set gains **one** entry (§10).

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

- **The tone `m` is exact.** The basis is orthonormal and `e1`, `e2` are orthogonal to `(1,1,1)`, so `v'` has the same mean as `v`: the step moves hue and chroma only. **It does not restore Rec.709 luma** as the parametric look does (the look does it on the LogC4 codes, with a dot product and an add; the draft's reason, a decode and an encode per pixel, was wrong; `docs/32` §3.1 point 3 argues for leaving it). A hue move therefore changes luma a little, and the fit sees that, because it runs the real chain (§5); the luma curve covers the rest. Flagged: question 13, and §15, decision 19 (the POC's step restored luma).
- **No neutral guard is needed.** The step scales and rotates the vector `(p, q)`, so as `r → 0` the move tends to 0: it is continuous at grey and exact at `r = 0`. A pixel with `r = 0` is returned as it is.
- **The identity is exact.** A node with both curves empty (or Strength 0) runs no hue step (`N_HUE` is not set; §7).
- **The curves are evaluated from baked tables**, one of 1024 f64 samples around the circle for each curve, built from the spline in f64 by one function, with linear interpolation between samples and wrap-around. **The table is the definition**: the CPU reference reads the same table in f64 and the device reads it as f32, so the baking error is not a parity term. The error of the table against the exact spline is measured in RG-02 and stated in `docs/STATUS.md` (§10).

### 4.3 Where else

Only the Reference Grade node has `hue_curves` and `elements` in this version. Correctors and Colour Uniform do not; giving them the field is a later idea (§13), the same work in another kind.

## 4a. The element grades

The element grades are the node's local part: a small grade for each of six kinds of thing in the picture and one for the subject, each acting only where its **soft mask** says that thing is. The POC's round 5 added them to the global grade, and every later round kept them (§16).

### 4a.1 The fields

`elements` has seven members. The first six are the **elements**; the seventh is the **subject**.

| Member | What it is (§4b.2, §4b.3) | `exposure_ev` | `hue_deg` | `chroma` |
|---|---|---|---|---|
| `sky` | sky | −1 to 1, 0 | ±40, 0 | 0.5 to 1.6, 1 |
| `vegetation` | trees, grass, flowers, plants | −1 to 1, 0 | ±35, 0 | 0.5 to 1.6, 1 |
| `water` | sea, river, other water | −1 to 1, 0 | ±35, 0 | 0.5 to 1.6, 1 |
| `ground` | sand, earth, road, rock, mountain, snow | −1 to 1, 0 | ±30, 0 | 0.5 to 1.6, 1 |
| `built` | buildings, walls, roofs; and the surfaces of interiors | −1 to 1, 0 | ±25, 0 | 0.5 to 1.6, 1 |
| `skin` | skin of people (MediaPipe) | −1 to 1, 0 | ±15, 0 | 0.5 to 1.6, 1 |
| `subject` | the person, or the main object | −1 to 1, 0 | – | – |

- **`exposure_ev`** is an exposure offset in stops, in the scene's log space: it adds `exposure_ev · 0.0648` to the three LogC4 codes (0.0648 is the POC's code step of one stop, `V5_STOP_CODE`).
- **`hue_deg`** turns the hue by that many degrees in the chroma plane (positive turns red towards yellow, as `hue_vs_hue`). The range of each element is its **cap**, the largest turn the network lets the fit make for it (§4b.5): a sky may turn 40°, skin 15°.
- **`chroma`** multiplies the chroma, as a ratio (1 is unchanged), in the same plane.
- **The ranges are the fit's own bounds.** The panel's sliders and the validation use the same numbers (§3.4); a hand edit cannot go further than a fit can.
- Defaults are the identity. A member at the identity runs no step and makes no mask.

### 4a.2 The step

For one element, on the pixel's LogC4 triple `v` (the basis of §4.1):

```
m = (v_R + v_G + v_B) / 3        p = v · e1        q = v · e2
(p', q') = chroma · R(hue_deg) · (p, q)
local(v) = (m + exposure_ev · 0.0648) · (1, 1, 1) + p' e1 + q' e2
```

The subject's step is the exposure term alone. The mean `m` moves only by the exposure offset, so a hue or chroma change keeps the tone. As with the hue curves, **no luma restoration** is made here, and the POC did none either (§16.9).

### 4a.3 The soft masks

Each member has a **soft mask** `a_e` per pixel, 0 to 1, made like this:

1. **A binary mask** from the scene: the pixels the segmenter calls that element (§4b.2), skin from MediaPipe (§4b.3), the subject by the rule of §4b.3. At each size the label map of the segmenter's own grid is carried by nearest neighbour (centre-mapped: source index `⌊(x + ½)/W · S⌋`).
2. **A guided filter** (radius `r = max(1, round(0.02 · short side))` px, ε = 1e-3), with the picture's **L\*/100 as the guide**: box means with the window cut at the frame (a mean over the pixels that exist), `mean_I`, `mean_p`, `corr_I`, `corr_Ip`; `var = corr_I − mean_I²`, `cov = corr_Ip − mean_I mean_p`; `a = cov / (var + ε)`, `b = mean_p − a · mean_I`; `q = mean(a) · I + mean(b)`; then `a_e = clamp(q, 0, 1)`. A member whose binary mask is empty has `a_e = 0`. The arithmetic is f32 with f64 box sums, in one fixed order.
3. **The guide.** The guide is the L\* (CIELAB, D65, of the sRGB display codes) of **the Input's picture as `docs/23` §4.7 renders it**, the same picture the segmenter reads. This is one place where the product departs from the POC, which took the guide from the photo's default render through the person's look (§15, decision 15). The reason is the rule of `docs/23` §4.7: grading a node never moves its own mask.
4. **Where and when.** Masks are made **per photo, in the background, at the size they are needed**: the analysis size for a fit, the viewport's size for the viewer and **the full size at render and export** (the radius is a share of the short side, so the same filter at every size). Like the Subject and People masks they are held in memory for the session only, per photo, and dropped with the photo (`docs/23` §3.11): never saved, never written to `ym:Settings`.
5. **Only what is used.** A mask is made only for an element whose effective grade (§3.2) is not the identity, and for a fit, for every element of at least 2 % of the picture. A photo with no element grade at the identity pays nothing.
6. **Tiles.** An export cuts the planes per tile with the filter's halo (`2r + 1` px) as raster masks do (`docs/algorithms/06` §5.2). Bits must not depend on the tile size (§10, `any_graph`). The box sums are therefore made with arithmetic that gives the same bits for a window however the plane was cut (RG-04 chooses it); if none does, the planes are made whole, in the memory ledger.
7. **Where the model is missing** (a damaged install): there are no masks, the element step is skipped, the viewer says so and an export is refused (§8).

### 4a.4 What an element grade is not

- It is **not a mask row**. The node's `masks` (`docs/23` §3.3) limit the whole node as for any node: `out = v_in + m (w' − v_in)` (§2.3). The element masks are inside.
- It is not applied where `a_e = 0`, and a pixel in two elements (skin lies inside the person; `built` joins two label groups) takes the steps in the order of §2.3, each by its own mask.
- It does not move geometry and does not change what the segmenter sees.

## 4b. The scene

"The scene" is what the fit knows about a picture beyond its pixels: which pixels are sky, foliage, water, ground, built surfaces, skin and subject, and the measurements made on each. It is made the same way for the photo and for the reference.

### 4b.1 The segmenter

- **Model.** **EoMT small** (Encoder-only Mask Transformer, CVPR 2025), the COCO panoptic checkpoint `tue-mps/coco_panoptic_eomt_small_640_2x`: a plain DINOv2 ViT-S/14 with 4 register tokens (hidden size 384, 12 blocks, 6 heads, MLP ratio 4, LayerScale), 200 learned queries entering for the last 3 blocks, two upscaling blocks, a class head of 133 COCO panoptic classes plus "none" and a mask head. **24.0 M parameters** (23,976,079), 96 MB at f32 and **about 48 MB at f16**. Panoptic quality 46.7 on COCO (the authors' model zoo).
- **Licences** (`docs/14`). The EoMT code and weights are **MIT** (Mobile Perception Systems Lab at TU/e; the repository's `LICENSE` and the model card's `license: mit`). The backbone is DINOv2, **Apache-2.0**. The training set is COCO panoptic 2017: its annotations are CC BY 4.0 and its images are under Flickr's terms, noted as for SAM's SA-1B. The DINOv3 variants (`…_dinov3`) carry the DINOv3 licence and are **not used**. Mask2Former, SegFormer and the ADE20K-trained options were rejected for non-commercial terms or cost (`research/` notes of the POC).
- **Runtime.** Our own pure-Rust CPU code, `ym_ml::eomt`, f32, like `sam2` and `depth`. It **reuses the depth model's DINOv2 ViT-S encoder code** (`crates/ym-ml/src/depth/`: patch embedding, LayerScale, positional embedding resize, attention blocks, `sam2`'s packed matrix product) and adds the register tokens, the query tokens of the last blocks with the masked attention the model uses at inference, the upscaling blocks (a transposed convolution, a depthwise 3 × 3 convolution, a layer norm over channels), the class and mask heads, and the semantic merge: per class `softmax(class)[:, :−1] · sigmoid(mask)` summed over the queries, and the panoptic post-processing (the masks upsampled bilinearly, then per pixel the best segment). No deformable attention and no custom operator.
- **Input.** The Input's picture as `docs/23` §4.7 renders it (Neutral look, no node, no output sharpening), at 1024 px on the long edge as SAM reads it, resized to the model's own 640 × 640 pre-processing (RG-03 reads the exact resize and padding from the checkpoint's processor and states it here). **Open point for RG-03:** the checkpoint's configuration says patch 16, the training configuration says `vit_small_patch14_reg4_dinov2`; the patch embedding's kernel size in the safetensors file decides, and it sets the token count (1600 or about 2116).
- **Output.** One element label per pixel of a label map at the model's grid, carried to any size by nearest neighbour (§4a.3). A pixel no segment claims is *unlabelled*.
- **Determinism** as `sam2` and `depth`: f32 in a fixed order, no fused multiply-add, `libm` for `exp`, `erf` and `tanh`, every output row computed by one task whatever the pool, so the same picture gives the same labels on every run, on any number of threads and, as far as IEEE 754 goes, on every platform. The app and `ym` give the same masks.
- **Weights and shipping, as SAM 2.1.** `data/models/eomt/` holds `eomt_coco_panoptic_small.f16.safetensors` (about 48 MB; the header carries `format` `yellowmoon-eomt`, `version`, the source URL and SHA-256, the commit of the upstream code and the configuration), `SHA256SUMS`, `MODEL_CARD.md` (`data/models/README.md`) and a `.gitignore` for the weights. They are converted once from the authors' checkpoint (file name and SHA-256 pinned in the script) by `training/eomt/export.py`, which runs in CI (`.github/workflows/models.yml`; Python is for converting and fixtures only), published as an asset of a Yellowmoon prerelease (never "latest", which the updater reads; the name is RG-03's), **fetched by `tools/fetch/eomt.sh` and pinned by SHA-256**, and bundled in the app and next to `ym`. They are **not committed** (`docs/12`). The app is offline and needs no first-run download (`AGENTS.md` rule 7). `NOTICE` carries the MIT and Apache-2.0 notices (item 15; written in RG-01, as the Depth model's item 13 was written by its spec task).
- **Parity.** Fixtures from the authors' code (a script under `tools/`, CPU, f32, one thread, at a pinned commit; test tooling only, never run by the app or `ym`), written once and published beside the weights: the label maps and the per-class scores of 20 public photos. Gates, stated by RG-03 and measured: the class scores within a stated bound, **the element masks the same on at least 99 % of pixels (starting; RG-03 measures and states the bound)**, a second run bit-identical, the same bits on 1, 2 and 8 threads. Without the fixtures the tests print `SKIPPED` and pass; `YM_REQUIRE_EOMT=1` makes that a failure.
- **Cost.** By our count about 60 GMAC at 640 (patch 16) to 90 GMAC (patch 14); an **estimate** of 1.5 to 2.2 s on a 4-core machine, **not measured**; no Apple-silicon timing exists. It runs once per photo, in the background, and once for each reference.

### 4b.2 Classes to elements

The 133 COCO panoptic classes map to **elements** by name. The mapping is the POC's (a table in the Rust code, tested); any class not listed is `other_thing` (the 80 thing classes) or `other_stuff` (the 53 stuff classes) and has no grade of its own.

| Element | Classes |
|---|---|
| `sky` | `sky-other-merged` |
| `vegetation` | `tree-merged`, `grass-merged`, `flower`, `potted plant` |
| `water` | `sea`, `river`, `water-other` |
| `ground` | `sand`, `dirt-merged`, `gravel`, `road`, `pavement-merged`, `mountain-merged`, `rock-merged`, `snow`, `playingfield`, `railroad` |
| `architecture` | `building-other-merged`, `house`, `roof`, `bridge`, `fence-merged`, `wall-brick`, `wall-stone`, `wall-tile`, `wall-wood`, `wall-other-merged`, `window-other`, `platform`, `stairs`, `tent` |
| `interior` | `ceiling-merged`, `floor-wood`, `floor-other-merged`, `curtain`, `rug-merged`, `cabinet-merged`, `counter`, `shelf`, `window-blind`, `door-stuff`, `mirror-stuff`, `pillow`, `blanket`, `towel`, `table-merged`, `light` |
| `person` | `person` |
| `other_stuff` | `banner`, `cardboard`, `fruit`, `net`, `paper-merged`, `food-other-merged` and every other stuff class (ids 80 to 132) |
| `other_thing` | every other thing class (ids 0 to 79) |

**`built` is `architecture` plus `interior`.** `skin` is not a class: it is the next section. `person` minus `skin` is *clothing*, which has no grade but is watched by the walls (§5.4). `unlabelled`, `other_thing` and `other_stuff` have no grade; they stay in the frame's measures.

### 4b.3 Skin and the subject

- **Skin** is MediaPipe Selfie Multiclass (`ym_ml::mediapipe`, shipped, deterministic; never Apple Vision, `docs/23` §4.5) classes `BodySkin` and `FaceSkin`, **intersected with `person`**. It runs on each connected component of `person` that covers at least 0.5 % of the picture, on a square crop of the component's box times 1.15; each pixel takes the class of the highest score. It runs only when `person` covers at least 3 % of the picture; otherwise there is no skin. **If the MediaPipe weights are missing** (a damaged install) there is no skin and the status line says so (§9.3); the POC had a colour-rule fallback and the product does not.
- **The subject** is `person` when it covers at least 3 % of the picture; otherwise the `other_thing` pixels inside the central 60 % of the frame (x and y from 0.2 to 0.8) when they cover at least 1 %; otherwise there is no subject.
- **The two pictures**: the segmenter and MediaPipe run on the photo (§4b.1) and on the reference's analysis picture (§6.5), each once. Every number below is a measure of one picture with its scene.

### 4b.4 The measures (`X.*`)

Measured in **CIELAB with the D65 white** on the picture's sRGB display codes, clamped to 0 to 1, as the fit's own pictures are (§5.1). The photo `P` is measured on the **8192 sample pixels** of §5.1 rendered with the node at its identity (so after the real chain, not before it); the reference `R` on **all pixels** of its analysis picture. An element that is absent has share 0 and every other value 0. The elements below are the six of §4a.1 (`built` = architecture and interior; `skin` the skin plane).

| Measure | Definition |
|---|---|
| `p2`, `p50`, `p98` | L\* at 2, 50 and 98 % of the pixels' ranks, linear between ranks (position `q · (n − 1)`) |
| `iqr_rel` | `(p75 − p25) / max(p98 − p2, 1)` |
| `logsat` | `ln(max(median C\*, 1))` of the pixels of vegetation, water, ground, built, `other_thing` and `other_stuff` when they are at least 1 % of the picture; otherwise of all pixels that are neither sky nor person (unlabelled pixels count) |
| `tint_{s,m,h}_{a,b}` | the neutral cast: the picture's pixels in three L\* terciles by rank (shadows, mids, highlights); in each, among the pixels of the neutral-capable groups (`unlabelled`, `ground`, `architecture`, `interior`, `other_thing`, `other_stuff`, and `person` minus skin) those with `C* ≤ max(8, the C\* at the 15 % rank of that group's pixels)`; their mean `a*` and `b*` (0 if none) |
| `cast_a`, `cast_b`, `conc` | the frame's cast: mean `a*`, `b*` over the pixels that are not skin and have L\* 15 to 80 (when those are at least 10 % of the picture; else all non-skin pixels; else all); `conc` is the length of that mean over `max(mean C\*, 1)`: 0 when colours cancel, 1 when every pixel has one hue |
| `magenta_share` | the share of the picture with `C* > 8` and **Oklab** hue 300° to 355° |
| `subj`, `sep` | `subj` is 1 when the subject and the background (labelled pixels that are not the subject) each hold at least `max(1 %, 30 samples)`; `sep` is the subject's median L\* minus the background's (0 without a subject) |
| `{e}.share`, `.L`, `.C`, `.logC` | the element's share of the picture; its median L\*; its median C\*; `ln(max(C, 1))` |
| `{e}.h` | the circular mean, in degrees 0 to 360, of the unit vectors `(a*, b*)/C*` of the element's pixels with `C* > 5`; 0 if there are none |
| `s{j}.share`, `.logC`, and its hue | 60° sectors of CIELAB hue centred on 0, 60, … 300 (`j = ⌊((h + 30) mod 360) / 60⌋`): over pixels with `C* > 5`, **not skin**, and **not in a sector-excluded element**. Share of the picture; `ln(max(median C\*, 1))`; the circular mean hue. An element is excluded when both pictures hold it at 2 % or more of the picture (sky, vegetation, water, ground, built; first pass, then measured again) |
| `D.{e}.dh`, `D.s{j}.dh` | `wrap(R.h − P.h)` into (−180°, 180°], 0 if either picture has no hue pixel there |

A group's *hue* in these measures is the circular mean of unit vectors; the walls of §5.4 use another (the angle of the mean Oklab vector).

### 4b.5 The network

A **hand-written feed-forward network** reads these measures of `P` and `R` and writes the grading **targets** (`T.*`) and **gates** (`G.*`). Every weight and bias in it was written by hand; nothing is trained. It makes every grading decision: what to copy from the reference, how far, where to stop, and which gates are open. The objective and the solver (§5) make none.

- **The file.** `data/refgrade/network.ywn`, text, ours, Apache-2.0: round 15's file, which is the POC's v14 file unchanged (432 neurons after expansion, 62 targets and 21 gates; the header of the file lists what each round added and why). It ships in the app and next to `ym`. Its SHA-256 and its expanded text are pinned by a test. **A weight changes only by a spec change with the owner's agreement and a pass of §10's acceptance rules**; it is never changed to fit the owner's own photos (§10, *Data rules*).
- **The grammar** (`.ywn`), one neuron per line, in file order, evaluated in f64:
  - `name = relu( bias + w*node - node + … )` or `name = lin( … )`. `relu(x) = max(0, x)`; `lin(x) = x`. The bias is a number, then terms joined by `+` or `-`; a term is `w*node` or `node` (weight 1). A `node` is an input (§4b.4) or a neuron defined **earlier**. Weights and biases are decimal numbers, a leading `-` allowed.
  - `const NAME key=value key=value …` names a row of constants; `template NAME(var)` … `end` is a block of neuron lines; `apply NAME v1 v2 …` expands the block once per value, with `{var}` replaced by the value and `{key}` replaced from the `const` row named by that value (`-{key}` is minus the constant). Lines starting with `#` and blank lines are ignored.
  - Every neuron named `T.*` or `G.*` is an output.
  - **A file that breaks a rule is refused with the line and the name, never patched**: a node not defined earlier, a template value with a missing constant, a duplicate name (also against an input), an unknown activation, a template without `end`, a term that is not a number times a node.
- **Inputs** (names the network may read): for `X` in `P`, `R`: `X.p2`, `.p50`, `.p98`, `.iqr_rel`, `.logsat`, `.subj`, `.sep`, `.cast_a`, `.cast_b`, `.conc`, `.magenta_share`, `.tint_{s,m,h}_{a,b}`, `X.{e}.{share,L,h,C,logC}` for the six elements, `X.s{j}.{share,logC}` for six sectors; and `D.{e}.dh`, `D.s{j}.dh`.
- **Outputs.** Targets: `T.p2`, `T.p98`, `T.iqr_rel`, `T.p50` (the photo keeps its own median), `T.logsat`, `T.tint_{s,m,h}_{a,b}`, `T.cast_a`, `T.cast_b`, `T.wall_deg` (the hue-drift wall's limit, 25° and up to 45° when unify is open), `T.hl_cmax`, `T.white_a`, `T.white_b`, `T.white_tol`, `T.sep`, `T.s{j}.dh`, `T.s{j}.dlogC`, `T.{e}.h`, `T.{e}.logC`, `T.{e}.off`; and the wall parameters `T.purple_ok`, `T.purple_max_share`, `T.purple_rot_deg`, `T.skin_drift`, `T.skin_dlogC_max`, `T.mem.sky_deg`, `T.mem.vegetation_deg`, `T.turn_deg`, `T.turn_share`, `T.turn_cref`, `T.turn_cap`, `T.turn_full`, `T.clip_extra`. Gates: `G.key_mismatch`, `G.unify`, `G.subj`, `G.s{j}`, `G.{e}.matched`, `G.{e}.hue`.
- **What its parts do, in a sentence each.** *Key mismatch*: a low-key reference (median L\* ≤ 23 to 25) on a high-key photo (≥ 40 to 42) means its exposure relationships are content, not grade; they are halved or dropped. *Floor, ceiling, contrast*: move 0.8, 0.8 and 0.85 of the difference, clipped (±12 L\*, ±0.15). *Saturation*: 0.85 of the log ratio, the ratio held to 0.5 to 1.5. *Tint*: the full difference of the near-neutral cast per band, ±20. *Unify*: opens when the reference's whole frame leans to one colour, and then lets the frame cast and the hue-drift wall move. *Sectors*: the palette of pixels no element covers, gated by both pictures holding 2 to 3 % in a sector. *Elements*: matched when both pictures hold the element at 2 to 3 % (sky also only when lit: median L\* above about 18 to 22 in both, so a black backdrop is not a sky); hue moves 0.85 (sky), 0.75 (vegetation, water), 0.65 (ground), 0.45 (built) and 0.35 (skin) of the difference, up to the caps of §4a.1, inside windows of plausible hue for sky (175° to 285°), vegetation (70° to 180°), water (150° to 285°) and skin (30° to 70°); chroma by a power of the ratio (0.9 to 0.5), exposure relative to the frame. *Purple*: a colour may not turn into purple unless the reference holds it. *Whites*: they may take 0.8 of the reference's own highlight tint, no more. *Memory colours*: sky and foliage hold their hue within 10° and 12°, 45° and 40° when matched.

## 5. The fit

**Every number in this section is the POC's round 15, taken from its code** where the POC's notes and its code differed (§16.3). Nothing here is a design target: it is what was measured to work (§16.8). RG-06 writes it again in Rust under the house rules; the POC's code is not merged. A constant changes only by the rule of §10.

The fit **searches** for the node's numbers. The hand-written network (§4b.5) turns the measures of the photo and the reference into targets and gates; an objective measures how far a candidate's picture is from the targets; walls make harm costly; a staged search moves the numbers until the objective stops falling. The fit makes no grading decision of its own.

### 5.1 Pictures and the sample

| | Photo | Reference |
|---|---|---|
| Source | the node's input rendered by the plan (§2.2), once per fit | the loaded reference (§6) |
| Analysis size | **512 px on the long edge** (the deterministic area average of §6.5, in linear light) | the same, never enlarged |
| Pixels read | a **seeded stratified sample of 8192 pixels**, the same for every candidate | **all pixels** of the analysis picture |
| Scene | the label map and skin of §4b, gathered at the 8192 pixels (the soft masks of §4a.3 at the same pixels) | the label map and skin of §4b on the analysis picture |

- **The sample.** A grid of 128 columns by 64 rows over the analysis picture, one pixel per cell: for each cell, row by row, `fx = (gx + u) / 128`, `fy = (gy + u') / 64` with `u`, `u'` the next two draws of **SplitMix64 seeded with `0x5EED_2026_1124`** (`next_u64() >> 11` over 2^53), and the pixel `(⌊fx · W⌋, ⌊fy · H⌋)` clamped to the picture. The seed is a constant: the sample depends on the analysis picture's size and nothing else.
- **The photo's chain-before-the-node is rendered once**; only the node and the steps after it run per candidate (§2.2). `P`, the photo as it stands, is the sample with the node at its identity.
- **Colour.** Each candidate's pixels go through the Output encoded as **sRGB display codes, clamped to 0 to 1** (the range of an 8-bit file) whatever the viewport's space is, then to linear sRGB, XYZ with the Rec.709 primaries and **CIELAB with the D65 white**, and to Oklab (Ottosson's matrices) for the hue walls. The reference goes to the same space (§6.3). *Clipped white* is any encoded channel at 0.995 or more; *crushed black* is L\* below 2.5.
- **The fit ignores the node's own masks** (`masks`): a reference describes a whole picture, so the grade is fitted as if the masks were 1 everywhere, and the masks then limit where it acts. **It does use the element masks** of §4a.3: they are part of the candidate (§14, question 8).
- **The fit runs the node at Strength 100**; Strength is applied afterwards (§3.2).
- **The statistics are of distributions, not of pixel pairs**: the two pictures may show different things (§12).

### 5.2 The parameters

**49 normalised numbers `x`, each from −1 to 1, the identity at 0**: 30 global and 19 local. Each maps to the node's fields; a hand edit may sit anywhere in a field's range, the fit's bounds are these.

| Group | Index | Field | Mapping from `x` |
|---|---|---|---|
| Luma curve | 0 | `grade.curve.luma`, 8 knots at the stops −6, −4, −2, −1, 0, 1, 2, 4 from scene grey 0.18 (input codes: LogC4 of the stop) | `x₀`: the first knot's output stop is `−6 + 2 x₀`. |
| | 1 to 7 | | the slope of segment `k` in stop space is `exp(1.1 x_k)`: `y_k = y_{k−1} + (s_k − s_{k−1}) · exp(1.1 x_k)`; the output code is LogC4 of `y_k`, held between the previous code and 1. The end points (0, 0) and (1, 1) are added. A curve is increasing by construction |
| Channel curves | 8 to 10, 11 to 13, 14 to 16 | `grade.curve.red`, `.green`, `.blue`, 3 knots each at the stops −3, 0, 3 | the output stop is `s + x` (±1 stop); the output code is LogC4 of it, held to 0 to 1; (0, 0) and (1, 1) added |
| Saturation | 17 | `grade.saturation` (manual, percent) | `60 x` |
| `hue_vs_sat` | 18 to 23 | 6 knots at 0°, 60°, … 300° | `σ = 1 + 0.7 x` (0.3 to 1.7) |
| `hue_vs_hue` | 24 to 29 | 6 knots at 0°, 60°, … 300° | `δ = 30° · x` |
| Element grades | 30 + 3k (k = 0 to 5: sky, vegetation, water, ground, built, skin) | `elements.{e}.exposure_ev` | `x` (±1 stop) |
| | 31 + 3k | `.hue_deg` | `cap_e · x` (caps 40, 35, 35, 30, 25, 15) |
| | 32 + 3k | `.chroma` | `1.6^x` for `x ≥ 0`, `0.5^(−x)` for `x < 0` |
| Subject | 48 | `elements.subject.exposure_ev` | `x` (±1 stop) |

A fit writes the knots above and no others; knots a person adds by hand are theirs.

### 5.3 The objective: the look terms

The **objective is a weighted sum**; every term is `term(err; scale, dead) = ((max(|err| − dead, 0)) / scale)²`. Measures are those of §4b.4 taken on the candidate (`m.*`); targets are the network's. `w_term` is sky 1.5, vegetation 1.2, water 1.0, ground 1.0, built 0.6, skin 2.0. `√share` is the square root of the photo's own share of the element, `P.{e}.share`.

| Term | Error | Scale | Dead | Weight |
|---|---|---|---|---|
| floor | `m.p2 − T.p2` | 2 | 1 | 1.0 |
| ceiling | `m.p98 − T.p98` | 2 | 1 | 1.0 |
| contrast | `m.iqr_rel − T.iqr_rel` | 0.03 | 0.01 | 1.0 |
| anchor | `m.p50 − T.p50` | 1 | 5 | 4.0 |
| tint (6 channels) | `m.tint − T.tint` | 3 | 1 | shadows 1.0, mids 0.7, highlights 1.0 |
| saturation | `m.logsat − T.logsat` | 0.05 | 0.03 | 1.0 |
| frame cast (`a*`, `b*`) | `m.cast − T.cast` | 2 | 1 | 1.2 · `G.unify` |
| sector hue, per open sector `j` | the mean over the sector's pixels (by `P`'s hue) of `wrap(candidate hue − P hue)`, minus `T.s{j}.dh` | 8° | 2° | 0.8 · `G.s{j}` · √(P.s{j}.share) |
| sector chroma | `ln(max(median C\*, 1))` of those pixels minus `P`'s, minus `T.s{j}.dlogC` | 0.08 | 0.03 | 0.8 · `G.s{j}` · √(P.s{j}.share) |
| element hue (the candidate has hue pixels) | `wrap(m.{e}.h − T.{e}.h)` | 10° | 3° | `w_term · √share · G.{e}.hue` |
| element chroma | `m.{e}.logC − T.{e}.logC` | 0.10 | 0.03 | `w_term · √share · G.{e}.matched` |
| element exposure | `(m.{e}.L − m.p50) − T.{e}.off` | 3 | 2 | `w_term · √share · G.{e}.matched` |
| subject | `m.sep − T.sep` | 3 | 2 | 1.0 · `G.subj` |

**The regulariser** is `0.5 · mean(x_global²) + 0.5 · mean(x_local²)` over the 30 global and the 19 local numbers: the fit prefers the smallest grade that reaches the targets.

**The full objective** is the sum of the look terms, the walls of §5.4, the regulariser and `3.0 ·` the clip term of §5.4. **The look-only objective** is the same sum **without the walls and without the clip term** (the regulariser stays); the second search of §5.5 uses it. A candidate the chain refuses (a non-finite result) gets a clip term of 10⁶ and is worse than anything.

### 5.4 The walls

The walls make harm costly. They are not grading decisions: they are what the owner's rule "film grades don't turn content colours into purple" and its like became. They read the network's wall parameters (`T.*`). In every wall *the hue of a group* is the **Oklab hue of the group's mean Oklab (a, b) vector** (`atan2` of the mean `b` and `a`), measured on the pixels of the group in `P` and on the same pixels in the candidate; chroma thresholds stay in CIELAB `C*`. `wrap` folds a difference into (−180°, 180°]. `sq(x) = x²`.

**Groups.** An *element group* is one of the six elements, taken only when the photo's share is at least 2 % and the candidate has pixels in it; *clothing* is `person` minus skin (at least 2 %); a *sector group* is `P`'s pixels of a sector with at least 4 % share; the *whites* are `P`'s pixels with L\* ≥ 80 and C\* < 10 (at least 2 % of the sample); the *dark colours* are `P`'s non-skin pixels with L\* < 35 and C\* ≥ 2 (at least 2 % of the sample). A group counts in a wall only when the length of `P`'s mean (a\*, b\*) vector over it is at least 2.5. The whites' three walls apply only when the whites group exists (at least 2 % of the sample).

| Wall | Formula | Constants |
|---|---|---|
| **Hue drift** (per group) | `2 · sq((\|wrap(Δh)\| − limit)₊ / 5)` | `limit = T.wall_deg` (25°, 45° when unify is open); for an element with `G.{e}.hue > 0.5`, `max(T.wall_deg, cap_e + 5)` |
| **Memory colours** (sky, vegetation) | `6 · sq((\|wrap(Δh)\| − T.mem.{sky,vegetation}_deg)₊ / 2)` | limits 10° and 12°, 45° and 40° when matched |
| **Whites: chroma cap** | `3 · sq((C̄ − T.hl_cmax)₊ / 2)` with `C̄` the **mean per-pixel C\*** of the photo's non-skin whites with L\* ≥ 80 and C\* < 8 | opposite tints do not cancel |
| **Whites: direction** | `d` = the **mean per-pixel distance** of those whites' (a\*, b\*) to (`T.white_a`, `T.white_b`); if `d > T.white_tol`: `3 · sq((d − T.white_tol) / 2)` | |
| **Whites: hue** | the hue-drift wall on the whites group, when `P`'s mean-vector C\* there is at least 4 | |
| **Purple, per group** | in the **Oklab band 300° to 355°**: a group whose candidate hue is in the band adds `3 · sq((\|wrap(Δh)\| − T.purple_rot_deg)₊ / 5)`; a group not in the band at either end but approaching it (within 15° of the band, and nearer than in `P`) adds `1 · sq((15 − dist) / 5) · (1 − T.purple_ok)` | `T.purple_rot_deg` 10°, 45° when the reference holds purple |
| **Purple, pixels** | the share `s` of the sample's non-skin pixels whose candidate C\* ≥ 4, Oklab hue in the band, and either `P` C\* < 2 or a hue turn of at least `T.purple_rot_deg`; if `s > T.purple_max_share`: `6 · sq((s − T.purple_max_share) / 0.01)` | `T.purple_max_share` 1 %, 25 % when the reference holds purple |
| **Skin: absolute** (Skin protection on; skin ≥ 2 %) | skin's mean-vector CIELAB hue outside 30° to 70° by `eh`, chroma outside 12 to 38 by `ec`: `12 · (sq(eh / 3) + sq(ec / 3))` | |
| **Skin: drift** (on) | `12 · sq((\|wrap(Δh)\| − T.skin_drift)₊ / 3)` | `T.skin_drift` 6°, 10° when the reference's skin is matched with reliable hue |
| **Skin: chroma** (on; skin ≥ 2 %) | `r = ln(max(C_cand, 1) / max(C_P, 1))` over the mean-vector chroma; `12 · sq((r − T.skin_dlogC_max)₊ / 0.05)` | `T.skin_dlogC_max` 0.12, 0.30 when matched |
| **Sky chroma** (sky ≥ 2 %) | `limit = max(1.3 · C_P, exp(T.sky.logC))`; `2 · sq((C_cand − limit)₊ / 2)` over the mean-vector chroma | |
| **Content turn budget** | see below | `T.turn_*` |
| **Clip** | see below | `T.clip_extra` |

**The content turn budget (P2).** A strong colour keeps its hue family; big hue moves belong to weak colours, which is what a cast is. For each non-skin sample pixel with `P` C\* = C ≥ 8, with `Δh` the absolute Oklab hue turn between `P` and the candidate:

```
allowed(C) = clamp( asin( min(1, T.turn_cref · sin(T.turn_deg) / C) ), T.turn_deg, max(T.turn_cap, T.turn_deg) )     (C = 0 gives the cap)
w          = clamp( (Δh − allowed(C)) / (T.turn_full − allowed(C)), 0, 1 )          (a step if T.turn_full ≤ allowed)
s          = Σ w / N                                                                 N: all sample pixels
cost       = 6 · sq((s − T.turn_share) / 0.01)   when s > T.turn_share
```

`T.turn_deg` is 20° (30° when unify is open), `T.turn_share` 0.03, `T.turn_cref` 25, `T.turn_cap` 45°, `T.turn_full` 60° (where a turn is a reversal). The allowed turn is:

| C\* | 8 to 12 | 15 | 20 | 25 and above |
|---|---|---|---|---|
| unify shut | 45° | 35° | 25° | 20° |
| unify open | 45° | 45° | 39° | 30° |

A strong colour turned 30° counts as a quarter of a pixel, 40° as half, 60° as a whole one. A frame where 10 % of strong colours turn 30° costs nothing; a frame where 4 % reverse costs as before.

**The clip term (P5).** A grade does not destroy detail the photo has. With `hi` the share of pixels with any encoded channel ≥ 0.995 and `lo` the share with L\* < 2.5, `p` the photo's own share, `r` the reference's:

```
allow(p, r)  = max( p, min( r, p + T.clip_extra ) )              T.clip_extra = 0.02
over(o, a)   = sq( (o − a − 0.005)₊ / 0.01 )                     the slack is 0.005, CLIP_SCALE is 0.01
clip         = over(hi_cand, allow(hi_P, hi_R)) + over(lo_cand, allow(lo_P, lo_R))
```

The reference may license at most two points of new clipping, never more than the reference itself has: a black night sky is its content, not a licence to crush a daylight photo. The clip term enters the objective at weight 3.0.

### 5.5 The search

**Golden section, one parameter at a time, in stages, with fixed counts.** One coordinate step `golden(i, half, n)` searches `[x_i − half, x_i + half]` clipped to [−1, 1]: two interior points at 0.618 and 0.382, then `n` more steps, every evaluation of the full chain on the 8192 pixels; it keeps the **best of all points seen, the start included**, so a step never raises the objective and a tie keeps the start. Nothing needs to be differentiable.

1. **Global stages**, on the full objective, each parameter of the stage in index order, each stage starting from the last:

   | Stage | Parameters | Pass 1 (half, n) | Pass 2 (half, n) |
   |---|---|---|---|
   | tone | 0 to 7 | 1.0, 8 | 0.4, 6 |
   | colour per band | 8 to 16 | 1.0, 7 | 0.4, 6 |
   | saturation and `hue_vs_sat` | 17 to 23 | 1.0, 7 | 0.4, 6 |
   | `hue_vs_hue` | 24 to 29 | 1.0, 7 | 0.4, 6 |

2. **The need test.** A matched element (`G.{e}.matched > 0.5`) whose binary mask covers at least 2 % of the picture gets a local grade only if, after the global stages, **its error is over a threshold**: the hue error `|wrap(m.{e}.h − T.{e}.h)|` is more than 6° (and `G.{e}.hue > 0.5`), or the log-chroma error is more than 0.14, or the exposure error `|(m.{e}.L − m.p50) − T.{e}.off|` is more than 4 L\* (and the exposure target is open, `G.{e}.matched − G.key_mismatch > 0.5`). The subject gets its exposure grade when `G.subj > 0.5`, its mask covers at least 2 %, and `|m.sep − T.sep|` is more than 4. **An element that does not pass keeps its parameters at exactly 0 and they are not searched.**
3. **Local stage**, for the needed elements in the order sky, vegetation, water, ground, built, skin, subject, their three parameters (the subject's one): pass (1.0, 7), then pass (0.4, 6).
4. **Joint polish**: the 30 global parameters and the needed local ones, one pass (0.25, 6).

**By count from the stages above** (not measured): one search evaluates about 770 candidates with no local grade, plus about 50 for each needed element and 24 for the polish of its parameters; the fit is three searches (below), so on the order of 2400 candidates, each one the chain on 8192 pixels and a set of measures.

**Two starts, keep the better.** A search from the neutral grade stops in the first basin it reaches, and a grade that carries a strong cast lies far from neutral, behind walls the path must cross. So the fit searches **twice** and keeps the better result:

1. **From neutral**: the search above on the full objective, from `x = 0`.
2. **From the look**: first the same search on the **look-only objective** from `x = 0`; then the search on the **full objective** starting from that result.
3. **Keep the one with the lower full objective. A tie keeps (1).**

This is a property of the search, not of any photo. It cost the POC about five times the time of one search (§16.7) and won on 333 of 728 dev fits and 461 of 962 test fits.

**A result no better than the identity is the identity.** If every parameter is still 0 at the end, the fit writes the identity and the status says "The photo is already close to the reference. Nothing was changed." (§9.3). That is not a failure. **A non-finite value, a segmenter or classifier error that cannot be recovered, or a search that cannot evaluate the chain is a failure** and writes nothing (§9.3).

### 5.6 Skin protection

- **What the switch does.** `skin_protection` on adds the three **skin walls** of §5.4 (absolute, drift, chroma) to the objective. Off, they are not added; the skin element's own wall as an element group, its grade caps (±15°) and every other wall stay. It is read **at fit time**: changing the switch changes nothing until the next Fit (§9.2).
- **Source: MediaPipe, never Apple Vision** (§4b.3). Apple Vision is not reproducible across Macs (`docs/23` §4.5).
- **No skin found** (no person, or the weights missing): the walls are not added and the status says "No skin was found, so Skin protection did nothing." when the switch is on.
- **Round 15's skin is not final.** Skin (H2, 7.9 % of held-out fits) and tinted whites (H4, 7.8 %) are nearly all of the harm that is left; RG-13 is the quality round on them (§16.8).

### 5.7 Determinism, progress and time

- **CPU, f64 for the measures and the objective, a fixed order of operations, no data-dependent threading.** A candidate's pixels are split in fixed chunks of 256 whatever the thread count, each chunk computed by one task, and every statistic is reduced in pixel order, so 1 and N threads give the same bits (as the CPU chain's tiles do). The skin plane and the segmenter are deterministic. The same photo, reference, settings and version give the same fitted numbers on every run (rule 8).
- **Progress and Cancel.** The fit reports where it is: the scene (segmenter and skin for the photo and the reference, once each and then kept), then search 1 of 3, 2 of 3, 3 of 3 and the stage within it. Cancel is checked between candidates, takes effect within a fraction of a second, and **changes nothing**.
- **Time: several seconds on a Mac with all cores is accepted** (the owner, 2026-11-25). The POC measured **about 30 s per fit on one Linux thread** for round 15 and about 6 s for the single search of rounds 11 to 14 (§16.7); those are Linux, single-thread figures and not Mac figures. The segmenter's cost is an estimate (§4b.1). RG-11 is the speed task and RG-H3 the owner's timing. The order of remedies, the quality gate re-run after each: evaluate the candidates across all cores (the two independent searches in parallel, and the pixels of a candidate across chunks), cheaper chain steps for the candidate, the scene cached per photo and per reference, and only then fewer evaluations.

## 6. References

### 6.1 Image files

- **Formats: JPEG, PNG, TIFF**, 8 or 16 bits per sample, RGB or grey (a grey picture matches as black and white; the status line says so). Decoders: `zune-jpeg`, `png` and `tiff`, already in the workspace.
- **No HEIC.** libheif is LGPL (`docs/14`, rule 1). HEIC and every other format get the "not supported" error (§9.3), which says to save as JPEG or PNG first. No RAW file is accepted as a reference: a raw's picture depends on its settings; use a photo in the strip, or export it first.
- **Orientation is applied** (the EXIF tag of a JPEG or TIFF, values 1 to 8), because the scene model expects an upright picture (§15, decision 18).
- **A file is only read.** Rule 5 holds: nothing is written, moved or renamed. The file is read once, reduced and let go of (§6.5).

### 6.2 What a reference is read as

An image gives a grid of display-referred code values in its own colour space. §6.3 turns them into display-linear sRGB, the space the fit measures in (§5.1); §6.5 reduces them. **The reference is used at run time and never stored, uploaded or cached** (§6.6).

### 6.3 Colour handling

1. **Untagged files are taken as sRGB** (the sRGB curve and primaries, IEC 61966-2.1). This is a guess and is listed as a risk (§12): a wrongly tagged or untagged picture shifts the whole fit.
2. **Tags that are read**, in this order: an embedded ICC profile (JPEG `APP2`, PNG `iCCP`, TIFF tag 34675); PNG `cICP` (the code points our exports write: sRGB 1/13, Display P3 12/13, Rec.2020; `docs/09`); PNG `sRGB`. The others (`gAMA`, `cHRM`) are not read.
3. **Profiles that are handled: RGB matrix/TRC profiles**: the colorant tags `rXYZ`, `gXYZ`, `bXYZ` (already in the profile connection space, D50) and the curves `rTRC`, `gTRC`, `bTRC` as `curv` (the identity, a gamma or a table) or `para` (types 0 to 4). That covers sRGB, Display P3, Adobe RGB, Rec.2020 and **the "Display" profile macOS puts in screenshots** (the research set had one; RG-05 adds it, and the others, as fixtures). **A profile with a LUT (`A2B0`), a CMYK or a named-colour profile is not handled.**
4. **A profile that cannot be read** (damaged, or not a matrix/TRC kind) is **not an error**: the file is taken as sRGB and the card says so ("The picture's colour profile couldn't be read, so it is taken as sRGB.", §9.3). Stopping would block a picture that is usually fine, and the person sees the note.
5. **The conversion** is in f64, in a fixed order: code values → linear by the curves → XYZ D50 by the matrix → **linear sRGB** (the Bradford adaptation from D50 to D65, then the sRGB matrix), clamped to 0 to 1 as the fit's own pictures are (§5.1). The fit's CIELAB (D65) is made from that. No chromatic adaptation is needed for an ICC profile to reach D50 (its colorants are D50); an untagged picture is already sRGB, and a `cICP` picture goes from its primaries' D65 white to sRGB directly. Colours a wider profile has outside sRGB are clamped.
6. **A 16-bit picture is read at its full precision**; 8-bit pictures are read as they are (no dithering, no smoothing).

**Decision: our own minimal ICC reader, not a crate** (§14, question 10). The reader is a few hundred lines in `ym-color` (the tag table, `XYZ `, `curv`, `para`), in f64 and a fixed order. Why: (1) **determinism**: a general colour-management crate may choose SIMD or f32 paths by CPU, and the fit's bits must not depend on them (rule 8); (2) **scope**: we need exactly matrix/TRC, and refuse the rest; (3) **no new dependency** for `cargo deny` and `docs/14`; (4) our own export already writes such profiles (`ym-export` `icc.rs`: sRGB, Display P3, Rec.2020), so the layout is known and the reader is tested against them. `moxcms` is listed in `docs/14` among the permitted dependencies (MIT/Apache-2.0/BSD) but is **not in the workspace manifest today**; if the owner prefers a crate, it is the candidate, with the determinism check as an acceptance test. The **oracle** for the reader's tests is an independent tool in CI only (RG-05 picks: `moxcms` as a dev dependency, or littlecms' command-line tool), never a dependency of a shipped crate.

### 6.4 A photo in the filmstrip

- **It is rendered through its own current settings** to a display picture at the scene-picture size (§6.5): the same chain, the same Output, the same C13 space the photo of §5.1 uses, so no profile is involved. The result is a snapshot (§1.3).
- **The picture is held in the core, in memory, for the session**; it is the reduced pictures (§6.5), not the file and not a link. The page asks the core to set it and gets back a thumbnail and a short label.
- **Rules.** The photo being graded cannot be its own reference ("That's the photo you're grading. Choose another."). A photo that is still developing, pending or failed cannot be chosen (its tile is disabled with "This photo isn't ready yet."). The chooser is a mode like Pick colour: Esc ends it.

### 6.5 Letterbox, size and reduction

- **Letterbox removal: uniform bars only.** Rows and columns are examined **from each edge inward**. A row is part of a bar when the **standard deviation of its 8-bit codes (the three channels together) is below 2.0 and their mean is below 16**; a column is tested the same way between the bars already found at the top and bottom. The bar is the run of such lines from the edge and ends at the first line that fails; **a bar is never more than 35 % of a side**. These two thresholds and the 35 % are the POC's (§16.5).
  - **A dark row is not a bar.** Night frames have many dark rows that are picture. The rule needs **both** a standard deviation below the noise of a real picture **and** a mean near black. The POC had nothing beyond those two, and it was not tested on night frames: **RG-05 adds a minimum bar width, a step at the bar's inner edge (the first lines inside differ from the bar by more than noise) and a margin of a few pixels against JPEG ringing, sets them from public data and tests them on the night-frame cases of §10**, writing the constants into §16.5.
  - A picture in which every line qualifies has no bars removed and then fails the "empty" check below.
  - The card says "Black bars were removed from the reference." when it removed any.
- **Size.** After removal, the picture's **short side must be at least *N* pixels** or the reference is refused ("too small", §9.3). *N* is not set (§16.6). A picture with no area inside its bars is refused ("empty", §9.3).
- **Reduction.** A deterministic area average (a box filter with exact fractional weights) **in linear light**, in f64, to two pictures: the **scene picture** (long edge at most 1024 px) that the segmenter and MediaPipe read, and the **analysis picture** (long edge at most 512 px) that the fit measures (§5.1). A smaller picture is not enlarged. The core holds both for the session (§6.6), about 8 MB.

### 6.6 Copyright and what is never kept

References are supplied by the person, used at run time and never stored: **not in the settings, not in a preset or an album, not in a log, not in the XMP, not in a cache.** The core holds the reduced pictures and the scene made from them in memory for the session and drops it when the node is deleted, the photo is closed or the app quits. A film still is usually copyrighted; Yellowmoon reads it as the person's own tool does, and keeps no copy. **Evaluation stills are local only** (§10.1): never committed, never cached by CI, never uploaded.

## 7. Rendering

### 7.1 The plan and the CPU reference

- **The plan** (`docs/23` §5.2). The node is a corrector-like per-pixel step with a picture input, a key input and a key output. It fuses into the colour pass like a corrector's grade. `StepKind` gains `ReferenceGrade` (`ym_color::plan`, `pipeline::schedule::step_kind`; their exhaustive matches fail to compile without it): per pixel, reads the picture and the planes of the live elements. A step at its identity (every step at its identity after `effective`, §3.2) is skipped.
- **The CPU reference** is `ColorChain::run_nodes` calling `PreparedNode::apply` (`ym_color::display::node`): the corrector's steps in their order from `effective` values, then `hue::apply` in f64 (§4.2), then the element grades in f64 (§4a.2) through the planes, then the node's mask blend. `color_plan` fills `trim`, `grade`, `warp` and `curves` through `grade_params` as Colour Uniform's does, plus the baked hue tables, the element parameters and the element planes (§4a.3). `nodes_key` (`pipeline/color.rs`) adds the node's stored values and `strength` to the key, or an edit would not invalidate the frame; the element planes' key is the photo's scene key (the segmenter's input and rules version) and the radius.
- **Show mask, thumbnails, histogram, scatter, pixel readout, ancestors, dead nodes**: as a corrector's. The node's thumbnail is the picture after it.

### 7.2 The device: the least change

The colour kernel binds all 8 storage buffers (K3: 8 storage buffers and 1 uniform; `docs/17`). **No binding is added.** The llvmpipe rule (`docs/17`, *A lavapipe miscompile on AMD Zen 4 and 5*, `docs/STATUS.md`: the 2026-11-20 and 2026-11-21 miscompiles of `run_nodes` after small control-flow changes) applies in full: **change `run_nodes`, `node_apply` and `node_mask` as little as possible, prefer host packing, and pass `diag-alf` before a release.** The design:

1. **The hue tables are packed by the host into binding 2**, the curve-table buffer (K3 is full, so nothing new is bound). Today binding 2 holds, in plan order, four 1024-sample `f32` tables for every node that runs a curve, and a record names its first table. **After all of the curve tables**, the host appends **two tables of 1024 `f32`** (`hue_vs_hue` in radians, `hue_vs_sat` as a multiplier) for each live Reference Grade node whose hue curves are not the identity, in plan order. Because they come last, **no existing node's table offset moves and the buffer's bytes for every existing graph are unchanged.** `ChainTables::prepare` writes them from the f64 tables of §4.2, rounded once to `f32`.
2. **One record flag and one row.** A new bit of `ctl.x`, **`N_HUE`**, set by the host for a live node with a non-identity hue step. The first hue table's index is written to a row the node's kind never uses: **`R_HUE` = `R_JOIN + J_IN`, lane x** (the mixers' rows; a Reference Grade node is neither a mixer nor a Colour Uniform node, and `docs/32` §12a.4 already moved the pull's rows there). The constant is written by the host from `offset_of!(NodeRecU, join)`; **the record stays 1328 bytes**, `REC_ROWS` is unchanged, and a test pins the rows and the size.
3. **The fewest new branches.** The kernel gains **one `if` on `N_HUE`** and, inside the new function, **no branch on the data**: the hue is read as `h = select(0.0, atan2(q, p), r > 0.0)` (a `select`, the shape that fixed the Blur node's miscompile), so the step has no `r == 0` branch. The step is: `p`, `q`, `r`; `h`; two table reads with linear interpolation and wrap-around; one `sincos`; the rotation and scale; the add back along `e1` and `e2`. It resembles the Colour Uniform pull (one `atan2`, one `sincos`, one square root; `docs/32` §9), without `exp2` or `log2`.
4. **The shape, in order of preference** (each tried on the same runners; the first that passes is built):
   - **H1 (recommended): an early return, the shape of W1** (`docs/32` §12a.4, `docs/17`). Below the `N_SKIP` return in `node_apply`: `if ((bits & (N_HUE | N_ELEM)) != 0u) { return rg_apply(ni, v_in, bits); }`, and `rg_apply` is a **copy** of `cu_graded_apply` with the pull replaced by the hue step at the end: trims, C9, C9b and C11, then `hue_node`, then the element step (item 5). **The body of `node_apply` below the early return, `run_nodes`, `node_mask` and the mask code are byte-identical to main**, so every other node runs the code it did. This is the shape that passed on all four runners for the graded Colour Uniform node; it costs a second copy of the four steps in WGSL, kept honest by the parity test.
   - **H2: a block last in `node_apply`**, after C11, in the shape of `uniform_node`'s block, guarded by `(bits & N_HUE) != 0u`. Smaller, but it changes `node_apply` itself.
   - **H3: refuse on the device.** If neither passes, the device refuses a Reference Grade node with non-identity hue curves or element grades (`InvalidInput`, which the render takes as it takes any refusal: the CPU makes the frame and says why, `RenderReport::cpu_reason`). The node, the fit and the Hue tab stay; the owner decides whether a slow node is acceptable. The remedy order is that of the 2026-11-21 round.
5. **The element grades** (§4a). The same early return carries them. For each live node, for each element whose effective grade (§3.2) is not the identity: four `f32` (the exposure shift in LogC4 code, the `cos` and `sin` of the rotation, the chroma ratio; the subject has its one), written by the host into binding 2 **after the hue tables**, so again no existing offset moves; a second record flag, **`N_ELEM`**, in `ctl.x`, and the first block's index in lane y of the row `R_HUE`; the early return tests `(bits & (N_HUE | N_ELEM))`. An element's soft mask is a **raster plane** of the window (binding 7, `docs/17` §11a, the mechanism of the Brush row), `u16`, quantised by the same rule on the CPU reference so that the two read the same numbers. The step is a short loop inside `rg_apply`, in the order of §2.3, whose bound is a host integer, with no branch on pixel data beyond the plane read. **RG-08 settles the packing, the plane table and the record's rows, and states them in `docs/17` §11a;** if a row or a binding cannot be found without changing the record, H3 is the fallback (the device refuses the element step) and the owner is asked.

### 7.3 Parity and the release gate

- **A CPU f64 reference and a parity test** (`crates/ym-gpu/tests/gpu_reference_grade.rs`): the device against `PreparedNode::apply` at **2e-4 in LogC4** (the bound of `docs/02`), and one 10-bit code after C12 and C13; the identity cases (empty curves, Strength 0, a node with no hue step) **exact**. Cases, on lavapipe and on the Mac GPU CI (Metal, fast-math, K8): `hue_vs_hue` alone, `hue_vs_sat` alone, both, with trims, grade and curves, each element alone and all seven, planes of mask 0, partial and 1, a pixel in two elements, a pixel at `r = 0` and near grey, colours at 0°, 60° and 359.99° (the seam), negative codes, a node after a node, a node with a mask. The measured worst error is written into `docs/STATUS.md`; none is claimed here.
- **K8 (Metal's fast-math).** Nothing is decided by a rounding that changes the result beyond the gate: the `select` on `r > 0` chooses between results that agree to rounding near `r = 0`, because the step scales and rotates a vector of length `r`. The flag, the row and the table index are integers written by the host.
- **`diag-alf` must pass before any release that contains the change.** RG-08 runs the workflow on the four GitHub runner types (AMD EPYC 9V74 and 9V45, EPYC 7763 and the Xeon Platinum 8573C), comparing `main`'s commit with the RG-08 commit, counting the failures of `gpu_alf_look`, `gpu_masks`, `gpu_chain`, the graded Colour Uniform test and the new `gpu_reference_grade` in three runs each. **The gate is 0 failures on every runner**, reported in `docs/STATUS.md` as the earlier rounds were. The Mac GPU CI runs the same suites. A failed gate is reported, never loosened.
- **Speed.** The hue step's cost per megapixel is measured in RG-08 on lavapipe and, as a Mac figure, in RG-H3; the starting budget is the Colour Uniform pull's (`docs/32` §9), which the step resembles. No figure is claimed here.

## 8. `ym`

- **The flags.** `ym render <photo> --reference <file> [--reference-strength <0–100>] [--no-skin-protection] [--reference-node <id>]`. The flags set fields of the node and run the fit; they are not new settings. The strength is in percent, as `--intensity <0–100>` is (`docs/10`); it sets `strength`.
- **What the fit does.** `--reference` fits **exactly as the app does** (§5): the same analysis size, loader, ICC reader, letterbox rule, scene model, network, chain and search. A graph with **one** Reference Grade node is fitted; a graph with **none** gets one added after Main (as Add node does in a default graph); with **several**, `--reference-node <id>` is required (exit 2: "The graph has more than one Reference Grade node. Say which with --reference-node."). The fit overwrites the node's fitted fields as Fit again does (§3.3). With several photos on the command line each is fitted to the same reference.
- **Bit-identity.** For **the same reference picture**, the same photo, settings and version, `ym` and the app write the same fitted numbers and so the same bytes (rule 6). **An export from the app pins the fitted values in `ym:Settings`**, like the Auto answers (`docs/19`, `docs/22`), so `ym render photo --settings exported.xmp` (with no `--reference`) renders the same bytes as the app's export. A reference from the filmstrip is a display picture held in memory, not a file, so `ym` cannot take it; the pinned settings reproduce its fit.
- **The scene model.** `ym` runs the segmenter and MediaPipe as the app does (§4b). A graph whose Reference Grade node has element grades, or a `--reference` fit, needs the scene model; without it `ym` stops with `error.kind` `mask` and the sentence "Node 3 has element grades, and the scene model is missing from this install. Reinstall Yellowmoon." (the node's number filled in).
- **The reference is a file only.** Errors have the app's texts (§9.3) after "The reference can't be used:" and exit 2 as an unusable `--settings` file does. `ym render --json` and `ym info --json` print the node in `settings` as the document holds it, with no reference.
- **The help** names the kind in the closing paragraph ("other nodes (correctors, Colour Uniform nodes, Reference Grade nodes, Blur nodes, mixers, …)") and `docs/10` quotes it and gains a *Reference Grade* section and the example of §3.1 as a JSON block that `crates/ym-cli/tests/docs.rs` reads and validates. The snapshot `cli__help_render` changes with it. **RG-02 changes the paragraph and the snapshot with the kind (the docs test compares them); RG-09 adds the section and the example.** Until RG-09 no CLI example of this node is in `docs/10`, so `crates/ym-cli/tests/docs.rs` has no command to run against unbuilt code.
- **Version.** `ym --version`, `ym doctor` and `ym info` print the new pipeline version from RG-02 (§3.8).

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
| **Elements** | **new**: seven cards, Sky, Vegetation, Water, Ground, Built, Skin and Subject (§9.4a) |
| **Mask** | the corrector's Mask tab, as before |

Seven tabs: 40 letters in a 400 px panel; the simple view's seven tabs are 33 and fit (`docs/32` §12a.2), so this row is 7 letters wider and **may clip**; RG-12 tests that no label is clipped at 1440 × 900, 1180 × 760 and 1800 × 1100, and the fix, if one clipped, is the row's type size for this kind, never a shortened label. ⌘1 to ⌘7 select the tabs in the order of the row; the remembered tab is per kind. The tab components read the node's numbers through `resolvedGraded` (Colour Uniform's widening of `resolvedCorrector`), which learns this kind.

### 9.2 The Reference tab

One card, **Reference**:

1. **The thumbnail** of the reference (session only; empty until chosen), with its name under it (§3.7).
2. **Choose reference…** (a button that opens a menu): **Image file…** (the system open dialog, JPEG, PNG and TIFF) and **Photo in the strip** (the filmstrip becomes a chooser, like Pick colour: "Click a photo in the strip. Esc cancels."; the photo being graded and the unready ones are disabled, §6.4).
3. **Fit** (glasscn button; "Fit again" after a fit). Disabled without a reference, with the tooltip "Choose a reference first." While fitting it shows a **progress bar** with the step ("Finding what is in the pictures…", then "Fitting… 1 of 3", 2 of 3, 3 of 3) and a **Cancel** (Esc); a cancel changes nothing.
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
| making the scene | "Finding what is in the pictures…" |
| fitting | "Fitting… 2 of 3" (the search's number; with the progress bar and Cancel) |
| fitted | "Fitted to the reference." |
| nothing to change | "The photo is already close to the reference. Nothing was changed." |
| edited since the fit | "Edited by hand since the fit. Fit again to start over." |
| **look changed since the fit** | "The look changed since the fit. Fit again to match it with the new look." |
| values, no reference (after a restart, a paste or a preset) | "The values are kept. Choose a reference to fit again." |
| spatial nodes after the node | "Nodes after this one that blur or add film aren't part of the fit." (a note) |
| element grades, the scene model missing | "The scene model is missing from this install, so element grades aren't applied. Reinstall Yellowmoon." (a note) |

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
| the scene model is missing (Fit) | "The scene model is missing from this install. Reinstall Yellowmoon." |

Strings are in `src/` with the rest of the copy; no exclamation marks. A test pins each text.

### 9.4 The Hue tab

Two cards on the Curve card's plot widget, **taught a periodic x axis** (0° to 360° with the rainbow track the Hue slider of Colour Uniform uses): **Hue vs hue** (y: −180° to +180°, a detent at 0) and **Hue vs saturation** (y: 0 to 8, a detent at 1, the axis drawn on a log or a linear scale chosen in RG-07). Points are added by a click on the plot, dragged, removed by Delete or a double click, and moved by the arrow keys; at most 16 per curve ("A curve can have up to 16 points."). Each card has its own Reset. The plot's accessible names and the keyboard path follow the Curve card. The fit's knots are at fixed hues but the person's points are free.

### 9.4a The Elements tab

Seven cards on the corrector's slider rows, in the order of §2.3: **Sky**, **Vegetation**, **Water**, **Ground**, **Built**, **Skin** and **Subject**. Each card has **Exposure** (−1 to +1 EV), **Hue** (± the element's cap, in degrees) and **Chroma** (a ratio, 0.5 to 1.6, shown as a percentage of the photo's own, 50 to 160 %); the subject has Exposure only. Each card has its own Reset. When the element covers under 2 % of the photo, its card says **"Not found in this photo."** and its sliders still work (a grade there acts on nothing); a fit leaves it at the identity. RG-07 draws whether a card can show its element's mask like Show mask does. While the scene is being made the cards show **"Looking at the photo…"**.

### 9.5 The rest

- **Masks work as on any node.** The Mask tab is the corrector's; the node's Show mask is the ordinary one.
- **Thumbnails.** The card's thumbnail is the picture after the node. The reference's thumbnail is the Reference tab's.
- **IPC.** The commands are additive and named in RG-12 (`docs/01`): choose a file, choose a strip photo, `fit_reference(node_id)` (progress as events), `cancel_fit(node_id)`, `clear_reference(node_id)`; the core holds the reference per node for the session, the page holds only a thumbnail and a label, and the mock holds a stand-in picture so that the e2e tests run (the mock's fit returns a fixed patch). Nothing the page does saves the reference.
- **Accessibility** (`docs/28`): the glass audit and axe run on the new states; the status line announces; the chooser mode is keyboard-reachable; the Hue plots have the Curve card's keyboard path; the progress bar is named and its step text is announced politely; the Elements cards are rows of sliders as the corrector's are.

## 10. Tests, evaluation and gates

Every gate must pass in CI (or the named job). **A failed gate is reported, never loosened** (`AGENTS.md`). The numbers of the quality gates are the POC's and the owner's rule of 2026-11-25 (§16.8); changing one needs the owner.

| Area | Test | Gate | Task |
|---|---|---|---|
| Format and rules | the fields, defaults and clamping; hue points (modulo, duplicates, limits, 16 points); the `elements` members (names, ranges per element, the subject's one field, the texts); a value not finite; an unknown field and its text; the 64-node limit; the schema; the written order | every refusal atomic with its plain text | RG-02 |
| Patches | add and remove; a partial `hue_curves`; a partial `elements`; the fit as one patch; Reset; copy and paste; undo (one step for a fit) | tested | RG-02 |
| Identity | empty curves; Strength 0; all elements at the identity; a bypassed node; a pixel with mask 0; a default node | bit for bit | RG-02 |
| Hue maths | one point is a constant; the periodic seam (0° and 360° give one value, a sweep has no step above 5e-3 of a code); `r = 0` and grey give no NaN and are unchanged; the tone `m` is preserved to 1e-12; rotation by 180°; the table against the exact spline (error stated) | 1e-12 where stated | RG-02 |
| Element step | `local` by hand on known triples; exposure only; rotation by 180°; the tone preserved by hue and chroma; masks 0, 0.5 and 1 | 1e-12 | RG-02 |
| Strength | each fitted field's formula of §3.2 (the chroma power included); monotone at every Strength; Strength 0 is the identity; the same `effective` on the CPU and in the host packing | 1e-12 | RG-02 |
| Order | `apply` equals `elements(hue(C11(C9b(C9(trims(v))))))` and differs from the hue step first, and from the elements first, by more than 1e-3 somewhere | 1e-12 | RG-02 |
| Segmenter | parity with the authors' code on the fixtures of §4b.1 (the element masks equal on at least 99 % of pixels); a second run bit-identical; 1, 2 and 8 threads give the same bits; the class table of §4b.2 against the checkpoint's `id2label` | stated bounds, measured | RG-03 |
| Element masks | skin from MediaPipe inside `person`; the subject rule; the guided filter on known inputs (a step edge, a flat guide, an empty mask); the same bits at every tile size; session cache and drop with the photo | exact; against the POC's masks on the local set, equal on at least 99 % of pixels (starting) | RG-04 |
| Network | the file parses; the expansion has 432 neurons and its hash is pinned; every refusal of §4b.5; the **network check**: the POC's recorded input vectors (a neutral photo against a one-colour amber reference opens `G.unify`, moves `T.cast_b` up and gives `T.wall_deg` near 45; the turn parameters for unify 0 and 1; the key-mismatch ramps; the purple and skin outputs) give the same outputs to 1e-9 | exact on the vectors | RG-06 |
| Objective | each term and wall of §5.3 and §5.4 on a fixed candidate (the turn weight and `allowed(C)` at C\* 8, 15, 25, 60 for unify shut and open; a pale wall pixel turned 35° does not count, a strong colour turned 25° counts; the clip allowance formula; opposite-tinted whites do not cancel; a purple turn; the look-only objective equals the full one minus the walls and the clip term) | 1e-12 | RG-06 |
| The fit: properties | the identity photo (reference = the photo's own display picture) gives the identity and "already close"; **the final objective is no greater than the neutral search's**; no worse than the identity objective; every parameter in [−1, 1]; knots monotone; an element that fails the need test is exactly 0; **1 and N threads give the same bits; two runs give the same bits**; Cancel changes nothing | exact | RG-06 |
| Reference loading | JPEG, PNG and TIFF, 8 and 16 bits, grey; EXIF orientation; untagged → sRGB; ICC matrix/TRC fixtures (our own generated sRGB, Display P3 and Rec.2020, a `para` variant, a **screenshot "Display" profile**), checked against the oracle; an unreadable profile → sRGB with the note; HEIC refused; **letterbox**: a synthetic picture with black bars is trimmed, a **night frame with many dark rows** (noise, a dark sky) is not, a flat dark picture is not trimmed into nothing, pillarbox, JPEG ringing | the bars exact; the ICC conversion within the stated bound of the oracle (starting: set in RG-05) | RG-05 |
| Plan and random graphs | the node in the plan, live and dead, numbering, `picture_ancestors`; `any_graph.rs` draws `reference_grade` from `NodeKind::ALL` with fitted-style values, hue curves and element grades (a seed list joins the regression seeds; the existing draws do not change) | CPU: tiles and threads bit for bit; device: not refused, within the bounds, exports byte for byte across tile sizes | RG-02, RG-08 |
| Device parity | §7.3 | 2e-4 in LogC4; one 10-bit code after C12 and C13; identity exact | RG-08 |
| `diag-alf` | §7.3 | 0 failures on each of the four runners, 3 runs | RG-08 |
| App against `ym` | the example of §3.1 in a graph: JPEG, TIFF 16, PNG 16 and DNG; a fit through the app's path and through `ym render --reference` on the same file | byte-identical, on the CPU and on the device (`parity.rs`, `parity_gpu.rs`) | RG-09 |
| Golden | **one new entry**, `node-reference-grade`: a fixed set of fitted-style values with element grades on a synthetic scene (with a stand-in element map, not the segmenter's), rendered by `ym` and pinned in `expected.json`; the device checked against it on the GPU subset | `golden check`: the current entries unchanged and the new one (so *N + 1* of *N + 1*, where *N* is the count when RG-09 lands); **no existing hash moves** | RG-09 |
| Presets and albums | a preset saved from a photo with the node holds it in `nodes`, applies the numbers elsewhere (no re-fit); an album round-trips and an older build's refusal text is the album's; the field test of `docs/30` §14 | tested | RG-02 |
| CLI | the example read and validated (`docs/10`, `crates/ym-cli/tests/docs.rs`); the help's paragraph; every error text; `--reference-node`; a graph with none, one and two nodes; the missing scene model | the test fails on any difference | RG-09 |
| **Against round 15** | the evaluation tool below, run on the same local photos and references as the POC: the Rust node's fits against round 15's | the three rules below with round 15 as the baseline, **no worse on any line** | RG-10 |
| **The quality round** | the evaluation tool on a **fresh** held-out set, the changed network and objective against the baseline of RG-10 | the three rules below | RG-13 |
| Interface | unit: the tabs, `NODE_TABS.reference_grade`, `resolvedGraded` for the kind, the shortcuts (⌘1 to ⌘7), `graphOps` (add after the selected node, the 64-node limit, reset), the status-line state machine (including "look changed", "edited" and the progress text), each copy string, the Hue plots' keyboard path, the Elements cards; Playwright against the mock: the menu, choosing, Fit and Cancel, Strength, Skin protection, each error, the strip chooser, Reset, Copy and Paste, undo; **no tab clipped** at three window sizes; the glass audit and a11y; reference screenshots `design/screenshots-c2/photos-reference-grade-*.png` that **the owner looks at before they are committed** | green | RG-12 |
| Speed (on a Mac) | the fit's time (§5.7); a slider move in the fit view with 8 nodes, one a Reference Grade node with both hue curves and element grades; the node's cost per megapixel | the node view's 100 ms (`docs/23` §5.9); the fit's time is the owner's to accept | RG-H3 |
| Blind A/B [HUMAN] | the owner picks references and keeps them local; `tools/eval/ab` shows the reference and two candidates (the fit, and a baseline) in random order; the rater picks the closer one; against no change and against the Reinhard map | the existing provisional gate: win share at least 60 % and the Wilson lower bound above 50 % (`docs/19` §13); **thresholds the owner's** | RG-H4 |
| Licences | `cargo deny check` and the npm licence check; `tools/licences/check_bundle.py` for the scene model (card, `SHA256SUMS`, `.gitignore`, `NOTICE` item 15) | pass; no new dependency (if one is chosen, it is recorded in the PR) | RG-03 to RG-12 |

### 10.1 The evaluation tool (RG-10)

The POC's evaluation harness is rebuilt in the repository as **`tools/eval/refgrade`** (Rust, a binary that links the core; it never runs in the app or `ym`). It is the measure every change to the network or the objective is judged by. It reads a **local evaluation set of public raw photos and reference stills, never committed** (`testdata/`, git-ignored; its sources are not named in the repository, and the tool takes the set's description from a local `split.json`). It reports **harms** and **look transfer**.

**The set.** `split.json` lists each photo with its raw path, its set (`dev` or `test`) and content tags (people, landscape, city, interiors, hard light, colourful). The POC's set was 130 public photos, 56 in dev and 74 held out in test, each fitted to 13 reference stills (728 and 962 fits). The reference stills are copyrighted pictures the owner and the repository keep local (§6.6); they are used for evaluation only, never committed or cached by CI, never used for any shipped number. **The owner's own photos sit in the held-out set and are never looked at while designing** ("I won't let you tune specifically for my photos"); the tool gives only aggregate numbers on the held-out set, and a seeded random sample of 12 photos (seed `20261124`) for the owner's eyes.

**The working rule** (the owner, 2026-11-24): *no rule targets a photo.* Every change to the network or the objective states a general grading principle ("film grades don't turn content colours into purple"), is found on dev, and is judged on the held-out set. A held-out set used three times (round 15 used the POC's third check) is spent: later rounds are checked on a **fresh** set (RG-13).

**Harms** (per fit, on the full renders of the photo's default and the fit, in display sRGB; C\* and L\* in CIELAB, hue in **Oklab**; deliberately different from the walls so a fit cannot pass them just by satisfying its own walls):

| ID | Harm | Flag when |
|---|---|---|
| H1 | Purple turn | the share of non-skin pixels with fit C\* ≥ 5 and Oklab hue in 300° to 355°, where the photo's C\* was < 2 or its hue differs by ≥ 15°, is > 2 % of the frame, and the reference's own share (Oklab 300° to 355°, C\* > 8) is < 5 % |
| H2 | Skin colour | skin ≥ 1 % of the frame, and the skin's Oklab mean-vector hue moves by > 10° (only tested when its mean-vector C\* is at least 4), or its mean-vector chroma ratio (fit over photo) is > 1.25 or < 0.55 |
| H3 | Memory colour | sky (or vegetation) ≥ 2 % of the photo and < 2 % of the reference, and its Oklab mean-vector hue moves by > 15° (a sky with median L\* under 20 and a group with mean-vector C\* under 4 are not tested) |
| H4 | Tinted whites | pixels with the photo's L\* ≥ 80 and C\* < 8 cover ≥ 1 % of the frame, and their fit mean C\* is > max(8, the reference's own mean C\* on its whites + 4) |
| H5 | Clipping | the fit's share of L\* > 98, or of L\* < 2, exceeds max(photo share, reference share) by > 3 points |
| H6 | Colour reversal | the share of non-skin pixels with the photo's C\* ≥ 10 whose Oklab hue turns by > 60° is > 3 % of the frame |

**Look transfer** (a fit that does nothing passes every harm check, so how much of the look arrived is measured too):

| ID | Metric | Definition |
|---|---|---|
| T1 | Colour transfer | `1 − SW(fit, ref) / SW(photo, ref)`; `SW` is the sliced Wasserstein-1 distance between the (a\*, b\*) distributions, 4096 seeded pixel samples per picture, 32 fixed directions. 1 is a matched palette, 0 no change, below 0 further away |
| T2 | Tone transfer | the same form on the key-preserving tone shape `(p98 − p2, (p50 − p2) / (p98 − p2))` of L\*, scaled by 1/50 and 1 before the L1 distance. The key stays with the photo |
| T3 | Cast transfer | `1 − \|cast(fit) − cast(ref)\| / \|cast(photo) − cast(ref)\|`, the cast being the mean (a\*, b\*) of non-skin pixels with L\* 15 to 80 |

Where a denominator is under 1 (under 0.01 for T2), that metric is skipped for the fit. The tool also reports **E2**, the key-preserving CIEDE2000 (`crates/ym-color/src/deltae.rs`), for photos with labels, and the **fit time**.

**The acceptance rules** (the owner's, 2026-11-25), for a new version against a baseline on a held-out set, over all of its fits:

1. The share of fits with any harm **falls** (for RG-10, against the POC's round 15: does not rise).
2. No single harm's share of fits **rises by more than 1 point**.
3. Mean T1, T2 and T3 each **fall by no more than 0.01, measured on the fits the baseline left harm-free.** Look a baseline got by doing harm is not a loss. The all-fits reading is reported beside it.

If (1) and (2) hold but (3) fails, the round is not accepted as is; the next step is an ablation on dev to find which principle costs transfer, **not** a retune on the held-out set.

**Outputs**: `metrics.json` (per fit: every harm value and flag, T1 to T3, the time), `summary.json` and `summary.md` (the shares of fits flagged per harm and for any, the share of photos with a flagged fit, the means and medians of T1 to T3, a per-reference table so references that cause harm stand out, per-tag tables), the dev grids (references, the previous round, this round), the owner's sample for the held-out set, and `--compare` for two runs. Threads: each fit stays single-threaded and deterministic; the set runs on a pool.

**Data rules.**

- **Public data only for choosing constants** (§16). Owner's photos are a held-out check at most, reported and not used to tune (`docs/32` §2.3).
- **The evaluation set and its stills are local only**: never committed, never cached by CI, never uploaded.
- **No measurement of our own is needed**; if a gate seems to need one, use public data and note the limitation in `docs/STATUS.md` (`AGENTS.md`, V1 data rule).

## 11. Speed

| Where | Budget |
|---|---|
| **The fit** | **Several seconds on a Mac with all cores, with progress and Cancel** (the owner, 2026-11-25). The POC's figure is about **30 s on one Linux thread** (§16.7); no Mac figure exists. RG-11 does the speed work and RG-H3 is the owner's timing |
| The scene | the segmenter once per photo and once per reference, in the background, then kept in memory for the session; an **estimate** of 1.5 to 2.2 s each on 4 cores (§4b.1), not measured; MediaPipe skin on each person component |
| The photo's chain before the node | rendered once per fit at the analysis size |
| The node on the GPU | the hue step is one `atan2`, one `sincos`, one square root and two table reads; the element step adds up to seven plane reads, a rotation and a scale for each live element; budgets set in RG-08 from the Colour Uniform pull's (`docs/32` §9) |
| The node on the CPU reference | the corrector's steps, the hue step and the element steps in f64, split across rows and threads; measured in RG-02 on the Linux machine |
| Element masks | the guided filter by box sums is linear in the pixels whatever the radius; measured in RG-04 at a small, a medium and a 24 MP size |
| Memory | the scene model about 96 MB at f32 while loaded (48 MB file); each reference's two pictures (up to 1024 px, about 8 MB); the label maps; an element plane is `u16` per pixel (48 MB at 24 MP), made only for a live element and held within the memory ledger (`docs/17`, H-13); two 1024-sample tables per node on the device |
| Thumbnail | a few milliseconds, as for any node |

## 12. Risks

| Risk | What happens | What the design does |
|---|---|---|
| **Content mismatch** | A night reference could darken a beach; a reference full of one colour pulls the whole photo | the network keeps the photo's median, matches shape and not level, and drops exposure relations under a key mismatch (§4b.5); the content turn budget and the clip term (§5.4); **Strength** scales the result; the status "Nothing was changed" when the fit does not help |
| **Look dependence** | A fit is valid for one Output look | the fit runs the real chain (§2.2); changing the look marks the fit stale and shows a hint; nothing is re-fitted by itself |
| **Colour tagging of references** | An untagged file is a guess (sRGB); a wrong guess shifts the whole fit | tags are read (§6.3); the note when a profile cannot be read; Strength; the person can see the thumbnail. The POC read 8-bit sRGB only (§16.9) |
| **Fit time** | About 30 s on one Linux thread; unmeasured on a Mac | progress and Cancel, the remedies of §5.7, RG-11, RG-H3 |
| **The scene model is wrong** | A sky called water, foliage called ground, a missing person, a mask that bleeds | a wrong element only changes where a grade acts; the walls watch hue drift, memory colours and skin by measuring the picture, not the label; the hard cases are named in the model card; the guided filter takes the masks to the picture's edges |
| **Skin** | A global grade can push skin toward the reference's cast; skin was 7.9 % of held-out fits flagged in round 15 | the three skin walls and the skin element's caps (§5.4, §4a.1); Strength and the switch; **RG-13 is the quality round on it** |
| **Tinted whites** | White surfaces take the reference's cast; 7.8 % of held-out fits flagged in round 15 | the whites walls (§5.4); RG-13 |
| **Local looks** | Vignettes, a graded sky gradient, split looks in one half cannot be copied | out of scope (§13); the elements cover sky, foliage, water, ground, built, skin and the subject only; the status never promises more; masks and the person's hand cover some |
| **Spatial nodes after the node** | Film, Blur and Face Refinement are not run in the fit | a note on the card; the golden entry has none after the node; question 7 |
| **The kernel change** | llvmpipe has miscompiled small control-flow changes in `run_nodes` | §7: host packing, one `if`, `select`, H1 first, `diag-alf` before release, H3 as the last resort. **No release contains the kernel change until `diag-alf` is clean** |
| **Table resolution** | Hue tables of 1024 samples | the error against the exact spline is measured in RG-02 and stated |
| **Hue step and luma** | The step keeps the mean (§4.2); the POC's step restored the Rec.709 luma of the LogC4 triple | §15, decision 19: asked in `docs/STATUS.md`; RG-10's gate shows whether it matters |
| **Memory** | Seven full-size planes at 24 MP | planes only for live elements, the memory ledger, tiles with a halo (§4a.3) |
| **A hand-written network is not tuned to every picture** | A weight is wrong for a kind of photo | the evaluation tool finds it (§10.1); a weight changes only by a spec change under the acceptance rules; never for the owner's photos |
| **The held-out set is spent** | Three checks were made on the POC's | later rounds use a fresh set (RG-13) |
| **Reference copyright** | A film still is copyrighted | used at run time, never stored (§6.6); evaluation stills local only |

## 13. Not in this version

- **A learned grading model.** The network is hand-written. A trained predictor, or any model that sets targets or settings, is out; only if the owner asks and with a budget.
- **Spatial or local adaptation beyond the seven element grades** (region-aware fits, a bilateral grid pass, gradients). It would cost a binding, the `diag-alf` gate again and a new pipeline step.
- **Any provider's involvement**: Claude, Antigravity and Apple Intelligence receive no reference and write no numbers here (`docs/22`).
- **HEIC and other formats; RAW files as references; a reference from the web or a URL.**
- **Hue curves and element grades on correctors and Colour Uniform nodes.**
- **Batch matching** (apply one reference to a selection in one step). Each photo is fitted on its own node; a preset or Paste carries the values.
- **Matching inside the node's masks, or a reference region.**
- **Choosing a network weight by looking at the owner's photos** (§10.1).
- **Our own measurements of any kind** (V2).

## 14. Questions for the owner (RG-H1), each with a recommended default

**All 18 answered on 2026-11-25 (RG-H1): every recommended default is accepted as written.** The table keeps each question and its default; the last column says what round 15 changed in what the default means.

| # | Question | Recommended default (accepted) | Why, and what changes if the answer differs | Since round 15 |
|---|---|---|---|---|
| 1 | **The name.** | **"Reference Grade"** in the UI; kind `reference_grade`; "a Reference Grade node"; default node name "Match" | never an ARRI product name (`AGENTS.md` rule 3). Another UI name changes the strings and no code | unchanged |
| 2 | **Strength: which fields, what scale.** | **Strength scales only what the fit writes** (the four curves, saturation, both hue curves), from the identity; stored and typed as 0 to 100 % like a look's intensity | the research and the request say "0 to 1"; the house rule is that a file reads like the panel (`docs/32` §5.1), so the stored number is 100 times larger. The alternative, scaling **every** field of the node, is a change to `effective` and more cases | **the element grades are fitted fields too** (exposure and hue scale linearly, chroma as a power, §3.2) |
| 3 | **What an edit of a fitted value does.** | **It changes that stored value only; Strength still scales it; Fit starts again from the identity and overwrites the fitted fields; a session flag says "Edited by hand"** (§3.3) | the alternatives are a second stored copy ("fitted" and "edited") or an automatic re-fit; both are more state | the fitted fields include `elements` |
| 4 | **Is a label of the reference stored** (a file name)? | **No** (§3.7) | it would travel in every exported file; if wanted, a `reference_label` string, written to `ym:Settings`, presets and albums, with the privacy cost | unchanged |
| 5 | **When the Output look changes after a fit.** | **A hint; no automatic re-fit** (§2.2); the look's id and intensity (and anything replacing C12) are what count | automatic re-fit would overwrite hand edits and spend seconds on each look change; the alternative is "re-fit while the reference is still in memory" | a re-fit takes several seconds, which makes the default firmer |
| 6 | **Does an upstream edit also mark the fit stale?** | **No** | an exposure tweak is a normal step after a fit; the person can press Fit again | unchanged |
| 7 | **Spatial nodes after the node in the fit.** | **Not run; a note on the card** (§2.2) | running them per candidate is far too slow; the alternative is to refuse Fit when one is after the node | unchanged |
| 8 | **The fit and the node's masks.** | **Ignored** (§5.1) | the reference describes a whole picture; fitting inside masks needs a matching region | **the fit ignores the node's own `masks` but uses its element masks** (§4a.3): they are part of the candidate, and the person's masks still limit the result |
| 9 | **Skin protection.** | **On by default; MediaPipe Selfie Multiclass skin classes; read at the next fit** (§5.6) | off by default is possible; the term can be removed with its test | **it is the three skin walls** of §5.4; MediaPipe also makes the skin element; with no MediaPipe weights there is no skin and no fallback |
| 10 | **ICC: our own minimal reader or a crate.** | **Our own** (§6.3) | `moxcms` is the candidate crate (licence class allowed in `docs/14`, not in the manifest); it needs a determinism check | unchanged; the POC read 8-bit sRGB only and did not test it |
| 11 | **Letterbox removal: a switch?** | **Automatic, no switch**; a note says when it happened (§6.5) | a "Remove black bars" switch is cheap if the rule misfires; the person can also crop the file | the POC's two constants are adopted (§16.5); RG-05 adds and tests the rest |
| 12 | **A reference from the strip is a snapshot, not a link.** | **Snapshot** (§1.3, §6.4) | a link would re-fit or go stale when the other photo changes, and would need the other photo's settings kept in memory | unchanged |
| 13 | **Hue step: keep the mean, or restore luma as the look does?** | **Keep the mean** (§4.2) | the draft said restoring luma needs a decode and an encode per pixel in the kernel | **the draft's reason is wrong, and round 15 restored luma.** The look's step does it on the LogC4 codes, with a dot product and an add, no decode; the POC's hue step was the look's and restored it. The owner's answer stands (§4.2 keeps the mean); the difference is asked again in `docs/STATUS.md` and RG-10 shows whether it matters (§15, decision 19) |
| 14 | **Hue curves only on this node** | **Yes** (§4.3) | the same field on correctors and Colour Uniform is the same work again, later | element grades are on this node only too |
| 15 | **The thresholds** of E1, E2 and E3. | **Set after the POC numbers (§16.8) by the owner.** E3 starts from the existing provisional gate: win share at least 60 % and the Wilson lower bound above 50 % against no change and against the Reinhard map | the research gives no published bar for this shape of problem (`research/reference-grade-2026-11-23.md`, *Methods*) | **the owner's three acceptance rules of 2026-11-25 (§10.1) are the gates for the fit's quality**; E1 and E2 are not gated (the POC measured neither as a gate); the blind A/B keeps the provisional bar |
| 16 | **The pipeline version.** | **0.17.0**, the next minor step (§3.8) | a patch step is allowed if the lead prefers; the number only has to be above 0.16.0 | set by RG-02 |
| 17 | **A fit-time limit.** | **Target 1 to 2 s on an M1, a Cancel button, no hard limit** | a hard limit would need a partial-result rule; decide after RG-H3 | **the target no longer holds:** the owner's decision 2 is "several seconds on a Mac with all cores, with progress and Cancel". No hard limit stays |
| 18 | **Mock-ups before the build.** | **Yes** (RG-07, RG-H2) | the node has new tabs and a new chooser mode | the Elements tab is part of them |

## 15. Decisions made while writing the spec

The simplest readings where the request, the research or the POC was silent, and **where this file differs from the research or from its own first draft**.

| # | Choice | Why |
|---|---|---|
| 1 | **The research says Phase 1 needs "no WGSL change"; this spec has one** (§7). | The owner chose hue curves over warp pins on 2026-11-23, and the research itself said that choice is a kernel change. The element grades add to it (decision 21). The research text of *Parameter set* and *Effort* is superseded; the warp-pin parameter set is not used |
| 2 | **The research says the fit runs "through any nodes after it and the Output".** The spec runs the **per-pixel** steps only (§2.2). | Spatial nodes make their picture before the colour pass and cannot be run per candidate |
| 3 | **The fit measures in CIELAB with the D65 white, on sRGB display codes clamped to 0 to 1.** The first draft said CIELAB D50 on the display's own codes (Display P3). | The POC measured in D65 on sRGB and every constant of §5 and every threshold of §10 was set there. Following it keeps the numbers true. The reference is converted to the same space (§6.3) |
| 4 | **MediaPipe's skin is `BodySkin` (2) and `FaceSkin` (3) of Selfie Multiclass** (`crates/ym-ml/src/mediapipe/selfie.rs`). | Read from the code |
| 5 | **The research asks for ICC handling for "sRGB, Display P3 and Rec.709".** The spec reads any **RGB matrix/TRC** profile and refuses the rest (§6.3). | The macOS "Display" profile of screenshots is a matrix/TRC profile; a fixed list of three would miss it |
| 6 | **`strength` is stored 0 to 100, not 0 to 1.** | The house rule for stored units (§14, question 2) |
| 7 | **The hue step is last in the corrector's part** (after C11), not before the curve (§2.3). | The proven kernel shape; the fit's search stages |
| 8 | **The hue curves are baked to 1024-sample tables that are the definition** (§4.2). | One definition for the CPU and the device; the table error is not a parameter of the parity gate. The POC evaluated the exact spline; RG-02 measures the difference |
| 9 | **The node writes every field** (a new kind), unlike Colour Uniform's `trim` and `grade` (§3.1). | The omit-at-default convention existed for a kind that older builds already read |
| 10 | **A result no better than the identity is the identity**, not a failure (§5.5). | A reference that matches already is a normal answer; the search keeps its start on a tie |
| 11 | **The hue knots of the fit are at fixed hues, the person's points are free** (§9.4). | The fit's parameter count is fixed; the field is not |
| 12 | **`moxcms` is named in `docs/14` but is not in the workspace manifest.** | Found by reading `Cargo.toml` and `docs/14`; stated in §6.3 |
| 13 | **The first draft's "no learned model, global looks only" is withdrawn** (the owner's decision 1, 2026-11-25). | The POC showed that per-element grades need a scene model, and that the grading rules need a network of their own. The draft's six objective terms (quantile shape, band means, sector chroma, regulariser, clip, skin) were the POC's round 1 and were replaced by the network's targets in the rounds after it (§16.3) |
| 14 | **The segmenter reads the Input's picture as `docs/23` §4.7 renders it, at 1024 px on the long edge; for a reference, the reference's own picture after the bars are removed, also up to 1024 px.** The POC labelled the photo's default render. | The rule of `docs/23` §4.7 for every AI mask: grading never moves a mask. The two differ only by the look and the node's own tone, which the model barely sees |
| 15 | **The guide of the soft masks is the L\* of the same Input picture.** The POC took it from the photo's default render through the person's look (`P`'s L\*). | The same rule: a node's grade and the person's look must not move its masks. RG-04 measures the difference against the POC's masks on the local set; RG-10's gate catches harm. If the guide has to be a function of a different picture, the fit and the render must use the same one |
| 16 | **An element's field range is the fit's own bound** (±1 EV, the element's cap, a ratio 0.5 to 1.6) (§4a.1). | The simplest rule with no invented number; a hand edit can go as far as a fit |
| 17 | **Strength scales the stored values, and a chroma ratio as a power** (§3.2). The POC scaled its search parameters. | The node has the stored values only; the two agree at 100 %, where the gates are |
| 18 | **A reference's EXIF orientation is applied.** The first draft said orientation is ignored. | The segmenter expects an upright picture, which the POC (stills with no orientation tag) never tested |
| 19 | **The node keeps the mean in the hue step (question 13); round 15 restored luma.** The POC's hue step was `ParametricLook::evaluate`, which restores the Rec.709 luma of the LogC4 triple after the move (`look.rs`), on the codes with no decode. The element grades restore nothing, as in the POC. | The owner accepted "keep the mean" with a reason that the code contradicts (the draft thought a decode and an encode were needed). Both can be built; the evaluation will tell. Asked in `docs/STATUS.md`; if the answer is "restore", §4.2 and RG-02 change |
| 20 | **Skin protection is a switch on three walls; with no MediaPipe weights there is no skin and no fallback.** The POC had a colour-rule fallback (person pixels with L\* 25 to 85, C\* 8 to 45, hue 20° to 80°). | MediaPipe is bundled; a damaged install says so and a rule that guesses skin by colour is the kind of guess the owner does not want |
| 21 | **The element grades run in the same early return of the kernel** (§7.2, item 5): host-packed parameters in the table buffer, planes as raster planes, one more flag, no new binding. | The least change to the code that llvmpipe miscompiled before; `diag-alf` decides |
| 22 | **The network file is `data/refgrade/network.ywn`**, round 15's file unchanged, hash-pinned. | The owner: the hand-written network ships as a text file in the repository |
| 23 | **`NOTICE` item 15 is written now**, by this task (the EoMT weights, MIT, and the DINOv2 backbone, Apache-2.0), before RG-03 ships the model. | The precedent is `docs/14` and `NOTICE` item 13, which D-01's spec task wrote before the Depth weights shipped; `tools/licences/check_bundle.py` gets its check from RG-03 |
| 24 | **The fit's sample seed is one constant** (`0x5EED_2026_1124`), not one per pair of photo and reference. | That is what the code does; the sample then depends on the analysis picture's size only |
| 25 | **`PIPELINE_VERSION` takes the next minor step with RG-02**, the task that writes the kind, not at the end. | The rule of `docs/01` *pipeline_version* and `docs/23` is to set it when a kind lands; the golden gains its entry in RG-09 |
| 26 | **A node with element grades and no scene model is refused for export and by `ym`; the viewer draws it without the element step and says so** (§4a.3, §8). | The Depth mask's rule for missing weights |

## 16. POC findings

The POC is the Rust example `crates/ym-core/examples/refgrade_poc.rs` (with `refgrade_poc/eval.rs`) on the branch `refgrade-poc`, built on the real colour chain (`ColorChain`, `ParametricLook`, the looks' library, the MediaPipe segmenter). It ran on the CPU on **Linux**, one thread per fit, on a **local evaluation set of public raw photos and reference stills, never committed** (§10.1). The scene labels were made once by the authors' EoMT checkpoint in Python and cached as label maps the POC read; the POC did not run a segmenter of its own (§16.9). Every number below is the POC's; where this file gives a number the POC did not measure, it says so. **Round numbers** are the POC's: rounds 2 to 5 replaced the first draft's quantile terms by a scene-aware objective and then by the network; rounds 6 to 10 added guards against the harms the owner saw; rounds 11 to 15 are measured on the harms and transfer of §10.1.

### 16.1 The parameter list and the knots

49 normalised parameters, 30 global and 19 local, exactly as §5.2: luma curve at the stops −6, −4, −2, −1, 0, 1, 2, 4 (8); red, green and blue at the stops −3, 0, 3 (9); saturation (1); `hue_vs_sat` and `hue_vs_hue` at 0°, 60°, … 300° (6 each); per element exposure, hue and chroma (6 × 3); the subject's exposure (1). Ranges: the first luma knot ±2 stops, a luma segment's log slope ±1.1, a channel knot ±1 stop, saturation ±60, `hue_vs_sat` 1 ± 0.7, `hue_vs_hue` ±30°, local exposure ±1 stop, local hue ± the element's cap (40, 35, 35, 30, 25, 15), local chroma ratio 0.5 to 1.6. The regulariser scale is the range itself (parameters are normalised). **The draft's list (30 numbers, a luma curve on fixed scene stops about middle grey, three-knot channel curves, six hue knots) held; the 19 local numbers are new.**

### 16.2 Analysis sizes and sampling

The photo's node picture is made at 1600 px on the long edge and reduced to **512 px** (area average, linear light) for the fit; the **sample is 8192 pixels** (a 128 × 64 grid, one jittered pixel per cell, SplitMix64 seed `0x5EED_2026_1124`); the reference is reduced to **512 px** (never enlarged) and measured on all of its pixels. The area average is a box filter with exact fractional weights, f64 accumulation. The soft masks were made at the fit size and at the 1600 px render size, with the same relative radius.

### 16.3 The objective

Rounds 2 to 5 replaced the first draft's six terms by the network's targets (§4b.5) and the look terms of §5.3; rounds 6 to 15 added the walls of §5.4 one by one, each against a harm the owner or the dev set showed. **Weights, scales and dead bands are in §5.3 and §5.4 and come from the POC's code.** The weights were chosen by hand from development reference stills and the public photos: no weight was changed by looking at the owner's photos or at the held-out stills. A sliced Wasserstein term was not used in the objective (it is the transfer metric T1 of the evaluation, §10.1). The POC's notes and its code differed in a few constants; **§5 follows the code** (wall limits as network outputs, the skin wall weight 12, the single polish pass, the cast measured on L\* 15 to 80 and not on skin).

### 16.4 The optimiser

Golden-section coordinate search, stage table and counts of §5.5, a start-inclusive best-of so a step never raises the objective, the need test for local grades, a joint polish, and round 15's **two starts**, keeping the lower full objective and the neutral search on a tie. No randomness beyond the sample; the seed is the sample's.

### 16.5 Letterbox constants

The POC's rule, from each edge inward: a row (or a column, between the bars found so far) is part of a bar when the **standard deviation of its 8-bit codes is below 2.0 and their mean is below 16**, over the three channels; a bar is at most **35 %** of a side. It was used on reference stills with black bars. **Not covered by the POC:** a minimum bar width, a step test at the bar's inner edge, a JPEG-ringing margin, and night frames with many dark rows; RG-05 adds and sets them from public data and tests them on the night-frame cases of §10 (§6.5).

### 16.6 The minimum reference size

The POC accepted every size and never enlarged one. *N* of the "too small" text is **not set**; RG-05 sets it from public data (the smallest reference whose fits stay inside the harm rules) and writes it here. Until then §9.3's text keeps *N*.

### 16.7 Time

**Linux, one thread per fit, not a Mac.** Round 15: **about 30 s per fit** (three searches); rounds 11 to 14, one search: about **6 s**. Round 15's look start won on 461 of the 962 held-out fits and 333 of the 728 dev fits. The sets ran four fits at once; each fit was timed as one thread and is deterministic. The time is almost all candidate evaluation (the chain on 8192 pixels and the measures): by count about 770 evaluations per search before local grades (§5.5). **No time was measured for the segmenter, the guided filter at full size, the ICC reader, or any Mac or GPU path.** The owner decided (2026-11-25) that several seconds on a Mac with all cores is acceptable; RG-11 and RG-H3 follow.

### 16.8 Quality

The POC's measures are the harms H1 to H6 and the transfers T1 to T3 (§10.1), on the held-out set unless said (**74 photos × 13 reference stills = 962 fits**) and on dev (**56 photos × 13 = 728 fits**). Baseline: round 11, the first round measured this way (its any-harm share 34.8 % on held-out, 28.7 % on dev). The owner's rule of 2026-11-24: *no rule targets a photo; every change states a general principle*; and of 2026-11-25: *transfer is judged on the fits the baseline left harm-free.*

**Held-out set, against round 11.**

| Round | What it did | Any harm | Change per harm (points) H1 H2 H3 H4 H5 H6 | Transfer on the 627 fits round 11 left harm-free, T1 T2 T3 | Verdict |
|---|---|---|---|---|---|
| 11 | the baseline: Oklab purple band, turns not entries, a skin chroma wall | 34.8 % | | 0.233, 0.131, 0.242 | baseline |
| 12 | five principles: turns anywhere, a content turn budget, one hue space and stiffer memory walls, whites by pixel, a clip allowance | 12.5 % | +0.0, −0.1, −11.0, −2.1, −8.6, −4.9 | all fits: 0.236, 0.122, 0.230 (round 11: 0.251, 0.124, 0.254) | line 3 failed (T1 −0.015, T3 −0.024) |
| 13 | withdrew the either-end purple turn; the allowed turn depends on the colour's strength (`allowed(C)`) | 13.5 % | +0.0, +0.0, −11.0, −1.0, −8.6, −4.7 | T1 −0.006, T2 +0.002, **T3 −0.016** | **not accepted** (line 3) |
| 14 | the cost of a turn grows with how far it goes (`turn_weight`) | not run on the held-out set (its dev gate failed) | | | stopped on dev |
| 15 | **the same network and objective as 14; a search from both ends** | **15.7 %** | **+0.1, +0.7, −10.8, +0.2, −8.6, −4.7** | **0.241 (+0.008), 0.149 (+0.019), 0.250 (+0.009)** | **accepted** |

Round 15 on the all-fits reading of line 3: T1 +0.000, T2 +0.011, T3 −0.001 against round 11, so it also passes under the first form of the rule. Against round 12 (never accepted) it carries +0.015 to +0.026 more look on each T, over all fits, and more harm (12.5 % against 15.7 %, almost all of it H4, +2.3, and H2, +0.8).

**Dev set (728 fits), the gates that sent each round to the held-out set.**

| Round | Any harm | H1 … H6 | T1, T2, T3 (all fits) | Gate |
|---|---|---|---|---|
| 11 | 28.7 % | 0.0, 6.2, 4.7, 3.6, 11.4, 9.2 | 0.222, 0.103, 0.221 | baseline |
| 12 | 7.7 % | 0.7, 4.8, 0.0, 2.9, 0.0, 0.8 | 0.215, 0.101, 0.199 | an ablation of the five principles showed two costing look |
| 13 | 7.1 % | 0.3, 4.5, 0.0, 2.6, 0.0, 0.0 | 0.217, 0.101, 0.209 | T3 gate missed (0.2087 against 0.209); the owner then chose to judge transfer on clean fits |
| 14 | 8.1 % | H1 −0.3, H2 +0.8, H3 +0.0, H4 +0.3 against round 13 | 0.220, 0.106, 0.215 | **failed**: the 36 tail fits recovered 23 % of their T3 gap (0.279 against 0.333 needed); the other two lines passed |
| 15 | 8.2 % | +0.1, +0.3 against round 14, the others +0.0 | 0.228, 0.121, 0.234 | **passed**: the 36 tail fits reach T3 0.371 (round 11 0.431, round 13 0.234, round 14 0.279; medians 0.423 against 0.339 and 0.482); the 519 clean fits within 0.005 of round 14 (T1 +0.006, T2 +0.018, T3 +0.016) |

**What the diagnosis found (dev only).** On 36 dev fits where round 13 lost the reference's cast, round 11's own grade scored better on round 13's *look* terms on 31 of them; the walls rejected it: they cost round 11's grade a median of 51 under round 13 and over 100 on 17 fits, and the content turn budget was the largest single penalty on 17 fits (a mean of 489; under round 14's graded budget it was the largest on 9, at a mean of 53). In 14 of 56 diagnosed fits (8 of the 36 tail fits, 6 of 20 controls) round 11's grade was **cheaper** under round 14's objective than round 14's own: the search missed it. That is the finding behind the two starts.

**What is left on the held-out set after round 15** (aggregates; the tags are the set's content tags): people 31.5 % with any harm (H2 21.5 %, H4 10.4 %), hard light 21.2 % (H2), city 19.9 % (H4 14.7 %), landscape 2.7 %, interiors 5.1 %. **Skin (H2, 7.9 %) and tinted whites (H4, 7.8 %) are nearly all of the remaining harm.** That is the work of RG-13.

**What did not work.** A rule that counts every purple turn in either direction (round 12's first principle) fought the turn budget and sent the skin of a magenta-lit portrait through 70° to 150°; it was withdrawn in round 13. A hard yes-or-no turn budget (rounds 12 and 13) rejected moderate turns that carry a cast. A single search from neutral missed grades behind the walls. **The held-out set has been used three times** (rounds 12, 13 and 15); later rounds use a fresh set.

**Not measured.** E2 (key-preserving CIEDE2000) was reported by the tool without a gate; no E1-style synthetic recovery was gated; no blind A/B was run. The thresholds the draft left to this section (E1, E2) are replaced by the three rules of §10.1 (§14, question 15).

### 16.9 What the POC did not cover

- **The GPU and Metal:** no kernel change and no parity were made; the hue step ran as the parametric look's exact CPU function (not the 1024-sample table), the element grades as a CPU step; neither wrote a settings file.
- **The hue step's luma:** the POC's hue step restored the Rec.709 luma of the LogC4 triple; this spec's does not (§15, decision 19).
- **The ICC reader, 16-bit and TIFF input, EXIF orientation, a reference without sRGB:** the POC read 8-bit sRGB JPEG and PNG.
- **The strip as a source, the UI, Strength, Reset, presets, albums, the `ym` flags and the byte-identity of app and CLI.**
- **The scene model in Rust:** the labels were cached from the original checkpoint in Python. The segmenter's input (§4b.1), the 640 grid, its time and its parity are RG-03's to measure.
- **Soft masks at full size and in tiles:** the POC made them at 512 and 1600 px, with the guide of the default render (§15, decision 15).
- **Any Mac timing** (the fit, the segmenter, the node), and **the letterbox rule's night frames and minimum size** (§16.5, §16.6).
- **The owner's own photos** sat in the held-out set and were never looked at while designing; the POC gives them only as part of aggregate numbers.

## 17. Mock-ups

None yet. **RG-07** makes them in `design/reference-grade/` (pages at 1440 × 900 built from the app's DOM and CSS, shot to JPEGs, as `design/colour-uniform/` and `design/blur/`): the node card, the Reference tab in each state of §9.3 (including the progress while fitting), the chooser mode in the strip, the Hue tab, the Elements tab (the cards, an element not found in the photo, the scene being made) and the error notices. The pictures are synthetic scenes and references, with no film still and no likeness. The owner reviews them (**RG-H2**) before RG-12 builds the panel.

## 18. Tasks

Task format of `docs/15`, milestone RG. The work goes on its own branch, one reviewed commit per task with its checks green (`cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`, `npm test`, `npm run build`). Every task that changes what `ym` does or prints updates `docs/10` and the help in the same commit. The tasks and their *Done when* are in `docs/15` (milestone RG: **RG-01** to **RG-14**, with **[HUMAN] RG-H1** to **RG-H4**).

| Task | What | After |
|---|---|---|
| RG-01 | this spec | – |
| **RG-H1** | the owner's answers (done 2026-11-25) | RG-01 |
| RG-02 | the node format and CPU path: global grade, hue curves, element grades, Strength; validation, Reset, Copy and Paste, presets, albums; `PIPELINE_VERSION` | RG-H1 |
| RG-03 | the scene segmenter in Rust: port, weights asset, fetch script, model card, parity fixtures | RG-H1 |
| RG-04 | element masks: classes to elements, skin, guided-filter soft edges, full size at render and export, session cache | RG-03 |
| RG-05 | reading references: JPEG, PNG, TIFF, ICC, black bars, a strip photo as a snapshot | RG-H1 |
| RG-06 | the fit: the network evaluator, the measures, the objective and walls, the two-start search, progress and Cancel | RG-02, RG-04, RG-05 |
| RG-07 | mock-ups | any time |
| **RG-H2** | the owner reviews the mock-ups | RG-07 |
| RG-08 | the GPU path: hue curves and element grades in the colour kernel, parity, **`diag-alf`** | RG-02, RG-04 |
| RG-09 | `ym --reference`; app and `ym` byte-identical; one new golden entry | RG-06, RG-08 |
| RG-10 | the evaluation tool; the Rust node against round 15, no worse on any line | RG-06 |
| RG-11 | speed | RG-10 |
| RG-12 | the panel: Reference, Hue and Elements tabs, Fit, Strength, status lines, Playwright, a11y, screenshots the owner sees before commit | RG-H2, RG-09 |
| **RG-H3** | the owner times a fit on a Mac and tries the build | RG-12 |
| RG-13 | the quality round on skin and tinted whites, on a fresh held-out set | RG-10 |
| **RG-H4** | the owner's blind A/B | RG-13 |
| RG-14 | release notes, the as-built section, the release | all |

**The `diag-alf` gate (RG-08) blocks any release that contains the kernel change.**

## 19. As built

*(Filled in by the tasks as they land.)*

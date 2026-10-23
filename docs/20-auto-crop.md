# 20 · Auto crop

Auto crop levels a tilted photo and proposes a frame for it. It is an **Auto** toggle in Crop mode: while it is on, the straighten angle and the frame follow the core's proposal for the chosen aspect, and moving either by hand turns it off. This file covers what the person sees, the setting, the core's API, what the analysis looks at, the straightener, the subject map, the frame proposer, the command line, determinism and the tests.

Everything here is deterministic geometry and image statistics in Rust, the same in the app and `ym` on every platform. No model weights ship (`research/` has the background: no published cropper's weights pass `docs/14` rule 5). The owner asked for it on 2026-10-22 ("The mode auto should be available in crop mode, as a toggle button"). The ranked plan it starts is in the research report of that day: this file is ranks 0 to 3 (camera hints, the straightener, the classical proposer, the toggle). A learned scorer, Vision evidence on Macs and a Foundation Models chooser are later steps (section 11), each gated by the owner's blind A/B. **2026-10-23: the owner stopped this classical proposer** ("crop composition is pretty bad, cutting objects, no thirds rule"); a learned cropper is being researched (`research/crop-models-2026-10-23.md`), and this file describes the proposer as it stands on the branch.

Auto crop does not change `PIPELINE_VERSION`: a crop is a setting, and a photo whose crop is not Auto renders as before.

## 1. What the person sees

The Crop panel's **Frame** card (`docs/11` §Crop) gets an `AutoToggle` in its header, labelled **Auto** with the accessible name "Auto crop". Unlike the AUTO buttons of the other cards it switches both ways.

| Action | What happens |
|---|---|
| Press Auto while it is off | The toggle shows pressed at once and the card's note reads "Finding a frame…". The core's proposal for the draft's aspect and orientation arrives (section 3): the draft's straighten angle becomes the proposal's angle and its frame the proposal's first frame, the viewport shows the straightened photo, and the note says what was done (below). The draft as it was before the press is kept for turning Auto off |
| Change the aspect while Auto is on | A new proposal for that aspect; the frame jumps to it. The angle is the same (it does not depend on the aspect) |
| Rotate 90° while Auto is on | A new proposal for the new orientation |
| Drag the frame or a handle, move it with the arrow keys, move Straighten | Auto turns off; what was moved stays |
| Reset (the card's link) | Auto turns off and the draft is the whole photo, straight, Original, as today |
| Press Auto while it is on | Auto turns off and the draft goes back to what it was just before Auto was turned on in this panel. When the panel opened with Auto already on, it goes back to the whole photo at 0° with the current aspect |
| Done | The crop is sent with `auto: true` while the toggle is on (section 2) |
| Cancel | As today: the crop and orientation from before the panel opened |

The **note** is a muted `RowNote` at the bottom of the Frame card, shown while Auto is on. One line for the frame, one for the angle:

| Result | Line |
|---|---|
| asking | "Finding a frame…" |
| a frame other than the whole photo | "Cropped to frame the subject." |
| the whole photo kept | "No better frame found, so the whole photo is kept." |
| levelled | "Levelled by 1.4°." (the magnitude of the angle, one decimal, `angleLabel` without its sign) |
| already level | "Already level." |
| no answer | "No horizon or straight lines found, so the photo is not turned." |
| the request failed | the toggle goes off and the note reads "Couldn't suggest a frame for this photo." for 4 s |

**No border in photos** (the owner's rule of 2026-10-22). A straightened photo leaves empty corners on its canvas, and no frame Auto proposes reaches them: every frame lies inside the photo straightened at its full size (section 7, step 8), so an Auto crop never exports a black edge. The Crop panel keeps a frame the person moves or straightens inside the photo by the same rule (`docs/11` §Crop).

When the panel opens on a photo whose crop has `auto: true`, the toggle is on and the panel asks for the proposal again to fill the note (the answer is the same: section 9). Auto is locked, like Crop itself, while the photo is developing. A press is never needed for anything else: Auto crop is never applied at import, on export or by Re-run.

**Keyboard.** The toggle is in the tab order before the card's Reset, where it is drawn (the header's order, as in the other cards); Space and Return press it. `A` while the Crop panel is open toggles it (no other shortcut uses A in Crop mode).

## 2. The setting

`Crop` (`ym_core::settings::Crop`, the wire `Crop` and `src/api/types.ts`) gains one field:

```text
auto: bool   // the rect and angle came from Auto crop and follow it while the panel is open
```

- JSON: absent reads as `false`, and `false` is not written, so every existing settings document, `ym:Settings` packet and golden manifest is unchanged. `true` is written as `"auto": true`.
- The rect and angle in the setting are always the concrete values in effect: rendering never looks at `auto`, and an export, a thumbnail and `ym render --settings` give the same pixels whatever it says. `ym:Settings` pins them as it pins every Auto value.
- Copy settings and Paste still leave the crop out (`docs/11`), so a proposal never travels to another photo.
- Undo takes a Done back in one step, `auto` with it.

## 3. The core's API

`Session::suggest_crop(id, aspect, orientation) -> Result<CropSuggestion, SessionError>` (`ym_core::auto_crop`):

```text
CropSuggestion {
  straighten: "level" | "rotate" | "unsure",
  angle_deg: f64,            // the straighten angle the frames are made for: the proposal's when "rotate", else 0
  frames: [CropFrame],       // best first, at most 3, never empty; each a rect of the canvas (Crop.rect's space)
  keep: bool,                // the first frame is the largest frame of the aspect at that angle
  subject: "camera" | "picture",  // where the subject came from: the camera's focus point, or the picture alone
}
CropFrame { rect: NormRect, score: f64 }   // score rounded to 4 decimals; 0 for the kept frame
```

- `aspect` and `orientation` are the draft's: the user's quarter turns decide the canvas, and the aspect decides the ratio (section 7). The analysis of the photo (sections 4 to 6) does not depend on them and is kept in memory for the photo's life in the session (never on disk, hard rule 4); a request after the first costs only the proposer.
- Errors: `UnknownPhoto`; `NotReady` while the photo is developing ("The photo is still developing."); a render error of the picture.
- The app's command is `suggest_crop(photo_id, aspect, orientation)` with the same answer (wire `CropSuggestion`); the mock answers a fixed proposal (section 10).

## 4. What the analysis looks at

**The picture.** Apple Auto's picture (`docs/19` §3, `AdvisorPictures`): the photo at its default settings, the Signature look, our ACES 2 display rendering, 768 px on the long edge, the camera's orientation applied, no turn, straightening or crop of the user's, rendered on the CPU reference path. The session makes its own `AdvisorPictures` on first use, on every platform; it does not need Apple Auto. Its bytes are the same in the app and `ym` (`docs/19` §7), so the proposal is too.

**The camera's hints** (`ym_raw::makernote::SonyFocus`, `docs/04` §Container). From the Sony maker note:

| Tag | Format | Use |
|---|---|---|
| `0x2027` FocusLocation | int16u[4]: width, height, x, y | the focus point: `x / width`, `y / height` of the camera's frame, in sensor orientation |
| `0x204a` FocusLocation2 | int16u[4] | preferred when present ("the same within one pixel") |
| `0x2037` FocusFrameSize | six undefined bytes read as int16u[3]: width, height, valid (`257`; `0` is n/a) | the AF box around the point, in the same units |
| `0x201c` AFAreaModeSetting | int8u | whether the photographer placed the AF area (below) |

- The point's fractions are taken as fractions of the developed photo (the camera's frame is its default crop; the lens stage's auto crop moves a point by well under 1 % of the frame) and turned by the camera's orientation (`geometry::sensor_to_oriented`) into the picture's space. The box's sides swap with the orientation.
- A point of `0 0 0 0`, a width or height of 0, or a point outside the frame is no point. **A point exactly at the centre** (within one unit of `width / 2`, `height / 2`) is no point either: Sony writes the centre "if the focus location information cannot be obtained".
- **Placed on purpose:** the AF-area setting is Flexible Spot (3), Flexible Spot with LA-EA4 (4), Zone (11) or Expanded Flexible Spot (12) on NEX, ILCE, ILME and ZV bodies, or Flexible Spot (4), Zone (8) or Expanded Flexible Spot (12) on ILCA bodies, and the point is at least 10 % of the short side from the centre. The proposer then keeps the frame more readily (section 7).
- Bodies without these tags (older A and NEX bodies, files from other makers) give no hint; the analysis works from the picture alone and the answer says `subject: "picture"`.

## 5. The straightener

On the picture's luma `Y = 0.2126 R' + 0.7152 G' + 0.0722 B'` of the 8-bit values over 255 (gamma-encoded: edges, not light, matter here), at the picture's size.

1. **Gradients.** A Gaussian blur of σ = 1.0 px (a 7-tap kernel, normalised, edges clamped), then Sobel 3 × 3 divided by 8, so the magnitude is the change of luma per pixel. Pixels within 2 px of the border get no gradient.
2. **Line support regions** (after von Gioi et al., *LSD*, TPAMI 2010, written from the paper; no code of the IPOL reference, which is AGPL). A pixel takes part when its gradient magnitude is at least `max(0.03, 2 × the median magnitude)` (the median over the pixels that have a gradient). Seeds are visited in order of falling magnitude (256 magnitude bins from the largest down, a pixel's bin `⌊256 m / m_max⌋` and at most 255; raster order within a bin). From each unused seed a region grows, breadth first, over the 8-neighbours (in raster order around each pixel) whose gradient direction is within 22.5° of the region's mean direction (the mean of the unit vectors so far, updated as pixels join; directions are compared over the full circle, so the two sides of a thin line make two regions). A pixel a region took stays taken, even when the region is dropped. A region of fewer than 12 pixels is dropped.
3. **Segments.** Each region's magnitude-weighted centroid and second moments give its axis (the eigenvector of the larger eigenvalue). Length and width are the spans of its pixels (at their centres) along and across the axis, plus 1; its midpoint is the middle of its span along the axis. A region is a segment when `length ≥ max(20 px, 0.04 × the long side)`, `length ≥ 5 × width` and it holds at least half of `length × width` pixels. Its weight is `length × min(1, mean magnitude / 0.15)`; its angle `θ` is the axis's, in (−90°, 90°], measured in the picture's coordinates (y down), so a positive angle is a line turned clockwise.
4. **Verticals.** Segments within 12° of vertical, with deviation `δ = θ − 90°` for `θ > 0` and `θ + 90°` otherwise. A camera pitched up or down makes verticals lean apart about the centre column, so the model is `δ = τ + k · (x − W/2) / W`, with `x` the segment's midpoint, fitted by 8 iterations of weighted least squares whose weights are the segments' times Tukey's biweight of their residuals (c = 1.5°), starting from the weighted median of `δ` (the first `δ`, in ascending order, where the cumulative weight reaches half) with `k = 0`. `k` is held at 0 when the weighted standard deviation of `x / W` is below 0.1. The estimate `τ_v` counts when there are at least 3 near-vertical segments of total length at least 0.6 H, the inliers (residual within c) carry at least 75 % of their weight (raised from 60 % after the real-photo evaluation of 2026-10-23: a squirrel on a leaning tree trunk, whose bark and a feeder's cage bars seen in perspective agreed at only 62 %, was turned 2.9° the wrong way while every photo it straightened correctly had 79 % or more) and the inliers' RMS residual, weighted by the segments' weights, is at most 1.0°.
5. **Horizon.** Segments within 12° of horizontal (`δ = θ`). Only collinear horizontals say the camera's roll; the edges of a façade converge. The segments within 0.6° of the peak of their weight histogram (bins of 0.2° over ±12°, smoothed by `[1, 2, 3, 2, 1] / 9`; the peak is the centre of the highest bin, the first on a tie) are grouped by the height at which their line crosses the centre column, in clusters 2 % of H tall (sorted; a cluster takes the next segments while they cross within 2 % of H of its first). The heaviest cluster (by weight, the first on a tie) gives `τ_h`, its weighted mean `δ`, and counts when its segments' total length is at least 0.35 W. When it counts, the **horizon's height** is the mean of its crossings of the centre column weighted as the segments, as a fraction of H; the proposer keeps the horizon where the photographer put it (section 7), whatever the decision below.
6. **Decision.** Both counted: when they agree within 1.0°, `τ` is their mean weighted by inlier length (the verticals' inliers' total length, the horizon cluster's total length); otherwise **unsure**. One counted: that one. None: **unsure**. Then `|τ| ≤ 0.2°` is **level**, `|τ| > 10°` is **unsure** (a tilt that large is more often meant than a mistake), and anything else is **rotate** by `angle_deg = −τ` (the straighten angle turns the photo clockwise for positive angles, `geometry`).

The band 0.2° to 10° and the agreement of 1.0° are the owner's to change (section 12).

## 6. The subject map

On a map of the picture at a quarter of its size (4 × 4 boxes averaged; the remainder at the right and bottom dropped), long side 192 cells for a 768 px picture.

1. **Saliency.** Each cell's colour in CIELAB (sRGB decoded, the 16 pixels averaged in linear light, D65). The minimum barrier distance of each cell to the picture's border (Zhang et al., *Minimum Barrier Salient Object Detection at 80 FPS*, ICCV 2015, written from the paper) on L, a and b separately, by its raster-scan approximation: border cells 0, every other cell infinite, then three pairs of passes (forward from the top left using the upper and left neighbours, backward from the bottom right using the lower and right ones), each keeping per cell the running maximum and minimum along its best path. The sum over the three channels, divided by its 99th percentile and clamped to 1, is the saliency `M`.
2. **Focus.** The absolute Laplacian of luma at the picture's size (4-neighbour, edges clamped), averaged over each cell, divided by its 95th percentile and clamped to 1, then a 3 × 3 box mean (over the cells inside the map): `F`. When the photo is sharp everywhere this says little, so its weight is `f = clamp((p95 − p50) / p95, 0, 1)` of the cell values before division.
3. **Evidence.** `E = (M + f · F) / (1 + f)`.
4. **The camera's point.** With a point (section 4), the evidence's mass (not where the components are, step 5) is weighted by `0.35 + 0.65 · exp(−d² / 2σ²)`, `d` the distance from the cell's centre to the point in cells, `σ` the larger of 15 % of the map's short side and 0.75 × the box's larger side in cells.
5. **Subject and distractions.** Cells whose unweighted `E` is at least half of its 99th percentile form components (4-connected, labelled in raster order), each carrying the weighted mass of step 4; so a part of the subject away from the camera's point (a skull above the ribs it focused on) is still a component. The **core** is the component holding the cell under the camera's point when it holds at least 10 % of the thresholded mass, otherwise the heaviest (ties: the lowest label). Every component is part of the **subject** except a **distraction**: a component, not the core, that reaches into the band of 3 % of the map's side along its border and has less than half the core's mass (a branch, a post or a bright patch the frame cuts into). The subject map `S` is the weighted `E` on the subject's cells, 0 elsewhere, summing to 1; the distraction map `D` likewise on the distractions, summing to 1 when there are any. `ρ`, the distractions' share of the thresholded mass, says how much they matter.

## 7. The frame proposer

For a request (aspect, quarter turns, angle):

1. **The canvas.** The maps turned by the user's quarter turns (an exact permutation), then sampled on the canvas at the straighten angle: each canvas cell takes the bilinear value of the turned map at the cell centre rotated back about the canvas centre; cells that fall outside the photo are 0. The canvas has as many cells as the turned map. Summed-area tables of `S` and `D` make every sum below a constant-time lookup; they are read bilinearly, as if each cell's value were spread evenly over it, so a frame's edges need not fall on cell edges. (Frames are checked against the photo exactly, step 4, so no table of valid cells is needed.)
2. **The ratio** (width over height, in pixels of the canvas): Original and Free the canvas's own; Square 1; 4:5 is 4/5 on a portrait canvas and 5/4 on a landscape one; 16:9 is 16/9 on a landscape canvas and 9/16 on a portrait one. (Free proposes in the canvas's shape: the person can drag from there.)
3. **The largest frame.** For a W × H canvas turned by θ, the largest frame of ratio r centred on the canvas with no empty corner has `h = min(W / (r |cos θ| + |sin θ|), H / (r |sin θ| + |cos θ|))` and `w = r h`; at θ = 0 that is the largest rectangle of ratio r inside the canvas. The frame also stays inside the canvas (`h ≤ H`, `w ≤ W`), which binds only when `tan(|θ| / 2)` exceeds `min(r, 1 / r)`, far beyond 10°. This is the **kept frame**.
4. **Candidates.** Six sizes, 1.0, 0.94, 0.88, 0.82, 0.76 and 0.70 times the kept frame's sides; for each, a 9 × 9 grid over the positions where a frame of that size fits in the canvas (the near edge at `i / 8` of the room, `i` from 0 to 8; one centred position along an axis where it fits exactly), keeping those whose four corners are inside the turned photo. Order: sizes from the largest, then rows from the top, then columns from the left; the kept frame first among the largest.
5. **Score** of a frame R, with `a` its area over the kept frame's. The photographer framed the photo, so the kept frame's centre is where a frame starts from, and it moves only for what it holds, slices or leaves out:
   - `lost` = the subject mass the kept frame holds less the mass R holds, when that is positive (else 0); `gained` = the mass R holds beyond the kept frame's, when positive (else 0). A loss weighs four times a gain: a frame does not give up what the photographer put in to take in a little of what they left out, but takes in a large part (a quarter of the subject, say) that the kept frame of a narrower aspect leaves out;
   - `cut` = the subject mass in the bands 3 % of the canvas's short side wide inside R's edges (their union), counting only the edges more than one cell inside the canvas (an edge on the canvas's border slices nothing), less the same for the kept frame: negative for a frame that slices less of the subject than the kept frame does (the kept square of a landscape photo can slice through it);
   - `removed` = `ρ ×` the distraction mass outside R, less the distraction mass outside the kept frame;
   - `shift` = the distance from R's centre to the kept frame's, over the kept frame's longer side;
   - `horizon` (with a horizon, section 5 step 5) = how much further from where the photographer put it R moves the horizon than the kept frame does, when that is positive (else 0). The horizon's place is its height as a fraction of the canvas (across it, `1 −` or the height itself, when one or three quarter turns stand it up); its place in a frame is the same fraction of the frame. The straighten angle moves its crossing of the centre line by under 2 % of its distance from the centre at 10°, so it is left out. A frame that keeps the horizon at the same height, Square sliding sideways for one, pays nothing; a 16:9 frame sliding down a landscape to hold a sharp field lifts the horizon in the frame and pays;
   - `score = −4 lost + gained − 2 cut + removed − 0.5 shift − 1.5 (1 − a) − 2 horizon`, which is 0 for the kept frame.
   - Why so: on 57 real photos (12 local, 45 from Play Raw), in blind pairs of Auto's Square and 16:9 frames against the centred frame of the same size and angle, the first version (the subject on the thirds) won 18, lost 20 and tied 15 of the 53 pairs where it moved; the second (the subject centred in the frame) won 14, lost 21 and tied 13 of 48; a third, this one with a gain weighing as much as a loss, won 14, lost 14 and tied 15 of 43. They moved the photographer's framing for small differences in the maps, pushing a bee or two runners to the frame's side. This version won 12, lost 3 and tied 5 of the 20 pairs where it moved; in the other 94 its frame was the centred one. That was on the photos it was tuned on. On 150 further Play Raw photos nobody had looked at (a held-out test, the same judge, the same pairs), it won 21, lost 18 and tied 20 of the 59 pairs where it moved: no better than the centred frame. With the thresholds of step 6 it moves only on the 30 of those pairs it is surest of, and wins 14, loses 6 and ties 10 of them (63 % with a tie as half; the 95 % interval runs from 46 % to 78 %), with the centred frame on the other 270; the threshold was read off these same pairs as well as the development photos, so that figure is not a clean test. One judge, the agent; the owner's A/B decides (section 12).
   - With a camera box, frames that do not hold the box grown by 10 % on each side (and clamped to the picture) are dropped, the kept frame too. When every frame is dropped, the kept frame alone is the answer.
6. **Choice.** Frames sorted by score (ties: the lower index); a frame is dropped when its intersection over union with one already chosen exceeds 0.6; at most 3. The kept frame is first unless the best score exceeds **0.15**, or **0.3** when the point was placed on purpose (section 4), or the camera box dropped it. (The score says how sure Auto is, and on real photos it says it well: in the blind pairs below, frames that scored under 0.15 lost to the centred frame 15 times and won 12 times; from 0.15 to 0.3 they won 12 and lost 6; above 0.3 they won 9 and never lost. The development photos alone gave the same line: their three losses all scored under 0.14. The thresholds were 0.05 and 0.15.) `keep` says the kept frame is first.
7. **Rounding.** Rects are written in canvas fractions: the edges rounded to 6 decimals and clamped to `[0, 1]`, the width and height their differences. Scores are rounded to 4 decimals.
8. **Inside the photo at its full size** (`ym_core::auto_crop::suggestion`, "No border in photos"). Steps 1 to 7 work on the picture, whose sides are rounded to whole pixels, so a frame that touches the turned photo there can pass its edge by a pixel or more at the photo's own size (6048 × 4024 is not exactly the 768 × 511 picture's shape). Each frame is therefore fitted on the canvas of the photo's own size (`PhotoSummary::width_px` × `height_px`, turned by the quarter turns) with `ym_core::pipeline::geometry::fit_inside`: shrunk about its centre, keeping its shape, by as little as makes its four corners sample the straightened photo at least `FIT_MARGIN_PX` = 2 px from its edge (the margin covers the reduced views, whose canvases are rounded too), then rounded to 6 decimals again. At an angle of 0 nothing changes. Scores and `keep` are those of step 6.

## 8. The command line

`ym info FILE --suggest-crop[=ASPECT]` (ASPECT one of `original` (the default), `free`, `1:1` or `square`, `4:5`, `16:9`) adds the proposal for the photo as the camera turned it (no quarter turn of the user's; `ym info` takes no settings):

- text: `Auto crop    Levelled by 1.4° · frame 0.0412 0.0310 0.9176 0.9380 (from the camera's focus point)`, or `Already level · whole photo kept (from the picture)`, or `No horizon found · frame …`;
- `--json`: `"auto_crop": {CropSuggestion}`, the app's answer field for field.

Without the flag `ym info` does not render the picture.

`ym render FILE… --auto-crop[=ASPECT]` (the same aspects) gives each photo the crop the panel's Auto gives it: the proposal's angle and first frame for that aspect at the photo's orientation (from `--settings`, else the camera's), with `auto: true`; the whole photo at 0° is no crop, as the panel's Done sends it. It replaces a crop from `--settings`. With `--json` the file's record gains `"auto_crop"`, the proposal as above, and its `settings` show the crop. A photo with no proposal is exported as its settings say, with a warning, and `"auto_crop": {"error": "…"}`. The same crop given as `--settings` writes the same bytes, which is what the app's Done does (`crates/ym-cli/tests/render.rs`). The blind A/B of Auto crop renders with it (`tools/eval/ab`, the crop test).

## 9. Determinism and speed

Every step is f64 on the CPU, single-threaded, with fixed iteration orders and explicit tie rules; no random numbers. A percentile `p` of `n` values is the value at index `⌊p (n − 1)⌋` of the values sorted in ascending order. The picture is the CPU reference render. Same file, settings and version give the same `CropSuggestion` in the app and `ym`, on every platform. (The steps use `exp`, `sin`, `cos`, `atan2`, `cbrt` and `powf` from the system's maths library; two platforms whose libraries differ in a last bit could differ only where a value sits exactly on a threshold or a rounding step. The app and `ym` on one Mac always agree.) Budget on one core of an M1 (estimates until measured): the picture as Apple Auto's (the reduced raw path), then about 30 ms for the straightener, 20 ms for the maps and under 5 ms per proposal. Measured on one core of a 2.8 GHz Xeon (Linux, release build, a 768 × 512 synthetic picture, 2026-10-22): the straightener 40 ms, the maps 16 ms, three proposals 4.5 ms together.

## 10. The app

- `src-tauri`: the command `suggest_crop` and the wire types `CropSuggestion` and `CropFrame` (`bindings.ts` regenerated).
- `src/api.ts` (`suggestCrop`), the mock (`src/api.mock.ts`): a fixed proposal (rotate 1.4°, a frame of 90 % centred a little to the left) after 150 ms, or `keep` for photos whose name has "keep" in it, so tests and screenshots can show both.
- `src/screens/crop/useCropSession.ts` (the draft's `auto` and the toggle; Auto's frame is fitted before it becomes the base frame), `CropPanel.tsx` (the toggle and the note), `cropMath.ts` (`fitInside`, the port of `geometry::fit_inside`, and `constrainFrame` for drags: "No border in photos", `docs/11` §Crop).
- e2e: the state `36-crop-auto` (the panel with Auto on and the note), its glass audit and a11y check.

## 11. Not in this version

| Step | Why later |
|---|---|
| A learned frame scorer (about 0.3 to 0.8 M parameters, trained from scratch on CC0, CC BY and CC BY-SA photos) | it must beat this proposer in the owner's blind A/B first |
| Vision evidence on Macs (saliency, faces, people, horizon) | a second opinion that must agree with the straightener; measured on Macs |
| A Foundation Models chooser among the three frames | macOS 27 only; latency |
| Keystone (Upright-style) correction | a different tool; Adobe holds patents near it |
| `ym render --crop auto` | after the A/B |

## 12. Decisions for the owner

Recorded in `docs/STATUS.md` with the defaults this version uses: the straighten band (0.2° to 10°) and agreement (1.0°); Free proposing in the canvas's shape; the proposer's weights and margins (section 7, set by eye on synthetic scenes); and the blind A/B (about 150 photos, Auto crop against the original framing, the gate of `tools/eval/ab`) before Auto crop goes to `main`.

## 13. Tests

- `crates/ym-compose`: straightener on synthetic pictures drawn analytically (4 × 4 samples per pixel, fixed-seed noise) and turned by known angles of 0.5° to 9° either way (a façade with converging verticals seen by a camera pitched up 8°, a sea horizon, both, texture without lines, a 15° tilt, level scenes): within 0.15° of the truth where it answers, unsure on the texture and the 15° tilt, level when level, the sign of the angle; the largest-frame formula against a brute-force search; the proposer on synthetic maps and rendered pictures (a centred subject is kept; a strong distraction at one edge (40 % of the thresholded mass) is cropped away without losing the subject, and with the camera's point placed on purpose only when the score passes 0.3; a weak rendered one moves no frame; a subject the centred square slices through is brought inside whole, and one inside the square, off its centre, is left where it is (with Original at 0° the whole photo is kept); a part of the subject outside the centred square moves the frame when it is 27 % of the mass and not when it is 6 %; a subject wholly outside the square is brought in (the quarter-turn tests); a strip of the subject across the kept 16:9 frame's bottom edge (20 % of the mass) pulls the frame down without a horizon and not with one across the middle, a strip of 35 % still does, a quarter turn makes the horizon hold the frame across instead, and Square sliding sideways is the same with a horizon as without; the horizon's height is where its cluster crosses the centre column; a second part of the subject away from the camera's point (a skull above focused-on ribs) is never cut, for Original, Square and 4:5; a small component at the border distracts while one inside the picture or of half the core's mass or more joins the subject; the camera box is never cut; frames inside `[0, 1]` and the turned photo, with the requested ratio; the quarter turns map as `geometry` does); determinism (two runs, the same bytes of the serialised answer).
- `crates/ym-raw`: the four tags from synthetic maker notes (`testgen`), and the values exiftool prints for the local real files when they are present.
- `crates/ym-core` geometry: `fit_inside` on random frames, angles up to 45° and four photo shapes leaves no output pixel on the frame's edge without a source, at full resolution (pixel-exact) and in views reduced 3, 8 and 16 times; it keeps the frame's shape, shrinks no more than it must (0.1 % larger no longer fits), gives the largest frame of step 3 for the whole photo, changes nothing at 0°, and replaces a frame whose centre is outside by the centred largest. `auto_crop`: a sea horizon tilted 3° in a 768 × 512 picture, for a 6048 × 4024 photo, Original, Square and 16:9, unturned and turned: every frame's export at full size has no empty edge pixel (the test fails without the fit).
- `crates/ym-core`: `suggest_crop` on a synthetic ARW (frames inside the canvas, the kept frame's ratio, the same answer from two sessions); the camera point's orientation mapping; `Crop.auto` in JSON (absent is false, false is not written).
- `crates/ym-cli`: `ym info --suggest-crop --json` equals the session's answer (app–CLI parity); `ym render --auto-crop=1:1 --json` reports the proposal and crops to its first frame and angle with `auto`, and the same crop through `--settings` writes the same bytes; the whole photo at 0° is no crop and a crop from `--settings` is replaced; an unknown aspect is a usage error.
- Frontend: the panel's toggle (on applies the frame and the angle; an aspect change asks again; a drag or Straighten turns it off; off restores; Done sends `auto`), the note's lines, the mock, e2e `36-crop-auto`.

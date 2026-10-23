# 20 · Auto crop

Auto crop levels a tilted photo and proposes a frame for it. It is an **Auto** toggle in Crop mode: while it is on, the straighten angle and the frame follow the core's proposal for the chosen aspect, and moving either by hand turns it off. This file covers what the person sees, the setting, the core's API, what the analysis looks at, the straightener, the crop model, the frame proposer, the command line, determinism and the tests.

Everything here runs in Rust, the same in the app and `ym` on every platform: the straightener is deterministic geometry, and the frame comes from a small crop model of our own (section 6, 1.1M parameters, `data/models/crop/`) run on the CPU. The owner asked for Auto crop on 2026-10-22 ("The mode auto should be available in crop mode, as a toggle button"). **2026-10-23: the owner stopped the first, classical proposer** ("crop composition is pretty bad, cutting objects, no thirds rule"), and chose a learned one (`research/crop-models-2026-10-23.md`): a student of FrameLite's crops, pretrained on Open Images' boxes, then "Let's put the model into app". The classical subject map and scoring are gone; the toggle, the straightener and the frame geometry stay. Vision evidence on Macs and a Foundation Models chooser are later steps (section 11). The owner's blind A/B decides whether Auto crop goes to `main`.

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
  subject: "camera" | "picture",  // where the subject came from: always "picture" (the crop model reads the picture alone)
}
CropFrame { rect: NormRect, score: f64 }   // its intersection over union with the crop model's crop, rounded to 4 decimals
```

- `aspect` and `orientation` are the draft's: the user's quarter turns decide the canvas, and the aspect decides the ratio (section 7). The analysis of the photo (sections 4 to 6) does not depend on them and is kept in memory for the photo's life in the session (never on disk, hard rule 4); a request after the first costs only the proposer.
- Errors: `UnknownPhoto`; `NotReady` while the photo is developing ("The photo is still developing."); a render error of the picture.
- The app's command is `suggest_crop(photo_id, aspect, orientation)` with the same answer (wire `CropSuggestion`); the mock answers a fixed proposal (section 10).

## 4. What the analysis looks at

**The picture.** Apple Auto's picture (`docs/19` §3, `AdvisorPictures`): the photo at its default settings, the Signature look, our ACES 2 display rendering, 768 px on the long edge, the camera's orientation applied, no turn, straightening or crop of the user's, rendered on the CPU reference path. The session makes its own `AdvisorPictures` on first use, on every platform; it does not need Apple Auto. Its bytes are the same in the app and `ym` (`docs/19` §7), so the proposal is too.

**The camera's focus point** (`ym_raw::makernote::SonyFocus`: FocusLocation `0x2027`/`0x204a`, FocusFrameSize `0x2037`, AFAreaModeSetting `0x201c`, `docs/04` §Container) is still read from the maker note but no longer used: the classical proposer weighted its subject map with it, and the crop model looks at the picture alone. A model that also reads the point is a later step (section 11).

## 5. The straightener

On the picture's luma `Y = 0.2126 R' + 0.7152 G' + 0.0722 B'` of the 8-bit values over 255 (gamma-encoded: edges, not light, matter here), at the picture's size.

1. **Gradients.** A Gaussian blur of σ = 1.0 px (a 7-tap kernel, normalised, edges clamped), then Sobel 3 × 3 divided by 8, so the magnitude is the change of luma per pixel. Pixels within 2 px of the border get no gradient.
2. **Line support regions** (after von Gioi et al., *LSD*, TPAMI 2010, written from the paper; no code of the IPOL reference, which is AGPL). A pixel takes part when its gradient magnitude is at least `max(0.03, 2 × the median magnitude)` (the median over the pixels that have a gradient). Seeds are visited in order of falling magnitude (256 magnitude bins from the largest down, a pixel's bin `⌊256 m / m_max⌋` and at most 255; raster order within a bin). From each unused seed a region grows, breadth first, over the 8-neighbours (in raster order around each pixel) whose gradient direction is within 22.5° of the region's mean direction (the mean of the unit vectors so far, updated as pixels join; directions are compared over the full circle, so the two sides of a thin line make two regions). A pixel a region took stays taken, even when the region is dropped. A region of fewer than 12 pixels is dropped.
3. **Segments.** Each region's magnitude-weighted centroid and second moments give its axis (the eigenvector of the larger eigenvalue). Length and width are the spans of its pixels (at their centres) along and across the axis, plus 1; its midpoint is the middle of its span along the axis. A region is a segment when `length ≥ max(20 px, 0.04 × the long side)`, `length ≥ 5 × width` and it holds at least half of `length × width` pixels. Its weight is `length × min(1, mean magnitude / 0.15)`; its angle `θ` is the axis's, in (−90°, 90°], measured in the picture's coordinates (y down), so a positive angle is a line turned clockwise.
4. **Verticals.** Segments within 12° of vertical, with deviation `δ = θ − 90°` for `θ > 0` and `θ + 90°` otherwise. A camera pitched up or down makes verticals lean apart about the centre column, so the model is `δ = τ + k · (x − W/2) / W`, with `x` the segment's midpoint, fitted by 8 iterations of weighted least squares whose weights are the segments' times Tukey's biweight of their residuals (c = 1.5°), starting from the weighted median of `δ` (the first `δ`, in ascending order, where the cumulative weight reaches half) with `k = 0`. `k` is held at 0 when the weighted standard deviation of `x / W` is below 0.1. The estimate `τ_v` counts when there are at least 3 near-vertical segments of total length at least 0.6 H, the inliers (residual within c) carry at least 75 % of their weight (raised from 60 % after the real-photo evaluation of 2026-10-23: a squirrel on a leaning tree trunk, whose bark and a feeder's cage bars seen in perspective agreed at only 62 %, was turned 2.9° the wrong way while every photo it straightened correctly had 79 % or more) and the inliers' RMS residual, weighted by the segments' weights, is at most 1.0°.
5. **Horizon.** Segments within 12° of horizontal (`δ = θ`). Only collinear horizontals say the camera's roll; the edges of a façade converge. The segments within 0.6° of the peak of their weight histogram (bins of 0.2° over ±12°, smoothed by `[1, 2, 3, 2, 1] / 9`; the peak is the centre of the highest bin, the first on a tie) are grouped by the height at which their line crosses the centre column, in clusters 2 % of H tall (sorted; a cluster takes the next segments while they cross within 2 % of H of its first). The heaviest cluster (by weight, the first on a tie) gives `τ_h`, its weighted mean `δ`, and counts when its segments' total length is at least 0.35 W. When it counts, the **horizon's height** is the mean of its crossings of the centre column weighted as the segments, as a fraction of H (the classical proposer used it; the crop model does not).
6. **Decision.** Both counted: when they agree within 1.0°, `τ` is their mean weighted by inlier length (the verticals' inliers' total length, the horizon cluster's total length); otherwise **unsure**. One counted: that one. None: **unsure**. Then `|τ| ≤ 0.2°` is **level**, `|τ| > 10°` is **unsure** (a tilt that large is more often meant than a mistake), and anything else is **rotate** by `angle_deg = −τ` (the straighten angle turns the photo clockwise for positive angles, `geometry`).

The band 0.2° to 10° and the agreement of 1.0° are the owner's to change (section 12).

## 6. The crop model

A small convolutional network of our own (`ym_ml::crop_net`, `data/models/crop/cropnet.safetensors`, its model card beside it) looks at the picture of section 4 and answers one free crop: `[x0, y0, x1, y1]` as fractions of the picture's width and height.

- **Input.** The picture letterboxed into a 256 × 256 square: scaled to fit (its long side 256 px, the short side rounded half to even) with the bilinear filter of Pillow's `Image.resize`, reproduced to the bit (`crop_net::letterbox`; the training data went through Pillow), centred on black; then `(v / 255 − 0.5) / 0.25` per channel.
- **Network.** A 3 × 3 stem of stride 2 (24 channels), seventeen inverted-residual blocks in five stages (expansion 1 then 4; 24, 32, 48, 96, 160 channels; strides 1, 2, 2, 2, 2): a 1 × 1 expansion, a 3 × 3 depthwise convolution, a 1 × 1 projection and a residual when the shape is kept; a 1 × 1 neck to 32 channels at 8 × 8; linear layers of 256 and 4. SiLU after every layer but the projections and the last; the batch norms are folded into the convolutions. The last four outputs, through a sigmoid, are `p` in `[0, 1]`, and `x0 = p0`, `x1 = x0 + (1 − x0) p2`, the same for y: every answer is a box inside the picture. 1,101,852 parameters, 263 M multiply-adds.
- **Training** (`training/crop/`, the model card). The backbone first learned, from 96,000 Open Images photos and the boxes people drew on them, where objects, people, faces, animals, vehicles and buildings are and which of 600 classes a photo holds; then the network learned to crop from FrameLite's crops of 11,999 other Open Images photos (FrameLite: an Apache-2.0 cropper run offline as a teacher, the owner's decision of 2026-10-23). Photos CC BY 2.0, boxes CC BY 4.0, so the weights keep Apache-2.0 (`docs/14` rule 5), with the photos credited in the model card.
- **Loading.** `SessionEnv::load` reads it from `<data_root>/models/crop/`, checked against `SHA256SUMS` there. A missing or damaged file is not fatal: Auto crop then answers "Couldn't suggest a frame for this photo: the crop model could not be loaded (…)".
- **Parity.** The training run's own answers (PyTorch) for five drawn pictures are a fixture (`crates/ym-ml/tests/data/cropnet_parity.json`, written by `training/crop/export.py`): the letterbox's bytes must match Pillow's exactly and the four numbers within 2e-4 (measured: 2.2e-7).

## 7. The frame proposer

For a request (aspect, quarter turns, angle), from the crop model's crop:

1. **The canvas.** The picture turned by the user's quarter turns, then by the straighten angle about its centre.
2. **The target.** The crop's corners turned by the quarter turns (`geometry`'s exact map), its centre moved where the straighten angle takes it, its width and height kept: an upright box `T` on the canvas.
3. **The ratio** (width over height, in pixels of the canvas): Original the canvas's own; Square 1; 4:5 is 4/5 on a portrait canvas and 5/4 on a landscape one; 16:9 is 16/9 on a landscape canvas and 9/16 on a portrait one. **Free** is the crop's own shape: the answer is `T` itself, shrunk about its centre (keeping its shape, by bisection) by as little as puts its corners inside the canvas and the turned photo, a single frame.
4. **The largest frame.** For a W × H canvas turned by θ, the largest frame of ratio r centred on the canvas with no empty corner has `h = min(W / (r |cos θ| + |sin θ|), H / (r |sin θ| + |cos θ|))` and `w = r h`; at θ = 0 that is the largest rectangle of ratio r inside the canvas. The frame also stays inside the canvas (`h ≤ H`, `w ≤ W`), which binds only when `tan(|θ| / 2)` exceeds `min(r, 1 / r)`, far beyond 10°. This is the **kept frame**.
5. **Candidates.** Six sizes, 1.0, 0.94, 0.88, 0.82, 0.76 and 0.70 times the kept frame's sides; for each, a 9 × 9 grid over the positions where a frame of that size fits in the canvas (the near edge at `i / 8` of the room, `i` from 0 to 8; one centred position along an axis where it fits exactly), keeping those whose four corners are inside the turned photo. Order: sizes from the largest, then rows from the top, then columns from the left; the kept frame first among the largest.
6. **Choice.** Each candidate's score is its intersection over union with `T`. Frames sorted by score (ties: the lower index, so the larger frame); a frame is dropped when its intersection over union with one already chosen exceeds 0.6; at most 3. `keep` says the kept frame is first (for Free: `T` is the kept frame of the canvas's shape).
7. **Rounding.** Rects are written in canvas fractions: the edges rounded to 6 decimals and clamped to `[0, 1]`, the width and height their differences. Scores are rounded to 4 decimals.
8. **Inside the photo at its full size** (`ym_core::auto_crop::suggestion`, "No border in photos"). Steps 1 to 7 work on the picture, whose sides are rounded to whole pixels, so a frame that touches the turned photo there can pass its edge by a pixel or more at the photo's own size (6048 × 4024 is not exactly the 768 × 511 picture's shape). Each frame is therefore fitted on the canvas of the photo's own size (`PhotoSummary::width_px` × `height_px`, turned by the quarter turns) with `ym_core::pipeline::geometry::fit_inside`: shrunk about its centre, keeping its shape, by as little as makes its four corners sample the straightened photo at least `FIT_MARGIN_PX` = 2 px from its edge (the margin covers the reduced views, whose canvases are rounded too), then rounded to 6 decimals again. At an angle of 0 nothing changes. Scores and `keep` are those of step 6.

**How good it is** (`research/crop-models-2026-10-23.md`, the agent's blind pairs on 150 Play Raw photos nobody trained on, judged and then unblinded). The free crop against the whole photo: won 56, lost 10, tied 39 of 105 (72 %, a tie as half; FrameLite itself 67 %). With Original, the closest frame of the photo's own shape against the whole photo, a fully blind test: it kept the whole photo on 37, and of the other 113 won 36, lost 15 and tied 62 (59 %). One judge who had seen the photos before: a pointer, not the gate. The owner's A/B (section 12) decides.

## 8. The command line

`ym info FILE --suggest-crop[=ASPECT]` (ASPECT one of `original` (the default), `free`, `1:1` or `square`, `4:5`, `16:9`) adds the proposal for the photo as the camera turned it (no quarter turn of the user's; `ym info` takes no settings):

- text: `Auto crop    Levelled by 1.4° · frame 0.0412 0.0310 0.9176 0.9380 (from the picture)`, or `Already level · whole photo kept (from the picture)`, or `No horizon found · frame …`;
- `--json`: `"auto_crop": {CropSuggestion}`, the app's answer field for field.

Without the flag `ym info` does not render the picture.

`ym render FILE… --auto-crop[=ASPECT]` (the same aspects) gives each photo the crop the panel's Auto gives it: the proposal's angle and first frame for that aspect at the photo's orientation (from `--settings`, else the camera's), with `auto: true`; the whole photo at 0° is no crop, as the panel's Done sends it. It replaces a crop from `--settings`. With `--json` the file's record gains `"auto_crop"`, the proposal as above, and its `settings` show the crop. A photo with no proposal is exported as its settings say, with a warning, and `"auto_crop": {"error": "…"}`. The same crop given as `--settings` writes the same bytes, which is what the app's Done does (`crates/ym-cli/tests/render.rs`). The blind A/B of Auto crop renders with it (`tools/eval/ab`, the crop test).

## 9. Determinism and speed

The straightener and the proposer are f64 on the CPU, single-threaded, with fixed iteration orders and explicit tie rules; no random numbers. A percentile `p` of `n` values is the value at index `⌊p (n − 1)⌋` of the values sorted in ascending order. The crop model is f32 on one thread in a fixed order (no fused multiply-adds, no SIMD paths chosen at run time), so the same picture gives the same bits on every run. The picture is the CPU reference render. Same file, settings and version give the same `CropSuggestion` in the app and `ym`, on every platform. (The steps use `exp`, `sin`, `cos`, `atan2` and `powf` from the system's maths library; two platforms whose libraries differ in a last bit could differ only where a value sits exactly on a threshold or a rounding step. The app and `ym` on one Mac always agree.) Measured on one core of a 2.8 GHz Xeon (Linux, release build, 2026-10-22 and 2026-10-23): the straightener 40 ms on a 768 × 512 picture, the crop model 120 ms, three proposals under 5 ms together; on top comes the picture, as Apple Auto's (the reduced raw path).

## 10. The app

- `src-tauri`: the command `suggest_crop` and the wire types `CropSuggestion` and `CropFrame` (`bindings.ts` regenerated).
- `src/api.ts` (`suggestCrop`), the mock (`src/api.mock.ts`): a fixed proposal (rotate 1.4°, a frame of 90 % centred a little to the left, then the kept frame) after 150 ms, or `keep` for photos whose name has "keep" in it, so tests and screenshots can show both.
- `src/screens/crop/useCropSession.ts` (the draft's `auto` and the toggle; Auto's frame is fitted before it becomes the base frame), `CropPanel.tsx` (the toggle and the note), `cropMath.ts` (`fitInside`, the port of `geometry::fit_inside`, and `constrainFrame` for drags: "No border in photos", `docs/11` §Crop).
- e2e: the state `36-crop-auto` (the panel with Auto on and the note), its glass audit and a11y check.

## 11. Not in this version

| Step | Why later |
|---|---|
| The camera's focus point as an input of the crop model | needs training photos with AF data; FrameLite's labels and Open Images have none |
| A second, larger pretraining of the crop model (about twice the photos and time) | the step that helped; question 23 in `docs/STATUS.md` |
| Vision evidence on Macs (saliency, faces, people, horizon) | a second opinion that must agree with the straightener; measured on Macs |
| A Foundation Models chooser among the three frames | macOS 27 only; latency |
| Keystone (Upright-style) correction | a different tool; Adobe holds patents near it |
| `ym render --crop auto` | after the A/B |

## 12. Decisions for the owner

Recorded in `docs/STATUS.md` with the defaults this version uses: the straighten band (0.2° to 10°) and agreement (1.0°); the aspects' frames as the candidates closest to the crop model's crop (section 7); and the blind A/B (about 150 photos, Auto crop against the original framing, the gate of `tools/eval/ab`) before Auto crop goes to `main`.

## 13. Tests

- `crates/ym-ml` (`tests/crop_net.rs`): the shipped weights are the fixture's; the letterbox gives Pillow's bytes exactly for five drawn pictures (768 × 512, 511 × 768, 768 × 768, 300 × 199, 97 × 64); the four numbers and the box agree with PyTorch within 2e-4; the same picture gives the same bits; a damaged or truncated file is refused; the timing (ignored, release). In the module: the fixed-point coefficients sum to one, a flat picture stays flat and centred.
- `crates/ym-compose`: straightener on synthetic pictures drawn analytically (4 × 4 samples per pixel, fixed-seed noise) and turned by known angles of 0.5° to 9° either way (a façade with converging verticals seen by a camera pitched up 8°, a sea horizon, both, texture without lines, a 15° tilt, level scenes): within 0.15° of the truth where it answers, unsure on the texture and the 15° tilt, level when level, the sign of the angle; the largest-frame formula against a brute-force search; the proposer (`frame.rs`): a crop of the whole picture keeps the photo; the first frame is the candidate closest to the crop, for four crops, four turns, three ratios and three angles (brute force), the frames best first; Free gives the crop itself, turned with the photo, and shrinks it into the photo at an angle; a crop on one side moves the square there under each quarter turn; frames inside `[0, 1]` and the turned photo, with the requested ratio; determinism (two runs, the same bytes of the serialised answer).
- `crates/ym-raw`: the four maker-note tags from synthetic maker notes (`testgen`), and the values exiftool prints for the local real files when they are present.
- `crates/ym-core` geometry: `fit_inside` on random frames, angles up to 45° and four photo shapes leaves no output pixel on the frame's edge without a source, at full resolution (pixel-exact) and in views reduced 3, 8 and 16 times; it keeps the frame's shape, shrinks no more than it must (0.1 % larger no longer fits), gives the largest frame of step 4 for the whole photo, changes nothing at 0°, and replaces a frame whose centre is outside by the centred largest. `auto_crop`: a sea horizon tilted 3° in a 768 × 512 picture, for a 6048 × 4024 photo, Original, Free, Square and 16:9, unturned and turned: every frame's export at full size has no empty edge pixel.
- `crates/ym-core` (`tests/auto_crop.rs`): `suggest_crop` on a synthetic ARW with the crop model loaded (frames inside the canvas, the ratio of each aspect but Free, scores in `[0, 1]` best first, the subject from the picture, the same answer from two sessions); a missing model is the reason given; `Crop.auto` in JSON (absent is false, false is not written).
- `crates/ym-cli`: `ym info --suggest-crop --json` equals the session's answer (app–CLI parity); `ym render --auto-crop=1:1 --json` reports the proposal and crops to its first frame and angle with `auto`, and the same crop through `--settings` writes the same bytes; the whole photo at 0° is no crop and a crop from `--settings` is replaced; an unknown aspect is a usage error.
- Frontend: the panel's toggle (on applies the frame and the angle; an aspect change asks again; a drag or Straighten turns it off; off restores; Done sends `auto`), the note's lines, the mock, e2e `36-crop-auto`.

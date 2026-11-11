# 27 · Face skin retouching

**Status (2026-11-11): round 1 in progress on branch `skin-retouch` (the library, CPU, mock models).** From the owner's brief of 2026-11-11: "In a new branch, launch agents that will implement face skin retouching node, here is the brief. See if this makes sense and is realistic", and "adapt it for Yellowmoon, if needed". This file is that brief adapted to Yellowmoon's conventions. Its formulas and constants are normative; where a decision was needed, section 9 records it.

Three tools share one set of masks per face:

1. **Blemish removal.** A proposer (a small network, or a classical stand-in) makes a cleaned copy of the face; the library turns the difference into a gain map and applies it through a strength threshold.
2. **Dark circles.** Lifts the under-eye area toward its regional average colour.
3. **Even skin**, with a **Texture** control. Flattens low-frequency luminance across the skin and keeps the fine detail.

---

## 1. Assessment (2026-11-11)

**The maths is coherent and realistic.** Sections 3 to 7 define a complete, testable CPU library: mask arithmetic and feathering, the blemish map (proposer, chroma guard, gain, strength) and its thresholded apply, and one low-frequency operator for dark circles (colour) and even skin (luminance) with an asymmetric blend. Every formula has its constants, and the tests of section 8 can be written against them. With mock models the library is a few days of agent work.

**The models are the hard part.** Without them the node does nothing on a real photo. The licence-clean routes, for the owner to choose before round 2 (section 12):

| Model | What it gives | Licence-clean route |
|---|---|---|
| Face finder | face rectangles and sizes | Apple Vision face rectangles (macOS, through `ym-apple`, as Subject and People do), or MediaPipe face detection (Apache-2.0 weights) run in Rust |
| `FaceMesh` | a dense mesh and two per-vertex tables (under-eye, inverted skin uniformity) | MediaPipe Face Landmarker, 468 vertices (Apache-2.0 weights), run in Rust like SAM 2.1 and Depth Anything (`docs/07`); the two tables are ours to author |
| `FaceParser` | skin without ears, eyes, neck skin | From the mesh (the face oval minus eyes, brows and lips) and, for the neck, the People mask below the jaw. The common face-parsing models are trained on CelebAMask-HQ (non-commercial), which `docs/14` rules out for shipped weights |
| `BeautyMarkDetector` | boxes to leave alone | None licence-clean known; the tools work without it, and a Brush row painted as protection covers it |
| `BlemishProposer` | a cleaned face | **No licence-clean training data is known** (FFHQ is CC BY-NC-SA). A classical proposer (a morphological closing and an edge-aware smoothing at blemish scale) implements the same trait and keeps every later stage; a trained network replaces it later, and training on Modal needs the owner's yes |

**Fit with the pipeline.** A node's picture is LogC4 code values (scene-referred AWG4). The retouch step decodes it to linear AWG4, runs the library, and encodes back; values above 1 are expected and allowed. Yellowmoon renders every node structure on the device (`docs/STATUS.md` question 39). This round is CPU only, with pure per-pixel kernels ready for GPU kernels in a later round. Until then a graph with this node renders on the CPU, and the "Rendering on the CPU" pill says so.

---

## 2. Scope and rules (round 1)

- A new crate, **`crates/ym-retouch`** (Apache-2.0, in the workspace). Nothing in the app, `ym` or the node graph changes in round 1, and no existing graph changes by a bit.
- CPU first. Rows in parallel with `rayon`. Each per-pixel kernel is a pure function, so a GPU kernel can follow without changing the maths. No `unsafe`.
- Colour maths in `f64` on this CPU reference path (`AGENTS.md`); image buffers are `f32` to keep a 24 MP image at 288 MB. Masks are `u8` with the integer arithmetic of section 5.
- **Determinism** (`AGENTS.md` rule 8): every reduction (the mean of section 6.1 step 2) sums in a fixed order, independent of the thread count; parallel work writes disjoint rows.
- The models are traits (section 4). The crate bundles, downloads and hard-codes no weights; it ships deterministic mocks for tests.
- Errors: one `thiserror` enum, `RetouchError`. No panic on bad input; sizes are checked.
- Names: colour spaces in type names, units in field names (`_px`), American spelling in identifiers.

---

## 3. Types

```rust
/// Linear AWG4, f32, row-major, no alpha. Nominally 0..1; values above 1 are allowed.
pub struct LinearAwg4Image { pub width_px: u32, pub height_px: u32, pub data: Vec<[f32; 3]> }

/// 8-bit coverage. 0 = outside, 255 = fully inside.
pub struct Mask { pub width_px: u32, pub height_px: u32, pub data: Vec<u8> }

/// Axis-aligned rectangle in full-resolution image pixels.
pub struct Rect { pub x_px: i32, pub y_px: i32, pub width_px: u32, pub height_px: u32 }

/// A mask over `region` only; outside it the mask is 0.
pub struct BoundedMask { pub region: Rect, pub mask: Mask }

pub struct Face {
    pub rect: Rect,
    /// Face size as a fraction of the longer image side; clamped to 0.05..=1.0 where used.
    pub scale: f32,
}

pub struct SkinSettings {
    pub blemish: f32,      // 0..=100
    pub dark_circles: f32, // 0..=100
    pub even_skin: f32,    // 0..=100
    pub texture: f32,      // -100..=100
    pub include_neck_blemish: bool,
    pub include_neck_even_skin: bool,
}
```

Every setting is clamped to its range on entry. Derived values:

```
a  = blemish / 100
dc = dark_circles / 100
se = even_skin / 100
t  = (texture + 100) / 200          // 0..=1
```

**`render_scale`** (0.0001 ≤ s ≤ 1, else an error) is the working image size over the full-resolution size, so a preview and an export with the same settings look the same.

```
D = max(full_width_px, full_height_px) * render_scale * clamp(face.scale, 0.05, 1.0)
```

`D` is the face size in working pixels; every radius is proportional to it.

---

## 4. Models (traits)

```rust
pub trait FaceParser {
    /// Soft mask of one face part, full image coordinates; None when the part is not found.
    fn part_mask(&self, image: &LinearAwg4Image, face: &Face, part: FacePart) -> Result<Option<BoundedMask>>;
}
pub enum FacePart { SkinWithoutEars, Eyes, NeckSkin }

pub trait FaceMesh {
    /// Rasterises the face mesh into an 8-bit mask with a per-vertex brightness table.
    fn render(&self, face: &Face, table: MeshTable, size_px: (u32, u32)) -> Result<BoundedMask>;
}
pub enum MeshTable { UnderEye, InvertedSkinUniformity }

pub trait BeautyMarkDetector {
    /// Boxes of moles and beauty marks that blemish removal leaves alone.
    fn detect(&self, image: &LinearAwg4Image, face: &Face) -> Result<Vec<Rect>>;
}

pub trait BlemishProposer {
    /// Side of the square output tile, in pixels.
    fn output_size_px(&self) -> u32;
    /// Border consumed on each side: input side = output side + 2 * border.
    fn border_px(&self) -> u32;
    /// Planar RRR..GGG..BBB in, planar out, both in the encoding of 6.1 step 2.
    fn run(&self, input: &[f32]) -> Result<Vec<f32>>;
}
```

A failing beauty-mark detector or neck parser is logged (`tracing::warn!`) and the tools go on without it.

**A trained proposer**, for whoever trains one (round 2 or later): fully convolutional, residual, no padding. Twelve plain 3×3 convolutions (3→64, ten 64→64, 64→3); between each consecutive pair, one depthwise 3×3 convolution with dilation 2 (ten in all); per block: plain convolution, LeakyReLU with its own learned slope (after each of the first 11), depthwise convolution. The last convolution predicts a residual; the output is the input centre-cropped by 32 px per side minus the residual, so `border_px() == 32`. Tile input side 256 to 1024 px.

---

## 5. Masks

Integer arithmetic on `u8`:

```
mul(a, b)      = (a * b) / 255                 // integer division
mul_inv(a, b)  = (a * (255 - b)) / 255
max(a, b)      = per-pixel maximum
```

### 5.1 Feather

`feather(mask, r_px)`, where `r_px` may be negative (positive grows and softens outward, negative shrinks and softens inward):

1. `f = max(512 / width_px, 512 / height_px)`; resize the mask by `f` (bilinear).
2. `r = trunc(r_px * f)`. If `r < 0`, invert the mask and use `|r|`.
3. Dilate by `r` (a disc).
4. Gaussian blur, square kernel of side `k = 6r + ((6r) & 1) + 1` (always odd), `sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8`, reflecting at the borders.
5. If inverted in step 2, invert back.
6. Resize back to the original size (bilinear).

Feather radii below are fractions of `F = trunc(face.scale * max(full_width_px, full_height_px))`; masks are built at full resolution (section 9).

### 5.2 Skin

```
skin = part_mask(SkinWithoutEars)
with the neck: skin = max(skin, part_mask(NeckSkin))   // over the union of both regions
```

### 5.3 Blemish mask

`skin` (with the neck per `include_neck_blemish`), with a soft elliptical hole for each beauty-mark box:

- Inner ellipse inscribed in the box: 0 inside.
- Outer ellipse: same centre, axes times √2, clipped to the mask.
- Between them, with `d` the normalised elliptical distance (1 at the inner edge, √2 at the outer): `pixel = pixel * (d - 1) / (√2 - 1)`.

No other blur or erosion.

### 5.4 Dark-circle mask

```
m = mul(mesh.render(UnderEye), skin_without_ears)
if the eyes mask exists:
    m = mul_inv(m, feather(eyes, +0.03 * F))
```

### 5.5 Even-skin mask

```
u    = mesh.render(InvertedSkinUniformity)
core = mul_inv(skin, u)                       // skin with the neck per include_neck_even_skin
core = mul_inv(core, exclude)                 // only when an exclusion mask is given (default none)
core = feather(core, -0.011 * F)
edge = feather(skin, -0.065 * F)
m    = mul(core, edge)
```

After building any mask, its region is cropped to the bounding box of its non-zero pixels.

---

## 6. Blemish removal

Two phases: an expensive **map** per face, cached, and a cheap **apply**.

### 6.1 Map

Inputs: the full-resolution linear AWG4 crop over the face rect, the blemish mask over the same rect, `rgb_gain` (default `[1.0; 3]`: the input is already white-balanced), and `face_px`, the face size in pixels of that crop.

1. **Size.** `k = 450 / max(1, face_px)`. If `k < 1`, area-downscale the crop and the mask by `k`. Never upscale.
2. **Encode.** Per channel: times `rgb_gain`, then `sqrt`. Take the mean of `(R + 2G + B) / 4` over the crop (a fixed-order sum) and multiply the image by `0.5 / mean` (not when the mean is 0). This is `IN`.
3. **Propose.** Run the proposer over `IN` in a grid of tiles of `output_size_px()`; each tile's input is the tile plus `border_px()` on every side, edges replicated outside the image. The last row and column of tiles move inward to end at the image edge. An image smaller than one tile is padded by edge replication. The outputs stitch into `OUT`, the size of `IN`.
4. **Chroma guard**, when `max(w, h) >= 80`. `radius = max(w, h) / 80` (integer).
   - Per pixel and channel: `r_c = ((max(OUT_c, 0) + 1e-5) / (max(IN_c, 0) + 1e-5))^2`.
   - `conf = clamp(1.1 - 10 * (1 - min(r) / max(r)), 1e-5, 1.0)`; `conf = 1` when `min == max`.
   - `W = OUT * conf` per channel; `B = box_blur(W, radius)`.
   - For every pixel at least `radius` from each edge, where `conf < 1`:
     ```
     Lb = 0.299*B.r + 0.587*B.g + 0.114*B.b      // skip the pixel if Lb <= 0
     k  = (0.299*OUT.r + 0.587*OUT.g + 0.114*OUT.b) / Lb
     OUT_c = lerp(B_c * k, OUT_c, conf^2)
     ```
   Colour-shifted pixels get the chroma of a local average and keep their luma.
5. **Gain.** Per channel, `m` the mask byte:
   ```
   g = clamp((max(OUT_c, 1/256) / max(IN_c, 1/256))^2, 0.5, 2.5)
   g = 1 + (m / 255) * (g - 1)                    // g = 1 where m == 0
   d = min(x, w-1-x, y, h-1-y)
   if d < 2: g = 1 + (d / 2) * (g - 1)            // fades to 1 at the border
   ```
   Squaring undoes the `sqrt`: `g` is a linear-light gain.
6. **Strength.** `s = max(|log2 g.r|, |log2 g.g|, |log2 g.b|, 1e-6)`, then a max filter of radius 3 and a box blur of radius 3: `alpha`.
7. **Result.** `BlemishMap { rect, width_px, height_px, data: Vec<[f32; 4]> }` of `(g.r, g.g, g.b, alpha)` at the size of step 1 (not resized back).

The map is cached. Its key: the image's identity, the face count, each face rect, and everything upstream that changes the pixels the proposer sees. It does not depend on the Blemish amount.

### 6.2 Apply

For each face with `a > 0`, for each working pixel `(px, py)`:

```
u = (map.width  / rect.width ) * (px / render_scale - rect.x + 0.5) - 0.5
v = (map.height / rect.height) * (py / render_scale - rect.y + 0.5) - 0.5
skip unless 0 <= u < map.width and 0 <= v < map.height
(g, alpha) = bilinear at (u, v), the +1 neighbour clamped to the last index

s = clamp(alpha / max(0.01, 1 - a), 0, 1)
if a < 0.25: s = s * 4 * a
if a > 0.75: s = lerp(s, 1, 4 * (a - 0.75))
with a protection mask: s = s * (1 - protect / 255)

rgb = rgb * lerp(1, g, s)        // per channel
```

The amount is a threshold on strength: low values fix only strong blemishes, high values apply every correction. The protection mask (working coordinates) is what the person paints as "leave alone".

## 7. Dark circles and even skin

One operator, per face, after blemish removal, on the working image. A tool is active when its amount is above 0 and its mask exists and is not all zeros; with neither active, the face is skipped.

### 7.1 Parameters

```
tex_frac   = min(0.004 + 0.02 * t, 0.024)
r_tex_full = D * tex_frac
half       = r_tex_full >= 8
ds         = if half { 2 } else { 1 }

r_tex = max(1, trunc(r_tex_full / ds))
r_dc  = max(1, trunc(0.077 * D / ds))     // when dark circles is active
r_se  = max(1, trunc(0.044 * D / ds))     // when even skin is active

k_dc  = 1 - (1 - dc)^3
sp_dc = 1 - 0.75 * dc^4
k_se  = se
sp_se = 1
```

`blur(X, r)` is two box filters in a row with radii `r / 2` and `r - r / 2` (integer division); a box of radius `q` has side `2q + 1`. The working region is padded by the sum of the radii so tile edges do not show.

### 7.2 Low-frequency layers

1. When `half`, area-downscale the source and both masks by 2; this is `S`.
2. `T = blur(S, r_tex)`. `src - T` is the texture that is kept; Texture moves this radius from 0.4 % to 2.4 % of the face.
3. Mask-normalised blur of a layer `X` with `m = mask / 255`:
   ```
   C = m * blur(X * m, r) / max(blur(m, r), 1/255) + (1 - m) * X
   ```
   Inside the mask, the mask-weighted regional average; outside, `X`.

### 7.3 Dark circles (colour)

```
C_dc = masked_blur(T, dark_circle_mask, r_dc)      // RGB
new  = src - T + C_dc
```

### 7.4 Even skin (luminance)

```
Y    = (T.r + T.g + T.b) / 3
C_se = masked_blur(Y, even_skin_mask, r_se)        // one channel
new  = src * (C_se / max(Y, 0.001))                // the same gain on all channels
```

### 7.5 Blend

For both tools, with that tool's `k` and `sp`; `Sum(p) = max(p.r + p.g + p.b, 0.001)`:

```
if Sum(new) < Sum(src):
    w = (Sum(new) / Sum(src))^16                    // darkening almost fully suppressed
else:
    w = (1 - sp) + sp * (Sum(src) / Sum(new))^2     // brightening rolled off
out = lerp(src, new, k * w)
```

When `half`, `src` is the full-resolution pixel at `(x, y)` and `T`, `C_dc`, `C_se`, `Y` are read at `(x / 2, y / 2)` (integer division, nearest).

### 7.6 Order

Per face: `T` once, from the image as it is on entry; dark circles in place; then even skin in place, with the dark-circle result as `src` and the same `T`.

## 8. Pipeline, entry point and tests

```
for each face: masks (5), cached per image; blemish map (6.1), cached per image and face rect
output = copy of input
for each face: blemish apply (6.2)
for each face: dark circles, then even skin (7)
```

```rust
pub fn retouch_skin(
    image: &LinearAwg4Image,          // the working image
    full_size_px: (u32, u32),         // the full-resolution size (masks and maps are built there)
    faces: &[(Face, SkinSettings)],
    render_scale: f32,
    models: &Models,                  // the trait objects of section 4
    cache: &mut RetouchCache,
    protect: Option<&Mask>,
) -> Result<LinearAwg4Image, RetouchError>;
```

Tests (all required):

- **Identity:** all amounts 0 return the input bit for bit.
- **Mask locality:** pixels where every mask is 0 do not change.
- **Blemish threshold:** with a synthetic map, `s` at `a` = 0.1, 0.5 and 0.9 equals hand-computed values, the `< 0.25` and `> 0.75` branches included.
- **Gain clamp:** gains stay in 0.5..=2.5 and are 1 on the outer ring.
- **Chroma guard:** a mock proposer that tints one channel by 15 % has its chroma pulled back; one that scales all channels equally is left alone (`conf == 1`).
- **Masked blur:** a constant image is a fixed point for any mask; outside the mask the layer is unchanged.
- **Blend asymmetry:** a `new` 20 % darker than `src` moves the pixel by under 3 % of the difference; a brighter one moves it by the formula.
- **Texture:** zero-mean noise with a period well below `r_tex` on a flat skin patch keeps its amplitude within 5 % after Even skin at 100.
- **Scale:** `render_scale` 1.0 and 0.5, the first downscaled, differ by a mean absolute difference under 1 %.
- **Feather:** a positive radius never lowers a pixel; a negative one never raises one.
- **Determinism:** the same inputs give the same bytes with 1, 3 and 4 threads.
- **Properties** (`proptest`) of the pure kernels: finite outputs for finite inputs, monotonic in the amount.
- A `criterion` benchmark: 24 MP, one face.

## 9. Decisions (where the brief left room)

1. **Masks and `F` at full resolution.** The `FaceParser` and `FaceMesh` traits give full image coordinates, so `F` uses the full-resolution size; section 7 resamples a mask by `render_scale` (bilinear) to the working image.
2. **The blemish map from the full-resolution crop.** Never from a preview, so a preview and an export use the same map (the scale test needs it). This is why `retouch_skin` takes `full_size_px` and the cache holds maps per face.
3. **Open points:** bilinear for mask resizing; a disc for dilation; radius, not side, for the max filter and box blur of 6.1 step 6; `rgb_gain` 1; the mesh tables belong to the `FaceMesh` implementation (the mock draws an analytic table).
4. **Colour space:** linear AWG4 from the node's LogC4 picture (`logc4_decode_rgb`), encoded back after (round 2).
6. **S-02 choices.** Feather: a radius that truncates to 0 returns the mask unchanged (no resize round trip); the result is clamped to never go below the input for a positive radius nor above it for a negative one (the Gaussian of a thin mask would otherwise lower it); borders reflect without repeating the edge sample (reflect-101); bilinear resizes are pixel-centre aligned and round to nearest; the disc dilation ignores pixels outside the mask. Ellipse holes use pixel centres and truncate the ramp. `F` and `D` clamp `face.scale` to 0.05..=1; NaN settings become 0. The neck comes from an optional second `FaceParser`, absent means no neck. Mesh and parser regions may extend outside the image and are clipped.
5. **The proposer's name:** `BlemishProposer`, since a classical implementation fills it until a trained one exists.

## 10. Tasks

Each task by a Sonnet agent on `skin-retouch` (or a branch off it); the lead reviews and commits.

- **S-01 · Spec.** This file. *Done when:* committed.
- **S-02 · Crate, types, traits, mocks, masks** (sections 2 to 5). *Done when:* `ym-retouch` builds in the workspace; types, errors, traits and mocks are documented; mask arithmetic, feather and the three mask builders pass their tests (feather monotonicity included); `cargo deny check` passes.
  *As built (S-02):* `crates/ym-retouch` has modules `types`, `error`, `models`, `mock` (feature `mock`), `feather` and `mask`; S-03 to S-05 add `blemish`, `lowfreq` and `pipeline`. `Models` holds the parser, an optional separate neck parser, the mesh, an optional beauty-mark detector and the proposer. Builders work on full-size masks and `face_masks` returns three `Option<BoundedMask>` cropped to their bounding boxes. 20 unit and property tests pass (u8 arithmetic with `proptest`, feather, hole ramp, builders with the mocks, crop, clamping, `render_scale`, determinism with 1 and 4 threads). `cargo deny` is not installed in this environment, so it was not run; the crate adds no dependency that was not already in the workspace (rayon, thiserror, tracing, proptest).
- **S-03 · Blemish removal** (section 6), after S-02. *Done when:* map and apply match the formulas; the threshold, gain-clamp and chroma-guard tests pass; the map cache works.
- **S-04 · Dark circles and even skin** (section 7), after S-02, alongside S-03. *Done when:* masked blur, blend and order match; the masked-blur, blend and texture tests pass.
- **S-05 · Pipeline, the remaining tests, benchmark** (section 8), after S-03 and S-04. *Done when:* identity, locality, scale and determinism pass; proptest for the pure kernels; the 24 MP benchmark runs and its time is written here.

**Round 2** (needs the owner's answers in section 11): the models (section 1), the classical proposer, the node in the graph (settings, plan, cache, `ym`, interface with mockups first), GPU kernels and parity.

## 11. For the owner (`docs/STATUS.md` question 45)

- **Free to implement:** confirm the brief is yours to use (not taken from another product's code or binaries).
- **The models for round 2** (section 1): Vision or MediaPipe for faces; MediaPipe's mesh; masks from the mesh; a classical proposer first, a trained one only with a training budget.
- **Where the tools live:** a node of their own ("Skin", settings per face) or a tab of the corrector.

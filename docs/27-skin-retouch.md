# 27 · Face skin retouching

**Status (2026-11-11): round 1 done on branch `skin-retouch` (the library `ym-retouch`, CPU, mock models; S-01 to S-05); round 2 started (models, the Skin node, tuning; section 10), with the owner's answers in section 11.** From the owner's brief of 2026-11-11: "In a new branch, launch agents that will implement face skin retouching node, here is the brief. See if this makes sense and is realistic", and "adapt it for Yellowmoon, if needed". This file is that brief adapted to Yellowmoon's conventions. Its formulas and constants are normative; where a decision was needed, section 9 records it.

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
| `BlemishProposer` | a cleaned face | **No licence-clean training data is known** (FFHQ is CC BY-NC-SA). A classical proposer (gated detect-and-fill: only detected spots are filled, everything else is returned unchanged; S-07) implements the same trait and keeps every later stage; a trained network may replace it later (S-14, the owner's yes of 2026-11-11) |

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
    /// Planar RRR..GGG..BBB in, planar out, both in the encoding of 6.1 step 2. Away from blemishes the
    /// output equals the input (no veil over clean skin at high amounts; round 2, S-06).
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
7. **S-04 choices.** Edges of every box filter replicate the edge pixel (the masked ratio uses the same rule, so it stays a weighted mean). A tool never changes a pixel whose own mask byte (at the working resolution) is 0, even on the half path, where the half-resolution mask can be non-zero one pixel outside the mask; this keeps "outside the masks, bit-identical" exact and stops the 0.001 floor of `Y` from darkening unmasked pixels. Dark circles hand their result to even skin in `f64`, without rounding to `f32` between the two. A mask that resamples to nothing (off the image) makes its tool inactive.
8. **S-03 choices.** The strength max filter is a square window of side 7 (radius 3) and ignores pixels outside the map; the box blurs (strength and chroma guard) shrink the window at the borders instead of replicating. The area downscale gives `round(w * k)` by `round(h * k)` (at least 1) and averages the covered source area with fractional edge weights; the mask is downscaled the same way and rounded to `u8`. A non-finite `face_px` counts as 1. NaN or negative values encode as 0. The chroma guard changes only pixels at least `radius` from every edge with `conf < 1`; others keep the proposer output. The map cache holds one key at a time (image key, face count, face rects, upstream key; the two keys are opaque `u64` from the caller) and one `Arc<BlemishMap>` per face index.

9. **S-05 choices: the entry point and the full-resolution image.** `retouch_skin` keeps the signature of section 8; the full-resolution image reaches it through the cache. `prepare_skin(full_image, faces, models, cache)` builds the masks and the blemish maps of every face from the full-resolution image and stores them in the `RetouchCache`. A preview then calls `retouch_skin` on its working image, which only reads the cache; if something is missing it fails with `RetouchError::NotPrepared` instead of building from a preview. When the working image is itself full size (an export), `retouch_skin` builds what is missing from it and caches it, so the CLI needs one call. The cache is named by two opaque keys (`set_keys(image_key, upstream_key)`; a change drops everything). Masks are rebuilt when the faces, the full size or either neck switch change; a map is rebuilt when the faces or the blemish neck switch change (that switch is folded into the map key), so the even-skin neck switch and every amount reuse the maps. A map covers the face rectangle joined with the blemish mask's region (the neck lies below the face), clipped to the image, and `face_px` is `D` at full resolution. Maps are built lazily and only for faces with Blemish above 0 on the full-image path, and for every face in `prepare_skin`. With every amount 0 the call returns a copy of the input before touching the cache. The working size must equal `full_size_px * render_scale` within 2 px (`RetouchError::WorkingSize`); a protection mask must have the working size.
10. **S-05 choices: border conventions.** The blemish box blurs shrink the window at the borders (decision 8) and the low-frequency filters replicate the edge pixel (decision 7). Both are kept: they are separate stages with their own tests, the blemish blurs run on a per-face map whose outer pixels are faded to gain 1 anyway, and no formula changes. Nothing shares a helper.

## 10. Tasks

Each task by a Sonnet agent on `skin-retouch` (or a branch off it); the lead reviews and commits.

- **S-01 · Spec.** This file. *Done when:* committed.
- **S-02 · Crate, types, traits, mocks, masks** (sections 2 to 5). *Done when:* `ym-retouch` builds in the workspace; types, errors, traits and mocks are documented; mask arithmetic, feather and the three mask builders pass their tests (feather monotonicity included); `cargo deny check` passes.
  *As built (S-02):* `crates/ym-retouch` has modules `types`, `error`, `models`, `mock` (feature `mock`), `feather` and `mask`; S-03 to S-05 add `blemish`, `lowfreq` and `pipeline`. `Models` holds the parser, an optional separate neck parser, the mesh, an optional beauty-mark detector and the proposer. Builders work on full-size masks and `face_masks` returns three `Option<BoundedMask>` cropped to their bounding boxes. 20 unit and property tests pass (u8 arithmetic with `proptest`, feather, hole ramp, builders with the mocks, crop, clamping, `render_scale`, determinism with 1 and 4 threads). `cargo deny` is not installed in this environment, so it was not run; the crate adds no dependency that was not already in the workspace (rayon, thiserror, tracing, proptest).
- **S-03 · Blemish removal** (section 6), after S-02. *Done when:* map and apply match the formulas; the threshold, gain-clamp and chroma-guard tests pass; the map cache works.
  *As built (S-03):* module `blemish` in `crates/ym-retouch`. `blemish_map(crop, mask, rect, rgb_gain, face_px, proposer) -> BlemishMap` follows 6.1 step by step; `apply_blemish(&mut image, &map, a, render_scale, protect)` follows 6.2 (a protection mask must have the image's size). The pure kernels are free functions for a later GPU port: `encode_channel`, `chroma_confidence`, `chroma_pull`, `blemish_gain`, `gain_strength`, `threshold_strength`, `apply_gain`. `BlemishMapCache::get_or_build(key, face_index, build)` with `BlemishMapKey` is what S-05's `RetouchCache` holds. 21 new unit and property tests (41 in the crate) cover the threshold by hand values, the gain clamp, ring and fade, the chroma guard (tint pulled back with luma kept, equal scaling untouched), identity, tiling with four tile sizes and awkward image sizes, never upscaling, 1 against 4 threads, the u/v mapping and bilinear clamp, and the cache. See decision 7.
- **S-04 · Dark circles and even skin** (section 7), after S-02, alongside S-03. *Done when:* masked blur, blend and order match; the masked-blur, blend and texture tests pass.
  *As built (S-04):* module `lowfreq` in `ym-retouch`. `retouch_low_frequency(image, full_size_px, face, settings, render_scale, dark_circle_mask, even_skin_mask)` works in place on the working image and returns whether any tool ran; `LowFreqParams` holds the parameters of 7.1; `mask_to_working` resamples a full-resolution `BoundedMask` by `render_scale` (bilinear, pixel-centre aligned) to a `BoundedMask` in working pixels; the pure kernels `blend_weight`, `blend_pixel`, `masked_blur_combine`, `dark_circles_target`, `even_skin_target` and `dark_circle_k_sp` are free functions for a later GPU kernel. Only the padded bounding box of the masks is processed (pad = `(r_tex + r_tool) * ds + ds`, origin rounded down to even on the half path so the 2 x 2 blocks sit on the whole image's grid), so a small face on a large image costs little; the result equals processing the whole image. Layers are `f64` planes; blurs are box filters with a running sum in fixed order (rows in parallel, writes disjoint). 14 tests pass (34 in the crate): parameters, blur as two boxes, masked blur fixed point and locality, blend asymmetry, mask resampling, amounts 0 and empty masks, locality outside both masks, texture within 5 % (half, full and automatic path), half against full path (mean absolute difference under 2e-3), cropped against whole region, 1 and 4 threads, and `proptest` for finite kernels, monotonic moved fraction `k * w` (dark circles and even skin) and the constant-image fixed point of the whole operator.
- **S-05 · Pipeline, the remaining tests, benchmark** (section 8), after S-03 and S-04. *Done when:* identity, locality, scale and determinism pass; proptest for the pure kernels; the 24 MP benchmark runs and its time is written here.
  *As built (S-05):* module `pipeline`: `retouch_skin`, `prepare_skin` and `RetouchCache` (decisions 9 and 10). 10 new tests (65 in the crate, about 4 s): identity (also with NaN and negative amounts, and no faces), locality through the whole pipeline with the mocks (exact outside every mask with the identity proposer; with a proposer that scales by 0.9, nothing moves more than 3 px from a mask, because the map is sampled bilinearly on a grid coarser than the image), scale consistency (1200 x 900 with a 500 px face, amounts 60, preview prepared from the full image: mean absolute difference under 1 % and under half of the change itself), determinism with 1, 3 and 4 threads, cache reuse and switches, protection mask, input errors, a failing detector and an absent neck, mask cropping, and a `proptest` that the output is finite for any settings (and the identity when no amount is above 0). The pure kernels already carry their own `proptest`s from S-03 and S-04. Benchmark `benches/retouch.rs` (`cargo bench -p ym-retouch --features mock --bench retouch`; criterion 0.5, MIT OR Apache-2.0, a dev-dependency only): 6000 x 4000, one face 1500 x 1800 (`scale` 0.3), mock models (identity-shaped proposer scaling by 0.9, 192 px tiles, border 32), blemish, dark circles and even skin at 60. Linux container, CPU only, 4 cores (`nproc` 4), Intel Xeon at 2.1 GHz, release build, 10 samples (the container is noisy; the ranges are criterion's): face masks alone 650 to 850 ms; `prepare_skin` (masks and blemish map) 370 to 450 ms (the mask run above was measured at another moment, so the two overlap); apply on the full image with a warm cache 414 to 427 ms, which includes the 288 MB copy of the image; apply at `render_scale` 0.25 on a 1500 x 1000 preview 54 to 66 ms. The mock mesh and parser rasterise per pixel, so a real model changes the preparation time; the apply time does not depend on the models.

**Round 2** (the owner's answers of 2026-11-11, section 11). The research is `research/reports/Skin retouching models and data.md` (notes in `research/research_notes/Skin retouching models and data/`). Tasks, each by a Sonnet agent; the lead reviews and commits:

- **S-06 · Spec for round 2.** This section and section 11; the spec edits the research asked for: the blemish scale is 0.7 % to 3.5 % of `D` (the earlier sketch sat at pore scale); a proposer returns `OUT = IN` away from blemishes, so no veil reaches the clean skin at high amounts; a proposer is built per face with the face size in map pixels. *Done when:* committed.
- **S-07 · Classical blemish proposer and beauty-mark detector** in `ym-retouch`. The proposer: a gated detect-and-fill (dark spots by a per-channel black top-hat, red spots by a red-excess test, at blemish scale; spots under about 3 px dropped; inside each spot the blurred base is filled smoothly from its rim; everywhere else the output equals the input). The detector: dark, compact, brown, isolated blobs of 1.2 % to 4.5 % of the face, ranked by contrast, at most about 10, boxes padded about 30 %. Deterministic, `f64`, rows in parallel. *Done when:* synthetic tests with known lesions and moles pass (spots removed, moles and the clean skin untouched), and the constants are documented as starting values for S-12.
- **S-08 · MediaPipe weights and the face finder.** BlazeFace full range, Face Mesh V2 (from `face_landmarker.task`) and Selfie Multiclass, Apache-2.0 per their model cards; a CI-only script converts the TFLite files to safetensors (Python for exporting weights only, rule 6); fetch scripts with SHA-256, a release asset and the bundle as for SAM 2.1 (the owner: "bundle"), model cards and NOTICE lines; BlazeFace ported to Rust in `ym-ml` with tiling for group photos, our own NMS and face cap; parity against the reference outputs. *Done when:* faces found on public test portraits match the reference within the stated tolerance, and the weights are bundled by the release workflow.
  *As built (S-08):* **Weights.** `tools/mediapipe/convert.py` (Python, CI only; `tflite` 2.18.0 and `numpy` pinned in `tools/mediapipe/requirements.txt`) downloads the three originals from `storage.googleapis.com/mediapipe-models/.../1/` (the versioned URLs), refuses a file whose SHA-256 is not the pinned one (and checks the member `face_landmarks_detector.tflite` of `face_landmarker.task` too), and writes `blaze_face_full_range.f16.safetensors` (1,067,186 bytes, 514,069 values), `face_mesh_v2.f16.safetensors` (2,448,944 bytes, 1,199,316 values) and `selfie_multiclass_256.f32.safetensors` (16,321,304 bytes, 4,070,284 values, f32 because Google ships it in f32). Constants are `t<TFLite tensor index>` (the f16 weights behind DEQUANTIZE are stored as they are); the header holds the operator graph (types, tensor indices, every builtin option, the integer constants, inputs and outputs). The writer is our own with sorted keys (the `safetensors` package orders its header at random), so the output is byte-identical on every run and the three hashes are **committed now** in `data/models/mediapipe/SHA256SUMS` (`0261507f…`, `f56049da…`, `a889f4ef…`); the `mediapipe` job of `models.yml` compares its output with them and publishes only equal bytes. Model card, `NOTICE` item 14, `docs/14` rows, `docs/07`, `docs/12`, `tools/fetch/mediapipe.sh` (with `test_mediapipe.py`, `--only`, `--verify`, `--photos`) and `check_bundle.py --require-mediapipe` are in place; `release.yml` and `build.yml` cache, download (GitHub CLI), fetch, check and verify the files in the bundle **once the release exists**, and warn and go on without them before that. **To publish (the lead):** push a branch named `models/mediapipe-faces-1` (for example `git push origin skin-retouch:models/mediapipe-faces-1`) or run Actions > models > Run workflow with the job `mediapipe`; the job converts, checks the hashes against the committed ones, and creates the prerelease `models-mediapipe-faces-1` with the three files and a `SHA256SUMS`. Nothing else needs copying. From then on the CI job `mediapipe` fetches the weights and the photos and sets `YM_REQUIRE_MEDIAPIPE`. **The runtime.** `ym_ml::mediapipe::graph` is a runner for such a file (`CONV_2D`, `DEPTHWISE_CONV_2D`, `ADD`, `RELU`, `PRELU`, `LOGISTIC`, `MAX_POOL_2D`, `PAD`, `RESIZE_BILINEAR`, `RESHAPE`, `CONCATENATION`; a graph with another operator is refused when it loads), instead of one hand-written forward pass per network, so S-09 and S-10 add only what they lack (Selfie Multiclass: `MUL`, `SUM`, `SOFTMAX`, `TRANSPOSE`, `TRANSPOSE_CONV`, nearest resize) and no weights file changes. As a smoke check (a throwaway example, not committed) Face Mesh V2 already runs through it and agrees with the TFLite interpreter on a synthetic input to 1.5e-4 absolute on values up to 221. Bits are fixed (f32, one chain per output, no fused multiply-add, `sam2`'s packed product for the 1 x 1 and im2col convolutions); 1 and 4 threads give equal bits. **The finder.** `ym_ml::mediapipe::blazeface`: the letterbox of MediaPipe's `ImageToTensor` into 192 x 192, `x * 2 - 1` (found by trying both ranges against MediaPipe's task: `[-1, 1]` matches, `[0, 1]` does not), black bars; MediaPipe samples bilinearly at `(dst - offset) / scale` with no half-pixel shift and no filter (OpenCV `warpAffine`), which `Resample::MediaPipe` reproduces and `Resample::Box` (the default) repeats on a picture first reduced by an integer box filter of `floor(0.5 / scale)` pixels, so that a 24 MP photo is not sampled one pixel in 30; 2304 anchors (48 x 48, stride 4, fixed size, offset 0.5); box and six keypoints from 16 regressors; score = sigmoid of the logit clipped to +-100; a weighted NMS of our own (best score first, group = IoU above 0.3, score-weighted average of box and keypoints, the best score kept), score threshold 0.5. **Tiling:** the network finds faces from about 7 % to 50 % of its frame (measured: 10 px of 192 start to be found, 12 px reliably, the score falls past 150 px), so `plan_windows` gives windows of `min_face / 0.07` pixels (default minimum 1.5 % of the longer side, at least 24 px) stepping by two thirds of their size, each next size four times larger while under three quarters of the picture, and the whole picture last (6000 x 4000: 7 x 5 windows of 1286 px and the whole, 36 passes). In each window a detection under 6 % of the window is dropped (its box is coarse: a 219 px face seen in a 6000 px window came out 281 px and, merged by score, spoiled the good boxes from the 1286 px windows) and so is one that touches an inner border of the window (it is whole in a neighbour); the rest is merged by the same NMS, smaller than the minimum is dropped, and the best 32 are kept. `FaceFinder::find` returns `DetectedFace { rect, scale, keypoints_px, score }`: `rect` has the fields of `ym_retouch::Rect` (whole pixels, rounded outwards and clipped to the picture), `scale` is the longer side of the box over the longer side of the picture; the type stays in `ym-ml` so that neither crate depends on the other, and the adapter to `ym_retouch::Face` is two lines in the place that wires the Skin node (S-11). `PixelSource` takes 8-bit or float RGB in display-referred 0..1 (the caller converts from LogC4 or linear AWG4). **Parity** (`crates/ym-ml/tests/blazeface.rs` against `tests/fixtures/blazeface.json`, 48 KB, written by `tools/mediapipe/reference.py` from Google's interpreter `ai-edge-litert` 2.3.0 and Google's `FaceDetector` task of `mediapipe` 1.2.2; weights and photos from `tools/fetch/mediapipe.sh`, the tests skip with a plain line without them and `YM_REQUIRE_MEDIAPIPE=1` makes that a failure): the raw network on a fixed synthetic input (a formula both languages compute bit for bit): largest logit difference **6.2e-6** (bound 1e-4), regressors **8.9e-7** of the largest (bound 1e-5); three photos with MediaPipe's own resampling against the interpreter on its tensor: IoU at least **0.9931** (bound 0.98), score difference at most **0.0025** (bound 0.01); against MediaPipe's task, whose box is whole pixels: worst edge **1.45 px** (bound 2.5 px or 2 %), worst keypoint **0.33 %** of the box (bound 2 %), score difference at most **0.005** (bound 0.02); the finder's defaults on the same photos against the task: same number of faces, IoU at least **0.93** (bound 0.85); a 24 MP picture with seven faces from 49 to 390 px pasted into a noisy background: every face of 90 px or more found once and no other (5 of 5, IoU at least **0.877**, bound 0.8), all 7 with `min_face_px` 30; a cap of 2 returns the best two; 1 and 4 threads give equal results. **Photos:** Wikimedia Commons answered 429 to every request from this environment for the whole session, so the CC0 portraits of `training/depth/oracle_photos.json` could not be fetched; the three photos used are MediaPipe's own test images (`storage.googleapis.com/mediapipe-assets/portrait.jpg`, `man-woman-okay.jpg` with two faces, `woman_hands.jpg`; part of the Apache-2.0 MediaPipe repository's test data, pinned by SHA-256 in `tools/fetch/mediapipe-photos.tsv`, downloaded at test time and never committed). Adding CC0 Commons portraits is a matter of listing them in the TSV and running `reference.py` again. **Timing** (Linux container, 4 cores, Intel Xeon 2.1 GHz, release build, `cargo run --release -p ym-ml --example faces_time`): one 192 x 192 pass **12 ms** on 4 threads (20 ms on 1; 105.7 M multiply-adds); the letterbox of an 820 x 1024 photo 0.9 ms (2.7 ms on 1); the finder on 820 x 1024 **84 ms** (280 ms on 1); on a **24 MP** (6000 x 4000) picture, 36 passes, **240 ms** on 4 threads (0.8 s on 1). Open: the default minimum face (1.5 % of the longer side) and the score threshold 0.5 are starting values for S-11 and S-12; the Mac timing is M0-H4.
- **S-09 · Face mesh in Rust**, after S-08: Face Mesh V2 ported, two passes per face, the canonical topology (Apache-2.0) for region polygons, the mesh rasterised into a `BoundedMask`; the two tables (`UnderEye`, `InvertedSkinUniformity`) authored as our own UV textures. *Done when:* parity with the reference landmarks; the masks look right on a contact sheet of public portraits.
- **S-10 · Selfie Multiclass in Rust and the real `FaceParser`**, after S-08: skin from the face-skin class inside the mesh oval plus a forehead band, eyes and other cut-outs from the mesh polygons, the neck from the body-skin component joined to the jaw, a per-face colour model at the edges and the existing guided filter (`docs/algorithms/06`). *Done when:* parity of the segmenter; masks on the contact sheet.
- **S-11 · The Skin node.** A node kind of its own, **`skin`**, in the node graph only: it is not offered in the simple main view (the owner, 2026-11-11). Settings per face (Blemish, Dark circles, Even skin, Texture, the two neck switches), faces found automatically and listed in its panel, the protection mask from a Brush row; LogC4 decoded to linear AWG4 for `ym-retouch` and encoded back; the plan step, the caches, `ym` settings and the interface. Mockups first (S-H1). *Done when:* app and `ym` give the same bytes; the node's tests pass; the node renders on the CPU with the "Rendering on the CPU" pill until S-13.
- **S-12 · Tuning the constants.** The owner, 2026-11-11: "you can change constants for optimal result, try it." The brief's constants are starting values: tune them (blemish scale, gain clamp, threshold curve, low-frequency radii, blend exponents, detector thresholds) against the synthetic protocol of the research (lesions with known ground truth in ten skin tones, texture energy, colour shift, mole survival) and contact sheets of public portraits; each change recorded here with before and after numbers. *Done when:* the synthetic gates pass and the owner has seen the sheets.
- **S-13 · GPU kernels** for the apply of section 6.2 and the operator of section 7, with CPU parity tests, so the node runs on the device.
- **S-14 · A trained proposer (optional, Modal).** The network of section 4 trained on synthetic chromophore lesions over Open Images faces (CC BY 2.0), per the research plan: $15 planned, $20 hard cap, a ledger that refuses a run past the cap. Shipped only if it beats the classical proposer in the owner's blind A/B, and only if its speed on a Mac is acceptable (a 48-channel variant is allowed).
- **[HUMAN] S-H1** mockups of the Skin node; **S-H2** blind A/B and the test build.

## 11. The owner's answers (2026-11-11, `docs/STATUS.md` question 45)

1. **Provenance and constants.** "Brief came from research" and "you can change constants for optimal result, try it": the brief is the owner's, from research; its constants are starting values that S-12 tunes.
2. **Models.** "yes": MediaPipe BlazeFace, Face Mesh V2 and Selfie Multiclass ship (Apache-2.0 cards; Google's own training data, as for SAM 2.1 and Depth Anything).
3. **Where the tools live.** "own node for this, not available in simple main view": the `skin` node, in the node graph only.
4. **Modal.** "yes": a trained proposer may be trained, $15 planned, $20 hard cap (S-14).
5. **Research-only sets.** "yes": FFHQR, Fitzpatrick17k and HAM10000 for local measurement only, never shipped or committed.
6. **Weights.** "bundle": bundled with the app like SAM 2.1 (a release asset fetched by the release workflow), not committed.

# 27 · Face Refinement (face skin retouching)

**Status (2026-11-11): round 1 done on branch `skin-retouch` (the library `ym-retouch`, CPU, mock models; S-01 to S-05); round 2 started (models, the Face Refinement node, tuning; section 10; the node's format is S-11a, its render S-11b, its interface S-11c), with the owner's answers in section 11.** From the owner's brief of 2026-11-11: "In a new branch, launch agents that will implement face skin retouching node, here is the brief. See if this makes sense and is realistic", and "adapt it for Yellowmoon, if needed". This file is that brief adapted to Yellowmoon's conventions. Its formulas and constants are normative; where a decision was needed, section 9 records it.

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
| `FaceParser` | skin without ears, eyes, the mouth's interior | From the mesh (the face oval minus eyes, brows and lips; the inner lip line for the mouth) and the face-skin class of Selfie Multiclass. No neck: the owner dropped the neck switches on 2026-11-11. The common face-parsing models are trained on CelebAMask-HQ (non-commercial), which `docs/14` rules out for shipped weights |
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
    pub skin_contrast: f32,       // -100..=100, 0 = unchanged
    pub shadow_desaturation: f32, // 0..=100
    pub shine_removal: f32,       // 0..=100
    pub teeth_whitening: f32,     // 0..=100
}
```

Every setting is clamped to its range on entry. Derived values:

```
a  = blemish / 100
dc = dark_circles / 100
se = even_skin / 100
t  = (texture + 100) / 200          // 0..=1
gc = skin_contrast / 100            // -1..=1
gd = shadow_desaturation / 100
gs = shine_removal / 100
tw = teeth_whitening / 100
```

The neck switches of the brief are gone (the owner, 2026-11-11: "Don't do include neck"); the skin is the face's skin only. The four grading and teeth settings are the owner's additions of the same day, after DaVinci Resolve's Face Refinement (section 7.7 and 7.8).

**`render_scale`** (0.0001 ≤ s ≤ 16, else an error; S-11b raised the limit from 1, decision 13) is the working image size over the size the masks were built at, so a preview and an export with the same settings look the same.

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
pub enum FacePart { SkinWithoutEars, Eyes, MouthInterior }

pub trait FaceMesh {
    /// Rasterises the face mesh into an 8-bit mask with a per-vertex brightness table.
    fn render(&self, image: &LinearAwg4Image, face: &Face, table: MeshTable, size_px: (u32, u32)) -> Result<BoundedMask>;
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

A failing beauty-mark detector is logged (`tracing::warn!`) and the tools go on without it; a missing mouth interior only turns teeth whitening off.

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
```

### 5.3 Blemish mask

`skin`, with a soft elliptical hole for each beauty-mark box:

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
core = mul_inv(skin, u)
core = mul_inv(core, exclude)                 // only when an exclusion mask is given (default none)
core = feather(core, -0.011 * F)
edge = feather(skin, -0.065 * F)
m    = mul(core, edge)
```

### 5.6 Grading and teeth masks

The skin grading of 7.7 uses the even-skin edge mask: `grade = feather(skin, -0.065 * F)` (the `edge` of 5.5), so it fades out before the hairline, the brows and the jaw.

The teeth mask starts from the mouth's interior and keeps what looks like teeth:

```
mouth = feather(part_mask(MouthInterior), -0.004 * F)
per pixel, in linear light with Y = (r + g + b) / 3 and sat = 1 - min(r, g, b) / max(r, g, b):
  bright = smoothstep(0.6, 1.0, Y / Ymouth)        // Ymouth = the mouth region's 80th percentile of Y
  pale   = 1 - smoothstep(0.35, 0.6, sat)           // gums, tongue and lips are saturated
  notred = 1 - smoothstep(0.08, 0.2, (r - g) / max(Y, 1e-4))
teeth = mul(mouth, round(255 * bright * pale * notred)), then feather(teeth, +0.002 * F)
```

All thresholds here are starting values for S-12.

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

The amount is a threshold on strength: low values fix only strong blemishes, high values apply every correction. The protection mask (working coordinates) is what the person paints as "leave alone". It covers every tool, not only this one (the owner, 2026-11-11, section 11 item 9): see 7.5, 7.7 and 7.8.

## 7. Dark circles, even skin, skin grading and teeth

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

With a protection mask, `k` is multiplied by `1 - protect / 255` at each pixel (so are `m` in 7.7 and 7.8); a pixel with `protect = 255` is not touched at all and is bit-identical to its input.

When `half`, `src` is the full-resolution pixel at `(x, y)` and `T`, `C_dc`, `C_se`, `Y` are read at `(x / 2, y / 2)` (integer division, nearest).

### 7.6 Order

Per face: `T` once, from the image as it is on entry; dark circles in place; then even skin in place, with the dark-circle result as `src` and the same `T`.

### 7.7 Skin grading (the owner, 2026-11-11, after Resolve)

Three controls on the face's skin, working like even skin on the low-frequency layer `T` of 7.2, so pores and fine detail are kept. Mask `m = grade / 255` (5.6). `Y = (T.r + T.g + T.b) / 3` and the regional average `C_g = masked_blur(Y, grade, r_g)` with `r_g = max(1, trunc(0.15 * D / ds))`, wide enough to be the face's own level. In order, each reading the previous one's result as `src` and the same `T`:

**Contrast** (`gc`, -1..=1): the skin's low-frequency light spread around its own level, in log space:

```
Y'  = C_g * (Y / C_g)^(1 + 0.5 * gc)
new = src * (Y' / max(Y, 0.001))
out = lerp(src, new, m)                    // both directions: contrast may lift or lower
```

**Shadow desaturation** (`gd`, 0..=1): less colour in the skin's shadows, where cameras and mixed light leave muddy or green casts:

```
w   = clamp((C_g - Y) / (0.5 * C_g), 0, 1)          // 0 at the skin's level, 1 at half of it
L   = (src.r + src.g + src.b) / 3
out = L + (1 - 0.6 * gd * w * m) * (src - L)        // luminance kept
```

**Shine removal** (`gs`, 0..=1): specular sheen on the forehead, nose and cheeks pulled back toward the skin around it:

```
sat  = 1 - min(T) / max(T)
e    = ln(Y / C_g) - 0.1                              // more than about 10 % above the skin's level
w    = smoothstep(0, 0.4, e) * (1 - smoothstep(0.15, 0.35, sat)) * m
C    = masked_blur(T, grade, r_se)                    // the local skin colour (RGB)
T'   = lerp(T, C * (C_g * e^0.1) / max(mean(C), 0.001), gs * w)
out  = src - T + T'                                   // texture (src - T) kept
```

### 7.8 Teeth whitening (the owner, 2026-11-11)

In the teeth mask of 5.6, `m = teeth / 255`: less yellow, a little more light, never past the brightest tooth:

```
L    = (src.r + src.g + src.b) / 3
c    = L + (1 - 0.7 * tw * m) * (src - L)             // chroma toward neutral (yellow first, as it is the main cast)
cap  = Lmax / max(L, 1e-4)                            // Lmax = the mask's 99th percentile of L
out  = c * min(1 + 0.15 * tw * m, max(cap, 1))
```

### 7.9 Order

Per face: blemish removal (6), then dark circles and even skin (7.1 to 7.6), then skin grading (7.7: contrast, shadow desaturation, shine removal), then teeth whitening (7.8). A tool whose amount is 0 or whose mask is empty does nothing; with every amount 0 the image is unchanged bit for bit.

## 8. Pipeline, entry point and tests

```
for each face: masks (5), cached per image; blemish map (6.1), cached per image and face rect
output = copy of input
for each face: blemish apply (6.2)
for each face: dark circles, even skin, skin grading, teeth whitening (7, in the order of 7.9)
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
- **Skin grading:** contrast at +100 widens and at -100 narrows the spread of low-frequency skin luminance around `C_g`, with high-frequency texture energy kept within 5 %; shadow desaturation lowers chroma only below the skin's level and keeps luminance; shine removal lowers a synthetic specular patch toward the skin around it and leaves skin at its level unchanged.
- **Teeth:** whitening lowers the chroma of yellow teeth on a synthetic mouth, keeps the lips and gums (outside the gated mask) unchanged, and never raises a tooth above the brightest one.
- **Properties** (`proptest`) of the pure kernels: finite outputs for finite inputs, monotonic in the amount.
- A `criterion` benchmark: 24 MP, one face.

## 9. Decisions (where the brief left room)

1. **Masks and `F` at full resolution.** The `FaceParser` and `FaceMesh` traits give full image coordinates, so `F` uses the full-resolution size; section 7 resamples a mask by `render_scale` (bilinear) to the working image.
2. **The blemish map from the full-resolution crop.** Never from a preview, so a preview and an export use the same map (the scale test needs it). This is why `retouch_skin` takes `full_size_px` and the cache holds maps per face.
3. **Open points:** bilinear for mask resizing; a disc for dilation; radius, not side, for the max filter and box blur of 6.1 step 6; `rgb_gain` 1; the mesh tables belong to the `FaceMesh` implementation (the mock draws an analytic table).
4. **Colour space:** linear AWG4 from the node's LogC4 picture (`logc4_decode_rgb`), encoded back after (round 2).
6. **S-02 choices.** Feather: a radius that truncates to 0 returns the mask unchanged (no resize round trip); the result is clamped to never go below the input for a positive radius nor above it for a negative one (the Gaussian of a thin mask would otherwise lower it); borders reflect without repeating the edge sample (reflect-101); bilinear resizes are pixel-centre aligned and round to nearest; the disc dilation ignores pixels outside the mask. Ellipse holes use pixel centres and truncate the ramp. `F` and `D` clamp `face.scale` to 0.05..=1; NaN settings become 0. (The neck, an optional second `FaceParser`, was removed by S-15.) Mesh and parser regions may extend outside the image and are clipped.
5. **The proposer's name:** `BlemishProposer`, since a classical implementation fills it until a trained one exists.
7. **S-04 choices.** Edges of every box filter replicate the edge pixel (the masked ratio uses the same rule, so it stays a weighted mean). A tool never changes a pixel whose own mask byte (at the working resolution) is 0, even on the half path, where the half-resolution mask can be non-zero one pixel outside the mask; this keeps "outside the masks, bit-identical" exact and stops the 0.001 floor of `Y` from darkening unmasked pixels. Dark circles hand their result to even skin in `f64`, without rounding to `f32` between the two. A mask that resamples to nothing (off the image) makes its tool inactive.
8. **S-03 choices.** The strength max filter is a square window of side 7 (radius 3) and ignores pixels outside the map; the box blurs (strength and chroma guard) shrink the window at the borders instead of replicating. The area downscale gives `round(w * k)` by `round(h * k)` (at least 1) and averages the covered source area with fractional edge weights; the mask is downscaled the same way and rounded to `u8`. A non-finite `face_px` counts as 1. NaN or negative values encode as 0. The chroma guard changes only pixels at least `radius` from every edge with `conf < 1`; others keep the proposer output. The map cache holds one key at a time (image key, face count, face rects, upstream key; the two keys are opaque `u64` from the caller) and one `Arc<BlemishMap>` per face index.

9. **S-05 choices: the entry point and the full-resolution image.** `retouch_skin` keeps the signature of section 8; the full-resolution image reaches it through the cache. `prepare_skin(full_image, faces, models, cache)` builds the masks and the blemish maps of every face from the full-resolution image and stores them in the `RetouchCache`. A preview then calls `retouch_skin` on its working image, which only reads the cache; if something is missing it fails with `RetouchError::NotPrepared` instead of building from a preview. When the working image is itself full size (an export), `retouch_skin` builds what is missing from it and caches it, so the CLI needs one call. The cache is named by two opaque keys (`set_keys(image_key, upstream_key)`; a change drops everything). Masks and maps are rebuilt when the faces or the full size change, so every amount reuses them (the two neck switches of the first version were removed by S-15). A map covers the face rectangle joined with the blemish mask's region, clipped to the image, and `face_px` is `D` at full resolution. Maps are built lazily and only for faces with Blemish above 0 on the full-image path, and for every face in `prepare_skin`. With every amount 0 the call returns a copy of the input before touching the cache. The working size must equal `full_size_px * render_scale` within 2 px (`RetouchError::WorkingSize`); a protection mask must have the working size.
10. **S-05 choices: border conventions.** The blemish box blurs shrink the window at the borders (decision 8) and the low-frequency filters replicate the edge pixel (decision 7). Both are kept: they are separate stages with their own tests, the blemish blurs run on a per-face map whose outer pixels are faded to gain 1 anyway, and no formula changes. Nothing shares a helper.

12. **S-09 and S-10 choices.** (a) **The real models live in `ym-retouch` behind the feature `mediapipe`** (`ym-ml` as an optional dependency; `ym-ml` does not depend on `ym-retouch`, so there is no cycle). The maths that needs no network (rasteriser, tables, parts, guided filter) is in the crate without the feature and is tested without weights; `ym-core` (S-11) turns the feature on in one line and needs no adapter module of its own. The detector's `DetectedFace` still converts to `Face` in two lines at the wiring. (b) **`FaceMesh::render` takes the image** (`render(image, face, table, size_px)`): a mesh from a network needs the picture, and the trait gave it none; the mock ignores it, `face_masks` passes it. This changes the trait of section 4 for the callers of S-11. (c) **One analysis per face** is shared by the three parser calls and the two mesh calls through a cache in `MediaPipeFaces` (the last 8 faces) keyed by the rectangle, the picture's size and an FNV-1a hash of every pixel of the rectangle grown by 1.5 of its size on each side, so the answer depends on the picture only (determinism) and never on call order. (d) `Face` carries no eye keypoints, so the parser's first mesh pass is the unturned box and two refinements follow (S-09). (e) Coordinates: the networks count pixel centres as integers, the masks as halves; the adapter adds 0.5 to the landmarks and the crop's centre. (f) `InvertedSkinUniformity` is bright where the skin is not uniform (the trait's text and the mock), not where it is.

11. **S-15 choices.** (a) The teeth mask is built from the full-resolution image, cached with the other masks and rebuilt only when the faces or the size change; it is built from the image as it enters the pipeline, not after blemish removal. `Ymouth` and `Lmax` are nearest-rank percentiles (sorted with a total order, `ceil(p * n)`), so they do not depend on the thread count; `Ymouth` is taken over the pixels where the feathered mouth mask is not 0, `Lmax` over the pixels where the working teeth mask is not 0, from the image as it is when whitening runs (after grading). (b) A mouth interior that is missing or whose parser fails only turns whitening off (logged). (c) In 7.7 the per-pixel `m` is the mask byte at the working resolution (not the half-resolution mask), so a byte of 0 is exactly untouched; `C_g` and `C` use the layer-resolution mask. The padded region grows by `r_g` (0.15 D) when grading is active. (d) Non-positive `Y` or `C_g` leave a pixel alone in all three grading steps (the formulas divide by them); `w` of shine removal is exactly 0 when `e <= 0`, so `e^0.1` is never taken of a negative number. (e) Contrast, shadow desaturation and shine removal do not use the asymmetric blend of 7.5; only the protection scales them. (f) Protect: a protected pixel (`protect = 255`) is skipped before any arithmetic, so it is bit-identical to the input; between, `k` (dark circles, even skin) and `m` (7.7, 7.8) are scaled by `1 - protect / 255`. The `retouch_tones` entry takes the protection mask and checks its size.

13. **S-11b choices: what the models see, and when they run.** The brief for S-11b asked that dragging any slider (the node's, a corrector's upstream, the look) never re-run a network, and that a 24 MP export stay cheap. The measured numbers are in the last paragraph.
    (a) **The analysis picture.** The finder, the face mesh, the segmenter, the mole detector and the blemish proposer run on a picture that depends on **no node's grade, no amount and no look**: the Input's picture in the photo's **final frame** (crop, turn and lens kept, so a face is where the render has it; `graph/faces.rs`, `face_source`), Neutral look, no output sharpening, at most `FACE_ANALYSIS_LONG_EDGE_PX` = **1400** px on the long edge (never enlarged), made by the renderer of its own that has no graphics device (`Renderer::object_reference`), so the picture, the faces and the masks are the CPU reference path's bytes whatever the machine and the app and `ym` agree. It is two renderings of **one sampling**: sRGB 8 bits (what the finder looks at) and the LogC4 / AWG4 value before the look in `f32`, decoded to linear AWG4 (what the masks and the blemish maps are made on, in the encoding the node's own input has). **Why 1400:** the raw group a picture is made from is the largest reduction that leaves the long edge twice the picture (`thumbnail_scale`), so 1400 reads the half-resolution group of a 24 MP photo and 3072 read the full-resolution one: the first analysis of the 24 MP photo took **37 s** at 3072 and **1.3 s** at 1400.
    (b) **The key** names what the picture was made from: the file, the raw group's numbers, the Input's colour, the final frame's size, geometry and lens, the weights (the three `SHA256SUMS` lines and the smallest face), the formats and the rules' version (`FACES_FORMAT_VERSION`, `FACES_RULES_VERSION`) and the picture's size. A node's amounts, the look and a corrector upstream are not in it. A change of the Input's exposure or the crop makes a new analysis (debounced 150 ms); a viewer keeps the latest analysis meanwhile **while the frame is the same**.
    (c) **The faces.** Found on the 8-bit picture by the finder (`FaceFinder`, tiled; `min_face_px` = 0.8 of the floor in the picture's pixels); **dropped under `MIN_FACE_FULL_PX` = 200** px across in the full-resolution photo, **the best `MAX_FACES_KEPT` = 32 kept** by score, **numbered 1 to 32 from left to right** by the centre of the box (ties by the top edge, then the score). `DetectedFace` becomes `ym_retouch::Face` in two lines (decision 12). All of these constants are starting values for S-12.
    (d) **The masks and the blemish maps are built once per analysis, at the analysis frame's size** (`prepare_skin` on the analysis picture, the `RetouchCache` kept in the photo's graph), not at the photo's full resolution: the frame is `F` of section 5 and `render_scale` is the working image's size over it, now up to **16** (a 24 MP export of a 1400 px analysis is x4.3). Every mask and map is resampled bilinearly to the working image (`mask_to_working`, `apply_blemish` already sampled bilinearly), so a preview, a viewport frame and an export share the same masks and maps, which is what the scale test of section 8 needs. Consequence: the masks of a face are as sharp as 1400 px allows (a 1370 px face is read at 350 px and its edges are soft by about four export pixels; every radius is a fraction of the face, at least 1.1 % of it, so they are broader than that), and a face of 200 px in a 6000 px photo is 47 px in the analysis. `BlemishProposer::for_face` makes the classical proposer per face with the face size in map pixels (it had been built by the caller).
    (e) **What runs on every frame** is the cheap apply only: the node's input (LogC4 from the chain stopped at the node, `film_pass`'s prefix chain) is decoded to linear AWG4 over the face's **region**, `ym_retouch::retouch_skin_region` applies the cached masks and maps to it (`apply_blemish_at`, `retouch_tones_at`: the same functions as the whole-image path with the part's place), and the result is encoded back. No network runs: the MediaPipe cache that hashes the pixels of the image it is given is never given the node's input. **The gain map from another grade:** a gain is a ratio of the same pixel before and after, so a change of exposure leaves it exact (the map of the neutral picture applied to a picture at +2 EV equals four times the neutral result, bit for bit: `a_map_of_one_grade_applies_to_another`); a tone curve changes the ratio: through a strong power curve (0.7) the papule's contrast of -0.205 becomes +0.111 with the neutral map (an over-correction to the other sign and about half the size) against -0.022 with the map of its own picture. A corrector upstream with a gentle curve is well inside that; S-12 can measure real grades.
    (f) **Regions, not the whole frame.** Every tool changes only pixels inside a rectangle around its face: the padded region of the low-frequency tools and the footprint of the blemish map (`tones_region`, `retouch_regions`). Regions that overlap are merged, every origin is made even (the half-resolution path works on 2 x 2 blocks of the whole image's grid), and the picture is made over the regions only; the rest is the node's input, passed through **exactly** (a pixel the retouch left as it was is marked `NaN` in the node's picture and the chain keeps its own value, so there is no round trip through `f32` outside the effect). **The part equals the whole image bit for bit** (`parts_around_the_faces_equal_the_whole_picture_bit_for_bit`: four faces, two of them overlapping, every tool on, a protection plane, at scales 0.5, 1 and 2). The working size must equal `full * render_scale` within two pixels, and within `ceil(scale)` above scale 2.
    (g) **The step and the scheduler.** The node is a picture step of the chain like a Film node's (`FilmStep` with Strength 1, the chain blends `in + m (made - in)` by the node's mask and gives the key output as for a corrector). Its reach is 0: a render that asks for an area is given the area grown to hold every region it meets (`FaceWork::widen`, with the reach of the protection plane's Grow, Feather and Refine around it), and a region an earlier tile of the same export made is kept and not read again. An export in tiles, in bands or in one piece gives the same bytes (`one_and_four_threads_give_the_same_bytes`, tiles of 256 px included).
    (h) **Protection** is rasterised by the same brush rasteriser as a Brush row (`RasterRow`, Grow, Feather and Refine included) over each region at the render's resolution, quantised to 0..255, and given to the library as the protection plane. A protected pixel is bit-identical to the input.
    (i) **A Layer fed by the node's picture** gets what a corrector's gives it (`plan_joins`): the node's picture before its own mask blend (the retouched picture where it has faces, its input elsewhere) with its `m` as the alpha. A Layer at Normal and 100 % over the node's input with the node above it is the node alone, with or without a mask on the node (`a_layer_fed_by_the_nodes_picture_gets_what_it_gets_from_a_corrector`).
    (j) **CPU only until S-13.** A frame with a live node (not bypassed, an amount above 0 or a face with settings, a model on this machine) is made wholly on the CPU: `Job::view_gpu` answers `None` with `device_refused` and the reason "Face Refinement runs on the CPU", which is how the viewport says "Rendering on the CPU" (`views_on_cpu` stays true until a device frame is made); a linear DNG's colour stage is the CPU's too. Taking the whole frame to the CPU is simpler than a mixed frame, and it is exact: the device and the CPU renderers agree to the byte for such a graph (`any_graph.rs`, where a device opens). A node that does nothing costs the device nothing.
    (k) **No models.** A copy without the weights has no `FaceModel`: the node passes its picture through in a view, `face_state` says `models_missing` with "Face Refinement needs the face models, and this copy of Yellowmoon has none. Reinstall the app.", and an export or `ym render` of a photo with a live node stops with `error.kind` `mask` and "Node 2 is a Face Refinement node, and the face models are missing. Reinstall Yellowmoon, or remove the node." (as a Depth row without its model does); a bypassed node, one with every amount at 0 and a photo without the node go on. The weights are found as the depth model's are (`<data root>/models/mediapipe/`, the bundle's resources or `data/models/`), each checked against `SHA256SUMS` on first use; a damaged file takes the model away for the run.
    (l) **The phase.** The core tells its listener `FaceEvent::Started` when the worker begins and `FaceEvent::Finished(FaceState)` when it ends (one for the other); `FaceState` is the phase (`finding`, `ready` with N faces, `failed` with the sentence, `models_missing`) and the numbered faces with their boxes in the oriented photo frame, as shares of its width and height (`Renderer::face_state`, `Core::face_state`; the app's `face_state` event and command, section 9 of `docs/23`). The worker is started when the photo being looked at has a Face Refinement node (added, or a photo with one opened or pasted to: `Core::prepare_masks`), stops without a word when the person leaves the photo, and is not started by the thumbnails of other photos.
    (m) **Measured** (this container: 4 cores, Intel Xeon 2.1 GHz, no graphics device, release build; a 6000 x 4000 raw file made from the public portraits, faces of 890 px and, in the second, two of 630 to 780 px; `tests/face_timing.rs`, ignored, `YELLOWMOON_DATA=$PWD/data cargo test -p ym-core --release --test face_timing -- --ignored --nocapture`; the container is noisy, differences under 40 ms are not significant). **The first analysis** (once per photo and key, in the background): 1.7 s with one face (the picture 1.25 s, the finder 0.12 s, the masks and maps 0.32 s) and 1.9 s with two (1.2, 0.07, 0.6). **A viewport frame** in a 1600 x 1000 cell, the same upstream moves with the node at 0 (no face step) against with it live: at Fit, 413 ms against 384 ms after a slider of the node and 451 ms after a corrector upstream with one face, and 404 against 424 and 472 with two; at 100 %, 529 against 512 and 533 with one, 476 against 586 and 632 with two. **The step is 20 to 70 ms at Fit and 0 to 150 ms at 100 %**; a region of 244 x 325 px costs 4 to 6 ms for the prefix chain, 11 to 17 ms for the retouch, 1 ms each to decode and encode; one of 968 x 1300 px (a face of 890 px at 100 %) 65 ms, 114 ms, 11 ms and 8 ms. The 400 ms the frame costs without the node is the CPU colour pass of a container with no device. **An export of the 24 MP photo**: 29.3 s without the node and 29.8 s with it (one face), 29.7 s and 32.8 s (two): the node adds its own analysis (1.7 to 1.9 s, not kept: an export keeps nothing), the pixels of each region read once (a region is not read again by the next tile), and 0.2 to 0.3 s of retouch. (Before regions were made once and the tiles' widening was limited to the regions a tile itself meets, the two-face export took 8 s more.) The numbers on a Mac are M0-H4.

## 10. Tasks

Each task by a Sonnet agent on `skin-retouch` (or a branch off it); the lead reviews and commits.

- **S-01 · Spec.** This file. *Done when:* committed.
- **S-02 · Crate, types, traits, mocks, masks** (sections 2 to 5). *Done when:* `ym-retouch` builds in the workspace; types, errors, traits and mocks are documented; mask arithmetic, feather and the three mask builders pass their tests (feather monotonicity included); `cargo deny check` passes.
  *As built (S-02):* `crates/ym-retouch` has modules `types`, `error`, `models`, `mock` (feature `mock`), `feather` and `mask`; S-03 to S-05 add `blemish`, `lowfreq` and `pipeline`. `Models` holds the parser (the optional neck parser was removed by S-15), the mesh, an optional beauty-mark detector and the proposer. Builders work on full-size masks and `face_masks` returns three `Option<BoundedMask>` cropped to their bounding boxes. 20 unit and property tests pass (u8 arithmetic with `proptest`, feather, hole ramp, builders with the mocks, crop, clamping, `render_scale`, determinism with 1 and 4 threads). `cargo deny` is not installed in this environment, so it was not run; the crate adds no dependency that was not already in the workspace (rayon, thiserror, tracing, proptest).
- **S-03 · Blemish removal** (section 6), after S-02. *Done when:* map and apply match the formulas; the threshold, gain-clamp and chroma-guard tests pass; the map cache works.
  *As built (S-03):* module `blemish` in `crates/ym-retouch`. `blemish_map(crop, mask, rect, rgb_gain, face_px, proposer) -> BlemishMap` follows 6.1 step by step; `apply_blemish(&mut image, &map, a, render_scale, protect)` follows 6.2 (a protection mask must have the image's size). The pure kernels are free functions for a later GPU port: `encode_channel`, `chroma_confidence`, `chroma_pull`, `blemish_gain`, `gain_strength`, `threshold_strength`, `apply_gain`. `BlemishMapCache::get_or_build(key, face_index, build)` with `BlemishMapKey` is what S-05's `RetouchCache` holds. 21 new unit and property tests (41 in the crate) cover the threshold by hand values, the gain clamp, ring and fade, the chroma guard (tint pulled back with luma kept, equal scaling untouched), identity, tiling with four tile sizes and awkward image sizes, never upscaling, 1 against 4 threads, the u/v mapping and bilinear clamp, and the cache. See decision 7.
- **S-04 · Dark circles and even skin** (section 7), after S-02, alongside S-03. *Done when:* masked blur, blend and order match; the masked-blur, blend and texture tests pass.
  *As built (S-04):* module `lowfreq` in `ym-retouch`. `retouch_low_frequency(image, full_size_px, face, settings, render_scale, dark_circle_mask, even_skin_mask)` works in place on the working image and returns whether any tool ran; `LowFreqParams` holds the parameters of 7.1; `mask_to_working` resamples a full-resolution `BoundedMask` by `render_scale` (bilinear, pixel-centre aligned) to a `BoundedMask` in working pixels; the pure kernels `blend_weight`, `blend_pixel`, `masked_blur_combine`, `dark_circles_target`, `even_skin_target` and `dark_circle_k_sp` are free functions for a later GPU kernel. Only the padded bounding box of the masks is processed (pad = `(r_tex + r_tool) * ds + ds`, origin rounded down to even on the half path so the 2 x 2 blocks sit on the whole image's grid), so a small face on a large image costs little; the result equals processing the whole image. Layers are `f64` planes; blurs are box filters with a running sum in fixed order (rows in parallel, writes disjoint). 14 tests pass (34 in the crate): parameters, blur as two boxes, masked blur fixed point and locality, blend asymmetry, mask resampling, amounts 0 and empty masks, locality outside both masks, texture within 5 % (half, full and automatic path), half against full path (mean absolute difference under 2e-3), cropped against whole region, 1 and 4 threads, and `proptest` for finite kernels, monotonic moved fraction `k * w` (dark circles and even skin) and the constant-image fixed point of the whole operator.
- **S-05 · Pipeline, the remaining tests, benchmark** (section 8), after S-03 and S-04. *Done when:* identity, locality, scale and determinism pass; proptest for the pure kernels; the 24 MP benchmark runs and its time is written here.
  *As built (S-05):* module `pipeline`: `retouch_skin`, `prepare_skin` and `RetouchCache` (decisions 9 and 10). 10 new tests (65 in the crate, about 4 s): identity (also with NaN and negative amounts, and no faces), locality through the whole pipeline with the mocks (exact outside every mask with the identity proposer; with a proposer that scales by 0.9, nothing moves more than 3 px from a mask, because the map is sampled bilinearly on a grid coarser than the image), scale consistency (1200 x 900 with a 500 px face, amounts 60, preview prepared from the full image: mean absolute difference under 1 % and under half of the change itself), determinism with 1, 3 and 4 threads, cache reuse and switches, protection mask, input errors, a failing detector and an absent neck, mask cropping, and a `proptest` that the output is finite for any settings (and the identity when no amount is above 0). The pure kernels already carry their own `proptest`s from S-03 and S-04. Benchmark `benches/retouch.rs` (`cargo bench -p ym-retouch --features mock --bench retouch`; criterion 0.5, MIT OR Apache-2.0, a dev-dependency only): 6000 x 4000, one face 1500 x 1800 (`scale` 0.3), mock models (identity-shaped proposer scaling by 0.9, 192 px tiles, border 32), blemish, dark circles and even skin at 60. Linux container, CPU only, 4 cores (`nproc` 4), Intel Xeon at 2.1 GHz, release build, 10 samples (the container is noisy; the ranges are criterion's): face masks alone 650 to 850 ms; `prepare_skin` (masks and blemish map) 370 to 450 ms (the mask run above was measured at another moment, so the two overlap); apply on the full image with a warm cache 414 to 427 ms, which includes the 288 MB copy of the image; apply at `render_scale` 0.25 on a 1500 x 1000 preview 54 to 66 ms. The mock mesh and parser rasterise per pixel, so a real model changes the preparation time; the apply time does not depend on the models.

**Round 2** (the owner's answers of 2026-11-11, section 11). The research is `research/reports/Skin retouching models and data.md` (notes in `research/research_notes/Skin retouching models and data/`). Tasks, each by a Sonnet agent; the lead reviews and commits:

- **S-06 · Spec for round 2.** This section and section 11; the spec edits the research asked for: the blemish scale is 0.7 % to 3.5 % of `D` (the earlier sketch sat at pore scale); a proposer returns `OUT = IN` away from blemishes, so no veil reaches the clean skin at high amounts; a proposer is built per face with the face size in map pixels. *Done when:* committed.
- **S-07 · Classical blemish proposer and beauty-mark detector** in `ym-retouch`. The proposer: a gated detect-and-fill (dark spots by a per-channel black top-hat, red spots by a red-excess test, at blemish scale; spots under about 3 px dropped; inside each spot the blurred base is filled smoothly from its rim; everywhere else the output equals the input). The detector: dark, compact, brown, isolated blobs of 1.2 % to 4.5 % of the face, ranked by contrast, at most about 10, boxes padded about 30 %. Deterministic, `f64`, rows in parallel. *Done when:* synthetic tests with known lesions and moles pass (spots removed, moles and the clean skin untouched), and the constants are documented as starting values for S-12.
  *As built (S-07):* module `classical` in `ym-retouch` (`ClassicalProposer::new(face_px)` and `ClassicalBeautyMarks`, both re-exported; every constant is a named, documented starting value in `classical::constants` for S-12). The proposer is built per face with the face size in map pixels (`min(face, 450)`), because the trait passes none; its tile is 320 px and its border is derived from the face size (56 px at 450). It works in the received encoding on natural logs: Gaussian smoothing (0.35 % of the face), a per-channel black top-hat with a square window of 3.5 % of the face, a red-excess top-hat of `ln R - ln G` (opening), a score (deepest channel over 0.05, or red excess over 0.06), hysteresis at 0.55 of the seed, 8-connected components dropped under 3 px, over 5 % of the face or beyond aspect 3.5, a disc growth of 0.4 %; inside the mask the base (Gaussian of the pixels outside the mask, sigma 0.3 % of the face) is filled harmonically (120 over-relaxation sweeps from the rim) and the detail is kept, clamped to 0.03. Output outside the mask is the input, copied bit for bit. The detector crops the face rectangle (at most 320 px), takes log luminance, finds local maxima in space and scale of six differences of Gaussians (sigma 0.4 % to 2.6 % of the face), measures each of the 64 strongest against the median of its surroundings (extent = region darker than half its depth), and accepts compact, round, dark (0.4 to 1.4 in `ln` luminance), brown (red drop over 0.04 and at least 0.35 of the green drop) blobs of 1.2 % to 4.5 % of the face; strongest first, overlapping boxes dropped, at most 10, padded by 30 %. Decisions: lighter brown marks (contrast under 0.4) are not moles and go to blemish removal; the detector sees no skin mask, so it can box pupils and nostrils, which only punches holes where the mask is already 0 (they are refused above 1.4); square, not disc, windows for the top-hat (van Herk, linear time); detail is clamped so a spot's crisp rim does not return as a ring. Tests (synthetic skin in linear light, seeded: tone, gradient, 1.5 % texture, 40 pores; papules `[0.97, 0.72, 0.80]`, brown spots `[0.82, 0.68, 0.60]`, moles `[0.50, 0.38, 0.30]`): 20 new tests (85 in the crate, about 6 s). Measured on this container (4 cores, Linux, optimised test profile): worst residual contrast inside a removed spot 0.28 of the original (papules and dark spots from 3 to 14 px: 0.0 to 0.17; the smallest brown spot, 5 px across, 0.28; limit asserted 0.3); on 8 clean patches and everywhere farther than `2R + 8` px from a lesion the output equals the input exactly; texture inside a spot keeps 76 % of the high-pass amplitude; 1 and 4 threads give the same bytes; a 128 px tile equals one big tile to 6e-8; through `prepare_skin` and `retouch_skin` at Blemish 60 the papule contrast falls from -0.28 to +0.006 and the brown spot from -0.32 to -0.04, clean skin is bit-identical, and at Blemish 100 a mole beside a papule keeps its contrast (-0.62 to -0.62) while the papule goes (-0.28 to -0.003); detector on 8 patches with 24 moles (radius 3.5 to 8), 32 papules and 32 brown spots: recall 1.000 (24/24), precision 1.000 (no papule or brown spot boxed); a 16-mole patch returns 10 boxes; a mole of core radius 6 gets a 17 x 16 px box. Time per 450 px face (one 450 x 450 map, four 320 px tiles): proposer about 0.27 s, detector about 0.06 s. Limits for S-12: the synthetic skin has no hair, lips, brows or shadows, so thresholds are untested against real portraits; flat large red areas (wider than the window) are not blemishes by design.
- **S-08 · MediaPipe weights and the face finder.** BlazeFace full range, Face Mesh V2 (from `face_landmarker.task`) and Selfie Multiclass, Apache-2.0 per their model cards; a CI-only script converts the TFLite files to safetensors (Python for exporting weights only, rule 6); fetch scripts with SHA-256, a release asset and the bundle as for SAM 2.1 (the owner: "bundle"), model cards and NOTICE lines; BlazeFace ported to Rust in `ym-ml` with tiling for group photos, our own NMS and face cap; parity against the reference outputs. *Done when:* faces found on public test portraits match the reference within the stated tolerance, and the weights are bundled by the release workflow.
  *As built (S-08):* **Weights.** `tools/mediapipe/convert.py` (Python, CI only; `tflite` 2.18.0 and `numpy` pinned in `tools/mediapipe/requirements.txt`) downloads the three originals from `storage.googleapis.com/mediapipe-models/.../1/` (the versioned URLs), refuses a file whose SHA-256 is not the pinned one (and checks the member `face_landmarks_detector.tflite` of `face_landmarker.task` too), and writes `blaze_face_full_range.f16.safetensors` (1,067,186 bytes, 514,069 values), `face_mesh_v2.f16.safetensors` (2,448,944 bytes, 1,199,316 values) and `selfie_multiclass_256.f32.safetensors` (16,321,304 bytes, 4,070,284 values, f32 because Google ships it in f32). Constants are `t<TFLite tensor index>` (the f16 weights behind DEQUANTIZE are stored as they are); the header holds the operator graph (types, tensor indices, every builtin option, the integer constants, inputs and outputs). The writer is our own with sorted keys (the `safetensors` package orders its header at random), so the output is byte-identical on every run and the three hashes are **committed now** in `data/models/mediapipe/SHA256SUMS` (`0261507f…`, `f56049da…`, `a889f4ef…`); the `mediapipe` job of `models.yml` compares its output with them and publishes only equal bytes. Model card, `NOTICE` item 14, `docs/14` rows, `docs/07`, `docs/12`, `tools/fetch/mediapipe.sh` (with `test_mediapipe.py`, `--only`, `--verify`, `--photos`) and `check_bundle.py --require-mediapipe` are in place; `release.yml` and `build.yml` cache, download (GitHub CLI), fetch, check and verify the files in the bundle **once the release exists**, and warn and go on without them before that. **To publish (the lead):** push a branch named `models/mediapipe-faces-1` (for example `git push origin skin-retouch:models/mediapipe-faces-1`) or run Actions > models > Run workflow with the job `mediapipe`; the job converts, checks the hashes against the committed ones, and creates the prerelease `models-mediapipe-faces-1` with the three files and a `SHA256SUMS`. Nothing else needs copying. From then on the CI job `mediapipe` fetches the weights and the photos and sets `YM_REQUIRE_MEDIAPIPE`. **The runtime.** `ym_ml::mediapipe::graph` is a runner for such a file (`CONV_2D`, `DEPTHWISE_CONV_2D`, `ADD`, `RELU`, `PRELU`, `LOGISTIC`, `MAX_POOL_2D`, `PAD`, `RESIZE_BILINEAR`, `RESHAPE`, `CONCATENATION`; a graph with another operator is refused when it loads), instead of one hand-written forward pass per network, so S-09 and S-10 add only what they lack (Selfie Multiclass: `MUL`, `SUM`, `SOFTMAX`, `TRANSPOSE`, `TRANSPOSE_CONV`, nearest resize) and no weights file changes. As a smoke check (a throwaway example, not committed) Face Mesh V2 already runs through it and agrees with the TFLite interpreter on a synthetic input to 1.5e-4 absolute on values up to 221. Bits are fixed (f32, one chain per output, no fused multiply-add, `sam2`'s packed product for the 1 x 1 and im2col convolutions); 1 and 4 threads give equal bits. **The finder.** `ym_ml::mediapipe::blazeface`: the letterbox of MediaPipe's `ImageToTensor` into 192 x 192, `x * 2 - 1` (found by trying both ranges against MediaPipe's task: `[-1, 1]` matches, `[0, 1]` does not), black bars; MediaPipe samples bilinearly at `(dst - offset) / scale` with no half-pixel shift and no filter (OpenCV `warpAffine`), which `Resample::MediaPipe` reproduces and `Resample::Box` (the default) repeats on a picture first reduced by an integer box filter of `floor(0.5 / scale)` pixels, so that a 24 MP photo is not sampled one pixel in 30; 2304 anchors (48 x 48, stride 4, fixed size, offset 0.5); box and six keypoints from 16 regressors; score = sigmoid of the logit clipped to +-100; a weighted NMS of our own (best score first, group = IoU above 0.3, score-weighted average of box and keypoints, the best score kept), score threshold 0.5. **Tiling:** the network finds faces from about 7 % to 50 % of its frame (measured: 10 px of 192 start to be found, 12 px reliably, the score falls past 150 px), so `plan_windows` gives windows of `min_face / 0.07` pixels (default minimum 1.5 % of the longer side, at least 24 px) stepping by two thirds of their size, each next size four times larger while under three quarters of the picture, and the whole picture last (6000 x 4000: 7 x 5 windows of 1286 px and the whole, 36 passes). In each window a detection under 6 % of the window is dropped (its box is coarse: a 219 px face seen in a 6000 px window came out 281 px and, merged by score, spoiled the good boxes from the 1286 px windows) and so is one that touches an inner border of the window (it is whole in a neighbour); the rest is merged by the same NMS, smaller than the minimum is dropped, and the best 32 are kept. `FaceFinder::find` returns `DetectedFace { rect, scale, keypoints_px, score }`: `rect` has the fields of `ym_retouch::Rect` (whole pixels, rounded outwards and clipped to the picture), `scale` is the longer side of the box over the longer side of the picture; the type stays in `ym-ml` so that neither crate depends on the other, and the adapter to `ym_retouch::Face` is two lines in the place that wires the Skin node (S-11). `PixelSource` takes 8-bit or float RGB in display-referred 0..1 (the caller converts from LogC4 or linear AWG4). **Parity** (`crates/ym-ml/tests/blazeface.rs` against `tests/fixtures/blazeface.json`, 48 KB, written by `tools/mediapipe/reference.py` from Google's interpreter `ai-edge-litert` 2.3.0 and Google's `FaceDetector` task of `mediapipe` 1.2.2; weights and photos from `tools/fetch/mediapipe.sh`, the tests skip with a plain line without them and `YM_REQUIRE_MEDIAPIPE=1` makes that a failure): the raw network on a fixed synthetic input (a formula both languages compute bit for bit): largest logit difference **6.2e-6** (bound 1e-4), regressors **8.9e-7** of the largest (bound 1e-5); three photos with MediaPipe's own resampling against the interpreter on its tensor: IoU at least **0.9931** (bound 0.98), score difference at most **0.0025** (bound 0.01); against MediaPipe's task, whose box is whole pixels: worst edge **1.45 px** (bound 2.5 px or 2 %), worst keypoint **0.33 %** of the box (bound 2 %), score difference at most **0.005** (bound 0.02); the finder's defaults on the same photos against the task: same number of faces, IoU at least **0.93** (bound 0.85); a 24 MP picture with seven faces from 49 to 390 px pasted into a noisy background: every face of 90 px or more found once and no other (5 of 5, IoU at least **0.877**, bound 0.8), all 7 with `min_face_px` 30; a cap of 2 returns the best two; 1 and 4 threads give equal results. **Photos:** Wikimedia Commons answered 429 to every request from this environment for the whole session, so the CC0 portraits of `training/depth/oracle_photos.json` could not be fetched; the three photos used are MediaPipe's own test images (`storage.googleapis.com/mediapipe-assets/portrait.jpg`, `man-woman-okay.jpg` with two faces, `woman_hands.jpg`; part of the Apache-2.0 MediaPipe repository's test data, pinned by SHA-256 in `tools/fetch/mediapipe-photos.tsv`, downloaded at test time and never committed). Adding CC0 Commons portraits is a matter of listing them in the TSV and running `reference.py` again. **Timing** (Linux container, 4 cores, Intel Xeon 2.1 GHz, release build, `cargo run --release -p ym-ml --example faces_time`): one 192 x 192 pass **12 ms** on 4 threads (20 ms on 1; 105.7 M multiply-adds); the letterbox of an 820 x 1024 photo 0.9 ms (2.7 ms on 1); the finder on 820 x 1024 **84 ms** (280 ms on 1); on a **24 MP** (6000 x 4000) picture, 36 passes, **240 ms** on 4 threads (0.8 s on 1). Open: the default minimum face (1.5 % of the longer side) and the score threshold 0.5 are starting values for S-11 and S-12; the Mac timing is M0-H4.
- **S-09 · Face mesh in Rust**, after S-08: Face Mesh V2 ported, two passes per face, the canonical topology (Apache-2.0) for region polygons, the mesh rasterised into a `BoundedMask`; the two tables (`UnderEye`, `InvertedSkinUniformity`) authored as our own UV textures. *Done when:* parity with the reference landmarks; the masks look right on a contact sheet of public portraits.
  *As built (S-09):* **Runtime.** `ym_ml::mediapipe::facemesh` (`FaceMeshNet`, `FaceLandmarker`) on the graph runner; no new operator was needed. The crop is `ym_ml::mediapipe::roi` (`Roi`, `crop_tensor`): a rotated square sampled bilinearly at `i / 256` (MediaPipe's `ImageToTensor`), values `0..1` (checked against the task: `x / 255`), black outside, `Resample::Box` reducing by whole blocks first. First pass: the detector's box times 1.5 turned by `atan2` of the eye keypoints (`landmarks`, when the finder's keypoints are known) or the box unturned (`landmarks_from_box`, what `FaceParser` has, since `Face` carries no keypoints); then refinement passes, each on the box around all 478 points times 1.5 turned to level points 33 and 263 (MediaPipe's `face_landmarks_to_roi`). **Finding:** with only a box, one refinement leaves 0.84 % of the face's size of error on a face tilted 29 degrees and two leave 0.16 %, so the `FaceParser` path makes three passes (first plus two); with the keypoints two passes suffice. Face presence is the second output's logit (about 15 to 30 on a face, negative on a gradient). **Topology.** `tools/mediapipe/gen_topology.py` (pinned SHA-256 of `canonical_face_model.obj` and `face_mesh_connections.py`, Apache-2.0, MediaPipe repository) writes `crates/ym-retouch/src/face_topology.rs`: the 898 triangles, the UV of the 468 vertices (v down) and the closed index rings of the face oval, outer and inner lips, eyes, brows and irises; no vertex positions. **Rasteriser.** `face_raster::render_mesh`: front-facing triangles only (winding sign `FRONT_FACING_SIGN`), per pixel the texture at the interpolated UV weighted by the triangle's share of the pixel (a 4 x 4 grid on the pixels an edge crosses, the whole pixel elsewhere), shares added in triangle order, so seams vanish and the outer border is anti-aliased; rows in bands of 32 on `rayon`, 1 and 4 threads equal. `fill_polygon` is the exact-coverage polygon filler (8 sub-scanlines, exact horizontally). **Tables** (`face_tables`): procedural 256 x 256 textures over UV, built from the UV of landmarks (eye corners and lower lid, nostrils, nose wings, mouth corners, chin crease): `UnderEye` is a crescent from the lower lid down to about half an eye's width, deepest toward the nose, fading at both corners; `InvertedSkinUniformity` is bright (the trait's meaning: bright where the skin is not uniform, so `mul_inv` keeps even skin off it; the mock agrees) on the lids and sockets, the sides of the nose and the nostrils, the folds from nose to mouth, the mouth corners, the philtrum and the crease under the lip, and 0 on cheeks, forehead, temples and jaw. All shapes are ours, starting values for S-12. **Adapter.** See decision 12. **Parity** (`crates/ym-ml/tests/facemesh.rs` against `tests/fixtures/faces.json`, 105 KB, written by `tools/mediapipe/reference_faces.py` from Google's interpreter and `FaceLandmarker` task; weights and photos as S-08, skipped without them, `YM_REQUIRE_MEDIAPIPE=1` fails instead): the network on a fixed synthetic input, largest landmark difference **2.6e-6** of the largest landmark (220.7 px; bound 1e-5), presence logit 1.8e-5; one pass on the reference's own crop (OpenCV `warpAffine`), mean / worst landmark distance **0.00041 / 0.00185** of the crop's side (bounds 0.0010 / 0.0040); our finder and two passes against the reference's own chain, mean **0.0014** of the face (bound 0.004); against MediaPipe's task (its own short-range detector, one pass), mean **0.0126** of the face (bound 0.02); the box-only path, mean **0.0047** (bound 0.01); a gradient crop has presence 0; 1 and 4 threads equal bits. **Timing** (Linux container, 4 cores, Xeon 2.1 GHz, release; `cargo run --release -p ym-ml --example mesh_time`): one 256 x 256 pass **25 ms** on 4 threads (21 ms on 1); a face's whole analysis (three mesh passes, the segmenter, the crops) **250 ms** for a 230 px face, **380 ms** for 690 px, **690 ms** for 1370 px; the two tables **33 / 124 / 298 ms** for those faces. **Photos:** Wikimedia Commons answered 429 again, so the CC0 portraits of `training/depth/oracle_photos.json` could not be fetched; the three MediaPipe test images of S-08 are used (faces of 72 to 230 px), and an enlarged copy of one (`face_masks_dump --scale`) stands in for a real 1000 px face in the timings and the contact sheet. `tools/mediapipe/contact_sheet.py` with the example `face_masks_dump` make the contact sheets (dev only, nothing committed).
- **S-10 · Selfie Multiclass in Rust and the real `FaceParser`**, after S-08: skin from the face-skin class inside the mesh oval plus a forehead band, eyes and other cut-outs from the mesh polygons, the mouth interior from the inner lip line, a per-face colour model at the edges and the existing guided filter (`docs/algorithms/06`). *Done when:* parity of the segmenter; masks on the contact sheet.
  *As built (S-10):* **Operators.** `graph.rs` gained `MUL`, `SUM` (any axes, `KeepDims`), `SOFTMAX`, `TRANSPOSE` (any permutation), `TRANSPOSE_CONV` (as a gather in a fixed order, SAME padding derived as TFLite does) and `RESIZE_NEAREST_NEIGHBOR` (`floor((i + 0.5) * scale)` with half-pixel centres); `ADD` and `MUL` now broadcast numpy-style and take a constant as either operand (Selfie Multiclass adds per-channel constants); unit tests for each, the transposed convolution against TFLite's scatter form for kernels 2, 3 and 4. The TFLite graph ends **before** the softmax (its outputs are logits, 2.97 and -1.02 on the first pixel of a test input): `SelfieNet::forward` applies the softmax over the six classes, `forward_logits` does not. **Segmenter** (`ym_ml::mediapipe::selfie`): the crop is the landmarks' box times 1.8 turned like the face (looked at at 1.4, 1.8 and 2.4: 1.4 cuts the hair off the top, 2.4 shrinks the face; at 1.8 the face fills about 55 % of the crop, so a 1000 px face is read at about 4 px per crop pixel). **The parser** (`ym-retouch::face_parts`, the maths without a network) follows section 1: the face-skin probability, bilinear at full resolution, times the mesh's face oval whose upper vertices move up by up to **0.16** of the face's height (weight squared in the height above the eye line; the ears and the neck stay outside the oval), minus the polygons of the eyes (margin 0.012 of the height), brows (0.010), outer lips (0.008) and the two nostrils (0.004; indices 98, 240, 75, 60, 99 and the mirrored 327, 460, 305, 290, 328, placed by eye on one photo), a Gaussian of the skin's colour (log brightness and two log chromaticity ratios, fitted to the solid core of the mask at a fixed stride, at most 20,000 samples) that trims, in a band of 0.07 of the height along the edge, pixels from 3 to 5 standard deviations away (a spot inside the face is never trimmed), the guided filter of `docs/algorithms/06` (`guided.rs`, written again on `f32` planes with the same maths and the Fast Guided Filter above 16 px; checked against an explicit double loop to 2e-5) with radius **0.02** of the face's height, `eps` 3.2e-4 and the picture's encoded luma as the guide, and the polygons cut once more without margins so the filter cannot pull skin back over a lid or a lip. `Eyes` is the two lid contours; `MouthInterior` the inner lip contour, `None` when the opening is under 0.4 % of the face's height. **The real models** (`ym_retouch::mediapipe`, feature `mediapipe`): `MediaPipeFaces` (the two networks and a cache), `MediaPipeParser`, `MediaPipeMesh`; the networks see the linear AWG4 picture as sRGB-encoded Rec.709 (`DisplayView`; the matrix is `ym-color`'s, checked by a test). **Parity** (`tests/facemesh.rs`): the segmenter on a fixed synthetic input, largest probability difference **5.0e-6** (bound 1e-4); on the reference's own crops of the photos (our chain to the crop), mean probability difference **0.0011** at most (bound 0.004), argmax agreement **0.9961** at least (bound 0.99), all samples within 0.05; 1 and 4 threads equal bits. **Masks** (`crates/ym-retouch/tests/mediapipe.rs`, on the four faces of the three photos): skin 65 % of the face box on the portrait, the eyes 1 % of the skin and under 10 % of them skin, the mouth interior inside the face and under 35 % skin, the under-eye crescent below the eyes and more than half on skin, the non-uniform table centred in the lower face, a picture without a face gives no parts, the same bits on 1 and 4 threads and from the cache, a changed picture never gets an old analysis. **Timing:** one segmenter pass **159 ms** on 4 threads (258 ms on 1); the whole parser per face is the analysis of S-09 plus the skin mask: first call **250 ms** (230 px face), **380 ms** (690 px), **690 ms** (1370 px); the following parts of the same face (eyes, mouth, tables) come from the cache: 2 to 80 ms. **Looked at** (contact sheets of the four faces and of 3x and 6x copies of the portrait): the skin covers the face to the hairline and the jaw, leaves out the ears, eyes, brows, lips and nostrils, and follows the tilted face. **Known limits:** the face-skin class includes a beard (the beard's hair is "skin" to the network), so a bearded face keeps its beard in the skin mask; hair over the forehead is cut by the class, the trim and the filter only as well as the network sees it; the nostril rings and every table shape were judged by eye on one photo; no portrait of 1000 px or more could be fetched.
- **S-11 · The Face Refinement node.** A node kind of its own, **`face_refinement`**, named **Face Refinement** (the owner, 2026-11-11: "Node name should be Face Refinement, since it is about faces and not generally skin"), in the node graph only: it is not offered in the simple main view (the owner, 2026-11-11). Settings per face (Blemish, Dark circles, Even skin, Texture; Skin contrast, Shadow desaturation, Shine removal; Teeth whitening), faces found automatically and listed in its panel, the protection mask from a Brush row; LogC4 decoded to linear AWG4 for `ym-retouch` and encoded back; the plan step, the caches, `ym` settings and the interface. Mockups first (S-H1). *Done when:* app and `ym` give the same bytes; the node's tests pass; the node renders on the CPU with the "Rendering on the CPU" pill until S-13.
  *As built (S-11a), the node's format, validation, patches, wire and CLI settings only (S-11b is the render step and the face finding, S-11c the interface; nothing of either exists yet).* The kind `face_refinement` is a ninth `NodeKind`; the nodes live in `NodeGraph::faces` (`BTreeMap<u32, FaceRefinementNode>`, `crates/ym-core/src/settings/face.rs`), sharing the id space and the 64-node limit with the correctors, the Film nodes and the mixers. JSON, with every field at its default (a bare `{ "id": 2, "kind": "face_refinement" }` is a valid item):

```json
{ "id": 2, "kind": "face_refinement", "name": "Face Refinement", "bypass": false, "pos": [1, 0],
  "same_for_all": true,
  "all": { "blemish": 30, "dark_circles": 30, "even_skin": 25, "texture": 0,
           "skin_contrast": 0, "shadow_desaturation": 0, "shine_removal": 0, "teeth_whitening": 0 },
  "faces": [],
  "protect": { "strokes": [], "feather": 0, "grow": 0, "refine": 0 },
  "masks": { "invert": false, "items": [] } }
```

  `all` is the `FaceSettings` every face takes while `same_for_all` is true and when `faces` is empty; `faces[n - 1]` is face `n`'s (faces are numbered from 1, left to right; at most 32 entries, 33 are refused with "A Face Refinement node can hold settings for up to 32 faces."); a face past the end of the list takes face 1's settings (`FaceRefinementNode::settings_for_face(face_number)`, face 0 reads as 1). A `FaceSettings` that leaves an amount out takes its default (30, 30, 25, 0, 0, 0, 0, 0), in `all` and in each `faces` entry alike. The ranges are those of section 3 (Blemish, Dark circles, Even skin, Shadow desaturation, Shine removal and Whitening 0 to 100; Texture and Skin contrast -100 to 100); a number out of range is clamped and reported like every setting (`SettingsField::{AllFaces, Faces, Protect}`), a number that is not a number is refused. `FaceSettings::to_skin_settings()` gives the `ym_retouch::SkinSettings` (`f32`, the same names), so `ym-core` depends on `ym-retouch` (no cycle: `ym-retouch` depends on no workspace crate). `protect` is a `BrushMask`, the type of a Brush row's strokes, feather, grow and refine, so the stroke rasteriser can make its plane (`docs/algorithms/06`); its numbers are clamped as a Brush row's are, with the same 100,000-point limit. The node has the ports of a corrector (`picture`, `key`), a Mask tab (`masks`), Bypass and a name; the card is numbered like a Film node's. Fields in the patch (`NodePatch`): `same_for_all`, `all` (a partial: the amounts to change), `faces` (the whole list, replaced) and `protect` (`strokes`, `strokes_append`, `strokes_keep`, `feather`, `grow`, `refine` as a Brush row's patch has them); undo is `SettingsPatch::between`; Copy takes the node whole and a paste follows the face numbers (nothing about faces is stored per photo); **Reset** of the node returns `same_for_all`, `all` and `faces` to the defaults above and leaves the name, Bypass, place, masks and **protection** alone ("Masks and protection stay", `design/skin/mockup-notes.md`), while Reset of the `protect` field alone is Clear. **`PIPELINE_VERSION` 0.12.0** (`docs/01` *pipeline_version*, the rule the Film node and the mixers followed: an older build refuses a kind it does not know; no existing graph changes by a bit, only the written version string). The wire types (`src-tauri/src/wire.rs`, uncompiled on the machine that wrote them), `bindings.ts`, `types.ts` (with the mock's rules in `fixtures/graph.ts`, a clamp in `api.mock.ts`, `newFaceRefinement` and `faceSettingsFor` in `state/settingsGraph.ts`), the JSON schema (`face_refinement_node`, `face_settings`, `protect_plane`), `ym --settings` (the core reads it; `docs/10` has the example and the help's closing paragraph names the kind) and `docs/23` section 3.4 follow. **Until S-11b the node passes its picture through unchanged**: the plan compiler runs it as a bypassed corrector (a step that keeps its picture and still gives its mask on the key output, so key links from it work), and the tests `a_face_refinement_node_passes_its_picture_through_until_s11b` (`render_nodes.rs`: the display render, the viewport frame and the linear DNG picture equal the graph without it, with the node first, last, at its defaults, bypassed, per face, with strong amounts, protection and a mask row) and `a_settings_file_with_a_face_refinement_node_is_read_and_the_photo_renders` (`ym-cli`'s `render.rs`: `ym render --settings` accepts the node) prove it. S-11b replaces that step with the retouching one, adds the node to the random-graph generator (`any_graph.rs` lists it under `NOT_GENERATED_NODES` until then) and decides what a Layer fed by the node reads. The node is in the node view only; the simple view never lists it, and the interface does not draw it before S-11c (the tab table gives it its Mask tab only).
  *As built (S-11b), the render step, the face finding and the app's API (the interface is S-11c).* **Files.** `crates/ym-core/src/pipeline/faces.rs` (`FaceModel`: the three networks and their weights, loaded on first use, or a fake; `FaceBox`, `FacePhase`, `FaceState`, `FaceEvent`; the constants of decision 13; `select_faces`), `pipeline/graph/faces.rs` (the key, the slot in the photo's graph, the worker, `Job::faces_for`, `face_source`), `pipeline/frame/face_pass.rs` (the node's regions, protection and retouch) and `film_pass.rs` (a node of the chain stopped at its input, run over the regions: the Film node's pattern), `ym-retouch` (`retouch_regions`, `retouch_skin_region`, `retouch_tones_at`, `apply_blemish_at`, `Placement`, `BlemishProposer::for_face`, `MAX_RENDER_SCALE`). **Behaviour:** the decisions of section 9, decision 13, which says what runs when, with the numbers. `ym-core` turns on `ym-retouch`'s `mediapipe` and `mock` features (the mock models are the fake model's, as `DepthModel::fake` is for Depth); `Core::open` finds the weights as it finds the depth model's, and the test hook `YELLOWMOON_TEST_FACES=fake` gives `ym` and the app the fake finder (two faces at fixed places, 24 px floor). **Plan and keys:** `step_kind` stays `Corrector` (the plan compiler runs the node as a corrector step; `node_set` gives the chain a Film-like step when the node is live and the photo has faces, and a bypassed corrector otherwise, so the key output is the node's mask in every case); `nodes_key` has the node's Bypass, `same_for_all`, `all`, the list of faces, the protection (a hash of the strokes and its Grow, Feather and Refine), the masks and the analysis' own key. `PIPELINE_VERSION` is unchanged (0.12.0): a graph without the node is the bytes it was (the golden tests pass). **Tests:** `tests/face_render.rs` (12, the fake finder and the mock models: only the faces change, bit for bit; each face takes its own settings by its number, a face past the list takes face 1's, `same_for_all` ignores the list; protection holds; an inert node, no face and a face under the floor are exact pass-throughs; the node's masks act as a corrector's; a Layer fed by the node; a viewer never waits and the worker finds the faces once, with its two events, and no slider of the node or of a corrector upstream runs a network; leaving the photo; the grade key sees every field; 1 and 4 threads and 256 px tiles give equal bytes; no models: the frame is the input, `face_state` is `models_missing` and an export refuses), `tests/face_real.rs` (the three public portraits as raw files, the real models: the faces are found and numbered, within an IoU of 0.6 of the finder on the JPEG, the 200 px floor keeps the faces that big, and an export changes nothing away from the faces; skips with a plain line without the weights or the photos, `YM_REQUIRE_MEDIAPIPE=1` makes that a failure), `tests/any_graph.rs` (the node is generated; a graph with a live node is checked to be the CPU's, refused with its reason, `views_on_cpu` true, and the CPU renderer's bytes), `ym-cli`'s `render.rs` (the settings file renders with the fake finder; without the models the photo stops with the `mask` error and the others go on) and `parity.rs` (the app's export and `ym render` write the same JPEG, 16-bit TIFF and DNG), `ym-retouch` (the part equals the whole image; a part that cuts a face or starts odd is refused; a map of one grade on another), and `src/api.mock.faces.test.ts`. **The app:** `face_state { photo_id, phase, faces, message }` is an event (sent when the worker starts, `finding`, and when it ends) and a command of the same name (the state now); the viewport is busy and finding while the worker runs; the TypeScript bindings, types, contract and mock follow (the mock finds two faces after 400 ms, `?mockFaces=ok|none|fail|missing`). `src-tauri` does not build in the environment that wrote it (no GTK): its Rust is the mirror of the depth worker's and `bindings.ts` was written by hand in specta's format; CI's macOS job compiles it and the bindings test (`YM_UPDATE_BINDINGS=1`) will say if one character differs. **Open:** the device kernels (S-13); real-portrait thresholds (S-12); a Mac's timings (M0-H4); the interface (S-11c); a Film node after a Face Refinement node whose halo cuts a face region leaves that region's part unretouched (logged at debug level), a combination no test draws.
- **S-12 · Tuning the constants.** The owner, 2026-11-11: "you can change constants for optimal result, try it." The brief's constants are starting values: tune them (blemish scale, gain clamp, threshold curve, low-frequency radii, blend exponents, detector thresholds) against the synthetic protocol of the research (lesions with known ground truth in ten skin tones, texture energy, colour shift, mole survival) and contact sheets of public portraits; each change recorded here with before and after numbers. *Done when:* the synthetic gates pass and the owner has seen the sheets.
- **S-13 · GPU kernels** for the apply of section 6.2 and the operator of section 7, with CPU parity tests, so the node runs on the device.
- **S-14 · A trained proposer (optional, Modal).** The network of section 4 trained on synthetic chromophore lesions over Open Images faces (CC BY 2.0), per the research plan: $15 planned, $20 hard cap, a ledger that refuses a run past the cap. Shipped only if it beats the classical proposer in the owner's blind A/B, and only if its speed on a Mac is acceptable (a 48-channel variant is allowed).
- **S-15 · No neck; skin grading and teeth whitening** in `ym-retouch` (the owner, 2026-11-11): remove the neck switches and the `NeckSkin` part everywhere (types, masks, mocks, pipeline, cache keys, tests); add `MouthInterior`, the grading and teeth masks of 5.6, the operators of 7.7 and 7.8 in the order of 7.9, and their tests (section 8); mocks draw a synthetic mouth with yellow teeth and red lips. *Done when:* the tests pass and the identity test still holds with every amount 0.
  *As built (S-15):* the neck is gone: `SkinSettings` lost both switches, `FacePart::NeckSkin`, `Models::neck`, `SkinParts` and the neck folding in the cache keys. New: `FacePart::MouthInterior`; `SkinSettings::{skin_contrast, shadow_desaturation, shine_removal, teeth_whitening}` with `gc()`, `gd()`, `gs()`, `tw()` and `grading_active()`; masks `grading_mask` (the `edge` of 5.5, built once and shared with the even-skin mask), `teeth_gate`, `teeth_mask`, `smoothstep`, `skin_mask`; `FaceMasks` gained `grading` and `teeth`; `face_masks` no longer takes the settings (the masks do not depend on them). Module `lowfreq`: `retouch_tones(image, full, face, settings, render_scale, &ToolMasks, protect)` runs 7.1 to 7.9 on one `T`; the pure kernels `contrast_target`, `shadow_desaturation_pixel`, `shine_removal_texture` and `teeth_pixel`; `retouch_low_frequency` stays as the dark-circles and even-skin entry. `retouch_skin` calls `retouch_tones` per face and passes `protect` to every tool (section 11 item 9). Mocks: `mock::mouth_ellipse`, `mock::paint_mouth` (red lips, yellowish teeth shaded darker toward the corners, dark gap) and `mock::mouth_colors`. 100 tests pass in the crate (85 before S-15, so 15 net new after the neck tests were replaced): grading (contrast widens by x1.31 at +100 and narrows to x0.69 at -100 on a synthetic light field with period 90 to 110 px, since the regional average `C_g` absorbs part of it; texture energy x1.0009 and x0.9995; shadow desaturation lowers chroma next to a step on the dark side by over 10 % at 100 and leaves the bright side and the far dark side within 1e-6; shine removal takes a +0.40 highlight on 0.32 skin to 0.42 at 100 and 0.57 at 50, and leaves skin at its level within 1e-6), teeth (the cast `chroma / light` of the mock teeth drops below 0.6 of its value at 100, lips, gap, corners and skin are bit-identical, no tooth ends above the brightest one, half the amount whitens less), masks (teeth mask covers the teeth band only and is invariant under a gain of 2 on the picture, so the 80th percentile adapts; the gate of the mock colours: teeth 150 or more of 255, lips and gap 0), identity with every amount 0 (also with NaN), locality over all five masks, 1 against 4 threads with every tool on, half against full path for grading (mean absolute difference under 3e-3), `proptest` for the four kernels (finite; contrast monotonic in the amount, fixed at the skin's level; shadow desaturation keeps the channel sum and lowers chroma monotonically; shine removal moves monotonically and not at all below 1.1 x `C_g`; teeth monotonic in light and in relative chroma, capped by `L_max`, identity at 0), and the pipeline test that a fully protected image is unchanged to the bit with every tool at 100 (a half-protected one is unchanged on its protected side, a 128 protection moves the pixels less). `cargo bench --no-run` builds. The numbers come from synthetic pictures; S-12 tunes the constants on real ones.
- **[HUMAN] S-H1** (approved 2026-11-11, section 11 item 9) mockups of the Face Refinement node (revised: no Include neck; a Skin grading card with Contrast, Shadow desaturation and Shine removal; a Teeth card with Whitening); **S-H2** blind A/B and the test build.

## 11. The owner's answers (2026-11-11, `docs/STATUS.md` question 45)

1. **Provenance and constants.** "Brief came from research" and "you can change constants for optimal result, try it": the brief is the owner's, from research; its constants are starting values that S-12 tunes.
2. **Models.** "yes": MediaPipe BlazeFace, Face Mesh V2 and Selfie Multiclass ship (Apache-2.0 cards; Google's own training data, as for SAM 2.1 and Depth Anything).
3. **Where the tools live.** "own node for this, not available in simple main view": the Face Refinement node (`face_refinement`), in the node graph only.
4. **Modal.** "yes": a trained proposer may be trained, $15 planned, $20 hard cap (S-14).
5. **Research-only sets.** "yes": FFHQR, Fitzpatrick17k and HAM10000 for local measurement only, never shipped or committed.
6. **Weights.** "bundle": bundled with the app like SAM 2.1 (a release asset fetched by the release workflow), not committed.
7. **On the mockups (2026-11-11):** "Don't do include neck. But let's add some things from Resolve: teeth whitening, skin grading (contrast, shadow desaturation, shine removal)." Sections 3, 5.6, 7.7 to 7.9 and task S-15.
8. **The node's name (2026-11-11):** "Node name should be Face Refinement, since it is about faces and not generally skin." The node is **Face Refinement**, kind `face_refinement`; its first panel tab is **Face**. The library stays `ym-retouch`.
9. **The mockups approved (S-H1, 2026-11-11):** "approve with your recommendations" on `design/skin/` (31fd47c): "Same for all faces" on by default when a photo has several faces; a new node retouches at once at Blemish 30, Dark circles 30, Even skin 25, Texture 0 and every grading and teeth control at 0, and Reset returns there; Protect covers every tool (blemish, dark circles, even skin, grading, teeth), not Blemish only; Skin grading and Teeth follow the per-face switch like the face card; the eye stays "Show skin areas"; the Mask tab stays; faces under 200 px across are skipped and at most 32 are kept (starting values for S-12), numbered left to right; the card label follows the app's 10-letter rule ("Face Refi…"). **Faces are found automatically** (the owner asked; no button): when the node is added and when a photo that has one is opened or pasted to, in the background, cached per photo like the depth map; until then the node shows "Finding faces…" and the photo is shown without it. Pasted per-face settings follow the face numbers; faces beyond them take face 1's settings.
13. **What the models see, and when they run (S-11b)** is decision 13 of section 9.

# 21 · Lens character

Lens character gives a photo the look of another lens: how its highlights flare, how its edges soften and darken, how the out-of-focus background is shaped, and, with an anamorphic attachment, taller oval bokeh, horizontal streaks and the anamorphic bend. It is a **Lens** card on the Look tab: a lens choice and three sliders, **Flares**, **Edges** and **Anamorphic**. This file covers what the person sees, the setting, where the lens acts in the pipeline, the four stages, the core's API, the lens data, the bokeh detector, the command line, determinism and speed, the app, and the tests.

The reasons, sources and rejected routes (generative editors among them) are in `research/lens-character-2026-10-23.md`. The owner's decisions of 2026-10-24: lenses only, no film looks; existing bokeh is reshaped and none is ever added ("modification is a better route than adding fake bokeh"; "if photo has no bokeh - it will almost unmodified, but we also do other things for lens simulation"); lens data are synthetic, computed from public lens designs; "Flares and field effects need its controls. Bokeh reshaping will increase with raising anamorphic slider"; Anamorphic also shapes the flares; a lens choice above the sliders with seven lenses including Default; Anamorphic works with every lens and brings the anamorphic distortion; the picture zooms in so no corner is empty. Later the same day the owner dropped the depth model the first version of this spec used: "Why do we need depth model? We need only bokeh detection model so we can modify it"; "Depth on its own does not mean we need to modify bokeh. And we are adding no bokeh, remember? We just modify it." The bokeh detector (section 7) finds the blur already in the photo from the photo itself.

Everything here runs in Rust, the same in the app and `ym`. The optics are classical kernels (a CPU f64 reference and a WGSL kernel each, `docs/17` §6). There is no learned model and nothing to download: the bokeh detector is classical image analysis (section 7). Nothing is fitted or traced at run time: the lens data are formulas and small tables computed offline (section 6).

**Status: specified, not built.** Lens character does not change `PIPELINE_VERSION`: with the three sliders at 0 the lens stages do not run, and every pixel is what it is today.

## 1. What the person sees

The Look tab's fourth card, under Tone: **Lens**.

| Part | What it is |
|---|---|
| Header | "Lens" and the card's Reset link |
| **Lens** row | a glasscn `Select` with the seven lenses below; the menu shows each lens's line under its name |
| **Flares** | 0 to 100, step 1, detent and Reset at 0, shown with `formats.integer` |
| **Edges** | the same |
| **Anamorphic** | the same |
| Note | a `RowNote`: the chosen lens's line, then the bokeh line when there is one (below) |

| Lens | Line |
|---|---|
| Default | "Clean and modern: even sharpness, faint glow." |
| Classic | "Warm and gentle: soft glow, amber flares, smooth falloff." |
| Vintage | "Glows wide open, lower contrast, soft edges, a little swirl." |
| Petzval | "Very sharp centre, strong swirl, fast falloff." |
| Triplet | "Soft corners, busy flares, a crisp centre." |
| Sonnar | "Smooth and glowing, with gentle edges." |
| Detuned | "Sharp only in the centre, with smeared flares." |

The names are the approved list; two of them need a check before they ship (section 12, decision 1).

**What each slider does.** The lens decides what the effects look like (the colour of its ghosts, how its edges soften, how much it swirls); the sliders set how much. 100 is the lens's full character, as its data describe it.

- **Flares**: glow around bright light, veiling glare (lower contrast near light), ghosts on the line from a light through the frame's centre, and starbursts on small lights. All of it grows with the light the photo really holds, including what the sensor clipped (section 4, L3).
- **Edges**: the softening of the edges, the darkening towards them, colour fringes, the lens's own distortion, and swirl and cat's-eye shapes in the out-of-focus background.
- **Anamorphic**: an anamorphic attachment in front of the chosen lens, as real adapters are used on spherical lenses. It makes out-of-focus highlights taller ovals, turns flares into horizontal streaks and oval ghosts (Flares sets how much), bends the picture the anamorphic way (a horizontal barrel: horizontal lines near the top and bottom bow, verticals stay nearly straight, and the left and right sides are drawn in a little), makes the edge darkening oval, softens the left and right edges more than the top and bottom, and adds horizontal colour fringes. The picture zooms in a little as it rises (a few per cent at 100; the attachment's data set it), so no corner is ever empty.

| Action | What happens |
|---|---|
| Choose a lens | `kind` changes, and Flares and Edges move to that lens's starting amounts (its data's `start`; Default's are 0 and 0). Anamorphic stays. One undo step |
| Move a slider | the picture follows, as with any slider; one undo step per drag |
| Reset (the card's link) | Default, and the three sliders at 0 |
| Before (`\`) | the camera's default picture, with no lens effect at all (`docs/02` *The Before view*) |

**The bokeh line** shows while a slider asks for bokeh reshaping:

| State | Line |
|---|---|
| little to reshape | "This photo has little out-of-focus background, so its bokeh changes little." |

Otherwise there is no bokeh line. A photo without bokeh still gets flares and edges: the owner's rule.

**Elsewhere.** Copy settings and Paste carry the lens like any setting; "Apply to" copies the look only (`docs/01` *Applying to several photos*); Re-run leaves it alone. The viewport, the filmstrip thumbnails and the look tiles show the lens; a thumbnail or look tile reshapes the bokeh only once the photo's blur map exists (it is made when the photo is shown in the viewport or exported). The automatic estimates (exposure, white lift, white balance, Apple Auto, Auto crop) are made without lens effects, so a lens slider never moves them.

## 2. The setting

| Field | Type / range | Default | UI |
|---|---|---|---|
| `lens` | `{ kind, flares 0–100, edges 0–100, anamorphic 0–100 }`; `kind` is one of `default`, `classic`, `vintage`, `petzval`, `triplet`, `sonnar`, `detuned` | `{ kind: "default", flares: 0, edges: 0, anamorphic: 0 }` | Look › Lens |

- **Rules.** As `color_warp` (`docs/01` *Settings in the core*): a document without `lens` reads as the default; numbers are clamped (`PatchReport::clamped`), a value that is not finite is `SettingsError::NotFinite("lens.<field>")`, an unknown `kind` or field is `SettingsError::Parse`; a patch may carry only the parts it changes (`{ "lens": { "flares": 40 } }`); writing writes every field, so the `ym:Settings` of every export gains `"lens":{...}` (no pixel changes).
- **No lens.** All three amounts at 0 is no lens, whatever `kind` says: no lens stage runs, and the render has the bytes it has today without the field (a test).
- `SettingsField::Lens` (`"lens"`) belongs to a new `StageGroup::Lens` between `Raw` and `Color`. It is in `SettingsField::ALL` and not in `COPY_LEAVES_OUT`; Reset gives the default; `ResolvedValues::lens` and `pinned_settings` carry it as stored. Choosing a lens in the card sends `kind` with the lens's `start` amounts in one patch; the core does not move amounts by itself.
- Wire, schema and types: `LensSettings` in `src/api/types.ts`, `$defs/lens` in `crates/ym-core/schema/settings.schema.json`, the field-name test of `docs/01`.

## 3. Where the lens acts

A new stage group, **Lens (L1 to L3)**, sits between the raw group (after R9) and the colour group (before C1), on scene-linear camera RGB balanced with `g0`: the light the emulated lens delivers to the sensor. **L4**, the anamorphic bend and the lens's distortion, is geometry, applied while sampling.

| # | Stage | Slider | Where |
|---|---|---|---|
| L1 | Bokeh reshaping | Anamorphic (taller ovals); Edges times the lens's swirl (tangential stretch, cat's-eye) | per tile, with lists made once per photo |
| L2 | Field optics: edge softness, colour fringes, falloff | Edges; Anamorphic's own terms | per tile |
| L3 | Highlight optics: glow, veiling glare, ghosts, streaks, starbursts | Flares, shaped by Anamorphic | a reduced plane and lists made once per photo, then per tile |
| L4 | Anamorphic bend, the lens's distortion, the zoom | Anamorphic; Edges | geometry (`docs/01` *Geometry*) |

Why there:

- Added light stays additive, and white balance, exposure, the sensor map, the highlight range (C7), the look and the display transform then treat flare light like the rest of the scene, as they would a real lens's.
- The clipped light is still recognisable (R5 leaves a pixel at its clip level in every channel only where every channel clipped), so it can be restored for the flares (L3).
- A colour, grade or look edit does not re-run the lens: the group's key is the raw group's key, the reduction, the lens setting and, when Anamorphic is above 0, the axis (below).
- The photo's own lens corrections (R3, R5b, R9) still run first. The emulated lens adds its character on top; the Sony lens's corrections are never switched off to get it (the research note's decision).

**Coordinates.** The optical centre is the centre of the photo's area after R9's auto crop (the sensor's centre), not of the person's crop. Every length in the lens data is in units of the half-diagonal of that area, and the field radius `ρ` is the distance from the centre over the half-diagonal, so a view reduced `k` times uses the same shapes at `1/k` of the pixels. The **anamorphic axis** is the horizontal of the canvas: the photo as the person sees it after the camera's and their own quarter turns, before straightening (section 12, decision 4). In the buffer it is the x or the y axis.

**Planes and lists, once per photo.** Some parts of the lens see the whole frame (a ghost lies across the centre from its light; glow reaches far), while tiles must equal the whole frame bit for bit (`docs/17` D5). Those parts are therefore made once per photo, on the CPU, from small whole-frame images, and shared by every tile and every scale:

- the **lens plane**: the raw group's image at `k_L`, the largest power of two (at least 2) that keeps the frame's long edge at or above 1024 px, so the plane is 1024 to 2047 px long (a 6000 px frame gives `k_L` = 4, an 8640 px frame 8; frames under 2048 px use 2, and test frames under 1024 px may use 1);
- the **tail plane**: the lens plane halved again until its long edge is at most 512 px;
- the **source list** (L3) and the **ball list** and **blur map** (L1).

Their key is the raw group's key, the lens `kind` and its data version, the camera profile (for the spectral projection, section 6) and the camera's focus point; the slider amounts are not in it, so moving a slider re-runs only the per-tile passes. Each tile then reads them at its own pixels' absolute positions.

**Regions.** A reduced view (the fit view, thumbnails, look tiles) runs the lens over its whole frame. At full resolution the lens output of a rectangle is computed from the raw group's region grown by the lens halo (`LensPlan::halo_px`, which depends on the amounts and the photo's blur map), assembled from the raw group's cached tiles as `Job::region` already does; the raw tiles keep their own cache and key, and the lens output has its own. An export tile grows its raw source rectangle by the halo the same way.

**Render graph.** `docs/01` *Render graph* gains the row "Lens (L1 to L3) | `lens`, and everything that invalidates the raw group; the orientation when Anamorphic is above 0", and its Geometry row gains L4.

## 4. The stages

Starting values below (thresholds, caps, the headroom) are set by eye on the CC0 night, backlit and shallow-focus photos of `docs/test-photo-sources.md` and kept in `ym_lens::consts` or the lens data, each with a doc comment.

### L1 · Bokeh reshaping

It reshapes the blur already in the photo and never adds blur where there is none. Blurring each out-of-focus pixel along one direction, by an amount proportional to its own blur, turns a round disc of radius `r` into an oval `r` wide and `q·r` tall, and leaves pixels in focus untouched (anisotropic defocus magnification; Bae and Durand, 2007). Narrowing a disc would need deconvolution and is not done.

1. **The blur map** (radius `b`, in half-diagonal units), on the lens plane.
   - The bokeh detector (section 7) gives the map and a confidence per pixel; nothing else goes into it.
   - With fewer than 50 measured edges, or less than 2 % of the frame at `b ≥ 2·b_min`, nothing is reshaped and the render says why (the note's "little out-of-focus background").
   - Per tile the map is upsampled with a joint bilateral filter guided by the tile's own pixels (radius 8, `docs/17` D4), so hair and fine edges follow the picture, and it is set to 0 where the tile's local gradient energy shows a sharp edge. No matting model is used (section 11).
2. **Bright balls** (the shapes a viewer reads first) are found on the lens plane where `b ≥ 2·b_min`: local peaks at least 1 EV above the median of a ring of radius `2b`, flat-topped (at least 60 % of the disc within 0.5 EV of the peak), with a radius 0.6 to 1.6 times `b`. At most 256, brightest first; of two closer than 1.2 radii the dimmer is dropped. Each records its centre, radius, energy per channel above its ring and the ring's plane fit.
3. **Take the balls out.** In each ball's disc (radius 1.15 r, soft edge) the pixels are replaced by the plane fit of its ring.
4. **Stretch.** Two one-dimensional gathers, first across the anamorphic axis (vertical on the canvas), then tangential. Each pixel `p` averages samples on a line through it of half-length `ℓ(p)`:
   - vertical: `ℓ = (q − 1)·b(p)`, where `q = 1 + (Q_max − 1)·anamorphic/100` and `Q_max = 2` (section 12, decision 3);
   - tangential: `ℓ = σ(ρ)·edges/100·b(p)`, with the lens's swirl profile `σ(ρ)` (0 at the centre);
   - a sample counts only when its blur is at least half the pixel's own, and the weights are renormalised, so a sharp subject never smears into the background and the background never spills onto it;
   - `ℓ` is 0 below `b_min` (0.004) and the blur is capped at `b_cap` (0.03), so the halo is at most `(Q_max − 1 + σ_max)·b_cap` plus the upsampling's 8 px; samples are spaced evenly, at most 65 a line, read with linear interpolation.
5. **Draw the balls back**, each with the energy it had: an ellipse `r` across and `q·r` along the vertical, turned towards the tangent by the swirl, cut by the lens's cat's-eye at its field radius (the pupil intersected with a second circle of radius `r·c(ρ)` set off from it radially; Edges scales the offset), with the lens's rim profile (`bokeh.edge`: a table of brightness against the normalised radius). Drawn analytically at each pixel from the list, so the edges are sharp at every scale and tiles agree.

Where the detector is not confident, `b` is taken as 0 and nothing is stretched.

### L2 · Field optics

With `e = edges/100`, and `e' = e + (1 − e)·κ·(q − 1)` the **edge amount**: Edges plus the attachment's own share, so that at Edges 0 the anamorphic terms below still act, alone (`e' = e` at Anamorphic 0; at `e' = 0` L2 is the identity). Every `e` below is `e'`.

- **Point spread.** Each lens has spread functions at nine field radii (`ρ` = 0, 1/8, …, 1) and eight wavelengths, 33 × 33 cells over ±`w(ρ)` (cell-centred, section 6) in a frame turned to the radial (x, outwards from the frame's centre) and tangential (y, anticlockwise on the screen) directions. The plan projects them onto the camera's three channels (section 6). A pixel gathers with the blend of its two neighbouring radii (linear in `ρ'`, held at the last table beyond 1), turned to its own direction and scaled in width to a half-width of `ω = min(e'·(1 + S)·w(ρ'), w_max)·H` pixels, `H` being the half-diagonal in pixels and `S` the side softness below. The channels' kernels have their own centroids, which is the lateral colour fringe; the anamorphic attachment multiplies the axis part of that offset by `1 + h·(q − 1)` (horizontal fringing). The CPU reference does so by reading each channel of the gathered picture that much further along the axis (linear interpolation between neighbouring pixels, the gain applied after): to first order the same as translating the kernel, for one gather instead of three. The width is capped at `w_max` = 0.012 (about 62 px on a 50 MP frame).
- **Taps.** A kernel of half-width `ω` is `n × n` point masses at its cell centres, `n = min(17, 2·⌈ω⌉ + 1)`: the 33 × 33 table resampled area-weighted and renormalised to sum 1, each mass read from the input by bilinear interpolation. A kernel below `ω` = 0.5 px is a copy of the input (the gain still applies). A kernel whose cells would be more than one texel apart at 17 taps (`ω` above about 8.5 px) reads a 2×, 4×, … box-reduced copy of the input instead, whose blocks are aligned to absolute frame coordinates so that tiles agree; the input is replicated beyond the frame edge. A kernel costs at most 289 taps; the 17 is a speed budget (`ym_lens::consts::MAX_TAPS_SIDE`) and can be raised to 33 where the cost is acceptable.
- **Falloff.** A gain `1 − e·(1 − T(ρ'))` on all three channels, with the lens's relative illumination `T` (read linearly between radii, held at `T(1)` beyond `ρ'` = 1).
- **Anamorphic.** The field radius becomes oval, `ρ' = √((α·x)² + y²)` in half-diagonal units along the axis (x) and across it (y), with `α = 1 + κ·(q − 1)`, so falloff and softness grow faster towards the left and right; the attachment adds its own side softness `S = a_s·(q − 1)·x²`, which widens the kernel (`1 + S` times). The kernel's direction is the geometric one, from the frame's centre to the pixel. At Edges 0 these terms act alone through `e'`.

Reduced views use the same kernels at `1/k` scale (a kernel below one pixel is a copy).

### L3 · Highlight optics

With `F = flares/100`:

- **Restored light** `R`: the image times `1 + f·(2^h − 1)`, `h = 3` EV (`LENS_HEADROOM_EV`), where `f` is the clip test: a ramp from 0 to 1 as `min_c(X_c / g0_c)` goes from 0.95 to 0.98 (`g0` is the clip level of each channel in the group's scale; R3's gain only raises clipped values, so they still pass; on the lens plane the ramp runs from 0.90 to 0.97, because binning mixes clipped and unclipped photosites). Only the flares read `R`; the picture's own clipped pixels are not raised.
- **Highlights** `H`: `R` weighted by a soft knee (a smoothstep) from 0 at half the clip level to 1 at the clip level of the pixel's brightest channel, so a coloured light that clipped in one channel keeps its colour in its ghosts.
- **The light budget.** The lens data give the fractions of light at Flares 100: glow `a_g`, veiling `a_v`, each ghost `a_i`, starburst `a_*`, and the attachment's streak `a_s`. With `Z` the output of L2 and `a` the sum of the fractions in use (the streak's with its Anamorphic factor) times `F`:

  `out = (1 − a)·Z + F·(a_g·K_g⊛R + a_v·K_v⊛R + Σ a_i·G_i[H] + a_*·B[H] + a_s·(q − 1)/(Q_max − 1)·S[H])`

  Every kernel and shape has unit energy, so the light the image loses comes back spread out, and the light the sensor clipped comes back too. The starburst's fraction is in use in proportion to its strength (below): `a` equals `LensPlan::light_fraction()` at full strength and is smaller under it, and the light a weak starburst does not take stays in the picture. A tint (a ghost's coating, the streak's) is the reflectance spectrum projected on the camera's channels and scaled to a mean of 1 over the channels, so it colours the light and does not change its energy. The light a ghost's clipping circle blocks is lost.
- **Glow** `K_g`: a core within 8 px (and at most 0.0015 half-diagonals), gathered per tile at full resolution, plus the rest, which is smooth and is convolved with the tail plane by an FFT. The tail falls as a power law and widens with shorter wavelengths, per the lens data; the camera's channel weights mix the eight wavelength kernels (each of unit energy) into one kernel per channel. A cosine window (1 up to 70 % of the core radius, 0 at it) splits the kernel into the core, `K_c s`, and the tail, `K_c (1 − s)`; the core radius is the smallest of `flare.glow.core_radius`, 0.0015 and 8 px of the full-resolution frame. Both parts are normalised on the grid they are used on: the core's taps at the tile's scale sum to `a_g·m_c` (`m_c` is the core's share of the kernel's energy), the tail kernel on the tail plane sums to `a_g·(1 − m_c)`, so they add up to exactly `a_g` at every scale. The tail plane is read at each pixel's position with a Catmull-Rom curve, held at 0 or above. Detuned's glow and starburst come from a pupil with a seeded random phase (computed offline, so it is data), which gives its smeared flares.
- **Veiling** `K_v`: a very wide, nearly flat kernel on the tail plane (a flat disc of `flare.veil.radius` with a cosine edge over its outer 40 %, the same in every channel); the contrast the lens loses near light. It is folded into the tail plane with the glow's tail, both already weighted by their fractions.
- **The source list**: the connected components of the pixels of the lens plane whose `H` reaches 0.02 clip levels in some channel (8-connected, in raster order), each larger than 0.02 half-diagonals split into cells of that size on a grid fixed to the plane, each with its energy per channel (`H` summed, times the area of a plane pixel in half-diagonals squared), centroid and second moments (with the footprint of a plane pixel) and its extent, the larger side of its bounding box; at most 256, by energy, ties by position.
- **Ghosts** `G_i`: per source and ghost, the lens's aperture shape (blade count, roundness, rim profile) of apothem `s_i` (the radius of its inscribed circle; `roundness` blends the polygon with that circle), centred at `c + m_i·(p_s − c)` on the line from the source `p_s` through the centre `c`, cut by a circle of radius `R_i` centred at `c + n_i·(p_s − c)` (the arcs of ghosts from lights near the edge), tinted by the ghost's coating (section 6). The ghost's density is the rim profile times a Gaussian step at the edge (the steps of the shape and of the clipping circle have a standard deviation of `√((soft·s)² + the source's variance + (1 px)²)`: the ghost is widened by the source's extent, and an edge is never sharper than a pixel, which keeps the drawn energy exact on any pixel grid), divided by its exact integral. With Anamorphic, ghosts widen along the axis by `1 + κ_g·(q − 1)` and the attachment adds its own oval ghosts (round ellipses, widened the same way, at the Anamorphic factor of their fractions).
- **Starbursts** `B`: for compact sources (extent at most 0.004), the lens's spike table (its blade count, `|F{pupil}|²` over wavelengths, radius at most 0.08, unrotated in the canvas's orientation), stronger as the photo's f-number grows: the strength is `clamp((N − 2)/(f_full − 2), 0, 1)` for the f-number `N` (EXIF; f/4 when it has none). A pixel receives the exact mass of the table's piecewise-linear reconstruction over its area.
- **Streaks** `S` (Anamorphic above 0 only): for every source, a band along the axis through its centre, as tall as the source (a Gaussian across the axis with the source's standard deviation across it), with the attachment's profile `(1 + |a|/core)^−power` along the axis (a sharp core and a long tail, cut by a cosine over its last fifth at the reach `reach·(q − 1)/(Q_max − 1)`, that is one half-diagonal at Anamorphic 100) and its tint. A pixel receives the exact mass of the band over its area.
- Ghosts, starbursts and streaks are drawn analytically per pixel from the list; each tile gets, from the plan, the items whose bounds overlap it, in list order.
- **Energy on a pixel grid.** The glow's core taps and tail kernel are normalised on the grid they are used on, and the streak and the starburst give each pixel the exact mass over its area, so these parts return exactly their light at every scale (to the rounding of `f32`). A ghost is a smooth density sampled at the pixel centres: its raster sums to its light within 1e-7 for a round aperture with the lens's rim and edges up to a tenth of the radius (1e-11 with a flat rim), and within about 1e-5 for a polygon aperture, whose corners are creases of the gauge. Light that leaves the frame, and the part of a ghost its clipping circle blocks, is lost. The tail plane is made once per photo whatever Flares says (the planes' key leaves the amounts out), about 0.3 s for a 6000 × 4000 frame (lens plane 1500 × 1000) on a loaded 4-core machine.

### L4 · Anamorphic bend, distortion and zoom

The map from the photo to the canvas, in half-diagonal units with x along the anamorphic axis:

- the attachment: `x' = x·(1 + μ·x²)`, `y' = y·(1 − β·x²)`, with `μ = μ₀·(q − 1)` (the horizontal barrel; the traced attachment gives `μ₀ < 0`, the sides drawn in) and `β = β₀·(q − 1)` (horizontal lines bowing);
- the lens's own radial distortion `r' = r·(1 + e·(δ₁r² + δ₂r⁴))`.

The lens's distortion applies first and the attachment's bend after it. In half-diagonal units of the canvas, centred on its centre, the canvas position of a photo position `p` is `z·F(p)` (`F` the two steps and `z` the zoom below), and a canvas position `c` reads the photo at `F⁻¹(c / z)` (`LensWarp::photo_to_canvas` and `canvas_to_photo`). A reduced view takes the warp of the full-resolution canvas (`LensWarp::rescaled`) so every scale shares one zoom.

Sampling needs the inverse, found by three Newton steps from the output position (f64 in the CPU reference; on the device with the per-tile rebase of D4). The **zoom** `z` is the smallest factor of at least 1 for which every point of the canvas edge (200 samples a side), divided by `z`, maps inside the photo with a 1 px margin, so the canvas has no empty corner. The map sits in the geometry sampler between the canvas and the photo (`pipeline/geometry.rs`): straightening and the crop act on the bent, zoomed canvas, so the crop frame and the panel work on the bent picture.

- At full resolution the sampling is then Catmull-Rom everywhere, as for a straightened photo.
- A tile's source rectangle is the bounding box of its mapped edge (64 samples a side), plus the filter's support and 2 px.
- `pick_point_at` and `white_pick` go through the map.
- Auto crop analyses the picture without lens effects (`docs/20` §4) and its frame is applied to the canvas as it is. The bend is mild and the canvas is full, so the frame holds nearly the same content and "No border in photos" still holds.
- The Before view has no bend.
- A DNG carries the bend in its pixels: `Geometry::upright` gets it in the buffer's own orientation, with the axis the canvas gives.

## 5. The core's API

New crate **`ym-lens`** (data, plans, planes and the CPU references; it depends on `ym-color` for the spectral projection and not on `ym-gpu`, and works on planar `f32` slices with their place in the frame). Kernels in `ym-gpu`, which depends on `ym-lens` for the plans (`kernels/lens_reshape.wgsl`, `lens_field.wgsl`, `lens_flare.wgsl`; L4 in the sampler).

```rust
pub struct LensLibrary { /* data/lenses, loaded and checked once */ }
impl LensLibrary {
    pub fn load(data_root: &Path) -> Result<Self, LensError>;
    pub fn lenses(&self) -> &[LensInfo];          // id, name, line, start, provenance
}
pub struct LensPlan { /* one photo's lens: kernels projected on its camera, axis, zoom */ }
impl LensPlan {
    pub fn new(lib: &LensLibrary, settings: &LensSettings, frame: FrameInfo,
               camera: &SpectralCurves, f_number: Option<f64>) -> Result<Self, LensError>;
    pub fn is_identity(&self) -> bool;            // all three amounts at 0
}
pub struct LensPlanes { /* lens plane, tail plane FFT, source list, blur map, ball list */ }
impl LensPlanes {
    pub fn build(plan: &LensPlan, lens_plane: &RgbImage, clipped: &Plane,
                 focus: Option<(f64, f64)>, cancel: &CancelToken) -> Result<Self, LensError>;
}
pub mod cpu { /* reshape, field, flare: run on any rectangle of the frame; warp: Lens4Map */ }
```

`ym-core` keeps the plan and the planes in the photo's graph (*Render graph*; in memory only, hard rule 4). `Session::lens_library()` backs the IPC command of section 10.

## 6. The lens data

`data/lenses/`, committed, about 6 MB in all:

- `<id>/lens.json`: `id`, `version`, `name`, `line`, `start` `{ flares, edges }`, `provenance`, `field` (the radii, `T(ρ)`, the distortion `δ₁, δ₂`, the swirl `σ(ρ)`, the cat's-eye `c(ρ)`), `flare` (glow and veiling fractions and tails, the ghosts `{ m, s, n, R, a, coating }`, the starburst's blade count, rotation and strength), `bokeh.rim`.
- `<id>/tables.safetensors`: the spread functions (9 radii × 8 wavelengths × 33 × 33) and the spike table, f32 (the glow's tail per wavelength is a formula, `flare.glow` with its `chroma`).
- `anamorphic.json`: the attachment (`Q_max`, `μ₀`, `β₀`, `κ`, `κ_g`, `a_s`, `h`, the streak's profile, fraction and tint, its oval ghosts).
- `SHA256SUMS` and `README.md` (what each file holds, the statement below, the designs).

### File format

All lengths are in half-diagonal units of the emulated frame (a 36 × 24 mm frame, half-diagonal 21.633 mm, at the design's focal length); spectra on the 31-sample grid 400, 410, …, 700 nm; the eight PSF wavelengths are 420, 460, 500, 540, 580, 620, 660 and 700 nm.

`index.json`: `{ "version": 1, "lenses": ["default", "classic", "vintage", "petzval", "triplet", "sonnar", "detuned"], "attachment": "anamorphic.json" }` (the menu order).

`<id>/lens.json`:

| Field | Meaning |
|---|---|
| `id`, `version`, `name`, `line` | as in section 1; `version` is an integer, raised when the data change |
| `start` | `{ "flares": 0–100, "edges": 0–100 }` |
| `provenance` | `statement` (exactly the sentence above), `design` `{ type, source, year, focal_mm, f_number, format }`, `coatings`, `generated` (a date), `method` (one sentence) |
| `field.radii` | 9 field radii, 0 to 1 |
| `field.psf_half_width` | `w(ρ)` per radius: the PSF table spans `±w` |
| `field.relative_illumination` | `T(ρ)` per radius, `T(0) = 1`, read linearly between radii |
| `field.distortion` | `{ "k1": δ₁, "k2": δ₂ }` |
| `field.swirl` | `σ(ρ)` per radius |
| `field.cats_eye` | `c(ρ)` per radius: the offset of the second circle in ball radii (0: none) |
| `bokeh.rim` | 16 samples of a ball's brightness against its normalised radius (0 the centre, 1 the edge), mean 1 |
| `flare.glow` | `{ "fraction", "core_radius", "r0", "power", "reach", "chroma": [8] }`: the core is a disc of `core_radius` with a soft edge; the tail falls as `(1 + r/r0)^−power` out to `reach`; `chroma` scales `r0` per PSF wavelength |
| `flare.veil` | `{ "fraction", "radius" }` |
| `flare.aperture` | `{ "blades": 0 for round or a count, "rotation_deg", "roundness": 0–1 }` |
| `flare.ghosts` | a list of `{ "m", "n", "s", "clip_radius", "fraction", "soft", "coating": [31] }` (section 4, L3; `soft` is the edge's width in ghost radii) |
| `flare.starburst` | `{ "fraction", "radius", "f_number_full": the f-number at which it reaches its full strength }` |

`<id>/tables.safetensors` (f32, little-endian, each slice summing to 1):

| Tensor | Shape | Meaning |
|---|---|---|
| `psf` | `[9, 8, 33, 33]` | per radius and wavelength, samples on a 33 × 33 grid over `±w(ρ)`, row-major (index `y·33 + x`) with x radial (outwards) and y tangential (anticlockwise on the screen) |
| `spikes` | `[8, 129, 129]` | per wavelength, the starburst over `±flare.starburst.radius`, row-major with x to the right and y down |

**Grid of the tables.** A table is a set of cell masses: sample `k` of `n` (33 for `psf`, 129 for `spikes`) is the mass of the cell `[−1 + 2k/n, −1 + 2(k + 1)/n]` in units of the half-width, so the table spans exactly `±w` (or `±radius`), the middle sample is centred on the origin and a slice sums to 1 (1e-4). A table written with samples *at* `−w … +w` would be 3 % too wide.

**Checks.** The loader refuses a lens folder (`LensError`, never a panic) whose `SHA256SUMS` (`sha256sum` format, `<hex>  <relative path>`, one line for every file it reads and for `README.md`) does not match, whose `index.json` does not list all seven lenses once each, whose statement is not the exact sentence, whose radii do not run strictly increasing from 0 to 1, whose `T` is not 1 at the centre, in (0, 1] and falling, whose `σ(0)` is not 0, whose numbers are not finite, whose lists or tensors have another length or shape or hold a negative sample, or whose fractions of light at Flares 100 and Anamorphic 100 add up to 1 or more. `bokeh.rim` is the key of the file format; the stage text's `bokeh.edge` is read as the same table.

`anamorphic.json`: `{ "id": "anamorphic", "version", "provenance", "q_max": 2.0, "bend": { "mu0", "beta0" }, "field": { "kappa", "side_softness", "fringe" }, "ghost_stretch": κ_g, "streak": { "fraction", "reach", "core", "power", "tint": [31] }, "ghosts": [ … as above ] }`.

**Colour.** Coatings and tints are reflectance spectra (31 samples, 400 to 700 nm), and the spread functions and tails are given per wavelength. The plan projects them onto the photo's camera channels with the profile's spectral curves (`data/cameras/sony/`, `docs/03`), so a blue streak is the blue that body's sensor would record. Stand-in bodies use their profile's curves.

**Provenance.** Every file says, in `provenance.statement`: **"Synthetic, computed from public lens designs, not measured."** The designs are prescriptions Yellowmoon transcribes itself from expired patents or nineteenth-century publications; each lens names its source (patent number or publication, year, focal length, f-number) and its design type in generic words (double Gauss, triplet, Petzval portrait lens). Lens designs never come from third-party lens libraries. Coatings are assumed (single-layer MgF₂ quarter-wave films, multi-layer for Default), tuned by eye and documented as an artistic choice. Default is a well-corrected double Gauss with multi-layer coatings; Detuned is Default with its elements decentred and tilted.

**How they are made.** The data are computed offline by a lens simulator that is not part of the repository or the app, and are fitted into the formulas and tables above; only those are committed. They cannot be regenerated from the repository alone, and `README.md` says so. Tests check what can be checked from the data alone (section 14).

**Licences.** `docs/14` gains a row: lens prescriptions transcribed from expired patents and old publications (facts of a design; provenance per lens; [HUMAN] check, section 13).

## 7. The bokeh detector

Reshaping needs to know where the photo is already out of focus and by how much, not how far away things are: a far wall in a photo taken at f/11 is sharp and must stay as it is. The detector therefore measures the blur the Sony lens left in the photo, from the photo alone, once per photo on the lens plane (`LensPlanes::build`), deterministically in f64 on the CPU:

1. **Blur at edges.** The re-blur gradient ratio of Zhuo and Sim (2011, implemented from the paper): the plane's luminance is blurred again with a Gaussian of 1 px, and at each strong edge (local maxima of the gradient magnitude above a threshold set from the plane's noise) the ratio `ρ_g` of the gradient magnitudes gives the blur `σ = 1 / √(ρ_g² − 1)` px, a sparse map with a confidence from the edge's contrast.
2. **The subject anchor.** The camera's focus point (`SonyFocus`, `docs/04` §Container, as a fraction of the frame in the buffer's orientation) marks its area as in focus with high confidence; without one, nothing is anchored.
3. **Spread to every pixel.** The sparse values and confidences are spread over the plane by edge-aware normalised convolution (a joint bilateral filter guided by the plane's colours, run over a fixed pyramid), so a value travels along its own surface and stops at colour edges. The result is `b` (in half-diagonal units) and a confidence `c` per pixel.
4. **Bright balls are measured by their size, not their edges.** A defocused light is a disc with a sharp rim, so the edge measure of step 1 reads it as sharp (`research/bokeh-map-2026-10-24.md` §1). Bright, flat-topped blobs are therefore found directly (a scale-space blob search with a disc template) and their measured radius is written into the map over the disc and its ring, with high confidence; on a dark night background they are often the only out-of-focus evidence there is.
5. **Caution.** `b` is set to 0 where `c` is low, below `b_min`, and where the plane shows sharp texture; balls (section 4, L1 step 2) are looked for only where `b ≥ 2·b_min`.

The report says "little out-of-focus background" when less than 2 % of the frame has `b ≥ 2·b_min` with confidence. Its tests use synthetic scenes with a known blur per region (section 14). The owner allows a learned model here ("Bokeh map extraction can use ML if needed, of course"). No pretrained model qualifies: every defocus-map and blur-detection model found has a copyleft or missing code licence, or weights trained on data with no grant or a non-commercial one (`research/bokeh-map-2026-10-24.md`). The learned route is therefore a small model of our own (about 2 M parameters, a crop-net-style encoder with a small decoder, input the lens plane, output log blur radius and a confidence), trained on synthetic defocus with a known blur made from licence-clean sharp photos (Open Images, CC BY 2.0, with its CC BY 4.0 masks; DIODE, MIT), bundled and run in Rust like the crop model (section 11). The classical detector above is the first version, the fallback, and the baseline the model must beat. Research-only blur sets (CUHK, DUT-DBD, RTF) may measure both locally (the owner, 2026-10-24); they live only in the git-ignored `testdata/`, are never committed and never trained on.

## 8. The command line

`ym render` gains:

| Option | Values | Effect |
|---|---|---|
| `--lens <name>` | `default`, `classic`, `vintage`, `petzval`, `triplet`, `sonnar`, `detuned` | the lens; Flares and Edges take its starting amounts unless `--flares` or `--edges` is given, as in the card |
| `--flares <0–100>` | | Flares |
| `--edges <0–100>` | | Edges |
| `--anamorphic <0–100>` | | Anamorphic |

They are the `lens` field of the settings object; `--settings` takes it too. New commands: `ym lenses list [--json]` (id, name, line, start amounts, design source).

## 9. Determinism and speed

- The planes and lists are made on the CPU in f64 with a fixed order (lists sorted by energy, ties by position), the FFT with `rustfft`'s scalar planner (MIT OR Apache-2.0) and `libm` for transcendental functions, so they are the same bits on every platform.
- Every per-tile pass is a function of the planes, the lists, the tile's own pixels and absolute coordinates: tiles equal the whole frame bit for bit at 512 and 1024 px (D5), with the halos of section 4 (L1's stretch, L2's `w_max`, L3's 8 px core).
- Each stage has a CPU f64 reference and a WGSL kernel, parity within 1e-3 (the f16 storage gate of `docs/05`), and holds under Metal fast-math (K8). The app and `ym` export the same bytes (rule 6).
- **Targets** (Apple M1, 50 MP, Metal): the planes in at most 0.5 s after a change of lens or photo (off the foreground worker; the fit view shows the lens when they are ready); a slider move in the fit view in at most 100 ms once they exist; a 100 % tile at most twice the cost of the raw tile; an export at most 4 s longer. CI on Linux measures the CPU and lavapipe as a guard against regressions only; the Mac numbers are the owner's check (section 13).

## 10. The app

- IPC: `lens_list` → `LensInfo[]` (read once). The render report carries `lens: { reshaped, reason }`, which the note reads ("little out-of-focus background").
- The browser build's mock serves `lens_list` and the report.
- `docs/11` gets the card under *Look tab* and the tab table's Look row.

## 11. Not in this version

| Step | Why later |
|---|---|
| Adding bokeh to a photo that has none | the owner's direction: reshape what is there, never add |
| A learned bokeh detector of our own (a small defocus model trained on synthetic defocus with a known blur, bundled) | allowed by the owner (section 7); after the licence check of pretrained candidates, and training needs GPU time and the owner's yes before any Modal money is spent |
| A matting model for hair | the guided upsampling and the sharpness test first; added if tests show halos at hair |
| A sharp area away from the centre (Detuned's sweet spot moved) | needs a control |
| A lens with a front effect plate (abstract smeared flares) | Detuned's smear covers a mild form |
| Narrowing bokeh or giving it an exact new shape | needs deconvolution |
| Breathing and focus pulls | they do not exist in a still |
| Replicas of named lenses | no public prescriptions or coating data for them |
| Film looks, grain, halation | out of scope (the owner: lenses only) |
| A generative layer | fails the pipeline's constraints (research note); the owner, 2026-10-24: "No need for generative model bakeoff" |
| A 2.39:1 crop aspect | a separate request (section 12, decision 5) |

## 12. Decisions for the owner

The spec uses the first option of each until the owner says otherwise.

1. **Two names.** "Sonnar" is a Zeiss trademark and "Petzval" a person's name that a lens maker also sells under; the research note's rule was descriptive names only. Keep both after a [HUMAN] check, or rename them (for example **Glow** and **Swirl**)? The ids (`--lens sonnar`) change with the names, so this is decided before LC-03.
2. **"Edges"** as the label of the field-effects slider.
3. **Anamorphic 100 is a 2× squeeze**: bokeh twice as tall as it is wide.
4. **The anamorphic axis follows the picture as seen**, so a portrait photo gets horizontal streaks; a real attachment would turn with the camera and give vertical ones.
5. **No 2.39:1 crop aspect** in this work; the Crop panel keeps its five.
6. **Choosing a lens moves Flares and Edges** to its starting amounts (card and `--lens`).

## 13. Tasks

Task format of `docs/15`. Each task updates the specs it touches and `docs/STATUS.md`.

**LC-01 · Lens library** (`ym-lens`): the data format, loading, checks and `SHA256SUMS`, the spectral projection. *Done when:* the property tests of section 14 pass on synthetic lens files.
**LC-02 · Lens data** (LC-01): the seven lenses and the attachment, with provenance and `README.md`. *Done when:* every file passes LC-01's checks; contact sheets of the CC0 test photos at 0, 50 and 100 for each slider and lens are in `docs/images/lens/`.
**LC-03 · The setting** (decision 1): `lens` in `settings.rs`, the schema, `types.ts`, patches, Copy and Paste, `StageGroup::Lens`, the render graph's keys. *Done when:* the settings tests and the no-lens identity test pass.
**LC-04 · L4** (LC-03): the map, the zoom, the sampler on the CPU and the device, pick, tile rectangles, `Geometry::upright`. *Done when:* the geometry tests of section 14 pass.
**LC-05 · L2** (LC-01, LC-03): CPU reference, kernel, parity, tiles. *Done when:* its tests pass.
**LC-06 · L3** (LC-01, LC-03): restored light, planes, FFT, source list, the drawn items; CPU reference, kernels, parity, tiles. *Done when:* its tests pass.
**LC-07 · Bokeh detector** (LC-01): the blur map and the ball list of section 7 and L1 step 2. *Done when:* its tests pass.
**LC-08 · L1** (LC-07): blur map, balls, stretch, redraw; CPU reference, kernels, parity, tiles. *Done when:* its tests pass.
**LC-09 · The card** (LC-03): Lens card, IPC, the note, the mock, e2e, screenshots. *Done when:* its tests pass and the screenshots are reviewed.
**LC-10 · The command line** (LC-03): the options, `ym lenses`, `ym models`, `docs/10`. *Done when:* its tests pass.

**[HUMAN] LC-H1** Check the transcribed prescriptions and the two names (decision 1). **LC-H2** Look at each lens on your own photos and say which ship. **LC-H3** The Mac speed targets (section 9), with `ym doctor --bench`.

## 14. Tests

- `ym-lens`: every shipped and synthetic lens file loads, matches `SHA256SUMS` and has `provenance.statement` exactly; spread functions, tails, ghosts, spikes and streak profiles have unit energy (1e-9) and no negative sample; `T(ρ)` is 1 at the centre and falls; the projection of a flat spectrum is neutral for every profile; a damaged file is refused, never a panic.
- **Identity:** all three amounts at 0, for every lens, gives the bytes of a render without the field (CPU and device, viewport, 100 % and export), and creates no lens pass (`RenderReport::runs`).
- **Energy (L3):** on synthetic frames whose flares stay inside the frame, the sum of `out` is `(1 − a)·ΣZ` plus each part's fraction times `F` times the light it spreads (`ΣR` for glow and veiling, `ΣH` for the rest), to 1e-6; a clipped point source gets its 3 EV of headroom in the flare and not in its own pixels; a ghost's centre lies on the line from its source through the centre (0.5 px), on the other side for negative `m`; with Anamorphic at 0 there is no streak, and at 100 the streak lies along the axis, also on a turned canvas.
- **L1:** a synthetic scene with a sharp subject before a background of known disc blur (drawn analytically): background discs come out `q` times taller within 5 %, their energy kept within 1 %; the subject's pixels unchanged to 1e-6; a scene in focus everywhere is unchanged and reports "little out-of-focus background".
- **L2:** a flat field gets exactly the falloff gain; a point grid gets each lens's spread at its field radius; Edges 0 with Anamorphic 0 is the identity.
- **L4:** no output pixel of the canvas lacks a source, at full resolution and reduced 3, 8 and 16 times, for every Anamorphic amount, four turns and three straighten angles; the inverse map's round trip within 1e-9 (f64); a straight horizontal line near the top bows and a vertical stays within 0.5 px of straight; `pick_point_at` round trip.
- **Tiles and parity:** each stage's tiles at 512 and 1024 equal the whole frame bit for bit on the device; CPU and device within 1e-3; the poison and interleave tests of D7; the far-corner test of D4.
- **Settings:** reading with and without `lens`, partial patches, clamping, an unknown kind, Copy and Paste, Reset, the schema.
- **CLI and parity:** `ym render --lens vintage --flares 60 --anamorphic 50` equals the same settings through `--settings` and the app's export byte for byte; `ym lenses list --json` equals `lens_list`.
- **Bokeh detector:** synthetic scenes drawn analytically (textured regions blurred by discs of known radii, a sharp subject with fine strands against a blurred background, an evenly sharp scene, flat regions, noise at several ISO-like levels): isolated bright discs of known radius on a dark noisy ground measured within 15 %; `b` within 25 % of the truth where the detector is confident, below `b_min` on every sharp region and strand, no reshaping on the sharp scene; the focus point anchors the subject; the same bits in two runs.
- **Frontend:** the card (the menu and its lines, a lens choice sends the start amounts in one patch and is one undo step, sliders, Reset, the bokeh lines), the mock, e2e `lens.spec.ts`.
- **Goldens:** the golden set is unchanged (no lens); a lens golden per stage on synthetic scenes joins it.

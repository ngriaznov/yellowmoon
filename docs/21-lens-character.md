# 21 · Lens character

Lens character gives a photo the look of another lens: how its highlights flare, how its edges soften and darken, how the out-of-focus background is shaped, and, with an anamorphic attachment, taller oval bokeh, horizontal streaks and the anamorphic bend. It is a **Lens** card on the Look tab: a lens choice and three sliders, **Flares**, **Edges** and **Anamorphic**. This file covers what the person sees, the setting, where the lens acts in the pipeline, the four stages, the core's API, the lens data, the depth model, the command line, determinism and speed, the app, and the tests.

The reasons, sources and rejected routes (generative editors among them) are in `research/lens-character-2026-10-23.md`. The owner's decisions of 2026-10-24: lenses only, no film looks; the depth model is downloaded on first use; existing bokeh is reshaped and none is ever added ("modification is a better route than adding fake bokeh"; "if photo has no bokeh - it will almost unmodified, but we also do other things for lens simulation"); lens data are synthetic, computed from public lens designs; "Flares and field effects need its controls. Bokeh reshaping will increase with raising anamorphic slider"; Anamorphic also shapes the flares; a lens choice above the sliders with seven lenses including Default; Anamorphic works with every lens and brings the anamorphic distortion and edge stretch; the picture zooms in so no corner is empty.

Everything here runs in Rust, the same in the app and `ym`. The optics are classical kernels (a CPU f64 reference and a WGSL kernel each, `docs/17` §6). Machine learning has one job, depth for the bokeh reshaping, and runs on the CPU in Rust (section 7). Nothing is fitted or traced at run time: the lens data are formulas and small tables computed offline (section 6).

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
- **Anamorphic**: an anamorphic attachment in front of the chosen lens, as real adapters are used on spherical lenses. It makes out-of-focus highlights taller ovals, turns flares into horizontal streaks and oval ghosts (Flares sets how much), bends the picture the anamorphic way (horizontal lines near the top and bottom bow, verticals stay nearly straight, things near the left and right edges look slightly wider), makes the edge darkening oval, softens the left and right edges more than the top and bottom, and adds horizontal colour fringes. The picture zooms in a little as it rises (a few per cent at 100; the attachment's data set it), so no corner is ever empty.

| Action | What happens |
|---|---|
| Choose a lens | `kind` changes, and Flares and Edges move to that lens's starting amounts (its data's `start`; Default's are 0 and 0). Anamorphic stays. One undo step |
| Move a slider | the picture follows, as with any slider; one undo step per drag |
| Anamorphic above 0, or Edges above 0 on a lens that swirls, while the depth model is not installed | the download starts (section 7). Flares, Edges and the anamorphic bend show at once; the bokeh reshapes when the model is ready |
| Reset (the card's link) | Default, and the three sliders at 0 |
| Before (`\`) | the camera's default picture, with no lens effect at all (`docs/02` *The Before view*) |

**The bokeh line** shows while a slider asks for bokeh reshaping:

| State | Line |
|---|---|
| downloading | "Downloading the depth model for bokeh (50 MB)… 42%" |
| the download failed | "Couldn't download the depth model, so bokeh is left as it is." and a **Try again** link |
| reading depth | "Reading depth…" |
| little to reshape | "This photo has little out-of-focus background, so its bokeh changes little." |

Otherwise there is no bokeh line. A photo without bokeh still gets flares and edges: the owner's rule.

**Elsewhere.** Copy settings and Paste carry the lens like any setting; "Apply to" copies the look only (`docs/01` *Applying to several photos*); Re-run leaves it alone. The viewport, the filmstrip thumbnails and the look tiles show the lens; a thumbnail or look tile leaves the bokeh as it is until the photo's depth exists (it is made when the photo is shown in the viewport or exported). The automatic estimates (exposure, white lift, white balance, Apple Auto, Auto crop) are made without lens effects, so a lens slider never moves them.

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

- the **lens plane**: the raw group's image at `k_L`, the smallest power of two that gives a long edge of at most 1024 px;
- the **tail plane**: the lens plane halved again until its long edge is at most 512 px;
- the **source list** (L3) and the **ball list** and **blur map** (L1).

Their key is the raw group's key, the lens `kind` and its data version, the camera profile (for the spectral projection, section 6) and the depth model's version; the slider amounts are not in it, so moving a slider re-runs only the per-tile passes. Each tile then reads them at its own pixels' absolute positions.

**Regions.** A reduced view (the fit view, thumbnails, look tiles) runs the lens over its whole frame. At full resolution the lens output of a rectangle is computed from the raw group's region grown by the lens halo (`LensPlan::halo_px`, which depends on the amounts and the photo's blur map), assembled from the raw group's cached tiles as `Job::region` already does; the raw tiles keep their own cache and key, and the lens output has its own. An export tile grows its raw source rectangle by the halo the same way.

**Render graph.** `docs/01` *Render graph* gains the row "Lens (L1 to L3) | `lens`, and everything that invalidates the raw group; the orientation when Anamorphic is above 0", and its Geometry row gains L4.

## 4. The stages

Starting values below (thresholds, caps, the headroom) are set by eye on the CC0 night, backlit and shallow-focus photos of `docs/test-photo-sources.md` and kept in `ym_lens::consts` or the lens data, each with a doc comment.

### L1 · Bokeh reshaping

It reshapes the blur already in the photo and never adds blur where there is none. Blurring each out-of-focus pixel along one direction, by an amount proportional to its own blur, turns a round disc of radius `r` into an oval `r` wide and `q·r` tall, and leaves pixels in focus untouched (anisotropic defocus magnification; Bae and Durand, 2007). Narrowing a disc would need deconvolution and is not done.

1. **The blur map** (radius `b`, in half-diagonal units), on the lens plane.
   - The thin lens gives a blur radius proportional to the difference of inverse depths, `b = s·|D − D_f|`. The depth model (section 7) gives `D` up to an unknown scale and offset, the camera's focus point gives `D_f` (`SonyFocus`, `docs/04` §Container; without one, the `D` with the most sharp edges), and the scale `s` comes from the photo's own edges: the re-blur gradient ratio of Zhuo and Sim (2011, implemented from the paper) measures the blur at the plane's strong edges, and `s` is the median of `b_edge / |D − D_f|` over the edges where `b_edge` is at least `b_min`.
   - The sign of `D − D_f` says whether a pixel is in front of the focus or behind it.
   - With fewer than 50 measured edges, or less than 2 % of the frame at `b ≥ 2·b_min`, nothing is reshaped and the render says why (the note's "little out-of-focus background").
   - Per tile the map is upsampled with a joint bilateral filter guided by the tile's own pixels (radius 8, `docs/17` D4), so hair and fine edges follow the picture, and it is set to 0 where the tile's local gradient energy shows a sharp edge. No matting model is used (section 11).
2. **Bright balls** (the shapes a viewer reads first) are found on the lens plane where `b ≥ 2·b_min`: local peaks at least 1 EV above the median of a ring of radius `2b`, flat-topped (at least 60 % of the disc within 0.5 EV of the peak), with a radius 0.6 to 1.6 times `b`. At most 256, brightest first; of two closer than 1.2 radii the dimmer is dropped. Each records its centre, radius, energy per channel above its ring and the ring's plane fit.
3. **Take the balls out.** In each ball's disc (radius 1.15 r, soft edge) the pixels are replaced by the plane fit of its ring.
4. **Stretch.** Two one-dimensional gathers, first across the anamorphic axis (vertical on the canvas), then tangential. Each pixel `p` averages samples on a line through it of half-length `ℓ(p)`:
   - vertical: `ℓ = (q − 1)·b(p)`, where `q = 1 + (Q_max − 1)·anamorphic/100` and `Q_max = 2` (section 12, decision 3);
   - tangential: `ℓ = σ(ρ)·edges/100·b(p)`, with the lens's swirl profile `σ(ρ)` (0 at the centre);
   - a sample counts only when its blur is at least half the pixel's own and on the same side of the focus, and the weights are renormalised, so a sharp subject never smears into the background and the background never spills onto it;
   - `ℓ` is 0 below `b_min` (0.004) and the blur is capped at `b_cap` (0.03), so the halo is at most `(Q_max − 1 + σ_max)·b_cap` plus the upsampling's 8 px; samples are spaced evenly, at most 65 a line, read with linear interpolation.
5. **Draw the balls back**, each with the energy it had: an ellipse `r` across and `q·r` along the vertical, turned towards the tangent by the swirl, cut by the lens's cat's-eye at its field radius (the pupil intersected with a second circle of radius `r·c(ρ)` set off from it radially; Edges scales the offset), with the lens's rim profile (`bokeh.edge`: a table of brightness against the normalised radius). Drawn analytically at each pixel from the list, so the edges are sharp at every scale and tiles agree.

Without the depth model L1 is skipped, and L2 to L4 still run.

### L2 · Field optics

With `e = edges/100`:

- **Point spread.** Each lens has spread functions at nine field radii (`ρ` = 0, 1/8, …, 1) and eight wavelengths, 33 × 33 samples over ±`w(ρ)` in a frame turned to the radial (x) and tangential (y) directions. The plan projects them onto the camera's three channels (section 6) and scales their width by `e`. A pixel gathers with the blend of its two neighbouring radii, turned to its own direction. The channels' kernels have their own centroids, which is the lateral colour fringe; the anamorphic attachment multiplies the horizontal part of that offset by `1 + h·(q − 1)` (horizontal fringing). The width is capped at `w_max` = 0.012 (about 62 px on a 50 MP frame); the implementation may gather the wide part from a half-resolution copy of the tile, as long as D5 and the parity hold.
- **Falloff.** A gain `1 − e·(1 − T(ρ'))` on all three channels, with the lens's relative illumination `T`.
- **Anamorphic.** The field radius becomes oval, `ρ' = √((α·x)² + y²)` in half-diagonal units along the axis (x) and across it (y), with `α = 1 + κ·(q − 1)`, so falloff and softness grow faster towards the left and right; the attachment adds its own side softness, `w` times `1 + a_s·(q − 1)·x²`. At Edges 0 these terms act alone.

Reduced views use the same kernels at `1/k` scale (a kernel below one pixel is a copy).

### L3 · Highlight optics

With `F = flares/100`:

- **Restored light** `R`: the image times `1 + f·(2^h − 1)`, `h = 3` EV (`LENS_HEADROOM_EV`), where `f` is the clip test: a ramp from 0 to 1 as `min_c(X_c / g0_c)` goes from 0.95 to 0.98 (`g0` is the clip level of each channel in the group's scale; R3's gain only raises clipped values, so they still pass; on the lens plane the ramp runs from 0.90 to 0.97, because binning mixes clipped and unclipped photosites). Only the flares read `R`; the picture's own clipped pixels are not raised.
- **Highlights** `H`: `R` weighted by a soft knee from 0 at half the clip level to 1 at the clip.
- **The light budget.** The lens data give the fractions of light at Flares 100: glow `a_g`, veiling `a_v`, each ghost `a_i`, starburst `a_*`, and the attachment's streak `a_s`. With `Z` the output of L2 and `a` the sum of the fractions in use (the streak's with its Anamorphic factor) times `F`:

  `out = (1 − a)·Z + F·(a_g·K_g⊛R + a_v·K_v⊛R + Σ a_i·G_i[H] + a_*·B[H] + a_s·(q − 1)/(Q_max − 1)·S[H])`

  Every kernel and shape has unit energy, so the light the image loses comes back spread out, and the light the sensor clipped comes back too.
- **Glow** `K_g`: a core within 8 px (and at most 0.0015 half-diagonals), gathered per tile at full resolution, plus the rest, which is smooth and is convolved with the tail plane by an FFT. The plan splits the lens's kernel so that the two parts add up to it to within the tail plane's resolution. The tail falls as a power law and widens with shorter wavelengths, per the lens data. Detuned's glow and starburst come from a pupil with a seeded random phase (computed offline, so it is data), which gives its smeared flares.
- **Veiling** `K_v`: a very wide, nearly flat kernel on the tail plane; the contrast the lens loses near light.
- **The source list**: the connected components of `H` on the lens plane (8-connected, in raster order), each larger than 0.02 half-diagonals split into cells of that size, each with its energy per channel, centroid and second moments; at most 256, by energy, ties by position.
- **Ghosts** `G_i`: per source and ghost, the lens's aperture shape (blade count, roundness, rim profile) of radius `s_i`, centred at `c + m_i·(p_s − c)` on the line from the source `p_s` through the centre `c`, cut by a circle of radius `R_i` centred at `c + n_i·(p_s − c)` (the arcs of ghosts from lights near the edge), widened by the source's extent, tinted by the ghost's coating (section 6). With Anamorphic, ghosts widen along the axis by `1 + κ_g·(q − 1)` and the attachment adds its own oval ghosts.
- **Starbursts** `B`: for compact sources (extent at most 0.004), the lens's spike table (its blade count, `|F{pupil}|²` over wavelengths, radius at most 0.08), stronger as the photo's f-number grows (EXIF; f/4 when it has none).
- **Streaks** `S` (Anamorphic above 0 only): for every source, a band along the axis through its centre, as tall as the source, with the attachment's profile (a sharp core and a long tail, reaching one half-diagonal at Anamorphic 100) and its tint.
- Ghosts, starbursts and streaks are drawn analytically per pixel from the list; each tile gets, from the plan, the items whose bounds overlap it, in list order.

### L4 · Anamorphic bend, distortion and zoom

The map from the photo to the canvas, in half-diagonal units with x along the anamorphic axis:

- the attachment: `x' = x·(1 + μ·x²)`, `y' = y·(1 − β·x²)`, with `μ = μ₀·(q − 1)` (the edge stretch) and `β = β₀·(q − 1)` (horizontal lines bowing);
- the lens's own radial distortion `r' = r·(1 + e·(δ₁r² + δ₂r⁴))`.

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
                 depth: Option<&DepthMap>, cancel: &CancelToken) -> Result<Self, LensError>;
}
pub mod cpu { /* reshape, field, flare: run on any rectangle of the frame; warp: Lens4Map */ }
```

`ym-core` keeps the plan and the planes in the photo's graph (*Render graph*) and the depth map with the photo's estimates (in memory only; hard rule 4). `Session::lens_library()`, `Session::model_status(name)` and `Session::fetch_model(name)` back the IPC commands of section 10.

## 6. The lens data

`data/lenses/`, committed, about 2.5 MB in all:

- `<id>/lens.json`: `id`, `version`, `name`, `line`, `start` `{ flares, edges }`, `provenance`, `field` (the radii, `T(ρ)`, the distortion `δ₁, δ₂`, the swirl `σ(ρ)`, the cat's-eye `c(ρ)`), `flare` (glow and veiling fractions and tails, the ghosts `{ m, s, n, R, a, coating }`, the starburst's blade count, rotation and strength), `bokeh.edge`.
- `<id>/tables.safetensors`: the spread functions (9 radii × 8 wavelengths × 33 × 33), the glow tail per wavelength and the spike table, f32.
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
| `psf` | `[9, 8, 33, 33]` | per radius and wavelength, samples on a 33 × 33 grid over `±w(ρ)`, row-major with x radial (outwards) and y tangential (anticlockwise) |
| `spikes` | `[8, 129, 129]` | per wavelength, the starburst over `±flare.starburst.radius` |

`anamorphic.json`: `{ "id": "anamorphic", "version", "provenance", "q_max": 2.0, "bend": { "mu0", "beta0" }, "field": { "kappa", "side_softness", "fringe" }, "ghost_stretch": κ_g, "streak": { "fraction", "reach", "core", "power", "tint": [31] }, "ghosts": [ … as above ] }`.

**Colour.** Coatings and tints are reflectance spectra (31 samples, 400 to 700 nm), and the spread functions and tails are given per wavelength. The plan projects them onto the photo's camera channels with the profile's spectral curves (`data/cameras/sony/`, `docs/03`), so a blue streak is the blue that body's sensor would record. Stand-in bodies use their profile's curves.

**Provenance.** Every file says, in `provenance.statement`: **"Synthetic, computed from public lens designs, not measured."** The designs are prescriptions Yellowmoon transcribes itself from expired patents or nineteenth-century publications; each lens names its source (patent number or publication, year, focal length, f-number) and its design type in generic words (double Gauss, triplet, Petzval portrait lens). Lens designs never come from third-party lens libraries. Coatings are assumed (single-layer MgF₂ quarter-wave films, multi-layer for Default), tuned by eye and documented as an artistic choice. Default is a well-corrected double Gauss with multi-layer coatings; Detuned is Default with its elements decentred and tilted.

**How they are made.** The data are computed offline by a lens simulator that is not part of the repository or the app, and are fitted into the formulas and tables above; only those are committed. They cannot be regenerated from the repository alone, and `README.md` says so. Tests check what can be checked from the data alone (section 14).

**Licences.** `docs/14` gains a row: lens prescriptions transcribed from expired patents and old publications (facts of a design; provenance per lens; [HUMAN] check, section 13).

## 7. The depth model

- **Model.** Depth Anything V2-Small (Apache-2.0 code and weights, 24.8 M parameters). `training/depth/export.py` (offline, rule 6) writes its weights as f16 safetensors, about 50 MB, with the architecture in the header and deterministic bytes. Its model card is `data/models/depth/MODEL_CARD.md`; the weights are not committed (they are above the 30 MB line of `docs/12` *Bundle contents*), and `NOTICE` gains the model's notice.
- **Runtime.** `ym_ml::depth_net`, pure Rust on the CPU, f32 arithmetic in a fixed order, the same bits in the app and `ym` on every platform (as `crop_net`). Parity with PyTorch within 2e-3 of the output's range on five fixtures.
- **Input.** Apple Auto's picture (`docs/19` §3: the photo at default settings, 768 px, the camera's orientation), resized to 518 px on its short side with both sides multiples of 14. Output: relative inverse depth on that grid, kept with the photo's estimates in memory and never written.
- **Where it comes from.** A release asset of `github.com/ngriaznov/yellowmoon` under the tag `models-depth-1`, its SHA-256 compiled into the app and `ym`. `docs/12` adopts this first-run download (AGENTS rule 7): it is stored at `~/Library/Application Support/app.yellowmoon/models/depth/1/depth.safetensors`, which `docs/01` *Persistence* gains (`models/`). It is fetched over HTTPS to a temporary file, checked and renamed; a failure leaves nothing behind.
- **When.** The app downloads it the first time a photo's settings ask for reshaping (section 1), shows the progress in the card's note, and does not retry by itself. The CLI never downloads during a render: `ym models fetch` does (section 8).
- **Without it** L1 is skipped and everything else runs. An export waits for the photo's depth when the model is installed and the settings need it (as it waits for the shared estimates). The XMP records the lens data's id and version and the depth model's version and SHA-256 when it was used (`docs/01` *Versions written into outputs*).

## 8. The command line

`ym render` gains:

| Option | Values | Effect |
|---|---|---|
| `--lens <name>` | `default`, `classic`, `vintage`, `petzval`, `triplet`, `sonnar`, `detuned` | the lens; Flares and Edges take its starting amounts unless `--flares` or `--edges` is given, as in the card |
| `--flares <0–100>` | | Flares |
| `--edges <0–100>` | | Edges |
| `--anamorphic <0–100>` | | Anamorphic |

They are the `lens` field of the settings object; `--settings` takes it too. New commands: `ym lenses list [--json]` (id, name, line, start amounts, design source) and `ym models list|fetch [--json]` (what is installed, its path and SHA-256; `fetch` downloads the depth model with progress on stderr). A render that needs the model when it is missing warns, once per run: "The depth model is not installed, so bokeh was left as it is. Run `ym models fetch` to install it." (`--json`: a warning with kind `depth_model`). `ym doctor` lists downloaded models in `model_weights` with their paths.

## 9. Determinism and speed

- The planes and lists are made on the CPU in f64 with a fixed order (lists sorted by energy, ties by position), the FFT with `rustfft`'s scalar planner (MIT OR Apache-2.0) and `libm` for transcendental functions, so they are the same bits on every platform.
- Every per-tile pass is a function of the planes, the lists, the tile's own pixels and absolute coordinates: tiles equal the whole frame bit for bit at 512 and 1024 px (D5), with the halos of section 4 (L1's stretch, L2's `w_max`, L3's 8 px core).
- Each stage has a CPU f64 reference and a WGSL kernel, parity within 1e-3 (the f16 storage gate of `docs/05`), and holds under Metal fast-math (K8). The app and `ym` export the same bytes (rule 6).
- **Targets** (Apple M1, 50 MP, Metal): the planes in at most 0.5 s after a change of lens or photo (off the foreground worker; the fit view shows the lens when they are ready); a slider move in the fit view in at most 100 ms once they exist; a 100 % tile at most twice the cost of the raw tile; an export at most 4 s longer; depth at most 3 s once per photo. CI on Linux measures the CPU and lavapipe as a guard against regressions only; the Mac numbers are the owner's check (section 13).

## 10. The app

- IPC: `lens_list` → `LensInfo[]` (read once); `model_status(name)` and the event `model_download { name, state: idle | downloading { percent } | ready | failed { reason } }`; `fetch_model(name)` (the note's Try again). The render report carries `lens: { reshaped, reason }`, which the note reads ("little out-of-focus background", "depth model missing", "reading depth").
- The browser build's mock serves `lens_list`, a fake download and the report.
- `docs/11` gets the card under *Look tab* and the tab table's Look row.

## 11. Not in this version

| Step | Why later |
|---|---|
| Adding bokeh to a photo that has none | the owner's direction: reshape what is there, never add |
| A matting model for hair | the guided upsampling and the sharpness test first; added if tests show halos at hair |
| A sharp area away from the centre (Detuned's sweet spot moved) | needs a control |
| A lens with a front effect plate (abstract smeared flares) | Detuned's smear covers a mild form |
| Narrowing bokeh or giving it an exact new shape | needs deconvolution |
| Breathing and focus pulls | they do not exist in a still |
| Replicas of named lenses | no public prescriptions or coating data for them |
| Film looks, grain, halation | out of scope (the owner: lenses only) |
| A generative layer | fails the pipeline's constraints (research note); the bake-off is still open |
| A 2.39:1 crop aspect | a separate request (section 12, decision 5) |

## 12. Decisions for the owner

The spec uses the first option of each until the owner says otherwise.

1. **Two names.** "Sonnar" is a Zeiss trademark and "Petzval" a person's name that a lens maker also sells under; the research note's rule was descriptive names only. Keep both after a [HUMAN] check, or rename them (for example **Glow** and **Swirl**)? The ids (`--lens sonnar`) change with the names, so this is decided before LC-03.
2. **"Edges"** as the label of the field-effects slider.
3. **Anamorphic 100 is a 2× squeeze**: bokeh twice as tall as it is wide.
4. **The anamorphic axis follows the picture as seen**, so a portrait photo gets horizontal streaks; a real attachment would turn with the camera and give vertical ones.
5. **No 2.39:1 crop aspect** in this work; the Crop panel keeps its five.
6. **The depth model downloads without asking first** (the card says so while it downloads); or ask once.
7. **Choosing a lens moves Flares and Edges** to its starting amounts (card and `--lens`).
8. **AGENTS.md rule 4** says the only things that persist are imported looks and a few preferences; a downloaded model is a third kind. The owner's edit, if wanted.

## 13. Tasks

Task format of `docs/15`. Each task updates the specs it touches and `docs/STATUS.md`.

**LC-01 · Lens library** (`ym-lens`): the data format, loading, checks and `SHA256SUMS`, the spectral projection. *Done when:* the property tests of section 14 pass on synthetic lens files.
**LC-02 · Lens data** (LC-01): the seven lenses and the attachment, with provenance and `README.md`. *Done when:* every file passes LC-01's checks; contact sheets of the CC0 test photos at 0, 50 and 100 for each slider and lens are in `docs/images/lens/`.
**LC-03 · The setting** (decision 1): `lens` in `settings.rs`, the schema, `types.ts`, patches, Copy and Paste, `StageGroup::Lens`, the render graph's keys. *Done when:* the settings tests and the no-lens identity test pass.
**LC-04 · L4** (LC-03): the map, the zoom, the sampler on the CPU and the device, pick, tile rectangles, `Geometry::upright`. *Done when:* the geometry tests of section 14 pass.
**LC-05 · L2** (LC-01, LC-03): CPU reference, kernel, parity, tiles. *Done when:* its tests pass.
**LC-06 · L3** (LC-01, LC-03): restored light, planes, FFT, source list, the drawn items; CPU reference, kernels, parity, tiles. *Done when:* its tests pass.
**LC-07 · Depth model**: export, the release asset, `ym_ml::depth_net` and its parity, the download in the app and `ym models fetch`, `docs/12` and `docs/01` *Persistence*. *Done when:* parity and the download tests pass.
**LC-08 · L1** (LC-06, LC-07): blur map, balls, stretch, redraw; CPU reference, kernels, parity, tiles. *Done when:* its tests pass.
**LC-09 · The card** (LC-03): Lens card, IPC, the note, the mock, e2e, screenshots. *Done when:* its tests pass and the screenshots are reviewed.
**LC-10 · The command line** (LC-03): the options, `ym lenses`, `ym models`, `docs/10`. *Done when:* its tests pass.

**[HUMAN] LC-H1** Check the transcribed prescriptions and the two names (decision 1). **LC-H2** Look at each lens on your own photos and say which ship. **LC-H3** The Mac speed targets (section 9), with `ym doctor --bench`.

## 14. Tests

- `ym-lens`: every shipped and synthetic lens file loads, matches `SHA256SUMS` and has `provenance.statement` exactly; spread functions, tails, ghosts, spikes and streak profiles have unit energy (1e-9) and no negative sample; `T(ρ)` is 1 at the centre and falls; the projection of a flat spectrum is neutral for every profile; a damaged file is refused, never a panic.
- **Identity:** all three amounts at 0, for every lens, gives the bytes of a render without the field (CPU and device, viewport, 100 % and export), and creates no lens pass (`RenderReport::runs`).
- **Energy (L3):** on synthetic frames whose flares stay inside the frame, the sum of `out` is `(1 − a)·ΣZ` plus each part's fraction times `F` times the light it spreads (`ΣR` for glow and veiling, `ΣH` for the rest), to 1e-6; a clipped point source gets its 3 EV of headroom in the flare and not in its own pixels; a ghost's centre lies on the line from its source through the centre (0.5 px), on the other side for negative `m`; with Anamorphic at 0 there is no streak, and at 100 the streak lies along the axis, also on a turned canvas.
- **L1:** a synthetic scene with a sharp subject before a background of known disc blur (drawn analytically): background discs come out `q` times taller within 5 %, their energy kept within 1 %; the subject's pixels unchanged to 1e-6; a scene in focus everywhere is unchanged and reports "little out-of-focus background"; without the depth model L1 is skipped and the report says so.
- **L2:** a flat field gets exactly the falloff gain; a point grid gets each lens's spread at its field radius; Edges 0 with Anamorphic 0 is the identity.
- **L4:** no output pixel of the canvas lacks a source, at full resolution and reduced 3, 8 and 16 times, for every Anamorphic amount, four turns and three straighten angles; the inverse map's round trip within 1e-9 (f64); a straight horizontal line near the top bows and a vertical stays within 0.5 px of straight; `pick_point_at` round trip.
- **Tiles and parity:** each stage's tiles at 512 and 1024 equal the whole frame bit for bit on the device; CPU and device within 1e-3; the poison and interleave tests of D7; the far-corner test of D4.
- **Settings:** reading with and without `lens`, partial patches, clamping, an unknown kind, Copy and Paste, Reset, the schema.
- **CLI and parity:** `ym render --lens vintage --flares 60 --anamorphic 50` equals the same settings through `--settings` and the app's export byte for byte; `ym lenses list --json` equals `lens_list`; the missing-model warning; `ym models fetch` against a local test server (hash checked, a bad hash refused and nothing left behind).
- **Depth:** `depth_net` against PyTorch on five fixtures; the same bits in two runs and from two sessions.
- **Frontend:** the card (the menu and its lines, a lens choice sends the start amounts in one patch and is one undo step, sliders, Reset, the bokeh lines), the mock, e2e `lens.spec.ts`.
- **Goldens:** the golden set is unchanged (no lens); a lens golden per stage on synthetic scenes joins it.

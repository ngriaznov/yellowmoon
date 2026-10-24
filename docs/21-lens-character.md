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

**As built (L1 to L3 in the graph, CPU; `crates/ym-core/src/pipeline/graph/lens.rs`).** The decisions the build made:

- **Where.** The lens sits behind the five places the raw group leaves the graph (`docs/01` *The renderer and its keys*): `Job::lens_group_backing(k)` (fit view, thumbnails, look tiles, histogram and readout: `apply` over the whole group at 1/k, cached per lens setting), `Job::lens_region_pixels(rect)` (the 100 % view: lens tiles on the 512 px grid, cached), `Job::lens_region(rect, false)` and `Job::lens_region_resident(rect)` (exports, the export tail and both DNG targets: the raw region grown by `halo_px`, one call, `apply`, cropped to the rectangle; nothing cached), and `Job::lens_tiles_ready` / `Job::lens_warm_tiles` (so a 100 % view with a lens is not stuck coarse). Without a lens each is the raw method it mirrors. On the CPU route the lens's caches are host images; on the device route they are device images (the block *the lens on the device*, below).
- **Planes.** One set per photo in `GraphState`, outside the `ByteLru`s (and counted in `PhotoGraph::bytes`), made from the raw group at `k_L`; for a frame under 1024 px (`k_L` = 1, only tiny test frames) the whole frame is assembled from the raw tiles. An **export makes its planes itself** from the CPU reference raw group and keeps them for the request, so the bytes of an export do not depend on what the viewer looked at (and a device-made group never feeds them): the app and `ym` write the same file.
- **The axis is in the key whatever the amounts**, not only when Anamorphic is above 0: L3's starbursts and ghosts follow the canvas's orientation too. A quarter turn therefore makes the lens passes again.
- **The focus point** is the maker note's point as a fraction of the sensor frame, shifted and scaled by R9's crop and clamped to the frame (the distortion correction's few pixels are ignored: the detector needs the area). Sony's "centre when unknown" is not told apart from a real centre point.
- **The camera's colour** comes from the profile's spectral curves read from the data folder (`RenderOptions::data_root`), projected once per camera. When the curves cannot be read the lens uses a neutral camera and the render's notes say "The camera's colour curves could not be read, so the lens colours are neutral."
- **Thumbnails and look tiles** make the photo's planes like any other request (including the blur map when a slider asks for bokeh), so they show the lens exactly as the viewport does and do not depend on what was opened first; section 1's "only once the blur map exists" is not needed. The price is one set of planes per photo whose graph a thumbnail touches (the renderer keeps three graphs).
- **Budgets.** The reduced lens groups take up to 1/32 and the 100 % lens tiles up to 1/12 of the CPU budget, per photo; a slider drag leaves one entry per value until the oldest are evicted. An export adds the lens's halo to the 160 px its working set is reserved for.
- **Counters.** `StageRuns::lens_planes`, `lens_groups`, `lens_tiles`, and, for the stage caches below, `lens_fields` (the field stage ran over a group or a tile) and `lens_flares` (the flare terms did); `StageTimings::lens_ms`; all zero without a lens (a test pins it for every lens).
- **Measured (CPU reference, `cargo test -p ym-core --test render_lens -- --ignored --nocapture lens_timing`; 24 MP synthetic photo, Vintage 60/50/50, 4 shared cores at load 4 to 6, workspace crates at the dev profile's `opt-level = 1`, so a release build is several times faster).** Fit view (1500 x 1000, lens group at 1/4): 2.0 s without a lens (cold), 10.7 s with the lens the first time (planes and group), 10.1 s for a Flares move and 10.4 s for an Edges move (each move runs L1 to L3 over the 1500 x 1000 group again). 100 % view of 1500 x 1000: 12.6 s without a lens (12 raw tiles), 37.8 s with it the first time (6 more raw tiles for the halos, 12 lens tiles) and 21.3 s for a Flares move (12 lens tiles, about 1.8 s each). Export: 79.8 s without a lens (70.5 s in the raw stages), 289 s with it (88.5 s in the raw stages, the planes' group included; 190.9 s in the lens, 24 tiles). These are the CPU reference's numbers, the guard against regressions; the targets of section 9 (a slider move in at most 100 ms) are for the kernels on the device, which the blocks below describe; those numbers are from before the stage caches and the L3 changes, whose effect is measured there.

**As built (stage caches; `graph/lens.rs`, `LensPlan::{field_key, flares_key, field_is_identity}`).** `Z = L2(L1(X))` reads Edges and Anamorphic but not Flares; the flare terms of L3, `F (core + tail + items)` (`ym_lens::flare_terms`, `f64`), read Flares and Anamorphic but not Edges and not `Z`. The two halves are cached on their own, and the picture is `(1 − a) Z + terms` in the one expression and order `ym_lens::apply` uses (`apply` is `apply_field`, `flare_terms` and `add_flares`; a test pins that a cached `Z` or cached terms give the bits a fresh run gives):

- `lens_field_groups` and `lens_field_tiles` hold `Z`, keyed by the raw group's key, `LensPlan::field_key` (lens, data versions, Edges, Anamorphic, camera, frame, axis, the planes' key) and `k` or the tile; `lens_flare_groups` and `lens_flare_tiles` hold the terms, keyed by `LensPlan::flares_key` (lens, Flares, Anamorphic, camera, frame, axis, `g0`, f-number) and `k` or the tile. `lens_groups` and `lens_tiles` still hold the finished picture. With Edges and Anamorphic at 0 the field stage is the identity (`field_is_identity`) and `Z` is the raw group itself, not cached.
- A Flares move runs L3's terms over the cached `Z` (`lens_flares` counts one run, `lens_fields` none); an Edges move runs L1 and L2 and reuses the terms; an Anamorphic move runs both. On the device the terms are not cached (a Flares move uploads nothing and the kernel is cheap, below); `Z` is cached as device images.
- **Which stage costs what** (the CPU reference, `cargo run -p ym-lens --example bench_stages`, a 1500 × 1000 group of a 6000 × 4000 frame, 60/50/50, the dev profile, CPU seconds; before the L3 changes below / after): Vintage L1 1.8 / 1.9, L2 5.6 / 5.0, L3 6.1 / 0.4, the whole lens 14.1 / 8.6; Default L1 1.9 / 1.9, L2 2.0 / 1.5, L3 6.2 / 0.2, the whole lens 9.5 / 3.9. L3 dominated, because of the ghosts of many sources (the block under *L3*); after the change L2 does.
- **A slider move on that group** (the same measure): every move ran the whole lens before (Vintage 14.6 s, Default 9.8 s wall); now a Flares move is L3 alone (0.41 s Vintage, 0.29 s Default), an Edges move is L1 and L2 (4.7 s and 2.4 s, the terms reused), an Anamorphic move is both.

**As built (the lens on the device; `graph/lens_device.rs`, `ym-gpu` `kernels/lens_{field,flare}.rs`).** The decisions the build made:

- **When.** `GpuStages::LENS` (bit 7, in `ALL` and `USED_BY_EXPORTS`, not in `SIGNED_OFF` until the Mac run; `YELLOWMOON_GPU` names it `lens`) and a view whose results stay on the device (`Job::keeps_on_device`), or an export on the device. Otherwise the CPU route above, which is also the fallback.
- **A tile is one job.** A reduced group is made in output tiles of 1024 px, the 100 % view on the raw group's 512 px grid, an export in its rectangle. A job gathers the raw window (the tile grown by the halo of L2 or L3) from the raw group's pieces where they are (`encode_gather`; a raw tile that lives on the host is uploaded inside the job), runs L2 into `Z`, runs L3 over `Z`, and the result is a `ResidentImage` of pieces that the sampler and the colour pass read directly. No pixel is read back for the lens: a Flares move on the fit view reads back the finished frame and nothing else (a test counts the bytes). Every pixel is a function of its absolute position and of the input within the halo, so tiles of any size give the whole frame's bits (D5).
- **Per photo on the device** (`LensDevice`): the spread functions of the lens on the camera, the tail plane and spike tables of the planes, the lens's rim, aperture and normal-tail tables, and the scale data (the glow core's taps, the drawn items and their bins) of the last six plan shapes. None depends on Flares, so a Flares move uploads nothing (`F` and `1 − a` are uniforms); an Edges move changes a uniform.
- **L1 is hybrid.** Where L1 acts, the host runs `ym_lens::reshape` on the raw region L2 reads (its halo included), uploads the result as L2's input, and L3 still reads the raw picture; everything after L1 stays on the device. The slow part of an Anamorphic or Edges move on a photo with bokeh is therefore L1's 1.9 s on that group (CPU, dev profile); a kernel for it is the open item of this block.
- **Failures.** A device error in a tile falls back to the CPU lens for the request with a note (`The graphics card could not apply the lens, so the CPU did: …`) and ends the photo's use of the device lens, except an error about size; a shader that does not compile ends it for the process; a strict policy (`Require`) fails the request; an export starts again wholly on the CPU (`Job::take_gpu_restart`), so its file is the CPU export. `tests/gpu_faults.rs` pins each fault (`oom`, `shader`, `lost`, `tile@N`, `cancel@N`) on an export with a lens.
- **Exports** follow the other stages: the planes of an export are still made from the CPU reference group, so the app and `ym` write the same file (`crates/ym-cli/tests/parity_gpu.rs`). The device export equals the CPU export within the kernels' tolerance (the largest difference in a 16-bit PNG was 106 of 65535, none of 1.44 million samples over 256), not bit for bit: the two routes are different arithmetic, as with every other device stage.
- **Tests** (`tests/render_lens.rs`, device cases skip without an adapter): the CPU lens against the device lens at the fit view and the 100 % view (1 code of 255), with the raw group on the host and on the device; the reshaped case (1 code); the counters of a slider move on tiles and on a group; no group readback; identity at all-zero amounts with no lens pass; the export tail; viewport jobs between the tiles of an export change nothing (D7).

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

**As built** (`ym_lens::l1`; decisions the stage text left open):

- **Planes.** `LensPlanes::bokeh()` holds the map and the balls only when `LensPlan::needs_blur_map()`; `LensPlanes::serves(plan)` says whether planes still fit a plan (the planes' key leaves the amounts out, so a plan that gains Anamorphic needs planes built again; `l1::apply` refuses planes without the map with `LensError::Frame`). `ym_lens::reshape` and `reshape_halo` run L1 alone (tests and diagnostics).
- **Upsampling.** The map is read at the tile's scale from the coarsest of its reductions whose cells are at least a pixel apart; each pixel takes the cells within `min(8, 1.6 × the spacing)` pixels, weighted by distance and by how close the tile's own colours (3 x 3 smoothed square roots) are to the cell's footprint mean, and a cell's weight is its confidence (a confident 0 counts); with less than a quarter of the weight confident the pixel is 0, and so is a value below `b_min`. The sharp-edge test compares the picture's gradient energy with that of its 3 x 3 mean over 5 x 5 windows (a sharp edge loses more than 1.4 times over, above 4 noise units), only where the blur is at least 4 px at the tile's scale, dilated by 2 px; it is not applied inside a ball's disc and 7 px beyond (the ball's own rim is a sharp edge). The halo is the two gathers' reach plus `ceil(support) + ceil(cell/2) + 3` (at least 7) px of the tile's own picture, so a little more than the 8 px of the stage text.
- **Gather.** Where every pixel of the line passes the sample rule (checked with a table of range minima of the blur), the average is the exact mean of the piecewise-linear picture over the line, read from running integrals in 64-bit fixed point (2^-26), so any tiling reads the same integers; elsewhere it is the average of `2⌈ℓ⌉ + 1` evenly spaced taps (at most 65), the ends at half weight, the rule applied to each. The tangential gather always uses taps. The swirl profile is read at the plain field radius `ρ` (not the oval one of L2).
- **Balls.** Take-out radius `1.15 r` plus 1.5 pixels of the work grid; the map carries a ball's radius over `2.6 r` around it (the largest oval), except where it says confidently sharp. The drawn shape is the ellipse `r` across and `r + |s|` along, with `s = (q − 1) r` along the cross direction plus `σ(ρ) · edges · r` along the tangent (sign chosen to add); the cat's-eye is two copies of it, each half the offset `c(ρ) · edges · r` away radially, intersected; the rim table is read at `i / 15`; the energy is spread over the shape's analytic area (the lens-shaped area in closed form, the rim's mean over it on a fixed grid), so it is the ball's own to 1 %. A ball changes nothing (is neither taken out nor drawn) when `q`, the swirl and the cat's-eye leave its shape round.

**As built (device).** L1 has no kernel yet. Its per-tile pass (the bilateral upsampling of the blur map, the sharp-edge test, the take-out, the two gathers with their running integrals, the redraw) runs on the host where it acts, as `ym_lens::reshape` over the rectangle L2 reads, and its output is uploaded as L2's input (the block *the lens on the device*, section 3). The kernel is to be driven by the planes' data (the blur map levels and their confidence, the summed-area table of the blurred cells, the ball list) and not by constants copied into the shader, and to draw each ball over its own bounding box.

### L2 · Field optics

With `e = edges/100`, and `e' = e + (1 − e)·κ·(q − 1)` the **edge amount**: Edges plus the attachment's own share, so that at Edges 0 the anamorphic terms below still act, alone (`e' = e` at Anamorphic 0; at `e' = 0` L2 is the identity). Every `e` below is `e'`.

- **Point spread.** Each lens has spread functions at nine field radii (`ρ` = 0, 1/8, …, 1) and eight wavelengths, 33 × 33 cells over ±`w(ρ)` (cell-centred, section 6) in a frame turned to the radial (x, outwards from the frame's centre) and tangential (y, anticlockwise on the screen) directions. The plan projects them onto the camera's three channels (section 6). A pixel gathers with the blend of its two neighbouring radii (linear in `ρ'`, held at the last table beyond 1), turned to its own direction and scaled in width to a half-width of `ω = min(e'·(1 + S)·w(ρ'), w_max)·H` pixels, `H` being the half-diagonal in pixels and `S` the side softness below. The channels' kernels have their own centroids, which is the lateral colour fringe; the anamorphic attachment multiplies the axis part of that offset by `1 + h·(q − 1)` (horizontal fringing). The CPU reference does so by reading each channel of the gathered picture that much further along the axis (linear interpolation between neighbouring pixels, the gain applied after): to first order the same as translating the kernel, for one gather instead of three. The width is capped at `w_max` = 0.012 (about 62 px on a 50 MP frame).
- **Taps.** A kernel of half-width `ω` is `n × n` point masses at its cell centres, `n = min(17, 2·⌈ω⌉ + 1)`: the 33 × 33 table resampled area-weighted and renormalised to sum 1, each mass read from the input by bilinear interpolation. A kernel below `ω` = 0.5 px is a copy of the input (the gain still applies). A kernel whose cells would be more than one texel apart at 17 taps (`ω` above about 8.5 px) reads a 2×, 4×, … box-reduced copy of the input instead, whose blocks are aligned to absolute frame coordinates so that tiles agree; the input is replicated beyond the frame edge. A kernel costs at most 289 taps; the 17 is a speed budget (`ym_lens::consts::MAX_TAPS_SIDE`) and can be raised to 33 where the cost is acceptable.
- **Falloff.** A gain `1 − e·(1 − T(ρ'))` on all three channels, with the lens's relative illumination `T` (read linearly between radii, held at `T(1)` beyond `ρ'` = 1).
- **Anamorphic.** The field radius becomes oval, `ρ' = √((α·x)² + y²)` in half-diagonal units along the axis (x) and across it (y), with `α = 1 + κ·(q − 1)`, so falloff and softness grow faster towards the left and right; the attachment adds its own side softness `S = a_s·(q − 1)·x²`, which widens the kernel (`1 + S` times). The kernel's direction is the geometric one, from the frame's centre to the pixel. At Edges 0 these terms act alone through `e'`.

Reduced views use the same kernels at `1/k` scale (a kernel below one pixel is a copy).

**As built (device; `ym-gpu` `kernels/lens_field.rs`, `shaders/lens_field.wgsl`, `ym_lens::device`).**

- **Passes.** `lens_field_pyramid` once per level (level 0 is the input over the region the gathers read, rounded outwards to blocks of `2^lmax` frame pixels with the frame's edge replicated; level `l` the mean of the aligned 2 × 2 blocks of level `l − 1`), `lens_field_gather` over the output (the pixel's spread function resampled to `n × n` taps, turned to the radial direction and scaled to `ω`, read by bilinear taps from the level that keeps the taps a texel apart), and, when the attachment's fringe reads beyond the output, `lens_field_finish`: the gather is made over the output grown by the fringe's reach and each channel is read at its shift along the axis, times the gain. Without the fringe the gather applies the gain itself.
- **Data.** The spread functions on the camera's channels are flattened once per lens and camera (`LensPlan::field_tables_data`: a header of the radii and the cell tables of side 3, 5, …, 17 at the nine radii); the amounts, frame and axis are `FieldPass`, plain numbers (`LensPlan::field_pass`). The constants of the WGSL come from `ym_lens::consts`.
- **Positions** (D4, D5). A pixel is its absolute column and row; a tap lies a small offset (at most about 90 px) from it, formed in `f32` and split, for the level of the pyramid, into an exact integer and a fraction, so no `f32` of order 1e4 exists; the pyramid's texels are means of aligned blocks of absolute coordinates. The sums that decide a tap count or a level sit behind the OR barrier (K8).
- **Parity** (`crates/ym-gpu/tests/gpu_lens_field.rs`, lavapipe): 36 cases (every lens at four Edges/Anamorphic pairs on a landscape frame, two lenses on an odd portrait frame, two on the y axis), 27.5 million samples, largest difference 3.04e-6 absolute (white = 1; about 25 ulps of the sum of up to 289 products), no sample over the bound 6.1e-6 (twice the first measurement; a flipped decision is allowed in 1e-6 of the samples, none did); tiles of 512 and 1024 equal the whole frame bit for bit; a poisoned device equals a plain one; the far corner of an 8658 × 5772 frame is as exact as the near one (1.2e-7 and 1.2e-7 against 1.2e-6 in the centre); the binding stays inside the design size. On lavapipe (software) the 36 cases take 9.9 s against the CPU reference's 4.4 s; the Mac numbers are the owner's check (section 13, LC-H3).

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

**As built (L3, CPU and device; `ym_lens::l3`, `ym-gpu` `kernels/lens_flare.rs`, `shaders/lens_flare.wgsl`).** A photo with many lights took minutes (a night or neon photo: 11 to 30 lights 216 s on the CPU, a full list of 256 sources 8 to 10 minutes) because every ghost of every source was evaluated at every pixel it reaches, and the shipped lenses' ghosts reach most of the frame (an apothem of up to a whole half-diagonal, 12 of them, and the attachment's). Two changes, both in the item list (`l3/items.rs`), so the CPU reference and the kernel share them:

- **Ghosts too faint to see are not drawn.** Each ghost has a peak density (its brightest channel's amplitude times the rim's peak). The faintest are left out while their peaks add up to no more than `GHOST_PRUNE_BUDGET` (2^-18 clip levels, a quarter of the step of a 16-bit code at the clip level), a function of the whole list and never of the tile, so any tiling draws the same items. At most that much light is missing at any pixel. Default's ghosts have fractions of 1e-5, so for most photos nothing of them is drawn; Petzval's and Triplet's (up to 4e-3) are.
- **Ghosts that vary slowly are drawn on grids.** A ghost whose edge (a Gaussian step of at least a pixel, usually tens or hundreds) and rim curve change slowly, and whose pixel rectangle is at least 16 384 px, is evaluated at the nodes of a grid of spacing `L` = 2, 4, 8, 16 or 32 px (the largest that keeps 2.5 nodes across its narrowest feature) and read back by bilinear interpolation. The node `(j, k)` of a grid sits at the pixel corner `(L j + L/2, L k + L/2)` and holds the sum, in list order, of the densities of the grid's ghosts whose rectangle grown by `L` holds it; a pixel is a function of four nodes, a node of its place and the list, so any tiling gives the whole frame's bits (D5; items within 64 px of a tile count for its nodes). The device does it in one more dispatch per grid (`lens_flare_nodes`) before the main pass. The error of the interpolation is below 0.03 % of a ghost's peak and 5e-5 of its light on the test lights (`ym_lens::flare_terms_exact` is the exact sum). The streaks, starbursts, glow core and tail plane are as before.
- **Measured** (`cargo run -p ym-lens --example bench_sources`, a flat grey 1500 × 1000 group of a 6000 × 4000 frame with `n` clipped lights of 3 to 7 px, Flares 60, Edges 40, Anamorphic 50, the dev profile, wall seconds on 4 shared cores; the flare terms alone, before / after): Default 8 lights 11.7 / 0.24, 32 lights 40.1 / 0.58, 256 lights 239.7 / 4.5; Petzval 8 lights 4.9 / 0.42, 32 lights 18.7 / 1.06, 256 lights 268 / 7.5. The light added changed by at most 1.5e-4 of its total (Petzval 256 lights: 37 902 against 37 894). With no lights the pass is 0.1 to 0.3 s. What remains at 256 lights is the nodes of the grids (2.5 to 6.7 s thread time) and the streaks' profile sums (1.1 s).
- **Parity** (`crates/ym-gpu/tests/gpu_lens_flare.rs`, lavapipe): 19 cases (every lens at Flares 100 with Anamorphic 0 and 100, both axes, lights of every size and colour; 15 draw ghosts on grids): largest difference 1.15e-6 of values up to 0.165 (8.7e-6 relative; the bound is 2e-5 of the terms' peak, twice the first measurement); the ghosts alone on grids (a lens of only big soft ghosts, 4 cases): 9.8e-7 of their peak; mixed with a picture 1.5e-7; tiles of 512 and 1024 equal the whole frame bit for bit; the far corner of an 8658 × 5772 frame: 7.1e-8, 1.0e-7 and 2.4e-8 at the near corner, centre and far corner; poison equals plain. The 18 cases of 500 × 350 take 3.3 s on lavapipe (with uploads) against the CPU reference's 1.4 s (it took 100.9 s before the changes above).
- **The kernels.** `lens_flare_restore` (the restored light the glow core reads), `lens_flare_nodes` (one grid), `lens_flare` (a pixel: core, tail plane, the items of its 32 px bin in list order, the grids, mixed with `Z`). The item lists come from the CPU reference's own `Items` (`ym_lens::device::FlareScaleData`), so the two read one source; 8 storage bindings, the last being the restored light or the nodes. The tail plane, the spike tables and the lens's tables are uploaded once per photo and lens.
- **The light budget of a frame with nothing in it** (a question from the contact sheets). The picture's light is `(1 − a) Z + F (…)` and the glow's tail and the veil spread the light they take over the plane, so what they send beyond the frame's edge is lost (section 4, *Energy on a pixel grid*). That is by design and it is what a lens does: the veiling glare at a pixel is the light of the scene within the frame, so the corners get less than the centre. On a uniform frame at Flares 100 with Edges and Anamorphic at 0 the mean keeps 99.8 % of its light with Default, 97.2 % with Vintage (centre 98.4 %, corner 94 %), pinned by `a_uniform_frame_loses_only_the_light_that_leaves_it`. The larger loss seen on a flat frame at Default 60/40 is L2's falloff, the lens's relative illumination (Edges 40 alone leaves 93.8 % of the mean), not L3.

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

**As built (L4, CPU and device; `ym-gpu` `cpu/sampler.rs`, `kernels/sampler.rs`, `shaders/sampler.wgsl`; `ym-core` `pipeline/geometry.rs`).** The decisions the build made:

- **Where it lives.** `ym_gpu::cpu::sampler::SamplerWarp` is the map as plain `Copy` data (`mu`, `beta`, `d1`, `d2`, the zoom, the Newton steps, and whether the axis is the canvas's y): the same operations in the same order as `ym_lens::LensWarp` (a test holds them equal bit for bit, also on reduced canvases), because `ym-gpu` does not depend on `ym-lens`; `ym_core::pipeline::geometry::sampler_warp` builds it from a `LensWarp`, which gained four accessors (`mu`, `beta`, `distortion_k1`, `distortion_k2`) and nothing else. `Geometry::warp`, `Grid` (through its geometry) and `SamplerGrid::warp` carry it, so it is in every `{:?}` the frame and display keys print, and a slider move is a new key. The identity is `None`: a photo without a bend has exactly the geometry, the kernels and the bytes it had.
- **Built once per photo.** `Job::geometry()` (now `&mut self` and fallible, as it asks for the lens plan; `Job::base_geometry()` is the old one, enough for sides and sizes) takes the warp from the photo's `LensCtx`, which fits the zoom on the full-resolution canvas (`RawPlan::out_w x out_h`, sides swapped by the turns) when the plan is made and caches it with the plan. Anamorphic 0 on a lens without distortion, or Edges 0 with it, is the identity. A grid takes its own canvas from the buffer it samples (`rescaled`: the half-diagonal of the reduced canvas is the unit, as for L1 to L3), so every scale shares the zoom. `Renderer::geometry(src, cancel)` hands the geometry out for checks.
- **The order.** The inside test is on the canvas position after the crop and the straighten, before the bend, so the pixels that are inside, their intervals and `inside_bounds` are what they were. Then `canvas_to_photo`, the division by the canvas size, the turns, the EXIF orientation and the clamp to the buffer. Straightening and the crop act on the bent canvas; Auto crop is untouched.
- **The kernel.** A grid with a bend is never a pixel copy. Its shrink is the largest over the canvas of `|J^-1 v| / z`, `v` each of the output's two step vectors on the canvas and `J` the Jacobian of `F` at the photo position that canvas point reads, taken on a 9 x 9 lattice of the canvas that holds its corners and edges (`SamplerWarp::max_shrink`; a fine scan finds nothing above it); Catmull-Rom up to 1.25, `Box(n)` beyond, `n = ceil(shrink - 0.25)` from 2 to 4. It is one number for the grid, taken from the grid alone, so a tile, the whole frame, the CPU and the device choose alike. At full resolution it is 1.12 to 1.16 at Anamorphic 100 (the sides, where the attachment draws the picture in), so exports and 100 % views are Catmull-Rom for every shipped lens (a test).
- **Source rectangles.** `SamplerGrid::source_bounds` (and `Grid::source_bounds` for a grid with a bend) follows the boundary of what the rectangle samples through the bend: each edge of the rectangle where it lies on the canvas and each edge of the canvas where it lies in the rectangle, both clipped exactly to the other, 65 samples each, the result clamped to the buffer as the sampler clamps, plus the caller's margin and 2 px. The map is a diffeomorphism of the canvas, so the box of the boundary is the box of the region. Without a bend the four corners are used as before. An export's working-set reservation grows by a quarter of the tile for a bend.
- **Pick.** `Geometry::final_to_oriented` applies the inverse map to the canvas position before the turns; `white_pick` stays in the photo's own coordinates, before the bend.
- **The DNG.** `Geometry::upright` carries the warp with the axis swapped when the buffer's sides are (`SamplerWarp::transposed`); the DNG's preview is the baked picture bit for bit, turned back (a test, for the camera orientations that swap the sides, a user's turn and a straighten angle).
- **On the device.** The tables stay what they were: the exact integer-and-fraction position of the chain *without* the bend. The kernel (`warp_pos`) takes the canvas centre's offset to that position (the integer part exact, so no `f32` of order 1e4), in half-diagonal units, runs the same three Newton steps in `f32` written for the offset `e` from the start (the residual is `e + (F(x) - x)`, so nothing of order 1 is subtracted from something of order 1), and moves the exact position by the displacement `e - (z - 1) t`, in canvas pixels through the dihedral map's linear part (a signed permutation of unit scale: exact). Everything depends on the absolute column and row through the tables, so tiles equal the whole frame bit for bit (D5), and the poison test holds (D7). No decision compares sums, so no fast-math barrier is needed (K8); the clamp is on integers. CPU against device: the position differs by a few ulps of the displacement, about 1e-5 px on a 200 px frame and 1e-4 px at 50 MP, which `tests/gpu_sampler.rs` holds to the sampler's own bound of 1.2e-6 in value on a 1152-case matrix (measured 7.2e-7, against 6.0e-7 without a bend; 5e-6 on hardware until the Metal run) and to 1e-5 at the far corner of the 50 MP frame (measured 4.9e-6).

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

**As built** (`ym_lens::l1::{detector, disc, balls, spread}`; decisions the steps above left open):

- **Work plane.** The detector runs on the lens plane, or on its 2 x 2 reduction when the plane's long edge is 1536 px or more; the smallest radius it reports is `max(b_min, 2.2 px)` of that plane, because a sharp edge on a binned plane already reads as a disc of about a pixel.
- **Step 1.** One re-blur of 1 px cannot read the blurs of interest (a disc of 8 px gives a ratio of 1.01); the plane's luminance, in linear light (a defocus blur is a convolution there), therefore goes up a ladder of Gaussians of 0, 1, 2, 4, 8 and 16 px, and the gradient ratios of the five consecutive pairs (each with its own non-maximum suppression, 4 sectors, and the other level's peak over 3 x 3) give the radius through tables of the ratio of a step edge blurred by a *disc* (its half-circle line spread, a native blur of 0.6 px in quadrature) instead of `1 / √(ρ² − 1)`. Each reading is weighted by how well its ratio is conditioned (`|d ln r / d ln ρ|`) and its contrast over the noise (square-root luminance, judged by Immerkaer's Laplacian estimate); a reading is dropped when a weighty reading of a finer pair, within 1.5 of its radius, is below 0.55 of its radius (a region blurred by a disc holds no sharper edge: the coarse levels of a fine texture would otherwise read as blur).
- **Step 3 and the balls.** A scale-space maximum of the differences of the ladder's levels (square-root domain, a ratio of 2 in width) seeds a candidate, and the ball is measured on the linear plane by the area of its half-level contour (flood fill), so radius and centre are good to a few per cent whatever the seed's scale. Besides the stage text's tests a ball must be round (principal axes within 1.4), compact, and, when the edges around it (outside every candidate's footprint) weigh at least 1.5, agree with their radius within 1.6. A scene needs three balls, or balls those edges confirm: a lone round light (the moon, a lamp in focus) is more likely sharp. The edge readings inside a ball's take-out disc (1.5 times) are dropped and the ball's radius is added over its disc and ring; the rim is not an edge of the blur.
- **Step 4.** Pull-push over a pyramid of 2 x 2 means with a 3 x 3 bilateral window per level (colour tolerance 0.25 in square-root units: a textured background has many colours, so the guide stops only large differences); a level is trusted where the readings in its window weigh 0.25, finer levels first, the rest filled from the coarser level by a guided upsampling. The result per pixel is the mean and variance of the log radius and the cover; `c = cover / (1 + variance / 0.12)`.
- **Step 5.** `b = 0` where `c < 0.3`, in the anchor (radius 0.05 half-diagonals, in focus with `c = 1`), within 2 px of a strong sharp edge (a reading of a radius under half the smallest, 5 noise units), and where the blurred region (4-connected) is smaller than two discs of its own mean radius and holds no ball. The map holds `b` in half-diagonals; the largest `b` (held at `b_cap`) sets the stretch's halo. The planes are `active` (reshaping on) when at least 50 edges were measured (readings of weight 0.1 plus the balls' perimeters) and 2 % of the frame has `b ≥ 2·b_min` with `c ≥ 0.5`.

The report says "little out-of-focus background" when less than 2 % of the frame has `b ≥ 2·b_min` with confidence. Its tests use synthetic scenes with a known blur per region (section 14). The owner allows a learned model here ("Bokeh map extraction can use ML if needed, of course"). No pretrained model qualifies: every defocus-map and blur-detection model found has a copyleft or missing code licence, or weights trained on data with no grant or a non-commercial one (`research/bokeh-map-2026-10-24.md`). The learned route is therefore a small model of our own (about 2 M parameters, a crop-net-style encoder with a small decoder, input the lens plane, output log blur radius and a confidence), trained on synthetic defocus with a known blur made from licence-clean sharp photos (Open Images, CC BY 2.0, with its CC BY 4.0 masks; DIODE, MIT), bundled and run in Rust like the crop model (section 11). The owner, 2026-10-24: "Our blur map model can be bundled with Yellowmoon": its weights (about 1.5 M parameters, 6 MB in f32) go in `data/models/bokeh/` with a model card, as the crop model's do, with no download. The classical detector above is the first version, the fallback, and the baseline the model must beat. Research-only blur sets (CUHK, DUT-DBD, RTF) may measure both locally (the owner, 2026-10-24); they live only in the git-ignored `testdata/`, are never committed and never trained on.

**As built, the learned map's runtime** (`ym_ml::bokeh_net`, `crates/ym-ml/src/bokeh_net.rs`; not called by `LensPlanes` yet, the classical detector is still what runs):

- **The network.** `training/bokeh/export.py`'s file, `data/models/bokeh/bokehnet.safetensors` with a `SHA256SUMS` (`BokehNet::load_bundled(data_root)`; `from_file` and `from_bytes` skip or take a checksum), batch norms folded, 1,522,122 parameters, 3,364 multiply-adds per input pixel. The architecture is read from the header (stem, blocks, taps, decoder), and a file whose encoding constants, activation, strides or tensor shapes differ from what the runtime implements is refused at load (`MlError::Unsupported`, `Weights`, `TensorShape`). All maths is `f32` on the CPU in a fixed order (bias first, then input channels or taps in index order; the global mean in `f64`), with no fused multiply-add and `libm` for `exp` and `log2`, so the bits are the same on every run, on any number of threads (one task per output row) and, as far as IEEE 754 goes, on every platform.
- **From the plane to the map** (`BokehNet::map(&PlanarRgbRef) -> BokehMap`, one call, no half-diagonal to pass: `b` is a ratio). The lens plane (planar camera RGB, scene linear, any scale; 1024 to 2047 px long) is box-reduced by exact area weights to a long edge of 1024 px (a plane that is not longer is used as it is, never enlarged), encoded as `training/bokeh/encode.py` does (median-normalised, `log2(1 + v / K) / 16`), padded with replicated edge pixels to a multiple of 32, run, and the answer cropped to `⌈w/2⌉ × ⌈h/2⌉` cells (a 1024 x 683 plane gives 512 x 342). `b = max(r, 1 px) / hd`, `hd` the half-diagonal of the reduced plane in its pixels before padding; `c = exp(−β / 0.25)` with `β = exp(clamp(s, −6, 3))`. The map is raw: the rules of step 5 (`b = 0` where `c` is low, below `b_min`, near a sharp edge, in the anchor) stay with the caller, and the network is not told the focus point. `BokehMap::sample(u, v)` (fractions of the plane), `sample_px(x, y, w, h)` and `resample(w, h)` read it bilinearly at any plane size, held at the edge beyond the outer cell centres.
- **Measured** (the export of a 1,000-step smoke run, `testdata/bokeh-smoke/`; to be measured again with the final weights). Parity with PyTorch's folded network on the fixture's four cases (three integer patterns and one linear plane): largest difference of `(μ, s)` 2.4e-6 (limit 1e-4); the encoder's probes 4.7e-10 (limit 1e-6); a whole plane through `map` agrees with the fixture's `b` (in pixels) and `c` to 1.1e-5. Two runs and 1, 2, 3 and 5 threads give identical bits, on planes that need padding and on one that is reduced. Speed on a 4-core machine shared with other builds (load 6 to 9), workspace crates at the dev profile's `opt-level = 1`, so a release build is several times faster (not measured): 1024 x 683 plane 1.2 s (reduction 26 ms, encoding 51 ms, network 1.1 s, conversion 11 ms), on one thread 2.2 s; 1512 x 1006 plane 1.5 s; loading the weights 60 to 70 ms. Where the network's time goes: the stem 8 %, the encoder's twelve blocks 63 % (the last three, at stride 32 with 224 and 416 channels on 22 x 32 cells, alone take 25 % of the network's time for a thousandth of its pixels: a kernel that handles several rows per task would help there), the decoder 28 % (its last step, at half resolution, 14 %), the head under 1 %. A GPU path is not needed now.
- **Tests.** `crates/ym-ml/tests/bokeh_net.rs` takes the weights from `data/models/bokeh/` with the fixture `crates/ym-ml/tests/data/bokehnet_parity.json` (written by `export.py --fixture`) when both exist, else from `testdata/bokeh-smoke/` (`bokehnet.safetensors`, `parity.json`), else prints a note and passes. The dev example `crates/ym-ml/examples/bokeh_maps.rs` runs the network on the planes `lens_sheet.rs --dump-planes` writes and draws `model_b.png` and `model_c.png` in the detector's colour scales, for a side-by-side look.
- **Plugging it in** (`LensPlanes::build`): load one `Arc<BokehNet>` per session beside the crop model (`Core` reads `<data_root>/models/bokeh`), make the map from the plane `build` already has (`BokehNet::map`), resample it onto the detector's grid (`BokehMap::resample`) to fill the same `blur` and `confidence` rows, and then run step 5's rules and the ball search on top; without the weights, or when the model fails the go/no-go of `training/bokeh/README.md`, the classical detector stays.

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
- **Targets** (Apple M1, 50 MP, Metal): the planes in at most 0.5 s after a change of lens or photo (off the foreground worker; the fit view shows the lens when they are ready); a slider move in the fit view in at most 100 ms once they exist; a 100 % tile at most twice the cost of the raw tile; an export at most 4 s longer. CI on Linux measures the CPU and lavapipe as a guard against regressions only; the Mac numbers are the owner's check (section 13). **Status:** L2 and L3 are kernels with the parity above; L1 is on the host; the Mac numbers are not measured.

## 10. The app

- IPC: `lens_list` → `LensInfo[]` (read once). The render report carries `lens: { reshaped, reason }`, which the note reads ("little out-of-focus background"). As built: `RenderReport::lens` is `ym_lens::ReshapeReport`, and the `render_stats` event carries it as `lens: { reshaped: boolean, reason: "not_asked" | "little_blur" | null } | null` (`null` without a lens). A plan that does not ask for reshaping (no Anamorphic, and Edges on a lens without swirl) says `not_asked` whatever planes were made for an earlier setting; a frame served from the frame cache says what the photo's planes say.
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

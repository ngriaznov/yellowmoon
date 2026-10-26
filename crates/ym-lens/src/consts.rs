//! Constants of the lens character feature (`docs/21-lens-character.md`).
//!
//! Starting values set by eye live here or in the lens data, each with a doc comment (`docs/21` section 4).

/// The sentence every lens file carries as `provenance.statement` (`docs/21` section 6).
pub const PROVENANCE_STATEMENT: &str =
    "Synthetic, computed from public lens designs, not measured.";

/// Number of field radii (`rho` = 0, 1/8, ..., 1) the spread functions are given at.
pub const RADII: usize = 9;

/// Number of wavelengths the spread functions are given at.
pub const PSF_WAVELENGTHS: usize = 8;

/// The eight wavelengths of the spread functions, in nanometres.
pub const PSF_WAVELENGTHS_NM: [f64; PSF_WAVELENGTHS] =
    [420.0, 460.0, 500.0, 540.0, 580.0, 620.0, 660.0, 700.0];

/// Samples per side of a spread function table.
pub const PSF_SIZE: usize = 33;

/// Number of samples of a spectrum on the 31-sample grid (400, 410, ..., 700 nm).
pub const GRID_SAMPLES: usize = 31;

/// First wavelength of the 31-sample grid, in nanometres.
pub const GRID_START_NM: f64 = 400.0;

/// Step of the 31-sample grid, in nanometres.
pub const GRID_STEP_NM: f64 = 10.0;

/// Samples per side of a starburst table.
pub const SPIKE_SIZE: usize = 129;

/// Samples of a ball's rim profile (`bokeh.rim`).
pub const RIM_SAMPLES: usize = 16;

/// The largest radius of a starburst table, in half-diagonals (`docs/21` section 4, L3).
pub const STARBURST_MAX_RADIUS: f64 = 0.08;

/// Largest half-width of a field-optics kernel, in half-diagonals (`docs/21` section 4, L2).
///
/// The width is bent towards it, never cut at it: [`W_KNEE`].
pub const W_MAX: f64 = 0.012;

/// Where the half-width of a field-optics kernel starts to bend towards [`W_MAX`], in half-diagonals: up to
/// here it is what the lens's curves and the attachment say, beyond it it grows ever more slowly and
/// approaches [`W_MAX`] without reaching it, so the spread has no corner where a hard cap would engage
/// (`docs/21` section 4, L2, *As built (the field law)*).
pub const W_KNEE: f64 = 0.75 * W_MAX;

/// The share of the traced lateral colour the field optics keep (`docs/21` section 4, L2, *As built (the
/// field law)*; `data/lenses/README.md`, *Tuned by hand*).
///
/// A channel's kernel is moved, when the tables are projected on the camera, so that its centroid's offset from
/// the green channel's is this fraction of the traced one: 1 would draw the traced fringes, 0 none. The traced
/// lenses, widened to the spread the runtime draws, put red and blue 0.35 % (Vintage) to 0.39 % (Classic) of a
/// half-diagonal apart at the sides of the frame at Edges 100 with the attachment's doubling (3.1 to 3.5 px on
/// a 1500 px frame), far more than a real lens shows (a couple of pixels at the corner of 24 MP, 0.05 to
/// 0.1 %). With 0.3 the worst lens (Classic, Edges 100, Anamorphic 100) has 0.12 %. It is one number for every
/// lens, so each keeps its own lateral colour relative to the others.
pub const LATERAL_COLOR_SCALE: f64 = 0.3;

/// The table radius the middle of the long edge of the frame is given under the attachment (`docs/21` section 4, L2,
/// *As built (the field law)*).
///
/// The oval field radius puts the side of the frame past the traced lens's last radius (1.08 at Anamorphic 100 on
/// a 3:2 frame), where the lens has no data. The law reads the tables at the oval radius over a scale that puts the
/// side at this radius (at least 1, and at least the unstretched side's own radius), so that the side is as dark
/// as the traced lens is at 0.93 of its corner (its relative illumination there: Vintage 0.64, Default 0.47): dark
/// enough to read as the attachment's falloff, not so dark that a plain sky, fog or snow reads as a frame. At 1
/// the sky of the darkest test photo (46435) kept 0.47 to 0.49 of its brightness at the outer 20 px at Vintage
/// 100 / 100 / 100 against the 0.55 asked for.
pub const SIDE_TABLE_RADIUS: f64 = 0.93;

/// How far past the last table radius the corner of the frame may lie under the attachment, as a share of the
/// distance in which the traced relative illumination would fall to 0 along its end slope (`docs/21` section 4,
/// L2, *As built (the field law)*).
///
/// The oval field radius puts the corner of the frame past 1 (1.12 to 1.22 at Anamorphic 100), where the lens
/// has no data. The law carries the end slope straight on there (a falloff whose slope never lessens, so
/// that no band appears on a flat field); a straight line reaches 0, so the radius is scaled until the corner
/// is no farther out than this share of the way to 0: with 0.6 a corner has at least 0.4 of the traced
/// end value. On a 3:2 frame only the Triplet, whose corner is nearly dark, is scaled by it.
pub const T_OVERSHOOT: f64 = 0.6;

/// A field-optics kernel whose half-width is below this many pixels is a copy of the input.
pub const COPY_BELOW_PX: f64 = 0.5;

/// Most taps along a side of a field-optics kernel (a budget of 17 x 17 = 289 taps a pixel).
///
/// A kernel wider than this budget at one tap a pixel reads a reduced copy of the input instead
/// ([`TAP_SPACING_MAX_PX`]); the 33 x 33 tables are resampled area-weighted to this many cells at most.
pub const MAX_TAPS_SIDE: usize = 17;

/// Largest spacing of the taps of a kernel, in pixels of the mip level they read. A kernel whose cells would be
/// further apart reads the next, 2x reduced level, so that every input pixel contributes (the bilinear
/// footprints of taps one texel apart sum to a constant) and no detail is skipped.
pub const TAP_SPACING_MAX_PX: f64 = 1.0;

/// The lens plane keeps a long edge of at least this many pixels (`LensPlan::lens_plane_scale`).
pub const LENS_PLANE_LONG_EDGE: usize = 1024;

/// Samples per side of the canvas edge the zoom is fitted on (`docs/21` section 4, L4).
pub const ZOOM_EDGE_SAMPLES: usize = 200;

/// The margin, in pixels, the canvas edge keeps inside the photo (`docs/21` section 4, L4).
pub const ZOOM_MARGIN_PX: f64 = 1.0;

/// The zoom is never larger than this; a map that needs more is held at it.
pub const ZOOM_MAX: f64 = 4.0;

/// Newton steps of the inverse map (`docs/21` section 4, L4).
pub const NEWTON_STEPS: usize = 3;

/// Tolerance of "each slice sums to 1" in the lens files.
pub const UNIT_SUM_TOLERANCE: f64 = 1e-4;

// ---------------------------------------------------------------------------------------------------------
// L3: highlight optics (`docs/21` section 4, L3)

/// The headroom `h` of the restored light, in EV: a pixel the sensor clipped is raised by `2^h` for the flares.
pub const LENS_HEADROOM_EV: f64 = 3.0;

/// How much farther above the clip level a *small* clipped light is assumed to be, in EV at most: a clipped
/// patch whose equivalent diameter is at most [`LIGHT_BOOST_FULL_HD`] holds `2^(LENS_HEADROOM_EV +
/// LIGHT_BOOST_MAX_EV)` times its clipped light for the ghosts, starbursts and streaks, and [`GLOW_BOOST_SHARE`]
/// of those extra EV for the glow and the veil. A lamp or the sun sends light 10 to 17 EV above the clip level of
/// the exposure it is photographed at (`docs/21` section 4, L3, *As built (flare strength)*), and the elements
/// that spread a light over thousands of times its own area only show when the light is that bright.
pub const LIGHT_BOOST_MAX_EV: f64 = 9.0;

/// A clipped patch with an equivalent diameter (the diameter of the disc of the same area) of at most this, in
/// half-diagonals, gets the whole boost [`LIGHT_BOOST_MAX_EV`].
pub const LIGHT_BOOST_FULL_HD: f64 = 0.025;

/// A clipped patch with an equivalent diameter of at least this, in half-diagonals, gets no boost (a clipped
/// sky or wall is a bright surface, not a light); between the two the boost falls linearly with the logarithm of
/// the diameter, 2.25 EV per doubling, so the light a patch sends into its flares, its area times its
/// brightness, stays about the same from one to the other.
pub const LIGHT_BOOST_NONE_HD: f64 = 0.4;

/// The most light, in clip levels times half-diagonals squared (the mean over the channels, summed over the
/// plane), that the small lights' extra headroom may add to a photo's highlights. A photo of one sun, a lit
/// window or a road of lamps adds 1 to 3; one of forty fairy lights, a floodlit palace or a bright window with
/// a dog in it 20 to 80, and their ghosts and glow would turn the picture to haze. Above the cap every light's
/// extra is reduced by the same factor.
pub const LIGHT_BOOST_ENERGY_CAP: f64 = 3.0;

/// The share of the boost, in EV, that the restored light of the glow and the veil gets (the glow's own
/// fraction is a tuned artistic amount that already assumes a bright light).
pub const GLOW_BOOST_SHARE: f64 = 0.5;

/// At Flares 100 the ghosts of a lens together take at least this fraction of the light of a light: a lens whose
/// data hold less (a coated lens: Default 1.2e-4, Vintage 3.4e-4) has them all scaled by `floor / sum`, at most
/// [`GHOST_GAIN_MAX`], so the lens as built is a lens whose coatings are somewhat worse than its design's. The
/// proportions between its ghosts stay the data's. Twelve ghosts of two single-layer surfaces (1.3 % each) hold
/// 2e-3 together, of two multi-layer ones (0.3 % each) 1e-4, of two bare surfaces (4 %) 2e-2. The data are
/// never lowered.
pub const GHOST_FRACTION_FLOOR: f64 = 0.0012;

/// The most a lens's ghosts are scaled up to the floor ([`GHOST_FRACTION_FLOOR`]): Default and Detuned (1.2e-4)
/// and Vintage (3.4e-4) get three times their data's light, Sonnar (8.6e-4) 1.4 times, Classic (1.4e-3) and the
/// uncoated Triplet (5e-3) and Petzval (2.7e-2) are as they are.
pub const GHOST_GAIN_MAX: f64 = 3.0;

/// The same floor for the attachment's own oval ghosts at Anamorphic 100 (data 2.3e-4).
pub const ATTACHMENT_GHOST_FLOOR: f64 = 0.0006;

/// At Flares 100 the glow of a lens takes at least this fraction of the light (a veiling glare that the
/// literature rates "usually good", 1.5 to 3 %); a lens whose data hold less has its glow raised to it.
pub const GLOW_FRACTION_FLOOR: f64 = 0.02;

/// At Flares 100 the starburst of a lens takes at least this fraction of the light of a compact light.
pub const STARBURST_FRACTION_FLOOR: f64 = 0.02;

/// The clip test on the picture's own pixels: `f` ramps from 0 to 1 as `min_c(X_c / g0_c)` goes from the first
/// to the second value (per tile, at the tile's scale).
pub const CLIP_RAMP: (f64, f64) = (0.95, 0.98);

/// The clip test on the lens plane, where binning mixes clipped and unclipped photosites.
pub const CLIP_RAMP_PLANE: (f64, f64) = (0.90, 0.97);

/// A clipped pixel whose channels, in clip levels, differ by less than this share of the largest are at the clip
/// pattern (`g0`: a colour cast of the white balance, which highlight recovery removes) and are restored as a
/// neutral (`l3::restore`).
pub const CLIP_PATTERN_SPREAD: f64 = 0.1;

/// A channel counts as saturated (its value is a lower bound of the light, not a measurement) with a weight that
/// ramps from 0 to 1 as its value, in its own clip levels, goes from the first to the second number. The channel
/// with the smallest `g0` (green) saturates first for a white light, and the others are still measuring: the
/// light's colour is then the others' with the saturated channel raised (`l3::restore`, *the light's colour*).
pub const LIGHT_SATURATION_RAMP: (f64, f64) = (0.90, 1.0);

/// A pixel with a saturated channel counts as a clipped *white* light (it can hold light the sensor did not
/// record: [`FIELD_HEADROOM_SHARE`]) when the mean of the channels that measure is above the saturated channel by
/// this share of it or more, with a ramp from 0 up to it. A white surface that only just clips green (the
/// others at its level) and a coloured lamp (the others under it) are not.
pub const LIGHT_WHITE_RAMP: f64 = 0.1;

/// The share, in EV, of the boost a small clipped light is assumed to hold above its clip level
/// ([`LIGHT_BOOST_MAX_EV`] and `l3::lights::boost_ev` give it by the size of the light, and
/// [`LIGHT_BOOST_ENERGY_CAP`] reduces it for a photo of many lights) that L1 and L2 spread: the light the glow
/// takes ([`GLOW_BOOST_SHARE`]), since they spread it wide. A star the sensor clipped holds far more light than
/// its clipped core says, and the ring of unclipped, fringed pixels round it would otherwise carry more of
/// it than the core does (`docs/21`, *As built (the light's colour, the headroom of a clipped light)*).
pub const FIELD_HEADROOM_SHARE: f64 = GLOW_BOOST_SHARE;

/// The most light, in clip levels of the lens plane, that the headroom of a clipped light adds to one pixel
/// ([`FIELD_HEADROOM_SHARE`]): an upper bound on what the lens can make of a light in one pixel, a few tens of
/// times the clip level, where a star of a few pixels has 20 to 30 and a street lamp with the brightest recovered
/// core 60 and more without it. The bound keeps the peak the spread leaves round a light (a pixel beside it
/// where the kernel is narrow) a size a resampling after the lens does not ring below zero.
pub const FIELD_HEADROOM_MAX: f64 = 40.0;

/// The field stages read L3's estimate of a light's boost ([`LIGHT_BOOST_FULL_HD`], [`LIGHT_BOOST_NONE_HD`]) at
/// this many times the patch's diameter: L2 spreads the light of a patch by the lens's spread functions, which
/// reach tens of pixels, so only a patch of a few pixels (a star, a lamp, a glint) is a point whose clipped
/// core hides light that the spread would carry, and a window or a flower of a few tens of pixels is a surface
/// whose interior the spread does not change (full boost up to 0.006 half-diagonals across, none from 0.1).
pub const FIELD_LIGHT_SIZE_FACTOR: f64 = 4.0;

/// How much of the extra light the field stage spreads (the headroom of a clipped light and the light's colour,
/// `light::l2_input`) a pixel is given, the larger of two shares, each a smoothstep (none up to the first number,
/// all of it from the second).
///
/// The first is how much the extra that the spread brings to the pixel is of the picture that it brings: the
/// largest over the channels of `|gather(extra)| / gather(picture)`, from [`EXTRA_INFLOW_RAMP`]. Beside a star on a
/// dark ground the extra is 10 to 1000 times the picture's light and arrives whole; in a bright area the extra
/// only moves from pixel to pixel, a lift of a fraction of the picture, and the pixel is as the picture gives it.
///
/// The second is how much the spread changes the pixel's own picture: the largest over the channels of
/// `|gather(picture) - picture| / picture`, from [`EXTRA_CHANGE_RAMP`]. A pixel the spread changes by a half or
/// more (the dark ground and the dim edge round a clipped light, the pixels of a small one) has the light of its
/// neighbours in their own colour whatever share of the picture that is (a light at the clip pattern, whose green
/// is raised by most of its own value), where a pixel the spread changes by a quarter at most (the middle and
/// the edge of a window, whatever the width of the kernel; a bright pixel the look's highlight curve shows
/// unchanged) keeps its own light as recorded.
pub const EXTRA_INFLOW_RAMP: (f64, f64) = (0.5, 2.0);

/// The ramp of the second share, the change of the picture: see [`EXTRA_INFLOW_RAMP`].
pub const EXTRA_CHANGE_RAMP: (f64, f64) = (0.25, 1.0);

/// How wide the lens's spread must be at a pixel for the extra light to reach it, as a share of the half-diagonal
/// (the kernel's half-width over it): none up to the first number, all of it from the second, a smoothstep
/// between. Where the kernel is a pixel or two (the middle of the frame, a small Edges amount) the lens moves
/// the light of a clipped light no farther than its neighbours and the extra lands as a peak a hundred times
/// them, which a resampling of the picture after the lens rings round, and the ring goes below zero beside it:
/// a black or coloured pixel in a white light. There the lens leaves the light as the picture records it.
pub const EXTRA_KERNEL_RAMP: (f64, f64) = (0.001, 0.003);

/// How soft the headroom is, as a share of the half-diagonal: it is read from the pyramid level whose texels are the
/// largest power of two of pixels not above this (at least 1 px) above the lens's own kernel's. A light's headroom
/// is a peak of a few pixels; spread by the kernel alone it keeps that peak where the kernel is small, a hard white
/// sprite whose edge falls from clipped to sky in a pixel or two and whose steepness the resampling after the lens
/// rings below zero (black pixels inside and round it); read from texels of a few pixels it is spread over them
/// first, the same light in a wider, softer halo (`field::extra_level_bias`). The light's colour is not read
/// softer: it cancels the clip pattern's cast pixel by pixel against the picture's own spread.
pub const EXTRA_SOFTNESS: f64 = 0.005;

/// The least light, in clip levels of the lens plane, that the shares of [`EXTRA_INFLOW_RAMP`] and
/// [`EXTRA_CHANGE_RAMP`] divide by (a black pixel is not given every faint tail of a light).
pub const EXTRA_CHANGE_FLOOR: f64 = 1e-3;

/// L1 left a pixel as it is when its output is within this share of a clip level of its input in every
/// channel (1 at the first number, 0 from the second, a smoothstep between): the pixel's own light stays where
/// it is, and the headroom of a clipped light is spread from it (`light.rs`). The ramp lies just above what an
/// untouched pixel differs by (nothing: L1 passes it on bit for bit, on the CPU and on the device alike): the
/// CPU and the device reshape the pixels L1 does change within a few per cent of each other, and a wider ramp
/// gives the two different shares of the headroom there (3 codes of the lens picture against the limit of 2
/// at 1e-4 to 1e-3, 40 at 1e-3 to 3e-2; 1 at this ramp).
pub const L1_UNCHANGED_RAMP: (f64, f64) = (1e-5, 1e-4);

/// The highlights `H` are the restored light weighted by a soft knee that is 0 at this fraction of the clip
/// level and 1 at the clip level.
pub const KNEE_START: f64 = 0.5;

/// The tail plane is the restored lens plane halved until its long edge is at most this many pixels.
pub const TAIL_PLANE_LONG_EDGE: usize = 512;

/// The glow's core is at most this long, in half-diagonals.
pub const GLOW_CORE_MAX_HD: f64 = 0.0015;

/// The glow's core is at most this long, in pixels of the full-resolution frame.
pub const GLOW_CORE_MAX_PX: f64 = 8.0;

/// The core's window is 1 up to this fraction of the core radius and falls to 0 at the radius (cosine).
pub const GLOW_CORE_FLAT: f64 = 0.7;

/// Nodes of the glow's radial tables (the tail runs to `flare.glow.reach`).
pub const GLOW_TABLE_NODES: usize = 16384;

/// The glow's tail is cut smoothly over the last part of its reach: it is 1 up to this fraction of the reach and
/// falls to 0 at the reach (cosine).
pub const GLOW_TAPER_START: f64 = 0.85;

/// The veiling kernel is flat up to this fraction of `flare.veil.radius` and falls to 0 at the radius (cosine).
pub const VEIL_FLAT: f64 = 0.6;

/// The veil is a haze over the whole frame, not light from one place, so its colour is the picture's average
/// colour and not the lights': this share of it is the mean of its three channels (1: a neutral grey haze, 0:
/// the picture's average colour). The glow's tail round a light keeps the light's colour.
pub const HAZE_NEUTRAL: f64 = 1.0;

/// The glow's tail is the halo of a light up to this radius (half-diagonals) and haze from [`GLOW_HAZE_TO_HD`] on,
/// with a smoothstep between: the far part of the tail joins the veil in being neutral (`HAZE_NEUTRAL`), the
/// near part keeps the light's colour (a sodium lamp has an orange halo, and a dark trunk a hundred pixels
/// from a warm sun is not an orange patch).
pub const GLOW_HAZE_FROM_HD: f64 = 0.05;

/// See [`GLOW_HAZE_FROM_HD`]: the radius from which the glow's tail is haze.
pub const GLOW_HAZE_TO_HD: f64 = 0.15;

/// A pixel of the lens plane is a source pixel when its highlight, in clip levels, reaches this value in some
/// channel.
pub const SOURCE_FLOOR: f64 = 0.02;

/// Sources larger than this, in half-diagonals, are split into cells of this size.
pub const SOURCE_CELL_HD: f64 = 0.02;

/// The source list holds at most this many sources (the brightest, ties by position).
pub const SOURCES_MAX: usize = 256;

/// Sources at most this large, in half-diagonals, get a starburst.
pub const COMPACT_EXTENT_HD: f64 = 0.004;

/// The standard deviation, in pixels, below which a ghost's edge is not made sharper: an edge narrower than a
/// pixel cannot be drawn, and this keeps the drawn energy exact on any pixel grid.
pub const EDGE_FLOOR_PX: f64 = 1.0;

/// The light, in clip levels, that the ghosts left undrawn may add up to at the very brightest place of the
/// picture: the sum of their peak densities. The faintest ghosts are not drawn while that sum stays under this
/// (2^-18, a quarter of the step of a 16-bit code at the clip level), so a photo with hundreds of lights does
/// not pay for thousands of ghosts that cannot be seen. The choice depends on the plan and the planes only.
pub const GHOST_PRUNE_BUDGET: f64 = 1.0 / 262_144.0;

/// The least width of a ghost's edge, as the standard deviation of its Gaussian step in ghost radii (the lens
/// data give `soft`, some 0.03: a near-hard disc). The 10 to 90 % width of the step is 2.56 standard
/// deviations, so 0.10 is a ramp of 26 % of the radius (at least a quarter): a ghost is a defocused image of the
/// light, never a sticker with an edge.
pub const GHOST_SOFT_MIN: f64 = 0.10;

/// The most the normalised tint of a ghost's coating may differ between channels (largest over smallest). The
/// data's coatings, the product of two reflectance spectra, are far more saturated than a ghost looks (6 to 23
/// times between the channels); the tint is brought to this by raising it to a power below 1 (`l3::items`), so
/// a coating keeps its hue and its ghosts come out pale.
pub const GHOST_TINT_MAX_RATIO: f64 = 1.2;

/// The brightest a ghost of one light gets, in clip levels at Flares 100, whatever the light is (the same ghost
/// of the cells of a large light counts together): a ghost brighter than this is brought down to it. A small
/// clipped light is assumed thousands of times brighter than the clip level (`LIGHT_BOOST_MAX_EV`), which spread
/// over a ghost's area still made discs brighter than the scene around them. Applies to ghosts of apothem
/// [`GHOST_FULL_RADIUS`] and larger; smaller ones get less (`l3::items::ghost_peak_max`).
pub const GHOST_PEAK_MAX: f64 = 0.003;

/// The most light a ghost carries, in clip levels times half-diagonals squared at Flares 100: its brightest value
/// may not pass this over its area `pi s^2`, so a ghost of apothem over 0.25 is fainter than [`GHOST_PEAK_MAX`] in
/// proportion to one over its area (a quarter of it at an apothem of 0.5, 6 % at 1).
pub const GHOST_LIGHT_MAX: f64 = 0.000_59;

/// The apothem, in half-diagonals, from which a ghost may be as bright as [`GHOST_PEAK_MAX`].
pub const GHOST_FULL_RADIUS: f64 = 0.1;

/// The least share of [`GHOST_PEAK_MAX`] a small ghost is allowed.
pub const GHOST_SMALL_SHARE: f64 = 0.25;

/// The coating of ghost `i` is shifted along the wavelengths by a number of nanometres between these two (a
/// low-discrepancy sequence of the ghost's index, `l3::items::coating_shift_nm`): the surfaces of a lens are not
/// coated alike and a ray meets a coating at an angle, which moves its reflectance minimum toward blue, so the
/// ghosts of one light come out amber, magenta, blue and cyan in turn (pale, [`GHOST_TINT_MAX_RATIO`]) and do
/// not all read as one lilac.
pub const GHOST_COATING_SHIFT_NM: (f64, f64) = (-70.0, 120.0);

/// A ghost is brighter on the side that faces the optical axis by this share of its mean, and fainter on the
/// other by the same (a linear shading across its apothem): the image of the aperture is vignetted by the lens's
/// barrel, which is never even. It integrates to nothing over the ghost, so the light the ghost carries is the
/// same, and the brightest value is `1 +` this times the profile's (`l3::items`).
pub const GHOST_SHADE: f64 = 0.35;

/// The peak of a streak at its light, in clip levels, for a light that is bright and compact, per unit of the
/// attachment's streak `fraction` (data version 3: 0.004, so 0.4): the streak is a line two fifths of the clip
/// level bright at its core at most, never as bright as the light it comes from.
pub const STREAK_PEAK_PER_FRACTION: f64 = 100.0;

/// The light's energy (`mean_c(E_c / g0_c)`, clip levels times half-diagonals squared) at which a streak reaches
/// half its peak: the peak is `peak e / (e + STREAK_E_HALF)`, a law that saturates, so a light of ten thousand
/// pixels does not make a streak a thousand times stronger than one of ten, and a star (0.01) makes a faint one.
pub const STREAK_E_HALF: f64 = 0.02;

/// Lights with less energy than this (same unit) make no streak.
pub const STREAK_E_MIN: f64 = 0.004;

/// The least standard deviation of a streak across the axis, in half-diagonals (2.5 px of the 1500 px fit
/// frame, a line 6 px wide at half its height): a soft line however small the light is.
pub const STREAK_SIGMA_MIN_HD: f64 = 0.0028;

/// The streak is wider than its light's own spread by this share of the light's extent (the larger side of its
/// bounding box, half-diagonals), added in quadrature: a big light has a thicker streak than a point.
pub const STREAK_SIGMA_EXTENT: f64 = 0.1;

/// A streak whose light is wider than this (standard deviation across the axis, half-diagonals) is a band: its
/// peak, and the energy it counts with, are divided by `1 + (sigma / STREAK_SIGMA_REF_HD)^2`, so a large bright
/// area makes a soft faint band and a point makes a line.
pub const STREAK_SIGMA_REF_HD: f64 = 0.04;

/// A light whose light is spread wider than this (standard deviation across the axis, half-diagonals) starts to be a
/// surface, not a light, and its streak fades; from [`STREAK_SURFACE_END_HD`] on it makes none (a smooth step
/// between). A large bright area (a lit wall, a window, blurred pavement) does not draw a line or a band across
/// the picture. The spread is of the light (`H`), so a bright sun in a hazy sky counts by its core.
pub const STREAK_SURFACE_START_HD: f64 = 0.03;

/// See [`STREAK_SURFACE_START_HD`].
pub const STREAK_SURFACE_END_HD: f64 = 0.055;

/// Across the axis a streak is two Gaussians: the core (the light's width) and a halo `STREAK_HALO_WIDTH` times as
/// wide that holds this share of its light, so the line has no sharp feet.
pub const STREAK_HALO_SHARE: f64 = 0.2;

/// The halo's width over the core's (standard deviations). See [`STREAK_HALO_SHARE`].
pub const STREAK_HALO_WIDTH: f64 = 4.0;

/// The reach of a streak grows with the logarithm of the light's energy: `STREAK_REACH_AT_E0` (of the attachment's
/// reach) at the energy `STREAK_REACH_E0`, `STREAK_REACH_PER_DECADE` more for every factor 10, at least
/// `STREAK_REACH_MIN` and at most 1. The law passes through the smallest star that draws one (0.3 at an energy of
/// 0.014) and the moon of a dark night sky (0.82 at 0.5), and reaches 1 at an energy of 2.6, a sun or a lit tree:
/// before (0.5 and 0.5 per decade) every light above 0.45 had the full reach, and the moon's streak added 2 codes
/// or more over 57 % of the frame's width at Anamorphic 100 (60 % at Vintage 100 / 100 / 100), now 48 % (49 %)
/// (`docs/21`, *As built (the moon's reach)*).
pub const STREAK_REACH_E0: f64 = 0.045;
/// See [`STREAK_REACH_E0`].
pub const STREAK_REACH_AT_E0: f64 = 0.47;
/// See [`STREAK_REACH_E0`].
pub const STREAK_REACH_PER_DECADE: f64 = 0.33;
/// See [`STREAK_REACH_E0`].
pub const STREAK_REACH_MIN: f64 = 0.3;

/// The part of its reach over which a streak's profile along the axis is cut to 0 by a cosine, counted from the
/// reach inwards: the profile starts to fall to 0 at `STREAK_FLAT` of the reach (a smooth end, with a zero slope,
/// so the streak does not stop anywhere).
pub const STREAK_FLAT: f64 = 0.35;

/// Only this many lights of a frame, the brightest, make a streak: the streaks are the signature of the few
/// brightest lights, and fifty lamps do not draw fifty lines.
pub const STREAK_MAX_LIGHTS: usize = 5;

/// The sum over the lights of a frame of (peak in clip levels times reach in half-diagonals) is brought down to
/// this when it is larger, every streak by the same factor, so a frame of fifty lights does not fill with lines.
pub const STREAK_TOTAL_CAP: f64 = 0.4;

/// Each streak after the brightest is weaker by this factor than the one before it (`STREAK_RANK_DECAY^rank`): a
/// frame of equally bright lights (a Christmas tree) has one clear streak and fainter ones, never a ladder.
pub const STREAK_RANK_DECAY: f64 = 0.75;

/// A light's streak is weaker by `(its energy / the strongest light's)^STREAK_REL_POWER` (energies as the peak law
/// counts them, over the width penalty): a bright gap beside the sun does not draw a bar as strong as the sun's.
pub const STREAK_REL_POWER: f64 = 1.0;

/// A streak whose peak would be under this (clip levels) is not drawn: it is under what a dark sky shows.
pub const STREAK_PEAK_FLOOR: f64 = 0.03;

/// Two lights merge into one streak when their rows (the positions across the axis) are within
/// `STREAK_MERGE_BASE_HD + STREAK_MERGE_SIGMAS` of their widths' sum, and their positions along the axis within
/// `STREAK_MERGE_ALONG_HD`: lamps in a row, or the bright gaps of one patch of sky, are one band, not a set of
/// lines. The streak is at the brightest light's place along the axis and the energy-weighted row across.
pub const STREAK_MERGE_BASE_HD: f64 = 0.012;

/// See [`STREAK_MERGE_BASE_HD`].
pub const STREAK_MERGE_SIGMAS: f64 = 1.5;

/// See [`STREAK_MERGE_BASE_HD`].
pub const STREAK_MERGE_ALONG_HD: f64 = 0.35;

/// A light inside the bounding box of a larger one, this many times its size or more, belongs to it (a glint on
/// a lit window): it makes no streak of its own.
pub const STREAK_ABSORB_RATIO: f64 = 4.0;

/// The colour of a streak turns from the light's own to the tail's over this share of its reach (a smooth step).
pub const STREAK_TURN: f64 = 0.4;

/// The tail's colour is the coating's tint (a blue) at this share of its chroma, between white (0) and the tint (1):
/// the streak is never a saturated blue, whatever the coating.
pub const STREAK_TAIL_CHROMA: f64 = 0.5;

/// Ghosts whose edges and rim change slowly are drawn on a grid of nodes `L` pixels apart (`L` is 2, 4, 8, 16 or
/// 32, the largest that keeps this many nodes across the narrowest feature) and read back by bilinear interpolation:
/// a ghost covers most of the frame, and evaluating it at every pixel of every one of hundreds of sources is
/// what made a photo with many lights take minutes.
pub const COARSE_NODES_PER_FEATURE: f64 = 2.5;

/// The rim curve (`RIM_SAMPLES` nodes over the ghost's radius) is smooth at this fraction of the radius.
pub const COARSE_RIM_FEATURE: f64 = 0.1;

/// Ghosts whose pixel rectangle is smaller than this many pixels are drawn at every pixel: it costs little.
pub const COARSE_MIN_AREA_PX: usize = 16_384;

/// The f-number the starburst assumes when the photo has none.
pub const STARBURST_DEFAULT_F_NUMBER: f64 = 4.0;

/// The f-number below which the starburst has no strength.
pub const STARBURST_MIN_F_NUMBER: f64 = 2.0;

/// The drawn items stop where the edge profile is below this fraction (about seven standard deviations).
pub const EDGE_REACH_SIGMAS: f64 = 7.0;

// ---------------------------------------------------------------------------------------------------------
// L1: bokeh reshaping and the bokeh detector (`docs/21` section 4, L1, and section 7)

/// `b_min`: blur radii below this, in half-diagonals, are left alone (about 2.5 px on a 1024 px lens plane).
pub const B_MIN: f64 = 0.004;

/// `b_cap`: the blur radius the stretch uses is held at this many half-diagonals.
pub const B_CAP: f64 = 0.03;

/// The ball list holds at most this many balls (the brightest first).
pub const MAX_BALLS: usize = 512;

/// A line of the stretch has at most this many samples.
pub const MAX_STRETCH_TAPS: usize = 65;

/// The joint bilateral upsampling of the blur map reads the tile's own pixels within this many pixels of a
/// pixel at the tile's scale (`docs/17` D4).
pub const UPSAMPLE_RADIUS_PX: usize = 8;

/// With fewer measured edges than this, nothing is reshaped ("little out-of-focus background").
pub const MIN_EDGE_SAMPLES: usize = 50;

/// The blur that counts towards [`MIN_BLUR_AREA`], in `b_min`: the area at `b >= 1.5 b_min` (a disc of 5 px on a
/// 1000 px lens plane, 20 px on a 24 MP frame) with confidence. A photo whose background is blurred by less is
/// left alone: the stretch would move it by a pixel or two.
pub const MIN_BLUR_FROM: f64 = 1.5;

/// With less than this fraction of the frame at `b >= MIN_BLUR_FROM b_min` with confidence, nothing is reshaped (the
/// classical detector's rule; a learned map counts the cells it is sure of against a share of its own, see
/// `l1::detector::MAP_MIN_STRONG_AREA`).
/// (On 32 real photos, the ones with an out-of-focus background are above 2.5 %, the sharp ones below 2 %.)
pub const MIN_BLUR_AREA: f64 = 0.02;

// ---------------------------------------------------------------------------------------------------------
// L1: the per-tile pass (`l1::tile`), which the CPU reference and the kernel (`lens_reshape.wgsl`) share

/// The sharp-edge test runs only where the blur is at least this many pixels at the tile's scale.
pub const SHARP_TEST_FROM_PX: f64 = 4.0;

/// A pixel is sharp when the gradient energy of the picture exceeds that of its 3 x 3 mean by this factor. (At 1.4 the
/// crisp rims of defocused discs, edges of 2 to 3 px in a field blurred by ten or more, were flagged as detail and
/// held back as small unstretched bites in the stretched field; the edges of hair and fur lose more than 1.8 times.)
pub const SHARP_RATIO: f64 = 1.8;

/// ... and the mean gradient energy of its 5 x 5 window exceeds this many noise variances.
pub const SHARP_GATE: f64 = 16.0;

/// Dilation of the sharp places in pixels.
pub const SHARP_DILATE: usize = 2;

/// The sharp-edge test sees a ball's rim this far beyond it, in pixels (gradient, box, window and dilation).
pub const BALL_RIM_REACH_PX: f64 = 7.0;

/// A sharp pixel in or around a ball that is darker than this fraction of the pixels three either side of it along a
/// row or a column (by more than twelve noise units of them) is a strand in front of the ball and stays protected.
pub const STRAND_VALLEY: f64 = 0.6;

/// The noise units a strand must be darker than its surroundings by (see [`STRAND_VALLEY`]).
pub const STRAND_MARGIN_NOISE: f64 = 12.0;

/// A line shorter than this, in pixels, is not gathered.
pub const MIN_LINE_PX: f64 = 0.05;

/// A ball's own light and the take-out of it fade out together over this share of the zone between the ball's radius
/// and the take-out radius (the glow), and at least a pixel: where a neighbour's light overlaps the ball's disc, the
/// part of it that is taken out is put back by the same weight, so no edge of the disc shows.
pub const BALL_GLOW_FADE: f64 = 1.0;

/// The lens's rim profile (`bokeh.edge`) modulates a ball's light up to its edge, relative to the profile's mean over
/// the disc, and fades out over this many ball radii beyond the edge (a neighbour's light that overlaps the ball's
/// disc is not shaped by this ball's profile).
pub const RIM_FADE_RADII: f64 = 0.3;

/// The lens's rim profile is measured on crisp discs; a ball whose rim is as wide as this many of its radii or more
/// (a soft glow, the highlight of an ornament) has no rim for it to shape, and a ball between this and
/// [`RIM_SOFT_TO`] takes it in part (a smoothstep). Applied to a soft ball it draws faint rings.
pub const RIM_SOFT_FROM: f64 = 0.3;

/// See [`RIM_SOFT_FROM`]: the rim width, in radii, from which the lens's rim profile is not applied.
pub const RIM_SOFT_TO: f64 = 0.8;

/// The half-length of both gathers, in units of the half-length of the box line that has the same variance: the line
/// is a biweight `(1 - u^2)^2` (variance `1/7` of the squared half-length against `1/3` for a box), so a line of the
/// same spread is `sqrt(7/3)` times as long. The weights fall to zero smoothly at both ends, so that a small change in
/// a pixel's line (its length, a sample dropping out) changes its value a little (`docs/21` section 4, L1, *As built
/// (soft gathers)*).
pub const PROFILE_SCALE: f64 = 1.527_525_231_651_947;

/// The longest tangential gather, as the half-length of its biweight line in half-diagonals: the half-length
/// `A (k sigma(rho) b + a(rho) H)` of the field swirl, `A` the Anamorphic amount (`docs/21` section 4, L1, *As
/// built (field swirl)*), is held to it, so that the halo, and with it the device's largest binding, stays bounded
/// whatever the blur and the lens (at `b_cap` and Anamorphic 100 the Petzval's swirl at the corner, 2.45, reaches
/// 0.112, which the bound cuts to 0.10; the first swirl, 1.5, reached 0.069).
pub const TANGENT_REACH_MAX: f64 = 0.10;

/// A ball is stretched along the tangent to at most this many times its width (`q_t`): beyond it a small light of
/// the outer field, whose absolute astigmatism is many of its radii, would become a line, not a lemon.
pub const TANGENT_ASPECT_MAX: f64 = 4.0;

/// A sample counts with the weight of a smoothstep of its blur over the pixel's own, from nothing at this ratio ...
pub const SAMPLE_RATIO_FROM: f64 = 0.5;

/// ... to everything at this one (a hard rule at one half tore the picture along the edge of a subject and where a
/// speck of sharp detail sat in a blurred field).
pub const SAMPLE_RATIO_TO: f64 = 0.85;

/// The sharp places (the flags the edge test sets, dilated) hold back the stretch by their density in a window of
/// `2 SHARP_SOFT_HALF + 1` pixels a side, not one pixel at a time: the density is `d`, the protection `1 - smoothstep`
/// of `d` between these two bounds, so a speck of a few pixels barely holds anything back and a region of sharp detail
/// (fur, a strand's whole length) holds back everything, with edges as wide as the window.
pub const SHARP_SOFT_HALF: usize = 7;

/// See [`SHARP_SOFT_HALF`]: the density from which the protection starts.
pub const SHARP_SOFT_FROM: f64 = 0.2;

/// See [`SHARP_SOFT_HALF`]: the density from which the stretch is held back altogether.
pub const SHARP_SOFT_TO: f64 = 0.7;

/// A pixel takes a blur from the map only when its colour (square roots, three channels) is within this distance of
/// the footprint mean of some blurred cell of confidence at least [`ACCEPT_CONFIDENCE`] around it. In a region
/// blurred by more than a cell the picture is smooth at the cell's scale, so every pixel is near its cell's mean;
/// the rim of a sharp subject (a bright bud against a dark background) is not near any, and keeps no blur.
pub const ACCEPT_COLOUR: f64 = 0.25;

/// See [`ACCEPT_COLOUR`].
pub const ACCEPT_CONFIDENCE: f64 = 0.3;

/// A pixel is a subject (a sharp thing a drawn ball does not cover) when this share of the weight of the cells around
/// it, by the same colour-guided weights as the blur, is known to be in focus.
pub const SUBJECT_SHARE: f64 = 0.5;

/// The stretch fades out towards the border of the blur map (a subject's outline, a hole where the detector
/// finds sharp detail) over this fraction of the longest line, within [`FEATHER_MIN_PX`] and
/// [`FEATHER_MAX_PX`] pixels at the tile's scale: a hard border cuts the lines of neighbouring pixels differently
/// and tears the picture along it.
pub const FEATHER_FRACTION: f64 = 0.8;

/// The ramp that fades the stretch out at the border of the blur map is the share of the feather's window that has a
/// blur, from one half (the border) to this share, smoothed (`docs/21` section 4, L1, *As built (the ramp)*): with
/// the feather at [`FEATHER_FRACTION`] of the longest line, the stretch of a pixel grows by at most about one pixel of
/// line for every pixel it lies from a sharp subject, so the halo that touches an in-focus subject is not smeared
/// at full length beside it and no straight edge shows where the smear starts.
pub const RAMP_SHARE_TO: f64 = 1.0;

/// See [`FEATHER_FRACTION`].
pub const FEATHER_MIN_PX: usize = 2;

/// See [`FEATHER_FRACTION`].
pub const FEATHER_MAX_PX: usize = 96;

/// Pixels farther than this from the edge of a gather's reach are not read: a safety margin for the bilinear
/// neighbour of the last sample and the window of the restored grain ([`GRAIN_HALF`]).
pub const GATHER_PAD: usize = 5;

/// Half-width of the window of the grain a gather gives back: a pixel's own grain is its picture minus the mean of the
/// `2 GRAIN_HALF + 1` pixels around it, square roots (`docs/21` section 4, L1, *As built (round 4)*). The grain of a
/// photo is correlated over two or three pixels (demosaicing and the reduction), and what the eye reads is the grain
/// of the output, which is the lens plane averaged by its reduction (a plane of twice the frame's size shows only
/// its coarser half), so the window has to be wide for the residual to hold all of it. Of the water of 39211, on
/// the output scale, 7 x 7 gives back 0.89 of the grain, 9 x 9 0.96, 11 x 11 0.99 and 13 x 13 1.00.
pub const GRAIN_HALF: usize = 5;

/// The grain a gather gives back is held to the noise the photo has: a residual of up to this many noise units
/// (the detector's estimate, the noise of the square-root luminance, which is a mean of the channels: one
/// channel's is about 1.6 times as large, a white-balanced red or blue more) is grain and goes back whole, ...
pub const GRAIN_GATE_FROM: f64 = 6.0;

/// ... and a larger one (the crisp rim of a defocused disc, a speck of detail: not grain) goes back less, none from
/// this many units on (a smoothstep between).
pub const GRAIN_GATE_TO: f64 = 12.0;

/// Nor is grain given back where the window around the pixel holds structure: a window whose pixels spread (standard
/// deviation of the square roots) by this many noise units or more is no longer grain alone, and the residual
/// against its mean, which a bump or a sharp detail beside it pulls to the other side (a dark dash along the edge of
/// a bright sharp thing), is not given back; nothing from [`GRAIN_SPREAD_TO`] on (a smoothstep between). The grain of a
/// photo spreads by one to two units (39211: the median 1.0, the 99th percentile 1.2 to 2.0, by channel), so this is
/// set just above it: a faint thin line (the scratch beside the ice of 56893, 4 to 8 units of residual and 3 to 6 of
/// spread) is structure, and was given back at 4 to 8, the first form, as a line.
pub const GRAIN_SPREAD_FROM: f64 = 1.5;

/// See [`GRAIN_SPREAD_FROM`].
pub const GRAIN_SPREAD_TO: f64 = 3.0;

/// A pixel in the wedge beside a sharp subject (the ramp under 1) whose square-root luminance stands above the picture
/// on both sides of it, this share of its own line away along the gather, is a thin bright thing, narrower than its
/// line, and is held back (`docs/21` section 4, L1, *As built (round 4)*): at least [`THIN_MIN_PX`] and at most the
/// feather. (Above the mean of the two sides, the bright side of every blurred edge near a subject was held back
/// as well and gained a crisp rim; 0.35 of the line left a hook at the end of the leaf of 59499, 0.5 does not.)
pub const THIN_SCALE: f64 = 0.5;

/// See [`THIN_SCALE`]: the least distance, in pixels of the tile's scale.
pub const THIN_MIN_PX: f64 = 2.0;

/// See [`THIN_SCALE`]: the excess over the brighter side (square-root units) from which a pixel starts to be held back,
/// at least [`THIN_NOISE`] noise units (the end of the leaf of 59499 stands 0.05 above its ground) ...
pub const THIN_FROM: f64 = 0.009;

/// ... and the noise units that make the least start of the hold (the detector's estimate, see [`GRAIN_GATE_FROM`]).
pub const THIN_NOISE: f64 = 6.0;

/// ... and the hold is complete at this many times the start.
pub const THIN_SPAN: f64 = 4.0;

/// The least colour noise, in square-root units, the upsampling and the sharp-edge test assume.
pub const NOISE_FLOOR: f64 = 0.0015;

/// The least width of the upsampling's colour guide, in square-root units.
pub const GUIDE_SIGMA_MIN: f64 = 0.06;

/// Noise units in the width of the upsampling's colour guide.
pub const GUIDE_SIGMA_NOISE: f64 = 3.0;

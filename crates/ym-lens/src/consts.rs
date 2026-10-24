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
pub const W_MAX: f64 = 0.012;

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

/// The clip test on the picture's own pixels: `f` ramps from 0 to 1 as `min_c(X_c / g0_c)` goes from the first
/// to the second value (per tile, at the tile's scale).
pub const CLIP_RAMP: (f64, f64) = (0.95, 0.98);

/// The clip test on the lens plane, where binning mixes clipped and unclipped photosites.
pub const CLIP_RAMP_PLANE: (f64, f64) = (0.90, 0.97);

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

/// With less than this fraction of the frame at `b >= 2 b_min` with confidence, nothing is reshaped.
pub const MIN_BLUR_AREA: f64 = 0.02;

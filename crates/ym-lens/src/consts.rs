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

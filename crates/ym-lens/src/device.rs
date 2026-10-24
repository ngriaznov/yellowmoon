//! The lens's plan in the form the GPU kernels take (`docs/21` sections 5 and 9, `docs/17` section 11).
//!
//! The kernels in `ym-gpu` (`lens_field`, `lens_flare`, `lens_reshape`) compute what the CPU references in
//! this crate compute, in `f32`. They take their numbers from here, so the references and the kernels read one
//! source: plain data (no references into the plan), flat tables the device uploads once per photo, and the
//! constants of the CPU planners ([`crate::consts`]), which the kernels' WGSL is written from.

/// Four-lane vectors in the header of the field tables (`FieldTablesData::data`): the radii, `w(rho)` and
/// `T(rho)` padded to 12 values each (3 vectors apiece), then 27 centroids `(x, y, 0, 0)` indexed
/// `[radius * 3 + channel]`.
pub const FIELD_HEADER: usize = 36;

/// The sizes `n` of the cell tables, `3, 5, ..., MAX_TAPS_SIDE`, in the order they follow the header.
pub const FIELD_TABLE_SIZES: usize = crate::field::SIZES;

/// Offset, in four-lane vectors from the end of the header, of the cell tables of side `n` (odd, 3 to
/// [`MAX_TAPS_SIDE`](crate::consts::MAX_TAPS_SIDE)): each holds `RADII * n * n` vectors, one per radius and
/// cell, the three channels' weights and a zero.
#[must_use]
pub const fn field_table_offset(n: usize) -> usize {
    let mut m = 3;
    let mut offset = 0;
    while m < n {
        offset += crate::consts::RADII * m * m;
        m += 2;
    }
    offset
}

/// The lens's spread functions on a camera's three channels, flattened for the device: a header of
/// [`FIELD_HEADER`] four-lane vectors, then the cell tables of every size from
/// [`field_table_offset`]. They depend on the lens and the camera only.
#[derive(Clone, Debug, PartialEq)]
pub struct FieldTablesData {
    /// Names the lens, its data version and the camera: equal keys, equal tables.
    pub key: String,
    /// The vectors, `[f32; 4]` each.
    pub data: Vec<[f32; 4]>,
}

/// The field optics (L2) of one plan at one frame size: what the kernel computes a pixel's kernel from.
///
/// The pixel at integer frame position `(i, j)` has the field position
/// `u = (i + 0.5 - width / 2) / half_diag`, `v` likewise; `x_a, y_a` are `(u, v)` or `(v, u)` for an axis
/// along the buffer's y. The rest follows `docs/21` section 4, L2, and `crate::field`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FieldPass {
    /// Width of the frame at this scale in pixels.
    pub width: usize,
    /// Height of the frame at this scale in pixels.
    pub height: usize,
    /// The half-diagonal of that frame in pixels.
    pub half_diag: f64,
    /// True when the anamorphic axis is the buffer's y axis.
    pub axis_y: bool,
    /// The edge amount `e'`.
    pub e: f64,
    /// `alpha`: the stretch of the field radius along the axis.
    pub alpha: f64,
    /// The side softness's amount `a_s (q - 1)`.
    pub side: f64,
    /// `h (q - 1)`: the boost on the axis part of the channels' centroid offsets (0 without the
    /// attachment).
    pub fringe_boost: f64,
    /// The most mip levels any pixel of the frame can ask for.
    pub lmax: u32,
    /// Context the gather of a pixel reads, in pixels.
    pub gather_halo: usize,
    /// Pixels the fringe shift reads beyond the output.
    pub pad: usize,
}

impl FieldPass {
    /// Pixels of context an output rectangle needs around it (the CPU reference's `halo`).
    #[must_use]
    pub fn halo(&self) -> usize {
        self.gather_halo + self.pad
    }
}

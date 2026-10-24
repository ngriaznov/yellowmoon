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

// ---------------------------------------------------------------------------------------------------------
// L3: highlight optics

/// The most polygon edges of the aperture the flare kernel takes.
pub const MAX_APERTURE_NORMALS: usize = 16;

/// Vectors of the header of [`FlareScaleData::vectors`]: the streak's profile (`core`, `power`, `length`,
/// `flat`; the mass of the power law up to `flat` and of the whole half profile; a spare) and the cumulative
/// mass of its cut part at its 65 panel boundaries (17 vectors).
pub const FLARE_HEADER: usize = 20;

/// Vectors in [`FlareLensData::tab`] before the normal tail's nodes: the rim's samples (4) and second
/// derivatives (4), the aperture's count and roundness (1) and its normals (8).
pub const FLARE_TAB_HEADER: usize = 17;

/// Nodes of the normal tail's table: `(Q, dQ/dz, R, dR/dz)` at `z = -8 + i / 256` for `i` in `0..=4096`.
pub const NORMAL_TAIL_NODES: usize = 4097;

/// The tail plane and the spike tables of one photo (`LensPlanes::flare_plane_data`): made once, from the
/// lens plane, whatever the amounts.
#[derive(Clone, Debug, PartialEq)]
pub struct FlarePlaneData {
    /// Width and height of the tail plane.
    pub tail_size: (usize, usize),
    /// The glow and veil light per channel on the tail plane: red, green, blue planes, row-major,
    /// `3 * width * height` samples.
    pub tail: Vec<f32>,
    /// The starburst tables on the camera's channels: `3 * 129 * 129` samples, row-major, x to the right and
    /// y down.
    pub spikes: Vec<f32>,
}

/// What the flare kernel reads of a lens on a camera (`LensPlan::flare_lens_data`): the rim curve
/// (samples and second derivatives), the aperture and the normal tail's table. All in four-lane vectors
/// from [`FLARE_TAB_HEADER`] on: the table's [`NORMAL_TAIL_NODES`] nodes.
#[derive(Clone, Debug, PartialEq)]
pub struct FlareLensData {
    /// Names the lens, its data version and the camera: equal keys, equal data.
    pub key: String,
    /// The vectors.
    pub tab: Vec<[f32; 4]>,
}

/// The clip test of the restored light for the device (`docs/21` section 4, L3).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlareRestore {
    /// The clip level `g0` of each channel.
    pub clip: [f32; 3],
    /// The ramp of the clip test on the picture's own pixels.
    pub ramp: (f32, f32),
    /// `2^h - 1` for the headroom `h`.
    pub boost: f32,
}

/// The flare kernel's data for one plan shape at one scale (`LensPlan::flare_scale_data`): the glow core's
/// taps, the streak's profile and the drawn items with the bins that cull them. It does not depend on
/// Flares: [`LensPlan::flare_mix`] gives the weights of the sum.
#[derive(Clone, Debug, PartialEq)]
pub struct FlareScaleData {
    /// Width of the frame at this scale in pixels.
    pub width: usize,
    /// Height of the frame at this scale in pixels.
    pub height: usize,
    /// The half-diagonal of that frame in pixels.
    pub half_diag: f64,
    /// True when the anamorphic axis is the buffer's y axis.
    pub axis_y: bool,
    /// The glow core's reach in pixels (0: no core). This is the halo the pass needs around a rectangle.
    pub core_radius: usize,
    /// The header ([`FLARE_HEADER`] vectors), then the core's taps `(2 r + 1)^2` one vector each (red, green,
    /// blue weights and a zero), then six vectors per item (`ym_lens::l3::items`: four of numbers, one of
    /// kind, shape and grid spacing as bits (1: drawn at every pixel), one of the item's pixel rectangle
    /// `x0, x1, y0, y1` as bits).
    pub vectors: Vec<[f32; 4]>,
    /// Index of the first core tap in `vectors`.
    pub core_offset: usize,
    /// Index of the first item in `vectors`.
    pub record_offset: usize,
    /// Number of items.
    pub records: usize,
    /// The bins that cull the items: `bins_x * bins_y + 1` offsets, then the item indices of each bin in
    /// list order. A bin is `bin_px` pixels square on the absolute pixel grid.
    pub bins: Vec<u32>,
    /// Bins across.
    pub bins_x: usize,
    /// Bins down.
    pub bins_y: usize,
    /// Side of a bin in pixels.
    pub bin_px: usize,
    /// The radius of the starburst tables in half-diagonals.
    pub spike_radius: f64,
    /// The clip test.
    pub restore: FlareRestore,
    /// True when the plan draws streaks (the header holds their profile).
    pub has_streak: bool,
    /// Which grids of nodes (spacing 2, 4, 8, 16 and 32 px) hold ghosts: those ghosts are not drawn at every pixel
    /// but at the nodes, and read back by bilinear interpolation (`COARSE_LEVELS`).
    pub coarse: [bool; 5],
}

impl FlareScaleData {
    /// Pixels of context an output rectangle needs around it: the core's reach.
    #[must_use]
    pub fn halo(&self) -> usize {
        self.core_radius
    }
}

impl crate::planes::LensPlanes {
    /// The share, 0 to 1, of the extra headroom of the small clipped lights that this photo keeps: 1 unless its
    /// lights would add more than `consts::LIGHT_BOOST_ENERGY_CAP` of flare light, in which case every light's
    /// extra is reduced by this factor (`docs/21` section 4, L3, *As built (flare strength)*).
    #[must_use]
    pub fn light_boost_scale(&self) -> f64 {
        self.flare().boost_scale()
    }

    /// The tail plane and the spike tables for the device, or `None` for planes without highlight data (an
    /// identity plan's).
    #[must_use]
    pub fn flare_plane_data(&self) -> Option<FlarePlaneData> {
        crate::l3::plane_data(self)
    }
}

impl crate::plan::LensPlan {
    /// The lens's rim, aperture and normal tail for the device.
    #[must_use]
    pub fn flare_lens_data(&self) -> FlareLensData {
        crate::l3::lens_data(self)
    }

    /// The flare kernel's data for a frame of `width x height` pixels (the frame at the scale of the
    /// caller), or `None` when the flares add no light or the planes hold no highlight data.
    #[must_use]
    pub fn flare_scale_data(
        &self,
        planes: &crate::planes::LensPlanes,
        width: usize,
        height: usize,
    ) -> Option<FlareScaleData> {
        crate::l3::scale_data(self, planes, width, height)
    }

    /// `(F, 1 - a)`: the weight of the flare terms and the share of the picture's light that stays, for
    /// `out = (1 - a) Z + F (terms)`.
    #[must_use]
    pub fn flare_mix(&self) -> FlareMix {
        let (f, keep) = crate::l3::mix(self);
        FlareMix { f, keep }
    }

    /// What [`flare_scale_data`](LensPlan::flare_scale_data) depends on besides the planes and the frame
    /// size: the lens, its data versions, the camera, the axis, the clip levels, the f-number and Anamorphic
    /// (ghosts widen, streaks reach). Not Flares and not Edges.
    #[must_use]
    pub fn flare_shape_key(&self) -> String {
        let f = self.frame();
        format!(
            "lens-flare-shape:{}:v{}:att{}:{:?}:{}:{:?}:fn{:?}:clip{:?}",
            self.kind(),
            self.lens().info.version,
            self.attachment().version,
            self.amounts().anamorphic,
            self.camera().key(),
            f.axis,
            f.f_number,
            f.clip,
        )
    }
}

/// The weights of the flare sum, `out = keep * Z + f * terms` (`LensPlan::flare_mix`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlareMix {
    /// `F`: the weight of the flare terms (`flares / 100`).
    pub f: f64,
    /// `1 - a`: the share of the picture's light that stays.
    pub keep: f64,
}

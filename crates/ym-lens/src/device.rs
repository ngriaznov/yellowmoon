//! The lens's plan in the form the GPU kernels take (`docs/21` sections 5 and 9, `docs/17` section 11).
//!
//! The kernels in `ym-gpu` (`lens_field`, `lens_flare`, `lens_reshape`) compute what the CPU references in
//! this crate compute, in `f32`. They take their numbers from here, so the references and the kernels read one
//! source: plain data (no references into the plan), flat tables the device uploads once per photo, and the
//! constants of the CPU planners ([`crate::consts`]), which the kernels' WGSL is written from.

pub use crate::field::extra_level_bias;

/// Four-lane vectors in the header of the field tables (`FieldTablesData::data`): the radii, `w(rho)` and
/// `T(rho)` padded to 12 values each (3 vectors apiece), then 27 centroids `(x, y, 0, 0)` indexed
/// `[radius * 3 + channel]`, then the slopes of the monotone cubics through `w` and through `T` (3 vectors
/// apiece).
pub const FIELD_HEADER: usize = 42;

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
    /// What the oval field radius is divided by to read the relative illumination (1 without the
    /// attachment).
    pub radius_scale: f64,
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
/// Flares: [`LensPlan::flare_mix`](crate::LensPlan::flare_mix) gives the weights of the sum.
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

/// The clipped lights of a photo for the device: the extra headroom in EV of every pixel of the lens plane
/// (`light.rs`, `LightMap`), the plane's size and scale, and the cap's scale.
#[derive(Clone, Debug)]
pub struct LightMapData {
    /// The EV of every pixel of the lens plane, row-major; empty when no pixel is part of a light.
    pub ev: Vec<f32>,
    /// The lens plane's width in pixels.
    pub width: usize,
    /// The lens plane's height in pixels.
    pub height: usize,
    /// The lens plane's scale `k_L`.
    pub plane_scale: usize,
    /// The share of the extra headroom the photo keeps.
    pub scale: f64,
}

impl crate::planes::LensPlanes {
    /// The clipped lights of the photo for the device (`LightMapData`).
    #[must_use]
    pub fn light_map_data(&self) -> LightMapData {
        self.light().data()
    }

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

    /// What [`flare_scale_data`](crate::LensPlan::flare_scale_data) depends on besides the planes and the frame
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

// ---------------------------------------------------------------------------------------------------------
// L1: bokeh reshaping

/// Vectors in the header of [`BokehScaleData::vectors`]: the lens's field radii (3 vectors, 9 values and 3 of
/// padding), its swirl profile (3), its rim profile (4, 16 values), the profile's mean over a disc (1: the
/// value, three zeros) and its astigmatism profile (3).
pub const BOKEH_HEADER: usize = 14;

/// Vectors per ball in [`BokehScaleData::vectors`]. A record, with integers carried as whole numbers:
///
/// | vector | contents |
/// |---|---|
/// | 0 | centre column's integer part (a whole number), its fraction, centre row's integer part, its fraction |
/// | 1 | semi-axes `a` (along `m`) and `b` of the stretched patch, the unit vector `m` |
/// | 2 | half the cat's-eye offset (a vector), the offset in semi-axes, the width of the mask's edge in pixels |
/// | 3 | the stretch (`a / b`), the share of the lens's rim profile the ball takes (0 to 1), a zero, the take-out radius (the patch's) |
/// | 4 | the ball's radius, half its plateau, the reach of everything drawn, the blur a pixel inside the ball gets |
/// | 5 to 7 | the ring's plane fit of each channel: the value at the centre and the slopes per pixel, and a zero |
/// | 8 to 10 | the earlier balls whose discs overlap this one's, three at most: the offset of their centre from this one's, the radius of their disc (0 for none), and the width of their weight's fall |
pub const BOKEH_SHAPE_VECTORS: usize = 11;

/// Side in pixels of the bins that cull the balls, on the absolute pixel grid.
pub const BOKEH_BIN_PX: usize = 32;

/// A photo's blur map for the device (`LensPlanes::bokeh_plane_data`): made once, whatever the amounts.
#[derive(Clone, Debug, PartialEq)]
pub struct BokehPlaneData {
    /// Columns and rows of the map (level 0).
    pub size: (usize, usize),
    /// Per level of the map's reductions: columns, rows and the index in [`cells`](Self::cells) of its first
    /// cell.
    pub levels: Vec<(usize, usize, usize)>,
    /// `(a, c, s, 0)` per cell, the levels one after the other, each row-major: the mean of weight times blur, of
    /// the weight (the cell's confidence where it is blurred or confidently sharp) and of the share known to be in
    /// focus.
    pub cells: Vec<[f32; 4]>,
    /// The summed-area table of the level-0 cells that have a blur, `(columns + 1) * (rows + 1)` entries.
    pub nonzero: Vec<u32>,
}

/// The cells `[x_lo, x_hi)` x `[y_lo, y_hi)` of the map's level that a region reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BokehCells {
    /// First column.
    pub x_lo: usize,
    /// One past the last column.
    pub x_hi: usize,
    /// First row.
    pub y_lo: usize,
    /// One past the last row.
    pub y_hi: usize,
}

/// The rectangles of a pass over `want`, from the output outwards, and the cells of the blur map they read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BokehTile {
    /// The output.
    pub want: crate::PixelRect,
    /// The input of the tangential gather.
    pub r1: crate::PixelRect,
    /// The input of the vertical gather, over which the stretch's blur is used.
    pub r0: crate::PixelRect,
    /// The region the blur map is made over (`r0` and the feather), so that the stretch's soft edge is the same in
    /// every tile.
    pub rb: crate::PixelRect,
    /// `rb` grown by [`SHARP_SOFT_HALF`](crate::consts::SHARP_SOFT_HALF): the region the sharp places are known
    /// over, so that their density over the window around a pixel of `rb` is the same in every tile.
    pub rbs: crate::PixelRect,
    /// The picture the pass reads.
    pub need: crate::PixelRect,
    /// The cells of the map's level that pixels of `rbs` read.
    pub cells: BokehCells,
}

/// L1 at one scale (`LensPlan::bokeh_pass`): the numbers the per-tile passes take, in the form the CPU reference
/// uses them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BokehPass {
    /// Width of the frame at this scale in pixels.
    pub width: usize,
    /// Height of the frame at this scale in pixels.
    pub height: usize,
    /// The half-diagonal of that frame in pixels.
    pub half_diag: f64,
    /// The noise of the square-root luminance per pixel at this scale.
    pub noise: f64,
    /// Half-length of the vertical stretch per unit of blur.
    pub stretch: f64,
    /// The amount of the lens's own swirl and cat's-eye, 0 to 1: Anamorphic / 100 (Edges has none of it).
    pub swirl: f64,
    /// True when the vertical gather runs along the buffer's y (the anamorphic axis is the buffer's x).
    pub lines_along_y: bool,
    /// True when the vertical gather acts (Anamorphic above 0).
    pub vertical: bool,
    /// True when the tangential gather acts (Anamorphic on a lens with a swirl or an astigmatism).
    pub tangential: bool,
    /// Pixels of context the vertical gather needs.
    pub v_halo: usize,
    /// Pixels of context the tangential gather needs.
    pub t_halo: usize,
    /// Width in pixels of the soft edge the stretch has at the border of the blur map.
    pub feather: usize,
    /// Pixels of the tile's own picture the blur map's upsampling and the sharp-edge test read.
    pub guide_reach: usize,
    /// Columns and rows of the map (level 0).
    pub map: (usize, usize),
    /// The level of the map's reductions the upsampling reads.
    pub level: usize,
    /// Columns and rows of that level.
    pub level_size: (usize, usize),
    /// Index in [`BokehPlaneData::cells`] of that level's first cell.
    pub level_offset: usize,
    /// The level's cells in pixels of this scale, along x and along y.
    pub cell: (f64, f64),
    /// Spatial support of the upsampling in pixels.
    pub support: f64,
    /// Width of the upsampling's colour guide.
    pub sigma_guide: f64,
}

impl BokehPass {
    /// Pixels of context an output rectangle needs around it.
    #[must_use]
    pub fn halo(&self) -> usize {
        self.v_halo + self.t_halo + self.guide_reach + self.feather + crate::consts::SHARP_SOFT_HALF
    }

    /// The rectangles a pass over `want` works on, from the output outwards, and the map cells they read.
    #[must_use]
    pub fn tile(&self, want: crate::PixelRect) -> BokehTile {
        let (fw, fh) = (self.width, self.height);
        let r1 = want.grown(self.t_halo, fw, fh);
        let r0 = r1.grown(self.v_halo, fw, fh);
        let rb = r0.grown(self.feather, fw, fh);
        let rbs = rb.grown(crate::consts::SHARP_SOFT_HALF, fw, fh);
        let need = rbs.grown(self.guide_reach, fw, fh);
        BokehTile {
            want,
            r1,
            r0,
            rb,
            rbs,
            need,
            cells: self.cells_of(rbs),
        }
    }

    /// The cells of the map's level that pixels of `rb` read: those within the upsampling's support of any of
    /// them.
    #[must_use]
    pub fn cells_of(&self, rb: crate::PixelRect) -> BokehCells {
        let (lw, lh) = self.level_size;
        let sup = self.support;
        BokehCells {
            x_lo: ((rb.x as f64 - sup) / self.cell.0).floor().max(0.0) as usize,
            x_hi: (((rb.x1() as f64 + sup) / self.cell.0).ceil() as usize + 1).min(lw),
            y_lo: ((rb.y as f64 - sup) / self.cell.1).floor().max(0.0) as usize,
            y_hi: (((rb.y1() as f64 + sup) / self.cell.1).ceil() as usize + 1).min(lh),
        }
    }
}

/// The balls' shapes at one scale for the device (`LensPlan::bokeh_scale_data`): the lens's profiles, the cell
/// geometry of the map's level and the shape of every ball the plan changes, with the bins that cull them.
/// It depends on Anamorphic (the shapes), not on Edges and not on Flares.
#[derive(Clone, Debug, PartialEq)]
pub struct BokehScaleData {
    /// The pass these numbers belong to.
    pub pass: BokehPass,
    /// The header ([`BOKEH_HEADER`] vectors), then one vector per column of the map's level and one per row (the
    /// centre's integer part, its fraction, the first pixel the cell's footprint covers and one past the last, the
    /// whole numbers as `f32` values), then [`BOKEH_SHAPE_VECTORS`] per ball.
    pub vectors: Vec<[f32; 4]>,
    /// Index in `vectors` of the first column's cell.
    pub x_offset: usize,
    /// Index in `vectors` of the first row's cell.
    pub y_offset: usize,
    /// Index in `vectors` of the first ball.
    pub shape_offset: usize,
    /// Number of balls.
    pub shapes: usize,
    /// The bins that cull the balls: `bins_x * bins_y + 1` offsets, then the ball indices of each bin in list
    /// order. A bin is [`BOKEH_BIN_PX`] pixels square on the absolute pixel grid and lists every ball whose
    /// influence (what it draws, takes out or protects) can reach one of its pixels.
    pub bins: Vec<u32>,
    /// Bins across.
    pub bins_x: usize,
    /// Bins down.
    pub bins_y: usize,
}

impl crate::planes::LensPlanes {
    /// The blur map for the device, or `None` for planes without one (the plan did not ask to reshape bokeh, or
    /// the photo has little out-of-focus background).
    #[must_use]
    pub fn bokeh_plane_data(&self) -> Option<BokehPlaneData> {
        crate::l1::plane_data(self)
    }
}

impl crate::plan::LensPlan {
    /// L1's numbers for a frame of `width x height` pixels at scale `k`, or `None` when the reshaping leaves the
    /// picture alone for these planes (no blur map, little blur, or nothing that changes a ball or a pixel).
    #[must_use]
    pub fn bokeh_pass(
        &self,
        planes: &crate::planes::LensPlanes,
        k: usize,
        width: usize,
        height: usize,
    ) -> Option<BokehPass> {
        crate::l1::device_pass(self, planes, k, width, height)
    }

    /// The balls' shapes and the lens's profiles for the device at that scale, or `None` as
    /// [`bokeh_pass`](Self::bokeh_pass).
    #[must_use]
    pub fn bokeh_scale_data(
        &self,
        planes: &crate::planes::LensPlanes,
        k: usize,
        width: usize,
        height: usize,
    ) -> Option<BokehScaleData> {
        crate::l1::scale_data(self, planes, k, width, height)
    }

    /// What [`bokeh_scale_data`](Self::bokeh_scale_data) depends on besides the planes, the scale and the frame:
    /// the lens and its data versions, Anamorphic and the axis. Not Edges (the swirl and the cat's-eye are
    /// Anamorphic's) and not Flares.
    #[must_use]
    pub fn bokeh_shape_key(&self) -> String {
        format!(
            "lens-bokeh-shape:{}:v{}:att{}:{:?}:{:?}",
            self.kind(),
            self.lens().info.version,
            self.attachment().version,
            self.amounts().anamorphic,
            self.frame().axis,
        )
    }
}

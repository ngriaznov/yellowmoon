//! The Film node's passes on the device (`docs/24-film.md` sections 3.8c, 3.9 and 3.9a; `docs/17-gpu.md` section 11).
//!
//! Five pipelines, each the twin of `ym-film`'s CPU code (the parity tests, `tests/gpu_film*.rs`, hold them to it):
//!
//! | Kernel | Pass | CPU twin |
//! |---|---|---|
//! | [`KernelId::FilmColourIn`] | P1: LogC4 / AWG4 to linear, stage 1 (the soft gamut compression), T1, the raw exposure `E` | `BakedFilm::exposure_of` and `pow10` |
//! | [`KernelId::FilmSpatial`] | P2 and the diffusion of P3: the separable Gaussians of `spatial::mix` (a direct sum up to 4 px, else the coarse grid), the dye cloud's blur, the grain's add | `spatial::{blur, mix}` |
//! | [`KernelId::FilmDevelop`] | P3: the curves without couplers, the per-pixel coupler step, the inhibitor, the second develop | `BakedFilm::{film_density, coupler_inhibitor, density_with_inhibitor}` |
//! | [`KernelId::FilmGrain`] | P4: the counter-based hash (exact integer maths), the sum of four uniforms, the three sublayers | `grain::{hash32, normal}`, `FilmFrame::grain_pass` |
//! | [`KernelId::FilmColourOut`] | P5: T3a, the paper curves, the print shift, T3b′, the per-pixel return, LogC4 | `BakedFilm::colour_out` |
//!
//! [`encode_film_tile`] runs the five on a tile of the photo frame in the order of `FilmFrame::run_tile`, with the same
//! regions (the tile and the halos of the later passes, cut to the frame), so a tile is the whole frame's pixels bit for
//! bit (`docs/17` D4 and D5). The pieces are public so the plan loop of F-08b can place them: [`encode_colour_in`],
//! [`encode_mix`], [`encode_develop`], [`encode_grain`], [`encode_colour_out`].
//!
//! **Plain data in, no `ym-film` type.** [`FilmColourData`] (the baked tables and the scalars of the chain),
//! [`FilmPlan`] (the Gaussians, the dye cloud and the grain of a frame) are built by the caller from `ym-film`
//! (`BakedFilm::device_parts`, `FilmFrame::device_plan`; `tests/film_common/mod.rs` has the conversion).
//! [`FilmDevice::upload`] puts the tables in storage buffers (f32, `vec4` nodes for the 3D tables; K3: the colour-out
//! kernel binds seven storage buffers and the uniform).
//!
//! **Constants (K7).** The cine rendering's 57 constants and the derived scalars are the `CINE_*` of the common block
//! (`ym_color::display::cine::kernel_scalars`), LogC4's are `LC4_*`, the core table's axes, sizes and Newton steps, the
//! grain hash and the scene stage's knee and limit are written into the uniform block from the plain data. The settings
//! that only the kernels have (the locus tolerance, the coupler step's counts, the table sizes, `FX_*`) are generated here, once; the core's Newton
//! settings and the fixed counts of the display step come from `ym_color::display::cine::kernel_refine_scalars`.
//!
//! **Fixed counts (docs/17 K5, K8).** The CPU's loops with an exit on a tolerance run a fixed number of iterations on the
//! device: the coupler step [`COUPLER_ITERATIONS`] with [`COUPLER_TRIALS`] halvings, the shared luminance of the display
//! step, the cube-root inverse and the luminance floor (`cine::KERNEL_*_ITERATIONS`); the
//! core's Newton steps are `CoreTable::steps()` (5), each trying its candidates in the CPU's order.
//!
//! **Fast-math (K8).** Every `pow`, `exp`, `log`, `sqrt`, division and comparison of the new shaders, and why it is safe:
//!
//! * `exp`/`log`/`sqrt`/`pow` have positive arguments by construction: `exp` goes through `fr_exp`, which clamps its argument
//!   to +-80; `log` is of a value clamped or floored positive (`max(e, 1e-30)`, `max(yt, 1e-37)`, `1 + c exp(..)`, `chi`
//!   with `+ EPS_CHROMA`, `y2 >= Y2_MIN`, `s/kl` with `s > 0`); `sqrt` is of a sum of squares plus a floor or of a `max(.., 0)`
//!   (`sqrt(max(1 - u, 0))` at the folds of the ceiling and of the table's rays); the tables' warp is `sqrt` of a clamped
//!   `u` and `pow` as `exp(log(max(u, 1e-30)) / p)`. The cube root is `exp(log|x| / 3)` with the sign applied after.
//! * Every division is by a value bounded away from 0: a span (`hi - lo`, `s_span`, `4 w`), a sum of positive terms (`1 + e`),
//!   `exp(..)` clamped, `chi` (>= 1e-7), a determinant that is tested against 1e-30 first, `dg` floored at 1e-6 in
//!   magnitude, the sum `xyz.x + xyz.y + xyz.z` tested `> 0` before the chromaticities, `r >= 1e-12` for the unit direction,
//!   `den >= 1e-18` for the locus hit, `y2` (a `exp`), `step` of the table (a constant), a pixel count (>= 1).
//! * Comparisons: NaN is tested by the bits (`is_nan_f`) and never by a comparison. The tetrahedron of a 3D table is picked
//!   by comparing the fractions: at a tie the two choices give the same interpolant (continuous across the shared face).
//!   The locus segment hit compares `t` and `s` with a tolerance of 1e-5 (the CPU 1e-12): a ray through a vertex is
//!   accepted by both segments and gives the same `t`. `r <= knee` is continuous (`tanh(0) = 0`). The Newton candidate that
//!   lowers the residual (`cc < cost`): two candidates that tie are within the iteration's tolerance of each other. The
//!   comparison `rho <= rho_max` selects between the start and the refined colour: both give the pre-image of the boundary
//!   at equality to the table's accuracy. `s >= kl` in the floors of the display step joins two branches that are equal at
//!   `kl` (`kl (1 + ln 1) = kl`).
//! * No `fma`, no subgroup, no atomic. Sums are plain left-to-right sums in the CPU's order; nothing in the passes is an
//!   error-free transformation or a rank count, so the OR barrier of K8 is not needed: a fused or re-associated sum
//!   changes a value by its own rounding (<= 1e-7 relative) and no decision depends on it beyond the continuous ones above.
//!   The hash is integer arithmetic: exact under any compiler.
//! * The Gaussians' cell positions are exact: `f` is a power of two, so `(x - (f - 1) / 2) / f` is exact in f32 for any frame
//!   the design allows (x < 2^24).
//!
//! The mask blend of a corrector node is the plan loop's (`node_mask`, `docs/17` section 11a), not this module's: the Film
//! node's output is LogC4 in a pass buffer, which the plan blends with the node's input by `m`.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use super::shader::KernelId;
use super::util::GROUP;
use crate::cpu::Placement;
use crate::device::{GpuContext, GpuError};
use crate::image::{GpuImage, IMAGE_USAGE};
use crate::pool::PooledBuffer;
use crate::stream::{PassList, upload_bytes};

/// Entries of every 1D curve table (`ym_film::bake::CURVE_ENTRIES`).
pub const CURVE_ENTRIES: usize = 4096;
/// Entries of a grain amplitude table (`ym_film::grain::LUT_ENTRIES`).
pub const GRAIN_ENTRIES: usize = 1024;
/// Iterations of the coupler step (the CPU runs at most 16 and stops at 1e-12).
pub const COUPLER_ITERATIONS: u32 = 12;
/// Step halvings tried in each iteration of the coupler step (the CPU tries up to 12).
pub const COUPLER_TRIALS: u32 = 4;
/// The residual below which the coupler step stops moving.
pub const COUPLER_TOL: f32 = 1e-9;
/// The upper limit of `ln Ys` the scene side accepts (a scene luminance of 5e8).
pub const LN_YS_MAX: f64 = 20.0;
/// A Gaussian narrower than this (pixels) is a delta (`ym_film::spatial::MIN_SIGMA_PX`).
pub const MIN_SIGMA_PX: f64 = 0.2;
/// A Gaussian up to this wide (pixels) is a direct sum (`ym_film::spatial::DIRECT_MAX_SIGMA_PX`).
pub const DIRECT_MAX_SIGMA_PX: f64 = 4.0;

/// `sqrt(3) / 131072`: the factor that turns the integer sum of four centred 16-bit uniforms into a unit normal
/// (`ym_film::grain`, the same expression).
fn normal_scale() -> f64 {
    3.0_f64.sqrt() / 131_072.0
}

/// The generated `FX_*` constants block of the Film kernels.
pub(super) fn constants() -> String {
    use std::fmt::Write as _;
    use ym_color::display::cine::{self, consts};
    let f = |x: f64| format!("{:?}", x as f32);
    let mut s = String::new();
    for (n, v) in [
        ("LOG10_E", std::f64::consts::LOG10_E),
        ("PI", std::f64::consts::PI),
        ("D_MIN", cine::D_MIN),
        ("NORMAL_SCALE", normal_scale()),
        (
            "CHROMA_CAP",
            consts::CHI_K + (consts::CHI_M - consts::CHI_K) * 3.0_f64.tanh(),
        ),
        ("LN_YS_MAX", LN_YS_MAX),
        ("COUPLER_TOL", f64::from(COUPLER_TOL)),
    ] {
        let _ = writeln!(s, "const FX_{n}: f32 = {};", f(v));
    }
    for (n, v) in [
        ("CURVE_ENTRIES", CURVE_ENTRIES as u32),
        ("GRAIN_ENTRIES", GRAIN_ENTRIES as u32),
        ("COUPLER_ITERATIONS", COUPLER_ITERATIONS),
        ("COUPLER_TRIALS", COUPLER_TRIALS),
    ] {
        let _ = writeln!(s, "const FX_{n}: u32 = {v}u;");
    }
    // The Newton settings of the core and the fixed iteration counts: `ym_color::display::cine::kernel_refine_scalars`
    // (K7: the CPU's `refine` reads the same constants).
    for (n, v) in cine::kernel_refine_scalars() {
        if n.ends_with("ITERATIONS") {
            let _ = writeln!(s, "const FX_{n}: u32 = {}u;", v as u32);
        } else {
            let _ = writeln!(s, "const FX_{n}: f32 = {};", f(v));
        }
    }
    s
}

// ---------------------------------------------------------------------------------------------------------
// Plain data
// ---------------------------------------------------------------------------------------------------------

/// The 1D tables of three channels: [`CURVE_ENTRIES`] values each over `[lo, hi]`, linear between them, held at the edge.
#[derive(Debug, Clone, PartialEq)]
pub struct CurveSet {
    /// The lower end of each channel's axis.
    pub lo: [f64; 3],
    /// The upper end of each channel's axis.
    pub hi: [f64; 3],
    /// `3 * CURVE_ENTRIES` values, channel by channel.
    pub data: Vec<f32>,
}

/// A 3D table of `size^3` f32 `vec4` nodes, red fastest, tetrahedral interpolation, axes warped by `power`.
#[derive(Debug, Clone, PartialEq)]
pub struct Table3Data {
    /// Nodes per side.
    pub size: usize,
    /// The lower end of each axis.
    pub lo: [f64; 3],
    /// The upper end of each axis.
    pub hi: [f64; 3],
    /// The axis warp (1 or 2; any other value costs a `pow` per lookup).
    pub power: f64,
    /// `4 * size^3` values.
    pub data: Vec<f32>,
}

/// The core table of the per-pixel return (`ym_color::display::cine::CoreTable`).
#[derive(Debug, Clone, PartialEq)]
pub struct CoreData {
    /// Nodes along the level, the hue angle and the radius.
    pub dims: [usize; 3],
    /// The Newton steps after the table's start.
    pub steps: u32,
    /// The level `s` of the first row.
    pub s_lo: f64,
    /// The span of the level axis.
    pub s_span: f64,
    /// `ln rho_max` of every ray, `ns * nt` values.
    pub ln_reach: Vec<f32>,
    /// The nodes, three values each, `ns * nt * nk` of them, radius fastest.
    pub nodes: Vec<f32>,
}

/// The scalars of the colour passes: what changes without a re-bake, and the small matrices.
#[derive(Debug, Clone, PartialEq)]
pub struct FilmScalars {
    /// Rows of the matrix from linear AWG4 to the adapted XYZ (stage 1).
    pub scene_to_xyz: [[f64; 3]; 3],
    /// The reference white's chromaticity.
    pub white_xy: [f64; 2],
    /// The knee of the gamut compression (`ym_film::scene::KNEE`).
    pub knee: f64,
    /// The limit of the gamut compression (`ym_film::scene::LIMIT`).
    pub limit: f64,
    /// `log10` of the factor from `luminance / 0.18 * T1` to the raw exposure.
    pub log10_gain: f64,
    /// The coupler amount (0: no couplers).
    pub coupler_amount: f64,
    /// The coupler matrix.
    pub coupler_matrix: [[f64; 3]; 3],
    /// The print exposure shift after T3a, in `log10`.
    pub print_log10_shift: f64,
    /// Rows of the matrix from the output's RGB (its own primaries) to XYZ.
    pub output_to_xyz: [[f64; 3]; 3],
}

/// Everything the colour passes (P1, P3, P5) read: the baked tables and the scalars.
#[derive(Debug, Clone, PartialEq)]
pub struct FilmColourData {
    /// The scalars.
    pub scalars: FilmScalars,
    /// The locus polygon of the gamut compression.
    pub locus: Vec<[f64; 2]>,
    /// T1's nodes along x and y.
    pub t1_dims: [usize; 2],
    /// T1's node spacing.
    pub t1_step: f64,
    /// T1's nodes, `vec4`.
    pub t1: Vec<f32>,
    /// The curves without couplers (the argument is the effective exposure).
    pub film0: CurveSet,
    /// The paper's curves (none for a slide).
    pub paper: Option<CurveSet>,
    /// T3a (none for a slide).
    pub t3a: Option<Table3Data>,
    /// T3b′, the scan.
    pub t3b: Table3Data,
    /// The core table.
    pub core: CoreData,
}

impl FilmColourData {
    /// Whether this is a slide (no enlarger, no paper).
    #[must_use]
    pub fn is_slide(&self) -> bool {
        self.t3a.is_none()
    }
}

/// One Gaussian of a point spread function and its weight per channel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaussTerm {
    /// Sigma in pixels.
    pub sigma_px: f64,
    /// Weight of the term in each channel.
    pub weight: [f64; 3],
}

/// The grain of a frame.
#[derive(Debug, Clone, PartialEq)]
pub struct GrainData {
    /// The amplitude tables: for channel `c` and sublayer `s`, [`GRAIN_ENTRIES`] values at
    /// `(c * 3 + s) * GRAIN_ENTRIES`.
    pub lut: Vec<f32>,
    /// The density at which each channel's tables end.
    pub d_top: [f64; 3],
    /// Pixels per um times Amount times Size.
    pub scale: f64,
    /// The seed.
    pub seed: u32,
    /// The hash constants (`ym_film::grain::HASH_CONSTANTS`).
    pub hash: [u32; 8],
}

/// The passes of a frame (`ym_film::spatial::DevicePlan`).
#[derive(Debug, Clone, PartialEq)]
pub struct FilmPlan {
    /// The photo frame at the render's scale: width and height in pixels.
    pub frame: (usize, usize),
    /// P2's Gaussians.
    pub halation: Vec<GaussTerm>,
    /// The inhibitor's Gaussians.
    pub diffusion: Vec<GaussTerm>,
    /// Whether P3 spreads the inhibitor.
    pub diffuse: bool,
    /// The dye cloud's sigma in pixels.
    pub cloud_px: f64,
    /// The grain, when the pass runs.
    pub grain: Option<GrainData>,
}

/// A rectangle of the photo frame, `x1` and `y1` excluded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FRect {
    /// Left.
    pub x0: i32,
    /// Top.
    pub y0: i32,
    /// Right, excluded.
    pub x1: i32,
    /// Bottom, excluded.
    pub y1: i32,
}

impl FRect {
    /// Width.
    #[must_use]
    pub fn width(&self) -> i32 {
        self.x1 - self.x0
    }

    /// Height.
    #[must_use]
    pub fn height(&self) -> i32 {
        self.y1 - self.y0
    }

    /// The whole frame.
    #[must_use]
    pub fn frame(frame: (usize, usize)) -> Self {
        Self {
            x0: 0,
            y0: 0,
            x1: frame.0 as i32,
            y1: frame.1 as i32,
        }
    }

    /// `h` pixels more on every side, cut to the frame.
    #[must_use]
    pub fn expand(&self, h: usize, frame: (usize, usize)) -> Self {
        let h = h as i32;
        Self {
            x0: (self.x0 - h).max(0),
            y0: (self.y0 - h).max(0),
            x1: (self.x1 + h).min(frame.0 as i32),
            y1: (self.y1 + h).min(frame.1 as i32),
        }
    }

    /// Whether `other` lies inside this rectangle.
    #[must_use]
    pub fn contains(&self, other: &FRect) -> bool {
        self.x0 <= other.x0 && self.y0 <= other.y0 && self.x1 >= other.x1 && self.y1 >= other.y1
    }

    fn is_empty(&self) -> bool {
        self.width() <= 0 || self.height() <= 0
    }
}

/// A three-plane f32 image on the device that holds the rectangle `rect` of the photo frame (or of a cell grid).
#[derive(Debug)]
pub struct FilmImage {
    /// The planes.
    pub img: GpuImage,
    /// The frame rectangle the image holds.
    pub rect: FRect,
    /// The exact display colour of each pixel of a picture the colour-out pass made (three planes laid out as `img`): the
    /// print's colour in the output's primaries, held to the cine rendering's range, which the picture's LogC4 values map
    /// back to. The colour pass uses it for a pixel whose value reaches C12 unchanged (`docs/24` section 3.9c, F-08d).
    /// `None` for every other image.
    pub shadow: Option<GpuImage>,
}

impl FilmImage {
    /// An image of `rect` whose content is undefined until a pass writes it.
    ///
    /// # Errors
    ///
    /// [`GpuError::InvalidInput`] for an empty rectangle; the pool's errors.
    pub fn alloc(ctx: &GpuContext, label: &str, rect: FRect) -> Result<Self, GpuError> {
        if rect.is_empty() {
            return Err(GpuError::InvalidInput(format!(
                "{label}: the rectangle {rect:?} is empty"
            )));
        }
        let (w, h) = (rect.width() as usize, rect.height() as usize);
        let img = GpuImage::alloc(ctx, label, w, h, 3, Placement::whole(w, h))?;
        Ok(Self {
            img,
            rect,
            shadow: None,
        })
    }

    /// [`FilmImage::alloc`] with the shadow of the colour-out pass.
    ///
    /// # Errors
    ///
    /// As [`FilmImage::alloc`].
    pub fn alloc_with_shadow(ctx: &GpuContext, label: &str, rect: FRect) -> Result<Self, GpuError> {
        let mut image = Self::alloc(ctx, label, rect)?;
        let (w, h) = (rect.width() as usize, rect.height() as usize);
        image.shadow = Some(GpuImage::alloc(
            ctx,
            "film shadow",
            w,
            h,
            3,
            Placement::whole(w, h),
        )?);
        Ok(image)
    }

    /// Uploads dense planar samples (three planes of `rect`).
    ///
    /// # Errors
    ///
    /// As [`GpuImage::upload`].
    pub fn upload(
        ctx: &GpuContext,
        label: &str,
        rect: FRect,
        data: &[f32],
    ) -> Result<Self, GpuError> {
        let (w, h) = (rect.width() as usize, rect.height() as usize);
        let img = GpuImage::upload(ctx, label, w, h, 3, data, Placement::whole(w, h))?;
        Ok(Self {
            img,
            rect,
            shadow: None,
        })
    }

    /// Reads the image back as dense planar samples (after the pass list that wrote it has finished).
    ///
    /// # Errors
    ///
    /// As [`GpuImage::read`].
    pub fn read(&self, ctx: &GpuContext) -> Result<Vec<f32>, GpuError> {
        self.img.read(ctx)
    }

    fn desc(&self) -> [i32; 4] {
        [
            self.rect.x0,
            self.rect.y0,
            self.img.stride as i32,
            self.img.plane_len() as i32,
        ]
    }
}

// ---------------------------------------------------------------------------------------------------------
// The device tables
// ---------------------------------------------------------------------------------------------------------

fn storage(ctx: &GpuContext, label: &str, data: &[f32]) -> Result<Arc<PooledBuffer>, GpuError> {
    let bytes: &[u8] = bytemuck::cast_slice(data);
    let buf = ctx
        .pool()
        .acquire(label, (bytes.len() as u64).max(16), IMAGE_USAGE)?;
    if !bytes.is_empty() {
        upload_bytes(ctx, &buf, 0, bytes)?;
    }
    Ok(Arc::new(buf))
}

fn check_len(label: &str, got: usize, want: usize) -> Result<(), GpuError> {
    if got == want {
        Ok(())
    } else {
        Err(GpuError::InvalidInput(format!(
            "{label}: {got} values, expected {want}"
        )))
    }
}

/// The colour tables of a film in storage buffers, with the scalars. Cheap to clone (the buffers are shared).
#[derive(Debug, Clone)]
pub struct FilmDevice {
    scalars: FilmScalars,
    locus_len: u32,
    t1_dims: [usize; 2],
    t1_step: f64,
    film0: (CurveSet, Arc<PooledBuffer>),
    paper: Option<(CurveSet, Arc<PooledBuffer>)>,
    t3a: Option<(Table3Data, Arc<PooledBuffer>)>,
    t3b: (Table3Data, Arc<PooledBuffer>),
    core: CoreData,
    t1: Arc<PooledBuffer>,
    locus: Arc<PooledBuffer>,
    nodes: Arc<PooledBuffer>,
    reach: Arc<PooledBuffer>,
    dummy: Arc<PooledBuffer>,
    /// A view: the passes run the plain f32 arithmetic of before F-08d ([`FilmDevice::fast`]).
    fast: bool,
}

impl FilmDevice {
    /// Uploads the tables.
    ///
    /// # Errors
    ///
    /// [`GpuError::InvalidInput`] when a table has the wrong length; the pool's and the upload's errors.
    pub fn upload(ctx: &GpuContext, data: &FilmColourData) -> Result<Self, GpuError> {
        let [nx, ny] = data.t1_dims;
        check_len("T1", data.t1.len(), 4 * nx * ny)?;
        check_len("film0 curves", data.film0.data.len(), 3 * CURVE_ENTRIES)?;
        let t3 = |label: &str, t: &Table3Data| check_len(label, t.data.len(), 4 * t.size.pow(3));
        t3("T3b", &data.t3b)?;
        if let Some(t) = &data.t3a {
            t3("T3a", t)?;
        }
        if let Some(p) = &data.paper {
            check_len("paper curves", p.data.len(), 3 * CURVE_ENTRIES)?;
        }
        let [ns, nt, nk] = data.core.dims;
        check_len("core reach", data.core.ln_reach.len(), ns * nt)?;
        check_len("core nodes", data.core.nodes.len(), 3 * ns * nt * nk)?;
        let locus: Vec<f32> = data
            .locus
            .iter()
            .flat_map(|p| [p[0] as f32, p[1] as f32])
            .collect();
        Ok(Self {
            scalars: data.scalars.clone(),
            locus_len: data.locus.len() as u32,
            t1_dims: data.t1_dims,
            t1_step: data.t1_step,
            film0: (
                data.film0.clone_header(),
                storage(ctx, "film0 curves", &data.film0.data)?,
            ),
            paper: match &data.paper {
                Some(p) => Some((p.clone_header(), storage(ctx, "paper curves", &p.data)?)),
                None => None,
            },
            t3a: match &data.t3a {
                Some(t) => Some((t.clone_header(), storage(ctx, "T3a", &t.data)?)),
                None => None,
            },
            t3b: (
                data.t3b.clone_header(),
                storage(ctx, "T3b", &data.t3b.data)?,
            ),
            core: CoreData {
                ln_reach: Vec::new(),
                nodes: Vec::new(),
                ..data.core.clone()
            },
            t1: storage(ctx, "T1", &data.t1)?,
            locus: storage(ctx, "locus", &locus)?,
            nodes: storage(ctx, "core nodes", &data.core.nodes)?,
            reach: storage(ctx, "core reach", &data.core.ln_reach)?,
            dummy: storage(ctx, "film dummy", &[0.0; 4])?,
            fast: false,
        })
    }

    /// The scalars.
    #[must_use]
    pub fn scalars(&self) -> &FilmScalars {
        &self.scalars
    }

    /// The same tables with other scalars (the film exposure, the print exposure and the coupler amount change
    /// without a re-upload).
    #[must_use]
    pub fn with_scalars(&self, scalars: FilmScalars) -> Self {
        Self {
            scalars,
            ..self.clone()
        }
    }

    /// The same tables for a **view**: the colour-in, develop and colour-out passes run the plain f32 arithmetic they had
    /// before F-08d (the same bits, `docs/24` section 3.9c "Speed"), and the colour-out pass makes no shadow. The default is
    /// the compensated arithmetic of the export. Which one runs is a word of the uniform block (`h[2].w`), so the two share
    /// the pipelines.
    #[must_use]
    pub fn fast(&self, fast: bool) -> Self {
        Self {
            fast,
            ..self.clone()
        }
    }

    /// Whether the passes run the plain f32 arithmetic of a view ([`FilmDevice::fast`]).
    #[must_use]
    pub fn is_fast(&self) -> bool {
        self.fast
    }

    /// Whether this is a slide.
    #[must_use]
    pub fn is_slide(&self) -> bool {
        self.t3a.is_none()
    }

    /// The bytes of the largest storage binding of the colour passes.
    #[must_use]
    pub fn max_binding_bytes(&self) -> u64 {
        [&self.t3b.1, &self.nodes, &self.t1]
            .iter()
            .map(|b| b.size())
            .chain(self.t3a.iter().map(|t| t.1.size()))
            .max()
            .unwrap_or(0)
    }
}

impl CurveSet {
    fn clone_header(&self) -> CurveSet {
        CurveSet {
            lo: self.lo,
            hi: self.hi,
            data: Vec::new(),
        }
    }
}

impl Table3Data {
    fn clone_header(&self) -> Table3Data {
        Table3Data {
            data: Vec::new(),
            ..self.clone()
        }
    }
}

// ---------------------------------------------------------------------------------------------------------
// The uniform block
// ---------------------------------------------------------------------------------------------------------

/// The uniform block of every Film kernel (`FU` of `film.wgsl`), 496 bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FilmU {
    frame: [i32; 4],
    a: [i32; 4],
    b: [i32; 4],
    c: [i32; 4],
    win: [i32; 4],
    op: [u32; 4],
    p: [[f32; 4]; 16],
    m: [[f32; 4]; 6],
    h: [[u32; 4]; 3],
}

impl FilmU {
    fn new(frame: [i32; 4], b: &FilmImage, win: FRect) -> Self {
        let mut u = FilmU::zeroed();
        u.frame = frame;
        u.b = b.desc();
        u.win = [win.x0, win.y0, win.width(), win.height()];
        u
    }
}

fn v3(v: [f64; 3], w: f32) -> [f32; 4] {
    [v[0] as f32, v[1] as f32, v[2] as f32, w]
}

fn rows(m: &[[f64; 3]; 3]) -> [[f32; 4]; 3] {
    [v3(m[0], 0.0), v3(m[1], 0.0), v3(m[2], 0.0)]
}

fn groups(win: FRect) -> (u32, u32, u32) {
    (
        (win.width() as u32).div_ceil(GROUP),
        (win.height() as u32).div_ceil(GROUP),
        1,
    )
}

fn check_window(what: &str, win: FRect, images: &[&FilmImage]) -> Result<(), GpuError> {
    if win.is_empty() {
        return Err(GpuError::InvalidInput(format!(
            "{what}: the window {win:?} is empty"
        )));
    }
    for im in images {
        if !im.rect.contains(&win) {
            return Err(GpuError::InvalidInput(format!(
                "{what}: the window {win:?} is outside an image of {:?}",
                im.rect
            )));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------------------
// P1: colour in
// ---------------------------------------------------------------------------------------------------------

/// P1: LogC4 / AWG4 `src` to the raw exposure `E` (linear, at least 1e-30) over `win` of `dst`.
///
/// # Errors
///
/// [`GpuError::InvalidInput`] when the window is empty or outside either image.
pub fn encode_colour_in(
    passes: &mut PassList<'_>,
    dev: &FilmDevice,
    frame: (usize, usize),
    src: &FilmImage,
    dst: &FilmImage,
    win: FRect,
) -> Result<(), GpuError> {
    check_window("film colour in", win, &[src, dst])?;
    let s = &dev.scalars;
    let mut u = FilmU::new([frame.0 as i32, frame.1 as i32, 0, 0], dst, win);
    u.h[2][3] = u32::from(dev.fast);
    if !dev.fast {
        passes.context().counters_shared().count_precise();
    }
    u.a = src.desc();
    u.op = [
        0,
        dev.locus_len,
        dev.t1_dims[0] as u32,
        dev.t1_dims[1] as u32,
    ];
    u.m[..3].copy_from_slice(&rows(&s.scene_to_xyz));
    // The low parts of the matrix and of the scale: the exposure is the product of a matrix row and a scale, and the f32
    // rounding of a constant is a systematic error of 3e-8 in every pixel (more where a row cancels).
    let lo_rows = s.scene_to_xyz.map(|r| r.map(|v| v - f64::from(v as f32)));
    u.m[3..6].copy_from_slice(&rows(&lo_rows));
    let scale = 10.0_f64.powf(s.log10_gain) / 0.18;
    u.p[0] = [
        s.white_xy[0] as f32,
        s.white_xy[1] as f32,
        scale as f32,
        (scale - f64::from(scale as f32)) as f32,
    ];
    u.p[1] = [dev.t1_step as f32, s.knee as f32, s.limit as f32, 0.0];
    // 1 / step as a hi + lo pair: the position of a chromaticity in T1 is x * (1 / step), carried without a rounding of its own.
    let k = 1.0 / dev.t1_step;
    u.p[2] = [k as f32, (k - f64::from(k as f32)) as f32, 0.0, 0.0];
    let ub = passes.uniform("film colour in uniform", &u)?;
    passes.hold(dev.t1.clone());
    passes.hold(dev.locus.clone());
    passes.dispatch(
        KernelId::FilmColourIn,
        "film colour in",
        &[
            (0, ub.binding()),
            (1, src.img.buf.binding()),
            (2, dst.img.buf.binding()),
            (3, dev.t1.binding()),
            (4, dev.locus.binding()),
        ],
        groups(win),
    )
}

// ---------------------------------------------------------------------------------------------------------
// P2 and the diffusion: the Gaussians
// ---------------------------------------------------------------------------------------------------------

/// The factor of the coarse grid of a Gaussian of `sigma` pixels: 1 for a direct sum, else a power of two
/// (`ym_film::spatial` `decim_factor`).
#[must_use]
pub fn decim_factor(sigma: f64) -> usize {
    if sigma <= DIRECT_MAX_SIGMA_PX {
        1
    } else {
        1 << ((sigma / 2.5).log2().floor().max(1.0) as u32)
    }
}

/// The pixels a Gaussian of `sigma` pixels reads beyond its output (`ym_film::spatial::blur_halo`).
#[must_use]
pub fn blur_halo(sigma_px: f64) -> usize {
    if sigma_px < MIN_SIGMA_PX {
        return 0;
    }
    let f = decim_factor(sigma_px);
    let cells = if f > 1 { 3 * f } else { 0 };
    (6.0 * sigma_px).ceil() as usize + cells
}

/// The largest halo of a set of Gaussians.
#[must_use]
pub fn terms_halo(terms: &[GaussTerm]) -> usize {
    terms
        .iter()
        .map(|t| blur_halo(t.sigma_px))
        .max()
        .unwrap_or(0)
}

/// The one-sided weights of the sampled Gaussian, `w[0..=r]` with `r = ceil(6 sigma)`, normalised so that
/// `w0 + 2 sum w = 1` (`ym_film::spatial` `kernel`).
#[must_use]
pub fn gauss_taps(sigma: f64) -> Vec<f64> {
    let r = (6.0 * sigma).ceil() as usize;
    let mut w: Vec<f64> = (0..=r)
        .map(|i| (-0.5 * (i as f64 / sigma).powi(2)).exp())
        .collect();
    let total: f64 = w[0] + 2.0 * w[1..].iter().sum::<f64>();
    for v in &mut w {
        *v /= total;
    }
    w
}

/// The sum of the squared weights of the whole kernel (`ym_film::spatial::kernel_power`).
#[must_use]
pub fn gauss_power(sigma: f64) -> f64 {
    let w = gauss_taps(sigma);
    w[0] * w[0] + 2.0 * w[1..].iter().map(|v| v * v).sum::<f64>()
}

const SP_SCALE: u32 = 0;
const SP_ROWS: u32 = 1;
const SP_COLS: u32 = 2;
const SP_CELLS: u32 = 3;
const SP_UPSAMPLE: u32 = 4;
const SP_ADD: u32 = 5;

struct Spatial<'a> {
    op: u32,
    radius: u32,
    accumulate: bool,
    factor: u32,
    weight: [f32; 3],
    frame: [i32; 4],
    src: &'a FilmImage,
    src2: Option<&'a FilmImage>,
    dst: &'a FilmImage,
    win: FRect,
    taps: Option<&'a Arc<PooledBuffer>>,
}

fn dispatch_spatial(
    passes: &mut PassList<'_>,
    dummy: &Arc<PooledBuffer>,
    s: &Spatial<'_>,
    what: &str,
) -> Result<(), GpuError> {
    let mut u = FilmU::new(s.frame, s.dst, s.win);
    u.a = s.src.desc();
    if let Some(c) = s.src2 {
        u.c = c.desc();
    }
    u.op = [s.op, s.radius, u32::from(s.accumulate), s.factor];
    u.p[0] = [s.weight[0], s.weight[1], s.weight[2], 0.0];
    let ub = passes.uniform("film spatial uniform", &u)?;
    let taps = s.taps.unwrap_or(dummy);
    passes.hold(taps.clone());
    passes.dispatch(
        KernelId::FilmSpatial,
        what,
        &[
            (0, ub.binding()),
            (1, s.src.img.buf.binding()),
            (2, s.dst.img.buf.binding()),
            (3, taps.binding()),
            (
                4,
                s.src2
                    .map_or_else(|| dummy.binding(), |c| c.img.buf.binding()),
            ),
        ],
        groups(s.win),
    )
}

fn taps_buffer(
    ctx: &GpuContext,
    passes: &mut PassList<'_>,
    sigma: f64,
) -> Result<(Arc<PooledBuffer>, u32), GpuError> {
    let w = gauss_taps(sigma);
    let f: Vec<f32> = w.iter().map(|&v| v as f32).collect();
    let buf = storage(ctx, "film taps", &f)?;
    passes.hold(buf.clone());
    Ok((buf, (w.len() - 1) as u32))
}

/// A direct separable blur of `src` over `out`, reads clamped into `[0, bounds)`; written to `dst` (which holds `out`),
/// or added to it times `weight`.
#[allow(clippy::too_many_arguments)]
fn encode_direct(
    ctx: &GpuContext,
    passes: &mut PassList<'_>,
    dummy: &Arc<PooledBuffer>,
    src: &FilmImage,
    out: FRect,
    bounds: (i32, i32),
    sigma: f64,
    dst: &FilmImage,
    weight: Option<[f32; 3]>,
) -> Result<(), GpuError> {
    let (taps, r) = taps_buffer(ctx, passes, sigma)?;
    let ri = r as i32;
    let (ty0, ty1) = ((out.y0 - ri).max(0), (out.y1 + ri).min(bounds.1));
    let trect = FRect {
        x0: out.x0,
        y0: ty0,
        x1: out.x1,
        y1: ty1,
    };
    let tmp = FilmImage::alloc(ctx, "film blur rows", trect)?;
    let frame = [bounds.0, bounds.1, 0, 0];
    dispatch_spatial(
        passes,
        dummy,
        &Spatial {
            op: SP_ROWS,
            radius: r,
            accumulate: false,
            factor: 0,
            weight: [0.0; 3],
            frame,
            src,
            src2: None,
            dst: &tmp,
            win: trect,
            taps: Some(&taps),
        },
        "film blur rows",
    )?;
    dispatch_spatial(
        passes,
        dummy,
        &Spatial {
            op: SP_COLS,
            radius: r,
            accumulate: weight.is_some(),
            factor: 0,
            weight: weight.unwrap_or([0.0; 3]),
            frame,
            src: &tmp,
            src2: None,
            dst,
            win: out,
            taps: Some(&taps),
        },
        "film blur columns",
    )?;
    passes.hold(taps);
    // The temporary lives until the list is submitted: keep its buffer.
    let FilmImage { img, .. } = tmp;
    passes.keep(img.buf);
    Ok(())
}

/// A Gaussian of `sigma` pixels of `src` over `out` (`ym_film::spatial::blur`): a direct sum up to
/// [`DIRECT_MAX_SIGMA_PX`], else on the coarse grid. `src` must cover `out` and [`blur_halo`] pixels more (cut to the
/// frame); the result is written to `dst`, or added to it times `weight`.
///
/// # Errors
///
/// [`GpuError::InvalidInput`] when `src` does not cover what the Gaussian reads or `dst` does not hold `out`; the
/// pool's errors.
#[allow(clippy::too_many_arguments)]
pub fn encode_gauss(
    ctx: &GpuContext,
    passes: &mut PassList<'_>,
    dev: &FilmDevice,
    frame: (usize, usize),
    src: &FilmImage,
    out: FRect,
    sigma: f64,
    dst: &FilmImage,
    weight: Option<[f32; 3]>,
) -> Result<(), GpuError> {
    encode_gauss_with(ctx, passes, &dev.dummy, frame, src, out, sigma, dst, weight)
}

#[allow(clippy::too_many_arguments)]
fn encode_gauss_with(
    ctx: &GpuContext,
    passes: &mut PassList<'_>,
    dummy: &Arc<PooledBuffer>,
    frame: (usize, usize),
    src: &FilmImage,
    out: FRect,
    sigma: f64,
    dst: &FilmImage,
    weight: Option<[f32; 3]>,
) -> Result<(), GpuError> {
    let fr = FRect::frame(frame);
    let need = out.expand(blur_halo(sigma), frame);
    check_window("film blur", out, &[dst])?;
    if !src.rect.contains(&need) || !fr.contains(&out) {
        return Err(GpuError::InvalidInput(format!(
            "film blur: the source {:?} does not cover {need:?}",
            src.rect
        )));
    }
    let f = decim_factor(sigma);
    if f == 1 {
        return encode_direct(
            ctx,
            passes,
            dummy,
            src,
            out,
            (frame.0 as i32, frame.1 as i32),
            sigma,
            dst,
            weight,
        );
    }
    let ff = f as f64;
    let sigma_c = (sigma * sigma - ff * ff / 4.0).max(0.0).sqrt() / ff;
    let rc = (6.0 * sigma_c).ceil() as i32;
    let (ncx, ncy) = (frame.0.div_ceil(f) as i32, frame.1.div_ceil(f) as i32);
    let centre = |x: i32| (f64::from(x) - 0.5 * (ff - 1.0)) / ff;
    let range = |lo: i32, hi: i32, n: i32| -> (i32, i32) {
        let a = centre(lo).floor().max(0.0) as i32;
        let b = ((centre(hi - 1).floor() + 1.0).max(0.0) as i32).min(n - 1);
        (a.min(n - 1), b)
    };
    let (cx0, cx1) = range(out.x0, out.x1, ncx);
    let (cy0, cy1) = range(out.y0, out.y1, ncy);
    let needc = FRect {
        x0: (cx0 - rc).max(0),
        y0: (cy0 - rc).max(0),
        x1: (cx1 + rc + 1).min(ncx),
        y1: (cy1 + rc + 1).min(ncy),
    };
    let cells = FilmImage::alloc(ctx, "film cells", needc)?;
    dispatch_spatial(
        passes,
        dummy,
        &Spatial {
            op: SP_CELLS,
            radius: 0,
            accumulate: false,
            factor: f as u32,
            weight: [0.0; 3],
            frame: [frame.0 as i32, frame.1 as i32, 0, 0],
            src,
            src2: None,
            dst: &cells,
            win: needc,
            taps: None,
        },
        "film cells",
    )?;
    let want = FRect {
        x0: cx0,
        y0: cy0,
        x1: cx1 + 1,
        y1: cy1 + 1,
    };
    let blurred = FilmImage::alloc(ctx, "film blurred cells", want)?;
    encode_direct(
        ctx,
        passes,
        dummy,
        &cells,
        want,
        (ncx, ncy),
        sigma_c,
        &blurred,
        None,
    )?;
    dispatch_spatial(
        passes,
        dummy,
        &Spatial {
            op: SP_UPSAMPLE,
            radius: 0,
            accumulate: weight.is_some(),
            factor: f as u32,
            weight: weight.unwrap_or([0.0; 3]),
            frame: [frame.0 as i32, frame.1 as i32, ncx, ncy],
            src: &blurred,
            src2: None,
            dst,
            win: out,
            taps: None,
        },
        "film upsample",
    )?;
    passes.keep(cells.img.buf);
    passes.keep(blurred.img.buf);
    Ok(())
}

/// `centre * src + sum_t weight_t * (gauss_t * src)` over `out`, with the centre weight `1 - sum weight` (so a uniform area
/// is unchanged); the terms narrower than [`MIN_SIGMA_PX`] stay at the pixel (`ym_film::spatial` `mix`). `dst` holds
/// `out`; `src` covers `out` and the largest halo.
///
/// # Errors
///
/// As [`encode_gauss`].
#[allow(clippy::too_many_arguments)]
pub fn encode_mix(
    ctx: &GpuContext,
    passes: &mut PassList<'_>,
    dev: &FilmDevice,
    frame: (usize, usize),
    src: &FilmImage,
    out: FRect,
    terms: &[GaussTerm],
    dst: &FilmImage,
) -> Result<(), GpuError> {
    let mut centre = [1.0_f64; 3];
    for t in terms {
        for (c, w) in centre.iter_mut().zip(t.weight) {
            *c -= w;
        }
    }
    let mut live = Vec::new();
    for t in terms {
        if t.sigma_px >= MIN_SIGMA_PX {
            live.push(*t);
        } else {
            for (c, w) in centre.iter_mut().zip(t.weight) {
                *c += w;
            }
        }
    }
    check_window("film mix", out, &[src, dst])?;
    dispatch_spatial(
        passes,
        &dev.dummy,
        &Spatial {
            op: SP_SCALE,
            radius: 0,
            accumulate: false,
            factor: 0,
            weight: centre.map(|c| c as f32),
            frame: [frame.0 as i32, frame.1 as i32, 0, 0],
            src,
            src2: None,
            dst,
            win: out,
            taps: None,
        },
        "film mix centre",
    )?;
    for t in live {
        encode_gauss(
            ctx,
            passes,
            dev,
            frame,
            src,
            out,
            t.sigma_px,
            dst,
            Some(t.weight.map(|w| w as f32)),
        )?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------------------
// P3: develop
// ---------------------------------------------------------------------------------------------------------

/// What [`encode_develop`] computes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DevelopMode {
    /// The film density with the couplers solved per pixel (`BakedFilm::film_density`).
    Density,
    /// The inhibitor the couplers release for that density (`coupler_inhibitor(film_density(..))`).
    Inhibitor,
    /// The density given the spread inhibitor `q` (`density_with_inhibitor`).
    WithInhibitor,
}

/// P3: the exposure `e` (linear, from P2) to a density or an inhibitor over `win` of `dst`. `q` is the spread inhibitor
/// of [`DevelopMode::WithInhibitor`].
///
/// # Errors
///
/// [`GpuError::InvalidInput`] when the window is outside an image or `q` is missing.
pub fn encode_develop(
    passes: &mut PassList<'_>,
    dev: &FilmDevice,
    mode: DevelopMode,
    e: &FilmImage,
    q: Option<&FilmImage>,
    dst: &FilmImage,
    win: FRect,
) -> Result<(), GpuError> {
    check_window("film develop", win, &[e, dst])?;
    if mode == DevelopMode::WithInhibitor {
        let Some(q) = q else {
            return Err(GpuError::InvalidInput(
                "film develop: the inhibitor is missing".into(),
            ));
        };
        check_window("film develop", win, &[q])?;
    }
    let s = &dev.scalars;
    let mut u = FilmU::new([0; 4], dst, win);
    u.h[2][3] = u32::from(dev.fast);
    if !dev.fast {
        passes.context().counters_shared().count_precise();
    }
    u.a = e.desc();
    if let Some(q) = q {
        u.c = q.desc();
    }
    u.op = [
        match mode {
            DevelopMode::Density => 0,
            DevelopMode::Inhibitor => 1,
            DevelopMode::WithInhibitor => 2,
        },
        0,
        0,
        0,
    ];
    let (c0, buf) = &dev.film0;
    u.m[..3].copy_from_slice(&rows(&s.coupler_matrix));
    u.p[0] = [s.coupler_amount as f32, 0.0, 0.0, 0.0];
    u.p[1] = v3(c0.lo, 0.0);
    let k = |c: usize| (CURVE_ENTRIES - 1) as f64 / (c0.hi[c] - c0.lo[c]);
    u.p[2] = v3([k(0), k(1), k(2)], 0.0);
    // The axis start and the factor as hi + lo pairs (the position in the curve is carried without a rounding of its own).
    for c in 0..3 {
        u.p[3][c] = (c0.lo[c] - f64::from(c0.lo[c] as f32)) as f32;
        u.p[4][c] = (k(c) - f64::from(k(c) as f32)) as f32;
    }
    let ub = passes.uniform("film develop uniform", &u)?;
    passes.hold(buf.clone());
    passes.dispatch(
        KernelId::FilmDevelop,
        "film develop",
        &[
            (0, ub.binding()),
            (1, e.img.buf.binding()),
            (2, dst.img.buf.binding()),
            (3, buf.binding()),
            (
                4,
                q.map_or_else(|| dev.dummy.binding(), |q| q.img.buf.binding()),
            ),
        ],
        groups(win),
    )
}

// ---------------------------------------------------------------------------------------------------------
// P4: grain
// ---------------------------------------------------------------------------------------------------------

/// The uniform words of the grain hash.
fn hash_words(hash: &[u32; 8]) -> [[u32; 4]; 3] {
    [
        [hash[0], hash[1], hash[2], hash[3]],
        [hash[4], hash[5], hash[6], hash[7]],
        [0; 4],
    ]
}

/// The grain pass: adds the grain to the film density `d` over `tile` (`FilmFrame::apply_grain`). `d` covers the tile and
/// the dye cloud's halo (cut to the frame). The noise is computed over `d`'s whole rectangle, blurred by the dye cloud (which
/// keeps the variance: the blur is divided by the kernel's power) and added to `d` over `tile`.
///
/// # Errors
///
/// [`GpuError::InvalidInput`] when `d` does not cover the tile and the cloud's halo; the pool's errors.
#[allow(clippy::too_many_arguments)]
pub fn encode_grain(
    ctx: &GpuContext,
    passes: &mut PassList<'_>,
    dev: &FilmDevice,
    plan: &FilmPlan,
    grain: &GrainData,
    d: &FilmImage,
    tile: FRect,
    dst: &FilmImage,
) -> Result<(), GpuError> {
    let frame = plan.frame;
    let cloud_on = plan.cloud_px >= MIN_SIGMA_PX;
    let halo = if cloud_on {
        blur_halo(plan.cloud_px)
    } else {
        0
    };
    check_window("film grain", tile, &[dst])?;
    if !d.rect.contains(&tile.expand(halo, frame)) {
        return Err(GpuError::InvalidInput(format!(
            "film grain: the density {:?} does not cover the tile {tile:?} and the cloud's halo {halo}",
            d.rect
        )));
    }
    let lut = storage(ctx, "film grain lut", &grain.lut)?;
    passes.hold(lut.clone());
    let noise = FilmImage::alloc(ctx, "film grain noise", d.rect)?;
    let mut u = FilmU::new([0; 4], &noise, d.rect);
    u.a = d.desc();
    u.op = [0, 0, 0, grain.seed];
    u.p[0] = [grain.scale as f32, 0.0, 0.0, 0.0];
    u.p[1] = v3(grain.d_top, 0.0);
    u.h = hash_words(&grain.hash);
    let ub = passes.uniform("film grain uniform", &u)?;
    passes.dispatch(
        KernelId::FilmGrain,
        "film grain noise",
        &[
            (0, ub.binding()),
            (1, d.img.buf.binding()),
            (2, noise.img.buf.binding()),
            (3, lut.binding()),
        ],
        groups(d.rect),
    )?;
    let (cloud, norm) = if cloud_on {
        let blurred = FilmImage::alloc(ctx, "film grain cloud", tile)?;
        encode_gauss(
            ctx,
            passes,
            dev,
            frame,
            &noise,
            tile,
            plan.cloud_px,
            &blurred,
            None,
        )?;
        (blurred, 1.0 / gauss_power(plan.cloud_px))
    } else {
        (noise, 1.0)
    };
    dispatch_spatial_add(passes, &dev.dummy, frame, d, &cloud, norm as f32, dst, tile)?;
    passes.keep(cloud.img.buf);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn dispatch_spatial_add(
    passes: &mut PassList<'_>,
    dummy: &Arc<PooledBuffer>,
    frame: (usize, usize),
    a: &FilmImage,
    c: &FilmImage,
    norm: f32,
    dst: &FilmImage,
    win: FRect,
) -> Result<(), GpuError> {
    check_window("film grain add", win, &[a, c, dst])?;
    dispatch_spatial(
        passes,
        dummy,
        &Spatial {
            op: SP_ADD,
            radius: 0,
            accumulate: false,
            factor: 0,
            weight: [norm, 0.0, 0.0],
            frame: [frame.0 as i32, frame.1 as i32, 0, 0],
            src: a,
            src2: Some(c),
            dst,
            win,
            taps: None,
        },
        "film grain add",
    )
}

/// The probe of the tests: for every pixel of `win`, plane 0 holds the bits of the hash word 0 of
/// `(seed, x, y, channel, sublayer)`, plane 1 the bits of word 1 and plane 2 the integer sum of the draw as an f32 (exact).
///
/// # Errors
///
/// [`GpuError::InvalidInput`] when the window is outside `dst`.
#[allow(clippy::too_many_arguments)]
pub fn encode_grain_probe(
    ctx: &GpuContext,
    passes: &mut PassList<'_>,
    grain: &GrainData,
    channel: u32,
    sublayer: u32,
    dst: &FilmImage,
    win: FRect,
) -> Result<(), GpuError> {
    check_window("film grain probe", win, &[dst])?;
    let lut = storage(ctx, "film grain lut", &[0.0; 4])?;
    passes.hold(lut.clone());
    let mut u = FilmU::new([0; 4], dst, win);
    u.a = dst.desc();
    u.op = [1, channel, sublayer, grain.seed];
    u.h = hash_words(&grain.hash);
    let ub = passes.uniform("film grain probe uniform", &u)?;
    // The kernel binds the density input too; the probe reads none of it.
    let scratch = passes.scratch("film grain probe input", 16)?;
    passes.dispatch(
        KernelId::FilmGrain,
        "film grain probe",
        &[
            (0, ub.binding()),
            (1, scratch.binding()),
            (2, dst.img.buf.binding()),
            (3, lut.binding()),
        ],
        groups(win),
    )
}

// ---------------------------------------------------------------------------------------------------------
// P5: colour out
// ---------------------------------------------------------------------------------------------------------

/// P5: the film density `d` to LogC4 / AWG4 over `win` of `dst`: T3a, the paper's curves and the print shift (a negative),
/// T3b′ and the per-pixel return.
///
/// # Errors
///
/// [`GpuError::InvalidInput`] when the window is outside an image.
pub fn encode_colour_out(
    passes: &mut PassList<'_>,
    dev: &FilmDevice,
    d: &FilmImage,
    dst: &FilmImage,
    win: FRect,
) -> Result<(), GpuError> {
    check_window("film colour out", win, &[d, dst])?;
    let s = &dev.scalars;
    let mut u = FilmU::new([0; 4], dst, win);
    u.h[2][3] = u32::from(dev.fast);
    if !dev.fast {
        passes.context().counters_shared().count_precise();
    }
    u.a = d.desc();
    u.op = [0, 0, u32::from(dev.is_slide()), dev.core.steps];
    u.m[..3].copy_from_slice(&rows(&s.output_to_xyz));
    // The print exposure shift as a hi + lo pair in the third and fourth lanes.
    let shift = s.print_log10_shift;
    u.p[0] = [
        dev.core.s_lo as f32,
        dev.core.s_span as f32,
        shift as f32,
        (shift - f64::from(shift as f32)) as f32,
    ];
    let [ns, nt, nk] = dev.core.dims;
    u.h[0] = [ns as u32, nt as u32, nk as u32, 0];
    // The tables' axes as pairs (lo, and 1 / (hi - lo)): the position of a density in a table is carried without the
    // rounding of its own (`rt_*_pair`, docs/24 section 3.9c).
    let axes = |t: &Table3Data, r_hi: usize, lo_lo: usize, r_lo: usize, u: &mut FilmU| {
        for c in 0..3 {
            let l = t.lo[c];
            let r = 1.0 / (t.hi[c] - t.lo[c]);
            u.p[lo_lo][c] = (l - f64::from(l as f32)) as f32;
            u.p[r_hi][c] = r as f32;
            u.p[r_lo][c] = (r - f64::from(r as f32)) as f32;
        }
    };
    let (t3a_buf, t3a_size) = match &dev.t3a {
        Some((t, b)) => {
            u.p[1] = v3(t.lo, t.power as f32);
            u.p[2] = v3(t.hi, 0.0);
            axes(t, 8, 7, 9, &mut u);
            (b.clone(), t.size as u32)
        }
        None => (dev.dummy.clone(), 2),
    };
    let (t3b, t3b_buf) = &dev.t3b;
    u.p[3] = v3(t3b.lo, t3b.power as f32);
    u.p[4] = v3(t3b.hi, 0.0);
    axes(t3b, 11, 10, 12, &mut u);
    u.h[1] = [t3a_size, t3b.size as u32, 0, 0];
    let curves = match &dev.paper {
        Some((c, b)) => {
            u.p[5] = v3(c.lo, 0.0);
            let k = |i: usize| (CURVE_ENTRIES - 1) as f64 / (c.hi[i] - c.lo[i]);
            u.p[6] = v3([k(0), k(1), k(2)], 0.0);
            for i in 0..3 {
                u.p[13][i] = (c.lo[i] - f64::from(c.lo[i] as f32)) as f32;
                u.p[14][i] = (k(i) - f64::from(k(i) as f32)) as f32;
            }
            b.clone()
        }
        None => dev.dummy.clone(),
    };
    // The exact display colour goes to the shadow (a pass given no shadow writes one nobody reads).
    let scratch = match &dst.shadow {
        Some(_) => None,
        None => {
            // A view writes no shadow: the binding gets the smallest image.
            let (w, h) = if dev.fast {
                (1, 1)
            } else {
                (dst.img.width, dst.img.height)
            };
            Some(GpuImage::alloc(
                passes.context(),
                "film shadow scratch",
                w,
                h,
                3,
                Placement::whole(w, h),
            )?)
        }
    };
    let shadow: &GpuImage = dst
        .shadow
        .as_ref()
        .or(scratch.as_ref())
        .ok_or_else(|| GpuError::InvalidInput("film colour out: no shadow".to_owned()))?;
    u.c = [
        dst.rect.x0,
        dst.rect.y0,
        shadow.stride as i32,
        shadow.plane_len() as i32,
    ];
    let ub = passes.uniform("film colour out uniform", &u)?;
    for b in [&curves, &t3a_buf, t3b_buf, &dev.nodes, &dev.reach] {
        passes.hold(b.clone());
    }
    let shadow_binding = shadow.buf.binding();
    let dispatched = passes.dispatch(
        KernelId::FilmColourOut,
        "film colour out",
        &[
            (0, ub.binding()),
            (1, d.img.buf.binding()),
            (2, dst.img.buf.binding()),
            (3, curves.binding()),
            (4, t3a_buf.binding()),
            (5, t3b_buf.binding()),
            (6, dev.nodes.binding()),
            (7, dev.reach.binding()),
            (8, shadow_binding),
        ],
        groups(win),
    );
    if let Some(img) = scratch {
        let _ = passes.keep(img.buf);
    }
    dispatched
}

// ---------------------------------------------------------------------------------------------------------
// The whole node on a tile
// ---------------------------------------------------------------------------------------------------------

/// The halos of the three spatial passes (`ym_film::spatial::Halos`): halation, diffusion, grain.
#[must_use]
pub fn halos(plan: &FilmPlan) -> [usize; 3] {
    [
        terms_halo(&plan.halation),
        terms_halo(&plan.diffusion),
        if plan.grain.is_some() {
            blur_halo(plan.cloud_px)
        } else {
            0
        },
    ]
}

/// The part of the LogC4 input a tile needs: the tile and every halo, cut to the frame.
#[must_use]
pub fn required_input(plan: &FilmPlan, tile: FRect) -> FRect {
    tile.expand(halos(plan).iter().sum(), plan.frame)
}

/// The Film node's passes P1 to P5 on `tile` of the photo frame, in the order and with the regions of
/// `FilmFrame::run_tile`. `input` is LogC4 / AWG4 and covers at least [`required_input`]; the result is LogC4 / AWG4 over
/// `tile`, the same bits the whole frame gives at those pixels.
///
/// # Errors
///
/// [`GpuError::InvalidInput`] when the tile is empty or outside the frame or the input is too small; the pool's errors.
pub fn encode_film_tile(
    ctx: &GpuContext,
    passes: &mut PassList<'_>,
    dev: &FilmDevice,
    plan: &FilmPlan,
    input: &FilmImage,
    tile: FRect,
) -> Result<FilmImage, GpuError> {
    let frame = plan.frame;
    if tile.is_empty() || !FRect::frame(frame).contains(&tile) {
        return Err(GpuError::InvalidInput(format!(
            "film: the tile {tile:?} is empty or outside the frame"
        )));
    }
    let h = halos(plan);
    let need = required_input(plan, tile);
    if !input.rect.contains(&need) {
        return Err(GpuError::InvalidInput(format!(
            "film: the input {:?} does not cover the {need:?} the tile needs",
            input.rect
        )));
    }
    let r_d = tile.expand(h[2], frame);
    let r_q = r_d.expand(h[1], frame);
    let r_e = r_q.expand(h[0], frame);
    // P1.
    let e = FilmImage::alloc(ctx, "film E", r_e)?;
    encode_colour_in(passes, dev, frame, input, &e, r_e)?;
    // P2.
    let e2 = if plan.halation.is_empty() {
        e
    } else {
        let m = FilmImage::alloc(ctx, "film E'", r_q)?;
        encode_mix(ctx, passes, dev, frame, &e, r_q, &plan.halation, &m)?;
        passes.keep(e.img.buf);
        m
    };
    // P3.
    let d = FilmImage::alloc(ctx, "film D", r_d)?;
    if plan.diffuse && !plan.diffusion.is_empty() {
        let q0 = FilmImage::alloc(ctx, "film Q0", r_q)?;
        encode_develop(passes, dev, DevelopMode::Inhibitor, &e2, None, &q0, r_q)?;
        let q = FilmImage::alloc(ctx, "film Q", r_d)?;
        encode_mix(ctx, passes, dev, frame, &q0, r_d, &plan.diffusion, &q)?;
        encode_develop(
            passes,
            dev,
            DevelopMode::WithInhibitor,
            &e2,
            Some(&q),
            &d,
            r_d,
        )?;
        passes.keep(q0.img.buf);
        passes.keep(q.img.buf);
    } else {
        encode_develop(passes, dev, DevelopMode::Density, &e2, None, &d, r_d)?;
    }
    passes.keep(e2.img.buf);
    // P4.
    let dg = match &plan.grain {
        Some(grain) => {
            let out = FilmImage::alloc(ctx, "film Dg", tile)?;
            encode_grain(ctx, passes, dev, plan, grain, &d, tile, &out)?;
            passes.keep(d.img.buf);
            out
        }
        None => d,
    };
    // P5.
    let out = if dev.fast {
        FilmImage::alloc(ctx, "film out", tile)?
    } else {
        FilmImage::alloc_with_shadow(ctx, "film out", tile)?
    };
    encode_colour_out(passes, dev, &dg, &out, tile)?;
    passes.keep(dg.img.buf);
    Ok(out)
}

/// The WGSL of one Film kernel.
pub(super) fn body(id: KernelId) -> String {
    const FILM: &str = include_str!("../shaders/film.wgsl");
    let section = |name: &str| -> &'static str {
        let marker = format!("//@@ {name}\n");
        let start = FILM.find(&marker).map_or(0, |i| i + marker.len());
        let rest = &FILM[start..];
        let end = rest.find("//@@ ").unwrap_or(rest.len());
        &rest[..end]
    };
    let mut s = constants();
    s.push_str(section("common"));
    let (decls, sections, entry): (&str, &[&str], &str) = match id {
        KernelId::FilmColourIn => (
            "@group(0) @binding(1) var<storage, read> src: array<f32>;\n\
             @group(0) @binding(2) var<storage, read_write> dst: array<f32>;\n\
             @group(0) @binding(3) var<storage, read> t1: array<f32>;\n\
             @group(0) @binding(4) var<storage, read> locus: array<f32>;\n",
            &["colour_in"],
            "ci_main",
        ),
        KernelId::FilmSpatial => (
            "@group(0) @binding(1) var<storage, read> src: array<f32>;\n\
             @group(0) @binding(2) var<storage, read_write> dst: array<f32>;\n\
             @group(0) @binding(3) var<storage, read> taps: array<f32>;\n\
             @group(0) @binding(4) var<storage, read> src2: array<f32>;\n",
            &["spatial"],
            "sp_main",
        ),
        KernelId::FilmDevelop => (
            "@group(0) @binding(1) var<storage, read> src: array<f32>;\n\
             @group(0) @binding(2) var<storage, read_write> dst: array<f32>;\n\
             @group(0) @binding(3) var<storage, read> curves: array<f32>;\n\
             @group(0) @binding(4) var<storage, read> src2: array<f32>;\n",
            &["develop"],
            "dv_main",
        ),
        KernelId::FilmGrain => (
            "@group(0) @binding(1) var<storage, read> src: array<f32>;\n\
             @group(0) @binding(2) var<storage, read_write> dst: array<f32>;\n\
             @group(0) @binding(3) var<storage, read> lut: array<f32>;\n",
            &["grain"],
            "gr_main",
        ),
        _ => (
            "@group(0) @binding(1) var<storage, read> src: array<f32>;\n\
             @group(0) @binding(2) var<storage, read_write> dst: array<f32>;\n\
             @group(0) @binding(3) var<storage, read> curves: array<f32>;\n\
             @group(0) @binding(4) var<storage, read> t3a: array<f32>;\n\
             @group(0) @binding(5) var<storage, read> t3b: array<f32>;\n\
             @group(0) @binding(6) var<storage, read> nodes: array<f32>;\n\
             @group(0) @binding(7) var<storage, read> reach: array<f32>;\n\
             @group(0) @binding(8) var<storage, read_write> shadow: array<f32>;\n",
            &[],
            "rt_main",
        ),
    };
    s.push_str("@group(0) @binding(0) var<uniform> fu: FU;\n");
    s.push_str(decls);
    if matches!(
        id,
        KernelId::FilmColourIn | KernelId::FilmDevelop | KernelId::FilmColourOut
    ) {
        // Accurate log, exp and LogC4 (F-08d): the barrier word is the frame's last lane, 0 in these three passes.
        s.push_str("fn px_zero() -> u32 {\n    return bitcast<u32>(fu.frame.w);\n}\n\n");
        // A view runs the plain f32 arithmetic (`FilmDevice::fast`), an export the compensated one (`h[2].w`).
        s.push_str("fn px_fast() -> bool {\n    return fu.h[2].w == 1u;\n}\n\n");
        s.push_str(include_str!("../shaders/precise.wgsl"));
    }
    for sec in sections {
        s.push_str(section(sec));
    }
    if id == KernelId::FilmColourOut {
        let t3 = include_str!("../shaders/film_t3.wgsl");
        for (name, tab) in [("t3a", "t3a"), ("t3b", "t3b")] {
            s.push_str(&t3.replace("@NAME@", name).replace("@TAB@", tab));
        }
        s.push_str(include_str!("../shaders/film_return.wgsl"));
    }
    s.push_str(&format!(
        "\n@compute @workgroup_size({GROUP}, {GROUP}, 1)\n\
         fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{ {entry}(gid); }}\n"
    ));
    s
}

#[cfg(test)]
mod tests {
    use std::mem::{offset_of, size_of};

    use naga::valid::{Capabilities, ValidationFlags, Validator};

    use super::*;
    use crate::device::StorageMode;
    use crate::kernels::shader_source;

    #[test]
    fn the_uniform_block_has_the_layout_of_the_wgsl_struct() {
        let source = shader_source(StorageMode::F32, KernelId::FilmColourOut);
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
        Validator::new(ValidationFlags::all(), Capabilities::empty())
            .validate(&module)
            .expect("valid");
        let var = module
            .global_variables
            .iter()
            .map(|(_, v)| v)
            .find(|v| v.name.as_deref() == Some("fu"))
            .expect("the uniform fu");
        let naga::TypeInner::Struct { members, span } = &module.types[var.ty].inner else {
            panic!("fu is not a struct");
        };
        let fields = [
            ("frame", offset_of!(FilmU, frame)),
            ("a", offset_of!(FilmU, a)),
            ("b", offset_of!(FilmU, b)),
            ("c", offset_of!(FilmU, c)),
            ("win", offset_of!(FilmU, win)),
            ("op", offset_of!(FilmU, op)),
            ("p", offset_of!(FilmU, p)),
            ("m", offset_of!(FilmU, m)),
            ("h", offset_of!(FilmU, h)),
        ];
        assert_eq!(members.len(), fields.len());
        for (member, (name, offset)) in members.iter().zip(fields) {
            assert_eq!(member.name.as_deref(), Some(name));
            assert_eq!(member.offset as usize, offset, "offset of {name}");
        }
        assert_eq!(*span as usize, size_of::<FilmU>());
        assert_eq!(size_of::<FilmU>(), 496);
    }

    #[test]
    fn the_newton_settings_are_generated_from_ym_color_not_typed() {
        use ym_color::display::cine;
        let src = shader_source(StorageMode::F32, KernelId::FilmColourOut);
        for (n, v) in cine::kernel_refine_scalars() {
            let want = if n.ends_with("ITERATIONS") {
                format!("const FX_{n}: u32 = {}u;", v as u32)
            } else {
                format!("const FX_{n}: f32 = {:?};", v as f32)
            };
            assert!(src.contains(&want), "missing {want}");
        }
        // The shader's own text uses the names, not the numbers.
        let ret = include_str!("../shaders/film_return.wgsl");
        for typed in ["0.7", "0.25", "0.001", "1.0e-13", "10.0, 0.1"] {
            assert!(!ret.contains(typed), "{typed} is typed in film_return.wgsl");
        }
        assert_eq!(cine::STEP_LIMIT, 0.7);
        assert_eq!(cine::STEP_SCALES, [1.0, 0.5, 0.25]);
    }

    #[test]
    fn a_taps_buffer_sums_to_one() {
        for sigma in [0.3, 1.0, 3.9, 12.0] {
            let w = gauss_taps(sigma);
            let total = w[0] + 2.0 * w[1..].iter().sum::<f64>();
            assert!((total - 1.0).abs() < 1e-12, "{sigma}: {total}");
        }
    }
}

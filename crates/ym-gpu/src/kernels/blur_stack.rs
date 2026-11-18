//! The Blur nodes of a plan on the device (`docs/33-blur.md` section 6, `docs/17-gpu.md` section 11b): the scaffold every type
//! plugs its kernel into.
//!
//! A Blur node is a spatial step on the Film node's route (`kernels::film_stack`): the render makes the node's picture **before
//! the colour pass** and the colour kernel's step takes it as it is (`N_BLUR`: the blend in linear light is done here, with
//! the node's mask). For each live Blur node, over the area `read` it reads (the area `made` the nodes after it need, grown by
//! the node's reach and cut to the photo's extent, from the scheduler):
//!
//! 1. the node's *prefix* chain colours the picture it reads (LogC4 / AWG4, [`FilmNodeSpec::prefix`](super::film_stack)) and
//!    the node's *mask* chain (the plan stopped at the node's `m`, as the Show mask view's) makes `m`, both over `read`;
//! 2. **prep** ([`encode_prep`]): decode to linear, the highlight weight, `omega = h m` (Respect on) or `h`, and the weighted
//!    planes `omega X` and `omega` (four planes);
//! 3. the **pyramid** ([`Pyramid`], [`encode_pyramid_level`]): the 2 x 2 box means of those planes at absolute alignment,
//!    level 0 (the planes) up to the level the type's kernel asks for ([`BlurKernel::levels`]); a level is a [`CellGrid`], and
//!    the coarse grid of Soft, Lens and Motion is one of its levels;
//! 4. **the type's kernel** ([`BlurKernel::encode`]) writes the sums `N` and `D` over `made` (four planes: `N` in three, `D` in
//!    the fourth) from the pyramid: the one pass that differs between types, **behind a small interface** (below);
//! 5. **blend** ([`encode_blend`]): `out = X + m (N - X D) / (D + EPS)` in linear light and the encode back to LogC4, with the
//!    exact `m = 0` branch (the input's bits) and `m = 1` branch (`encode(N / (D + EPS))`), into the node's picture over `made`.
//!
//! Every pass has a CPU twin in `f64` ([`crate::cpu::blur`]) and a parity test (`tests/gpu_blur.rs`).
//!
//! # The seam for a type
//!
//! A type (BL-06 Soft, BL-07 Lens, BL-08 Motion, BL-09 Zoom) is one [`BlurKernel`]: it names the highest pyramid level it
//! reads, and records its passes into the list, reading the pyramid and writing `N` and `D` over the made area. It owns its
//! WGSL (a `KernelId` of its own, `blur_soft.wgsl`, ...) and its uniform; it plugs in at one place, [`device_kernel`], which
//! maps a node's [`BlurParams`] to its kernel. A type with no kernel there (`None`) is **refused on the device**
//! ([`GpuError::InvalidInput`], "a Soft Blur node is not on the device yet"): the frame falls back to the CPU with that reason
//! (`RenderReport::cpu_reason`), the other types of the same graph unaffected. The helpers a kernel uses are
//! [`encode_upsample`] (a level read back to pixels, nearest or bilinear) and the pyramid's levels (`Pyramid::level`).
//!
//! # Tiles
//!
//! Cells and levels are at **absolute positions** ([`CellGrid`]), the taps and weights are functions of the pixel and the
//! settings alone, and the sums of a pixel run in a fixed order, so a tile of any size equals the whole frame bit for bit. The
//! kernel's reach is the scheduler's ([`BlurWork::reach`], `ym_color::blur::reach_px`), which already holds three cells of
//! slack beyond what a kernel reads; a kernel reads a cell only inside the grid the read area gives it (a cell the frame's edge
//! cuts counts its missing pixels as 0).

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use ym_color::blur::{
    AWG4_LUMA, BlurKind, BlurParams, EPS, HIGHLIGHT_EV, HIGHLIGHT_GAIN, is_identity,
};
use ym_color::display::ColorChain;
use ym_color::mask_raster::Area;

use super::shader::KernelId;
use super::util::GROUP;
use crate::cpu::Placement;
use crate::cpu::blur::CellGrid;
use crate::cpu::tiling::Rect;
use crate::device::{GpuContext, GpuError};
use crate::image::GpuImage;
use crate::kernels::film::{FRect, FilmImage};
use crate::stream::PassList;

/// The sums `N` and `D` of a node over the area it makes: four planes, `N` in planes 0 to 2 (linear AWG4, weighted) and `D` in
/// plane 3, over `area` (absolute pixels of the photo's extent). A type's kernel writes every pixel of it.
#[derive(Debug)]
pub struct SumsImage {
    /// The planes: `area.w x area.h`, four planes.
    pub img: GpuImage,
    /// The pixels it holds.
    pub area: Area,
}

/// One level of the pyramid: the cells of `grid` as a four-plane image (`omega X` and `omega` averaged over the cell).
#[derive(Debug)]
pub struct PyramidLevel {
    /// The cells the image holds.
    pub grid: CellGrid,
    /// The planes: `grid.w x grid.h`, four planes. Level 0 is the pixels of the read area.
    pub img: GpuImage,
}

/// The weighted planes of a node's read area and the levels above them (`docs/33` section 3.5).
#[derive(Debug)]
pub struct Pyramid {
    /// The photo's extent: cells cut by its edge count their missing pixels as 0.
    pub frame: Area,
    /// The levels, level 0 first.
    pub levels: Vec<PyramidLevel>,
}

impl Pyramid {
    /// The level `level`, or `None` when it has no whole cell in the read area.
    #[must_use]
    pub fn level(&self, level: u32) -> Option<&PyramidLevel> {
        self.levels.get(level as usize)
    }

    /// The pixels of the read area (level 0).
    #[must_use]
    pub fn read_area(&self) -> Area {
        self.levels[0].grid.pixels()
    }
}

/// A type's device kernel: the one pass of a Blur node that differs between types (`docs/33` section 3.5). It reads the pyramid of
/// the weighted planes and writes the sums `N` and `D` over the area the node makes; the scaffold does everything else.
pub trait BlurKernel: Send + Sync + std::fmt::Debug {
    /// The type's name in the refusals and the labels ("Soft", "Lens", "Motion", "Zoom").
    fn name(&self) -> &'static str;

    /// The highest level of the pyramid the kernel reads (0 when it reads the pixels only). The scaffold builds the levels up to
    /// it; a level with no whole cell in the read area is absent and [`BlurKernel::encode`] is given a shorter pyramid.
    fn levels(&self) -> u32;

    /// Records the passes that write `N` and `D` over `out.area` from `src`. A pixel of `out.area` reads the pixels (and the
    /// cells) within the node's reach of it; what lies outside the pyramid's grids has no weight.
    ///
    /// # Errors
    ///
    /// [`GpuError::InvalidInput`] when the pyramid lacks a level the kernel needs; the pool's and the passes' errors.
    fn encode(
        &self,
        ctx: &GpuContext,
        passes: &mut PassList<'_>,
        src: &Pyramid,
        out: &SumsImage,
    ) -> Result<(), GpuError>;
}

/// The device kernel of the type `params` names, or `None` while its task has not landed. **The one place a type plugs in**
/// (BL-06 Soft, BL-07 Lens, BL-08 Motion, BL-09 Zoom): add its arm here. A `None` is a refusal per type, not per node: the
/// plan's other nodes still run on the device, and a frame that holds the node is made by the CPU.
#[must_use]
pub fn device_kernel(params: &BlurParams) -> Option<Arc<dyn BlurKernel>> {
    match params.kind {
        BlurKind::Soft { .. }
        | BlurKind::Lens { .. }
        | BlurKind::Motion { .. }
        | BlurKind::Zoom { .. } => None,
    }
}

/// The name of a type, as the refusals write it.
#[must_use]
pub fn type_name(params: &BlurParams) -> &'static str {
    match params.kind {
        BlurKind::Soft { .. } => "Soft",
        BlurKind::Lens { .. } => "Lens",
        BlurKind::Motion { .. } => "Motion",
        BlurKind::Zoom { .. } => "Zoom",
    }
}

/// A Blur node's work: its numbers, the chain that makes its mask `m`, its reach and its kernel.
#[derive(Debug, Clone)]
pub struct BlurWork {
    /// The type and its numbers (`ym_color::blur`): Highlights (Lens) and Respect mask edges are read by the scaffold.
    pub params: BlurParams,
    /// The colour chain stopped at the node's `m` (the plan compiled to the node's mask, no look, the node bypassed), which
    /// makes the node's mask in all three planes at `Stop::C10`.
    pub mask: Arc<ColorChain>,
    /// The pixels the node reads beyond the pixels it makes (`ym_color::blur::reach_px`).
    pub reach: usize,
    /// The type's kernel, `None` while it has none ([`device_kernel`]): the node is refused on the device.
    pub kernel: Option<Arc<dyn BlurKernel>>,
}

// ---------------------------------------------------------------------------------------------------------
// The uniform block
// ---------------------------------------------------------------------------------------------------------

/// The uniform block of `blur.wgsl` (`BU`), 112 bytes. An image is its place `(x, y)` of pixel (or cell) (0, 0), its row stride
/// and its plane length.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct BlurU {
    a: [i32; 4],
    b: [i32; 4],
    c: [i32; 4],
    d: [i32; 4],
    win: [i32; 4],
    p: [f32; 4],
    op: [u32; 4],
}

/// An image placed at `(x, y)`.
fn placed(img: &GpuImage, x: i64, y: i64) -> [i32; 4] {
    [
        x as i32,
        y as i32,
        img.stride as i32,
        img.plane_len() as i32,
    ]
}

fn groups(w: usize, h: usize) -> (u32, u32, u32) {
    ((w as u32).div_ceil(GROUP), (h as u32).div_ceil(GROUP), 1)
}

fn size_error(what: &str, img: &GpuImage, w: usize, h: usize, planes: usize) -> GpuError {
    GpuError::InvalidInput(format!(
        "blur {what}: a {} x {} image of {} planes where {w} x {h} of at least {planes} are needed",
        img.width, img.height, img.planes
    ))
}

// ---------------------------------------------------------------------------------------------------------
// The passes
// ---------------------------------------------------------------------------------------------------------

/// Records the prep pass: from `picture` (LogC4 / AWG4, three or four planes over `read`) and `mask` (plane 0 over `read`) the
/// four weighted planes `omega X` and `omega` into `dst` (four planes over `read`). `highlights` is Lens' control (0 to 100,
/// 0 for the other types) and `respect` is Respect mask edges.
///
/// # Errors
///
/// [`GpuError::InvalidInput`] when an image is smaller than `read` or has too few planes; shader errors.
pub fn encode_prep(
    passes: &mut PassList<'_>,
    picture: &GpuImage,
    mask: &GpuImage,
    dst: &GpuImage,
    read: Area,
    (highlights, respect): (f64, bool),
) -> Result<(), GpuError> {
    let (w, h) = (read.w, read.h);
    if picture.planes < 3 || picture.width < w || picture.height < h {
        return Err(size_error("prep picture", picture, w, h, 3));
    }
    if mask.width < w || mask.height < h {
        return Err(size_error("prep mask", mask, w, h, 1));
    }
    if dst.planes != 4 || dst.width < w || dst.height < h {
        return Err(size_error("prep planes", dst, w, h, 4));
    }
    let u = BlurU {
        a: placed(picture, read.x, read.y),
        b: placed(mask, read.x, read.y),
        c: placed(dst, read.x, read.y),
        win: [read.x as i32, read.y as i32, w as i32, h as i32],
        p: [
            (highlights / 100.0) as f32,
            f32::from(u8::from(respect)),
            0.0,
            0.0,
        ],
        ..BlurU::default()
    };
    let ub = passes.uniform("blur prep uniform", &u)?;
    passes.dispatch(
        KernelId::BlurPrep,
        "blur prep",
        &[
            (0, ub.binding()),
            (1, picture.buf.binding()),
            (2, mask.buf.binding()),
            (3, dst.buf.binding()),
        ],
        groups(w, h),
    )
}

/// Records one level of the pyramid: the cells of `upper` as the 2 x 2 box means of `lower` (the level below it).
///
/// # Errors
///
/// [`GpuError::InvalidInput`] when the levels are not consecutive or an image is too small; shader errors.
pub fn encode_pyramid_level(
    passes: &mut PassList<'_>,
    lower: &PyramidLevel,
    upper: &PyramidLevel,
) -> Result<(), GpuError> {
    if upper.grid.level != lower.grid.level + 1 {
        return Err(GpuError::InvalidInput(format!(
            "blur pyramid: level {} is not above level {}",
            upper.grid.level, lower.grid.level
        )));
    }
    for (what, l) in [("lower", lower), ("upper", upper)] {
        if l.img.planes != 4 || l.img.width < l.grid.w || l.img.height < l.grid.h {
            return Err(size_error(what, &l.img, l.grid.w, l.grid.h, 4));
        }
    }
    let u = BlurU {
        a: placed(&lower.img, lower.grid.i0, lower.grid.j0),
        b: placed(&upper.img, upper.grid.i0, upper.grid.j0),
        win: [
            upper.grid.i0 as i32,
            upper.grid.j0 as i32,
            upper.grid.w as i32,
            upper.grid.h as i32,
        ],
        op: [0, 0, lower.grid.w as u32, lower.grid.h as u32],
        ..BlurU::default()
    };
    let ub = passes.uniform("blur pyramid uniform", &u)?;
    passes.dispatch(
        KernelId::BlurPyramid,
        "blur pyramid",
        &[
            (0, ub.binding()),
            (1, lower.img.buf.binding()),
            (2, upper.img.buf.binding()),
        ],
        groups(upper.grid.w, upper.grid.h),
    )
}

/// Records the read-back of `cells` to the pixels of `area` of `dst` (four planes, the image's pixel (0, 0) at `area`'s corner):
/// the cell that holds each pixel (`bilinear` false) or the four cells around its centre (`bilinear` true; a cell's centre at
/// `(I + 1/2) f`). A cell outside the grid reads 0. The helper the types' kernels use to put a blurred coarse grid back on
/// the pixels.
///
/// # Errors
///
/// [`GpuError::InvalidInput`] when `dst` does not hold `area` in four planes; shader errors.
pub fn encode_upsample(
    passes: &mut PassList<'_>,
    cells: &PyramidLevel,
    dst: &GpuImage,
    area: Area,
    bilinear: bool,
) -> Result<(), GpuError> {
    if dst.planes != 4 || dst.width < area.w || dst.height < area.h || area.is_empty() {
        return Err(size_error("upsample", dst, area.w, area.h, 4));
    }
    let u = BlurU {
        a: placed(&cells.img, cells.grid.i0, cells.grid.j0),
        b: placed(dst, area.x, area.y),
        win: [area.x as i32, area.y as i32, area.w as i32, area.h as i32],
        op: [
            cells.grid.level,
            u32::from(bilinear),
            cells.grid.w as u32,
            cells.grid.h as u32,
        ],
        ..BlurU::default()
    };
    let ub = passes.uniform("blur upsample uniform", &u)?;
    passes.dispatch(
        KernelId::BlurUpsample,
        "blur upsample",
        &[
            (0, ub.binding()),
            (1, cells.img.buf.binding()),
            (2, dst.buf.binding()),
        ],
        groups(area.w, area.h),
    )
}

/// Records the blend and the encode over `made`: `picture` and `mask` are over `read` (LogC4 / AWG4, and `m` in plane 0),
/// `sums` is `N` and `D` over `made`, and `out` (three planes over `made`) gets the node's picture.
///
/// # Errors
///
/// [`GpuError::InvalidInput`] when `made` is not inside `read` or an image is too small; shader errors.
pub fn encode_blend(
    passes: &mut PassList<'_>,
    (picture, mask): (&GpuImage, &GpuImage),
    read: Area,
    sums: &SumsImage,
    out: &GpuImage,
) -> Result<(), GpuError> {
    let made = sums.area;
    if !read.contains(&made) || made.is_empty() {
        return Err(GpuError::InvalidInput(format!(
            "blur blend: the area {made:?} is empty or outside the {read:?} the node reads"
        )));
    }
    if picture.planes < 3
        || picture.width < read.w
        || picture.height < read.h
        || mask.width < read.w
        || mask.height < read.h
    {
        return Err(size_error("blend inputs", picture, read.w, read.h, 3));
    }
    if sums.img.planes != 4 || sums.img.width < made.w || sums.img.height < made.h {
        return Err(size_error("blend sums", &sums.img, made.w, made.h, 4));
    }
    if out.planes < 3 || out.width < made.w || out.height < made.h {
        return Err(size_error("blend output", out, made.w, made.h, 3));
    }
    let u = BlurU {
        a: placed(picture, read.x, read.y),
        b: placed(mask, read.x, read.y),
        c: placed(&sums.img, made.x, made.y),
        d: placed(out, made.x, made.y),
        win: [made.x as i32, made.y as i32, made.w as i32, made.h as i32],
        ..BlurU::default()
    };
    let ub = passes.uniform("blur blend uniform", &u)?;
    passes.dispatch(
        KernelId::BlurBlend,
        "blur blend",
        &[
            (0, ub.binding()),
            (1, picture.buf.binding()),
            (2, mask.buf.binding()),
            (3, sums.img.buf.binding()),
            (4, out.buf.binding()),
        ],
        groups(made.w, made.h),
    )
}

/// Records the weighted planes and the pyramid of a node over `read`: the prep pass into level 0 and the levels up to `levels`
/// (fewer when a level has no whole cell in `read`).
///
/// # Errors
///
/// [`GpuError::InvalidInput`] for an empty `read` or a `frame` that does not hold it; the pool's and the passes' errors.
pub fn encode_pyramid(
    ctx: &GpuContext,
    passes: &mut PassList<'_>,
    (picture, mask): (&GpuImage, &GpuImage),
    (read, frame): (Area, Area),
    (highlights, respect): (f64, bool),
    levels: u32,
) -> Result<Pyramid, GpuError> {
    if read.is_empty() || !frame.contains(&read) {
        return Err(GpuError::InvalidInput(format!(
            "blur: the area {read:?} is empty or outside the frame {frame:?}"
        )));
    }
    let Some(grid0) = CellGrid::of(read, frame, 0) else {
        return Err(GpuError::InvalidInput("blur: no pixel to read".to_owned()));
    };
    let base = GpuImage::alloc(
        ctx,
        "blur weighted planes",
        read.w,
        read.h,
        4,
        Placement::whole(read.w, read.h),
    )?;
    encode_prep(passes, picture, mask, &base, read, (highlights, respect))?;
    let mut out = Pyramid {
        frame,
        levels: vec![PyramidLevel {
            grid: grid0,
            img: base,
        }],
    };
    for level in 1..=levels {
        let Some(grid) = CellGrid::of(read, frame, level) else {
            break;
        };
        let img = GpuImage::alloc(
            ctx,
            "blur pyramid level",
            grid.w,
            grid.h,
            4,
            Placement::whole(grid.w, grid.h),
        )?;
        let upper = PyramidLevel { grid, img };
        encode_pyramid_level(passes, &out.levels[level as usize - 1], &upper)?;
        out.levels.push(upper);
    }
    Ok(out)
}

/// The picture a Blur node makes over `made` (`docs/33` section 3): `picture` and `mask` over `read` (LogC4 / AWG4 and `m`, both
/// made by the caller's chains), the node's `work` with its kernel, the photo's extent `frame`. The prep, the pyramid, the
/// type's kernel and the blend, as passes of `passes`; the intermediate planes are kept until the list has run. The node's
/// picture is a three-plane image over `made`, as a Film node's.
///
/// # Errors
///
/// [`GpuError::InvalidInput`] when the node has no kernel on the device (the type's refusal: "a Soft Blur node is not on the
/// device yet"), when the node is at its identity (it is not run), or when `made` is not inside `read` inside `frame`; the
/// pool's and the passes' errors.
pub fn encode_blur_picture(
    ctx: &GpuContext,
    passes: &mut PassList<'_>,
    work: &BlurWork,
    (picture, mask): (&GpuImage, &GpuImage),
    (read, made, frame): (Area, Area, Area),
) -> Result<FilmImage, GpuError> {
    let Some(kernel) = work.kernel.as_ref() else {
        return Err(GpuError::InvalidInput(format!(
            "a {} Blur node is not on the device yet",
            type_name(&work.params)
        )));
    };
    if is_identity(&work.params) {
        return Err(GpuError::InvalidInput(
            "a Blur node at its identity is not run".to_owned(),
        ));
    }
    if !read.contains(&made) || made.is_empty() {
        return Err(GpuError::InvalidInput(format!(
            "blur: the area {made:?} is empty or outside the {read:?} the node reads"
        )));
    }
    let highlights = match work.params.kind {
        BlurKind::Lens { highlights, .. } => highlights,
        _ => 0.0,
    };
    let pyramid = encode_pyramid(
        ctx,
        passes,
        (picture, mask),
        (read, frame),
        (highlights, work.params.respect_mask_edges),
        kernel.levels(),
    )?;
    let sums = SumsImage {
        img: GpuImage::alloc(
            ctx,
            "blur sums",
            made.w,
            made.h,
            4,
            Placement::whole(made.w, made.h),
        )?,
        area: made,
    };
    kernel.encode(ctx, passes, &pyramid, &sums)?;
    let rect = FRect {
        x0: made.x as i32,
        y0: made.y as i32,
        x1: made.x1() as i32,
        y1: made.y1() as i32,
    };
    let out = FilmImage::alloc(ctx, "blur picture", rect)?;
    encode_blend(passes, (picture, mask), read, &sums, &out.img)?;
    // The planes the passes read stay out of the pool until the list has run.
    for level in pyramid.levels {
        let _ = passes.keep(level.img.buf);
    }
    let _ = passes.keep(sums.img.buf);
    Ok(out)
}

/// The window of `a` in the image a chain wrote over `outer` (for the chains that colour the node's read area).
#[must_use]
pub fn window_in(outer: Area, a: Area) -> Rect {
    Rect::new((a.x - outer.x) as usize, (a.y - outer.y) as usize, a.w, a.h)
}

// ---------------------------------------------------------------------------------------------------------
// The shader
// ---------------------------------------------------------------------------------------------------------

/// The generated constants of the Blur passes, from `ym_color::blur` (so the numbers are not typed twice).
fn constants() -> String {
    let f = |v: f64| format!("{:?}", v as f32);
    format!(
        "const BLUR_LUMA: vec3<f32> = vec3<f32>({}, {}, {});\n\
         const BLUR_HIGHLIGHT_GAIN: f32 = {};\n\
         const BLUR_HIGHLIGHT_EV: vec2<f32> = vec2<f32>({}, {});\n\
         const BLUR_EPS: f32 = {};\n",
        f(AWG4_LUMA[0]),
        f(AWG4_LUMA[1]),
        f(AWG4_LUMA[2]),
        f(HIGHLIGHT_GAIN),
        f(HIGHLIGHT_EV[0]),
        f(HIGHLIGHT_EV[1]),
        format_args!("{:e}", EPS as f32),
    )
}

/// The WGSL of one Blur pass: the constants, the shared block and the pass's own file. Called by
/// [`super::shader::shader_source`]; any other id is a programming error and gets an empty body.
pub(super) fn body(id: KernelId) -> String {
    let pass = match id {
        KernelId::BlurPrep => include_str!("../shaders/blur_prep.wgsl"),
        KernelId::BlurPyramid => include_str!("../shaders/blur_pyramid.wgsl"),
        KernelId::BlurUpsample => include_str!("../shaders/blur_upsample.wgsl"),
        KernelId::BlurBlend => include_str!("../shaders/blur_blend.wgsl"),
        _ => return String::new(),
    };
    let mut s = constants();
    s.push_str(include_str!("../shaders/blur.wgsl"));
    s.push('\n');
    s.push_str(pass);
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
        let source = shader_source(StorageMode::F32, KernelId::BlurPrep);
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
        Validator::new(ValidationFlags::all(), Capabilities::empty())
            .validate(&module)
            .expect("valid");
        let var = module
            .global_variables
            .iter()
            .map(|(_, v)| v)
            .find(|v| v.name.as_deref() == Some("bu"))
            .expect("the uniform bu");
        let naga::TypeInner::Struct { members, span } = &module.types[var.ty].inner else {
            panic!("bu is not a struct");
        };
        let fields = [
            ("a", offset_of!(BlurU, a)),
            ("b", offset_of!(BlurU, b)),
            ("c", offset_of!(BlurU, c)),
            ("d", offset_of!(BlurU, d)),
            ("win", offset_of!(BlurU, win)),
            ("p", offset_of!(BlurU, p)),
            ("op", offset_of!(BlurU, op)),
        ];
        assert_eq!(members.len(), fields.len());
        for (member, (name, offset)) in members.iter().zip(fields) {
            assert_eq!(member.name.as_deref(), Some(name));
            assert_eq!(member.offset as usize, offset, "offset of {name}");
        }
        assert_eq!(*span as usize, size_of::<BlurU>());
        assert_eq!(size_of::<BlurU>(), 112);
    }

    #[test]
    fn every_pass_is_valid_wgsl() {
        for id in [
            KernelId::BlurPrep,
            KernelId::BlurPyramid,
            KernelId::BlurUpsample,
            KernelId::BlurBlend,
        ] {
            let source = shader_source(StorageMode::F32, id);
            let module = naga::front::wgsl::parse_str(&source)
                .unwrap_or_else(|e| panic!("{id:?}: {}", e.emit_to_string(&source)));
            Validator::new(ValidationFlags::all(), Capabilities::empty())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{id:?}: {e:?}"));
        }
    }

    #[test]
    fn no_type_has_a_device_kernel_until_its_task_lands() {
        use ym_color::blur::BlurKind;
        for kind in [
            BlurKind::Soft { radius: 10.0 },
            BlurKind::Lens {
                radius: 10.0,
                blades: 0,
                rotation_deg: 0.0,
                highlights: 0.0,
            },
            BlurKind::Motion {
                length: 10.0,
                angle_deg: 0.0,
            },
            BlurKind::Zoom {
                amount: 10.0,
                center: [0.5, 0.5],
            },
        ] {
            let params = BlurParams {
                kind,
                respect_mask_edges: true,
            };
            assert!(device_kernel(&params).is_none(), "{}", type_name(&params));
        }
    }
}

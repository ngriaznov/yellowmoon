//! The Face Refinement node's retouch on the device (`docs/27-skin-retouch.md` sections 6.2 and 7, S-13).
//!
//! The CPU twin is `ym-retouch`: `retouch_skin_region_toned` (blemish removal of every face of a region, then the tools of each
//! face) on the linear AWG4 pixels of a region of the working image, with the LogC4 decode and encode `film_pass` puts around it.
//! This module makes the CPU's *plan* for the region (`ym_retouch::plan_skin_region`: the faces that meet it, the cached maps,
//! the masks resampled to the working image and cut to the region, the padded regions and radii: all the CPU's own code and
//! f64 maths) and runs the per-pixel and per-plane maths of the plan on the device, in f32, in `face.wgsl`:
//!
//! 1. **decode**: the node's input (LogC4, from the prefix chain) to linear AWG4;
//! 2. **the tone relation** ([`ToneSource`]): the plain gain, the CPU's curves, or the device's own estimate from its
//!    quantiles of the region's pixels against the analysis picture's (`tone.rs`'s second half);
//! 3. **blemish removal** of each face in order (6.2): the map bilinear at the pixel, the strength, the gain;
//! 4. **the tools of each face** in order (7): the layer `S` (the region, or its 2 x 2 area downscale), `T = blur(S)`, the
//!    mask-normalised blurs, the per-pixel blend of dark circles, even skin and grading; then teeth whitening with `L_max` the
//!    nearest-rank 99th percentile found by a radix select;
//! 5. **encode**: LogC4 again where a tool changed the pixel, a picture of zeros elsewhere ("no picture here": the colour
//!    pass keeps the node's input for it, `NODE_FACE`).
//!
//! The box filter is separable and slides a sum that restarts at every 64th output; the CPU's is the same filter in f64.
//! Sizes: the largest buffer is the layer scratch, up to twenty planes of the (half-resolution) padded region; a region that
//! passes the device's binding is refused with [`GpuError::BufferTooLarge`] and the frame is made on the CPU.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use super::mask_raster::RasterPlanes;
use super::shader::KernelId;
use super::util::{GROUP, STORAGE_IN, STORAGE_SCRATCH, storage_buffer};
use crate::cpu::Placement;
use crate::device::{GpuContext, GpuError};
use crate::image::GpuImage;
use crate::pool::PooledBuffer;
use crate::stream::PassList;
use crate::tiles::device_binding_bytes;

/// Equal-count bins of the tone relation (`ym_retouch::TONE_BINS`).
pub const TONE_BINS: usize = 32;

/// The pixels of a blemish map, `(gain.r, gain.g, gain.b, alpha)` row by row (the CPU's `BlemishMap::data`).
pub trait MapData: AsRef<[[f32; 4]]> + Send + Sync {}
impl<T: AsRef<[[f32; 4]]> + Send + Sync> MapData for T {}

/// A rectangle in whole pixels: column, row, width, height.
pub type PixelRect = (i32, i32, u32, u32);

/// One tone channel as plain numbers (`ym_retouch::ToneCurveData`).
#[derive(Debug, Clone)]
pub struct ToneCurve {
    /// `ln A` of each knot, strictly increasing.
    pub ln_analysis: Vec<f64>,
    /// `ln X` of each knot, non-decreasing.
    pub ln_input: Vec<f64>,
    /// The slope below the first knot.
    pub slope_lo: f64,
    /// The slope above the last knot.
    pub slope_hi: f64,
}

/// Blemish removal of one face on a region (6.2).
#[derive(Clone)]
pub struct BlemishStep {
    /// The map's rectangle in the pixels the map was made at (`BlemishMap::rect`).
    pub rect: PixelRect,
    /// The map's size in pixels (`BlemishMap::width_px`, `height_px`).
    pub size: (u32, u32),
    /// The map's pixels.
    pub data: Arc<dyn MapData>,
    /// The Blemish amount, 0..=1 (above 0).
    pub amount: f64,
}

impl std::fmt::Debug for BlemishStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlemishStep")
            .field("rect", &self.rect)
            .field("size", &self.size)
            .field("amount", &self.amount)
            .finish_non_exhaustive()
    }
}

/// Dark circles, even skin and grading of one face on a region (7.1 to 7.7): the padded region, the radii and strengths of
/// `ym_retouch::LowFreqParams`, the amounts, and the tools' masks over the padded region.
#[derive(Debug, Clone)]
pub struct LowStep {
    /// The padded region, in pixels of the region.
    pub rect: PixelRect,
    /// The downscale of the layers: 2 on the half-resolution path, else 1.
    pub ds: usize,
    /// The texture, dark-circle, even-skin and grading radii at the layer resolution.
    pub radii: [usize; 4],
    /// `k_dc`, `sp_dc`, `k_se`, `sp_se`.
    pub strengths: [f64; 4],
    /// Skin contrast `gc` (0: off), shadow desaturation `gd`, shine removal `gs`.
    pub grading: [f64; 3],
    /// The dark-circle mask over `rect`, when the tool is active.
    pub dc: Option<Vec<u8>>,
    /// The even-skin mask over `rect`, when active.
    pub se: Option<Vec<u8>>,
    /// The grading mask over `rect`, when any grading amount is active.
    pub grade: Option<Vec<u8>>,
}

/// Teeth whitening of one face on a region (7.8).
#[derive(Debug, Clone)]
pub struct TeethStep {
    /// The mask's rectangle in the region.
    pub rect: PixelRect,
    /// The mask over `rect`.
    pub bytes: Vec<u8>,
    /// Teeth whitening, 0..=1.
    pub amount: f64,
}

/// What the CPU's plan says one face does on a region (`ym_retouch::DeviceFace`).
#[derive(Debug, Clone, Default)]
pub struct FaceStep {
    /// Blemish removal.
    pub blemish: Option<BlemishStep>,
    /// Dark circles, even skin and grading.
    pub low: Option<LowStep>,
    /// Teeth whitening.
    pub teeth: Option<TeethStep>,
}

/// The plan of a region of the working image (`ym_retouch::DevicePlan`): the faces that meet it, in order.
#[derive(Debug, Clone, Default)]
pub struct FacePlan {
    /// The working pixel of the region's first pixel.
    pub origin_px: (i32, i32),
    /// The size of the region.
    pub size_px: (u32, u32),
    /// The working image over the size the maps and masks were made at.
    pub render_scale: f64,
    /// The faces that meet the region.
    pub faces: Vec<FaceStep>,
}

/// The floats of a tone channel's block in the device's table: status, knots, the two end slopes, `ln A`, `ln X` and the
/// analysis picture's quantiles (`face.wgsl`, `FC_BLOCK`).
const BLOCK: usize = 4 + 3 * TONE_BINS;
/// Values the tone relation is estimated from, at most (`ym-core`'s `TONE_SAMPLES`, the CPU's grid).
pub const TONE_SAMPLES: usize = 65_536;

/// What the blemish gain is applied through (`docs/27` S-12).
#[derive(Debug, Clone)]
pub enum ToneSource {
    /// The plain multiplication (an exposure change upstream, or nothing known).
    Plain,
    /// The CPU's relation, as numbers (a test of the blemish step alone, or a relation made elsewhere).
    Curves(Box<[ToneCurve; 3]>),
    /// The device makes it: `ln` of the analysis picture's [`TONE_BINS`] quantiles of each channel over the region
    /// (`ym_retouch::tone_quantiles`), against its own quantiles of the region's decoded input.
    Estimate(Box<[[f64; TONE_BINS]; 3]>),
}

/// The protection plane of a region: 0 leaves a pixel to the tools, 255 leaves it alone.
#[derive(Clone, Copy)]
pub enum Protect<'a> {
    /// No protection.
    None,
    /// One byte a pixel, rows of the region's width (the CPU's mask as it takes it; tests).
    Bytes(&'a [u8]),
    /// The raster planes of the protection row (`kernels::mask_raster`), u16 codes read as the CPU quantises them to a byte;
    /// `origin` is the plane pixel of the region's first pixel.
    Raster {
        /// The planes.
        planes: &'a RasterPlanes,
        /// The plane of the row.
        plane: usize,
        /// The plane pixel of the region's first pixel.
        origin: (i32, i32),
    },
}

/// One region of the working image to retouch.
pub struct FaceRegion<'a> {
    /// The plan: the faces that meet the region and what they do (`ym_retouch::plan_skin_region`, converted).
    pub plan: &'a FacePlan,
    /// How the blemish gain meets the picture's tone.
    pub tone: &'a ToneSource,
    /// The protection.
    pub protect: Protect<'a>,
}

/// The uniform block of `face.wgsl` (`FU`), 240 bytes: fifteen `vec4`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct FaceU {
    op: [u32; 4],
    pic: [u32; 4],
    src: [u32; 4],
    outi: [i32; 4],
    outs: [i32; 4],
    prot: [u32; 4],
    pro2: [u32; 4],
    a: [i32; 4],
    b: [i32; 4],
    c: [i32; 4],
    d: [i32; 4],
    f: [f32; 4],
    g: [f32; 4],
    h: [f32; 4],
}

#[derive(Clone, Copy)]
enum Op {
    Decode = 0,
    Blemish = 1,
    Down = 2,
    Box = 3,
    Mul = 4,
    MComb = 5,
    YPlane = 6,
    LowFinal = 7,
    TSelect = 8,
    TApply = 9,
    Encode = 10,
    Tone = 11,
    Copy = 12,
}

fn groups2(w: usize, h: usize) -> (u32, u32, u32) {
    ((w as u32).div_ceil(GROUP), (h as u32).div_ceil(GROUP), 1)
}

/// A one-dimensional op over `n` elements: rows of 4096 threads (`fc_lin`).
fn groups_flat(n: usize) -> (u32, u32, u32) {
    (256, (n as u32).div_ceil(4096 * GROUP).max(1), 1)
}

fn i(v: usize) -> i32 {
    v as i32
}

/// The bytes of the masks, the protection and the floats of the maps and tone blocks of a region, with where each is.
struct Tables {
    bytes: Vec<u8>,
    floats: Vec<f32>,
    faces: Vec<FaceOffsets>,
    protect_bytes: Option<usize>,
    tone_base: usize,
}

#[derive(Default, Clone, Copy)]
struct FaceOffsets {
    map: usize,
    dc: Option<usize>,
    se: Option<usize>,
    grade: Option<usize>,
    teeth: Option<usize>,
}

fn push_bytes(bytes: &mut Vec<u8>, data: &[u8]) -> usize {
    let at = bytes.len();
    bytes.extend_from_slice(data);
    while !bytes.len().is_multiple_of(4) {
        bytes.push(0);
    }
    at
}

fn tables_of(region: &FaceRegion<'_>) -> Tables {
    let mut bytes = Vec::new();
    let mut floats: Vec<f32> = Vec::new();
    let mut faces = Vec::with_capacity(region.plan.faces.len());
    let wants_tone = region.plan.faces.iter().any(|f| f.blemish.is_some());
    let mut tone_base = 0;
    if wants_tone {
        tone_base = floats.len();
        floats.resize(tone_base + 3 * BLOCK, 0.0);
        match region.tone {
            ToneSource::Plain => {}
            ToneSource::Curves(curves) => {
                for (c, curve) in curves.iter().enumerate() {
                    let b = tone_base + c * BLOCK;
                    let n = curve.ln_analysis.len().min(TONE_BINS);
                    floats[b] = 2.0;
                    floats[b + 1] = n as f32;
                    floats[b + 2] = curve.slope_lo as f32;
                    floats[b + 3] = curve.slope_hi as f32;
                    for k in 0..n {
                        floats[b + 4 + k] = curve.ln_analysis[k] as f32;
                        floats[b + 4 + TONE_BINS + k] = curve.ln_input[k] as f32;
                    }
                }
            }
            ToneSource::Estimate(q) => {
                for (c, row) in q.iter().enumerate() {
                    let b = tone_base + c * BLOCK;
                    for (k, v) in row.iter().enumerate() {
                        floats[b + 4 + 2 * TONE_BINS + k] = *v as f32;
                    }
                }
            }
        }
    }
    for face in &region.plan.faces {
        let mut o = FaceOffsets::default();
        if let Some(b) = &face.blemish {
            o.map = floats.len();
            for px in b.data.as_ref().as_ref() {
                floats.extend_from_slice(px);
            }
        }
        if let Some(low) = &face.low {
            o.dc = low.dc.as_deref().map(|m| push_bytes(&mut bytes, m));
            o.se = low.se.as_deref().map(|m| push_bytes(&mut bytes, m));
            o.grade = low.grade.as_deref().map(|m| push_bytes(&mut bytes, m));
        }
        if let Some(teeth) = &face.teeth {
            o.teeth = Some(push_bytes(&mut bytes, &teeth.bytes));
        }
        faces.push(o);
    }
    let protect_bytes = match region.protect {
        Protect::Bytes(b) => Some(push_bytes(&mut bytes, b)),
        _ => None,
    };
    Tables {
        bytes,
        floats,
        faces,
        protect_bytes,
        tone_base,
    }
}

/// The layer planes a face's low-frequency pass needs and where they are (in elements of the scratch).
struct LayerMap {
    /// Elements in a layer plane.
    plane: usize,
    lw: usize,
    lh: usize,
    t: usize,
    tmp: usize,
    y: usize,
    m: usize,
    den: usize,
    num: usize,
    c_dc: Option<usize>,
    c_se: Option<usize>,
    c_g: Option<usize>,
    c_shine: Option<usize>,
    /// Elements in all.
    total: usize,
}

fn layer_map(low: &LowStep) -> LayerMap {
    let (rw, rh) = (low.rect.2 as usize, low.rect.3 as usize);
    let ds = low.ds;
    let (lw, lh) = (rw.div_ceil(ds), rh.div_ceil(ds));
    let plane = lw * lh;
    let mut next = 0;
    let mut take = |planes: usize| {
        let at = next * plane;
        next += planes;
        at
    };
    let t = take(3);
    let tmp = take(3);
    let y = take(1);
    let m = take(1);
    let den = take(1);
    let shine = low.grade.is_some() && low.grading[2] > 0.0;
    let num = take(if low.dc.is_some() || shine { 3 } else { 1 });
    let c_dc = low.dc.is_some().then(|| take(3));
    let c_se = low.se.is_some().then(|| take(1));
    let c_g = low.grade.is_some().then(|| take(1));
    let c_shine = shine.then(|| take(3));
    LayerMap {
        plane,
        lw,
        lh,
        t,
        tmp,
        y,
        m,
        den,
        num,
        c_dc,
        c_se,
        c_g,
        c_shine,
        total: next * plane,
    }
}

struct Rec<'a, 'p> {
    passes: &'a mut PassList<'p>,
    base: FaceU,
    pic: &'a GpuImage,
    src: &'a GpuImage,
    out: &'a GpuImage,
    lay: Arc<PooledBuffer>,
    bytes: Arc<PooledBuffer>,
    fd: Arc<PooledBuffer>,
    words: Arc<PooledBuffer>,
}

impl Rec<'_, '_> {
    fn run(&mut self, op: Op, u: FaceU, groups: (u32, u32, u32)) -> Result<(), GpuError> {
        let mut u = u;
        u.op[0] = op as u32;
        let ub = self.passes.uniform("face uniform", &u)?;
        self.passes.dispatch(
            KernelId::Face,
            match op {
                Op::Decode => "face decode",
                Op::Blemish => "face blemish",
                Op::Down => "face downscale",
                Op::Box => "face box",
                Op::Mul | Op::MComb | Op::YPlane => "face layer",
                Op::LowFinal => "face tools",
                Op::TSelect | Op::TApply => "face teeth",
                Op::Encode => "face encode",
                Op::Tone => "face tone",
                Op::Copy => "face place",
            },
            &[
                (0, ub.binding()),
                (1, self.pic.buf.binding()),
                (2, self.lay.binding()),
                (3, self.src.buf.binding()),
                (4, self.out.buf.binding()),
                (5, self.bytes.binding()),
                (6, self.fd.binding()),
                (7, self.words.binding()),
            ],
            groups,
        )
    }

    fn base(&self) -> FaceU {
        self.base
    }

    /// A layer plane: the mean of the `ds x ds` blocks of a plane of the working image (`plane`) or of a mask the size of the
    /// rectangle (`mask`: its first byte).
    #[allow(clippy::too_many_arguments)]
    fn down(
        &mut self,
        rect: (usize, usize, usize, usize),
        source: Result<usize, usize>,
        dst: usize,
        (lw, lh): (usize, usize),
        ds: usize,
    ) -> Result<(), GpuError> {
        let mut u = self.base();
        u.a = [i(rect.0), i(rect.1), i(rect.2), i(rect.3)];
        u.b = [i(dst), i(lw), i(lh), i(ds)];
        match source {
            Ok(plane) => u.c = [i(plane), 0, 0, 0],
            Err(byte) => u.c = [i(byte), 1, 0, 0],
        }
        self.run(Op::Down, u, groups2(lw, lh))
    }

    #[allow(clippy::too_many_arguments)]
    fn box_pass(
        &mut self,
        (src, dst): (usize, usize),
        (lw, lh): (usize, usize),
        q: usize,
        vertical: bool,
        planes: usize,
    ) -> Result<(), GpuError> {
        let mut u = self.base();
        u.a = [i(src), i(dst), i(lw), i(lh)];
        u.b = [i(q), i32::from(vertical), i(planes), i(lw * lh)];
        let (lines, len) = if vertical { (lw, lh) } else { (lh, lw) };
        self.run(
            Op::Box,
            u,
            (
                (lines as u32).div_ceil(GROUP),
                (len as u32).div_ceil(64).div_ceil(GROUP),
                planes as u32,
            ),
        )
    }

    /// `blur(X, r)` of 7.1 in place over `planes` planes at `off`: two boxes of radii `r / 2` and `r - r / 2` along rows, then
    /// along columns, through the scratch at `tmp`.
    fn blur(
        &mut self,
        off: usize,
        tmp: usize,
        size: (usize, usize),
        planes: usize,
        r: usize,
    ) -> Result<(), GpuError> {
        let (q1, q2) = (r / 2, r - r / 2);
        self.box_pass((off, tmp), size, q1, false, planes)?;
        self.box_pass((tmp, off), size, q2, false, planes)?;
        self.box_pass((off, tmp), size, q1, true, planes)?;
        self.box_pass((tmp, off), size, q2, true, planes)
    }

    /// The mask-normalised blur of a layer `x` (`planes` planes at `x_off`) with the mask plane `m` and `den = blur(m, r)` ready:
    /// `dst = m blur(x m, r) / max(den, 1/255) + (1 - m) x`, through `num` (as many planes as `x`).
    #[allow(clippy::too_many_arguments)]
    fn masked_blur(
        &mut self,
        map: &LayerMap,
        x_off: usize,
        planes: usize,
        dst: usize,
        r: usize,
    ) -> Result<(), GpuError> {
        let mut u = self.base();
        u.a = [i(x_off), i(map.m), i(map.num), i(map.plane)];
        u.b = [i(planes), 0, 0, 0];
        self.run(Op::Mul, u, groups_flat(map.plane))?;
        self.blur(map.num, map.tmp, (map.lw, map.lh), planes, r)?;
        let mut u = self.base();
        u.a = [i(map.m), i(map.num), i(map.den), i(x_off)];
        u.b = [i(dst), i(map.plane), i(planes), 0];
        self.run(Op::MComb, u, groups_flat(map.plane))
    }
}

/// The picture one region made: three planes of the region's size, zeros where no tool changed the pixel.
#[derive(Debug)]
pub struct RegionPicture {
    buf: Arc<PooledBuffer>,
    width: usize,
    height: usize,
    stride: usize,
    plane_len: usize,
}

/// The pictures an export has made of a node's regions, so that a tile that meets a region another tile met already copies the
/// picture and does not make it again (the node is a region whole in whichever tile meets it, and the bytes are the same: nothing
/// in a region's maths depends on the tile). One for each region of one render's node; a picture is kept once the list that fills
/// it has been submitted, so a later list is ordered after it on the queue.
#[derive(Debug, Default)]
pub struct KeptRegion {
    picture: std::sync::Mutex<Option<Arc<RegionPicture>>>,
}

impl KeptRegion {
    /// An empty slot.
    #[must_use]
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// The picture a list that was submitted made, if any.
    #[must_use]
    pub fn get(&self) -> Option<Arc<RegionPicture>> {
        self.picture
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    fn keep(&self, picture: Arc<RegionPicture>) {
        *self
            .picture
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(picture);
    }
}

/// Makes the picture of one region and, when `kept` is given, keeps it there once the list has been submitted.
///
/// # Errors
///
/// As [`retouch_region`].
pub fn make_region(
    ctx: &GpuContext,
    passes: &mut PassList<'_>,
    region: &FaceRegion<'_>,
    src: &GpuImage,
    src_at: (usize, usize),
    kept: Option<&Arc<KeptRegion>>,
) -> Result<Arc<RegionPicture>, GpuError> {
    let picture = Arc::new(retouch_region(ctx, passes, region, src, src_at)?);
    if let Some(kept) = kept {
        let (kept, picture) = (Arc::clone(kept), Arc::clone(&picture));
        passes.on_submit(move || kept.keep(picture));
    }
    Ok(picture)
}

/// Copies a region's picture into the node's picture `out` with the region's first pixel at `out_at` (the part outside `out` is
/// dropped; a pixel of zeros is not written, `out` has zeros there).
///
/// # Errors
///
/// [`GpuError::InvalidInput`] for an `out` with fewer than three planes; the pool's and the passes' errors.
pub fn place_region(
    ctx: &GpuContext,
    passes: &mut PassList<'_>,
    picture: &Arc<RegionPicture>,
    out: &GpuImage,
    out_at: (i64, i64),
) -> Result<(), GpuError> {
    if out.planes < 3 {
        return Err(GpuError::InvalidInput(
            "face: the output has fewer than three planes".to_owned(),
        ));
    }
    let stand_in =
        |passes: &mut PassList<'_>, label: &str| -> Result<Arc<PooledBuffer>, GpuError> {
            let buffer = ctx.pool().acquire(label, 4, STORAGE_SCRATCH)?;
            Ok(passes.keep(buffer))
        };
    let u = FaceU {
        op: [Op::Copy as u32, 0, 0, 0],
        pic: [0, 0, picture.width as u32, picture.height as u32],
        src: [picture.stride as u32, picture.plane_len as u32, 0, 0],
        outi: [
            out.stride as i32,
            out.plane_len() as i32,
            out_at.0 as i32,
            out_at.1 as i32,
        ],
        outs: [out.width as i32, out.height as i32, 0, 0],
        ..FaceU::default()
    };
    let (pic, lay, bytes, fd, words) = (
        stand_in(passes, "face stand-in")?,
        stand_in(passes, "face stand-in")?,
        stand_in(passes, "face stand-in")?,
        stand_in(passes, "face stand-in")?,
        stand_in(passes, "face stand-in")?,
    );
    passes.hold(Arc::clone(&picture.buf));
    let ub = passes.uniform("face uniform", &u)?;
    passes.dispatch(
        KernelId::Face,
        "face place",
        &[
            (0, ub.binding()),
            (1, pic.binding()),
            (2, lay.binding()),
            (3, picture.buf.binding()),
            (4, out.buf.binding()),
            (5, bytes.binding()),
            (6, fd.binding()),
            (7, words.binding()),
        ],
        groups2(picture.width, picture.height),
    )
}

/// Records the retouch of one region into `out` (the region's picture is made and placed): [`retouch_region`] and
/// [`place_region`].
///
/// # Errors
///
/// As [`retouch_region`] and [`place_region`].
pub fn encode_face_region(
    ctx: &GpuContext,
    passes: &mut PassList<'_>,
    region: &FaceRegion<'_>,
    src: &GpuImage,
    src_at: (usize, usize),
    out: &GpuImage,
    out_at: (i64, i64),
) -> Result<(), GpuError> {
    let picture = Arc::new(retouch_region(ctx, passes, region, src, src_at)?);
    place_region(ctx, passes, &picture, out, out_at)
}

/// Records the retouch of one region: `src` is the node's input (LogC4 / AWG4, planar, at least three planes) and `src_at` the
/// pixel of it where the region starts. The result is the region's picture: LogC4 where a tool changed the pixel, zeros elsewhere
/// ("no picture here"). Nothing is read back.
///
/// # Errors
///
/// [`GpuError::InvalidInput`] for a region `src` does not cover or a protection plane that does not cover the region;
/// [`GpuError::BufferTooLarge`] when the layer scratch passes the device's binding; the pool's and the passes' errors.
pub fn retouch_region(
    ctx: &GpuContext,
    passes: &mut PassList<'_>,
    region: &FaceRegion<'_>,
    src: &GpuImage,
    src_at: (usize, usize),
) -> Result<RegionPicture, GpuError> {
    let (w, h) = (
        region.plan.size_px.0 as usize,
        region.plan.size_px.1 as usize,
    );
    if w == 0 || h == 0 {
        return Err(GpuError::InvalidInput("face: an empty region".to_owned()));
    }
    if src.planes < 3 || src_at.0 + w > src.width || src_at.1 + h > src.height {
        return Err(GpuError::InvalidInput(format!(
            "face: the {w} x {h} region at {src_at:?} is not inside the {} x {} input",
            src.width, src.height
        )));
    }
    if let Protect::Bytes(b) = region.protect
        && b.len() != w * h
    {
        return Err(GpuError::InvalidInput(format!(
            "face: {} protection bytes for a {w} x {h} region",
            b.len()
        )));
    }
    let tables = tables_of(region);
    let limits = &ctx.info().limits;
    let binding = device_binding_bytes(limits);
    // The layer scratch: the largest any face needs.
    let layers: Vec<Option<LayerMap>> = region
        .plan
        .faces
        .iter()
        .map(|f| f.low.as_ref().map(layer_map))
        .collect();
    let lay_floats = layers
        .iter()
        .flatten()
        .map(|m| m.total)
        .max()
        .unwrap_or(0)
        .max(4);
    let lay_bytes = (lay_floats * 4) as u64;
    if lay_bytes > binding {
        return Err(GpuError::BufferTooLarge {
            requested: lay_bytes,
            limit: binding,
        });
    }
    let pic = GpuImage::alloc(ctx, "face working", w, h, 4, Placement::whole(w, h))?;
    // The region's picture: zeros, then the pixels a tool changed.
    let out = GpuImage::alloc(ctx, "face region picture", w, h, 3, Placement::whole(w, h))?;
    passes.encoder().clear_buffer(&out.buf, 0, None);
    let lay = Arc::new(
        ctx.pool()
            .acquire("face layers", lay_bytes, STORAGE_SCRATCH)?,
    );
    let bytes = Arc::new(storage_buffer(ctx, "face bytes", &tables.bytes)?);
    let fd_bytes: &[u8] = bytemuck::cast_slice(&tables.floats);
    let fd = Arc::new(storage_buffer(ctx, "face tables", fd_bytes)?);
    let (words, word_plane): (Arc<PooledBuffer>, usize) = match region.protect {
        Protect::Raster {
            planes,
            plane,
            origin,
        } => {
            if plane >= planes.planes
                || planes.row0 != 0
                || planes.rows_held != planes.height
                || origin.0 < 0
                || origin.1 < 0
                || origin.0 as usize + w > planes.width
                || origin.1 as usize + h > planes.height
            {
                return Err(GpuError::InvalidInput(format!(
                    "face: the protection planes ({} x {}) do not cover the {w} x {h} region at {origin:?}",
                    planes.width, planes.height
                )));
            }
            (
                Arc::clone(&planes.buf),
                plane * planes.stride * planes.height / 2,
            )
        }
        _ => (
            Arc::new(ctx.pool().acquire("face no planes", 4, STORAGE_IN)?),
            0,
        ),
    };
    passes.hold(Arc::clone(&lay));
    passes.hold(Arc::clone(&bytes));
    passes.hold(Arc::clone(&fd));
    passes.hold(Arc::clone(&words));
    let mut base = FaceU {
        pic: [
            pic.stride as u32,
            pic.plane_len() as u32,
            w as u32,
            h as u32,
        ],
        src: [
            src.stride as u32,
            src.plane_len() as u32,
            src_at.0 as u32,
            src_at.1 as u32,
        ],
        outi: [out.stride as i32, out.plane_len() as i32, 0, 0],
        outs: [w as i32, h as i32, 0, 0],
        ..FaceU::default()
    };
    match region.protect {
        Protect::None => {}
        Protect::Bytes(_) => {
            base.prot = [2, w as u32, 0, 0];
            base.pro2[0] = tables.protect_bytes.unwrap_or(0) as u32;
        }
        Protect::Raster { planes, origin, .. } => {
            base.prot = [1, planes.stride as u32, origin.0 as u32, origin.1 as u32];
            base.pro2[0] = word_plane as u32;
        }
    }
    let mut rec = Rec {
        passes,
        base,
        pic: &pic,
        src,
        out: &out,
        lay,
        bytes,
        fd,
        words,
    };
    rec.run(Op::Decode, rec.base(), groups2(w, h))?;
    // The tone relation, once for the region, from the decoded picture before any face changes it.
    let wants_tone = region.plan.faces.iter().any(|f| f.blemish.is_some());
    let tone_mode = i32::from(!matches!(region.tone, ToneSource::Plain));
    if wants_tone && let ToneSource::Estimate(_) = region.tone {
        let step = (((w * h) as f64 / TONE_SAMPLES as f64).sqrt().ceil() as usize).max(1);
        let mut u = rec.base();
        u.a = [i(w.div_ceil(step)), i(h.div_ceil(step)), i(step), 0];
        u.b = [i(tables.tone_base), 0, 0, 0];
        rec.run(Op::Tone, u, (3, 1, 1))?;
    }
    // 6.2: blemish removal of every face, in order.
    let scale = region.plan.render_scale;
    for (face, off) in region.plan.faces.iter().zip(&tables.faces) {
        let Some(b) = &face.blemish else {
            continue;
        };
        let sx = f64::from(b.size.0) / f64::from(b.rect.2);
        let sy = f64::from(b.size.1) / f64::from(b.rect.3);
        let (rx, ry) = (f64::from(b.rect.0), f64::from(b.rect.1));
        let (ox, oy) = (
            f64::from(region.plan.origin_px.0),
            f64::from(region.plan.origin_px.1),
        );
        let u0 = sx * (ox / scale - rx + 0.5) - 0.5;
        let v0 = sy * (oy / scale - ry + 0.5) - 0.5;
        let mut u = rec.base();
        u.a = [i(off.map), b.size.0 as i32, b.size.1 as i32, tone_mode];
        u.b = [i(tables.tone_base), 0, 0, 0];
        u.f = [
            u0 as f32,
            (sx / scale) as f32,
            v0 as f32,
            (sy / scale) as f32,
        ];
        u.g = [b.amount as f32, 0.0, 0.0, 0.0];
        rec.run(Op::Blemish, u, groups2(w, h))?;
    }
    // 7: the tools of each face, in order.
    for ((face, off), map) in region.plan.faces.iter().zip(&tables.faces).zip(&layers) {
        if let (Some(low), Some(map)) = (&face.low, map) {
            record_low(&mut rec, low, map, off)?;
        }
        if let (Some(teeth), Some(at)) = (&face.teeth, off.teeth) {
            record_teeth(&mut rec, teeth, at)?;
        }
    }
    rec.run(Op::Encode, rec.base(), groups2(w, h))?;
    let _ = passes.keep(pic.buf);
    let (stride, plane_len) = (out.stride, out.plane_len());
    Ok(RegionPicture {
        buf: passes.keep(out.buf),
        width: w,
        height: h,
        stride,
        plane_len,
    })
}

/// Dark circles, even skin and grading of one face (7.1 to 7.7) over its padded region.
fn record_low(
    rec: &mut Rec<'_, '_>,
    low: &LowStep,
    map: &LayerMap,
    off: &FaceOffsets,
) -> Result<(), GpuError> {
    let rect = (
        low.rect.0 as usize,
        low.rect.1 as usize,
        low.rect.2 as usize,
        low.rect.3 as usize,
    );
    let [r_tex, r_dc, r_se, r_g] = low.radii;
    let [k_dc, sp_dc, k_se, sp_se] = low.strengths;
    let [gc, gd, _gs] = low.grading;
    let ds = low.ds;
    let size = (map.lw, map.lh);
    // S, then T = blur(S, r_tex).
    for c in 0..3 {
        rec.down(rect, Ok(c), map.t + c * map.plane, size, ds)?;
    }
    rec.blur(map.t, map.tmp, size, 3, r_tex)?;
    let mut u = rec.base();
    u.a = [i(map.t), i(map.y), i(map.plane), 0];
    rec.run(Op::YPlane, u, groups_flat(map.plane))?;
    // The mask `m` at the layer resolution and `den = blur(m, r)` of a tool.
    let prepare = |rec: &mut Rec<'_, '_>, byte: usize, r: usize| -> Result<(), GpuError> {
        rec.down(rect, Err(byte), map.m, size, ds)?;
        rec.down(rect, Err(byte), map.den, size, ds)?;
        rec.blur(map.den, map.tmp, size, 1, r)
    };
    if let (Some(byte), Some(dst)) = (off.dc, map.c_dc) {
        prepare(rec, byte, r_dc)?;
        rec.masked_blur(map, map.t, 3, dst, r_dc)?;
    }
    if let (Some(byte), Some(dst)) = (off.se, map.c_se) {
        prepare(rec, byte, r_se)?;
        rec.masked_blur(map, map.y, 1, dst, r_se)?;
    }
    if let (Some(byte), Some(dst)) = (off.grade, map.c_g) {
        prepare(rec, byte, r_g)?;
        rec.masked_blur(map, map.y, 1, dst, r_g)?;
        if let Some(shine) = map.c_shine {
            prepare(rec, byte, r_se)?;
            rec.masked_blur(map, map.t, 3, shine, r_se)?;
        }
    }
    let mut flags = 0;
    if gc != 0.0 {
        flags |= 1;
    }
    if gd > 0.0 {
        flags |= 2;
    }
    if map.c_shine.is_some() {
        flags |= 4;
    }
    let none = -1;
    let at = |o: Option<usize>| o.map_or(none, i);
    let mut u = rec.base();
    u.op[1] = map.plane as u32;
    u.a = [i(rect.0), i(rect.1), i(rect.2), i(rect.3)];
    u.b = [i(map.lw), i(ds), i(map.t), i(map.y)];
    u.c = [at(map.c_dc), at(map.c_se), at(map.c_g), at(map.c_shine)];
    u.d = [at(off.dc), at(off.se), at(off.grade), flags];
    u.f = [k_dc as f32, sp_dc as f32, k_se as f32, sp_se as f32];
    u.g = [gc as f32, gd as f32, low.grading[2] as f32, 0.0];
    rec.run(Op::LowFinal, u, groups2(rect.2, rect.3))
}

/// Teeth whitening (7.8): `L_max`, then the pixels.
fn record_teeth(rec: &mut Rec<'_, '_>, teeth: &TeethStep, at: usize) -> Result<(), GpuError> {
    let rect = (
        teeth.rect.0 as usize,
        teeth.rect.1 as usize,
        teeth.rect.2 as usize,
        teeth.rect.3 as usize,
    );
    let n = teeth.bytes.iter().filter(|&&b| b != 0).count();
    let rank = ((0.99 * n as f64).ceil() as usize).clamp(1, n.max(1));
    let mut u = rec.base();
    u.a = [i(rect.0), i(rect.1), i(rect.2), i(rect.3)];
    u.b = [i(at), i(rank), 0, 0];
    rec.run(Op::TSelect, u, (1, 1, 1))?;
    let mut u = rec.base();
    u.a = [i(rect.0), i(rect.1), i(rect.2), i(rect.3)];
    u.b = [i(at), 0, 0, 0];
    u.f = [teeth.amount as f32, 0.0, 0.0, 0.0];
    rec.run(Op::TApply, u, groups2(rect.2, rect.3))
}

/// Retouches a region on the device and reads it back: `logc4` is the node's input over the region (planar, `3 * w * h`), the
/// result the node's picture in the same layout, zeros where no tool changed the pixel. The thin wrapper the tests use.
///
/// # Errors
///
/// As [`encode_face_region`] and the upload and readback.
pub fn face_region_host(
    ctx: &GpuContext,
    region: &FaceRegion<'_>,
    logc4: &[f32],
) -> Result<Vec<f32>, GpuError> {
    let (w, h) = (
        region.plan.size_px.0 as usize,
        region.plan.size_px.1 as usize,
    );
    let src = GpuImage::upload(ctx, "face input", w, h, 3, logc4, Placement::whole(w, h))?;
    let out = GpuImage::alloc(ctx, "face output", w, h, 3, Placement::whole(w, h))?;
    let mut passes = PassList::new(ctx, "face region");
    passes.encoder().clear_buffer(&out.buf, 0, None);
    encode_face_region(ctx, &mut passes, region, &src, (0, 0), &out, (0, 0))?;
    let read = out.encode_read(&mut passes)?;
    let done = passes.submit()?.finish()?;
    out.take(&done, read)
}

/// The WGSL of the face kernel.
pub(super) fn body() -> String {
    let mut s = String::from(include_str!("../shaders/face.wgsl"));
    s.push_str(&format!(
        "\n@compute @workgroup_size({GROUP}, {GROUP}, 1)\n\
         fn main(@builtin(global_invocation_id) gid: vec3<u32>, \
         @builtin(local_invocation_index) local: u32, \
         @builtin(workgroup_id) group: vec3<u32>) {{ face_main(gid, local, group); }}\n"
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
        let source = shader_source(StorageMode::F32, KernelId::Face);
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
        Validator::new(ValidationFlags::all(), Capabilities::empty())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
        let var = module
            .global_variables
            .iter()
            .map(|(_, v)| v)
            .find(|v| v.name.as_deref() == Some("FU_"))
            .expect("the uniform FU_");
        let naga::TypeInner::Struct { members, span } = &module.types[var.ty].inner else {
            panic!("FU_ is not a struct");
        };
        let fields = [
            ("op", offset_of!(FaceU, op)),
            ("pic", offset_of!(FaceU, pic)),
            ("src", offset_of!(FaceU, src)),
            ("outi", offset_of!(FaceU, outi)),
            ("outs", offset_of!(FaceU, outs)),
            ("prot", offset_of!(FaceU, prot)),
            ("pro2", offset_of!(FaceU, pro2)),
            ("a", offset_of!(FaceU, a)),
            ("b", offset_of!(FaceU, b)),
            ("c", offset_of!(FaceU, c)),
            ("d", offset_of!(FaceU, d)),
            ("f", offset_of!(FaceU, f)),
            ("g", offset_of!(FaceU, g)),
            ("h", offset_of!(FaceU, h)),
        ];
        assert_eq!(members.len(), fields.len());
        for (member, (name, offset)) in members.iter().zip(fields) {
            assert_eq!(member.name.as_deref(), Some(name));
            assert_eq!(member.offset as usize, offset, "offset of {name}");
        }
        assert_eq!(*span as usize, size_of::<FaceU>());
    }
}

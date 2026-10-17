//! The photo's raw words on the device, and R1 normalise, R3 vignetting, R4 gains and the clip mask in
//! one pass (`docs/17-gpu.md` section 11).
//!
//! # The raw words, uploaded once
//!
//! [`DeviceRaw`] is the readout of one photo as the decoder made it: two `u16` per `u32`, one or two
//! row slabs of at most one design binding (128 MiB, [`crate::tiles::DESIGN_BINDING_BYTES`]) each, and
//! for a cRAW photo the quantisation step of every photosite next to them. A 61 MP plane of `u16` is
//! 115 MiB and fits one slab; a 70 MP plane is cut at a row into two. The step is kept as a byte per
//! photosite that indexes a table when the photo has at most 256 distinct steps, else as the `f16`
//! bits the decoder made it from (so the upload is exact). Every kernel of the raw front binds the
//! slabs the same way ([`front_raw.wgsl`](../../shaders/front_raw.wgsl)), a photo with one slab
//! binding it twice.
//!
//! # The pass
//!
//! [`encode_prep`] reads a rectangle of the frame from the resident words and writes the balanced
//! mosaic of that rectangle: `max((raw - black) * inv, -0.05)`, times the vignetting gain, times the
//! R4 gain, each product rounded as the CPU rounds it (`opaque`, K8). Without a vignetting profile the
//! output has the CPU's bits; with one the gain is the profile evaluated in `f32` from an exact
//! integer radius (`front_vig.wgsl`). With a mask the pass also writes the clip bits of R1 (one bit per
//! photosite of the rectangle) for R5 ([`super::highlights`]).
//!
//! The CPU reference is `cpu::normalize::normalize_tile` followed by `cpu::vignette::apply_vignette`
//! and `cpu::gains::apply_gains`; a tile of any rectangle equals the whole frame bit for bit (the
//! stage is pointwise and positions are integers, D4/D5).

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use super::shader::KernelId;
use super::util::GROUP;
use crate::cpu::normalize::{CLIP_FRACTION, Levels, NEGATIVE_FLOOR, QSTEP_MARGIN};
use crate::cpu::tiling::Rect;
use crate::cpu::vignette::{MIN_ATTENUATION, VignetteProfile};
use crate::cpu::{Cfa, Placement};
use crate::device::{GpuContext, GpuError};
use crate::image::{GpuImage, IMAGE_USAGE};
use crate::pool::PooledBuffer;
use crate::stream::{PassList, read_bytes, upload_bytes};
use crate::tiles::DESIGN_BINDING_BYTES;

/// The most samples one slab of raw words (or of steps) holds: a design binding of `u16`.
pub const MAX_SLAB_SAMPLES: usize = (DESIGN_BINDING_BYTES / 2) as usize;

/// The most knots the vignetting table takes (the uniform holds 32).
pub const MAX_VIGNETTE_KNOTS: usize = 32;

/// How the quantisation step of a cRAW photo is stored on the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantMode {
    /// The photo has no step plane: R1 clips at 0.995 of the range.
    None,
    /// `f16` bits, two per word, the decoder's own values.
    F16,
    /// One byte per photosite, an index into a table of at most 256 steps.
    Table,
}

impl QuantMode {
    /// The value of `RawFrame::dims.w` in the shaders.
    const fn code(self) -> u32 {
        match self {
            QuantMode::None => 0,
            QuantMode::F16 => 1,
            QuantMode::Table => 2,
        }
    }
}

/// One slab of rows of the raw words (and of the steps).
struct RawSlab {
    words: PooledBuffer,
    steps: Option<PooledBuffer>,
    first_row: usize,
    rows: usize,
}

/// The raw words of one photo on the device (see the module notes).
pub struct DeviceRaw {
    slabs: Vec<RawSlab>,
    width: usize,
    height: usize,
    cfa: Cfa,
    quant: QuantMode,
    /// The distinct steps in raw units, when `quant` is [`QuantMode::Table`].
    table: Vec<f32>,
}

impl std::fmt::Debug for DeviceRaw {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceRaw")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("slabs", &self.slabs.len())
            .field("quant", &self.quant)
            .finish()
    }
}

impl DeviceRaw {
    /// Uploads a `width x height` readout once. `qstep` is the cRAW step per photosite in raw units
    /// (`f32` values that came from `f16`, as `FrameData` holds them).
    ///
    /// # Errors
    ///
    /// [`GpuError::InvalidInput`] for an empty readout, data of the wrong length, a readout that does
    /// not fit two slabs, or a step that is not an `f16` value; the pool's errors.
    pub fn upload(
        ctx: &GpuContext,
        label: &str,
        width: usize,
        height: usize,
        cfa: Cfa,
        data: &[u16],
        qstep: Option<&[f32]>,
    ) -> Result<DeviceRaw, GpuError> {
        DeviceRaw::upload_in_slabs(
            ctx,
            label,
            width,
            height,
            cfa,
            data,
            qstep,
            MAX_SLAB_SAMPLES,
        )
    }

    /// [`upload`](Self::upload) with slabs of at most `max_slab_samples` samples (tests cut a small
    /// frame in two this way).
    ///
    /// # Errors
    ///
    /// As [`upload`](Self::upload).
    #[allow(clippy::too_many_arguments)]
    pub fn upload_in_slabs(
        ctx: &GpuContext,
        label: &str,
        width: usize,
        height: usize,
        cfa: Cfa,
        data: &[u16],
        qstep: Option<&[f32]>,
        max_slab_samples: usize,
    ) -> Result<DeviceRaw, GpuError> {
        if width == 0 || height == 0 || data.len() != width * height {
            return Err(GpuError::InvalidInput(format!(
                "{label}: {} samples for a {width} x {height} readout",
                data.len()
            )));
        }
        // The hook of the `upload@N` fault: a photo's words that find no room on the device.
        ctx.inject_upload_fault(2 * data.len() as u64)?;
        if let Some(q) = qstep
            && q.len() != data.len()
        {
            return Err(GpuError::InvalidInput(format!(
                "{label}: {} steps for {} samples",
                q.len(),
                data.len()
            )));
        }
        let rows_per_slab = max_slab_samples / width;
        if rows_per_slab == 0 {
            return Err(GpuError::InvalidInput(format!(
                "{label}: a row of {width} samples does not fit a slab of {max_slab_samples}"
            )));
        }
        if height > 2 * rows_per_slab {
            return Err(GpuError::InvalidInput(format!(
                "{label}: {width} x {height} samples need more than two slabs of {max_slab_samples}"
            )));
        }
        // The step representation: a table of bytes when the photo has at most 256 distinct steps.
        let (quant, table, index_of) = match qstep {
            None => (QuantMode::None, Vec::new(), None),
            Some(q) => {
                let mut seen = vec![0u64; 1024];
                for v in q {
                    let bits = half::f16::from_f32(*v).to_bits();
                    if half::f16::from_bits(bits).to_f32().to_bits() != v.to_bits() {
                        return Err(GpuError::InvalidInput(format!(
                            "{label}: the step {v} is not an f16 value"
                        )));
                    }
                    seen[usize::from(bits >> 6)] |= 1 << (bits & 63);
                }
                let count: u32 = seen.iter().map(|w| w.count_ones()).sum();
                if count <= 256 {
                    let mut table = Vec::with_capacity(count as usize);
                    let mut index = vec![0u8; 65536];
                    for bits in 0..=u16::MAX {
                        if seen[usize::from(bits >> 6)] >> (bits & 63) & 1 == 1 {
                            index[usize::from(bits)] = table.len() as u8;
                            table.push(half::f16::from_bits(bits).to_f32());
                        }
                    }
                    (QuantMode::Table, table, Some(index))
                } else {
                    (QuantMode::F16, Vec::new(), None)
                }
            }
        };
        let mut slabs = Vec::new();
        let mut row = 0usize;
        while row < height {
            let rows = (height - row).min(rows_per_slab);
            let (a, b) = (row * width, (row + rows) * width);
            let bytes = (b - a) as u64 * 2;
            let words = ctx
                .pool()
                .acquire(label, bytes.div_ceil(4) * 4, IMAGE_USAGE)?;
            upload_bytes(ctx, &words, 0, bytemuck::cast_slice(&data[a..b]))?;
            let steps = match (qstep, quant) {
                (Some(q), QuantMode::F16) => {
                    let bits: Vec<u16> = q[a..b]
                        .iter()
                        .map(|v| half::f16::from_f32(*v).to_bits())
                        .collect();
                    let buf =
                        ctx.pool()
                            .acquire("raw steps", bytes.div_ceil(4) * 4, IMAGE_USAGE)?;
                    upload_bytes(ctx, &buf, 0, bytemuck::cast_slice(&bits))?;
                    Some(buf)
                }
                (Some(q), QuantMode::Table) => {
                    let index = index_of.as_ref().ok_or_else(|| {
                        GpuError::InvalidInput("the step table has no index".into())
                    })?;
                    let bytes_idx: Vec<u8> = q[a..b]
                        .iter()
                        .map(|v| index[usize::from(half::f16::from_f32(*v).to_bits())])
                        .collect();
                    let buf = ctx.pool().acquire(
                        "raw steps",
                        ((b - a) as u64).div_ceil(4) * 4,
                        IMAGE_USAGE,
                    )?;
                    upload_bytes(ctx, &buf, 0, &bytes_idx)?;
                    Some(buf)
                }
                _ => None,
            };
            slabs.push(RawSlab {
                words,
                steps,
                first_row: row,
                rows,
            });
            row += rows;
        }
        Ok(DeviceRaw {
            slabs,
            width,
            height,
            cfa,
            quant,
            table,
        })
    }

    /// Width in photosites.
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Height in photosites.
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    /// The pattern at photosite (0, 0).
    #[must_use]
    pub fn cfa(&self) -> Cfa {
        self.cfa
    }

    /// How many slabs the words are in (1 or 2).
    #[must_use]
    pub fn slab_count(&self) -> usize {
        self.slabs.len()
    }

    /// How the quantisation step is stored.
    #[must_use]
    pub fn quant_mode(&self) -> QuantMode {
        self.quant
    }

    /// Bytes the photo holds on the device (words and steps).
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.slabs
            .iter()
            .map(|s| s.words.size() + s.steps.as_ref().map_or(0, PooledBuffer::size))
            .sum()
    }

    /// Reads the words back, for the tests and the host fallback.
    ///
    /// # Errors
    ///
    /// The errors of [`read_bytes`].
    pub fn read(&self, ctx: &GpuContext) -> Result<Vec<u16>, GpuError> {
        let mut out = Vec::with_capacity(self.width * self.height);
        for s in &self.slabs {
            let n = s.rows * self.width;
            let bytes = read_bytes(ctx, &s.words, 0, n as u64 * 2)?;
            out.extend(
                bytes
                    .chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]])),
            );
        }
        Ok(out)
    }

    /// The row where the second slab starts (the height for a photo in one slab).
    fn split_row(&self) -> usize {
        self.slabs.get(1).map_or(self.height, |s| s.first_row)
    }

    /// The four bindings of the words and steps (`front_raw.wgsl` bindings 1 to 4).
    pub(crate) fn bindings(&self) -> [wgpu::BindingResource<'_>; 4] {
        let first = &self.slabs[0];
        let second = self.slabs.get(1).unwrap_or(first);
        [
            first.words.binding(),
            second.words.binding(),
            first
                .steps
                .as_ref()
                .map_or_else(|| first.words.binding(), PooledBuffer::binding),
            second
                .steps
                .as_ref()
                .or(first.steps.as_ref())
                .map_or_else(|| first.words.binding(), PooledBuffer::binding),
        ]
    }
}

/// The vignetting profile as the shader reads it: the attenuation `G_i` at each knot, in `f64` (the
/// shader takes each as a pair of `f32`, so that it forms the gain in double-floats as the CPU forms it
/// in `f64`).
#[derive(Clone, Debug, PartialEq)]
pub struct VignetteTable {
    att: Vec<f64>,
}

impl VignetteTable {
    /// The knots of `profile`.
    ///
    /// # Errors
    ///
    /// [`GpuError::InvalidInput`] for a profile with more than [`MAX_VIGNETTE_KNOTS`] knots.
    pub fn from_profile(profile: &VignetteProfile) -> Result<VignetteTable, GpuError> {
        let ys = profile.knot_attenuations();
        if ys.len() < 2 || ys.len() > MAX_VIGNETTE_KNOTS {
            return Err(GpuError::InvalidInput(format!(
                "the vignetting table has {} knots (the kernel takes 2 to {MAX_VIGNETTE_KNOTS})",
                ys.len()
            )));
        }
        Ok(VignetteTable { att: ys.to_vec() })
    }

    /// The number of knots.
    #[must_use]
    pub fn len(&self) -> usize {
        self.att.len()
    }

    /// Always `false`: a table has at least two knots.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        false
    }
}

/// What R1, R3 and R4 need besides the words: the levels at the frame's parity, the gains and the
/// profile.
#[derive(Clone, Debug)]
pub struct PrepParams {
    /// R1 levels indexed by the frame's own parity (`Chain::levels`).
    pub levels: Levels,
    /// R4 gains `g0` (red, green, blue).
    pub gains: [f32; 3],
    /// Whether R4 is applied in this pass (the sigma plane of the fit view is made from R1 and R4
    /// too; a test can leave it out).
    pub apply_gains: bool,
    /// R3, `None` for a photo without a vignetting profile (or with `decode.lens_corrections` off).
    pub vignette: Option<VignetteTable>,
}

/// The rows of the shader's `RawFrame` block.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct FrameU {
    dims: [u32; 4],
    black: [f32; 4],
    inv: [f32; 4],
    white: [f32; 4],
    clip: [f32; 4],
    qtab: [[f32; 4]; 64],
}

/// The shader's `VigU` block.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct VigU {
    info: [u32; 4],
    k: [f32; 4],
    att: [[f32; 4]; 16],
}

/// `v` as a pair of `f32` whose sum is `v` to 48 bits.
fn split_f64(v: f64) -> (f32, f32) {
    let hi = v as f32;
    (hi, (v - f64::from(hi)) as f32)
}

impl FrameU {
    /// The block for `raw` and `levels`. The inverse range is formed in `f32` as the CPU forms it.
    pub(crate) fn new(raw: &DeviceRaw, levels: &Levels) -> FrameU {
        let mut qtab = [[0.0f32; 4]; 64];
        for (i, v) in raw.table.iter().enumerate() {
            qtab[i / 4][i % 4] = *v;
        }
        FrameU {
            dims: [
                raw.width as u32,
                raw.height as u32,
                raw.split_row() as u32,
                raw.quant.code(),
            ],
            black: levels.black,
            inv: std::array::from_fn(|i| 1.0 / (levels.white[i] - levels.black[i])),
            white: levels.white,
            clip: [QSTEP_MARGIN, CLIP_FRACTION, 0.0, 0.0],
            qtab,
        }
    }
}

impl VigU {
    /// The block for a frame of `width x height` photosites; with no table the profile is off.
    pub(crate) fn new(table: Option<&VignetteTable>, width: usize, height: usize) -> VigU {
        let mut att = [[0.0f32; 4]; 16];
        let Some(t) = table else {
            return VigU {
                info: [0; 4],
                k: [0.0; 4],
                att,
            };
        };
        for (i, v) in t.att.iter().enumerate() {
            let (hi, lo) = split_f64(*v);
            att[i / 2][2 * (i % 2)] = hi;
            att[i / 2][2 * (i % 2) + 1] = lo;
        }
        // r = sqrt(X^2 + Y^2) / sqrt(w^2 + h^2) for the doubled offsets X, Y; u = r (n - 1) - 0.5.
        let diag = ((width as f64).powi(2) + (height as f64).powi(2)).sqrt();
        let (scale_hi, scale_lo) = split_f64((t.att.len() - 1) as f64 / diag);
        VigU {
            info: [t.att.len() as u32, 0, 0, 0],
            k: [scale_hi, scale_lo, MIN_ATTENUATION as f32, 0.0],
            att,
        }
    }
}

/// The uniform block of `prep.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PrepU {
    /// The runtime zero of the barrier (always +0.0 bits).
    zero: [u32; 4],
    fr: FrameU,
    vig: VigU,
    /// Rectangle x, y, width, height.
    rect: [u32; 4],
    /// Mosaic row stride, mask words per row.
    dst: [u32; 4],
    /// Vignetting on, gains on.
    flags: [u32; 4],
    /// R4 gain per CFA position of the frame.
    gain: [f32; 4],
    /// The floor of R1.
    k: [f32; 4],
}

/// The clip bits of a rectangle, one bit per photosite, row after row (see `prep.wgsl`).
#[derive(Clone, Debug)]
pub struct ClipBits {
    /// The words: `rows * words_per_row` of them, bit `x % 32` of word `x / 32` of a row.
    pub buf: Arc<PooledBuffer>,
    /// Words per row.
    pub words_per_row: usize,
    /// Width of the rectangle in photosites.
    pub width: usize,
    /// Height of the rectangle in photosites.
    pub height: usize,
}

impl ClipBits {
    /// Bytes of the bits of a `width x height` rectangle.
    #[must_use]
    pub fn bytes_for(width: usize, height: usize) -> u64 {
        (width.div_ceil(32) * height) as u64 * 4
    }
}

/// What [`encode_prep`] records.
#[derive(Debug)]
pub struct PrepOut {
    /// The balanced mosaic of the rectangle (one plane).
    pub mosaic: GpuImage,
    /// The clip bits of the rectangle, when they were asked for.
    pub mask: Option<ClipBits>,
}

/// The CFA position `2 * (y & 1) + (x & 1)` gains of a frame whose first photosite has the pattern
/// `cfa`.
fn gains_by_position(cfa: Cfa, gains: [f32; 3]) -> [f32; 4] {
    std::array::from_fn(|p| gains[cfa.color_at(p & 1, p >> 1).plane()])
}

/// The largest storage binding (bytes) a prep pass needs for a `rect` of photosites: the mosaic (the
/// raw slabs are bounded by the design binding themselves).
#[must_use]
pub fn max_binding_bytes(rect: Rect) -> u64 {
    GpuImage::buffer_bytes(rect.w, rect.h, 1)
}

/// Records R1, R3 and R4 for `rect` (frame coordinates) of the resident photo, and with `with_mask`
/// the clip bits of R1 for the same rectangle.
///
/// # Errors
///
/// [`GpuError::InvalidInput`] for an empty rectangle outside the frame, a frame whose diagonal does
/// not fit 32 bits (`w^2 + h^2 >= 2^32`), or levels without a range; the pool's errors; shader errors.
pub fn encode_prep(
    passes: &mut PassList<'_>,
    raw: &DeviceRaw,
    rect: Rect,
    params: &PrepParams,
    with_mask: bool,
) -> Result<PrepOut, GpuError> {
    let ctx = passes.context();
    if rect.is_empty() || rect.x1() > raw.width || rect.y1() > raw.height {
        return Err(GpuError::InvalidInput(format!(
            "prep: the rectangle {rect:?} is empty or outside the {} x {} frame",
            raw.width, raw.height
        )));
    }
    params
        .levels
        .validate()
        .map_err(|e| GpuError::InvalidInput(format!("prep: {e}")))?;
    if (raw.width as u64).pow(2) + (raw.height as u64).pow(2) >= 1 << 32 {
        return Err(GpuError::InvalidInput(format!(
            "prep: a {} x {} frame is too large for the 32-bit radius of the vignetting gain",
            raw.width, raw.height
        )));
    }
    let place = Placement::whole(raw.width, raw.height).offset(rect.x, rect.y);
    let mosaic = GpuImage::alloc(ctx, "prep mosaic", rect.w, rect.h, 1, place)?;
    let words_per_row = rect.w.div_ceil(32);
    let u = PrepU {
        zero: [0; 4],
        fr: FrameU::new(raw, &params.levels),
        vig: VigU::new(params.vignette.as_ref(), raw.width, raw.height),
        rect: [rect.x as u32, rect.y as u32, rect.w as u32, rect.h as u32],
        dst: [mosaic.stride as u32, words_per_row as u32, 0, 0],
        flags: [
            u32::from(params.vignette.is_some()),
            u32::from(params.apply_gains),
            0,
            0,
        ],
        gain: gains_by_position(raw.cfa, params.gains),
        k: [NEGATIVE_FLOOR, 0.0, 0.0, 0.0],
    };
    let bindings = raw.bindings();
    let ub = passes.uniform("prep uniform", &u)?;
    passes.dispatch(
        KernelId::Prep,
        "prep r1 r3 r4",
        &[
            (0, ub.binding()),
            (1, bindings[0].clone()),
            (2, bindings[1].clone()),
            (5, mosaic.buf.binding()),
        ],
        (
            (rect.w as u32).div_ceil(GROUP),
            (rect.h as u32).div_ceil(GROUP),
            1,
        ),
    )?;
    let mask = if with_mask {
        let buf = passes.keep(ctx.pool().acquire(
            "clip bits",
            ClipBits::bytes_for(rect.w, rect.h),
            IMAGE_USAGE,
        )?);
        passes.dispatch(
            KernelId::PrepMask,
            "prep clip bits",
            &[
                (0, ub.binding()),
                (1, bindings[0].clone()),
                (2, bindings[1].clone()),
                (3, bindings[2].clone()),
                (4, bindings[3].clone()),
                (6, buf.binding()),
            ],
            (
                (words_per_row as u32).div_ceil(GROUP),
                (rect.h as u32).div_ceil(GROUP),
                1,
            ),
        )?;
        Some(ClipBits {
            buf,
            words_per_row,
            width: rect.w,
            height: rect.h,
        })
    } else {
        None
    };
    Ok(PrepOut { mosaic, mask })
}

/// R1, R3 and R4 of `rect` on the host: upload nothing (the words are resident), run
/// [`encode_prep`], read the mosaic back, and with `with_mask` the clip flags as one byte per
/// photosite (0 or 1, the CPU's `ClipMask` layout). For the parity tests.
///
/// # Errors
///
/// As [`encode_prep`], plus the readback's.
pub fn prep(
    ctx: &GpuContext,
    raw: &DeviceRaw,
    rect: Rect,
    params: &PrepParams,
    with_mask: bool,
) -> Result<(Vec<f32>, Option<Vec<u8>>), GpuError> {
    let mut passes = PassList::new(ctx, "prep");
    let out = encode_prep(&mut passes, raw, rect, params, with_mask)?;
    let read = out.mosaic.encode_read(&mut passes)?;
    let mask_read = match &out.mask {
        Some(m) => Some(passes.read(m.buf.as_ref(), 0, ClipBits::bytes_for(m.width, m.height))?),
        None => None,
    };
    let done = passes.submit()?.finish()?;
    let mosaic = out.mosaic.take(&done, read)?;
    let mask = match (&out.mask, mask_read) {
        (Some(m), Some(id)) => Some(unpack_mask(done.read(id), m.width, m.height)),
        _ => None,
    };
    Ok((mosaic, mask))
}

/// The clip bits as one byte per photosite.
fn unpack_mask(bytes: &[u8], width: usize, height: usize) -> Vec<u8> {
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let per_row = width.div_ceil(32);
    let mut out = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            out.push(((words[y * per_row + x / 32] >> (x % 32)) & 1) as u8);
        }
    }
    out
}

/// The WGSL of a prep-module kernel: the shared blocks, the shader file and the kernel's `main`.
pub(super) fn body(id: KernelId) -> String {
    let mut s = String::new();
    s.push_str(include_str!("../shaders/front_opaque.wgsl"));
    s.push_str(include_str!("../shaders/front_exact.wgsl"));
    s.push_str(include_str!("../shaders/front_raw.wgsl"));
    s.push_str(include_str!("../shaders/front_vig.wgsl"));
    s.push_str(include_str!("../shaders/prep.wgsl"));
    let entry = match id {
        KernelId::PrepMask => "prep_mask",
        _ => "prep_main",
    };
    s.push_str(&format!(
        "\n@compute @workgroup_size({GROUP}, {GROUP}, 1)\n\
         fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{ {entry}(gid); }}\n"
    ));
    s
}

#[cfg(test)]
pub(crate) mod tests {
    use std::mem::{offset_of, size_of};

    use naga::valid::{Capabilities, ValidationFlags, Validator};

    use super::*;
    use crate::device::StorageMode;
    use crate::kernels::shader_source;

    /// The members of the struct bound as `name` in `id`'s module, with their offsets and its size.
    pub(crate) fn uniform_layout(id: KernelId, name: &str) -> (Vec<(String, u32)>, u32) {
        let source = shader_source(StorageMode::F32, id);
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{id:?}: {}", e.emit_to_string(&source)));
        Validator::new(ValidationFlags::all(), Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{id:?}: {e:?}"));
        let var = module
            .global_variables
            .iter()
            .map(|(_, v)| v)
            .find(|v| v.name.as_deref() == Some(name))
            .unwrap_or_else(|| panic!("{id:?}: no uniform `{name}`"));
        let naga::TypeInner::Struct { members, span } = &module.types[var.ty].inner else {
            panic!("`{name}` is not a struct");
        };
        (
            members
                .iter()
                .map(|m| (m.name.clone().unwrap_or_default(), m.offset))
                .collect(),
            *span,
        )
    }

    #[test]
    fn the_uniform_block_has_the_layout_of_its_rust_struct() {
        for id in [KernelId::Prep, KernelId::PrepMask] {
            let (fields, span) = uniform_layout(id, "pu");
            let want = [
                ("zero", offset_of!(PrepU, zero)),
                ("fr", offset_of!(PrepU, fr)),
                ("vig", offset_of!(PrepU, vig)),
                ("rect", offset_of!(PrepU, rect)),
                ("dst", offset_of!(PrepU, dst)),
                ("flags", offset_of!(PrepU, flags)),
                ("gain", offset_of!(PrepU, gain)),
                ("k", offset_of!(PrepU, k)),
            ];
            assert_eq!(fields.len(), want.len(), "{id:?}");
            for ((name, offset), (want_name, want_offset)) in fields.iter().zip(want) {
                assert_eq!(name, want_name);
                assert_eq!(*offset as usize, want_offset, "{id:?}: offset of {name}");
            }
            assert_eq!(span as usize, size_of::<PrepU>(), "{id:?}");
        }
        // The nested blocks.
        let source = shader_source(StorageMode::F32, KernelId::Prep);
        let module = naga::front::wgsl::parse_str(&source).expect("parse");
        let find = |name: &str| {
            module
                .types
                .iter()
                .find_map(|(_, t)| match (&t.name, &t.inner) {
                    (Some(n), naga::TypeInner::Struct { members, span }) if n == name => {
                        Some((members.clone(), *span))
                    }
                    _ => None,
                })
                .unwrap_or_else(|| panic!("no struct {name}"))
        };
        let (m, span) = find("RawFrame");
        let want = [
            offset_of!(FrameU, dims),
            offset_of!(FrameU, black),
            offset_of!(FrameU, inv),
            offset_of!(FrameU, white),
            offset_of!(FrameU, clip),
            offset_of!(FrameU, qtab),
        ];
        assert_eq!(m.len(), want.len());
        for (a, b) in m.iter().zip(want) {
            assert_eq!(a.offset as usize, b);
        }
        assert_eq!(span as usize, size_of::<FrameU>());
        let (m, span) = find("VigU");
        let want = [
            offset_of!(VigU, info),
            offset_of!(VigU, k),
            offset_of!(VigU, att),
        ];
        assert_eq!(m.len(), want.len());
        for (a, b) in m.iter().zip(want) {
            assert_eq!(a.offset as usize, b);
        }
        assert_eq!(span as usize, size_of::<VigU>());
    }

    #[test]
    fn the_mask_unpacks_by_row_and_bit() {
        // 33 columns: two words a row, the second word holds one bit.
        let mut words = [0u32; 2 * 2];
        words[0] = 0b101;
        words[1] = 1;
        words[2] = 1 << 31;
        words[3] = 0;
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let m = unpack_mask(&bytes, 33, 2);
        assert_eq!(m.len(), 66);
        assert_eq!((m[0], m[1], m[2], m[3]), (1, 0, 1, 0));
        assert_eq!(m[32], 1);
        assert_eq!(m[33 + 31], 1);
        assert_eq!(m[33 + 32], 0);
    }
}

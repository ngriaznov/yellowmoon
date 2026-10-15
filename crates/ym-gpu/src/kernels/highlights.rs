//! R5 highlight reconstruction on the device (G-07, `docs/05-raw-processing.md` section R5).
//!
//! The CPU reference is [`crate::cpu::highlights`]: superpixels of 2 x 2 photosites, the opposed
//! estimate from the unclipped other channels over the 3 x 3 neighbourhood, the global chroma offsets
//! and the levelling of fully clipped quads. The device runs it as
//!
//! * [`encode_highlights`]: the superpixel planes of the input rectangle, then the apply pass over a
//!   core, with the offsets in a small buffer ([`Offsets`]). The full chain gets the CPU's offsets from
//!   `Chain::prepare` ([`Offsets::from_host`]); the clip bits are those of [`super::prep`];
//! * [`OffsetReducer`]: the reduced path's own offsets, a reduction over the reference superpixels in
//!   a fixed order (D2 of `docs/gpu-plan.md`): per workgroup a binary tree in thread-index order, the
//!   partials of every workgroup of every slab, then one workgroup that folds them in index order with
//!   compensated summation. The differences `value - estimate` are formed as exact two-float
//!   sums, so the result does not depend on the order of the terms beyond the final rounding.
//!
//! Applying needs one superpixel of context beyond the core (2 px; the stage list asks for 4); the
//! superpixel grid clamps at the edge of the input rectangle as the CPU's does at the edge of the
//! buffer it is given, so a tile with its halo equals the whole frame bit for bit.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use super::prep::ClipBits;
use super::shader::KernelId;
use super::util::GROUP;
use crate::cpu::tiling::Rect;
use crate::cpu::{Cfa, ClipMask, Mosaic};
use crate::device::{GpuContext, GpuError};
use crate::image::{GpuImage, IMAGE_USAGE};
use crate::pool::PooledBuffer;
use crate::stream::{Completed, PassList, ReadId, upload_bytes};

/// Words of an offsets buffer: three offsets, a pad, three counts.
const OFFSET_WORDS: usize = 8;

/// Words of one partial: three sums, three compensations, three counts.
const PARTIAL_WORDS: usize = 9;

/// The uniform block of `highlights.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HlU {
    /// The runtime zero of the barrier (always +0.0 bits).
    zero: [u32; 4],
    /// Input width, height, row stride.
    dims: [u32; 4],
    /// Core x, y, width, height in the input.
    core: [u32; 4],
    /// Output row stride, superpixel columns and rows, mask words per row.
    dst: [u32; 4],
    /// The plane index at (0, 0), (1, 0), (0, 1), (1, 1) of the input.
    cfa: [u32; 4],
    /// Reduction: first and one-past-last superpixel row, partial count, first partial.
    rows: [u32; 4],
}

/// The global chroma offsets on the device (three `f32`, the counts after them).
#[derive(Clone, Debug)]
pub struct Offsets {
    buf: Arc<PooledBuffer>,
}

impl Offsets {
    /// Offsets made on the host (the CPU's `ChromaOffsets` rounded to `f32`, as the CPU rounds them
    /// where it adds them).
    ///
    /// # Errors
    ///
    /// The pool's errors.
    pub fn from_host(passes: &mut PassList<'_>, offsets: [f32; 3]) -> Result<Offsets, GpuError> {
        let ctx = passes.context();
        let words: [u32; OFFSET_WORDS] = [
            offsets[0].to_bits(),
            offsets[1].to_bits(),
            offsets[2].to_bits(),
            0,
            0,
            0,
            0,
            0,
        ];
        let buf =
            ctx.pool()
                .acquire("highlight offsets", (OFFSET_WORDS * 4) as u64, IMAGE_USAGE)?;
        upload_bytes(ctx, &buf, 0, bytemuck::cast_slice(&words))?;
        Ok(Offsets {
            buf: passes.keep(buf),
        })
    }

    /// Asks `passes` to copy the offsets back after the passes recorded so far.
    ///
    /// # Errors
    ///
    /// The errors of [`PassList::read`].
    pub fn encode_read(&self, passes: &mut PassList<'_>) -> Result<ReadId, GpuError> {
        passes.read(self.buf.as_ref(), 0, (OFFSET_WORDS * 4) as u64)
    }

    /// The offsets and the counts of the reference superpixels out of a finished list.
    #[must_use]
    pub fn take(done: &Completed, id: ReadId) -> ([f32; 3], [u32; 3]) {
        let w: Vec<u32> = done
            .read(id)
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        (
            [
                f32::from_bits(w[0]),
                f32::from_bits(w[1]),
                f32::from_bits(w[2]),
            ],
            [w[4], w[5], w[6]],
        )
    }
}

impl ClipBits {
    /// The clip bits of a host mask, uploaded and kept by `passes`.
    ///
    /// # Errors
    ///
    /// The pool's errors.
    pub fn upload(passes: &mut PassList<'_>, mask: &ClipMask) -> Result<ClipBits, GpuError> {
        let ctx = passes.context();
        let per_row = mask.width.div_ceil(32);
        let mut words = vec![0u32; per_row * mask.height];
        for y in 0..mask.height {
            for x in 0..mask.width {
                if mask.data[y * mask.width + x] != 0 {
                    words[y * per_row + x / 32] |= 1 << (x % 32);
                }
            }
        }
        let buf = ctx.pool().acquire(
            "clip bits",
            ClipBits::bytes_for(mask.width, mask.height),
            IMAGE_USAGE,
        )?;
        upload_bytes(ctx, &buf, 0, bytemuck::cast_slice(&words))?;
        Ok(ClipBits {
            buf: passes.keep(buf),
            words_per_row: per_row,
            width: mask.width,
            height: mask.height,
        })
    }
}

/// The superpixel planes of a rectangle (see `highlights.wgsl`).
#[derive(Debug)]
pub struct Superpixels {
    planes: Arc<PooledBuffer>,
    flags: Arc<PooledBuffer>,
    /// Superpixel columns.
    pub sw: usize,
    /// Superpixel rows.
    pub sh: usize,
}

/// The largest storage binding (bytes) the highlight passes need for a rectangle of `rect` photosites:
/// the six superpixel planes (one and a half times the mosaic).
#[must_use]
pub fn max_binding_bytes(rect: Rect) -> u64 {
    6 * (rect.w / 2) as u64 * (rect.h / 2) as u64 * 4
}

fn check_inputs(mosaic: &GpuImage, mask: &ClipBits, cfa: Cfa, what: &str) -> Result<HlU, GpuError> {
    if mosaic.planes != 1 {
        return Err(GpuError::InvalidInput(format!(
            "{what}: the input has {} planes, not one",
            mosaic.planes
        )));
    }
    if (mask.width, mask.height) != (mosaic.width, mosaic.height) {
        return Err(GpuError::InvalidInput(format!(
            "{what}: the clip bits are for {} x {}, the mosaic is {} x {}",
            mask.width, mask.height, mosaic.width, mosaic.height
        )));
    }
    if !mosaic.place.x0.is_multiple_of(2) || !mosaic.place.y0.is_multiple_of(2) {
        return Err(GpuError::InvalidInput(format!(
            "{what}: needs an even-aligned rectangle, got origin ({}, {})",
            mosaic.place.x0, mosaic.place.y0
        )));
    }
    let plane = |dx: usize, dy: usize| cfa.color_at(dx, dy).plane() as u32;
    Ok(HlU {
        zero: [0; 4],
        dims: [
            mosaic.width as u32,
            mosaic.height as u32,
            mosaic.stride as u32,
            0,
        ],
        core: [0, 0, mosaic.width as u32, mosaic.height as u32],
        dst: [
            0,
            (mosaic.width / 2) as u32,
            (mosaic.height / 2) as u32,
            mask.words_per_row as u32,
        ],
        cfa: [plane(0, 0), plane(1, 0), plane(0, 1), plane(1, 1)],
        rows: [0; 4],
    })
}

/// Records the superpixel planes of `mosaic` (the rectangle of the clip bits `mask`; `cfa` the
/// pattern at its first photosite).
///
/// # Errors
///
/// [`GpuError::InvalidInput`] for an input that is not one plane, bits of another size, a rectangle
/// that starts at an odd photosite, or one below 2 x 2; the pool's errors; shader errors.
pub fn encode_superpixels(
    passes: &mut PassList<'_>,
    mosaic: &GpuImage,
    mask: &ClipBits,
    cfa: Cfa,
) -> Result<Superpixels, GpuError> {
    let u = check_inputs(mosaic, mask, cfa, "highlights")?;
    let (sw, sh) = (mosaic.width / 2, mosaic.height / 2);
    if sw == 0 || sh == 0 {
        return Err(GpuError::InvalidInput(format!(
            "highlights: a {} x {} rectangle holds no superpixel",
            mosaic.width, mosaic.height
        )));
    }
    let planes = passes.scratch("superpixel planes", 6 * (sw * sh) as u64 * 4)?;
    let flags = passes.scratch("superpixel flags", (sw * sh) as u64 * 4)?;
    let ub = passes.uniform("highlights uniform", &u)?;
    passes.dispatch(
        KernelId::HighlightsSuper,
        "highlights superpixels",
        &[
            (0, ub.binding()),
            (1, mosaic.buf.binding()),
            (2, mask.buf.binding()),
            (3, planes.binding()),
            (4, flags.binding()),
        ],
        ((sw as u32).div_ceil(GROUP), (sh as u32).div_ceil(GROUP), 1),
    )?;
    Ok(Superpixels {
        planes,
        flags,
        sw,
        sh,
    })
}

/// Records R5 for the `core` rectangle of `mosaic` (device in, device out): the superpixels of the
/// whole input, then the apply pass. Returns the core as a new one-plane image whose placement is the
/// input's moved to the core. A photosite beyond the last whole superpixel of an odd-sized input is
/// copied.
///
/// # Errors
///
/// As [`encode_superpixels`], plus [`GpuError::InvalidInput`] for a core that is empty or outside the
/// input.
pub fn encode_highlights(
    passes: &mut PassList<'_>,
    mosaic: &GpuImage,
    mask: &ClipBits,
    cfa: Cfa,
    core: Rect,
    offsets: &Offsets,
) -> Result<GpuImage, GpuError> {
    let ctx = passes.context();
    if core.is_empty() || core.x1() > mosaic.width || core.y1() > mosaic.height {
        return Err(GpuError::InvalidInput(format!(
            "highlights: the core {core:?} is empty or outside the {} x {} input",
            mosaic.width, mosaic.height
        )));
    }
    let sp = encode_superpixels(passes, mosaic, mask, cfa)?;
    let out = GpuImage::alloc(
        ctx,
        "highlights mosaic",
        core.w,
        core.h,
        1,
        mosaic.place.offset(core.x, core.y),
    )?;
    let mut u = check_inputs(mosaic, mask, cfa, "highlights")?;
    u.core = [core.x as u32, core.y as u32, core.w as u32, core.h as u32];
    u.dst[0] = out.stride as u32;
    let ub = passes.uniform("highlights uniform", &u)?;
    passes.dispatch(
        KernelId::Highlights,
        "highlights apply",
        &[
            (0, ub.binding()),
            (1, mosaic.buf.binding()),
            (2, mask.buf.binding()),
            (3, sp.planes.binding()),
            (4, sp.flags.binding()),
            (5, offsets.buf.binding()),
            (6, out.buf.binding()),
        ],
        (
            (core.w as u32).div_ceil(GROUP),
            (core.h as u32).div_ceil(GROUP),
            1,
        ),
    )?;
    Ok(out)
}

/// R5 on the host: uploads `mosaic` and `mask`, applies `offsets` on `core` and reads the core back.
/// For the parity tests.
///
/// # Errors
///
/// As [`encode_highlights`], plus the readback's.
pub fn highlights(
    ctx: &GpuContext,
    mosaic: &Mosaic,
    mask: &ClipMask,
    offsets: [f32; 3],
    core: Rect,
) -> Result<Vec<f32>, GpuError> {
    let input = GpuImage::upload(
        ctx,
        "highlights input",
        mosaic.width,
        mosaic.height,
        1,
        &mosaic.data,
        mosaic.place,
    )?;
    let mut passes = PassList::new(ctx, "highlights");
    let bits = ClipBits::upload(&mut passes, mask)?;
    let off = Offsets::from_host(&mut passes, offsets)?;
    let out = encode_highlights(&mut passes, &input, &bits, mosaic.cfa, core, &off)?;
    let read = out.encode_read(&mut passes)?;
    let done = passes.submit()?.finish()?;
    out.take(&done, read)
}

/// The reduction of the reduced path's chroma offsets over several slabs (see the module notes). The
/// caller keeps the reducer alive until the lists that use it have finished.
#[derive(Debug)]
pub struct OffsetReducer {
    partials: Arc<PooledBuffer>,
    capacity: usize,
    used: usize,
}

impl OffsetReducer {
    /// Partials a slab of `sw` superpixel columns and `core_rows` photosite rows of core makes.
    #[must_use]
    pub fn partials_for(sw: usize, core_rows: usize) -> usize {
        sw.div_ceil(16) * (core_rows / 2).div_ceil(16)
    }

    /// A reducer with room for `capacity` partials (the sum of [`partials_for`](Self::partials_for)
    /// over the slabs).
    ///
    /// # Errors
    ///
    /// The pool's errors.
    pub fn new(ctx: &GpuContext, capacity: usize) -> Result<OffsetReducer, GpuError> {
        let partials = ctx.pool().acquire(
            "offset partials",
            (capacity.max(1) * PARTIAL_WORDS * 4) as u64,
            IMAGE_USAGE,
        )?;
        Ok(OffsetReducer {
            partials: Arc::new(partials),
            capacity,
            used: 0,
        })
    }

    /// Partials recorded so far.
    #[must_use]
    pub fn used(&self) -> usize {
        self.used
    }

    /// Records the reference terms of the superpixels whose rows lie in the photosite rows
    /// `core_rows` of `mosaic` (even, in the mosaic's own coordinates); the rest of the mosaic is
    /// the context of the 3 x 3 neighbourhood (two photosites above and below).
    ///
    /// # Errors
    ///
    /// [`GpuError::InvalidInput`] for rows that are not even, empty or outside the mosaic, or a reducer
    /// that is full; as [`encode_superpixels`].
    pub fn encode_slab(
        &mut self,
        passes: &mut PassList<'_>,
        mosaic: &GpuImage,
        mask: &ClipBits,
        cfa: Cfa,
        core_rows: (usize, usize),
    ) -> Result<(), GpuError> {
        let (r0, r1) = core_rows;
        if !r0.is_multiple_of(2) || r1 <= r0 || r1 > mosaic.height || !(r1 - r0).is_multiple_of(2) {
            return Err(GpuError::InvalidInput(format!(
                "the reduction rows {r0}..{r1} must be even and inside the {} rows of the slab",
                mosaic.height
            )));
        }
        let sp = encode_superpixels(passes, mosaic, mask, cfa)?;
        let n = Self::partials_for(sp.sw, r1 - r0);
        if self.used + n > self.capacity {
            return Err(GpuError::InvalidInput(format!(
                "the reducer has room for {} partials, {} are used and the slab makes {n}",
                self.capacity, self.used
            )));
        }
        let mut u = check_inputs(mosaic, mask, cfa, "offset reduction")?;
        u.rows = [(r0 / 2) as u32, (r1 / 2) as u32, 0, self.used as u32];
        let ub = passes.uniform("reduction uniform", &u)?;
        passes.dispatch(
            KernelId::Reduce,
            "highlight offsets partials",
            &[
                (0, ub.binding()),
                (3, sp.planes.binding()),
                (4, sp.flags.binding()),
                (7, self.partials.binding()),
            ],
            (
                (sp.sw as u32).div_ceil(GROUP),
                (((r1 - r0) / 2) as u32).div_ceil(GROUP),
                1,
            ),
        )?;
        self.used += n;
        Ok(())
    }

    /// Records the fold of every partial recorded so far and returns the offsets.
    ///
    /// # Errors
    ///
    /// The pool's errors; shader errors.
    pub fn encode_fold(&self, passes: &mut PassList<'_>) -> Result<Offsets, GpuError> {
        let ctx = passes.context();
        let buf = passes.keep(ctx.pool().acquire(
            "highlight offsets",
            (OFFSET_WORDS * 4) as u64,
            IMAGE_USAGE,
        )?);
        let mut u = HlU::zeroed();
        u.rows = [0, 0, self.used as u32, 0];
        let ub = passes.uniform("fold uniform", &u)?;
        passes.dispatch(
            KernelId::ReduceFold,
            "highlight offsets fold",
            &[
                (0, ub.binding()),
                (5, buf.binding()),
                (7, self.partials.binding()),
            ],
            (1, 1, 1),
        )?;
        Ok(Offsets { buf })
    }
}

/// The offsets of a whole host mosaic through the device reduction, in slabs of `slab_rows` rows (a
/// multiple of 2) with their context; and the counts. For the parity tests.
///
/// # Errors
///
/// As [`OffsetReducer`], plus the readbacks'.
pub fn reduce_offsets(
    ctx: &GpuContext,
    mosaic: &Mosaic,
    mask: &ClipMask,
    slab_rows: usize,
) -> Result<([f32; 3], [u32; 3]), GpuError> {
    let h = mosaic.height;
    let slab_rows = slab_rows.max(2) & !1;
    let mut slabs = Vec::new();
    let mut r = 0;
    while r < h / 2 * 2 {
        slabs.push((r, (r + slab_rows).min(h / 2 * 2)));
        r += slab_rows;
    }
    let sw = mosaic.width / 2;
    let capacity: usize = slabs
        .iter()
        .map(|&(a, b)| OffsetReducer::partials_for(sw, b - a))
        .sum();
    let mut reducer = OffsetReducer::new(ctx, capacity)?;
    for &(a, b) in &slabs {
        // The slab with two photosite rows of context on each side, clamped to the mosaic.
        let top = a.saturating_sub(2);
        let bottom = (b + 2).min(h);
        let rect = Rect::new(0, top, mosaic.width, bottom - top);
        let sub = mosaic.extract(rect);
        let sub_mask = mask.extract(rect);
        let input = GpuImage::upload(
            ctx,
            "reduction slab",
            sub.width,
            sub.height,
            1,
            &sub.data,
            sub.place,
        )?;
        let mut passes = PassList::new(ctx, "offset reduction slab");
        let bits = ClipBits::upload(&mut passes, &sub_mask)?;
        reducer.encode_slab(&mut passes, &input, &bits, sub.cfa, (a - top, b - top))?;
        passes.submit()?.finish()?;
    }
    let mut passes = PassList::new(ctx, "offset fold");
    let off = reducer.encode_fold(&mut passes)?;
    let read = off.encode_read(&mut passes)?;
    let done = passes.submit()?.finish()?;
    Ok(Offsets::take(&done, read))
}

/// The WGSL of a highlights-module kernel.
pub(super) fn body(id: KernelId) -> String {
    let mut s = String::new();
    s.push_str(include_str!("../shaders/front_opaque.wgsl"));
    s.push_str(include_str!("../shaders/front_exact.wgsl"));
    s.push_str(include_str!("../shaders/highlights.wgsl"));
    let main = match id {
        KernelId::HighlightsSuper => format!(
            "@compute @workgroup_size({GROUP}, {GROUP}, 1)\n\
             fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{ hl_super(gid); }}\n"
        ),
        KernelId::Reduce => format!(
            "@compute @workgroup_size({GROUP}, {GROUP}, 1)\n\
             fn main(\n    @builtin(global_invocation_id) gid: vec3<u32>,\n    \
             @builtin(local_invocation_id) lid: vec3<u32>,\n    \
             @builtin(workgroup_id) wid: vec3<u32>,\n    \
             @builtin(num_workgroups) groups: vec3<u32>,\n) {{ hl_reduce_partial(gid, lid, wid, groups); }}\n"
        ),
        KernelId::ReduceFold => "@compute @workgroup_size(256, 1, 1)\n\
             fn main(@builtin(local_invocation_index) t: u32) { hl_fold(t); }\n"
            .to_string(),
        _ => format!(
            "@compute @workgroup_size({GROUP}, {GROUP}, 1)\n\
             fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{ hl_apply(gid); }}\n"
        ),
    };
    s.push('\n');
    s.push_str(&main);
    s
}

#[cfg(test)]
mod tests {
    use std::mem::{offset_of, size_of};

    use super::*;
    use crate::kernels::prep::tests::uniform_layout;

    #[test]
    fn the_uniform_block_has_the_layout_of_its_rust_struct() {
        for id in [
            KernelId::HighlightsSuper,
            KernelId::Highlights,
            KernelId::Reduce,
            KernelId::ReduceFold,
        ] {
            let (fields, span) = uniform_layout(id, "pu");
            let want = [
                ("zero", offset_of!(HlU, zero)),
                ("dims", offset_of!(HlU, dims)),
                ("core", offset_of!(HlU, core)),
                ("dst", offset_of!(HlU, dst)),
                ("cfa", offset_of!(HlU, cfa)),
                ("rows", offset_of!(HlU, rows)),
            ];
            assert_eq!(fields.len(), want.len(), "{id:?}");
            for ((name, offset), (want_name, want_offset)) in fields.iter().zip(want) {
                assert_eq!(name, want_name);
                assert_eq!(*offset as usize, want_offset, "{id:?}: offset of {name}");
            }
            assert_eq!(span as usize, size_of::<HlU>(), "{id:?}");
        }
    }

    #[test]
    fn the_partial_count_matches_the_dispatch() {
        assert_eq!(OffsetReducer::partials_for(33, 64), 3 * 2);
        assert_eq!(OffsetReducer::partials_for(16, 32), 1);
        assert_eq!(OffsetReducer::partials_for(16, 34), 2);
    }
}

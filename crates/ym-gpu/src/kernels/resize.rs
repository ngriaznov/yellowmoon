//! The Lanczos-3 resize of the export tail (`docs/gpu-plan.md` G-08, `docs/09-export.md` section
//! Pixel pipeline per format): the planner that both the CPU resize (`ym_export::resize`) and the
//! kernel read their taps from, and the kernel itself.
//!
//! # The planner
//!
//! [`Taps`] holds, for every output index of one axis, the first input index it reads and its taps:
//! the Lanczos window of half-width 3 scaled by the shrink factor (so a shrink also filters
//! aliasing), taps outside the image dropped and the rest normalised to sum to 1. It is planned in
//! `f64` and shared by the CPU resize and the kernel, so the two cannot disagree on a weight
//! (`K7` of the plan: constants are written once, from the CPU planner). The kernel gets the taps
//! rounded to `f32` ([`PreparedTaps`]).
//!
//! # The kernel
//!
//! One shader, `resize.wgsl`, one pass per axis. The horizontal pass ([`encode_resize_h`]) turns
//! rows of an image `in_w` wide into rows `out_w` wide; the vertical pass ([`encode_resize_v`])
//! turns rows into output rows from the run of rows each one needs, which the caller gathers into
//! one image ([`Taps::input_span`] says which). The sum is in difference form
//! (`r + sum w (v - r)`), so a flat field comes out exactly: the property the CPU gets from `f64`
//! accumulation. On other content the result is within about 1e-6 of the CPU's, relative to the
//! local range (`tests/gpu_resize.rs`).
//!
//! [`resize`] is the thin host wrapper (upload, both passes, readback) that the tests use.

use std::f64::consts::PI;
use std::ops::Range;

use bytemuck::{Pod, Zeroable};

use super::shader::KernelId;
use super::util::{even, storage_buffer};
use crate::cpu::tiling::Rect;
use crate::cpu::{Placement, RgbImage};
use crate::device::{GpuContext, GpuError};
use crate::image::GpuImage;
use crate::pool::PooledBuffer;
use crate::stream::PassList;

/// The Lanczos window half-width `a`.
pub const LANCZOS_A: f64 = 3.0;

fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        let px = x * PI;
        px.sin() / px
    }
}

/// The Lanczos-3 kernel.
#[must_use]
pub fn lanczos3(x: f64) -> f64 {
    if (-LANCZOS_A..LANCZOS_A).contains(&x) {
        sinc(x) * sinc(x / LANCZOS_A)
    } else {
        0.0
    }
}

/// Per output index of one axis: the first input index and its normalised taps (see the module
/// notes). The index of the first tap and the index after the last one never decrease with the
/// output index.
#[derive(Debug, Clone, PartialEq)]
pub struct Taps {
    in_len: usize,
    start: Vec<usize>,
    offsets: Vec<usize>,
    weights: Vec<f64>,
}

impl Taps {
    /// The taps that resize `in_len` samples to `out_len`.
    ///
    /// # Panics
    ///
    /// Never for `in_len >= 1` and `out_len >= 1`; the callers refuse empty images.
    #[must_use]
    pub fn new(in_len: usize, out_len: usize) -> Self {
        let scale = in_len as f64 / out_len as f64;
        let filter_scale = scale.max(1.0);
        let support = LANCZOS_A * filter_scale;
        let mut start = Vec::with_capacity(out_len);
        let mut offsets = Vec::with_capacity(out_len + 1);
        let mut weights = Vec::new();
        offsets.push(0);
        for i in 0..out_len {
            let center = (i as f64 + 0.5) * scale;
            let lo = ((center - support + 0.5).floor().max(0.0)) as usize;
            let hi = (((center + support + 0.5).floor()) as usize).min(in_len);
            let first = weights.len();
            let mut sum = 0.0;
            for x in lo..hi {
                let w = lanczos3((x as f64 - center + 0.5) / filter_scale);
                weights.push(w);
                sum += w;
            }
            if sum != 0.0 {
                for w in &mut weights[first..] {
                    *w /= sum;
                }
            }
            start.push(lo);
            offsets.push(weights.len());
        }
        Self {
            in_len,
            start,
            offsets,
            weights,
        }
    }

    /// Number of input samples.
    #[must_use]
    pub fn in_len(&self) -> usize {
        self.in_len
    }

    /// Number of output samples.
    #[must_use]
    pub fn out_len(&self) -> usize {
        self.start.len()
    }

    /// The first input index of output `i` and its taps.
    #[must_use]
    pub fn row(&self, i: usize) -> (usize, &[f64]) {
        (
            self.start[i],
            &self.weights[self.offsets[i]..self.offsets[i + 1]],
        )
    }

    /// The most taps any output has.
    #[must_use]
    pub fn max_taps(&self) -> usize {
        self.offsets
            .windows(2)
            .map(|w| w[1] - w[0])
            .max()
            .unwrap_or(0)
    }

    /// The input indices the outputs `outs` read together: from the first tap of the first output
    /// to one past the last tap of the last (both ends only grow with the output index). Empty for
    /// an empty range.
    #[must_use]
    pub fn input_span(&self, outs: Range<usize>) -> Range<usize> {
        if outs.is_empty() {
            return 0..0;
        }
        let last = outs.end - 1;
        self.start[outs.start]..self.start[last] + (self.offsets[last + 1] - self.offsets[last])
    }
}

/// The uniform block of `resize.wgsl` (`ResizeU`): 12 words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct ResizeU {
    dims: [u32; 4],
    src: [u32; 4],
    dst: [u32; 4],
}

/// The taps of one axis on the device: one `(first input index, count, weight offset, 0)` entry per
/// output index and the weights rounded to `f32`. Immutable; share it between passes and between
/// the slabs of one export.
#[derive(Debug)]
pub struct PreparedTaps {
    entries: PooledBuffer,
    weights: PooledBuffer,
    taps: Taps,
}

impl PreparedTaps {
    /// Uploads `taps`.
    ///
    /// # Errors
    ///
    /// The pool's errors for the two buffers ([`GpuError::OutOfBudget`], [`GpuError::BufferTooLarge`]).
    pub fn new(ctx: &GpuContext, taps: Taps) -> Result<Self, GpuError> {
        let entries: Vec<[u32; 4]> = (0..taps.out_len())
            .map(|i| {
                [
                    taps.start[i] as u32,
                    (taps.offsets[i + 1] - taps.offsets[i]) as u32,
                    taps.offsets[i] as u32,
                    0,
                ]
            })
            .collect();
        let weights: Vec<f32> = taps.weights.iter().map(|&w| w as f32).collect();
        Ok(Self {
            entries: storage_buffer(ctx, "resize taps", bytemuck::cast_slice(&entries))?,
            weights: storage_buffer(ctx, "resize weights", bytemuck::cast_slice(&weights))?,
            taps,
        })
    }

    /// The planned taps.
    #[must_use]
    pub fn taps(&self) -> &Taps {
        &self.taps
    }

    /// The device bytes the two buffers take.
    #[must_use]
    pub fn byte_len(&self) -> u64 {
        self.entries.size() + self.weights.size()
    }
}

/// The WGSL of the resize kernel.
pub(super) fn body() -> String {
    String::from(include_str!("../shaders/resize.wgsl"))
}

/// The size of the largest storage binding a resize pass makes: the larger of the source and the
/// destination image (three planes of f32) for `window` pixels of the larger side.
pub fn max_binding_bytes(window: Rect) -> u64 {
    12 * even(window.w) as u64 * window.h as u64
}

#[allow(clippy::too_many_arguments)] // the natural argument list of the two passes
fn dispatch(
    passes: &mut PassList<'_>,
    taps: &PreparedTaps,
    src: &GpuImage,
    dst: &GpuImage,
    horizontal: bool,
    columns: usize,
    rows: usize,
    src_z: usize,
    dst_row: usize,
    first_entry: usize,
) -> Result<(), GpuError> {
    let ctx = passes.context();
    let limit = ctx.info().limits.max_compute_workgroups_per_dimension as usize;
    let (gx, gy) = (columns.div_ceil(16), rows.div_ceil(16));
    if gx > limit || gy > limit {
        return Err(GpuError::InvalidInput(format!(
            "a {columns} x {rows} resize needs more workgroups than the device allows ({limit})"
        )));
    }
    let u = ResizeU {
        dims: [columns as u32, rows as u32, u32::from(!horizontal), 3],
        src: [src.stride as u32, src.plane_len() as u32, src_z as u32, 0],
        dst: [
            dst.stride as u32,
            dst.plane_len() as u32,
            dst_row as u32,
            first_entry as u32,
        ],
    };
    let ub = passes.uniform("resize uniform", &u)?;
    passes.dispatch(
        KernelId::Resize,
        if horizontal {
            "resize horizontal"
        } else {
            "resize vertical"
        },
        &[
            (0, ub.binding()),
            (1, src.buf.binding()),
            (2, taps.entries.binding()),
            (3, taps.weights.binding()),
            (4, dst.buf.binding()),
        ],
        (gx.max(1) as u32, gy.max(1) as u32, 1),
    )
}

/// Records the horizontal pass: rows `src_rows` of `src` (an image `taps.in_len()` wide, at least
/// three planes) become rows `dst_row..` of `dst` (an image `taps.out_len()` wide). The planes
/// beyond the third are not read or written.
///
/// # Errors
///
/// [`GpuError::InvalidInput`] for an image of the wrong width or with fewer than three planes, rows
/// outside the images, or a pass that needs more workgroups than the device allows.
pub fn encode_resize_h(
    passes: &mut PassList<'_>,
    taps: &PreparedTaps,
    src: &GpuImage,
    src_rows: Range<usize>,
    dst: &GpuImage,
    dst_row: usize,
) -> Result<(), GpuError> {
    let rows = src_rows.len();
    if src.width != taps.taps.in_len() || dst.width != taps.taps.out_len() {
        return Err(GpuError::InvalidInput(format!(
            "a horizontal resize from {} to {} columns was given images {} and {} wide",
            taps.taps.in_len(),
            taps.taps.out_len(),
            src.width,
            dst.width
        )));
    }
    if src.planes < 3 || dst.planes < 3 {
        return Err(GpuError::InvalidInput(
            "a resize reads and writes three planes".into(),
        ));
    }
    if rows == 0 || src_rows.end > src.height || dst_row + rows > dst.height {
        return Err(GpuError::InvalidInput(format!(
            "rows {src_rows:?} of a {} row image into rows {dst_row}.. of a {} row image",
            src.height, dst.height
        )));
    }
    dispatch(
        passes,
        taps,
        src,
        dst,
        true,
        dst.width,
        rows,
        src_rows.start,
        dst_row,
        0,
    )
}

/// Records the vertical pass: the output rows `out_rows` of the resize become rows `dst_row..` of
/// `dst`. `src` holds the input rows from `src_first` on in its rows 0.. and has to hold every row
/// the outputs read ([`Taps::input_span`]); `src` and `dst` have the same width.
///
/// # Errors
///
/// [`GpuError::InvalidInput`] for images of different widths or with fewer than three planes, output
/// rows outside the resize, input rows the source does not hold, or rows outside `dst`.
pub fn encode_resize_v(
    passes: &mut PassList<'_>,
    taps: &PreparedTaps,
    src: &GpuImage,
    src_first: usize,
    out_rows: Range<usize>,
    dst: &GpuImage,
    dst_row: usize,
) -> Result<(), GpuError> {
    let rows = out_rows.len();
    if src.width != dst.width {
        return Err(GpuError::InvalidInput(format!(
            "a vertical resize keeps the width: {} and {} columns",
            src.width, dst.width
        )));
    }
    if src.planes < 3 || dst.planes < 3 {
        return Err(GpuError::InvalidInput(
            "a resize reads and writes three planes".into(),
        ));
    }
    if rows == 0 || out_rows.end > taps.taps.out_len() || dst_row + rows > dst.height {
        return Err(GpuError::InvalidInput(format!(
            "output rows {out_rows:?} of {} into rows {dst_row}.. of a {} row image",
            taps.taps.out_len(),
            dst.height
        )));
    }
    let span = taps.taps.input_span(out_rows.clone());
    if span.start < src_first || span.end > src_first + src.height {
        return Err(GpuError::InvalidInput(format!(
            "the outputs {out_rows:?} read input rows {span:?}, the source holds {src_first}..{}",
            src_first + src.height
        )));
    }
    dispatch(
        passes,
        taps,
        src,
        dst,
        false,
        dst.width,
        rows,
        src_first,
        dst_row,
        out_rows.start,
    )
}

/// Resizes `image` to `out_width` x `out_height` on the device (both passes, one submission) and
/// reads the result back: the host wrapper the tests use. The horizontal pass runs first, into an
/// intermediate image `out_width` wide, as the CPU resize does.
///
/// # Errors
///
/// [`GpuError::InvalidInput`] for an empty image or size, and the errors of the passes, the upload
/// and the readback.
pub fn resize(
    ctx: &GpuContext,
    image: &RgbImage,
    out_width: usize,
    out_height: usize,
) -> Result<RgbImage, GpuError> {
    if image.width == 0 || image.height == 0 || out_width == 0 || out_height == 0 {
        return Err(GpuError::InvalidInput(
            "a resize needs a picture and a size".into(),
        ));
    }
    let xs = PreparedTaps::new(ctx, Taps::new(image.width, out_width))?;
    let ys = PreparedTaps::new(ctx, Taps::new(image.height, out_height))?;
    let src = GpuImage::upload_rgb(ctx, "resize input", image)?;
    let mid = GpuImage::alloc(
        ctx,
        "resize mid",
        out_width,
        image.height,
        3,
        Placement::whole(out_width, image.height),
    )?;
    let dst = GpuImage::alloc(
        ctx,
        "resize output",
        out_width,
        out_height,
        3,
        Placement::whole(out_width, out_height),
    )?;
    let mut passes = PassList::new(ctx, "resize");
    encode_resize_h(&mut passes, &xs, &src, 0..image.height, &mid, 0)?;
    encode_resize_v(&mut passes, &ys, &mid, 0, 0..out_height, &dst, 0)?;
    let read = dst.encode_read(&mut passes)?;
    let done = passes.submit()?.finish()?;
    let data = dst.take(&done, read)?;
    Ok(RgbImage {
        width: out_width,
        height: out_height,
        place: Placement::whole(out_width, out_height),
        data,
    })
}

#[cfg(test)]
mod tests {
    use std::mem::{offset_of, size_of};

    use naga::valid::{Capabilities, ValidationFlags, Validator};

    use super::*;
    use crate::device::StorageMode;
    use crate::kernels::shader_source;

    #[test]
    fn kernel_values() {
        assert_eq!(lanczos3(0.0), 1.0);
        for n in [-2.0, -1.0, 1.0, 2.0] {
            assert!(lanczos3(n).abs() < 1e-15, "zero at integers: {n}");
        }
        assert_eq!(lanczos3(3.0), 0.0);
        assert_eq!(lanczos3(-3.5), 0.0);
        assert!((lanczos3(0.5) - 0.607_927_101_854_026_6).abs() < 1e-12);
        assert!(lanczos3(1.5) < 0.0 && lanczos3(2.5) > 0.0);
    }

    #[test]
    fn weights_sum_to_one_and_spans_grow_with_the_output() {
        for (i, o) in [(100, 37), (64, 64), (1000, 999), (10, 3), (7, 70), (1, 1)] {
            let t = Taps::new(i, o);
            assert_eq!((t.in_len(), t.out_len()), (i, o));
            let mut prev = (0, 0);
            for x in 0..o {
                let (first, w) = t.row(x);
                let sum: f64 = w.iter().sum();
                assert!((sum - 1.0).abs() < 1e-14, "{i}->{o} at {x}: {sum}");
                assert!(!w.is_empty() && first + w.len() <= i);
                assert!(
                    first >= prev.0 && first + w.len() >= prev.1,
                    "{i}->{o} at {x}"
                );
                prev = (first, first + w.len());
                assert_eq!(t.input_span(x..x + 1), first..first + w.len());
            }
            assert!(t.max_taps() >= 1);
            assert_eq!(t.input_span(0..0), 0..0);
            let all = t.input_span(0..o);
            assert_eq!(all.start, 0);
            assert_eq!(all.end, i);
        }
    }

    #[test]
    fn the_uniform_block_has_the_layout_of_the_wgsl_struct() {
        let source = shader_source(StorageMode::F32, KernelId::Resize);
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
        Validator::new(ValidationFlags::all(), Capabilities::empty())
            .validate(&module)
            .expect("valid");
        let var = module
            .global_variables
            .iter()
            .map(|(_, v)| v)
            .find(|v| v.name.as_deref() == Some("ru"))
            .expect("the uniform ru");
        let naga::TypeInner::Struct { members, span } = &module.types[var.ty].inner else {
            panic!("ru is not a struct");
        };
        let fields: [(&str, usize); 3] = [
            ("dims", offset_of!(ResizeU, dims)),
            ("src", offset_of!(ResizeU, src)),
            ("dst", offset_of!(ResizeU, dst)),
        ];
        assert_eq!(members.len(), fields.len());
        for (member, (name, offset)) in members.iter().zip(fields) {
            assert_eq!(member.name.as_deref(), Some(name));
            assert_eq!(member.offset as usize, offset, "offset of {name}");
        }
        assert_eq!(*span as usize, size_of::<ResizeU>());
    }

    #[test]
    fn the_largest_binding_is_three_planes_of_the_window() {
        let window = Rect::new(0, 0, 4001, 1000);
        assert_eq!(max_binding_bytes(window), 12 * 4002 * 1000);
    }
}

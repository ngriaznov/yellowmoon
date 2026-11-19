//! The Lens type of a Blur node on the device (`docs/33-blur.md` section 3.5, 11 items 75 to 80, BL-07b): the coverage of an
//! aperture by row segments.
//!
//! The kernel reads the pixels (level 0 of the scaffold's pyramid, `levels() = 0`) in `blur_lens.wgsl`, which has three modes of
//! one entry point:
//!
//! 0. up to `R = 8` px the exact table: the nonzero taps `ym_color::blur::lens::exact_taps`, summed in the oracle's order;
//! 1. above it the **scan**: one thread per row of the read area makes the row's prefix sums in 64-bit fixed point (two `u32`
//!    words a plane) and the raw values, so that any difference of two prefixes is exact and a tile of any origin has the whole
//!    frame's bits;
//! 2. the **rows**: every pixel of the made area adds the sub-rows of `ym_color::blur::lens::row_table` (an f32 copy of the CPU's
//!    f64 table) in table order, each `w * ((whole + vb ub) - va ua)`.
//!
//! The CPU twin is [`crate::cpu::blur_lens`], which is `ym_color::blur::lens::gather` in f64 (the CPU run's own fast method).
//! The scheduler's reach is `ceil(R) + 1` (`ym_color::blur::lens::reach`): the made area grown by it holds every pixel and
//! every end pixel a chord reaches into, and the prefix is the read area's own (no origin beyond it matters).
//!
//! `R` is in pixels of the photo's extent (`Pyramid::frame`), the unit of `docs/33` section 2.2.

use bytemuck::{Pod, Zeroable};
use ym_color::blur::lens::{exact_taps, row_table, uses_rows};
use ym_color::blur::{BlurKind, BlurParams, radius_px};
use ym_color::mask_raster::Area;

use super::blur_stack::{BlurKernel, BlurU, Pyramid, SumsImage, groups, placed};
use super::shader::KernelId;
use super::util::{GROUP, STORAGE_IN};
use crate::device::{GpuContext, GpuError};
use crate::stream::PassList;

/// One vec4 of the table buffer.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vec4([f32; 4]);

/// The device kernel of a Lens node: the aperture's numbers (`R` is worked out from the photo's extent when the node runs).
#[derive(Debug, Clone, Copy)]
pub struct LensKernel {
    /// Radius, 0 to 100 (the panel's number).
    radius: f64,
    /// 0 for a round aperture, else the blades.
    blades: u32,
    /// The polygon's rotation, degrees.
    rotation_deg: f64,
}

impl LensKernel {
    /// The kernel of `params`, or `None` when they are not a Lens.
    #[must_use]
    pub fn of(params: &BlurParams) -> Option<Self> {
        match params.kind {
            BlurKind::Lens {
                radius,
                blades,
                rotation_deg,
                ..
            } => Some(Self {
                radius,
                blades,
                rotation_deg,
            }),
            _ => None,
        }
    }

    /// The aperture's radius in pixels of a photo whose extent is `frame`.
    #[must_use]
    pub fn r_px(&self, frame: Area) -> f64 {
        radius_px(self.radius, frame.w.max(frame.h) as f64)
    }
}

impl BlurKernel for LensKernel {
    fn name(&self) -> &'static str {
        "Lens"
    }

    fn levels(&self) -> u32 {
        // The taps and the rows read the pixels (level 0).
        0
    }

    fn encode(
        &self,
        ctx: &GpuContext,
        passes: &mut PassList<'_>,
        src: &Pyramid,
        out: &SumsImage,
    ) -> Result<(), GpuError> {
        let (read, frame, made) = (src.read_area(), src.frame, out.area);
        let r = self.r_px(frame);
        let base = &src.levels[0].img;
        // The table: the exact taps (mode 0), or the sub-rows as two vec4 each (mode 2).
        let (list, count, mode): (Vec<Vec4>, usize, u32) = if uses_rows(r) {
            let rows = row_table(r, self.blades, self.rotation_deg);
            let mut v = Vec::with_capacity(2 * rows.len());
            for e in &rows {
                v.push(Vec4([e.ka as f32, e.kb as f32, e.fa as f32, e.fb as f32]));
                v.push(Vec4([e.dy as f32, e.w as f32, 0.0, 0.0]));
            }
            (v, rows.len(), 2)
        } else {
            let taps = exact_taps(r, self.blades, self.rotation_deg);
            let v = taps
                .iter()
                .map(|&(dx, dy, k)| Vec4([dx as f32, dy as f32, k as f32, 0.0]))
                .collect::<Vec<_>>();
            (v, taps.len(), 0)
        };
        let bytes: &[u8] = bytemuck::cast_slice(&list);
        let buffer =
            ctx.pool()
                .acquire("blur lens table", (bytes.len() as u64).max(16), STORAGE_IN)?;
        if !bytes.is_empty() {
            ctx.queue.write_buffer(&buffer, 0, bytes);
        }
        let table = passes.keep(buffer);
        // The scan's buffers (mode 2 only): the prefix words and the raw values, `W + 2` columns a row of the read area.
        let columns = (read.w + 2) * read.h;
        let scratch_bytes = if mode == 2 { 16 * columns as u64 } else { 16 };
        let phi = passes.scratch("blur lens prefix hi", scratch_bytes)?;
        let plo = passes.scratch("blur lens prefix lo", scratch_bytes)?;
        let pv = passes.scratch("blur lens values", scratch_bytes)?;
        let uniform = |mode: u32| BlurU {
            a: placed(base, read.x, read.y),
            b: placed(&out.img, made.x, made.y),
            c: [read.x as i32, read.y as i32, read.w as i32, read.h as i32],
            win: [made.x as i32, made.y as i32, made.w as i32, made.h as i32],
            op: [mode, count as u32, 0, 0],
            ..BlurU::default()
        };
        if mode == 2 {
            let ub = passes.uniform("blur lens scan uniform", &uniform(1))?;
            passes.dispatch(
                KernelId::BlurLens,
                "blur lens scan",
                &[
                    (0, ub.binding()),
                    (1, base.buf.binding()),
                    (2, out.img.buf.binding()),
                    (3, table.binding()),
                    (4, phi.binding()),
                    (5, plo.binding()),
                    (6, pv.binding()),
                ],
                (1, (read.h as u32).div_ceil(GROUP), 1),
            )?;
        }
        let ub = passes.uniform("blur lens uniform", &uniform(mode))?;
        passes.dispatch(
            KernelId::BlurLens,
            "blur lens",
            &[
                (0, ub.binding()),
                (1, base.buf.binding()),
                (2, out.img.buf.binding()),
                (3, table.binding()),
                (4, phi.binding()),
                (5, plo.binding()),
                (6, pv.binding()),
            ],
            groups(made.w, made.h),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::mem::size_of;

    use super::*;

    #[test]
    fn the_table_entries_are_whole_vec4s() {
        assert_eq!(size_of::<Vec4>(), 16);
    }
}

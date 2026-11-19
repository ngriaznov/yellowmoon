//! Motion on the device (`docs/33-blur.md` 3.5, 11 item 50, BL-08): the type's [`BlurKernel`].
//!
//! The exact line of the oracle, gathered over the pixels it touches, in `blur_motion.wgsl`: one pass over the made area, each
//! pixel adding the taps of `ym_color::blur::motion::taps` (the nonzero weights of the oracle's table, in its order) over the
//! weighted planes at level 0 of the pyramid. The taps are `f32` copies of the CPU's `f64` weights in a storage buffer of
//! `(dx, dy, w, 0)`; there are at most about `1.5 l + 4` of them (a thousand at the largest length on 50 MP).
//!
//! The CPU twin is [`crate::cpu::blur_motion`], which is `ym_color::blur::motion::gather` in `f64` (the CPU run's own fast
//! method). A pixel's sums depend on its place and the settings alone, so a tile of any size equals the frame bit for bit; the
//! scheduler's reach (`ym_color::blur::reach_px`, `ceil(l / 2) + 1 + 3 f`) holds every pixel a pixel reads, with room.

use bytemuck::{Pod, Zeroable};
use ym_color::blur::length_px;
use ym_color::blur::motion::taps;

use super::blur_stack::{BlurKernel, BlurU, Pyramid, SumsImage, groups, placed};
use super::shader::KernelId;
use super::util::STORAGE_IN;
use crate::device::{GpuContext, GpuError};
use crate::stream::PassList;

/// One tap as the shader reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TapGpu([f32; 4]);

/// Motion's device kernel: Length in the stored units (0 to 100) and the Angle in degrees.
#[derive(Debug, Clone, Copy)]
pub struct BlurMotion {
    length: f64,
    angle_deg: f64,
}

impl BlurMotion {
    /// The kernel of a Motion node of `length` (0 to 100) at `angle_deg`.
    #[must_use]
    pub fn new(length: f64, angle_deg: f64) -> Self {
        Self { length, angle_deg }
    }

    /// The line's length in pixels on a frame of the long edge `long_edge_px`.
    #[must_use]
    pub fn length_px(&self, long_edge_px: f64) -> f64 {
        length_px(self.length, long_edge_px)
    }
}

impl BlurKernel for BlurMotion {
    fn name(&self) -> &'static str {
        "Motion"
    }

    fn levels(&self) -> u32 {
        // The taps read the pixels (level 0).
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
        if frame.x != 0 || frame.y != 0 {
            return Err(GpuError::InvalidInput(
                "a Motion Blur node reads a photo whose extent starts at (0, 0)".to_owned(),
            ));
        }
        let l = self.length_px(frame.w.max(frame.h) as f64);
        let list: Vec<TapGpu> = taps(l, self.angle_deg)
            .iter()
            .map(|t| TapGpu([t.dx as f32, t.dy as f32, t.w as f32, 0.0]))
            .collect();
        let bytes: &[u8] = bytemuck::cast_slice(&list);
        let buffer = ctx
            .pool()
            .acquire("blur motion taps", bytes.len() as u64, STORAGE_IN)?;
        ctx.queue.write_buffer(&buffer, 0, bytes);
        let taps_buf = passes.keep(buffer);
        let base = &src.levels[0].img;
        let u = BlurU {
            a: placed(base, read.x, read.y),
            b: placed(&out.img, made.x, made.y),
            c: [read.x as i32, read.y as i32, read.w as i32, read.h as i32],
            win: [made.x as i32, made.y as i32, made.w as i32, made.h as i32],
            op: [0, list.len() as u32, 0, 0],
            ..BlurU::default()
        };
        let ub = passes.uniform("blur motion uniform", &u)?;
        passes.dispatch(
            KernelId::BlurMotion,
            "blur motion",
            &[
                (0, ub.binding()),
                (1, base.buf.binding()),
                (2, out.img.buf.binding()),
                (3, taps_buf.binding()),
            ],
            groups(made.w, made.h),
        )
    }
}

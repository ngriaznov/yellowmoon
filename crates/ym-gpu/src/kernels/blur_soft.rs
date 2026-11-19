//! Soft on the device (`docs/33-blur.md` 3.5, BL-06): the type's [`BlurKernel`].
//!
//! The pixel-integrated Gaussian of `sigma = R / 2` cut at 4 sigma, from the scaffold's weighted planes (level 0 of the
//! pyramid) to the sums `N` and `D` over the area the node makes, in `blur_soft.wgsl`:
//!
//! * **`sigma <= 4` px: a direct separable sum**, two passes over the pixels (at most 33 taps), the CPU's `gauss_weights`;
//! * **wider: the coarse grid** of cells of `f x f` pixels at absolute multiples of `f` (`ym_color::blur::soft`): the quadratic
//!   B-spline reduction of the planes to the cells (two passes), the Gaussian of the cells (two passes, at most 33 taps over
//!   `1 / f^2` of the pixels) and the B-spline read-back (one pass).
//!
//! The CPU twin is [`crate::cpu::blur_soft`], which is `ym_color::blur::soft` in `f64` (the CPU run's own fast method). The
//! weights are `f32` copies of its tables and the spline's are integers over a power of two. A pixel's sums depend on its place
//! and the settings alone, so a tile of any size equals the frame bit for bit; the scheduler's reach (`ym_color::blur::reach_px`)
//! holds every cell a pixel reads.

use bytemuck::{Pod, Zeroable};
use ym_color::blur::soft::{cell_level, cell_range, cell_weights};
use ym_color::blur::{gauss_weights, radius_px};
use ym_color::mask_raster::Area;

use super::blur_stack::{BlurKernel, BlurU, Pyramid, SumsImage, groups, placed};
use super::shader::KernelId;
use crate::cpu::Placement;
use crate::device::{GpuContext, GpuError};
use crate::image::GpuImage;
use crate::stream::PassList;

/// The most taps each side of the Gaussian (`ceil(4 * 4)` on the pixels or the cells).
const MAX_HALF: usize = 16;

/// The one-sided weights of the Gaussian as the shader's uniform: `w[0]` the centre, `w[k]` at distance `k`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Weights([[f32; 4]; 9]);

/// Soft's device kernel: Radius in the stored units (0 to 100); the pixels are the frame's long edge's share of it.
#[derive(Debug, Clone, Copy)]
pub struct BlurSoft {
    radius: f64,
}

impl BlurSoft {
    /// The kernel of a Soft node of `radius` (0 to 100).
    #[must_use]
    pub fn new(radius: f64) -> Self {
        Self { radius }
    }

    /// The Gaussian's sigma in pixels on a frame of the long edge `long_edge_px`.
    #[must_use]
    pub fn sigma_px(&self, long_edge_px: f64) -> f64 {
        radius_px(self.radius, long_edge_px) / 2.0
    }
}

/// The mode of a pass of `blur_soft.wgsl` (`op.x`).
#[derive(Clone, Copy)]
enum Mode {
    ReduceX = 0,
    ReduceY = 1,
    GaussX = 2,
    GaussY = 3,
    ReadBack = 4,
}

/// One image a pass reads or writes: the image, the place of its pixel (or cell) (0, 0).
type Placed<'a> = (&'a GpuImage, i64, i64);

/// The range a pass reads (first column, first row, width, height) and the window it writes, in the result's coordinates.
struct PassArea {
    range: [i32; 4],
    win: Area,
}

#[allow(clippy::too_many_arguments)]
fn pass(
    ctx: &GpuContext,
    passes: &mut PassList<'_>,
    label: &str,
    mode: Mode,
    (f, half): (u32, u32),
    (from, to): (Placed<'_>, Placed<'_>),
    area: &PassArea,
    weights: &Weights,
) -> Result<(), GpuError> {
    let u = BlurU {
        a: placed(from.0, from.1, from.2),
        b: placed(to.0, to.1, to.2),
        c: area.range,
        win: [
            area.win.x as i32,
            area.win.y as i32,
            area.win.w as i32,
            area.win.h as i32,
        ],
        op: [mode as u32, f, half, 0],
        ..BlurU::default()
    };
    let ub = passes.uniform("blur soft uniform", &u)?;
    let wimg = GpuImage::upload(
        ctx,
        "blur soft weights",
        36,
        1,
        1,
        bytemuck::cast_slice::<[f32; 4], f32>(&weights.0),
        Placement::whole(36, 1),
    )?;
    let done = passes.dispatch(
        KernelId::BlurSoft,
        label,
        &[
            (0, ub.binding()),
            (1, from.0.buf.binding()),
            (2, to.0.buf.binding()),
            (3, wimg.buf.binding()),
        ],
        groups(area.win.w, area.win.h),
    );
    let _ = passes.keep(wimg.buf);
    done
}

fn range(a: Area) -> [i32; 4] {
    [a.x as i32, a.y as i32, a.w as i32, a.h as i32]
}

fn weights_of(w: &[f64]) -> Result<(Weights, u32), GpuError> {
    let half = w.len() / 2;
    if half > MAX_HALF {
        return Err(GpuError::InvalidInput(format!(
            "a Soft Blur node of {half} taps each side (at most {MAX_HALF})"
        )));
    }
    let mut out = Weights::zeroed();
    for k in 0..=half {
        out.0[k / 4][k % 4] = w[half + k] as f32;
    }
    Ok((out, half as u32))
}

fn alloc(ctx: &GpuContext, label: &str, w: usize, h: usize) -> Result<GpuImage, GpuError> {
    GpuImage::alloc(ctx, label, w, h, 4, Placement::whole(w, h))
}

impl BlurKernel for BlurSoft {
    fn name(&self) -> &'static str {
        "Soft"
    }

    fn levels(&self) -> u32 {
        // The reduction reads the pixels (level 0): the B-spline cells are made here, not by the scaffold's box pyramid.
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
                "a Soft Blur node reads a photo whose extent starts at (0, 0)".to_owned(),
            ));
        }
        let sigma = self.sigma_px(frame.w.max(frame.h) as f64);
        let level = cell_level(sigma);
        let base = &src.levels[0].img;
        let px = (base, read.x, read.y);
        let mut keep: Vec<GpuImage> = Vec::new();
        if level == 0 {
            let (wts, half) = weights_of(&gauss_weights(sigma))?;
            let h = i64::from(half);
            let (ry0, ry1) = ((made.y - h).max(read.y), (made.y1() + h).min(read.y1()));
            let rows = Area {
                x: made.x,
                y: ry0,
                w: made.w,
                h: (ry1 - ry0) as usize,
            };
            let tmp = alloc(ctx, "blur soft rows", rows.w, rows.h)?;
            pass(
                ctx,
                passes,
                "blur soft x",
                Mode::GaussX,
                (1, half),
                (px, (&tmp, rows.x, rows.y)),
                &PassArea {
                    range: range(read),
                    win: rows,
                },
                &wts,
            )?;
            pass(
                ctx,
                passes,
                "blur soft y",
                Mode::GaussY,
                (1, half),
                ((&tmp, rows.x, rows.y), (&out.img, made.x, made.y)),
                &PassArea {
                    range: range(rows),
                    win: made,
                },
                &wts,
            )?;
            keep.push(tmp);
        } else {
            let f = 1i64 << level;
            let (ci0, cw) = cell_range(read.x, read.x1(), frame.w as i64, f);
            let (cj0, ch) = cell_range(read.y, read.y1(), frame.h as i64, f);
            if cw <= 0 || ch <= 0 {
                return Err(GpuError::InvalidInput(
                    "a Soft Blur node reads no whole cell".to_owned(),
                ));
            }
            let cells = Area {
                x: ci0,
                y: cj0,
                w: cw as usize,
                h: ch as usize,
            };
            let columns = Area {
                x: ci0,
                y: read.y,
                w: cw as usize,
                h: read.h,
            };
            let (wts, half) = weights_of(&cell_weights(sigma, f as usize))?;
            let none = Weights::zeroed();
            let t1 = alloc(ctx, "blur soft columns", columns.w, columns.h)?;
            let c1 = alloc(ctx, "blur soft cells", cells.w, cells.h)?;
            let c2 = alloc(ctx, "blur soft cells x", cells.w, cells.h)?;
            let c3 = alloc(ctx, "blur soft cells blurred", cells.w, cells.h)?;
            let fu = f as u32;
            pass(
                ctx,
                passes,
                "blur soft reduce x",
                Mode::ReduceX,
                (fu, 0),
                (px, (&t1, columns.x, columns.y)),
                &PassArea {
                    range: range(read),
                    win: columns,
                },
                &none,
            )?;
            pass(
                ctx,
                passes,
                "blur soft reduce y",
                Mode::ReduceY,
                (fu, 0),
                ((&t1, columns.x, columns.y), (&c1, cells.x, cells.y)),
                &PassArea {
                    range: range(columns),
                    win: cells,
                },
                &none,
            )?;
            pass(
                ctx,
                passes,
                "blur soft cells x",
                Mode::GaussX,
                (fu, half),
                ((&c1, cells.x, cells.y), (&c2, cells.x, cells.y)),
                &PassArea {
                    range: range(cells),
                    win: cells,
                },
                &wts,
            )?;
            pass(
                ctx,
                passes,
                "blur soft cells y",
                Mode::GaussY,
                (fu, half),
                ((&c2, cells.x, cells.y), (&c3, cells.x, cells.y)),
                &PassArea {
                    range: range(cells),
                    win: cells,
                },
                &wts,
            )?;
            pass(
                ctx,
                passes,
                "blur soft read-back",
                Mode::ReadBack,
                (fu, 0),
                ((&c3, cells.x, cells.y), (&out.img, made.x, made.y)),
                &PassArea {
                    range: range(cells),
                    win: made,
                },
                &none,
            )?;
            keep.extend([t1, c1, c2, c3]);
        }
        // The planes the passes read stay out of the pool until the list has run.
        for img in keep {
            let _ = passes.keep(img.buf);
        }
        Ok(())
    }
}

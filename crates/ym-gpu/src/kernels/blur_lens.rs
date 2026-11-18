//! The Lens type of a Blur node on the device (`docs/33-blur.md` section 3.5, BL-07): the coverage of an aperture over cells.
//!
//! The scaffold ([`super::blur_stack`]) makes the weighted planes and the pyramid; this kernel is the one pass that differs:
//!
//! 1. the cells of the level `L = ym_color::blur::lens::level_for(R)` (the smallest `f = 2^L` with `R / f <= 8`; level 0 when
//!    `R <= 8`: the pixels) are replaced by the table's weighted sums of the cells around them ([`LensU`], `blur_lens.wgsl`),
//!    over the cells that the pixels of the made area read back (a cell is the same number in every tile);
//! 2. at a level above 0 they are read back to pixels bilinearly ([`encode_upsample`]); at level 0 the sums are the pixels.
//!
//! The table is `ym_color::blur::lens::cell_table` (f64, 8 x 8 supersampled) sent as f32: the CPU twin's weights to rounding.
//! `R` is in pixels of the photo's extent (`Pyramid::frame`), the unit of `docs/33` section 2.2.

use bytemuck::{Pod, Zeroable};
use ym_color::blur::lens::{cell_table, level_for};
use ym_color::blur::{BlurKind, BlurParams, radius_px};
use ym_color::mask_raster::Area;

use super::blur_stack::{BlurKernel, Pyramid, PyramidLevel, SumsImage, encode_upsample};
use super::util::GROUP;
use crate::cpu::Placement;
use crate::cpu::blur::CellGrid;
use crate::device::{GpuContext, GpuError};
use crate::image::GpuImage;
use crate::stream::PassList;

use super::shader::KernelId;

/// Entries the uniform's table holds: 17 x 17 cells, padded to whole vectors.
const TABLE_VECTORS: usize = 73;

/// The uniform block of `blur_lens.wgsl` (`LU`), 1232 bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LensU {
    src: [i32; 4],
    dst: [i32; 4],
    win: [i32; 4],
    dims: [i32; 4],
    table: [[f32; 4]; TABLE_VECTORS],
}

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
    fn r_px(&self, frame: Area) -> f64 {
        radius_px(self.radius, frame.w.max(frame.h) as f64)
    }
}

/// The cells of the level `level` (grid `src`) that the pixels of `made` read back bilinearly, cut to `src`: the output grid
/// of the gather. `None` when none is in `src`.
pub fn output_grid(src: &CellGrid, made: Area) -> Option<CellGrid> {
    let level = src.level;
    let f = 1i64 << level;
    // The cell holding a pixel's centre minus a half: floor((x + 1/2) / f - 1/2) = floor((2x + 1 - f) / (2f)).
    let first = |a: i64| (2 * a + 1 - f).div_euclid(2 * f);
    let (i_lo, i_hi) = (first(made.x), first(made.x1() - 1) + 1);
    let (j_lo, j_hi) = (first(made.y), first(made.y1() - 1) + 1);
    let (i0, i1) = (i_lo.max(src.i0), i_hi.min(src.i0 + src.w as i64 - 1));
    let (j0, j1) = (j_lo.max(src.j0), j_hi.min(src.j0 + src.h as i64 - 1));
    (i1 >= i0 && j1 >= j0).then_some(CellGrid {
        level,
        i0,
        j0,
        w: (i1 - i0 + 1) as usize,
        h: (j1 - j0 + 1) as usize,
    })
}

impl BlurKernel for LensKernel {
    fn name(&self) -> &'static str {
        "Lens"
    }

    fn levels(&self) -> u32 {
        // The level depends on the photo's extent: `levels_for`.
        0
    }

    fn levels_for(&self, frame: Area) -> u32 {
        level_for(self.r_px(frame))
    }

    fn encode(
        &self,
        ctx: &GpuContext,
        passes: &mut PassList<'_>,
        src: &Pyramid,
        out: &SumsImage,
    ) -> Result<(), GpuError> {
        let r = self.r_px(src.frame);
        let level = level_for(r);
        let Some(level_src) = src.level(level) else {
            return Err(GpuError::InvalidInput(format!(
                "a Lens Blur node: the pyramid has no level {level} over the area it reads"
            )));
        };
        let table = cell_table(r, self.blades, self.rotation_deg, level);
        if table.half > 8 {
            return Err(GpuError::InvalidInput(format!(
                "a Lens Blur node: the aperture table is {} cells wide, at most 17 are made",
                2 * table.half + 1
            )));
        }
        let mut packed = [[0.0f32; 4]; TABLE_VECTORS];
        for (i, w) in table.weights.iter().enumerate() {
            packed[i / 4][i % 4] = *w as f32;
        }
        // Level 0: the sums are the gather itself over the made pixels. Else the cells the made pixels read.
        let (grid, direct) = if level == 0 {
            (
                CellGrid {
                    level: 0,
                    i0: out.area.x,
                    j0: out.area.y,
                    w: out.area.w,
                    h: out.area.h,
                },
                true,
            )
        } else {
            let Some(g) = output_grid(&level_src.grid, out.area) else {
                return Err(GpuError::InvalidInput(
                    "a Lens Blur node: no cell of the pyramid is read by the made area".to_owned(),
                ));
            };
            (g, false)
        };
        let cells_img = if direct {
            None
        } else {
            Some(GpuImage::alloc(
                ctx,
                "blur lens cells",
                grid.w,
                grid.h,
                4,
                Placement::whole(grid.w, grid.h),
            )?)
        };
        let dst: &GpuImage = cells_img.as_ref().unwrap_or(&out.img);
        let u = LensU {
            src: [
                level_src.grid.i0 as i32,
                level_src.grid.j0 as i32,
                level_src.img.stride as i32,
                level_src.img.plane_len() as i32,
            ],
            dst: [
                grid.i0 as i32,
                grid.j0 as i32,
                dst.stride as i32,
                dst.plane_len() as i32,
            ],
            win: [grid.i0 as i32, grid.j0 as i32, grid.w as i32, grid.h as i32],
            dims: [
                level_src.grid.w as i32,
                level_src.grid.h as i32,
                table.half as i32,
                0,
            ],
            table: packed,
        };
        let ub = passes.uniform("blur lens uniform", &u)?;
        passes.dispatch(
            KernelId::BlurLens,
            "blur lens",
            &[
                (0, ub.binding()),
                (1, level_src.img.buf.binding()),
                (2, dst.buf.binding()),
            ],
            (
                (grid.w as u32).div_ceil(GROUP),
                (grid.h as u32).div_ceil(GROUP),
                1,
            ),
        )?;
        if let Some(img) = cells_img {
            let blurred = PyramidLevel { grid, img };
            encode_upsample(passes, &blurred, &out.img, out.area, true)?;
            let _ = passes.keep(blurred.img.buf);
        }
        Ok(())
    }
}

/// The WGSL of the Lens gather (it has its own uniform, so it does not share the scaffold's block).
pub(super) fn body() -> String {
    include_str!("../shaders/blur_lens.wgsl").to_owned()
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
        let source = shader_source(StorageMode::F32, KernelId::BlurLens);
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
        Validator::new(ValidationFlags::all(), Capabilities::empty())
            .validate(&module)
            .expect("valid");
        let var = module
            .global_variables
            .iter()
            .map(|(_, v)| v)
            .find(|v| v.name.as_deref() == Some("lu"))
            .expect("the uniform lu");
        let naga::TypeInner::Struct { members, span } = &module.types[var.ty].inner else {
            panic!("lu is not a struct");
        };
        let fields = [
            ("src", offset_of!(LensU, src)),
            ("dst", offset_of!(LensU, dst)),
            ("win", offset_of!(LensU, win)),
            ("dims", offset_of!(LensU, dims)),
            ("table", offset_of!(LensU, table)),
        ];
        assert_eq!(members.len(), fields.len());
        for (member, (name, offset)) in members.iter().zip(fields) {
            assert_eq!(member.name.as_deref(), Some(name));
            assert_eq!(member.offset as usize, offset, "offset of {name}");
        }
        assert_eq!(*span as usize, size_of::<LensU>());
        assert_eq!(size_of::<LensU>(), 1232);
    }

    #[test]
    fn the_output_grid_holds_the_cells_the_pixels_read() {
        let src = CellGrid {
            level: 2,
            i0: 0,
            j0: 0,
            w: 100,
            h: 100,
        };
        // Pixels 40..60 of a 4-px cell grid: the centre of pixel 40 is 40.5 / 4 - 0.5 = 9.6 (cells 9 and 10), of pixel 59 is 14.2.
        let g = output_grid(
            &src,
            Area {
                x: 40,
                y: 40,
                w: 20,
                h: 20,
            },
        )
        .expect("grid");
        assert_eq!((g.i0, g.w), (9, 7));
        // A made area at the grid's edge reads only the cells that exist.
        let g = output_grid(
            &src,
            Area {
                x: 0,
                y: 0,
                w: 4,
                h: 4,
            },
        )
        .expect("grid");
        assert_eq!((g.i0, g.w), (0, 2));
    }
}

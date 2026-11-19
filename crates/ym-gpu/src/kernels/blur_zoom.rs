//! The Zoom type of a Blur node on the device (`docs/33-blur.md` section 3.5, BL-09): a radial line of up to 32 taps read
//! trilinearly from the scaffold's pyramid.
//!
//! The scaffold ([`super::blur_stack`]) makes the weighted planes and the pyramid up to the level
//! `L = ym_color::blur::zoom::top_level` (the smallest cell that holds the largest spacing of the taps in the frame, from the
//! span and the centre). The kernel is one pass per level `k = 0 ..= L` ([`ZoomU`], `blur_zoom.wgsl`): a pixel is made by the pass
//! of the lower level `l0` its taps read (the others return at once), which reads the levels `l0` and `l0 + 1` and writes the
//! sums `N` and `D` over the made area, so a pass binds two levels whatever `L` is (K3: one uniform and a few storage buffers).
//!
//! The CPU twin is `ym_color::blur::zoom::gather`, in f64, which is also the CPU run's own fast method. The centre is `c = center
//! x extent - 1/2` in pixel coordinates, sent as an integer and a fraction, so that a pixel's offset from it is exact in f32.

use bytemuck::{Pod, Zeroable};
use ym_color::blur::zoom::top_level;
use ym_color::blur::{BlurKind, BlurParams, zoom_span};
use ym_color::mask_raster::Area;

use super::blur_stack::{BlurKernel, Pyramid, SumsImage, groups, placed};
use super::shader::KernelId;
use crate::device::{GpuContext, GpuError};
use crate::stream::PassList;

/// The uniform block of `blur_zoom.wgsl` (`ZU`), 112 bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ZoomU {
    lo: [i32; 4],
    hi: [i32; 4],
    dst: [i32; 4],
    win: [i32; 4],
    dims: [i32; 4],
    cen: [i32; 4],
    p: [f32; 4],
}

/// The device kernel of a Zoom node: the Amount and the centre (a fraction of the frame, through crop and rotation already).
#[derive(Debug, Clone, Copy)]
pub struct ZoomKernel {
    /// Amount, 0 to 100 (the panel's number).
    amount: f64,
    /// The centre as a fraction of the photo's extent.
    center: [f64; 2],
}

impl ZoomKernel {
    /// The kernel of `params`, or `None` when they are not a Zoom.
    #[must_use]
    pub fn of(params: &BlurParams) -> Option<Self> {
        match params.kind {
            BlurKind::Zoom { amount, center } => Some(Self { amount, center }),
            _ => None,
        }
    }

    /// The span `a` of the line (`ym_color::blur::zoom_span`).
    fn span(&self) -> f64 {
        zoom_span(self.amount)
    }
}

impl BlurKernel for ZoomKernel {
    fn name(&self) -> &'static str {
        "Zoom"
    }

    fn levels(&self) -> u32 {
        // The top level depends on the photo's extent: `levels_for`.
        0
    }

    fn levels_for(&self, frame: Area) -> u32 {
        top_level(self.span(), self.center, frame.w, frame.h)
    }

    fn encode(
        &self,
        _ctx: &GpuContext,
        passes: &mut PassList<'_>,
        src: &Pyramid,
        out: &SumsImage,
    ) -> Result<(), GpuError> {
        let top = self.levels_for(src.frame);
        if src.levels.len() <= top as usize {
            return Err(GpuError::InvalidInput(format!(
                "a Zoom Blur node: the pyramid has {} levels over the area it reads, {} are needed",
                src.levels.len(),
                top + 1
            )));
        }
        let cx = self.center[0] * src.frame.w as f64 - 0.5;
        let cy = self.center[1] * src.frame.h as f64 - 0.5;
        let (cxi, cyi) = (cx.floor(), cy.floor());
        for k in 0..=top {
            let lo = &src.levels[k as usize];
            let hi = &src.levels[(k + 1).min(top) as usize];
            let u = ZoomU {
                lo: placed(&lo.img, lo.grid.i0, lo.grid.j0),
                hi: placed(&hi.img, hi.grid.i0, hi.grid.j0),
                dst: placed(&out.img, out.area.x, out.area.y),
                win: [
                    out.area.x as i32,
                    out.area.y as i32,
                    out.area.w as i32,
                    out.area.h as i32,
                ],
                dims: [
                    lo.grid.w as i32,
                    lo.grid.h as i32,
                    hi.grid.w as i32,
                    hi.grid.h as i32,
                ],
                cen: [cxi as i32, cyi as i32, k as i32, top as i32],
                p: [
                    self.span() as f32,
                    (cx - cxi) as f32,
                    (cy - cyi) as f32,
                    0.0,
                ],
            };
            let ub = passes.uniform("blur zoom uniform", &u)?;
            passes.dispatch(
                KernelId::BlurZoom,
                "blur zoom",
                &[
                    (0, ub.binding()),
                    (1, lo.img.buf.binding()),
                    (2, hi.img.buf.binding()),
                    (3, out.img.buf.binding()),
                ],
                groups(out.area.w, out.area.h),
            )?;
        }
        Ok(())
    }
}

/// The WGSL of the Zoom gather (it has its own uniform, so it does not share the scaffold's block).
pub(super) fn body() -> String {
    include_str!("../shaders/blur_zoom.wgsl").to_owned()
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
        let source = shader_source(StorageMode::F32, KernelId::BlurZoom);
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
        Validator::new(ValidationFlags::all(), Capabilities::empty())
            .validate(&module)
            .expect("valid");
        let var = module
            .global_variables
            .iter()
            .map(|(_, v)| v)
            .find(|v| v.name.as_deref() == Some("zu"))
            .expect("the uniform zu");
        let naga::TypeInner::Struct { members, span } = &module.types[var.ty].inner else {
            panic!("zu is not a struct");
        };
        let fields = [
            ("lo", offset_of!(ZoomU, lo)),
            ("hi", offset_of!(ZoomU, hi)),
            ("dst", offset_of!(ZoomU, dst)),
            ("win", offset_of!(ZoomU, win)),
            ("dims", offset_of!(ZoomU, dims)),
            ("cen", offset_of!(ZoomU, cen)),
            ("p", offset_of!(ZoomU, p)),
        ];
        assert_eq!(members.len(), fields.len());
        for (member, (name, offset)) in members.iter().zip(fields) {
            assert_eq!(member.name.as_deref(), Some(name));
            assert_eq!(member.offset as usize, offset, "offset of {name}");
        }
        assert_eq!(*span as usize, size_of::<ZoomU>());
        assert_eq!(size_of::<ZoomU>(), 112);
    }

    #[test]
    fn the_levels_follow_the_span_and_the_centre() {
        let big = Area::whole(6000, 4000);
        let corner = ZoomKernel {
            amount: 100.0,
            center: [0.0, 0.0],
        };
        assert_eq!(corner.levels_for(big), 5);
        let middle = ZoomKernel {
            amount: 100.0,
            center: [0.5, 0.5],
        };
        assert_eq!(middle.levels_for(big), 4);
        let small = ZoomKernel {
            amount: 5.0,
            center: [0.5, 0.5],
        };
        assert_eq!(small.levels_for(Area::whole(400, 300)), 0);
    }
}

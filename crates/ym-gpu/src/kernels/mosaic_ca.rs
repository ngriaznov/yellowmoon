//! R5b, the lateral chromatic aberration corrected on the mosaic, on the device
//! (`docs/05-raw-processing.md` section R9, "Real files").
//!
//! The CPU reference is [`crate::cpu::lens::correct_mosaic_ca`]: every red and blue photosite takes its
//! colour's value at the position where green shows the same feature, a Catmull-Rom sample of the
//! colour's own lattice at `site + shift(r) d / (2 r)` lattice steps, green untouched. The curves are
//! the CPU's ([`CaCurves`], from `LensGeometry::ca_curves`), uploaded as `f32` knots;
//! the shader finds the shift with the same linear interpolation and end-segment extension
//! (`RadialCurve::shift_at`). The position of a site never forms a large float: the offset from the
//! optical centre is an integer plus a constant fraction, and the displacement is split into its
//! integer part and its fraction before it is added to the integer lattice index (D4), so a tile with
//! the stage's halo ([`crate::cpu::lens::mosaic_ca_halo_px`]) equals the whole frame bit for bit.

use bytemuck::{Pod, Zeroable};

use super::shader::KernelId;
use super::util::{GROUP, storage_buffer};
use crate::cpu::demosaic::CaCurves;
use crate::cpu::tiling::Rect;
use crate::cpu::{Cfa, CfaColor, Mosaic};
use crate::device::{GpuContext, GpuError};
use crate::image::GpuImage;
use crate::stream::PassList;

/// The uniform block of `mosaic_ca.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CaU {
    /// The runtime zero of the barrier (always +0.0 bits).
    zero: [u32; 4],
    /// Input width, height, row stride.
    dims: [u32; 4],
    /// Core x, y, width, height in the input.
    core: [u32; 4],
    /// Output row stride, red knots, blue knots.
    dst: [u32; 4],
    /// The plane index at (0, 0), (1, 0), (0, 1), (1, 1).
    cfa: [u32; 4],
    /// Red lattice phase (x, y), blue lattice phase (x, y).
    lat: [u32; 4],
    /// Frame column of the input's column 0 and frame row of its row 0, each minus `floor(centre)`.
    org: [i32; 4],
    /// `0.5 - fract(cx)`, `0.5 - fract(cy)`.
    frac: [f32; 4],
}

/// The CA curves of a photo as the shader reads them.
#[derive(Clone, Debug)]
pub struct CaTable {
    /// Red knots, then blue knots: `(radius_px, shift_px)` rounded to `f32`.
    knots: Vec<[f32; 2]>,
    n_red: u32,
    n_blue: u32,
    centre_int: [i32; 2],
    centre_frac: [f32; 2],
}

impl CaTable {
    /// The table of `curves`.
    ///
    /// # Errors
    ///
    /// [`GpuError::InvalidInput`] for an optical centre out of range or knots that do not fit `f32`.
    pub fn from_curves(curves: &CaCurves) -> Result<CaTable, GpuError> {
        let (cx, cy) = (curves.centre_x.floor(), curves.centre_y.floor());
        let limit = f64::from(i32::MAX / 2);
        if !(cx.abs() < limit && cy.abs() < limit) {
            return Err(GpuError::InvalidInput(format!(
                "mosaic CA: the optical centre ({}, {}) is out of range",
                curves.centre_x, curves.centre_y
            )));
        }
        let mut knots = Vec::with_capacity(curves.red.len() + curves.blue.len());
        for curve in [&curves.red, &curves.blue] {
            for (&r, &s) in curve.radius_px().iter().zip(curve.shift_px()) {
                let k = [r as f32, s as f32];
                if !(k[0].is_finite() && k[1].is_finite()) {
                    return Err(GpuError::InvalidInput(
                        "mosaic CA: the knots do not fit f32".into(),
                    ));
                }
                knots.push(k);
            }
        }
        Ok(CaTable {
            knots,
            n_red: curves.red.len() as u32,
            n_blue: curves.blue.len() as u32,
            centre_int: [cx as i32, cy as i32],
            centre_frac: [
                (0.5 - (curves.centre_x - cx)) as f32,
                (0.5 - (curves.centre_y - cy)) as f32,
            ],
        })
    }
}

/// The phase `(x, y)` of the lattice of `colour` in the 2 x 2 cell of `cfa`.
fn phase(cfa: Cfa, colour: CfaColor) -> Result<(u32, u32), GpuError> {
    (0..2usize)
        .flat_map(|dy| (0..2usize).map(move |dx| (dx, dy)))
        .find(|&(dx, dy)| cfa.color_at(dx, dy) == colour)
        .map(|(dx, dy)| (dx as u32, dy as u32))
        .ok_or_else(|| {
            GpuError::InvalidInput(format!(
                "mosaic CA: the colour filter pattern has no {colour:?} photosite"
            ))
        })
}

/// The largest storage binding (bytes) the pass needs for an input rectangle of `rect` photosites:
/// the mosaic itself.
#[must_use]
pub fn max_binding_bytes(rect: Rect) -> u64 {
    GpuImage::buffer_bytes(rect.w, rect.h, 1)
}

/// Records R5b for the `core` rectangle of `input` (device in, device out). `cfa` is the pattern at
/// the input's first photosite. The rest of the input is the context the correction reads (the
/// stage's halo); lattice sites beyond its edge are replicated, as the CPU does at the edge of the
/// buffer it gets.
///
/// # Errors
///
/// [`GpuError::InvalidInput`] for an input that is not one plane or is below 2 x 2, an empty core or
/// one outside the input, or a pattern without red or blue; the pool's errors; shader errors.
pub fn encode_mosaic_ca(
    passes: &mut PassList<'_>,
    input: &GpuImage,
    cfa: Cfa,
    core: Rect,
    table: &CaTable,
) -> Result<GpuImage, GpuError> {
    let ctx = passes.context();
    if input.planes != 1 || input.width < 2 || input.height < 2 {
        return Err(GpuError::InvalidInput(format!(
            "mosaic CA: the input must be one plane of at least 2 x 2 photosites, got {} plane(s) of {} x {}",
            input.planes, input.width, input.height
        )));
    }
    if core.is_empty() || core.x1() > input.width || core.y1() > input.height {
        return Err(GpuError::InvalidInput(format!(
            "mosaic CA: the core {core:?} is empty or outside the {} x {} input",
            input.width, input.height
        )));
    }
    let (rx, ry) = phase(cfa, CfaColor::Red)?;
    let (bx, by) = phase(cfa, CfaColor::Blue)?;
    let out = GpuImage::alloc(
        ctx,
        "mosaic ca",
        core.w,
        core.h,
        1,
        input.place.offset(core.x, core.y),
    )?;
    let plane = |dx: usize, dy: usize| cfa.color_at(dx, dy).plane() as u32;
    let u = CaU {
        zero: [0; 4],
        dims: [
            input.width as u32,
            input.height as u32,
            input.stride as u32,
            0,
        ],
        core: [core.x as u32, core.y as u32, core.w as u32, core.h as u32],
        dst: [out.stride as u32, table.n_red, table.n_blue, 0],
        cfa: [plane(0, 0), plane(1, 0), plane(0, 1), plane(1, 1)],
        lat: [rx, ry, bx, by],
        org: [
            input.place.x0 as i32 - table.centre_int[0],
            input.place.y0 as i32 - table.centre_int[1],
            0,
            0,
        ],
        frac: [table.centre_frac[0], table.centre_frac[1], 0.0, 0.0],
    };
    let knots = passes.keep(storage_buffer(
        ctx,
        "mosaic ca curves",
        bytemuck::cast_slice(&table.knots),
    )?);
    let ub = passes.uniform("mosaic ca uniform", &u)?;
    passes.dispatch(
        KernelId::MosaicCa,
        "mosaic ca",
        &[
            (0, ub.binding()),
            (1, input.buf.binding()),
            (2, knots.binding()),
            (3, out.buf.binding()),
        ],
        (
            (core.w as u32).div_ceil(GROUP),
            (core.h as u32).div_ceil(GROUP),
            1,
        ),
    )?;
    Ok(out)
}

/// R5b on the host: uploads `mosaic`, corrects the `core` and reads it back. For the parity tests.
///
/// # Errors
///
/// As [`encode_mosaic_ca`], plus the readback's.
pub fn mosaic_ca(
    ctx: &GpuContext,
    mosaic: &Mosaic,
    core: Rect,
    curves: &CaCurves,
) -> Result<Vec<f32>, GpuError> {
    let table = CaTable::from_curves(curves)?;
    let input = GpuImage::upload(
        ctx,
        "mosaic ca input",
        mosaic.width,
        mosaic.height,
        1,
        &mosaic.data,
        mosaic.place,
    )?;
    let mut passes = PassList::new(ctx, "mosaic ca");
    let out = encode_mosaic_ca(&mut passes, &input, mosaic.cfa, core, &table)?;
    let read = out.encode_read(&mut passes)?;
    let done = passes.submit()?.finish()?;
    out.take(&done, read)
}

/// The WGSL of the mosaic-CA kernel.
pub(super) fn body() -> String {
    let mut s = String::new();
    s.push_str(include_str!("../shaders/front_opaque.wgsl"));
    s.push_str(include_str!("../shaders/mosaic_ca.wgsl"));
    s.push_str(&format!(
        "\n@compute @workgroup_size({GROUP}, {GROUP}, 1)\n\
         fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{ ca_main(gid); }}\n"
    ));
    s
}

#[cfg(test)]
mod tests {
    use std::mem::{offset_of, size_of};

    use super::*;
    use crate::kernels::prep::tests::uniform_layout;

    #[test]
    fn the_uniform_block_has_the_layout_of_its_rust_struct() {
        let (fields, span) = uniform_layout(KernelId::MosaicCa, "pu");
        let want = [
            ("zero", offset_of!(CaU, zero)),
            ("dims", offset_of!(CaU, dims)),
            ("core", offset_of!(CaU, core)),
            ("dst", offset_of!(CaU, dst)),
            ("cfa", offset_of!(CaU, cfa)),
            ("lat", offset_of!(CaU, lat)),
            ("org", offset_of!(CaU, org)),
            ("frac", offset_of!(CaU, frac)),
        ];
        assert_eq!(fields.len(), want.len());
        for ((name, offset), (want_name, want_offset)) in fields.iter().zip(want) {
            assert_eq!(name, want_name);
            assert_eq!(*offset as usize, want_offset, "offset of {name}");
        }
        assert_eq!(span as usize, size_of::<CaU>());
    }
}

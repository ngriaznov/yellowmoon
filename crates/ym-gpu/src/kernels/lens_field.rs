//! L2, the lens's field optics, on the device (`docs/21` section 4, L2; `docs/17-gpu.md` section 11).
//!
//! The CPU reference is `ym_lens::field` (reached through [`ym_lens::apply_field`] with no reshaping):
//! each pixel gathers the input with the lens's spread function at its field radius, resampled to
//! `n x n` taps, turned to the radial direction and scaled to a half-width of `omega` pixels, from a
//! box-reduced pyramid of the input; the channels' own centroids are the lateral colour, the attachment
//! reads each channel further along the axis, and the gain `1 - e (1 - T)` is applied last. This module
//! is the same computation in `f32`.
//!
//! # Passes
//!
//! `lens_field_pyramid` once per level (level 0 is the input over the region the gathers read, rounded
//! outwards to blocks of `2^lmax` frame pixels; level `l` the mean of the aligned 2 x 2 blocks of level
//! `l - 1`), then `lens_field_gather` over the output, and, when the attachment's fringe reads beyond the
//! output, `lens_field_finish`: the gather is made over the output grown by the fringe's reach and each
//! channel is read from it at its shift, times the gain. Without the fringe the gather applies the gain
//! itself.
//!
//! # Parameters
//!
//! The spread functions on the camera's channels are [`FieldTablesBuffer`], uploaded once per lens and
//! camera ([`ym_lens::device::FieldTablesData`]); the amounts, the frame and the axis are
//! [`FieldPass`], plain numbers from the plan (`LensPlan::field_pass`). The constants of the WGSL
//! (the tap budget, `w_max`, the copy threshold) are written from `ym_lens::consts`.
//!
//! # Positions
//!
//! D4 and D5 of `docs/17-gpu.md`: a pixel is its absolute column and row; a tap lies a small offset
//! (at most about 90 px) from it, formed in `f32` and split, for the level of the pyramid, into an
//! exact integer and a fraction. The pyramid's texels are means of aligned blocks of absolute frame
//! coordinates, so a tile and the whole frame give every pixel the same bits.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use ym_lens::consts::{
    COPY_BELOW_PX, EXTRA_INFLOW_FLOOR, EXTRA_INFLOW_RAMP, MAX_TAPS_SIDE, RADII, TAP_SPACING_MAX_PX,
    W_KNEE, W_MAX,
};
use ym_lens::device::{FIELD_HEADER, FIELD_TABLE_SIZES, field_table_offset};
pub use ym_lens::device::{FieldPass, FieldTablesData};

use super::shader::KernelId;
use super::util::{GROUP, storage_buffer};
use crate::cpu::Placement;
use crate::cpu::tiling::Rect;
use crate::device::{GpuContext, GpuError};
use crate::image::GpuImage;
use crate::pool::PooledBuffer;
use crate::stream::PassList;

/// The most levels of the pyramid (`lmax` is at most 12).
const MAX_LEVELS: usize = 13;

/// The uniform block of `lens_field.wgsl` (`FieldU`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FieldU {
    /// The runtime zero of the barrier (always +0.0 bits).
    zero: [u32; 4],
    /// Frame width, height, half-diagonal in pixels, 0.
    frame: [f32; 4],
    /// `e'`, `alpha`, side softness, fringe boost.
    plan: [f32; 4],
    /// Frame width, height, `lmax`, 1 when the axis is the buffer's y.
    ints: [i32; 4],
    /// The input: frame column and row of its first pixel, width, height.
    src: [i32; 4],
    /// The input: row stride, plane length.
    srcs: [u32; 4],
    /// The rectangle written: frame column and row, width, height.
    win: [i32; 4],
    /// Output row stride, plane length, 1 to apply the gain in the gather, the level to build.
    dst: [u32; 4],
    /// The gather: frame column and row of its first pixel, width, height.
    gth: [i32; 4],
    /// The gather: row stride, plane length.
    gs: [u32; 4],
    /// Per level: offset in the pyramid (vectors), width, height.
    lvl: [[i32; 4]; MAX_LEVELS],
    /// Per level: column and row of the first texel in blocks of that level.
    org: [[i32; 4]; MAX_LEVELS],
    /// 2 when the gather is the extra light, added to the output planes; 0 otherwise.
    ex: [u32; 4],
}

/// The lens's spread functions on the device: uploaded once per lens and camera, shared by every amount
/// and every tile of a photo.
#[derive(Clone)]
pub struct FieldTablesBuffer {
    buf: Arc<PooledBuffer>,
    key: String,
}

impl std::fmt::Debug for FieldTablesBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FieldTablesBuffer")
            .field("key", &self.key)
            .field("bytes", &self.buf.size())
            .finish()
    }
}

impl FieldTablesBuffer {
    /// Uploads the tables.
    ///
    /// # Errors
    ///
    /// [`GpuError::InvalidInput`] for tables of the wrong length; the pool's errors.
    pub fn upload(ctx: &GpuContext, data: &FieldTablesData) -> Result<Self, GpuError> {
        let want = FIELD_HEADER + total_cells();
        if data.data.len() != want {
            return Err(GpuError::InvalidInput(format!(
                "lens field tables: {} vectors, expected {want}",
                data.data.len()
            )));
        }
        let buf = storage_buffer(ctx, "lens field tables", bytemuck::cast_slice(&data.data))?;
        Ok(FieldTablesBuffer {
            buf: Arc::new(buf),
            key: data.key.clone(),
        })
    }

    /// Names the lens, its data version and the camera the tables were made for.
    #[must_use]
    pub fn key(&self) -> &str {
        &self.key
    }

    /// Bytes on the device.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.buf.size()
    }
}

/// Four-lane vectors in all the cell tables: `RADII * (9 + 25 + ... + 289)`.
const fn total_cells() -> usize {
    field_table_offset(MAX_TAPS_SIDE) + RADII * MAX_TAPS_SIDE * MAX_TAPS_SIDE
}

/// Where the pyramid's levels lie, for an output rectangle.
struct Pyramid {
    /// Per level: offset in vectors, width, height, first texel column and row.
    levels: Vec<[i32; 5]>,
    /// Vectors in all.
    vectors: usize,
}

fn floor_div(a: i64, b: i64) -> i64 {
    a.div_euclid(b)
}

fn ceil_div(a: i64, b: i64) -> i64 {
    -((-a).div_euclid(b))
}

/// The pyramid the gathers of `gathered` (the output grown by the fringe's reach) read: the region grown
/// by the gather's halo, rounded outwards to blocks of `2^lmax`, its levels one after the other.
fn pyramid_layout(pass: &FieldPass, gathered: Rect) -> Pyramid {
    let halo = pass.gather_halo as i64;
    let need = [
        gathered.x as i64 - halo,
        gathered.y as i64 - halo,
        gathered.x1() as i64 + halo,
        gathered.y1() as i64 + halo,
    ];
    let b = 1i64 << pass.lmax;
    let x0 = floor_div(need[0], b) * b;
    let y0 = floor_div(need[1], b) * b;
    let x1 = ceil_div(need[2], b) * b;
    let y1 = ceil_div(need[3], b) * b;
    let (mut w, mut h) = (x1 - x0, y1 - y0);
    let (mut bx, mut by) = (x0, y0);
    let mut offset = 0usize;
    let mut levels = Vec::new();
    for _ in 0..=pass.lmax {
        levels.push([offset as i32, w as i32, h as i32, bx as i32, by as i32]);
        offset += (w * h) as usize;
        w /= 2;
        h /= 2;
        bx /= 2;
        by /= 2;
    }
    Pyramid {
        levels,
        vectors: offset,
    }
}

/// The largest storage binding (bytes) the pass needs for an output of `rect` pixels: the pyramid or
/// an image, whichever is larger.
#[must_use]
pub fn max_binding_bytes(rect: Rect, pass: &FieldPass) -> u64 {
    let gathered = grown(rect, pass.pad, pass.width, pass.height);
    let pyramid = pyramid_layout(pass, gathered).vectors as u64 * 16;
    pyramid
        .max(GpuImage::buffer_bytes(gathered.w, gathered.h, 3))
        .max(GpuImage::buffer_bytes(rect.w, rect.h, 3))
}

fn grown(r: Rect, px: usize, fw: usize, fh: usize) -> Rect {
    let x = r.x.saturating_sub(px);
    let y = r.y.saturating_sub(px);
    let x1 = (r.x1() + px).min(fw);
    let y1 = (r.y1() + px).min(fh);
    Rect::new(x, y, x1 - x, y1 - y)
}

/// Records L2 over `out` (frame coordinates at the scale of `pass`): device in, device out.
///
/// `src` is a three-plane image whose first pixel is the frame pixel `at`, in a frame of the pass's size;
/// it must cover `out` grown by [`FieldPass::halo`] and clamped to the frame. (The image's own
/// `place` is not read: a window gathered from pieces has none.) The result is `out.w x out.h`, placed
/// at `(out.x, out.y)` in that frame.
///
/// `extra` is light the picture does not record and L2 spreads (`ym_lens::l2_input`: the headroom of a clipped
/// light, and the light's colour where a channel clipped): an image and its first frame pixel, covering what `src`
/// does (`out` grown by [`FieldPass::halo`]). It is gathered with the same spread functions and added to the
/// gather of `src` by the share of how much of it the lens moves to the pixel (`EXTRA_INFLOW_RAMP`); a pixel with
/// no kernel gets none.
///
/// # Errors
///
/// [`GpuError::InvalidInput`] for a source that is not three planes or does not cover the halo, an extra light
/// that does not, or an empty output or one outside the frame; the pool's errors; shader errors.
#[allow(clippy::too_many_arguments)]
pub fn encode_field(
    passes: &mut PassList<'_>,
    tables: &FieldTablesBuffer,
    pass: &FieldPass,
    src: &GpuImage,
    at: (usize, usize),
    extra: Option<(&GpuImage, (usize, usize))>,
    out: Rect,
) -> Result<GpuImage, GpuError> {
    let ctx = passes.context();
    let (fw, fh) = (pass.width, pass.height);
    if src.planes != 3 {
        return Err(GpuError::InvalidInput(format!(
            "lens field: the source has {} planes, not 3",
            src.planes
        )));
    }
    if out.is_empty() || out.x1() > fw || out.y1() > fh {
        return Err(GpuError::InvalidInput(format!(
            "lens field: the output {out:?} is empty or outside the {fw} x {fh} frame"
        )));
    }
    let need = grown(out, pass.halo(), fw, fh);
    let have = Rect::new(at.0, at.1, src.width, src.height);
    if need.x < have.x || need.y < have.y || need.x1() > have.x1() || need.y1() > have.y1() {
        return Err(GpuError::InvalidInput(format!(
            "lens field: the source {have:?} does not cover the output {out:?} grown by {} px",
            pass.halo()
        )));
    }
    if pass.lmax as usize >= MAX_LEVELS {
        return Err(GpuError::InvalidInput(format!(
            "lens field: {} pyramid levels",
            pass.lmax + 1
        )));
    }
    let gathered = grown(out, pass.pad, fw, fh);
    if let Some((img, o)) = extra {
        let have = Rect::new(o.0, o.1, img.width, img.height);
        if img.planes != 3
            || need.x < have.x
            || need.y < have.y
            || need.x1() > have.x1()
            || need.y1() > have.y1()
        {
            return Err(GpuError::InvalidInput(format!(
                "lens field: the extra light {have:?} ({} planes) does not cover the output {out:?} grown by {} px",
                img.planes,
                pass.halo()
            )));
        }
    }
    let pyramid = pyramid_layout(pass, gathered);
    let pyr = passes.scratch("lens field pyramid", pyramid.vectors as u64 * 16)?;
    let dst = GpuImage::alloc(
        ctx,
        "lens field",
        out.w,
        out.h,
        3,
        Placement {
            x0: out.x,
            y0: out.y,
            frame_width: fw,
            frame_height: fh,
        },
    )?;
    // With no fringe reading beyond the output, and no extra light to add to the gather, the gather writes the
    // result itself.
    let fused = pass.pad == 0 && extra.is_none();
    let mid = if fused {
        None
    } else {
        Some(GpuImage::alloc(
            ctx,
            "lens field gather",
            gathered.w,
            gathered.h,
            3,
            Placement {
                x0: gathered.x,
                y0: gathered.y,
                frame_width: fw,
                frame_height: fh,
            },
        )?)
    };
    let mut base = FieldU::zeroed();
    base.frame = [
        fw as f32,
        fh as f32,
        pass.half_diag as f32,
        pass.radius_scale as f32,
    ];
    base.plan = [
        pass.e as f32,
        pass.alpha as f32,
        pass.side as f32,
        pass.fringe_boost as f32,
    ];
    base.ints = [
        fw as i32,
        fh as i32,
        pass.lmax as i32,
        i32::from(pass.axis_y),
    ];
    base.src = [
        at.0 as i32,
        at.1 as i32,
        src.width as i32,
        src.height as i32,
    ];
    base.srcs = [src.stride as u32, src.plane_len() as u32, 0, 0];
    for (l, lv) in pyramid.levels.iter().enumerate() {
        base.lvl[l] = [lv[0], lv[1], lv[2], 0];
        base.org[l] = [lv[3], lv[4], 0, 0];
    }
    // The pyramid, one pass per level.
    for (l, lv) in pyramid.levels.iter().enumerate() {
        let mut u = base;
        u.dst[3] = l as u32;
        let ub = passes.uniform("lens field pyramid uniform", &u)?;
        passes.dispatch(
            KernelId::LensFieldPyramid,
            "lens field pyramid",
            &[
                (0, ub.binding()),
                (2, pyr.binding()),
                (3, src.buf.binding()),
            ],
            (
                (lv[1] as u32).div_ceil(GROUP),
                (lv[2] as u32).div_ceil(GROUP),
                1,
            ),
        )?;
    }
    // The gather.
    let target = mid.as_ref().unwrap_or(&dst);
    let region = if fused { out } else { gathered };
    let mut u = base;
    u.win = [
        region.x as i32,
        region.y as i32,
        region.w as i32,
        region.h as i32,
    ];
    u.dst = [
        target.stride as u32,
        target.plane_len() as u32,
        u32::from(fused),
        0,
    ];
    let ub = passes.uniform("lens field gather uniform", &u)?;
    passes.dispatch(
        KernelId::LensFieldGather,
        "lens field gather",
        &[
            (0, ub.binding()),
            (1, tables.buf.binding()),
            (2, pyr.binding()),
            (4, target.buf.binding()),
        ],
        (
            (region.w as u32).div_ceil(GROUP),
            (region.h as u32).div_ceil(GROUP),
            1,
        ),
    )?;
    // The extra light: its own pyramid and gather, added to the picture's gather by the share the lens moves.
    if let Some((img, o)) = extra {
        let pyr_e = passes.scratch("lens field extra pyramid", pyramid.vectors as u64 * 16)?;
        let mut eb = base;
        eb.src = [o.0 as i32, o.1 as i32, img.width as i32, img.height as i32];
        eb.srcs = [img.stride as u32, img.plane_len() as u32, 0, 0];
        for (l, lv) in pyramid.levels.iter().enumerate() {
            let mut u = eb;
            u.dst[3] = l as u32;
            let ub = passes.uniform("lens field extra pyramid uniform", &u)?;
            passes.dispatch(
                KernelId::LensFieldPyramid,
                "lens field extra pyramid",
                &[
                    (0, ub.binding()),
                    (2, pyr_e.binding()),
                    (3, img.buf.binding()),
                ],
                (
                    (lv[1] as u32).div_ceil(GROUP),
                    (lv[2] as u32).div_ceil(GROUP),
                    1,
                ),
            )?;
        }
        let mut u = base;
        u.win = [
            region.x as i32,
            region.y as i32,
            region.w as i32,
            region.h as i32,
        ];
        u.dst = [target.stride as u32, target.plane_len() as u32, 0, 0];
        u.ex = [2, 0, 0, 0];
        let ub = passes.uniform("lens field extra gather uniform", &u)?;
        passes.dispatch(
            KernelId::LensFieldGather,
            "lens field extra gather",
            &[
                (0, ub.binding()),
                (1, tables.buf.binding()),
                (2, pyr_e.binding()),
                (4, target.buf.binding()),
            ],
            (
                (region.w as u32).div_ceil(GROUP),
                (region.h as u32).div_ceil(GROUP),
                1,
            ),
        )?;
    }
    passes.hold(Arc::clone(&tables.buf));
    // The fringe shift and the gain.
    if let Some(mid) = &mid {
        let mut u = base;
        u.win = [out.x as i32, out.y as i32, out.w as i32, out.h as i32];
        u.dst = [dst.stride as u32, dst.plane_len() as u32, 0, 0];
        u.gth = [
            gathered.x as i32,
            gathered.y as i32,
            gathered.w as i32,
            gathered.h as i32,
        ];
        u.gs = [mid.stride as u32, mid.plane_len() as u32, 0, 0];
        let ub = passes.uniform("lens field finish uniform", &u)?;
        passes.dispatch(
            KernelId::LensFieldFinish,
            "lens field finish",
            &[
                (0, ub.binding()),
                (1, tables.buf.binding()),
                (4, dst.buf.binding()),
                (5, mid.buf.binding()),
            ],
            (
                (out.w as u32).div_ceil(GROUP),
                (out.h as u32).div_ceil(GROUP),
                1,
            ),
        )?;
        // The gather is only read by the pass above; the list keeps it until the work is done.
    }
    Ok(dst)
}

/// L2 on the host's behalf: uploads `input` (three planes, `width x height`, placed at `(x0, y0)` of the
/// frame of `pass`), runs the pass over `out` and reads the planes back (`3 * out.w * out.h` samples).
/// The GPU counterpart of `ym_lens::apply_field` for a plan with no reshaping; for tests and the first
/// wiring.
///
/// # Errors
///
/// As [`encode_field`], plus the readback's.
pub fn field(
    ctx: &GpuContext,
    tables: &FieldTablesBuffer,
    pass: &FieldPass,
    input: &[f32],
    size: (usize, usize),
    at: (usize, usize),
    out: Rect,
) -> Result<Vec<f32>, GpuError> {
    let src = GpuImage::upload(
        ctx,
        "lens field source",
        size.0,
        size.1,
        3,
        input,
        Placement {
            x0: at.0,
            y0: at.1,
            frame_width: pass.width,
            frame_height: pass.height,
        },
    )?;
    let mut passes = PassList::new(ctx, "lens field");
    let img = encode_field(&mut passes, tables, pass, &src, at, None, out)?;
    let read = img.encode_read(&mut passes)?;
    let done = passes.submit()?.finish()?;
    img.take(&done, read)
}

/// L2 with an extra light on the host's behalf: as [`field`], with `extra` (three planes, its size, its first
/// frame pixel) spread besides `input` ([`encode_field`]). For tests.
///
/// # Errors
///
/// As [`encode_field`], plus the readback's.
#[allow(clippy::too_many_arguments)]
pub fn field_with_extra(
    ctx: &GpuContext,
    tables: &FieldTablesBuffer,
    pass: &FieldPass,
    input: &[f32],
    size: (usize, usize),
    at: (usize, usize),
    extra: (&[f32], (usize, usize), (usize, usize)),
    out: Rect,
) -> Result<Vec<f32>, GpuError> {
    let place = |at: (usize, usize)| Placement {
        x0: at.0,
        y0: at.1,
        frame_width: pass.width,
        frame_height: pass.height,
    };
    let src = GpuImage::upload(ctx, "lens field source", size.0, size.1, 3, input, place(at))?;
    let more = GpuImage::upload(
        ctx,
        "lens field extra light",
        extra.1.0,
        extra.1.1,
        3,
        extra.0,
        place(extra.2),
    )?;
    let mut passes = PassList::new(ctx, "lens field");
    let img = encode_field(&mut passes, tables, pass, &src, at, Some((&more, extra.2)), out)?;
    let read = img.encode_read(&mut passes)?;
    let done = passes.submit()?.finish()?;
    img.take(&done, read)
}

fn constants() -> String {
    let offsets: Vec<String> = (0..FIELD_TABLE_SIZES)
        .map(|s| format!("{}u", field_table_offset(3 + 2 * s)))
        .collect();
    format!(
        "const LF_RADII: u32 = {}u;\n\
         const LF_W_MAX: f32 = {:?};\n\
         const LF_W_KNEE: f32 = {:?};\n\
         const LF_COPY_BELOW_PX: f32 = {:?};\n\
         const LF_MAX_TAPS: u32 = {}u;\n\
         const LF_SPACING_MAX: f32 = {:?};\n\
         const LF_HEADER: u32 = {}u;\n\
         const LF_INFLOW_FROM: f32 = {:?};\n\
         const LF_INFLOW_TO: f32 = {:?};\n\
         const LF_INFLOW_FLOOR: f32 = {:?};\n\
         var<private> LF_OFF: array<u32, {}> = array<u32, {}>({});\n",
        RADII,
        W_MAX as f32,
        W_KNEE as f32,
        COPY_BELOW_PX as f32,
        MAX_TAPS_SIDE,
        TAP_SPACING_MAX_PX as f32,
        FIELD_HEADER,
        EXTRA_INFLOW_RAMP.0 as f32,
        EXTRA_INFLOW_RAMP.1 as f32,
        EXTRA_INFLOW_FLOOR as f32,
        FIELD_TABLE_SIZES,
        FIELD_TABLE_SIZES,
        offsets.join(", "),
    )
}

/// The WGSL of a field kernel.
pub(super) fn body(id: KernelId) -> String {
    let main = match id {
        KernelId::LensFieldPyramid => "lf_pyramid_main",
        KernelId::LensFieldGather => "lf_gather_main",
        _ => "lf_finish_main",
    };
    let mut s = constants();
    s.push_str(include_str!("../shaders/lens_field.wgsl"));
    s.push_str(&format!(
        "\n@compute @workgroup_size({GROUP}, {GROUP}, 1)\n\
         fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{ {main}(gid); }}\n"
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
        for id in [
            KernelId::LensFieldPyramid,
            KernelId::LensFieldGather,
            KernelId::LensFieldFinish,
        ] {
            let (fields, span) = uniform_layout(id, "pu");
            let want = [
                ("zero", offset_of!(FieldU, zero)),
                ("frame", offset_of!(FieldU, frame)),
                ("plan", offset_of!(FieldU, plan)),
                ("ints", offset_of!(FieldU, ints)),
                ("src", offset_of!(FieldU, src)),
                ("srcs", offset_of!(FieldU, srcs)),
                ("win", offset_of!(FieldU, win)),
                ("dst", offset_of!(FieldU, dst)),
                ("gth", offset_of!(FieldU, gth)),
                ("gs", offset_of!(FieldU, gs)),
                ("lvl", offset_of!(FieldU, lvl)),
                ("org", offset_of!(FieldU, org)),
                ("ex", offset_of!(FieldU, ex)),
            ];
            assert_eq!(fields.len(), want.len(), "{id:?}");
            for ((name, offset), (want_name, want_offset)) in fields.iter().zip(want) {
                assert_eq!(name, want_name);
                assert_eq!(*offset as usize, want_offset, "{id:?}: offset of {name}");
            }
            assert_eq!(span as usize, size_of::<FieldU>(), "{id:?}");
        }
    }

    #[test]
    fn the_table_layout_matches_the_planner() {
        // 9 radii of (9 + 25 + 49 + 81 + 121 + 169 + 225 + 289) cells.
        assert_eq!(total_cells(), RADII * 968);
        assert_eq!(field_table_offset(3), 0);
        assert_eq!(field_table_offset(5), RADII * 9);
    }

    #[test]
    fn the_pyramid_covers_the_gathers_and_halves_exactly() {
        let pass = FieldPass {
            width: 6000,
            height: 4000,
            half_diag: 3605.5,
            axis_y: false,
            e: 1.0,
            alpha: 1.0,
            radius_scale: 1.0,
            side: 0.0,
            fringe_boost: 0.0,
            lmax: 3,
            gather_halo: 70,
            pad: 0,
        };
        let p = pyramid_layout(&pass, Rect::new(0, 0, 1024, 1024));
        assert_eq!(p.levels.len(), 4);
        for l in &p.levels {
            assert!(l[1] > 0 && l[2] > 0);
        }
        // Level 0 starts at a multiple of 8 at or below -70 and holds the grown region.
        assert_eq!(p.levels[0][3], -72);
        assert!(p.levels[0][1] as usize >= 1024 + 140);
        assert_eq!(p.levels[0][1] % 8, 0);
        // Each level halves the one before.
        for w in p.levels.windows(2) {
            assert_eq!(w[1][1], w[0][1] / 2);
            assert_eq!(w[1][3], w[0][3] / 2);
        }
    }
}

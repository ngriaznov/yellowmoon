//! The geometry sampler on the device (`docs/17-gpu.md` section 11): camera RGB through a grid that
//! maps the pixels of a view or an export to positions in a pixel buffer, through the camera's EXIF
//! orientation, the user's quarter turns, the straighten angle and the crop. The CPU reference is
//! [`crate::cpu::sampler`], whose module notes describe the model and the three kernels (a pixel
//! copy, Catmull-Rom, a box of bilinear sub-samples).
//!
//! The output is a four-plane [`GpuImage`]: red, green, blue, and a plane that is 1.0 where the pixel
//! is outside the photo (the letterbox) and 0.0 inside. That plane travels with the pixels through
//! the colour pass ([`PreparedChain::encode_stage`](super::color::PreparedChain::encode_stage)) and
//! output sharpening to the C13 pass ([`encode_c13`](super::encode::encode_c13)), which paints it.
//!
//! # Positions and the inside test
//!
//! The host collapses the position chain into one `f64` affine map
//! ([`SamplerGrid::affine_to_buffer`]) and hands the kernel two tables, one entry per column and one
//! per row of the picture (per sub-sample of a box): what that column or row contributes to the source
//! position, split in `f64` into an integer part and a fraction (the row entries also carry the offset
//! and the integer origin of the source image). A position is the sum of one column entry and one row
//! entry, the integers added as integers and the fractions as `f32`, so no `f32` value of order 1e4
//! occurs (D4 of `docs/17-gpu.md`) and the position is exact to a fraction's precision, about
//! 1e-7 px. The entries depend on the absolute column and row only, not on the window: a tile and
//! the whole frame sample every pixel with the same bits (D5). The inside test is not an `f32`
//! comparison either: the host gives the exact interval of columns the CPU predicate accepts for
//! every row and sub-sample ([`SamplerGrid::inside_rows`]), so the mask equals the CPU's bit for
//! bit.
//!
//! # The lens's bend
//!
//! A grid with a [`SamplerWarp`] (L4, `docs/21` section 4) is not
//! separable: the photo position a canvas position reads is the root of a nonlinear map. The tables
//! stay what they are (the chain *without* the warp, exact), and the kernel adds the warp's
//! displacement to that position: it takes the canvas offset from the centre in half-diagonal units
//! (the integer part of the offset exact), runs the CPU's Newton steps in `f32` on the offset from the
//! start so that nothing of order 1 is subtracted from something of order 1, and moves the exact
//! position by the displacement (a few percent of the half-diagonal at most). The uniform block
//! carries the coefficients, the zoom, the dihedral map's linear part ([`SamplerGrid::warp_frame`]),
//! the canvas centre and the clamp to the buffer. Positions then differ from the CPU's by an `f32`
//! rounding of the displacement, about 1e-5 px at 50 MP, and a pixel is still a function of the
//! absolute column and row only, so a tile and the whole frame agree bit for bit.
//!
//! # Entry points
//!
//! [`encode_sample`] records the pass into a [`PassList`] and returns the
//! four-plane image; [`sample`] is the thin host wrapper (upload, pass, readback) that tests and the
//! first wiring call.

use bytemuck::{Pod, Zeroable};

use super::shader::KernelId;
use super::util::{STORAGE_IN, even};
use crate::cpu::sampler::{Affine2, Kernel, Sampled, SamplerGrid, SamplerWarp};
use crate::cpu::tiling::Rect;
use crate::cpu::{Placement, RgbImage};
use crate::device::{GpuContext, GpuError};
use crate::image::GpuImage;
use crate::pool::PooledBuffer;
use crate::stream::{PassList, upload_bytes};

/// The uniform block of `sampler.wgsl` (`SamplerU`): 208 bytes. The last ten members are the lens's
/// bend and are zero without one.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct SamplerU {
    dims: [u32; 4],
    src: [u32; 4],
    /// Kernel, sub-samples per axis, 1 with a warp.
    mode: [u32; 4],
    /// Newton steps, 1 when the anamorphic axis is the canvas's y.
    wk: [u32; 4],
    /// `a1 .. a4`: the first four terms of `x'` (`x^2, y^2, x^4, x^2 y^2`).
    wx: [f32; 4],
    /// `b1 .. b4`: the first four terms of `y'`.
    wy: [f32; 4],
    /// `a5`, `b5` (the `y^4` terms), `d1`, `d2`.
    wc: [f32; 4],
    /// Zoom, zoom - 1, half-diagonal in canvas pixels, its inverse.
    wb: [f32; 4],
    /// Canvas displacement to buffer displacement, row-major.
    sl: [f32; 4],
    /// Its inverse, row-major.
    si: [f32; 4],
    /// Integer part of the canvas centre's buffer position, relative to the source window.
    ci: [i32; 4],
    /// Its fraction.
    cf: [f32; 4],
    /// The buffer's corners relative to the source window: `x0, y0, x1, y1`.
    lim: [i32; 4],
}

impl SamplerU {
    /// The bend's part of the block for `grid`, whose source window starts at `origin`.
    fn set_warp(&mut self, grid: &SamplerGrid, origin: (usize, usize)) {
        let Some(w) = &grid.warp else {
            return;
        };
        let frame = grid.warp_frame();
        let h = SamplerWarp::half_diagonal(grid.canvas);
        let (ox, oy) = (origin.0 as f64, origin.1 as f64);
        let (cxi, cxf) = split(frame.centre.0 - ox);
        let (cyi, cyf) = split(frame.centre.1 - oy);
        self.mode[2] = 1;
        self.wk = [u32::from(w.steps), u32::from(w.axis_y), 0, 0];
        let (a, b) = (w.bend_x, w.bend_y);
        self.wx = [a[0] as f32, a[1] as f32, a[2] as f32, a[3] as f32];
        self.wy = [b[0] as f32, b[1] as f32, b[2] as f32, b[3] as f32];
        self.wc = [a[4] as f32, b[4] as f32, w.d1 as f32, w.d2 as f32];
        self.wb = [
            w.zoom as f32,
            (w.zoom - 1.0) as f32,
            h as f32,
            (1.0 / h) as f32,
        ];
        let m = |a: [[f64; 2]; 2]| {
            [
                a[0][0] as f32,
                a[0][1] as f32,
                a[1][0] as f32,
                a[1][1] as f32,
            ]
        };
        self.sl = m(frame.linear);
        self.si = m(frame.inverse);
        self.ci = [cxi, cyi, 0, 0];
        self.cf = [cxf, cyf, 0.0, 0.0];
        self.lim = [
            (-ox) as i32,
            (-oy) as i32,
            (grid.buffer.0 - ox) as i32,
            (grid.buffer.1 - oy) as i32,
        ];
    }
}

/// One entry of the position tables: an integer part and a fraction in `[0, 1)` for the source x and
/// y (16 bytes, `PosEntry` of the shader).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default, Debug, PartialEq)]
struct PosEntry {
    int: [i32; 2],
    frac: [f32; 2],
}

fn split(v: f64) -> (i32, f32) {
    let fl = v.floor();
    (fl as i32, (v - fl) as f32)
}

fn entry(x: f64, y: f64) -> PosEntry {
    let (xi, xf) = split(x);
    let (yi, yf) = split(y);
    PosEntry {
        int: [xi, yi],
        frac: [xf, yf],
    }
}

/// What each column of the window contributes to the source position, per sub-sample: entry
/// `x * taps + i` for the output position `X = off_x + x + (i + 0.5) / taps`.
fn column_entries(m: &Affine2, off_x: usize, w: usize, taps: usize) -> Vec<PosEntry> {
    let mut v = Vec::with_capacity(w * taps);
    for x in 0..w {
        for i in 0..taps {
            let xs = (off_x + x) as f64 + (i as f64 + 0.5) / taps as f64;
            v.push(entry(m[0][0] * xs, m[1][0] * xs));
        }
    }
    v
}

/// What each row contributes, with the offset of the map and less the integer origin of the source:
/// entry `y * taps + j` for the output position `Y = off_y + y + (j + 0.5) / taps`.
fn row_entries(
    m: &Affine2,
    off_y: usize,
    h: usize,
    taps: usize,
    origin: (usize, usize),
) -> Vec<PosEntry> {
    let mut v = Vec::with_capacity(h * taps);
    for y in 0..h {
        for j in 0..taps {
            let ys = (off_y + y) as f64 + (j as f64 + 0.5) / taps as f64;
            v.push(entry(
                m[0][1] * ys + m[0][2] - origin.0 as f64,
                m[1][1] * ys + m[1][2] - origin.1 as f64,
            ));
        }
    }
    v
}

/// The WGSL of the sampler kernel.
pub(super) fn body() -> String {
    String::from(include_str!("../shaders/sampler.wgsl"))
}

/// The size of the largest storage binding a sampling pass writing the `window` makes (K4 of
/// `docs/17-gpu.md`): the four-plane output, 16 bytes a pixel. (The source image is the caller's.)
pub fn max_binding_bytes(window: Rect) -> u64 {
    16 * even(window.w) as u64 * window.h as u64
}

fn kernel_code(kernel: Kernel) -> u32 {
    match kernel {
        Kernel::Copy => 0,
        Kernel::Bicubic => 1,
        Kernel::Box(_) => 2,
    }
}

/// Records the sampling pass: the `w x h` output pixels at offset `off` of `grid`, read from `src`
/// (three planes of camera RGB), an image whose pixel (0, 0) is buffer pixel `src_origin`. Returns the
/// four-plane result (module notes). The only upload is the uniform block and the row intervals of
/// the inside test (`8 h n^2` bytes, `n` the box size).
///
/// # Errors
///
/// [`GpuError::InvalidInput`] for an empty output or a source that is not a three- or four-plane
/// image, or an output too big for one dispatch; the pool's errors for the output.
pub fn encode_sample(
    passes: &mut PassList<'_>,
    src: &GpuImage,
    src_origin: (usize, usize),
    grid: &SamplerGrid,
    off: (usize, usize),
    w: usize,
    h: usize,
) -> Result<GpuImage, GpuError> {
    let ctx = passes.context();
    if w == 0 || h == 0 {
        return Err(GpuError::InvalidInput(format!(
            "the sampler needs a non-empty output, not {w} x {h}"
        )));
    }
    if src.planes < 3 {
        return Err(GpuError::InvalidInput(format!(
            "the sampler reads three planes of camera RGB, the source has {}",
            src.planes
        )));
    }
    let limit = ctx.info().limits.max_compute_workgroups_per_dimension as usize;
    if w.div_ceil(16) > limit || h.div_ceil(16) > limit {
        return Err(GpuError::InvalidInput(format!(
            "a {w} x {h} output needs more workgroups than the device allows ({limit})"
        )));
    }
    let kernel = grid.kernel();
    let taps = kernel.taps();
    let m = grid.affine_to_buffer();
    let inside = grid.inside_rows(off, w, h, taps);
    let cols = column_entries(&m, off.0, w, taps);
    let rows = row_entries(&m, off.1, h, taps, src_origin);
    let mut table = |label: &str, bytes: &[u8]| -> Result<_, GpuError> {
        let buf = ctx.pool().acquire(label, bytes.len() as u64, STORAGE_IN)?;
        upload_bytes(ctx, &buf, 0, bytes)?;
        Ok(passes.keep(buf))
    };
    let cols_buf = table("sampler columns", bytemuck::cast_slice(&cols))?;
    let rows_buf = table("sampler rows", bytemuck::cast_slice(&rows))?;
    let inside_buf = table("sampler inside", bytemuck::cast_slice(&inside))?;
    let out = GpuImage::alloc(ctx, "sampled", w, h, 4, Placement::whole(w, h))?;
    let mut u = SamplerU {
        dims: [
            w as u32,
            h as u32,
            out.stride as u32,
            out.plane_len() as u32,
        ],
        src: [
            src.width as u32,
            src.height as u32,
            src.stride as u32,
            src.plane_len() as u32,
        ],
        mode: [kernel_code(kernel), taps as u32, 0, 0],
        ..SamplerU::default()
    };
    u.set_warp(grid, src_origin);
    let ub = passes.uniform("sampler uniform", &u)?;
    passes.dispatch(
        KernelId::Sampler,
        "sampler",
        &[
            (0, ub.binding()),
            (1, src.buf.binding()),
            (2, cols_buf.binding()),
            (3, rows_buf.binding()),
            (4, inside_buf.binding()),
            (5, out.buf.binding()),
        ],
        (w.div_ceil(16) as u32, h.div_ceil(16) as u32, 1),
    )?;
    Ok(out)
}

/// Uploads `image` (an image whose pixel (0, 0) is buffer pixel `origin`), samples the `w x h` output
/// pixels at offset `off` of `grid` on the device and reads the result back: the host wrapper of
/// [`encode_sample`], with the same type as the CPU port ([`crate::cpu::sampler::sample_to_camera`]).
///
/// # Errors
///
/// The errors of [`encode_sample`] and of the upload and readback.
pub fn sample(
    ctx: &GpuContext,
    image: &RgbImage,
    origin: (usize, usize),
    grid: &SamplerGrid,
    off: (usize, usize),
    w: usize,
    h: usize,
) -> Result<Sampled, GpuError> {
    let src = GpuImage::upload_rgb(ctx, "sampler input", image)?;
    let mut passes = PassList::new(ctx, "sampler");
    let out = encode_sample(&mut passes, &src, origin, grid, off, w, h)?;
    let id = out.encode_read(&mut passes)?;
    let _src: std::sync::Arc<PooledBuffer> = passes.keep(src.buf);
    let done = passes.submit()?.finish()?;
    let planar = out.take(&done, id)?;
    let n = w * h;
    let img = RgbImage::from_planar(w, h, planar[..3 * n].to_vec())?;
    let mask: Vec<u8> = planar[3 * n..]
        .iter()
        .map(|&v| u8::from(v != 0.0))
        .collect();
    let any = mask.iter().any(|&v| v != 0);
    Ok(Sampled {
        img,
        outside: any.then_some(mask),
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
    fn the_uniform_block_has_the_layout_of_the_wgsl_struct() {
        let source = shader_source(StorageMode::F32, KernelId::Sampler);
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
        Validator::new(ValidationFlags::all(), Capabilities::empty())
            .validate(&module)
            .expect("valid");
        let var = module
            .global_variables
            .iter()
            .map(|(_, v)| v)
            .find(|v| v.name.as_deref() == Some("su"))
            .expect("the uniform su");
        let naga::TypeInner::Struct { members, span } = &module.types[var.ty].inner else {
            panic!("su is not a struct");
        };
        let fields: [(&str, usize); 13] = [
            ("dims", offset_of!(SamplerU, dims)),
            ("src", offset_of!(SamplerU, src)),
            ("mode", offset_of!(SamplerU, mode)),
            ("wk", offset_of!(SamplerU, wk)),
            ("wx", offset_of!(SamplerU, wx)),
            ("wy", offset_of!(SamplerU, wy)),
            ("wc", offset_of!(SamplerU, wc)),
            ("wb", offset_of!(SamplerU, wb)),
            ("sl", offset_of!(SamplerU, sl)),
            ("si", offset_of!(SamplerU, si)),
            ("ci", offset_of!(SamplerU, ci)),
            ("cf", offset_of!(SamplerU, cf)),
            ("lim", offset_of!(SamplerU, lim)),
        ];
        assert_eq!(members.len(), fields.len());
        for (member, (name, offset)) in members.iter().zip(fields) {
            assert_eq!(member.name.as_deref(), Some(name));
            assert_eq!(member.offset as usize, offset, "offset of {name}");
        }
        assert_eq!(*span as usize, size_of::<SamplerU>());
        // The position entry: two i32, then two f32.
        assert_eq!(size_of::<PosEntry>(), 16);
        assert_eq!(offset_of!(PosEntry, frac), 8);
    }

    #[test]
    fn a_position_splits_into_an_integer_and_a_fraction() {
        assert_eq!(split(12.75), (12, 0.75));
        assert_eq!(split(-0.25), (-1, 0.75));
        assert_eq!(split(-3.0), (-3, 0.0));
        let e = entry(1.5, -2.25);
        assert_eq!((e.int, e.frac), ([1, -3], [0.5, 0.75]));
    }
}

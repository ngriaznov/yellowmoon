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
//! # Snapped positions
//!
//! The split above gives the same pixel different `f32` fractions when it is reached through different columns and
//! rows: a window of a level has its origin in the map's offset (row entries), the level's band has it in its columns
//! (`n / z` against `(1 / z) * (n + x)`), and `f32(a) + f32(b)` is not `f32(a + b)`, so a crop of a level and the
//! window render of the same pixels sampled up to 6e-8 px apart, which a code boundary somewhere in the picture
//! turned into a one-code byte on Metal. A grid with [`SamplerGrid::snapped_positions`] (the viewport's views, never
//! the exports) and no rotation puts each source coordinate whole into the one table it depends on, rounded to a
//! multiple of 2^-24 px (exact in an `f32` fraction), and the other table gives 0 for it: the same position, from
//! the same lattice value, whichever way the render reached it (it differs only when the two `f64` evaluations
//! straddle a lattice midpoint, about 2e-6 of the positions, by one step). A position that is a whole number comes
//! out whole, which the split did not guarantee.
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
use crate::cpu::sampler::{
    Affine2, Kernel, SNAPPED_POSITION_STEP, Sampled, SamplerGrid, SamplerWarp,
};
use crate::cpu::tiling::Rect;
use crate::cpu::{Placement, RgbImage};
use crate::device::{GpuContext, GpuError};
use crate::image::GpuImage;
use crate::pool::PooledBuffer;
use crate::stream::{PassList, upload_bytes};

/// The uniform block of `sampler.wgsl` (`SamplerU`): 256 bytes. Members 4 to 13 are the lens's bend and are
/// zero without one; the last three place the pixel in the photo frame and are zero unless the photo-frame
/// planes are asked for.
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
    /// The source window's origin in the buffer (`x, y`).
    po: [i32; 4],
    /// The camera's orientation undone on fractions, row-major.
    pm: [f32; 4],
    /// Its offset (`x, y`) and `1 / buffer width`, `1 / buffer height`.
    pt: [f32; 4],
}

impl SamplerU {
    /// The photo-frame positions' part of the block for `grid`, whose source window starts at `origin`: the
    /// kernel writes planes 4 and 5.
    fn set_positions(&mut self, grid: &SamplerGrid, origin: (usize, usize)) {
        let (m, t) = grid.sensor_to_oriented_map();
        self.mode[3] = 1;
        self.po = [origin.0 as i32, origin.1 as i32, 0, 0];
        self.pm = [
            m[0][0] as f32,
            m[0][1] as f32,
            m[1][0] as f32,
            m[1][1] as f32,
        ];
        self.pt = [
            t[0] as f32,
            t[1] as f32,
            (1.0 / grid.buffer.0) as f32,
            (1.0 / grid.buffer.1) as f32,
        ];
    }

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

/// Which table carries each source coordinate when a grid's positions are snapped
/// ([`SamplerGrid::snapped_positions`]): `true` for the column table (the coordinate depends on the output's x
/// alone), `false` for the row table (on y alone). `None` for a grid that is not axis-aligned (a straightened
/// photo): its coordinates depend on both, and the tables keep the split of the module notes.
fn snapped_carriers(grid: &SamplerGrid, m: &Affine2) -> Option<[bool; 2]> {
    if !grid.snapped_positions {
        return None;
    }
    let one = |row: &[f64; 3]| {
        if row[1] == 0.0 {
            Some(true)
        } else if row[0] == 0.0 {
            Some(false)
        } else {
            None
        }
    };
    Some([one(&m[0])?, one(&m[1])?])
}

/// A source coordinate rounded to a multiple of [`SNAPPED_POSITION_STEP`] (2^-24 px, exact in an `f32` fraction): the
/// same number for the same position whatever `f64` rounding the way to it took, to within one step with a
/// probability of about 1e-13 / 6e-8 per position. A position that is a whole number comes out whole.
fn snap_position(v: f64) -> f64 {
    (v / SNAPPED_POSITION_STEP).round() * SNAPPED_POSITION_STEP
}

/// What each column of the window contributes to the source position, per sub-sample: entry
/// `x * taps + i` for the output position `X = off_x + x + (i + 0.5) / taps`. With `snapped` (see
/// [`snapped_carriers`]) a coordinate that depends on x alone is carried here whole: the map's offset, less the integer
/// origin of the source, in one entry, snapped, and the row entries give 0 for it.
fn column_entries(
    m: &Affine2,
    off_x: usize,
    w: usize,
    taps: usize,
    origin: (usize, usize),
    snapped: Option<[bool; 2]>,
) -> Vec<PosEntry> {
    let mut v = Vec::with_capacity(w * taps);
    for x in 0..w {
        for i in 0..taps {
            let xs = (off_x + x) as f64 + (i as f64 + 0.5) / taps as f64;
            let one = |c: usize, o: usize| match snapped {
                Some(carried) if carried[c] => snap_position(m[c][0] * xs + m[c][2]) - o as f64,
                Some(_) => 0.0,
                None => m[c][0] * xs,
            };
            v.push(entry(one(0, origin.0), one(1, origin.1)));
        }
    }
    v
}

/// What each row contributes, with the offset of the map and less the integer origin of the source:
/// entry `y * taps + j` for the output position `Y = off_y + y + (j + 0.5) / taps`. With `snapped`, a coordinate that
/// depends on y alone is carried here whole (snapped), and the column entries give 0 for it.
fn row_entries(
    m: &Affine2,
    off_y: usize,
    h: usize,
    taps: usize,
    origin: (usize, usize),
    snapped: Option<[bool; 2]>,
) -> Vec<PosEntry> {
    let mut v = Vec::with_capacity(h * taps);
    for y in 0..h {
        for j in 0..taps {
            let ys = (off_y + y) as f64 + (j as f64 + 0.5) / taps as f64;
            let one = |c: usize, o: usize| match snapped {
                Some(carried) if !carried[c] => snap_position(m[c][1] * ys + m[c][2]) - o as f64,
                Some(_) => 0.0,
                None => m[c][1] * ys + m[c][2] - o as f64,
            };
            v.push(entry(one(0, origin.0), one(1, origin.1)));
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
    encode_sample_at(passes, src, src_origin, grid, off, w, h, false)
}

/// [`encode_sample`], and with `positions` a six-plane result: the fifth and sixth planes hold each pixel's
/// position in the photo frame (the oriented fractions the mask shapes of the nodes are placed in;
/// [`SamplerGrid::photo_position`] is the CPU twin), for the colour pass to read.
///
/// # Errors
///
/// As [`encode_sample`].
#[allow(clippy::too_many_arguments)] // the natural argument list; a struct would only move them
pub fn encode_sample_at(
    passes: &mut PassList<'_>,
    src: &GpuImage,
    src_origin: (usize, usize),
    grid: &SamplerGrid,
    off: (usize, usize),
    w: usize,
    h: usize,
    positions: bool,
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
    let snapped = snapped_carriers(grid, &m);
    let cols = column_entries(&m, off.0, w, taps, src_origin, snapped);
    let rows = row_entries(&m, off.1, h, taps, src_origin, snapped);
    let mut table = |label: &str, bytes: &[u8]| -> Result<_, GpuError> {
        let buf = ctx.pool().acquire(label, bytes.len() as u64, STORAGE_IN)?;
        upload_bytes(ctx, &buf, 0, bytes)?;
        Ok(passes.keep(buf))
    };
    let cols_buf = table("sampler columns", bytemuck::cast_slice(&cols))?;
    let rows_buf = table("sampler rows", bytemuck::cast_slice(&rows))?;
    let inside_buf = table("sampler inside", bytemuck::cast_slice(&inside))?;
    let out = GpuImage::alloc(
        ctx,
        "sampled",
        w,
        h,
        if positions { 6 } else { 4 },
        Placement::whole(w, h),
    )?;
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
    if positions {
        u.set_positions(grid, src_origin);
    }
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
        let fields: [(&str, usize); 16] = [
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
            ("po", offset_of!(SamplerU, po)),
            ("pm", offset_of!(SamplerU, pm)),
            ("pt", offset_of!(SamplerU, pt)),
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

    /// The position the shader forms from a column entry and a row entry: the integers added as integers, the fractions
    /// as `f32`, carried into the integer (the shader floors the fraction where it needs the tap).
    fn position(c: &PosEntry, r: &PosEntry, axis: usize) -> (i64, u32) {
        let f = c.frac[axis] + r.frac[axis];
        let fl = f.floor();
        (
            i64::from(c.int[axis]) + i64::from(r.int[axis]) + fl as i64,
            (f - fl).to_bits(),
        )
    }

    /// A pixel of a window of a level and the same pixel of a band of the whole level are given the same source position, bit
    /// for bit, once the grid's positions are snapped: the window's map has the origin `(n, m)` in its offset
    /// (`n / z`, as `ViewportState::view_to_final` writes it), the band's map has the band's first row (`row0 * (1 / z)`, as
    /// the fill composes it), and the unsnapped tables split the same position differently between a column entry and a row
    /// entry. The numbers are those of the 75 % case of `ym-core`'s `view_level` test, whose pan (53.5, -9.25) failed on
    /// Metal by one code.
    #[test]
    fn a_window_and_a_band_of_a_level_get_the_same_positions_once_snapped() {
        let buffer = (640.0, 420.0);
        let grid = |s: f64, tx: f64, ty: f64, snapped: bool| {
            SamplerGrid::new([[s, 0.0, tx], [0.0, s, ty]], buffer, buffer, 1, 0, true)
                .with_snapped_positions(snapped)
        };
        let (w, h, taps) = (300usize, 200usize, 2usize);
        let mut unsnapped_differ = 0usize;
        for z in [0.6f64, 0.75, 0.9] {
            let s = 1.0 / z;
            for (n, m) in [(74usize, 46usize), (127, 36), (85, 94), (26, 14), (90, 57)] {
                for row0 in [0usize, 100, 160] {
                    for snapped in [false, true] {
                        let window_grid = grid(s, n as f64 / z, m as f64 / z, snapped);
                        let band_grid = grid(s, 0.0, s * row0 as f64, snapped);
                        let (wm, bm) =
                            (window_grid.affine_to_buffer(), band_grid.affine_to_buffer());
                        let carriers = |g: &SamplerGrid, a: &Affine2| snapped_carriers(g, a);
                        let origin = (0, 0);
                        let wc =
                            column_entries(&wm, 0, w, taps, origin, carriers(&window_grid, &wm));
                        let wr = row_entries(&wm, 0, h, taps, origin, carriers(&window_grid, &wm));
                        let bw = (640.0 * z).round() as usize;
                        let bc =
                            column_entries(&bm, 0, bw, taps, origin, carriers(&band_grid, &bm));
                        let bh = 150usize;
                        let br = row_entries(&bm, 0, bh, taps, origin, carriers(&band_grid, &bm));
                        for y in 0..h {
                            let yl = m + y;
                            if yl < row0 || yl - row0 >= bh {
                                continue;
                            }
                            for x in 0..w {
                                if n + x >= bw {
                                    continue;
                                }
                                for j in 0..taps {
                                    for i in 0..taps {
                                        let (wce, wre) = (&wc[x * taps + i], &wr[y * taps + j]);
                                        let (bce, bre) =
                                            (&bc[(n + x) * taps + i], &br[(yl - row0) * taps + j]);
                                        for axis in 0..2 {
                                            let same = position(wce, wre, axis)
                                                == position(bce, bre, axis);
                                            if snapped {
                                                assert!(
                                                    same,
                                                    "z {z} origin ({n}, {m}) row0 {row0} pixel ({x}, {y}) axis {axis}: {wce:?} {wre:?} against {bce:?} {bre:?}"
                                                );
                                            } else if !same {
                                                unsnapped_differ += 1;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(
            unsnapped_differ > 0,
            "the unsnapped tables are what the snapping fixes"
        );
    }

    /// A snapped position is a whole number where the position is whole, however its `f64` evaluation rounded, and sits within
    /// half a step of the exact one.
    #[test]
    fn a_snapped_position_is_whole_where_the_exact_one_is_and_close_elsewhere() {
        for k in 0..2000i64 {
            let exact = k as f64 * 4.0 / 3.0 + 0.25;
            for noise in [-3e-13, 0.0, 2e-13] {
                let q = snap_position(exact + noise);
                assert!((q - exact).abs() <= 0.5 * SNAPPED_POSITION_STEP + 1e-12);
                assert_eq!(snap_position(q), q, "idempotent");
            }
        }
        for k in -50..50i64 {
            for noise in [-3e-13, 0.0, 2e-13] {
                assert_eq!(snap_position(k as f64 + noise), k as f64);
            }
        }
    }
}
